"""
LOTUS, learned world model: TD-MPC2 (Hansen, Su & Wang, ICLR 2024) for the self-driving car, extended (v2) with
physics, several futures, risk-aware planning and scenario evolution inside the model.

The agent learns a model of the world in a latent space and drives by planning in it:
  * encoder     DeepSets over the nearest road users the car can see -> latent z
  * dynamics    "what happens 0.2 s after action a", as several hypotheses with their odds (multiple-choice learning):
                the different ways the world may go on (the motorbike cuts in, or it does not)
  * reward, Q   ensemble of 5 values (two-hot), as in TD-MPC2
  * physics     from z: the car's speed and where the 6 nearest road users are and how they move; from (z, a): how the
                car itself moves in 0.2 s. Rolled through the dynamics this is a prediction of the physical future
  * collision   from z: probability of contact within 1, 2 and 3 s
  * genome      dynamics, reward, Q and collision are conditioned on the scenario genome (who is on the road and how
                they ride, the 13 genes of phase 3), so the model is a learned simulator of *any* traffic culture
  * policy      the prior that proposes plans (and drives alone in the browser)

Planning (every 0.2 s): MPPI over 6 decisions (1.2 s), each candidate plan imagined in 3 sampled futures; a plan is
scored by its expected return minus predicted contact, aggregated between the mean and the worst future (CVaR), so the
car prefers what is safe in the bad futures too.

Evolution inside the world model: every --evo-every steps, MAP-Elites proposes scenario genomes and scores them in the
model's imagination (thousands per second: how much return would the car get, how often would it touch someone, in
this traffic culture). The most promising are checked in the real simulation (difficulty for the current car, and the
realism filter of phase 3), kept in an archive of diverse hard scenarios, and mixed into the training environments.

Reward v2 (only for this driver): crashes cost 25, hitting a person 40, being hit 5, plus defensive driving (slow near
people, a time gap to the vehicle ahead, stopping within sight); the demonstrations come from the defensive teacher.

  python tdmpc2.py --steps 1000000 --workers 16 --demos 40000          # full run (GPU recommended)
  python tdmpc2.py --steps 3000 --workers 2 --envs 2 --demos 400 --samples 64 --eval-n 4   # smoke test
  python tdmpc2.py --eval runs/tdmpc2/final.pt --eval-n 24              # evaluate with planning

Outputs in --out: checkpoints (.pt), policy.json (the policy prior alone, runs in the browser without planning),
log.json, eval_*.json, episodes/ (training episodes with what the car imagined, replayed on the monitor's /sim page),
evo/ (the scenario archive and its log).
"""
import argparse, base64, json, math, os, random, subprocess, time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train import VecEnv
from distill import HEAD, K, NB, OBS, note, HERE
import coevolve as CO   # the scenario genome of phase 3: SPACE, DEFAULT, mutate, random_genome, descriptor, evaluate, score, Archive

ACT = 2
GKEYS = list(CO.SPACE.keys()); GD = len(GKEYS)
GSPACE = {"keys": GKEYS, "lo": [CO.SPACE[k][0] for k in GKEYS], "hi": [CO.SPACE[k][1] for k in GKEYS], "def": [CO.DEFAULT[k] for k in GKEYS]}
def gnorm(g): return np.array([(g.get(k, CO.DEFAULT[k]) - CO.SPACE[k][0]) / (CO.SPACE[k][1] - CO.SPACE[k][0]) * 2 - 1 for k in GKEYS], np.float32)
G_DEFAULT = gnorm(CO.DEFAULT)
NPH = 6                                   # road users decoded by the physics head
PHYS_F = (0, 1, 2, 3, 13, 15)             # of each: ahead, sideways, relative speed (2), is a person, present
PHYS_IDX = [0] + [HEAD + j * NB + f for j in range(NPH) for f in PHYS_F]
PD = len(PHYS_IDX)
COLL_K = (5, 10, 15)                      # contact within 1, 2, 3 s (decisions of 0.2 s)
KIN_SCALE = np.array([4.0, 1.0, 0.5], np.float32)   # the car's move in one decision (dx, dy, dh), normalised
RANGE = 30.0                              # ENV.RANGE: neighbour positions are divided by it
ROOT = os.path.dirname(HERE)


# ---------------------------------------------------------------- building blocks
class SimNorm(nn.Module):
    """simplicial normalisation: the latent is split into groups of 8 and each group is a softmax"""
    def __init__(self, dim=8): super().__init__(); self.dim = dim
    def forward(self, x): s = x.shape; return F.softmax(x.view(*s[:-1], -1, self.dim), -1).view(*s)

def mlp(inp, hidden, out, act_out=None, dropout=0.0):
    layers, d = [], inp
    for i, h in enumerate(hidden):
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.Mish()]
        if dropout and i == 0: layers.insert(len(layers) - 2, nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, out))
    if act_out is not None: layers += [nn.LayerNorm(out), act_out] if isinstance(act_out, SimNorm) else [act_out]
    return nn.Sequential(*layers)

def symlog(x): return torch.sign(x) * torch.log(1 + x.abs())
def symexp(x): return torch.sign(x) * (torch.exp(x.abs()) - 1)

class TwoHot:
    """values as a categorical distribution over bins evenly spaced in symlog space (cross-entropy regression)"""
    def __init__(self, n=101, vmin=-10.0, vmax=10.0, device="cpu"):
        self.n, self.vmin, self.vmax = n, vmin, vmax; self.bins = torch.linspace(vmin, vmax, n, device=device); self.w = (vmax - vmin) / (n - 1)
    def encode(self, x):
        x = symlog(x).clamp(self.vmin, self.vmax).squeeze(-1); idx = ((x - self.vmin) / self.w).floor().long().clamp(0, self.n - 2)
        off = (x - self.bins[idx]) / self.w; t = torch.zeros(*x.shape, self.n, device=x.device)
        t.scatter_(-1, idx.unsqueeze(-1), (1 - off).unsqueeze(-1)); t.scatter_(-1, (idx + 1).unsqueeze(-1), off.unsqueeze(-1)); return t
    def decode(self, logits): return symexp((F.softmax(logits, -1) * self.bins).sum(-1, keepdim=True))
    def ce(self, logits, target): return -(self.encode(target) * F.log_softmax(logits, -1)).sum(-1, keepdim=True)


class SetEncoder(nn.Module):
    """DeepSets over the neighbours (shared MLP, max + mean pooling over present slots), then an MLP to the latent"""
    def __init__(self, latent, h_phi=64, hidden=256, simnorm=8):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(NB, h_phi), nn.ReLU(), nn.Linear(h_phi, h_phi), nn.ReLU())
        self.head = mlp(HEAD + 2 * h_phi, [hidden], latent, SimNorm(simnorm))
        self.register_buffer("mean", torch.zeros(OBS)); self.register_buffer("std", torch.ones(OBS))
    def forward(self, obs):
        x = ((obs - self.mean) / self.std).clamp(-8, 8)
        head, nb = x[..., :HEAD], x[..., HEAD:].reshape(*x.shape[:-1], K, NB)
        present = obs[..., HEAD:].reshape(*obs.shape[:-1], K, NB)[..., NB - 1:NB] > 0.5
        h = self.phi(nb)
        mx = torch.where(present, h, torch.zeros_like(h)).max(-2).values
        mn = (h * present.float()).sum(-2) / present.float().sum(-2).clamp(min=1)
        return self.head(torch.cat([head, mx, mn], -1))


class Dynamics(nn.Module):
    """several hypotheses of the next latent and their odds (multiple-choice learning): the futures the world may take"""
    def __init__(self, latent, hidden, heads, simnorm):
        super().__init__()
        self.M, self.L = heads, latent
        self.trunk = nn.Sequential(nn.Linear(latent + ACT + GD, hidden), nn.LayerNorm(hidden), nn.Mish(), nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Mish())
        self.out = nn.Linear(hidden, heads * latent); self.ln = nn.LayerNorm(latent); self.sn = SimNorm(simnorm)
        self.logit = nn.Linear(hidden, heads)
    def forward(self, z, a, g):
        h = self.trunk(torch.cat([z, a, g], -1))
        return self.sn(self.ln(self.out(h).view(*h.shape[:-1], self.M, self.L))), self.logit(h)   # (..., M, L), (..., M)


class WorldModel(nn.Module):
    def __init__(self, latent=128, hidden=256, num_q=5, bins=101, simnorm=8, dropout=0.01, heads=4):
        super().__init__()
        self.L = latent
        self.enc = SetEncoder(latent, hidden=hidden, simnorm=simnorm)
        self.dyn = Dynamics(latent, hidden, heads, simnorm)
        self.rew = mlp(latent + ACT + GD, [hidden, hidden], bins)
        self.pi = mlp(latent, [hidden, hidden], 2 * ACT)
        self.qs = nn.ModuleList([mlp(latent + ACT + GD, [hidden, hidden], bins, dropout=dropout) for _ in range(num_q)])
        self.coll = mlp(latent + GD, [hidden], len(COLL_K))
        self.phys = mlp(latent, [hidden], PD)
        self.kin = mlp(latent + ACT, [hidden], 3)
        for q in self.qs: nn.init.zeros_(q[-1].weight); nn.init.zeros_(q[-1].bias)   # start every value at 0
        nn.init.zeros_(self.rew[-1].weight); nn.init.zeros_(self.rew[-1].bias)
        nn.init.constant_(self.coll[-1].bias, -4.0)                                # contact is rare
        self.num_q = num_q
    def pick(self, zs, i): return zs.gather(-2, i[..., None, None].expand(*i.shape, 1, self.L)).squeeze(-2)
    def next(self, z, a, g, mode="sample"):
        """one future: the most likely hypothesis ("best"), a sampled one ("sample"), or the given indices"""
        zs, lg = self.dyn(z, a, g)
        i = lg.argmax(-1) if mode == "best" else torch.distributions.Categorical(logits=lg).sample() if mode == "sample" else mode
        return self.pick(zs, i)
    def reward(self, z, a, g): return self.rew(torch.cat([z, a, g], -1))
    def Q(self, z, a, g, qs=None):
        x = torch.cat([z, a, g], -1); return torch.stack([q(x) for q in (qs if qs is not None else self.qs)])
    def collision(self, z, g): return self.coll(torch.cat([z, g], -1))
    def physics(self, z): return self.phys(z)
    def kinematics(self, z, a): return self.kin(torch.cat([z, a], -1))
    def policy(self, z, sample=True):
        mu, log_std = self.pi(z).chunk(2, -1); log_std = -10 + 6 * (torch.tanh(log_std) + 1)   # in [-10, 2]
        eps = torch.randn_like(mu) if sample else torch.zeros_like(mu)
        u = mu + eps * log_std.exp(); a = torch.tanh(u)
        logp = (-0.5 * eps.pow(2) - log_std - 0.5 * math.log(2 * math.pi)).sum(-1, keepdim=True) - torch.log(F.relu(1 - a.pow(2)) + 1e-6).sum(-1, keepdim=True)
        return a, logp, torch.tanh(mu)


# ---------------------------------------------------------------- the agent
class TDMPC2:
    def __init__(self, a, device):
        self.a, self.dev = a, device
        self.m = WorldModel(a.latent, a.hidden, a.num_q, heads=a.heads).to(device)
        self.target_qs = [q for q in torch.nn.ModuleList([__import__("copy").deepcopy(q) for q in self.m.qs]).to(device)]
        for q in self.target_qs:
            for p in q.parameters(): p.requires_grad_(False)
        self.th = TwoHot(device=device)
        enc_params = list(self.m.enc.parameters())
        rest = [p for n, p in self.m.named_parameters() if not n.startswith("enc.") and not n.startswith("pi.")]
        self.opt = torch.optim.Adam([{"params": enc_params, "lr": a.lr * 0.3}, {"params": rest, "lr": a.lr}], lr=a.lr)
        self.pi_opt = torch.optim.Adam(self.m.pi.parameters(), lr=a.lr, eps=1e-5)
        self.scale = 1.0   # running 5-95 percentile range of Q, normalises the policy objective
        self.prev_mean = None
        self.imagined = None   # what the car imagined for environment 0 at the last decision (for the replay page)

    def set_norm(self, obs):
        self.m.enc.mean.copy_(torch.tensor(obs.mean(0), device=self.dev)); self.m.enc.std.copy_(torch.tensor(obs.std(0) + 1e-3, device=self.dev))

    def q_value(self, z, a, g, target=False, avg=True):
        idx = random.sample(range(self.a.num_q), 2)
        qs = [self.target_qs[i] for i in idx] if target else [self.m.qs[i] for i in idx]
        v = self.th.decode(self.m.Q(z, a, g, qs))
        return v.mean(0) if avg else v.min(0).values

    def futures(self, z, a, g):
        """next latents of F parallel futures (dim -2 of z): future 0 follows the most likely hypothesis, the others are sampled"""
        zs, lg = self.m.dyn(z, a, g)
        i = torch.distributions.Categorical(logits=lg).sample(); i[..., 0] = lg[..., 0, :].argmax(-1)
        return self.m.pick(zs, i)

    @torch.no_grad()
    def plan(self, obs, g, explore=True, record=False):
        """risk-aware MPPI in the latent space, batched over environments: obs (B, OBS), genomes g (B, GD) -> actions (B, 2)"""
        a, B = self.a, obs.shape[0]; H, S, E, P, Fn = a.horizon, a.samples, a.elites, a.pi_trajs, a.futures
        z0 = self.m.enc(torch.tensor(obs, device=self.dev)); g0 = torch.tensor(g, device=self.dev)
        # trajectories proposed by the policy prior (in the most likely future)
        pi_act = torch.empty(B, P, H, ACT, device=self.dev); z = z0.unsqueeze(1).expand(B, P, -1); gP = g0.unsqueeze(1).expand(B, P, -1)
        for t in range(H):
            pi_act[:, :, t], _, _ = self.m.policy(z)
            if t < H - 1: z = self.m.next(z, pi_act[:, :, t], gP, "best")
        mean = torch.zeros(B, H, ACT, device=self.dev); std = torch.full((B, H, ACT), a.max_std, device=self.dev)
        if self.prev_mean is not None and self.prev_mean.shape[0] == B: mean[:, :-1] = self.prev_mean[:, 1:]
        zS = z0[:, None, None].expand(B, S, Fn, -1); gS = g0[:, None, None].expand(B, S, Fn, -1)
        nworst = max(1, int(round(a.cvar * Fn)))
        for it in range(a.iters):
            acts = (mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(B, S - P, H, ACT, device=self.dev)).clamp(-1, 1)
            acts = torch.cat([acts, pi_act], 1)
            G, disc, z = torch.zeros(B, S, Fn, 1, device=self.dev), 1.0, zS
            for t in range(H):
                at = acts[:, :, None, t].expand(B, S, Fn, ACT)
                G = G + disc * self.th.decode(self.m.reward(z, at, gS))
                z = self.futures(z, at, gS)
                G = G - disc * a.coll_cost * torch.sigmoid(self.m.collision(z, gS)[..., :1])   # predicted contact within 1 s
                disc *= a.gamma
            api, _, _ = self.m.policy(z); G = G + disc * self.q_value(z, api, gS)
            Gs = G.squeeze(-1); worst = Gs.sort(-1).values[..., :nworst].mean(-1)
            val = (1 - a.risk) * Gs.mean(-1) + a.risk * worst                                    # mean vs the worst futures (CVaR)
            top = val.topk(E, 1).indices
            ev = val.gather(1, top); ea = acts.gather(1, top[..., None, None].expand(-1, -1, H, ACT))
            w = torch.exp(a.temp * (ev - ev.max(1, keepdim=True).values)); w = w / w.sum(1, keepdim=True)
            mean = (w[..., None, None] * ea).sum(1)
            std = (w[..., None, None] * (ea - mean.unsqueeze(1)) ** 2).sum(1).sqrt().clamp(a.min_std, a.max_std)
        idx = torch.multinomial(w, 1).squeeze(1) if explore else w.argmax(1)
        act = ea[torch.arange(B), idx, 0]
        if explore: act = (act + std[:, 0] * torch.randn_like(act)).clamp(-1, 1)
        self.prev_mean = mean
        if record: self.imagined = self.imagine(z0[:1], g0[:1], ea[0, idx[0]])
        return act.cpu().numpy()

    @torch.no_grad()
    def imagine(self, z0, g0, plan):
        """the physical futures the car imagines for one plan: its own path in each future, the predicted contact risk,
        and ghosts of the nearest road users in the most likely future (positions in the car's frame at decision time)"""
        Fn, H = max(1, self.a.futures), plan.shape[0]
        z = z0.expand(Fn, -1).unsqueeze(0); g = g0.expand(Fn, -1).unsqueeze(0)
        x = torch.zeros(Fn, device=self.dev); y = torch.zeros(Fn, device=self.dev); h = torch.zeros(Fn, device=self.dev)
        paths, risk, ghosts, ks = [[] for _ in range(Fn)], 0.0, [], {max(1, H // 2) - 1, H - 1}
        ksc = torch.tensor(KIN_SCALE, device=self.dev)
        for t in range(H):
            at = plan[t].expand(1, Fn, ACT)
            d = self.m.kinematics(z, at)[0] * ksc
            x = x + d[:, 0] * torch.cos(h) - d[:, 1] * torch.sin(h); y = y + d[:, 0] * torch.sin(h) + d[:, 1] * torch.cos(h); h = h + d[:, 2]
            z = self.futures(z, at, g)
            risk = max(risk, float(torch.sigmoid(self.m.collision(z, g)[0, :, 1]).max()))   # contact within 2 s, any future
            for f in range(Fn): paths[f].append([round(float(x[f]), 2), round(float(y[f]), 2)])
            if t in ks:   # ghosts: the most likely future at mid-horizon, every future at the end
                phs = self.m.physics(z[0]).cpu().numpy()
                for f in (range(Fn) if t == H - 1 else [0]):
                    ph, gs = phs[f], []; c, s = math.cos(float(h[f])), math.sin(float(h[f]))
                    for j in range(NPH):
                        fx, fy, ped, pres = float(ph[1 + j * 6]) * RANGE, float(ph[2 + j * 6]) * RANGE, float(ph[5 + j * 6]), float(ph[6 + j * 6])
                        if pres > 0.5: gs.append([round(float(x[f]) + fx * c - fy * s, 2), round(float(y[f]) + fx * s + fy * c, 2), int(ped > 0.5)])
                    ghosts.append([t + 1, gs, f])
        return {"paths": paths, "risk": round(risk, 3), "ghosts": ghosts}

    @torch.no_grad()
    def dream(self, gs, obs, horizon, chunk=32):
        """score scenario genomes in imagination: from real situations, how much return would the car get (policy prior,
        sampled futures) and how likely is contact, if the traffic followed genome g. gs (C, GD), obs (N, OBS)"""
        z0 = self.m.enc(torch.tensor(obs, device=self.dev)); N = z0.shape[0]; vals, cols = [], []
        for c0 in range(0, len(gs), chunk):
            g = torch.tensor(gs[c0:c0 + chunk], device=self.dev); C = g.shape[0]
            z = z0.unsqueeze(0).expand(C, N, -1); g = g.unsqueeze(1).expand(C, N, -1)
            G, disc, pmax = torch.zeros(C, N, 1, device=self.dev), 1.0, torch.zeros(C, N, 1, device=self.dev)
            for t in range(horizon):
                _, _, act = self.m.policy(z)
                G = G + disc * self.th.decode(self.m.reward(z, act, g)); z = self.m.next(z, act, g, "sample")
                pc = torch.sigmoid(self.m.collision(z, g)[..., :1]); G = G - disc * self.a.coll_cost * pc; pmax = torch.maximum(pmax, pc); disc *= self.a.gamma
            _, _, act = self.m.policy(z); G = G + disc * self.q_value(z, act, g)
            vals.append(G.mean((1, 2)).cpu()); cols.append(pmax.mean((1, 2)).cpu())
        return torch.cat(vals).numpy(), torch.cat(cols).numpy()

    def update(self, batch, bc=None, bc_w=0.0):
        a = self.a; o, act, r, term, no, aux, g, clab = batch   # (H, B, ...): obs, action, reward, terminal, next obs, [contact, dx, dy, dh], genome, contact-within labels
        H, B = act.shape[0], act.shape[1]
        with torch.no_grad():
            nz = self.m.enc(no)                                   # latent of every next observation (consistency targets)
            pa, _, _ = self.m.policy(nz)
            td = r + a.gamma * (1 - term) * self.q_value(nz, pa, g, target=True, avg=False)
        self.m.train()
        z = self.m.enc(o[0]); zs = []; cons = 0; mode_l = 0; ar = torch.arange(B, device=self.dev)
        for t in range(H):
            zs.append(z)
            cand, lg = self.m.dyn(z, act[t], g[t])
            err = ((cand - nz[t].unsqueeze(1)) ** 2).mean(-1)     # (B, M): how far each hypothesis is from what happened
            best = err.argmin(-1)
            cons = cons + (a.rho ** t) * (err[ar, best].mean() + a.wta_eps * err.mean())   # the winner learns (others a little)
            mode_l = mode_l + (a.rho ** t) * F.cross_entropy(lg, best)                      # odds: which hypothesis came true
            z = cand[ar, best]
        zs = torch.stack(zs)
        rl = sum((a.rho ** t) * self.th.ce(self.m.reward(zs[t], act[t], g[t]), r[t]).mean() for t in range(H))
        qs = self.m.Q(zs, act, g)                                 # (num_q, H, B, bins)
        vl = sum((a.rho ** t) * self.th.ce(qs[:, t], td[t].unsqueeze(0).expand(a.num_q, -1, -1)).mean() for t in range(H))
        cl = sum((a.rho ** t) * F.binary_cross_entropy_with_logits(self.m.collision(zs[t], g[t]), clab[t]) for t in range(H))
        ksc = torch.tensor(KIN_SCALE, device=self.dev)
        pl_ = sum((a.rho ** t) * F.mse_loss(self.m.physics(zs[t]), o[t][..., PHYS_IDX]) for t in range(H))
        kl = sum((a.rho ** t) * F.mse_loss(self.m.kinematics(zs[t], act[t]), aux[t][..., 1:4] / ksc) for t in range(H))
        loss = (a.cons_coef * cons + a.mode_coef * mode_l + a.rew_coef * rl + a.val_coef * vl + a.coll_coef * cl + a.phys_coef * pl_ + a.kin_coef * kl) / H
        self.opt.zero_grad(); loss.backward(); gn = nn.utils.clip_grad_norm_(self.m.parameters(), a.grad_clip); self.opt.step()
        # policy prior: maximise Q (normalised by its running scale) plus a little entropy; optional cloning of the teacher
        zd = zs.detach(); pa, logp, mu = self.m.policy(zd)
        qv = self.q_value(zd, pa, g)
        with torch.no_grad():
            rng = torch.quantile(qv[0].flatten(), 0.95) - torch.quantile(qv[0].flatten(), 0.05)
            self.scale = 0.99 * self.scale + 0.01 * max(1.0, float(rng))
        w = torch.tensor([a.rho ** t for t in range(H)], device=self.dev).view(H, 1, 1)
        pl = (-(a.ent_coef * -logp + qv / self.scale) * w).mean()
        if bc is not None and bc_w > 0:
            bo, ba = bc; _, _, bmu = self.m.policy(self.m.enc(bo).detach()); pl = pl + bc_w * F.mse_loss(bmu, ba)
        self.pi_opt.zero_grad(); pl.backward(); nn.utils.clip_grad_norm_(self.m.pi.parameters(), a.grad_clip); self.pi_opt.step()
        with torch.no_grad():
            for q, tq in zip(self.m.qs, self.target_qs):
                for p, tp in zip(q.parameters(), tq.parameters()): tp.lerp_(p, a.tau)
        f = lambda x: float(x.detach()) / H
        return {"loss": loss.item(), "cons": f(cons), "mode": f(mode_l), "rew": f(rl), "val": f(vl), "coll": f(cl), "phys": f(pl_), "kin": f(kl), "pi": pl.item(), "grad": float(gn), "qscale": self.scale}

    def save(self, path, steps):
        torch.save({"model": self.m.state_dict(), "target": [q.state_dict() for q in self.target_qs], "scale": self.scale, "steps": steps, "args": vars(self.a)}, path)
    def load(self, path):
        ck = torch.load(path, map_location=self.dev); self.m.load_state_dict(ck["model"])
        for q, s in zip(self.target_qs, ck["target"]): q.load_state_dict(s)
        self.scale = ck.get("scale", 1.0); return ck.get("steps", 0)


# ---------------------------------------------------------------- replay buffer of sequences
class Buffer:
    """per environment streams of (obs, action, reward, terminal, next obs, aux, genome); sequences never cross an episode
    end. Contact-within-k labels look ahead up to 3 s in the same stream."""
    def __init__(self, n_env, cap, device):
        self.n, self.cap, self.dev = n_env, cap, device
        self.o = np.zeros((n_env, cap, OBS), np.float16); self.no = np.zeros((n_env, cap, OBS), np.float16)
        self.a = np.zeros((n_env, cap, ACT), np.float32); self.r = np.zeros((n_env, cap), np.float32)
        self.term = np.zeros((n_env, cap), np.float32); self.done = np.ones((n_env, cap), bool); self.len = 0; self.ptr = 0
        self.aux = np.zeros((n_env, cap, 4), np.float32); self.g = np.zeros((n_env, cap, GD), np.float16)
    def add(self, o, a, r, done, term, no, aux, g):
        i = self.ptr
        self.o[:, i], self.a[:, i], self.r[:, i], self.done[:, i], self.term[:, i], self.no[:, i], self.aux[:, i], self.g[:, i] = o, a, r, done, term, no, aux, g
        self.ptr = (self.ptr + 1) % self.cap; self.len = min(self.cap, self.len + 1)
    def sample(self, B, H):
        L = max(COLL_K); span = H + L
        hi = self.len - span if self.len < self.cap and self.len > span + 1 else max(1, self.len - H)
        e = np.random.randint(0, self.n, B * 3); t = np.random.randint(0, max(1, hi), B * 3)
        if self.len == self.cap: t = (self.ptr + t) % self.cap
        idx = (t[:, None] + np.arange(H)[None]) % self.cap
        ok = ~self.done[e[:, None], idx[:, :-1]].any(1) if H > 1 else np.ones(len(e), bool)
        if self.len == self.cap: ok &= ((idx - self.ptr) % self.cap)[:, -1] >= ((idx - self.ptr) % self.cap)[:, 0]   # not across the write head
        e, t, idx = e[ok][:B], t[ok][:B], idx[ok][:B]
        # contact within k decisions of each step, without crossing the end of its episode
        w = (t[:, None] + np.arange(span)[None]) % self.cap
        c, d = self.aux[e[:, None], w, 0], self.done[e[:, None], w].astype(np.float32)
        lab = np.zeros((len(e), H, len(COLL_K)), np.float32)
        for h in range(H):
            sc, sd = c[:, h:h + L], d[:, h:h + L]
            alive = np.concatenate([np.ones((len(e), 1), np.float32), np.cumprod(1 - sd[:, :-1], 1)], 1)
            hit = sc * alive
            for j, k in enumerate(COLL_K): lab[:, h, j] = hit[:, :k].max(1)
        T = lambda x: torch.tensor(np.ascontiguousarray(np.swapaxes(x, 0, 1)), dtype=torch.float32, device=self.dev)
        ix = (e[:, None], idx)
        return (T(self.o[ix]), T(self.a[ix]), T(self.r[ix])[..., None], T(self.term[ix])[..., None], T(self.no[ix]), T(self.aux[ix]), T(self.g[ix]),
                torch.tensor(np.ascontiguousarray(np.swapaxes(lab, 0, 1)), device=self.dev))
    def states(self, n):
        """random observations, for scoring genomes in imagination"""
        e = np.random.randint(0, self.n, n); t = np.random.randint(0, max(1, self.len), n)
        return self.o[e, t].astype(np.float32)


def teacher_demos(n, workers, out, seed=5, reward="v2", defensive=1):
    """the (defensive) look-ahead teacher drives; every decision is stored with its reward and physics (gen_demos.js)"""
    os.makedirs(out, exist_ok=True); per = max(1, n // workers); ps = []
    for w in range(workers):
        ps.append(subprocess.Popen(["node", os.path.join(HERE, "gen_demos.js"), "--decisions", str(per), "--seed", str(seed * 100 + w), "--reward", reward, "--defensive", str(defensive),
                                    "--out", os.path.join(out, f"demo_w{w}")], stdout=subprocess.DEVNULL))
    for p in ps: p.wait()
    rows = []
    for w in range(workers):
        meta = json.load(open(os.path.join(out, f"demo_w{w}.json"))); d = np.fromfile(os.path.join(out, f"demo_w{w}.bin"), np.float32).reshape(meta["rows"], -1)
        rows.append(d)
    return np.concatenate(rows) if rows else np.zeros((0, 2 * OBS + ACT + 7), np.float32)   # obs, action, reward, done, terminal, next obs, contact, dx, dy, dh


def evaluate(agent, n, workers, device, explore=False):
    """the fixed suite of eval.js (same seeds, densities and maps), driven with planning (the default population)"""
    DENS = [0.5, 0.8, 1.05]
    specs = [{"seed": 90000 + i, "density": DENS[i % 3]} for i in range(n)]
    eps = []
    for s0 in range(0, n, max(1, workers) * 4):
        chunk = specs[s0:s0 + max(1, workers) * 4]
        env = VecEnv(1, len(chunk), extra={"specs": chunk})
        obs = env.reset(); alive = np.ones(len(chunk), bool); agent.prev_mean = None; res = [None] * len(chunk)
        g = np.tile(G_DEFAULT, (len(chunk), 1))
        while alive.any():
            act = agent.plan(obs, g, explore)
            obs, rew, done, infos = env.step(act)
            for i, d in enumerate(done):
                if d and alive[i]: alive[i] = False; res[i] = infos[i]
        env.close(); eps += [dict(r, i=s0 + i) for i, r in enumerate(res)]
    m = lambda k: sum(e["result"] == k for e in eps) / len(eps)
    return {"episodes": len(eps), "ok": m("ok"), "crash": m("crash"), "red": sum(bool(e.get("red")) for e in eps) / len(eps), "timeout": m("timeout"), "hit": m("hit")}, eps


def export_policy(agent, path, name, steps):
    """the policy prior (encoder + policy head) as JSON for the browser: it drives without planning"""
    b64 = lambda t: base64.b64encode(np.ascontiguousarray(t.detach().cpu().numpy(), np.float32).tobytes()).decode()
    def seq(s):
        out, mods = [], list(s)
        for i, m in enumerate(mods):
            if isinstance(m, nn.Linear):
                L = {"W": b64(m.weight), "b": b64(m.bias), "inp": m.in_features, "out": m.out_features, "act": "none"}
                if i + 1 < len(mods) and isinstance(mods[i + 1], nn.LayerNorm): L["ln"] = [b64(mods[i + 1].weight), b64(mods[i + 1].bias)]
                nxt = mods[i + 2] if i + 2 < len(mods) else None
                L["act"] = "mish" if isinstance(nxt, nn.Mish) else "simnorm" if isinstance(nxt, SimNorm) else "relu" if i + 1 < len(mods) and isinstance(mods[i + 1], nn.ReLU) else "none"
                out.append(L)
        return out
    e = agent.m.enc
    json.dump({"type": "tdmpc", "name": name, "steps": steps, "egoDim": HEAD, "K": K, "nbDim": NB, "obsMean": b64(e.mean), "obsStd": b64(e.std),
               "phi": seq(e.phi), "head": seq(e.head), "pi": seq(agent.m.pi), "simnorm": 8}, open(path + ".tmp", "w"))
    os.replace(path + ".tmp", path)   # atomic: the monitor never reads a half-written file


def save_episode(out, i, steps, info, rp, keep):
    """one training episode (seed + the actions TD-MPC2 chose + what it imagined), for an exact replay on /sim"""
    d = os.path.join(out, "episodes"); os.makedirs(d, exist_ok=True)
    meta = {"id": i, "steps": steps, "result": info.get("result"), "time": info.get("time"), "legs": info.get("legs"), "red": bool(info.get("red")), "evolved": bool(rp.get("opts", {}).get("genome"))}
    f = os.path.join(d, f"ep_{i:06d}.json"); json.dump(dict(meta, replay=rp), open(f + ".tmp", "w")); os.replace(f + ".tmp", f)
    idx = os.path.join(d, "index.json")
    try: L = json.load(open(idx))
    except (OSError, ValueError): L = []
    L.append(meta)
    for old in L[:-keep]:
        try: os.remove(os.path.join(d, f"ep_{old['id']:06d}.json"))
        except OSError: pass
    json.dump(L[-keep:], open(idx + ".tmp", "w")); os.replace(idx + ".tmp", idx)


# ---------------------------------------------------------------- evolution inside the world model
class Evolution:
    """MAP-Elites over scenario genomes, scored first in the world model's imagination, then checked in the simulation"""
    def __init__(self, a, out):
        self.a, self.dir = a, os.path.join(out, "evo"); os.makedirs(self.dir, exist_ok=True)
        self.rng = random.Random(a.seed * 7 + 99); self.archive = CO.Archive(); self.gen = 0; self.log = []
        pf = a.priors if os.path.isabs(a.priors) else os.path.join(ROOT, a.priors)
        self.priors = json.load(open(pf)) if os.path.exists(pf) else {}

    def generation(self, agent, buf, env, steps):
        a, t0 = self.a, time.time(); self.gen += 1
        el = self.archive.elites()
        cands = [CO.mutate(self.rng.choice(el)["genome"], self.rng) if el and self.rng.random() < 0.8 else CO.random_genome(self.rng) for _ in range(a.evo_cands)]
        # 1. imagination: thousands of rollouts in the world model, per genome
        vals, cols = agent.dream(np.stack([gnorm(g) for g in cands]), buf.states(a.evo_states), a.dream_h)
        base, _ = agent.dream(G_DEFAULT[None], buf.states(a.evo_states), a.dream_h)
        dd = (base[0] - vals) + a.coll_cost * cols                       # how much worse than ordinary traffic, in imagination
        order = np.argsort(-dd)
        # 2. the most promising of every archive cell, then the best overall, are checked in the simulation
        pick, seen = [], set()
        for i in order:
            c = CO.descriptor(cands[i])
            if c not in seen: seen.add(c); pick.append(int(i))
            if len(pick) >= a.evo_real: break
        for i in order:
            if len(pick) >= a.evo_real: break
            if int(i) not in pick: pick.append(int(i))
        driver = os.path.join(self.dir, "driver.json"); export_policy(agent, driver, "TD-MPC2 prior", steps)
        genomes = [cands[i] for i in pick] + [r["genome"] for r in el]   # the archive is re-scored against the current car
        res = CO.score(CO.evaluate(genomes, driver, a.workers, a.evo_episodes, self.dir, f"g{self.gen}", 700000 + self.gen * 1000), self.priors, a.realism)
        new, old = res[:len(pick)], res[len(pick):]
        self.archive = CO.Archive()
        for r in old + new:
            if r["difficulty"] >= a.evo_min: self.archive.offer(r)          # scenarios the car has mastered drop out
        pool = [r["genome"] for r in self.archive.elites()]
        if pool: env.set_pool(pool, a.pool_p)
        real = np.array([r["difficulty"] for r in new]); imag = dd[pick]
        corr = float(np.corrcoef(real, imag)[0, 1]) if len(new) > 2 and real.std() > 1e-6 and imag.std() > 1e-6 else None
        row = {"gen": self.gen, "steps": steps, "seconds": round(time.time() - t0, 1), "cells": len(self.archive.cells), "qd": round(self.archive.qd(), 3),
               "feasible": sum(r["feasible"] for r in new), "checked": len(new), "imagined": len(cands), "dream_vs_real": None if corr is None else round(corr, 3),
               "best": max(new, key=lambda r: r["difficulty"] * r["feasible"])["difficulty"] if new else 0, "pool": len(pool)}
        self.log.append(row); print(json.dumps({"evo": row}), flush=True)
        json.dump(self.log, open(os.path.join(self.dir, "log.json"), "w"), indent=1)
        json.dump([{k: r[k] for k in ("genome", "cell", "difficulty", "realism", "crash", "ok")} for r in self.archive.elites()], open(os.path.join(self.dir, "archive.json"), "w"), indent=1)
        return row


# ---------------------------------------------------------------- training loop
def train(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    dev = torch.device(a.device); os.makedirs(a.out, exist_ok=True)
    agent = TDMPC2(a, dev)
    extra = {"genomeSpace": GSPACE, "reward": a.reward, "reseedEvery": a.reseed}
    env = VecEnv(a.workers, a.envs, a.seed, density_range=[0.4, 1.1], extra=extra, extra0={"showcase": 1} if a.replays else None)
    n_ep, imag = 0, []
    N = env.n; buf = Buffer(N, max(1000, a.buffer // N), dev)
    evo = Evolution(a, a.out) if a.evo_every else None
    demo = None
    if a.demos > 0:
        note(a.out, phase="demos", step=0)
        d = teacher_demos(a.demos, a.workers, os.path.join(a.out, "demos"), reward=a.reward, defensive=a.defensive)
        demo = d; print(json.dumps({"demos": int(len(d))}), flush=True)
    obs = env.reset(); gcur = env.gen
    agent.set_norm(np.concatenate([obs] + ([demo[:, :OBS]] if demo is not None and len(demo) else [])))
    dbuf = None
    if demo is not None and len(demo):   # a buffer of the teacher's drives, mixed into every update
        n1 = len(demo) // a.workers
        dbuf = Buffer(a.workers, max(2, n1), dev)
        for w in range(a.workers):
            for t in range(n1):
                row = demo[w * n1 + t]
                dbuf.o[w, t], dbuf.a[w, t], dbuf.r[w, t], dbuf.done[w, t], dbuf.term[w, t] = row[:OBS], row[OBS:OBS + 2], row[OBS + 2], row[OBS + 3] > 0.5, row[OBS + 4]
                dbuf.no[w, t], dbuf.aux[w, t], dbuf.g[w, t] = row[OBS + 5:2 * OBS + 5], row[2 * OBS + 5:2 * OBS + 9], G_DEFAULT
        dbuf.len = n1
    hist, log, steps, upd, t0, last_eval, stats = deque(maxlen=200), [], 0, 0, time.time(), 0, {}
    while steps < a.steps:
        rec = bool(a.replays)
        act = np.random.uniform(-1, 1, (N, ACT)).astype(np.float32) if steps < a.seed_steps and demo is None else agent.plan(obs, gcur, record=rec)
        if rec: imag.append(agent.imagined if not (steps < a.seed_steps and demo is None) else None)
        nobs, rew, done, infos = env.step(act)
        term = np.array([bool(d) and (inf or {}).get("result") not in ("timeout",) for d, inf in zip(done, infos)], np.float32)
        nxt = nobs.copy()
        for i, inf in enumerate(infos):
            rp = inf.pop("replay", None) if inf else None
            if rp and rp.get("centre"): rp["imagined"] = imag[-len(rp["actions"]):] if len(imag) >= len(rp["actions"]) else None; save_episode(a.out, n_ep, steps, inf, rp, a.replays); n_ep += 1
            if rp is not None: imag = []
            if inf: hist.append(inf); nxt[i] = np.array(inf.get("terminal", nobs[i]), np.float32)
        buf.add(obs, act, rew, done, term, nxt, env.aux, gcur); obs = nobs; gcur = env.gen; steps += N
        if any(done): agent.prev_mean = None if all(done) else agent.prev_mean
        if steps >= a.seed_steps or demo is not None:
            for _ in range(max(1, int(N * a.utd))):
                H1 = a.horizon
                if dbuf is not None and random.random() < a.demo_frac * max(0.2, 1 - steps / max(1, a.steps)):
                    batch = dbuf.sample(a.batch, H1)
                else: batch = buf.sample(a.batch, H1) if buf.len > H1 + 2 else dbuf.sample(a.batch, H1)
                if batch[0].shape[1] < 8: continue
                bc = None; bcw = a.bc * max(0.0, 1 - steps / max(1, a.bc_steps))
                if dbuf is not None and bcw > 0:
                    b2 = dbuf.sample(a.batch, 1); bc = (b2[0][0], b2[1][0])
                stats = agent.update(batch, bc, bcw); upd += 1
        if evo and steps >= a.evo_start and steps // a.evo_every != (steps - N) // a.evo_every and buf.len > 100:
            note(a.out, phase="evolve", total=a.steps, steps=steps); evo.generation(agent, buf, env, steps); agent.prev_mean = None
        if steps // a.log_every != (steps - N) // a.log_every:
            m = lambda k: sum(e["result"] == k for e in hist) / max(1, len(hist))
            row = {"steps": steps, "minutes": round((time.time() - t0) / 60, 1), "ok": round(m("ok"), 3), "crash": round(m("crash"), 3), "hit": round(m("hit"), 3), "timeout": round(m("timeout"), 3),
                   "updates": upd, **{k: round(v, 4) for k, v in stats.items()}, **({"evo_cells": len(evo.archive.cells)} if evo else {})}
            log.append(row); print(json.dumps(row), flush=True); json.dump(log, open(os.path.join(a.out, "log.json"), "w"), indent=1)
            note(a.out, phase="train", total=a.steps, **row)
        if a.export_every and upd and steps // a.export_every != (steps - N) // a.export_every:   # a snapshot for the live simulator page (no evaluation)
            export_policy(agent, os.path.join(a.out, "policy.json"), f"TD-MPC2 prior {steps // 1000}k", steps)
        if a.eval_every and steps - last_eval >= a.eval_every:
            last_eval = steps; agent.save(os.path.join(a.out, f"ckpt_{steps}.pt"), steps); export_policy(agent, os.path.join(a.out, "policy.json"), f"TD-MPC2 prior {steps // 1000}k", steps)
            ev, _ = evaluate(agent, a.eval_n, a.workers, dev); ev["steps"] = steps; print(json.dumps({"eval": ev}), flush=True)
            json.dump(ev, open(os.path.join(a.out, f"eval_{steps}.json"), "w")); agent.prev_mean = None
    env.close()
    agent.save(os.path.join(a.out, "final.pt"), steps); export_policy(agent, os.path.join(a.out, "policy.json"), "TD-MPC2 prior", steps)
    ev, eps = evaluate(agent, a.eval_n, a.workers, dev); ev["steps"] = steps
    json.dump({"summary": ev, "episodes": eps}, open(os.path.join(a.out, "eval_final.json"), "w")); print(json.dumps({"final": ev}), flush=True)
    note(a.out, phase="done", **ev)
    return ev


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1_000_000); p.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 1)))
    p.add_argument("--replays", type=int, default=40, help="training episodes kept for the replay on the monitor page (0 = none)")
    p.add_argument("--export-every", type=int, default=2000, help="write policy.json for the live simulator page this often (0 = only at evaluations)")
    p.add_argument("--envs", type=int, default=4); p.add_argument("--seed", type=int, default=1)
    p.add_argument("--reward", default="v2", help="v2: heavier crashes (people most), being hit, defensive driving; v1: the reward of phases 2-3")
    p.add_argument("--defensive", type=int, default=1, help="demonstrations by the defensive teacher")
    p.add_argument("--reseed", type=int, default=600, help="decisions between fresh maps per environment (evolved scenarios enter with new maps)")
    p.add_argument("--demos", type=int, default=40000, help="teacher decisions to start the buffer with (0 = learn from scratch)")
    p.add_argument("--demo-frac", type=float, default=0.5); p.add_argument("--bc", type=float, default=1.0, help="weight of cloning the teacher in the policy loss (fades out)")
    p.add_argument("--bc-steps", type=int, default=300_000)
    p.add_argument("--seed-steps", type=int, default=5000); p.add_argument("--buffer", type=int, default=1_000_000); p.add_argument("--batch", type=int, default=256)
    p.add_argument("--utd", type=float, default=0.5, help="gradient updates per environment step")
    p.add_argument("--latent", type=int, default=128); p.add_argument("--hidden", type=int, default=256); p.add_argument("--num-q", type=int, default=5)
    p.add_argument("--heads", type=int, default=4, help="hypotheses of the dynamics (different futures)")
    p.add_argument("--horizon", type=int, default=6, help="planning horizon in decisions of 0.2 s"); p.add_argument("--iters", type=int, default=5); p.add_argument("--samples", type=int, default=256)
    p.add_argument("--futures", type=int, default=3, help="sampled futures per candidate plan"); p.add_argument("--risk", type=float, default=0.5, help="weight of the worst futures vs the mean")
    p.add_argument("--cvar", type=float, default=0.34, help="share of futures that count as the worst"); p.add_argument("--coll-cost", type=float, default=8.0, help="planning cost of predicted contact")
    p.add_argument("--elites", type=int, default=32); p.add_argument("--pi-trajs", type=int, default=16); p.add_argument("--temp", type=float, default=0.5)
    p.add_argument("--min-std", type=float, default=0.05); p.add_argument("--max-std", type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=0.99); p.add_argument("--rho", type=float, default=0.5); p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--grad-clip", type=float, default=20.0)
    p.add_argument("--cons-coef", type=float, default=20.0); p.add_argument("--rew-coef", type=float, default=0.1); p.add_argument("--val-coef", type=float, default=0.1)
    p.add_argument("--mode-coef", type=float, default=0.1); p.add_argument("--wta-eps", type=float, default=0.05)
    p.add_argument("--coll-coef", type=float, default=0.5); p.add_argument("--phys-coef", type=float, default=2.0); p.add_argument("--kin-coef", type=float, default=1.0)
    p.add_argument("--ent-coef", type=float, default=1e-4)
    # evolution inside the world model
    p.add_argument("--evo-every", type=int, default=50_000, help="steps between generations of scenario evolution (0 = off)"); p.add_argument("--evo-start", type=int, default=100_000)
    p.add_argument("--evo-cands", type=int, default=512, help="genomes imagined per generation"); p.add_argument("--evo-states", type=int, default=256, help="real situations each genome is imagined from")
    p.add_argument("--evo-real", type=int, default=12, help="genomes checked in the simulation per generation"); p.add_argument("--evo-episodes", type=int, default=2)
    p.add_argument("--evo-min", type=float, default=0.15, help="scenarios easier than this for the current car leave the archive")
    p.add_argument("--dream-h", type=int, default=10, help="imagined decisions per genome rollout"); p.add_argument("--pool-p", type=float, default=0.4, help="share of new maps that use an evolved scenario")
    p.add_argument("--realism", type=float, default=0.5); p.add_argument("--priors", default="calib/priors_vn.json")
    p.add_argument("--log-every", type=int, default=5000); p.add_argument("--eval-every", type=int, default=100_000); p.add_argument("--eval-n", type=int, default=48)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=os.path.join(HERE, "runs", "tdmpc2")); p.add_argument("--eval", default=None, help="evaluate a checkpoint with planning")
    a = p.parse_args()
    a.elites = min(a.elites, a.samples // 2); a.pi_trajs = min(a.pi_trajs, a.samples // 4)
    return a


if __name__ == "__main__":
    a = parse()
    if a.eval:
        dev = torch.device(a.device); ag = TDMPC2(a, dev); ag.load(a.eval)
        ev, eps = evaluate(ag, a.eval_n, a.workers, dev); print(json.dumps(ev))
        json.dump({"summary": ev, "episodes": eps}, open(a.eval + ".eval.json", "w"))
    else: train(a)
