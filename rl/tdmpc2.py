"""
LOTUS, learned world model: TD-MPC2 (Hansen, Su & Wang, ICLR 2024) for the self-driving car.

The agent learns a model of the world in a latent space: an encoder (DeepSets over the 16 nearest road users, the same
observation the other drivers get), a latent dynamics model "what happens 0.2 s after action a", a reward model, an
ensemble of Q functions and a policy prior. It drives by planning: every decision it imagines 512 short futures in the
latent space (MPPI over 3 decisions, refined 6 times, then a learned value for what comes after) and takes the best first
action. Optionally the replay buffer starts with drives by the look-ahead teacher, which shortens learning a lot.

  python tdmpc2.py --steps 1000000 --workers 16 --demos 40000          # full run (GPU recommended)
  python tdmpc2.py --steps 3000 --workers 2 --envs 2 --demos 400 --samples 64 --eval-n 4   # smoke test
  python tdmpc2.py --eval runs/tdmpc2/final.pt --eval-n 24              # evaluate with planning

Outputs in --out: checkpoints (.pt), policy.json (the policy prior alone, runs in the browser without planning),
log.json, eval_*.json.
"""
import argparse, base64, json, math, os, random, subprocess, time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from train import VecEnv
from distill import HEAD, K, NB, OBS, note, HERE

ACT = 2


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


class WorldModel(nn.Module):
    def __init__(self, latent=128, hidden=256, num_q=5, bins=101, simnorm=8, dropout=0.01):
        super().__init__()
        self.enc = SetEncoder(latent, hidden=hidden, simnorm=simnorm)
        self.dyn = mlp(latent + ACT, [hidden, hidden], latent, SimNorm(simnorm))
        self.rew = mlp(latent + ACT, [hidden, hidden], bins)
        self.pi = mlp(latent, [hidden, hidden], 2 * ACT)
        self.qs = nn.ModuleList([mlp(latent + ACT, [hidden, hidden], bins, dropout=dropout) for _ in range(num_q)])
        for q in self.qs: nn.init.zeros_(q[-1].weight); nn.init.zeros_(q[-1].bias)   # start every value at 0
        nn.init.zeros_(self.rew[-1].weight); nn.init.zeros_(self.rew[-1].bias)
        self.num_q = num_q
    def next(self, z, a): return self.dyn(torch.cat([z, a], -1))
    def reward(self, z, a): return self.rew(torch.cat([z, a], -1))
    def Q(self, z, a, qs=None):
        za = torch.cat([z, a], -1); return torch.stack([q(za) for q in (qs if qs is not None else self.qs)])
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
        self.m = WorldModel(a.latent, a.hidden, a.num_q).to(device)
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

    def set_norm(self, obs):
        self.m.enc.mean.copy_(torch.tensor(obs.mean(0), device=self.dev)); self.m.enc.std.copy_(torch.tensor(obs.std(0) + 1e-3, device=self.dev))

    def q_value(self, z, a, target=False, avg=True):
        idx = random.sample(range(self.a.num_q), 2)
        qs = [self.target_qs[i] for i in idx] if target else [self.m.qs[i] for i in idx]
        v = self.th.decode(self.m.Q(z, a, qs))
        return v.mean(0) if avg else v.min(0).values

    @torch.no_grad()
    def plan(self, obs, explore=True):
        """MPPI in the latent space, batched over environments: obs (B, OBS) -> actions (B, 2)"""
        a, B, H, S, E, P = self.a, obs.shape[0], self.a.horizon, self.a.samples, self.a.elites, self.a.pi_trajs
        z0 = self.m.enc(torch.tensor(obs, device=self.dev))
        # trajectories proposed by the policy prior
        pi_act = torch.empty(B, P, H, ACT, device=self.dev); z = z0.unsqueeze(1).expand(B, P, -1)
        for t in range(H):
            pi_act[:, :, t], _, _ = self.m.policy(z); z = self.m.next(z, pi_act[:, :, t]) if t < H - 1 else z
        mean = torch.zeros(B, H, ACT, device=self.dev); std = torch.full((B, H, ACT), a.max_std, device=self.dev)
        if self.prev_mean is not None and self.prev_mean.shape[0] == B: mean[:, :-1] = self.prev_mean[:, 1:]
        zS = z0.unsqueeze(1).expand(B, S, -1)
        for it in range(a.iters):
            acts = (mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(B, S - P, H, ACT, device=self.dev)).clamp(-1, 1)
            acts = torch.cat([acts, pi_act], 1)
            G, disc, z = torch.zeros(B, S, 1, device=self.dev), 1.0, zS
            for t in range(H):
                G = G + disc * self.th.decode(self.m.reward(z, acts[:, :, t])); z = self.m.next(z, acts[:, :, t]); disc *= a.gamma
            api, _, _ = self.m.policy(z); G = G + disc * self.q_value(z, api)
            val = G.squeeze(-1); top = val.topk(E, 1).indices
            ev = val.gather(1, top); ea = acts.gather(1, top[..., None, None].expand(-1, -1, H, ACT))
            w = torch.exp(a.temp * (ev - ev.max(1, keepdim=True).values)); w = w / w.sum(1, keepdim=True)
            mean = (w[..., None, None] * ea).sum(1)
            std = (w[..., None, None] * (ea - mean.unsqueeze(1)) ** 2).sum(1).sqrt().clamp(a.min_std, a.max_std)
        idx = torch.multinomial(w, 1).squeeze(1) if explore else w.argmax(1)
        act = ea[torch.arange(B), idx, 0]
        if explore: act = (act + std[:, 0] * torch.randn_like(act)).clamp(-1, 1)
        self.prev_mean = mean
        return act.cpu().numpy()

    def update(self, batch, bc=None, bc_w=0.0):
        a = self.a; o, act, r, term, no = batch   # o (H, B, OBS), act (H, B, 2), r (H, B, 1), term (H, B, 1), no (H, B, OBS)
        H = act.shape[0]
        with torch.no_grad():
            nz = self.m.enc(no)                                   # latent of every next observation (consistency targets)
            pa, _, _ = self.m.policy(nz)
            td = r + a.gamma * (1 - term) * self.q_value(nz, pa, target=True, avg=False)
        self.m.train()
        z = self.m.enc(o[0]); zs = []; cons = 0
        for t in range(H):
            zs.append(z); z = self.m.next(z, act[t]); cons = cons + (a.rho ** t) * F.mse_loss(z, nz[t])
        zs = torch.stack(zs)
        rl = sum((a.rho ** t) * self.th.ce(self.m.reward(zs[t], act[t]), r[t]).mean() for t in range(H))
        qs = self.m.Q(zs, act)                                    # (num_q, H, B, bins)
        vl = sum((a.rho ** t) * self.th.ce(qs[:, t], td[t].unsqueeze(0).expand(a.num_q, -1, -1)).mean() for t in range(H))
        loss = (a.cons_coef * cons + a.rew_coef * rl + a.val_coef * vl) / H
        self.opt.zero_grad(); loss.backward(); gn = nn.utils.clip_grad_norm_(self.m.parameters(), a.grad_clip); self.opt.step()
        # policy prior: maximise Q (normalised by its running scale) plus a little entropy; optional cloning of the teacher
        zd = zs.detach(); pa, logp, mu = self.m.policy(zd)
        qv = self.q_value(zd, pa)
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
        return {"loss": loss.item(), "cons": float(cons.detach()), "rew": float(rl.detach()), "val": float(vl.detach()), "pi": pl.item(), "grad": float(gn), "qscale": self.scale}

    def save(self, path, steps):
        torch.save({"model": self.m.state_dict(), "target": [q.state_dict() for q in self.target_qs], "scale": self.scale, "steps": steps, "args": vars(self.a)}, path)
    def load(self, path):
        ck = torch.load(path, map_location=self.dev); self.m.load_state_dict(ck["model"])
        for q, s in zip(self.target_qs, ck["target"]): q.load_state_dict(s)
        self.scale = ck.get("scale", 1.0); return ck.get("steps", 0)


# ---------------------------------------------------------------- replay buffer of sequences
class Buffer:
    """per environment streams of (obs, action, reward, terminal, next obs); sequences never cross an episode end"""
    def __init__(self, n_env, cap, device):
        self.n, self.cap, self.dev = n_env, cap, device
        self.o = np.zeros((n_env, cap, OBS), np.float16); self.no = np.zeros((n_env, cap, OBS), np.float16)
        self.a = np.zeros((n_env, cap, ACT), np.float32); self.r = np.zeros((n_env, cap), np.float32)
        self.term = np.zeros((n_env, cap), np.float32); self.done = np.ones((n_env, cap), bool); self.len = 0; self.ptr = 0
    def add(self, o, a, r, done, term, no):
        i = self.ptr
        self.o[:, i], self.a[:, i], self.r[:, i], self.done[:, i], self.term[:, i], self.no[:, i] = o, a, r, done, term, no
        self.ptr = (self.ptr + 1) % self.cap; self.len = min(self.cap, self.len + 1)
    def sample(self, B, H):
        e = np.random.randint(0, self.n, B * 3); t = np.random.randint(0, max(1, self.len - H), B * 3)
        if self.len == self.cap: t = (self.ptr + t) % self.cap
        idx = (t[:, None] + np.arange(H)[None]) % self.cap
        ok = ~self.done[e[:, None], idx[:, :-1]].any(1) if H > 1 else np.ones(len(e), bool)
        if self.len == self.cap: ok &= ((idx - self.ptr) % self.cap)[:, -1] >= ((idx - self.ptr) % self.cap)[:, 0]   # not across the write head
        e, idx = e[ok][:B], idx[ok][:B]
        T = lambda x: torch.tensor(np.ascontiguousarray(np.swapaxes(x, 0, 1)), dtype=torch.float32, device=self.dev)
        return T(self.o[e[:, None], idx]), T(self.a[e[:, None], idx]), T(self.r[e[:, None], idx])[..., None], T(self.term[e[:, None], idx])[..., None], T(self.no[e[:, None], idx])


def teacher_demos(n, workers, out, seed=5):
    """the look-ahead teacher drives; every decision is stored with its reward (gen_demos.js)"""
    os.makedirs(out, exist_ok=True); per = max(1, n // workers); ps = []
    for w in range(workers):
        ps.append(subprocess.Popen(["node", os.path.join(HERE, "gen_demos.js"), "--decisions", str(per), "--seed", str(seed * 100 + w), "--out", os.path.join(out, f"demo_w{w}")], stdout=subprocess.DEVNULL))
    for p in ps: p.wait()
    rows = []
    for w in range(workers):
        meta = json.load(open(os.path.join(out, f"demo_w{w}.json"))); d = np.fromfile(os.path.join(out, f"demo_w{w}.bin"), np.float32).reshape(meta["rows"], -1)
        rows.append(d)
    return np.concatenate(rows) if rows else np.zeros((0, 2 * OBS + ACT + 3), np.float32)   # obs, action, reward, done, terminal, next obs


def evaluate(agent, n, workers, device, explore=False):
    """the fixed suite of eval.js (same seeds, densities and maps), driven with planning"""
    DENS = [0.5, 0.8, 1.05]
    specs = [{"seed": 90000 + i, "density": DENS[i % 3]} for i in range(n)]
    eps = []
    for s0 in range(0, n, max(1, workers) * 4):
        chunk = specs[s0:s0 + max(1, workers) * 4]
        env = VecEnv(1, len(chunk), extra={"specs": chunk})
        obs = env.reset(); alive = np.ones(len(chunk), bool); agent.prev_mean = None; res = [None] * len(chunk)
        while alive.any():
            act = agent.plan(obs, explore)
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


# ---------------------------------------------------------------- training loop
def train(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    dev = torch.device(a.device); os.makedirs(a.out, exist_ok=True)
    agent = TDMPC2(a, dev)
    env = VecEnv(a.workers, a.envs, a.seed, density_range=[0.4, 1.1])
    N = env.n; buf = Buffer(N, max(1000, a.buffer // N), dev)
    demo = None
    if a.demos > 0:
        note(a.out, phase="demos", step=0)
        d = teacher_demos(a.demos, a.workers, os.path.join(a.out, "demos"))
        demo = d; print(json.dumps({"demos": int(len(d))}), flush=True)
    obs = env.reset()
    agent.set_norm(np.concatenate([obs] + ([demo[:, :OBS]] if demo is not None and len(demo) else [])))
    dbuf = None
    if demo is not None and len(demo):   # a buffer of the teacher's drives, mixed into every update
        n1 = len(demo) // a.workers
        dbuf = Buffer(a.workers, max(2, n1), dev)
        for w in range(a.workers):
            for t in range(n1):
                row = demo[w * n1 + t]; o, act, r, done, term, no = row[:OBS], row[OBS:OBS + 2], row[OBS + 2], row[OBS + 3] > 0.5, row[OBS + 4], row[OBS + 5:]
                dbuf.o[w, t], dbuf.a[w, t], dbuf.r[w, t], dbuf.done[w, t], dbuf.term[w, t], dbuf.no[w, t] = o, act, r, done, term, no
        dbuf.len = n1
    hist, log, steps, upd, t0, last_eval = deque(maxlen=200), [], 0, 0, time.time(), 0
    while steps < a.steps:
        act = np.random.uniform(-1, 1, (N, ACT)).astype(np.float32) if steps < a.seed_steps and demo is None else agent.plan(obs)
        nobs, rew, done, infos = env.step(act)
        term = np.array([bool(d) and (inf or {}).get("result") not in ("timeout",) for d, inf in zip(done, infos)], np.float32)
        nxt = nobs.copy()
        for i, inf in enumerate(infos):
            if inf: hist.append(inf); nxt[i] = np.array(inf.get("terminal", nobs[i]), np.float32)
        buf.add(obs, act, rew, done, term, nxt); obs = nobs; steps += N
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
        if steps // a.log_every != (steps - N) // a.log_every:
            m = lambda k: sum(e["result"] == k for e in hist) / max(1, len(hist))
            row = {"steps": steps, "minutes": round((time.time() - t0) / 60, 1), "ok": round(m("ok"), 3), "crash": round(m("crash"), 3), "timeout": round(m("timeout"), 3), "updates": upd, **({k: round(v, 4) for k, v in stats.items()} if upd else {})}
            log.append(row); print(json.dumps(row), flush=True); json.dump(log, open(os.path.join(a.out, "log.json"), "w"), indent=1)
            note(a.out, phase="train", total=a.steps, **row)
        if a.export_every and upd and steps // a.export_every != (steps - N) // a.export_every:   # a snapshot for the live simulator page (no evaluation)
            export_policy(agent, os.path.join(a.out, "policy.json"), f"TD-MPC2 prior {steps // 1000}k", steps)
        if a.eval_every and steps - last_eval >= a.eval_every:
            last_eval = steps; agent.save(os.path.join(a.out, f"ckpt_{steps}.pt"), steps); export_policy(agent, os.path.join(a.out, "policy.json"), f"TD-MPC2 prior {steps // 1000}k", steps)
            ev, _ = evaluate(agent, a.eval_n, a.workers, dev); ev["steps"] = steps; print(json.dumps({"eval": ev}), flush=True)
            json.dump(ev, open(os.path.join(a.out, f"eval_{steps}.json"), "w"))
    env.close()
    agent.save(os.path.join(a.out, "final.pt"), steps); export_policy(agent, os.path.join(a.out, "policy.json"), "TD-MPC2 prior", steps)
    ev, eps = evaluate(agent, a.eval_n, a.workers, dev); ev["steps"] = steps
    json.dump({"summary": ev, "episodes": eps}, open(os.path.join(a.out, "eval_final.json"), "w")); print(json.dumps({"final": ev}), flush=True)
    note(a.out, phase="done", **ev)
    return ev


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1_000_000); p.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 1)))
    p.add_argument("--export-every", type=int, default=10000, help="write policy.json for the live simulator page this often (0 = only at evaluations)")
    p.add_argument("--envs", type=int, default=4); p.add_argument("--seed", type=int, default=1)
    p.add_argument("--demos", type=int, default=40000, help="teacher decisions to start the buffer with (0 = learn from scratch)")
    p.add_argument("--demo-frac", type=float, default=0.5); p.add_argument("--bc", type=float, default=1.0, help="weight of cloning the teacher in the policy loss (fades out)")
    p.add_argument("--bc-steps", type=int, default=300_000)
    p.add_argument("--seed-steps", type=int, default=5000); p.add_argument("--buffer", type=int, default=1_000_000); p.add_argument("--batch", type=int, default=256)
    p.add_argument("--utd", type=float, default=0.5, help="gradient updates per environment step")
    p.add_argument("--latent", type=int, default=128); p.add_argument("--hidden", type=int, default=256); p.add_argument("--num-q", type=int, default=5)
    p.add_argument("--horizon", type=int, default=3); p.add_argument("--iters", type=int, default=6); p.add_argument("--samples", type=int, default=512)
    p.add_argument("--elites", type=int, default=64); p.add_argument("--pi-trajs", type=int, default=24); p.add_argument("--temp", type=float, default=0.5)
    p.add_argument("--min-std", type=float, default=0.05); p.add_argument("--max-std", type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=0.97); p.add_argument("--rho", type=float, default=0.5); p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--grad-clip", type=float, default=20.0)
    p.add_argument("--cons-coef", type=float, default=20.0); p.add_argument("--rew-coef", type=float, default=0.1); p.add_argument("--val-coef", type=float, default=0.1)
    p.add_argument("--ent-coef", type=float, default=1e-4)
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
