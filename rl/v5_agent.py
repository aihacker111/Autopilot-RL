"""Recurrent TD-MPC controller: learned traffic context, bootstrap dynamics
ensemble, multi-future risk planning and conservative planner distillation.
No true scenario genome is supplied to the controller during inference.
"""
import base64
import copy
import json
from contextlib import contextmanager
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tdmpc2 import (WorldModel, SetEncoder, Dynamics, SimNorm, TwoHot, GD,
                    PHYS_IDX, KIN_SCALE, COLL_K, mlp)
from distill import HEAD, K, NB, OBS


@contextmanager
def frozen(modules):
    params = [p for m in modules for p in m.parameters()]
    old = [p.requires_grad for p in params]
    for p in params: p.requires_grad_(False)
    try: yield
    finally:
        for p, flag in zip(params, old): p.requires_grad_(flag)


class BeliefEncoder(nn.Module):
    def __init__(self, latent, hidden, memory=True):
        super().__init__()
        self.base = SetEncoder(latent, hidden=hidden)
        self.gru = nn.GRUCell(latent + 2, latent)
        self.posterior = nn.Sequential(nn.Linear(latent, latent), nn.LayerNorm(latent), SimNorm(8))
        self.context = nn.Sequential(nn.Linear(latent, hidden), nn.Mish(), nn.Linear(hidden, GD), nn.Tanh())
        self.latent = latent; self.memory = memory

    def step(self, obs, previous, h=None):
        if h is None or not self.memory: h = obs.new_zeros(obs.shape[0], self.latent)
        h = self.gru(torch.cat([self.base(obs), previous], -1), h)
        return self.posterior(h), h, self.context(h)

    def forward(self, obs):
        shape = obs.shape[:-1]; flat = obs.reshape(-1, OBS)
        z, _, _ = self.step(flat, flat[:, 1:3])
        return z.reshape(*shape, self.latent)


class Agent:
    def __init__(self, cfg, device):
        self.c, self.device = cfg, torch.device(device)
        L, H = cfg['latent'], cfg['hidden']
        self.model = WorldModel(L, H, cfg['num_q'], heads=cfg['heads']).to(self.device)
        self.model.enc = BeliefEncoder(L, H, cfg['memory']).to(self.device)
        self.extra_dynamics = nn.ModuleList([Dynamics(L, H, cfg['heads'], 8) for _ in range(cfg['ensemble'] - 1)]).to(self.device)
        self.extra_collision = nn.ModuleList([mlp(L + GD, [H], len(COLL_K)) for _ in range(cfg['ensemble'] - 1)]).to(self.device)
        for head in self.extra_collision: nn.init.constant_(head[-1].bias, -4.)
        self.target_q = copy.deepcopy(self.model.qs).eval()
        self.target_q.requires_grad_(False)
        self.model_params = [p for n, p in self.model.named_parameters() if not n.startswith('pi.')] + list(self.extra_dynamics.parameters()) + list(self.extra_collision.parameters())
        self.opt = torch.optim.Adam(self.model_params, lr=cfg['lr'])
        self.pi_opt = torch.optim.Adam(self.model.pi.parameters(), lr=cfg['lr'])
        self.twohot = TwoHot(device=self.device)
        self.h = self.previous = self.mean = None
        self.last_weight = self.last_uncertainty = self.last_plan = None
        self.qscale = 1.
        self.updates = 0; self.steps = 0

    @property
    def dynamics(self): return [self.model.dyn, *self.extra_dynamics]

    def collision_logits(self, z, g):
        x = torch.cat([z, g], -1)
        return torch.stack([self.model.coll(x), *[head(x) for head in self.extra_collision]])

    def q(self, z, a, g, target=False):
        qs = self.target_q if target else self.model.qs
        values = self.twohot.decode(self.model.Q(z, a, g, qs))
        return values.min(0).values if target else values.mean(0)

    def reset(self, mask=None):
        if mask is None: self.h = self.previous = self.mean = None; return
        mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        if self.h is not None: self.h[mask] = 0; self.previous[mask] = 0
        if self.mean is not None: self.mean[mask] = 0

    def observe_execution(self, executed, done):
        self.previous = torch.as_tensor(executed.copy(), device=self.device)
        self.reset(done)

    @torch.no_grad()
    def act(self, obs, planner=True, explore=False):
        self.model.eval(); self.extra_dynamics.eval(); self.extra_collision.eval()
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device); B = len(o)
        if self.h is None or len(self.h) != B:
            self.h = o.new_zeros(B, self.c['latent']); self.previous = o.new_zeros(B, 2); self.mean = None
        z, self.h, context = self.model.enc.step(o, self.previous, self.h)
        _, _, prior = self.model.policy(z, sample=False)
        if not planner:
            self.last_plan = prior.cpu().numpy(); self.last_weight = np.zeros(B, np.float32); self.last_uncertainty = np.zeros(B, np.float32)
            return prior.cpu().numpy()
        c = self.c; H, S, E, Fcount = c['horizon'], c['samples'], min(c['elites'], c['samples']), c['futures']
        mean = o.new_zeros(B, H, 2) if self.mean is None else torch.cat([self.mean[:, 1:], self.mean[:, -1:]], 1)
        std = torch.ones_like(mean)
        initial_uncertainty = self.collision_logits(z, context).sigmoid()[..., 0].std(0, unbiased=False)
        iterations = c['iterations'] + (c['adaptive_iterations'] if initial_uncertainty.max().item() > c['uncertainty_threshold'] else 0)
        g = context[:, None, None].expand(B, S, Fcount, GD)
        prior_plans = max(1, min(c['prior_plans'], S // 4))
        proposals = o.new_empty(B, prior_plans, H, 2)
        zp = z[:, None].expand(B, prior_plans, -1); gp = context[:, None].expand(B, prior_plans, GD)
        for t in range(H):
            aa, _, mu = self.model.policy(zp, sample=explore)
            proposals[:, :, t] = aa if explore else mu
            zp = self.model.next(zp, proposals[:, :, t], gp, 'best')
        # Keep the same ensemble member along each future, rather than switching models every step.
        for _ in range(iterations):
            actions = (mean[:, None] + std[:, None] * torch.randn(B, S, H, 2, device=self.device)).clamp(-1, 1)
            actions[:, -prior_plans:] = proposals
            latent = z[:, None, None].expand(B, S, Fcount, -1)
            returns = o.new_zeros(B, S, Fcount, 1); uncertainty = o.new_zeros(B, S, Fcount, 1); discount = 1.
            for t in range(H):
                a = actions[:, :, None, t].expand(B, S, Fcount, 2)
                returns += discount * self.twohot.decode(self.model.reward(latent, a, g))
                predictions = []; choices = []
                for dyn in self.dynamics:
                    hypotheses, logits = dyn(latent, a, g)
                    index = logits.argmax(-1) if not explore else torch.distributions.Categorical(logits=logits).sample()
                    # Multiple plausible futures are sampled even in evaluation; no action exploration.
                    if Fcount > 1:
                        sampled = torch.distributions.Categorical(logits=logits).sample(); index[..., 1:] = sampled[..., 1:]
                    choices.append(self.model.pick(hypotheses, index))
                    predictions.append((hypotheses * logits.softmax(-1).unsqueeze(-1)).sum(-2))
                disagreement = torch.stack(predictions).var(0, unbiased=False).mean(-1, keepdim=True)
                latent = torch.stack([choices[f % len(choices)][..., f, :] for f in range(Fcount)], -2)
                cp = self.collision_logits(latent, g).sigmoid()[..., :1]
                u = cp.std(0, unbiased=False) + c['latent_uncertainty_scale'] * disagreement
                upper = (cp.mean(0) + c['uncertainty_beta'] * u).clamp(0, 1)
                returns -= discount * (c['collision_cost'] * upper + c['uncertainty_cost'] * u)
                uncertainty += u / H; discount *= c['gamma']
            api, _, _ = self.model.policy(latent, sample=False)
            returns += discount * self.q(latent, api, g, target=True)
            scores = returns.squeeze(-1); worst = scores.sort(-1).values[..., :max(1, int(np.ceil(Fcount * c['cvar'])))].mean(-1)
            score = (1 - c['risk']) * scores.mean(-1) + c['risk'] * worst
            top = score.topk(E, 1).indices; elite = actions.gather(1, top[..., None, None].expand(B, E, H, 2))
            weights = (c['temperature'] * score.gather(1, top)).softmax(1)
            mean = (elite * weights[..., None, None]).sum(1)
            std = ((elite - mean[:, None]).square() * weights[..., None, None]).sum(1).sqrt().clamp(.05, 1.)
        index = weights.argmax(1); rows = torch.arange(B, device=self.device)
        action = elite[rows, index, 0]; chosen = top[rows, index]
        u = uncertainty[rows, chosen].mean((-1, -2))
        self.mean = mean
        self.last_plan = action.cpu().numpy().copy()
        self.last_uncertainty = u.cpu().numpy()
        self.last_weight = (torch.exp(-c['distill_uncertainty_scale'] * u) * (u < c['distill_max_uncertainty'])).cpu().numpy()
        if explore: action = (action + c['exploration'] * torch.randn_like(action)).clamp(-1, 1)
        return action.cpu().numpy()

    def update(self, batch):
        c, m = self.c, self.model; burn = batch['burn']; T, B = batch['obs'].shape[:2]
        m.train(); self.extra_dynamics.train(); self.extra_collision.train()
        h = batch['obs'].new_zeros(B, c['latent']); zs, next_zs, contexts = [], [], []
        for t in range(T):
            if t < burn:
                with torch.no_grad(): _, h, _ = m.enc.step(batch['obs'][t], batch['previous'][t], h)
                continue
            z, h, g = m.enc.step(batch['obs'][t], batch['previous'][t], h)
            with torch.no_grad(): nz, _, _ = m.enc.step(batch['next_obs'][t], batch['executed'][t], h.detach())
            zs.append(z); next_zs.append(nz); contexts.append(g)
        z, nz, g = torch.stack(zs), torch.stack(next_zs), torch.stack(contexts)
        ex = batch['executed'][burn:]; rewards = batch['reward'][burn:, :, None]
        terminal = batch['terminal'][burn:, :, None].float()
        # A map-window boundary is a deliberate independent segment. No value is
        # bootstrapped into a geographically unrelated state; timeouts bootstrap.
        with torch.no_grad():
            pa, _, _ = m.policy(nz); targets = rewards + c['gamma'] * (1 - terminal) * self.q(nz, pa, g.detach(), True)
        dynamics_loss = z.new_zeros(())
        contact = batch['collision'][burn:]; known = contact >= 0
        bootstrap_masks = []
        for _ in self.dynamics:
            mask = (torch.rand(B, device=self.device) < .8).float(); mask[0] = 1; bootstrap_masks.append(mask)
        for i, dyn in enumerate(self.dynamics):
            rollout = z[0]; loss = z.new_zeros(())
            for t in range(len(z)):
                cand, logits = dyn(rollout, ex[t], g[t]); err = (cand - nz[t, :, None]).square().mean(-1)
                best = err.argmin(-1); rows = torch.arange(B, device=self.device)
                per = err[rows, best] + .05 * err.mean(-1) + .1 * F.cross_entropy(logits, best, reduction='none')
                loss += c['rho'] ** t * (per * bootstrap_masks[i]).sum() / bootstrap_masks[i].sum()
                rollout = m.pick(cand, best)
            dynamics_loss += loss / len(z) / len(self.dynamics)
        reward_loss = self.twohot.ce(m.reward(z, ex, g), rewards).mean()
        value_loss = self.twohot.ce(m.Q(z, ex, g), targets.unsqueeze(0).expand(c['num_q'], -1, -1, -1)).mean()
        logits = self.collision_logits(z, g)
        collision_loss = z.new_zeros(())
        for i, lg in enumerate(logits):
            weights = known.float() * bootstrap_masks[i][None, :, None]
            ce = F.binary_cross_entropy_with_logits(lg, contact.clamp(min=0), reduction='none')
            collision_loss += (ce * weights).sum() / weights.sum().clamp(min=1) / len(logits)
        physics_loss = F.smooth_l1_loss(m.physics(z), batch['obs'][burn:, :, PHYS_IDX])
        kinematics_loss = F.smooth_l1_loss(m.kinematics(z, ex), batch['aux'][burn:, :, 1:4] / torch.as_tensor(KIN_SCALE, device=self.device))
        context_loss = F.mse_loss(g, batch['genome'][burn:])
        loss = (c['consistency_coef'] * dynamics_loss + c['reward_coef'] * reward_loss + c['value_coef'] * value_loss
                + c['collision_coef'] * collision_loss + c['physics_coef'] * physics_loss
                + c['kinematics_coef'] * kinematics_loss + c['context_coef'] * context_loss)
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite world-model loss')
        self.opt.zero_grad(set_to_none=True); loss.backward(); grad = nn.utils.clip_grad_norm_(self.model_params, c['grad_clip'], error_if_nonfinite=True); self.opt.step()
        # The actor learns through Q w.r.t. action; critic/model parameters are frozen.
        m.eval()
        with frozen([m.qs]):
            action, logp, mu = m.policy(z.detach()); q = self.q(z.detach(), action, g.detach())
            with torch.no_grad():
                spread = (torch.quantile(q, .95) - torch.quantile(q, .05)).clamp(min=1).item()
                self.qscale = .99 * self.qscale + .01 * spread
            policy_loss = (-q / self.qscale + c['entropy_coef'] * logp).mean()
            teacher_mask = batch['aux'][burn:, :, 8:9]
            teacher_loss = ((mu - batch['aux'][burn:, :, 6:8]).square() * teacher_mask).sum() / (2 * teacher_mask.sum().clamp(min=1))
            pw = batch['planner_weight'][burn:, :, None]
            distill = ((mu - batch['planner'][burn:]).square() * pw).sum() / (2 * pw.sum().clamp(min=1))
            intervention = ((batch['request'][burn:] - ex).abs().sum(-1, keepdim=True) > .05).float()
            correction = ((mu - ex).square() * intervention).sum() / (2 * intervention.sum().clamp(min=1))
            policy_loss += c['teacher_coef'] * teacher_loss + c['planner_coef'] * distill + c['shield_coef'] * correction
            self.pi_opt.zero_grad(set_to_none=True); policy_loss.backward(); nn.utils.clip_grad_norm_(m.pi.parameters(), c['grad_clip'], error_if_nonfinite=True); self.pi_opt.step()
        with torch.no_grad():
            for q, tq in zip(m.qs.parameters(), self.target_q.parameters()): tq.lerp_(q, c['tau'])
        self.updates += 1
        return {k: float(v) for k, v in dict(loss=loss, dynamics=dynamics_loss, reward=reward_loss, value=value_loss,
                collision=collision_loss, physics=physics_loss, kinematics=kinematics_loss, context=context_loss,
                policy=policy_loss, teacher=teacher_loss, distill=distill, correction=correction, grad=grad).items()}

    def state_dict(self):
        return dict(model=self.model.state_dict(), dynamics=self.extra_dynamics.state_dict(), collision=self.extra_collision.state_dict(),
                    target_q=self.target_q.state_dict(), optimizer=self.opt.state_dict(), pi_optimizer=self.pi_opt.state_dict(),
                    h=self.h, previous=self.previous, mean=self.mean, qscale=self.qscale, updates=self.updates, steps=self.steps)

    def load_state_dict(self, state):
        self.model.load_state_dict(state['model']); self.extra_dynamics.load_state_dict(state['dynamics'])
        self.extra_collision.load_state_dict(state['collision']); self.target_q.load_state_dict(state['target_q'])
        self.opt.load_state_dict(state['optimizer']); self.pi_opt.load_state_dict(state['pi_optimizer'])
        self.h, self.previous, self.mean = state['h'], state['previous'], state['mean']
        self.qscale, self.updates = state['qscale'], state['updates']

    def export(self, path):
        def b64(t): return base64.b64encode(t.detach().cpu().float().contiguous().numpy().tobytes()).decode()
        def layers(seq):
            out = []
            mods = list(seq)
            for i, mod in enumerate(mods):
                if not isinstance(mod, nn.Linear): continue
                spec = dict(inp=mod.in_features, out=mod.out_features, W=b64(mod.weight), b=b64(mod.bias), act='none')
                j = i + 1
                if j < len(mods) and isinstance(mods[j], nn.LayerNorm): spec['ln'] = [b64(mods[j].weight), b64(mods[j].bias)]; j += 1
                if j < len(mods): spec['act'] = 'relu' if isinstance(mods[j], nn.ReLU) else 'mish' if isinstance(mods[j], nn.Mish) else 'simnorm' if isinstance(mods[j], SimNorm) else 'none'
                out.append(spec)
            return out
        enc = self.model.enc
        spec = dict(type='tdmpc_v5', schema=5, name='LOTUS v5 recurrent prior', steps=self.steps, egoDim=HEAD, K=K, nbDim=NB,
                    obsMean=b64(enc.base.mean), obsStd=b64(enc.base.std), phi=layers(enc.base.phi), head=layers(enc.base.head),
                    posterior=layers(enc.posterior), pi=layers(self.model.pi), simnorm=8, latent=self.c['latent'],
                    memory=self.c['memory'], gru={k: b64(v) for k, v in enc.gru.state_dict().items()}, tracking=True)
        from pathlib import Path
        p = Path(path); tmp = p.with_suffix(p.suffix + '.tmp'); tmp.write_text(json.dumps(spec)); tmp.replace(p)
