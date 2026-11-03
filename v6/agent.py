"""V6 learning contract:

D/R/kinematics take EXECUTED actions. Q takes REQUESTED actions plus a shield-mode
bit, and targets physical reward minus an explicit intervention cost. A learned
projection predicts executed commands for latent planning; it never replaces the
real environment guard. Counterfactual branches use shield_mode=0. Heads are
supervised on ensemble multi-step latents, with a posterior auxiliary objective.
"""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from v5_agent import Agent as V5Agent, frozen
from tdmpc2 import GD, PHYS_IDX, KIN_SCALE, mlp


class Agent(V5Agent):
    def __init__(self, cfg, device):
        super().__init__(cfg, device)
        L, H = cfg['latent'], cfg['hidden']
        self.model.qs = nn.ModuleList([mlp(L + 2 + GD + 1, [H, H], 101, dropout=.01) for _ in range(cfg['num_q'])]).to(self.device)
        for q in self.model.qs: nn.init.zeros_(q[-1].weight); nn.init.zeros_(q[-1].bias)
        self.target_q = copy.deepcopy(self.model.qs).eval(); self.target_q.requires_grad_(False)
        self.projector = mlp(L + 2, [H], 2).to(self.device)
        nn.init.zeros_(self.projector[-1].weight); nn.init.zeros_(self.projector[-1].bias)
        self.model_params = ([p for n, p in self.model.named_parameters() if not n.startswith('pi.')]
                             + list(self.extra_dynamics.parameters()) + list(self.extra_collision.parameters()) + list(self.projector.parameters()))
        self.opt = torch.optim.Adam(self.model_params, lr=cfg['lr'])
        self.execution_shield = bool(cfg['shield'])
        self.last_z = self.last_context = None

    def mode_flag(self, z, enabled=None):
        if enabled is None: enabled = self.execution_shield
        if isinstance(enabled, torch.Tensor): return enabled.to(z).expand(*z.shape[:-1], 1)
        return z.new_full((*z.shape[:-1], 1), float(enabled))

    def project(self, z, request, enabled=None):
        corrected = (request + 2 * torch.tanh(self.projector(torch.cat([z, request], -1)))).clamp(-1, 1)
        flag = self.mode_flag(z, enabled)
        return flag * corrected + (1 - flag) * request

    def intervention_cost(self, request, executed):
        return self.c['intervention_cost'] * (request - executed).square().mean(-1, keepdim=True)

    def q_logits(self, z, request, context, target=False, enabled=None):
        x = torch.cat([z, request, context, self.mode_flag(z, enabled)], -1)
        return torch.stack([q(x) for q in (self.target_q if target else self.model.qs)])

    def q(self, z, a, g, target=False, enabled=None):
        values = self.twohot.decode(self.q_logits(z, a, g, target, enabled))
        # Match TD-MPC2's random two-Q reduction, rather than min over all heads.
        subset = torch.randperm(len(values), device=self.device)[:min(2, len(values))]
        chosen = values[subset]
        return chosen.min(0).values if target else chosen.mean(0)

    @torch.no_grad()
    def act(self, obs, planner=True, explore=False):
        self.model.eval(); self.extra_dynamics.eval(); self.extra_collision.eval(); self.projector.eval()
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device); B = len(o)
        if self.h is None or len(self.h) != B:
            self.h = o.new_zeros(B, self.c['latent']); self.previous = o.new_zeros(B, 2); self.mean = None
        z, self.h, context = self.model.enc.step(o, self.previous, self.h)
        self.last_z, self.last_context = z, context
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
            zp = self.model.next(zp, self.project(zp, proposals[:, :, t]), gp, 'best')
        # Keep the same ensemble member along each future, rather than switching models every step.
        for _ in range(iterations):
            actions = (mean[:, None] + std[:, None] * torch.randn(B, S, H, 2, device=self.device)).clamp(-1, 1)
            actions[:, -prior_plans:] = proposals
            latent = z[:, None, None].expand(B, S, Fcount, -1)
            returns = o.new_zeros(B, S, Fcount, 1); uncertainty = o.new_zeros(B, S, Fcount, 1); discount = 1.
            for t in range(H):
                requested = actions[:, :, None, t].expand(B, S, Fcount, 2)
                a = self.project(latent, requested)
                returns += discount * (self.twohot.decode(self.model.reward(latent, a, g)) - self.intervention_cost(requested, a))
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

    def _posterior_sequence(self, batch):
        m = self.model; burn = batch['burn']; T, B = batch['obs'].shape[:2]
        h = batch['obs'].new_zeros(B, self.c['latent']); zs, nzs, gs, ngs = [], [], [], []
        for t in range(T):
            if t < burn:
                with torch.no_grad(): _, h, _ = m.enc.step(batch['obs'][t], batch['previous'][t], h)
                continue
            z, h, g = m.enc.step(batch['obs'][t], batch['previous'][t], h)
            with torch.no_grad(): nz, _, ng = m.enc.step(batch['next_obs'][t], batch['executed'][t], h.detach())
            zs.append(z); nzs.append(nz); gs.append(g); ngs.append(ng)
        return tuple(torch.stack(x) for x in (zs, nzs, gs, ngs))

    def update(self, batch):
        c, m, burn = self.c, self.model, batch['burn']
        m.train(); self.extra_dynamics.train(); self.extra_collision.train(); self.projector.train()
        z, nz, context, next_context = self._posterior_sequence(batch)
        H, B = z.shape[:2]; ex = batch['executed'][burn:]; request = batch['request'][burn:]
        reward = batch['reward'][burn:, :, None]; terminal = batch['terminal'][burn:, :, None].float()
        enabled = batch['shield_enabled'][burn:, :, None].float()
        clab = batch['collision'][burn:]; known = clab >= 0
        adjusted_reward = reward - self.intervention_cost(request, ex)
        with torch.no_grad():
            pa, _, _ = m.policy(nz)
            target = adjusted_reward + c['gamma'] * (1 - terminal) * self.q(nz, pa, next_context, True, enabled)
        ar = torch.arange(B, device=self.device); temporal = z.new_tensor([c['rho'] ** t for t in range(H)])
        g_roll = context[0:1].expand_as(context)  # the same fixed traffic context used during planning
        kin_scale = torch.as_tensor(KIN_SCALE, device=self.device)

        def heads(states, gs, mask, member=None):
            # Per-timestep averages; bootstrap masks are fixed along each sequence.
            def reduce(per, valid=None):
                weights = mask[None].expand(H, B)
                if valid is not None: weights = weights * valid
                avg = (per * weights).sum(1) / weights.sum(1).clamp(min=1)
                return (avg * temporal).sum() / H
            losses = {}
            losses['reward'] = reduce(self.twohot.ce(m.reward(states, ex, gs), reward).squeeze(-1))
            qloss = self.twohot.ce(self.q_logits(states, request, gs, enabled=enabled), target.unsqueeze(0).expand(c['num_q'], -1, -1, -1)).mean(0).squeeze(-1)
            losses['value'] = reduce(qloss)
            lg = self.collision_logits(states, gs)
            ce = F.binary_cross_entropy_with_logits(lg, clab.clamp(min=0).unsqueeze(0).expand_as(lg), reduction='none')
            ce = ce.mean(0) if member is None else ce[member]
            valid_count = known.sum(-1)
            per = (ce * known).sum(-1) / valid_count.clamp(min=1)
            losses['collision'] = reduce(per, valid_count > 0)
            losses['physics'] = reduce(F.smooth_l1_loss(m.physics(states), batch['obs'][burn:, :, PHYS_IDX], reduction='none').mean(-1))
            losses['kinematics'] = reduce(F.smooth_l1_loss(m.kinematics(states, ex), batch['aux'][burn:, :, 1:4] / kin_scale, reduction='none').mean(-1))
            projection = self.project(states, request, enabled)
            losses['projection'] = reduce((projection - ex).square().mean(-1), enabled.squeeze(-1) > .5)
            return losses

        components = {k: z.new_zeros(()) for k in ('dynamics', 'reward', 'value', 'collision', 'physics', 'kinematics', 'projection')}
        step_errors = z.new_zeros(H)
        for member, dyn in enumerate(self.dynamics):
            mask = (torch.rand(B, device=self.device) < .8).float(); mask[0] = 1
            rollout = z[0]; imagined = []; consistency = z.new_zeros(())
            for t in range(H):
                imagined.append(rollout)
                candidates, logits = dyn(rollout, ex[t], g_roll[t])
                error = (candidates - nz[t, :, None]).square().mean(-1); winner = error.argmin(-1)
                wta = error[ar, winner] + .05 * error.mean(-1) + .1 * F.cross_entropy(logits, winner, reduction='none')
                consistency += temporal[t] * (wta * mask).sum() / mask.sum()
                step_errors[t] += error[ar, winner].detach().mean() / len(self.dynamics)
                rollout = m.pick(candidates, winner)
            components['dynamics'] += consistency / H / len(self.dynamics)
            terms = heads(torch.stack(imagined), g_roll, mask, member)
            for name, loss in terms.items(): components[name] += loss / len(self.dynamics)
        posterior_terms = heads(z, context, z.new_ones(B))
        core = z.new_zeros(())
        for name, coefficient in [('dynamics','consistency_coef'), ('reward','reward_coef'), ('value','value_coef'),
                                  ('collision','collision_coef'), ('physics','physics_coef'), ('kinematics','kinematics_coef'), ('projection','projection_coef')]:
            core += c[coefficient] * components[name]
            if name in posterior_terms: core += c['posterior_coef'] * c[coefficient] * posterior_terms[name]
        context_loss = F.mse_loss(context, batch['genome'][burn:]); core += c['context_coef'] * context_loss
        if not torch.isfinite(core): raise FloatingPointError('Nonfinite v6 world-model loss')
        self.opt.zero_grad(set_to_none=True); core.backward()
        grad = nn.utils.clip_grad_norm_(self.model_params, c['grad_clip'], error_if_nonfinite=True); self.opt.step()
        m.eval(); self.projector.eval()
        # Q is explicitly Q(request, shield_mode); actor and TD targets use that
        # same contract. CF data train the unshielded model/critic, not the actor.
        actor_mask = (batch['source'][burn:] != 2).float()[..., None]
        with frozen([m.qs]):
            action, logp, mu = m.policy(z.detach()); qvalue = self.q(z.detach(), action, context.detach(), enabled=enabled)
            with torch.no_grad():
                spread = (torch.quantile(qvalue, .95) - torch.quantile(qvalue, .05)).clamp(min=1).item()
                self.qscale = .99 * self.qscale + .01 * spread
            denominator = actor_mask.sum().clamp(min=1)
            policy = ((-qvalue / self.qscale + c['entropy_coef'] * logp) * actor_mask).sum() / denominator
            safe_label = (known[..., 0:1] & (clab[..., 0:1] == 0)).float()
            teacher_weight = batch['teacher_weight'][burn:, :, None] * safe_label * actor_mask
            teacher_loss = ((mu - batch['aux'][burn:, :, 6:8]).square() * teacher_weight).sum() / (2 * teacher_weight.sum().clamp(min=1))
            pw = batch['planner_weight'][burn:, :, None] * actor_mask
            distill = ((mu - batch['planner'][burn:]).square() * pw).sum() / (2 * pw.sum().clamp(min=1))
            intervention = ((request - ex).abs().sum(-1, keepdim=True) > .05).float() * actor_mask
            correction = ((mu - ex).square() * intervention).sum() / (2 * intervention.sum().clamp(min=1))
            policy += c['teacher_coef'] * teacher_loss + c['planner_coef'] * distill + c['shield_coef'] * correction
            self.pi_opt.zero_grad(set_to_none=True)
            if actor_mask.any():
                policy.backward(); nn.utils.clip_grad_norm_(m.pi.parameters(), c['grad_clip'], error_if_nonfinite=True); self.pi_opt.step()
        with torch.no_grad():
            for q, tq in zip(m.qs.parameters(), self.target_q.parameters()): tq.lerp_(q, c['tau'])
            probabilities = self.collision_logits(z.detach(), context.detach()).sigmoid().mean(0)
            positive = known & (clab > .5)
            projection_error = (self.project(z.detach(), request, enabled) - ex).abs().mean()
        self.updates += 1
        metrics = dict(loss=core, **components, posterior_value=posterior_terms['value'], context=context_loss,
                       policy=policy, teacher=teacher_loss, distill=distill, correction=correction, grad=grad,
                       q_actor=qvalue.mean(), q_target=target.mean(), projection_mae=projection_error,
                       collision_known_fraction=known.float().mean(), collision_positive_fraction=positive.float().sum()/known.sum().clamp(min=1),
                       collision_pred_positive=(probabilities*positive).sum()/positive.sum().clamp(min=1),
                       collision_pred_negative=(probabilities*(known & ~positive)).sum()/(known & ~positive).sum().clamp(min=1),
                       demo_fraction=(batch['source'][burn:] == 1).float().mean(), counterfactual_fraction=(batch['source'][burn:] == 2).float().mean())
        metrics.update({f'latent_error_{i+1}': error for i, error in enumerate(step_errors)})
        return {k: v.detach().item() if isinstance(v, torch.Tensor) else float(v) for k, v in metrics.items()}

    def state_dict(self):
        return dict(super().state_dict(), projector=self.projector.state_dict())

    def load_state_dict(self, state):
        super().load_state_dict(state); self.projector.load_state_dict(state['projector']); self.steps = state['steps']

    def initialize_weights(self, state):
        loaded, skipped = [], []
        current = self.model.state_dict()
        for key, value in state['model'].items():
            if key in current and current[key].shape == value.shape: current[key] = value; loaded.append(key)
            else:
                if not key.startswith('qs.'): raise ValueError(f'Incompatible initialization tensor: {key}')
                skipped.append(key)
        self.model.load_state_dict(current)
        for name, module in [('dynamics',self.extra_dynamics), ('collision',self.extra_collision)]:
            if name in state: module.load_state_dict(state[name])
        if 'projector' in state: self.projector.load_state_dict(state['projector'])
        self.target_q.load_state_dict(self.model.qs.state_dict()); self.reset()
        return dict(loaded=loaded, skipped=skipped, optimizer_replay_rng_restored=False)
