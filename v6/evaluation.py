"""V6 validation: exact shield-mode agreement and measured control diagnostics."""
import time
import numpy as np
import torch
from train import VecEnv
from v5_eval import ROOT, build_manifest, specs, isolated, aggregate, rank


def evaluate(agent, suite, cfg, planner=True, shield=True, seed=991):
    mode = agent.execution_shield; projection_mode = agent.projector.training
    old_z, old_context = agent.last_z, agent.last_context
    try:
        agent.execution_shield = bool(shield)
        return _evaluate(agent, suite, cfg, planner, shield, seed)
    finally:
        agent.projector.train(projection_mode)
        agent.execution_shield = mode; agent.last_z, agent.last_context = old_z, old_context


def _evaluate(agent, suite, cfg, planner=True, shield=True, seed=991):
    import time
    from tdmpc2 import GSPACE
    episodes, predictions, outcomes = [], [], []; projection_errors = []
    with isolated(agent, seed):
        for start in range(0, len(suite), cfg['eval_batch']):
            chunk = suite[start:start + cfg['eval_batch']]
            env = VecEnv(1, len(chunk), seed=seed, extra=dict(specs=chunk, genomeSpace=GSPACE,
                reward='v5', shield=shield, lateralShield=shield and cfg['lateral_shield'], tracking=cfg['tracking'], singleEpisode=True))
            try:
                obs = env.reset(); agent.reset(); alive = np.ones(len(chunk), bool)
                returns = [[] for _ in chunk]; cuts = [[] for _ in chunk]; stationary = [[] for _ in chunk]; initial_q = [None for _ in chunk]
                latencies = [[] for _ in chunk]; cp = [[] for _ in chunk]; contacts = [[] for _ in chunk]
                max_decisions = cfg.get('eval_max_decisions', 2000)
                for _ in range(max_decisions):
                    t0 = time.perf_counter(); actions = agent.act(obs, planner, explore=False)
                    latency = (time.perf_counter() - t0) * 1000 / max(1, int(alive.sum()))
                    with torch.no_grad():
                        z = agent.model.enc.posterior(agent.h); context = agent.model.enc.context(agent.h)
                        probabilities = agent.collision_logits(z, context).sigmoid().mean(0)[:, 0].cpu().numpy()
                    with torch.no_grad():
                        predicted_execution = agent.project(agent.last_z, torch.as_tensor(actions, device=agent.device)).cpu().numpy()
                        predicted_q = agent.q(agent.last_z, torch.as_tensor(actions, device=agent.device), agent.last_context).squeeze(-1).cpu().numpy()
                    obs, rewards, done, infos = env.step(actions)
                    actual = env.aux[:, 4:6]
                    projection_errors.extend(np.abs(predicted_execution[alive] - actual[alive]).mean(-1).tolist())
                    cost = agent.intervention_cost(torch.as_tensor(actions, device=agent.device), torch.as_tensor(actual, device=agent.device)).squeeze(-1).cpu().numpy()
                    for i in np.flatnonzero(alive):
                        if not returns[i]: initial_q[i] = float(predicted_q[i])
                        returns[i].append(float(rewards[i] - cost[i])); cuts[i].append(bool(np.abs(actions[i]-actual[i]).sum() > .05))
                        stationary[i].append(bool(obs[i,0] < .05))
                    agent.observe_execution(actual, done)
                    for i in np.flatnonzero(alive):
                        latencies[i].append(latency); cp[i].append(float(probabilities[i])); contacts[i].append(float(env.aux[i, 0]))
                        if done[i]:
                            row = dict(infos[i], group=chunk[i].get('group', 'curriculum'), latency_ms=float(np.mean(latencies[i])))
                            row.update(initial_q=initial_q[i], discounted_return=sum(cfg['gamma']**t * r for t,r in enumerate(returns[i])), decisions=len(returns[i]), intervention_fraction=float(np.mean(cuts[i])), stationary_fraction=float(np.mean(stationary[i])))
                            row.pop('terminal', None); episodes.append(row); alive[i] = False
                            # Label contact-within-1s at each state; censor end-of-timeout.
                            for j in range(len(cp[i])):
                                future = contacts[i][j:j + 5]
                                if len(future) == 5 or any(future) or infos[i]['result'] not in ('timeout', 'stage'):
                                    predictions.append(cp[i][j]); outcomes.append(float(any(future)))
                    if not alive.any(): break
                if alive.any(): raise RuntimeError('Evaluation exceeded eval_max_decisions; increase the cap. Incomplete episodes are not scored as safe.')
            finally: env.close()
    result = aggregate(episodes)
    result.update(controller='planner' if planner else 'prior', shield=bool(shield),
                  groups={g: aggregate([e for e in episodes if e['group'] == g]) for g in sorted({e['group'] for e in episodes})})
    result.update(collision_brier=None, collision_ece=None, calibration_samples=0)
    if predictions:
        p, y = np.array(predictions), np.array(outcomes)
        ece = 0.
        for lo in np.linspace(0, .9, 10):
            mask = (p >= lo) & (p < lo + .1 if lo < .9 else p <= 1)
            if mask.any(): ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
        result.update(collision_brier=float(np.mean((p - y) ** 2)), collision_ece=float(ece), calibration_samples=len(p))
    result.update(projection_mae=float(np.mean(projection_errors)) if projection_errors else None, intervention_fraction=float(np.mean([e['intervention_fraction'] for e in episodes])), stationary_fraction=float(np.mean([e['stationary_fraction'] for e in episodes])))
    complete = [e for e in episodes if e['result'] not in ('timeout', 'stage')]
    result['initial_q_minus_return'] = float(np.mean([e['initial_q']-e['discounted_return'] for e in complete])) if complete else None
    return result, episodes

