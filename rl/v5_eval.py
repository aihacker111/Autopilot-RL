"""Frozen geographic/seed splits, controller-isolated evaluation and safety metrics."""
import copy
import hashlib
import json
import math
import random
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from train import VecEnv

ROOT = Path(__file__).resolve().parents[1]


def geographic_split(x, y):
    # Large spatial cells; callers exclude windows within a halo of boundaries.
    cell = f'{math.floor(x / 8000)},{math.floor(y / 8000)}'
    value = int(hashlib.sha256(cell.encode()).hexdigest()[:8], 16) % 10
    return 'test' if value < 2 else 'validation' if value < 4 else 'train'


def build_manifest(seed=2026, half=850, generated=False):
    groups = {k: [] for k in ('train', 'validation', 'test')}
    if not generated:
        base = ROOT / 'web/hcm'; index = json.loads((base / 'index.json').read_text())
        margin = half + 400
        for key, meta in sorted(index['tiles'].items()):
            if meta['j'] <= 400: continue
            tile = json.loads((base / 't' / f'{key}.json').read_text())
            for j in tile['J']:
                if j['t'] not in ('signal', 'roundabout', 'rail', 'priority'): continue
                x, y = j['x'], j['y']
                if min(x % 8000, 8000 - x % 8000, y % 8000, 8000 - y % 8000) < margin: continue
                groups[geographic_split(x, y)].append([x, y])
        if any(not v for v in groups.values()): raise ValueError('Not enough geographically isolated HCM centres; reduce half or provide a split manifest')
        # Deduplicate exact centres, keep deterministic bounded manifests.
        for key in groups:
            points = sorted(set(map(tuple, groups[key]))); random.Random(seed).shuffle(points)
            groups[key] = [list(p) for p in points[:512]]
    return dict(schema=5, seed=seed, half=half, generated=generated, centres=groups,
                seed_bases=dict(train=100000, validation=300000000, test=600000000),
                journey_semantics='independent map segments; multi-window routes are not geographically continuous')


def specs(manifest, split, count, genomes=None):
    if split not in manifest['centres']: raise ValueError(split)
    rng = random.Random(manifest['seed'] + manifest['seed_bases'][split]); points = manifest['centres'][split]
    presets = ['city', 'alley', 'roundabout', 'highway', 'rail', 'rural', 'pass']
    out = []
    for i in range(count):
        seed = manifest['seed_bases'][split] + i
        # A generated domain is held out by both map seed and traffic seed.
        row = dict(seed=seed, mapSeed=seed, density=[.5, .8, 1.05][i % 3], half=manifest['half'], routeM=500, routeLegs=12)
        if points: row.update(preset='hcm', centre=points[i % len(points)], group='hcm')
        else: row.update(preset=presets[i % len(presets)], group=presets[i % len(presets)])
        if genomes: row.update(genome=genomes[i % len(genomes)], density=genomes[i % len(genomes)]['density'])
        out.append(row)
    return out


@contextmanager
def isolated(agent, seed):
    py, npstate, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    online = {k: copy.deepcopy(getattr(agent, k)) for k in ('h', 'previous', 'mean', 'last_weight', 'last_uncertainty', 'last_plan')}
    modes = [m.training for m in (agent.model, agent.extra_dynamics, agent.extra_collision)]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); agent.reset()
    try: yield
    finally:
        for k, v in online.items(): setattr(agent, k, v)
        for m, flag in zip((agent.model, agent.extra_dynamics, agent.extra_collision), modes): m.train(flag)
        random.setstate(py); np.random.set_state(npstate); torch.set_rng_state(cpu)
        if cuda is not None: torch.cuda.set_rng_state_all(cuda)


def aggregate(episodes):
    n = max(1, len(episodes)); km = sum(e.get('travelled', 0) for e in episodes) / 1000
    contact = sum(e.get('contacts', e['result'] in ('crash', 'hit')) for e in episodes)
    # Event rates must be accompanied by exposure and count, including zero events.
    result = dict(episodes=len(episodes), km=km, contacts=int(contact),
                  contact_per_km=contact / max(km, 1e-9),
                  person_contacts=sum(e.get('personContacts', 0) for e in episodes),
                  shield_per_km=sum(e.get('shield', 0) + e.get('lateralShield', 0) for e in episodes) / max(km, 1e-9),
                  mean_jerk=sum(e.get('meanJerk', 0) for e in episodes) / n,
                  route_progress=sum(e.get('progressFraction', min(1, e.get('dist', 0) / max(1, e.get('routeLen', 1)))) for e in episodes) / n,
                  latency_ms=float(np.mean([e.get('latency_ms', 0) for e in episodes])))
    for key in ('ok', 'crash', 'hit', 'red', 'timeout'):
        result[key] = sum(bool(e.get('red')) if key == 'red' else e['result'] == key for e in episodes) / n
    return result


def evaluate(agent, suite, cfg, planner=True, shield=True, seed=991):
    import time
    from tdmpc2 import GSPACE
    episodes, predictions, outcomes = [], [], []
    with isolated(agent, seed):
        for start in range(0, len(suite), cfg['eval_batch']):
            chunk = suite[start:start + cfg['eval_batch']]
            env = VecEnv(1, len(chunk), seed=seed, extra=dict(specs=chunk, genomeSpace=GSPACE,
                reward='v5', shield=shield, lateralShield=shield and cfg['lateral_shield'], tracking=cfg['tracking'], singleEpisode=True))
            try:
                obs = env.reset(); agent.reset(); alive = np.ones(len(chunk), bool)
                latencies = [[] for _ in chunk]; cp = [[] for _ in chunk]; contacts = [[] for _ in chunk]
                max_decisions = cfg.get('eval_max_decisions', 2000)
                for _ in range(max_decisions):
                    t0 = time.perf_counter(); actions = agent.act(obs, planner, explore=False)
                    latency = (time.perf_counter() - t0) * 1000 / max(1, int(alive.sum()))
                    with torch.no_grad():
                        z = agent.model.enc.posterior(agent.h); context = agent.model.enc.context(agent.h)
                        probabilities = agent.collision_logits(z, context).sigmoid().mean(0)[:, 0].cpu().numpy()
                    obs, _, done, infos = env.step(actions); agent.observe_execution(env.aux[:, 4:6], done)
                    for i in np.flatnonzero(alive):
                        latencies[i].append(latency); cp[i].append(float(probabilities[i])); contacts[i].append(float(env.aux[i, 0]))
                        if done[i]:
                            row = dict(infos[i], group=chunk[i].get('group', 'curriculum'), latency_ms=float(np.mean(latencies[i])))
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
    return result, episodes


def rank(result, cfg):
    # Safety eligibility first, then completion/progress. No test data in selection.
    feasible = (result['person_contacts'] == 0 and result['contact_per_km'] <= cfg['max_contact_per_km'] and result['red'] <= cfg['max_red'])
    return (int(feasible), -result['person_contacts'], -result['contact_per_km'] if not feasible else result['ok'],
            result['route_progress'], -result['red'], -result['shield_per_km'], -result['mean_jerk'])
