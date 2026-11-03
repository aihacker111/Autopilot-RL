"""MAP-Elites curriculum selected with the deployed full planner.

The prior is a cheap first screen. Dynamics imagination ranks a diverse candidate
set; full-controller rollouts and measured realism decide archive membership.
Solvability is a teacher estimate, never a permanent impossibility judgement.
"""
import copy
import hashlib
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
import coevolve as co
from tdmpc2 import gnorm, GD
from v5_eval import specs, evaluate, isolated, ROOT


class Curriculum:
    def __init__(self, cfg, manifest):
        self.c, self.manifest = cfg, manifest
        self.rng = random.Random(cfg['seed'] + 717)
        self.cells, self.pending, self.history, self.mastered = {}, [], [], []
        self.generation = 0
        self.calibration = []
        self.priors = json.loads((ROOT / cfg['priors']).read_text())
        self.realism_source = self.priors.get('_about', 'user supplied priors')

    @torch.no_grad()
    def imagine(self, agent, candidates, replay):
        states = replay.states(self.c['evo_states']); c = self.c
        if not len(states): return np.zeros(len(candidates))
        o = torch.as_tensor(states, device=agent.device); m = agent.model
        m.eval(); z0, _, _ = m.enc.step(o, o[:, 1:3]); results = []
        for start in range(0, len(candidates), c['dream_batch']):
            gs = torch.as_tensor(np.stack([gnorm(g) for g in candidates[start:start + c['dream_batch']]]), device=agent.device)
            C, N = len(gs), len(o); g = gs[:, None].expand(C, N, GD); z = z0[None].expand(C, N, -1)
            ret = z.new_zeros(C, N, 1); discount = 1.
            for _ in range(c['dream_horizon']):
                action, _, _ = m.policy(z, sample=False)
                ret += discount * agent.twohot.decode(m.reward(z, action, g))
                z = m.next(z, action, g, 'sample')
                collision = agent.collision_logits(z, g).sigmoid()
                ret -= discount * c['collision_cost'] * (collision.mean(0)[..., :1] + c['uncertainty_beta'] * collision.std(0, unbiased=False)[..., :1])
                discount *= c['gamma']
            action, _, _ = m.policy(z, sample=False); ret += discount * agent.q(z, action, g, True)
            results.extend((-ret.mean((1, 2))).cpu().tolist())
        return np.array(results)

    def predicted_success(self, difficulty):
        if len(self.calibration) < 8:
            ranks = np.argsort(np.argsort(difficulty)) / max(1, len(difficulty) - 1)
            return 1 - ranks
        x, y = np.array(self.calibration).T; mu, sd = x.mean(), x.std() + 1e-6
        w, b = 0., float(np.log(np.clip(y.mean(), .01, .99) / np.clip(1 - y.mean(), .01, .99)))
        for _ in range(100):
            pred = 1 / (1 + np.exp(-np.clip(w * (x - mu) / sd + b, -30, 30)))
            w -= .2 * np.mean((pred - y) * (x - mu) / sd); b -= .2 * np.mean(pred - y)
        return 1 / (1 + np.exp(-np.clip(w * (difficulty - mu) / sd + b, -30, 30)))

    def suite(self, genome, n, offset=0):
        suite = specs(self.manifest, 'train', n, [genome])
        key = json.dumps(genome, sort_keys=True); salt = int(hashlib.sha256(key.encode()).hexdigest()[:6], 16)
        for i, row in enumerate(suite):
            row['seed'] += salt + offset + i * 997; row['mapSeed'] = row['seed']
        return suite

    def teacher_success(self, suite):
        from train import VecEnv
        env = VecEnv(1, len(suite), seed=self.c['seed'], extra=dict(specs=suite, reward='v5', shield=self.c['shield'],
                     lateralShield=self.c['lateral_shield'], tracking=self.c['tracking'], singleEpisode=True))
        alive = np.ones(len(suite), bool); results = []
        try:
            env.reset()
            for _ in range(self.c.get('eval_max_decisions', 2000)):
                actions = env.ws[0].call(dict(cmd='teacher'))['actions']
                _, _, done, infos = env.step(np.asarray(actions, np.float32))
                for i in np.flatnonzero(alive & done): results.append(infos[i]['result'] == 'ok' and not infos[i]['red']); alive[i] = False
                if not alive.any(): break
            if alive.any(): raise RuntimeError('Teacher evaluation exceeded the decision cap; this does not prove unsolvability')
            return sum(results) / len(suite)
        finally: env.close()

    def realism(self, episodes):
        stats = [e['realismStats'] for e in episodes if 'realismStats' in e]
        if not stats: return 0., {'missing_stats': 1}
        raw = [e['realismExposure'] for e in episodes if 'realismExposure' in e]
        if len(raw) == len(episodes):
            total = {k: sum(r[k] for r in raw) for k in raw[0]}
            mean = dict(instantStopsPerVehMin=total['instantStops'] / max(1e-6, total['vehSeconds'] / 60),
                        cutInsPerVehMin=total['cutIns'] / max(1e-6, total['vehSeconds'] / 60),
                        suddenBrakesPerTwoMin=total['suddenBrakes'] / max(1e-6, total['twoSeconds'] / 60),
                        npcCollisionsPerMin=total['collisions'] / max(1e-6, total['seconds'] / 60),
                        wrongWayShare=total['wrongSteps'] / max(1, total['twoSteps']),
                        hitRate=sum(e['result'] == 'hit' for e in episodes) / len(episodes))
        else: mean = {k: float(np.mean([s[k] for s in stats])) for k in stats[0]}
        return co.realism(mean, self.priors)

    def update(self, agent, replay, env, steps):
        self.generation += 1; c = self.c; parents = list(self.cells.values())
        candidates = []
        for _ in range(c['evo_candidates']):
            candidates.append(co.mutate(self.rng.choice(parents)['genome'], self.rng) if parents and self.rng.random() < .8 else co.random_genome(self.rng))
        # Restore agent modes/RNG afterwards: evaluation must not alter the training stream.
        with isolated(agent, c['seed'] + self.generation):
            difficulty = self.imagine(agent, candidates, replay)
        p = self.predicted_success(difficulty); scores = p * (1 - p)
        order = np.argsort(-scores); selected, seen = [], set()
        for i in order:
            cell = co.descriptor(candidates[i])
            if cell not in seen: selected.append(int(i)); seen.add(cell)
            if len(selected) >= c['evo_screen']: break
        # Cheap prior check; the full planner, not this screen, supplies p for curation.
        screened = []
        for i in selected:
            report, _ = evaluate(agent, self.suite(candidates[i], c['evo_prior_episodes']), c, planner=False, shield=c['shield'])
            screened.append((abs(report['ok'] - .5), i))
        checked = [candidates[i] for _, i in sorted(screened)[:c['evo_real']]]
        source = {json.dumps(candidates[i], sort_keys=True): float(difficulty[i]) for _, i in screened}
        old = list(self.cells.values())
        retry = self.pending[:c['evo_retry']]; self.pending = self.pending[c['evo_retry']:]
        genomes = checked + [r['genome'] for r in old + retry]
        records = []; unsolvable = unrealistic = 0
        old_by_key = {json.dumps(r['genome'], sort_keys=True): r for r in old + retry}
        for genome in genomes:
            key = json.dumps(genome, sort_keys=True); previous = old_by_key.get(key)
            suite = self.suite(genome, c['evo_min_episodes'])
            report, episodes = evaluate(agent, suite, c, planner=True, shield=c['shield'])
            n, wins = len(episodes), sum(e['result'] == 'ok' and not e.get('red') for e in episodes)
            # Beta posterior shrinks tiny-sample extremes; allocate more checks until
            # uncertainty is small enough or this generation's budget is exhausted.
            while n < c['evo_max_episodes']:
                alpha, beta = wins + 1, n - wins + 1
                sd = math.sqrt(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)))
                if 3.92 * sd <= c['evo_interval_width']: break
                more = min(c['evo_min_episodes'], c['evo_max_episodes'] - n)
                _, extra = evaluate(agent, self.suite(genome, more, offset=n * 104729), c, planner=True, shield=c['shield'])
                episodes += extra; n += len(extra); wins += sum(e['result'] == 'ok' and not e.get('red') for e in extra)
            success = (wins + 1) / (n + 2)
            teacher_ok = previous['teacher_ok'] if previous and previous['teacher_ok'] >= c['teacher_min'] else self.teacher_success(suite[:c['evo_teacher_episodes']])
            realistic, parts = self.realism(episodes)
            progress = max(0., success - previous['p']) if previous else 0.
            learn = success * (1 - success)
            novelty = 1. if str(co.descriptor(genome)) not in self.cells else 0.
            score = learn + c['progress_weight'] * progress + c['novelty_weight'] * novelty
            record = dict(genome=genome, p=success, n=n, wins=wins, teacher_ok=teacher_ok, realism=realistic,
                          realism_parts=parts, progress=progress, learn=learn, score=score, last_seen=steps,
                          cell=list(co.descriptor(genome)), solvability='teacher_estimate')
            if teacher_ok < c['teacher_min']: self.pending.append(record); unsolvable += 1
            elif realistic < c['realism_min']: unrealistic += 1
            elif learn >= c['learn_min']: records.append(record)
            elif success > .5: self.mastered.append(record)
            if key in source: self.calibration.append((source[key], success))
        self.mastered = self.mastered[-c['pending_capacity']:]
        self.calibration = self.calibration[-512:]; self.pending = self.pending[-c['pending_capacity']:]
        self.cells = {}
        for r in records:
            key = str(tuple(r['cell']))
            if key not in self.cells or r['score'] > self.cells[key]['score']: self.cells[key] = r
        self.apply(env, steps)
        row = dict(steps=steps, generation=self.generation, cells=len(self.cells), checked=len(genomes),
                   teacher_failed=unsolvable, unrealistic=unrealistic, pending=len(self.pending),
                   mean_p=float(np.mean([r['p'] for r in records])) if records else None,
                   realism_source=self.realism_source)
        self.history.append(row); return row

    def apply(self, env, steps):
        elites = list(self.cells.values())
        mastered = self.mastered[-max(1, len(elites) // 4):] if self.mastered else []
        # An empty archive explicitly clears an earlier pool.
        weights = [r['score'] + self.c['staleness_weight'] * min(1, (steps - r['last_seen']) / max(1, self.c['evo_every'])) for r in elites]
        pool = elites + mastered
        if mastered:
            mass = sum(weights) or 1.
            weights += [mass * .1 / len(mastered)] * len(mastered)
        env.set_pool([r['genome'] for r in pool], self.c['pool_probability'] if pool else 0., weights)

    def state_dict(self):
        return dict(cells=self.cells, pending=self.pending, mastered=self.mastered, history=self.history, generation=self.generation,
                    calibration=self.calibration, rng=self.rng.getstate())

    def load_state_dict(self, state):
        for key in ('cells', 'pending', 'mastered', 'history', 'generation', 'calibration'): setattr(self, key, state[key])
        self.rng.setstate(state['rng'])
