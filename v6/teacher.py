"""Competence-gated, reversible teacher-to-planner handover (validation only)."""
class Handover:
    def __init__(self, cfg):
        self.c = cfg; self.probability = 1.; self.passes = 0; self.baseline = None; self.last_report = None

    def update(self, report, steps):
        c = self.c; self.last_report = report
        eligible = (steps >= c['teacher_steps'] and report['episodes'] >= c['handover_min_episodes']
                    and report['ok'] >= c['handover_min_ok'] and report['contact_per_km'] <= c['handover_max_contact_per_km']
                    and report['red'] <= c['handover_max_red'] and report.get('projection_mae', float('inf')) <= c['handover_max_projection_mae'])
        if eligible:
            self.passes += 1
            if self.passes >= c['handover_required_passes']:
                self.probability = max(c['teacher_probability_floor'], self.probability - c['handover_decrement']); self.passes = 0
        else:
            self.passes = 0
            if self.probability < 1.: self.probability = min(1., self.probability + c['handover_decrement'])
        return dict(teacher_probability=self.probability, gate_passed=eligible, consecutive_passes=self.passes)

    def state_dict(self): return dict(probability=self.probability, passes=self.passes, baseline=self.baseline, last_report=self.last_report)

    def load_state_dict(self, state):
        for k, v in state.items(): setattr(self, k, v)


def evaluate_teacher(suite, cfg):
    import numpy as np
    import time
    from train import VecEnv
    from tdmpc2 import GSPACE
    from v5_eval import aggregate
    episodes = []
    for start in range(0, len(suite), cfg['eval_batch']):
        chunk = suite[start:start+cfg['eval_batch']]
        env = VecEnv(1, len(chunk), seed=991, extra=dict(specs=chunk, genomeSpace=GSPACE, reward='v5', shield=cfg['shield'],
                     lateralShield=cfg['lateral_shield'], tracking=cfg['tracking'], singleEpisode=True))
        alive = np.ones(len(chunk), bool)
        try:
            env.reset()
            for _ in range(cfg['eval_max_decisions']):
                t0 = time.perf_counter(); actions = np.asarray(env.ws[0].call(dict(cmd='teacher'))['actions'], np.float32)
                latency = (time.perf_counter()-t0)*1000/max(1,int(alive.sum()))
                _, _, done, infos = env.step(actions)
                for i in np.flatnonzero(alive & done):
                    row = dict(infos[i], latency_ms=latency); row.pop('terminal',None); episodes.append(row); alive[i] = False
                if not alive.any(): break
            if alive.any(): raise RuntimeError('Teacher baseline exceeded decision cap')
        finally: env.close()
    return aggregate(episodes), episodes
