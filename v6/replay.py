"""Protected asynchronous teacher streams plus balanced recurrent batch sampling."""
import numpy as np
import torch
from v5_replay import SequenceReplay


class Replay(SequenceReplay):
    def __init__(self, n, capacity, device='cpu'):
        super().__init__(n, capacity, device)
        for name in ('shield_enabled', 'teacher_weight', 'source'):
            self.data[name] = np.zeros((n, self.cap), np.float32)


class Demonstrations:
    def __init__(self, n, capacity, device='cpu'):
        self.streams = [Replay(1, max(32, capacity // n), device) for _ in range(n)]
        self.skipping = np.zeros(n, bool)

    def add(self, row, eligible):
        for e, (buffer, use) in enumerate(zip(self.streams, eligible)):
            if not use:
                # A gap censors future labels and cuts recurrence. Mark the last
                # stored transition, rather than joining nonconsecutive teacher states.
                if not self.skipping[e] and buffer.clock:
                    i = (buffer.clock - 1) % buffer.cap
                    if not buffer.data['done'][0, i]:
                        buffer.data['done'][0, i] = True; buffer.data['terminal'][0, i] = False; buffer.episode += 1
                self.skipping[e] = True
                continue
            values = {k: np.asarray(v)[e:e+1] for k, v in row.items()}
            values['source'] = np.ones(1, np.float32)
            buffer.add(**values); self.skipping[e] = False

    def sample(self, count, horizon, burn):
        if count <= 0: return None
        candidates = [b for b in self.streams if b.clock >= horizon + burn]
        if not candidates: return None
        chunks = []
        # A few chunks reduce small-buffer sampling overhead while retaining diversity.
        remaining = count
        for _ in range(max(4, len(candidates) * 2)):
            if remaining <= 0: break
            buffer = candidates[np.random.randint(len(candidates))]
            sample = buffer.sample(remaining, horizon, burn, priority=0.)
            if sample is not None:
                chunks.append(sample); remaining -= sample['obs'].shape[1]
        return combine(chunks) if chunks else None

    def state_dict(self): return dict(streams=[b.state_dict() for b in self.streams], skipping=self.skipping.copy())

    def load_state_dict(self, state):
        if len(state['streams']) != len(self.streams): raise ValueError('Demonstration stream count changed')
        for b, s in zip(self.streams, state['streams']): b.load_state_dict(s)
        self.skipping[:] = state['skipping']

    @property
    def size(self): return sum(min(b.clock, b.cap) for b in self.streams)


def combine(chunks):
    chunks = [c for c in chunks if c is not None]
    if not chunks: return None
    if any(c['burn'] != chunks[0]['burn'] for c in chunks): raise ValueError('Burn-in differs across batch sources')
    return {k: chunks[0]['burn'] if k == 'burn' else torch.cat([c[k] for c in chunks], dim=1) for k in chunks[0]}


def sample_mixture(online, demos, counterfactual, cfg):
    B, H, burn = cfg['batch'], cfg['horizon'], cfg['burn_in']
    nd = int(round(B * cfg['demo_fraction'])); nc = int(round(B * cfg['counterfactual_fraction']))
    demo = demos.sample(nd, H, burn)
    used_d = demo['obs'].shape[1] if demo else 0
    cf = counterfactual.sample(nc, H, burn, cfg['priority_fraction']) if nc else None
    used_c = cf['obs'].shape[1] if cf else 0
    main = online.sample(max(1, B - used_d - used_c), H, burn, cfg['priority_fraction'])
    return combine([main, demo, cf])
