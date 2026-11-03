"""Sequence replay with absolute slot generations and censored contact labels.

All streams use a monotonically increasing clock. A sampled history never crosses
an episode or map-window boundary. Future labels are unknown until the requested
horizon is observed, a contact is seen, or a genuine absorbing terminal occurs.
"""
from collections import deque
import numpy as np
import torch
from distill import OBS
from tdmpc2 import GD, AUXD, COLL_K


class SequenceReplay:
    def __init__(self, n, capacity, device='cpu'):
        self.n, self.cap, self.device = n, max(32, capacity // n), device
        self.clock = 0
        self.stamp = np.full(self.cap, -1, np.int64)
        self.episode = np.zeros(n, np.int64)
        self.contact_refs = deque(maxlen=max(1024, capacity))
        self.data = {}
        for k, shape, dtype in [('obs', (OBS,), np.float32), ('next_obs', (OBS,), np.float32),
                                ('request', (2,), np.float32), ('executed', (2,), np.float32),
                                ('previous', (2,), np.float32), ('reward', (), np.float32),
                                ('done', (), bool), ('terminal', (), bool), ('episode', (), np.int64),
                                ('aux', (AUXD,), np.float32), ('genome', (GD,), np.float32),
                                ('planner', (2,), np.float32), ('planner_weight', (), np.float32)]:
            self.data[k] = np.zeros((n, self.cap, *shape), dtype=dtype)

    def add(self, **row):
        i = self.clock % self.cap
        for k, v in row.items():
            if k not in self.data: raise KeyError(k)
            self.data[k][:, i] = v
        self.data['episode'][:, i] = self.episode
        self.stamp[i] = self.clock
        for e in np.flatnonzero(self.data['aux'][:, i, 0] > .5): self.contact_refs.append((int(e), self.clock))
        self.episode += self.data['done'][:, i]
        self.clock += 1
        oldest = self.clock - self.cap
        while self.contact_refs and self.contact_refs[0][1] < oldest: self.contact_refs.popleft()

    def valid(self, e, start, length):
        t = np.arange(start, start + length); ix = t % self.cap
        return (start >= max(0, self.clock - self.cap) and start + length <= self.clock
                and np.all(self.stamp[ix] == t)
                and not self.data['done'][e, ix[:-1]].any())

    def labels(self, e, t):
        labels = np.full(len(COLL_K), -1., np.float32)
        for q, horizon in enumerate(COLL_K):
            for d in range(horizon):
                s = t + d; i = s % self.cap
                if self.stamp[i] != s: break
                if self.data['aux'][e, i, 0] > .5: labels[q] = 1; break
                if self.data['done'][e, i]:
                    if self.data['terminal'][e, i]: labels[q] = 0
                    break
                if d == horizon - 1: labels[q] = 0
        return labels

    def sample(self, batch, horizon, burn=0, priority=.25, rng=None):
        rng = rng or np.random
        length = burn + horizon; hi = self.clock - length + 1; lo = max(0, self.clock - self.cap)
        if hi <= lo: return None
        picks = []; refs = list(self.contact_refs)
        for _ in range(batch * 40):
            if refs and rng.random() < priority:
                e, c = refs[rng.randint(len(refs))]; start = c - rng.randint(length)
            else: e, start = rng.randint(self.n), rng.randint(lo, hi)
            if self.valid(e, start, length): picks.append((e, start))
            if len(picks) == batch: break
        if len(picks) < max(1, batch // 4): return None
        es = np.array([e for e, t in picks]); ts = np.array([t for e, t in picks])
        ix = (ts[:, None] + np.arange(length)) % self.cap
        out = {k: torch.as_tensor(np.ascontiguousarray(v[es[:, None], ix].swapaxes(0, 1)), device=self.device) for k, v in self.data.items()}
        lab = np.array([[self.labels(e, t + h) for h in range(length)] for e, t in picks], np.float32)
        out['collision'] = torch.as_tensor(lab.swapaxes(0, 1).copy(), device=self.device)
        out['burn'] = burn
        return out

    def states(self, count):
        if not self.clock: return np.zeros((0, OBS), np.float32)
        ts = np.random.randint(max(0, self.clock - self.cap), self.clock, count)
        es = np.random.randint(self.n, size=count)
        return self.data['obs'][es, ts % self.cap]

    def state_dict(self):
        # Only occupied physical slots; the stamp restores exact ring position.
        size = min(self.cap, self.clock)
        return dict(n=self.n, cap=self.cap, clock=self.clock, stamp=self.stamp[:size].copy(),
                    episode=self.episode.copy(), contact_refs=list(self.contact_refs),
                    data={k: v[:, :size].copy() for k, v in self.data.items()})

    def load_state_dict(self, state):
        if (state['n'], state['cap']) != (self.n, self.cap): raise ValueError('Replay/environment layout changed on resume')
        self.clock = state['clock']; self.episode[:] = state['episode']; self.stamp.fill(-1)
        size = len(state['stamp']); self.stamp[:size] = state['stamp']
        for k, v in state['data'].items(): self.data[k][:, :size] = v
        self.contact_refs = deque(state['contact_refs'], maxlen=self.contact_refs.maxlen)
