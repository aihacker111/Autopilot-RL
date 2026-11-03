import numpy as np
from v5_replay import SequenceReplay
from distill import OBS
from tdmpc2 import AUXD, GD


def add(buf, done=False, terminal=False, contact=False):
    t = buf.clock
    aux = np.zeros((buf.n, AUXD), np.float32); aux[:, 0] = contact
    buf.add(obs=np.full((buf.n, OBS), t, np.float32), next_obs=np.full((buf.n, OBS), t + 1, np.float32),
            previous=np.zeros((buf.n, 2), np.float32), request=np.zeros((buf.n, 2), np.float32),
            executed=np.zeros((buf.n, 2), np.float32), reward=np.ones(buf.n), done=np.full(buf.n, done),
            terminal=np.full(buf.n, terminal), aux=aux, genome=np.zeros((buf.n, GD), np.float32),
            planner=np.zeros((buf.n, 2), np.float32), planner_weight=np.ones(buf.n))


def test_unknown_future_is_not_negative():
    b = SequenceReplay(1, 64)
    for _ in range(3): add(b)
    assert (b.labels(0, 0) == -1).all()
    add(b, done=True, terminal=False)
    assert (b.labels(0, 0) == -1).all()  # truncated, not known-safe
    add(b, done=True, terminal=True)
    assert (b.labels(0, 4) == 0).all()


def test_contact_labels_stop_at_episode_and_are_positive_before_contact():
    b = SequenceReplay(1, 64)
    for _ in range(6): add(b)
    add(b, done=True, terminal=True, contact=True)
    assert b.labels(0, 0).tolist() == [0, 1, 1]
    add(b, done=True, terminal=True)
    assert b.labels(0, 7).tolist() == [0, 0, 0]
    assert not b.valid(0, 5, 4)


def test_wraparound_keeps_absolute_priority_indices_and_drops_stale_refs():
    b = SequenceReplay(1, 32)
    for t in range(100): add(b, done=t in (5, 90), terminal=t in (5, 90), contact=t in (5, 90))
    assert list(b.contact_refs) == [(0, 90)]
    assert not b.valid(0, 0, 4)
    for _ in range(5):
        batch = b.sample(16, 4, burn=2, priority=1)
        values = batch['obs'][:, :, 0].cpu().numpy()
        np.testing.assert_array_equal(np.diff(values, axis=0), np.ones_like(values[1:]))
        assert not batch['done'][:-1].any()
    restored = SequenceReplay(1, 32); restored.load_state_dict(b.state_dict())
    np.testing.assert_array_equal(restored.data['obs'], b.data['obs'])
    assert restored.clock == b.clock
