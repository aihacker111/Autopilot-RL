import copy
import json
from pathlib import Path
import numpy as np
import torch
from v5_train import load_config, train
from v5_agent import Agent
from v5_replay import SequenceReplay
from v5_eval import ROOT, build_manifest, evaluate
from v5_curriculum import Curriculum
from train import VecEnv
from tdmpc2 import GSPACE
from test_v5_replay import add


def cfg(): return load_config(ROOT / 'configs/v5_smoke.yaml')


def test_full_training_resume_matches_uninterrupted(tmp_path):
    first = dict(cfg(), out=str(tmp_path / 'split'), steps=24)
    train(first)
    train(dict(first, steps=48), str(tmp_path / 'split/latest.pt'))
    train(dict(first, steps=48, out=str(tmp_path / 'continuous')))
    a = torch.load(tmp_path / 'split/latest.pt', weights_only=False)
    b = torch.load(tmp_path / 'continuous/latest.pt', weights_only=False)
    for name in ('model', 'dynamics', 'collision', 'target_q'):
        for k in a['agent'][name]: assert torch.equal(a['agent'][name][k], b['agent'][name][k]), (name, k)
    assert a['agent']['updates'] == b['agent']['updates'] > 0
    for k in ('obs', 'previous', 'genome'): np.testing.assert_array_equal(a['state'][k], b['state'][k])
    assert torch.equal(a['rng']['torch'], b['rng']['torch'])
    # Serializer alias tables can differ (newly spawned immutable vehicle type
    # objects); compare future behavior, not opaque serialization bytes.
    left = VecEnv(1, 1, extra=dict(genomeSpace=GSPACE, preset='city', reward='v5'))
    right = VecEnv(1, 1, extra=dict(genomeSpace=GSPACE, preset='city', reward='v5'))
    try:
        left.restore(a['environment']); right.restore(b['environment'])
        for _ in range(12):
            action = np.array([[.3, -.2]], np.float32)
            l = left.step(action); r = right.step(action)
            for x, y in zip(l[:3], r[:3]): np.testing.assert_array_equal(x, y)
            assert l[3] == r[3] and left.cf == right.cf
    finally: left.close(); right.close()


def test_four_controller_modes_and_real_full_planner_curriculum(tmp_path):
    c = dict(cfg(), eval_max_decisions=12, eval_batch=1,
             evo_candidates=2, evo_states=2, dream_batch=2, dream_horizon=2, evo_screen=2,
             evo_real=1, evo_retry=1, evo_prior_episodes=1, evo_min_episodes=1,
             evo_max_episodes=2, evo_teacher_episodes=1, realism_min=0.)
    agent = Agent(c, 'cpu'); suite = [dict(seed=17, preset='city', density=.5, maxTime=.4)]
    for planner in (False, True):
        for shield in (False, True):
            summary, rows = evaluate(agent, suite, c, planner, shield)
            assert summary['episodes'] == 1 and rows[0]['result'] == 'timeout'
            assert 'realismStats' in rows[0] and 'collision_brier' in summary
    replay = SequenceReplay(1, 64)
    for _ in range(20): add(replay)
    manifest = build_manifest(generated=True); curriculum = Curriculum(c, manifest)
    original = curriculum.suite
    def short(genome, n, offset=0):
        out = original(genome, n, offset)
        for row in out: row.update(preset='city', maxTime=.4)
        return out
    curriculum.suite = short
    env = VecEnv(1, 1, 19, extra=dict(genomeSpace=GSPACE, preset='city', reward='v5'))
    try:
        env.reset(); report = curriculum.update(agent, replay, env, 20)
        assert report['checked'] == 1 and report['generation'] == 1
        assert len(curriculum.calibration) == 1
        restored = Curriculum(c, manifest); restored.load_state_dict(curriculum.state_dict())
        assert restored.pending == curriculum.pending
    finally: env.close()


def test_validation_saves_best_and_resume_checks_schema(tmp_path, monkeypatch):
    import v5_train
    def short_specs(manifest, split, count):
        assert split == 'validation'
        return [dict(seed=23, preset='city', density=.5, maxTime=.6)]
    monkeypatch.setattr(v5_train,'specs',short_specs)
    c = dict(cfg(), out=str(tmp_path / 'run'), steps=16, eval_every=16, eval_episodes=1)
    result = train(c)
    assert result['best_validation']['split'] == 'validation'
    checkpoint = torch.load(tmp_path / 'run/best.pt', weights_only=False)
    assert checkpoint['state']['last_eval'] == 16
    assert (tmp_path / 'run/best_policy.json').exists()
    import pytest
    with pytest.raises(ValueError, match='Resume config changed'):
        train(dict(c, latent=40), str(tmp_path / 'run/latest.pt'))
