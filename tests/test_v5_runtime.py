import copy
import json
from pathlib import Path
import random
import subprocess
import numpy as np
import torch
from train import VecEnv
from tdmpc2 import GSPACE
from v5_agent import Agent
from v5_train import load_config
from v5_eval import ROOT, isolated, geographic_split, build_manifest, rank


def config(): return load_config(ROOT / 'configs/v5_smoke.yaml')


def test_worker_snapshot_restores_rng_contacts_and_counterfactuals():
    env = VecEnv(1, 1, seed=14, extra=dict(genomeSpace=GSPACE, reward='v5', preset='city', shield=True,
                 tracking=True, dagger=.1, cf=1., cfLen=4))
    try:
        env.reset()
        for _ in range(3): env.step(np.array([[.2, .1]], np.float32))
        snap = env.snapshot()
        first = []
        for _ in range(6):
            obs, rew, done, infos = env.step(np.array([[.3, -.2]], np.float32))
            first.append((obs.copy(), rew.copy(), done.copy(), copy.deepcopy(infos), copy.deepcopy(env.cf)))
        env.restore(snap)
        for expected in first:
            obs, rew, done, infos = env.step(np.array([[.3, -.2]], np.float32))
            for actual, exp in zip((obs, rew, done), expected[:3]): np.testing.assert_array_equal(actual, exp)
            assert infos == expected[3] and env.cf == expected[4]
    finally: env.close()


def test_exported_recurrent_policy_matches_python(tmp_path):
    cfg = config(); torch.manual_seed(2); agent = Agent(cfg, 'cpu'); agent.export(tmp_path / 'policy.json')
    rng = np.random.RandomState(17); rows = []
    for i in range(8):
        obs = rng.normal(0, .1, (1, 299)).astype(np.float32)
        for j in range(16): obs[0, 43 + j * 16 + 15] = int(j < i)
        executed = rng.uniform(-1, 1, (1, 2)).astype(np.float32)
        if i == 4: agent.reset()
        action = agent.act(obs, planner=False)[0]
        agent.observe_execution(executed, np.array([False]))
        rows.append(dict(obs=obs[0].tolist(), executed=executed[0].tolist(), expected=action.tolist(), reset=i == 4))
    (tmp_path / 'rows.json').write_text(json.dumps(rows))
    script = '''require('./rl/common.js');const fs=require('fs');const p=makePolicy(JSON.parse(fs.readFileSync(process.argv[1])));const rows=JSON.parse(fs.readFileSync(process.argv[2]));console.log(JSON.stringify(rows.map(r=>{if(r.reset)p.reset();const a=p.act(r.obs);p.observeExecution(r.executed);return a;})));'''
    result = subprocess.check_output([__import__('os').environ['LOTUS_NODE'], '-e', script, str(tmp_path / 'policy.json'), str(tmp_path / 'rows.json')], cwd=ROOT, text=True)
    np.testing.assert_allclose(json.loads(result), [r['expected'] for r in rows], atol=2e-5, rtol=2e-5)


def test_evaluation_isolation_restores_rng_and_recurrent_state():
    torch.manual_seed(3); agent = Agent(config(), 'cpu'); agent.act(np.zeros((1, 299), np.float32), planner=False)
    hidden = agent.h.clone(); state = torch.get_rng_state().clone(); ns = np.random.get_state(); ps = random.getstate()
    with isolated(agent, 99):
        agent.act(np.ones((1, 299), np.float32), planner=False); torch.rand(12); np.random.rand(); random.random()
    assert torch.equal(hidden, agent.h) and torch.equal(state, torch.get_rng_state())
    assert np.array_equal(ns[1], np.random.get_state()[1]) and ps == random.getstate()


def test_split_halos_and_safety_ranking():
    manifest = build_manifest()
    groups = manifest['centres']
    for split, points in groups.items():
        assert points
        for x, y in points:
            assert geographic_split(x, y) == split
            assert min(x % 8000, 8000-x % 8000, y % 8000, 8000-y % 8000) >= manifest['half'] + 400
    cfg = config()
    safe = dict(person_contacts=0, contact_per_km=0, red=0, ok=.8, route_progress=.9, shield_per_km=1, mean_jerk=1)
    risky = dict(safe, person_contacts=1, ok=1.)
    assert rank(safe, cfg) > rank(risky, cfg)


def test_counterfactual_does_not_mutate_live_state_and_keeps_learned_brain():
    script = '''require('./rl/common.js');const assert=require('assert');const e=new EgoEnv({seed:31,mapOpts:{preset:'city'},reward:'v5',tracking:true});e.reset();e.step([.1,0]);const state=JSON.stringify({ep:e.ep,rand:e.rand.st,simrand:e.sim.rand.st,x:e.ego.x,y:e.ego.y,slots:e.trackSlots});const marker={types:{}};e.sim.brain=marker;assert.strictEqual(e.sim.cloneLocal(e.ego.x,e.ego.y,45).brain,marker);e.branch([1,1],6);assert.strictEqual(JSON.stringify({ep:e.ep,rand:e.rand.st,simrand:e.sim.rand.st,x:e.ego.x,y:e.ego.y,slots:e.trackSlots}),state);console.log('ok');'''
    assert subprocess.check_output([__import__('os').environ['LOTUS_NODE'], '-e', script], cwd=ROOT, text=True).strip() == 'ok'


def test_hcm_cache_seed_and_snapshot_across_new_maps():
    manifest = build_manifest(); centre = manifest['centres']['train'][0]
    env = VecEnv(1, 1, seed=44, extra=dict(genomeSpace=GSPACE, reward='v5', preset='hcm',
        allowedCentres=[centre], half=600, reseedEvery=2, shield=True, tracking=True))
    try:
        env.reset(); snap = env.snapshot(); expected = []
        for _ in range(4):
            env.ws[0].call({'cmd':'reset'})
            expected.append(env.step(np.array([[.2,0]], np.float32))[0])
        env.restore(snap)
        for row in expected:
            env.ws[0].call({'cmd':'reset'})
            actual = env.step(np.array([[.2,0]], np.float32))[0]
            np.testing.assert_array_equal(actual,row)
    finally: env.close()
