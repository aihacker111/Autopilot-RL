"""LOTUS v5 training/evaluation entry point. Trusted local checkpoints only.

python rl/v5_train.py --config configs/v5_full.yaml
python rl/v5_train.py --config configs/v5_full.yaml --resume rl/runs/lotus_v5_full/latest.pt
python rl/v5_train.py --config configs/v5_smoke.yaml
python rl/v5_train.py --evaluate RUN/best.pt --matrix --split test
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
from collections import deque
import numpy as np
import torch
import yaml
from train import VecEnv
from tdmpc2 import GSPACE, GD
from distill import OBS
from v5_replay import SequenceReplay
from v5_agent import Agent
from v5_eval import ROOT, build_manifest, specs, evaluate, rank, aggregate
from v5_curriculum import Curriculum

# The UI/HTTP helpers live outside the training-source fingerprint. A pinned
# migration below permits this observability-only change for existing runs.
sys.path.insert(0, str(ROOT / 'monitor'))
from v5 import TrainingMonitor, compatible_sources

SCHEMA = 5


def atomic_json(path, data):
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)); tmp.replace(path)


def fingerprint():
    paths = [ROOT / 'pipeline.py', ROOT / 'requirements.txt']
    for directory, pattern in [('rl', '*.py'), ('rl', '*.js'), ('web', '*.js')]: paths += sorted((ROOT / directory).glob(pattern))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def map_fingerprint():
    base = ROOT / 'web/hcm'; index = json.loads((base / 'index.json').read_text()); digest = hashlib.sha256()
    for path in [base / 'index.json', *[base / 't' / f'{k}.json' for k in sorted(index['tiles'])]]:
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    return digest.hexdigest()


def metadata(cfg, manifest):
    try: commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError): commit = None
    node = os.environ.get('LOTUS_NODE') or shutil.which('node') or '/opt/homebrew/bin/node'
    return dict(schema=SCHEMA, observation=OBS, action='target speed and lateral position; executed action conditions dynamics',
                git=commit, files=fingerprint(), config=cfg, manifest_hash=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                versions=dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, yaml=yaml.__version__,
                              node=subprocess.check_output([node, '--version'], text=True).strip()),
                priors_hash=hashlib.sha256((ROOT / cfg['priors']).read_bytes()).hexdigest(),
                map_hash=map_fingerprint() if not cfg['generated_maps'] else None,
                npc_hash=hashlib.sha256((ROOT / cfg['npc_policy']).read_bytes()).hexdigest() if cfg['traffic'] == 'learned' else None)


def teacher(env, mask=None):
    actions = []
    for w in env.ws:
        m = mask[len(actions):len(actions) + w.n].tolist() if mask is not None else None
        actions += w.call(dict(cmd='teacher', mask=m))['actions']
    return actions


def load_config(path):
    defaults = yaml.safe_load((ROOT / 'configs/v5_full.yaml').read_text())['v5']
    custom = yaml.safe_load(Path(path).read_text())['v5']
    unknown = set(custom) - set(defaults)
    if unknown: raise ValueError(f'Unknown v5 config settings: {sorted(unknown)}')
    cfg = dict(defaults, **custom)
    if sys.version_info < (3, 12): raise RuntimeError('LOTUS v5 requires Python 3.12+')
    if cfg['traffic'] not in ('rules', 'learned'): raise ValueError('traffic must be rules or learned')
    if not cfg['generated_maps'] and not (ROOT / 'web/hcm/index.json').is_file(): raise FileNotFoundError('HCM map missing; install web/hcm or set generated_maps: true')
    if cfg['latent'] % 8: raise ValueError('latent must be divisible by SimNorm group size 8')
    for key in ('workers', 'envs', 'batch', 'horizon', 'samples', 'ensemble', 'eval_batch'):
        if cfg[key] < 1: raise ValueError(f'{key} must be positive')
    if cfg['export_every'] <= 0 or cfg['checkpoint_every'] <= 0 or cfg['log_every'] <= 0: raise ValueError('export/checkpoint/log intervals must be positive')
    if cfg['ensemble'] > cfg['futures']: raise ValueError('Use at least one imagined future per ensemble member')
    if len(cfg['routes']) != len(cfg['route_weights']) or min(cfg['route_weights']) < 0 or sum(cfg['route_weights']) <= 0: raise ValueError('Invalid route distribution')
    if cfg['counterfactual_length'] < cfg['horizon'] + cfg['burn_in']: raise ValueError('Counterfactual branch shorter than training sequence')
    return cfg


def device_for(name):
    if name == 'auto': return 'cuda' if torch.cuda.is_available() else 'cpu'
    return name


def save_checkpoint(path, cfg, meta, agent, replay, cf_replay, curriculum, env, state):
    rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
               cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    payload = dict(schema=SCHEMA, cfg=cfg, metadata=meta, agent=agent.state_dict(), replay=replay.state_dict(),
                   counterfactual=cf_replay.state_dict(), curriculum=curriculum.state_dict(), environment=env.snapshot(),
                   rng=rng, state=state, manifest=curriculum.manifest)
    path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp'); torch.save(payload, tmp); tmp.replace(path)


def restore_checkpoint(payload, agent, replay, cf_replay, curriculum, env):
    agent.load_state_dict(payload['agent']); replay.load_state_dict(payload['replay']); cf_replay.load_state_dict(payload['counterfactual'])
    curriculum.load_state_dict(payload['curriculum']); env.restore(payload['environment'])
    rng = payload['rng']; random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['torch'].cpu())
    if rng['cuda'] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all([x.cpu() for x in rng['cuda']])
    return payload['state']


def add_counterfactual(buffer, initial_obs, previous, genome, branch):
    # Each branch is an independent episode. Short/censored branches keep unknown
    # future labels and are never padded with fictitious absorbing transitions.
    obs = initial_obs.copy(); prev = previous.copy()
    for j, st in enumerate(branch):
        no = np.asarray(st[0], np.float32); executed = np.asarray(st[1:3], np.float32)
        aux = np.array([st[6], *st[7:10], *executed, 0., 0., 0., 0.], np.float32)
        last = bool(st[4]) or j == len(branch) - 1
        buffer.add(obs=obs[None], next_obs=no[None], previous=prev[None], request=executed[None], executed=executed[None],
                   reward=[st[3]], done=[last], terminal=[bool(st[5])], aux=aux[None], genome=genome[None], planner=executed[None], planner_weight=[0.])
        obs, prev = no, executed


def train(cfg, resume=None):
    cfg = dict(cfg); cfg['device'] = device_for(cfg['device']); torch.set_num_threads(cfg['threads'])
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    out = ROOT / cfg['out']; out.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(resume, map_location=cfg['device'], weights_only=False) if resume else None
    if not resume and ((out / 'latest.pt').exists() or (out / 'metadata.json').exists()): raise FileExistsError(f'{out} already contains a run; use --resume or a new --out')
    manifest_path = out / 'splits.json'
    manifest = checkpoint['manifest'] if checkpoint else build_manifest(cfg['split_seed'], cfg['half'], cfg['generated_maps'])
    meta = metadata(cfg, manifest)
    if checkpoint:
        if checkpoint.get('schema') != SCHEMA: raise ValueError('Incompatible checkpoint schema')
        mutable = {'steps', 'device', 'log_every', 'export_every', 'checkpoint_every', 'eval_every', 'out'}
        changed = {k for k, v in cfg.items() if k not in mutable and checkpoint['cfg'].get(k) != v}
        if changed: raise ValueError(f'Resume config changed: {sorted(changed)}')
        if not compatible_sources(checkpoint['metadata']['files'], meta['files']): raise ValueError('Source changed since checkpoint; exact resume refused (start a new experiment)')
        for key in ('manifest_hash', 'priors_hash', 'map_hash', 'npc_hash'):
            if checkpoint['metadata'].get(key) != meta[key]: raise ValueError(f'{key} changed on resume')
    if checkpoint and checkpoint['metadata']['files'] != meta['files']:
        atomic_json(out / 'resume_monitor_migration.json', dict(previous=checkpoint['metadata']['files'], current=meta['files'], reason='Pinned observability-only monitor integration'))
    if checkpoint and checkpoint['metadata']['versions'] != meta['versions']: raise ValueError('Runtime versions changed; exact resume refused')
    if not checkpoint or not manifest_path.exists():
        atomic_json(manifest_path, manifest); atomic_json(out / 'metadata.json', meta)
        (out / 'config.yaml').write_text(yaml.safe_dump({'v5': cfg}, sort_keys=False))
    if cfg['traffic'] == 'learned':
        path = ROOT / cfg['npc_policy']
        if not path.is_file(): raise FileNotFoundError(f'Learned NPC policy missing: {path}')
        os.environ['LOTUS_NPC'] = str(path)
    else: os.environ.pop('LOTUS_NPC', None)
    N = cfg['workers'] * cfg['envs']
    extra = dict(genomeSpace=GSPACE, reward='v5', shield=cfg['shield'], lateralShield=cfg['lateral_shield'], tracking=cfg['tracking'],
                 routeMs=cfg['routes'], routeW=cfg['route_weights'], half=cfg['half'], reseedEvery=cfg['reseed'],
                 dagger=cfg['teacher_probability'], cf=cfg['counterfactual_probability'], cfLen=cfg['counterfactual_length'],
                 allowedCentres=manifest['centres']['train'])
    if cfg['generated_maps']: extra['preset'] = 'city'
    env = VecEnv(cfg['workers'], cfg['envs'], cfg['seed'], density_range=[.4, 1.1], extra=extra)
    agent = Agent(cfg, cfg['device']); replay = SequenceReplay(N, cfg['replay_capacity'], cfg['device'])
    cf_replay = SequenceReplay(1, cfg['counterfactual_capacity'], cfg['device']); curriculum = Curriculum(cfg, manifest)
    state = dict(steps=0, obs=None, previous=np.zeros((N, 2), np.float32), genome=None, credit=0., best=None, best_validation=None,
                 history=[], last_eval=0, last_checkpoint=0, last_export=0, last_log=0, last_evo=0)
    status = 'failed'; t0 = time.monotonic(); stats = {}
    try:
        if checkpoint: state = restore_checkpoint(checkpoint, agent, replay, cf_replay, curriculum, env)
        else: state['obs'] = env.reset(); state['genome'] = env.gen.copy()
        agent.export(out / 'policy.json')  # visible before the first scheduled export
        atomic_json(out / 'status.json', dict(phase='train', steps=state['steps'], total=cfg['steps'], t=time.time()))
        baseline_step = state['steps']; reports = deque(state['history'], maxlen=200)
        while state['steps'] < cfg['steps']:
            obs, previous, genome = state['obs'], state['previous'], state['genome']
            warm = state['steps'] < cfg['teacher_steps']
            actions = agent.act(obs, planner=not warm, explore=True)
            labels = None
            if warm:
                labels = np.asarray(teacher(env), np.float32); actions = labels
            else:
                ask = (agent.last_uncertainty > cfg['uncertainty_threshold']) & (np.random.rand(N) < cfg['teacher_uncertainty_probability'])
                if ask.any():
                    raw = teacher(env, ask)
                    labels = np.array([x if x is not None else [0., 0.] for x in raw], np.float32)
                else: ask = np.zeros(N, bool)
            next_obs, rewards, done, infos = env.step(actions)
            terminal_obs = next_obs.copy(); terminal = np.zeros(N, bool)
            for i, info in enumerate(infos):
                if info:
                    terminal_obs[i] = np.asarray(info['terminal'], np.float32)
                    terminal[i] = info['result'] != 'timeout'  # 'stage' is a boundary between independent geographic segments
                    if info['result'] != 'stage': reports.append({k: v for k, v in info.items() if k not in ('terminal', 'replay')})
            aux = env.aux.copy()
            if labels is not None:
                mask = np.ones(N, bool) if warm else ask
                aux[mask, 6:8] = labels[mask]; aux[mask, 8] = 1.
            weights = agent.last_weight.copy()
            weights[warm | (np.abs(actions - aux[:, 4:6]).sum(-1) > .05)] = 0.
            if state['steps'] < cfg['learning_starts']: weights[:] = 0.
            replay.add(obs=obs, next_obs=terminal_obs, previous=previous, request=actions, executed=aux[:, 4:6], reward=rewards,
                       done=done, terminal=terminal, aux=aux, genome=genome, planner=agent.last_plan, planner_weight=weights)
            for branch in env.cf: add_counterfactual(cf_replay, obs[branch['i']], previous[branch['i']], genome[branch['i']], branch['steps'])
            agent.observe_execution(aux[:, 4:6], done)
            state.update(steps=state['steps'] + N, obs=next_obs, previous=aux[:, 4:6].copy(), genome=env.gen.copy())
            state['previous'][done] = 0
            if state['steps'] >= cfg['learning_starts']:
                state['credit'] += N * cfg['updates_per_transition']
                while state['credit'] >= 1:
                    source = cf_replay if cf_replay.clock >= cfg['burn_in'] + cfg['horizon'] and np.random.rand() < cfg['counterfactual_fraction'] else replay
                    batch = source.sample(cfg['batch'], cfg['horizon'], cfg['burn_in'], cfg['priority_fraction'])
                    if batch is not None: stats = agent.update(batch)
                    state['credit'] -= 1
            steps = state['steps']; agent.steps = steps
            if cfg['evo_every'] and steps >= cfg['evo_start'] and steps - state['last_evo'] >= cfg['evo_every']:
                print(json.dumps({'curriculum': curriculum.update(agent, replay, env, steps)}), flush=True)
                state['last_evo'] = steps; atomic_json(out / 'curriculum.json', curriculum.state_dict() | {'rng': None})
            if cfg['eval_every'] and steps - state['last_eval'] >= cfg['eval_every']:
                report, episodes = evaluate(agent, specs(manifest, 'validation', cfg['eval_episodes']), cfg, planner=True, shield=cfg['shield'])
                report.update(steps=steps, split='validation'); atomic_json(out / f'validation_{steps}.json', dict(summary=report, episodes=episodes))
                state['last_eval'] = steps
                score = rank(report, cfg)
                if state['best'] is None or score > tuple(state['best']):
                    state.update(best=score, best_validation=report)
                    # Save the full training state; the same file can be evaluated or resumed.
                    state['history'] = list(reports)
                    save_checkpoint(out / 'best.pt', cfg, meta, agent, replay, cf_replay, curriculum, env, state)
                    agent.export(out / 'best_policy.json')
                print(json.dumps({'validation': report}), flush=True)
            if steps - state['last_export'] >= cfg['export_every']:
                agent.export(out / 'policy.json'); state['last_export'] = steps
            if steps - state['last_log'] >= cfg['log_every']:
                state['last_log'] = steps; report = aggregate(list(reports)) if reports else {}
                row = dict(steps=steps, updates=agent.updates, seconds=time.monotonic() - t0,
                           transitions_per_second=(steps - baseline_step) / max(.001, time.monotonic() - t0), **stats, **report)
                print(json.dumps(row), flush=True)
                with (out / 'metrics.jsonl').open('a') as f: f.write(json.dumps(row) + '\n')
                atomic_json(out / 'status.json', dict(phase='train', total=cfg['steps'], t=time.time(), **row))
            if steps - state['last_checkpoint'] >= cfg['checkpoint_every']:
                state['last_checkpoint'] = steps; state['history'] = list(reports)
                save_checkpoint(out / 'latest.pt', cfg, meta, agent, replay, cf_replay, curriculum, env, state)
        state['history'] = list(reports)
        save_checkpoint(out / 'latest.pt', cfg, meta, agent, replay, cf_replay, curriculum, env, state)
        agent.export(out / 'policy.json')
        result = dict(steps=state['steps'], updates=agent.updates, best_validation=state['best_validation'],
                      test_evaluated=False, checkpoint=str(out / 'latest.pt'), realism_source=curriculum.realism_source,
                      journey_semantics=manifest['journey_semantics'])
        atomic_json(out / 'result.json', result); status = 'done'; return result
    except KeyboardInterrupt:
        # A signal can arrive halfway through a simulator RPC/update. Only the last
        # committed checkpoint is an exact resumable transaction.
        atomic_json(out / 'interrupted.json', dict(last_committed_checkpoint=str(out / 'latest.pt'), steps_seen=state['steps']))
        raise
    finally:
        env.close()
        final_status = json.loads((out / 'status.json').read_text()) if (out / 'status.json').exists() else {}
        atomic_json(out / 'status.json', dict(final_status, phase=status, steps=state['steps'], total=cfg['steps'], t=time.time()))


def evaluate_checkpoint(path, split, matrix, episodes=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    cfg = checkpoint['cfg']; cfg['device'] = device_for('auto'); torch.set_num_threads(cfg['threads'])
    agent = Agent(cfg, cfg['device']); agent.load_state_dict(checkpoint['agent']); agent.reset()
    out = Path(path).resolve().parent; manifest = json.loads((out / 'splits.json').read_text())
    if cfg['traffic'] == 'learned': os.environ['LOTUS_NPC'] = str(ROOT / cfg['npc_policy'])
    else: os.environ.pop('LOTUS_NPC', None)
    reports = {}
    for planner, shield in ([(p, s) for p in (False, True) for s in (False, True)] if matrix else [(True, True)]):
        name = f'{"planner" if planner else "prior"}_shield{int(shield)}'
        summary, rows = evaluate(agent, specs(manifest, split, episodes or cfg['eval_episodes']), cfg, planner, shield)
        reports[name] = dict(summary=summary, episodes=rows)
        print(json.dumps({name: summary}), flush=True)
    target = out / f'evaluation_{split}_{Path(path).stem}.json'; atomic_json(target, reports)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default=str(ROOT / 'configs/v5_full.yaml'))
    parser.add_argument('--seed', type=int); parser.add_argument('--resume'); parser.add_argument('--out'); parser.add_argument('--steps', type=int); parser.add_argument('--device')
    parser.add_argument('--evaluate'); parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--port', type=int, default=8765, help='training monitor port; 0 disables the HTTP server')
    parser.add_argument('--bind', default='0.0.0.0', help='monitor listen address')
    parser.add_argument('--keep-monitor', action='store_true', help='keep the monitor after training ends until Ctrl+C')
    parser.add_argument('--matrix', action='store_true'); parser.add_argument('--episodes', type=int)
    args = parser.parse_args()
    if args.evaluate: evaluate_checkpoint(args.evaluate, args.split, args.matrix, args.episodes); return
    cfg = load_config(args.config)
    for key in ('out', 'steps', 'device', 'seed'):
        if getattr(args, key) is not None: cfg[key] = getattr(args, key)
    with TrainingMonitor(ROOT / cfg['out'], args.port, args.bind, cfg['generated_maps']) as monitor:
        try: print(json.dumps(train(cfg, args.resume)), flush=True)
        finally:
            if args.keep_monitor: monitor.keep_open()


if __name__ == '__main__': main()
