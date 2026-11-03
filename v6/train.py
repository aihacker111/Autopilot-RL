"""LOTUS v6. New experiments use --init-from for weights, --resume only for v6.
python train_v6.py --config configs/v6_full.yaml
python train_v6.py --config configs/v6_full.yaml --resume rl/runs/lotus_v6_full/latest.pt
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
from collections import deque
import numpy as np
import torch
import yaml
from train import VecEnv
from tdmpc2 import GSPACE
from v5_train import atomic_json, metadata as v5_metadata, device_for, teacher
from v5_eval import ROOT, build_manifest, specs, aggregate, rank
from v5_curriculum import Curriculum
from v5 import TrainingMonitor
from v6.config import load_config
from v6.agent import Agent
from v6.replay import Replay, Demonstrations, sample_mixture
from v6.teacher import Handover, evaluate_teacher
from v6.evaluation import evaluate

SCHEMA = 6


def metadata(cfg, manifest):
    out = v5_metadata(cfg, manifest)
    out.update(schema=SCHEMA, action='physical model: executed; Q: requested + shield flag; reward_Q = physical_reward - intervention_cost')
    for path in [ROOT/'train_v6.py', *sorted((ROOT/'v6').glob('*.py'))]:
        out['files'][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def save_checkpoint(path, cfg, meta, agent, replay, cf_replay, demos, handover, curriculum, env, state):
    payload = dict(schema=SCHEMA, cfg=cfg, metadata=meta, manifest=curriculum.manifest, agent=agent.state_dict(), replay=replay.state_dict(),
                   counterfactual=cf_replay.state_dict(), demonstrations=demos.state_dict(), handover=handover.state_dict(),
                   curriculum=curriculum.state_dict(), environment=env.snapshot(), state=state,
                   rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None))
    path = Path(path); tmp = path.with_suffix(path.suffix+'.tmp'); torch.save(payload,tmp); tmp.replace(path)


def restore_checkpoint(payload, agent, replay, cf_replay, demos, handover, curriculum, env):
    agent.load_state_dict(payload['agent']); replay.load_state_dict(payload['replay']); cf_replay.load_state_dict(payload['counterfactual'])
    demos.load_state_dict(payload['demonstrations']); handover.load_state_dict(payload['handover']); curriculum.load_state_dict(payload['curriculum'])
    env.restore(payload['environment']); rng = payload['rng']
    random.setstate(rng['python']); np.random.set_state(rng['numpy']); torch.set_rng_state(rng['torch'].cpu())
    if rng['cuda'] is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all([x.cpu() for x in rng['cuda']])
    return payload['state']


def add_counterfactual(buffer, initial_obs, previous, genome, branch):
    obs, prev = initial_obs.copy(), previous.copy()
    for j, st in enumerate(branch):
        no, ex = np.asarray(st[0],np.float32), np.asarray(st[1:3],np.float32)
        aux = np.asarray([st[6],*st[7:10],*ex,0.,0.,0.,0.],np.float32)
        buffer.add(obs=obs[None], next_obs=no[None], previous=prev[None], request=ex[None], executed=ex[None], reward=[st[3]],
                   done=[bool(st[4]) or j==len(branch)-1], terminal=[bool(st[5])], aux=aux[None], genome=genome[None],
                   planner=ex[None], planner_weight=[0.], teacher_weight=[0.], shield_enabled=[False], source=[2.])
        obs, prev = no, ex


def validate_resume(checkpoint, cfg, meta):
    if checkpoint.get('schema') != SCHEMA: raise ValueError('V6 exact resume requires schema 6; use --init-from for v5 weights in a new run')
    mutable = {'steps','device','log_every','export_every','checkpoint_every','eval_every','out'}
    changed = [k for k,v in cfg.items() if k not in mutable and checkpoint['cfg'].get(k)!=v]
    if changed: raise ValueError(f'Resume config changed: {changed}')
    for k in ('files','versions','manifest_hash','priors_hash','map_hash','npc_hash'):
        if checkpoint['metadata'].get(k) != meta.get(k): raise ValueError(f'V6 exact resume refused: {k} changed')


def train(cfg, resume=None, init_from=None):
    if resume and init_from: raise ValueError('--resume and --init-from are mutually exclusive')
    cfg=dict(cfg); cfg['device']=device_for(cfg['device']); torch.set_num_threads(cfg['threads'])
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8'); torch.use_deterministic_algorithms(True)
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    out=ROOT/cfg['out']; out.mkdir(parents=True,exist_ok=True)
    checkpoint=torch.load(resume,map_location=cfg['device'],weights_only=False) if resume else None
    if not resume and ((out/'metadata.json').exists() or (out/'latest.pt').exists()): raise FileExistsError('Use a new v6 run directory or --resume')
    if checkpoint and checkpoint.get('schema')!=SCHEMA: raise ValueError('V5 checkpoints require --init-from, not --resume')
    manifest=checkpoint['manifest'] if checkpoint else build_manifest(cfg['split_seed'],cfg['half'],cfg['generated_maps'])
    meta=metadata(cfg,manifest)
    if checkpoint: validate_resume(checkpoint,cfg,meta)
    if not checkpoint or not (out/'metadata.json').exists():
        atomic_json(out/'metadata.json',meta); atomic_json(out/'splits.json',manifest)
        (out/'config.yaml').write_text(yaml.safe_dump({'v6':cfg},sort_keys=False))
    if cfg['traffic']=='learned':
        npc=ROOT/cfg['npc_policy']
        if not npc.is_file(): raise FileNotFoundError(npc)
        os.environ['LOTUS_NPC']=str(npc)
    else: os.environ.pop('LOTUS_NPC',None)
    N=cfg['workers']*cfg['envs']; agent=Agent(cfg,cfg['device'])
    replay=Replay(N,cfg['replay_capacity'],cfg['device']); cf=Replay(1,cfg['counterfactual_capacity'],cfg['device'])
    demos=Demonstrations(N,cfg['demo_capacity'],cfg['device']); handover=Handover(cfg); curriculum=Curriculum(cfg,manifest)
    if init_from:
        weights=torch.load(init_from,map_location=cfg['device'],weights_only=False)
        if weights.get('schema') not in (5,6): raise ValueError('Initialization requires a LOTUS v5/v6 checkpoint')
        atomic_json(out/'initialization.json',dict(source=str(init_from),**agent.initialize_weights(weights['agent'])))
    extra=dict(genomeSpace=GSPACE,reward='v5',shield=cfg['shield'],lateralShield=cfg['lateral_shield'],tracking=cfg['tracking'],
               routeMs=cfg['routes'],routeW=cfg['route_weights'],half=cfg['half'],reseedEvery=cfg['reseed'],dagger=cfg['teacher_probability'],
               cf=cfg['counterfactual_probability'],cfLen=cfg['counterfactual_length'],allowedCentres=manifest['centres']['train'])
    if cfg['generated_maps']: extra['preset']='city'
    env=VecEnv(cfg['workers'],cfg['envs'],cfg['seed'],density_range=[.4,1.1],extra=extra)
    state=dict(steps=0,obs=None,previous=np.zeros((N,2),np.float32),genome=None,credit=0.,best=None,best_validation=None,history=[],
               last_eval=0,last_checkpoint=0,last_export=0,last_log=0,last_evo=0,stats_sum={},stats_count=0,
               control=dict(decisions=0,cuts=0,stationary=0,latency=0.,teacher=0))
    phase='failed'; t0=time.monotonic(); reports=deque(maxlen=200)
    def save(name='latest.pt'):
        state['history']=list(reports)
        save_checkpoint(out/name,cfg,meta,agent,replay,cf,demos,handover,curriculum,env,state)
    def status(phase_name,**kw):
        atomic_json(out/'status.json',dict(phase=phase_name,steps=state['steps'],total=cfg['steps'],t=time.time(),teacher_probability=handover.probability,**kw))
    try:
        if checkpoint:
            state=restore_checkpoint(checkpoint,agent,replay,cf,demos,handover,curriculum,env); reports=deque(state['history'],maxlen=200)
        else: state['obs']=env.reset(); state['genome']=env.gen.copy()
        agent.export(out/'policy.json'); status('train')
        if not checkpoint and cfg['teacher_baseline_episodes']:
            status('teacher_baseline'); print('Evaluating the teacher on the fixed validation suite...',flush=True)
            report,episodes=evaluate_teacher(specs(manifest,'validation',cfg['teacher_baseline_episodes']),cfg)
            handover.baseline=report; atomic_json(out/'teacher_baseline.json',dict(summary=report,episodes=episodes))
            acceptable=report['ok']>=cfg['teacher_baseline_min_ok'] and report['contact_per_km']<=cfg['teacher_baseline_max_contact_per_km']
            print(json.dumps({'teacher_baseline':report,'acceptable':acceptable}),flush=True)
            if cfg['require_teacher_baseline'] and not acceptable:
                save(); raise RuntimeError('Teacher failed its baseline gate. Check teacher_baseline.json and improve demonstrations/scenarios before a long run.')
        initial_steps=state['steps']; status('train')
        while state['steps']<cfg['steps']:
            obs,prev,genome=state['obs'],state['previous'],state['genome']
            began=time.perf_counter()
            if cfg['device'].startswith('cuda'): torch.cuda.synchronize()
            actions=agent.act(obs,planner=handover.probability<1.,explore=True)
            drives=np.random.rand(N)<handover.probability
            asks=drives | ((agent.last_uncertainty>cfg['uncertainty_threshold']) & (np.random.rand(N)<cfg['teacher_uncertainty_probability']))
            labels=None
            if asks.any():
                raw=teacher(env,asks); labels=np.asarray([r if r is not None else [0.,0.] for r in raw],np.float32); actions[drives]=labels[drives]
            if cfg['device'].startswith('cuda'): torch.cuda.synchronize()
            latency=(time.perf_counter()-began)*1000
            new_obs,rewards,done,infos=env.step(actions); aux=env.aux.copy()
            if labels is not None: aux[asks,6:8]=labels[asks]; aux[asks,8]=1.
            terminal_obs=new_obs.copy(); terminal=np.zeros(N,bool)
            for i,info in enumerate(infos):
                if info:
                    terminal_obs[i]=np.asarray(info['terminal'],np.float32); terminal[i]=info['result'] not in ('timeout','stage')
                    if info['result']!='stage': reports.append({k:v for k,v in info.items() if k not in ('terminal','replay')})
            cuts=np.abs(actions-aux[:,4:6]).sum(-1)>.05
            weight=agent.last_weight.copy(); weight[drives | cuts]=0
            if state['steps']<cfg['learning_starts']: weight[:]=0
            quality=aux[:,8]*np.exp(-np.abs(actions-aux[:,4:6]).sum(-1)); quality[aux[:,0]>.5]=0
            row=dict(obs=obs,next_obs=terminal_obs,previous=prev,request=actions,executed=aux[:,4:6],reward=rewards,done=done,terminal=terminal,
                     aux=aux,genome=genome,planner=agent.last_plan,planner_weight=weight,teacher_weight=quality,
                     shield_enabled=np.full(N,cfg['shield'],np.float32),source=np.zeros(N,np.float32))
            replay.add(**row); demos.add(row,drives)
            for branch in env.cf: add_counterfactual(cf,obs[branch['i']],prev[branch['i']],genome[branch['i']],branch['steps'])
            agent.observe_execution(aux[:,4:6],done)
            state.update(steps=state['steps']+N,obs=new_obs,previous=aux[:,4:6].copy(),genome=env.gen.copy()); state['previous'][done]=0
            agent.steps=state['steps']; control=state['control']
            control['decisions']+=N; control['cuts']+=int(cuts.sum()); control['stationary']+=int((obs[:,0]<.05).sum())
            control['latency']+=latency; control['teacher']+=int(drives.sum())
            if state['steps']>=cfg['learning_starts']:
                state['credit']+=N*cfg['updates_per_transition']
                while state['credit']>=1:
                    batch=sample_mixture(replay,demos,cf,cfg)
                    if batch is not None:
                        metrics=agent.update(batch); state['stats_count']+=1
                        for k,v in metrics.items(): state['stats_sum'][k]=state['stats_sum'].get(k,0.)+v
                    state['credit']-=1
            steps=state['steps']
            if cfg['evo_every'] and steps>=cfg['evo_start'] and steps-state['last_evo']>=cfg['evo_every']:
                status('evolve'); print(json.dumps({'curriculum':curriculum.update(agent,replay,env,steps)}),flush=True); state['last_evo']=steps
            if cfg['eval_every'] and steps-state['last_eval']>=cfg['eval_every']:
                status('validation')
                validation,episodes=evaluate(agent,specs(manifest,'validation',cfg['eval_episodes']),cfg,True,cfg['shield'])
                readiness,_=evaluate(agent,specs(manifest,'validation',cfg['handover_episodes']),cfg,False,cfg['shield'])
                gate_report=dict(readiness,ok=min(readiness['ok'],validation['ok']),contact_per_km=max(readiness['contact_per_km'],validation['contact_per_km']),red=max(readiness['red'],validation['red']),projection_mae=max(readiness['projection_mae'],validation['projection_mae']))
                gate=handover.update(gate_report,steps); state['last_eval']=steps
                validation.update(steps=steps,split='validation')
                atomic_json(out/f'validation_{steps}.json',dict(summary=validation,episodes=episodes,prior=readiness,handover=gate))
                score=rank(validation,cfg)
                if state['best'] is None or score>tuple(state['best']):
                    state.update(best=score,best_validation=validation); save('best.pt'); agent.export(out/'best_policy.json')
                print(json.dumps({'validation':validation,'prior':readiness,'handover':gate}),flush=True); status('train')
            if steps-state['last_export']>=cfg['export_every']: agent.export(out/'policy.json'); state['last_export']=steps
            if steps-state['last_log']>=cfg['log_every']:
                averaged={k:v/max(1,state['stats_count']) for k,v in state['stats_sum'].items()}; control=state['control']; n=max(1,control['decisions'])
                line=dict(steps=steps,updates=agent.updates,seconds=time.monotonic()-t0,transitions_per_second=(steps-initial_steps)/max(.001,time.monotonic()-t0),
                          **averaged,**(aggregate(list(reports)) if reports else {}))
                line.update(teacher_probability=handover.probability,teacher_drove_fraction=control['teacher']/n,intervention_fraction=control['cuts']/n,
                            stationary_fraction=control['stationary']/n,latency_ms=control['latency']/n,demo_size=demos.size,metric_updates=state['stats_count'])
                print(json.dumps(line),flush=True)
                with (out/'metrics.jsonl').open('a') as f: f.write(json.dumps(line)+'\n')
                state['last_log']=steps; state['stats_sum']={}; state['stats_count']=0
                state['control']=dict(decisions=0,cuts=0,stationary=0,latency=0.,teacher=0); status('train',metrics=line)
            if steps-state['last_checkpoint']>=cfg['checkpoint_every']:
                state['last_checkpoint']=steps; save()
            if cfg['bootstrap_limit'] and steps>=cfg['bootstrap_limit'] and handover.probability>=1.:
                save(); raise RuntimeError('Prior has not passed competence gates within bootstrap_limit. Review validation and teacher baseline before extending training.')
        save(); agent.export(out/'policy.json'); phase='done'
        result=dict(schema=6,steps=state['steps'],updates=agent.updates,best_validation=state['best_validation'],teacher_probability=handover.probability,
                    test_evaluated=False,checkpoint=str(out/'latest.pt'),demonstrations=demos.size)
        atomic_json(out/'result.json',result); return result
    except BaseException as exc:
        atomic_json(out/'error.json',dict(error=str(exc),steps=state['steps'],last_committed_checkpoint=str(out/'latest.pt')))
        raise
    finally:
        env.close(); status(phase)


def evaluate_checkpoint(path,split,matrix,episodes=None):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck.get('schema')!=SCHEMA: raise ValueError('Use the matching v5 evaluator for a v5 checkpoint')
    cfg=ck['cfg']; cfg['device']=device_for('auto'); torch.set_num_threads(cfg['threads'])
    if cfg['traffic']=='learned': os.environ['LOTUS_NPC']=str(ROOT/cfg['npc_policy'])
    else: os.environ.pop('LOTUS_NPC',None)
    agent=Agent(cfg,cfg['device']); agent.load_state_dict(ck['agent']); agent.reset(); reports={}
    for planner,shield in ([(p,s) for p in (False,True) for s in (False,True)] if matrix else [(True,cfg['shield'])]):
        key=f'{"planner" if planner else "prior"}_shield{int(shield)}'
        summary,rows=evaluate(agent,specs(ck['manifest'],split,episodes or cfg['eval_episodes']),cfg,planner,shield)
        reports[key]=dict(summary=summary,episodes=rows); print(json.dumps({key:summary}),flush=True)
    atomic_json(Path(path).resolve().parent/f'evaluation_{split}_{Path(path).stem}.json',reports)
    return reports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/v6_full.yaml')); p.add_argument('--out'); p.add_argument('--steps',type=int); p.add_argument('--seed',type=int); p.add_argument('--device')
    mode=p.add_mutually_exclusive_group(); mode.add_argument('--resume'); mode.add_argument('--init-from')
    p.add_argument('--evaluate'); p.add_argument('--split',choices=['validation','test'],default='validation'); p.add_argument('--matrix',action='store_true'); p.add_argument('--episodes',type=int)
    p.add_argument('--port',type=int,default=8765); p.add_argument('--bind',default='0.0.0.0'); p.add_argument('--keep-monitor',action='store_true')
    a=p.parse_args()
    if a.evaluate: evaluate_checkpoint(a.evaluate,a.split,a.matrix,a.episodes); return
    cfg=load_config(a.config)
    for k in ('out','steps','seed','device'):
        if getattr(a,k) is not None: cfg[k]=getattr(a,k)
    with TrainingMonitor(ROOT/cfg['out'],a.port,a.bind,cfg['generated_maps'], label='LOTUS v6') as monitor:
        try: print(json.dumps(train(cfg,a.resume,a.init_from)),flush=True)
        finally:
            if a.keep_monitor: monitor.keep_open()
