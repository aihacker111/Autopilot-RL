import json
import warnings
import numpy as np
import pytest
import torch
from v6.config import load_config
from v6.agent import Agent
from v6.replay import Replay, Demonstrations, sample_mixture
from v6.teacher import Handover
from v6.train import train
from v5_eval import ROOT
from distill import OBS
from tdmpc2 import GD,AUXD


def cfg(): return load_config(ROOT/'configs/v6_smoke.yaml')


def row(clock=0, n=1, source=0):
    rng=np.random.RandomState(clock+7)
    obs=rng.normal(0,.1,(n,OBS)).astype(np.float32)
    for j in range(16): obs[:,43+j*16+15]=int(j<3)
    aux=np.zeros((n,AUXD),np.float32); aux[:,4:6]=.1; aux[:,6:8]=.1; aux[:,8]=1
    return dict(obs=obs,next_obs=obs+.01,previous=np.zeros((n,2),np.float32),request=np.full((n,2),.1,np.float32),
                executed=np.full((n,2),.1,np.float32),reward=np.full(n,.1,np.float32),done=np.zeros(n,bool),terminal=np.zeros(n,bool),
                aux=aux,genome=np.zeros((n,GD),np.float32),planner=np.full((n,2),.1,np.float32),planner_weight=np.ones(n,np.float32),
                teacher_weight=np.ones(n,np.float32),shield_enabled=np.ones(n,np.float32),source=np.full(n,source,np.float32))


def replay():
    b=Replay(1,128)
    for i in range(40): b.add(**row(i))
    return b


def test_reward_multistep_supervision_updates_dynamics_without_consistency_loss():
    c=dict(cfg(),horizon=3,burn_in=1,posterior_coef=0.,consistency_coef=0.,value_coef=0.,collision_coef=0.,physics_coef=0.,kinematics_coef=0.,projection_coef=0.,context_coef=0.)
    torch.set_num_threads(1); torch.manual_seed(1); a=Agent(c,'cpu')
    torch.nn.init.normal_(a.model.rew[-1].weight,std=.1)
    before=a.model.dyn.out.weight.detach().clone()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always'); report=a.update(replay().sample(4,3,1,0))
    assert not torch.equal(before,a.model.dyn.out.weight)
    assert report['grad']>0 and all(np.isfinite(v) for v in report.values())
    assert 'latent_error_3' in report
    assert not any('requires_grad' in str(w.message) for w in caught)


def test_requested_action_q_target_includes_intervention_cost_and_shield_bit():
    c=cfg();a=Agent(c,'cpu'); batch=replay().sample(4,c['horizon'],c['burn_in'],0)
    batch['request'][:]=torch.tensor([1.,0.]);batch['executed'][:]=torch.tensor([-1.,0.]);batch['terminal'][:]=True
    batch['reward'][:]=.1
    # cost=.5*mean([4,0])=1; target=-.9, not the .1 physical reward.
    report=a.update(batch);assert report['q_target']==pytest.approx(-.9,abs=1e-6)
    z=torch.zeros(2,c['latent']);request=torch.tensor([[.8,.2],[-.2,.3]]);g=torch.zeros(2,GD)
    with torch.no_grad():
        a.projector[-1].bias.fill_(1.)
        assert torch.equal(a.project(z,request,False),request)
    seen=[];hook=a.model.qs[0][0].register_forward_pre_hook(lambda m,args:seen.append(args[0].detach()))
    a.q_logits(z,request,g,enabled=True);a.q_logits(z,request,g,enabled=False);hook.remove()
    assert (seen[0][...,-1]==1).all() and (seen[1][...,-1]==0).all()
    np.testing.assert_allclose(seen[0][...,c['latent']:c['latent']+2].numpy(),request.numpy())


def test_protected_demos_survive_online_data_and_do_not_bridge_teacher_gaps():
    d=Demonstrations(1,128)
    for i in range(8):d.add(row(i),[True])
    d.add(row(8),[False])
    assert d.streams[0].data['done'][0,7] and not d.streams[0].data['terminal'][0,7]
    assert (d.streams[0].labels(0,7)==-1).all()
    for i in range(100):d.add(row(i),[False])
    assert d.size==8
    d.add(row(110),[True]); assert not d.streams[0].valid(0,6,3)
    restored=Demonstrations(1,128);restored.load_state_dict(d.state_dict());assert restored.size==9
    assert np.array_equal(restored.streams[0].data['obs'],d.streams[0].data['obs'])


def test_batch_mixes_protected_demos_and_online_and_counterfactual():
    online=replay();cf=Replay(1,128);d=Demonstrations(1,128)
    for i in range(40):
        cf.add(**row(i,source=2));d.add(row(i),[True])
    batch=sample_mixture(online,d,cf,cfg()); source=batch['source']
    assert (source==1).float().mean()==.5
    assert (source==2).float().mean()==.125
    assert (source==0).float().mean()==.375


def test_handover_requires_repeated_validation_and_recovers_on_regression():
    c=cfg(); h=Handover(c)
    good=dict(episodes=12,ok=.9,contact_per_km=0.,red=0.,projection_mae=.01)
    h.update(good,0); assert h.probability==1.
    h.update(good,20); assert h.probability==1.
    h.update(good,40); assert h.probability==.75
    h.update(dict(good,contact_per_km=2.),60); assert h.probability==1.


def test_v6_resume_restores_demos_handover_and_optimizers(tmp_path):
    c=dict(cfg(),out=str(tmp_path/'resumed'),steps=24)
    train(c);train(dict(c,steps=48),str(tmp_path/'resumed/latest.pt'))
    train(dict(c,steps=48,out=str(tmp_path/'continuous')))
    a=torch.load(tmp_path/'resumed/latest.pt',weights_only=False);b=torch.load(tmp_path/'continuous/latest.pt',weights_only=False)
    for kind in ('model','dynamics','collision','projector','target_q'):
        for key in a['agent'][kind]: assert torch.equal(a['agent'][kind][key],b['agent'][kind][key]),(kind,key)
    assert a['handover']==b['handover']
    for key in a['demonstrations']['streams'][0]['data']:
        np.testing.assert_array_equal(a['demonstrations']['streams'][0]['data'][key],b['demonstrations']['streams'][0]['data'][key])
    assert torch.equal(a['rng']['torch'],b['rng']['torch'])
    assert a['agent']['updates']==b['agent']['updates']>0
    report=json.loads((tmp_path/'continuous/metrics.jsonl').read_text().splitlines()[-1]);assert report['metric_updates']==2
    assert report['latency_ms']>0 and report['demo_fraction']>.4


def test_v6_rejects_v5_exact_resume(tmp_path):
    old=tmp_path/'v5.pt';torch.save({'schema':5},old)
    with pytest.raises(ValueError,match='--init-from'):train(dict(cfg(),out=str(tmp_path/'new')),str(old))


def test_evaluation_matrix_preserves_training_state_and_rng():
    from v6.evaluation import evaluate
    a=Agent(cfg(),'cpu'); torch.manual_seed(17)
    a.act(np.zeros((1,OBS),np.float32),False)
    hidden=a.h.clone();last=a.last_z; rng=torch.get_rng_state().clone(); a.projector.train()
    suite=[dict(seed=23,preset='city',density=.5,maxTime=.6)]
    for planner in (False,True):
        for shield in (False,True):
            report,rows=evaluate(a,suite,cfg(),planner,shield)
            assert report['episodes']==1 and report['shield']==shield
            assert np.isfinite(report['projection_mae']) and report['latency_ms']>0
            assert report['intervention_fraction']>=0
            assert len(rows)==1 and rows[0]['decisions']>0
            if not shield: assert report['projection_mae']<=5.1e-5  # worker rounds executed actions to 1e-4
            assert torch.equal(a.h,hidden) and a.last_z is last and a.projector.training
            assert a.execution_shield==cfg()['shield'] and torch.equal(torch.get_rng_state(),rng)


def test_v5_weight_initialization_resets_incompatible_critic():
    from v5_agent import Agent as OldAgent
    old=OldAgent(cfg(),'cpu');new=Agent(cfg(),'cpu')
    before=new.model.qs[0][0].weight.detach().clone()
    report=new.initialize_weights(old.state_dict())
    assert any(k.startswith('qs.') for k in report['skipped'])
    assert torch.equal(new.model.qs[0][0].weight,before)
    assert torch.equal(new.model.enc.state_dict()['gru.weight_ih'],old.model.enc.state_dict()['gru.weight_ih'])
    assert not report['optimizer_replay_rng_restored'] and new.updates==0
    with pytest.raises(ValueError,match='Incompatible initialization'):
        Agent(dict(cfg(),latent=40),'cpu').initialize_weights(old.state_dict())


def test_teacher_baseline_failure_stops_before_collecting(tmp_path,monkeypatch):
    import v6.train as trainer
    def bad_teacher(suite,c):return dict(ok=0.,contact_per_km=2.),[]
    monkeypatch.setattr(trainer,'evaluate_teacher',bad_teacher)
    c=dict(cfg(),out=str(tmp_path/'bad_teacher'),teacher_baseline_episodes=1,require_teacher_baseline=True)
    with pytest.raises(RuntimeError,match='Teacher failed'):train(c)
    checkpoint=torch.load(tmp_path/'bad_teacher/latest.pt',weights_only=False)
    assert checkpoint['state']['steps']==0 and checkpoint['handover']['baseline']['ok']==0
    assert json.loads((tmp_path/'bad_teacher/status.json').read_text())['phase']=='failed'


def test_validation_saves_best_and_gates_on_both_controllers(tmp_path,monkeypatch):
    import v6.train as trainer
    from v6.teacher import evaluate_teacher
    suite=[dict(seed=23,preset='city',density=.5,maxTime=.6)]
    teacher_report,_=evaluate_teacher(suite,cfg()); assert teacher_report['episodes']==1
    monkeypatch.setattr(trainer,'specs',lambda *args:suite)
    c=dict(cfg(),out=str(tmp_path/'validation'),steps=16,eval_every=8,eval_episodes=1,handover_episodes=1)
    train(c)
    ck=torch.load(tmp_path/'validation/best.pt',weights_only=False)
    assert ck['state']['last_eval'] in (8,16)
    report=json.loads((tmp_path/'validation/validation_16.json').read_text())
    assert report['summary']['controller']=='planner' and report['prior']['controller']=='prior'
    assert not report['handover']['gate_passed'] and (tmp_path/'validation/best_policy.json').exists()


def test_map_stage_is_a_truncation_not_absorbing_terminal(tmp_path,monkeypatch):
    import v6.train as trainer
    Base=trainer.VecEnv
    class StageEnv(Base):
        def step(self,actions):
            obs,reward,done,infos=super().step(actions)
            done[0]=True;infos[0]=dict(result='stage',terminal=obs[0].tolist())
            return obs,reward,done,infos
    monkeypatch.setattr(trainer,'VecEnv',StageEnv)
    train(dict(cfg(),steps=1,out=str(tmp_path/'stage')))
    ck=torch.load(tmp_path/'stage/latest.pt',weights_only=False)
    assert ck['replay']['data']['done'][0,0]
    assert not ck['replay']['data']['terminal'][0,0]
