"""
LOTUS, Phase 2 (v2), last step: PPO fine-tuning that starts from the distilled student.

  python finetune.py --init runs/teacher_v2/student_r3.pt --steps 3000000 --workers 30

The actor starts as the student; a behaviour-cloning term keeps it close to the student at
first and fades out, so PPO can improve on the teacher without forgetting what it learned.
Exports policy JSON files the web page can run (same set-encoder format as distill.py).
"""
import argparse, json, os, time, copy
from collections import deque
import numpy as np
import torch
import torch.nn as nn
from train import VecEnv
from distill import SetPolicy, export, HERE


class ActorCritic(nn.Module):
    def __init__(self, student):
        super().__init__()
        self.actor = copy.deepcopy(student)
        self.critic = copy.deepcopy(student); self.critic.rho[-1] = nn.Linear(self.critic.rho[-1].in_features, 1)
        self.log_std = nn.Parameter(torch.full((2,), -1.2))
    def dist(self, obs): return torch.distributions.Normal(self.actor(obs), self.log_std.exp())
    def value(self, obs): return self.critic.rho(self.critic.features(obs)).squeeze(-1)


def main(a):
    torch.set_num_threads(2); torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device)
    student = SetPolicy(); sd = torch.load(a.init, map_location="cpu")
    if any(k.startswith("actor.") for k in sd): sd = {k[6:]: v for k, v in sd.items() if k.startswith("actor.")}   # a fine-tuned checkpoint
    student.load_state_dict(sd)
    frozen = copy.deepcopy(student).to(dev).eval()
    for p_ in frozen.parameters(): p_.requires_grad_(False)
    model = ActorCritic(student).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5)
    pool = json.load(open(a.pool)) if a.pool else None   # Phase 3: evolved scenario genomes mixed into training
    env = VecEnv(a.workers, a.envs, a.seed, density_range=[0.4, 1.1], extra={"pool": pool, "pPool": a.p_pool} if pool else None)
    N, T = env.n, a.n_steps
    obs = env.reset(); hist = deque(maxlen=300); log = []; steps, upd, t0 = 0, 0, time.time()
    ret_rms_var, disc = 1.0, np.zeros(N)
    milestones = sorted(int(x) for x in a.milestones.split(","))
    n_upd = max(1, a.steps // (N * T))
    while steps < a.steps:
        frac = 1 - upd / n_upd
        for g in opt.param_groups: g["lr"] = a.lr * max(0.1, frac)
        bc_w = a.bc * max(0.0, 1 - upd / (0.5 * n_upd))   # behaviour cloning fades out over the first half
        B = {k: [] for k in ("o", "a", "lp", "r", "d", "v")}
        for t in range(T):
            ot = torch.tensor(obs, device=dev)
            with torch.no_grad(): d = model.dist(ot); act = d.sample(); lp = d.log_prob(act).sum(-1); v = model.value(ot)
            nobs, rew, done, infos = env.step(np.clip(act.cpu().numpy(), -1, 1))
            disc = disc * 0.99 + rew; ret_rms_var = 0.999 * ret_rms_var + 0.001 * float((disc ** 2).mean()); disc[done] = 0
            for k, x in (("o", ot), ("a", act), ("lp", lp), ("v", v)): B[k].append(x)
            B["r"].append(torch.tensor(rew / np.sqrt(ret_rms_var + 1e-8), device=dev)); B["d"].append(torch.tensor(done, dtype=torch.float32, device=dev))
            for inf in infos:
                if inf: hist.append(inf)
            obs = nobs
        steps += N * T; upd += 1
        with torch.no_grad(): last_v = model.value(torch.tensor(obs, device=dev))
        O, A, LP, R, D, V = (torch.stack(B[k]) for k in ("o", "a", "lp", "r", "d", "v"))
        adv = torch.zeros_like(R); g = 0
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else V[t + 1]
            delta = R[t] + 0.99 * nv * (1 - D[t]) - V[t]; g = delta + 0.99 * 0.95 * (1 - D[t]) * g; adv[t] = g
        ret = adv + V
        O, A, LP, adv, ret = O.reshape(T * N, -1), A.reshape(T * N, -1), LP.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        for _ in range(a.epochs):
            perm = torch.randperm(T * N, device=dev)
            for i in range(0, T * N, a.mb):
                j = perm[i:i + a.mb]; d = model.dist(O[j]); nlp = d.log_prob(A[j]).sum(-1); ratio = (nlp - LP[j]).exp()
                Aj = (adv[j] - adv[j].mean()) / (adv[j].std() + 1e-8)
                pl = -torch.min(ratio * Aj, ratio.clamp(0.8, 1.2) * Aj).mean()
                vl = 0.5 * (model.value(O[j]) - ret[j]).pow(2).mean()
                bc = ((d.mean - frozen(O[j])) ** 2).mean()
                loss = pl + 0.5 * vl - a.ent * d.entropy().sum(-1).mean() + bc_w * bc
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()
        if hist:
            H = list(hist); n = len(H); m = lambda k: round(sum(h["result"] == k for h in H) / n, 3)
            row = {"steps": steps, "minutes": round((time.time() - t0) / 60, 1), "ok": m("ok"), "crash": m("crash"), "red": round(sum(bool(h["red"]) for h in H) / n, 3),
                   "timeout": m("timeout"), "return": round(float(np.mean([h["ret"] for h in H])), 2), "bc_w": round(bc_w, 3), "fps": int(steps / (time.time() - t0))}
            log.append(row); print(json.dumps(row), flush=True); json.dump(log, open(os.path.join(a.out, "log.json"), "w"))
        while milestones and steps >= milestones[0]:
            m_ = milestones.pop(0); export(model.actor.cpu(), os.path.join(a.out, f"step_{m_}.json"), f"PPO {m_ // 1000}k", m_); model.actor.to(dev)
    export(model.actor.cpu(), os.path.join(a.out, "final.json"), "PPO tinh chỉnh", steps)
    torch.save(model.state_dict(), os.path.join(a.out, "final.pt")); env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--init", required=True)
    p.add_argument("--steps", type=int, default=3_000_000)
    p.add_argument("--workers", type=int, default=max(1, min(32, (os.cpu_count() or 2) - 1)))
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--mb", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ent", type=float, default=0.0)
    p.add_argument("--bc", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--pool", default=None, help="JSON list of scenario genomes to train among (Phase 3)")
    p.add_argument("--p-pool", type=float, default=0.5, help="share of training scenes drawn from the pool")
    p.add_argument("--milestones", default="500000,1500000")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=os.path.join(HERE, "runs", "finetune_v2"))
    main(p.parse_args())
