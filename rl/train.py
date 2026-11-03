"""
LOTUS, Phase 2: PPO teacher for the self-driving car.

The environment (traffic, motorbike swarm, signals) lives in JavaScript and is
shared with the web simulator. Each Node worker runs several copies; this script
steps them in parallel and trains the policy with PyTorch.

  python train.py --steps 1200000                # train (MacBook: ~1 h on 4 workers)
  python train.py --eval runs/teacher/final.pt   # evaluate a checkpoint

Requires: Python 3.9+, torch, numpy, Node.js 18+ on the PATH.
"""
import argparse, base64, json, os, subprocess, sys, time
from collections import deque
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- environment
class Worker:
    def __init__(self, n, seed, density=None, density_range=None, ramp=0, extra=None):
        self.p = subprocess.Popen(["node", os.path.join(HERE, "worker.js")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.n = n
        msg = {"cmd": "init", "n": n, "seed": seed}
        if density is not None: msg["density"] = density
        if density_range is not None: msg["densityRange"] = density_range
        if ramp: msg["ramp"] = ramp
        if extra: msg.update(extra)
        self.info = self.call(msg)

    def send(self, m): self.p.stdin.write(json.dumps(m) + "\n"); self.p.stdin.flush()
    def recv(self): return json.loads(self.p.stdout.readline())
    def call(self, m): self.send(m); return self.recv()
    def close(self):
        try: self.send({"cmd": "close"})
        except Exception: pass


class VecEnv:
    """W Node processes x n environments each, stepped in lockstep."""
    def __init__(self, workers, envs_per_worker, seed=1, density=None, density_range=None, curriculum=0, extra=None, extra0=None):
        ramp = int(curriculum / max(1, workers * envs_per_worker))
        self.ws = [Worker(envs_per_worker, seed * 100 + i, density, density_range, ramp, dict(extra or {}, **(extra0 if i == 0 and extra0 else {})) or None) for i in range(workers)]
        self.n = workers * envs_per_worker
        self.obs_dim = self.ws[0].info["obs_dim"]; self.act_dim = self.ws[0].info["act_dim"]

    def reset(self):
        for w in self.ws: w.send({"cmd": "reset"})
        rs = [w.recv() for w in self.ws]
        self.gen = np.array(sum([r["gen"] for r in rs], []), np.float32) if "gen" in rs[0] else None   # world model: scenario genomes
        return np.array(sum([r["obs"] for r in rs], []), dtype=np.float32)

    def step(self, actions):
        k = 0
        for w in self.ws:
            w.send({"cmd": "step", "actions": actions[k:k + w.n].tolist()}); k += w.n
        obs, rew, done, infos, aux, gen, cf, base = [], [], [], [], [], [], [], 0
        for w in self.ws:   # workers run in parallel; we only collect here
            r = w.recv(); obs += r["obs"]; rew += r["rew"]; done += r["done"]; infos += r["infos"]; aux += r.get("aux", []); gen += r.get("gen", [])
            cf += [dict(c, i=c["i"] + base) for c in r.get("cf", [])]; base += w.n   # counterfactual branches (tdmpc2.py v4), by global env index
        self.cf = cf
        self.aux = np.array(aux, np.float32) if aux else None; self.gen = np.array(gen, np.float32) if gen else None   # world model extras (tdmpc2.py)
        return np.array(obs, np.float32), np.array(rew, np.float32), np.array(done, bool), infos

    def set_pool(self, pool, p, w=None):
        """evolved scenario genomes (optionally weighted), mixed into the next map of every environment"""
        for wk in self.ws: wk.send({"cmd": "pool", "pool": pool, "p": p, **({"w": list(map(float, w))} if w is not None else {})})
        for wk in self.ws: wk.recv()

    def close(self):
        for w in self.ws: w.close()


# ---------------------------------------------------------------- model
class RunningNorm:
    def __init__(self, dim):
        self.mean = np.zeros(dim, np.float64); self.var = np.ones(dim, np.float64); self.count = 1e-4
    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        d = bm - self.mean; tot = self.count + bc
        self.mean = self.mean + d * bc / tot
        self.var = (self.var * self.count + bv * bc + d ** 2 * self.count * bc / tot) / tot
        self.count = tot
    @property
    def std(self): return np.sqrt(self.var + 1e-8)
    def __call__(self, x): return np.clip((x - self.mean) / self.std, -8, 8).astype(np.float32)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hp=128, hv=256):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(obs_dim, hp), nn.Tanh(), nn.Linear(hp, hp), nn.Tanh(), nn.Linear(hp, act_dim))
        self.v = nn.Sequential(nn.Linear(obs_dim, hv), nn.Tanh(), nn.Linear(hv, hv), nn.Tanh(), nn.Linear(hv, 1))
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.orthogonal_(m.weight, 2 ** 0.5); nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.pi[-1].weight, 0.01); nn.init.orthogonal_(self.v[-1].weight, 1.0)

    def dist(self, x):
        return torch.distributions.Normal(self.pi(x), self.log_std.exp())


def export_json(model, norm, path, name, steps):
    """Weights as base64 float32 so the browser can run the policy (see MLPPolicy in env.js)."""
    b64 = lambda a: base64.b64encode(np.ascontiguousarray(a, np.float32).tobytes()).decode()
    layers = [{"W": b64(m.weight.detach().cpu().numpy()), "b": b64(m.bias.detach().cpu().numpy()), "inp": m.in_features, "out": m.out_features}
              for m in model.pi if isinstance(m, nn.Linear)]
    with open(path, "w") as f:
        json.dump({"name": name, "steps": steps, "obsMean": b64(norm.mean), "obsStd": b64(norm.std), "layers": layers}, f)


def save(model, norm, path, steps):
    torch.save({"model": model.state_dict(), "mean": norm.mean, "var": norm.var, "count": norm.count, "steps": steps}, path)


def load(path, obs_dim, act_dim):
    ck = torch.load(path, weights_only=False, map_location="cpu")
    model = ActorCritic(obs_dim, act_dim); model.load_state_dict(ck["model"])
    norm = RunningNorm(obs_dim); norm.mean, norm.var, norm.count = ck["mean"], ck["var"], ck["count"]
    return model, norm, ck["steps"]


# ---------------------------------------------------------------- training
def train(a):
    torch.set_num_threads(1); torch.manual_seed(a.seed); np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    env = VecEnv(a.workers, a.envs, a.seed, density_range=[0.4, 1.05], curriculum=a.curriculum)
    N, T = env.n, a.n_steps
    dev = torch.device(a.device)
    if a.resume:
        model, norm, start = load(a.resume, env.obs_dim, env.act_dim); print(f"resume from {a.resume} at {start} steps", flush=True)
    else:
        model, norm, start = ActorCritic(env.obs_dim, env.act_dim), RunningNorm(env.obs_dim), 0
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=1e-5)
    ret_rms, disc_ret = RunningNorm(1), np.zeros(N)
    milestones = sorted(set([0] + [int(x) for x in a.milestones.split(",")] + [a.steps]))
    raw = env.reset()
    if not a.resume: norm.update(raw)
    hist = deque(maxlen=200); log = []
    steps, upd, t0 = start, 0, time.time()
    if not a.resume: save(model, norm, os.path.join(a.out, "step_0.pt"), 0); export_json(model, norm, os.path.join(a.out, "step_0.json"), "Untrained", 0)
    milestones = [m for m in milestones if m > start]
    n_upd = max(1, (a.steps - start) // (N * T))
    while steps < a.steps:
        frac = 1 - upd / max(1, n_upd)
        for g in opt.param_groups: g["lr"] = a.lr * max(0.05, frac)
        B_obs = np.zeros((T, N, env.obs_dim), np.float32); B_act = np.zeros((T, N, env.act_dim), np.float32)
        B_logp = np.zeros((T, N), np.float32); B_rew = np.zeros((T, N), np.float32); B_done = np.zeros((T, N), np.float32); B_val = np.zeros((T + 1, N), np.float32)
        raws = []
        for t in range(T):
            o = norm(raw); B_obs[t] = o; raws.append(raw)
            with torch.no_grad():
                ot = torch.from_numpy(o).to(dev); d = model.dist(ot); act = d.sample()
                B_logp[t] = d.log_prob(act).sum(-1).cpu().numpy(); B_val[t] = model.v(ot).squeeze(-1).cpu().numpy()
            B_act[t] = act.cpu().numpy()
            raw, rew, done, infos = env.step(np.clip(B_act[t], -1, 1))
            disc_ret = disc_ret * a.gamma + rew; ret_rms.update(disc_ret[:, None]); disc_ret[done] = 0
            B_rew[t] = rew / float(ret_rms.std[0]); B_done[t] = done
            for inf in infos:
                if inf: hist.append(inf)
        steps += N * T; upd += 1
        norm.update(np.concatenate(raws))
        with torch.no_grad(): B_val[T] = model.v(torch.from_numpy(norm(raw)).to(dev)).squeeze(-1).cpu().numpy()
        # GAE
        adv = np.zeros((T, N), np.float32); last = 0
        for t in reversed(range(T)):
            nonterm = 1 - B_done[t]
            delta = B_rew[t] + a.gamma * B_val[t + 1] * nonterm - B_val[t]
            last = delta + a.gamma * a.lam * nonterm * last; adv[t] = last
        ret = adv + B_val[:T]
        o, ac, lp, ad, rt = (torch.from_numpy(x.reshape(T * N, *x.shape[2:])).to(dev) for x in (B_obs, B_act, B_logp, adv, ret))
        for _ in range(a.epochs):
            idx = torch.randperm(T * N, device=dev)
            for s in range(0, T * N, a.mb):
                j = idx[s:s + a.mb]
                d = model.dist(o[j]); nlp = d.log_prob(ac[j]).sum(-1); ratio = (nlp - lp[j]).exp()
                A = (ad[j] - ad[j].mean()) / (ad[j].std() + 1e-8)
                pl = -torch.min(ratio * A, ratio.clamp(1 - a.clip, 1 + a.clip) * A).mean()
                vl = 0.5 * (model.v(o[j]).squeeze(-1) - rt[j]).pow(2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pl + a.vf * vl - a.ent * ent
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 0.5); opt.step()
        # log
        if hist:
            H = list(hist); n = len(H)
            row = {"steps": steps, "minutes": round((time.time() - t0) / 60, 2), "episodes": n,
                   "return": round(float(np.mean([h["ret"] for h in H])), 3),
                   "ok": round(sum(h["result"] == "ok" for h in H) / n, 3), "crash": round(sum(h["result"] == "crash" for h in H) / n, 3),
                   "timeout": round(sum(h["result"] == "timeout" for h in H) / n, 3), "red": round(sum(h["red"] for h in H) / n, 3),
                   "time_ok": round(float(np.mean([h["time"] for h in H if h["result"] == "ok"] or [0])), 2),
                   "std": [round(x, 3) for x in model.log_std.exp().cpu().tolist()], "fps": int(steps / (time.time() - t0))}
            log.append(row)
            with open(os.path.join(a.out, "log.json"), "w") as f: json.dump(log, f)
            print(json.dumps(row), flush=True)
        while milestones and steps >= milestones[0]:
            m = milestones.pop(0)
            save(model, norm, os.path.join(a.out, f"step_{m}.pt"), steps)
            export_json(model, norm, os.path.join(a.out, f"step_{m}.json"), f"{m // 1000}k steps", steps)
    save(model, norm, os.path.join(a.out, "final.pt"), steps); export_json(model, norm, os.path.join(a.out, "final.json"), "Fully trained", steps)
    env.close()


def evaluate(a):
    env = VecEnv(a.workers, a.envs, seed=999, density=a.density)
    model, norm, st = load(a.eval, env.obs_dim, env.act_dim)
    raw = env.reset(); res = []
    while len(res) < a.episodes:
        with torch.no_grad(): act = model.pi(torch.from_numpy(norm(raw))).numpy() if not a.random else np.random.uniform(-1, 1, (env.n, env.act_dim))
        raw, _, _, infos = env.step(np.clip(act, -1, 1))
        res += [i for i in infos if i]
    res = res[:a.episodes]; n = len(res)
    out = {"checkpoint": os.path.basename(a.eval), "steps": st, "episodes": n, "density": a.density,
           "ok": sum(r["result"] == "ok" for r in res) / n, "crash": sum(r["result"] == "crash" for r in res) / n,
           "timeout": sum(r["result"] == "timeout" for r in res) / n, "red": sum(r["red"] for r in res) / n,
           "time_ok": float(np.mean([r["time"] for r in res if r["result"] == "ok"] or [0]))}
    print(json.dumps(out)); env.close()
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1_200_000)
    p.add_argument("--workers", type=int, default=max(1, min(24, (os.cpu_count() or 2) - 1)), help="Node processes; the simulation is CPU-bound, so this is what makes training fast")
    p.add_argument("--envs", type=int, default=4, help="environments per worker")
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--mb", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--vf", type=float, default=0.5)
    p.add_argument("--ent", type=float, default=0.003)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--milestones", default="150000,500000")
    p.add_argument("--out", default=os.path.join(HERE, "runs", "teacher"))
    p.add_argument("--eval", default=None)
    p.add_argument("--curriculum", type=int, default=300_000, help="steps over which traffic grows from light to full (0 = off)")
    p.add_argument("--resume", default=None, help="continue from a .pt checkpoint")
    p.add_argument("--device", default="cpu", help="cpu or cuda; the policy is small, so cuda only helps with many workers")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--density", type=float, default=0.9)
    p.add_argument("--random", action="store_true")
    a = p.parse_args()
    evaluate(a) if a.eval else train(a)
