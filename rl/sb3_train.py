"""
LOTUS, Phase 2 with Stable-Baselines3: PPO or SAC for the self-driving car.

The traffic simulation stays in JavaScript (shared with the web page); each Node
worker process runs several environments, exposed to SB3 as one VecEnv.

  python sb3_train.py --algo sac --device cuda --steps 1000000       # SAC on a GPU
  python sb3_train.py --algo ppo --steps 5000000 --workers 30         # PPO on many CPU cores
  python sb3_train.py --eval runs/sb3_sac/final.zip --algo sac        # evaluate

Outputs in --out: step_*.zip / final.zip (SB3 models), vecnormalize.pkl,
step_*.json / final.json (weights for the web page), log.json (learning curve).
"""
import argparse, base64, json, os, time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from train import Worker   # Node worker process wrapper

HERE = os.path.dirname(os.path.abspath(__file__))


class NodeVecEnv(VecEnv):
    """W Node processes x n environments each. step_async sends to every worker, step_wait collects."""
    def __init__(self, workers, envs_per_worker, seed=1, density=None, density_range=(0.4, 1.05), curriculum=0):
        ramp = int(curriculum / max(1, workers * envs_per_worker))
        self.ws = [Worker(envs_per_worker, seed * 100 + i, density, list(density_range) if density is None else None, ramp) for i in range(workers)]
        od, ad = self.ws[0].info["obs_dim"], self.ws[0].info["act_dim"]
        obs_space = gym.spaces.Box(-np.inf, np.inf, (od,), np.float32)
        act_space = gym.spaces.Box(-1.0, 1.0, (ad,), np.float32)
        super().__init__(workers * envs_per_worker, obs_space, act_space)
        self._actions = None

    def reset(self):
        for w in self.ws: w.send({"cmd": "reset"})
        return np.array(sum([w.recv()["obs"] for w in self.ws], []), dtype=np.float32)

    def step_async(self, actions): self._actions = np.clip(actions, -1, 1)

    def step_wait(self):
        k = 0
        for w in self.ws:
            w.send({"cmd": "step", "actions": self._actions[k:k + w.n].tolist()}); k += w.n
        obs, rew, done, infos = [], [], [], []
        for w in self.ws:
            r = w.recv(); obs += r["obs"]; rew += r["rew"]; done += r["done"]
            for inf in r["infos"]:
                if inf:
                    inf = dict(inf); inf["terminal_observation"] = np.array(inf.pop("terminal"), np.float32)
                    inf["TimeLimit.truncated"] = inf["result"] == "timeout"
                    inf["episode_stats"] = {k2: inf[k2] for k2 in ("result", "ret", "time", "red")}
                    infos.append(inf)
                else: infos.append({})
        return np.array(obs, np.float32), np.array(rew, np.float32), np.array(done, bool), infos

    def close(self):
        for w in self.ws: w.close()
    # SB3 bookkeeping not needed for this env
    def get_attr(self, name, indices=None): return [None] * self.num_envs
    def set_attr(self, name, value, indices=None): pass
    def env_method(self, name, *a, indices=None, **k): return [None] * self.num_envs
    def env_is_wrapped(self, wrapper_class, indices=None): return [False] * self.num_envs
    def seed(self, seed=None): return [None] * self.num_envs


# ---------------------------------------------------------------- export for the web page
def export_json(model, venv, algo, path, name, steps):
    b64 = lambda a: base64.b64encode(np.ascontiguousarray(a, np.float32).tobytes()).decode()
    if algo == "ppo":
        mods = [m for m in model.policy.mlp_extractor.policy_net if isinstance(m, nn.Linear)] + [model.policy.action_net]
        act, squash = "tanh", False
    else:
        mods = [m for m in model.actor.latent_pi if isinstance(m, nn.Linear)] + [model.actor.mu]
        act, squash = "relu", True
    layers = [{"W": b64(m.weight.detach().cpu().numpy()), "b": b64(m.bias.detach().cpu().numpy()), "inp": m.in_features, "out": m.out_features} for m in mods]
    rms = venv.obs_rms
    with open(path, "w") as f:
        json.dump({"name": name, "steps": steps, "obsMean": b64(rms.mean), "obsStd": b64(np.sqrt(rms.var + venv.epsilon)),
                   "layers": layers, "act": act, "squash": squash, "algo": algo}, f)


class Progress(BaseCallback):
    """Writes log.json (same format as train.py) and saves checkpoints at milestones."""
    def __init__(self, out, algo, milestones, every=2048):
        super().__init__()
        self.out, self.algo, self.milestones, self.every = out, algo, sorted(milestones), every
        self.hist, self.log, self.t0, self.next_log = deque(maxlen=200), [], time.time(), every

    def _save(self, tag, label):
        self.model.save(os.path.join(self.out, tag + ".zip"))
        self.training_env.save(os.path.join(self.out, "vecnormalize.pkl"))
        export_json(self.model, self.training_env, self.algo, os.path.join(self.out, tag + ".json"), label, self.num_timesteps)

    def _on_training_start(self): self._save("step_0", "Mới học")

    def _on_step(self):
        for inf in self.locals["infos"]:
            if "episode_stats" in inf: self.hist.append(inf["episode_stats"])
        n = self.num_timesteps
        if n >= self.next_log and self.hist:
            self.next_log = n + self.every
            H = list(self.hist); k = len(H)
            row = {"steps": n, "minutes": round((time.time() - self.t0) / 60, 2), "episodes": k,
                   "return": round(float(np.mean([h["ret"] for h in H])), 3),
                   "ok": round(sum(h["result"] == "ok" for h in H) / k, 3), "crash": round(sum(h["result"] == "crash" for h in H) / k, 3),
                   "timeout": round(sum(h["result"] == "timeout" for h in H) / k, 3), "red": round(sum(h["red"] for h in H) / k, 3),
                   "time_ok": round(float(np.mean([h["time"] for h in H if h["result"] == "ok"] or [0])), 2),
                   "fps": int(n / (time.time() - self.t0))}
            self.log.append(row); print(json.dumps(row), flush=True)
            with open(os.path.join(self.out, "log.json"), "w") as f: json.dump(self.log, f)
        while self.milestones and n >= self.milestones[0]:
            m = self.milestones.pop(0); self._save(f"step_{m}", f"{m // 1000}k bước")
        return True


def train(a):
    os.makedirs(a.out, exist_ok=True)
    if a.device == "cpu": torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2) // 4)))
    venv = VecNormalize(NodeVecEnv(a.workers, a.envs, a.seed, curriculum=a.curriculum), norm_obs=True, norm_reward=(a.algo == "ppo"), clip_obs=8.0, gamma=0.99)
    if a.algo == "ppo":
        n_steps = max(64, 2048 // venv.num_envs) if a.n_steps is None else a.n_steps
        model = PPO("MlpPolicy", venv, n_steps=n_steps, batch_size=512, n_epochs=8, learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                    clip_range=0.2, ent_coef=0.003, vf_coef=0.5, max_grad_norm=0.5, device=a.device, seed=a.seed, verbose=0,
                    policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[256, 256]), activation_fn=nn.Tanh, log_std_init=-0.5, ortho_init=True))
    else:
        # SAC: off-policy, reuses every sample many times, so it needs far fewer simulator steps; the GPU does the extra gradient work
        model = SAC("MlpPolicy", venv, learning_rate=3e-4, buffer_size=a.buffer, batch_size=a.batch, learning_starts=a.learning_starts, gamma=0.99, tau=0.005,
                    train_freq=1, gradient_steps=a.utd * venv.num_envs if a.utd else -1, ent_coef="auto", device=a.device, seed=a.seed, verbose=0,
                    policy_kwargs=dict(net_arch=dict(pi=[128, 128], qf=[256, 256])))
    cb = Progress(a.out, a.algo, [int(x) for x in a.milestones.split(",")], every=a.log_every)
    model.learn(total_timesteps=a.steps, callback=cb)
    cb._save("final", "Đã học xong")
    venv.close()


def evaluate(a):
    base = NodeVecEnv(a.workers, a.envs, seed=999, density=a.density)
    venv = VecNormalize.load(os.path.join(os.path.dirname(a.eval), "vecnormalize.pkl"), base); venv.training = False; venv.norm_reward = False
    model = (PPO if a.algo == "ppo" else SAC).load(a.eval, device="cpu")
    obs = venv.reset(); res = []
    while len(res) < a.episodes:
        act, _ = model.predict(obs, deterministic=True)
        obs, _, _, infos = venv.step(act)
        res += [i["episode_stats"] for i in infos if "episode_stats" in i]
    res = res[:a.episodes]; n = len(res)
    out = {"checkpoint": os.path.basename(a.eval), "episodes": n, "density": a.density, "ok": sum(r["result"] == "ok" for r in res) / n,
           "crash": sum(r["result"] == "crash" for r in res) / n, "timeout": sum(r["result"] == "timeout" for r in res) / n, "red": sum(r["red"] for r in res) / n}
    print(json.dumps(out)); venv.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ppo", "sac"], default="sac")
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--workers", type=int, default=max(1, min(24, (os.cpu_count() or 2) - 1)))
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--n-steps", type=int, default=None, help="PPO rollout length per env")
    p.add_argument("--batch", type=int, default=512, help="SAC batch size")
    p.add_argument("--utd", type=int, default=0, help="SAC gradient steps per env step per env (0 = one per env step)")
    p.add_argument("--curriculum", type=int, default=300_000, help="steps over which traffic grows from light to full (0 = off)")
    p.add_argument("--learning-starts", type=int, default=10_000)
    p.add_argument("--buffer", type=int, default=1_000_000)
    p.add_argument("--log-every", type=int, default=2048)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--milestones", default="120000,400000")
    p.add_argument("--out", default=None)
    p.add_argument("--eval", default=None)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--density", type=float, default=0.9)
    a = p.parse_args()
    a.out = a.out or os.path.join(HERE, "runs", f"sb3_{a.algo}")
    evaluate(a) if a.eval else train(a)
