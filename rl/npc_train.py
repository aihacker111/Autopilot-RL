"""
LOTUS: learned road users (motorbikes and cars), web/npc.js.

  python npc_train.py --bc-steps 60000 --iters 40 --workers 16        # full run
  python npc_train.py --bc-steps 3000 --iters 2 --ppo-steps 600 --workers 2   # smoke test

1. Imitation: the rules drive, every decision is recorded, and a network learns to reproduce it (behaviour cloning).
2. Multi-agent RL (PPO with shared weights): every motorbike and car in the city is driven by the network with a little
   exploration. Each one is rewarded for keeping up with its own preferred speed and penalised for collisions, near
   misses and, in proportion to how law-abiding it is, for running red lights. A pull toward the rules (the label of the
   rule-based decision in the same state) keeps it human-like; it fades to a floor but never to zero.
3. Realism report: traffic statistics with the rules and with the network (collisions, near misses, red runs, speed,
   share stopped), and the realism score of the phase 3 filter (calib/priors_vn.json).

Outputs in --out: npc_bc.json, npc.json (the final network, also picked up by web/build.py), log.json, report.json.
"""
import argparse, base64, json, os, subprocess, time
import numpy as np
import torch
import torch.nn as nn
from distill import note

HERE = os.path.dirname(os.path.abspath(__file__))
EGO, K, NB = 20, 10, 12          # must match NPC in web/npc.js
OBS = EGO + K * NB; ROW = OBS + 9


class NpcNet(nn.Module):
    def __init__(self, out=2, h_phi=48, h=128):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(NB, h_phi), nn.ReLU(), nn.Linear(h_phi, h_phi), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(EGO + 2 * h_phi, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, out))
        self.register_buffer("mean", torch.zeros(OBS)); self.register_buffer("std", torch.ones(OBS))
    def forward(self, obs):
        x = ((obs - self.mean) / self.std).clamp(-8, 8)
        head, nb = x[:, :EGO], x[:, EGO:].reshape(-1, K, NB)
        present = obs[:, EGO:].reshape(-1, K, NB)[:, :, NB - 1:NB] > 0.5
        h = self.phi(nb)
        mx = torch.where(present, h, torch.zeros_like(h)).max(1).values
        mn = (h * present.float()).sum(1) / present.float().sum(1).clamp(min=1)
        return self.rho(torch.cat([head, mx, mn], 1))


def export(net, log_std, path, name):
    b64 = lambda t: base64.b64encode(np.ascontiguousarray(t.detach().cpu().numpy(), np.float32).tobytes()).decode()
    L = lambda s: [{"W": b64(m.weight), "b": b64(m.bias), "inp": m.in_features, "out": m.out_features} for m in s if isinstance(m, nn.Linear)]
    json.dump({"type": "npc", "name": name, "obsMean": b64(net.mean), "obsStd": b64(net.std), "phi": L(net.phi), "rho": L(net.rho),
               "logStd": [float(x) for x in log_std], "types": {"moto": 1, "car": 1, "taxi": 1}}, open(path, "w"))


def rollouts(mode, steps, workers, out, tag, policy=None, std=None, seed0=0):
    os.makedirs(out, exist_ok=True); ps = []
    for w in range(workers):
        cmd = ["node", os.path.join(HERE, "gen_npc.js"), "--mode", mode, "--steps", str(steps), "--seed", str(seed0 + w + 1), "--out", os.path.join(out, f"{tag}_w{w}")]
        if policy: cmd += ["--policy", policy]
        if std is not None: cmd += ["--std", ",".join(f"{x:.3f}" for x in std)]
        ps.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True))
    rows, sums = [], []
    for w, p in enumerate(ps):
        o, _ = p.communicate(); sums.append(json.loads(o.strip().splitlines()[-1]))
        f = os.path.join(out, f"{tag}_w{w}.bin")
        if os.path.exists(f):
            d = np.fromfile(f, np.float32).reshape(-1, ROW); d[:, OBS + 7] += w * 1e9; rows.append(d); os.remove(f)
    keys = [k for k in sums[0] if isinstance(sums[0][k], (int, float))]
    summ = {k: float(np.mean([s[k] for s in sums])) for k in keys}
    return (np.concatenate(rows) if rows else np.zeros((0, ROW), np.float32)), summ


def bc_train(net, X, Y, epochs, dev):
    net.mean.copy_(torch.tensor(X.mean(0), device=dev)); net.std.copy_(torch.tensor(X.std(0) + 1e-3, device=dev))
    Xt, Yt = torch.tensor(X, device=dev), torch.tensor(Y, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4); n = len(Xt); idx = torch.randperm(n, device=dev); nv = max(512, n // 10)
    va, tr = idx[:nv], idx[nv:]; best = (1e9, None)
    for ep in range(epochs):
        perm = tr[torch.randperm(len(tr), device=dev)]
        for i in range(0, len(perm), 1024):
            j = perm[i:i + 1024]; loss = ((torch.tanh(net(Xt[j])) - Yt[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        with torch.no_grad(): vl = float(((torch.tanh(net(Xt[va])) - Yt[va]) ** 2).mean())
        if vl < best[0]: best = (vl, {k: v.clone() for k, v in net.state_dict().items()})
    net.load_state_dict(best[1]); return best[0]


def gae(rew, done, trunc, vid, val, gamma, lam):
    """advantages per vehicle trajectory: rows of one vehicle are in time order inside each worker file"""
    adv = np.zeros_like(rew); ret = np.zeros_like(rew)
    order = np.lexsort((np.arange(len(vid)), vid))
    v_sorted = vid[order]
    starts = np.r_[0, np.nonzero(np.diff(v_sorted))[0] + 1, len(order)]
    for a, b in zip(starts[:-1], starts[1:]):
        ix = order[a:b]; last = 0.0
        for k in range(len(ix) - 1, -1, -1):
            i = ix[k]
            if k == len(ix) - 1: nv = 0.0 if done[i] else val[i]   # truncated at the end of the rollout: bootstrap on itself
            else: nv = val[ix[k + 1]]
            d = rew[i] + gamma * nv - val[i]
            last = d + gamma * lam * (0.0 if k == len(ix) - 1 else last); adv[i] = last; ret[i] = last + val[i]
    return adv, ret


def realism(summary, priors_path):
    """the phase 3 realism score on the traffic statistics (1 = inside every threshold)"""
    try:
        P = json.load(open(priors_path))
    except Exception:
        return None
    checks = {"collisionsPerVehMin": P.get("npc_collisions_per_veh_min", 0.01), "instantStopsPerVehMin": P.get("instant_stops_per_veh_min", 0.5), "redRunsPerMin": P.get("red_runs_per_min", 6)}
    score = 1.0
    for k, lim in checks.items():
        if k in summary and summary[k] > lim: score *= max(0.0, 1 - (summary[k] - lim) / max(1e-6, lim))
    return round(score, 3)


def main(a):
    torch.manual_seed(a.seed); np.random.seed(a.seed); dev = torch.device(a.device); os.makedirs(a.out, exist_ok=True)
    data = os.path.join(a.out, "data"); log = []
    # 1. imitation
    t0 = time.time(); D, s_rules = rollouts("bc", a.bc_steps, a.workers, data, "bc", seed0=a.seed * 1000)
    X, Y = D[:, :OBS], D[:, OBS + 2:OBS + 4]
    pi = NpcNet().to(dev); vl = bc_train(pi, X, Y, a.bc_epochs, dev)
    log_std = torch.full((2,), a.std0, device=dev, requires_grad=True)
    export(pi, log_std.tolist(), os.path.join(a.out, "npc_bc.json"), "Learned traffic (imitation)")
    note(a.out, stage="bc", iters=a.iters)
    row = {"stage": "bc", "samples": int(len(X)), "val_mse": round(vl, 4), "minutes": round((time.time() - t0) / 60, 1), "rules": s_rules}; log.append(row); print(json.dumps(row), flush=True)
    # 2. multi-agent PPO with a pull toward the rules
    vf = NpcNet(out=1).to(dev); vf.mean.copy_(pi.mean); vf.std.copy_(pi.std)
    opt = torch.optim.Adam(list(pi.parameters()) + [log_std], lr=a.lr); vopt = torch.optim.Adam(vf.parameters(), lr=a.lr * 3)
    for it in range(a.iters):
        t0 = time.time(); cur = os.path.join(a.out, "npc_cur.json"); export(pi, log_std.tolist(), cur, f"Learned traffic (RL {it})")
        D, s = rollouts("ppo", a.ppo_steps, a.workers, data, f"ppo{it}", policy=cur, std=log_std.tolist(), seed0=a.seed * 1000 + 100 * (it + 1))
        D = D[np.isfinite(D).all(1)]
        if len(D) < 256: continue
        O, U, LAB, R, DN, TR, VID = D[:, :OBS], D[:, OBS:OBS + 2], D[:, OBS + 2:OBS + 4], D[:, OBS + 4], D[:, OBS + 5] > 0.5, D[:, OBS + 6] > 0.5, D[:, OBS + 7]
        Ot, Ut, Lt = (torch.tensor(x, device=dev) for x in (O, U, LAB))
        with torch.no_grad():
            val = torch.cat([vf(Ot[i:i + 8192]).squeeze(-1) for i in range(0, len(Ot), 8192)]).cpu().numpy()
            dist = torch.distributions.Normal(torch.cat([pi(Ot[i:i + 8192]) for i in range(0, len(Ot), 8192)]), log_std.exp())
            old_lp = dist.log_prob(Ut).sum(-1)
        adv, ret = gae(R, DN, TR, VID, val, a.gamma, a.lam)
        At = torch.tensor((adv - adv.mean()) / (adv.std() + 1e-8), device=dev); Rt = torch.tensor(ret, device=dev)
        bc_w = max(a.bc_floor, a.bc_w * (1 - it / max(1, a.iters * 0.6)))
        n = len(Ot)
        for ep in range(a.epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, a.mb):
                j = perm[i:i + a.mb]; mu = pi(Ot[j]); d = torch.distributions.Normal(mu, log_std.exp())
                lp = d.log_prob(Ut[j]).sum(-1); ratio = (lp - old_lp[j]).exp()
                pl = -torch.min(ratio * At[j], ratio.clamp(1 - a.clip, 1 + a.clip) * At[j]).mean()
                loss = pl - a.ent * d.entropy().sum(-1).mean() + bc_w * ((torch.tanh(mu) - Lt[j]) ** 2).mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(list(pi.parameters()) + [log_std], 0.5); opt.step()
                with torch.no_grad(): log_std.clamp_(-2.5, -0.5)
                vl_ = ((vf(Ot[j]).squeeze(-1) - Rt[j]) ** 2).mean(); vopt.zero_grad(); vl_.backward(); nn.utils.clip_grad_norm_(vf.parameters(), 0.5); vopt.step()
        row = {"stage": "ppo", "iter": it, "samples": int(n), "reward": round(float(R.mean()), 4), "bc_w": round(bc_w, 3), "std": [round(float(x), 3) for x in log_std.detach().exp()],
               "minutes": round((time.time() - t0) / 60, 1), "traffic": {k: round(v, 4) for k, v in s.items()}}
        log.append(row); print(json.dumps(row), flush=True); json.dump(log, open(os.path.join(a.out, "log.json"), "w"), indent=1)
        note(a.out, stage="ppo", iter=it, iters=a.iters, reward=row["reward"])
        torch.save({"pi": pi.state_dict(), "vf": vf.state_dict(), "log_std": log_std.detach().cpu()}, os.path.join(a.out, "npc.pt"))
    final = os.path.join(a.out, "npc.json"); export(pi, log_std.tolist(), final, "Learned traffic")
    # 3. realism report: the same windows with the rules and with the network (no exploration)
    _, rules = rollouts("stats", a.stats_steps, a.workers, data, "stats_rules", seed0=777)
    _, learned = rollouts("stats", a.stats_steps, a.workers, data, "stats_net", policy=final, seed0=777)
    pri = os.path.join(HERE, "..", "calib", "priors_vn.json")
    rep = {"rules": rules, "learned": learned, "realism_rules": realism(rules, pri), "realism_learned": realism(learned, pri)}
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w"), indent=1); json.dump(log, open(os.path.join(a.out, "log.json"), "w"), indent=1)
    print(json.dumps({"report": rep}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bc-steps", type=int, default=60000, help="simulation steps per worker recorded with the rules")
    p.add_argument("--bc-epochs", type=int, default=30)
    p.add_argument("--iters", type=int, default=40); p.add_argument("--ppo-steps", type=int, default=3000, help="simulation steps per worker per PPO iteration")
    p.add_argument("--stats-steps", type=int, default=3000)
    p.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 1)))
    p.add_argument("--lr", type=float, default=3e-4); p.add_argument("--epochs", type=int, default=4); p.add_argument("--mb", type=int, default=4096)
    p.add_argument("--clip", type=float, default=0.2); p.add_argument("--ent", type=float, default=0.001); p.add_argument("--gamma", type=float, default=0.95); p.add_argument("--lam", type=float, default=0.9)
    p.add_argument("--bc-w", type=float, default=2.0); p.add_argument("--bc-floor", type=float, default=0.3, help="the pull toward the rules never goes below this")
    p.add_argument("--std0", type=float, default=-1.4)
    p.add_argument("--seed", type=int, default=1); p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=os.path.join(HERE, "runs", "npc"))
    main(p.parse_args())
