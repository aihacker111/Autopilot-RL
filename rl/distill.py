"""
LOTUS, Phase 2 (v2): distil the look-ahead teacher into a fast network, with DAgger.

  python distill.py --rounds 4 --decisions 25000 --workers 30      # full run on a server
  python distill.py --rounds 2 --decisions 1500 --workers 2         # smoke test

Round 0: the teacher drives and labels. Round k: the current student drives part of the
time (beta falls each round) and the teacher labels every state the student reaches,
which teaches it to recover from its own mistakes. After each round the student is
retrained on everything gathered so far and evaluated on the fixed suite.

Outputs in --out: student_r{k}.pt / .json (set-encoder policy, runs in the browser),
data/, eval_r{k}.json, log.json.
"""
import argparse, base64, json, os, subprocess, time
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
EGO_DIM, WP, K, NB = 23, 10, 16, 16   # must match ENV in web/env.js (observation v3)
HEAD = EGO_DIM + WP * 2          # ego features + route waypoints
OBS = HEAD + K * NB


def note(out, name="status.json", **kw):
    """small progress file for the training monitor (atomic write, so a reader never sees half a file)"""
    p = os.path.join(out, name); tmp = p + ".tmp"
    with open(tmp, "w") as f: json.dump({**kw, "t": time.time()}, f)
    os.replace(tmp, p)


class SetPolicy(nn.Module):
    """DeepSets: shared MLP per neighbour, max+mean pooling, MLP head; tanh output in [-1, 1]."""
    def __init__(self, h_phi=64, h=256):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(NB, h_phi), nn.ReLU(), nn.Linear(h_phi, h_phi), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(HEAD + 2 * h_phi, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 2))
        self.register_buffer("mean", torch.zeros(OBS)); self.register_buffer("std", torch.ones(OBS))

    def features(self, obs):
        x = ((obs - self.mean) / self.std).clamp(-8, 8)
        head, nb = x[:, :HEAD], x[:, HEAD:].reshape(-1, K, NB)
        present = obs[:, HEAD:].reshape(-1, K, NB)[:, :, NB - 1:NB] > 0.5
        h = self.phi(nb)
        mx = torch.where(present, h, torch.zeros_like(h)).max(1).values   # ReLU outputs are >= 0, so zero is a safe fill
        cnt = present.float().sum(1).clamp(min=1)
        mn = (h * present.float()).sum(1) / cnt
        return torch.cat([head, mx, mn], 1)

    def forward(self, obs): return torch.tanh(self.rho(self.features(obs)))


def export(model, path, name, steps):
    b64 = lambda a: base64.b64encode(np.ascontiguousarray(a.detach().cpu().numpy(), np.float32).tobytes()).decode()
    L = lambda seq: [{"W": b64(m.weight), "b": b64(m.bias), "inp": m.in_features, "out": m.out_features} for m in seq if isinstance(m, nn.Linear)]
    with open(path, "w") as f:
        json.dump({"type": "set", "name": name, "steps": steps, "egoDim": HEAD, "K": K, "nbDim": NB, "obsMean": b64(model.mean), "obsStd": b64(model.std),
                   "phi": L(model.phi), "rho": L(model.rho)}, f)


def gen(round_, n_total, workers, out, student=None, beta=1.0, seed0=0):
    per = max(1, n_total // workers); procs = []
    for w in range(workers):
        cmd = ["node", os.path.join(HERE, "gen_dagger.js"), "--decisions", str(per), "--seed", str(seed0 + round_ * 1000 + w + 1), "--out", os.path.join(out, "data", f"r{round_}_w{w}")]
        if student: cmd += ["--student", student, "--beta", str(beta)]
        procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for p in procs: p.wait()


def load_data(out):
    X, Y, eps = [], [], []
    d = os.path.join(out, "data")
    for f in sorted(os.listdir(d)):
        if not f.endswith(".bin"): continue
        meta = json.load(open(os.path.join(d, f[:-4] + ".json")))
        a = np.fromfile(os.path.join(d, f), np.float32).reshape(meta["rows"], meta["obs_dim"] + meta["act_dim"])
        X.append(a[:, :OBS]); Y.append(a[:, OBS:]); eps += meta["results"]
    return np.concatenate(X), np.concatenate(Y), eps


def train_student(X, Y, epochs, device, init=None):
    model = SetPolicy().to(device)
    if init is not None: model.load_state_dict(init.state_dict())
    model.mean.copy_(torch.tensor(X.mean(0))); model.std.copy_(torch.tensor(X.std(0) + 1e-3))
    Xt, Yt = torch.tensor(X, device=device), torch.tensor(Y, device=device)
    n = len(Xt); idx = torch.randperm(n, device=device); nv = max(256, n // 10)
    va, tr = idx[:nv], idx[nv:]
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state = 1e9, None
    for ep in range(epochs):
        model.train(); perm = tr[torch.randperm(len(tr), device=device)]
        for i in range(0, len(perm), 512):
            j = perm[i:i + 512]; pred = model(Xt[j])
            loss = ((pred - Yt[j]) ** 2 * torch.tensor([2.0, 1.0], device=device)).mean()   # speed matters more than sideways position
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad(): vl = float(((model(Xt[va]) - Yt[va]) ** 2).mean())
        if vl < best: best, best_state = vl, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best


def evaluate(policy_json, n, workers):
    shards = [subprocess.Popen(["node", os.path.join(HERE, "eval.js"), "--driver", policy_json, "--n", str(n), "--shard", str(s), "--shards", str(workers), "--out", policy_json + f".eval{s}"], stdout=subprocess.PIPE) for s in range(workers)]
    for p in shards: p.wait()
    eps = []
    for s in range(workers): eps += json.load(open(policy_json + f".eval{s}"))["episodes"]
    m = lambda k: sum(e["result"] == k for e in eps) / len(eps)
    return {"episodes": len(eps), "ok": m("ok"), "crash": m("crash"), "red": sum(bool(e["red"]) for e in eps) / len(eps), "timeout": m("timeout"), "hit": m("hit")}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--decisions", type=int, default=25000, help="teacher-labelled decisions per round")
    p.add_argument("--workers", type=int, default=max(1, min(32, (os.cpu_count() or 2) - 1)))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--eval-n", type=int, default=120)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=os.path.join(HERE, "runs", "teacher_v2"))
    a = p.parse_args()
    os.makedirs(os.path.join(a.out, "data"), exist_ok=True)
    log, model, student = [], None, None
    for r in range(a.rounds):
        beta = 1.0 if r == 0 else max(0.1, 0.6 ** r)
        note(a.out, round=r, rounds=a.rounds, step="gen", beta=beta)
        t0 = time.time(); gen(r, a.decisions, a.workers, a.out, student, beta)
        X, Y, eps = load_data(a.out)
        note(a.out, round=r, rounds=a.rounds, step="train", samples=int(len(X)))
        t1 = time.time(); model, vl = train_student(X, Y, a.epochs, a.device, model)
        student = os.path.join(a.out, f"student_r{r}.json")
        export(model.cpu(), student, f"Student round {r + 1}", int(len(X))); model.to(a.device)
        torch.save(model.state_dict(), os.path.join(a.out, f"student_r{r}.pt"))
        note(a.out, round=r, rounds=a.rounds, step="eval")
        ev = evaluate(student, a.eval_n, a.workers)
        teach = [e for e in eps if e]; tm = lambda k: sum(e["result"] == k for e in teach) / max(1, len(teach))
        row = {"round": r, "beta": beta, "samples": int(len(X)), "val_mse": round(vl, 4), "gen_min": round((t1 - t0) / 60, 1), "train_min": round((time.time() - t1) / 60, 1),
               "rollout_ok": round(tm("ok"), 3), "rollout_crash": round(tm("crash"), 3), "eval": ev}
        log.append(row); print(json.dumps(row), flush=True)
        json.dump(log, open(os.path.join(a.out, "distill_log.json"), "w"), indent=1)
    note(a.out, round=a.rounds - 1, rounds=a.rounds, step="done")
