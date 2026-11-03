"""
LOTUS, Phase 3: co-evolution of the road-user population (Artificial Life) and the self-driving car.

  python coevolve.py --policy runs/finetune_v2/final.json --init runs/finetune_v2/final.pt --gens 4 --workers 30

Each generation:
  1. MAP-Elites searches scenario genomes (who is on the road and how they ride) that make the
     current car fail, keeping the hardest *realistic* scenario in every cell of a grid
     (traffic density x dominant behaviour), so the archive stays diverse.
  2. The realism filter rejects scenarios whose behaviour rates fall outside plausible bounds
     (calib/priors_vn.json, or statistics measured from real trajectories).
  3. The car is fine-tuned with PPO among the elite scenarios (half of the training scenes).
  4. The whole archive is re-scored against the new car; cells it has mastered become easy again
     and the search moves on.

Outputs in --out: archive_g{k}.json, pool_g{k}.json, policy_g{k}.json/.pt, coevolve_log.json,
and heldout.json (robustness on scenarios evolved against a different seed, before vs after).
"""
import argparse, json, math, os, random, subprocess, time
import numpy as np
from distill import note

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SPACE = {"density": [0.3, 1.25], "motoShare": [0.6, 0.95], "aggrMu": [0.1, 0.95], "aggrSd": [0.05, 0.4], "complyMu": [0.05, 0.95],
         "patienceMu": [0.05, 0.95], "speedMu": [0.1, 0.95], "pWrong": [0, 0.25], "pCutIn": [0, 0.6], "pEarlyLeft": [0, 0.9], "pBrake": [0, 0.08],
         "pedRate": [0.3, 2.0], "pJaywalk": [0, 0.6], "pDelivery": [0, 0.02]}   # pDelivery: HCMC delivery riders stopping at the curb
DEFAULT = {"density": 0.8, "motoShare": 0.835, "aggrMu": 0.5, "aggrSd": 0.29, "complyMu": 0.5, "patienceMu": 0.5, "speedMu": 0.5,
           "pWrong": 0, "pCutIn": 0, "pEarlyLeft": 0, "pBrake": 0, "pedRate": 1.0, "pJaywalk": 0.15, "pDelivery": 0}
BEHAVIOURS = ["none", "wrong-way", "cut-in", "early-left", "sudden-brake", "aggressive", "jaywalk", "delivery-stop"]
DENS_BINS = 5
PROGRESS = []   # MAP-Elites batches, for the training monitor


def random_genome(rng):
    g = {k: rng.uniform(*r) for k, r in SPACE.items()}
    for k in ("pWrong", "pCutIn", "pEarlyLeft", "pBrake", "pJaywalk", "pDelivery"):   # most scenarios use only some behaviours
        if rng.random() < 0.5: g[k] = 0.0
    return g


def mutate(g, rng, sigma=0.12):
    c = dict(g)
    for k, (lo, hi) in SPACE.items():
        if rng.random() < 0.5: c[k] = float(np.clip(c[k] + rng.gauss(0, sigma * (hi - lo)), lo, hi))
    return c


def descriptor(g):
    """cell of the archive: traffic density bin x the behaviour that stands out most"""
    d = min(DENS_BINS - 1, int((g["density"] - SPACE["density"][0]) / (SPACE["density"][1] - SPACE["density"][0]) * DENS_BINS))
    emph = {"wrong-way": g["pWrong"] / 0.25, "cut-in": g["pCutIn"] / 0.6, "early-left": g["pEarlyLeft"] / 0.9, "sudden-brake": g["pBrake"] / 0.08,
            "aggressive": max(0.0, (g["aggrMu"] - 0.5) / 0.45 + (0.5 - g["complyMu"]) / 0.45) / 2,
            "jaywalk": max(0.0, g.get("pJaywalk", 0) / 0.6 * min(1.0, g.get("pedRate", 1) / 1.2)), "delivery-stop": g.get("pDelivery", 0) / 0.02}
    b, v = max(emph.items(), key=lambda kv: kv[1])
    return d, (BEHAVIOURS.index(b) if v >= 0.25 else 0)


def realism(stats, priors):
    """product of soft constraints: 1 inside the plausible bound, decaying outside it"""
    score, why = 1.0, {}
    for k, p in priors.items():
        if k.startswith("_") or k not in stats: continue
        x = stats[k]
        f = 1.0 if x <= p["max"] else math.exp(-(x - p["max"]) / p["scale"])
        score *= f; why[k] = round(f, 3)
    return score, why


def evaluate(genomes, driver, workers, episodes, out_dir, tag, seed0, shield=False):
    os.makedirs(os.path.join(out_dir, "tmp"), exist_ok=True)
    items = [{"id": i, "seed": seed0 + i, "genome": g} for i, g in enumerate(genomes)]
    shards = [items[w::workers] for w in range(workers)]
    procs = []
    for w, sh in enumerate(shards):
        if not sh: continue
        fi, fo = os.path.join(out_dir, "tmp", f"{tag}_in{w}.json"), os.path.join(out_dir, "tmp", f"{tag}_out{w}.json")
        json.dump(sh, open(fi, "w"))
        procs.append((subprocess.Popen(["node", os.path.join(HERE, "evolve_eval.js"), "--in", fi, "--out", fo, "--driver", driver, "--episodes", str(episodes)] + (["--shield", "1"] if shield else []), stdout=subprocess.DEVNULL), fo))
    res = []
    for p, fo in procs: p.wait(); res += json.load(open(fo))
    res.sort(key=lambda r: r["id"])
    return res


class Archive:
    def __init__(self): self.cells = {}
    def offer(self, r):
        key = f"{r['cell'][0]},{r['cell'][1]}"
        if not r["feasible"]: return False
        cur = self.cells.get(key)
        if cur is None or r["difficulty"] > cur["difficulty"]: self.cells[key] = r; return True
        return False
    def elites(self): return list(self.cells.values())
    def coverage(self): return len(self.cells) / (DENS_BINS * len(BEHAVIOURS))
    def qd(self): return sum(r["difficulty"] for r in self.cells.values())


def score(results, priors, thr):
    for r in results:
        r["realism"], r["realism_parts"] = realism(r["stats"], priors)
        r["feasible"] = r["realism"] >= thr
        r["cell"] = descriptor(r["genome"])
    return results


def map_elites(archive, driver, a, priors, rng, tag, seed0):
    evals, t0 = 0, time.time()
    while evals < a.budget:
        if not archive.cells or evals < a.n_init and len(archive.cells) < 3:
            batch = [random_genome(rng) for _ in range(a.batch)]
        else:
            el = archive.elites()
            batch = [mutate(rng.choice(el)["genome"], rng) if rng.random() < 0.85 else random_genome(rng) for _ in range(a.batch)]
        res = score(evaluate(batch, driver, a.workers, a.episodes, a.out, f"{tag}_{evals}", seed0 + evals), priors, a.realism)
        added = sum(archive.offer(r) for r in res); evals += len(batch)
        row = {"stage": tag, "evals": evals, "added": added, "coverage": round(archive.coverage(), 3), "qd": round(archive.qd(), 3),
               "feasible": round(sum(r["feasible"] for r in res) / len(res), 2), "min": round((time.time() - t0) / 60, 1)}
        print(json.dumps(row), flush=True)
        PROGRESS.append({**row, "t": time.time()}); note(a.out, "progress.json", rows=PROGRESS)
        note(a.out, "archive_live.json", stage=tag, budget=a.budget, cells=[{"cell": r["cell"], "difficulty": r["difficulty"], "realism": r["realism"], "crash": r["crash"], "ok": r["ok"]} for r in archive.elites()])
    return archive


def rescore(archive, driver, a, priors, tag, seed0):
    el = archive.elites()
    if not el: return archive
    res = score(evaluate([r["genome"] for r in el], driver, a.workers, a.episodes, a.out, tag, seed0), priors, a.realism)
    new = Archive()
    for r in res: new.offer(r)
    return new


def summary(results):
    if not results: return {}
    n = len(results); m = lambda k: round(sum(r[k] for r in results) / n, 3)
    return {"scenarios": n, "difficulty": m("difficulty"), "crash": m("crash"), "ok": m("ok"), "red": m("red"), "timeout": m("timeout")}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, help="policy JSON driving in the evaluations (set-encoder student or fine-tuned)")
    p.add_argument("--init", required=True, help="matching .pt checkpoint to fine-tune from")
    p.add_argument("--gens", type=int, default=4)
    p.add_argument("--budget", type=int, default=600, help="scenario evaluations per generation")
    p.add_argument("--batch", type=int, default=60)
    p.add_argument("--n-init", type=int, default=120)
    p.add_argument("--episodes", type=int, default=4, help="episodes per scenario evaluation")
    p.add_argument("--realism", type=float, default=0.5, help="minimum realism score to enter the archive")
    p.add_argument("--priors", default=os.path.join(ROOT, "calib", "priors_vn.json"))
    p.add_argument("--ft-steps", type=int, default=1_000_000, help="PPO steps per generation")
    p.add_argument("--p-pool", type=float, default=0.5)
    p.add_argument("--heldout", type=int, default=60, help="size of the held-out evolved test set")
    p.add_argument("--workers", type=int, default=max(1, min(32, (os.cpu_count() or 2) - 1)))
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--out", default=os.path.join(HERE, "runs", "coevolve"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed); priors = json.load(open(a.priors))
    policy, ckpt, log = a.policy, a.init, []
    note(a.out, gen=0, gens=a.gens, step="heldout")
    # held-out test set: scenarios evolved against the starting car with a different seed, never trained on
    held = map_elites(Archive(), policy, argparse.Namespace(**{**vars(a), "budget": max(a.heldout * 2, a.batch)}), priors, random.Random(a.seed + 999), "heldout", 900000)
    held_genomes = [r["genome"] for r in held.elites()][: a.heldout]
    json.dump(held_genomes, open(os.path.join(a.out, "heldout.json"), "w"))
    before = summary(score(evaluate(held_genomes, policy, a.workers, a.episodes, a.out, "held_before", 800000), priors, a.realism))
    default_before = summary(score(evaluate([DEFAULT] * 20, policy, a.workers, a.episodes, a.out, "def_before", 700000), priors, 0))
    archive = Archive()
    for g in range(a.gens):
        note(a.out, gen=g, gens=a.gens, step="search")
        archive = rescore(archive, policy, a, priors, f"rescore{g}", 100000 * (g + 1)) if g else archive
        archive = map_elites(archive, policy, a, priors, rng, f"gen{g}", 10000 * (g + 1))
        el = archive.elites(); json.dump(el, open(os.path.join(a.out, f"archive_g{g}.json"), "w"))
        pool = [r["genome"] for r in el]; pool_f = os.path.join(a.out, f"pool_g{g}.json"); json.dump(pool, open(pool_f, "w"))
        # fine-tune the car among the elite scenarios
        ft_out = os.path.join(a.out, f"ft_g{g}")
        note(a.out, gen=g, gens=a.gens, step="finetune")
        subprocess.run(["python3", os.path.join(HERE, "finetune.py"), "--init", ckpt, "--pool", pool_f, "--p-pool", str(a.p_pool), "--steps", str(a.ft_steps),
                        "--workers", str(a.workers), "--bc", "0.3", "--milestones", str(a.ft_steps * 2), "--out", ft_out], check=True)
        policy, ckpt = os.path.join(ft_out, "final.json"), os.path.join(ft_out, "final.pt")
        held_now = summary(score(evaluate(held_genomes, policy, a.workers, a.episodes, a.out, f"held_g{g}", 800000), priors, a.realism))
        row = {"gen": g, "cells": len(el), "coverage": round(archive.coverage(), 3), "qd": round(archive.qd(), 3), "elite_difficulty": round(float(np.mean([r["difficulty"] for r in el])) if el else 0, 3),
               "heldout_before": before, "heldout_now": held_now, "default_before": default_before}
        log.append(row); print(json.dumps(row), flush=True)
        json.dump(log, open(os.path.join(a.out, "coevolve_log.json"), "w"), indent=1)
    default_after = summary(score(evaluate([DEFAULT] * 20, policy, a.workers, a.episodes, a.out, "def_after", 700000), priors, 0))
    json.dump({"heldout_before": before, "heldout_after": log[-1]["heldout_now"] if log else None, "default_before": default_before, "default_after": default_after,
               "final_policy": policy}, open(os.path.join(a.out, "result.json"), "w"), indent=1)
    note(a.out, gen=a.gens - 1, gens=a.gens, step="done")
    print(json.dumps({"done": True, "final_policy": policy}))
