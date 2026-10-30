"""
Autopilot-RL: run the whole pipeline (phases 2 -> 3 -> 4) with gates between stages.

  python pipeline.py                                  # everything, configs/default.yaml
  python pipeline.py --stages 2,3                     # stop before CARLA
  python pipeline.py --stages 4 --teacher rl/runs/coevolve/ft_g3/final.json
  python pipeline.py --smoke                          # tiny settings, no CARLA (dry run): checks the installation in minutes

A stage whose result file already exists is skipped; --redo 3 reruns phase 3. A failed gate stops the run
(--force continues anyway) so a weak teacher never feeds the next stage.
"""
import argparse, json, os, subprocess, sys, time
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
RL, CARLA, RUNS = os.path.join(ROOT, "rl"), os.path.join(ROOT, "carla"), os.path.join(ROOT, "rl", "runs")
PY = sys.executable


def sh(cmd, cwd=ROOT):
    print("\n\033[1m$ " + " ".join(str(c) for c in cmd) + "\033[0m", flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=cwd)


def node_eval(driver, n, workers, out):
    procs = [subprocess.Popen(["node", os.path.join(RL, "eval.js"), "--driver", driver, "--n", str(n), "--shard", str(s), "--shards", str(workers), "--out", f"{out}.s{s}"], stdout=subprocess.DEVNULL) for s in range(workers)]
    for p in procs: p.wait()
    eps = sum([json.load(open(f"{out}.s{s}"))["episodes"] for s in range(workers)], [])
    m = lambda k: round(sum(e["result"] == k for e in eps) / len(eps), 3)
    res = {"driver": os.path.basename(driver), "episodes": len(eps), "ok": m("ok"), "crash": m("crash"), "red": round(sum(bool(e["red"]) for e in eps) / len(eps), 3), "timeout": m("timeout"), "hit": m("hit")}
    json.dump(res, open(out, "w"), indent=1); print(json.dumps(res)); return res


def gate(name, ok, msg, force):
    print(f"\n{'PASS' if ok else 'FAIL'} gate {name}: {msg}", flush=True)
    if not ok and not force: sys.exit(f"Stopped at gate {name}. Improve this stage or rerun with --force.")


def phase2(c, W, dev, force, redo):
    out = os.path.join(RUNS, "eval", "phase2_final.json")
    if os.path.exists(out) and not redo: print("phase 2 already done:", json.load(open(out))); return json.load(open(out))
    os.makedirs(os.path.join(RUNS, "eval"), exist_ok=True)
    base = node_eval("idm", c["eval_n"], W, os.path.join(RUNS, "eval", "idm.json"))
    teach = node_eval("planner", c["eval_n"], W, os.path.join(RUNS, "eval", "planner.json"))
    sh([PY, os.path.join(RL, "distill.py"), "--rounds", c["rounds"], "--decisions", c["decisions"], "--workers", W, "--epochs", c["epochs"], "--eval-n", c["eval_n"], "--device", dev, "--out", os.path.join(RUNS, "teacher_v2")])
    last = os.path.join(RUNS, "teacher_v2", f"student_r{c['rounds'] - 1}.pt")
    sh([PY, os.path.join(RL, "finetune.py"), "--init", last, "--steps", c["ft_steps"], "--workers", W, "--device", dev, "--out", os.path.join(RUNS, "finetune_v2")])
    final = node_eval(os.path.join(RUNS, "finetune_v2", "final.json"), c["eval_n"], W, out)
    final.update(baseline=base, teacher=teach); json.dump(final, open(out, "w"), indent=1)
    gate("phase 2", final["ok"] >= c["gate"]["min_ok"] and final["crash"] <= c["gate"]["max_crash"], f"ok {final['ok']} (>= {c['gate']['min_ok']}), crash {final['crash']} (<= {c['gate']['max_crash']})", force)
    return final


def phase3(c, W, dev, force, redo, policy, init):
    res = os.path.join(RUNS, "coevolve", "result.json")
    if not (os.path.exists(res) and not redo):
        sh([PY, os.path.join(RL, "coevolve.py"), "--policy", policy, "--init", init, "--gens", c["gens"], "--budget", c["budget"], "--batch", c["batch"], "--episodes", c["episodes"],
            "--realism", c["realism"], "--priors", os.path.join(ROOT, c["priors"]), "--ft-steps", c["ft_steps"], "--heldout", c["heldout"], "--workers", W, "--out", os.path.join(RUNS, "coevolve")])
    r = json.load(open(res)); hb, ha, db, da = r["heldout_before"], r["heldout_after"], r["default_before"], r["default_after"]
    gate("phase 3", (ha or {}).get("difficulty", 9) <= hb.get("difficulty", 0) and da.get("ok", 0) >= db.get("ok", 0) - c["gate"]["max_default_drop"],
         f"held-out difficulty {hb.get('difficulty')} -> {(ha or {}).get('difficulty')}, ordinary traffic ok {db.get('ok')} -> {da.get('ok')}", force)
    return r


def phase4(c, dev, teacher, smoke):
    cmd = [PY, os.path.join(CARLA, "phase4.py"), "--teacher", teacher, "--rounds", c["rounds"], "--episodes", c["episodes"], "--eval-episodes", c["eval_episodes"], "--encoder", c["encoder"],
           "--epochs", c["epochs"], "--map-seed", c["map_seed"], "--nx", c["nx"], "--ny", c["ny"], "--host", c["carla_host"], "--port", c["carla_port"], "--device", dev, "--out", os.path.join(RUNS, "phase4")]
    sh(cmd + (["--dry-run"] if smoke else []))
    return json.load(open(os.path.join(RUNS, "phase4", "result.json")))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    p.add_argument("--stages", default="2,3,4"); p.add_argument("--redo", default=""); p.add_argument("--force", action="store_true")
    p.add_argument("--teacher", default=None, help="policy JSON to use as the phase 4 teacher (default: phase 3 result)")
    p.add_argument("--smoke", action="store_true", help="tiny settings and no CARLA, to check that everything runs")
    a = p.parse_args()
    cfg = yaml.safe_load(open(a.config))
    if a.smoke:
        RUNS = os.path.join(ROOT, "rl", "runs_smoke")
        cfg["phase2"].update(eval_n=6, rounds=1, decisions=300, epochs=5, ft_steps=2048); cfg["phase2"]["gate"].update(min_ok=0, max_crash=1)
        cfg["phase3"].update(gens=1, budget=8, batch=4, episodes=1, ft_steps=1024, heldout=2); cfg["phase3"]["gate"]["max_default_drop"] = 1
        cfg["phase4"].update(encoder="tiny", rounds=1, episodes=2, eval_episodes=1, epochs=3, nx=2, ny=2)
        a.force = True
    W = (os.cpu_count() or 4) - 2 if cfg["workers"] == "auto" else int(cfg["workers"]); W = max(1, min(W, 64))
    try: import torch; dev = ("cuda" if torch.cuda.is_available() else "cpu") if cfg["device"] == "auto" else cfg["device"]
    except ImportError: dev = "cpu"
    stages, redo, t0 = a.stages.split(","), a.redo.split(","), time.time()
    summary = {}
    if "2" in stages: summary["phase2"] = phase2(cfg["phase2"], W, dev, a.force, "2" in redo)
    policy, init = os.path.join(RUNS, "finetune_v2", "final.json"), os.path.join(RUNS, "finetune_v2", "final.pt")
    if "3" in stages:
        r = phase3(cfg["phase3"], W, dev, a.force, "3" in redo, policy, init); summary["phase3"] = r; policy = r["final_policy"]
    elif os.path.exists(os.path.join(RUNS, "coevolve", "result.json")): policy = json.load(open(os.path.join(RUNS, "coevolve", "result.json")))["final_policy"]
    if "4" in stages: summary["phase4"] = phase4(cfg["phase4"], dev, a.teacher or policy, a.smoke)
    summary["minutes"] = round((time.time() - t0) / 60, 1)
    os.makedirs(RUNS, exist_ok=True); json.dump(summary, open(os.path.join(RUNS, "pipeline_summary.json"), "w"), indent=1)
    print("\n" + json.dumps(summary, indent=1))
