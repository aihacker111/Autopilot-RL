"""
Autopilot-RL: run the whole pipeline (phases 2 -> 3 -> 4) with gates between stages.

  python pipeline.py                                  # everything, configs/default.yaml
  python pipeline.py --stages 2,3                     # stop before CARLA
  python pipeline.py --stages 4 --teacher rl/runs/coevolve/ft_g3/final.json
  python pipeline.py --smoke                          # tiny settings, no CARLA (dry run): checks the installation in minutes
  python pipeline.py --port 9000                      # training monitor on another port (--port 0 turns it off)

At start it prints a link to the training monitor (http://<server>:8765): a dashboard with the current
stage, learning curves, the scenario archive, server load and the log, plus the simulator driving with
the newest checkpoint. The monitor keeps running after the pipeline ends until Ctrl+C (--exit-when-done).

A stage whose result file already exists is skipped; --redo 3 reruns phase 3. A failed gate stops the run
(--force continues anyway) so a weak teacher never feeds the next stage.
"""
import argparse, json, os, subprocess, sys, time
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
RL, CARLA, RUNS = os.path.join(ROOT, "rl"), os.path.join(ROOT, "carla"), os.path.join(ROOT, "rl", "runs")
PY = sys.executable


class Tee:
    """everything printed goes to the terminal and to RUNS/pipeline.log (the monitor shows it)"""
    def __init__(self, path): self.term, self.f = sys.stdout, open(path, "a", buffering=1, encoding="utf-8")
    def write(self, s): self.term.write(s); self.f.write(s)
    def flush(self): self.term.flush(); self.f.flush()
    def isatty(self): return self.term.isatty()


STATUS = {}


def stage(key, state="active", **kw):
    """records stage transitions in RUNS/pipeline_status.json for the monitor"""
    now = time.time(); h = STATUS.setdefault("history", [])
    if state == "active": h.append({"key": key, "state": "active", "start": now})
    else:
        for e in reversed(h):
            if e["key"] == key and e["state"] == "active": e.update(state=state, end=now); break
        else: h.append({"key": key, "state": state, "start": now, "end": now})
    STATUS.update(current=key if state == "active" else STATUS.get("current"), **kw); save()


def save(**kw):
    STATUS.update(updated=time.time(), **kw)
    os.makedirs(RUNS, exist_ok=True); tmp = os.path.join(RUNS, "pipeline_status.json.tmp")
    json.dump(STATUS, open(tmp, "w"), indent=1); os.replace(tmp, os.path.join(RUNS, "pipeline_status.json"))


def sh(cmd, cwd=ROOT):
    print("\n\033[1m$ " + " ".join(str(c) for c in cmd) + "\033[0m", flush=True)
    p = subprocess.Popen([str(c) for c in cmd], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                         env={**os.environ, "PYTHONUNBUFFERED": "1"})
    for line in p.stdout: sys.stdout.write(line); sys.stdout.flush()
    if p.wait(): raise subprocess.CalledProcessError(p.returncode, cmd)


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
    if os.path.exists(out) and not redo:
        for k in ("2.baseline", "2.distill", "2.finetune"): stage(k, "done")
        print("phase 2 already done:", json.load(open(out))); return json.load(open(out))
    os.makedirs(os.path.join(RUNS, "eval"), exist_ok=True)
    stage("2.baseline")
    base = node_eval("idm", c["eval_n"], W, os.path.join(RUNS, "eval", "idm.json"))
    teach = node_eval("planner", c["eval_n"], W, os.path.join(RUNS, "eval", "planner.json"))
    stage("2.baseline", "done"); stage("2.distill")
    sh([PY, os.path.join(RL, "distill.py"), "--rounds", c["rounds"], "--decisions", c["decisions"], "--workers", W, "--epochs", c["epochs"], "--eval-n", c["eval_n"], "--device", dev, "--out", os.path.join(RUNS, "teacher_v2")])
    last = os.path.join(RUNS, "teacher_v2", f"student_r{c['rounds'] - 1}.pt")
    stage("2.distill", "done"); stage("2.finetune")
    sh([PY, os.path.join(RL, "finetune.py"), "--init", last, "--steps", c["ft_steps"], "--workers", W, "--device", dev, "--out", os.path.join(RUNS, "finetune_v2")])
    final = node_eval(os.path.join(RUNS, "finetune_v2", "final.json"), c["eval_n"], W, out)
    final.update(baseline=base, teacher=teach); json.dump(final, open(out, "w"), indent=1)
    stage("2.finetune", "done")
    gate("phase 2", final["ok"] >= c["gate"]["min_ok"] and final["crash"] <= c["gate"]["max_crash"], f"ok {final['ok']} (>= {c['gate']['min_ok']}), crash {final['crash']} (<= {c['gate']['max_crash']})", force)
    return final


def phase3(c, W, dev, force, redo, policy, init):
    res = os.path.join(RUNS, "coevolve", "result.json")
    stage("3.coevolve")
    if not (os.path.exists(res) and not redo):
        sh([PY, os.path.join(RL, "coevolve.py"), "--policy", policy, "--init", init, "--gens", c["gens"], "--budget", c["budget"], "--batch", c["batch"], "--episodes", c["episodes"],
            "--realism", c["realism"], "--priors", os.path.join(ROOT, c["priors"]), "--ft-steps", c["ft_steps"], "--heldout", c["heldout"], "--workers", W, "--out", os.path.join(RUNS, "coevolve")])
    stage("3.coevolve", "done")
    r = json.load(open(res)); hb, ha, db, da = r["heldout_before"], r["heldout_after"], r["default_before"], r["default_after"]
    gate("phase 3", (ha or {}).get("difficulty", 9) <= hb.get("difficulty", 0) and da.get("ok", 0) >= db.get("ok", 0) - c["gate"]["max_default_drop"],
         f"held-out difficulty {hb.get('difficulty')} -> {(ha or {}).get('difficulty')}, ordinary traffic ok {db.get('ok')} -> {da.get('ok')}", force)
    return r


def phase4(c, dev, teacher, smoke):
    cmd = [PY, os.path.join(CARLA, "phase4.py"), "--teacher", teacher, "--rounds", c["rounds"], "--episodes", c["episodes"], "--eval-episodes", c["eval_episodes"], "--encoder", c["encoder"],
           "--epochs", c["epochs"], "--map-seed", c["map_seed"], "--nx", c["nx"], "--ny", c["ny"], "--host", c["carla_host"], "--port", c["carla_port"], "--device", dev, "--out", os.path.join(RUNS, "phase4")]
    stage("4.carla")
    sh(cmd + (["--dry-run"] if smoke else []))
    stage("4.carla", "done")
    return json.load(open(os.path.join(RUNS, "phase4", "result.json")))


def run(stages, redo, cfg, W, dev, a, t0):
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
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    p.add_argument("--stages", default="2,3,4"); p.add_argument("--redo", default=""); p.add_argument("--force", action="store_true")
    p.add_argument("--teacher", default=None, help="policy JSON to use as the phase 4 teacher (default: phase 3 result)")
    p.add_argument("--smoke", action="store_true", help="tiny settings and no CARLA, to check that everything runs")
    p.add_argument("--port", type=int, default=8765, help="training monitor port (0 = no monitor)")
    p.add_argument("--bind", default="0.0.0.0", help="monitor address; 127.0.0.1 allows only an SSH tunnel")
    p.add_argument("--exit-when-done", action="store_true", help="stop the monitor when the pipeline ends")
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
    os.makedirs(RUNS, exist_ok=True); sys.stdout = Tee(os.path.join(RUNS, "pipeline.log"))
    print(f"\n===== LOTUS pipeline {time.strftime('%Y-%m-%d %H:%M:%S')} · stages {a.stages} · {W} workers · {dev} =====", flush=True)
    save(started=t0, stages=stages, state="running", workers=W, device=dev, smoke=a.smoke, history=[])
    srv = None
    if a.port:
        sys.path.insert(0, os.path.join(ROOT, "monitor")); import server as monitor
        srv, url = monitor.start(RUNS, a.port, a.bind)
    try:
        summary = run(stages, redo, cfg, W, dev, a, t0)
        save(state="done", ended=time.time())
    except BaseException as e:   # gate failures and errors: keep the dashboard up so the reason can be read
        msg = str(e) if not isinstance(e, subprocess.CalledProcessError) else f"{os.path.basename(str(e.cmd[1]))} exited with code {e.returncode}"
        if isinstance(e, KeyboardInterrupt): msg = "stopped with Ctrl+C"
        cur = STATUS.get("current")
        if cur: stage(cur, "failed", state="failed", error=msg, ended=time.time())
        else: save(state="failed", error=msg, ended=time.time())
        print("\n" + msg, flush=True)
        if srv and not a.exit_when_done and not isinstance(e, KeyboardInterrupt):
            print(f"Dashboard still open at {url} · Ctrl+C to exit", flush=True)
            try:
                while True: time.sleep(3600)
            except KeyboardInterrupt: pass
        sys.exit(1)
    if srv and not a.exit_when_done:
        print(f"\nDone. Dashboard still open at {url} · Ctrl+C to exit", flush=True)
        try:
            while True: time.sleep(3600)
        except KeyboardInterrupt: pass

