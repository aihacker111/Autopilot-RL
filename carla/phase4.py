"""
LOTUS, Phase 4: DAgger for the camera student in CARLA.

  python carla/phase4.py --teacher rl/runs/coevolve/ft_g3/final.json --rounds 4 --episodes 40 --encoder dinov3
  python carla/phase4.py --teacher idm --rounds 2 --episodes 2 --encoder tiny --dry-run      # pipeline test without CARLA

Round 0: the teacher drives in CARLA and labels every frame. Round k: the student drives part of the
time (beta halves each round) and the teacher keeps labelling. After each round the frames are encoded
with the frozen vision model and the head is retrained on all data. Finally the student and the teacher
drive the same number of closed-loop episodes and are compared.
"""
import argparse, json, os, subprocess, sys
HERE = os.path.dirname(os.path.abspath(__file__))


def sh(cmd):
    print("$ " + " ".join(cmd), flush=True); subprocess.run(cmd, check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True); p.add_argument("--rounds", type=int, default=4); p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--eval-episodes", type=int, default=30); p.add_argument("--encoder", default="dinov3"); p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--map-seed", type=int, default=7); p.add_argument("--nx", type=int, default=3); p.add_argument("--ny", type=int, default=3)
    p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=2000)
    p.add_argument("--device", default="cuda"); p.add_argument("--dry-run", action="store_true")
    p.add_argument("--out", default=os.path.join(HERE, "..", "rl", "runs", "phase4"))
    a = p.parse_args()
    data = os.path.join(a.out, "data"); os.makedirs(data, exist_ok=True)
    common = ["--teacher", a.teacher, "--map-seed", str(a.map_seed), "--nx", str(a.nx), "--ny", str(a.ny), "--host", a.host, "--port", str(a.port), "--device", a.device, "--out-dir", a.out] + (["--dry-run"] if a.dry_run else [])
    py = sys.executable; prev = None; log = []
    for r in range(a.rounds):
        coll = [py, os.path.join(HERE, "bridge.py"), "collect", "--data", data, "--episodes", str(a.episodes), "--seed", str(100 + r), "--result", os.path.join(a.out, f"collect_r{r}.json")] + common
        if prev: coll += ["--student", prev, "--beta", str(max(0.1, 0.5 ** r))]
        sh(coll)
        sh([py, os.path.join(HERE, "student.py"), "encode", "--data", data, "--encoder", a.encoder, "--device", a.device])
        ck = os.path.join(a.out, f"student_r{r}.pt")
        sh([py, os.path.join(HERE, "student.py"), "train", "--data", data, "--encoder", a.encoder, "--device", a.device, "--epochs", str(a.epochs), "--out", ck] + (["--init", prev] if prev else []))
        prev = ck
        log.append({"round": r, "collect": json.load(open(os.path.join(a.out, f"collect_r{r}.json")))})
        json.dump(log, open(os.path.join(a.out, "phase4_log.json"), "w"), indent=1)
    ev = [py, os.path.join(HERE, "bridge.py"), "drive", "--episodes", str(a.eval_episodes), "--seed", "999"] + common
    sh(ev + ["--student", prev, "--result", os.path.join(a.out, "eval_student.json")])
    sh(ev + ["--result", os.path.join(a.out, "eval_teacher.json")])
    res = {"student": json.load(open(os.path.join(a.out, "eval_student.json"))), "teacher": json.load(open(os.path.join(a.out, "eval_teacher.json"))), "final_student": prev}
    json.dump(res, open(os.path.join(a.out, "result.json"), "w"), indent=1); print(json.dumps(res))
