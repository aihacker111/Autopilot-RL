"""
LOTUS test drive: the best world-model checkpoint drives a long journey (10-20 km) through every kind of test scenario,
with its full planner (risk-aware MPPI in several imagined futures), and the journey is recorded for the monitor's
Test drive page (/drive), where it is replayed exactly, stretch by stretch, with what the car imagined.

The journey is a chain of ~1 km stretches. Each stretch is a real place of Ho Chi Minh City (signalised crossroads,
roundabouts, a six-way junction, a flyover, a level crossing, residential streets) with a traffic culture to test
against (ordinary traffic, rush hour, a school gate, night speeders, motorbikes cutting in, wrong-way riders, sudden
braking, early left turns, red-light runners) and, when the run has them, the hardest scenarios the evolution found.

  python rl/drive_test.py                                   # best.pt of rl/runs/tdmpc2 (picked if missing), ~15 km
  python rl/drive_test.py --km 20                           # longer
  python rl/drive_test.py --ckpt rl/runs/tdmpc2/final.pt    # another checkpoint
Then open the link it prints (the monitor): /drive.
"""
import argparse, glob, json, math, os, sys, time
import numpy as np
import torch
import tdmpc2 as T
from train import VecEnv, Worker
CO = T.CO


class Journey(VecEnv):
    """the stretches split over several Node workers (one CPU core each), stepped in lockstep like VecEnv"""
    def __init__(self, js, workers, seed, extra):
        W = max(1, min(len(js), workers)); k = math.ceil(len(js) / W)
        self.ws = [Worker(len(c), seed * 100 + i, None, None, 0, dict(extra, journey=c)) for i, c in enumerate(js[j:j + k] for j in range(0, len(js), k))]
        self.n = len(js); self.obs_dim = self.ws[0].info["obs_dim"]; self.act_dim = self.ws[0].info["act_dim"]
        self.built = sum([w.info.get("journey", []) for w in self.ws], [])   # seed, environment options and map centre of every stretch


class Live:
    """the test drive written as it happens, for the monitor's /drive page: per stretch a head file and chunks of 50
    decisions (actions + what the car imagined), and live.json with the list of stretches"""
    def __init__(self, out, info):
        self.dir = os.path.join(out, "live"); os.makedirs(self.dir, exist_ok=True)
        for f in glob.glob(os.path.join(self.dir, "*.json")): os.remove(f)
        self.info, self.rows, self.chunks = info, [], {}
        self.write()
    def dump(self, name, obj):
        f = os.path.join(self.dir, name); json.dump(obj, open(f + ".tmp", "w"), ensure_ascii=False); os.replace(f + ".tmp", f)
    def write(self): self.dump("live.json", dict(self.info, status="driving", segments=self.rows))
    def add(self, sid, spec, built, steps):
        meta = {"id": sid, "steps": steps, "traffic": spec["traffic"], "place": spec["place"], "kind": spec["kind"], "evolved": spec["evolved"]}
        self.rows.append(dict(meta, done=False)); self.chunks[sid] = 0
        self.dump(f"seg_{sid:03d}.json", {"meta": meta, "replay": {"seed": built["seed"], "opts": built["opts"], "centre": built["centre"], "npc": built["npc"], "check": []}, "chunks": 0})
    def push(self, sid, actions, imagined):
        if not actions: return
        k = self.chunks[sid]; self.dump(f"seg_{sid:03d}_c{k}.json", {"actions": actions, "imagined": imagined}); self.chunks[sid] = k + 1
        h = json.load(open(os.path.join(self.dir, f"seg_{sid:03d}.json"))); h["chunks"] = k + 1; self.dump(f"seg_{sid:03d}.json", h)
    def finish(self, sid, seg):
        for r in self.rows:
            if r["id"] == sid: r.update(seg, done=True)

TRAFFIC = [
    ("Ordinary traffic", {}),
    ("Rush hour", {"density": 1.2, "motoShare": 0.92, "aggrMu": 0.7, "patienceMu": 0.25, "pCutIn": 0.3}),
    ("School gate", {"density": 0.9, "pedRate": 1.9, "pJaywalk": 0.45}),
    ("Night speeders", {"density": 0.45, "speedMu": 0.9, "complyMu": 0.25, "aggrMu": 0.75}),
    ("Motorbikes cutting in", {"pCutIn": 0.55, "aggrMu": 0.7}),
    ("Wrong-way riders", {"pWrong": 0.2}),
    ("Sudden braking", {"pBrake": 0.07}),
    ("Early left turns", {"pEarlyLeft": 0.8}),
    ("Red-light runners", {"complyMu": 0.1}),
]
PLACES = [("Ngã tư Bảy Hiền", "signalised crossroads"), ("Vòng xoay Lăng Cha Cả", "roundabout"), ("Ngã sáu Cộng Hòa", "six-way junction"),
          ("Cầu vượt Nguyễn Thái Sơn - Nguyễn Kiệm", "flyover"), ("rail", "level crossing"), ("Ngã tư Hàng Xanh", "signalised crossroads"),
          ("Vòng xoay Dân Chủ", "roundabout"), ("Thảo Điền", "residential streets"), ("Ngã tư An Sương", "city gateway"), ("Ngã tư Gò Dưa", "suburban crossroads"),
          ("Vòng xoay Phú Lâm", "roundabout"), ("Vòng xoay Cây Gõ", "roundabout"), ("Ngã tư Bốn Xã", "signalised crossroads")]


def load_agent(ckpt, device):
    argv, sys.argv = sys.argv, [sys.argv[0]]
    try: a0 = T.parse()
    finally: sys.argv = argv
    ck = torch.load(ckpt, map_location=device); saved = ck.get("args", {})
    b = argparse.Namespace(**{**vars(a0), **{k: v for k, v in saved.items() if k not in ("device", "out", "workers", "record", "record_n", "record_seed", "eval", "steps", "best", "port")}})
    b.device = str(device)
    if "shield" not in saved: b.shield = 0   # a checkpoint from before the safety shield drives as it was trained
    ag = T.TDMPC2(b, device); steps = ag.load(ckpt)
    return ag, steps


def stretches(n, evolved, seed0):
    """n stretches: places and traffic cycled so that the combinations vary; every 4th uses an evolved scenario when there are some"""
    out = []
    for i in range(n):
        place, kind = PLACES[i % len(PLACES)]
        if evolved and i % 4 == 3:
            e = evolved[(i // 4) % len(evolved)]; name, g = f"Evolved scenario (cell {e['cell'][0]},{e['cell'][1]}, difficulty {e['difficulty']:.2f})", dict(e["genome"])
        else:
            name, ov = TRAFFIC[i % len(TRAFFIC)]; g = dict(CO.DEFAULT, **ov) if ov else None
        out.append({"seed": seed0 + i, "place": place, "kind": kind, "traffic": name, "genome": g, "evolved": bool(evolved and i % 4 == 3)})
    return out


def drive(agent, specs, a, live=None, steps=0):
    """all stretches at once, in lockstep: one batched plan per decision for every car still driving; streamed to the
    live page every 50 decisions"""
    js = [{"seed": s["seed"], "place": s["place"], "genome": s["genome"], "routeM": a.stretch_m, "half": a.half, "maxTime": a.max_time} for s in specs]
    env = Journey(js, a.workers, a.seed, {"genomeSpace": T.GSPACE, "reward": agent.a.reward, "shield": agent.a.shield})
    B = len(js)
    if live:
        for s, b in zip(specs, env.built): live.add(s["id"], s, b, steps)
        live.write()
    obs = env.reset(); g = env.gen; alive = np.ones(B, bool); imag = [[] for _ in range(B)]; res = [None] * B; agent.prev_mean = None; n = 0; t0 = time.time()
    pa, pi = [[] for _ in range(B)], [[] for _ in range(B)]   # decisions not yet streamed
    while alive.any():
        act = agent.plan(obs, g, explore=False, record="all" if a.imagine else False)
        for b in range(B):
            if alive[b]:
                im = agent.imagined_all[b] if a.imagine else None
                imag[b].append(im); pa[b].append(act[b].tolist()); pi[b].append(im)
        obs, rew, done, infos = env.step(act); g = env.gen; n += 1
        ended = [b for b in range(B) if alive[b] and done[b]]
        for b in ended:
            alive[b] = False; res[b] = infos[b]; r = infos[b]
            if live: live.finish(specs[b]["id"], {"result": r["result"], "time": round(r["time"], 1), "km_route": round(r.get("routeLen", 0) / 1000, 3),
                                                  "km": round(min(r.get("dist", 0), r.get("routeLen", 1e9)) / 1000, 3), "red": bool(r.get("red"))})
        if live and (n % 50 == 0 or ended):
            for b in range(B):
                if pa[b] and (n % 50 == 0 or b in ended): live.push(specs[b]["id"], pa[b], pi[b]); pa[b], pi[b] = [], []
            live.write()
        if n % 100 == 0: print(json.dumps({"decisions": n, "sim_seconds": round(n * 0.2), "stretches_done": int(B - alive.sum()), "of": B, "minutes": round((time.time() - t0) / 60, 1)}), flush=True)
    env.close()
    for b in range(B):
        if res[b] and res[b].get("replay") and a.imagine: res[b]["replay"]["imagined"] = imag[b][-len(res[b]["replay"]["actions"]):]
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default=os.path.join(T.HERE, "runs", "tdmpc2"), help="the world-model run folder")
    p.add_argument("--ckpt", default=None, help="checkpoint (default: best.pt of the run, picked first if missing)")
    p.add_argument("--km", type=float, default=15.0, help="journey length to reach (km)"); p.add_argument("--max-km", type=float, default=20.0)
    p.add_argument("--stretch-m", type=float, default=1000, help="route length of one stretch (m)"); p.add_argument("--half", type=int, default=900, help="half size of each map window (m)")
    p.add_argument("--max-time", type=float, default=900, help="seconds before a stretch times out"); p.add_argument("--batch", type=int, default=8, help="stretches added per round")
    p.add_argument("--evolved", type=int, default=3, help="hardest evolved scenarios to include (from the run's evo/archive.json)")
    p.add_argument("--imagine", type=int, default=1, help="record what the car imagined (for the page)"); p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 1)), help="CPU cores for the simulation (the stretches are split over them)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=os.path.join(T.HERE, "runs", "drive")); p.add_argument("--port", type=int, default=8765, help="serve the monitor afterwards (0 = do not)")
    p.add_argument("--shield", type=int, default=-1, help="safety shield: 1 on, 0 off, -1 as the checkpoint was trained")
    p.add_argument("--bind", default="0.0.0.0", help="address of the page: 127.0.0.1 = only through an SSH tunnel (safer on a public server)")
    a = p.parse_args()
    dev = torch.device(a.device)
    ckpt = a.ckpt
    if not ckpt:
        ckpt = os.path.join(a.run, "best.pt")
        if not os.path.exists(ckpt):
            ns = argparse.Namespace(out=a.run, device=a.device)
            argv, sys.argv = sys.argv, [sys.argv[0]]
            try: base = T.parse()
            finally: sys.argv = argv
            T.select_best(argparse.Namespace(**{**vars(base), **vars(ns)}), dev)
        if not os.path.exists(ckpt): ckpt = os.path.join(a.run, "final.pt")
    agent, steps = load_agent(ckpt, dev)
    if a.shield >= 0: agent.a.shield = a.shield
    evolved = []
    af = os.path.join(a.run, "evo", "archive.json")
    if a.evolved and os.path.exists(af): evolved = sorted(json.load(open(af)), key=lambda r: -r["difficulty"])[: a.evolved]
    os.makedirs(a.out, exist_ok=True)
    for f in glob.glob(os.path.join(a.out, "seg_*.json")) + glob.glob(os.path.join(a.out, "journey.json")): os.remove(f)
    print(json.dumps({"test_drive": os.path.basename(ckpt), "steps": steps, "target_km": a.km, "evolved_scenarios": len(evolved), "workers": a.workers, "shield": bool(agent.a.shield)}), flush=True)
    srv, url = None, None
    if a.port:   # the page first, so the drive can be watched while it happens
        sys.path.insert(0, os.path.join(T.ROOT, "monitor")); import server as monitor
        srv, url = monitor.start(os.path.dirname(os.path.abspath(a.out)), a.port, a.bind, quiet=True)
        if srv:
            print(f"\nWatch the test drive live: {url}/drive", flush=True)
            print(f"  (from another computer: ssh -L {url.rsplit(':', 1)[1]}:localhost:{url.rsplit(':', 1)[1]} {os.environ.get('USER', 'user')}@<this server>, then http://localhost:{url.rsplit(':', 1)[1]}/drive)\n", flush=True)
    live = Live(a.out, {"ckpt": os.path.relpath(ckpt, T.ROOT), "steps": steps, "target_km": a.km})
    print("Driving all stretches in parallel; a progress line every 100 decisions (20 s of simulated time).", flush=True)
    segs, planned, t0, i0 = [], 0.0, time.time(), 0
    while planned < a.km * 1000 and planned < a.max_km * 1000:
        n = a.batch if segs else max(a.batch, math.ceil(a.km * 1000 / (a.stretch_m * 0.85)))
        specs = stretches(i0 + n, evolved, a.seed * 1000)[i0:]
        for j, sp in enumerate(specs): sp["id"] = i0 + j
        i0 += n
        for s, r in zip(specs, drive(agent, specs, a, live, steps)):
            if not r or not r.get("replay"): continue
            if planned >= a.max_km * 1000: break
            rp = r.pop("replay"); planned += r.get("routeLen", 0)
            seg = {"id": s["id"], "steps": steps, "traffic": s["traffic"], "place": s["place"], "kind": s["kind"], "evolved": s["evolved"],
                   "result": r["result"], "time": round(r["time"], 1), "km_route": round(r.get("routeLen", 0) / 1000, 3), "km": round(min(r.get("dist", 0), r.get("routeLen", 1e9)) / 1000, 3),
                   "red": bool(r.get("red")), "minGap": round(r.get("minGap", 9), 2), "with": r.get("with"), "kmh": round(r.get("dist", 0) / max(1e-6, r["time"]) * 3.6, 1)}
            json.dump(dict(seg, replay=rp), open(os.path.join(a.out, f"seg_{seg['id']:03d}.json"), "w")); segs.append(seg); live.finish(seg["id"], seg)
            print(json.dumps({k: seg[k] for k in ("id", "place", "traffic", "result", "km", "time")}, ensure_ascii=False), flush=True)
    km = sum(s["km"] for s in segs); hours = sum(s["time"] for s in segs) / 3600; cnt = lambda k: sum(s["result"] == k for s in segs)
    inc = cnt("crash") + sum(s["red"] for s in segs)
    by = {}
    for s in segs:
        k = "Evolved scenarios" if s["evolved"] else s["traffic"]; d = by.setdefault(k, {"stretches": 0, "ok": 0, "crash": 0, "hit": 0, "red": 0, "km": 0.0})
        d["stretches"] += 1; d["km"] = round(d["km"] + s["km"], 3); d["red"] += s["red"]
        if s["result"] in ("ok", "crash", "hit"): d[s["result"]] += 1
    summary = {"ckpt": os.path.relpath(ckpt, T.ROOT), "steps": steps, "stretches": len(segs), "km": round(km, 2), "km_planned": round(planned / 1000, 2), "hours": round(hours, 2),
               "completed": cnt("ok"), "crashes": cnt("crash"), "hit_by_others": cnt("hit"), "red_lights": sum(s["red"] for s in segs), "timeouts": cnt("timeout"),
               "crashes_per_100km": round(cnt("crash") / max(1e-6, km) * 100, 1), "km_per_incident": round(km / inc, 2) if inc else None,
               "mean_kmh": round(km / max(1e-6, hours), 1), "by_traffic": by, "seconds": round(time.time() - t0)}
    json.dump({"summary": summary, "segments": segs}, open(os.path.join(a.out, "journey.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({"summary": {k: v for k, v in summary.items() if k != "by_traffic"}}), flush=True)
    if srv:
        print(f"\nDone. The page stays open: {url}/drive  (Ctrl+C to stop)", flush=True)
        try:
            while True: time.sleep(3600)
        except KeyboardInterrupt: print("\n[monitor] stopped")


if __name__ == "__main__":
    main()
