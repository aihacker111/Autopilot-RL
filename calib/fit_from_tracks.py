"""
Calibrate the Phase 3 realism filter from real trajectories (for example drone footage of a
Hanoi or Saigon junction tracked with your own multi-object tracker).

  python calib/fit_from_tracks.py tracks.csv --fps 30 --out calib/measured.json [--tolerance 1.5]
  python calib/fit_from_tracks.py calib/tracks_delhi.csv calib/tracks_onsitevru.csv --merge calib/priors_vn.json

Public datasets are converted to this format by calib/convert_tracks.py (it also records the frame rate,
so --fps is not needed). With several inputs each is measured on its own and the loosest rate is kept;
--merge combines the result with other priors files (for example the Vietnamese ones) the same way.

Input CSV columns: frame,id,x,y[,cls]   x, y in metres on the ground plane (top-down; any origin).
cls (optional): moto, bike, car, bus, truck ... ; without it every track counts as a two-wheeler.

Measured, with the same definitions the simulator reports:
  wrongWayShare          share of two-wheeler time moving against the dominant flow of its 5 m cell
  cutInsPerVehMin        a vehicle slides sideways into the space less than 3 m in front of a car
  suddenBrakesPerTwoMin  decelerations above 3.5 m/s^2 lasting at least 0.3 s, per two-wheeler-minute
plus descriptive statistics (speed and acceleration percentiles, time headways) printed for reference.
The output keeps the priors format: max = measured x tolerance.
"""
import argparse, csv, json, math, os
from collections import defaultdict
import numpy as np

TWO = {"moto", "motorbike", "motorcycle", "bike", "bicycle", "scooter"}
CARS = {"car", "taxi", "bus", "truck", "van"}


def load(path):
    tr = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            tr[row["id"]].append((int(row["frame"]), float(row["x"]), float(row["y"]), row.get("cls", "moto").strip().lower()))
    out = {}
    for k, v in tr.items():
        v.sort()
        if len(v) < 10: continue
        a = np.array([[p[0], p[1], p[2]] for p in v], float)
        out[k] = {"f": a[:, 0], "xy": a[:, 1:], "cls": max(set(p[3] for p in v), key=[p[3] for p in v].count)}
    return out


def smooth(x, k=5):
    if len(x) < k: return x
    ker = np.ones(k) / k
    return np.stack([np.convolve(np.pad(x[:, i], (k // 2, k // 2), mode="edge"), ker, mode="valid") for i in range(x.shape[1])], 1)


def measure(path, fps):
    T = load(path); dt = 1.0 / fps
    if not T: raise SystemExit(f"{path}: no usable tracks (need at least 10 points each)")
    for t in T.values():
        xy = smooth(t["xy"]); v = np.gradient(xy, dt, axis=0); sp = np.linalg.norm(v, axis=1)
        acc = np.gradient(smooth(sp[:, None])[:, 0], dt)
        t.update(xy=xy, v=v, sp=sp, acc=acc, two=t["cls"] in TWO or t["cls"] not in CARS)
    # dominant flow per 5 m cell (unit heading vectors summed over moving samples)
    flow = defaultdict(lambda: np.zeros(2))
    for t in T.values():
        m = t["sp"] > 1.0
        if not m.any(): continue
        cells = np.floor(t["xy"][m] / 5).astype(np.int64); u = t["v"][m] / t["sp"][m][:, None]
        for key, vec in zip(map(tuple, cells), u): flow[key] += vec
    # wrong-way share: two-wheeler samples moving against the dominant flow of their cell
    wrong = tot = 0
    for t in T.values():
        if not t["two"]: continue
        m = t["sp"] >= 1.0
        if not m.any(): continue
        cells = np.floor(t["xy"][m] / 5).astype(np.int64)
        D = np.array([flow[k] for k in map(tuple, cells)]); n = np.linalg.norm(D, axis=1)
        cos = (t["v"][m] * D).sum(1) / (t["sp"][m] * np.maximum(n, 1e-9))
        tot += int(m.sum()); wrong += int(((n > 5) & (cos < -0.5)).sum())
    # sudden brakes per two-wheeler-minute
    brakes, two_min, veh_min = 0, 0.0, 0.0
    for t in T.values():
        mins = len(t["sp"]) * dt / 60; veh_min += mins
        if not t["two"]: continue
        two_min += mins
        hard = t["acc"] < -3.5; run = 0
        for h in hard:
            run = run + 1 if h else 0
            if run == max(1, int(0.3 / dt)): brakes += 1
    # cut-ins: someone slides sideways into < 3 m in front of a car (each car/other pair counted once)
    frames = defaultdict(list)
    for k, t in T.items():
        for i, f in enumerate(t["f"]): frames[int(f)].append((k, i))
    cut, seen = 0, set()
    for f, lst in frames.items():
        if len(lst) < 2: continue
        P = np.array([T[k]["xy"][i] for k, i in lst]); V = np.array([T[k]["v"][i] for k, i in lst]); S = np.array([T[k]["sp"][i] for k, i in lst])
        car = np.array([T[k]["cls"] in CARS for k, _ in lst]) & (S > 2)
        if not car.any(): continue
        for ci in np.flatnonzero(car):
            h = V[ci] / S[ci]; nrm = np.array([-h[1], h[0]]); d = P - P[ci]
            lon, lat = d @ h, d @ nrm; lat_v = (V - V[ci]) @ nrm
            hit = (lon > 2.5) & (lon < 5.5) & (np.abs(lat) < 1.0) & (np.abs(lat_v) > 0.5) & (np.sign(lat_v) == -np.sign(lat + 1e-6))
            hit[ci] = False
            for oi in np.flatnonzero(hit):
                pair = (lst[ci][0], lst[oi][0])
                if pair not in seen: seen.add(pair); cut += 1
    meas = {"wrongWayShare": wrong / max(1, tot), "cutInsPerVehMin": cut / max(1e-6, veh_min), "suddenBrakesPerTwoMin": brakes / max(1e-6, two_min)}
    sp_all = np.concatenate([t["sp"] for t in T.values()]); acc_all = np.concatenate([t["acc"] for t in T.values()])
    desc = {"tracks": len(T), "two_wheeler_share": round(sum(t["two"] for t in T.values()) / len(T), 3), "minutes_of_tracks": round(veh_min, 1),
            "speed_kmh_p50": round(float(np.percentile(sp_all, 50) * 3.6), 1), "speed_kmh_p90": round(float(np.percentile(sp_all, 90) * 3.6), 1),
            "accel_p95": round(float(np.percentile(acc_all, 95)), 2), "decel_p05": round(float(np.percentile(acc_all, 5)), 2)}
    return meas, desc


def fps_of(path, given):
    if given: return given
    side = path + ".json"
    if os.path.exists(side): return float(json.load(open(side)).get("fps", 30))
    return 30.0


def main(a):
    here = os.path.dirname(os.path.abspath(__file__))
    per = {}
    for path in a.csv:
        fps = fps_of(path, a.fps); meas, desc = measure(path, fps)
        per[os.path.basename(path)] = {"fps": fps, "measured": {k: round(v, 4) for k, v in meas.items()}, "describe": desc}
        print(json.dumps({os.path.basename(path): per[os.path.basename(path)]}, indent=1), flush=True)
    # several sources: keep the loosest (largest) rate per metric, so no source's ordinary behaviour is rejected
    keys = ["wrongWayShare", "cutInsPerVehMin", "suddenBrakesPerTwoMin"]
    meas = {k: max(p["measured"][k] for p in per.values()) for k in keys}
    base = json.load(open(os.path.join(here, "priors_vn.json")))
    out = {"_about": f"Measured from {', '.join(per)}; max = measured x {a.tolerance}" + (f", never stricter than {', '.join(os.path.basename(m) for m in a.merge)}" if a.merge else "") + ".",
           "_measured": meas, "_sources": per}
    for k, p in base.items():
        if k.startswith("_"): continue
        out[k] = dict(p)
        if k in meas: out[k]["max"] = round(max(meas[k] * a.tolerance, 1e-3), 4); out[k]["scale"] = round(max(meas[k] * 0.5, 1e-3), 4)
    # --merge: combine with other priors files (e.g. the Vietnamese priors) by taking the looser bound
    for mf in a.merge or []:
        other = json.load(open(mf))
        for k, p in other.items():
            if k.startswith("_") or not isinstance(p, dict) or "max" not in p: continue
            if k not in out: out[k] = dict(p); continue
            out[k]["max"] = max(out[k]["max"], p["max"]); out[k]["scale"] = max(out[k]["scale"], p.get("scale", 0))
    json.dump(out, open(a.out, "w"), indent=1)
    print("\nwrote", a.out); print(json.dumps({k: out[k]["max"] for k in keys}, indent=1))
    rel = os.path.relpath(os.path.abspath(a.out), os.path.dirname(here))
    print(f"use it in configs/default.yaml:  phase3.priors: {a.out if rel.startswith('..') else rel}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv", nargs="+", help="one or more tracks CSVs (each measured separately)")
    p.add_argument("--fps", type=float, default=None, help="frame rate of the CSV (default: from <csv>.json written by convert_tracks.py, else 30)")
    p.add_argument("--out", default="calib/measured.json")
    p.add_argument("--tolerance", type=float, default=1.5, help="max = measured x tolerance")
    p.add_argument("--merge", nargs="+", default=None, help="priors files to combine with (looser bound wins), e.g. calib/priors_vn.json")
    main(p.parse_args())
