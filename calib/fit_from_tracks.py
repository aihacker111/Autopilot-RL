"""
Calibrate the Phase 3 realism filter from real trajectories (for example drone footage of a
Hanoi or Saigon junction tracked with your own multi-object tracker).

  python calib/fit_from_tracks.py tracks.csv --fps 30 --out calib/measured.json [--tolerance 1.5]

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


def main(a):
    T = load(a.csv); dt = 1.0 / a.fps
    for t in T.values():
        xy = smooth(t["xy"]); v = np.gradient(xy, dt, axis=0); sp = np.linalg.norm(v, axis=1)
        acc = np.gradient(smooth(sp[:, None])[:, 0], dt)
        t.update(xy=xy, v=v, sp=sp, acc=acc, two=t["cls"] in TWO or t["cls"] not in CARS)
    # dominant flow per 5 m cell
    flow = defaultdict(lambda: np.zeros(2))
    for t in T.values():
        m = t["sp"] > 1.0
        for (x, y), (vx, vy), s in zip(t["xy"][m], t["v"][m], t["sp"][m]): flow[(int(x // 5), int(y // 5))] += np.array([vx, vy]) / s
    # wrong-way share
    wrong = tot = 0
    for t in T.values():
        if not t["two"]: continue
        for (x, y), (vx, vy), s in zip(t["xy"], t["v"], t["sp"]):
            if s < 1.0: continue
            d = flow[(int(x // 5), int(y // 5))]; n = np.linalg.norm(d)
            tot += 1
            if n > 5 and (vx * d[0] + vy * d[1]) / (s * n) < -0.5: wrong += 1
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
    # cut-ins: someone slides sideways into < 3 m in front of a car
    frames = defaultdict(list)
    for k, t in T.items():
        for i, f in enumerate(t["f"]): frames[int(f)].append((k, i))
    cut, seen = 0, set()
    for f, lst in frames.items():
        cars = [(k, i) for k, i in lst if T[k]["cls"] in CARS and T[k]["sp"][i] > 2]
        for kc, ic in cars:
            c = T[kc]; h = c["v"][ic] / c["sp"][ic]; nrm = np.array([-h[1], h[0]])
            for ko, io in lst:
                if ko == kc or (kc, ko) in seen: continue
                o = T[ko]; d = o["xy"][io] - c["xy"][ic]; lon, lat = d @ h, d @ nrm
                lat_v = (o["v"][io] - c["v"][ic]) @ nrm
                if 2.5 < lon < 5.5 and abs(lat) < 1.0 and abs(lat_v) > 0.5 and np.sign(lat_v) == -np.sign(lat + 1e-6): cut += 1; seen.add((kc, ko))
    meas = {"wrongWayShare": wrong / max(1, tot), "cutInsPerVehMin": cut / max(1e-6, veh_min), "suddenBrakesPerTwoMin": brakes / max(1e-6, two_min)}
    sp_all = np.concatenate([t["sp"] for t in T.values()]); acc_all = np.concatenate([t["acc"] for t in T.values()])
    desc = {"tracks": len(T), "minutes_of_tracks": round(veh_min, 1), "speed_kmh_p50": round(float(np.percentile(sp_all, 50) * 3.6), 1), "speed_kmh_p90": round(float(np.percentile(sp_all, 90) * 3.6), 1),
            "accel_p95": round(float(np.percentile(acc_all, 95)), 2), "decel_p05": round(float(np.percentile(acc_all, 5)), 2)}
    base = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "priors_vn.json")))
    out = {"_about": f"Measured from {os.path.basename(a.csv)} ({desc['tracks']} tracks, {desc['minutes_of_tracks']} track-minutes); max = measured x {a.tolerance}.", "_measured": {k: round(v, 4) for k, v in meas.items()}, "_describe": desc}
    for k, p in base.items():
        if k.startswith("_"): continue
        out[k] = dict(p)
        if k in meas: out[k]["max"] = round(max(meas[k] * a.tolerance, 1e-3), 4); out[k]["scale"] = round(max(meas[k] * 0.5, 1e-3), 4)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({"measured": out["_measured"], "describe": desc}, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("csv"); p.add_argument("--fps", type=float, default=30); p.add_argument("--out", default="calib/measured.json"); p.add_argument("--tolerance", type=float, default=1.5)
    main(p.parse_args())
