"""
Convert public trajectory datasets into the CSV that fit_from_tracks.py reads (frame,id,x,y,cls).

  python calib/convert_tracks.py --preset onsitevru data/onsitevru/*.csv --out calib/tracks_onsitevru.csv
  python calib/convert_tracks.py --preset delhi     data/delhi/*.csv     --out calib/tracks_delhi.csv
  python calib/convert_tracks.py --preset hetrod    data/hetrod/*.csv    --out calib/tracks_hetrod.csv
  python calib/convert_tracks.py --preset generic   my.csv --id-col track --t-col time --x-col px --y-col py --scale 0.05

Presets (column names are matched loosely: case, spaces, units in brackets and punctuation are ignored):
  onsitevru  OnSiteVRU, Shanghai (Kaggle): vehicle_id, frame_time, world_x, world_y, vehicle_type, video_id
  delhi      UAV trajectories, Delhi/Noida (Zenodo 10.5281/zenodo.17745347): vehicle ID, timestamp, local X/Y, vehicle type
  hetrod     HetroD, Taiwan: track/agent id, timestamp or frame, x/y in the local metric frame, agent type, location
  generic    anything else; give the columns with --id-col --t-col (or --frame-col) --x-col --y-col [--cls-col]
The column guesses come from the dataset papers; the script prints what it picked, and every --*-col
option overrides it. Check the printed speed statistics: they reveal a wrong unit at once.

What the converter does
  - one recording per input file (or per value of the group column, e.g. video_id): recordings are moved
    10 km apart and their frame numbers do not overlap, so vehicles of different sites never "meet"
  - time is detected as seconds, milliseconds or frame index and resampled to --hz (default 10 Hz)
  - classes are mapped to moto / bike / car / bus / truck; pedestrians are dropped; e-bikes and scooters
    count as moto; three-wheelers (auto-rickshaws, tricycles) count as car
  - writes <out>.json beside the CSV with the frame rate, so fit_from_tracks.py needs no --fps
Needs only the Python standard library and numpy (pandas only for .parquet / .feather inputs).
"""
import argparse, csv, glob, io, json, os, re, sys
from collections import Counter, defaultdict
import numpy as np

norm = lambda s: re.sub(r"[^a-z0-9]", "", re.sub(r"[\[\(].*?[\]\)]", "", s.lower()))

ALIASES = {   # in priority order; matched against normalised column names
    "id": ["vehicleid", "trackid", "agentid", "objectid", "vehid", "trajectoryid", "id", "vid", "track"],
    "t": ["frametime", "timestamp", "times", "time", "timesec", "t", "ts"],
    "frame": ["frameid", "frame", "framenum", "framenumber", "frameidx"],
    "x": ["worldx", "localx", "xm", "posx", "positionx", "centerx", "xcenter", "utmx", "x"],
    "y": ["worldy", "localy", "ym", "posy", "positiony", "centery", "ycenter", "utmy", "y"],
    "cls": ["vehicletype", "agenttype", "objecttype", "type", "class", "category", "cls", "label", "vehicleclass"],
    "group": ["videoid", "recordingid", "recording", "location", "locationid", "sceneid", "scene", "site", "clip"],
}
PRESETS = {
    "onsitevru": {"fps": 25, "prefer": {"t": ["frametime"], "x": ["worldx"], "y": ["worldy"], "group": ["videoid"]}},
    "delhi": {"fps": 30, "prefer": {"x": ["localx", "xm", "x"], "y": ["localy", "ym", "y"]}},
    "hetrod": {"fps": 30, "prefer": {"group": ["location", "locationid", "recordingid"]}},
    "generic": {"fps": 30, "prefer": {}},
}
# order matters: the first rule whose keyword appears in the class name wins
CLASS_RULES = [("drop", ["pedestrian", "person", "walker", "ped", "human", "people"]),
               ("car", ["tricycle", "m3w", "3w", "threewheel", "autorickshaw", "rickshaw", "tuktuk", "auto"]),
               ("moto", ["electric", "ebike", "ebicycle", "scooter", "moped", "motor", "m2w", "2w", "twowheel", "mc"]),
               ("bike", ["bicycle", "cycle", "bike", "cyclist"]),
               ("bus", ["bus", "minibus", "coach"]),
               ("truck", ["truck", "lorry", "hcv", "lcv", "tractor", "trailer", "heavy", "lgv", "hgv"]),
               ("car", ["car", "taxi", "van", "suv", "jeep", "sedan", "pickup", "vehicle", "lmv"])]


def map_class(name, custom):
    n = str(name).strip()
    if n in custom: return custom[n]
    k = norm(n)
    if k in custom: return custom[k]
    for out, keys in CLASS_RULES:
        if any(w in k for w in keys): return out
    return None


def read_table(path):
    """yields dict rows from csv/tsv/txt (delimiter sniffed) or parquet/feather (pandas)"""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".feather", ".pq"):
        import pandas as pd
        df = pd.read_parquet(path) if ext != ".feather" else pd.read_feather(path)
        for r in df.to_dict("records"): yield {str(k): v for k, v in r.items()}
        return
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        head = f.read(65536); f.seek(0)
        try: dialect = csv.Sniffer().sniff(head.split("\n", 1)[0] + "\n", delimiters=",;\t| ")
        except csv.Error: dialect = csv.excel
        for r in csv.DictReader(f, dialect=dialect): yield r


def pick(cols, role, a, prefer):
    forced = getattr(a, f"{role}_col", None)
    if forced:
        hit = [c for c in cols if c == forced or norm(c) == norm(forced)]
        if not hit: sys.exit(f"--{role}-col {forced!r} not found; columns are: {cols}")
        return hit[0]
    by = {norm(c): c for c in cols}
    for key in prefer.get(role, []) + ALIASES[role]:
        if key in by: return by[key]
    return None


def fnum(v):
    try: return float(v)
    except (TypeError, ValueError): return float("nan")


def convert(a):
    files = sorted(set(sum([glob.glob(p) for p in a.inputs], [])))
    if not files: sys.exit("no input files")
    P = PRESETS[a.preset]; custom = {}
    for kv in (a.cls_map.split(",") if a.cls_map else []):
        k, v = kv.split("="); custom[k.strip()] = v.strip(); custom[norm(k)] = v.strip()
    recs = defaultdict(lambda: defaultdict(list))   # recording -> track id -> [(t, x, y, cls)]
    raw_cls, picked = Counter(), None
    for path in files:
        rows = read_table(path); first = next(rows, None)
        if first is None: continue
        cols = list(first.keys())
        c = {r: pick(cols, r, a, P["prefer"]) for r in ("id", "t", "frame", "x", "y", "cls", "group")}
        if a.frame_col: c["t"] = None
        if not (c["id"] and c["x"] and c["y"] and (c["t"] or c["frame"])):
            sys.exit(f"{os.path.basename(path)}: could not find id/time/x/y columns among {cols}; use --id-col --t-col/--frame-col --x-col --y-col")
        if picked != c: print(f"{os.path.basename(path)}: id={c['id']!r} time={c['t'] or c['frame']!r} x={c['x']!r} y={c['y']!r} class={c['cls']!r} recording={c['group']!r}", flush=True); picked = c
        tc = c["t"] or c["frame"]
        for r in [first] + [x for x in rows]:
            x, y, t = fnum(r[c["x"]]), fnum(r[c["y"]]), fnum(r[tc])
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(t)): continue
            cl = r[c["cls"]] if c["cls"] else "moto"
            raw_cls[str(cl)] += 1
            g = f"{os.path.basename(path)}|{r[c['group']]}" if c["group"] else os.path.basename(path)
            recs[g][str(r[c["id"]])].append((t, x * a.scale, (-y if a.flip_y else y) * a.scale, cl))
    # time unit: frame index, seconds or milliseconds, from the typical step inside a track
    steps = []
    for tr in list(recs.values())[:5]:
        for pts in list(tr.values())[:200]:
            ts = np.unique([p[0] for p in pts])
            if len(ts) > 2: steps.append(np.median(np.diff(ts)))
    step = float(np.median(steps)) if steps else 1.0
    if a.time_unit != "auto": unit = a.time_unit
    elif not c["t"] or (abs(step - round(step)) < 1e-6 and step >= 1 and step <= 5): unit = "frame"
    elif step > 5: unit = "ms"
    else: unit = "s"
    src_fps = a.src_fps or P["fps"]
    to_s = {"frame": 1.0 / src_fps, "ms": 1e-3, "s": 1.0}[unit]
    print(f"time column read as {unit} (typical step {step:g}{'' if unit != 'frame' else f', source {src_fps:g} fps'}); output resampled to {a.hz:g} Hz", flush=True)
    # write
    out_rows, cls_count, frame0, n_tracks, mapped_unknown = [], Counter(), 0, 0, Counter()
    for ri, (g, tracks) in enumerate(sorted(recs.items())):
        xs = [p[1] for pts in tracks.values() for p in pts]; ys = [p[2] for pts in tracks.values() for p in pts]
        # recentre each recording and move recordings 10 km apart; shifts are whole multiples of the 5 m cells
        # fit_from_tracks.py uses, so the dominant-flow grid stays aligned with the data
        ox, oy = -5.0 * round(float(np.median(xs)) / 5) + ri * 10000.0, -5.0 * round(float(np.median(ys)) / 5)
        t0 = min(p[0] for pts in tracks.values() for p in pts) * to_s; fmax = 0
        for tid, pts in tracks.items():
            cl = Counter(p[3] for p in pts).most_common(1)[0][0]
            m = map_class(cl, custom) if cl != "moto" or c["cls"] else "moto"
            if m is None: mapped_unknown[str(cl)] += 1; m = a.unknown
            if m == "drop": continue
            pts.sort(key=lambda p: p[0]); seen = {}
            for t, x, y, _ in pts:
                f = int(round((t * to_s - t0) * a.hz))
                seen[f] = (x + ox, y + oy)   # several samples in one output frame: keep the last
            if len(seen) < 5: continue
            n_tracks += 1; cls_count[m] += 1; tag = f"{ri}_{tid}"
            for f, (x, y) in sorted(seen.items()):
                out_rows.append((frame0 + f, tag, round(x, 3), round(y, 3), m)); fmax = max(fmax, f)
        frame0 += fmax + int(60 * a.hz)   # a minute of empty frames between recordings
    if not out_rows: sys.exit("no tracks left after conversion")
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["frame", "id", "x", "y", "cls"]); w.writerows(out_rows)
    meta = {"fps": a.hz, "preset": a.preset, "sources": [os.path.basename(p) for p in files], "recordings": len(recs), "tracks": n_tracks,
            "classes": dict(cls_count), "raw_classes": dict(raw_cls.most_common(30)), "time_unit": unit, "scale": a.scale}
    json.dump(meta, open(a.out + ".json", "w"), indent=1)
    # sanity: two-wheeler speeds reveal unit mistakes at once
    by = defaultdict(list)
    for fr, tid, x, y, cl in out_rows: by[tid].append((fr, x, y, cl))
    sp = []
    for pts in by.values():
        if pts[0][3] not in ("moto", "bike") or len(pts) < 3: continue
        q = np.array([[p[0], p[1], p[2]] for p in pts], float); d = np.diff(q, axis=0)
        ok = d[:, 0] > 0; sp += list(np.hypot(d[ok, 1], d[ok, 2]) / (d[ok, 0] / a.hz) * 3.6)
    p90 = float(np.percentile(sp, 90)) if sp else float("nan")   # p90: queues make the median ~0 in dense traffic
    print(json.dumps({"tracks": n_tracks, "recordings": len(recs), "classes": dict(cls_count), "two_wheeler_speed_kmh_p90": round(p90, 1)}, ensure_ascii=False))
    if mapped_unknown: print(f"unrecognised classes counted as {a.unknown!r}: {dict(mapped_unknown.most_common(10))} (fix with --cls-map 'name=moto,...')")
    if sp and not (8 <= p90 <= 120):
        print(f"WARNING: two-wheeler speed p90 {p90:.1f} km/h is implausible; coordinates are probably not in metres (try --scale) or the time unit is wrong (try --time-unit / --src-fps).")
    print(f"\nwrote {a.out} ({len(out_rows)} rows). Next:\n  python calib/fit_from_tracks.py {a.out} --merge calib/priors_vn.json --out calib/measured.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="files or glob patterns (csv, tsv, txt, parquet, feather)")
    p.add_argument("--preset", choices=list(PRESETS), default="generic")
    p.add_argument("--out", required=True)
    p.add_argument("--hz", type=float, default=10, help="output frame rate (10 Hz is plenty for the statistics and much faster)")
    p.add_argument("--id-col"); p.add_argument("--t-col"); p.add_argument("--frame-col"); p.add_argument("--x-col"); p.add_argument("--y-col")
    p.add_argument("--cls-col"); p.add_argument("--group-col")
    p.add_argument("--cls-map", help="explicit class mapping, e.g. '1=car,2=moto,3=bus,7=drop'")
    p.add_argument("--unknown", default="moto", choices=["moto", "bike", "car", "bus", "truck", "drop"], help="class for unrecognised labels")
    p.add_argument("--scale", type=float, default=1.0, help="multiply coordinates by this to get metres (e.g. 0.01 for cm, metres-per-pixel for pixels)")
    p.add_argument("--flip-y", action="store_true", help="negate y (for image coordinates); the statistics do not need it")
    p.add_argument("--time-unit", choices=["auto", "s", "ms", "frame"], default="auto")
    p.add_argument("--src-fps", type=float, default=None, help="source frame rate when the time column is a frame index")
    convert(p.parse_args())
