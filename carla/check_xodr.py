"""Sanity-check an exported OpenDRIVE file without CARLA: references resolve, and every connecting road starts
and ends exactly on the centre of the lanes it links, with matching headings.
  python carla/check_xodr.py city7.xodr"""
import math, sys, xml.etree.ElementTree as ET

def geom_point(g, s):
    x, y, h = float(g.get("x")), float(g.get("y")), float(g.get("hdg"))
    if g.find("line") is not None: return x + s * math.cos(h), y + s * math.sin(h), h
    p = g.find("paramPoly3"); L = float(g.get("length")); t = s / L
    U = lambda q: float(p.get("a" + q)) + float(p.get("b" + q)) * t + float(p.get("c" + q)) * t * t + float(p.get("d" + q)) * t ** 3
    dU = lambda q: float(p.get("b" + q)) + 2 * float(p.get("c" + q)) * t + 3 * float(p.get("d" + q)) * t * t
    u, v = U("U"), U("V"); hu, hv = dU("U"), dU("V")
    return x + u * math.cos(h) - v * math.sin(h), y + u * math.sin(h) + v * math.cos(h), h + math.atan2(hv, hu)

def lane_point(road, s, lane):
    g = road.find("planView/geometry"); x, y, h = geom_point(g, s)
    w = 0.0
    sec = road.find("lanes/laneSection")
    side = sec.find("right" if lane < 0 else "left")
    widths = {int(l.get("id")): float(l.find("width").get("a")) for l in side.findall("lane")}
    off = sum(widths[k] for k in (range(-1, lane, -1) if lane < 0 else range(1, lane))) + widths[lane] / 2
    t = -off if lane < 0 else off
    return x - t * math.sin(h), y + t * math.cos(h), h if lane < 0 else h + math.pi

def main(path):
    root = ET.parse(path).getroot(); roads = {r.get("id"): r for r in root.findall("road")}
    worst_p, worst_h, n = 0, 0, 0
    for r in roads.values():
        if r.get("junction") == "-1": continue
        L = float(r.get("length")); pre, suc = r.find("link/predecessor"), r.find("link/successor")
        ln = r.find("lanes/laneSection/right/lane/link")
        a, b = roads[pre.get("elementId")], roads[suc.get("elementId")]
        li, lo = int(ln.find("predecessor").get("id")), int(ln.find("successor").get("id"))
        sa = float(a.get("length")) if pre.get("contactPoint") == "end" else 0.0
        sb = float(b.get("length")) if suc.get("contactPoint") == "end" else 0.0
        ax, ay, ah = lane_point(a, sa, li); bx, by, bh = lane_point(b, sb, lo)
        g = r.find("planView/geometry"); x0, y0, h0 = geom_point(g, 0); x1, y1, h1 = geom_point(g, L)
        dh = lambda p, q: abs(math.atan2(math.sin(p - q), math.cos(p - q)))
        worst_p = max(worst_p, math.hypot(x0 - ax, y0 - ay), math.hypot(x1 - bx, y1 - by)); worst_h = max(worst_h, dh(h0, ah), dh(h1, bh)); n += 1
    jids = {j.get("id") for j in root.findall("junction")}
    bad = [r.get("id") for r in roads.values() for e in r.findall("link/*") if e.get("elementType") == "junction" and e.get("elementId") not in jids]
    print(f"roads {len(roads)} connecting {n} junctions {len(jids)} | worst endpoint gap {worst_p:.3f} m, worst heading gap {math.degrees(worst_h):.2f} deg | dangling refs {len(bad)}")
    return worst_p < 0.05 and worst_h < math.radians(1) and not bad

if __name__ == "__main__": sys.exit(0 if main(sys.argv[1]) else 1)
