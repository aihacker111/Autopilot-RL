"""Extract Ho Chi Minh City (old city boundary, as a bounding box) from a Vietnam OSM extract into compact gzipped JSON
lines for the LOTUS map builder. Runs in ~1 minute with pyosmium (pip install osmium).
  python data/extract_hcm.py data/osm/vietnam-261031.osm.pbf data/hcm
Coordinates are local metres (x east, y south) around Notre-Dame Cathedral, stored as integer decimetres."""
import gzip, json, math, os, sys, time
import osmium

W, S, E, N = 106.35, 10.35, 107.05, 11.18          # old HCMC (incl. Cu Chi and Can Gio), as a box
LAT0, LON0 = 10.77978, 106.69902
KX, KY = 111320.0 * math.cos(math.radians(LAT0)), 110574.0
def xy(loc): return [round((loc.lon - LON0) * KX * 10), round(-(loc.lat - LAT0) * KY * 10)]
def inside(loc): return loc.valid() and W < loc.lon < E and S < loc.lat < N

ROAD = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential", "living_street", "service", "road", "busway",
        "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link", "track", "construction", "pedestrian", "footway", "path"}
DROP_SERVICE = {"parking_aisle", "drive-through", "emergency_access"}
RAIL = {"rail", "subway", "light_rail", "narrow_gauge", "monorail", "construction", "tram"}
ROAD_KEYS = ["highway", "name", "ref", "lanes", "lanes:forward", "lanes:backward", "oneway", "maxspeed", "layer", "bridge", "tunnel", "junction", "width",
             "service", "access", "motor_vehicle", "motorcar", "motorcycle", "surface", "construction", "bridge:name", "turn:lanes", "sidewalk", "covered", "level"]
RAIL_KEYS = ["railway", "name", "layer", "bridge", "tunnel", "service", "usage", "gauge", "electrified", "construction"]
NODE_KEEP = {"highway": {"traffic_signals", "stop", "give_way", "crossing", "mini_roundabout", "turning_circle", "bus_stop", "motorway_junction"},
             "railway": {"level_crossing", "crossing", "station", "halt", "stop", "subway_entrance"},
             "public_transport": {"station", "stop_position"}}
BIG_AMENITY = {"hospital", "university", "college", "marketplace", "townhall", "bus_station", "place_of_worship", "school", "fuel", "police", "stadium", "cinema"}

def tags(o, keys): return {k: o.tags[k] for k in keys if k in o.tags}

def main(src, out):
    t0 = time.time(); os.makedirs(out, exist_ok=True)
    f = {k: gzip.open(os.path.join(out, k + ".jsonl.gz"), "wt", encoding="utf-8") for k in ["roads", "rails", "nodes", "buildings", "areas", "water", "places"]}
    cnt = {k: 0 for k in f}
    fp = osmium.FileProcessor(src).with_locations().with_areas().with_filter(osmium.filter.EmptyTagFilter())
    for o in fp:
        if o.is_node():
            if not inside(o.location): continue
            tg = dict(o.tags); keep = any(tg.get(k) in v for k, v in NODE_KEEP.items()) or "traffic_sign" in tg
            if keep: f["nodes"].write(json.dumps({"id": o.id, "p": xy(o.location), "t": tg}, ensure_ascii=False) + "\n"); cnt["nodes"] += 1
            nm = tg.get("name")
            if nm and ("place" in tg or tg.get("amenity") in BIG_AMENITY or "tourism" in tg or "historic" in tg or tg.get("junction") or tg.get("aeroway") or tg.get("railway") in ("station", "halt")
                       or nm.split(" ")[0].lower() in ("ngã", "vòng", "cầu", "chợ", "ga", "bến", "công", "nút", "hầm")):
                f["places"].write(json.dumps({"p": xy(o.location), "name": nm, "t": {k: tg[k] for k in ("place", "amenity", "tourism", "historic", "junction", "aeroway", "railway", "name:en") if k in tg}}, ensure_ascii=False) + "\n"); cnt["places"] += 1
        elif o.is_way():
            try: nd = o.nodes; loc0 = nd[0].location; locm = nd[len(nd) // 2].location
            except Exception: continue
            if not (inside(loc0) or inside(locm)): continue
            hw, rw = o.tags.get("highway"), o.tags.get("railway")
            try: pts = [xy(n.location) for n in nd]
            except Exception: continue
            if hw in ROAD:
                if hw == "service" and o.tags.get("service") in DROP_SERVICE: continue
                if o.tags.get("area") == "yes": continue
                f["roads"].write(json.dumps({"id": o.id, "n": [n.ref for n in nd], "p": pts, "t": tags(o, ROAD_KEYS)}, ensure_ascii=False) + "\n"); cnt["roads"] += 1
            elif rw in RAIL:
                f["rails"].write(json.dumps({"id": o.id, "n": [n.ref for n in nd], "p": pts, "t": tags(o, RAIL_KEYS)}, ensure_ascii=False) + "\n"); cnt["rails"] += 1
            elif o.tags.get("waterway") in ("river", "canal", "stream", "drain"):
                f["water"].write(json.dumps({"line": pts, "t": tags(o, ["waterway", "name", "width"])}, ensure_ascii=False) + "\n"); cnt["water"] += 1
            if "building" in o.tags and o.is_closed():
                f["buildings"].write(json.dumps({"p": pts[:-1], "t": tags(o, ["building", "building:levels", "height", "name", "amenity", "shop"])}, ensure_ascii=False) + "\n"); cnt["buildings"] += 1
            nm = o.tags.get("name")
            if nm and hw is None and rw is None and (o.tags.get("amenity") in BIG_AMENITY or "tourism" in o.tags or "aeroway" in o.tags or "leisure" in o.tags or "historic" in o.tags):
                cx = sum(p[0] for p in pts) // len(pts); cy = sum(p[1] for p in pts) // len(pts)
                f["places"].write(json.dumps({"p": [cx, cy], "name": nm, "t": tags(o, ["amenity", "tourism", "aeroway", "leisure", "historic", "name:en"])}, ensure_ascii=False) + "\n"); cnt["places"] += 1
        elif o.is_area():
            tg = o.tags; kind = None
            if tg.get("natural") == "water" or tg.get("waterway") == "riverbank" or tg.get("water"): kind = "water"
            elif tg.get("leisure") in ("park", "garden", "golf_course", "pitch", "stadium") or tg.get("landuse") in ("grass", "recreation_ground", "cemetery", "village_green"): kind = "park"
            elif tg.get("landuse") in ("farmland", "meadow", "orchard", "aquaculture", "salt_pond") or tg.get("natural") in ("wetland", "scrub", "grassland"): kind = "farm"
            elif tg.get("natural") == "wood" or tg.get("landuse") == "forest": kind = "forest"
            elif tg.get("landuse") in ("industrial", "railway", "port", "military"): kind = "industrial"
            elif tg.get("aeroway") in ("aerodrome",): kind = "airport"
            if not kind: continue
            for ring in o.outer_rings():
                try:
                    p = [xy(n) for n in ring]
                except Exception: continue
                if not any(W < n.lon < E and S < n.lat < N for n in list(ring)[:: max(1, len(ring) // 8)]): continue
                f["water" if kind == "water" else "areas"].write(json.dumps({"poly": p, "kind": kind, "name": tg.get("name")}, ensure_ascii=False) + "\n")
                cnt["water" if kind == "water" else "areas"] += 1
    for v in f.values(): v.close()
    # turn restrictions (relations): a second, cheap pass
    rels = []
    for o in osmium.FileProcessor(src, osmium.osm.RELATION).with_filter(osmium.filter.TagFilter(("type", "restriction"))):
        r = o.tags.get("restriction") or o.tags.get("restriction:motorcar") or o.tags.get("restriction:motorcycle")
        m = {"from": [], "via": [], "to": []}
        for mem in o.members:
            if mem.role in m: m[mem.role].append([mem.type, mem.ref])
        rels.append({"id": o.id, "r": r, "m": m, "except": o.tags.get("except")})
    json.dump(rels, gzip.open(os.path.join(out, "restrictions.json.gz"), "wt"))
    hdr = osmium.io.Reader(src, osmium.osm.NOTHING).header()
    meta = {"source": os.path.basename(src), "timestamp": hdr.get("osmosis_replication_timestamp") or hdr.get("timestamp"), "bbox": [W, S, E, N], "origin": [LAT0, LON0],
            "kx": KX, "ky": KY, "units": "dm", "counts": cnt, "restrictions": len(rels), "licence": "© OpenStreetMap contributors, ODbL 1.0", "seconds": round(time.time() - t0, 1)}
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(meta, ensure_ascii=False))

if __name__ == "__main__": main(sys.argv[1], sys.argv[2])
