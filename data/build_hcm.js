// Ho Chi Minh City from OpenStreetMap -> the LOTUS road network, cut into tiles.
//   node data/build_hcm.js [data/hcm] [web/hcm]
// Input: the gzipped JSON lines written by data/extract_hcm.py. Output: web/hcm/index.json (metadata, places for search,
// an overview of the main roads) and web/hcm/t/<i>_<j>.json tiles (junctions, links, buildings, areas, water, rails, signs).
//
// What happens here:
//  1. every OSM way that vehicles use gets a cross-section (half-width, lanes, one-way, speed limit, class, layer);
//  2. ways are split where they share a node, and joined again where two pieces simply continue each other;
//  3. the OSM nodes of one real junction (a dual carriageway crossing is 4 nodes, a roundabout a whole ring) are merged
//     into one junction, at the same level only, so a flyover never merges with the junction under it;
//  4. junctions get a type: signal (an OSM signal at or just before it), roundabout, level crossing, priority (main road
//     first) or yield, plus the turn restrictions that apply there;
//  5. bridges and tunnels get a deck height; junction heights follow the links that meet there;
//  6. everything is cut into 4 km tiles so a window of the city can be loaded quickly.
"use strict";
const fs = require("fs"), path = require("path"), zlib = require("zlib");
const SRC = process.argv[2] || path.join(__dirname, "hcm"), OUT = process.argv[3] || path.join(__dirname, "..", "web", "hcm");
const TILE = 5000;
const readL = (f) => zlib.gunzipSync(fs.readFileSync(path.join(SRC, f))).toString("utf8").split("\n").filter(Boolean).map((l) => JSON.parse(l));
const t0 = Date.now(), log = (...a) => console.log(((Date.now() - t0) / 1000).toFixed(1) + "s", ...a);

/* ---------- 1. cross-sections */
const num = (s) => { const v = parseFloat(String(s || "").replace(",", ".")); return isFinite(v) ? v : null; };
const isHem = (nm) => /^(hẻm|hem|kiệt|ngõ|ngách)\b/i.test(nm || "");
const LANE_W = 3.3;
function section(t) {
  const hw = t.highway; let base = hw.replace("_link", ""), link = hw.endsWith("_link");
  if (["footway", "path", "pedestrian", "construction", "steps"].includes(hw)) return null;
  if (t.access === "no" || t.access === "private") { if (!(t.motorcycle === "yes" || t.motor_vehicle === "yes")) return null; }
  const motorNo = t.motor_vehicle === "no", carNo = motorNo || t.motorcar === "no", motoNo = motorNo || t.motorcycle === "no" || base === "motorway";
  if (carNo && motoNo) return null;
  let ow = 0;
  if (["yes", "1", "true"].includes(t.oneway) || t.junction === "roundabout" || t.junction === "circular" || (base === "motorway" && t.oneway !== "no")) ow = 1;
  else if (t.oneway === "-1" || t.oneway === "reverse") ow = -1;
  const lanesTot = num(t.lanes), width = num(t.width);
  let cls, kind = "urban", nDir, sw, vlim, shared = false;
  const name = t.name || "";
  if (base === "motorway") { cls = "highway"; kind = "highway"; nDir = 2; sw = 0; vlim = 22; }
  else if (base === "trunk") { cls = "avenue"; nDir = 2; sw = 3; vlim = 16.7; }
  else if (base === "primary") { cls = "main"; nDir = 2; sw = 4; vlim = 13.9; }
  else if (base === "secondary") { cls = "main"; nDir = 2; sw = 3.5; vlim = 13.9; }
  else if (base === "tertiary") { cls = "sec"; nDir = 1; sw = 3; vlim = 12.5; }
  else if (base === "unclassified" || base === "road" || base === "busway") { cls = "sec"; nDir = 1; sw = 2.5; vlim = 11; }
  else if (base === "residential" && name && !isHem(name)) { cls = "small"; nDir = 1; sw = 2.5; vlim = 11; }
  else if (base === "track") { cls = "rural"; kind = "rural"; nDir = 1; sw = 0; vlim = 8; shared = true; }
  else { cls = "alley"; kind = "alley"; nDir = 1; sw = 0; vlim = 5.5; shared = true; }   // hẻm, unnamed residential, service, living street
  if (link) nDir = 1;
  if (lanesTot) nDir = ow ? Math.max(1, Math.round(lanesTot)) : Math.max(1, Math.round(lanesTot / 2));
  if (ow && !lanesTot && !link && cls !== "alley") nDir = Math.max(nDir, cls === "sec" || cls === "small" ? 2 : 3);   // one-way streets are wide
  let hwid;
  if (shared) { hwid = width ? width / 2 : cls === "rural" ? 2.4 : base === "service" ? 1.7 : 2.0; if (ow) shared = false; }
  else hwid = width ? width / 2 : ow ? nDir * LANE_W / 2 : nDir * LANE_W;
  hwid = Math.max(1.3, Math.min(hwid, 16));
  const ms = num(t.maxspeed); if (ms && ms > 4) vlim = ms / 3.6;
  let level = num(t.layer) || 0; const bridge = !!t.bridge && t.bridge !== "no", tunnel = !!t.tunnel && t.tunnel !== "no" && t.tunnel !== "building_passage";
  if (bridge && level < 1) level = 1; if (tunnel && level > -1) level = -1; if (!bridge && !tunnel) level = 0;   // layers of ground roads only order drawing
  return { cls, kind, ow, lanes: nDir, hw: +hwid.toFixed(2), sw, vlim: +vlim.toFixed(1), shared, noCar: carNo, noMoto: motoNo, level, bridge, tunnel,
    name: name || t["bridge:name"] || "", ref: t.ref || "", round: t.junction === "roundabout" || t.junction === "circular" };
}
const sameSec = (p, q) => p.cls === q.cls && p.lanes === q.lanes && p.hw === q.hw && p.level === q.level && p.noCar === q.noCar && p.noMoto === q.noMoto && p.round === q.round && Math.abs(p.ow) === Math.abs(q.ow) && (p.name === q.name || !p.name || !q.name);

/* ---------- read */
const roads = readL("roads.jsonl.gz"); log("roads", roads.length);
const tagged = readL("nodes.jsonl.gz"), nodeTag = new Map(); for (const n of tagged) nodeTag.set(n.id, n);
const ways = [], cons = [];
for (const r of roads) {
  if (r.t.highway === "construction") { cons.push({ p: r.p, name: r.t.name || "", what: r.t.construction || "" }); continue; }
  const s = section(r.t); if (!s) continue;
  let n = r.n, p = r.p.map((q) => [q[0] / 10, q[1] / 10]);
  if (s.ow === -1) { n = n.slice().reverse(); p = p.reverse(); s.ow = 1; }
  ways.push({ id: r.id, n, p, s });
}
log("drivable ways", ways.length, "under construction", cons.length);

/* ---------- 2. split at shared nodes and at level crossings */
const use = new Map();
for (const w of ways) for (let i = 0; i < w.n.length; i++) { const k = w.n[i]; use.set(k, (use.get(k) || 0) + (i === 0 || i === w.n.length - 1 ? 1 : 2)); }
const isLC = (id) => { const t = nodeTag.get(id); return t && t.t.railway === "level_crossing"; };
let segs = [];
for (const w of ways) {
  let a = 0;
  for (let i = 1; i < w.n.length; i++) {
    const k = w.n[i];
    if (i === w.n.length - 1 || use.get(k) > 2 || isLC(k)) { segs.push({ way: w.id, ways: [w.id], a: w.n[a], b: k, n: w.n.slice(a, i + 1), p: w.p.slice(a, i + 1), s: w.s }); a = i; }
  }
}
log("segments", segs.length);
// node positions
const pos = new Map(); for (const g of segs) { pos.set(g.a, g.p[0]); pos.set(g.b, g.p[g.p.length - 1]); }
const plen = (p) => { let L = 0; for (let i = 1; i < p.length; i++) L += Math.hypot(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1]); return L; };
// join pieces that simply continue each other (a node used by exactly two compatible pieces, no level crossing, no signal)
function joinContinuations() {
  const inc = new Map(); const add = (k, g) => { let l = inc.get(k); if (!l) inc.set(k, l = []); l.push(g); };
  for (const g of segs) { add(g.a, g); add(g.b, g); }
  const dead = new Set();
  for (const [k, l] of inc) {
    if (l.length !== 2 || isLC(k)) continue;
    const t = nodeTag.get(k); if (t && (t.t.highway === "traffic_signals" || t.t.highway === "stop" || t.t.highway === "give_way")) continue;
    let [g, h] = l; if (g === h || dead.has(g) || dead.has(h) || !sameSec(g.s, h.s)) continue;
    // orient: g ends at k, h starts at k
    const rev = (x) => { x.n.reverse(); x.p.reverse(); [x.a, x.b] = [x.b, x.a]; };
    if (g.b !== k) { if (g.s.ow) { [g, h] = [h, g]; } }
    if (g.b !== k) { if (g.s.ow) continue; rev(g); }
    if (h.a !== k) { if (h.s.ow) continue; rev(h); }
    if (g.b !== k || h.a !== k || g.a === h.b) continue;
    g.n = g.n.concat(h.n.slice(1)); g.p = g.p.concat(h.p.slice(1)); g.b = h.b; if (!g.s.name && h.s.name) g.s = h.s; for (const w of h.ways) if (!g.ways.includes(w)) g.ways.push(w);
    dead.add(h);
    // h's far end now belongs to g
    const lb = inc.get(h.b); if (lb) { const i = lb.indexOf(h); if (i >= 0) lb[i] = g; }
  }
  segs = segs.filter((g) => !dead.has(g));
}
joinContinuations(); log("after joining continuations", segs.length);

/* ---------- 3. junction clusters (union-find over OSM nodes) */
const parent = new Map(); const find = (k) => { let r = k; while (parent.has(r) && parent.get(r) !== r) r = parent.get(r); let c = k; while (parent.has(c) && parent.get(c) !== r) { const n = parent.get(c); parent.set(c, r); c = n; } return r; };
const union = (a, b) => { const ra = find(a), rb = find(b); if (ra !== rb) parent.set(ra, rb); };
const deg = new Map(), maxHw = new Map(), lvl = new Map();
for (const g of segs) for (const k of [g.a, g.b]) { deg.set(k, (deg.get(k) || 0) + 1); maxHw.set(k, Math.max(maxHw.get(k) || 0, g.s.hw)); (lvl.get(k) || lvl.set(k, new Set()).get(k)).add(g.s.level); }
const segLen = new Map(); for (const g of segs) segLen.set(g, plen(g.p));
// roundabout rings: all their nodes become one junction
for (const g of segs) if (g.s.round) union(g.a, g.b);
// short links between junction nodes at the same level (dual carriageways, staggered side streets)
for (const g of segs) {
  if (g.s.round || g.a === g.b) continue;
  const L = segLen.get(g), da = deg.get(g.a), db = deg.get(g.b);
  if (da < 3 || db < 3) continue;
  const J = Math.min(30, Math.max(9, 0.9 * (maxHw.get(g.a) + maxHw.get(g.b))));
  if (L < J && g.s.level === 0) union(g.a, g.b);
}
// cluster members and diameter guard (a chain of short links must not grow into a district)
const roundRoots = new Set(); for (const g of segs) if (g.s.round) roundRoots.add(find(g.a));
const members = new Map(); for (const k of pos.keys()) { const r = find(k); let l = members.get(r); if (!l) members.set(r, l = []); l.push(k); }
for (const [r, l] of members) {
  if (l.length < 2) continue;
  const isRound = roundRoots.has(r);
  let cx = 0, cy = 0; for (const k of l) { cx += pos.get(k)[0]; cy += pos.get(k)[1]; } cx /= l.length; cy /= l.length;
  const dmax = Math.max(...l.map((k) => Math.hypot(pos.get(k)[0] - cx, pos.get(k)[1] - cy)));
  if (!isRound && dmax > 35) { for (const k of l) parent.set(k, k); members.set(r, null); for (const k of l) members.set(k, [k]); }
}
log("clusters ready");

/* ---------- 4. junctions and links */
const J = new Map();   // cluster root -> junction
const jOf = (k) => { const r = find(k); let j = J.get(r); if (!j) { j = { id: J.size, root: r, nodes: [], x: 0, y: 0 }; J.set(r, j); } return j; };
for (const g of segs) { for (const k of [g.a, g.b]) { const j = jOf(k); if (!j.nodes.includes(k)) j.nodes.push(k); } }
for (const j of J.values()) { let x = 0, y = 0; for (const k of j.nodes) { x += pos.get(k)[0]; y += pos.get(k)[1]; } j.x = x / j.nodes.length; j.y = y / j.nodes.length; }
// roundabouts: centre and ring size from the ring geometry
for (const g of segs) if (g.s.round) { const j = jOf(g.a); j.round = j.round || { pts: [], lanes: g.s.lanes, hw: g.s.hw, name: "" }; for (const p of g.p) j.round.pts.push(p); if (g.s.name) j.round.name = g.s.name; }
for (const j of J.values()) if (j.round) {
  const P = j.round.pts; let x = 0, y = 0; for (const p of P) { x += p[0]; y += p[1]; } x /= P.length; y /= P.length; j.x = x; j.y = y;
  const r = P.reduce((s, p) => s + Math.hypot(p[0] - x, p[1] - y), 0) / P.length;
  j.rw = Math.max(5, Math.min(14, j.round.hw * 2)); j.R = Math.max(3, r - j.rw / 2);
}
// links: segments between different junctions, endpoints moved to the junction centres
const links = [];
for (const g of segs) {
  if (g.s.round) continue;
  const A = jOf(g.a), B = jOf(g.b); if (A === B) continue;
  let p = g.p.slice();
  // replace the ends inside the junction area by straight runs from the centre
  const trim = (pts, j, fromStart) => {
    const arr = fromStart ? pts : pts.slice().reverse(), c = [j.x, j.y], rr = j.round ? j.R + j.rw : Math.max(...j.nodes.map((k) => Math.hypot(pos.get(k)[0] - j.x, pos.get(k)[1] - j.y)), 0);
    let i = 0; while (i < arr.length - 2 && Math.hypot(arr[i][0] - c[0], arr[i][1] - c[1]) < rr + 1) i++;
    const out = [c, ...arr.slice(i)]; return fromStart ? out : out.reverse();
  };
  p = trim(p, A, true); p = trim(p, B, false);
  const L = plen(p); if (L < 2) continue;
  links.push({ id: links.length, a: A.id, b: B.id, p, s: g.s, way: g.way, ways: g.ways, n0: g.n[0], n1: g.n[g.n.length - 1], osm: g.n, len: L });
}
log("junctions", J.size, "links", links.length);
// very short links left between two junctions: merge those junctions too (a box cannot fit on a 6 m road)
{
  const jl = [...J.values()], jp = new Map(jl.map((j) => [j.id, j]));
  const p2 = new Map(); const f2 = (k) => { while (p2.has(k) && p2.get(k) !== k) k = p2.get(k); return k; };
  for (const L of links) if (L.len < 7 && !jp.get(L.a).round && !jp.get(L.b).round && L.s.level === 0) { const ra = f2(L.a), rb = f2(L.b); if (ra !== rb) p2.set(ra, rb); }
  const keep = [];
  for (const L of links) { L.a = f2(L.a); L.b = f2(L.b); if (L.a !== L.b) keep.push(L); }
  // centre of merged junctions: mean of the merged centres
  const grp = new Map(); for (const j of jl) { const r = f2(j.id); if (!grp.has(r)) grp.set(r, []); grp.get(r).push(j); }
  for (const [r, l] of grp) if (l.length > 1) { const t = jp.get(r); t.x = l.reduce((s, j) => s + j.x, 0) / l.length; t.y = l.reduce((s, j) => s + j.y, 0) / l.length; for (const j of l) if (j !== t) { for (const k of j.nodes) t.nodes.push(k); j.dead = true; } }
  for (const L of keep) { const A = jp.get(L.a), B = jp.get(L.b); L.p[0] = [A.x, A.y]; L.p[L.p.length - 1] = [B.x, B.y]; L.len = plen(L.p); }
  links.length = 0; for (const L of keep) if (L.len >= 2) links.push(L); links.forEach((L, i) => (L.id = i));
  for (const j of jl) if (j.dead) J.delete(j.root);
}
// re-number junctions densely
const JL = [...J.values()]; const remap = new Map(); JL.forEach((j, i) => { remap.set(j.id, i); j.id = i; });
for (const L of links) { L.a = remap.get(L.a); L.b = remap.get(L.b); }
const inc = JL.map(() => []); for (const L of links) { inc[L.a].push(L); inc[L.b].push(L); }
log("after merging short links: junctions", JL.length, "links", links.length);

/* ---------- 5. junction types, heights, restrictions */
const nodeJ = new Map(); for (const j of JL) for (const k of j.nodes) nodeJ.set(k, j.id);
// signals: an OSM signal node inside the junction, or on an incoming link within 30 m of it
const sig = new Set();
for (const n of tagged) if (n.t.highway === "traffic_signals") {
  if (nodeJ.has(n.id)) { sig.add(nodeJ.get(n.id)); continue; }
}
const sigNodes = new Set(tagged.filter((n) => n.t.highway === "traffic_signals").map((n) => n.id));
for (const L of links) {
  const pts = L.osm; let s = 0;
  for (let i = 0; i < pts.length; i++) if (sigNodes.has(pts[i])) {
    const tn = nodeTag.get(pts[i]); if (!tn) continue; const p = [tn.p[0] / 10, tn.p[1] / 10];
    const dA = Math.hypot(p[0] - JL[L.a].x, p[1] - JL[L.a].y), dB = Math.hypot(p[0] - JL[L.b].x, p[1] - JL[L.b].y);
    if (Math.min(dA, dB) < 35) sig.add(dA < dB ? L.a : L.b);
  }
}
const signNode = (id) => { const t = nodeTag.get(id); return t ? t.t.highway : null; };
const rank = { highway: 6, avenue: 5, main: 4, sec: 3, small: 2, rural: 1, alley: 0 };
for (const j of JL) {
  const l = inc[j.id], d = l.length;
  if (!d) { j.t = "none"; continue; }
  // height: the level most of its links have at this end
  const zs = l.map((L) => (L.s.bridge ? 1 : L.s.tunnel ? -1 : 0) * Math.abs(L.s.level));
  const cnt = {}; for (const z of zs) cnt[z] = (cnt[z] || 0) + 1;
  j.lv = +Object.entries(cnt).sort((p, q) => q[1] - p[1] || Math.abs(q[0]) - Math.abs(p[0]))[0][0];
  if (d === 2 && zs[0] !== zs[1]) j.lv = 0.5 * (zs[0] + zs[1]);   // where a bridge meets its approach: half way
  if (j.round) j.t = "roundabout";
  else if (j.nodes.some(isLC)) j.t = "rail";
  else if (d <= 1) j.t = "end";
  else if (d === 2) j.t = "bend";
  else if (sig.has(j.id)) j.t = "signal";
  else { const r = l.map((L) => rank[L.s.cls]).sort((a, b) => b - a); j.t = r[0] > r[2] ? "priority" : "yield"; }
  if (j.nodes.some((k) => signNode(k) === "mini_roundabout") && d >= 3) { j.t = "roundabout"; j.R = 3; j.rw = 6; }
}
// names of junctions: a named OSM node inside (junction=yes) or a place within 40 m whose name says it is a junction
const places = readL("places.jsonl.gz").map((p) => ({ x: p.p[0] / 10, y: p.p[1] / 10, name: p.name, t: p.t }));
const juncWords = /^(ngã|vòng xoay|bùng binh|công trường|nút giao|giao lộ|cầu vượt|hầm chui)/i;
{ const grid = new Map(); const key = (x, y) => Math.floor(x / 100) + "," + Math.floor(y / 100);
  for (const j of JL) { const k = key(j.x, j.y); if (!grid.has(k)) grid.set(k, []); grid.get(k).push(j); }
  for (const p of places) if (juncWords.test(p.name) || p.t.junction) {
    let best = null, bd = p.t.junction ? 60 : 45;
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) for (const j of grid.get((Math.floor(p.x / 100) + dx) + "," + (Math.floor(p.y / 100) + dy)) || []) { const d = Math.hypot(j.x - p.x, j.y - p.y); if (d < bd && inc[j.id].length >= 3) { bd = d; best = j; } }
    if (best && !best.name) best.name = p.name;
  } }
for (const j of JL) if (j.round && j.round.name && !j.name) j.name = j.round.name;
// turn restrictions: from way / via node / to way -> pairs of links at the junction
const rels = JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(SRC, "restrictions.json.gz"))).toString());
let nBan = 0; const why = {};
const W_ = (k) => (why[k] = (why[k] || 0) + 1);
for (const r of rels) {
  if (!r.r || r.r.includes("@") || !r.m.via.length || r.m.via[0][0] !== "n" || !r.m.from.length || !r.m.to.length) { W_("form"); continue; }
  if (r.except && /motorcar/.test(r.except)) continue;
  const jid = nodeJ.get(r.m.via[0][1]); if (jid == null) { W_(pos.has(r.m.via[0][1]) ? "viaNotJ" : "viaUnknown"); continue; }
  const near = inc[jid], fw = r.m.from[0][1], tw = r.m.to[0][1];
  const fl = near.filter((L) => L.ways.includes(fw)), tl = near.filter((L) => L.ways.includes(tw));
  if (!fl.length || !tl.length) { W_(!fl.length && !tl.length ? "noBoth" : !fl.length ? "noFrom" : "noTo"); continue; }
  const j = JL[jid];
  for (const F of fl) for (const T of tl) { (r.r.startsWith("only") ? (j.only || (j.only = [])) : (j.ban || (j.ban = []))).push([F.id, T.id, r.r]); nBan++; }
}
log("signals", [...sig].length, "restrictions applied", nBan, JSON.stringify(why));

/* ---------- rails: chains of OSM rail ways */
const railsRaw = readL("rails.jsonl.gz").filter((r) => r.t.railway !== "construction" && !["yard", "siding", "spur"].includes(r.t.service));
const rails = railsRaw.map((r, i) => ({ id: i, kind: r.t.railway === "rail" ? "rail" : "metro", name: r.t.name || "", lv: (r.t.bridge ? 1 : r.t.tunnel ? -1 : 0) * Math.max(1, Math.abs(num(r.t.layer) || 1)) * (r.t.bridge || r.t.tunnel ? 1 : 0), n: r.n, p: r.p.map((q) => [q[0] / 10, q[1] / 10]) }));
// chain consecutive pieces of the same kind into lines
{ const ends = new Map(); const add = (k, r) => { if (!ends.has(k)) ends.set(k, []); ends.get(k).push(r); };
  for (const r of rails) { r.lvs = r.p.map(() => r.lv); add(r.n[0], r); add(r.n[r.n.length - 1], r); }
  const used = new Set(), chains = [];
  for (const r of rails) {
    if (used.has(r)) continue; used.add(r);
    let n = r.n.slice(), p = r.p.slice(), lv = r.lvs.slice();
    for (const dir of [1, -1]) {
      for (;;) {
        const k = dir > 0 ? n[n.length - 1] : n[0], c = (ends.get(k) || []).filter((q) => !used.has(q) && q.kind === r.kind);
        if (c.length !== 1) break; const q = c[0]; used.add(q);
        let qn = q.n.slice(), qp = q.p.slice(), ql = q.lvs.slice();
        if (dir > 0) { if (qn[0] !== k) { qn.reverse(); qp.reverse(); ql.reverse(); } n = n.concat(qn.slice(1)); p = p.concat(qp.slice(1)); lv = lv.concat(ql.slice(1)); }
        else { if (qn[qn.length - 1] !== k) { qn.reverse(); qp.reverse(); ql.reverse(); } n = qn.slice(0, -1).concat(n); p = qp.slice(0, -1).concat(p); lv = ql.slice(0, -1).concat(lv); }
      }
    }
    if (plen(p) > 150) chains.push({ id: chains.length, kind: r.kind, name: r.name, n, p, lv });
  }
  rails.length = 0; rails.push(...chains);
}
for (const j of JL) if (j.t === "rail") { let best = null; for (const R of rails) if (R.kind === "rail") { const i = R.n.findIndex((k) => j.nodes.includes(k)); if (i >= 0) best = R.id; } if (best == null) { j.t = inc[j.id].length === 2 ? "bend" : "yield"; } else j.rail = best; }
log("rail lines", rails.length, "level crossings", JL.filter((j) => j.t === "rail").length);

/* ---------- other layers */
const buildings = readL("buildings.jsonl.gz").map((b) => { const lv = num(b.t["building:levels"]), h = num(b.t.height); return { p: b.p.map((q) => [q[0] / 10, q[1] / 10]), h: h || (lv ? lv * 3.5 : 0), name: b.t.name || "", kind: b.t.amenity || b.t.shop || b.t.building }; });
const areas = readL("areas.jsonl.gz").map((a) => ({ p: a.poly.map((q) => [q[0] / 10, q[1] / 10]), kind: a.kind, name: a.name || "" }));
const water = readL("water.jsonl.gz").map((w) => w.poly ? { p: w.poly.map((q) => [q[0] / 10, q[1] / 10]), kind: "water", name: w.name || "" } : { line: w.line.map((q) => [q[0] / 10, q[1] / 10]), w: num(w.t.width) || ({ river: 40, canal: 14, stream: 5, drain: 3 }[w.t.waterway] || 5), name: w.t.name || "", kind: w.t.waterway });
const signs = tagged.filter((n) => n.t.highway === "bus_stop" || n.t.highway === "stop" || n.t.highway === "give_way" || n.t.traffic_sign || n.t.highway === "crossing")
  .map((n) => ({ x: n.p[0] / 10, y: n.p[1] / 10, k: n.t.highway === "bus_stop" ? "bus" : n.t.highway === "crossing" ? "zebra" : n.t.highway === "stop" ? "stop" : n.t.highway === "give_way" ? "yield" : "sign", v: n.t.traffic_sign || n.t.name || "" }));
log("buildings", buildings.length, "areas", areas.length, "water", water.length, "signs", signs.length);

/* ---------- simplify and write tiles */
function simplify(p, tol) {   // Douglas-Peucker
  if (p.length < 3) return p;
  const keep = new Uint8Array(p.length); keep[0] = keep[p.length - 1] = 1;
  const st = [[0, p.length - 1]];
  while (st.length) {
    const [a, b] = st.pop(); let best = -1, bd = tol;
    const [ax, ay] = p[a], [bx, by] = p[b], dx = bx - ax, dy = by - ay, L = Math.hypot(dx, dy) || 1e-9;
    for (let i = a + 1; i < b; i++) { const d = Math.abs((p[i][0] - ax) * dy - (p[i][1] - ay) * dx) / L; if (d > bd) { bd = d; best = i; } }
    if (best >= 0) { keep[best] = 1; st.push([a, best], [best, b]); }
  }
  return p.filter((_, i) => keep[i]);
}
const dm = (p) => { const o = []; for (const q of p) o.push(Math.round(q[0] * 10), Math.round(q[1] * 10)); return o; };
const tileOf = (x, y) => Math.floor(x / TILE) + "_" + Math.floor(y / TILE);
const tiles = new Map(); const T = (k) => { let t = tiles.get(k); if (!t) tiles.set(k, t = { J: [], L: [], B: [], A: [], W: [], R: [], S: [], C: [] }); return t; };
const cen = (p) => { let x = 0, y = 0; for (const q of p) { x += q[0]; y += q[1]; } return [x / p.length, y / p.length]; };
for (const j of JL) {
  if (j.t === "none") continue;
  const o = { id: j.id, x: +j.x.toFixed(1), y: +j.y.toFixed(1), t: j.t };
  if (j.name) o.nm = j.name; if (j.lv) o.lv = j.lv; if (j.R) { o.R = +j.R.toFixed(1); o.rw = +j.rw.toFixed(1); } if (j.ban) o.ban = j.ban.map((b) => [b[0], b[1]]); if (j.only) o.only = j.only.map((b) => [b[0], b[1]]); if (j.rail != null) o.rail = j.rail;
  T(tileOf(j.x, j.y)).J.push(o);
}
for (const L of links) {
  const s = L.s, p = simplify(L.p, s.cls === "alley" ? 0.5 : 0.35), m = p[Math.floor(p.length / 2)];
  const o = { id: L.id, a: L.a, b: L.b, p: dm(p), c: s.cls, k: s.kind, hw: s.hw, ln: s.lanes, sw: s.sw, vl: s.vlim };
  if (s.ow) o.ow = 1; if (s.shared) o.sh = 1; if (s.noCar) o.nc = 1; if (s.noMoto) o.nm2 = 1; if (s.level) o.lv = s.level; if (s.bridge) o.br = 1; if (s.tunnel) o.tn = 1;
  if (s.name) o.nm = s.name; if (s.ref) o.ref = s.ref;
  T(tileOf(m[0], m[1])).L.push(o);
}
for (const b of buildings) { const c = cen(b.p); const o = { p: dm(simplify(b.p.concat([b.p[0]]), 0.3).slice(0, -1)) }; if (b.h) o.h = +b.h.toFixed(1); if (b.name) o.nm = b.name; if (b.kind && b.kind !== "yes") o.k = b.kind; T(tileOf(c[0], c[1])).B.push(o); }
for (const a of areas) { const c = cen(a.p); const o = { p: dm(simplify(a.p, 1.5)), k: a.kind }; if (a.name) o.nm = a.name; T(tileOf(c[0], c[1])).A.push(o); }
for (const w of water) {
  if (w.line) { const p = simplify(w.line, 1.0); const c = p[Math.floor(p.length / 2)]; const o = { l: dm(p), w: w.w, k: w.kind }; if (w.name) o.nm = w.name; T(tileOf(c[0], c[1])).W.push(o); }
  else { const c = cen(w.p); const o = { p: dm(simplify(w.p, 1.0)), k: "water" }; if (w.name) o.nm = w.name; T(tileOf(c[0], c[1])).W.push(o); }
}
// rail lines are few and long: they go in the index, so every window sees the whole line
const railOut = rails.map((R) => { const s = simplify(R.p.map((q, i) => [q[0], q[1], i]), 0.5); return { id: R.id, k: R.kind, nm: R.name, p: dm(s), lv: s.map((q) => R.lv[q[2]]) }; });
for (const s of signs) T(tileOf(s.x, s.y)).S.push([Math.round(s.x * 10), Math.round(s.y * 10), s.k, s.v]);
for (const c of cons) { const p = simplify(c.p.map((q) => [q[0] / 10, q[1] / 10]), 0.5); const m = p[Math.floor(p.length / 2)]; T(tileOf(m[0], m[1])).C.push({ p: dm(p), nm: c.name }); }
fs.mkdirSync(path.join(OUT, "t"), { recursive: true });
const tileIdx = {};
for (const [k, t] of tiles) { const s = JSON.stringify(t); fs.writeFileSync(path.join(OUT, "t", k + ".json"), s); tileIdx[k] = { j: t.J.length, l: t.L.length, kb: Math.round(s.length / 1024) }; }
// overview: main roads, rails and big water for the city-wide map
const ov = { roads: [], rails: [], water: [] };
for (const L of links) if (rank[L.s.cls] >= 3) ov.roads.push([L.s.cls, dm(simplify(L.p, 6))]);
for (const R of rails) ov.rails.push([R.kind, dm(simplify(R.p, 6))]);
for (const w of water) if (w.line && w.w >= 14) ov.water.push([w.w, dm(simplify(w.line, 8))]); else if (w.p && w.p.length > 20) ov.water.push([0, dm(simplify(w.p, 8))]);
const meta = JSON.parse(fs.readFileSync(path.join(SRC, "meta.json"), "utf8"));
const placeOut = [];
for (const j of JL) if (j.name) placeOut.push([j.name, Math.round(j.x), Math.round(j.y), "junction"]);
for (const p of places) if (p.t.place || p.t.railway || p.t.aeroway || p.t.tourism === "attraction" || /^(chợ|ga|bến xe|cầu|công viên|bệnh viện|đại học|sân bay)/i.test(p.name)) placeOut.push([p.name, Math.round(p.x), Math.round(p.y), p.t.place || p.t.railway || p.t.aeroway || "poi"]);
const index = { meta, tile: TILE, tiles: tileIdx, places: placeOut, rails: railOut, built: new Date().toISOString(), counts: { junctions: JL.length, links: links.length, signals: JL.filter((j) => j.t === "signal").length, roundabouts: JL.filter((j) => j.t === "roundabout").length, levelCrossings: JL.filter((j) => j.t === "rail").length, restrictions: nBan, rails: rails.length } };
fs.writeFileSync(path.join(OUT, "index.json"), JSON.stringify(index));
fs.writeFileSync(path.join(OUT, "overview.json"), JSON.stringify(ov));
log("tiles", tiles.size, "index kB", Math.round(JSON.stringify(index).length / 1024), "overview kB", Math.round(JSON.stringify(ov).length / 1024), JSON.stringify(index.counts));
