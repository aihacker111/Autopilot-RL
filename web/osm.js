/* ======================================================================
   LOTUS: Ho Chi Minh City from OpenStreetMap (data © OpenStreetMap
   contributors, ODbL). The whole city is stored as tiles (web/hcm/, built
   by data/build_hcm.js); a map for the simulation is a window of the city,
   about 1.2 km across, around a chosen point. Roads crossing the edge of the
   window become entries and exits, dead ends become alley homes.

   HcmCity holds the index and the tiles loaded so far. In Node the tiles
   are read from disk on demand; in the browser they must be fetched first
   (await city.ensure(x, y)) and then hcmMap() runs synchronously.
   ====================================================================== */
"use strict";
class HcmCity {
  constructor(index, readTile) {
    this.index = index; this.T = index.tile; this.tiles = new Map(); this.readTile = readTile || null;
    this.places = index.places.map(([name, x, y, kind]) => ({ name, x, y, kind }));
  }
  keys(x0, y0, x1, y1) {
    const out = [];
    for (let i = Math.floor(x0 / this.T); i <= Math.floor(x1 / this.T); i++) for (let j = Math.floor(y0 / this.T); j <= Math.floor(y1 / this.T); j++) { const k = i + "_" + j; if (this.index.tiles[k]) out.push(k); }
    return out;
  }
  // browser: fetch the tiles a window needs (base is the folder holding index.json)
  async ensure(x, y, half = 700, base = "hcm/") {
    const need = this.keys(x - half - 400, y - half - 400, x + half + 400, y + half + 400).filter((k) => !this.tiles.has(k));
    await Promise.all(need.map(async (k) => { const r = await fetch(base + "t/" + k + ".json"); this.tiles.set(k, await r.json()); }));
  }
  tile(k) { if (!this.tiles.has(k) && this.readTile) this.tiles.set(k, this.readTile(k)); return this.tiles.get(k); }
  // a random window centre on a junction of a real street (weighted toward busy, signalised areas)
  randomCentre(r) {
    const keys = Object.keys(this.index.tiles).filter((k) => this.index.tiles[k].j > 400);
    for (let tries = 0; tries < 50; tries++) {
      const k = keys[Math.floor(r() * keys.length)], t = this.tile(k); if (!t) continue;
      const J = t.J.filter((j) => j.t === "signal" || j.t === "roundabout" || (j.t === "rail" && r() < 0.5) || (j.t === "priority" && r() < 0.15));
      if (J.length) { const j = J[Math.floor(r() * J.length)]; return [j.x, j.y]; }
    }
    return [0, 0];
  }
  findPlace(q) {
    const n = (s) => s.normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/đ/g, "d").replace(/Đ/g, "D").toLowerCase();
    const k = n(q).trim(); if (!k) return [];
    return this.places.filter((p) => n(p.name).includes(k)).sort((a, b) => (a.kind === "junction" ? 0 : 1) - (b.kind === "junction" ? 0 : 1) || a.name.length - b.name.length).slice(0, 12);
  }
}

const dmPts = (a) => { const o = []; for (let i = 0; i < a.length; i += 2) o.push([a[i] / 10, a[i + 1] / 10]); return o; };
const DECK = 5.6;   // metres per bridge level
// a rail line through the given points (not smoothed: OSM already follows the track), heights resampled with it
function addRail(W, P, o) {
  const pts = resamplePoly(P, 2), S0 = cumLen(P), z0 = o.z || P.map(() => 0), z = [];
  let s = 0, k = 0;
  for (let i = 0; i < pts.length; i++) {
    if (i) s += Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]);
    while (k < S0.length - 2 && S0[k + 1] < s) k++;
    const f = Math.min(1, Math.max(0, (s - S0[k]) / Math.max(1e-6, S0[k + 1] - S0[k]))); z.push(z0[k] + (z0[k + 1] - z0[k]) * f);
  }
  const R = Object.assign({ id: W.rails.length, v: 11, cars: 6, carLen: 14, period: 170, phase: W.r() * 170 }, o, { pts, z });
  W.rails.push(R); return R;
}
/* build the simulation map for the window of half-size `half` metres around (cx, cy) */
function hcmMap(city, cx, cy, opt = {}) {
  const half = opt.half || 600, x0 = cx - half, y0 = cy - half, x1 = cx + half, y1 = cy + half;
  const inside = (p, m = 0) => p[0] > x0 + m && p[0] < x1 - m && p[1] > y0 + m && p[1] < y1 - m;
  const W = new World(opt.seed || 1, { preset: "hcm" });
  const keys = city.keys(x0 - 400, y0 - 400, x1 + 400, y1 + 400), tiles = keys.map((k) => city.tile(k)).filter(Boolean);
  const J = new Map(), Ls = [];
  for (const t of tiles) { for (const j of t.J) J.set(j.id, j); for (const L of t.L) Ls.push(L); }
  // clip a polyline that starts inside the window where it first leaves it
  const clip = (P) => {
    const out = [P[0]];
    for (let i = 1; i < P.length; i++) {
      const a = P[i - 1], b = P[i];
      if (inside(b)) { out.push(b); continue; }
      let lo = 0, hi = 1; for (let k = 0; k < 20; k++) { const m = (lo + hi) / 2; if (inside([a[0] + (b[0] - a[0]) * m, a[1] + (b[1] - a[1]) * m])) lo = m; else hi = m; }
      out.push([a[0] + (b[0] - a[0]) * lo, a[1] + (b[1] - a[1]) * lo]); break;
    }
    return out;
  };
  const plen = (p) => { let s = 0; for (let i = 1; i < p.length; i++) s += Math.hypot(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1]); return s; };
  // decide which links are in and how: whole (both junctions inside), or a stub from the inside junction to the edge
  const use = [];
  for (const L of Ls) {
    const A = J.get(L.a), B = J.get(L.b), P = dmPts(L.p);
    const ai = A && A.t !== "end" && inside([A.x, A.y], 8), bi = B && B.t !== "end" && inside([B.x, B.y], 8);
    if (ai && bi) use.push({ L, P, a: L.a, b: L.b });
    else if (ai || bi) {
      const Q = ai ? clip(P) : clip(P.slice().reverse()), len = plen(Q);
      if (len < 18) continue;
      // a stub runs from the junction outward; a one-way link keeps its direction (+1 leaving the junction)
      use.push({ L, P: Q, a: ai ? L.a : L.b, b: null, rev: !ai, dead: (ai ? B : A) && (ai ? B : A).t === "end" });
    }
  }
  // junctions left with a single arm become the end of a stub
  const deg = new Map(); for (const u of use) { deg.set(u.a, (deg.get(u.a) || 0) + 1); if (u.b != null) deg.set(u.b, (deg.get(u.b) || 0) + 1); }
  for (let it = 0; it < 3; it++) for (const u of use) {
    if (u.b != null && deg.get(u.b) === 1 && deg.get(u.a) > 1) { deg.set(u.b, 0); u.b = null; u.dead = true; }
    else if (u.b != null && deg.get(u.a) === 1 && deg.get(u.b) > 1) { deg.set(u.a, 0); u.P = u.P.slice().reverse(); u.a = u.b; u.b = null; u.rev = !u.rev; u.dead = true; }
  }
  const levels = new Map(); for (const u of use) for (const id of [u.a, u.b]) if (id != null) { if (!levels.has(id)) levels.set(id, []); levels.get(id).push(u.L.lv || 0); }
  const nodes = new Map();
  const typeOf = (j, d) => d <= 2 && j.t !== "rail" && j.t !== "roundabout" ? "bend" : j.t === "end" || j.t === "bend" ? (d >= 3 ? "yield" : "bend") : j.t;
  const nodeFor = (id) => {
    if (nodes.has(id)) return nodes.get(id);
    const j = J.get(id), d = deg.get(id) || 0;
    const n = W.node(j.x, j.y, typeOf(j, d), { gid: j.id, name: j.nm || "", kind: "urban" });
    if (j.R) { n.R = j.R; n.rw = j.rw; }
    // a junction is raised only when every road meeting there is (ramps joining on a flyover); where a bridge ends at a
    // street junction the junction stays on the ground and the bridge climbs within its own length
    const lvs = (levels.get(id) || []);
    if (lvs.length && lvs.every((l) => l > 0)) n.zFix = Math.min(...lvs) * DECK * 0.8;
    else if (lvs.length && lvs.every((l) => l < 0)) n.zFix = Math.max(...lvs) * DECK;
    if (j.ban) n.banLinks = j.ban; if (j.only) n.onlyLinks = j.only;
    if (j.rail != null) n.railGid = j.rail;
    nodes.set(id, n); return n;
  };
  const flowOf = { highway: 0.45, avenue: 0.4, main: 0.32, sec: 0.16, small: 0.08, alley: 0.012, rural: 0.02 };
  for (const u of use) {
    const L = u.L, A = nodeFor(u.a);
    const o = { gid: L.id, hw: L.hw, lanes: L.ln, sw: L.sw || 0, vlim: L.vl, kind: L.k, med: 0, bl: 0, shared: !!L.sh, noCar: !!L.nc, noMoto: !!L.nm2,
      name: L.nm || L.ref || "", level: L.lv || 0, bridge: !!L.br, tunnel: !!L.tn, flow: u.b == null ? (u.dead ? 0.01 : flowOf[L.c] || 0.05) : 0 };
    // one-way along the stored direction; the stored polyline runs from junction a to junction b
    if (L.ow) o.ow = (u.b == null ? (u.rev ? -1 : 1) : (u.a === L.a ? 1 : -1));
    if (L.lv) { const len = plen(u.P); o.zDeck = Math.sign(L.lv) * Math.min(DECK * Math.abs(L.lv), 1.2 + 0.07 * len / 2); }
    if (L.c === "highway") o.bl = 0;
    const P = u.a === L.a || u.b == null ? u.P : u.P.slice().reverse();
    const via = P.slice(1, -1);
    const lk = u.b == null ? W.road(A, P[P.length - 1], L.c, Object.assign(o, { via })) : W.road(A, nodeFor(u.b), L.c, Object.assign(o, { via }));
    if (u.b == null && u.dead && L.c === "alley") lk.home = true;
  }
  // rails: the part of each line inside the window (plus a margin so trains enter from off-screen)
  const rails = city.index.rails || [];
  const railIx = new Map();
  for (const R of rails) {
    const P = dmPts(R.p), lv = R.lv || P.map(() => 0); let best = null, cur = null;
    for (let i = 0; i < P.length; i++) {
      if (inside(P[i], -150)) { if (!cur) cur = { pts: [], lv: [] }; cur.pts.push(P[i]); cur.lv.push(lv[i] || 0); }
      else if (cur) { if (!best || cur.pts.length > best.pts.length) best = cur; cur = null; }
    }
    if (cur && (!best || cur.pts.length > best.pts.length)) best = cur;
    if (!best || best.pts.length < 2 || plen(best.pts) < 200) continue;
    const metro = R.k === "metro", hi = Math.max(...best.lv), lo = Math.min(...best.lv);
    const o = metro ? { v: 13, cars: 3, carLen: 20, period: 300, metro: true, name: R.nm, z: best.lv.map((l) => l * 7) } : { v: 8, cars: 8, carLen: 18, period: 360, name: R.nm, z: best.lv.map((l) => l * DECK) };
    if (lo < 0 && hi <= 0) o.tunnel = true;
    const r1 = addRail(W, best.pts, o); railIx.set(R.id, r1.id);
    if (metro) addRail(W, best.pts.slice().reverse(), Object.assign({}, o, { z: o.z.slice().reverse(), phase: W.r() * 300 }));   // the other track
  }
  for (const n of W.nodes) if (n.type === "rail" && !railIx.has(n.railGid)) n.type = "bend";
  // OpenStreetMap layers inside the window
  const box = (P) => P.some((p) => inside(p, -60));
  W.osm = { buildings: [], areas: [], water: [], cons: [] };
  for (const t of tiles) {
    for (const b of t.B) { const P = dmPts(b.p); if (box(P)) W.osm.buildings.push({ p: P, h: b.h || 0, nm: b.nm || "", k: b.k || "" }); }
    for (const a of t.A) { const P = dmPts(a.p); if (box(P)) W.osm.areas.push({ p: P, k: a.k, nm: a.nm || "" }); }
    for (const w of t.W) { if (w.l) { const P = dmPts(w.l); if (box(P)) W.osm.water.push({ l: P, w: w.w, nm: w.nm || "" }); } else { const P = dmPts(w.p); if (box(P)) W.osm.water.push({ p: P, nm: w.nm || "" }); } }
    for (const c of t.C) { const P = dmPts(c.p); if (box(P)) W.osm.cons.push({ p: P, nm: c.nm || "" }); }
    for (const s of t.S) { const p = [s[0] / 10, s[1] / 10]; if (inside(p)) (W.signs || (W.signs = [])).push({ x: p[0], y: p[1], k: s[2], v: s[3] }); }
  }
  // place labels inside the window (one per name nearby)
  W.labels = [];
  for (const p of city.places) if (inside([p.x, p.y], 20) && !W.labels.some((q) => q.name === p.name && Math.hypot(q.x - p.x, q.y - p.y) < 300)) W.labels.push({ x: p.x, y: p.y, name: p.name, kind: p.kind });
  // the city is far denser than OpenStreetMap's footprints: fill the free land with row houses turned to the street grid
  { const hist = new Float64Array(18);
    for (const lk of W.links) if (lk.kind !== "alley") for (let i = 1; i < lk.pts.length; i += 4) { const a = Math.atan2(lk.pts[i][1] - lk.pts[i - 1][1], lk.pts[i][0] - lk.pts[i - 1][0]); hist[Math.floor(((a % (Math.PI / 2)) + Math.PI / 2) % (Math.PI / 2) / (Math.PI / 36))]++; }
    let bi = 0; for (let i = 1; i < 18; i++) if (hist[i] > hist[bi]) bi = i;
    W.urbanFill.push({ x0: x0 - 60, y0: y0 - 60, x1: x1 + 60, y1: y1 + 60, ang: (bi + 0.5) * Math.PI / 36, dense: true }); }
  const map = W.finish(opt.name || "TP. Hồ Chí Minh");
  map.centre = [cx, cy]; map.half = half; map.pedScale = 0.3; map.city = { stamp: city.index.meta.timestamp, licence: city.index.meta.licence };
  // level crossings: the rail they sit on (finishWorld picked the nearest line; keep it on the main line)
  for (const n of map.nodes) if (n.type === "rail" && n.railGid != null && railIx.has(n.railGid)) {
    const R = map.rails[railIx.get(n.railGid)]; let best = null, s = 0;
    for (let i = 0; i < R.pts.length; i++) { if (i) s += Math.hypot(R.pts[i][0] - R.pts[i - 1][0], R.pts[i][1] - R.pts[i - 1][1]); const d = Math.hypot(R.pts[i][0] - n.x, R.pts[i][1] - n.y); if (!best || d < best.d) best = { d, rail: R.id, s }; }
    n.rail = best;
  }
  return map;
}

// a fixed set of busy places, so every scenario genome in phase 3 is judged on the same streets
const HCM_BENCH = ["Ngã tư Bảy Hiền", "Ngã tư Hàng Xanh", "Vòng xoay Lăng Cha Cả", "Vòng xoay Dân Chủ", "Ngã sáu Cộng Hòa", "Ngã tư An Sương", "Ngã tư Gò Dưa",
  "Vòng xoay Phú Lâm", "Vòng xoay Cây Gõ", "Thảo Điền", "Cầu vượt Nguyễn Thái Sơn - Nguyễn Kiệm", "Ngã tư Bốn Xã"];
function benchCentre(city, k) { const name = HCM_BENCH[((k % HCM_BENCH.length) + HCM_BENCH.length) % HCM_BENCH.length], p = city.findPlace(name)[0]; return p ? [p.x, p.y] : city.randomCentre(rngFrom(k + 1)); }
// windows are reused (a map is never changed by a simulation), so phase 2/3 workers do not rebuild the same streets
const HCM_CACHE = new Map();
function hcmMapCached(city, cx, cy, opt = {}) {
  const k = Math.round(cx) + "," + Math.round(cy) + "," + (opt.half || 600);
  if (HCM_CACHE.has(k)) { const m = HCM_CACHE.get(k); HCM_CACHE.delete(k); HCM_CACHE.set(k, m); return m; }
  const m = hcmMap(city, cx, cy, opt); HCM_CACHE.set(k, m);
  while (HCM_CACHE.size > (typeof window === "undefined" ? 6 : 2)) HCM_CACHE.delete(HCM_CACHE.keys().next().value);
  return m;
}
let HCM = null;
function setHcmCity(c) { HCM = c; }
function getHcmCity() {
  if (HCM) return HCM;
  if (typeof require === "function" && typeof module !== "undefined") {
    try {
      const fs = require("fs"), path = require("path"), base = path.join(__dirname, "hcm");
      const index = JSON.parse(fs.readFileSync(path.join(base, "index.json"), "utf8"));
      HCM = new HcmCity(index, (k) => JSON.parse(fs.readFileSync(path.join(base, "t", k + ".json"), "utf8")));
    } catch (e) { HCM = null; }
  }
  return HCM;
}
if (typeof module !== "undefined") module.exports = { HcmCity, hcmMap, hcmMapCached, setHcmCity, getHcmCity, HCM_BENCH, benchCentre };
