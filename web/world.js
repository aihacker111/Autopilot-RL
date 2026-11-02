/* ======================================================================
   LOTUS world: procedural Vietnamese road networks (no rendering; runs in
   the browser or Node). Units: metres. x east, y south. Right-hand traffic.

   A network is a set of nodes joined by links. A link is a smooth polyline
   (curves, hairpins) with a cross-section: carriageway half-width, median,
   motorbike lane, sidewalk, two-way shared use (alleys, rural roads), speed
   class and zones along it (market, flood, bus stop). Every link is split at
   its middle into two "arms", one owned by each end node; roads leaving the
   map are arms without a far node (where traffic enters and leaves).

   Nodes are junctions of any number of arms at any angle: signalised
   (phases grouped by direction), unsignalised, priority (an alley mouth:
   traffic on the main road goes first), roundabout, median U-turn opening,
   or level crossing (a barrier closes the road while a train passes).

   Terrain gives every point a height; roads follow it with limited grade.
   One raster over the map holds the ground and the top of anything solid
   (buildings, walls, hillsides). It decides where houses fit and answers
   line-of-sight questions: a crest, a blind bend or the corner of an alley
   hides what is behind it.

   cityMap(seed, {preset}) builds one map: "region" (one mixed area with
   every kind of road) or a smaller focused preset ("city", "alley",
   "roundabout", "highway", "rail", "rural", "pass").
   buildPath(map, node, fromArm, toArm) gives the reference path of one
   movement through a node, from the middle of the incoming road to the
   middle of the outgoing road; consecutive movements meet exactly.
   ====================================================================== */
"use strict";
const clampN = (v, a, b) => v < a ? a : v > b ? b : v;
// seeded RNG whose state can be copied (needed to clone the world for look-ahead planning)
function rngFrom(seed) {
  const st = { a: seed >>> 0 };
  const f = () => { let a = st.a | 0; a = a + 0x6D2B79F5 | 0; st.a = a; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  f.st = st; return f;
}
const rightNormal = (t) => [-t[1], t[0]];             // y points south, so this is the driver's right
const COMPASS = ["E", "SE", "S", "SW", "W", "NW", "N", "NE"];
const COMPASS_NAME = { E: "east", SE: "south-east", S: "south", SW: "south-west", W: "west", NW: "north-west", N: "north", NE: "north-east" };
const wrapA = (a) => { while (a > Math.PI) a -= 2 * Math.PI; while (a <= -Math.PI) a += 2 * Math.PI; return a; };

/* cross-sections. hw: half of the whole carriageway (both directions, median included); med: half-width of the
   median; lanes: per direction; bl: motorbike lane width per direction (at the outer edge); sw: sidewalk width;
   shared: two-way road without a centre line, used across its whole width; vlim: speed limit (m/s);
   speedK: scales how fast drivers like to go; flow: vehicles/s entering on a road leaving the map */
const ROADS = {
  main: { hw: 7.5, med: 0, lanes: 2, bl: 0, sw: 4.5, vlim: 15, speedK: 1.0, flow: 0.5, kind: "urban" },
  sec: { hw: 5.5, med: 0, lanes: 1, bl: 0, sw: 4.0, vlim: 13, speedK: 1.0, flow: 0.3, kind: "urban" },
  small: { hw: 4.0, med: 0, lanes: 1, bl: 0, sw: 3.0, vlim: 11, speedK: 0.9, flow: 0.14, kind: "urban" },
  avenue: { hw: 8.6, med: 1.1, lanes: 2, bl: 0, sw: 5.0, vlim: 16, speedK: 1.1, flow: 0.55, kind: "urban" },
  highway: { hw: 11.5, med: 1.5, lanes: 2, bl: 3.0, sw: 0, vlim: 18, speedK: 1.45, flow: 0.8, kind: "highway" },
  alley: { hw: 2.1, med: 0, lanes: 1, bl: 0, sw: 0, vlim: 5, speedK: 0.55, flow: 0.04, shared: true, kind: "alley" },
  lane: { hw: 1.5, med: 0, lanes: 1, bl: 0, sw: 0, vlim: 4.5, speedK: 0.5, flow: 0.03, shared: true, noCar: true, kind: "alley" },
  rural: { hw: 2.7, med: 0, lanes: 1, bl: 0, sw: 0, vlim: 12, speedK: 0.85, flow: 0.12, shared: true, kind: "rural" },
  pass: { hw: 3.7, med: 0, lanes: 1, bl: 0, sw: 0, vlim: 14, speedK: 0.95, flow: 0.14, kind: "pass", maxGrade: 0.11 },
};

/* ---------------- geometry helpers */
function polyLen(p) { let L = 0; for (let i = 1; i < p.length; i++) L += Math.hypot(p[i][0] - p[i - 1][0], p[i][1] - p[i - 1][1]); return L; }
function resamplePoly(p, step) {
  const out = [p[0]]; let carry = 0;
  for (let i = 1; i < p.length; i++) {
    const a = p[i - 1], b = p[i], L = Math.hypot(b[0] - a[0], b[1] - a[1]); if (L < 1e-9) continue;
    let s = step - carry;
    while (s <= L) { out.push([a[0] + (b[0] - a[0]) * s / L, a[1] + (b[1] - a[1]) * s / L]); s += step; }
    carry = L - (s - step);
  }
  const last = p[p.length - 1], q = out[out.length - 1];
  if (Math.hypot(last[0] - q[0], last[1] - q[1]) > step * 0.35) out.push([last[0], last[1]]); else out[out.length - 1] = [last[0], last[1]];
  return out;
}
/* centripetal Catmull-Rom through control points, resampled every `step` metres */
function smoothPath(ctrl, step = 1) {
  if (ctrl.length < 3) return resamplePoly(ctrl, step);
  const c = ctrl, n = c.length, ext = [[2 * c[0][0] - c[1][0], 2 * c[0][1] - c[1][1]], ...c, [2 * c[n - 1][0] - c[n - 2][0], 2 * c[n - 1][1] - c[n - 2][1]]];
  const raw = [];
  for (let i = 1; i < ext.length - 2; i++) {
    const P0 = ext[i - 1], P1 = ext[i], P2 = ext[i + 1], P3 = ext[i + 2];
    const tj = (a, b) => Math.pow(Math.hypot(b[0] - a[0], b[1] - a[1]) + 1e-6, 0.5);
    const t0 = 0, t1 = t0 + tj(P0, P1), t2 = t1 + tj(P1, P2), t3 = t2 + tj(P2, P3);
    const N = Math.max(4, Math.ceil(Math.hypot(P2[0] - P1[0], P2[1] - P1[1]) / step * 2));
    for (let k = 0; k < N; k++) {
      const t = t1 + (t2 - t1) * k / N, L = (A, B, ta, tb) => [((tb - t) * A[0] + (t - ta) * B[0]) / (tb - ta), ((tb - t) * A[1] + (t - ta) * B[1]) / (tb - ta)];
      const A1 = L(P0, P1, t0, t1), A2 = L(P1, P2, t1, t2), A3 = L(P2, P3, t2, t3), B1 = L(A1, A2, t0, t2), B2 = L(A2, A3, t1, t3);
      raw.push(L(B1, B2, t1, t2));
    }
  }
  raw.push(c[n - 1]);
  return resamplePoly(raw, step);
}
/* offset of a polyline to its right (d > 0) using averaged vertex normals */
function offsetPoly(p, d) {
  const out = [];
  for (let i = 0; i < p.length; i++) {
    const a = p[Math.max(0, i - 1)], b = p[Math.min(p.length - 1, i + 1)], L = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
    const n = rightNormal([(b[0] - a[0]) / L, (b[1] - a[1]) / L]); out.push([p[i][0] + n[0] * d, p[i][1] + n[1] * d]);
  }
  return out;
}
/* point and unit tangent at arc length s on an arm (arrays X, Y, Z, S) */
function armAt(arm, s) {
  const S = arm.S, n = S.length; s = clampN(s, 0, S[n - 1]);
  let lo = 0, hi = n - 1; while (hi - lo > 1) { const m = (lo + hi) >> 1; if (S[m] <= s) lo = m; else hi = m; }
  const f = (s - S[lo]) / Math.max(1e-9, S[hi] - S[lo]);
  const dx = arm.X[hi] - arm.X[lo], dy = arm.Y[hi] - arm.Y[lo], L = Math.hypot(dx, dy) || 1;
  return { x: arm.X[lo] + dx * f, y: arm.Y[lo] + dy * f, z: arm.Z[lo] + (arm.Z[hi] - arm.Z[lo]) * f, tx: dx / L, ty: dy / L };
}

/* ---------------- the network builder */
class World {
  constructor(seed, opt = {}) {
    this.seed = seed; this.r = rngFrom(seed * 7919 + 17); this.opt = opt;
    this.nodes = []; this.links = []; this.hills = []; this.rails = []; this.areas = []; this.villages = []; this.urbanFill = [];
  }
  node(x, y, type = "yield", o = {}) { const n = Object.assign({ id: this.nodes.length, x, y, type, kind: "urban", arms: {} }, o); this.nodes.push(n); return n; }
  /* road from node a to node b, or to a point [x, y] (a road leaving the map) */
  road(a, b, cls, o = {}) {
    const stub = Array.isArray(b), ctrl = [[a.x, a.y], ...(o.via || []), stub ? b : [b.x, b.y]];
    const L = Object.assign({ id: this.links.length, a: a.id, b: stub ? null : b.id, cls }, ROADS[cls], o, { pts: smoothPath(ctrl, 1.0), zones: o.zones || [] });
    delete L.via; this.links.push(L); return L;
  }
  hill(x, y, h, r) { this.hills.push({ x, y, h, r }); }
  H(x, y) { let z = 0; for (const h of this.hills) { const d2 = (x - h.x) ** 2 + (y - h.y) ** 2; z += h.h * Math.exp(-d2 / (2 * h.r * h.r)); } return z; }
  area(kind, x0, y0, x1, y1) { this.areas.push({ kind, x0: Math.min(x0, x1), y0: Math.min(y0, y1), x1: Math.max(x0, x1), y1: Math.max(y0, y1) }); }
  rail(ctrl, o = {}) { const pts = smoothPath(ctrl, 2); const R = Object.assign({ id: this.rails.length, pts, v: 11, cars: 6, carLen: 14, period: 170, phase: this.r() * 170 }, o); this.rails.push(R); return R; }
  finish(name) { return finishWorld(this, name); }
}

function finishWorld(W, name) {
  const r = W.r, nodes = W.nodes, links = W.links;
  // 1. heights: nodes on the terrain, roads following it with limited grade, then smoothed
  for (const n of nodes) n.z = n.zFix != null ? n.zFix : W.H(n.x, n.y);
  for (const L of links) {
    const P = L.pts, m = P.length, z = P.map((p) => L.zDeck != null ? L.zDeck : W.H(p[0], p[1]));   // a bridge or tunnel keeps its deck height
    z[0] = nodes[L.a].z; if (L.b != null) z[m - 1] = nodes[L.b].z;
    const g = L.maxGrade || 0.07;
    for (let it = 0; it < 3; it++) {
      for (let i = 1; i < m; i++) z[i] = clampN(z[i], z[i - 1] - g, z[i - 1] + g);
      if (L.b != null) { z[m - 1] = nodes[L.b].z; for (let i = m - 2; i >= 0; i--) z[i] = clampN(z[i], z[i + 1] - g, z[i + 1] + g); }
      z[0] = nodes[L.a].z;
    }
    for (let it = 0; it < 3; it++) { const c = z.slice(); for (let i = 1; i < m - 1; i++) { let s = 0, k = 0; for (let j = Math.max(0, i - 6); j <= Math.min(m - 1, i + 6); j++) { s += c[j]; k++; } z[i] = s / k; } }
    L.z = z; const S = [0]; for (let i = 1; i < m; i++) S.push(S[i - 1] + Math.hypot(P[i][0] - P[i - 1][0], P[i][1] - P[i - 1][1])); L.S = S; L.len = S[m - 1];
  }
  // 2. arms: each link is split at its middle; roads leaving the map stay whole
  const mkArm = (n, L, idx, dirLink) => {
    const X = [], Y = [], Z = [], S = [];
    for (const i of idx) { X.push(L.pts[i][0]); Y.push(L.pts[i][1]); Z.push(L.z[i]); }
    S.push(0); for (let i = 1; i < X.length; i++) S.push(S[i - 1] + Math.hypot(X[i] - X[i - 1], Y[i] - Y[i - 1]));
    let k = 1; while (k < X.length - 1 && S[k] < 3) k++;
    const dx = X[k] - X[0], dy = Y[k] - Y[0], dl = Math.hypot(dx, dy) || 1;
    const arm = { node: n.id, link: L.id, dirLink, X, Y, Z, S, len: S[S.length - 1], dir: [dx / dl, dy / dl], ang: Math.atan2(dy, dx), to: null, flow: 0,
      cls: L.cls, kind: L.kind, hw: L.hw, med: L.med, lanes: L.lanes, bl: L.bl, sw: L.sw, vlim: L.vlim, speedK: L.speedK, shared: !!L.shared, noCar: !!L.noCar,
      market: !!L.market, encroached: !!L.encroached, walls: L.walls || (L.kind === "alley" ? "buildings" : null), zones: [],
      // one-way: +1 traffic only leaves the node along this arm, -1 traffic only arrives; 0 both ways
      ow: L.ow ? L.ow * dirLink : 0, noMoto: !!L.noMoto, name: L.name || "", level: L.level || 0 };
    arm.inOK = arm.ow !== 1; arm.outOK = arm.ow !== -1;
    // zones are given in link arc length; convert to this arm's arc length
    const Lz = L.len, half = L.b == null ? Lz : Lz / 2;
    for (const zn of L.zones) {
      const a0 = dirLink > 0 ? zn.s0 : Lz - zn.s1, a1 = dirLink > 0 ? zn.s1 : Lz - zn.s0;
      if (a1 <= 0 || a0 >= half + 0.5) continue;
      arm.zones.push(Object.assign({}, zn, { s0: Math.max(0, a0), s1: Math.min(arm.len, a1), dirArm: zn.dir ? zn.dir * dirLink : 0 }));
    }
    return arm;
  };
  for (const L of links) {
    const m = L.pts.length, A = nodes[L.a];
    if (L.b == null) { const arm = mkArm(A, L, [...Array(m).keys()], 1); arm.flow = L.flow; L.armA = arm; addArm(A, arm); continue; }
    let mid = 0; while (mid < m - 1 && L.S[mid] < L.len / 2) mid++;
    const ia = [], ib = []; for (let i = 0; i <= mid; i++) ia.push(i); for (let i = m - 1; i >= mid; i--) ib.push(i);
    const B = nodes[L.b], aA = mkArm(A, L, ia, 1), aB = mkArm(B, L, ib, -1);
    addArm(A, aA); addArm(B, aB); L.armA = aA; L.armB = aB;
    aA.to = { j: B.id, arm: aB.id }; aB.to = { j: A.id, arm: aA.id };
  }
  function addArm(n, arm) {
    let base = COMPASS[((Math.round(arm.ang / (Math.PI / 4)) % 8) + 8) % 8], id = base, k = 2;
    while (n.arms[id]) id = base + k++;
    arm.id = id; n.arms[id] = arm;
  }
  // 3. junction geometry: where each arm meets the junction area
  for (const n of nodes) {
    const arms = Object.values(n.arms).sort((p, q) => p.ang - q.ang); n.order = arms.map((a) => a.id);
    const deg = arms.length;
    if (n.type === "roundabout") { n.R = n.R || 11; n.rw = n.rw || 8; for (const a of arms) a.box = n.R + n.rw + 3.5; }
    else if (n.type === "rail") { for (const a of arms) a.box = 4.5; }
    else if (n.type === "uturn") { for (const a of arms) a.box = 6; }
    else if (deg <= 2) { for (const a of arms) a.box = 1.5; }
    else {
      for (let i = 0; i < deg; i++) {
        const a = arms[i]; let box = 1.5;
        for (const j of [(i + 1) % deg, (i + deg - 1) % deg]) {
          const b = arms[j]; let th = Math.abs(wrapA(b.ang - a.ang)); if (th > 2.6) continue; th = Math.max(th, 0.5);
          box = Math.max(box, (b.hw + a.hw * Math.cos(th)) / Math.sin(th));
        }
        a.box = Math.min(box, 40);
      }
    }
    for (const a of arms) a.box = Math.min(a.box, a.len - 4);
    n.r = Math.max(...arms.map((a) => a.box + 1), 3);
    // priority junction (alley mouth or side road): the two widest arms form the main road
    if (n.type === "priority") { const by = arms.slice().sort((p, q) => q.hw - p.hw); by[0].major = by[1].major = true; }
    if (n.type === "signal") makePhases(n, arms, r);
  }
  // 4. level crossings: where the rail line passes the node
  for (const n of nodes) if (n.type === "rail") {
    let best = null; for (const R of W.rails) { let s = 0; for (let i = 0; i < R.pts.length; i++) { if (i) s += Math.hypot(R.pts[i][0] - R.pts[i - 1][0], R.pts[i][1] - R.pts[i - 1][1]); const d = Math.hypot(R.pts[i][0] - n.x, R.pts[i][1] - n.y); if (!best || d < best.d) best = { d, rail: R.id, s }; } }
    n.rail = best;
  }
  for (const R of W.rails) { const S = [0]; for (let i = 1; i < R.pts.length; i++) S.push(S[i - 1] + Math.hypot(R.pts[i][0] - R.pts[i - 1][0], R.pts[i][1] - R.pts[i - 1][1])); R.S = S; R.len = S[S.length - 1]; }
  // 5. entries: roads leaving the map
  const entries = [];
  for (const n of nodes) for (const a of Object.values(n.arms)) if (!a.to && a.flow > 0 && a.inOK) entries.push({ j: n.id, arm: a.id, flow: a.flow, kind: a.kind });
  // turn restrictions given as pairs of link ids (gid) -> pairs of arm ids at the node
  for (const n of nodes) {
    if (!n.banLinks && !n.onlyLinks) continue;
    const armOf = (gid) => Object.values(n.arms).find((a) => links[a.link].gid === gid);
    n.ban = new Set(); n.only = {};
    for (const [f, t] of n.banLinks || []) { const A = armOf(f), B = armOf(t); if (A && B) n.ban.add(A.id + ">" + B.id); }
    for (const [f, t] of n.onlyLinks || []) { const A = armOf(f), B = armOf(t); if (A && B) (n.only[A.id] || (n.only[A.id] = new Set())).add(B.id); }
  }
  // 6. bounds
  let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  const grow = (x, y, m = 0) => { x0 = Math.min(x0, x - m); y0 = Math.min(y0, y - m); x1 = Math.max(x1, x + m); y1 = Math.max(y1, y + m); };
  for (const L of links) for (const p of L.pts) grow(p[0], p[1], 6);
  for (const n of nodes) grow(n.x, n.y, n.r + 10);
  const bounds = { x0: Math.floor(x0 - 20), y0: Math.floor(y0 - 20), x1: Math.ceil(x1 + 20), y1: Math.ceil(y1 + 20) };
  const map = { name: name || "LOTUS", seed: W.seed, preset: W.opt.preset, nodes, juncs: nodes, links, entries, bounds, rails: W.rails, hills: W.hills, areas: W.areas,
    osm: W.osm || null, labels: W.labels || [], signs: W.signs || [],
    sidewalk: 4.0, cornerR: 6, zebra: [1.0, 4.0], stop: 4.6, turn: { straight: 0.55, right: 0.25, left: 0.2 }, villages: W.villages };
  map.H = (x, y) => W.H(x, y);
  buildRaster(map, W, r);
  return map;
}

/* signal phases: arms facing each other share a phase; others get their own */
function makePhases(n, arms, r) {
  const left = arms.slice(), groups = [];
  while (left.length) {
    const a = left.shift(); let best = -1, bd = 0.75;
    left.forEach((b, i) => { const d = Math.abs(Math.abs(wrapA(b.ang - a.ang)) - Math.PI); if (d < bd) { bd = d; best = i; } });
    groups.push(best >= 0 ? [a, left.splice(best, 1)[0]] : [a]);
  }
  while (groups.length > 3) { groups.sort((p, q) => p.length - q.length); const g = groups.shift(); groups[0].push(...g); }
  n.phases = [];
  for (const g of groups) {
    const big = Math.max(...g.map((a) => a.hw)), t = Math.round(big > 7 ? 26 + r() * 8 : big > 5 ? 20 + r() * 6 : 15 + r() * 5);
    n.phases.push({ green: g.map((a) => a.id), t }, { yellow: g.map((a) => a.id), t: 3 }, { allRed: true, t: 2 });
  }
  n.cycle = n.phases.reduce((s, p) => s + p.t, 0); n.offset = Math.floor(r() * n.cycle);
}

/* ---------------- raster: ground height, top of solid things, what occupies each cell */
const OCC = { FREE: 0, ROAD: 1, WALK: 2, BUILDING: 3, RAIL: 4, STALL: 5, KEEP: 6 };
function buildRaster(map, W, r) {
  const B = map.bounds, res = 1, nx = Math.ceil((B.x1 - B.x0) / res) + 1, ny = Math.ceil((B.y1 - B.y0) / res) + 1;
  const G = new Float32Array(nx * ny), O = new Float32Array(nx * ny), C = new Uint8Array(nx * ny);
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { const z = W.H(B.x0 + i * res, B.y0 + j * res); G[j * nx + i] = z; O[j * nx + i] = z; }
  const occ = { x0: B.x0, y0: B.y0, res, nx, ny, G, O, C };
  map.occ = occ;
  const cellOf = (x, y) => { const i = Math.round((x - B.x0) / res), j = Math.round((y - B.y0) / res); return i >= 0 && j >= 0 && i < nx && j < ny ? j * nx + i : -1; };
  occ.cellOf = cellOf;
  // stamp a polyline of half-width w: code, and optionally the ground height along it (road cut and fill)
  const stampPoly = (pts, zs, w, code, setGround) => {
    for (let k = 1; k < pts.length; k++) {
      const a = pts[k - 1], b = pts[k], za = zs ? zs[k - 1] : 0, zb = zs ? zs[k] : 0;
      const i0 = Math.floor((Math.min(a[0], b[0]) - w - B.x0) / res), i1 = Math.ceil((Math.max(a[0], b[0]) + w - B.x0) / res);
      const j0 = Math.floor((Math.min(a[1], b[1]) - w - B.y0) / res), j1 = Math.ceil((Math.max(a[1], b[1]) + w - B.y0) / res);
      const dx = b[0] - a[0], dy = b[1] - a[1], L2 = dx * dx + dy * dy || 1;
      for (let j = Math.max(0, j0); j <= Math.min(ny - 1, j1); j++) for (let i = Math.max(0, i0); i <= Math.min(nx - 1, i1); i++) {
        const x = B.x0 + i * res, y = B.y0 + j * res, t = clampN(((x - a[0]) * dx + (y - a[1]) * dy) / L2, 0, 1);
        const d = Math.hypot(x - a[0] - dx * t, y - a[1] - dy * t); if (d > w) continue;
        const c = j * nx + i;
        if (code === OCC.ROAD || C[c] === OCC.FREE || (code === OCC.WALK && C[c] !== OCC.ROAD)) C[c] = Math.max(C[c] === OCC.ROAD ? OCC.ROAD : 0, code);
        if (setGround) { const z = za + (zb - za) * t; G[c] = z; O[c] = z; }
      }
    }
  };
  occ.stampPoly = stampPoly;
  // roads with their sidewalks, then junction areas (on the ground they cut through)
  // water, parks and real building footprints (OpenStreetMap) go in first, so nothing is generated on top of them
  const fillPoly = (P, code, top, keepRoads) => {
    let y0 = 1e9, y1 = -1e9; for (const p of P) { y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); }
    for (let j = Math.max(0, Math.floor((y0 - B.y0) / res)); j <= Math.min(ny - 1, Math.ceil((y1 - B.y0) / res)); j++) {
      const y = B.y0 + j * res, xs = [];
      for (let k = 0, m = P.length; k < m; k++) { const a = P[k], b = P[(k + 1) % m]; if ((a[1] <= y) !== (b[1] <= y)) xs.push(a[0] + (y - a[1]) / (b[1] - a[1]) * (b[0] - a[0])); }
      xs.sort((p, q) => p - q);
      for (let k = 0; k + 1 < xs.length; k += 2) for (let i = Math.max(0, Math.ceil((xs[k] - B.x0) / res)); i <= Math.min(nx - 1, Math.floor((xs[k + 1] - B.x0) / res)); i++) {
        const c = j * nx + i; if (keepRoads && (C[c] === OCC.ROAD || C[c] === OCC.WALK)) continue;   // a footprint drawn over the street does not block it
        C[c] = code; if (top) O[c] = Math.max(O[c], G[c] + top);
      }
    }
  };
  occ.fillPoly = fillPoly;
  if (map.osm) {
    for (const a of map.osm.areas) if (a.k === "park" || a.k === "water" || a.k === "airport") fillPoly(a.p, OCC.KEEP, 0);
    for (const w of map.osm.water) { if (w.p) fillPoly(w.p, OCC.KEEP, 0); else stampPoly(w.l, null, w.w / 2, OCC.KEEP, false); }
  }
  const ground = map.links.filter((L) => !L.level), raised = map.links.filter((L) => L.level);
  for (const L of ground) { stampPoly(L.pts, L.z, L.hw + (L.sw || 0) + 0.6, OCC.WALK, true); }
  for (const L of ground) stampPoly(L.pts, L.z, L.hw + 0.2, OCC.ROAD, true);
  for (const L of raised) if (L.level > 0) stampPoly(L.pts, null, L.hw + 0.2, OCC.ROAD, false);   // under a flyover nothing is built
  for (const n of map.nodes) {
    const rr = n.type === "roundabout" ? n.R + n.rw + 1 : n.r + 1.5;
    stampPoly([[n.x - 0.01, n.y], [n.x + 0.01, n.y]], [n.z, n.z], rr + (n.type === "roundabout" ? 0 : 3), OCC.WALK, true);
    stampPoly([[n.x - 0.01, n.y], [n.x + 0.01, n.y]], [n.z, n.z], rr, OCC.ROAD, true);
  }
  for (const R of map.rails) if (!R.tunnel) stampPoly(R.pts, null, 3.2, OCC.RAIL, false);
  if (map.osm) for (const b of map.osm.buildings) fillPoly(b.p, OCC.BUILDING, b.h || 10, true);
  // ---- buildings: rows of houses along the frontage of town roads and alleys, then back houses filling blocks
  map.buildings = [];
  const TON = ["#b8452e", "#c75b39", "#8a3a2a", "#3a6ea5", "#4f7f9e", "#9aa6ad", "#a8b0b5", "#b5a27e", "#6f8b5c", "#d2c7b0"];
  const fits = (cx, cy, ang, w, d, pad) => {
    const c = Math.cos(ang), s = Math.sin(ang), R = Math.hypot(w, d) / 2 + pad;
    for (let y = cy - R; y <= cy + R; y += res) for (let x = cx - R; x <= cx + R; x += res) {
      const lx = (x - cx) * c + (y - cy) * s, ly = -(x - cx) * s + (y - cy) * c;
      if (Math.abs(lx) > w / 2 + pad || Math.abs(ly) > d / 2 + pad) continue;
      const k = cellOf(x, y); if (k < 0 || C[k] !== OCC.FREE) return false;
    }
    return true;
  };
  const mark = (cx, cy, ang, w, d, top, code) => {
    const c = Math.cos(ang), s = Math.sin(ang), R = Math.hypot(w, d) / 2 + 1;
    for (let y = cy - R; y <= cy + R; y += res) for (let x = cx - R; x <= cx + R; x += res) {
      const lx = (x - cx) * c + (y - cy) * s, ly = -(x - cx) * s + (y - cy) * c;
      if (Math.abs(lx) > w / 2 + 0.3 || Math.abs(ly) > d / 2 + 0.3) continue;
      const k = cellOf(x, y); if (k < 0) continue; C[k] = code; O[k] = Math.max(O[k], G[k] + top);
    }
  };
  occ.fits = fits; occ.mark = mark;
  const addHouse = (cx, cy, ang, w, d, front, kind) => {
    const fl = kind === "alley" ? 2 + Math.floor(r() * 3) : kind === "village" ? 1 + Math.floor(r() * 2) : kind === "industrial" ? 1 + Math.floor(r() * 2) : 2 + Math.floor(r() * 5);
    const h = { x: cx, y: cy, ang, w, d, fl, front, kind, style: kind === "industrial" ? "ton" : r() < 0.45 ? "ton" : r() < 0.82 ? "flat" : "tile", c: TON[Math.floor(r() * TON.length)],
      awning: !!front && kind !== "industrial" && r() < 0.55, seed: Math.floor(r() * 1e9) };
    map.buildings.push(h); mark(cx, cy, ang, w, d, fl * 3.2, OCC.BUILDING); return h;
  };
  for (const L of map.links) {
    const frontage = L.kind === "alley" ? "alley" : L.kind === "urban" ? "town" : L.kind === "highway" ? "industrial" : null;
    if (!frontage) continue;
    for (const side of [1, -1]) {
      const off = L.hw + (L.sw || 0) + (frontage === "alley" ? 0.25 : 0.35) + (frontage === "industrial" ? 6 : 0);
      let s = 2;
      while (s < L.len - 2) {
        const w = frontage === "alley" ? 3.2 + r() * 1.6 : frontage === "industrial" ? 14 + r() * 16 : 3.6 + r() * 1.9;
        const d = frontage === "alley" ? 7 + r() * 5 : frontage === "industrial" ? 16 + r() * 14 : 12 + r() * 7;
        const sm = Math.min(L.len - 1, s + w / 2), i = Math.min(L.pts.length - 2, Math.round(sm));
        const p = L.pts[i], q = L.pts[i + 1], tl = Math.hypot(q[0] - p[0], q[1] - p[1]) || 1, t = [(q[0] - p[0]) / tl, (q[1] - p[1]) / tl], nn = rightNormal(t);
        const cx = p[0] + nn[0] * side * (off + d / 2), cy = p[1] + nn[1] * side * (off + d / 2), ang = Math.atan2(t[1], t[0]);
        if ((frontage !== "industrial" || r() < 0.6) && fits(cx, cy, ang, w - 0.1, d, 0.05)) addHouse(cx, cy, ang, w - 0.1, d, side > 0 ? "right" : "left", frontage === "town" ? "town" : frontage);
        s += w + (frontage === "industrial" ? 6 + r() * 20 : 0);
      }
    }
  }
  for (const f of W.urbanFill) {   // back houses on a jittered grid (optionally turned to the street grid), leaving gaps
    const a = f.ang || 0, ca = Math.cos(a), sa = Math.sin(a), cx0 = (f.x0 + f.x1) / 2, cy0 = (f.y0 + f.y1) / 2, R = Math.hypot(f.x1 - f.x0, f.y1 - f.y0) / 2;
    if (f.dense) {   // rows of narrow tube houses back to back, as in the old districts
      for (let v = -R; v < R; v += 10.4) for (let u = -R; u < R;) {
        const w = 3.6 + r() * 2, d = 8.6 + r() * 1.4, cx = cx0 + (u + w / 2) * ca - v * sa, cy = cy0 + (u + w / 2) * sa + v * ca;
        u += w + 0.25 + (r() < 0.04 ? 3 + r() * 6 : 0);
        if (cx < f.x0 || cx > f.x1 || cy < f.y0 || cy > f.y1) continue;
        if (fits(cx, cy, a, w, d, -0.4)) addHouse(cx, cy, a, w, d, null, "alley");   // terraced: neighbours touch
      }
      continue;
    }
    const sx = 9, sy = 8;
    for (let u = -R; u < R; u += sx + r() * (f.dense ? 1.2 : 3)) for (let v = -R; v < R; v += sy + r() * (f.dense ? 2 : 4)) {
      if (r() < (f.dense ? 0.06 : 0.1)) continue;
      const w = f.dense ? 4 + r() * 1.6 : 6 + r() * 3.5, d = f.dense ? 8 + r() * 4 : 5.5 + r() * 4;
      const cx = cx0 + u * ca - v * sa, cy = cy0 + u * sa + v * ca;
      if (cx < f.x0 || cx > f.x1 || cy < f.y0 || cy > f.y1) continue;
      if (fits(cx, cy, a, w, d, f.dense ? 0.35 : 0.8)) addHouse(cx, cy, a, w, d, null, f.dense ? "alley" : "back");
    }
  }
  for (const v of W.villages) {   // a few houses with yards around village junctions
    for (let k = 0; k < v.n; k++) {
      const a = r() * 6.28, dd = 14 + r() * v.r, w = 6 + r() * 4, d = 6 + r() * 4, cx = v.x + Math.cos(a) * dd, cy = v.y + Math.sin(a) * dd;
      if (fits(cx, cy, a, w, d, 1.5)) addHouse(cx, cy, a + Math.PI / 2 * Math.round(r() * 3), w, d, null, "village");
    }
  }
  // ---- static obstacles: parked cars at the curb, market stalls, parked motorbikes in alleys, stalled motorbikes in floods
  map.statics = [];
  const atLink = (L, s, d) => { const i = clampN(Math.round(s), 0, L.pts.length - 2), p = L.pts[i], q = L.pts[i + 1], tl = Math.hypot(q[0] - p[0], q[1] - p[1]) || 1, t = [(q[0] - p[0]) / tl, (q[1] - p[1]) / tl], nn = rightNormal(t); return { x: p[0] + nn[0] * d, y: p[1] + nn[1] * d, ang: Math.atan2(t[1], t[0]), z: L.z[i] }; };
  const clearOfNodes = (L, s, m) => s > (L.armA ? L.armA.box : 0) + m && s < L.len - (L.b != null && L.armB ? L.armB.box : 0) - m;
  for (const L of map.links) {
    if (L.market) {
      for (const side of [1, -1]) for (let s = 8; s < L.len - 8; s += 3.5 + r() * 2.5) {
        if (!clearOfNodes(L, s, 6)) continue;
        const k = r(), dep = 0.6 + r() * 0.8;
        if (k < 0.55) { const p = atLink(L, s, side * (L.hw - dep / 2 + 0.05)); map.statics.push({ kind: "stall", x: p.x, y: p.y, z: p.z, h: p.ang, l: 2.2 + r() * 1.2, w: dep, seed: Math.floor(r() * 1e9) }); }
        else if (k < 0.85) { const p = atLink(L, s, side * (L.hw - 0.45)); map.statics.push({ kind: "moto", x: p.x, y: p.y, z: p.z, h: p.ang + side * 1.2, l: 1.9, w: 0.7, seed: Math.floor(r() * 1e9) }); }
      }
    } else if (L.kind === "urban" && L.cls !== "avenue") {
      for (const side of [1, -1]) {
        if (r() > (L.encroached ? 0.6 : 0.3)) continue;
        let s = 20 + r() * 25;
        while (s < L.len - 20) {
          const n = 1 + Math.floor(r() * 3);
          for (let k = 0; k < n && clearOfNodes(L, s, 18); k++, s += 5.6) {
            const p = atLink(L, s, side * (L.hw - 1.05));
            map.statics.push({ kind: "parked", x: p.x, y: p.y, z: p.z, h: side > 0 ? p.ang : p.ang + Math.PI, l: 4.4, w: 1.8, seed: Math.floor(r() * 1e9) });
          }
          s += 30 + r() * 60;
        }
      }
    } else if (L.kind === "alley") {
      for (let s = 10 + r() * 20; s < L.len - 8; s += 26 + r() * 30) {
        if (!clearOfNodes(L, s, 4)) continue;
        const side = r() < 0.5 ? 1 : -1, p = atLink(L, s, side * (L.hw - 0.45));
        map.statics.push({ kind: "moto", x: p.x, y: p.y, z: p.z, h: p.ang + side * 1.25, l: 1.9, w: 0.7, seed: Math.floor(r() * 1e9) });
      }
    }
    for (const zn of L.zones) if (zn.type === "flood") {
      const n = 1 + Math.floor(r() * 2);
      for (let k = 0; k < n; k++) { const s = zn.s0 + (zn.s1 - zn.s0) * (0.2 + 0.6 * r()), side = r() < 0.5 ? 1 : -1, p = atLink(L, s, side * (L.hw - 0.5)); map.statics.push({ kind: "stalled", x: p.x, y: p.y, z: p.z, h: p.ang + (side > 0 ? 0 : Math.PI), l: 1.9, w: 0.7, seed: Math.floor(r() * 1e9) }); }
    }
  }
  for (const st of map.statics) if (st.kind === "stall") mark(st.x, st.y, st.h, st.l, st.w, 2.2, OCC.STALL);
  // ---- where people walk: sidewalks, or the edge of the road where the sidewalk is taken (or there is none)
  map.walks = [];
  for (const L of map.links) {
    if (L.kind === "highway" || L.kind === "pass") continue;
    const onRoad = !L.sw || L.encroached || L.market, off = L.market ? L.hw - 1.9 : onRoad ? L.hw - 0.55 : L.hw + L.sw * 0.5;
    const dens = L.market ? 0.09 : L.kind === "alley" ? 0.025 : L.kind === "rural" ? 0.006 : L.encroached ? 0.035 : 0.03;
    const s0 = (L.armA ? L.armA.box : 0) + 2, s1 = L.len - (L.b != null && L.armB ? L.armB.box : 0) - 2;
    if (s1 - s0 < 8) continue;
    const pair = map.walks.length;
    for (const side of [1, -1]) {
      const pts = offsetPoly(L.pts.slice(Math.floor(s0), Math.ceil(s1) + 1), side * off);
      map.walks.push({ link: L.id, side, pts, S: cumLen(pts), off, onRoad, dens, pair, market: !!L.market, kind: L.kind });
    }
  }
}
function cumLen(pts) { const S = [0]; for (let i = 1; i < pts.length; i++) S.push(S[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])); return S; }

/* line of sight between two points (x, y, eye height above the local ground); false when terrain, a building or a
   stall on the raster rises above the straight line between them */
function lineOfSight(map, ax, ay, ah, bx, by, bh) {
  const o = map.occ; if (!o) return true;
  const ka = o.cellOf(ax, ay), kb = o.cellOf(bx, by); if (ka < 0 || kb < 0) return true;
  const za = o.G[ka] + ah, zb = o.G[kb] + bh, d = Math.hypot(bx - ax, by - ay), n = Math.floor(d / 1.5);
  for (let i = 1; i < n; i++) {
    const t = i / n; if (t * d < 1.5 || (1 - t) * d < 1.5) continue;
    const k = o.cellOf(ax + (bx - ax) * t, ay + (by - ay) * t); if (k < 0) continue;
    if (o.O[k] > za + (zb - za) * t + 0.05) return false;
  }
  return true;
}
function groundAt(map, x, y) { const o = map.occ; const k = o ? o.cellOf(x, y) : -1; return k >= 0 ? o.G[k] : 0; }

/* ---------------- one movement through a node */
function laneOffset(arm) { return arm.ow ? 0 : arm.shared ? arm.hw * 0.5 : (arm.med + arm.hw) / 2; }
function bandOf(arm, c) {
  // relative to a path running at offset c (driver's right positive): own lanes, extended (oncoming half or whole shared road), motorbike lane edge
  if (arm.shared) return { lo: -arm.hw - c + 0.3, hi: arm.hw - c - 0.3, loX: -arm.hw - c + 0.3, bike: NaN, ctr: -c };
  if (arm.ow) { const lo = -arm.hw - c + 0.35; return { lo, hi: arm.hw - c - 0.35, loX: lo, bike: arm.bl > 0 ? arm.hw - arm.bl - c : NaN, ctr: lo - 1 }; }   // one-way: the whole carriageway
  const lo = arm.med > 0 ? arm.med - c + 0.45 : -c + 0.35;
  return { lo, hi: arm.hw - c - 0.35, loX: arm.med > 0 ? lo : -arm.hw - c + 0.4, bike: arm.bl > 0 ? arm.hw - arm.bl - c : NaN, ctr: -c };
}
function turnKind(map, jn, A, B) {
  if (A === B) return "uturn";
  const tA = [-A.dir[0], -A.dir[1]], cr = tA[0] * B.dir[1] - tA[1] * B.dir[0], dt = tA[0] * B.dir[0] + tA[1] * B.dir[1];
  if (dt > 0.82) return "straight";
  if (dt < -0.85) return "uturn";
  return cr > 0 ? "right" : "left";
}
function buildPath(map, jn, from, to) {
  const A = jn.arms[from], B = jn.arms[to], z0 = map.zebra[0];
  const cA = laneOffset(A), cB = laneOffset(B), kind = turnKind(map, jn, A, B);
  const pts = [], seg = [];   // seg: 0 incoming, 1 junction, 2 outgoing
  const sA = Math.min(A.len - 0.5, A.box + z0), sB = Math.min(B.len - 0.5, B.box + z0);
  // incoming half: travel toward the node, so the driver's right is the arm's left
  for (let s = A.len; s > sA; s -= 0.5) { const p = armAt(A, s), n = rightNormal([p.tx, p.ty]); pts.push([p.x - n[0] * cA, p.y - n[1] * cA, p.z]); seg.push(0); }
  const pa = armAt(A, sA), na = rightNormal([pa.tx, pa.ty]), pin = [pa.x - na[0] * cA, pa.y - na[1] * cA, pa.z], tA = [-pa.tx, -pa.ty];
  const pb = armAt(B, sB), nb = rightNormal([pb.tx, pb.ty]), pout = [pb.x + nb[0] * cB, pb.y + nb[1] * cB, pb.z], tB = [pb.tx, pb.ty];
  pts.push(pin); seg.push(1);
  let ring = null;
  if (jn.type === "roundabout") {
    // join the ring, go round it anticlockwise on the map (decreasing angle with y south), leave it
    const Rr = jn.R + jn.rw * 0.5, ea = Math.atan2(pin[1] - jn.y, pin[0] - jn.x), xa = Math.atan2(pout[1] - jn.y, pout[0] - jn.x), blend = 0.42;
    let a0 = ea - blend, a1 = xa + blend, sweep = a0 - a1; while (sweep <= 0.15) sweep += 2 * Math.PI; while (sweep > 2 * Math.PI + 0.15) sweep -= 2 * Math.PI;
    const ringPt = (a) => [jn.x + Math.cos(a) * Rr, jn.y + Math.sin(a) * Rr];
    const e0 = ringPt(a0), e0t = [Math.sin(a0), -Math.cos(a0)], k1 = Math.hypot(e0[0] - pin[0], e0[1] - pin[1]) * 0.5;
    bez(pts, seg, pin, [pin[0] + tA[0] * k1, pin[1] + tA[1] * k1], [e0[0] - e0t[0] * k1, e0[1] - e0t[1] * k1], e0, jn.z, 1);
    const N = Math.max(6, Math.ceil(sweep * Rr / 0.5));
    for (let i = 1; i <= N; i++) { const a = a0 - sweep * i / N, p = ringPt(a); pts.push([p[0], p[1], jn.z]); seg.push(3); }
    const a1e = a0 - sweep, e1 = ringPt(a1e), e1t = [Math.sin(a1e), -Math.cos(a1e)], k2 = Math.hypot(pout[0] - e1[0], pout[1] - e1[1]) * 0.5;
    bez(pts, seg, e1, [e1[0] + e1t[0] * k2, e1[1] + e1t[1] * k2], [pout[0] - tB[0] * k2, pout[1] - tB[1] * k2], pout, jn.z, 1);
    ring = { Rr };
  } else {
    const dist = Math.hypot(pout[0] - pin[0], pout[1] - pin[1]);
    const k = kind === "uturn" ? Math.max(3.5, dist * 0.75) : dist * (kind === "straight" ? 0.36 : 0.5);
    bez(pts, seg, pin, [pin[0] + tA[0] * k, pin[1] + tA[1] * k], [pout[0] - tB[0] * k, pout[1] - tB[1] * k], pout, (pin[2] + pout[2]) / 2, 1);
  }
  for (let s = sB + 0.5; s <= B.len + 1e-6; s += 0.5) { const p = armAt(B, s), n = rightNormal([p.tx, p.ty]); pts.push([p.x + n[0] * cB, p.y + n[1] * cB, p.z]); seg.push(2); }
  // resample every 0.5 m
  const raw = [0]; for (let i = 1; i < pts.length; i++) raw.push(raw[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
  const len = raw[raw.length - 1], ds = 0.5, n = Math.floor(len / ds) + 1;
  const X = new Float32Array(n), Y = new Float32Array(n), Z = new Float32Array(n), TX = new Float32Array(n), TY = new Float32Array(n), K = new Float32Array(n), GR = new Float32Array(n);
  const DMIN = new Float32Array(n), DMAX = new Float32Array(n), DMINX = new Float32Array(n), BIKE = new Float32Array(n), CTR = new Float32Array(n), VMAX = new Float32Array(n), SPD = new Float32Array(n), FLOOD = new Float32Array(n), MKT = new Uint8Array(n), SEG = new Uint8Array(n);
  let j = 0;
  for (let i = 0; i < n; i++) {
    const s = i * ds; while (j < raw.length - 2 && raw[j + 1] < s) j++;
    const f = (s - raw[j]) / Math.max(1e-6, raw[j + 1] - raw[j]);
    X[i] = pts[j][0] + (pts[j + 1][0] - pts[j][0]) * f; Y[i] = pts[j][1] + (pts[j + 1][1] - pts[j][1]) * f; Z[i] = pts[j][2] + (pts[j + 1][2] - pts[j][2]) * f; SEG[i] = seg[f < 0.5 ? j : j + 1];
  }
  for (let i = 0; i < n; i++) { const a = Math.max(0, i - 1), b = Math.min(n - 1, i + 1); const dx = X[b] - X[a], dy = Y[b] - Y[a], l = Math.hypot(dx, dy) || 1; TX[i] = dx / l; TY[i] = dy / l; }
  for (let i = 0; i < n; i++) {
    const a = Math.max(0, i - 2), b = Math.min(n - 1, i + 2);
    let dth = Math.atan2(TY[b], TX[b]) - Math.atan2(TY[a], TX[a]); dth = Math.atan2(Math.sin(dth), Math.cos(dth));
    K[i] = Math.abs(dth) / Math.max(ds, (b - a) * ds);
    const a2 = Math.max(0, i - 4), b2 = Math.min(n - 1, i + 4); GR[i] = (Z[b2] - Z[a2]) / Math.max(ds, (b2 - a2) * ds);
  }
  let sIn = 0, sOut = len;
  { let best = 1e9; for (let i = 0; i < n; i++) { const d = Math.hypot(X[i] - pin[0], Y[i] - pin[1]); if (d < best) { best = d; sIn = i * ds; } } }
  { let best = 1e9; for (let i = 0; i < n; i++) { const d = Math.hypot(X[i] - pout[0], Y[i] - pout[1]); if (d < best) { best = d; sOut = i * ds; } } }
  const sStop = sIn - (map.stop - z0);
  const bA = bandOf(A, cA), bB = bandOf(B, cB), Rr = ring ? ring.Rr : 0;
  const bR = ring ? { lo: jn.R - Rr + 0.35, hi: jn.R + jn.rw - Rr - 0.35 } : null;
  const priority = jn.type === "priority" ? (A.major && B.major ? 2 : A.major ? 1 : 0) : jn.type === "uturn" ? (kind === "uturn" ? 0 : 2) : jn.type === "roundabout" ? 0 : 1;
  for (let i = 0; i < n; i++) {
    const s = i * ds;
    let b;
    if (s <= sIn) b = bA; else if (s >= sOut) b = bB;
    else if (ring && SEG[i] === 3) b = { lo: bR.lo, hi: bR.hi, loX: bR.lo, bike: NaN, ctr: bR.lo };
    else { const f = (s - sIn) / Math.max(1e-6, sOut - sIn); b = { lo: Math.min(bA.lo, bB.lo) * 0 + (bA.lo + (bB.lo - bA.lo) * f) - 0.8, hi: bA.hi + (bB.hi - bA.hi) * f + 0.8, bike: NaN, ctr: bA.ctr + (bB.ctr - bA.ctr) * f }; b.loX = b.lo; if (kind === "uturn") { b.lo -= 0.6; b.loX = b.lo; } }
    DMIN[i] = b.lo; DMAX[i] = b.hi; DMINX[i] = b.loX; BIKE[i] = b.bike; CTR[i] = b.ctr;
    const arm = s <= (sIn + sOut) / 2 ? A : B;
    SPD[i] = arm.speedK;
    VMAX[i] = Math.min(arm.vlim, Math.sqrt(2.6 / Math.max(K[i], 1e-3)));
  }
  // zones along the two halves: flood depth, market, bus stops for this direction of travel
  const busStops = [];
  const zoneOnPath = (arm, half) => {
    for (const zn of arm.zones) {
      let p0, p1;
      if (half === 0) { p0 = A.len - zn.s1; p1 = A.len - zn.s0; } else { p0 = sOut + (zn.s0 - sB); p1 = sOut + (zn.s1 - sB); }
      if (half === 0 && (p1 < 0 || p0 > sIn)) continue; if (half === 1 && (p1 < sOut || p0 > len)) continue;
      const i0 = Math.max(0, Math.floor(p0 / ds)), i1 = Math.min(n - 1, Math.ceil(p1 / ds));
      if (zn.type === "flood") for (let i = i0; i <= i1; i++) { const u = (i * ds - p0) / Math.max(1, p1 - p0), edge = Math.min(1, Math.min(u, 1 - u) * 5); FLOOD[i] = Math.max(FLOOD[i], zn.depth * edge); }
      if (zn.type === "market") for (let i = i0; i <= i1; i++) MKT[i] = 1;
      if (zn.type === "bus") { const travelDir = half === 0 ? -1 : 1; if (!zn.dirArm || zn.dirArm === travelDir) busStops.push({ s0: Math.max(0, p0), s1: Math.min(len, p1) }); }
    }
  };
  zoneOnPath(A, 0); zoneOnPath(B, 1);
  if (A.market) for (let i = 0; i < n && i * ds <= sIn; i++) MKT[i] = 1;
  if (B.market) for (let i = Math.floor(sOut / ds); i < n; i++) MKT[i] = 1;
  for (let i = 0; i < n; i++) {
    const s = i * ds;
    if (FLOOD[i] > 0.02) VMAX[i] = Math.min(VMAX[i], clampN(3.6 - FLOOD[i] * 5, 1.6, 3.6));
    if (MKT[i]) VMAX[i] = Math.min(VMAX[i], 4.5);
    if (s > sStop - 1 && s < sIn + 4) {
      if (jn.type === "yield") VMAX[i] = Math.min(VMAX[i], 4.2);
      if (jn.type === "priority" && !A.major) VMAX[i] = Math.min(VMAX[i], 3.0);
      if (jn.type === "rail") VMAX[i] = Math.min(VMAX[i], 5.5);
    }
    if (jn.type === "roundabout" && s > sStop - 6 && s < sIn + 2) VMAX[i] = Math.min(VMAX[i], 5.0);
    if (kind === "uturn" && s > sIn && s < sOut) VMAX[i] = Math.min(VMAX[i], 3.5);
    if (jn.type === "rail" && s > sStop && s < sOut + 2) VMAX[i] = Math.min(VMAX[i], 5.5);
  }
  return { junc: jn.id, signal: jn.type === "signal" || jn.type === "rail", rail: jn.type === "rail", round: jn.type === "roundabout", ntype: jn.type, from, to, kind, priority,
    len: (n - 1) * ds, ds, n, X, Y, Z, TX, TY, K, GR, VMAX, SPD, DMIN, DMAX, DMINX, BIKE, CTR, FLOOD, MKT, sIn, sOut, sStop, exitTo: B.to, busStops,
    kindIn: A.kind, kindOut: B.kind, shared: A.shared || B.shared, noCar: A.noCar || B.noCar, hasBike: A.bl > 0 || B.bl > 0 };
}
function bez(pts, seg, p0, c1, c2, p3, z, tag) {
  const L = Math.hypot(p3[0] - p0[0], p3[1] - p0[1]), N = Math.max(8, Math.ceil(L / 0.3));
  for (let i = 1; i <= N; i++) { const t = i / N, u = 1 - t; pts.push([u * u * u * p0[0] + 3 * u * u * t * c1[0] + 3 * u * t * t * c2[0] + t * t * t * p3[0], u * u * u * p0[1] + 3 * u * u * t * c1[1] + 3 * u * t * t * c2[1] + t * t * t * p3[1], (p0[2] ?? z) + ((p3[2] ?? z) - (p0[2] ?? z)) * t]); seg.push(tag); }
}

/* ---------------- districts */
/* a town grid with jittered, slightly skewed blocks. Options: diag (an avenue across, giving 5- and 6-arm junctions),
   roundabout [i, j], market [[i, j], [i, j]], alleyBlock [i, j] (an alley network inside that block), stubs on the sides */
function townGrid(W, o) {
  const r = W.r, nx = o.nx, ny = o.ny, N = [];
  const colCls = [], rowCls = [];
  const pick = () => { const q = r(); return q < 0.35 ? "main" : q < 0.75 ? "sec" : "small"; };
  for (let i = 0; i < nx; i++) colCls.push(o.colCls ? o.colCls[i] : pick());
  for (let j = 0; j < ny; j++) rowCls.push(o.rowCls ? o.rowCls[j] : pick());
  if (!colCls.includes("main") && !rowCls.includes("main")) rowCls[Math.floor(r() * ny)] = "main";
  const skew = (r() - 0.5) * (o.skew ?? 0.18);
  for (let i = 0; i < nx; i++) { N.push([]); for (let j = 0; j < ny; j++) {
    const x = o.x0 + i * o.sx + j * o.sy * skew + (r() - 0.5) * 2 * (o.jit ?? 10), y = o.y0 + j * o.sy + (r() - 0.5) * 2 * (o.jit ?? 10);
    const big = (rowCls[j] !== "small") + (colCls[i] !== "small");
    const type = big === 2 ? (r() < 0.88 ? "signal" : "yield") : big === 1 ? (r() < 0.45 ? "signal" : "yield") : "yield";
    N[i].push(W.node(x, y, type, { kind: "urban" }));
  } }
  if (o.roundabout) { const n = N[o.roundabout[0]][o.roundabout[1]]; n.type = "roundabout"; n.R = 10 + r() * 6; n.rw = 8; }
  if (o.diag) for (const [i, j] of o.diag) if (N[i][j].type !== "roundabout") N[i][j].type = "signal";
  const mk = (ma, mb) => o.market && o.market.some((m) => (m[0] === ma[0] && m[1] === ma[1] && o.market.some((q) => q[0] === mb[0] && q[1] === mb[1])));
  const ab = o.alleyBlock, mouths = {};
  const link = (a, b, cls, extra) => {
    const via = []; if (r() < 0.45) { const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2, L = Math.hypot(b.x - a.x, b.y - a.y), nn = rightNormal([(b.x - a.x) / L, (b.y - a.y) / L]), k = (r() - 0.5) * 10; via.push([mx + nn[0] * k, my + nn[1] * k]); }
    return W.road(a, b, cls, Object.assign({ via, encroached: r() < (o.encroach ?? 0.35) }, extra || {}));
  };
  // a block edge may get an alley mouth in it: the edge becomes two links meeting at a priority junction
  const edge = (a, b, cls, key, extra) => {
    if (mouths[key]) { const f = mouths[key], mx = a.x + (b.x - a.x) * f, my = a.y + (b.y - a.y) * f; const m = W.node(mx, my, "priority", { kind: "alley" }); link(a, m, cls, extra); link(m, b, cls, extra); return m; }
    link(a, b, cls, extra); return null;
  };
  if (ab) { mouths[`h${ab[0]},${ab[1]}`] = 0.38 + r() * 0.2; mouths[`h${ab[0]},${ab[1] + 1}`] = 0.42 + r() * 0.2; mouths[`v${ab[0]},${ab[1]}`] = 0.4 + r() * 0.2; }
  const M = {};
  for (let j = 0; j < ny; j++) for (let i = 0; i + 1 < nx; i++) { const m = mk([i, j], [i + 1, j]); M[`h${i},${j}`] = edge(N[i][j], N[i + 1][j], m ? "sec" : rowCls[j], `h${i},${j}`, m ? { market: true, zones: [] } : null); }
  for (let i = 0; i < nx; i++) for (let j = 0; j + 1 < ny; j++) { const m = mk([i, j], [i, j + 1]); M[`v${i},${j}`] = edge(N[i][j], N[i][j + 1], m ? "sec" : colCls[i], `v${i},${j}`, m ? { market: true } : null); }
  if (o.diag) for (let k = 0; k + 1 < o.diag.length; k++) { const [i0, j0] = o.diag[k], [i1, j1] = o.diag[k + 1]; W.road(N[i0][j0], N[i1][j1], "avenue", { encroached: false }); }
  if (ab) {   // the alley network: three mouths, one bend inside, one narrow lane
    const tl = N[ab[0]][ab[1]], br = N[ab[0] + 1][ab[1] + 1], m1 = M[`h${ab[0]},${ab[1]}`], m2 = M[`h${ab[0]},${ab[1] + 1}`], m3 = M[`v${ab[0]},${ab[1]}`];
    const cx = (tl.x + br.x) / 2 + (r() - 0.5) * 12, cy = (tl.y + br.y) / 2 + (r() - 0.5) * 10;
    const a1 = W.node(cx, cy, "yield", { kind: "alley" });
    if (m1) W.road(m1, a1, "alley", { via: [[m1.x, (m1.y + cy) / 2 + 4], [m1.x + (cx - m1.x) * 0.15, cy - 6]] });
    if (m2) W.road(a1, m2, "alley", { via: [[(cx + m2.x) / 2 + 5, cy + 3], [m2.x, (cy + m2.y) / 2 + 3]] });
    if (m3) W.road(a1, m3, "lane", { via: [[(cx + m3.x) / 2, cy - 1], [m3.x + 8, m3.y + 1]] });
    W.urbanFill.push({ x0: tl.x + 12, y0: tl.y + 12, x1: br.x - 12, y1: br.y - 12, alley: true });
  }
  // roads leaving the map from the outer junctions
  const st = o.stubs || {};
  const stub = (n, dir, cls, len) => W.road(n, [n.x + dir[0] * len, n.y + dir[1] * len], cls, { encroached: r() < 0.3 });
  for (let j = 0; j < ny; j++) { if (st.W && st.W(j) !== false) stub(N[0][j], [-1, 0], rowCls[j], o.stubLen || 80); if (st.E && st.E(j) !== false) stub(N[nx - 1][j], [1, 0], rowCls[j], o.stubLen || 80); }
  for (let i = 0; i < nx; i++) { if (st.N && st.N(i) !== false) stub(N[i][0], [0, -1], colCls[i], o.stubLen || 80); if (st.S && st.S(i) !== false) stub(N[i][ny - 1], [0, 1], colCls[i], o.stubLen || 80); }
  const xs = N.flat().map((n) => n.x), ys = N.flat().map((n) => n.y);
  W.area("urban", Math.min(...xs) - 90, Math.min(...ys) - 90, Math.max(...xs) + 90, Math.max(...ys) + 90);
  W.urbanFill.push({ x0: Math.min(...xs) - 70, y0: Math.min(...ys) - 70, x1: Math.max(...xs) + 70, y1: Math.max(...ys) + 70 });
  return { N, colCls, rowCls };
}
/* a dual carriageway along y with junctions at the given x, median U-turn openings between them, bus stops */
function highway(W, o) {
  const r = W.r, y = o.y, xs = o.joins.slice().sort((a, b) => a - b), J = {};
  const pts = []; let prev = W.node(o.x0, y + (r() - 0.5) * 6, "signal", { kind: "highway" });
  const first = prev; pts.push(prev);
  const stops = [];
  const nodesAt = [];
  for (let k = 0; k < xs.length; k++) {
    const x = xs[k];
    if (k > 0 || x - o.x0 > 160) { const u = W.node((prev.x + x) / 2 + (r() - 0.5) * 20, y + (r() - 0.5) * 4, "uturn", { kind: "highway" }); W.road(prev, u, "highway", { zones: busZone(prev, u) }); prev = u; }
    const n = W.node(x, y + (r() - 0.5) * 4, o.type || "signal", { kind: "highway" }); W.road(prev, n, "highway", { zones: busZone(prev, n) }); J[x] = n; prev = n; nodesAt.push(n);
  }
  if (o.x1 - prev.x > 160) { const u = W.node((prev.x + o.x1) / 2, y + (r() - 0.5) * 4, "uturn", { kind: "highway" }); W.road(prev, u, "highway", { zones: busZone(prev, u) }); prev = u; }
  W.road(prev, [o.x1 + 90, prev.y], "highway", { flow: 0.8 });
  W.road(first, [o.x0 - 90, first.y], "highway", { flow: 0.8 });
  first.type = "bend";   // the west end only joins the road leaving the map
  W.area("industrial", o.x0 - 90, y - 70, o.x1 + 90, y + 60);
  function busZone(a, b) { const L = Math.hypot(b.x - a.x, b.y - a.y); if (L < 120 || r() < 0.35) return []; const s = L * (0.3 + r() * 0.4); return [{ type: "bus", s0: s, s1: s + 22, dir: r() < 0.5 ? 1 : -1 }]; }
  return { J, first, last: prev };
}
/* a winding mountain road over a hill, from node a to node b, with hairpins */
function passRoad(W, a, b, o) {
  const r = W.r, cx = o.cx, cy = o.cy;
  W.hill(cx, cy, o.h || 46, o.R || 120);
  W.hill(cx + 60, cy - 70, (o.h || 46) * 0.45, 70);
  const via = o.via.map(([x, y]) => [x + (r() - 0.5) * 8, y + (r() - 0.5) * 8]);
  W.area("forest", cx - (o.R || 120) * 1.9, cy - (o.R || 120) * 1.9, cx + (o.R || 120) * 1.9, cy + (o.R || 120) * 1.9);
  return W.road(a, b, "pass", { via, walls: "hill" });
}

/* ---------------- presets */
function regionMap(W) {
  const r = W.r;
  // downtown: 3 x 3 junctions, an avenue across (5- and 6-arm junctions), a roundabout, a market street, an alley block
  const core = townGrid(W, { x0: -430, y0: -250, nx: 3, ny: 3, sx: 150, sy: 130, jit: 12, skew: 0.12, diag: [[0, 0], [1, 1], [2, 2]], roundabout: [2, 2], market: [[1, 1], [1, 2]], alleyBlock: [0, 1],
    colCls: ["sec", "main", "sec"], rowCls: ["main", "sec", "main"], stubs: { W: () => true, E: (j) => j === 2 }, stubLen: 150, encroach: 0.4 });
  const N = core.N;
  // highway along the north edge, joined by the middle avenue and the north-east corner; the mountain road comes down to its east end
  const hw = highway(W, { y: -380, x0: -650, x1: 560, joins: [N[1][0].x, N[2][0].x + 40, 420] });
  W.road(N[1][0], hw.J[N[1][0].x], "main", { encroached: false });
  W.road(N[2][0], hw.J[N[2][0].x + 40], "sec", { via: [[N[2][0].x + 25, (N[2][0].y + hw.J[N[2][0].x + 40].y) / 2]] });
  W.road(hw.J[N[1][0].x], [hw.J[N[1][0].x].x + 6, -470], "main", { flow: 0.35 });
  const j3 = hw.J[420]; j3.type = "priority";
  // mountain road from the east of downtown up over the hill to the highway
  const a = N[2][1];
  passRoad(W, a, j3, { cx: 210, cy: -170, h: 48, R: 118, via: [[-40, -118], [40, -96], [150, -70], [208, -88], [168, -120], [70, -140], [40, -170], [86, -206], [210, -226], [300, -214], [352, -262], [400, -318]] });
  // rail line between town and countryside; level crossings on the two roads going south
  const railY = 120;
  W.rail([[-700, railY + 12], [-300, railY], [100, railY - 4], [400, railY + 10], [700, railY + 34]], { period: 150 + r() * 60 });
  const r1 = W.node(N[0][2].x + 2, railY + 1, "rail", { kind: "rail" }), r2 = W.node(N[2][2].x + 4, railY - 2, "rail", { kind: "rail" });
  r1.x = N[0][2].x + 2; r2.x = N[2][2].x + 4; r1.y = railAt(W.rails[0], r1.x); r2.y = railAt(W.rails[0], r2.x);
  W.road(N[0][2], r1, "sec", { encroached: false }); W.road(N[2][2], r2, "main", { encroached: false });
  // countryside: a rural road through rice fields with three village junctions, a flooded dip, a loop to the south
  const ry = 275;
  W.hill(-190, ry, -3.2, 45);
  const v1 = W.node(N[0][2].x, ry + (r() - 0.5) * 8, "yield", { kind: "rural" }), v3 = W.node(-260, ry + 6, "yield", { kind: "rural" }), v2 = W.node(N[2][2].x + 6, ry - 4, "yield", { kind: "rural" }), v5 = W.node(170, ry + 10, "yield", { kind: "rural" });
  W.road(r1, v1, "rural"); W.road(r2, v2, "sec", { encroached: false });
  W.road(v1, [-700, ry + 10], "rural", { flow: 0.1 }); W.road(v1, v3, "rural", { via: [[(v1.x + v3.x) / 2, ry - 8]] });
  W.road(v3, v2, "rural", { via: [[-190, ry + 4]], zones: [{ type: "flood", s0: 0, s1: 0, depth: 0.32, auto: true }] });
  W.road(v2, v5, "rural", { via: [[50, ry + 16]] }); W.road(v5, [660, ry + 30], "rural", { flow: 0.1 });
  const v4 = W.node(-250, 360, "yield", { kind: "rural" }); W.road(v3, v4, "rural", { via: [[-268, 320]] }); W.road(v4, [-246, 470], "rural", { flow: 0.08 });
  W.road(v4, v5, "rural", { via: [[-60, 375], [100, 352]] });
  W.road(v5, [176, 470], "rural", { flow: 0.08 });
  for (const v of [v1, v3, v2, v5, v4]) W.villages.push({ x: v.x, y: v.y, n: 10, r: 40 });
  W.area("rural", -720, railY + 25, 700, 520);
  return "Mixed region";
}
function railAt(R, x) { const p = R.pts; for (let i = 1; i < p.length; i++) if ((p[i - 1][0] - x) * (p[i][0] - x) <= 0) { const t = (x - p[i - 1][0]) / ((p[i][0] - p[i - 1][0]) || 1); return p[i - 1][1] + (p[i][1] - p[i - 1][1]) * t; } return p[0][1]; }
/* flood zones marked "auto" cover the lowest stretch of their road */
function placeAutoZones(W) {
  for (const L of W.links) for (const zn of L.zones) if (zn.auto) {
    let lo = 0, zmin = 1e9; for (let i = 0; i < L.pts.length; i++) { const z = W.H(L.pts[i][0], L.pts[i][1]); if (z < zmin) { zmin = z; lo = i; } }
    const half = 22 + W.r() * 14; zn.s0 = Math.max(4, lo - half); zn.s1 = Math.min(L.pts.length - 4, lo + half); delete zn.auto;
  }
}

const PRESETS = {
  region: (W) => regionMap(W),
  city: (W) => {
    const r = W.r, nx = 2 + Math.floor(r() * 2), ny = 2 + Math.floor(r() * 2);
    const opts = { x0: -(nx - 1) * 70, y0: -(ny - 1) * 62, nx, ny, sx: 125 + r() * 30, sy: 110 + r() * 30, jit: 12, skew: 0.2, stubs: { W: () => true, E: () => true, N: () => r() < 0.7, S: () => r() < 0.7 } };
    if (r() < 0.45 && nx === 3 && ny === 3) opts.diag = [[0, 0], [1, 1], [2, 2]];
    else if (r() < 0.45) opts.roundabout = [Math.floor(r() * nx), Math.floor(r() * ny)];
    if (r() < 0.5) { const i = Math.floor(r() * nx); opts.market = [[i, 0], [i, 1]]; }
    if (r() < 0.4) opts.alleyBlock = [0, 0];
    townGrid(W, opts); return "Town";
  },
  alley: (W) => { townGrid(W, { x0: -70, y0: -60, nx: 2, ny: 2, sx: 150, sy: 125, jit: 8, alleyBlock: [0, 0], rowCls: ["sec", "small"], colCls: ["small", "sec"], stubs: { W: () => true, E: () => true, N: () => true, S: () => true } }); return "Alleys"; },
  roundabout: (W) => {
    const r = W.r, c = W.node(0, 0, "roundabout", { R: 11 + r() * 6, rw: 8 }), k = 4 + Math.floor(r() * 3), a0 = r() * 6.28;
    for (let i = 0; i < k; i++) {
      const a = a0 + i * 2 * Math.PI / k + (r() - 0.5) * 0.35, cls = i % 2 ? "sec" : "main", far = [Math.cos(a) * 150, Math.sin(a) * 150];
      if (r() < 0.4) { const m = W.node(far[0] * 0.9, far[1] * 0.9, r() < 0.5 ? "signal" : "yield"); W.road(c, m, cls); W.road(m, [far[0] * 1.6, far[1] * 1.6], cls, { flow: ROADS[cls].flow }); W.road(m, [far[0] * 0.9 - Math.sin(a) * 90, far[1] * 0.9 + Math.cos(a) * 90], "small", { flow: 0.1 }); }
      else W.road(c, [far[0] * 1.3, far[1] * 1.3], cls, { flow: ROADS[cls].flow });
    }
    W.area("urban", -260, -260, 260, 260); W.urbanFill.push({ x0: -230, y0: -230, x1: 230, y1: 230 });
    return "Roundabout";
  },
  highway: (W) => {
    const hw = highway(W, { y: 0, x0: -360, x1: 380, joins: [-60, 220] });
    const a = hw.J[-60], b = hw.J[220];
    W.road(a, [a.x - 4, 160], "main", { flow: 0.4 }); W.road(a, [a.x + 6, -150], "sec", { flow: 0.25 });
    W.road(b, [b.x + 10, 150], "sec", { flow: 0.25 });
    return "Highway";
  },
  rail: (W) => {
    const r = W.r;
    W.rail([[-500, 10], [-100, 0], [300, -12], [600, -30]], { period: 90 + r() * 40 });
    const n1 = W.node(-10, -120, "signal"), n2 = W.node(6, 120, "yield", { kind: "rural" }), x = railAt(W.rails[0], 0) ;
    const rc = W.node(0, x, "rail", { kind: "rail" });
    W.road(n1, rc, "main"); W.road(rc, n2, "main");
    W.road(n1, [-200, -126], "main", { flow: 0.26 }); W.road(n1, [200, -110], "sec", { flow: 0.15 }); W.road(n1, [-8, -260], "sec", { flow: 0.15 });
    W.road(n2, [-180, 130], "rural", { flow: 0.08 }); W.road(n2, [190, 112], "rural", { flow: 0.08 }); W.road(n2, [10, 260], "main", { flow: 0.2 });
    W.area("urban", -260, -300, 260, -20); W.area("rural", -560, 20, 560, 320); W.urbanFill.push({ x0: -230, y0: -280, x1: 230, y1: -40 });
    W.villages.push({ x: 6, y: 120, n: 12, r: 50 });
    return "Level crossing";
  },
  rural: (W) => {
    const r = W.r;
    W.hill(-30, 6, -3.2, 40);
    const a = W.node(-200, 0, "yield", { kind: "rural" }), b = W.node(160, 10, "yield", { kind: "rural" }), c = W.node(-40, 150, "yield", { kind: "rural" });
    W.road(a, b, "rural", { via: [[-30, 6]], zones: [{ type: "flood", s0: 0, s1: 0, depth: 0.25 + r() * 0.12, auto: true }] });
    W.road(a, c, "rural", { via: [[-140, 90]] }); W.road(b, c, "rural", { via: [[80, 100]] });
    W.road(a, [-420, -10], "rural", { flow: 0.12 }); W.road(b, [400, 20], "rural", { flow: 0.12 }); W.road(c, [-30, 320], "rural", { flow: 0.1 }); W.road(a, [-210, -200], "rural", { flow: 0.08 });
    for (const v of [a, b, c]) W.villages.push({ x: v.x, y: v.y, n: 12, r: 45 });
    W.area("rural", -460, -260, 440, 360);
    return "Countryside";
  },
  pass: (W) => {
    const a = W.node(-260, 160, "priority", { kind: "pass" }), b = W.node(240, -190, "priority", { kind: "pass" });
    W.road(a, [-460, 170], "rural", { flow: 0.12 }); W.road(a, [-250, 330], "rural", { flow: 0.1 });
    W.road(b, [440, -200], "rural", { flow: 0.12 }); W.road(b, [250, -380], "rural", { flow: 0.1 });
    passRoad(W, a, b, { cx: 0, cy: 0, h: 48, R: 125, via: [[-150, 140], [-40, 120], [70, 112], [110, 86], [60, 60], [-80, 46], [-120, 20], [-70, -8], [60, -14], [110, -50], [60, -80], [-30, -96], [-10, -136], [120, -160], [190, -182]] });
    return "Mountain pass";
  },
};
const PRESET_NAMES = Object.keys(PRESETS);

/* one map. preset: one of PRESET_NAMES (default "region") */
function cityMap(seed = 1, opt = {}) {
  const preset = opt.preset || "region";
  if (preset === "hcm") {   // a window of the real Ho Chi Minh City (web/osm.js), around opt.centre or a random junction
    const city = typeof getHcmCity === "function" ? getHcmCity() : null;
    if (!city) throw new Error("Ho Chi Minh City data not found (web/hcm/, built by data/build_hcm.js)");
    const c = opt.centre || city.randomCentre(rngFrom(seed * 104729 + 3));
    return hcmMapCached(city, c[0], c[1], { seed, half: opt.half, name: opt.name });
  }
  const W = new World(seed, { preset });
  const name = PRESETS[preset](W);
  placeAutoZones(W);
  const map = W.finish(opt.name || name);
  return map;
}

if (typeof module !== "undefined") module.exports = { World, finishWorld, cityMap, buildPath, rngFrom, clampN, rightNormal, armAt, offsetPoly, smoothPath, resamplePoly, polyLen, cumLen, lineOfSight, groundAt, ROADS, PRESETS, PRESET_NAMES, COMPASS, COMPASS_NAME, OCC, turnKind, laneOffset, wrapA };
