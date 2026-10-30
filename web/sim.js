/* ======================================================================
   LOTUS: 2D traffic core (no rendering; runs in the browser or Node).
   Units: metres, seconds. x east, y south (screen-like). Right-hand traffic.

   The world is a procedurally generated city: a grid of junctions joined by
   roads of three classes. Junctions are signalised (two-phase, countdown) or
   unsignalised; edge junctions may lose an arm and become T-junctions.

   A trip is a chain of legs. Each leg is a reference path through one
   junction, from the middle of the incoming road to the middle of the
   outgoing road; consecutive legs meet exactly, so a vehicle just switches
   path. Within a leg, vehicles move in Frenet coordinates (s along the path,
   d sideways). Motorbikes are not tied to lanes: they keep looking for the
   sideways offset with the most free road ahead.
   ====================================================================== */
"use strict";
const DT = 0.1;
const clampN = (v, a, b) => v < a ? a : v > b ? b : v;
// seeded RNG whose state can be copied (needed to clone the world for look-ahead planning)
function rngFrom(seed) {
  const st = { a: seed >>> 0 };
  const f = () => { let a = st.a | 0; a = a + 0x6D2B79F5 | 0; st.a = a; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
  f.st = st; return f;
}

const DIRS = { E: [1, 0], W: [-1, 0], N: [0, -1], S: [0, 1] };
const OPP = { E: "W", W: "E", N: "S", S: "N" };
const LEFT_OF = { E: "S", S: "W", W: "N", N: "E" };     // exit arm for a left turn when entering from key
const RIGHT_OF = { E: "N", N: "W", W: "S", S: "E" };
const rightNormal = (t) => [-t[1], t[0]];             // y points south, so this is the driver's right
const ROAD = {
  main: { cls: "main", hw: 7.5, lanes: 2, flow: 0.5 },
  sec: { cls: "sec", hw: 5.5, lanes: 1, flow: 0.3 },
  small: { cls: "small", hw: 4.0, lanes: 1, flow: 0.14 },
};

/* ---------------- procedural city */
function cityMap(seed = 1, opt = {}) {
  const r = rngFrom(seed * 7919 + 17);
  const nx = opt.nx ?? 3, ny = opt.ny ?? 3, EDGE = opt.edge ?? 70;
  const m = { name: opt.name || "Khu phố mẫu", seed, nx, ny, sidewalk: 4.5, cornerR: 7, zebra: [1.0, 4.0], stop: 4.6,
    turn: { straight: 0.58, right: 0.24, left: 0.18 }, juncs: [], links: [], entries: [] };
  const pickClasses = (n) => {
    const c = []; for (let i = 0; i < n; i++) { const q = r(); c.push(q < 0.35 ? ROAD.main : q < 0.75 ? ROAD.sec : ROAD.small); }
    if (!c.some((x) => x === ROAD.main) && !opt.noMain) c[Math.floor(r() * n)] = ROAD.main;
    return c;
  };
  const colCls = pickClasses(nx), rowCls = pickClasses(ny);
  const xs = [0], ys = [0];
  for (let i = 1; i < nx; i++) xs.push(xs[i - 1] + 78 + r() * 42);
  for (let j = 1; j < ny; j++) ys.push(ys[j - 1] + 74 + r() * 40);
  const cx = xs[nx - 1] / 2, cy = ys[ny - 1] / 2;
  const J = (i, j) => m.juncs[j * nx + i];
  for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    m.juncs.push({ id: j * nx + i, i, j, x: Math.round(xs[i] - cx), y: Math.round(ys[j] - cy), row: rowCls[j], col: colCls[i], arms: {} });
  }
  const addArm = (jn, a, road, len, to) => { jn.arms[a] = { id: a, dir: DIRS[a], hw: road.hw, lanes: road.lanes, cls: road.cls, len, to, flow: to ? 0 : road.flow }; };
  for (let j = 0; j < ny; j++) for (let i = 0; i + 1 < nx; i++) {   // east-west links
    const a = J(i, j), b = J(i + 1, j), L = b.x - a.x;
    addArm(a, "E", rowCls[j], L / 2, { j: b.id, arm: "W" }); addArm(b, "W", rowCls[j], L / 2, { j: a.id, arm: "E" });
    m.links.push({ a: a.id, b: b.id, axis: "x", road: rowCls[j] });
  }
  for (let i = 0; i < nx; i++) for (let j = 0; j + 1 < ny; j++) {   // north-south links
    const a = J(i, j), b = J(i, j + 1), L = b.y - a.y;
    addArm(a, "S", colCls[i], L / 2, { j: b.id, arm: "N" }); addArm(b, "N", colCls[i], L / 2, { j: a.id, arm: "S" });
    m.links.push({ a: a.id, b: b.id, axis: "y", road: colCls[i] });
  }
  // roads leaving the city: main roads always continue, smaller ones sometimes end in a T
  const edgeArm = (jn, a, road) => { if (road === ROAD.main || r() < (opt.edgeP ?? 0.72)) addArm(jn, a, road, EDGE, null); };
  for (let j = 0; j < ny; j++) { edgeArm(J(0, j), "W", rowCls[j]); edgeArm(J(nx - 1, j), "E", rowCls[j]); }
  for (let i = 0; i < nx; i++) { edgeArm(J(i, 0), "N", colCls[i]); edgeArm(J(i, ny - 1), "S", colCls[i]); }
  for (const jn of m.juncs) {
    // every junction needs both axes and at least three arms
    for (const [ax, alt, road] of [["E", "W", jn.row], ["N", "S", jn.col]]) if (!jn.arms[ax] && !jn.arms[alt]) addArm(jn, r() < 0.5 ? ax : alt, road, EDGE, null);
    if (Object.keys(jn.arms).length < 3) { const miss = ["E", "W", "N", "S"].find((a) => !jn.arms[a]); addArm(jn, miss, miss === "E" || miss === "W" ? jn.row : jn.col, EDGE, null); }
    jn.hwX = jn.row.hw; jn.hwY = jn.col.hw;
    for (const a of Object.values(jn.arms)) a.box = (a.id === "E" || a.id === "W") ? jn.hwY : jn.hwX;
    // signalised unless both roads are small (and some secondary crossings stay unsignalised)
    const big = (jn.row !== ROAD.small) + (jn.col !== ROAD.small);
    jn.type = opt.allSignal ? "signal" : big === 2 ? (r() < 0.9 ? "signal" : "yield") : big === 1 ? (r() < 0.45 ? "signal" : "yield") : "yield";
    if (jn.type === "signal") {
      const gx = jn.row === ROAD.main ? 28 + r() * 8 : 20 + r() * 6, gy = jn.col === ROAD.main ? 28 + r() * 8 : 18 + r() * 6;
      jn.phases = [{ green: ["E", "W"], t: Math.round(gx) }, { yellow: ["E", "W"], t: 3 }, { allRed: true, t: 2 },
        { green: ["N", "S"], t: Math.round(gy) }, { yellow: ["N", "S"], t: 3 }, { allRed: true, t: 2 }];
      jn.cycle = jn.phases.reduce((s, p) => s + p.t, 0); jn.offset = Math.floor(r() * jn.cycle);
    }
    for (const a of Object.values(jn.arms)) if (!a.to) m.entries.push({ j: jn.id, arm: a.id, flow: a.flow });
  }
  const X = m.juncs.map((q) => q.x), Y = m.juncs.map((q) => q.y), pad = EDGE + 26;
  m.bounds = { x0: Math.min(...X) - pad, x1: Math.max(...X) + pad, y0: Math.min(...Y) - pad, y1: Math.max(...Y) + pad };
  return m;
}

/* one leg: from the middle of the incoming road, through junction jn, to the middle of the outgoing road */
function buildPath(map, jn, from, to) {
  const A = jn.arms[from], B = jn.arms[to], O = [jn.x, jn.y];
  const tA = [-A.dir[0], -A.dir[1]], nA = rightNormal(tA), nB = rightNormal(B.dir);
  const cA = A.hw / 2, cB = B.hw / 2;
  const P = (dir, dist, n, c) => [O[0] + dir[0] * dist + n[0] * c, O[1] + dir[1] * dist + n[1] * c];
  const far = P(A.dir, A.len, nA, cA), pin = P(A.dir, A.box + map.zebra[0], nA, cA);
  const pout = P(B.dir, B.box + map.zebra[0], nB, cB), farOut = P(B.dir, B.len, nB, cB);
  const pts = [];
  const seg = (a, b) => { const n = Math.max(2, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1]) / 0.5)); for (let i = 0; i < n; i++) pts.push([a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n]); };
  seg(far, pin);
  const straight = to === OPP[from];
  const k = Math.hypot(pout[0] - pin[0], pout[1] - pin[1]) * (straight ? 0.33 : 0.5);
  const c1 = [pin[0] + tA[0] * k, pin[1] + tA[1] * k], c2 = [pout[0] - B.dir[0] * k, pout[1] - B.dir[1] * k];
  const nb = 40;
  for (let i = 0; i < nb; i++) { const t = i / nb, u = 1 - t; pts.push([u * u * u * pin[0] + 3 * u * u * t * c1[0] + 3 * u * t * t * c2[0] + t * t * t * pout[0], u * u * u * pin[1] + 3 * u * u * t * c1[1] + 3 * u * t * t * c2[1] + t * t * t * pout[1]]); }
  seg(pout, farOut); pts.push(farOut);
  const raw = [0]; for (let i = 1; i < pts.length; i++) raw.push(raw[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1]));
  const len = raw[raw.length - 1], ds = 0.5, n = Math.floor(len / ds) + 1;
  const X = new Float32Array(n), Y = new Float32Array(n), TX = new Float32Array(n), TY = new Float32Array(n), K = new Float32Array(n), DMIN = new Float32Array(n), DMAX = new Float32Array(n), VMAX = new Float32Array(n);
  let j = 0;
  for (let i = 0; i < n; i++) {
    const s = i * ds; while (j < raw.length - 2 && raw[j + 1] < s) j++;
    const f = (s - raw[j]) / Math.max(1e-6, raw[j + 1] - raw[j]);
    X[i] = pts[j][0] + (pts[j + 1][0] - pts[j][0]) * f; Y[i] = pts[j][1] + (pts[j + 1][1] - pts[j][1]) * f;
  }
  for (let i = 0; i < n; i++) { const a = Math.max(0, i - 1), b = Math.min(n - 1, i + 1); const dx = X[b] - X[a], dy = Y[b] - Y[a], l = Math.hypot(dx, dy) || 1; TX[i] = dx / l; TY[i] = dy / l; }
  for (let i = 0; i < n; i++) {
    const a = Math.max(0, i - 2), b = Math.min(n - 1, i + 2);
    let dth = Math.atan2(TY[b], TX[b]) - Math.atan2(TY[a], TX[a]); dth = Math.atan2(Math.sin(dth), Math.cos(dth));
    K[i] = Math.abs(dth) / ((b - a) * ds);
    VMAX[i] = Math.min(16, Math.sqrt(2.6 / Math.max(K[i], 1e-3)));
  }
  let sInV = 0, sOutV = len;
  { let best = 1e9; for (let i = 0; i < n; i++) { const d = Math.hypot(X[i] - pin[0], Y[i] - pin[1]); if (d < best) { best = d; sInV = i * ds; } } }
  { let best = 1e9; for (let i = 0; i < n; i++) { const d = Math.hypot(X[i] - pout[0], Y[i] - pout[1]); if (d < best) { best = d; sOutV = i * ds; } } }
  const sStop = sInV - (map.stop - map.zebra[0]);
  // unsignalised junction: everybody slows down to look before entering
  if (jn.type !== "signal") for (let i = 0; i < n; i++) { const s = i * ds; if (s > sStop - 1 && s < sInV + 4) VMAX[i] = Math.min(VMAX[i], 4.2); }
  const inA = [-cA + 0.35, A.hw - cA - 0.35], outB = [-cB + 0.35, B.hw - cB - 0.35];
  for (let i = 0; i < n; i++) {
    const s = i * ds;
    if (s <= sInV) { DMIN[i] = inA[0]; DMAX[i] = inA[1]; }
    else if (s >= sOutV) { DMIN[i] = outB[0]; DMAX[i] = outB[1]; }
    else { const f = (s - sInV) / (sOutV - sInV); DMIN[i] = inA[0] + (outB[0] - inA[0]) * f - 0.8; DMAX[i] = inA[1] + (outB[1] - inA[1]) * f + 0.8; }
  }
  const kind = straight ? "straight" : to === LEFT_OF[from] ? "left" : "right";
  return { junc: jn.id, signal: jn.type === "signal", from, to, kind, len: (n - 1) * ds, ds, n, X, Y, TX, TY, K, VMAX, DMIN, DMAX, sIn: sInV, sOut: sOutV, sStop, exitTo: B.to };
}

/* ---------------- vehicles */
const TYPES = {
  moto: { l: 1.9, w: 0.72, vDes: 9.5, acc: 2.6, brake: 4.5, latV: 1.2, step: 0.35 },
  bike: { l: 1.7, w: 0.6, vDes: 4.5, acc: 1.0, brake: 3.0, latV: 0.6, step: 0.35 },
  car: { l: 4.5, w: 1.8, vDes: 10.5, acc: 2.0, brake: 5.0, latV: 0.55, step: 1.0 },
  taxi: { l: 4.4, w: 1.75, vDes: 10.5, acc: 2.1, brake: 5.0, latV: 0.6, step: 1.0 },
  bus: { l: 10.5, w: 2.5, vDes: 8.0, acc: 1.1, brake: 3.5, latV: 0.35, step: 1.2 },
  truck: { l: 7.0, w: 2.3, vDes: 8.5, acc: 1.2, brake: 3.8, latV: 0.4, step: 1.2 },
};
const MIX = [["moto", 0.835], ["car", 0.075], ["taxi", 0.035], ["bike", 0.02], ["truck", 0.02], ["bus", 0.015]];

class Vehicle {
  constructor(sim, type, path, d) {
    const r = sim.rand, T = TYPES[type];
    this.id = sim.nextId++; this.type = type; this.T = T; this.path = path; this.legs = 1;
    this.l2 = T.l / 2; this.w2 = T.w / 2; this.s = 0; this.d = d; this.dTarget = d; this.v = T.vDes * 0.75;
    // personality: how bold, how patient, how law-abiding, how fast they like to go
    const G = sim.genome, two = type === "moto" || type === "bike";
    const nrm = (mu, sd) => { const u = Math.max(1e-6, r()), w = r(); return clampN(mu + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(6.2832 * w), 0, 1); };
    if (G) { this.aggr = nrm(G.aggrMu, G.aggrSd); this.patience = nrm(G.patienceMu, 0.25); this.comply = nrm(G.complyMu, 0.25); this.speedPref = 0.8 + 0.45 * nrm(G.speedMu, 0.25); }
    else { this.aggr = r(); this.patience = r(); this.comply = r(); this.speedPref = 0.8 + r() * 0.45; }
    if (type === "bus" || type === "truck") { this.aggr *= 0.6; this.comply = 0.4 + 0.6 * this.comply; }
    // Phase 3 behaviours (only when a scenario genome asks for them)
    this.wrongWay = !!(G && two && path.kind !== "left" && r() < G.pWrong);       // rides against traffic along the far curb up to the junction
    this.cutIn = !!(G && type === "moto" && r() < G.pCutIn);                      // cuts in front of a car, then brakes
    this.earlyLeft = !!(G && two && path.kind === "left" && r() < G.pEarlyLeft);   // cuts the corner on a left turn
    this.brakeRate = G && two ? G.pBrake * (0.5 + this.aggr) : 0;                 // sudden stops per second while moving
    this.vDes = T.vDes * this.speedPref;
    this.margin = 0.5 - 0.32 * this.aggr; this.minGap = (type === "moto" || type === "bike" ? 0.9 : 1.6) - 0.6 * this.aggr; this.headway = 0.85 - 0.45 * this.aggr;
    this.prio = r(); this.wait = 0; this.honk = 0; this.braking = false; this.redRun = null; this.phaseSeen = -1; this.committed = false;
    this.signal = path.kind !== "straight" && r() < 0.35 + 0.6 * this.comply;
    this.x = 0; this.y = 0; this.h = 0; this.px = 0; this.py = 0; this.ph = 0; this.vx = 0; this.vy = 0;
    this.nextLook = 0; this.ranRed = false;
    this.color = r(); this.color2 = r(); this.rider = r(); this.passenger = type === "moto" && r() < 0.28; this.cargo = type === "moto" && !this.passenger && r() < 0.12 ? (r() < 0.5 ? "box" : "gas") : null;
    this.delivery = type === "moto" && r() < 0.16;
    sim.place(this);
  }
}

class Sim {
  constructor(map, seed = 1, density = 1, opt = {}) {
    this.map = map; this.rand = rngFrom(seed * 7919 + 3); this.t = 0; this.nextId = 0;
    this.vehicles = []; this.density = density;
    this.paths = {};
    this.stats = { exited: [], redRuns: [], nearMiss: [], collisions: 0, honks: 0, squeezes: 0 };
    this.pairSeen = new Map(); this.grid = new Map(); this.boxStopped = {};
    // Phase 3: a scenario genome reshapes who is on the road and how they ride (null = the default population)
    this.genome = opt.genome || null;
    if (this.genome && this.genome.density != null) this.density = this.genome.density;
    for (let i = 0; i < (opt.warmup ?? 600); i++) this.step();
  }
  path(j, from, to) { const k = j * 100 + "NESW".indexOf(from) * 10 + "NESW".indexOf(to); return this.paths[k] || (this.paths[k] = buildPath(this.map, this.map.juncs[j], from, to)); }
  /* random exit at junction j for traffic entering from arm `from` (no U-turns) */
  chooseExit(j, from) {
    const jn = this.map.juncs[j], t = this.map.turn, opts = [];
    if (jn.arms[OPP[from]]) opts.push([OPP[from], t.straight]);
    if (jn.arms[RIGHT_OF[from]]) opts.push([RIGHT_OF[from], t.right]);
    if (jn.arms[LEFT_OF[from]]) opts.push([LEFT_OF[from], t.left]);
    let q = this.rand() * opts.reduce((s, o) => s + o[1], 0);
    for (const [a, w] of opts) { if (q < w) return a; q -= w; }
    return opts[0][0];
  }
  /* signal state of an arm of a junction: G / Y / R and seconds left; null if unsignalised */
  phaseAt(jn, t) { let x = (t + jn.offset) % jn.cycle, i = 0; for (; i < jn.phases.length; i++) { if (x < jn.phases[i].t) break; x -= jn.phases[i].t; } return { i, left: jn.phases[i].t - x }; }
  light(j, arm, t = this.t) {
    const jn = this.map.juncs[j]; if (jn.type !== "signal") return null;
    const ph = this.phaseAt(jn, t), p = jn.phases[ph.i];
    if (p.green && p.green.includes(arm)) return { c: "G", left: ph.left, idx: ph.i };
    if (p.yellow && p.yellow.includes(arm)) return { c: "Y", left: ph.left, idx: ph.i };
    let left = ph.left, k = ph.i;
    for (let n = 0; n < jn.phases.length; n++) { k = (k + 1) % jn.phases.length; const q = jn.phases[k]; if (q.green && q.green.includes(arm)) break; left += q.t; }
    return { c: "R", left, idx: ph.i };
  }
  place(v) {
    const p = v.path, i = Math.min(p.n - 2, Math.max(0, Math.floor(v.s / p.ds))), f = clampN(v.s / p.ds - i, 0, 1);
    const tx = p.TX[i] + (p.TX[i + 1] - p.TX[i]) * f, ty = p.TY[i] + (p.TY[i + 1] - p.TY[i]) * f, tl = Math.hypot(tx, ty) || 1;
    const x0 = p.X[i] + (p.X[i + 1] - p.X[i]) * f, y0 = p.Y[i] + (p.Y[i + 1] - p.Y[i]) * f;
    v.tx = tx / tl; v.ty = ty / tl; v.nx = -v.ty; v.ny = v.tx;
    v.x = x0 + v.nx * v.d; v.y = y0 + v.ny * v.d;
    v.h = Math.atan2(v.ty, v.tx) + clampN(Math.atan2(v.dDot || 0, Math.max(1.5, v.v)), -0.45, 0.45);
    v.vx = v.tx * v.v; v.vy = v.ty * v.v;
    v.ch = Math.cos(v.h); v.sh = Math.sin(v.h);
  }
  bounds(v) {
    const p = v.path, i = Math.min(p.n - 1, Math.max(0, Math.round(v.s / p.ds))); let lo = p.DMIN[i], hi = p.DMAX[i];
    if (v.comply < 0.12 && v.type === "moto" && v.s < p.sIn) lo -= 1.2;
    if (v.wrongWay && v.legs === 1 && v.s < p.sStop - 3) { const w = hi - lo; lo = lo - w - 0.35; hi = lo + 1.5; }   // the far curb of the other half
    if (v.earlyLeft && p.kind === "left" && v.s > p.sStop - 12 && v.s < p.sOut) lo -= 1.8;
    return [lo + v.w2 - 0.36, hi - v.w2 + 0.36];
  }
  buildGrid() {
    this.grid.clear();
    for (const v of this.vehicles) { const k = (Math.floor(v.x / 12) + 512) * 2048 + Math.floor(v.y / 12) + 512; let l = this.grid.get(k); if (!l) this.grid.set(k, l = []); l.push(v); }
  }
  near(v, r) {
    const out = [], gx = Math.floor(v.x / 12) + 512, gy = Math.floor(v.y / 12) + 512, c = Math.ceil(r / 12);
    for (let i = -c; i <= c; i++) for (let j = -c; j <= c; j++) { const l = this.grid.get((gx + i) * 2048 + gy + j); if (l) for (const u of l) if (u !== v) out.push(u); }
    return out;
  }
  hasPriority(v, u) {
    const pv = v.s - v.path.sStop, pu = u.s - u.path.sStop;
    if (v.wait > 3 && u.wait > 3) return v.prio > u.prio;
    if (pv > 0 && pu > 0) { const fv = (v.s - v.path.sIn) / (v.path.sOut - v.path.sIn), fu = (u.s - u.path.sIn) / (u.path.sOut - u.path.sIn); return fv !== fu ? fv > fu : v.prio > u.prio; }
    if (pv > 0 !== pu > 0) return pv > 0;
    return v.prio > u.prio;
  }
  scan(v, nb) {
    const out = [];
    for (const u of nb) {
      const rx = u.x - v.x, ry = u.y - v.y;
      if (rx * rx + ry * ry > 900) continue;
      // relative heading from cached cos/sin: crossing if the angle is between 0.6 and 2.5 rad
      const cd = u.ch * v.ch + u.sh * v.sh, sd = u.sh * v.ch - u.ch * v.sh, crossing = cd < 0.8253 && cd > -0.8011;
      const c = Math.abs(cd), sn = Math.abs(sd);
      const projL = c * u.l2 + sn * u.w2, projW = sn * u.l2 + c * u.w2;
      const fx0 = rx * v.tx + ry * v.ty, fy0 = rx * v.nx + ry * v.ny;
      const tau = 0.8, qx = rx + (u.vx - v.vx) * tau, qy = ry + (u.vy - v.vy) * tau;
      const fx1 = qx * v.tx + qy * v.ty, fy1 = qx * v.nx + qy * v.ny;
      const usePred = !(crossing && this.hasPriority(v, u));
      out.push({ u, fx0, fy0, fx1, fy1, projL, projW, usePred });
    }
    return out;
  }
  freeAhead(v, d, list, look) {
    let free = look;
    const dd = d - v.d, wv = v.w2 + v.margin, back = 0.05;
    for (const e of list) {
      if (e.fx0 > back && Math.abs(e.fy0 - dd) < wv + e.projW) { const g = e.fx0 - v.l2 - e.projL; if (g < free) free = g; }
      if (e.usePred && e.fx1 > back && Math.abs(e.fy1 - dd) < wv + e.projW) { const g = e.fx1 - v.l2 - e.projL; if (g < free) free = g; }
    }
    return free;
  }
  // neighbours that one step of motion (< 2 m) could bring into contact
  close(v, nb) { const out = []; for (const u of nb) { const dx = u.x - v.x, dy = u.y - v.y, rr = v.l2 + v.w2 + u.l2 + u.w2 + 2; if (dx * dx + dy * dy < rr * rr) out.push([u, this.boxGap(v, u)]); } return out; }
  // a move is refused if it brings us into (or deeper into) another body; road users keep 15 cm from the self-driving car
  worsens(v, before) { for (const [u, g0] of before) { const g = this.boxGap(v, u), lim = u.ego ? 0.15 : 0.02; if (g < lim && g < g0 - 1e-4) return true; } return false; }
  step() {
    const dt = DT;
    this.t += dt;
    if (!this.noSpawn) this.spawn(dt);
    this.buildGrid();
    this.boxStopped = {};
    for (const v of this.vehicles) {
      if (v.v >= 0.6) continue;
      const jn = this.map.juncs[v.path.junc];
      if (Math.abs(v.x - jn.x) < jn.hwY + 1 && Math.abs(v.y - jn.y) < jn.hwX + 1) (this.boxStopped[jn.id] || (this.boxStopped[jn.id] = [])).push(v);
    }
    for (const v of this.vehicles) { v.px = v.x; v.py = v.y; v.ph = v.h; }
    for (const v of this.vehicles) { if (v.ego) this.driveEgo(v, dt); else this.drive(v, dt); }
    // finished legs: continue into the next junction, or leave the city
    const keep = [];
    for (const v of this.vehicles) {
      const p = v.path;
      if (v.s < p.len) { keep.push(v); continue; }
      if (v.ego && !p.exitTo && !v.route) { v.s = Math.min(v.s, p.len - 1); keep.push(v); continue; }
      const nxt = v.route ? v.route[v.legs] : null;
      if (p.exitTo && (!v.route || nxt)) {
        const j2 = p.exitTo.j, from2 = p.exitTo.arm, to2 = nxt ? nxt.to : this.chooseExit(j2, from2);
        v.s -= p.len; v.path = this.path(j2, from2, to2); v.legs++;
        v.committed = false; v.phaseSeen = -1; v.redRun = null; v.ranRed = false; v.nextLook = 0;
        v.signal = v.path.kind !== "straight" && (v.comply > 0.4 || v.type !== "moto");
        keep.push(v);
      } else if (v.ego) { v.s = Math.min(v.s, p.len - 1); keep.push(v); }
      else this.stats.exited.push(this.t);
    }
    this.vehicles = keep;
    for (const v of this.vehicles) this.place(v);
    if (!this.noStats) {
      this.checkConflicts();
      const cut = this.t - 60;
      for (const k of ["exited", "redRuns", "nearMiss"]) { const a = this.stats[k]; while (a.length && a[0] < cut) a.shift(); }
    }
  }
  spawn(dt) {
    const m = this.map;
    for (const e of m.entries) {
      if (this.rand() > e.flow * this.density * dt) continue;
      let r = this.rand(), type = "moto";
      const mix = this.genome ? MIX.map(([t, p]) => [t, t === "moto" ? this.genome.motoShare : p * (1 - this.genome.motoShare) / (1 - MIX[0][1])]) : MIX;
      for (const [t, p] of mix) { if (r < p) { type = t; break; } r -= p; }
      const jn = m.juncs[e.j];
      if (type === "bus" && jn.arms[e.arm].cls === "small") type = "car";
      const path = this.path(e.j, e.arm, this.chooseExit(e.j, e.arm)), T = TYPES[type];
      const lo = path.DMIN[0] + T.w / 2, hi = path.DMAX[0] - T.w / 2;
      const lanes = jn.arms[e.arm].lanes;
      let d = type === "moto" || type === "bike" ? lo + this.rand() * (hi - lo) : (lanes > 1 ? (path.kind === "left" ? lo + 0.2 : path.kind === "right" ? hi - 0.2 : (this.rand() < 0.5 ? lo + 0.2 : hi - 0.2)) : (lo + hi) / 2);
      if (path.kind === "left" && (type === "moto" || type === "bike")) d = lo + (hi - lo) * this.rand() * 0.5;
      if (path.kind === "right" && (type === "moto" || type === "bike")) d = lo + (hi - lo) * (0.5 + this.rand() * 0.5);
      const nv = new Vehicle(this, type, path, d);
      let ok = true;
      for (const v of this.vehicles) if (v.path === path || (v.path.junc === e.j && v.path.from === e.arm)) { if (v.s < 25 && this.boxGap(nv, v) < 1.0) { ok = false; break; } }
      if (ok) this.vehicles.push(nv); else this.nextId--;
    }
  }
  drive(v, dt) {
    const p = v.path, T = v.T, two = v.type === "moto" || v.type === "bike";
    const nb = this.near(v, 24), list = this.scan(v, nb);
    const look = 22 + v.v * 1.2;
    // --- traffic light
    let stopS = Infinity;
    if (p.signal && !v.committed && v.s < p.sStop + 1) {
      const L = this.light(p.junc, p.from);
      if (L.c !== "G") {
        const toStop = p.sStop - v.s - v.l2;
        if (L.c === "Y" && toStop < v.v * v.v / (2 * T.brake) + 0.5) v.committed = true;
        else {
          if (v.phaseSeen !== L.idx) {
            v.phaseSeen = L.idx;
            let pRun = (1 - v.comply) ** 2 * (two ? 0.5 : 0.15);
            if (p.kind === "right") pRun = Math.max(pRun, (1 - v.comply) * (two ? 0.7 : 0.3));
            v.redRun = L.c === "R" && this.rand() < pRun;
          }
          if (!v.redRun) stopS = p.sStop + (two ? v.aggr * 2.6 : 0);
          else if (!v.ranRed && v.s > p.sStop - 3) { v.ranRed = true; this.stats.redRuns.push(this.t); }
        }
      }
    }
    if (v.s > p.sStop + 0.5) v.committed = true;
    // --- don't block the box
    if (v.s < p.sIn && v.s > p.sStop - 8 && !(two && v.aggr > 0.9)) {
      let jam = false;
      for (const u of (this.boxStopped[p.junc] || [])) {
        if (u === v) continue;
        for (let s2 = p.sIn; s2 <= p.sOut + 4 && !jam; s2 += 2) { const i = Math.min(p.n - 1, Math.round(s2 / p.ds)); if ((u.x - p.X[i]) ** 2 + (u.y - p.Y[i]) ** 2 < (u.l2 + v.w2 + 1.2) ** 2) jam = true; }
        if (jam) break;
      }
      if (jam) stopS = Math.min(stopS, p.sIn - 0.3);
    }
    // last resort against a permanent knot: squeeze through for a moment. Inside the junction after a short wait;
    // on the approach only after a long wait with nothing (no red light) holding the queue
    const knotIn = v.wait > 12 + 10 * v.patience && v.s > p.sStop && v.s < p.sOut + 3;
    const knotOut = v.wait > 45 + 10 * v.patience && (!p.signal || this.light(p.junc, p.from).c === "G");
    if ((knotIn || knotOut) && !(v.ghostUntil > this.t)) { v.ghostUntil = this.t + 3; this.stats.squeezes++; }
    const ghost = v.ghostUntil > this.t;
    // --- sideways offset: the corridor with the most free road
    const [lo, hi] = this.bounds(v);
    if (this.t >= v.nextLook) {
      v.nextLook = this.t + 0.25 + this.rand() * 0.2;
      let best = v.dTarget, bestScore = -1e9;
      for (let d = lo; d <= hi + 1e-6; d += T.step) {
        const free = this.freeAhead(v, d, list, look);
        let score = Math.min(free, look) - Math.abs(d - v.d) * (two ? 1.2 - 0.8 * v.aggr : 5) - Math.abs(d - v.dTarget) * 0.6;
        if (!two) score -= Math.abs(d - (lo + hi) / 2) * 0.2;
        if (p.kind !== "straight" && v.s > p.sIn - 30 && v.s < p.sIn) score -= (p.kind === "left" ? d - lo : hi - d) * 0.25;
        if (score > bestScore) { bestScore = score; best = d; }
      }
      if (v.cutIn && v.v > 2 && !(v.cutCool > this.t)) {
        for (const e of list) {
          const u = e.u; if (!(u.type === "car" || u.type === "taxi" || u.type === "bus" || u.type === "truck")) continue;
          if (e.fx0 < -9 || e.fx0 > -1 || Math.abs(e.fy0) > 2.6 || Math.abs(e.fy0) < 0.6 || u.ch * v.ch + u.sh * v.sh < 0.9 || u.v < 2) continue;
          best = v.d + e.fy0; v.cutBrakeAt = this.t + 0.9; v.cutCool = this.t + 10; this.stats.cutIns = (this.stats.cutIns || 0) + 1; break;
        }
      }
      v.dTarget = clampN(best, lo, hi);
    }
    // --- speed
    const freeNow = ghost ? Math.min(look, this.freeAhead(v, v.d, list.filter((e) => e.u.ego), look)) : Math.min(this.freeAhead(v, v.d, list, look), this.freeAhead(v, (v.d + v.dTarget) / 2, list, look));
    const impatient = v.wait > 3 + 6 * v.patience;
    const gap = Math.max(0.2, v.minGap * (impatient ? 0.45 : 1)) + v.v * v.headway;
    let vT = v.vDes;
    if (v.wrongWay && v.legs === 1 && v.s < p.sStop - 3) vT *= 0.6;
    if (v.brakeRate && v.v > 3 && !(v.brakeUntil > this.t) && this.rand() < v.brakeRate * dt) { v.brakeUntil = this.t + 0.7; this.stats.suddenBrakes = (this.stats.suddenBrakes || 0) + 1; }
    if (v.brakeUntil > this.t || (v.cutBrakeAt && this.t > v.cutBrakeAt && this.t < v.cutBrakeAt + 1.0)) vT = 0;
    vT = Math.min(vT, Math.sqrt(Math.max(0, 2 * T.brake * 0.7 * (freeNow - gap))));
    if (impatient && freeNow > 0.4) vT = Math.max(vT, 0.8);
    if (stopS < Infinity) vT = Math.min(vT, Math.sqrt(Math.max(0, 2 * T.brake * 0.7 * (stopS - v.s - v.l2 - 0.3))));
    const i0 = Math.min(p.n - 1, Math.max(0, Math.round(v.s / p.ds)));
    for (let k = 0; k < 24; k += 3) { const i = Math.min(p.n - 1, i0 + k), dist = k * p.ds; vT = Math.min(vT, Math.sqrt(p.VMAX[i] * p.VMAX[i] + 2 * T.brake * 0.5 * dist)); }
    const v0 = v.v;
    if (vT > v.v) v.v = Math.min(vT, v.v + T.acc * (0.7 + 0.5 * v.aggr) * dt);
    else v.v = Math.max(vT, v.v - (freeNow < gap * 0.6 ? T.brake * 1.6 : T.brake) * dt);
    // --- move forward, never into another body (a squeezing vehicle still respects the self-driving car)
    const s0 = v.s, dDot0 = v.dDot || 0;
    const before = this.close(v, ghost ? nb.filter((u) => u.ego) : nb);   // measured in the current pose
    v.dDot = 0;
    v.s += v.v * dt;
    this.place(v);
    if (this.worsens(v, before)) { if (v0 > 3) this.stats.instantStops = (this.stats.instantStops || 0) + 1; v.s = s0; v.v = 0; v.dDot = dDot0; this.place(v); }   // undo restores the exact old pose
    v.braking = v.v < v0 - 0.6 * dt || (v.v < 0.2 && stopS < Infinity);
    // --- sideways motion
    const lat = T.latV * (0.6 + 0.8 * v.aggr) * (v.v > 0.5 ? 1 : (two ? 0.35 : 0));
    const dd = v.dTarget - v.d;
    if (Math.abs(dd) > 1e-3 && lat > 0) {
      const d0 = v.d, dDotL = v.dDot || 0, before = this.close(v, ghost ? nb.filter((u) => u.ego) : nb);
      v.d = clampN(v.d + clampN(dd, -lat * dt, lat * dt), Math.min(lo, v.d), Math.max(hi, v.d));
      v.dDot = (v.d - d0) / dt;
      this.place(v); if (this.worsens(v, before)) { v.d = d0; v.dDot = dDotL; this.place(v); }
    }
    if (v.v < 0.3 && stopS === Infinity) v.wait += dt; else v.wait = Math.max(0, v.wait - dt * 2);
    if (two) { this.stats.twoSteps = (this.stats.twoSteps || 0) + 1; if (v.wrongWay && v.legs === 1 && v.s < p.sStop - 3) this.stats.wrongSteps = (this.stats.wrongSteps || 0) + 1; }
    if (v.honk > 0) v.honk -= dt;
    else if (v.wait > 1.5 + 4 * v.patience && this.rand() < 0.04 * (0.5 + v.aggr)) { v.honk = 0.6; this.stats.honks++; }
  }
  /* the self-driving car: target speed and target sideways position come from the policy; a low-level controller tracks them */
  driveEgo(v, dt) {
    const T = v.T, c = v.ctrl || { acc: 0, lat: 0 };
    const v0 = v.v;
    let a;
    if (c.vT != null) {   // target-speed mode (v2): proportional speed controller with comfort limits
      a = clampN(1.6 * (c.vT - v.v), -T.brake * 1.4, T.acc);
      if (v.jerkA != null) a = clampN(a, v.jerkA - 12 * dt, v.jerkA + 12 * dt);
      v.jerkA = a;
    } else a = c.acc >= 0 ? clampN(c.acc, -1, 1) * T.acc : clampN(c.acc, -1, 1) * T.brake * 1.4;
    v.v = clampN(v.v + a * dt, 0, 15);
    v.braking = v.v < v0 - 0.5 * dt;
    v.s += v.v * dt;
    const [lo, hi] = this.bounds(v);
    v.dTarget = lo + (clampN(c.lat, -1, 1) + 1) / 2 * (hi - lo);
    const d0 = v.d, lat = T.latV * (v.v > 0.5 ? 1 : 0), dd = v.dTarget - v.d;
    v.d = clampN(v.d + clampN(dd, -lat * dt, lat * dt), lo, hi); v.dDot = (v.d - d0) / dt;
    this.place(v);
    if (v.v < 0.3) v.wait += dt; else v.wait = 0;
  }
  /* a copy of the world around (cx, cy) for look-ahead planning: no spawning, no statistics, shared map and paths */
  cloneLocal(cx, cy, r) {
    const c = Object.create(Sim.prototype);
    c.map = this.map; c.paths = this.paths; c.t = this.t; c.density = 0; c.nextId = this.nextId;
    const st = this.rand.st.a; c.rand = rngFrom(0); c.rand.st.a = st;
    c.stats = { exited: [], redRuns: [], nearMiss: [], collisions: 0, honks: 0, squeezes: 0 };
    c.pairSeen = new Map(); c.grid = new Map(); c.boxStopped = {}; c.noSpawn = true; c.noStats = true;
    c.vehicles = [];
    for (const v of this.vehicles) if ((v.x - cx) ** 2 + (v.y - cy) ** 2 < r * r) { const u = Object.assign(Object.create(Vehicle.prototype), v); if (v.ctrl) u.ctrl = { ...v.ctrl }; c.vehicles.push(u); }
    return c;
  }
  checkConflicts() {
    for (const v of this.vehicles) {
      for (const u of this.near(v, 6)) {
        if (u.id <= v.id || u.ego || v.ego) continue;   // contacts with the self-driving car are judged by the environment
        const dx = u.x - v.x, dy = u.y - v.y, dist = Math.hypot(dx, dy);
        const rv = Math.min(v.w2, v.l2) + Math.min(u.w2, u.l2);
        if (dist > rv + 1.2) continue;
        const sep = this.boxGap(v, u);
        const key = v.id * 100000 + u.id, last = this.pairSeen.has(key) ? this.pairSeen.get(key) : -99;
        const rel = Math.hypot(u.vx - v.vx, u.vy - v.vy);
        const ghosted = (v.ghostUntil || -99) > this.t - 6 || (u.ghostUntil || -99) > this.t - 6;
        if (sep < -0.15) { if (last < 0 && !ghosted) { this.stats.collisions++; } this.pairSeen.set(key, this.t); }
        else if (sep < 0.35 && rel > 2.5 && this.t - last > 3) { this.stats.nearMiss.push(this.t); this.pairSeen.set(key, this.t); }
      }
    }
    if (this.pairSeen.size > 20000) this.pairSeen.clear();
  }
  boxGap(a, b) {
    const dx = b.x - a.x, dy = b.y - a.y, rr = a.l2 + a.w2 + b.l2 + b.w2;
    if (dx * dx + dy * dy > (rr + 1.5) * (rr + 1.5)) return Math.sqrt(dx * dx + dy * dy) - rr;
    const ac = a.ch ?? Math.cos(a.h), as = a.sh ?? Math.sin(a.h), bc = b.ch ?? Math.cos(b.h), bs = b.sh ?? Math.sin(b.h);
    const ax = [ac, as], ay = [-as, ac], bx = [bc, bs], by = [-bs, bc];
    const d = [dx, dy];
    let best = -1e9;
    for (const n of [ax, ay, bx, by]) {
      const ra = a.l2 * Math.abs(ax[0] * n[0] + ax[1] * n[1]) + a.w2 * Math.abs(ay[0] * n[0] + ay[1] * n[1]);
      const rb = b.l2 * Math.abs(bx[0] * n[0] + bx[1] * n[1]) + b.w2 * Math.abs(by[0] * n[0] + by[1] * n[1]);
      const g = Math.abs(d[0] * n[0] + d[1] * n[1]) - ra - rb;
      if (g > best) best = g;
    }
    return best;
  }
  summary() {
    const counts = {}; let vs = 0, nMoving = 0;
    for (const v of this.vehicles) { if (v.ego) continue; counts[v.type] = (counts[v.type] || 0) + 1; if (v.v > 0.3) { vs += v.v; nMoving++; } }
    const total = this.vehicles.length - this.vehicles.filter((v) => v.ego).length;
    return { counts, total, meanSpeed: nMoving ? vs / nMoving * 3.6 : 0, flowPerHour: this.stats.exited.length * 60,
      redRunsPerMin: this.stats.redRuns.length, nearMissPerMin: this.stats.nearMiss.length, collisions: this.stats.collisions, stopped: total - nMoving };
  }
}

/* Phase 3: the scenario genome. Ranges are the search space of the evolution. */
const GENOME_SPACE = {
  density: [0.3, 1.25], motoShare: [0.6, 0.95],
  aggrMu: [0.1, 0.95], aggrSd: [0.05, 0.4], complyMu: [0.05, 0.95], patienceMu: [0.05, 0.95], speedMu: [0.1, 0.95],
  pWrong: [0, 0.25], pCutIn: [0, 0.6], pEarlyLeft: [0, 0.9], pBrake: [0, 0.08],
};
const DEFAULT_GENOME = { density: 0.8, motoShare: 0.835, aggrMu: 0.5, aggrSd: 0.29, complyMu: 0.5, patienceMu: 0.5, speedMu: 0.5, pWrong: 0, pCutIn: 0, pEarlyLeft: 0, pBrake: 0 };

if (typeof module !== "undefined") module.exports = { Sim, Vehicle, cityMap, buildPath, TYPES, DT, rngFrom, OPP, LEFT_OF, RIGHT_OF, DIRS, ROAD, GENOME_SPACE, DEFAULT_GENOME };
