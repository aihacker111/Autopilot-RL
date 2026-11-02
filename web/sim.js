/* ======================================================================
   LOTUS: 2D traffic core (no rendering; runs in the browser or Node).
   Units: metres, seconds. x east, y south (screen-like). Right-hand traffic.
   The road network comes from world.js (cityMap, buildPath).

   A trip is a chain of legs. Each leg is a reference path through one
   node, from the middle of the incoming road to the middle of the
   outgoing road; consecutive legs meet exactly, so a vehicle just switches
   path. Within a leg, vehicles move in Frenet coordinates (s along the path,
   d sideways). Motorbikes are not tied to lanes: they keep looking for the
   sideways offset with the most free road ahead. On a road with a
   motorbike lane they keep to it (most of them); on alleys and rural roads
   everyone shares the whole width and keeps to the right when they can.
   Grade changes how hard vehicles can accelerate and brake.

   Besides vehicles, the road is shared with agents: pedestrians (walking
   along sidewalks or the road edge, crossing where they like), buffaloes on
   rural roads, trains on the rail line, and static obstacles (parked cars,
   market stalls, parked or stalled motorbikes). Vehicles see agents like
   any other body and steer around them; agents never walk into a vehicle.
   ====================================================================== */
"use strict";
const DT = 0.1;

/* ---------------- vehicles */
const TYPES = {
  moto: { l: 1.9, w: 0.72, vDes: 9.5, acc: 2.6, brake: 4.5, latV: 1.2, step: 0.35 },
  bike: { l: 1.7, w: 0.6, vDes: 4.5, acc: 1.0, brake: 3.0, latV: 0.6, step: 0.35 },
  car: { l: 4.5, w: 1.8, vDes: 10.5, acc: 2.0, brake: 5.0, latV: 0.55, step: 1.0 },
  taxi: { l: 4.4, w: 1.75, vDes: 10.5, acc: 2.1, brake: 5.0, latV: 0.6, step: 1.0 },
  bus: { l: 10.5, w: 2.5, vDes: 8.0, acc: 1.1, brake: 3.5, latV: 0.35, step: 1.2 },
  truck: { l: 7.0, w: 2.3, vDes: 8.5, acc: 1.2, brake: 3.8, latV: 0.4, step: 1.2 },
  tractor: { l: 3.4, w: 1.6, vDes: 4.8, acc: 0.8, brake: 3.0, latV: 0.3, step: 1.0 },
};
const HEAVY = { bus: 1, truck: 1, tractor: 1 };
const MIX = [["moto", 0.835], ["car", 0.075], ["taxi", 0.035], ["bike", 0.02], ["truck", 0.02], ["bus", 0.015]];
const MIXES = {   // who enters the map, by the kind of road they come in on
  urban: MIX,
  highway: [["moto", 0.6], ["car", 0.17], ["taxi", 0.04], ["truck", 0.11], ["bus", 0.06], ["bike", 0.02]],
  rural: [["moto", 0.74], ["bike", 0.1], ["tractor", 0.05], ["car", 0.05], ["truck", 0.04], ["bus", 0.02]],
  pass: [["moto", 0.6], ["car", 0.18], ["truck", 0.14], ["bus", 0.06], ["bike", 0.02]],
  alley: [["moto", 0.93], ["bike", 0.07]],
  rail: MIX, market: MIX,
};
const isTwo = (t) => t === "moto" || t === "bike";

class Vehicle {
  constructor(sim, type, path, d) {
    const r = sim.rand, T = TYPES[type];
    this.id = sim.nextId++; this.type = type; this.T = T; this.path = path; this.legs = 1;
    this.l2 = T.l / 2; this.w2 = T.w / 2; this.s = 0; this.d = d; this.dTarget = d; this.v = T.vDes * 0.75;
    // personality: how bold, how patient, how law-abiding, how fast they like to go
    const G = sim.genome, two = isTwo(type);
    const nrm = (mu, sd) => { const u = Math.max(1e-6, r()), w = r(); return clampN(mu + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(6.2832 * w), 0, 1); };
    if (G) { this.aggr = nrm(G.aggrMu, G.aggrSd); this.patience = nrm(G.patienceMu, 0.25); this.comply = nrm(G.complyMu, 0.25); this.speedPref = 0.8 + 0.45 * nrm(G.speedMu, 0.25); }
    else { this.aggr = r(); this.patience = r(); this.comply = r(); this.speedPref = 0.8 + r() * 0.45; }
    if (HEAVY[type]) { this.aggr *= 0.6; this.comply = 0.4 + 0.6 * this.comply; }
    // Phase 3 behaviours (only when a scenario genome asks for them)
    this.wrongWay = !!(G && two && path.kind !== "left" && r() < G.pWrong);       // rides against traffic along the far curb up to the junction
    this.cutIn = !!(G && type === "moto" && r() < G.pCutIn);                      // cuts in front of a car, then brakes
    this.earlyLeft = !!(G && two && path.kind === "left" && r() < G.pEarlyLeft);   // cuts the corner on a left turn
    this.brakeRate = G && two ? G.pBrake * (0.5 + this.aggr) : 0;                 // sudden stops per second while moving
    this.vDes = T.vDes * this.speedPref;
    this.margin = 0.5 - 0.32 * this.aggr; this.minGap = (two ? 0.9 : 1.6) - 0.6 * this.aggr; this.headway = 0.85 - 0.45 * this.aggr;
    this.prio = r(); this.wait = 0; this.honk = 0; this.braking = false; this.redRun = null; this.phaseSeen = -1; this.committed = false;
    this.signal = path.kind !== "straight" && r() < 0.35 + 0.6 * this.comply;
    this.x = 0; this.y = 0; this.z = 0; this.h = 0; this.px = 0; this.py = 0; this.ph = 0; this.vx = 0; this.vy = 0;
    this.nextLook = 0; this.ranRed = false; this.overtake = -1; this.served = null;
    this.color = r(); this.color2 = r(); this.rider = r(); this.passenger = type === "moto" && r() < 0.28; this.cargo = type === "moto" && !this.passenger && r() < 0.12 ? (r() < 0.5 ? "box" : "gas") : null;
    this.delivery = type === "moto" && r() < 0.16;
    sim.place(this);
  }
}

/* ---------------- agents: pedestrians, buffaloes, trains, static obstacles */
const AGENT_SIZE = { ped: [0.5, 0.5], buffalo: [2.4, 1.05], train: [14, 3.0], parked: [4.4, 1.8], stall: [2.4, 1.0], moto: [1.9, 0.7], stalled: [1.9, 0.7], pusher: [1.9, 0.8] };
class Agent {
  constructor(sim, kind, x, y, h, o = {}) {
    const sz = AGENT_SIZE[kind] || [1, 1], l = o.l || sz[0], w = o.w || sz[1];
    this.agent = true; this.kind = kind; this.type = kind; this.id = sim.nextAgent++;
    this.x = x; this.y = y; this.z = o.z || 0; this.h = h; this.T = { l, w }; this.l2 = l / 2; this.w2 = w / 2;
    this.v = 0; this.vx = 0; this.vy = 0; this.ch = Math.cos(h); this.sh = Math.sin(h); this.static = !!o.static; this.seed = o.seed || 0;
    this.px = x; this.py = y; this.ph = h; this.braking = false; this.dDot = 0;
    Object.assign(this, o.extra || {});
  }
  pose(x, y, h, v) { this.x = x; this.y = y; this.h = h; this.ch = Math.cos(h); this.sh = Math.sin(h); this.v = v; this.vx = this.ch * v; this.vy = this.sh * v; }
}

class Sim {
  constructor(map, seed = 1, density = 1, opt = {}) {
    this.map = map; this.rand = rngFrom(seed * 7919 + 3); this.t = 0; this.nextId = 0; this.nextAgent = 1000000;
    this.vehicles = []; this.agents = []; this.density = density;
    this.paths = {};
    this.stats = { exited: [], redRuns: [], nearMiss: [], collisions: 0, honks: 0, squeezes: 0, crossings: 0 };
    this.pairSeen = new Map(); this.grid = new Map(); this.boxStopped = {};
    // Phase 3: a scenario genome reshapes who is on the road and how they ride (null = the default population)
    this.genome = opt.genome || null;
    if (this.genome && this.genome.density != null) this.density = this.genome.density;
    this.pedRate = this.genome && this.genome.pedRate != null ? this.genome.pedRate : 1;
    this.pJay = this.genome && this.genome.pJaywalk != null ? this.genome.pJaywalk : 0.15;
    for (const st of map.statics || []) this.agents.push(new Agent(this, st.kind, st.x, st.y, st.h, { static: true, l: st.l, w: st.w, z: st.z, seed: st.seed }));
    // where vehicles appear inside the map: alley and village houses
    this.sources = [];
    for (const n of map.nodes) for (const a of Object.values(n.arms)) if (a.to && a.inOK && (a.kind === "alley" || a.kind === "rural")) this.sources.push({ j: n.id, arm: a.id, flow: a.kind === "alley" ? 0.012 : 0.006, kind: a.kind });
    this.walkPop = new Map();
    for (let i = 0; i < (opt.warmup ?? 600); i++) this.step();
  }
  path(j, from, to) { let a = this.paths[j]; if (!a) a = this.paths[j] = {}; const k = from + ">" + to; return a[k] || (a[k] = buildPath(this.map, this.map.nodes[j], from, to)); }
  /* random exit at node j for traffic entering from arm `from` */
  chooseExit(j, from, type = "moto") {
    const jn = this.map.nodes[j], A = jn.arms[from], car = !isTwo(type), opts = [], only = jn.only && jn.only[from];
    for (const id of jn.order) {
      const B = jn.arms[id];
      if (car && B.noCar) continue;
      if (!B.outOK || (!car && B.noMoto) || (jn.ban && jn.ban.has(from + ">" + id)) || (only && !only.has(id))) continue;   // one-way, motorway, turn restrictions
      if (HEAVY[type] && type !== "tractor" && B.kind === "alley") continue;
      if (type === "tractor" && B.kind === "highway") continue;
      const k = turnKind(this.map, jn, A, B);
      let w = k === "straight" ? 0.55 : k === "right" ? 0.25 : k === "left" ? 0.2 : 0;
      if (jn.type === "uturn") w = k === "uturn" ? (type === "moto" ? 0.12 : 0.05) : 0.9;
      else if (k === "uturn" && jn.type === "roundabout") w = 0.04;
      if (B.kind === "alley" && A.kind !== "alley") w *= 0.35;
      if (w > 0) opts.push([id, w]);
    }
    if (!opts.length) {   // nothing allowed by the usual weights: any legal way out, else turn back
      for (const id of jn.order) { const B = jn.arms[id]; if (id !== from && B.outOK && !(car && B.noCar) && !(!car && B.noMoto)) return id; }
      return A.outOK ? from : (jn.order.find((id) => jn.arms[id].outOK) || from);
    }
    let q = this.rand() * opts.reduce((s, o) => s + o[1], 0);
    for (const [a, w] of opts) { if (q < w) return a; q -= w; }
    return opts[0][0];
  }
  /* signal state of an arm of a node: G / Y / R and seconds left; null if unsignalised. A level crossing reads as a signal */
  phaseAt(jn, t) { let x = (t + jn.offset) % jn.cycle, i = 0; for (; i < jn.phases.length; i++) { if (x < jn.phases[i].t) break; x -= jn.phases[i].t; } return { i, left: jn.phases[i].t - x }; }
  light(j, arm, t = this.t) {
    const jn = this.map.nodes[j];
    if (jn.type === "rail") return this.barrier(jn, t);
    if (jn.type !== "signal") return null;
    const ph = this.phaseAt(jn, t), p = jn.phases[ph.i];
    if (p.green && p.green.includes(arm)) return { c: "G", left: ph.left, idx: ph.i };
    if (p.yellow && p.yellow.includes(arm)) return { c: "Y", left: ph.left, idx: ph.i };
    let left = ph.left, k = ph.i;
    for (let n = 0; n < jn.phases.length; n++) { k = (k + 1) % jn.phases.length; const q = jn.phases[k]; if (q.green && q.green.includes(arm)) break; left += q.t; }
    return { c: "R", left, idx: ph.i };
  }
  /* the barrier of a level crossing: one train per period runs the whole line; closed 14 s before it reaches the road
     until 3 s after its last car has passed, with 6 s of warning bells before closing */
  trainTau(R, t) { const p = Math.max(R.period, R.len / R.v + 25), c = (t + R.phase) / p, k = Math.floor(c); return { k, tau: (c - k) * p, p }; }
  barrier(jn, t) {
    const R = this.map.rails[jn.rail.rail], { k, tau, p } = this.trainTau(R, t), L = R.cars * R.carLen;
    const arrive = jn.rail.s / R.v, leave = (jn.rail.s + L) / R.v + 3, close = arrive - 14, warn = close - 6;
    if (tau >= close && tau < leave) return { c: "R", left: leave - tau, idx: k * 3 + 2, train: true };
    if (tau >= warn && tau < close) return { c: "Y", left: close - tau, idx: k * 3 + 1 };
    return { c: "G", left: tau < warn ? warn - tau : p - tau + warn, idx: k * 3 };
  }
  place(v) {
    const p = v.path, i = Math.min(p.n - 2, Math.max(0, Math.floor(v.s / p.ds))), f = clampN(v.s / p.ds - i, 0, 1);
    const tx = p.TX[i] + (p.TX[i + 1] - p.TX[i]) * f, ty = p.TY[i] + (p.TY[i + 1] - p.TY[i]) * f, tl = Math.hypot(tx, ty) || 1;
    const x0 = p.X[i] + (p.X[i + 1] - p.X[i]) * f, y0 = p.Y[i] + (p.Y[i + 1] - p.Y[i]) * f;
    v.tx = tx / tl; v.ty = ty / tl; v.nx = -v.ty; v.ny = v.tx;
    v.x = x0 + v.nx * v.d; v.y = y0 + v.ny * v.d; v.z = p.Z[i] + (p.Z[i + 1] - p.Z[i]) * f;
    v.h = Math.atan2(v.ty, v.tx) + clampN(Math.atan2(v.dDot || 0, Math.max(1.5, v.v)), -0.45, 0.45);
    v.vx = v.tx * v.v; v.vy = v.ty * v.v;
    v.ch = Math.cos(v.h); v.sh = Math.sin(v.h);
  }
  bounds(v) {
    const p = v.path, i = Math.min(p.n - 1, Math.max(0, Math.round(v.s / p.ds))), two = isTwo(v.type);
    let lo = p.DMIN[i], hi = p.DMAX[i];
    const bike = p.BIKE[i];
    if (bike === bike) {   // a motorbike lane: most two-wheelers keep to it, cars stay out of it
      if (two && v.comply > 0.25) lo = Math.max(lo, bike + 0.15);
      else if (!two) hi = Math.min(hi, bike - 0.1);
    }
    // going round an obstacle through the oncoming half (a two-wheeler only needs to edge a little over the centre)
    if (v.overtake > this.t) lo = Math.min(lo, two ? Math.max(p.DMINX[i], p.DMIN[i] - 1.2) : p.DMINX[i]);
    if (HEAVY[v.type] && p.K[i] > 0.045 && p.DMINX[i] < p.DMIN[i] - 1) lo = Math.min(lo, p.DMIN[i] - 1.5);   // long vehicles swing wide on hairpins
    if (v.comply < 0.12 && v.type === "moto" && v.s < p.sIn) lo -= 1.2;
    if (v.wrongWay && v.legs === 1 && v.s < p.sStop - 3 && p.DMINX[i] < p.DMIN[i] - 1) { const w = hi - lo; lo = lo - w - 0.35; hi = lo + 1.5; }   // the far curb of the other half
    if (v.earlyLeft && p.kind === "left" && v.s > p.sStop - 12 && v.s < p.sOut) lo -= 1.8;
    let a = lo + v.w2 - 0.36, b = hi - v.w2 + 0.36;
    if (a > b) a = b = (a + b) / 2;
    return [a, b];
  }
  buildGrid() {
    this.grid.clear();
    const add = (v) => { const k = (Math.floor(v.x / 12) + 512) * 2048 + Math.floor(v.y / 12) + 512; let l = this.grid.get(k); if (!l) this.grid.set(k, l = []); l.push(v); };
    for (const v of this.vehicles) add(v);
    for (const a of this.agents) add(a);
  }
  near(v, r) {
    const out = [], gx = Math.floor(v.x / 12) + 512, gy = Math.floor(v.y / 12) + 512, c = Math.ceil(r / 12);
    // only what is on the same level: a car on a flyover never meets the traffic underneath
    const z = v.z || 0;
    for (let i = -c; i <= c; i++) for (let j = -c; j <= c; j++) { const l = this.grid.get((gx + i) * 2048 + gy + j); if (l) for (const u of l) if (u !== v && Math.abs((u.z || 0) - z) < 3) out.push(u); }
    return out;
  }
  hasPriority(v, u) {
    if (u.agent) return false;   // never count on a pedestrian, an animal or a train giving way
    const pv = v.s - v.path.sStop, pu = u.s - u.path.sStop;
    if (v.path.junc === u.path.junc && v.path.priority !== u.path.priority && !(v.wait > 5 + 8 * v.patience)) {   // main road before side road, through traffic before U-turns (until patience runs out)
      if (v.path.priority > u.path.priority && !(u.s > u.path.sIn)) return true;
      if (v.path.priority < u.path.priority && !(v.s > v.path.sIn)) return false;
    }
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
      const usePred = u.agent ? !u.static : !(crossing && this.hasPriority(v, u));
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
  worsens(v, before) { for (const [u, g0] of before) { const g = this.boxGap(v, u), lim = u.ego ? 0.15 : u.kind === "ped" ? 0.1 : 0.02; if (g < lim && g < g0 - 1e-4) return true; } return false; }
  step() {
    const dt = DT;
    this.t += dt;
    if (!this.noSpawn) { this.spawn(dt); this.spawnAgents(dt); this.unjam(); }
    this.updateTrains();
    this.buildGrid();
    this.updateAgents(dt);
    this.boxStopped = {};
    for (const v of this.vehicles) {
      if (v.v >= 0.6) continue;
      const jn = this.map.nodes[v.path.junc];
      // only vehicles that have actually entered the box: a front vehicle waiting at its own stop line (or creeping a
      // little past it) must not count, or the fronts of all approaches hold each other forever
      if (v.s > v.path.sIn + 0.5 && (v.x - jn.x) ** 2 + (v.y - jn.y) ** 2 < (jn.r + 1) ** 2) (this.boxStopped[jn.id] || (this.boxStopped[jn.id] = [])).push(v);
    }
    for (const v of this.vehicles) { v.px = v.x; v.py = v.y; v.ph = v.h; }
    for (const v of this.vehicles) { if (v.ego) this.driveEgo(v, dt); else this.drive(v, dt); }
    // finished legs: continue into the next node, go home into an alley or a village, or leave the map
    const keep = [];
    for (const v of this.vehicles) {
      const p = v.path;
      if (v.s < p.len) { keep.push(v); continue; }
      if (v.ego && !p.exitTo && !v.route) { v.s = Math.min(v.s, p.len - 1); keep.push(v); continue; }
      const nxt = v.route ? v.route[v.legs] : null;
      if (p.exitTo && (!v.route || nxt)) {
        if (!v.ego && !v.phantom && this.sinkArm(p) && this.rand() < 0.3) { this.stats.exited.push(this.t); continue; }
        const j2 = p.exitTo.j, from2 = p.exitTo.arm, to2 = nxt ? nxt.to : this.chooseExit(j2, from2, v.type);
        v.s -= p.len; v.path = this.path(j2, from2, to2); v.legs++;
        v.committed = false; v.phaseSeen = -1; v.redRun = null; v.ranRed = false; v.nextLook = 0; v.served = null;
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
  sinkArm(p) { const B = this.map.nodes[p.junc].arms[p.to]; return B && B.to && (B.kind === "alley" || B.kind === "rural"); }
  pickType(kind) {
    let mix = MIXES[kind] || MIX;
    if (this.genome) { const ms = this.genome.motoShare, m0 = (mix.find((q) => q[0] === "moto") || [0, 0.8])[1]; mix = mix.map(([t, p]) => [t, t === "moto" ? ms : p * (1 - ms) / Math.max(1e-6, 1 - m0)]); }
    let r = this.rand(); for (const [t, p] of mix) { if (r < p) return t; r -= p; } return "moto";
  }
  spawnOn(j, arm, type, sAt) {
    const jn = this.map.nodes[j], A = jn.arms[arm];
    if (!isTwo(type) && A.noCar) type = "moto";
    if (isTwo(type) && A.noMoto) type = "car";
    if (type === "bus" && (A.cls === "small" || A.kind === "alley")) type = "car";
    const path = this.path(j, arm, this.chooseExit(j, arm, type)), T = TYPES[type];
    const i0 = Math.min(path.n - 1, Math.round((sAt || 0) / path.ds));
    let lo = path.DMIN[i0] + T.w / 2 + 0.05, hi = path.DMAX[i0] - T.w / 2 - 0.05;
    const bike = path.BIKE[i0];
    if (bike === bike) { if (isTwo(type)) lo = Math.max(lo, bike + T.w / 2 + 0.1); else hi = Math.min(hi, bike - T.w / 2 - 0.1); }
    if (A.shared) lo = Math.max(lo, path.CTR[i0] + T.w / 2 + 0.1);   // keep to the right half on shared roads
    if (hi < lo) hi = lo = (hi + lo) / 2;
    let d = isTwo(type) ? lo + this.rand() * (hi - lo) : (A.lanes > 1 && !A.shared ? (path.kind === "left" ? lo + 0.2 : path.kind === "right" ? hi - 0.2 : (this.rand() < 0.5 ? lo + 0.2 : hi - 0.2)) : (lo + hi) / 2);
    if (isTwo(type) && path.kind === "left") d = lo + (hi - lo) * this.rand() * 0.5;
    if (isTwo(type) && path.kind === "right") d = lo + (hi - lo) * (0.5 + this.rand() * 0.5);
    const nv = new Vehicle(this, type, path, clampN(d, Math.min(lo, hi), Math.max(lo, hi)));
    if (sAt) { nv.s = sAt; nv.v *= 0.4; this.place(nv); }
    for (const v of this.vehicles) if (v.path === path || (v.path.junc === j && v.path.from === arm)) { if (Math.abs(v.s - nv.s) < 25 && this.boxGap(nv, v) < 1.0) { this.nextId--; return null; } }
    for (const a of this.agents) if ((a.x - nv.x) ** 2 + (a.y - nv.y) ** 2 < 36 && this.boxGap(nv, a) < 0.8) { this.nextId--; return null; }
    this.vehicles.push(nv); return nv;
  }
  // demand cap: a closed street network keeps filling if more enters than leaves, and then locks up end to end.
  // The cap is one moving vehicle per ~30 m of lane at density 1 (motorbikes count half), scaled by the density.
  cap() {
    if (this.capN == null) { let L = 0; for (const l of this.map.links) L += l.len * (l.kind === "alley" ? 0.3 : l.shared ? 1.2 : l.ow ? l.lanes : l.lanes * 2); this.capN = L / 30; }
    return this.capN * Math.max(0.3, this.density);
  }
  load() { let n = 0; for (const v of this.vehicles) if (!v.static && !v.ego) n += isTwo(v.type) ? 0.5 : 1; return n; }
  // like SUMO's time-to-teleport: an NPC stuck far longer than any real driver would wait (a knot the rules cannot
  // untie) is taken off the map, but never within sight of the ego, so the agent never sees a car vanish
  unjam() {
    if (this.t < (this.nextUnjam || 0)) return; this.nextUnjam = this.t + 1;
    const ego = this.vehicles.find((v) => v.ego);
    this.vehicles = this.vehicles.filter((v) => {
      if (v.ego || v.static || v.phantom) return true;
      // a rider stuck nose to nose with the car in an alley for long turns into a house (it happens all day in a hẻm)
      if (v.egoBlock && v.wait > 18 + 10 * v.patience && isTwo(v.type) && v.path.kind !== undefined && (this.map.nodes[v.path.junc].arms[v.path.to] || {}).kind === "alley") { this.stats.home = (this.stats.home || 0) + 1; return false; }
      if (!(v.wait > 35 + 25 * v.patience)) return true;
      if (ego && (v.x - ego.x) ** 2 + (v.y - ego.y) ** 2 < 60 * 60 && v.wait < 75) return true;   // near the car only a hopeless knot is cleared
      this.stats.teleports = (this.stats.teleports || 0) + 1; return false;
    });
  }
  spawn(dt) {
    if (this.load() >= this.cap()) return;
    for (const e of this.map.entries) { if (this.rand() > e.flow * this.density * dt) continue; this.spawnOn(e.j, e.arm, this.pickType(e.kind)); }
    for (const sc of this.sources) { if (this.rand() > sc.flow * this.density * dt) continue; this.spawnOn(sc.j, sc.arm, sc.kind === "alley" ? (this.rand() < 0.92 ? "moto" : "bike") : this.pickType("rural"), 1); }
  }
  /* ---- agents */
  spawnAgents(dt) {
    const map = this.map, r = this.rand;
    // pedestrians: keep each walkway near its expected population
    for (let w = 0; w < (map.walks || []).length; w++) {
      const wk = map.walks[w], L = wk.S[wk.S.length - 1], target = wk.dens * L * this.pedRate * (map.pedScale || 1) * (0.5 + 0.5 * Math.min(1.4, this.density));
      const have = this.walkPop.get(w) || 0;
      if (have < target && r() < 0.03 * (target - have + 0.2)) this.addPed(w, r() * L, r() < 0.5 ? 1 : -1);
    }
    // buffaloes on rural roads, in small herds
    if (!this.herdT || this.t > this.herdT) {
      this.herdT = this.t + 20;
      const rural = map.links.filter((L) => L.kind === "rural" && L.len > 60);
      const want = Math.round(rural.reduce((s, L) => s + L.len, 0) / 320 * Math.min(1.5, this.pedRate)), have = this.agents.filter((a) => a.kind === "buffalo" && a.lead).length;
      if (rural.length && have < want) {
        const L = rural[Math.floor(r() * rural.length)], s = 30 + r() * (L.len - 60), side = r() < 0.5 ? 1 : -1, n = 1 + Math.floor(r() * 3), dir = r() < 0.5 ? 1 : -1;
        for (let k = 0; k < n; k++) this.addBuffalo(L, s - dir * k * 3.2, side * (L.hw + 0.4 - r() * 0.5), dir, k === 0);
      }
    }
  }
  addPed(w, s, dir) {
    const wk = this.map.walks[w], p = this.walkAt(wk, s);
    const a = new Agent(this, "ped", p.x, p.y, dir > 0 ? p.ang : p.ang + Math.PI, { seed: Math.floor(this.rand() * 1e9),
      extra: { walk: w, ws: s, wdir: dir, mode: "walk", speed: 1.0 + this.rand() * 0.6, jay: this.rand() < this.pJay, crossP: wk.market ? 0.05 : wk.onRoad ? 0.012 : 0.007, lat: (this.rand() - 0.5) * 0.8, wait: 0 } });
    a.z = groundAt(this.map, a.x, a.y);
    for (const v of this.near(a, 6)) if (this.boxGap(a, v) < 0.4) return null;
    this.agents.push(a); this.walkPop.set(w, (this.walkPop.get(w) || 0) + 1); return a;
  }
  addBuffalo(L, s, d, dir, lead) {
    if (s < 15 || s > L.len - 15) return;
    const p = linkPoint(L, s, d), a = new Agent(this, "buffalo", p.x, p.y, dir > 0 ? p.ang : p.ang + Math.PI, { seed: Math.floor(this.rand() * 1e9), extra: { link: L.id, ls: s, ld: d, side: Math.sign(d), dir, lead, graze: 0, cross: null } });
    a.z = p.z; this.agents.push(a);
  }
  walkAt(wk, s) {
    const S = wk.S, n = S.length; s = clampN(s, 0, S[n - 1]);
    let lo = 0, hi = n - 1; while (hi - lo > 1) { const m = (lo + hi) >> 1; if (S[m] <= s) lo = m; else hi = m; }
    const a = wk.pts[lo], b = wk.pts[hi], f = (s - S[lo]) / Math.max(1e-9, S[hi] - S[lo]);
    return { x: a[0] + (b[0] - a[0]) * f, y: a[1] + (b[1] - a[1]) * f, ang: Math.atan2(b[1] - a[1], b[0] - a[0]) };
  }
  updateTrains() {
    if (!this.map.rails || !this.map.rails.length) return;
    this.agents = this.agents.filter((a) => a.kind !== "train");
    for (const R of this.map.rails) {
      const { tau } = this.trainTau(R, this.t), front = R.v * tau;
      for (let c = 0; c < R.cars; c++) {
        const sc = front - R.carLen * (c + 0.5); if (sc < 0 || sc > R.len) continue;
        const p = railPoint(R, sc), a = new Agent(this, "train", p.x, p.y, p.ang, { l: R.carLen - 0.6, w: 3.0, extra: { car: c, rail: R.id, loco: c === 0 } });
        a.pose(p.x, p.y, p.ang, R.v); a.z = p.z; a.metro = !!R.metro; a.id = 900000 + R.id * 100 + c; this.agents.push(a);
      }
    }
  }
  updateAgents(dt) {
    const keep = [];
    for (const a of this.agents) {
      a.px = a.x; a.py = a.y; a.ph = a.h;
      if (a.static || a.kind === "train") { keep.push(a); continue; }
      if (a.kind === "ped" ? this.stepPed(a, dt) : this.stepBuffalo(a, dt)) keep.push(a);
      else if (a.kind === "ped") this.walkPop.set(a.walk, Math.max(0, (this.walkPop.get(a.walk) || 1) - 1));
    }
    this.agents = keep;
  }
  /* a pedestrian walks along a sidewalk or the road edge; now and then crosses to the other side, at a gap or (some) without waiting for one */
  stepPed(a, dt) {
    const wk = this.map.walks[a.walk], nb = this.near(a, 10);
    let nx, ny, nh;
    if (a.mode === "walk") {
      const L = wk.S[wk.S.length - 1];
      a.ws += a.wdir * a.speed * dt;
      if (a.ws < 0 || a.ws > L) return false;
      const p = this.walkAt(wk, a.ws), nrm = rightNormal([Math.cos(p.ang), Math.sin(p.ang)]);
      nx = p.x + nrm[0] * a.lat; ny = p.y + nrm[1] * a.lat; nh = a.wdir > 0 ? p.ang : p.ang + Math.PI;
      if (this.rand() < a.crossP * dt * (a.jay ? 2.5 : 1)) {
        const pw = this.map.walks[wk.pair + (wk.side > 0 ? 1 : 0)] || this.map.walks[wk.pair];
        const other = pw === wk ? null : pw;
        if (other) {   // target: the point straight across
          let bs = 0, bd = 1e9; for (let i = 0; i < other.pts.length; i += 2) { const d = (other.pts[i][0] - a.x) ** 2 + (other.pts[i][1] - a.y) ** 2; if (d < bd) { bd = d; bs = other.S[i]; } }
          const q = this.walkAt(other, bs);
          a.mode = "wait"; a.cross = { to: this.map.walks.indexOf(other), s: bs, x: q.x, y: q.y }; a.wait = 0;
        }
      }
    } else {
      const c = a.cross, dx = c.x - a.x, dy = c.y - a.y, d = Math.hypot(dx, dy);
      if (a.mode === "wait") {
        // look both ways: wait while a vehicle would reach the crossing line within the time it takes (jaywalkers accept less)
        const need = a.jay ? 1.2 : 2.6; let clear = true;
        for (const v of this.near(a, 30)) { if (v.agent) continue; const rx = a.x - v.x, ry = a.y - v.y, along = rx * v.tx + ry * v.ty; if (along < -3 || along > 30) continue; const side = Math.abs(rx * v.nx + ry * v.ny); if (side > 12) continue; if (along / Math.max(0.5, v.v) < need || along < 4) { clear = false; break; } }
        a.wait += dt;
        if (clear || a.wait > 25) { a.mode = "cross"; this.stats.crossings++; }
        nx = a.x; ny = a.y; nh = Math.atan2(dy, dx); a.v = 0;
      } else {
        if (d < 0.4) {
          this.walkPop.set(a.walk, Math.max(0, (this.walkPop.get(a.walk) || 1) - 1));
          a.walk = c.to; a.ws = c.s; a.mode = "walk"; a.wdir = this.rand() < 0.5 ? 1 : -1; a.cross = null; this.walkPop.set(a.walk, (this.walkPop.get(a.walk) || 0) + 1);
          return true;
        }
        const st = Math.min(d, a.speed * dt); nx = a.x + dx / d * st; ny = a.y + dy / d * st; nh = Math.atan2(dy, dx);
      }
    }
    // never step into a vehicle; stop and let it pass
    const ox = a.x, oy = a.y, oh = a.h, before = this.close(a, nb.filter((u) => !u.agent || u.kind === "train" || u.static));
    a.pose(nx, ny, nh, a.mode === "wait" ? 0 : a.speed);
    if (a.mode === "cross" && this.worsens(a, before)) {
      // boxed in mid-road: weave between the vehicles (try angled steps), and after a while squeeze past ones that stand still
      a.blocked = (a.blocked || 0) + dt;
      const st = Math.hypot(nx - ox, ny - oy), h0 = Math.atan2(ny - oy, nx - ox);
      let ok = false;
      for (const dh of [0.8, -0.8, 1.5, -1.5]) { a.pose(ox + Math.cos(h0 + dh) * st, oy + Math.sin(h0 + dh) * st, h0 + dh, a.speed); if (!this.worsens(a, before)) { ok = true; break; } }
      if (!ok && a.blocked > 6) { a.pose(nx, ny, nh, a.speed); ok = !this.worsens(a, before.filter(([u]) => u.ego || u.static || u.agent || u.v > 0.3)); }
      if (ok) return true;
      a.pose(nx, ny, nh, a.speed);
    } else if (a.mode === "cross") a.blocked = 0;
    if (this.worsens(a, before)) {
      a.pose(ox, oy, oh, 0);
      if (a.mode === "walk") {   // step aside: round a parked car toward the road, away from a vehicle toward the edge
        a.ws -= a.wdir * a.speed * dt;
        const byStatic = before.some(([u, g]) => u.static && g < 0.6);
        if (wk.onRoad || byStatic) a.lat = clampN(a.lat + (byStatic ? -wk.side * 0.2 : wk.side * 0.15), -2.6, 1.8);
      }
    } else if (a.mode === "walk" && Math.abs(a.lat) > 0.45) a.lat -= Math.sign(a.lat) * 0.02;
    return true;
  }
  /* a buffalo ambles along the edge of a rural road, stops to graze, and sometimes wanders across */
  stepBuffalo(a, dt) {
    const L = this.map.links[a.link], r = this.rand;
    if (a.graze > 0) { a.graze -= dt; a.v = 0; a.vx = a.vy = 0; return true; }
    if (!a.cross && r() < 0.03 * dt) a.graze = 4 + r() * 12;
    if (!a.cross && r() < 0.012 * dt) a.cross = { from: a.ld, to: -a.side * (L.hw + 0.4 - r() * 0.5), t: 0 };
    let spd = 0.55 + (a.seed % 100) / 400, ld = a.ld;
    if (a.cross) { a.cross.t += dt / 12; ld = a.cross.from + (a.cross.to - a.cross.from) * Math.min(1, a.cross.t); if (a.cross.t >= 1) { a.side = -a.side; a.ld = a.cross.to; a.cross = null; } spd *= 0.6; }
    const ls = a.ls + a.dir * spd * dt;
    if (ls < 12 || ls > L.len - 12) { a.dir = -a.dir; return true; }
    const p = linkPoint(L, ls, ld), ox = a.x, oy = a.y, oh = a.h, ang = Math.atan2(p.y - a.y, p.x - a.x);
    const before = this.close(a, this.near(a, 8).filter((u) => !u.agent));
    a.pose(p.x, p.y, Math.hypot(p.x - a.x, p.y - a.y) > 1e-3 ? ang : a.h, spd); a.z = p.z;
    if (this.worsens(a, before)) { a.pose(ox, oy, oh, 0); return true; }
    a.ls = ls; if (!a.cross) a.ld = ld;
    return true;
  }
  drive(v, dt) {
    const p = v.path, T = v.T, two = isTwo(v.type);
    const nb = this.near(v, 24), list = this.scan(v, nb);
    const look = 22 + v.v * 1.2;
    const i0 = Math.min(p.n - 1, Math.max(0, Math.round(v.s / p.ds))), gr = p.GR[i0];
    // grade: weaker acceleration uphill, longer braking downhill (heavy vehicles feel it most)
    const accK = two ? 0.35 : HEAVY[v.type] ? 0.95 : 0.55;
    const acc = Math.max(0.15, T.acc * (0.7 + 0.5 * v.aggr) - 9.81 * gr * accK), brake = Math.max(1.6, T.brake + 9.81 * Math.min(0, gr) * 0.6);
    // --- traffic light or level-crossing barrier
    let stopS = Infinity;
    if (p.signal && !v.committed && v.s < p.sStop + 1) {
      const L = this.light(p.junc, p.from);
      if (L.c !== "G") {
        const toStop = p.sStop - v.s - v.l2;
        if (L.c === "Y" && toStop < v.v * v.v / (2 * brake) + 0.5) v.committed = true;
        else {
          if (v.phaseSeen !== L.idx) {
            v.phaseSeen = L.idx;
            let pRun = (1 - v.comply) ** 2 * (two ? 0.5 : 0.15);
            if (p.kind === "right" && !p.rail) pRun = Math.max(pRun, (1 - v.comply) * (two ? 0.7 : 0.3));
            if (p.rail) pRun *= L.train ? 0.15 : 0.5;   // sneaking under a closing barrier happens, much less once the train is close
            v.redRun = L.c === "R" && this.rand() < pRun;
          }
          if (!v.redRun) { stopS = p.sStop + (two && !p.rail ? v.aggr * 2.6 : 0); v.why = p.rail ? "barrier" : "red"; }
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
      if (jam && p.sIn - 0.3 < stopS) { stopS = p.sIn - 0.3; v.why = "box"; }
    }
    // --- a bus pulls in at its stop
    if (v.type === "bus" && p.busStops.length) {
      for (const bs of p.busStops) {
        const key = bs.s0, mid = (bs.s0 + bs.s1) / 2;
        if (v.served === key || v.s > bs.s1 || v.s < bs.s0 - 45) continue;
        v.pullIn = true;
        if (v.dwell > this.t) { stopS = Math.min(stopS, v.s + v.l2 + 0.2); }
        else if (v.dwell && v.dwell <= this.t) { v.served = key; v.dwell = 0; v.pullIn = false; }
        else { stopS = Math.min(stopS, mid + v.l2); if (v.s > mid - 1.5 && v.v < 0.4) v.dwell = this.t + 8 + this.rand() * 8; }
      }
    } else v.pullIn = false;
    // last resort against a permanent knot: squeeze through for a moment. Inside the junction after a short wait;
    // on the approach only after a long wait with nothing (no red light) holding the queue
    const knotIn = v.wait > 12 + 10 * v.patience && v.s > p.sStop && v.s < p.sOut + 3;
    const knotOut = v.wait > 45 + 10 * v.patience && (!p.signal || this.light(p.junc, p.from).c === "G");
    if ((knotIn || knotOut) && !(v.ghostUntil > this.t)) { v.ghostUntil = this.t + 3; this.stats.squeezes++; }
    const ghost = v.ghostUntil > this.t;
    // --- sideways offset: the corridor with the most free road (keeping right on shared roads, to the crown in a flood)
    const [lo, hi] = this.bounds(v);
    if (this.t >= v.nextLook) {
      v.nextLook = this.t + 0.25 + this.rand() * 0.2;
      const ctr = p.CTR[i0], shared = p.DMINX[i0] === p.DMIN[i0] && p.DMIN[i0] < ctr - 0.5, flood = p.FLOOD[i0];
      let best = v.dTarget, bestScore = -1e9;
      for (let d = lo; d <= hi + 1e-6; d += T.step) {
        const free = this.freeAhead(v, d, list, look);
        let score = Math.min(free, look) - Math.abs(d - v.d) * (two ? 1.2 - 0.8 * v.aggr : 5) - Math.abs(d - v.dTarget) * 0.6;
        if (!two) score -= Math.abs(d - (lo + hi) / 2) * 0.2;
        if (p.kind !== "straight" && v.s > p.sIn - 30 && v.s < p.sIn) score -= (p.kind === "left" ? d - lo : hi - d) * 0.25;
        if (shared) score -= Math.max(0, ctr + v.w2 + 0.2 - d) * (two ? 1.2 : 2.2);   // keep right of the middle of a shared road
        if (v.overtake > this.t) score -= Math.max(0, p.DMIN[i0] + v.w2 - d) * 0.6;
        if (flood > 0.05) score -= Math.abs(d - (shared ? ctr : ctr + 1.5)) * flood * 4;
        if (v.pullIn) score += (d - lo) * 1.5;
        if (score > bestScore) { bestScore = score; best = d; }
      }
      if (v.cutIn && v.v > 2 && !(v.cutCool > this.t)) {
        for (const e of list) {
          const u = e.u; if (u.agent || !(u.type === "car" || u.type === "taxi" || u.type === "bus" || u.type === "truck")) continue;
          if (e.fx0 < -9 || e.fx0 > -1 || Math.abs(e.fy0) > 2.6 || Math.abs(e.fy0) < 0.6 || u.ch * v.ch + u.sh * v.sh < 0.9 || u.v < 2) continue;
          best = v.d + e.fy0; v.cutBrakeAt = this.t + 0.9; v.cutCool = this.t + 10; this.stats.cutIns = (this.stats.cutIns || 0) + 1; break;
        }
      }
      v.dTarget = clampN(best, lo, hi);
    }
    // --- speed
    if (stopS === Infinity) v.why = null;
    // standing nose to nose with the self-driving car (an alley standoff)?
    v.egoBlock = false; if (v.v < 0.3) for (const e of list) if (e.u.ego && e.fx0 > 0 && e.fx0 < 7 && Math.abs(e.fy0) < v.w2 + e.projW + 0.6) { v.egoBlock = true; break; }
    const freeNow = ghost ? Math.min(look, this.freeAhead(v, v.d, list.filter((e) => e.u.ego), look)) : Math.min(this.freeAhead(v, v.d, list, look), this.freeAhead(v, (v.d + v.dTarget) / 2, list, look));
    // blocked by something that will not move (parked car, stall, stalled bike): go round it through the oncoming half
    // (on the approach well before the stop line, or on the far side of the junction; two-wheelers only once close)
    if (freeNow < (two ? 3 : 8) && v.overtake < this.t && p.DMINX[i0] < p.DMIN[i0] - 1 && (v.s < p.sStop - 6 || v.s > p.sOut + 3)) {
      for (const e of list) if (e.u.static && e.fx0 > 0 && e.fx0 < 14 && Math.abs(e.fy0) < v.w2 + e.projW + 0.3) { v.overtake = this.t + 5; v.nextLook = 0; break; }
    }
    const impatient = v.wait > 3 + 6 * v.patience;
    const gap = Math.max(0.2, v.minGap * (impatient ? 0.45 : 1)) + v.v * v.headway;
    let vT = v.vDes * p.SPD[i0];
    if (HEAVY[v.type] && gr > 0.02) vT *= 1 - Math.min(0.65, gr * 5);
    if (HEAVY[v.type] && gr < -0.04) vT = Math.min(vT, 7);
    if (v.wrongWay && v.legs === 1 && v.s < p.sStop - 3) vT *= 0.6;
    if (v.brakeRate && v.v > 3 && !(v.brakeUntil > this.t) && this.rand() < v.brakeRate * dt) { v.brakeUntil = this.t + 0.7; this.stats.suddenBrakes = (this.stats.suddenBrakes || 0) + 1; }
    if (v.brakeUntil > this.t || (v.cutBrakeAt && this.t > v.cutBrakeAt && this.t < v.cutBrakeAt + 1.0)) vT = 0;
    vT = Math.min(vT, Math.sqrt(Math.max(0, 2 * brake * 0.7 * (freeNow - gap))));
    if (impatient && freeNow > 0.4) vT = Math.max(vT, 0.8);
    if (stopS < Infinity) vT = Math.min(vT, Math.sqrt(Math.max(0, 2 * brake * 0.7 * (stopS - v.s - v.l2 - 0.3))));
    for (let k = 0; k < 24; k += 3) { const i = Math.min(p.n - 1, i0 + k), dist = k * p.ds; vT = Math.min(vT, Math.sqrt(p.VMAX[i] * p.VMAX[i] + 2 * brake * 0.5 * dist)); }
    const v0 = v.v;
    if (vT > v.v) v.v = Math.min(vT, v.v + acc * dt);
    else v.v = Math.max(vT, v.v - (freeNow < gap * 0.6 ? brake * 1.6 : brake) * dt);
    // --- move forward, never into another body (a squeezing vehicle still respects the self-driving car)
    const s0 = v.s, dDot0 = v.dDot || 0;
    const before = this.close(v, ghost ? nb.filter((u) => u.ego || u.agent) : nb);   // measured in the current pose
    v.dDot = 0;
    v.s += v.v * dt;
    this.place(v);
    if (this.worsens(v, before)) { if (v0 > 3) this.stats.instantStops = (this.stats.instantStops || 0) + 1; v.s = s0; v.v = 0; v.dDot = dDot0; this.place(v); }   // undo restores the exact old pose
    v.braking = v.v < v0 - 0.6 * dt || (v.v < 0.2 && stopS < Infinity);
    // --- sideways motion
    const lat = T.latV * (0.6 + 0.8 * v.aggr) * (v.v > 0.5 ? 1 : (two ? 0.35 : 0));
    const dd = v.dTarget - v.d;
    if (Math.abs(dd) > 1e-3 && lat > 0) {
      const d0 = v.d, dDotL = v.dDot || 0, before = this.close(v, ghost ? nb.filter((u) => u.ego || u.agent) : nb);
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
    const T = v.T, c = v.ctrl || { acc: 0, lat: 0 }, p = v.path, gr = p.GR[Math.min(p.n - 1, Math.max(0, Math.round(v.s / p.ds)))];
    const v0 = v.v;
    let a;
    if (c.vT != null) {   // target-speed mode (v2): proportional speed controller with comfort limits
      a = clampN(1.6 * (c.vT - v.v), -T.brake * 1.4, T.acc);
      if (v.jerkA != null) a = clampN(a, v.jerkA - 12 * dt, v.jerkA + 12 * dt);
      v.jerkA = a;
    } else a = c.acc >= 0 ? clampN(c.acc, -1, 1) * T.acc : clampN(c.acc, -1, 1) * T.brake * 1.4;
    a -= 9.81 * gr * 0.45;   // the controller does not fully cancel the slope
    v.v = clampN(v.v + a * dt, 0, 15);
    v.braking = v.v < v0 - 0.5 * dt;
    v.s += v.v * dt;
    const [lo, hi] = this.bounds(v);
    v.dTarget = lo + (clampN(c.lat, -1, 1) + 1) / 2 * (hi - lo);
    const d0 = v.d, lat = T.latV * (v.v > 0.5 ? 1 : 0), dd = v.dTarget - v.d;
    v.d = clampN(v.d + clampN(dd, -lat * dt, lat * dt), Math.min(lo, v.d), Math.max(hi, v.d)); v.dDot = (v.d - d0) / dt;
    this.place(v);
    if (v.v < 0.3) v.wait += dt; else v.wait = 0;
  }
  /* a copy of the world around (cx, cy) for look-ahead planning: no spawning, no statistics, shared map and paths */
  cloneLocal(cx, cy, r) {
    const c = Object.create(Sim.prototype);
    c.map = this.map; c.paths = this.paths; c.t = this.t; c.density = 0; c.nextId = this.nextId; c.nextAgent = this.nextAgent; c.genome = this.genome;
    const st = this.rand.st.a; c.rand = rngFrom(0); c.rand.st.a = st;
    c.stats = { exited: [], redRuns: [], nearMiss: [], collisions: 0, honks: 0, squeezes: 0, crossings: 0 };
    c.pairSeen = new Map(); c.grid = new Map(); c.boxStopped = {}; c.noSpawn = true; c.noStats = true; c.sources = []; c.walkPop = new Map(); c.pedRate = this.pedRate; c.pJay = this.pJay;
    c.vehicles = []; c.agents = [];
    const r2 = r * r;
    for (const v of this.vehicles) if ((v.x - cx) ** 2 + (v.y - cy) ** 2 < r2) { const u = Object.assign(Object.create(Vehicle.prototype), v); if (v.ctrl) u.ctrl = { ...v.ctrl }; c.vehicles.push(u); }
    for (const a of this.agents) if (a.kind !== "train" && (a.x - cx) ** 2 + (a.y - cy) ** 2 < r2) c.agents.push(a.static ? a : Object.assign(Object.create(Agent.prototype), a, a.cross ? { cross: { ...a.cross } } : {}));
    return c;
  }
  checkConflicts() {
    for (const v of this.vehicles) {
      for (const u of this.near(v, 6)) {
        if (u.agent || u.id <= v.id || u.ego || v.ego) continue;   // contacts with the self-driving car are judged by the environment
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
    let peds = 0, animals = 0; for (const a of this.agents) { if (a.kind === "ped") peds++; else if (a.kind === "buffalo") animals++; }
    return { counts, total, peds, animals, meanSpeed: nMoving ? vs / nMoving * 3.6 : 0, flowPerHour: this.stats.exited.length * 60,
      redRunsPerMin: this.stats.redRuns.length, nearMissPerMin: this.stats.nearMiss.length, collisions: this.stats.collisions, stopped: total - nMoving };
  }
}
function linkPoint(L, s, d) {
  const i = clampN(Math.floor(s), 0, L.pts.length - 2), f = clampN(s - i, 0, 1), p = L.pts[i], q = L.pts[i + 1];
  const tl = Math.hypot(q[0] - p[0], q[1] - p[1]) || 1, t = [(q[0] - p[0]) / tl, (q[1] - p[1]) / tl], n = rightNormal(t);
  return { x: p[0] + (q[0] - p[0]) * f + n[0] * d, y: p[1] + (q[1] - p[1]) * f + n[1] * d, ang: Math.atan2(t[1], t[0]), z: L.z[i] + (L.z[i + 1] - L.z[i]) * f };
}
function railPoint(R, s) {
  const S = R.S, n = S.length; s = clampN(s, 0, S[n - 1]);
  let lo = 0, hi = n - 1; while (hi - lo > 1) { const m = (lo + hi) >> 1; if (S[m] <= s) lo = m; else hi = m; }
  const a = R.pts[lo], b = R.pts[hi], f = (s - S[lo]) / Math.max(1e-9, S[hi] - S[lo]);
  return { x: a[0] + (b[0] - a[0]) * f, y: a[1] + (b[1] - a[1]) * f, ang: Math.atan2(b[1] - a[1], b[0] - a[0]), z: R.z ? R.z[lo] + (R.z[hi] - R.z[lo]) * f : 0 };
}

/* Phase 3: the scenario genome. Ranges are the search space of the evolution. */
const GENOME_SPACE = {
  density: [0.3, 1.25], motoShare: [0.6, 0.95],
  aggrMu: [0.1, 0.95], aggrSd: [0.05, 0.4], complyMu: [0.05, 0.95], patienceMu: [0.05, 0.95], speedMu: [0.1, 0.95],
  pWrong: [0, 0.25], pCutIn: [0, 0.6], pEarlyLeft: [0, 0.9], pBrake: [0, 0.08], pedRate: [0.3, 2.0], pJaywalk: [0, 0.6],
};
const DEFAULT_GENOME = { density: 0.8, motoShare: 0.835, aggrMu: 0.5, aggrSd: 0.29, complyMu: 0.5, patienceMu: 0.5, speedMu: 0.5, pWrong: 0, pCutIn: 0, pEarlyLeft: 0, pBrake: 0, pedRate: 1, pJaywalk: 0.15 };

if (typeof module !== "undefined") module.exports = { Sim, Vehicle, Agent, TYPES, DT, MIX, MIXES, GENOME_SPACE, DEFAULT_GENOME, linkPoint, railPoint, isTwo };
