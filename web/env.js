/* ======================================================================
   LOTUS: RL environment v2 around one self-driving car (the "teacher").
   The car drives a route through several junctions of a generated city.
   It sees privileged state: its route as waypoints, the signal ahead with
   its countdown, and the nearest road users with conflict features (time
   and distance of closest approach). Shared by Node workers and the browser.
   ====================================================================== */
"use strict";
const ENV = {
  K: 16, RANGE: 30, REPEAT: 2, TIMEOUT: 120, START_BACK: 45, END_AFTER: 22, ROUTE_LEGS: 3, VMAX_CMD: 14,
};
ENV.EGO_DIM = 15; ENV.WP = 10; ENV.NB_DIM = 14;
ENV.OBS_DIM = ENV.EGO_DIM + ENV.WP * 2 + ENV.K * ENV.NB_DIM;
ENV.ACT_DIM = 2;

class EgoEnv {
  constructor(opts = {}) {
    this.seed = opts.seed || 1;
    this.rand = rngFrom(this.seed * 104729 + 11);
    this.densityRange = opts.densityRange || [0.4, 1.05];
    this.fixedDensity = opts.density;
    this.mapOpts = opts.mapOpts || null;               // fixed city options, or random sizes when null
    this.reseedEvery = opts.reseedEvery ?? 3000;       // decisions between fresh cities
    this.shaping = opts.shaping ?? true;
    this.rampDecisions = opts.ramp || 0;
    this.routeLegs = opts.routeLegs || ENV.ROUTE_LEGS;
    this.terminalRed = opts.terminalRed ?? true;
    this.decisions = 0; this.nextReseed = this.reseedEvery; this.episodes = 0;
    this.sim = opts.sim || null; this.ego = null;
    this.genome = opts.genome || null;
    this.mapSeed = opts.mapSeed ?? null;                 // a fixed scenario genome (evaluation of one scenario)
    this.pool = opts.pool || null; this.pPool = opts.pPool ?? 0.5;   // evolved scenarios mixed into training
  }
  newSim() {
    let [lo, hi] = this.densityRange;
    if (this.rampDecisions) hi = lo + (hi - lo) * (0.3 + 0.7 * Math.min(1, this.decisions / this.rampDecisions));
    const d = this.fixedDensity != null ? this.fixedDensity : lo + this.rand() * (hi - lo);
    const mo = this.mapOpts || { nx: 1 + Math.floor(this.rand() * 3), ny: 1 + Math.floor(this.rand() * 3) };
    const ms = this.mapSeed ?? Math.floor(this.rand() * 1e9);   // a fixed city (e.g. the one exported to CARLA) or a fresh one
    // Phase 3: sometimes drive among an evolved population instead of the default one
    let genome = this.genome || null;
    if (!genome && this.pool && this.pool.length && this.rand() < this.pPool) genome = this.pool[Math.floor(this.rand() * this.pool.length)];
    this.sim = new Sim(cityMap(ms, mo), Math.floor(this.rand() * 1e9), d, { warmup: 350, genome });
  }
  /* a route: start on an approach to junction j0 from arm a0, then follow random turns through up to n junctions */
  makeRoute(j0, a0) {
    const sim = this.sim, legs = [];
    let j = j0, from = a0;
    for (let k = 0; k < this.routeLegs; k++) {
      const to = sim.chooseExit(j, from); legs.push({ j, from, to });
      const nx = sim.map.juncs[j].arms[to].to; if (!nx) break;
      j = nx.j; from = nx.arm;
    }
    return legs;
  }
  reset(spec) {
    if (this.ego) this.removeEgo();
    if (!this.sim || (this.reseedEvery && this.decisions >= this.nextReseed)) { this.newSim(); this.nextReseed = this.decisions + this.reseedEvery; }
    this.episodes++;
    for (let tries = 0; ; tries++) {
      if (tries === 80 && this.reseedEvery) this.newSim();
      const sim = this.sim, m = sim.map;
      let j0, a0;
      if (spec && spec.j0 != null) { j0 = spec.j0; a0 = spec.a0; }
      else { const jn = m.juncs[Math.floor(this.rand() * m.juncs.length)], arms = Object.keys(jn.arms); j0 = jn.id; a0 = arms[Math.floor(this.rand() * arms.length)]; }
      const route = spec && spec.route ? spec.route : this.makeRoute(j0, a0);
      const path = sim.path(route[0].j, route[0].from, route[0].to), s0 = Math.max(0, path.sStop - ENV.START_BACK - (tries % 20) * 2);
      const T = TYPES.car, lo = path.DMIN[0] + T.w / 2, hi = path.DMAX[0] - T.w / 2;
      const lanes = m.juncs[route[0].j].arms[route[0].from].lanes;
      const d = lanes > 1 ? (path.kind === "left" ? lo + 0.25 : path.kind === "right" ? hi - 0.25 : (this.rand() < 0.5 ? lo + 0.25 : hi - 0.25)) : (lo + hi) / 2;
      const ego = new Vehicle(sim, "car", path, d);
      ego.ego = true; ego.s = s0; ego.v = 5 + this.rand() * 4; ego.ctrl = { vT: ego.v, lat: 0 }; ego.route = route;
      ego.comply = 1; ego.aggr = 0.3; ego.patience = 0.5; ego.signal = path.kind !== "straight";
      ego.wrongWay = ego.cutIn = ego.earlyLeft = false; ego.brakeRate = 0;
      sim.place(ego);
      let ok = true;
      for (const u of sim.vehicles) if (sim.boxGap(ego, u) < 1.2) { ok = false; break; }
      if (ok || tries > 400) {
        let gap = 60;
        for (const u of sim.vehicles) { const dx = u.x - ego.x, dy = u.y - ego.y, fx = dx * ego.tx + dy * ego.ty, fy = dx * ego.nx + dy * ego.ny; if (fx > 0 && Math.abs(fy) < 2 && fx < gap + 5) gap = Math.min(gap, fx - ego.l2 - u.l2); }
        ego.v = Math.min(ego.v, Math.sqrt(Math.max(0, 2 * 3 * (gap - 2)))); ego.ctrl.vT = ego.v;
        sim.place(ego); sim.vehicles.push(ego); this.ego = ego; break;
      }
      sim.nextId--;
      if (tries % 4 === 3) sim.step();
    }
    const e = this.ego;
    this.t0 = this.sim.t; this.prevAct = [this.speedToAct(e.v), 0]; this.lastProg = this.progress(); this.result = null; this.crashWith = null;
    this.redLegs = new Set(); this.ep = { ret: 0, steps: 0, red: 0, minGap: 9 };
    this.routeLen = this.totalLen();
    return this.observe();
  }
  speedToAct(v) { return v / ENV.VMAX_CMD * 2 - 1; }
  removeEgo() { const e = this.ego; this.sim.vehicles = this.sim.vehicles.filter((v) => v !== e); this.ego = null; }
  totalLen() { const e = this.ego; let L = 0; e.route.forEach((r, k) => { const p = this.sim.path(r.j, r.from, r.to); L += k === e.route.length - 1 ? p.sOut + ENV.END_AFTER : p.len; }); return L; }
  /* metres travelled along the whole route */
  progress() { const e = this.ego; let s = 0; for (let k = 0; k < e.legs - 1; k++) { const r = e.route[k]; s += this.sim.path(r.j, r.from, r.to).len; } return s + e.s; }
  lastLeg() { return this.ego.legs >= this.ego.route.length; }
  step(act) {
    const sim = this.sim, e = this.ego;
    const a0 = Math.max(-1, Math.min(1, act[0])), a1 = Math.max(-1, Math.min(1, act[1]));
    e.ctrl = { vT: (a0 + 1) / 2 * ENV.VMAX_CMD, lat: a1 };
    let r = 0, done = false, crash = null;
    const leg0 = e.legs, p0 = e.path, light0 = p0.signal ? sim.light(p0.junc, p0.from) : null, front0 = e.s + e.l2;
    for (let k = 0; k < ENV.REPEAT; k++) { sim.step(); crash = this.collision(); if (crash) break; }
    // progress along the route
    const prog = this.progress(); r += (prog - this.lastProg) * 0.1; this.lastProg = prog;
    // red light: the front bumper crosses the stop line of a signalised approach on red
    let ranRed = false;
    if (p0.signal && !this.redLegs.has(leg0)) {
      const crossed = e.legs === leg0 ? (front0 <= p0.sStop && e.s + e.l2 > p0.sStop) : front0 <= p0.sStop;
      if (crossed) { const L = sim.light(p0.junc, p0.from); if (L.c === "R" || (light0 && light0.c === "R")) { ranRed = true; this.redLegs.add(leg0); this.ep.red = 1; r -= 6; } }
    }
    // keep a margin, and react early to anyone we are closing in on
    let minGap = 9, warn = 0;
    for (const u of sim.near(e, 25)) {
      const g = sim.boxGap(e, u); if (g < minGap) minGap = g;
      if (!this.shaping) continue;
      const dx = u.x - e.x, dy = u.y - e.y, fx = dx * e.tx + dy * e.ty, fy = dx * e.nx + dy * e.ny;
      if (fx <= 0 || Math.abs(fy) > e.w2 + u.w2 + 1.2) continue;
      const closing = (e.vx - u.vx) * e.tx + (e.vy - u.vy) * e.ty;
      if (closing <= 0.2) continue;
      const ttc = Math.max(0.01, fx - e.l2 - u.T.l / 2) / closing;
      if (ttc < 2.5) warn = Math.max(warn, (2.5 - ttc) / 2.5);
    }
    if (minGap < 0.6) r -= 0.06 * (1 - minGap / 0.6);
    r -= 0.12 * warn;
    this.ep.minGap = Math.min(this.ep.minGap, minGap);
    r -= 0.02 * Math.abs(a0 - this.prevAct[0]) + 0.02 * Math.abs(a1 - this.prevAct[1]);
    this.prevAct = [a0, a1];
    const lp = e.path, li = Math.min(lp.n - 1, Math.round(e.s / lp.ds));
    if (e.v > lp.VMAX[li] + 1.5) r -= 0.04 * (e.v - lp.VMAX[li] - 1.5);
    if (crash) { done = true; this.crashWith = crash.u; this.result = crash.fault ? "crash" : "hit"; if (crash.fault) r -= 10; }
    else if (ranRed && this.terminalRed) { done = true; this.result = "red"; }
    else if (this.lastLeg() && e.s >= Math.min(e.path.len - 1, e.path.sOut + ENV.END_AFTER)) { r += 5; done = true; this.result = "ok"; }
    else if (sim.t - this.t0 > ENV.TIMEOUT) { r -= 2; done = true; this.result = "timeout"; }
    this.ep.ret += r; this.ep.steps++; this.decisions++;
    const info = done ? { result: this.result, ret: this.ep.ret, time: sim.t - this.t0, red: this.ep.red, minGap: this.ep.minGap, legs: e.route.length, dist: prog } : null;
    return { obs: this.observe(), r, done, info };
  }
  /* contact with another road user; fault unless we were hit from behind */
  collision() {
    const e = this.ego;
    for (const u of this.sim.near(e, 8)) {
      if (this.sim.boxGap(e, u) >= -0.05) continue;
      // fault goes to whoever was moving toward the other faster at the moment of contact
      const dx = u.x - e.x, dy = u.y - e.y, dist = Math.hypot(dx, dy) || 1;
      const egoToward = (e.vx * dx + e.vy * dy) / dist, otherToward = -(u.vx * dx + u.vy * dy) / dist;
      return { u, fault: egoToward > 0.3 && egoToward >= otherToward };
    }
    return null;
  }
  /* point `ahead` metres further along the route: [x, y, kappa] */
  routePoint(ahead) {
    const e = this.ego; let k = e.legs - 1, s = e.s + ahead;
    let p = this.sim.path(e.route[k].j, e.route[k].from, e.route[k].to);
    while (s > p.len && k + 1 < e.route.length) { s -= p.len; k++; p = this.sim.path(e.route[k].j, e.route[k].from, e.route[k].to); }
    const i = Math.max(0, Math.min(p.n - 1, Math.round(s / p.ds)));
    return [p.X[i] + -p.TY[i] * e.d * 0, p.Y[i] + p.TX[i] * e.d * 0, p.K[i]];
  }
  observe() {
    const sim = this.sim, e = this.ego, p = e.path, o = new Float32Array(ENV.OBS_DIM);
    const [lo, hi] = sim.bounds(e), L = p.signal ? sim.light(p.junc, p.from) : null;
    const clip = (x, a, b) => x < a ? a : x > b ? b : x;
    let k = 0;
    o[k++] = e.v / 10; o[k++] = this.prevAct ? this.prevAct[0] : 0; o[k++] = this.prevAct ? this.prevAct[1] : 0;
    o[k++] = hi > lo ? (e.d - lo) / (hi - lo) * 2 - 1 : 0; o[k++] = (e.dDot || 0) / 2;
    o[k++] = clip((p.sStop - e.s - e.l2) / 30, -1, 2); o[k++] = clip((p.sOut - e.s) / 40, -1, 2);
    o[k++] = clip((this.routeLen - this.progress()) / 100, 0, 3);
    o[k++] = p.signal ? 1 : 0;
    o[k++] = L && L.c === "G" ? 1 : 0; o[k++] = L && L.c === "Y" ? 1 : 0; o[k++] = L && L.c === "R" ? 1 : 0;
    o[k++] = L ? clip(L.left / 30, 0, 2) : 0;
    o[k++] = e.s + e.l2 > p.sStop ? 1 : 0;
    o[k++] = clip(e.wait / 20, 0, 2);
    // route waypoints in the car's frame (the policy steers sideways with its own action)
    for (let w = 1; w <= ENV.WP; w++) {
      const [x, y] = this.routePoint(w * 3), dx = x - e.x, dy = y - e.y;
      o[k++] = clip((dx * e.tx + dy * e.ty) / 30, -1.5, 1.5); o[k++] = clip((dx * e.nx + dy * e.ny) / 30, -1.5, 1.5);
    }
    // neighbours, nearest first, with constant-velocity closest approach
    const nb = [];
    for (const u of sim.near(e, ENV.RANGE)) { const dx = u.x - e.x, dy = u.y - e.y, d2 = dx * dx + dy * dy; if (d2 < ENV.RANGE * ENV.RANGE) nb.push([d2, u]); }
    nb.sort((a, b) => a[0] - b[0]);
    this.seen = [];
    k = ENV.EGO_DIM + ENV.WP * 2;
    for (let j = 0; j < Math.min(ENV.K, nb.length); j++) {
      const u = nb[j][1], dx = u.x - e.x, dy = u.y - e.y;
      const fx = dx * e.tx + dy * e.ty, fy = dx * e.nx + dy * e.ny;
      const rvx = u.vx - e.vx, rvy = u.vy - e.vy, dh = u.h - e.h;
      const rv2 = rvx * rvx + rvy * rvy, tc = rv2 > 1e-4 ? clip(-(dx * rvx + dy * rvy) / rv2, 0, 5) : 0;
      const dc = Math.hypot(dx + rvx * tc, dy + rvy * tc);
      o[k++] = fx / ENV.RANGE; o[k++] = fy / ENV.RANGE;
      o[k++] = (rvx * e.tx + rvy * e.ty) / 10; o[k++] = (rvx * e.nx + rvy * e.ny) / 10;
      o[k++] = Math.cos(dh); o[k++] = Math.sin(dh);
      o[k++] = u.T.l / 5; o[k++] = u.T.w / 2;
      o[k++] = clip(sim.boxGap(e, u) / 5, -0.2, 1);
      o[k++] = tc / 5; o[k++] = clip(dc / 10, 0, 1.5);
      o[k++] = clip((u.dDot || 0) / 2, -1, 1); o[k++] = u.braking ? 1 : 0;   // drifting sideways, braking: intent
      o[k++] = 1;
      this.seen.push(u);
    }
    return o;
  }
}

/* small MLP policy for the browser (weights exported from PyTorch); optional frame history */
class MLPPolicy {
  constructor(spec) {
    const dec = (b64) => { const bin = atobAny(b64), buf = new ArrayBuffer(bin.length), u8 = new Uint8Array(buf); for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i); return new Float32Array(buf); };
    this.mean = dec(spec.obsMean); this.std = dec(spec.obsStd);
    this.layers = spec.layers.map((l) => ({ W: dec(l.W), b: dec(l.b), inp: l.inp, out: l.out }));
    this.name = spec.name; this.steps = spec.steps; this.hidden = spec.act || "tanh"; this.squash = !!spec.squash;
  }
  act(obs) {
    let x = new Float32Array(obs.length);
    for (let i = 0; i < obs.length; i++) x[i] = Math.max(-8, Math.min(8, (obs[i] - this.mean[i]) / this.std[i]));
    this.layers.forEach((l, li) => {
      const y = new Float32Array(l.out);
      for (let o = 0; o < l.out; o++) { let s = l.b[o]; const row = o * l.inp; for (let i = 0; i < l.inp; i++) s += l.W[row + i] * x[i]; y[o] = li < this.layers.length - 1 ? (this.hidden === "relu" ? Math.max(0, s) : Math.tanh(s)) : (this.squash ? Math.tanh(s) : s); }
      x = y;
    });
    return [Math.max(-1, Math.min(1, x[0])), Math.max(-1, Math.min(1, x[1]))];
  }
}
/* set-encoder policy (DeepSets): a shared network per neighbour, max and mean pooling, then an MLP head */
class SetPolicy {
  constructor(spec) {
    const dec = (b64) => { const bin = atobAny(b64), buf = new ArrayBuffer(bin.length), u8 = new Uint8Array(buf); for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i); return new Float32Array(buf); };
    this.mean = dec(spec.obsMean); this.std = dec(spec.obsStd);
    const L = (ls) => ls.map((l) => ({ W: dec(l.W), b: dec(l.b), inp: l.inp, out: l.out }));
    this.phi = L(spec.phi); this.rho = L(spec.rho); this.egoDim = spec.egoDim; this.K = spec.K; this.nb = spec.nbDim;
    this.name = spec.name; this.steps = spec.steps;
  }
  mlp(layers, x, lastAct) {
    layers.forEach((l, li) => { const y = new Float32Array(l.out); for (let o = 0; o < l.out; o++) { let s = l.b[o]; const row = o * l.inp; for (let i = 0; i < l.inp; i++) s += l.W[row + i] * x[i]; y[o] = li < layers.length - 1 ? Math.max(0, s) : lastAct(s); } x = y; });
    return x;
  }
  act(obs) {
    const n = obs.length, x = new Float32Array(n);
    for (let i = 0; i < n; i++) x[i] = Math.max(-8, Math.min(8, (obs[i] - this.mean[i]) / this.std[i]));
    const H = this.phi[this.phi.length - 1].out, mx = new Float32Array(H).fill(0), mn = new Float32Array(H); let cnt = 0, first = true;
    for (let j = 0; j < this.K; j++) {
      const off = this.egoDim + j * this.nb;
      if (obs[off + this.nb - 1] < 0.5) continue;   // empty slot
      const h = this.mlp(this.phi, x.subarray(off, off + this.nb), (s) => Math.max(0, s));
      for (let i = 0; i < H; i++) { mx[i] = first ? h[i] : Math.max(mx[i], h[i]); mn[i] += h[i]; }
      first = false; cnt++;
    }
    if (cnt) for (let i = 0; i < H; i++) mn[i] /= cnt;
    const z = new Float32Array(this.egoDim + 2 * H); z.set(x.subarray(0, this.egoDim)); z.set(mx, this.egoDim); z.set(mn, this.egoDim + H);
    const y = this.mlp(this.rho, z, Math.tanh);
    return [y[0], y[1]];
  }
}
function makePolicy(spec) { return spec.type === "set" ? new SetPolicy(spec) : new MLPPolicy(spec); }
function atobAny(s) { return typeof atob === "function" ? atob(s) : Buffer.from(s, "base64").toString("binary"); }

if (typeof module !== "undefined") module.exports = { EgoEnv, ENV, MLPPolicy, SetPolicy, makePolicy };
