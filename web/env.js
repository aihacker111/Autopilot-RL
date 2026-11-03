/* ======================================================================
   LOTUS: RL environment v3 around one self-driving car (the "teacher").
   The car drives a route of about 280 m through a window of real Ho Chi Minh
   City (OpenStreetMap, web/osm.js) or, without the city data, a generated
   Vietnamese map (city grid, alleys, roundabout, highway, rail crossing,
   village road with buffalo and flooding, mountain pass, or all of them).
   It sees what a driver would see: its route as waypoints, the signal or
   barrier ahead, road grade, sight distance and width, flood and pedestrian
   zones, and the nearest road users with conflict features. Road users hidden
   behind buildings, stalls, hills or bigger vehicles are masked out.
   Shared by Node workers and the browser.
   ====================================================================== */
"use strict";
const ENV = {
  K: 16, RANGE: 30, REPEAT: 2, START_BACK: 45, END_AFTER: 22, ROUTE_M: 280, ROUTE_LEGS: 4, VMAX_CMD: 14, SIGHT: 60,
};
ENV.EGO_DIM = 23; ENV.WP = 10; ENV.NB_DIM = 16;
ENV.OBS_DIM = ENV.EGO_DIM + ENV.WP * 2 + ENV.K * ENV.NB_DIM;
ENV.ACT_DIM = 2;
// how often each kind of map is used for training (the mixed region is the slowest to simulate)
// training uses the real Ho Chi Minh City (OpenStreetMap, web/hcm/) when its data is present; the generated maps
// remain available for tests (--presets) and as the fallback when the city data is missing
ENV.PRESET_W = { hcm: 1 };
ENV.PRESET_W_GEN = { region: 0.26, city: 0.2, alley: 0.08, roundabout: 0.08, highway: 0.1, rail: 0.08, rural: 0.09, pass: 0.11 };
const EYE = { ego: 1.3, ped: 1.5, buffalo: 1.3, train: 3.0, bus: 2.7, truck: 2.6, tractor: 1.8, stall: 1.6, parked: 1.3, moto: 1.3, stalled: 0.9, bike: 1.4, car: 1.3, taxi: 1.3 };

/* Defensive driving, as a cost (reward v2 of the world-model driver, and the defensive teacher that demonstrates it):
   slow down near people on or beside the road, keep a time gap to the vehicle ahead, and never drive faster than one can
   stop within what one can see. Returns the cost of one decision (>= 0). */
function defensiveCost(sim, e, sight) {
  let c = 0;
  const v = e.v;
  for (const u of sim.near(e, 22)) {
    if (u === e || u.phantom) continue;
    const dx = u.x - e.x, dy = u.y - e.y, fx = dx * e.tx + dy * e.ty, fy = dx * e.nx + dy * e.ny;
    if (fx <= 0) continue;
    if ((u.kind === "ped" || u.kind === "buffalo") && fx < 18 && Math.abs(fy) < 5.5) {   // a person near our path: be ready to stop
      const vSafe = Math.max(1.5, Math.min(9, 1.5 + 0.45 * (fx - 2) + 0.6 * Math.max(0, Math.abs(fy) - 2)));
      if (v > vSafe) c += 0.05 * (v - vSafe);
    } else if (!u.static && Math.abs(fy) < e.w2 + (u.w2 || 0.4) + 0.3 && v > 2) {   // the vehicle ahead: at least ~1.2 s of gap
      const gap = fx - e.l2 - (u.l2 || 0.5), hw = gap / v;
      if (hw < 1.2) c += 0.05 * Math.min(1.5, (1.2 - hw) / 1.2);
    }
  }
  if (sight != null) { const need = v * 0.6 + v * v / (2 * 4.5) + 2; if (sight < need) c += 0.03 * Math.min(3, (need - sight) / 3); }
  return c;
}

class EgoEnv {
  constructor(opts = {}) {
    this.seed = opts.seed || 1;
    this.rand = rngFrom(this.seed * 104729 + 11);
    this.densityRange = opts.densityRange || [0.4, 1.05];
    this.fixedDensity = opts.density;
    this.allowedCentres = opts.allowedCentres || null; this.mapOpts = opts.mapOpts || null;               // fixed map options ({preset}), or a random preset when null
    this.reseedEvery = opts.reseedEvery ?? 3000;       // decisions between fresh maps
    this.shaping = opts.shaping ?? true;
    this.rampDecisions = opts.ramp || 0;
    this.routeM = opts.routeM || ENV.ROUTE_M;
    this.routeLegs = opts.routeLegs || ENV.ROUTE_LEGS;   // junctions per route (a test drive uses long routes)
    this.maxTime = opts.maxTime || 320;                 // seconds before a route times out
    this.terminalRed = opts.terminalRed ?? true;
    this.occlusion = opts.occlusion ?? true;           // the car only observes what it can see
    this.rewardV5 = opts.reward === "v5"; this.rewardV2 = opts.reward === "v2" || this.rewardV5;              // the world-model driver: heavier crashes (people most), defensive driving
    this.rewardCarl = opts.reward === "carl";          // CaRL (Jaeger et al. 2025): route completion x soft penalty factors; infractions end the trip
    this.decisions = 0; this.nextReseed = this.reseedEvery; this.episodes = 0;
    this.sim = opts.sim || null; this.ego = null;
    this.genome = opts.genome || null;
    this.mapSeed = opts.mapSeed ?? null;                 // a fixed map (e.g. the one exported to CARLA)
    this.pool = opts.pool || null; this.pPool = opts.pPool ?? 0.5;   // evolved scenarios mixed into training
    // v3 (the world-model driver): long routes of a random length, continued across map windows ("stages")
    this.routeMs = opts.routeMs || null; this.routeW = opts.routeW || null;   // route lengths (m) and their odds, e.g. [500, 1000, 2000, 5000]
    this.stageMax = opts.stageMax || 1000;             // metres of route planned at a time inside one window
    this.half = opts.half || null;                     // half size of the city windows (m)
    this.stageCentres = opts.stageCentres || null;     // replay: the window of every stage, as recorded
    this.stage = 0; this.target = 0; this.doneDist = 0; this.doneTime = 0;
    // v3: a safety shield on the speed command, from what the car senses (the light, the stop line, who is in or entering its path)
    this.lateralShield = !!opts.lateralShield; this.tracking = !!opts.tracking; this.trackSlots = []; this.shield = !!opts.shield; this.lastExec = [0, 0]; this.intervened = 0;
  }
  pickPreset() {
    if (this.mapOpts && this.mapOpts.preset) return this.mapOpts.preset;
    const hasCity = typeof getHcmCity === "function" && !!getHcmCity();
    const Wt = hasCity ? ENV.PRESET_W : ENV.PRESET_W_GEN;
    let q = this.rand() * Object.values(Wt).reduce((a, b) => a + b, 0);
    for (const [k, w] of Object.entries(Wt)) { if (k === "hcm" && !hasCity) continue; if (q < w) return k; q -= w; }
    return "city";
  }
  newSim() {
    let [lo, hi] = this.densityRange;
    if (this.rampDecisions) hi = lo + (hi - lo) * (0.3 + 0.7 * Math.min(1, this.decisions / this.rampDecisions));
    let d = this.fixedDensity != null ? this.fixedDensity : lo + this.rand() * (hi - lo);
    const preset = this.pickPreset();
    if (preset === "region") d *= 0.8;
    const ms = this.mapSeed ?? Math.floor(this.rand() * 1e9);
    // Phase 3: sometimes drive among an evolved population instead of the default one
    let genome = this.genome || null;
    if (!genome && this.pool && this.pool.length && this.rand() < this.pPool) {   // weighted by how much can be learnt there (PLR), when given
      if (this.poolW && this.poolW.length === this.pool.length) { let q = this.rand() * this.poolW.reduce((a, b) => a + b, 0), k = 0; while (k < this.pool.length - 1 && q >= this.poolW[k]) { q -= this.poolW[k]; k++; } genome = this.pool[k]; }
      else genome = this.pool[Math.floor(this.rand() * this.pool.length)];
    }
    const extra = preset !== "hcm" ? {} : Object.assign(this.allowedCentres && this.allowedCentres.length ? { centre: this.allowedCentres[Math.floor(this.rand() * this.allowedCentres.length)] } : {}, this.half ? { half: this.half } : {}, this.stageCentres && this.stageCentres[this.stage] ? { centre: this.stageCentres[this.stage] } : {});
    this.sim = new Sim(cityMap(ms, Object.assign({}, this.mapOpts || {}, { preset }, extra)), Math.floor(this.rand() * 1e9), d, { warmup: 350, genome });
    this.starts = null;
  }
  /* where the car may start: any arm a car may use, weighted so that every kind of road gets its share */
  startArms() {
    if (this.starts) return this.starts;
    const m = this.sim.map, list = [], byKind = {};
    const real = m.preset === "hcm";
    for (const n of m.nodes) for (const id of n.order) {
      const a = n.arms[id]; if (a.noCar || a.len < 30 || !a.inOK) continue;
      if (real && a.kind === "alley" && (a.hw < 2.2 || this.rand() < 0.7)) continue;   // in the real city, start mostly on streets
      const k = real ? a.cls : n.kind || a.kind; list.push({ j: n.id, arm: id, k }); byKind[k] = (byKind[k] || 0) + 1;
    }
    if (!list.length) for (const n of m.nodes) for (const id of n.order) { const a = n.arms[id]; if (!a.noCar && a.inOK && a.len >= 15) { list.push({ j: n.id, arm: id, k: "any" }); byKind.any = (byKind.any || 0) + 1; } }
    for (const s of list) s.w = 1 / byKind[s.k];
    return (this.starts = list);
  }
  /* a route: start on the approach to node j0 from arm a0, then follow random (car-legal) turns for about routeM metres */
  makeRoute(j0, a0) {
    const sim = this.sim, legs = [];
    let j = j0, from = a0, L = 0; const seen = new Set();
    for (let k = 0; k < this.routeLegs; k++) {
      let to = sim.chooseExit(j, from, "car");
      if (this.routeLegs > ENV.ROUTE_LEGS) {   // a long route (test drive): avoid dead ends and junctions already passed
        for (let t = 0; t < 8; t++) { const nx = sim.map.nodes[j].arms[to].to; if (nx && !seen.has(nx.j)) break; to = sim.chooseExit(j, from, "car"); }
      }
      seen.add(j);
      const p = sim.path(j, from, to); legs.push({ j, from, to });
      L += k === 0 ? p.len - Math.max(0, p.sStop - ENV.START_BACK) : p.len;
      const nx = sim.map.nodes[j].arms[to].to; if (!nx || L > this.routeM) break;
      j = nx.j; from = nx.arm;
    }
    return legs;
  }
  reset(spec) {
    if (this.ego) this.removeEgo();
    if (this.routeMs && !(spec && spec.stage)) {   // a new trip: draw its length
      const W = this.routeW || this.routeMs.map(() => 1); let q = this.rand() * W.reduce((a, b) => a + b, 0), L = this.routeMs[0];
      for (let k = 0; k < this.routeMs.length; k++) { if (q < W[k]) { L = this.routeMs[k]; break; } q -= W[k]; }
      this.target = L; this.doneTravel = 0; this.doneDist = 0; this.doneTime = 0; this.stage = 0;
    }
    if (this.target) { this.routeM = Math.min(this.stageMax, Math.max(150, this.target - this.doneDist)); this.routeLegs = 30; this.noExtend = false; }
    if (!this.sim || (this.reseedEvery && this.decisions >= this.nextReseed)) { this.newSim(); this.nextReseed = this.decisions + this.reseedEvery; }
    this.episodes++;
    for (let tries = 0; ; tries++) {
      if (tries === 80 && this.reseedEvery) this.newSim();
      const sim = this.sim, m = sim.map;
      let j0, a0;
      if (spec && spec.j0 != null) { j0 = spec.j0; a0 = spec.a0; }
      else { const st = this.startArms(); if (!st.length) { this.newSim(); continue; } let q = this.rand() * st.reduce((a, b) => a + b.w, 0), pick = st[0]; for (const s of st) { if (q < s.w) { pick = s; break; } q -= s.w; } j0 = pick.j; a0 = pick.arm; }
      const route = spec && spec.route ? spec.route : this.makeRoute(j0, a0);
      const path = sim.path(route[0].j, route[0].from, route[0].to), s0 = Math.max(0, path.sStop - ENV.START_BACK - (tries % 20) * 2);
      const T = TYPES.car, i0 = Math.min(path.n - 1, Math.round(s0 / path.ds));
      let lo = path.DMIN[i0] + T.w / 2, hi = path.DMAX[i0] - T.w / 2;
      if (path.BIKE[i0] === path.BIKE[i0]) hi = Math.min(hi, path.BIKE[i0] - T.w / 2 - 0.1);
      const arm = m.nodes[route[0].j].arms[route[0].from];
      if (arm.shared) lo = Math.max(lo, path.CTR[i0] + T.w / 2 + 0.1);
      if (hi < lo) hi = lo = (lo + hi) / 2;
      const d = arm.lanes > 1 && !arm.shared ? (path.kind === "left" ? lo + 0.25 : path.kind === "right" ? hi - 0.25 : (this.rand() < 0.5 ? lo + 0.25 : hi - 0.25)) : (lo + hi) / 2;
      const ego = new Vehicle(sim, "car", path, d);
      ego.ego = true; ego.s = s0; ego.v = 5 + this.rand() * 4; ego.ctrl = { vT: ego.v, lat: 0 }; ego.route = route;
      ego.comply = 1; ego.aggr = 0.3; ego.patience = 0.5; ego.signal = path.kind !== "straight";
      ego.wrongWay = ego.cutIn = ego.earlyLeft = false; ego.brakeRate = 0;
      sim.place(ego);
      let ok = true;
      for (const u of sim.vehicles) if (sim.boxGap(ego, u) < 1.2) { ok = false; break; }
      if (ok) for (const u of sim.agents) if ((u.x - ego.x) ** 2 + (u.y - ego.y) ** 2 < 100 && sim.boxGap(ego, u) < 1.0) { ok = false; break; }
      if (ok || tries > 400) {
        let gap = 60;
        for (const u of [...sim.vehicles, ...sim.agents]) { const dx = u.x - ego.x, dy = u.y - ego.y, fx = dx * ego.tx + dy * ego.ty, fy = dx * ego.nx + dy * ego.ny; if (fx > 0 && Math.abs(fy) < 2 && fx < gap + 5) gap = Math.min(gap, fx - ego.l2 - u.l2); }
        ego.v = Math.min(ego.v, Math.sqrt(Math.max(0, 2 * 3 * (gap - 2))), path.VMAX[i0]); ego.ctrl.vT = ego.v;
        sim.place(ego); sim.vehicles.push(ego); this.ego = ego; break;
      }
      sim.nextId--;
      if (tries % 4 === 3) sim.step();
    }
    const e = this.ego;
    this.t0 = this.sim.t; this.prevAct = [this.speedToAct(e.v), 0]; this.lastProg = this.progress(); this.result = null; this.crashWith = null;
    this.redLegs = new Set(); this.trackSlots = []; this.ep = { ret: 0, steps: 0, red: 0, minGap: 9, shield: 0, lateralShield: 0, jerk: 0, contact: 0, person: 0 }; this.startProg = this.progress(); this.lastAcc = 0;
    this.routeLen = this.totalLen();
    this.timeout = Math.min(this.maxTime, 60 + (this.routeLen - this.progress()) / 2.5);
    return this.observe();
  }
  speedToAct(v) { return v / ENV.VMAX_CMD * 2 - 1; }
  removeEgo() { const e = this.ego; this.sim.vehicles = this.sim.vehicles.filter((v) => v !== e); this.ego = null; }
  legPath(k) { const r = this.ego.route[k]; return this.sim.path(r.j, r.from, r.to); }
  totalLen() { const e = this.ego; let L = 0; e.route.forEach((r, k) => { const p = this.legPath(k); L += k === e.route.length - 1 ? Math.min(p.len - 1, p.sOut + ENV.END_AFTER) : p.len; }); return L; }
  /* metres travelled along the whole route */
  progress() { const e = this.ego; let s = 0; for (let k = 0; k < e.legs - 1; k++) s += this.legPath(k).len; return s + e.s; }
  lastLeg() { return this.ego.legs >= this.ego.route.length; }
  step(act) {
    const sim = this.sim, e = this.ego;
    const a0 = Math.max(-1, Math.min(1, act[0])); let a1 = Math.max(-1, Math.min(1, act[1])); const requestedLat = a1, beforeV = e.v;
    let vT = (a0 + 1) / 2 * ENV.VMAX_CMD; this.intervened = 0;
    if (this.shield) { const cap = this.shieldCap(); if (vT > cap + 0.3) { this.intervened = vT - cap; vT = Math.max(0, cap); } }   // never faster than it can stop for
    if (this.shield && this.lateralShield) {
      a1 = this.safeLat(a1); if (a1 !== requestedLat) this.ep.lateralShield++;
    }
    e.ctrl = { vT, lat: a1 }; this.lastExec = [vT / ENV.VMAX_CMD * 2 - 1, a1];
    let r = 0, done = false, crash = null;
    if (this.staticAhead(15)) e.overtake = sim.t + 3;   // own lane blocked by something parked: the oncoming half may be used to go round it
    const stats0 = { ...sim.stats };
    const leg0 = e.legs, p0 = e.path, light0 = p0.signal ? sim.light(p0.junc, p0.from) : null, front0 = e.s + e.l2;
    for (let k = 0; k < ENV.REPEAT; k++) { sim.step(); crash = this.collision(); if (crash) break; }
    // progress along the route
    const prog = this.progress(), progBefore = this.lastProg; r += (prog - this.lastProg) * 0.1; this.lastProg = prog;
    if (this.target && !this.noExtend && this.lastLeg() && this.routeLen - prog < 250 && this.doneDist + this.routeLen < this.target - 50) this.extendRoute();
    // red light or closed barrier: the front bumper crosses the stop line
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
      if (ttc < 2.5) warn = Math.max(warn, (2.5 - ttc) / 2.5 * (u.kind === "ped" || u.kind === "buffalo" ? 1.5 : 1));
    }
    if (minGap < 0.6) r -= 0.06 * (1 - minGap / 0.6);
    r -= 0.12 * warn;
    this.ep.minGap = Math.min(this.ep.minGap, minGap);
    const usedA0 = this.rewardV5 ? this.lastExec[0] : a0;
    r -= 0.02 * Math.abs(usedA0 - this.prevAct[0]) + 0.02 * Math.abs(a1 - this.prevAct[1]);
    this.prevAct = [usedA0, a1];
    const acceleration = (e.v - beforeV) / (DT * ENV.REPEAT);
    this.ep.jerk += Math.abs(acceleration - this.lastAcc) / (DT * ENV.REPEAT); this.lastAcc = acceleration;
    if (this.intervened) this.ep.shield++;
    const lp = e.path, li = Math.min(lp.n - 1, Math.round(e.s / lp.ds));
    if (e.v > lp.VMAX[li] + 1.5) r -= 0.04 * (e.v - lp.VMAX[li] - 1.5);   // too fast for the bend, the market, the flood
    if (this.rewardV2) r -= defensiveCost(sim, e, this.sight);
    if (this.intervened && !this.rewardV5) r -= 0.03 * Math.min(5, this.intervened);   // the shield had to step in: drive so that it does not
    if (crash) {
      done = true; this.ep.contact++; if (crash.u.kind === "ped" || crash.u.kind === "buffalo") this.ep.person++; this.crashWith = crash.u; this.result = crash.fault ? "crash" : "hit";
      if (!this.rewardV2) { if (crash.fault) r -= 10; }
      else r -= crash.fault ? (crash.u.kind === "ped" || crash.u.kind === "buffalo" ? 40 : 25) : 5;   // being hit: a defensive driver often avoids it
    }
    else if (ranRed && this.terminalRed) { done = true; this.result = "red"; }
    else if (this.lastLeg() && e.s >= Math.min(e.path.len - 1, e.path.sOut + ENV.END_AFTER)) {
      // the end of the planned route: the end of the trip, or of this window (the trip goes on in the next one: nextStage)
      if (this.target && this.doneDist + this.routeLen < this.target - 50) { r += 1; done = true; this.result = "stage"; }
      else { r += 5; done = true; this.result = "ok"; }
    }
    else if (sim.t - this.t0 > this.timeout) { r -= 2; done = true; this.result = "timeout"; }
    if (this.rewardCarl) {   // 1 per 100 m of route, scaled down by soft penalties; a crash, a red light or being hit ends the trip (the lost rest is the penalty)
      let f = 1;
      if (minGap < 0.6) f *= 0.8; if (e.v > lp.VMAX[li] + 1.5) f *= 0.8; if (this.intervened) f *= 0.9; if (warn > 0.5) f *= 0.9;
      r = Math.max(0, prog - progBefore) / 100 * f;
    }
    if (!this.ep.realism) this.ep.realism = { seconds: 0, vehSeconds: 0, twoSeconds: 0, instantStops: 0, cutIns: 0, suddenBrakes: 0, wrongSteps: 0, twoSteps: 0, collisions: 0 };
    const rs = this.ep.realism; rs.seconds += DT * ENV.REPEAT; rs.vehSeconds += sim.vehicles.filter(v => !v.ego).length * DT * ENV.REPEAT;
    rs.twoSeconds += sim.vehicles.filter(v => v.type === "moto" || v.type === "bike").length * DT * ENV.REPEAT;
    for (const key of ["instantStops", "cutIns", "suddenBrakes", "wrongSteps", "twoSteps", "collisions"]) rs[key] += (sim.stats[key] || 0) - (stats0[key] || 0);
    this.ep.ret += r; this.ep.steps++; this.decisions++;
    const info = done ? { result: this.result, ret: this.ep.ret, time: this.doneTime + sim.t - this.t0, red: this.ep.red, minGap: this.ep.minGap, legs: e.route.length, dist: this.doneDist + prog, routeLen: this.target || this.routeLen,
      shield: this.ep.shield, lateralShield: this.ep.lateralShield, meanJerk: this.ep.jerk / Math.max(1, this.ep.steps), contacts: this.ep.contact, personContacts: this.ep.person, travelled: (this.doneTravel || 0) + Math.max(0, prog - this.startProg), continuous: this.stage === 0, stages: this.stage + 1, preset: sim.map.preset, kind: sim.map.nodes[e.route[0].j].kind, with: crash ? crash.u.type : null } : null;
    if (info) { info.realismExposure = { ...rs }; info.progressFraction = this.result === "ok" ? 1 : Math.max(0, Math.min(1, info.travelled / Math.max(1, this.target || this.routeLen - this.startProg))); }
    if (info) info.realismStats = { instantStopsPerVehMin: rs.instantStops / Math.max(1e-6, rs.vehSeconds / 60), cutInsPerVehMin: rs.cutIns / Math.max(1e-6, rs.vehSeconds / 60), suddenBrakesPerTwoMin: rs.suddenBrakes / Math.max(1e-6, rs.twoSeconds / 60), npcCollisionsPerMin: rs.collisions / Math.max(1e-6, rs.seconds / 60), wrongWayShare: rs.wrongSteps / Math.max(1, rs.twoSteps), hitRate: this.result === "hit" ? 1 : 0 };
    return { obs: this.observe(), r, done, info };
  }
  /* v3 safety shield: the highest speed command that still lets the car stop for what it senses. A red light (or an
     amber it can still stop for) at the stop line; any road user in its path, or entering it before the car gets there
     (a jaywalker, a motorbike cutting in), with the room to stop behind it. Uses only what the car can see. */
  shieldCap() {
    // B: the deceleration it counts on, below what the car can do (its speed controller brakes progressively)
    const sim = this.sim, e = this.ego, p = e.path, B = 3.0, S0 = 2.5, stop = (d) => Math.sqrt(Math.max(0, 2 * B * Math.max(0, d)));
    let cap = Infinity;
    if (p.signal) {
      const L = sim.light(p.junc, p.from), d = p.sStop - e.s - e.l2;
      if (L.c !== "G" && d > -0.5 && (L.c === "R" || d > e.v * e.v / (2 * 6.5))) cap = Math.min(cap, d < 3 ? 0 : stop(d - 2.5));   // aim 2.5 m short: the car still rolls a little
    }
    const half = e.w2 + 0.6;
    for (const u of this.seen || []) {
      if (u === e || u.phantom) continue;
      const dx = u.x - e.x, dy = u.y - e.y, fx = dx * e.tx + dy * e.ty, fy = dx * e.nx + dy * e.ny;
      if (fx <= 0 || fx > 45) continue;
      if ((u.kind === "ped" || u.kind === "buffalo") && fx < 18 && Math.abs(fy) < 5.5)   // a person beside the path may step in: slow down beforehand
        cap = Math.min(cap, Math.max(1.5, 1.5 + 0.45 * (fx - 2) + 0.6 * Math.max(0, Math.abs(fy) - 2)));
      const w = half + (u.w2 || 0.4), vlat = (u.vx || 0) * e.nx + (u.vy || 0) * e.ny, tReach = fx / Math.max(1.5, e.v);
      const fyAt = fy + vlat * Math.min(3, tReach);   // where it will be sideways when the car gets there
      if (Math.abs(fy) > w && Math.abs(fyAt) > w && Math.sign(fy) === Math.sign(fyAt)) continue;
      const gap = fx - e.l2 - (u.l2 || 0.5), vAlong = Math.abs(fy) <= w ? Math.max(0, (u.vx || 0) * e.tx + (u.vy || 0) * e.ty) : 0;
      cap = Math.min(cap, Math.sqrt(vAlong * vAlong + 2 * B * Math.max(0, gap - S0)));
    }
    return cap;
  }
  safeLat(request) {
    const e = this.ego, [lo, hi] = this.sim.bounds(e), horizon = 1.2;
    const target = a => lo + (a + 1) / 2 * (hi - lo);
    const current = hi > lo ? Math.max(-1, Math.min(1, (e.d - lo) / (hi - lo) * 2 - 1)) : 0;
    const cost = a => {
      let risk = 0;
      for (const u of this.seen || []) {
        const dx = u.x - e.x, dy = u.y - e.y;
        const fx = dx * e.tx + dy * e.ty, fy = dx * e.nx + dy * e.ny;
        const rvx = (u.vx - e.vx) * e.tx + (u.vy - e.vy) * e.ty;
        const rvy = (u.vx - e.vx) * e.nx + (u.vy - e.vy) * e.ny;
        for (const t of [.4, .8, horizon]) {
          const sideways = Math.max(-e.T.latV * t, Math.min(e.T.latV * t, target(a) - e.d));
          const gapX = Math.abs(fx + rvx * t) - e.l2 - u.l2;
          const gapY = Math.abs(fy + rvy * t - sideways) - e.w2 - u.w2;
          if (gapX < .5 && gapY < .3) risk += Math.max(0, .3 - gapY) * (gapX < 0 ? 2 : 1);
        }
      }
      return risk;
    };
    const initial = cost(request); if (initial <= 0) return request;
    let best = request, bestCost = initial;
    for (const a of [current, -1, -.5, 0, .5, 1]) { const c = cost(a); if (c + .05 < bestCost) { best = a; bestCost = c; } }
    return best;
  }
  /* v4 counterfactual: what would have happened with another action, held for k decisions, on a copy of the world around
     the car, with no shield (the point is to see the consequence, e.g. squeezing into that gap). The real environment is
     not touched. Each transition: observation after it, executed action, reward, done, terminal, contact, move (dx, dy, dh). */
  branch(act, k, radius = 45) {
    const e = this.ego, c = this.sim.cloneLocal(e.x, e.y, radius), b = Object.assign(Object.create(EgoEnv.prototype), this);
    b.sim = c; b.ego = c.vehicles.find((v) => v.ego); b.shield = false; b.noExtend = true;
    b.ep = Object.assign({}, this.ep, this.ep.realism ? { realism: { ...this.ep.realism } } : {}); b.redLegs = new Set(this.redLegs); b.prevAct = this.prevAct.slice(); b.trackSlots = this.trackSlots.slice(); b.rand = rngFrom(0); b.rand.st.a = this.rand.st.a;
    const out = [];
    for (let i = 0; i < k; i++) {
      const x0 = b.ego.x, y0 = b.ego.y, h0 = b.ego.h, c0 = Math.cos(h0), s0 = Math.sin(h0);
      const r = b.step(act), dx = b.ego.x - x0, dy = b.ego.y - y0; let dh = b.ego.h - h0; while (dh > Math.PI) dh -= 2 * Math.PI; while (dh < -Math.PI) dh += 2 * Math.PI;
      const res = r.done ? r.info.result : null;
      out.push({ obs: r.obs, exec: b.lastExec.slice(), r: r.r, done: r.done, term: r.done && res !== "timeout" && res !== "stage", contact: res === "crash" || res === "hit", move: [dx * c0 + dy * s0, -dx * s0 + dy * c0, dh] });
      if (r.done) break;
    }
    return out;
  }
  /* v3: the route goes on from its last junction while the window allows (no dead ends, no loops) */
  extendRoute() {
    const e = this.ego, last = e.route[e.route.length - 1], nx = this.sim.map.nodes[last.j].arms[last.to].to;
    if (!nx) { this.noExtend = true; return; }
    const keep = this.routeM; this.routeM = Math.min(this.stageMax, this.target - this.doneDist - this.routeLen);
    const more = this.makeRoute(nx.j, nx.arm); this.routeM = keep;
    if (!more.length) { this.noExtend = true; return; }
    const before = this.routeLen; e.route.push(...more); this.routeLen = this.totalLen(); this.timeout += (this.routeLen - before) / 2.5;
  }
  /* v3: the trip continues in a fresh window (the "stage" result of step): the same trip, its totals kept */
  nextStage() {
    const keep = { ep: this.ep, doneDist: this.doneDist + this.routeLen, doneTime: this.doneTime + this.sim.t - this.t0, target: this.target, travelled: (this.doneTravel || 0) + Math.max(0, this.progress() - this.startProg) };
    this.doneDist = keep.doneDist; this.doneTravel = keep.travelled; this.doneTime = keep.doneTime;
    this.stage++; this.newSim(); this.nextReseed = this.decisions + this.reseedEvery;
    const o = this.reset({ stage: true });
    this.ep = keep.ep; this.target = keep.target;
    return o;
  }
  /* contact with another road user or obstacle; fault unless the other one was moving into us */
  collision() {
    const e = this.ego;
    for (const u of this.sim.near(e, 9)) {
      if (this.sim.boxGap(e, u) >= -0.05) continue;
      // fault goes to whoever was moving toward the other faster at the moment of contact
      const dx = u.x - e.x, dy = u.y - e.y, dist = Math.hypot(dx, dy) || 1;
      const egoToward = (e.vx * dx + e.vy * dy) / dist, otherToward = -(u.vx * dx + u.vy * dy) / dist;
      return { u, fault: egoToward > 0.3 && egoToward >= otherToward };
    }
    return null;
  }
  /* something that will not move (parked car, stall, stalled bike) in the car's lane within `range` metres, on an undivided road */
  staticAhead(range) {
    const e = this.ego, p = e.path, i = Math.min(p.n - 1, Math.max(0, Math.round(e.s / p.ds)));
    if (!(p.DMINX[i] < p.DMIN[i] - 1) || e.s > p.sStop - 4 && e.s < p.sOut) return false;
    for (const u of this.sim.near(e, range + 4)) {
      if (!u.static) continue;
      const dx = u.x - e.x, dy = u.y - e.y, fx = dx * e.tx + dy * e.ty, fy = dx * e.nx + dy * e.ny;
      if (fx > 0 && fx < range && Math.abs(fy) < e.w2 + u.w2 + 0.25) return true;
    }
    return false;
  }
  /* point `ahead` metres further along the route: [x, y, kappa, legIndex, s] */
  routePoint(ahead) {
    const e = this.ego; let k = e.legs - 1, s = e.s + ahead;
    let p = this.legPath(k);
    while (s > p.len && k + 1 < e.route.length) { s -= p.len; k++; p = this.legPath(k); }
    const i = Math.max(0, Math.min(p.n - 1, Math.round(s / p.ds)));
    return [p.X[i], p.Y[i], p.K[i], k, s, p, i];
  }
  /* can the car see u? (eyes 1.3 m up; close neighbours are always noticed) */
  visibleTo(e, u) {
    if (!this.occlusion) return true;
    const dx = u.x - e.x, dy = u.y - e.y; if (dx * dx + dy * dy < 49) return true;
    return lineOfSight(this.sim.map, e.x, e.y, EYE.ego, u.x, u.y, EYE[u.type] || 1.3);
  }
  /* how far along the route the road surface stays in view (crest, bend, corner) */
  sightDist() {
    const e = this.ego, m = this.sim.map;
    for (let d = 6; d <= ENV.SIGHT; d += 6) { const [x, y] = this.routePoint(d); if (!lineOfSight(m, e.x, e.y, EYE.ego, x, y, 0.7)) return d - 6; }
    return ENV.SIGHT;
  }
  /* the next node where the car has to give way (side road, roundabout entry, unsignalised junction) within `range` metres */
  yieldAhead(range = 40) {
    const e = this.ego; let dist = -e.s;
    for (let k = e.legs - 1; k < e.route.length; k++) {
      const p = this.legPath(k), toStop = dist + p.sStop - e.l2;
      if (toStop > range) break;
      if (toStop > -2 && (p.round || p.ntype === "yield" || (p.ntype === "priority" && p.priority === 0) || p.kind === "uturn")) return Math.max(0, toStop);
      dist += p.len;
    }
    return null;
  }
  observe() {
    const sim = this.sim, e = this.ego, p = e.path, o = new Float32Array(ENV.OBS_DIM);
    const [lo, hi] = sim.bounds(e), L = p.signal ? sim.light(p.junc, p.from) : null;
    const clip = (x, a, b) => x < a ? a : x > b ? b : x;
    const i0 = Math.min(p.n - 1, Math.max(0, Math.round(e.s / p.ds)));
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
    // the road itself: slope, how far one can see, width, giving way, water, people on the road
    const ahead = this.routePoint(25), pa = ahead[5], ia = ahead[6];
    o[k++] = clip(p.GR[i0] * 10, -2, 2); o[k++] = clip(pa.GR[ia] * 10, -2, 2);
    this.sight = this.occlusion ? this.sightDist() : ENV.SIGHT; o[k++] = this.sight / ENV.SIGHT;
    o[k++] = clip((hi - lo + e.T.w) / 6, 0, 2);
    const yd = this.yieldAhead(); o[k++] = yd == null ? 0 : 1 - yd / 40;
    let fl = 0; for (let d = 0; d <= 30; d += 5) { const q = this.routePoint(d); fl = Math.max(fl, q[5].FLOOD[q[6]]); }
    o[k++] = p.FLOOD[i0] * 3; o[k++] = fl * 3;
    o[k++] = p.MKT[i0] || p.DMINX[i0] === p.DMIN[i0] && p.DMIN[i0] < p.CTR[i0] - 0.5 ? 1 : 0;
    // route waypoints in the car's frame (the policy steers sideways with its own action)
    for (let w = 1; w <= ENV.WP; w++) {
      const [x, y] = this.routePoint(w * 3), dx = x - e.x, dy = y - e.y;
      o[k++] = clip((dx * e.tx + dy * e.ty) / 30, -1.5, 1.5); o[k++] = clip((dx * e.nx + dy * e.ny) / 30, -1.5, 1.5);
    }
    // neighbours the car can see, nearest first, with constant-velocity closest approach
    const nb = []; this.hidden = [];
    for (const u of sim.near(e, ENV.RANGE)) { const dx = u.x - e.x, dy = u.y - e.y, d2 = dx * dx + dy * dy; if (d2 < ENV.RANGE * ENV.RANGE) { if (this.visibleTo(e, u)) nb.push([d2, u]); else this.hidden.push(u); } }
    nb.sort((a, b) => a[0] - b[0]);
    if (this.tracking) {
      const nearest = nb.slice(0, ENV.K), byId = new Map(nearest.map(x => [x[1].id, x]));
      const ordered = this.trackSlots.filter(id => byId.has(id));
      for (const x of nearest) if (!ordered.includes(x[1].id)) ordered.push(x[1].id);
      this.trackSlots = ordered; nb.splice(0, nb.length, ...ordered.map(id => byId.get(id)));
    }
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
      o[k++] = u.kind === "ped" || u.kind === "buffalo" ? 1 : 0; o[k++] = u.static ? 1 : 0;
      o[k++] = 1;
      this.seen.push(u);
    }
    return o;
  }
  /* for the look-ahead teacher: who the car can see now, and "phantoms" where it cannot see (a motorbike coming out of
     a hidden side road, something stopped just beyond the crest or the bend) */
  planningView(radius = 32) {
    const sim = this.sim, e = this.ego, m = sim.map, vis = new Set(), ph = [];
    for (const u of sim.near(e, radius)) if (this.visibleTo(e, u)) vis.add((u.agent ? "a" : "v") + u.id);
    if (this.occlusion) {
      let dist = -e.s;
      for (let k = e.legs - 1; k < e.route.length && ph.length < 3; k++) {
        const p = this.legPath(k), toBox = dist + p.sIn;
        if (toBox > 45) break;
        if (toBox > -3) {
          const jn = m.nodes[p.junc];
          for (const id of jn.order) {
            if (id === p.from || id === p.to || ph.length >= 3) continue;
            const B = jn.arms[id]; if (B.len < B.box + 12) continue;
            const sB = Math.min(B.len - 1, B.box + 16), q = armAt(B, sB), n = rightNormal([q.tx, q.ty]), c = laneOffset(B);
            if (lineOfSight(m, e.x, e.y, EYE.ego, q.x - n[0] * c, q.y - n[1] * c, 1.3)) continue;
            ph.push({ j: jn.id, from: id, to: p.to, s: B.len - sB });
          }
        }
        dist += p.len;
      }
      const sd = this.sight ?? this.sightDist();
      if (sd < ENV.SIGHT - 6) { const q = this.routePoint(sd + 3); ph.push({ block: true, k: q[3], s: q[4] }); }
    }
    return { vis, ph };
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
/* the policy prior of the TD-MPC2 world model (rl/tdmpc2.py): set encoder -> latent (LayerNorm, Mish, SimNorm) -> policy
   head; drives without the planner (the planner needs the dynamics, reward and value networks, which stay in Python) */
class TDMPCPolicy extends SetPolicy {
  constructor(spec) {
    super(Object.assign({}, spec, { rho: spec.pi }));
    const dec = (b64) => { const bin = atobAny(b64), buf = new ArrayBuffer(bin.length), u8 = new Uint8Array(buf); for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i); return new Float32Array(buf); };
    const L = (ls) => ls.map((l) => ({ W: dec(l.W), b: dec(l.b), inp: l.inp, out: l.out, act: l.act, ln: l.ln ? [dec(l.ln[0]), dec(l.ln[1])] : null }));
    this.headL = L(spec.head); this.piL = L(spec.pi); this.sn = spec.simnorm || 8;
  }
  run(layers, x) {
    for (const l of layers) {
      let y = new Float32Array(l.out);
      for (let o = 0; o < l.out; o++) { let s = l.b[o]; const row = o * l.inp; for (let i = 0; i < l.inp; i++) s += l.W[row + i] * x[i]; y[o] = s; }
      if (l.ln) { let m = 0, v = 0; for (const t of y) m += t; m /= y.length; for (const t of y) v += (t - m) ** 2; v = Math.sqrt(v / y.length + 1e-5); for (let i = 0; i < y.length; i++) y[i] = (y[i] - m) / v * l.ln[0][i] + l.ln[1][i]; }
      if (l.act === "mish") for (let i = 0; i < y.length; i++) { const t = y[i]; y[i] = t * Math.tanh(Math.log1p(Math.exp(Math.min(20, t)))); }
      else if (l.act === "relu") for (let i = 0; i < y.length; i++) y[i] = Math.max(0, y[i]);
      else if (l.act === "simnorm") for (let g = 0; g < y.length; g += this.sn) { let mx = -1e9, s = 0; for (let i = g; i < g + this.sn; i++) mx = Math.max(mx, y[i]); for (let i = g; i < g + this.sn; i++) { y[i] = Math.exp(y[i] - mx); s += y[i]; } for (let i = g; i < g + this.sn; i++) y[i] /= s; }
      x = y;
    }
    return x;
  }
  encode(obs) {
    const n = obs.length, x = new Float32Array(n);
    for (let i = 0; i < n; i++) x[i] = Math.max(-8, Math.min(8, (obs[i] - this.mean[i]) / this.std[i]));
    const H = this.phi[this.phi.length - 1].out, mx = new Float32Array(H).fill(0), mn = new Float32Array(H); let cnt = 0;
    for (let j = 0; j < this.K; j++) {
      const off = this.egoDim + j * this.nb; if (obs[off + this.nb - 1] < 0.5) continue;
      const h = this.mlp(this.phi, x.subarray(off, off + this.nb), (s) => Math.max(0, s));
      for (let i = 0; i < H; i++) { mx[i] = Math.max(mx[i], h[i]); mn[i] += h[i]; } cnt++;
    }
    if (cnt) for (let i = 0; i < H; i++) mn[i] /= cnt;
    const f = new Float32Array(this.egoDim + 2 * H); f.set(x.subarray(0, this.egoDim)); f.set(mx, this.egoDim); f.set(mn, this.egoDim + H);
    return this.run(this.headL, f);
  }
  act(obs) {
    const out = this.run(this.piL, this.encode(obs));
    return [Math.tanh(out[0]), Math.tanh(out[1])];
  }
}
// Recurrent v5 prior. observeExecution must receive the action after the shield;
// reset at every episode/map boundary. Forward math matches torch.nn.GRUCell.
class RecurrentPolicy extends TDMPCPolicy {
  constructor(spec) {
    super(spec); this.latent = spec.latent; this.memory = spec.memory !== false;
    const dec = b => { const bin = atobAny(b), bytes = Uint8Array.from(bin, c => c.charCodeAt(0)); return new Float32Array(bytes.buffer); };
    this.gru = Object.fromEntries(Object.entries(spec.gru).map(([k, v]) => [k, dec(v)]));
    this.post = spec.posterior.map(l => ({ ...l, W: dec(l.W), b: dec(l.b), ln: l.ln ? l.ln.map(dec) : null }));
    this.reset();
  }
  reset() { this.h = new Float32Array(this.latent); this.previous = [0, 0]; }
  observeExecution(action) { this.previous = Array.from(action); }
  act(obs) {
    const z = this.encode(obs), x = new Float32Array(z.length + 2); x.set(z); x.set(this.previous, z.length);
    const L = this.latent, g = this.gru, ih = new Float64Array(3 * L), hh = new Float64Array(3 * L);
    for (let j = 0; j < 3 * L; j++) {
      ih[j] = g.bias_ih[j]; hh[j] = g.bias_hh[j];
      for (let k = 0; k < x.length; k++) ih[j] += g.weight_ih[j * x.length + k] * x[k];
      if (this.memory) for (let k = 0; k < L; k++) hh[j] += g.weight_hh[j * L + k] * this.h[k];
    }
    const next = new Float32Array(L), sigmoid = a => 1 / (1 + Math.exp(-a));
    for (let j = 0; j < L; j++) {
      const r = sigmoid(ih[j] + hh[j]), gate = sigmoid(ih[L + j] + hh[L + j]);
      const n = Math.tanh(ih[2 * L + j] + r * hh[2 * L + j]);
      next[j] = (1 - gate) * n + gate * (this.memory ? this.h[j] : 0);
    }
    this.h = next;
    const belief = this.run(this.post, next), out = this.run(this.piL, belief);
    return [Math.tanh(out[0]), Math.tanh(out[1])];
  }
}
function makePolicy(spec) { return spec.type === "tdmpc_v5" ? new RecurrentPolicy(spec) : spec.type === "tdmpc" ? new TDMPCPolicy(spec) : spec.type === "set" ? new SetPolicy(spec) : new MLPPolicy(spec); }
function atobAny(s) { return typeof atob === "function" ? atob(s) : Buffer.from(s, "base64").toString("binary"); }

if (typeof module !== "undefined") module.exports = { EgoEnv, ENV, MLPPolicy, SetPolicy, TDMPCPolicy, RecurrentPolicy, makePolicy, defensiveCost };
