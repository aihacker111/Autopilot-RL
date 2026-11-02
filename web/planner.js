/* ======================================================================
   LOTUS: the look-ahead teacher and a rule-based baseline.

   LookaheadPlanner: every decision it copies the world around the car and
   tries each candidate (target speed x sideways position) for a few seconds,
   several times with different random futures for the motorbikes. It keeps
   the candidate whose worst futures are still good. Slow but strong; its
   decisions are later distilled into a fast neural network.
   It plans only with what the car can see. Where the view is blocked (a
   side road behind a wall, the road beyond a crest or a blind bend) it
   assumes the worst: a motorbike coming out of the hidden road, something
   stopped just out of sight. So it slows down where a careful driver would.

   idmAct: a privileged rule-based driver (car following, stopping for red
   and yellow, yielding to crossing traffic). The baseline every learned
   policy has to beat.
   ====================================================================== */
"use strict";
class LookaheadPlanner {
  constructor(opt = {}) {
    this.vTs = opt.vTs || [0, 3, 6, 9, 12];
    this.lats = opt.lats || [-0.6, 0, 0.6];
    this.M = opt.M || 3;                 // random futures per candidate
    this.H = opt.H || 12;                // decisions of look-ahead (x 0.2 s)
    this.radius = opt.radius || 30;      // metres of world copied around the car
    this.mix = opt.mix ?? 0.6;           // weight of the worst future vs the average
    this.calls = 0;
  }
  /* the copied world as the car knows it: visible road users only, plus phantoms where it cannot see */
  view(env, c) {
    const V = this.cur; if (!V) return;
    c.vehicles = c.vehicles.filter((v) => v.ego || V.vis.has("v" + v.id));
    c.agents = c.agents.filter((a) => V.vis.has("a" + a.id));
    const ce = c.vehicles.find((v) => v.ego);
    for (const ph of V.ph) {
      if (ph.block) {
        const r = ce.route[ph.k], p = c.path(r.j, r.from, r.to), i = Math.max(0, Math.min(p.n - 1, Math.round(ph.s / p.ds)));
        const a = new Agent(c, "stalled", p.X[i] - p.TY[i] * ce.d, p.Y[i] + p.TX[i] * ce.d, Math.atan2(p.TY[i], p.TX[i]), { static: true });
        a.phantom = true; c.agents.push(a); continue;
      }
      const to = ph.to !== ph.from ? ph.to : c.chooseExit(ph.j, ph.from, "moto");
      const p = c.path(ph.j, ph.from, to), i = Math.max(0, Math.min(p.n - 1, Math.round(ph.s / p.ds)));
      const v = new Vehicle(c, "moto", p, (p.DMIN[i] + p.DMAX[i]) / 2);
      v.phantom = true; v.s = ph.s; v.v = 5.5; v.aggr = 0.8; v.comply = 0.3; v.vDes = 9; v.prio = 1; c.place(v); c.vehicles.push(v);
    }
  }
  rollout(env, vT, lat, m) {
    const sim = env.sim, e = env.ego;
    const c = sim.cloneLocal(e.x, e.y, this.radius);
    c.rand.st.a = (sim.rand.st.a + 7919 * (m + 1) + this.calls * 104729) >>> 0;
    this.view(env, c);
    const ce = c.vehicles.find((v) => v.ego);
    ce.ctrl = { vT, lat };
    let score = 0, dist = 0, legRed = new Set();
    for (let k = 0; k < this.H; k++) {
      const p0 = ce.path, leg0 = ce.legs, front0 = ce.s + ce.l2, light0 = p0.signal ? c.light(p0.junc, p0.from) : null;
      for (let q = 0; q < ENV.REPEAT; q++) { c.step(); dist += ce.v * DT; }
      const disc = Math.pow(0.97, k);
      // contact: large penalty, worse the sooner it happens
      for (const u of c.near(ce, 7)) {
        const g = c.boxGap(ce, u);
        if (g < -0.05) { const fx = (u.x - ce.x) * ce.tx + (u.y - ce.y) * ce.ty, closing = (u.vx - ce.vx) * ce.tx + (u.vy - ce.vy) * ce.ty; const dd = Math.hypot(u.x - ce.x, u.y - ce.y) || 1, et = (ce.vx * (u.x - ce.x) + ce.vy * (u.y - ce.y)) / dd, ot = -(u.vx * (u.x - ce.x) + u.vy * (u.y - ce.y)) / dd; if (et > 0.3 && et >= ot) return score - 12 * disc + dist * 0.1; }
        else if (g < 0.5) score -= 0.15 * (1 - g / 0.5) * disc;
      }
      if (p0.signal && !legRed.has(leg0)) {
        const crossed = ce.legs === leg0 ? (front0 <= p0.sStop && ce.s + ce.l2 > p0.sStop) : front0 <= p0.sStop;
        if (crossed) { const L = c.light(p0.junc, p0.from); if (L.c === "R" || (light0 && light0.c === "R")) { legRed.add(leg0); score -= 8 * disc; } }
      }
      const li = Math.min(ce.path.n - 1, Math.round(ce.s / ce.path.ds));
      if (ce.v > ce.path.VMAX[li] + 1.5) score -= 0.05 * (ce.v - ce.path.VMAX[li]) * disc;
    }
    // end state: can we still stop for what is ahead? penalise ending fast and close to something
    for (const u of c.near(ce, 20)) {
      const fx = (u.x - ce.x) * ce.tx + (u.y - ce.y) * ce.ty, fy = (u.x - ce.x) * ce.nx + (u.y - ce.y) * ce.ny;
      if (fx > 0 && Math.abs(fy) < ce.w2 + u.w2 + 0.6) { const need = ce.v * ce.v / (2 * 4.5) + 1.5, gap = fx - ce.l2 - u.l2; if (gap < need) score -= 1.5 * (1 - gap / need); }
    }
    const p = ce.path;
    if (p.signal && ce.s + ce.l2 < p.sStop) { const L = c.light(p.junc, p.from); if (L.c !== "G") { const need = ce.v * ce.v / (2 * 4.5); if (p.sStop - ce.s - ce.l2 < need) score -= 4; } }
    return score + dist * 0.1;
  }
  /* score a candidate: blend of its worst and average futures; stop early once a future is clearly a crash */
  evaluate(env, vT, lat, prevLat) {
    const s = [];
    for (let m = 0; m < this.M; m++) { const v = this.rollout(env, vT, lat, m); s.push(v); if (v < -6) break; }
    const worst = Math.min(...s), mean = s.reduce((a, b) => a + b, 0) / s.length;
    return this.mix * worst + (1 - this.mix) * mean - 0.15 * Math.abs(lat - prevLat) - 0.02 * Math.abs(vT - env.ego.v);
  }
  /* two stages: pick the target speed keeping the current sideways position, then try moving sideways at that speed and one step slower */
  plan(env) {
    this.calls++;
    this.cur = env.planningView ? env.planningView(this.radius) : null;
    const prevLat = env.prevAct ? env.prevAct[1] : 0;
    const latNow = this.lats.reduce((b, l) => Math.abs(l - prevLat) < Math.abs(b - prevLat) ? l : b, this.lats[0]);
    let best = [this.vTs[0], latNow], bestScore = -1e9;
    for (const vT of this.vTs) { const sc = this.evaluate(env, vT, latNow, prevLat); if (sc > bestScore) { bestScore = sc; best = [vT, latNow]; } }
    const vi = this.vTs.indexOf(best[0]), speeds = [best[0]]; if (vi > 0) speeds.push(this.vTs[vi - 1]); if (vi + 1 < this.vTs.length && bestScore < 0) speeds.push(this.vTs[vi + 1]);
    for (const lat of this.lats) { if (lat === latNow) continue; for (const vT of speeds) { const sc = this.evaluate(env, vT, lat, prevLat); if (sc > bestScore) { bestScore = sc; best = [vT, lat]; } } }
    this.last = { score: bestScore, vT: best[0], lat: best[1] };
    return [best[0] / ENV.VMAX_CMD * 2 - 1, best[1]];
  }
}

/* rule-based driver with privileged state (IDM car following, signals and barriers, giving way, going round obstacles) */
function idmAct(env) {
  const sim = env.sim, e = env.ego, p = e.path;
  const i0 = Math.min(p.n - 1, Math.round(e.s / p.ds));
  let v0 = 10; for (let k = 0; k < 30; k += 3) { const i = Math.min(p.n - 1, i0 + k); v0 = Math.min(v0, Math.sqrt(p.VMAX[i] ** 2 + 2 * 2.5 * k * p.ds)); }
  let gap = 60, dv = 0, blocker = null;
  for (const u of sim.near(e, 40)) {
    const dx = u.x - e.x, dy = u.y - e.y, fx = dx * e.tx + dy * e.ty, fy = dx * e.nx + dy * e.ny;
    const fx1 = fx + ((u.vx - e.vx) * e.tx + (u.vy - e.vy) * e.ty) * 1.0, fy1 = fy + ((u.vx - e.vx) * e.nx + (u.vy - e.vy) * e.ny) * 1.0;
    const w = e.w2 + u.w2 + 0.15;
    if ((fx > 0 && Math.abs(fy) < w) || (fx1 > 0 && Math.abs(fy1) < w)) {
      const g = Math.min(fx, Math.max(0, fx1)) - e.l2 - u.l2;
      if (g < gap) { gap = g; dv = e.v - ((u.vx * e.tx + u.vy * e.ty)); blocker = { u, fy }; }
    }
  }
  if (p.signal && e.s + e.l2 < p.sStop + 0.5) {
    const L = sim.light(p.junc, p.from), toStop = p.sStop - e.s - e.l2;
    if (L.c === "R" || (L.c === "Y" && toStop > e.v * e.v / (2 * 4.5))) { if (toStop < gap) { gap = Math.max(0.1, toStop); dv = e.v; } }
  }
  // give way: stop short of the junction while someone with priority would reach our path soon (accepting smaller gaps after waiting)
  const yd = env.yieldAhead ? env.yieldAhead(14) : null;
  if (yd != null) env.idmWait = e.v < 0.3 ? (env.idmWait || 0) + 0.2 : 0; else env.idmWait = 0;
  if (yd != null && yd < gap) {
    const conflict = [p.X[Math.min(p.n - 1, Math.round(p.sIn / p.ds))], p.Y[Math.min(p.n - 1, Math.round(p.sIn / p.ds))]], hurry = env.idmWait > 6 ? 0.45 : 1;
    for (const u of sim.near(e, 40)) {
      if (u.agent || u.ego || u.v < 0.5 || !sim.hasPriority(u, e)) continue;
      const dx = conflict[0] - u.x, dy = conflict[1] - u.y, dist = Math.hypot(dx, dy);
      if ((u.vx * dx + u.vy * dy) / Math.max(1e-3, dist) < 0.3 && dist > 5) continue;   // not heading our way
      if (dist < (u.v * 2.8 + 5) * hurry) { gap = Math.max(0.1, yd); dv = e.v; break; }
    }
  }
  // lateral: hug the right on shared roads, go round something that will not move
  let lat = 0;
  const shared = p.DMINX[i0] === p.DMIN[i0] && p.DMIN[i0] < p.CTR[i0] - 0.5;
  if (shared) lat = 1;
  const still = blocker && (blocker.u.static || (blocker.u.agent && blocker.u.v < 0.3));
  if (still && gap < 16) lat = blocker.fy > 0 ? -1 : 1;
  const s0 = 2, T = 1.2, a = 2.0, b = 3.0;
  const sStar = s0 + Math.max(0, e.v * T + e.v * dv / (2 * Math.sqrt(a * b)));
  const acc = a * (1 - Math.pow(e.v / Math.max(0.5, v0), 4) - Math.pow(sStar / Math.max(0.1, gap), 2));
  let vT = Math.max(0, Math.min(ENV.VMAX_CMD, e.v + acc / 1.6));
  if (still && gap > 0.3 && gap < 16 && Math.abs(lat) > 0) vT = Math.max(vT, gap > 1.5 ? 1.5 : 0.9);   // keep rolling while steering round it (a car cannot move sideways standing still)
  return [vT / ENV.VMAX_CMD * 2 - 1, lat];
}

if (typeof module !== "undefined") module.exports = { LookaheadPlanner, idmAct };
