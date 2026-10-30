/* ======================================================================
   LOTUS: the look-ahead teacher and a rule-based baseline.

   LookaheadPlanner: every decision it copies the world around the car and
   tries each candidate (target speed x sideways position) for a few seconds,
   several times with different random futures for the motorbikes. It keeps
   the candidate whose worst futures are still good. Slow but strong; its
   decisions are later distilled into a fast neural network.

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
  rollout(env, vT, lat, m) {
    const sim = env.sim, e = env.ego;
    const c = sim.cloneLocal(e.x, e.y, this.radius);
    c.rand.st.a = (sim.rand.st.a + 7919 * (m + 1) + this.calls * 104729) >>> 0;
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

/* rule-based driver with privileged state (IDM car following + signal + yielding) */
function idmAct(env) {
  const sim = env.sim, e = env.ego, p = e.path;
  const i0 = Math.min(p.n - 1, Math.round(e.s / p.ds));
  let v0 = 10; for (let k = 0; k < 30; k += 3) { const i = Math.min(p.n - 1, i0 + k); v0 = Math.min(v0, Math.sqrt(p.VMAX[i] ** 2 + 2 * 2.5 * k * p.ds)); }
  let gap = 60, dv = 0;
  for (const u of sim.near(e, 40)) {
    const dx = u.x - e.x, dy = u.y - e.y, fx = dx * e.tx + dy * e.ty, fy = dx * e.nx + dy * e.ny;
    const fx1 = fx + ((u.vx - e.vx) * e.tx + (u.vy - e.vy) * e.ty) * 1.0, fy1 = fy + ((u.vx - e.vx) * e.nx + (u.vy - e.vy) * e.ny) * 1.0;
    const w = e.w2 + u.w2 + 0.15;
    if ((fx > 0 && Math.abs(fy) < w) || (fx1 > 0 && Math.abs(fy1) < w)) {
      const g = Math.min(fx, Math.max(0, fx1)) - e.l2 - u.l2;
      if (g < gap) { gap = g; dv = e.v - ((u.vx * e.tx + u.vy * e.ty)); }
    }
  }
  if (p.signal && e.s + e.l2 < p.sStop + 0.5) {
    const L = sim.light(p.junc, p.from), toStop = p.sStop - e.s - e.l2;
    if (L.c === "R" || (L.c === "Y" && toStop > e.v * e.v / (2 * 4.5))) { if (toStop < gap) { gap = Math.max(0.1, toStop); dv = e.v; } }
  }
  const s0 = 2, T = 1.2, a = 2.0, b = 3.0;
  const sStar = s0 + Math.max(0, e.v * T + e.v * dv / (2 * Math.sqrt(a * b)));
  const acc = a * (1 - Math.pow(e.v / Math.max(0.5, v0), 4) - Math.pow(sStar / Math.max(0.1, gap), 2));
  const vT = Math.max(0, Math.min(ENV.VMAX_CMD, e.v + acc / 1.6));
  return [vT / ENV.VMAX_CMD * 2 - 1, 0];
}

if (typeof module !== "undefined") module.exports = { LookaheadPlanner, idmAct };
