/* ======================================================================
   LOTUS: learned road users. Motorbikes and cars can be driven by a
   neural network instead of the hand-written rules (rl/npc_train.py):
   first it imitates the rules, then it is fine-tuned by multi-agent RL
   (every rider learns to get where it is going quickly without crashing),
   with a pull toward the rules so it stays human-like.

   The network sees, for one vehicle: its speed, its personality (bold,
   patient, law-abiding, preferred speed: the scenario genomes of phase 3
   still shape the population), where it is across the road, the stop line
   and signal ahead, the turn it is making, the bend ahead, how long it has
   waited; and the 10 nearest road users around it. It outputs a target
   speed and a sideways target, every 0.3 s. The simulation still enforces
   physics: acceleration limits, the speed a bend allows, and that no body
   ever moves into another.
   ====================================================================== */
"use strict";
const NPC = { EGO: 20, K: 10, NB: 12, EVERY: 0.3, LAT: 1.5, TYPES: { moto: 1, car: 1, taxi: 1 } };
NPC.OBS = NPC.EGO + NPC.K * NPC.NB;

/* the observation of vehicle v; `list` is sim.scan(v, ...), [lo, hi] its sideways bounds, i0 its path index */
function npcObs(sim, v, list, lo, hi, i0, stopS) {
  const o = new Float32Array(NPC.OBS), p = v.path, two = isTwo(v.type);
  let k = 0;
  o[k++] = v.v / 15; o[k++] = v.vDes / 15; o[k++] = v.aggr; o[k++] = v.patience; o[k++] = v.comply; o[k++] = (v.speedPref - 1) * 2;
  o[k++] = two ? 1 : 0; o[k++] = (v.d - lo) / Math.max(0.5, hi - lo) * 2 - 1; o[k++] = (hi - lo) / 6;
  const toStop = p.sStop - v.s; o[k++] = Math.max(-1, Math.min(1, toStop / 40));
  const L = p.signal || p.rail ? sim.light(p.junc, p.from) : null;
  o[k++] = L && L.c === "G" ? 1 : 0; o[k++] = L && L.c === "Y" ? 1 : 0; o[k++] = L && L.c === "R" ? 1 : 0; o[k++] = L ? Math.min(1, L.left / 30) : 0;
  o[k++] = p.kind === "left" ? 1 : 0; o[k++] = p.kind === "right" ? 1 : 0;
  o[k++] = p.priority === 0 && toStop > -3 && toStop < 40 ? 1 : 0;   // a side road or roundabout entry: give way
  o[k++] = Math.min(1, p.VMAX[Math.min(p.n - 1, i0 + 20)] / 15);   // the bend 10 m ahead
  o[k++] = Math.min(1, (v.wait || 0) / 20); o[k++] = stopS < Infinity ? Math.max(-1, Math.min(1, (stopS - v.s) / 30)) : 1;
  // neighbours: the 10 nearest, in this vehicle's frame
  const nb = list.filter((e) => Math.abs(e.fx0) < 25 && Math.abs(e.fy0) < 12).sort((a, b) => a.fx0 * a.fx0 + a.fy0 * a.fy0 - b.fx0 * b.fx0 - b.fy0 * b.fy0);
  for (let j = 0; j < NPC.K; j++) {
    const off = NPC.EGO + j * NPC.NB; if (j >= nb.length) continue;
    const e = nb[j], u = e.u, rvx = (u.vx || 0) - v.vx, rvy = (u.vy || 0) - v.vy;
    o[off] = e.fx0 / 25; o[off + 1] = e.fy0 / 10;
    o[off + 2] = (rvx * v.tx + rvy * v.ty) / 10; o[off + 3] = (rvx * v.nx + rvy * v.ny) / 10;
    o[off + 4] = (u.ch || 1) * v.ch + (u.sh || 0) * v.sh; o[off + 5] = (u.sh || 0) * v.ch - (u.ch || 1) * v.sh;
    o[off + 6] = Math.min(3, (u.l2 || 0.5) / 2.5); o[off + 7] = Math.min(2, (u.w2 || 0.3) / 1);
    o[off + 8] = u.ego ? 1 : 0; o[off + 9] = u.agent && !u.static ? 1 : 0; o[off + 10] = u.static ? 1 : 0; o[off + 11] = 1;   // present
  }
  return o;
}

/* the network: DeepSets over the neighbours (shared MLP, max + mean pooling), then an MLP; tanh outputs */
class NpcBrain {
  constructor(spec) {
    const dec = (b64) => { const bin = typeof atob === "function" ? atob(b64) : Buffer.from(b64, "base64").toString("binary"), buf = new ArrayBuffer(bin.length), u8 = new Uint8Array(buf); for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i); return new Float32Array(buf); };
    const L = (ls) => ls.map((l) => ({ W: dec(l.W), b: dec(l.b), inp: l.inp, out: l.out }));
    this.phi = L(spec.phi); this.rho = L(spec.rho); this.mean = dec(spec.obsMean); this.std = dec(spec.obsStd);
    this.name = spec.name || "learned traffic"; this.logStd = spec.logStd || [-1.2, -1.2]; this.types = spec.types || NPC.TYPES;
  }
  static mlp(layers, x, last) {
    layers.forEach((l, li) => { const y = new Float32Array(l.out); for (let o = 0; o < l.out; o++) { let s = l.b[o]; const row = o * l.inp; for (let i = 0; i < l.inp; i++) s += l.W[row + i] * x[i]; y[o] = li < layers.length - 1 ? Math.max(0, s) : last(s); } x = y; });
    return x;
  }
  mean2(obs) {   // the mean action (before tanh is applied by the caller)
    const x = new Float32Array(obs.length); for (let i = 0; i < obs.length; i++) x[i] = Math.max(-8, Math.min(8, (obs[i] - this.mean[i]) / this.std[i]));
    const H = this.phi[this.phi.length - 1].out, mx = new Float32Array(H), mn = new Float32Array(H); let c = 0;
    for (let j = 0; j < NPC.K; j++) {
      const off = NPC.EGO + j * NPC.NB; if (obs[off + NPC.NB - 1] < 0.5) continue;
      const h = NpcBrain.mlp(this.phi, x.subarray(off, off + NPC.NB), (s) => Math.max(0, s));
      for (let i = 0; i < H; i++) { mx[i] = Math.max(mx[i], h[i]); mn[i] += h[i]; } c++;
    }
    if (c) for (let i = 0; i < H; i++) mn[i] /= c;
    const z = new Float32Array(NPC.EGO + 2 * H); z.set(x.subarray(0, NPC.EGO)); z.set(mx, NPC.EGO); z.set(mn, NPC.EGO + H);
    return NpcBrain.mlp(this.rho, z, (s) => s);
  }
}
/* the rule-based decision expressed in the network's action space (for imitation and as the anchor during RL) */
function npcLabel(v, vT, dT) { return [Math.max(-1, Math.min(1, vT / (v.T.vDes * 1.5) * 2 - 1)), Math.max(-1, Math.min(1, (dT - v.d) / NPC.LAT))]; }
function npcApply(v, a) { return { vT: (a[0] + 1) / 2 * v.T.vDes * 1.5, dT: v.d + a[1] * NPC.LAT }; }

let NPC_BRAIN = null;
function setNpcBrain(b) { NPC_BRAIN = b; }
function getNpcBrain() { return NPC_BRAIN; }
if (typeof module !== "undefined") module.exports = { NPC, npcObs, NpcBrain, npcLabel, npcApply, setNpcBrain, getNpcBrain };
