// Learned road users (web/npc.js): rollouts for imitation, multi-agent RL and the realism report.
//   node gen_npc.js --mode bc   --steps 3000 --seed 1 --out runs/npc/data/bc_w0          # the rules drive, every decision recorded
//   node gen_npc.js --mode ppo  --policy npc.json --std -1.2,-1.4 --steps 3000 --out ...  # the network drives, with exploration
//   node gen_npc.js --mode stats [--policy npc.json] --steps 3000 --out ...              # traffic statistics (rules, or the network)
// Each run drives a few windows of the city (or generated maps without the city data), with no self-driving car.
// Rows (float32): observation, pre-tanh action (2), rule label (2), reward, done, truncated, vehicle id, two-wheeler.
const fs = require("fs");
const { args } = require("./common.js");
const mode = args.mode || "bc", steps = +(args.steps || 3000), seed = +(args.seed || 1), out = args.out;
const brain = args.policy ? new NpcBrain(JSON.parse(fs.readFileSync(args.policy))) : null;
const std = args.std ? String(args.std).split(",").map(Number) : null;
const r = rngFrom(seed * 977 + 13), city = getHcmCity();
const D = NPC.OBS, ROW = D + 2 + 2 + 3 + 2;
let rows = [], nRows = 0;
const push = (arr) => { rows.push(arr); nRows++; };
const stats = { minutes: 0, vehMin: 0, collisions: 0, nearMiss: 0, redRuns: 0, instantStops: 0, moving: 0, samples: 0, speedSum: 0, stopped: 0, exited: 0, teleports: 0 };
const episodes = Math.max(1, Math.round(steps / 1500)), per = Math.ceil(steps / episodes);
for (let ep = 0; ep < episodes; ep++) {
  const dens = 0.4 + r() * 0.75, ms = Math.floor(r() * 1e9);
  const map = city ? cityMap(ms, { preset: "hcm" }) : cityMap(ms, { preset: ["city", "alley", "roundabout", "region"][Math.floor(r() * 4)] });
  const sim = new Sim(map, Math.floor(r() * 1e9), dens, { warmup: 200, brain: mode === "bc" ? null : brain });
  if (mode === "ppo" && std) sim.npcStd = std;
  const open = new Map();   // vehicle id -> {row, t, odo} of its last decision, waiting for its reward
  const reward = (v, rec, now) => {
    const dt = Math.max(0.05, now - rec.t), prog = Math.min(1.3, ((v.odo || 0) - rec.odo) / Math.max(1, v.vDes * dt));
    let rw = 0.5 * (prog - 1);   // time costs: being slower than one's own preferred speed is the loss
    if (v.hitT > rec.t) rw -= 4; if (v.nearT > rec.t) rw -= 0.5; if (v.redT > rec.t) rw -= 1.5 * v.comply;
    return rw;
  };
  if (mode !== "stats") sim.npcLog = (v, obs, u, lab) => {
    const prev = open.get(v.id);
    if (prev) prev.row[D + 4] = reward(v, prev, sim.t);
    const row = new Float32Array(ROW); row.set(obs); if (u) { row[D] = u[0]; row[D + 1] = u[1]; } row[D + 2] = lab[0]; row[D + 3] = lab[1];
    row[D + 7] = ep * 1e6 + v.id; row[D + 8] = isTwo(v.type) ? 1 : 0;
    push(row); open.set(v.id, { row, t: sim.t, odo: v.odo || 0, v });
  };
  const s0 = { ...sim.stats }, c0 = sim.stats.collisions, n0 = sim.stats.instantStops || 0, tp0 = sim.stats.teleports || 0;
  let nearN = 0, redN = 0, exitN = 0;
  for (let k = 0; k < per; k++) {
    const before = new Set(sim.vehicles.map((v) => v.id)), nm = sim.stats.nearMiss.length, rr = sim.stats.redRuns.length, ex = sim.stats.exited.length;
    sim.step();
    nearN += Math.max(0, sim.stats.nearMiss.length - nm); redN += Math.max(0, sim.stats.redRuns.length - rr); exitN += Math.max(0, sim.stats.exited.length - ex);
    // vehicles that left the map (or were cleared): their last decision ends the episode
    if (open.size && k % 5 === 0) { const alive = new Set(sim.vehicles.map((v) => v.id)); for (const [id, rec] of open) if (!alive.has(id)) { rec.row[D + 4] = reward(rec.v, rec, sim.t); rec.row[D + 5] = 1; open.delete(id); } }
    if (k % 10 === 0) { const mv = sim.vehicles.filter((v) => !v.ego && NPC.TYPES[v.type]); stats.samples++; stats.moving += mv.length; for (const v of mv) { stats.speedSum += v.v; if (v.v < 0.3) stats.stopped++; } }
  }
  for (const rec of open.values()) { rec.row[D + 4] = reward(rec.v, rec, sim.t); rec.row[D + 6] = 1; }   // still driving: truncated
  const mins = per * DT / 60;
  stats.minutes += mins; stats.vehMin += sim.vehicles.length * mins; stats.collisions += sim.stats.collisions - c0; stats.nearMiss += nearN; stats.redRuns += redN;
  stats.instantStops += (sim.stats.instantStops || 0) - n0; stats.exited += exitN; stats.teleports += (sim.stats.teleports || 0) - tp0;
}
const summary = { mode, policy: args.policy || null, rows: nRows, minutes: +stats.minutes.toFixed(1),
  collisionsPerVehMin: stats.collisions / Math.max(1e-6, stats.vehMin), nearMissPerMin: stats.nearMiss / Math.max(1e-6, stats.minutes), redRunsPerMin: stats.redRuns / Math.max(1e-6, stats.minutes),
  instantStopsPerVehMin: stats.instantStops / Math.max(1e-6, stats.vehMin), meanSpeedKmh: stats.speedSum / Math.max(1, stats.moving) * 3.6, stoppedShare: stats.stopped / Math.max(1, stats.moving),
  flowPerMin: stats.exited / Math.max(1e-6, stats.minutes), teleportsPerMin: stats.teleports / Math.max(1e-6, stats.minutes) };
if (out) {
  fs.mkdirSync(require("path").dirname(out), { recursive: true });
  if (nRows) { const buf = new Float32Array(nRows * ROW); rows.forEach((x, i) => buf.set(x, i * ROW)); fs.writeFileSync(out + ".bin", Buffer.from(buf.buffer)); }
  fs.writeFileSync(out + ".json", JSON.stringify({ rows: nRows, row: ROW, obs_dim: D, summary }));
}
console.log(JSON.stringify(summary));
