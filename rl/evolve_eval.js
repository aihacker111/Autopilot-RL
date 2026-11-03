// Phase 3: evaluate scenario genomes against one driver.
//   node evolve_eval.js --in batch.json --out result.json --driver policy.json|idm|planner --episodes 4 [--shield 1]
// For each genome: how hard the scenario is for the driver, and the behaviour statistics the realism filter needs.
const fs = require("fs");
const { args } = require("./common.js");
const batch = JSON.parse(fs.readFileSync(args.in)), K = +(args.episodes || 4);
let act;
if (args.driver === "planner") { const p = new LookaheadPlanner(); act = (env) => p.plan(env); }
else if (!args.driver || args.driver === "idm") act = (env) => idmAct(env);
else { const pol = makePolicy(JSON.parse(fs.readFileSync(args.driver))); act = (env, obs) => pol.act(obs); }
const PRESETS = ["city", "alley", "roundabout", "highway", "rail", "rural", "pass", "city"];
const out = [];
for (const item of batch) {
  const g = item.genome, eps = [];
  const acc = { vehMin: 0, twoMin: 0, instantStops: 0, cutIns: 0, suddenBrakes: 0, npcColl: 0, wrongSteps: 0, twoSteps: 0, redRuns: 0, minutes: 0 };
  for (let k = 0; k < K; k++) {
    // the real city when its data is present (the same busy places for every genome), else the generated maps
    const seed = (item.seed || 1) * 1000 + k, city = getHcmCity(), preset = city ? "hcm" : PRESETS[(seed + k) % PRESETS.length];
    const mapOpts = city ? { preset, centre: benchCentre(city, seed + k) } : { preset };
    const env = new EgoEnv(Object.assign({ seed, mapOpts, reseedEvery: 0, genome: g, density: g.density }, +args.shield ? { shield: true } : {}));   // --shield 1: as the world-model car drives
    let obs = env.reset(), done = false, info;
    const st0 = { ...env.sim.stats }, t0 = env.sim.t;
    let vehSum = 0, twoSum = 0, n = 0;
    while (!done) {
      const r = env.step(act(env, obs)); obs = r.obs; done = r.done; info = r.info;
      if (n++ % 10 === 0) { vehSum += env.sim.vehicles.length; twoSum += env.sim.vehicles.filter((v) => v.type === "moto" || v.type === "bike").length; }
    }
    const st = env.sim.stats, mins = (env.sim.t - t0) / 60, samples = Math.max(1, Math.ceil(n / 10));
    acc.minutes += mins; acc.vehMin += vehSum / samples * mins; acc.twoMin += twoSum / samples * mins;
    for (const key of ["instantStops", "cutIns", "suddenBrakes", "wrongSteps", "twoSteps"]) acc[key] += (st[key] || 0) - (st0[key] || 0);
    acc.npcColl += st.collisions - st0.collisions; acc.redRuns += st.redRuns.length;
    eps.push({ result: info.result, red: info.red, minGap: info.minGap, time: info.time });
  }
  const m = (k) => eps.filter((e) => e.result === k).length / eps.length;
  const difficulty = eps.reduce((s, e) => s + (e.result === "crash" ? 1 : 0) + (e.red ? 0.5 : 0) + (e.result === "timeout" ? 0.25 : 0) + 0.15 * Math.max(0, Math.min(1, 1 - e.minGap)), 0) / eps.length;
  out.push({ id: item.id, genome: g, difficulty, crash: m("crash"), hit: m("hit"), ok: m("ok"), clean: eps.filter((e) => e.result === "ok" && !e.red).length / eps.length, timeout: m("timeout"), red: eps.filter((e) => e.red).length / eps.length,
    stats: { instantStopsPerVehMin: acc.instantStops / Math.max(1e-6, acc.vehMin), cutInsPerVehMin: acc.cutIns / Math.max(1e-6, acc.vehMin), suddenBrakesPerTwoMin: acc.suddenBrakes / Math.max(1e-6, acc.twoMin),
      npcCollisionsPerMin: acc.npcColl / Math.max(1e-6, acc.minutes), wrongWayShare: acc.wrongSteps / Math.max(1, acc.twoSteps), hitRate: m("hit") } });
}
fs.writeFileSync(args.out, JSON.stringify(out));
console.log(JSON.stringify({ evaluated: out.length }));
