// Phase 4 bridge, simulation side. The 2D core stays the single source of truth: it moves every road user,
// runs the signals, and labels each moment with the teacher's action. CARLA (bridge.py) only renders cameras.
// Protocol: one JSON object per line on stdin/stdout.
//   {cmd:"init", mapSeed, nx, ny, density, genome, teacher, seed}   teacher: policy JSON path, or "planner"
//   {cmd:"reset"}                    -> state
//   {cmd:"step", action:[a0,a1]|null} -> state of the next decision (null = let the teacher drive)
const fs = require("fs"), path = require("path");
require(path.join(__dirname, "..", "rl", "common.js"));
const rl = require("readline").createInterface({ input: process.stdin });
let env = null, teacher = null, obs = null;
const send = (o) => process.stdout.write(JSON.stringify(o) + "\n");
const r4 = (x) => Math.round(x * 1e4) / 1e4;
function state(extra) {
  const e = env.ego, sim = env.sim, p = e.path;
  const label = teacher(env, obs);
  const L = p.signal ? sim.light(p.junc, p.from) : null;
  const lights = [];
  for (const jn of sim.map.juncs) if (jn.type === "signal") for (const a of Object.keys(jn.arms)) lights.push([jn.id, a, sim.light(jn.id, a).c]);
  return Object.assign({
    t: r4(sim.t), ego: { id: e.id, x: r4(e.x), y: r4(e.y), h: r4(e.h), v: r4(e.v), braking: e.braking },
    actors: sim.vehicles.filter((v) => !v.ego).map((v) => [v.id, v.type, r4(v.x), r4(v.y), r4(v.h), v.braking ? 1 : 0]),
    lights, label: label.map(r4),
    // what the camera student may know besides its images: its speed and the route from navigation
    nav: { speed: r4(e.v), waypoints: Array.from(obs.slice(ENV.EGO_DIM, ENV.EGO_DIM + ENV.WP * 2)).map(r4), toGoal: r4(obs[7]) },
    // privileged targets for auxiliary losses (never given to the student as input)
    aux: { light: L ? "GYR".indexOf(L.c) : 3, signal: p.signal ? 1 : 0 },
  }, extra || {});
}
rl.on("line", (line) => {
  const m = JSON.parse(line);
  if (m.cmd === "init") {
    env = new EgoEnv({ seed: m.seed || 1, mapSeed: m.mapSeed, mapOpts: { nx: m.nx || 3, ny: m.ny || 3 }, density: m.density ?? 0.8, genome: m.genome || null, reseedEvery: 0 });
    if (!m.teacher || m.teacher === "planner") { const pl = new LookaheadPlanner(); teacher = (env) => pl.plan(env); }
    else if (m.teacher === "idm") teacher = (env) => idmAct(env);
    else { const pol = makePolicy(JSON.parse(fs.readFileSync(m.teacher))); teacher = (env, o) => pol.act(o); }
    obs = env.reset();
    send({ ok: true, dt: DT * ENV.REPEAT, juncs: env.sim.map.juncs.map((j) => ({ id: j.id, x: j.x, y: j.y, type: j.type, arms: Object.keys(j.arms) })) });
  } else if (m.cmd === "reset") { obs = env.reset(); send(state({ reset: true })); }
  else if (m.cmd === "step") {
    const act = m.action || teacher(env, obs);
    const r = env.step(act); obs = r.obs;
    if (r.done) { const info = r.info; obs = env.reset(); send(state({ done: true, info, reset: true })); }
    else send(state({ done: false }));
  } else if (m.cmd === "close") process.exit(0);
});
