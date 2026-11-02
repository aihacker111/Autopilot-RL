// Node worker: runs N copies of the environment and talks JSON lines over stdin/stdout.
const path = require("path");
const W = path.join(__dirname, "..", "web");
Object.assign(global, require(path.join(W, "world.js")));
Object.assign(global, require(path.join(W, "osm.js")));
Object.assign(global, require(path.join(W, "sim.js")));
const { EgoEnv, ENV } = require(path.join(W, "env.js"));
const rl = require("readline").createInterface({ input: process.stdin });
let envs = [];
const pack = (a) => Array.from(a, (x) => Math.round(x * 1e4) / 1e4);
const send = (o) => process.stdout.write(JSON.stringify(o) + "\n");
rl.on("line", (line) => {
  const m = JSON.parse(line);
  if (m.cmd === "init") {
    envs = []; for (let i = 0; i < m.n; i++) envs.push(new EgoEnv({ seed: m.seed * 1000 + i, density: m.density, densityRange: m.densityRange, ramp: m.ramp || 0, shaping: m.shaping ?? true, pool: m.pool, pPool: m.pPool }));
    send({ obs_dim: ENV.OBS_DIM, act_dim: ENV.ACT_DIM });
  } else if (m.cmd === "reset") {
    send({ obs: envs.map((e) => pack(e.reset())) });
  } else if (m.cmd === "step") {
    const obs = [], rew = [], done = [], infos = [];
    envs.forEach((e, i) => {
      const r = e.step(m.actions[i]);
      let o = r.obs;
      if (r.done) { r.info.terminal = pack(o); o = e.reset(); }   // last observation of the episode, for bootstrapping on timeouts
      obs.push(pack(o)); rew.push(Math.round(r.r * 1e5) / 1e5); done.push(r.done); infos.push(r.done ? r.info : null);
    });
    send({ obs, rew, done, infos });
  } else if (m.cmd === "close") process.exit(0);
});
