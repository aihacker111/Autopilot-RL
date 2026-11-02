// Demonstrations for the world model (tdmpc2.py): the look-ahead teacher drives, and every decision is stored with what
// followed: observation, action, reward, done, terminal (done but not a timeout), next observation, and for the physics heads:
// contact (the decision ended in a collision), the car's move in its own frame (dx, dy, dh).
//   --reward v2 --defensive 1: the world-model driver's reward and the defensive teacher
//   node gen_demos.js --decisions 5000 --seed 1 --out runs/tdmpc2/demos/demo_w0
const fs = require("fs");
const { args } = require("./common.js");
const N = +(args.decisions || 2000), seed = +(args.seed || 1);
const planner = new LookaheadPlanner({ M: +(args.M || 3), H: +(args.H || 12), defensive: !!+(args.defensive || 0) });
const env = new EgoEnv({ seed: seed * 7 + 3, densityRange: [0.4, 1.1], reward: args.reward, reseedEvery: +(args.reseed || 3000) });
const D = ENV.OBS_DIM, row = 2 * D + ENV.ACT_DIM + 3 + 4, buf = new Float32Array(N * row);
let obs = env.reset(), n = 0, eps = 0;
while (n < N) {
  const x0 = env.ego.x, y0 = env.ego.y, h0 = env.ego.h;
  const act = planner.plan(env), r = env.step(act), o = n * row;
  buf.set(obs, o); buf[o + D] = act[0]; buf[o + D + 1] = act[1]; buf[o + D + 2] = r.r; buf[o + D + 3] = r.done ? 1 : 0;
  buf[o + D + 4] = r.done && r.info.result !== "timeout" ? 1 : 0; buf.set(r.obs, o + D + 5);
  const dx = env.ego.x - x0, dy = env.ego.y - y0; let dh = env.ego.h - h0; while (dh > Math.PI) dh -= 2 * Math.PI; while (dh < -Math.PI) dh += 2 * Math.PI;
  buf[o + 2 * D + 5] = r.done && (r.info.result === "crash" || r.info.result === "hit") ? 1 : 0;
  buf[o + 2 * D + 6] = dx * Math.cos(h0) + dy * Math.sin(h0); buf[o + 2 * D + 7] = -dx * Math.sin(h0) + dy * Math.cos(h0); buf[o + 2 * D + 8] = dh; n++;
  obs = r.obs; if (r.done) { obs = env.reset(); eps++; }
}
fs.mkdirSync(require("path").dirname(args.out), { recursive: true });
fs.writeFileSync(args.out + ".bin", Buffer.from(buf.buffer));
fs.writeFileSync(args.out + ".json", JSON.stringify({ rows: N, obs_dim: D, act_dim: ENV.ACT_DIM, episodes: eps }));
console.log(JSON.stringify({ out: args.out, rows: N, episodes: eps }));
