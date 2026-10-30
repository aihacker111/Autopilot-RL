// DAgger data: the look-ahead teacher labels every state; who actually drives is mixed (teacher with probability beta, else the student).
//   node gen_dagger.js --decisions 5000 --seed 1 --out data/r0_w1 [--student student.json --beta 0.5]
const fs = require("fs");
const { args } = require("./common.js");
const N = +(args.decisions || 2000), seed = +(args.seed || 1), beta = args.student ? +(args.beta ?? 0.5) : 1;
const student = args.student ? makePolicy(JSON.parse(fs.readFileSync(args.student))) : null;
const planner = new LookaheadPlanner({ M: +(args.M || 3), H: +(args.H || 12) });
const env = new EgoEnv({ seed: seed * 7 + 3, ramp: 0, densityRange: [0.4, 1.1] });
const rnd = rngFrom(seed * 31 + 5);
const rowLen = ENV.OBS_DIM + ENV.ACT_DIM, buf = new Float32Array(N * rowLen);
const results = [];
let obs = env.reset(), n = 0, t0 = Date.now();
while (n < N) {
  const label = planner.plan(env);
  buf.set(obs, n * rowLen); buf[n * rowLen + ENV.OBS_DIM] = label[0]; buf[n * rowLen + ENV.OBS_DIM + 1] = label[1]; n++;
  const act = student && rnd() > beta ? student.act(obs) : label;
  const r = env.step(act); obs = r.obs;
  if (r.done) { results.push(r.info); obs = env.reset(); }
  if (n % 500 === 0) process.stderr.write(`[w${seed}] ${n}/${N} ${((Date.now() - t0) / n).toFixed(0)} ms/decision\n`);
}
fs.mkdirSync(require("path").dirname(args.out), { recursive: true });
fs.writeFileSync(args.out + ".bin", Buffer.from(buf.buffer));
fs.writeFileSync(args.out + ".json", JSON.stringify({ rows: N, obs_dim: ENV.OBS_DIM, act_dim: ENV.ACT_DIM, beta, results }));
console.log(JSON.stringify({ out: args.out, rows: N, episodes: results.length }));
