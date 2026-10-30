// Fixed evaluation suite: the same scenarios for every driver.
//   node eval.js --driver planner|idm|random|<policy.json> --n 60 [--shard 0 --shards 4] [--out res.json]
const fs = require("fs");
const { args } = require("./common.js");
const n = +(args.n || 60), shard = +(args.shard || 0), shards = +(args.shards || 1);
const DENS = [0.5, 0.8, 1.05], SIZES = [[1, 1], [2, 2], [3, 3], [2, 3]];
let act;
if (args.driver === "planner") { const p = new LookaheadPlanner(); act = (env) => p.plan(env); }
else if (args.driver === "idm") act = (env) => idmAct(env);
else if (args.driver === "random") { const r = rngFrom(9); act = () => [r() * 2 - 1, r() * 2 - 1]; }
else { const pol = makePolicy(JSON.parse(fs.readFileSync(args.driver))); act = (env, obs) => pol.act(obs); }
const out = [];
for (let i = shard; i < n; i += shards) {
  const dens = DENS[i % DENS.length], [nx, ny] = SIZES[Math.floor(i / DENS.length) % SIZES.length];
  const env = new EgoEnv({ seed: 90000 + i, density: dens, mapOpts: { nx, ny }, reseedEvery: 0 });
  let obs = env.reset(), done = false, info;
  while (!done) { const r = env.step(act(env, obs)); obs = r.obs; done = r.done; info = r.info; }
  out.push({ i, dens, size: `${nx}x${ny}`, ...info });
}
const S = (k) => out.filter((r) => r.result === k).length / out.length;
const summary = { driver: args.driver, episodes: out.length, ok: S("ok"), crash: S("crash"), red: out.filter((r) => r.red).length / out.length, hit: S("hit"), timeout: S("timeout"),
  time_ok: out.filter((r) => r.result === "ok").reduce((a, r) => a + r.time, 0) / Math.max(1, out.filter((r) => r.result === "ok").length) };
if (args.out) fs.writeFileSync(args.out, JSON.stringify({ summary, episodes: out }));
console.log(JSON.stringify(summary));
