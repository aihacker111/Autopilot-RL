// Fixed evaluation suite: the same scenarios for every driver, spread over every kind of map and three densities.
//   node eval.js --driver planner|idm|random|<policy.json> --n 60 [--shard 0 --shards 4] [--out res.json] [--presets city,pass]
const fs = require("fs");
const { args } = require("./common.js");
const n = +(args.n || 60), shard = +(args.shard || 0), shards = +(args.shards || 1);
const DENS = [0.5, 0.8, 1.05], PRESETS = args.presets ? String(args.presets).split(",") : (getHcmCity() ? ["hcm"] : ["city", "region", "alley", "roundabout", "highway", "rail", "rural", "pass"]);
let act, policy = null;
if (args.driver === "planner") { const p = new LookaheadPlanner(); act = (env) => p.plan(env); }
else if (args.driver === "idm") act = (env) => idmAct(env);
else if (args.driver === "random") { const r = rngFrom(9); act = () => [r() * 2 - 1, r() * 2 - 1]; }
else { policy = makePolicy(JSON.parse(fs.readFileSync(args.driver))); act = (env, obs) => policy.act(obs); }
const out = [];
for (let i = shard; i < n; i += shards) {
  const dens = DENS[i % DENS.length], preset = PRESETS[Math.floor(i / DENS.length) % PRESETS.length];
  const env = new EgoEnv({ seed: 90000 + i, density: dens * (preset === "region" ? 0.8 : 1), mapOpts: { preset }, reseedEvery: 0 });
  if (policy && policy.reset) policy.reset();
  let obs = env.reset(), done = false, info;
  while (!done) { const r = env.step(act(env, obs)); if (policy && policy.observeExecution) policy.observeExecution(env.lastExec); obs = r.obs; done = r.done; info = r.info; }
  out.push({ i, dens, ...info, preset });
}
const S = (k) => out.filter((r) => r.result === k).length / out.length;
const summary = { driver: args.driver, episodes: out.length, ok: S("ok"), crash: S("crash"), red: out.filter((r) => r.red).length / out.length, hit: S("hit"), timeout: S("timeout"),
  time_ok: out.filter((r) => r.result === "ok").reduce((a, r) => a + r.time, 0) / Math.max(1, out.filter((r) => r.result === "ok").length) };
summary.by_preset = {};
for (const pr of PRESETS) { const e = out.filter((r) => r.preset === pr); if (e.length) summary.by_preset[pr] = { n: e.length, ok: e.filter((r) => r.result === "ok").length / e.length, crash: e.filter((r) => r.result === "crash").length / e.length, red: e.filter((r) => r.red).length / e.length, timeout: e.filter((r) => r.result === "timeout").length / e.length }; }
if (args.out) fs.writeFileSync(args.out, JSON.stringify({ summary, episodes: out }));
console.log(JSON.stringify(summary));
