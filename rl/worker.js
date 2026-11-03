// Node worker: runs N copies of the environment and talks JSON lines over stdin/stdout.
const path = require("path");
const W = path.join(__dirname, "..", "web");
Object.assign(global, require(path.join(W, "world.js")));
Object.assign(global, require(path.join(W, "osm.js")));
Object.assign(global, require(path.join(W, "npc.js")));
if (process.env.LOTUS_NPC) { try { setNpcBrain(new NpcBrain(JSON.parse(require("fs").readFileSync(process.env.LOTUS_NPC, "utf8")))); } catch (e) { process.stderr.write("LOTUS_NPC: " + e.message + "\n"); } }   // learned traffic (rl/npc_train.py)

Object.assign(global, require(path.join(W, "sim.js")));
const { EgoEnv, ENV } = require(path.join(W, "env.js"));
const rl = require("readline").createInterface({ input: process.stdin });
let envs = [];
// "showcase": env 0 of the first training worker starts every episode from a fresh, seeded environment and records the actions it
// is given, so the monitor page can rebuild exactly that episode (same map window, same traffic, same code) and replay it
let show = null, pool = null, pPool = 0;
function showEnv(k) {
  const seed = show.base + k, r = rngFrom(seed * 31 + 5), opts = Object.assign({}, show.opts, { seed });
  if (pool && pool.length && r() < pPool) opts.genome = pool[Math.floor(r() * pool.length)];   // an evolved scenario, recorded with the episode
  return new EgoEnv(opts);
}
// world-model extras ("aux"): per decision, contact (any collision), how the car moved in its own frame, and the scenario genome
let G = null;   // { keys, lo, hi, def } from tdmpc2.py
const gvec = (e) => { if (!G) return []; const g = e.sim && e.sim.genome; return G.keys.map((k, i) => { const v = g && g[k] != null ? g[k] : G.def[i]; return Math.round(((v - G.lo[i]) / (G.hi[i] - G.lo[i]) * 2 - 1) * 1e3) / 1e3; }); };
const pack = (a) => Array.from(a, (x) => Math.round(x * 1e4) / 1e4);
const send = (o) => process.stdout.write(JSON.stringify(o) + "\n");
rl.on("line", (line) => {
  const m = JSON.parse(line);
  if (m.cmd === "init") {
    envs = []; G = m.genomeSpace || null;
    const common = { shaping: m.shaping ?? true, reward: m.reward, defensive: m.defensive };
    if (m.specs) for (const sp of m.specs) envs.push(new EgoEnv(Object.assign({ seed: sp.seed, density: sp.density, mapOpts: { preset: sp.preset || (getHcmCity() ? "hcm" : "city") }, reseedEvery: 0 }, common)));   // a fixed evaluation suite (as eval.js)
    else for (let i = 0; i < m.n; i++) envs.push(new EgoEnv(Object.assign({ seed: m.seed * 1000 + i, density: m.density, densityRange: m.densityRange, ramp: m.ramp || 0, pool: m.pool, pPool: m.pPool, reseedEvery: m.reseedEvery }, common)));
    if (m.journey) {   // a test drive: one environment per stretch, at a named place, with a given traffic genome; every action recorded
      const city = getHcmCity();
      envs = m.journey.map((sp) => {
        let c = sp.centre;
        if (!c && city && sp.place === "rail") { for (const k of Object.keys(city.index.tiles).sort()) { const t = city.tile(k), j = t && t.J.find((q) => q.t === "rail"); if (j) { c = [j.x, j.y]; break; } } }
        else if (!c && city) { const p = city.findPlace(sp.place)[0]; if (p) c = [p.x, p.y]; }
        if (!c && city) c = city.randomCentre(rngFrom(sp.seed));
        const opts = Object.assign({ seed: sp.seed, mapOpts: { preset: city ? "hcm" : "city", centre: c, half: sp.half || 900 }, reseedEvery: 0, routeM: sp.routeM || 1000, routeLegs: 30, maxTime: sp.maxTime || 900 },
          sp.genome ? { genome: sp.genome, density: sp.genome.density } : {}, common);
        const e = new EgoEnv(opts); e.rec = { opts, centre: c, acts: [], chk: [] }; return e;
      });
    }
    if (m.showcase && !m.specs && envs.length) { show = { base: (m.seed * 1000 + 777) * 1000, k: 0, opts: Object.assign({ densityRange: m.densityRange, reseedEvery: 0 }, common), acts: [], chk: [] }; envs[0] = showEnv(0); }
    send({ obs_dim: ENV.OBS_DIM, act_dim: ENV.ACT_DIM });
  } else if (m.cmd === "pool") {   // evolved scenario genomes: mixed into every environment's next map (and the next showcase episode)
    pool = m.pool; pPool = m.p;
    for (const e of envs) { e.pool = pool; e.pPool = pPool; }
    send({ ok: true });
  } else if (m.cmd === "reset") {
    send({ obs: envs.map((e) => pack(e.reset())), ...(G ? { gen: envs.map(gvec) } : {}) });
  } else if (m.cmd === "step") {
    const obs = [], rew = [], done = [], infos = [], aux = [], gen = [];
    envs.forEach((e, i) => {
      if (e.finished) { obs.push(e.lastObs); rew.push(0); done.push(false); infos.push(null); if (G) { aux.push([0, 0, 0, 0]); gen.push(gvec(e)); } return; }   // a finished test-drive stretch waits for the others
      const x0 = e.ego.x, y0 = e.ego.y, h0 = e.ego.h, c0 = Math.cos(h0), s0 = Math.sin(h0);
      const r = e.step(m.actions[i]);
      let o = r.obs;
      if (G) {   // the move in the frame the car had before it, and whether this decision ended in contact
        const dx = e.ego.x - x0, dy = e.ego.y - y0; let dh = e.ego.h - h0; while (dh > Math.PI) dh -= 2 * Math.PI; while (dh < -Math.PI) dh += 2 * Math.PI;
        aux.push([r.done && (r.info.result === "crash" || r.info.result === "hit") ? 1 : 0, Math.round((dx * c0 + dy * s0) * 1e3) / 1e3, Math.round((-dx * s0 + dy * c0) * 1e3) / 1e3, Math.round(dh * 1e4) / 1e4]);
      }
      if (show && i === 0) { show.acts.push(m.actions[i]); if (show.acts.length % 10 === 0 && e.ego) show.chk.push([show.acts.length, Math.round(e.ego.x * 100) / 100, Math.round(e.ego.y * 100) / 100]); }
      if (e.rec) { e.rec.acts.push(m.actions[i]); if (e.rec.acts.length % 10 === 0 && e.ego) e.rec.chk.push([e.rec.acts.length, Math.round(e.ego.x * 100) / 100, Math.round(e.ego.y * 100) / 100]); }
      if (r.done && e.rec) {   // a test-drive stretch is over: report it for the replay, then wait
        r.info.replay = { seed: e.rec.opts.seed, opts: e.rec.opts, centre: e.rec.centre, npc: !!process.env.LOTUS_NPC, actions: e.rec.acts, check: e.rec.chk };
        e.finished = true; e.lastObs = pack(o);
        obs.push(e.lastObs); rew.push(Math.round(r.r * 1e5) / 1e5); done.push(true); infos.push(r.info); if (G) gen.push(gvec(e));
        return;
      }
      if (r.done) {
        r.info.terminal = pack(o);   // last observation of the episode, for bootstrapping on timeouts
        if (show && i === 0) {
          const c = e.sim.map.centre || null;
          r.info.replay = { seed: show.base + show.k, opts: Object.assign({}, show.opts, e.genome ? { genome: e.genome } : {}), centre: c, npc: !!process.env.LOTUS_NPC, actions: show.acts, check: show.chk };
          show.k++; show.acts = []; show.chk = []; envs[0] = showEnv(show.k); o = envs[0].reset();
        } else o = e.reset();
      }
      obs.push(pack(o)); rew.push(Math.round(r.r * 1e5) / 1e5); done.push(r.done); infos.push(r.done ? r.info : null);
      if (G) gen.push(gvec(envs[i]));   // the genome of the scenario the *next* observation belongs to
    });
    send({ obs, rew, done, infos, ...(G ? { aux, gen } : {}) });
  } else if (m.cmd === "close") process.exit(0);
});
