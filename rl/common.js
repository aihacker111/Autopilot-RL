// Loads the shared simulation, environment and planner (from ../web) into the global scope for Node scripts.
const path = require("path");
const W = path.join(__dirname, "..", "web");
Object.assign(global, require(path.join(W, "world.js")));
Object.assign(global, require(path.join(W, "osm.js")));
Object.assign(global, require(path.join(W, "sim.js")));
Object.assign(global, require(path.join(W, "env.js")));
Object.assign(global, require(path.join(W, "planner.js")));
const args = {}; for (let i = 2; i < process.argv.length; i++) { const a = process.argv[i]; if (a.startsWith("--")) { const v = process.argv[i + 1]; if (v === undefined || v.startsWith("--")) args[a.slice(2)] = true; else { args[a.slice(2)] = v; i++; } } }
module.exports = { args, W };
