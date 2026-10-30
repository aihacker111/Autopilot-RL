// Write simulated trajectories in the same CSV format as real tracks (frame,id,x,y,cls), e.g. to test the
// calibration script or to compare a scenario genome with real footage.
//   node calib/export_tracks.js --seconds 120 --out calib/sim_tracks.csv [--genome genome.json]
const fs = require("fs"), path = require("path");
const { args } = require(path.join(__dirname, "..", "rl", "common.js"));
const genome = args.genome ? JSON.parse(fs.readFileSync(args.genome)) : null;
const sim = new Sim(cityMap(+(args.seed || 5), { nx: 1, ny: 1 }), +(args.seed || 5), +(args.density || 0.8), { genome });
const rows = ["frame,id,x,y,cls"], N = Math.round(+(args.seconds || 120) / DT);
for (let f = 0; f < N; f++) { sim.step(); for (const v of sim.vehicles) rows.push(`${f},${v.id},${v.x.toFixed(2)},${v.y.toFixed(2)},${v.type}`); }
fs.writeFileSync(args.out || "sim_tracks.csv", rows.join("\n"));
const st = sim.stats; console.log(JSON.stringify({ rows: rows.length - 1, sim_suddenBrakes: st.suddenBrakes || 0, sim_cutIns: st.cutIns || 0, sim_wrongShare: (st.wrongSteps || 0) / Math.max(1, st.twoSteps || 1) }));
