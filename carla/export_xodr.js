// Phase 4: export a generated city as OpenDRIVE 1.4 so CARLA can build the same streets (generate_opendrive_world).
//   node carla/export_xodr.js --seed 7 --nx 3 --ny 3 --out city7.xodr
// Coordinates: OpenDRIVE x = sim x, OpenDRIVE y = -(sim y) (the sim's y points south). CARLA flips y again,
// so a CARLA world location (x, y) equals the sim position (x, y).
const fs = require("fs"), path = require("path");
const { args } = require(path.join(__dirname, "..", "rl", "common.js"));
function exportXodr(map) {
  const SW = map.sidewalk, X = [], f = (v) => v.toFixed(4);
  const roads = [], conns = {};
  let nextId = 1;
  const odr = ([x, y]) => [x, -y];
  const hdgOf = (dx, dy) => Math.atan2(-dy, dx);   // sim direction -> OpenDRIVE heading
  const lanesXml = (n, w, right, left) => {
    const lane = (id, type, width) => `<lane id="${id}" type="${type}" level="false"><link/><width sOffset="0" a="${f(width)}" b="0" c="0" d="0"/>${type === "driving" ? `<roadMark sOffset="0" type="${Math.abs(id) === n ? "solid" : "broken"}" weight="standard" color="white" width="0.12"/>` : ""}</lane>`;
    let L = "", R = "";
    if (left) { L += lane(n + 1, "sidewalk", SW); for (let k = n; k >= 1; k--) L += lane(k, "driving", w); }
    if (right) { for (let k = 1; k <= n; k++) R += lane(-k, "driving", w); R += lane(-(n + 1), "sidewalk", SW); }
    return `<lanes><laneSection s="0">${left ? `<left>${L}</left>` : ""}<center><lane id="0" type="none" level="false"><roadMark sOffset="0" type="${n > 1 ? "solid solid" : "broken"}" weight="standard" color="${n > 1 ? "yellow" : "white"}" width="0.12"/></lane></center>${right ? `<right>${R}</right>` : ""}</laneSection></lanes>`;
  };
  // straight roads: every link, plus every road leaving the city. A road runs from its start junction outward.
  const armRoad = {};   // `${junction}:${arm}` -> { id, atStart }  atStart: the junction is at s = 0 of that road
  for (const jn of map.juncs) for (const a of Object.values(jn.arms)) {
    const key = `${jn.id}:${a.id}`; if (armRoad[key]) continue;
    const d = a.dir, p0 = [jn.x + d[0] * a.box, jn.y + d[1] * a.box];
    let p1, other = null;
    if (a.to) { const jb = map.juncs[a.to.j], b = jb.arms[a.to.arm]; p1 = [jb.x - d[0] * b.box, jb.y - d[1] * b.box]; other = { j: jb.id, arm: a.to.arm }; }
    else { const B = map.bounds, far = d[0] > 0 ? B.x1 - jn.x : d[0] < 0 ? jn.x - B.x0 : d[1] > 0 ? B.y1 - jn.y : jn.y - B.y0; p1 = [jn.x + d[0] * far, jn.y + d[1] * far]; }
    const id = nextId++, len = Math.hypot(p1[0] - p0[0], p1[1] - p0[1]), [ox, oy] = odr(p0);
    armRoad[key] = { id, atStart: true, n: a.lanes, w: a.hw / a.lanes };
    if (other) armRoad[`${other.j}:${other.arm}`] = { id, atStart: false, n: a.lanes, w: a.hw / a.lanes };
    roads.push(`<road name="r${id}" length="${f(len)}" id="${id}" junction="-1"><link><predecessor elementType="junction" elementId="${100000 + jn.id}"/>${other ? `<successor elementType="junction" elementId="${100000 + other.j}"/>` : ""}</link>` +
      `<type s="0" type="town"><speed max="50" unit="km/h"/></type><planView><geometry s="0" x="${f(ox)}" y="${f(oy)}" hdg="${f(hdgOf(d[0], d[1]))}" length="${f(len)}"><line/></geometry></planView>` +
      `<elevationProfile/><lateralProfile/>${lanesXml(a.lanes, a.hw / a.lanes, true, true)}</road>`);
  }
  // connecting roads inside each junction: one per lane pair, a cubic Bezier written as paramPoly3
  for (const jn of map.juncs) {
    conns[jn.id] = [];
    for (const A of Object.values(jn.arms)) for (const B of Object.values(jn.arms)) {
      if (A === B) continue;
      const kind = B.id === OPP[A.id] ? "straight" : B.id === LEFT_OF[A.id] ? "left" : "right";
      const inR = armRoad[`${jn.id}:${A.id}`], outR = armRoad[`${jn.id}:${B.id}`];
      const nIn = A.lanes, nOut = B.lanes, pairs = [];
      if (kind === "straight") for (let k = 1; k <= Math.min(nIn, nOut); k++) pairs.push([k, k]);
      else if (kind === "left") pairs.push([1, 1]); else pairs.push([nIn, nOut]);
      for (const [ki, ko] of pairs) {
        const wi = A.hw / nIn, wo = B.hw / nOut, tA = [-A.dir[0], -A.dir[1]], nA = [-tA[1], tA[0]], nB = [-B.dir[1], B.dir[0]];
        const P0 = [jn.x + A.dir[0] * A.box + nA[0] * wi * (ki - 0.5), jn.y + A.dir[1] * A.box + nA[1] * wi * (ki - 0.5)];
        const P3 = [jn.x + B.dir[0] * B.box + nB[0] * wo * (ko - 0.5), jn.y + B.dir[1] * B.box + nB[1] * wo * (ko - 0.5)];
        const k = Math.hypot(P3[0] - P0[0], P3[1] - P0[1]) * (kind === "straight" ? 0.33 : 0.5);
        const P1 = [P0[0] + tA[0] * k, P0[1] + tA[1] * k], P2 = [P3[0] - B.dir[0] * k, P3[1] - B.dir[1] * k];
        const Q = [P0, P1, P2, P3].map(odr), h = Math.atan2(Q[1][1] - Q[0][1], Q[1][0] - Q[0][0]), c = Math.cos(h), s = Math.sin(h);
        const loc = (p) => [(p[0] - Q[0][0]) * c + (p[1] - Q[0][1]) * s, -(p[0] - Q[0][0]) * s + (p[1] - Q[0][1]) * c];
        const [q0, q1, q2, q3] = Q.map(loc);
        const bU = 3 * (q1[0] - q0[0]), cU = 3 * (q0[0] - 2 * q1[0] + q2[0]), dU = q3[0] - q0[0] + 3 * (q1[0] - q2[0]);
        const bV = 3 * (q1[1] - q0[1]), cV = 3 * (q0[1] - 2 * q1[1] + q2[1]), dV = q3[1] - q0[1] + 3 * (q1[1] - q2[1]);
        let len = 0, prev = q0; for (let i = 1; i <= 200; i++) { const t = i / 200, u = bU * t + cU * t * t + dU * t * t * t, v = bV * t + cV * t * t + dV * t * t * t; len += Math.hypot(u - prev[0], v - prev[1]); prev = [u, v]; }
        const id = nextId++;
        // incoming traffic uses the right lanes of a road ending at the junction, the left lanes of a road starting there
        const inLane = inR.atStart ? ki : -ki, outLane = outR.atStart ? -ko : ko;
        roads.push(`<road name="c${id}" length="${f(len)}" id="${id}" junction="${100000 + jn.id}"><link><predecessor elementType="road" elementId="${inR.id}" contactPoint="${inR.atStart ? "start" : "end"}"/><successor elementType="road" elementId="${outR.id}" contactPoint="${outR.atStart ? "start" : "end"}"/></link>` +
          `<type s="0" type="town"><speed max="30" unit="km/h"/></type><planView><geometry s="0" x="${f(Q[0][0])}" y="${f(Q[0][1])}" hdg="${f(h)}" length="${f(len)}"><paramPoly3 aU="0" bU="${f(bU)}" cU="${f(cU)}" dU="${f(dU)}" aV="0" bV="${f(bV)}" cV="${f(cV)}" dV="${f(dV)}" pRange="normalized"/></geometry></planView>` +
          `<elevationProfile/><lateralProfile/><lanes><laneSection s="0"><center><lane id="0" type="none" level="false"/></center><right><lane id="-1" type="driving" level="false"><link><predecessor id="${inLane}"/><successor id="${outLane}"/></link><width sOffset="0" a="${f((wi + wo) / 2)}" b="0" c="0" d="0"/></lane></right></laneSection></lanes></road>`);
        conns[jn.id].push(`<connection id="${conns[jn.id].length}" incomingRoad="${inR.id}" connectingRoad="${id}" contactPoint="start"><laneLink from="${inLane}" to="-1"/></connection>`);
      }
    }
  }
  const junctions = map.juncs.map((jn) => `<junction id="${100000 + jn.id}" name="${jn.type === "signal" ? "signal" : "yield"}_${jn.id}">${conns[jn.id].join("")}</junction>`).join("");
  const B = map.bounds;
  return `<?xml version="1.0" standalone="yes"?>\n<OpenDRIVE><header revMajor="1" revMinor="4" name="LOTUS_${map.seed}" version="1" date="${new Date().toISOString()}" north="${-B.y0}" south="${-B.y1}" east="${B.x1}" west="${B.x0}"><geoReference><![CDATA[+proj=tmerc +lat_0=21.0285 +lon_0=105.8542 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs]]></geoReference></header>\n${roads.join("\n")}\n${junctions}\n</OpenDRIVE>\n`;
}
if (require.main === module) {
  const map = cityMap(+(args.seed || 7), { nx: +(args.nx || 3), ny: +(args.ny || 3) });
  const xml = exportXodr(map), out = args.out || `city${map.seed}.xodr`;
  fs.writeFileSync(out, xml);
  fs.writeFileSync(out.replace(/\.xodr$/, "") + ".map.json", JSON.stringify({ seed: map.seed, nx: map.nx, ny: map.ny, bounds: map.bounds, juncs: map.juncs.map((j) => ({ id: j.id, x: j.x, y: j.y, type: j.type, arms: Object.keys(j.arms) })) }));
  console.log(JSON.stringify({ out, roads: (xml.match(/<road /g) || []).length, junctions: map.juncs.length, kb: Math.round(xml.length / 1024) }));
}
module.exports = { exportXodr };
