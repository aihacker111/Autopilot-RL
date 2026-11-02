// Phase 4: export a generated map as OpenDRIVE 1.4 so CARLA can build the same roads (generate_opendrive_world).
//   node carla/export_xodr.js --seed 7 --preset region --out region7.xodr
// Every link becomes a road made of cubic pieces (paramPoly3) with an elevation profile, and a cross-section of
// lanes: median, driving lanes, motorbike lane, sidewalk. Inside each node, one connecting road per movement follows
// the exact path the 2D core uses (bends, roundabout arcs, U-turns), shifted onto the centres of the lanes it links.
// Coordinates: OpenDRIVE x = sim x, OpenDRIVE y = -(sim y) (the sim's y points south). CARLA flips y again,
// so a CARLA world location (x, y) equals the sim position (x, y).
const fs = require("fs"), path = require("path");
const { args } = require(path.join(__dirname, "..", "rl", "common.js"));

function exportXodr(map) {
  const f = (v) => (Math.abs(v) < 5e-5 ? 0 : v).toFixed(4), z0 = map.zebra[0];
  const odr = ([x, y]) => [x, -y];
  const roads = [], conns = {};
  let nextId = 1;
  // a polyline [[x, y, z], ...] with a heading at every point (sim angles) as a chain of cubic Hermite pieces
  // (paramPoly3, normalised): every piece passes through its two points with the given headings, so the reference line
  // is continuous in position and heading and meets the lanes it links exactly. Elevation is linear per piece.
  const planView = (P, T) => {
    let s = 0, g = "", e = "";
    for (let i = 0; i + 1 < P.length; i++) {
      const [x0, y0] = odr(P[i]), [x1, y1] = odr(P[i + 1]), c = Math.hypot(x1 - x0, y1 - y0); if (c < 1e-3) continue;
      const h = -T[i], h1 = -T[i + 1], ch = Math.cos(h), sh = Math.sin(h);
      // end point and end tangent in the piece's local frame (u along the start heading)
      const ux = (x1 - x0) * ch + (y1 - y0) * sh, vy = -(x1 - x0) * sh + (y1 - y0) * ch, dh = h1 - h;
      const t1u = c * Math.cos(dh), t1v = c * Math.sin(dh);
      const bU = c, cU = 3 * ux - 2 * c - t1u, dU = -2 * ux + c + t1u, bV = 0, cV = 3 * vy - t1v, dV = -2 * vy + t1v;
      let L = 0, pu = 0, pv = 0; for (let k = 1; k <= 16; k++) { const t = k / 16, u = bU * t + cU * t * t + dU * t ** 3, v = bV * t + cV * t * t + dV * t ** 3; L += Math.hypot(u - pu, v - pv); pu = u; pv = v; }
      g += `<geometry s="${f(s)}" x="${f(x0)}" y="${f(y0)}" hdg="${f(h)}" length="${f(L)}"><paramPoly3 aU="0" bU="${f(bU)}" cU="${f(cU)}" dU="${f(dU)}" aV="0" bV="0" cV="${f(cV)}" dV="${f(dV)}" pRange="normalized"/></geometry>`;
      e += `<elevation s="${f(s)}" a="${f(P[i][2])}" b="${f((P[i + 1][2] - P[i][2]) / L)}" c="0" d="0"/>`;
      s += L;
    }
    return { xml: `<planView>${g}</planView><elevationProfile>${e}</elevationProfile>`, len: s };
  };
  // headings of a polyline from central differences, with the two ends given
  const headings = (P, hStart, hEnd) => P.map((q, i) => i === 0 && hStart != null ? hStart : i === P.length - 1 && hEnd != null ? hEnd :
    Math.atan2(P[Math.min(P.length - 1, i + 1)][1] - P[Math.max(0, i - 1)][1], P[Math.min(P.length - 1, i + 1)][0] - P[Math.max(0, i - 1)][0]));
  // cross-section of one direction, from the centre outward
  const section = (L) => {
    const out = [];
    if (L.ow) { const dw = 2 * L.hw / L.lanes; for (let k = 0; k < L.lanes; k++) out.push({ type: "driving", w: dw }); if (L.sw) out.push({ type: "sidewalk", w: L.sw }); return out; }   // one-way: every lane on one side
    if (L.shared) { out.push({ type: "driving", w: L.hw }); }
    else {
      if (L.med > 0) out.push({ type: "median", w: L.med });
      const dw = (L.hw - L.med - (L.bl || 0)) / L.lanes;
      for (let k = 0; k < L.lanes; k++) out.push({ type: "driving", w: dw });
      if (L.bl) out.push({ type: "biking", w: L.bl });
    }
    if (L.sw) out.push({ type: "sidewalk", w: L.sw });
    return out;
  };
  const laneXml = (id, l, mark) => `<lane id="${id}" type="${l.type}" level="false"><link/><width sOffset="0" a="${f(l.w)}" b="0" c="0" d="0"/>${mark ? `<roadMark sOffset="0" type="${mark}" weight="standard" color="white" width="0.12"/>` : ""}</lane>`;
  const lanesXml = (L) => {
    const sec = section(L); let left = "", right = "";
    if (L.ow) {   // lanes only on the side traffic uses, shifted (laneOffset) so the carriageway stays centred on the link
      const side = L.ow > 0 ? -1 : 1, lanes = sec.map((l, k) => laneXml(side * (k + 1), l, l.type === "driving" ? (k === L.lanes - 1 ? "solid" : "broken") : null));
      const sw = L.sw ? `<lane id="${-side}" type="sidewalk" level="false"><link/><width sOffset="0" a="${f(L.sw)}" b="0" c="0" d="0"/></lane>` : "";
      const lr = side < 0 ? { right: lanes.join(""), left: sw } : { left: lanes.slice().reverse().join(""), right: sw };
      return `<lanes><laneOffset s="0" a="${f(side < 0 ? L.hw : -L.hw)}" b="0" c="0" d="0"/><laneSection s="0"><left>${lr.left}</left><center><lane id="0" type="none" level="false"><roadMark sOffset="0" type="solid" weight="standard" color="white" width="0.12"/></lane></center><right>${lr.right}</right></laneSection></lanes>`;
    }
    sec.forEach((l, k) => { const outer = k === sec.length - 1 || sec[k + 1].type !== "driving"; const mark = l.type === "driving" ? (outer ? "solid" : "broken") : null; right += laneXml(-(k + 1), l, mark); });
    for (let k = sec.length - 1; k >= 0; k--) { const l = sec[k], outer = k === sec.length - 1 || sec[k + 1].type !== "driving"; left += laneXml(k + 1, l, l.type === "driving" ? (outer ? "solid" : "broken") : null); }
    const centre = L.med > 0 || L.shared ? "none" : L.lanes > 1 ? "solid solid" : "broken";
    return `<lanes><laneSection s="0"><left>${left}</left><center><lane id="0" type="none" level="false"><roadMark sOffset="0" type="${centre}" weight="standard" color="${L.lanes > 1 && !L.med ? "yellow" : "white"}" width="0.12"/></lane></center><right>${right}</right></laneSection></lanes>`;
  };
  // lane geometry of a link: id of the k-th driving lane from the centre, and its centre offset
  const driveLane = (L, k) => { const med = !L.ow && !L.shared && L.med > 0 ? 1 : 0; return med + k; };
  // offset of the k-th lane centre from the link centreline, toward the driver's right (one-way: lane 1 is the leftmost)
  const laneCentre = (L, k) => { if (L.ow) { const dw = 2 * L.hw / L.lanes; return dw * (k - 0.5) - L.hw; } if (L.shared) return L.hw / 2; const dw = (L.hw - L.med - (L.bl || 0)) / L.lanes; return L.med + dw * (k - 0.5); };
  // ---- roads: each link from the edge of its first junction (plus the zebra offset) to the edge of the second
  const roadOf = {};   // link id -> { id, sA, sB }
  for (const L of map.links) {
    const sA = (L.armA ? L.armA.box : 0) + z0, sB = L.b != null ? L.len - L.armB.box - z0 : L.len;
    if (sB - sA < 1) continue;
    // sample every 2 m; the heading at each sample is the link's smoothed tangent there
    const P = [], T = [], tan = (s) => { const a = linkPoint(L, Math.max(0, s - 1), 0), b = linkPoint(L, Math.min(L.len, s + 1), 0); return Math.atan2(b.y - a.y, b.x - a.x); };
    for (let s = sA; s < sB - 1; s += 2) { const p = linkPoint(L, s, 0); P.push([p.x, p.y, p.z]); T.push(tan(s)); }
    const pe = linkPoint(L, sB, 0); P.push([pe.x, pe.y, pe.z]); T.push(tan(sB));
    const pv = planView(P, T), id = nextId++;
    roadOf[L.id] = { id, sA, sB, len: pv.len, hA: T[0], hB: T[T.length - 1], pA: P[0], pB: P[P.length - 1] };
    const kind = { highway: "motorway", rural: "rural", pass: "rural", alley: "lowSpeed" }[L.kind] || "town";
    roads.push(`<road name="${L.cls}${L.id}" length="${f(pv.len)}" id="${id}" junction="-1"><link><predecessor elementType="junction" elementId="${100000 + L.a}"/>${L.b != null ? `<successor elementType="junction" elementId="${100000 + L.b}"/>` : ""}</link>` +
      `<type s="0" type="${kind}"><speed max="${Math.round(L.vlim * 3.6)}" unit="km/h"/></type>${pv.xml}<lateralProfile/>${lanesXml(L)}</road>`);
  }
  // ---- connecting roads inside every node: the core's own paths, shifted onto lane centres
  for (const n of map.nodes) {
    conns[n.id] = [];
    for (const from of n.order) for (const to of n.order) {
      const A = n.arms[from], B = n.arms[to], kind = turnKind(map, n, A, B);
      if (kind === "uturn" && n.type !== "uturn") continue;
      const LA = map.links[A.link], LB = map.links[B.link], rA = roadOf[LA.id], rB = roadOf[LB.id]; if (!rA || !rB) continue;
      const p = buildPath(map, n, from, to);
      if (!A.inOK || !B.outOK) continue;   // one-way: no movement against it
      if (n.ban && n.ban.has(from + ">" + to)) continue;
      if (n.only && n.only[from] && !n.only[from].has(to)) continue;
      const kIn = kind === "right" ? LA.lanes : 1, kOut = kind === "right" ? LB.lanes : 1;
      const dIn = laneCentre(LA, LA.shared ? 1 : kIn) - laneOffset(A), dOut = laneCentre(LB, LB.shared ? 1 : kOut) - laneOffset(B);
      // sample the path at exact arc lengths (sIn and sOut are where the link roads end), with the lane shift eased in
      // and out (smoothstep) so the connecting road leaves and joins its lanes tangentially
      const at = (s) => { const u = clampN(s / p.ds, 0, p.n - 1), i = Math.min(p.n - 2, Math.floor(u)), r = u - i, m = (A) => A[i] + (A[i + 1] - A[i]) * r;
        let tx = m(p.TX), ty = m(p.TY); const tl = Math.hypot(tx, ty) || 1; return [m(p.X), m(p.Y), m(p.Z), tx / tl, ty / tl]; };
      const P = [], span = p.sOut - p.sIn, nSeg = Math.max(3, Math.ceil(span / 1.5));
      for (let k = 0; k <= nSeg; k++) { const s = p.sIn + span * k / nSeg, t = k / nSeg, e = t * t * (3 - 2 * t), d = dIn + (dOut - dIn) * e, [x, y, z, tx, ty] = at(s); P.push([x - ty * d, y + tx * d, z]); }
      // incoming traffic uses the right lanes of a road that ends at this node, the left lanes of one that starts here
      const inEnd = A.dirLink < 0, outStart = B.dirLink > 0;
      // pin both ends exactly onto the lane centres of the link roads (the core path is snapped to its 0.5 m grid)
      const cIn = laneCentre(LA, LA.shared ? 1 : kIn), cOut = laneCentre(LB, LB.shared ? 1 : kOut);
      // lane centre at a road end: offset from the end point along the road's own end heading (right normal of the
      // reference direction is (-sin, cos) in sim coordinates); right lanes have positive offsets, left lanes negative
      const laneEnd = (r, atEnd, d) => { const q = atEnd ? r.pB : r.pA, h = atEnd ? r.hB : r.hA; return { x: q[0] - Math.sin(h) * d, y: q[1] + Math.cos(h) * d, z: q[2] }; };
      const e0 = laneEnd(rA, inEnd, inEnd ? cIn : -cIn), e1 = laneEnd(rB, !outStart, outStart ? cOut : -cOut);
      const g0 = [e0.x - P[0][0], e0.y - P[0][1]], g1 = [e1.x - P[P.length - 1][0], e1.y - P[P.length - 1][1]];
      for (let j = 0; j < P.length; j++) { const t = j / (P.length - 1), a = (1 - t) ** 3, b = t ** 3; P[j][0] += g0[0] * a + g1[0] * b; P[j][1] += g0[1] * a + g1[1] * b; }
      P[0][2] = e0.z; P[P.length - 1][2] = e1.z;
      // end headings: along the incoming lane (the link road's end heading, reversed for a road that starts here), and
      // along the outgoing lane
      const hIn = inEnd ? rA.hB : rA.hA + Math.PI, hOut = outStart ? rB.hA : rB.hB + Math.PI;
      const pv = planView(P, headings(P, hIn, hOut)), id = nextId++;
      const inLane = inEnd ? -driveLane(LA, LA.shared ? 1 : kIn) : driveLane(LA, LA.shared ? 1 : kIn), outLane = outStart ? -driveLane(LB, LB.shared ? 1 : kOut) : driveLane(LB, LB.shared ? 1 : kOut);
      const lw = (L) => L.ow ? 2 * L.hw / L.lanes : L.shared ? L.hw : (L.hw - L.med - (L.bl || 0)) / L.lanes;
      const w = (lw(LA) + lw(LB)) / 2;
      roads.push(`<road name="c${id}" length="${f(pv.len)}" id="${id}" junction="${100000 + n.id}"><link><predecessor elementType="road" elementId="${rA.id}" contactPoint="${inEnd ? "end" : "start"}"/><successor elementType="road" elementId="${rB.id}" contactPoint="${outStart ? "start" : "end"}"/></link>` +
        `<type s="0" type="town"><speed max="30" unit="km/h"/></type>${pv.xml}<lateralProfile/><lanes><laneSection s="0"><center><lane id="0" type="none" level="false"/></center><right><lane id="-1" type="driving" level="false"><link><predecessor id="${inLane}"/><successor id="${outLane}"/></link><width sOffset="0" a="${f(w)}" b="0" c="0" d="0"/></lane></right></laneSection></lanes></road>`);
      conns[n.id].push(`<connection id="${conns[n.id].length}" incomingRoad="${rA.id}" connectingRoad="${id}" contactPoint="start"><laneLink from="${inLane}" to="-1"/></connection>`);
    }
  }
  const junctions = map.nodes.map((n) => `<junction id="${100000 + n.id}" name="${n.type}_${n.id}">${conns[n.id].join("")}</junction>`).join("");
  const B = map.bounds;
  return `<?xml version="1.0" standalone="yes"?>\n<OpenDRIVE><header revMajor="1" revMinor="4" name="LOTUS_${map.preset}_${map.seed}" version="1" date="${new Date().toISOString()}" north="${-B.y0}" south="${-B.y1}" east="${B.x1}" west="${B.x0}"><geoReference><![CDATA[+proj=tmerc +lat_0=21.0285 +lon_0=105.8542 +k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs]]></geoReference></header>\n${roads.join("\n")}\n${junctions}\n</OpenDRIVE>\n`;
}
if (require.main === module) {
  // --preset hcm with --place "Ngã tư Bảy Hiền" or --at x,y exports a window of the real city
  let centre = null;
  if ((args.preset || "") === "hcm") { const city = getHcmCity(); if (args.at) centre = String(args.at).split(",").map(Number); else if (args.place) { const p = city.findPlace(String(args.place))[0]; if (p) centre = [p.x, p.y]; } }
  const map = cityMap(+(args.seed || 7), { preset: args.preset || "region", centre });
  const xml = exportXodr(map), out = args.out || `${map.preset}${map.seed}.xodr`;
  fs.writeFileSync(out, xml);
  fs.writeFileSync(out.replace(/\.xodr$/, "") + ".map.json", JSON.stringify({ seed: map.seed, preset: map.preset, bounds: map.bounds, nodes: map.nodes.map((j) => ({ id: j.id, x: j.x, y: j.y, z: j.z, type: j.type, arms: j.order })) }));
  console.log(JSON.stringify({ out, preset: map.preset, roads: (xml.match(/<road /g) || []).length, junctions: map.nodes.length, kb: Math.round(xml.length / 1024) }));
}
module.exports = { exportXodr };
