/* ======================================================================
   LOTUS: app loop, camera, HUD, minimap, inspector and the self-driving car.
   ====================================================================== */
(() => {
  const $ = (id) => document.getElementById(id);
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const ppm = Math.round(Math.min(14, Math.max(10, 7 * dpr)));
  let citySeed = 7, preset = "region", map, layers, sim, env, city = null, cityOv = null, centre = null, loadingMap = false;
  let qMap = null; try { qMap = new URLSearchParams(location.search).get("map"); if (qMap && PRESET_NAMES.includes(qMap)) preset = qMap; } catch (e) {}
  const AV = typeof AV_DATA !== "undefined" ? AV_DATA : null;
  // learned motorbikes and cars (rl/npc_train.py), when the page was built with them
  const NPCB = typeof NPC_DATA !== "undefined" ? new NpcBrain(NPC_DATA) : null;
  let learnedTraffic = false;
  if (NPCB) { $("trafficBox").hidden = false; $("trafficSep").hidden = false; }
  const setTraffic = (on) => { learnedTraffic = on && !!NPCB; setNpcBrain(learnedTraffic ? NPCB : null); if (sim) sim.brain = learnedTraffic ? NPCB : null; $("trRules").setAttribute("aria-pressed", String(!learnedTraffic)); $("trLearned").setAttribute("aria-pressed", String(learnedTraffic)); };
  $("trRules").onclick = () => setTraffic(false); $("trLearned").onclick = () => setTraffic(true);

  /* ---------- drivers for the self-driving car */
  const planner = new LookaheadPlanner({ vTs: [0, 4, 8, 12], lats: [-0.6, 0, 0.6], M: 2, H: 10 });
  const drivers = [{ id: "planner", short: "Teacher", name: "look-ahead teacher", desc: "Every 0.2 s it copies the world around the car, tries 12 speed and lane-offset plans over the next 2 s in 2 random futures, and picks the plan that is safest in the worst case.", act: (env) => planner.plan(env) },
    { id: "idm", short: "Rules", name: "rule-based", desc: "Hand-written driver: follows the vehicle ahead (IDM) and stops for red and amber. The baseline every learned driver has to beat.", act: (env) => idmAct(env) }];
  if (AV) for (const c of AV.ckpts) { const pol = makePolicy(c); drivers.push({ id: "nn" + c.steps, short: c.short, name: c.name || c.short, desc: c.desc, act: (env) => pol.act(curObs), nn: true, tdmpc: c.type === "tdmpc" }); }
  let drv = 0, curObs = null, lastAct = [0, 0], tally = null, crashMark = null, seeing = false;
  let replay = null, replayBusy = false;   // a recorded TD-MPC2 training episode being replayed (monitor page)
  let pickId = null, epOrder = [];          // an episode chosen in the list (null: the newest training episodes, in turn)
  const seenEp = new Set(), seenDrive = new Set(); let journey = null;   // journey: the test drive (rl/drive_test.py)
  try { if (new URLSearchParams(location.search).get("driver") === "latest" && drivers.length > 2) drv = drivers.length - 1; } catch (e) {}   // training monitor: start on the newest network

  /* Ho Chi Minh City: the index and an overview are fetched once; each window fetches the tiles it needs */
  async function loadCityData() {
    try {
      const r = await fetch("hcm/index.json"); if (!r.ok) throw new Error(r.status);
      city = new HcmCity(await r.json()); setHcmCity(city);
      fetch("hcm/overview.json").then((q) => q.json()).then((o) => { cityOv = o; mini.dirty = true; }).catch(() => {});
      return true;
    } catch (e) { const o = $("optHcm"); if (o) { o.disabled = true; o.textContent = "Ho Chi Minh City (needs a web server)"; } return false; }
  }
  async function openPlace(x, y, seed) {
    if (!city || loadingMap) return; loadingMap = true; $("loading").hidden = false;
    try { await city.ensure(x, y, 700); if (drivers[drv] && drivers[drv].replay) return; centre = [x, y]; preset = "hcm"; $("preset").value = "hcm"; newCity(seed ?? citySeed); } finally { loadingMap = false; $("loading").hidden = true; }
  }
  async function randomPlace(seed) { if (!city) return; const r = rngFrom(seed * 7 + 1); const keys = Object.keys(city.index.tiles).filter((k) => city.index.tiles[k].j > 400), k = keys[Math.floor(r() * keys.length)];
    const [i, j] = k.split("_").map(Number); await city.ensure((i + 0.5) * city.T, (j + 0.5) * city.T, city.T / 2); const c = city.randomCentre(r); await openPlace(c[0], c[1], seed); }
  function newCity(seed) {
    citySeed = seed;
    map = preset === "hcm" ? cityMap(seed, { preset, centre }) : cityMap(seed, { preset });
    const isReal = preset === "hcm"; $("searchBox").hidden = !isReal; $("attrib").hidden = !isReal; $("miniTabs").hidden = !isReal;
    if (isReal) {
      const d = new Date(map.city.stamp);
      $("attrib").textContent = `Map data © OpenStreetMap contributors (ODbL), as of ${d.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" })}. Roads, one-way streets, turn bans, signals, bridges and rail are real; traffic is simulated.`;
      $("brandP").textContent = "Ho Chi Minh City from OpenStreetMap: real streets and alleys, one-way streets, turn bans, signals, flyovers, the railway and Metro line 1. The car with the teal outline drives itself.";
    }
    layers = new CityLayers(map, seed * 3 + 1, ppm, 80, W < 680 ? 10 : 16);
    sim = new Sim(map, seed, +$("density").value * (preset === "region" ? 0.8 : 1), { warmup: preset === "hcm" ? 150 : 400 });
    env = new EgoEnv({ sim, reseedEvery: 0, seed: seed + 100 });
    curObs = env.reset();
    const real = map.nodes.filter((j) => j.type !== "bend"), sig = real.filter((j) => j.type === "signal").length;
    const named = preset === "hcm" ? (map.nodes.filter((j) => j.name).sort((a, b) => Math.hypot(a.x - map.centre[0], a.y - map.centre[1]) - Math.hypot(b.x - map.centre[0], b.y - map.centre[1]))[0] || {}).name : null;
    $("cityTag").lastChild.nodeValue = preset === "hcm" ? `${named || "HCMC"} · ${real.length} junctions · ${sig} signals` : `${map.name} #${seed} · ${real.length} junctions · ${sig} signals`;
    tally = { ok: 0, crash: 0, red: 0, timeout: 0, hit: 0, n: 0, tOk: 0 };
    updateTally(); speedHist.length = 0; mini.dirty = true;
    if (env.ego) { cam.x = env.ego.x; cam.y = env.ego.y; }
  }

  const cv = $("cv"), g = cv.getContext("2d"), stage = $("stage");
  let W = stage.clientWidth, H = stage.clientHeight;
  const cam = { x: 0, y: 0, z: 14 };
  const fitZoom = () => Math.min(W, H) / (W < 680 ? 46 : 70);
  function resize() { W = stage.clientWidth; H = stage.clientHeight; cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
  resize(); cam.z = fitZoom();
  window.addEventListener("resize", () => { const f = cam.z / fitZoom(); resize(); cam.z = fitZoom() * f; drawCurve(); mini.dirty = true; });

  let speed = 1, paused = false, acc = 0, last = performance.now(), sel = null, follow = false, followEgo = true;
  const speedHist = []; let nextSample = 0;
  const curve = $("curve"), cg = curve ? curve.getContext("2d") : null; let curveHover = null;
  const mini = { cv: $("mini"), dirty: true, base: null };

  /* ---------- controls */
  const playIcon = $("playIcon");
  function setPaused(p) { paused = p; playIcon.setAttribute("d", p ? "M4 2l10 6-10 6z" : "M4 2h3v12H4zM9 2h3v12H9z"); $("play").setAttribute("aria-label", p ? "Resume" : "Pause"); }
  $("play").onclick = () => setPaused(!paused);
  const speedBtns = document.querySelectorAll("#speedSeg button");
  speedBtns.forEach((b) => b.onclick = () => setSpeed(+b.dataset.speed));
  function setSpeed(s) { speed = s; speedBtns.forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.speed === s))); }
  const densNames = [[0.45, "Light"], [0.95, "Normal"], [1.15, "Busy"], [9, "Rush hour"]];
  function setDensity(d) { if (sim) sim.density = d; $("densName").textContent = densNames.find(([t]) => d <= t)[1]; }
  $("density").oninput = (e) => setDensity(+e.target.value);
  $("fit").onclick = () => { stopFollowing(); const B = map.bounds; animateCam((B.x0 + B.x1) / 2, (B.y0 + B.y1) / 2, Math.min(W / (B.x1 - B.x0), H / (B.y1 - B.y0)) * 1.02); };
  function leaveReplay() { if (drivers[drv] && drivers[drv].replay) { replay = null; drv = 0; renderDrivers(); $("avDesc").textContent = drivers[0].desc; $("epPick").hidden = true; $("driveBar").hidden = true; } }
  $("newCity").onclick = () => { leaveReplay(); if (preset === "hcm") randomPlace(citySeed + 1); else newCity(citySeed + 1); };
  $("preset").value = preset;
  $("preset").onchange = (e) => { leaveReplay(); if (e.target.value === "hcm") { randomPlace(citySeed); return; } preset = e.target.value; newCity(citySeed); const B = map.bounds; if (!followEgo) animateCam((B.x0 + B.x1) / 2, (B.y0 + B.y1) / 2, cam.z); };
  // place search (accents optional): junctions first
  const placeRes = $("placeRes");
  $("placeQ").oninput = (e) => {
    placeRes.innerHTML = ""; if (!city) return;
    const seen = []; for (const p of city.findPlace(e.target.value)) {
      if (seen.some((q) => q.name === p.name && Math.hypot(q.x - p.x, q.y - p.y) < 250)) continue; seen.push(p);
      const b = document.createElement("button"); b.type = "button"; b.setAttribute("role", "option");
      b.innerHTML = `<b></b><span></span>`; b.firstChild.textContent = p.name; b.lastChild.textContent = { junction: "junction", station: "station", stop: "stop", suburb: "area", quarter: "area", neighbourhood: "area", aerodrome: "airport" }[p.kind] || "place";
      b.onclick = () => { leaveReplay(); placeRes.innerHTML = ""; $("placeQ").value = p.name; openPlace(p.x, p.y); };
      placeRes.appendChild(b);
    }
  };
  $("placeQ").onkeydown = (e) => { if (e.key === "Enter" && placeRes.firstChild) placeRes.firstChild.click(); if (e.key === "Escape") placeRes.innerHTML = ""; };
  let miniCity = false;
  const setMiniTab = (c) => { miniCity = c; $("tabArea").setAttribute("aria-pressed", String(!c)); $("tabCity").setAttribute("aria-pressed", String(c)); mini.dirty = true; };
  $("tabArea").onclick = () => setMiniTab(false); $("tabCity").onclick = () => setMiniTab(true);
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT") return;
    if (e.code === "Space") { e.preventDefault(); setPaused(!paused); }
    if (e.key === "1" || e.key === "2" || e.key === "4") setSpeed(+e.key);
    if (e.key === "Escape") select(null);
  });
  function stopFollowing() { follow = false; $("iFollow").setAttribute("aria-pressed", "false"); followEgo = false; $("avFollow").setAttribute("aria-pressed", "false"); }

  /* ---------- camera */
  const toWorld = (sx, sy) => [(sx - W / 2) / cam.z + cam.x, (sy - H / 2) / cam.z + cam.y];
  const clampCam = () => { const B = map.bounds; cam.z = Math.max(Math.min(W / (B.x1 - B.x0), H / (B.y1 - B.y0)) * 0.9, Math.min(70, cam.z)); cam.x = Math.max(B.x0, Math.min(B.x1, cam.x)); cam.y = Math.max(B.y0, Math.min(B.y1, cam.y)); };
  let camAnim = null;
  function animateCam(x, y, z) { camAnim = { x0: cam.x, y0: cam.y, z0: cam.z, x, y, z, t: 0 }; }
  const pts = new Map(); let moved = 0, pinch0 = null;
  stage.addEventListener("pointerdown", (e) => { stage.setPointerCapture(e.pointerId); pts.set(e.pointerId, [e.clientX, e.clientY]); moved = 0; camAnim = null; if (pts.size === 2) { const [a, b] = [...pts.values()]; pinch0 = { d: Math.hypot(a[0] - b[0], a[1] - b[1]), z: cam.z }; } stage.classList.add("drag"); hideHint(); });
  stage.addEventListener("pointermove", (e) => {
    if (!pts.has(e.pointerId)) { hover(e); return; }
    const p = pts.get(e.pointerId), dx = e.clientX - p[0], dy = e.clientY - p[1];
    pts.set(e.pointerId, [e.clientX, e.clientY]);
    if (pts.size === 2 && pinch0) { const [a, b] = [...pts.values()]; cam.z = pinch0.z * Math.hypot(a[0] - b[0], a[1] - b[1]) / pinch0.d; moved += 10; }
    else { cam.x -= dx / cam.z; cam.y -= dy / cam.z; moved += Math.abs(dx) + Math.abs(dy); if (moved > 6) stopFollowing(); }
    clampCam();
  });
  const up = (e) => {
    if (!pts.has(e.pointerId)) return;
    pts.delete(e.pointerId); if (pts.size < 2) pinch0 = null;
    if (pts.size === 0) { stage.classList.remove("drag"); if (moved < 6) { const [wx, wy] = toWorld(e.clientX, e.clientY); select(pickAt(wx, wy)); } }
  };
  stage.addEventListener("pointerup", up); stage.addEventListener("pointercancel", up);
  stage.addEventListener("wheel", (e) => {
    e.preventDefault(); camAnim = null; hideHint();
    const [wx, wy] = toWorld(e.clientX, e.clientY);
    cam.z *= Math.exp(-e.deltaY * 0.0015); clampCam();
    const [nx, ny] = toWorld(e.clientX, e.clientY);
    if (!followEgo && !follow) { cam.x += wx - nx; cam.y += wy - ny; }
    clampCam();
  }, { passive: false });
  function pickAt(x, y) { let best = null, bd = 2.2; for (const v of [...sim.vehicles, ...sim.agents]) { if (v.static) continue; const d = Math.hypot(v.x - x, v.y - y) - (v.T.l / 2) * 0.6; if (d < bd) { bd = d; best = v; } } return best; }
  function hover(e) { const [wx, wy] = toWorld(e.clientX, e.clientY); stage.classList.toggle("hot", !!pickAt(wx, wy)); }
  let hintGone = false; function hideHint() { if (hintGone) return; hintGone = true; $("hint").style.opacity = 0; }
  setTimeout(hideHint, 9000);

  /* ---------- minimap: the whole city, junction lights, the car, and the camera window */
  const cityBox = () => ({ x0: -26000, y0: -32000, x1: 30000, y1: 30000 });
  mini.cv.addEventListener("click", (e) => {
    if (miniCity && preset === "hcm") {
      const b = mini.cv.getBoundingClientRect(), B = cityBox(), s = Math.min(b.width / (B.x1 - B.x0), b.height / (B.y1 - B.y0));
      const x = B.x0 + (e.clientX - b.left - (b.width - (B.x1 - B.x0) * s) / 2) / s, y = B.y0 + (e.clientY - b.top - (b.height - (B.y1 - B.y0) * s) / 2) / s;
      leaveReplay(); city.ensure(x, y, 700).then(() => { const c = nearestJunction(x, y); setMiniTab(false); openPlace(c[0], c[1]); }); return;
    }
    const b = mini.cv.getBoundingClientRect(), B = map.bounds, s = Math.min(b.width / (B.x1 - B.x0), b.height / (B.y1 - B.y0));
    const x = B.x0 + (e.clientX - b.left - (b.width - (B.x1 - B.x0) * s) / 2) / s, y = B.y0 + (e.clientY - b.top - (b.height - (B.y1 - B.y0) * s) / 2) / s;
    stopFollowing(); animateCam(x, y, Math.max(cam.z, fitZoom()));
  });
  function nearestJunction(x, y) { let best = [x, y], bd = 1e18; for (const k of city.keys(x - 700, y - 700, x + 700, y + 700)) { const t = city.tile(k); if (t) for (const j of t.J) { if (j.t === "end" || j.t === "bend") continue; const d = (j.x - x) ** 2 + (j.y - y) ** 2 - (j.t === "signal" ? 4e4 : 0); if (d < bd) { bd = d; best = [j.x, j.y]; } } } return best; }
  function drawCityMini() {   // the whole city: main roads, rivers, rail; the current window as a box
    const c = mini.cv, w = c.clientWidth, h = c.clientHeight, m = c.getContext("2d"), B = cityBox(), s = Math.min(w / (B.x1 - B.x0), h / (B.y1 - B.y0)), ox = (w - (B.x1 - B.x0) * s) / 2, oy = (h - (B.y1 - B.y0) * s) / 2;
    const X = (x) => ox + (x - B.x0) * s, Y = (y) => oy + (y - B.y0) * s;
    if (mini.dirty || !mini.cityBase) {
      const base = document.createElement("canvas"); base.width = c.width; base.height = c.height; const b = base.getContext("2d"); b.setTransform(dpr, 0, 0, dpr, 0, 0);
      b.fillStyle = "#c9c3b5"; b.fillRect(0, 0, w, h);
      const line = (a) => { b.beginPath(); for (let i = 0; i < a.length; i += 2) i ? b.lineTo(X(a[i] / 10), Y(a[i + 1] / 10)) : b.moveTo(X(a[0] / 10), Y(a[1] / 10)); };
      if (cityOv) {
        b.strokeStyle = "#7fb0c8"; b.fillStyle = "#7fb0c8"; for (const [wd, a] of cityOv.water) { line(a); if (wd) { b.lineWidth = Math.max(0.6, wd * s); b.stroke(); } else b.fill(); }
        for (const [cls, a] of cityOv.roads) { b.strokeStyle = cls === "highway" ? "#d9822b" : cls === "avenue" ? "#e0a03a" : cls === "main" ? "#6a6f75" : "#8d9095"; b.lineWidth = cls === "highway" || cls === "avenue" ? 1.1 : cls === "main" ? 0.7 : 0.35; line(a); b.stroke(); }
        b.strokeStyle = "#2f5d9a"; b.lineWidth = 1; for (const [k, a] of cityOv.rails) { b.setLineDash(k === "metro" ? [] : [2, 1.5]); line(a); b.stroke(); } b.setLineDash([]);
      } else { b.fillStyle = "#555"; b.font = "12px sans-serif"; b.fillText("Loading city overview…", 10, 20); }
      mini.cityBase = base; mini.dirty = false;
    }
    m.setTransform(1, 0, 0, 1, 0, 0); m.clearRect(0, 0, c.width, c.height); m.drawImage(mini.cityBase, 0, 0); m.setTransform(dpr, 0, 0, dpr, 0, 0);
    const Bw = map.bounds; m.strokeStyle = "#16b5a6"; m.lineWidth = 2; m.strokeRect(X(Bw.x0), Y(Bw.y0), Math.max(3, (Bw.x1 - Bw.x0) * s), Math.max(3, (Bw.y1 - Bw.y0) * s));
    m.fillStyle = "rgba(30,35,39,.7)"; m.font = "600 10px sans-serif"; m.fillText("Click to drive there", 8, h - 8);
  }
  function drawMini() {
    const c = mini.cv, w = c.clientWidth, h = c.clientHeight; if (!w) return;
    if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); mini.dirty = true; }
    if (miniCity && preset === "hcm") return drawCityMini();
    const m = c.getContext("2d"), B = map.bounds, s = Math.min(w / (B.x1 - B.x0), h / (B.y1 - B.y0)), ox = (w - (B.x1 - B.x0) * s) / 2, oy = (h - (B.y1 - B.y0) * s) / 2;
    const X = (x) => ox + (x - B.x0) * s, Y = (y) => oy + (y - B.y0) * s;
    if (mini.dirty) {
      const base = document.createElement("canvas"); base.width = c.width; base.height = c.height;
      const b = base.getContext("2d"); b.setTransform(dpr, 0, 0, dpr, 0, 0);
      b.fillStyle = "#b3ad9c"; b.fillRect(X(B.x0), Y(B.y0), (B.x1 - B.x0) * s, (B.y1 - B.y0) * s);
      const LANDC = { urban: "#b7b0a2", industrial: "#aaa79f", rural: "#8fa564", forest: "#5d7a45" };
      for (const ar of map.areas) { b.fillStyle = LANDC[ar.kind] || "#b3ad9c"; b.fillRect(X(ar.x0), Y(ar.y0), (ar.x1 - ar.x0) * s, (ar.y1 - ar.y0) * s); }
      const shape = (o) => { b.beginPath(); if (o.disk) b.arc(X(o.disk[0]), Y(o.disk[1]), o.disk[2] * s, 0, 6.3); else { b.moveTo(X(o.poly[0][0]), Y(o.poly[0][1])); for (const p of o.poly) b.lineTo(X(p[0]), Y(p[1])); b.closePath(); } b.fill(); };
      b.fillStyle = "#7fb0c8"; for (const q of layers.F.water) { if (q.poly) shape(q); else { b.strokeStyle = "#7fb0c8"; b.lineWidth = Math.max(0.8, q.w * s); b.beginPath(); q.pts.forEach((p, i) => i ? b.lineTo(X(p[0]), Y(p[1])) : b.moveTo(X(p[0]), Y(p[1]))); b.stroke(); } }
      b.fillStyle = "#8c867a"; for (const h of layers.F.houses) { const P = h.b.osm ? h.b.poly : h.corners; b.beginPath(); P.forEach((p, i) => i ? b.lineTo(X(p[0]), Y(p[1])) : b.moveTo(X(p[0]), Y(p[1]))); b.closePath(); b.fill(); }
      b.fillStyle = "#cdc6b8"; for (const q of layers.F.walks) shape(q);
      for (const q of layers.F.roads) { b.fillStyle = q.c === "#8f8b83" || q.c === "#97928a" ? "#7d7a74" : "#4a4e53"; shape(q); }
      for (const q of layers.F.decks) { b.fillStyle = "#5c6066"; shape(q); }
      b.strokeStyle = "#6b5a4a"; b.lineWidth = 1.2; for (const R of map.rails) { b.beginPath(); R.pts.forEach((p, i) => i ? b.lineTo(X(p[0]), Y(p[1])) : b.moveTo(X(p[0]), Y(p[1]))); b.stroke(); }
      mini.base = base; mini.dirty = false;
    }
    m.setTransform(1, 0, 0, 1, 0, 0); m.clearRect(0, 0, c.width, c.height); m.drawImage(mini.base, 0, 0);
    m.setTransform(dpr, 0, 0, dpr, 0, 0);
    m.fillStyle = "rgba(255,230,160,.55)"; for (const v of sim.vehicles) if (!v.ego) m.fillRect(X(v.x) - 0.6, Y(v.y) - 0.6, 1.2, 1.2);
    for (const jn of map.nodes) {
      if (jn.type === "bend") continue;
      const lE = sim.light(jn.id, jn.order[0]), col = !lE ? (jn.type === "roundabout" ? "#9ad0f0" : "#d8d3c4") : lE.c === "G" ? "#2ee66b" : lE.c === "Y" ? "#ffc21a" : "#ff4a3d";
      m.fillStyle = col; m.beginPath(); m.arc(X(jn.x), Y(jn.y), jn.type === "uturn" ? 1.6 : 2.6, 0, 6.3); m.fill(); m.strokeStyle = "rgba(0,0,0,.5)"; m.lineWidth = 0.8; m.stroke();
    }
    m.fillStyle = "#2f5d9a"; for (const a of sim.agents) if (a.kind === "train") m.fillRect(X(a.x) - 2, Y(a.y) - 2, 4, 4);
    if (env.ego) {
      const e = env.ego; m.strokeStyle = "rgba(34,200,185,.95)"; m.lineWidth = 2; m.beginPath();
      e.route.forEach((r, k) => { if (k < e.legs - 1) return; const p = sim.path(r.j, r.from, r.to); for (let i = k === e.legs - 1 ? Math.floor(e.s / p.ds) : 0; i < p.n; i += 6) m.lineTo(X(p.X[i]), Y(p.Y[i])); });
      m.stroke(); m.fillStyle = "#16b5a6"; m.beginPath(); m.arc(X(e.x), Y(e.y), 3.4, 0, 6.3); m.fill(); m.strokeStyle = "#fff"; m.lineWidth = 1.2; m.stroke();
    }
    m.strokeStyle = "rgba(255,255,255,.9)"; m.lineWidth = 1.2; m.strokeRect(X(cam.x - W / 2 / cam.z), Y(cam.y - H / 2 / cam.z), W / cam.z * s, H / cam.z * s);
  }

  /* ---------- inspector */
  const TYPE_NAME = { moto: "Motorbike", bike: "Bicycle", car: "Car", taxi: "Taxi", bus: "Bus", truck: "Truck", tractor: "Farm tractor", ped: "Pedestrian", buffalo: "Water buffalo", train: "Train", parked: "Parked car", stall: "Market stall", stalled: "Stalled motorbike" };
  const armName = (id) => "the " + (COMPASS_NAME[String(id).replace(/\d+$/, "")] || id);
  const KIND = { straight: "going straight", left: "turning left", right: "turning right", uturn: "making a U-turn" };
  const NODE_NAME = { signal: "signalised junction", yield: "junction", priority: "side-road junction", roundabout: "roundabout", uturn: "U-turn opening", rail: "level crossing", bend: "bend" };
  const traits = [["aggr", "Risk-taking", (v) => v.aggr], ["patience", "Patience", (v) => v.patience], ["comply", "Rule-abiding", (v) => v.comply], ["speed", "Likes speed", (v) => (v.speedPref - 0.8) / 0.45]];
  const egoBars = [["spd", "Target speed", () => (lastAct[0] + 1) / 2], ["lat", "Lane position", () => (lastAct[1] + 1) / 2]];
  function buildTraits(list) { $("iTraits").innerHTML = list.map(([k, n]) => `<div class="trait"><span>${n}</span><span class="bar"><b id="t_${k}"></b></span><span class="num" id="tn_${k}"></span></div>`).join(""); }
  function fillTraits(list, v) { for (const [k, , f] of list) { const x = Math.max(0, Math.min(1, f(v))); $("t_" + k).style.width = (x * 100).toFixed(0) + "%"; $("tn_" + k).textContent = Math.round(x * 100); } }
  function select(v) {
    sel = v; $("insp").hidden = !v; document.body.classList.toggle("inspecting", !!v);
    if (!v) { follow = false; $("iFollow").setAttribute("aria-pressed", "false"); return; }
    if (v.ego) { $("iName").textContent = "Self-driving car · " + drivers[drv].name; buildTraits(egoBars); $("iTraitsLbl").textContent = "Current command"; $("iFollow").hidden = true; }
    else if (v.agent) { $("iName").textContent = TYPE_NAME[v.kind] || v.kind; $("iTraits").innerHTML = ""; $("iTraitsLbl").textContent = v.kind === "ped" ? (v.jay ? "Crosses without waiting for a gap" : "Waits for a gap to cross") : v.kind === "buffalo" ? "Ambles along the verge, sometimes wanders across" : "Runs on a timetable; the barriers close for it"; $("iFollow").hidden = false; }
    else {
      const extra = v.type === "moto" ? (v.delivery ? " · delivery" : v.passenger ? " · with passenger" : v.cargo === "gas" ? " · carrying gas cylinders" : v.cargo ? " · carrying goods" : "") : "";
      $("iName").textContent = TYPE_NAME[v.type] + extra; buildTraits(traits); fillTraits(traits, v); $("iTraitsLbl").textContent = "Personality (fixed at spawn)"; $("iFollow").hidden = false;
    }
    updateInspector();
  }
  $("iClose").onclick = () => select(null);
  $("iFollow").onclick = () => { follow = !follow; $("iFollow").setAttribute("aria-pressed", String(follow)); if (follow && sel) { followEgo = false; $("avFollow").setAttribute("aria-pressed", "false"); animateCam(sel.x, sel.y, Math.max(cam.z, fitZoom() * 1.6)); } };
  function updateInspector() {
    if (!sel) return;
    if (sel.agent) {
      $("iRoute").textContent = sel.kind === "ped" ? (sel.mode === "walk" ? "Walking along the " + (map.walks[sel.walk].onRoad ? "road edge" : "sidewalk") : sel.mode === "wait" ? "Waiting to cross" : "Crossing the road") : sel.kind === "buffalo" ? (sel.graze > 0 ? "Grazing" : sel.cross ? "Wandering across the road" : "Walking along the verge") : "Passing through";
      $("iSpeed").textContent = Math.round(sel.v * 3.6) + " km/h"; $("iWait").textContent = Math.round(sel.wait || 0) + " s";
      const st = $("iState"); st.className = "state " + (sel.v > 0.2 ? "go" : "wait"); st.lastElementChild.textContent = sel.v > 0.2 ? "Moving" : "Standing";
      return;
    }
    const jn0 = map.nodes[sel.path.junc];
    $("iRoute").textContent = `${NODE_NAME[jn0.type][0].toUpperCase() + NODE_NAME[jn0.type].slice(1)} ${sel.path.junc + 1} · from ${armName(sel.path.from)} · ${KIND[sel.path.kind]}`;
    $("iSpeed").textContent = Math.round(sel.v * 3.6) + " km/h"; $("iWait").textContent = Math.round(sel.wait) + " s";
    const L = sel.path.signal ? sim.light(sel.path.junc, sel.path.from) : null;
    let cls = "go", txt = "Moving";
    if (sel.ghostUntil > sim.t) { cls = "wait"; txt = "Squeezing through"; }
    else if ((sel.ego ? env.redLegs && env.redLegs.size : sel.ranRed) && sel.s < sel.path.sOut + 5) { cls = "stop"; txt = "Running a red"; }
    else if (sel.v < 0.3 && sel.honk > 0) { cls = "wait"; txt = "Honking"; }
    else if (sel.v < 0.3 && sel.dwell > sim.t) { cls = "wait"; txt = "At the bus stop"; }
    else if (sel.v < 0.3) { cls = "wait"; txt = sel.s < sel.path.sIn && L && L.c !== "G" ? (sel.path.rail ? "Waiting for the train" : "Waiting at light") : !sel.path.signal && sel.s < sel.path.sIn + 2 && sel.s > sel.path.sStop - 4 ? "Yielding" : "Stuck in traffic"; }
    else if (sel.overtake > sim.t) { cls = "wait"; txt = "Going round an obstacle"; }
    else if (sel.braking) { cls = "wait"; txt = "Braking"; }
    const st = $("iState"); st.className = "state " + cls; st.lastElementChild.textContent = txt;
    if (sel.ego) fillTraits(egoBars, sel);
  }

  /* ---------- the self-driving car */
  $("av").hidden = false;
  function renderDrivers() {
    $("drvSeg").innerHTML = drivers.map((d, i) => `<button data-d="${i}" aria-pressed="${i === drv}">${d.short}</button>`).join("");
    $("drvSeg").querySelectorAll("button").forEach((b) => b.onclick = () => setDriver(+b.dataset.d));
  }
  renderDrivers();
  $("avFollow").onclick = () => { followEgo = !followEgo; $("avFollow").setAttribute("aria-pressed", String(followEgo)); if (followEgo) { follow = false; $("iFollow").setAttribute("aria-pressed", "false"); if (env.ego) animateCam(env.ego.x, env.ego.y, Math.max(cam.z, fitZoom())); } };
  $("avSee").onclick = () => { seeing = !seeing; $("avSee").setAttribute("aria-pressed", String(seeing)); };
  function setDriver(i) {
    const prevReplay = drivers[drv] && drivers[drv].replay, prevDrv = drv;
    drv = i; $("drvSeg").querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.d === i)));
    $("avDesc").textContent = drivers[i].desc;
    tally = { ok: 0, crash: 0, red: 0, timeout: 0, hit: 0, n: 0, tOk: 0 };
    const ev = AV && AV.evals ? AV.evals[drivers[i].id] : null;
    $("avEval").innerHTML = ev ? `Scored on ${ev.episodes} fixed episodes: completed <b>${pct(ev.ok)}</b> · crash <b>${pct(ev.crash)}</b> · red light <b>${pct(ev.red)}</b>` : "";
    $("epPick").hidden = !drivers[i].replay; $("driveBar").hidden = drivers[i].src !== "drive";
    if (prevReplay !== drivers[i].replay || (drivers[i].src || "") !== ((drivers[prevDrv] || {}).src || "")) pickId = null;
    if (drivers[i].replay) { replay = null; nextReplay(); }
    else if (replay || prevReplay) { replay = null; newCity(citySeed); }   // back to a live simulation
    else if (env && env.ego) { env.removeEgo(); curObs = env.reset(); }
    if (sel && sel.ego) select(env.ego);
    updateTally(); drawCurve();
  }
  /* the list of recorded episodes: the newest training episodes (auto), then the checkpoint episodes, grouped by scenario
     so that the next one is the same scenario at the next checkpoint */
  function fillEpPick(eps, src) {
    const R = { ok: "done", crash: "crash", hit: "hit", red: "red light", timeout: "timeout" }, k = (e) => Math.round(e.steps / 1000) + "k";
    if (src === "drive") {   // the stretches of the journey, in order
      epOrder = eps.map((e) => e.id);
      const el = $("epPick"), cur = el.value;
      el.innerHTML = `<option value="auto">Whole journey, stretch by stretch</option>` + eps.map((e) => `<option value="${e.id}">${e.id + 1}. ${e.traffic.replace(/ \(cell.*$/, "")} · ${e.place === "rail" ? "level crossing" : e.place} · ${e.result ? `${(e.km || 0).toFixed(1)} km · ${R[e.result] || e.result}${e.red && e.result !== "red" ? " · red light" : ""}` : "driving now"}</option>`).join("");
      el.value = cur || "auto"; if (el.selectedIndex < 0) el.value = "auto"; return;
    }
    const ms = eps.filter((e) => e.milestone).sort((a, b) => a.scenario - b.scenario || a.steps - b.steps), live = eps.filter((e) => !e.milestone).reverse();
    epOrder = [...ms, ...live].map((e) => e.id);
    const opt = (e, label) => `<option value="${e.id}">${label} · ${R[e.result] || e.result}</option>`, kk = k;
    const k2 = (e) => e.ckpt === "best.pt" ? "best " + kk(e) : kk(e);
    let html = `<option value="auto">Newest training episodes (in turn)</option>`;
    if (ms.length) html += `<optgroup label="Checkpoints, same scenarios">` + ms.map((e) => opt(e, `Scenario ${e.scenario + 1} · ${k2(e)}`)).join("") + `</optgroup>`;
    if (live.length) html += `<optgroup label="Recent training episodes">` + live.map((e) => opt(e, `#${e.id + 1} · ${k(e)}`)).join("") + `</optgroup>`;
    const el = $("epPick"), cur = el.value; el.innerHTML = html; el.value = cur || "auto"; if (el.selectedIndex < 0) el.value = "auto";
  }
  $("epPick").onchange = (e) => { pickId = e.target.value === "auto" ? null : +e.target.value; replay = null; tally = { ok: 0, crash: 0, red: 0, timeout: 0, hit: 0, n: 0, tOk: 0 }; updateTally(); nextReplay(); };

  /* a live stretch: fetch what has been driven since, and carry on (the beginning is identical) */
  async function extendLive(rp) {
    try {
      const ep = await (await fetch(`/api/drive?id=${rp.ep.id}`, { cache: "no-store" })).json();
      if (replay !== rp) return;
      if (ep.replay && ep.replay.actions.length > rp.i) { rp.ep = ep; rp.waiting = false; return; }
      if (ep.replay && !ep.live) { rp.waiting = false; rp.ep = ep; return; }   // final and nothing new: the next tick ends it
    } catch (e) {}
    setTimeout(() => { if (replay === rp) extendLive(rp); }, 2000);
  }

  /* replay a TD-MPC2 training episode exactly: the same seeded environment (map window, traffic) and the actions it chose, with its planner */
  async function nextReplay() {
    if (replayBusy) return; replayBusy = true; $("loading").hidden = false;
    try {
      const src = drivers[drv].src || "episodes", data = await (await fetch("/api/" + src, { cache: "no-store" })).json();
      const eps = (src === "drive" ? data.segments : data.episodes) || [];
      if (!city) { setTimeout(nextReplay, 1000); return; }   // the city data is still loading
      if (!eps.length) return;
      if (src === "drive") journey = data;
      fillEpPick(eps, src);
      let pick;
      if (src === "drive") { pick = (pickId != null && eps.find((e) => e.id === pickId)) || eps.find((e) => !seenDrive.has(e.id)); if (!pick) { seenDrive.clear(); tally = { ok: 0, crash: 0, red: 0, timeout: 0, hit: 0, n: 0, tOk: 0 }; updateTally(); pick = eps[0]; } seenDrive.add(pick.id); }
      else { const live = eps.filter((e) => !e.milestone); pick = (pickId != null && eps.find((e) => e.id === pickId)) || live.slice().reverse().find((e) => !seenEp.has(e.id)) || live[live.length - 1] || eps[eps.length - 1]; seenEp.add(pick.id); }
      $("epPick").value = pickId != null ? String(pick.id) : "auto";
      const ep = await (await fetch(`/api/${src}?id=${pick.id}`, { cache: "no-store" })).json(), R = ep.replay;
      const half = (R.opts.mapOpts && R.opts.mapOpts.half) || R.opts.half || 600;
      for (const c of R.opts.stageCentres || [R.centre]) if (c) await city.ensure(c[0], c[1], half + 100);   // every window of the trip (v3: long trips change window)
      const keep = getNpcBrain(); setNpcBrain(R.npc ? NPCB : null);
      const e = new EgoEnv(Object.assign({}, R.opts, { seed: R.seed, mapOpts: Object.assign({ centre: R.centre }, R.opts.mapOpts || {}) }));
      const o = e.reset(); setNpcBrain(keep);
      if (!drivers[drv].replay) return;   // switched to another driver meanwhile
      replay = { ep, i: 0, drift: false, src };
      env = e; sim = e.sim; map = sim.map; centre = R.centre; preset = "hcm"; curObs = o;
      $("preset").value = "hcm"; $("searchBox").hidden = false; $("attrib").hidden = false; $("miniTabs").hidden = false;
      layers = new CityLayers(map, R.seed * 3 + 1, ppm, 80, W < 680 ? 10 : 16); mini.dirty = true;
      const named = (map.nodes.filter((j) => j.name).sort((a, b) => Math.hypot(a.x - R.centre[0], a.y - R.centre[1]) - Math.hypot(b.x - R.centre[0], b.y - R.centre[1]))[0] || {}).name;
      $("cityTag").lastChild.nodeValue = src === "drive" ? `${named || "HCMC"} · test drive ${ep.id + 1}/${eps.length}` : ep.milestone ? `${named || "HCMC"} · ${Math.round(ep.steps / 1000)}k · scenario ${ep.scenario + 1}` : `${named || "HCMC"} · training episode ${ep.id + 1}`;
      const res = ep.live && !ep.result ? "is driving it right now (live: the replay follows the drive as it happens)" : { ok: "reached the end of its route", crash: "crashed", hit: "was hit by another road user", red: "ran a red light", timeout: "ran out of time" }[ep.result] || ep.result;
      if (src === "drive") {
        const S = journey.summary || {};
        $("avDesc").textContent = `Test drive of the ${/best/.test(S.ckpt || "") ? "best " : ""}checkpoint (${Math.round(ep.steps / 1000)}k training steps), full planner. Stretch ${ep.id + 1} of ${eps.length}: ${ep.traffic} at ${ep.place === "rail" ? "a level crossing" : ep.place} (${ep.kind})${ep.km_route ? `, a ${ep.km_route.toFixed(1)} km route` : ""}. It ${res}${ep.time && !ep.live ? " after " + (+ep.time).toFixed(1) + " s" : ""}.` + (R.imagined ? " Orange: the paths it imagined; violet: where it expected nearby road users; the badge: its predicted risk of contact." : "");
        $("driveSum").textContent = S.km != null ? `Whole journey: ${S.km} km in ${S.stretches} stretches · ${S.completed} completed · ${S.crashes} crashes (${S.crashes_per_100km} per 100 km) · ${S.red_lights} red lights · ${S.hit_by_others} hit by others · mean ${S.mean_kmh} km/h` : "";
        $("driveBar").hidden = false;
      } else $("avDesc").textContent = (ep.milestone ? `${ep.ckpt === "best.pt" ? "Best checkpoint (most routes completed minus crashes on the fixed evaluation)" : "Checkpoint"} at ${Math.round(ep.steps / 1000)}k training steps, scenario ${ep.scenario + 1}: every checkpoint drives these same scenarios, so you can compare them. Replayed exactly:` : `Training episode ${ep.id + 1}, recorded at ${Math.round(ep.steps / 1000)}k training steps and replayed exactly:`) + ` TD-MPC2 driving with its full planner (MPPI over imagined futures). It ${res}${ep.time ? " after " + (+ep.time).toFixed(1) + " s" : ""}.${R.imagined ? " Orange: the paths it imagined in its possible futures; violet: where it expected nearby road users to be (dashed pink: in its other possible futures); the badge is its predicted risk of contact within 2 s." : ""}${R.opts && R.opts.genome ? " This episode used an evolved scenario." : ""}` + (R.npc && !NPCB ? " (Learned traffic is not loaded here, so the replay may differ.)" : "");
      if (env.ego) { cam.x = env.ego.x; cam.y = env.ego.y; }
    } catch (err) { console.warn("replay", err); } finally { replayBusy = false; $("loading").hidden = true; }
  }
  /* training monitor: follow the TD-MPC2 run while it learns, swapping in each new snapshot of its network without restarting the drive */
  (function liveBrain() {
    if (!/^https?:$/.test(location.protocol)) return;
    let ver = null, busy = false, timer = null;
    const want = (() => { try { return new URLSearchParams(location.search).get("driver") === "latest"; } catch (e) { return false; } })();
    async function poll() {
      if (busy) return; busy = true;
      try {
        const r = await fetch("/api/policy?meta=1", { cache: "no-store" });
        if (!r.ok) { clearInterval(timer); return; }   // a plain web server, not the training monitor
        const m = await r.json();
        if (!drivers.some((x) => x.src === "drive")) {
          const J = await (await fetch("/api/drive", { cache: "no-store" })).json();
          if (J.segments && J.segments.length) {
            drivers.push({ id: "drive", replay: true, src: "drive", short: "Drive", name: "TD-MPC2 test drive", act: () => replay && replay.i < replay.ep.replay.actions.length ? replay.ep.replay.actions[replay.i++] : [-1, 0],
              desc: "A long test drive of the best checkpoint with its full planner, through every kind of scenario." });
            renderDrivers(); if (location.pathname === "/drive") setDriver(drivers.length - 1);
          }
        }
        if (!drivers.some((x) => x.replay && !x.src)) {
          const eps = (await (await fetch("/api/episodes", { cache: "no-store" })).json()).episodes || [];
          if (eps.length) {
            drivers.push({ id: "replay", replay: true, short: "Training", name: "TD-MPC2 in training", act: () => replay && replay.i < replay.ep.replay.actions.length ? replay.ep.replay.actions[replay.i++] : [-1, 0],
              desc: "Real TD-MPC2 training episodes, replayed exactly as they happened, with its full planner. Each one is the newest episode recorded so far." });
            renderDrivers(); if (want) setDriver(drivers.length - 1);
          }
        }
        if (m.v != null && m.v !== ver) {
          const spec = await (await fetch("/api/policy", { cache: "no-store" })).json();
          ver = m.v; if (spec.type === "tdmpc" && spec.egoDim === 43) swap(spec);
        }
      } catch (e) { console.warn("live poll", e); } finally { busy = false; }
    }
    function swap(spec) {
      const pol = makePolicy(spec), k = Math.round((spec.steps || 0) / 1000) + "k";
      const isBest = /best/i.test(spec.name || "");
      const d = { id: "tdmpc-live", nn: true, tdmpc: true, short: (isBest ? "Best " : "Prior ") + k, name: "TD-MPC2 policy prior " + k, act: (env) => pol.act(curObs),
        desc: isBest ? `TD-MPC2's policy prior from its best checkpoint (${k} steps; most routes completed minus crashes on the fixed evaluation). Here it drives with the prior alone; the Training button replays it with its full planner.` : `TD-MPC2's policy prior, live from training: ${k} steps so far. Each new snapshot replaces it while it drives. In training it also plans with its learned model of the world; here it drives with the prior alone.` };
      let i = drivers.findIndex((x) => x.tdmpc); const first = i < 0;
      if (first) { drivers.push(d); i = drivers.length - 1; } else drivers[i] = d;
      renderDrivers();
      if (drv === i) { $("avDesc").textContent = d.desc; if (!first) toast(`New brain: TD-MPC2 after ${k} steps`, "go"); }
      else if (want && !drivers[drv].replay) setDriver(i);
    }
    timer = setInterval(poll, 5000); poll();
  })();
  const pct = (x) => Math.round(x * 100) + "%";
  function updateTally() {
    for (const k of ["ok", "crash", "red", "timeout"]) $("ta_" + k).textContent = tally[k];
    $("avEp").textContent = tally.n + 1;
    $("avTime").textContent = tally.ok ? (tally.tOk / tally.ok).toFixed(1) + " s" : "–";
  }
  let toastT = null;
  function toast(text, kind) { const el = $("toast"); el.className = "toast " + kind; el.lastElementChild.textContent = text; el.hidden = false; clearTimeout(toastT); toastT = setTimeout(() => { el.hidden = true; }, 2400); }
  function onEpisodeEnd(info) {
    tally.n++; if (info.result !== "hit") tally[info.result === "red" ? "red" : info.result]++; else tally.hit++;
    if (info.red && info.result !== "red") tally.red++;
    const t = info.time.toFixed(1);
    if (info.result === "ok") { tally.tOk += info.time; toast(`Finished a ${info.legs}-junction route in ${t} s${info.red ? " · but ran a red" : ""}`, info.red ? "wait" : "go"); }
    else if (info.result === "crash" || info.result === "hit") {
      const u = env.crashWith; crashMark = { x: env.ego.x, y: env.ego.y, t: performance.now() };
      toast(info.result === "crash" ? `Crashed into a ${u ? TYPE_NAME[u.type].toLowerCase() : "vehicle"} after ${t} s` : `Hit by a ${u ? TYPE_NAME[u.type].toLowerCase() : "vehicle"} (not the self-driving car's fault)`, info.result === "crash" ? "stop" : "wait");
    }
    else if (info.result === "red") toast("Ran a red light, episode over", "stop");
    else toast("Ran out of time, episode skipped", "wait");
    updateTally();
  }
  function drawCurve() {
    const box = $("curveBox"); if (!AV || !AV.log || !AV.log.length) { box.hidden = true; return; } box.hidden = false;
    const w = curve.clientWidth, h = curve.clientHeight; if (!w) return;
    if (curve.width !== Math.round(w * dpr)) { curve.width = Math.round(w * dpr); curve.height = Math.round(h * dpr); }
    cg.setTransform(dpr, 0, 0, dpr, 0, 0); cg.clearRect(0, 0, w, h);
    const css = getComputedStyle(document.documentElement), col = (n) => css.getPropertyValue(n).trim();
    const L = 30, R = 8, T = 6, Bm = 18, maxS = AV.log[AV.log.length - 1].steps;
    const X = (s) => L + (s / maxS) * (w - L - R), Y = (p) => T + (1 - p) * (h - T - Bm);
    cg.font = "500 10px 'JetBrains Mono', ui-monospace, monospace"; cg.fillStyle = col("--ink-3"); cg.strokeStyle = col("--line"); cg.lineWidth = 1; cg.textAlign = "right"; cg.textBaseline = "middle";
    for (const p of [0, 0.5, 1]) { cg.beginPath(); cg.moveTo(L, Y(p)); cg.lineTo(w - R, Y(p)); cg.stroke(); cg.fillText(Math.round(p * 100) + "%", L - 5, Y(p)); }
    for (const [k, c] of [["ok", "--go"], ["crash", "--stop"]]) { cg.strokeStyle = col(c); cg.lineWidth = 2; cg.beginPath(); AV.log.forEach((r, i) => { const x = X(r.steps), y = Y(r[k]); i ? cg.lineTo(x, y) : cg.moveTo(x, y); }); cg.stroke(); }
    if (curveHover != null) { const r = AV.log.reduce((b, q) => Math.abs(X(q.steps) - curveHover) < Math.abs(X(b.steps) - curveHover) ? q : b, AV.log[0]); $("curveTip").textContent = `${Math.round(r.steps / 1000)}k steps · completed ${pct(r.ok)} · crash ${pct(r.crash)}`; }
    else $("curveTip").textContent = "Hover over the chart to inspect each checkpoint";
  }
  if (curve) { curve.addEventListener("pointermove", (e) => { curveHover = e.clientX - curve.getBoundingClientRect().left; drawCurve(); }); curve.addEventListener("pointerleave", () => { curveHover = null; drawCurve(); }); }

  /* ---------- one tick = one decision of the car (2 simulation steps) */
  const stepLen = DT * ENV.REPEAT;
  function tick() {
    if (drivers[drv].replay && (!replay || replayBusy || replay.waiting)) return;   // the next recorded episode is loading, or a live stretch has not driven further yet
    if (replay && env.ego) {   // a long trip records every k-th imagination: hold the last one (with the pose it was imagined from)
      const im = replay.ep.replay.imagined, q = im ? im[replay.i] : null;
      if (q || !im || replay.i === 0) { replay.imag = q; replay.anchor = [env.ego.x, env.ego.y, env.ego.h]; }
    }
    const snap = new Map(); for (const v of sim.vehicles) snap.set(v, [v.x, v.y, v.h]); for (const a of sim.agents) if (!a.static) snap.set(a, [a.x, a.y, a.h]);
    lastAct = drivers[drv].act(env);
    const r = env.step(lastAct); curObs = r.obs;
    if (replay && r.done && r.info.result === "stage") {   // v3: the trip goes on in the next window
      curObs = env.nextStage(); sim = env.sim; map = sim.map; layers = new CityLayers(map, replay.ep.replay.seed * 3 + 1 + env.stage, ppm, 80, W < 680 ? 10 : 16); mini.dirty = true;
      if (env.ego) { cam.x = env.ego.x; cam.y = env.ego.y; }
      toast(`New part of the trip (window ${env.stage + 1})`, "go"); return;
    }
    if (replay) {
      const c = replay.ep.replay.check.find((q) => q[0] === replay.i);
      if (c && env.ego && !replay.drift && Math.hypot(env.ego.x - c[1], env.ego.y - c[2]) > 1) { replay.drift = true; console.warn("replay drifted from the recording at decision", replay.i); }
      if (!r.done && replay.i >= replay.ep.replay.actions.length && replay.ep.live) { replay.waiting = true; extendLive(replay); }
      else if (r.done || replay.i >= replay.ep.replay.actions.length) {
        if (r.done) onEpisodeEnd(r.info);
        if (pickId != null) { const j = epOrder.indexOf(pickId); pickId = j >= 0 && j + 1 < epOrder.length ? epOrder[j + 1] : null; }
        replay = null; setTimeout(nextReplay, 1200);
      }
    }
    else if (r.done) { const old = env.ego; onEpisodeEnd(r.info); curObs = env.reset(); if (sel === old) select(env.ego); }
    for (const v of [...sim.vehicles, ...sim.agents]) { if (v.static) continue; const s = snap.get(v); if (s) { v.px = s[0]; v.py = s[1]; v.ph = s[2]; } else { v.px = v.x; v.py = v.y; v.ph = v.h; } }
  }

  /* ---------- HUD */
  const MIX_ORDER = ["moto", "car", "taxi", "bus", "truck", "bike", "tractor"];
  $("mix").innerHTML = MIX_ORDER.map((t) => `<div class="mixrow"><span>${TYPE_NAME[t]}</span><span class="bar"><b id="m_${t}"></b></span><span class="num" id="mn_${t}">0</span></div>`).join("");
  const spark = $("spark"), sg = spark.getContext("2d");
  function drawSpark() {
    const w = spark.clientWidth, h = spark.clientHeight; if (!w) return;
    if (spark.width !== Math.round(w * dpr)) { spark.width = Math.round(w * dpr); spark.height = Math.round(h * dpr); }
    sg.setTransform(dpr, 0, 0, dpr, 0, 0); sg.clearRect(0, 0, w, h);
    const css = getComputedStyle(document.documentElement), ink = css.getPropertyValue("--paint").trim() || "#c99a12", line = css.getPropertyValue("--line").trim();
    const N = 90, maxV = 40;
    sg.strokeStyle = line; sg.lineWidth = 1; for (const v of [10, 20, 30]) { const y = h - 3 - (v / maxV) * (h - 8); sg.beginPath(); sg.moveTo(0, y); sg.lineTo(w, y); sg.stroke(); }
    if (speedHist.length < 2) return;
    const X = (i) => w - (speedHist.length - 1 - i) * (w / (N - 1)), Y = (v) => h - 3 - Math.min(1, v / maxV) * (h - 8);
    const path = new Path2D(); speedHist.forEach((v, i) => i ? path.lineTo(X(i), Y(v)) : path.moveTo(X(i), Y(v)));
    sg.save(); const area = new Path2D(path); area.lineTo(X(speedHist.length - 1), h); area.lineTo(X(0), h); area.closePath(); sg.globalAlpha = 0.18; sg.fillStyle = ink; sg.fill(area); sg.restore();
    sg.strokeStyle = ink; sg.lineWidth = 2; sg.lineJoin = "round"; sg.stroke(path);
    sg.fillStyle = ink; sg.beginPath(); sg.arc(X(speedHist.length - 1) - 2, Y(speedHist[speedHist.length - 1]), 3, 0, 6.3); sg.fill();
  }
  let hudT = 0, honkLog = [], lastHonks = 0;
  function nearestJunction(x, y) { let best = map.nodes[0], bd = 1e9; for (const j of map.nodes) { if (j.type === "bend") continue; const d = (j.x - x) ** 2 + (j.y - y) ** 2; if (d < bd) { bd = d; best = j; } } return best; }
  function updateDriveBar() {
    if (!replay || replay.src !== "drive" || !journey) return;
    const segs = journey.segments, i = replay.ep.id, before = segs.filter((q) => q.id < i);
    if (!journey.summary) {   // the drive is still going
      const fin = segs.filter((q) => q.result).length, cr = segs.filter((q) => q.result === "crash").length, rd = segs.filter((q) => q.red).length;
      $("driveTxt").innerHTML = `<b>Test drive, live</b> · watching stretch ${i + 1}/${segs.length} · ${fin} finished so far: ${cr} crash${cr === 1 ? "" : "es"}, ${rd} red light${rd === 1 ? "" : "s"}${replay.waiting ? " · waiting for the car to drive on…" : ""}`;
      $("driveFill").style.width = (segs.length ? fin / segs.length * 100 : 0).toFixed(1) + "%"; return;
    }
    const total = segs.reduce((a, q) => a + q.km_route, 0);   // along the journey's route
    const now = env.ego ? Math.min(replay.ep.km_route, env.progress() / 1000) : replay.ep.km_route, done = before.reduce((a, q) => a + q.km_route, 0) + now;
    const cr = before.filter((q) => q.result === "crash").length, rd = before.filter((q) => q.red).length;
    $("driveTxt").innerHTML = `<b>Test drive</b> · ${done.toFixed(1)} / ${total.toFixed(1)} km · stretch ${i + 1}/${segs.length} · so far ${cr} crash${cr === 1 ? "" : "es"}, ${rd} red light${rd === 1 ? "" : "s"}`;
    $("driveFill").style.width = (total ? done / total * 100 : 0).toFixed(1) + "%";
  }
  function updateHud() {
    updateDriveBar();
    const s = sim.summary();
    $("sTotal").firstChild.nodeValue = s.total;
    $("sFlow").firstChild.nodeValue = s.flowPerHour.toLocaleString("en-US");
    $("sSpeed").firstChild.nodeValue = s.meanSpeed.toFixed(1);
    $("sRed").textContent = s.redRunsPerMin; $("sNear").textContent = s.nearMissPerMin;
    const nh = sim.stats.honks - lastHonks; lastHonks = sim.stats.honks; honkLog.push([sim.t, nh]); while (honkLog.length && honkLog[0][0] < sim.t - 60) honkLog.shift();
    $("sHonk").textContent = honkLog.reduce((a, b) => a + b[1], 0);
    for (const t of MIX_ORDER) { const c = s.counts[t] || 0; $("m_" + t).style.width = (s.total ? c / s.total * 100 : 0).toFixed(1) + "%"; $("mn_" + t).textContent = c; $("m_" + t).parentElement.parentElement.hidden = t === "tractor" && !c; }
    // the junction closest to the middle of the screen
    const jn = nearestJunction(cam.x, cam.y), grey = () => { $("lampMain").style.color = "var(--ink-3)"; $("phaseLeft").style.color = "var(--ink-3)"; $("phaseLeft").textContent = "–"; };
    $("phaseJ").textContent = `${jn.type === "rail" ? "Crossing" : jn.type === "roundabout" ? "Roundabout" : "Junction"} ${jn.id + 1}`;
    const names = (ids) => [...new Set(ids.map((id) => String(id).replace(/\d+$/, "")))].join(" – ");
    let col = null, left = null;
    if (jn.type === "signal") {
      const ph = sim.phaseAt(jn, sim.t), p = jn.phases[ph.i];
      if (p.allRed) { $("phaseName").textContent = "All red"; $("phaseSub").textContent = "Clearing the junction"; col = "R"; left = ph.left; }
      else { const ids = p.green || p.yellow; $("phaseName").textContent = names(ids); $("phaseSub").textContent = p.green ? `${names(ids)} is green · ${jn.phases.length / 3} phases` : `${names(ids)} turning red`; col = p.green ? "G" : "Y"; left = ph.left; }
    } else if (jn.type === "rail") {
      const L = sim.light(jn.id, jn.order[0]); col = L.c; left = L.left;
      $("phaseName").textContent = "Level crossing"; $("phaseSub").textContent = L.c === "G" ? "Barriers up" : L.c === "Y" ? "Bells ringing, barriers coming down" : "Barriers down, train passing";
    } else {
      const T = { roundabout: ["Roundabout", "Give way to traffic on the ring"], priority: [jn.kind === "alley" ? "Alley mouth" : "Side road", "Side traffic gives way to the main road"], uturn: ["U-turn opening", "U-turners give way to through traffic"], yield: ["No signal", "Vehicles yield to each other on entry"] }[jn.type] || ["Bend", ""];
      $("phaseName").textContent = T[0]; $("phaseSub").textContent = T[1];
    }
    if (col) { const c = col === "G" ? "var(--go)" : col === "Y" ? "var(--wait)" : "var(--stop)"; $("lampMain").style.color = c; $("phaseLeft").style.color = c; $("phaseLeft").textContent = String(Math.ceil(left)).padStart(2, "0"); } else grey();
    if (env.ego) {
      const e = env.ego;
      $("avRoute").textContent = `Junction ${e.legs}/${e.route.length} · ${KIND[e.path.kind]}`;
      $("avSpeed").textContent = Math.round(e.v * 3.6) + " km/h";
      $("avRet").textContent = (env.ep.ret >= 0 ? "+" : "") + env.ep.ret.toFixed(1);
    }
    updateInspector(); drawSpark();
  }

  /* ---------- what TD-MPC2 imagined at its last decision (recorded in training, replayed): its path in each sampled
     future, ghosts of the nearest road users where it expects them, and the predicted risk of contact within 2 s */
  function drawImagined(g, im, A) {
    const c = Math.cos(A[2]), s_ = Math.sin(A[2]), W_ = (p) => [A[0] + p[0] * c - p[1] * s_, A[1] + p[0] * s_ + p[1] * c];
    g.save(); g.lineCap = "round"; g.lineJoin = "round";
    (im.paths || []).forEach((P, f) => {
      if (!P.length) return;
      g.strokeStyle = f === 0 ? "rgba(255,140,40,.95)" : "rgba(255,170,90,.6)"; g.lineWidth = f === 0 ? 0.32 : 0.2; g.setLineDash(f === 0 ? [] : [0.5, 0.45]);
      g.beginPath(); const p0 = W_([0, 0]); g.moveTo(p0[0], p0[1]); for (const p of P) { const q = W_(p); g.lineTo(q[0], q[1]); } g.stroke();
      const e = W_(P[P.length - 1]); g.setLineDash([]); g.fillStyle = g.strokeStyle; g.beginPath(); g.arc(e[0], e[1], f === 0 ? 0.32 : 0.22, 0, 6.3); g.fill();
    });
    g.setLineDash([]);
    const tMax = Math.max(1, ...(im.ghosts || []).map((q) => q[0]));
    (im.ghosts || []).forEach(([t, gs, f]) => {   // most likely future: solid violet; the other futures: dashed pink, at the end of the horizon
      const a = t === tMax ? 0.4 : 0.65, rgb = f ? "235,90,170" : "150,110,255";
      g.setLineDash(f ? [0.35, 0.3] : []);
      for (const [x, y, ped, dir] of gs) {   // dir: the way it is moving (recorded since the ghosts skip standing things)
        const q = W_([x, y]), ang = A[2] + (dir ?? 0);
        g.strokeStyle = `rgba(${rgb},${a + 0.25})`; g.fillStyle = `rgba(${rgb},${f ? 0.06 : a * 0.35})`; g.lineWidth = 0.12;
        g.beginPath(); if (ped) g.arc(q[0], q[1], 0.45, 0, 6.3); else g.ellipse(q[0], q[1], 1.1, 0.5, ang, 0, 6.3); g.fill(); g.stroke();
        if (dir != null && !ped) { g.beginPath(); g.moveTo(q[0] + Math.cos(ang) * 1.1, q[1] + Math.sin(ang) * 1.1); g.lineTo(q[0] + Math.cos(ang) * 1.6, q[1] + Math.sin(ang) * 1.6); g.stroke(); }   // nose: which way it goes
      }
    });
    g.setLineDash([]);
    if (env.ego && im.risk != null) {
      const r = im.risk, col = r > 0.5 ? "#e5484d" : r > 0.2 ? "#f2a623" : "#2fb67c", z = cam.z, x = env.ego.x, y = env.ego.y - 3.2;
      g.font = `600 ${13 / z}px system-ui`; const txt = `risk ${Math.round(r * 100)}%`, w = g.measureText(txt).width + 10 / z;
      g.fillStyle = "rgba(20,24,28,.78)"; roundRect(g, x - w / 2, y - 9 / z, w, 18 / z, 9 / z); g.fill();
      g.fillStyle = col; g.textAlign = "center"; g.textBaseline = "middle"; g.fillText(txt, x, y + 0.5 / z);
    }
    g.restore();
  }

  /* ---------- drawing */
  const lerpAng = (a, b, t) => { let d = b - a; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI; return a + d * t; };
  let miniT = 0;
  function frame(now) {
    const real = Math.min(0.1, (now - last) / 1000); last = now;
    if (!paused) {
      acc += real * speed; let n = 0; const t0 = performance.now();
      while (acc >= stepLen && n < 8 && performance.now() - t0 < 45) { tick(); acc -= stepLen; n++; if (sim.t >= nextSample) { nextSample = sim.t + 2; speedHist.push(sim.summary().meanSpeed); if (speedHist.length > 90) speedHist.shift(); } }
      if (acc > stepLen * 2) acc = stepLen;   // the look-ahead teacher may not keep up at 4x: slow down rather than jump
    }
    if (sel && !sim.vehicles.includes(sel)) select(null);
    if (camAnim) { camAnim.t = Math.min(1, camAnim.t + real * 2.2); const e = 1 - Math.pow(1 - camAnim.t, 3); cam.x = camAnim.x0 + (camAnim.x - camAnim.x0) * e; cam.y = camAnim.y0 + (camAnim.y - camAnim.y0) * e; cam.z = camAnim.z0 + (camAnim.z - camAnim.z0) * e; if (camAnim.t >= 1) camAnim = null; }
    const a = paused ? 1 : Math.min(1, acc / stepLen), a_ = a, trainsVis = [];
    const tgt = follow && sel ? sel : followEgo && env.ego ? env.ego : null;
    if (tgt && !camAnim) { const k = 1 - Math.exp(-real * 4); cam.x += (tgt.x - cam.x) * k; cam.y += (tgt.y - cam.y) * k; }
    clampCam();

    g.setTransform(1, 0, 0, 1, 0, 0); g.fillStyle = "#b7b0a2"; g.fillRect(0, 0, cv.width, cv.height);
    const z = cam.z * dpr; g.setTransform(z, 0, 0, z, cv.width / 2 - cam.x * z, cv.height / 2 - cam.y * z);
    g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "high";
    const vx0 = cam.x - W / 2 / cam.z, vx1 = cam.x + W / 2 / cam.z, vy0 = cam.y - H / 2 / cam.z, vy1 = cam.y + H / 2 / cam.z;
    layers.draw(g, "ground", vx0, vy0, vx1, vy1, 2);
    const vis = [], pos = new Map(), m = 12; let overDone = false;
    for (const v of sim.vehicles) {
      let x = v.x, y = v.y, h = v.h;
      if (Math.hypot(v.x - v.px, v.y - v.py) < 6) { x = v.px + (v.x - v.px) * a; y = v.py + (v.y - v.py) * a; h = lerpAng(v.ph, v.h, a); }
      pos.set(v, [x, y]);
      if (x < vx0 - m || x > vx1 + m || y < vy0 - m || y > vy1 + m) continue;
      vis.push([v, x, y, h]);
    }
    const ego = env.ego;
    if (ego) {   // the route the car has been given, through every remaining junction
      g.save(); g.strokeStyle = "rgba(34,200,185,.5)"; g.lineWidth = 0.5; g.lineCap = "round"; g.lineJoin = "round"; g.beginPath();
      let first = true, end = null;
      ego.route.forEach((r, k) => {
        if (k < ego.legs - 1) return;
        const p = sim.path(r.j, r.from, r.to), i0 = k === ego.legs - 1 ? Math.floor(ego.s / p.ds) : 0, i1 = k === ego.route.length - 1 ? Math.min(p.n - 1, Math.round((p.sOut + ENV.END_AFTER) / p.ds)) : p.n - 1;
        for (let i = i0; i <= i1; i += 2) { const X = p.X[i] - p.TY[i] * ego.d, Y = p.Y[i] + p.TX[i] * ego.d; if (first) { g.moveTo(X, Y); first = false; } else g.lineTo(X, Y); end = [X, Y]; }
      });
      g.stroke();
      if (end) { g.fillStyle = "rgba(34,200,185,.95)"; g.beginPath(); g.arc(end[0], end[1], 0.7, 0, 6.3); g.fill(); g.fillStyle = "#fff"; g.beginPath(); g.arc(end[0], end[1], 0.28, 0, 6.3); g.fill(); }
      g.restore();
    }
    if (sel && !sel.ego) {
      const p = sel.path, i0 = Math.max(0, Math.floor(sel.s / p.ds));
      g.save(); g.strokeStyle = "rgba(255,214,70,.85)"; g.lineWidth = 0.22; g.setLineDash([0.8, 0.6]); g.lineCap = "round";
      g.beginPath(); for (let i = i0; i < p.n; i += 2) { const X = p.X[i] - p.TY[i] * sel.d, Y = p.Y[i] + p.TX[i] * sel.d; i === i0 ? g.moveTo(X, Y) : g.lineTo(X, Y); } g.stroke(); g.restore();
    }
    for (const a of sim.agents) {
      if (a.static) continue;
      let x = a.x, y = a.y, h = a.h;
      if (Math.hypot(a.x - a.px, a.y - a.py) < 6) { x = a.px + (a.x - a.px) * a_; y = a.py + (a.y - a.py) * a_; h = lerpAng(a.ph, a.h, a_); }
      pos.set(a, [x, y]);
      if (x < vx0 - 20 || x > vx1 + 20 || y < vy0 - 20 || y > vy1 + 20) continue;
      if (a.kind === "train") trainsVis.push([a, x, y, h]); else drawAgent(g, a, x, y, h, sim.t);
    }
    // levels: in a tunnel (only the car itself is shown, faded), on the ground, on a flyover or viaduct (drawn above its deck)
    const low = vis.filter(([v]) => (v.z || 0) < 2.5 && ((v.z || 0) > -2.5 || v.ego)), high = vis.filter(([v]) => (v.z || 0) >= 2.5);
    for (const [v, x, y, h] of low) drawShadow(g, v, x, y, h);
    low.sort((p, q) => (p[0].T.l > 3 ? 1 : 0) - (q[0].T.l > 3 ? 1 : 0));
    for (const [v, x, y, h] of low) { if ((v.z || 0) <= -2.5) g.globalAlpha = 0.45; drawVehicle(g, v, x, y, h, sim.t); g.globalAlpha = 1; }
    for (const [v, x, y] of low) if (v.honk > 0) drawHonk(g, v, x, y, sim.t);
    for (const [t_, x, y, h] of trainsVis) if ((t_.z || 0) < 2 && (t_.z || 0) > -2) drawAgent(g, t_, x, y, h, sim.t);
    if (high.length || trainsVis.some(([t_]) => (t_.z || 0) >= 2)) {
      layers.draw(g, "over", vx0, vy0, vx1, vy1, 0); overDone = true;
      for (const [t_, x, y, h] of trainsVis) if ((t_.z || 0) >= 2) drawAgent(g, t_, x, y, h, sim.t);
      for (const [v, x, y, h] of high) drawShadow(g, v, x, y, h);
      for (const [v, x, y, h] of high) drawVehicle(g, v, x, y, h, sim.t);
    }
    if (ego && seeing && env.seen) {
      const [ex, ey] = pos.get(ego), pk = pedScale(g);
      const ringR = (u) => u.kind === "ped" ? 0.32 * pk + 0.25 : Math.max(u.T.l, u.T.w) / 2 + 0.35;   // pedestrians are drawn larger than life
      g.save(); g.strokeStyle = "rgba(34,200,185,.5)"; g.lineWidth = 0.08; g.setLineDash([0.6, 0.5]); g.beginPath(); g.arc(ex, ey, ENV.RANGE, 0, 6.3); g.stroke(); g.setLineDash([]);
      if (env.hidden) for (const u of env.hidden) { const q = pos.get(u) || [u.x, u.y]; g.strokeStyle = "rgba(255,90,70,.75)"; g.lineWidth = 0.08; g.setLineDash([0.4, 0.35]); g.beginPath(); g.arc(q[0], q[1], ringR(u), 0, 6.3); g.stroke(); g.setLineDash([]); }
      if (env.sight != null && env.sight < ENV.SIGHT) { const q = env.routePoint(env.sight); g.fillStyle = "rgba(255,90,70,.85)"; g.beginPath(); g.arc(q[0], q[1], 0.6, 0, 6.3); g.fill(); g.strokeStyle = "rgba(255,90,70,.6)"; g.lineWidth = 0.1; g.beginPath(); g.moveTo(ex, ey); g.lineTo(q[0], q[1]); g.stroke(); }
      for (const u of env.seen) {
        const q = pos.get(u); if (!q) continue;
        const hot = Math.max(0, Math.min(1, 1 - sim.boxGap(ego, u) / 4));
        g.strokeStyle = `rgba(${Math.round(34 + 221 * hot)},${Math.round(200 - 130 * hot)},${Math.round(185 - 120 * hot)},${0.35 + 0.5 * hot})`; g.lineWidth = 0.06 + 0.1 * hot;
        g.beginPath(); g.moveTo(ex, ey); g.lineTo(q[0], q[1]); g.stroke(); g.beginPath(); g.arc(q[0], q[1], ringR(u), 0, 6.3); g.stroke();
      }
      g.restore();
    }
    if (crashMark) { const k = (now - crashMark.t) / 1600; if (k > 1) crashMark = null; else { g.strokeStyle = `rgba(255,70,55,${1 - k})`; g.lineWidth = 0.25; for (const r0 of [1.2, 2.4]) { g.beginPath(); g.arc(crashMark.x, crashMark.y, r0 + k * 3, 0, 6.3); g.stroke(); } } }
    if (sel) { const it = vis.find((q) => q[0] === sel); if (it) { const pulse = 0.5 + 0.5 * Math.sin(now / 220); g.strokeStyle = sel.ego ? `rgba(34,200,185,${0.6 + 0.4 * pulse})` : `rgba(255,214,70,${0.6 + 0.4 * pulse})`; g.lineWidth = 0.16; g.beginPath(); g.arc(it[1], it[2], sel.T.l / 2 + 0.9 + pulse * 0.2, 0, 6.3); g.stroke(); } }
    if (!overDone) layers.draw(g, "over", vx0, vy0, vx1, vy1, 0);
    drawSignals(g, sim, layers, vx0 - 2, vy0 - 2, vx1 + 2, vy1 + 2);
    if (replay && replay.imag && replay.anchor) drawImagined(g, replay.imag, replay.anchor);
    drawLabels(g, vx0, vy0, vx1, vy1);

    hudT -= real; if (hudT <= 0) { hudT = 0.25; updateHud(); }
    miniT -= real; if (miniT <= 0) { miniT = 0.2; drawMini(); }
    requestAnimationFrame(frame);
  }
  /* street and place names, drawn on top at a constant screen size; skipped where they would overlap */
  function drawLabels(g, vx0, vy0, vx1, vy1) {
    const F = layers.F; if (!F.labels || !F.labels.length || cam.z < 2.2) return;
    const placed = [], k = 1 / cam.z;
    g.save(); g.textAlign = "center"; g.textBaseline = "middle"; g.lineJoin = "round";
    for (const lb of F.labels) {
      if (lb.x < vx0 || lb.x > vx1 || lb.y < vy0 || lb.y > vy1) continue;
      if (!lb.place && !lb.major && cam.z < 5) continue;
      const fs = lb.place ? 13 : lb.major ? 12 : 10.5, wPx = lb.name.length * fs * 0.55 + 10;
      const sx = (lb.x - vx0) * cam.z, sy = (lb.y - vy0) * cam.z;
      if (placed.some((q) => Math.abs(q[0] - sx) < (q[2] + wPx) / 2 && Math.abs(q[1] - sy) < fs + 4)) continue;
      placed.push([sx, sy, wPx]);
      g.save(); g.translate(lb.x, lb.y); g.rotate(lb.ang); g.scale(k, k);
      g.font = `${lb.place ? 700 : 600} ${fs}px 'Be Vietnam Pro', system-ui, sans-serif`;
      g.strokeStyle = lb.place ? "rgba(255,255,255,.92)" : "rgba(28,30,33,.78)"; g.lineWidth = 3.5; g.strokeText(lb.name, 0, 0);
      g.fillStyle = lb.place ? "#1e2327" : "#f4f1e8"; g.fillText(lb.name, 0, 0);
      g.restore();
    }
    g.restore();
  }
  setDensity(+$("density").value);
  if (qMap === "hcm" || !qMap) {   // start in Ho Chi Minh City when its data can be fetched, else on a generated map
    loadCityData().then((ok) => { if (ok) { const qs = new URLSearchParams(location.search), want = qs.get("place"), at = qs.get("at"); if (at) { const [x, y] = at.split(",").map(Number); openPlace(x, y); return; } const hit = want && city.findPlace(want)[0]; const st = hit || city.findPlace("Ngã tư Bảy Hiền")[0]; if (st) openPlace(st.x, st.y); } });
  }
  newCity(citySeed);
  setDriver(drv);
  cam.z = fitZoom();
  updateHud(); drawCurve();
  requestAnimationFrame(frame);
  window.__nta = { sim: () => sim, env: () => env, map: () => map, cam, select, setSpeed, setPaused, setDriver, newCity, look: (x, y, z) => { stopFollowing(); cam.x = x; cam.y = y; if (z) cam.z = z; } };
})();
