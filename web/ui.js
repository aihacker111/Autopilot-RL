/* ======================================================================
   LOTUS: app loop, camera, HUD, minimap, inspector and the self-driving car.
   ====================================================================== */
(() => {
  const $ = (id) => document.getElementById(id);
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const ppm = Math.round(Math.min(14, Math.max(10, 7 * dpr)));
  let citySeed = 7, map, layers, sim, env;
  const AV = typeof AV_DATA !== "undefined" ? AV_DATA : null;

  /* ---------- drivers for the self-driving car */
  const planner = new LookaheadPlanner({ vTs: [0, 4, 8, 12], lats: [-0.6, 0, 0.6], M: 2, H: 10 });
  const drivers = [{ id: "planner", short: "Thầy nhìn trước", desc: "Mỗi 0,2 giây sao chép thế giới quanh xe, thử 12 phương án tốc độ và vị trí ngang trong 2 giây tới với 2 tương lai ngẫu nhiên, chọn phương án an toàn nhất trong trường hợp xấu nhất.", act: (env) => planner.plan(env) },
    { id: "idm", short: "Viết tay", desc: "Bộ lái viết tay: bám xe phía trước (IDM), dừng đèn đỏ và vàng. Mốc so sánh mà mọi bản học máy phải thắng.", act: (env) => idmAct(env) }];
  if (AV) for (const c of AV.ckpts) { const pol = makePolicy(c); drivers.push({ id: "nn" + c.steps, short: c.short, desc: c.desc, act: (env) => pol.act(curObs), nn: true }); }
  let drv = 0, curObs = null, lastAct = [0, 0], tally = null, crashMark = null, seeing = false;

  function newCity(seed) {
    citySeed = seed;
    map = cityMap(seed, { nx: 3, ny: 3, name: "Khu phố mẫu" });
    layers = new CityLayers(map, seed * 3 + 1, ppm, 80, W < 680 ? 10 : 16);
    sim = new Sim(map, seed, +$("density").value);
    env = new EgoEnv({ sim, reseedEvery: 0, seed: seed + 100 });
    curObs = env.reset();
    const sig = map.juncs.filter((j) => j.type === "signal").length, t3 = map.juncs.filter((j) => Object.keys(j.arms).length === 3).length;
    $("cityTag").lastChild.nodeValue = `Khu phố #${seed} · ${map.juncs.length} nút · ${sig} có đèn${t3 ? ` · ${t3} ngã ba` : ""}`;
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
  function setPaused(p) { paused = p; playIcon.setAttribute("d", p ? "M4 2l10 6-10 6z" : "M4 2h3v12H4zM9 2h3v12H9z"); $("play").setAttribute("aria-label", p ? "Chạy tiếp" : "Tạm dừng"); }
  $("play").onclick = () => setPaused(!paused);
  const speedBtns = document.querySelectorAll("#speedSeg button");
  speedBtns.forEach((b) => b.onclick = () => setSpeed(+b.dataset.speed));
  function setSpeed(s) { speed = s; speedBtns.forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.speed === s))); }
  const densNames = [[0.45, "Vắng"], [0.95, "Bình thường"], [1.15, "Đông"], [9, "Tan tầm"]];
  function setDensity(d) { if (sim) sim.density = d; $("densName").textContent = densNames.find(([t]) => d <= t)[1]; }
  $("density").oninput = (e) => setDensity(+e.target.value);
  $("fit").onclick = () => { stopFollowing(); const B = map.bounds; animateCam((B.x0 + B.x1) / 2, (B.y0 + B.y1) / 2, Math.min(W / (B.x1 - B.x0), H / (B.y1 - B.y0)) * 1.02); };
  $("newCity").onclick = () => { newCity(citySeed + 1); };
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
  function pickAt(x, y) { let best = null, bd = 2.2; for (const v of sim.vehicles) { const d = Math.hypot(v.x - x, v.y - y) - (v.T.l / 2) * 0.6; if (d < bd) { bd = d; best = v; } } return best; }
  function hover(e) { const [wx, wy] = toWorld(e.clientX, e.clientY); stage.classList.toggle("hot", !!pickAt(wx, wy)); }
  let hintGone = false; function hideHint() { if (hintGone) return; hintGone = true; $("hint").style.opacity = 0; }
  setTimeout(hideHint, 9000);

  /* ---------- minimap: the whole city, junction lights, the car, and the camera window */
  mini.cv.addEventListener("click", (e) => {
    const b = mini.cv.getBoundingClientRect(), B = map.bounds, s = Math.min(b.width / (B.x1 - B.x0), b.height / (B.y1 - B.y0));
    const x = B.x0 + (e.clientX - b.left - (b.width - (B.x1 - B.x0) * s) / 2) / s, y = B.y0 + (e.clientY - b.top - (b.height - (B.y1 - B.y0) * s) / 2) / s;
    stopFollowing(); animateCam(x, y, Math.max(cam.z, fitZoom()));
  });
  function drawMini() {
    const c = mini.cv, w = c.clientWidth, h = c.clientHeight; if (!w) return;
    if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); mini.dirty = true; }
    const m = c.getContext("2d"), B = map.bounds, s = Math.min(w / (B.x1 - B.x0), h / (B.y1 - B.y0)), ox = (w - (B.x1 - B.x0) * s) / 2, oy = (h - (B.y1 - B.y0) * s) / 2;
    const X = (x) => ox + (x - B.x0) * s, Y = (y) => oy + (y - B.y0) * s;
    if (mini.dirty) {
      const base = document.createElement("canvas"); base.width = c.width; base.height = c.height;
      const b = base.getContext("2d"); b.setTransform(dpr, 0, 0, dpr, 0, 0);
      b.fillStyle = "#b7b0a2"; b.fillRect(X(B.x0), Y(B.y0), (B.x1 - B.x0) * s, (B.y1 - B.y0) * s);
      b.fillStyle = "#8c867a"; for (const hs of layers.F.houses) b.fillRect(X(hs.x), Y(hs.y), Math.max(0.6, hs.w * s), Math.max(0.6, hs.h * s));
      b.fillStyle = "#cdc6b8"; for (const q of layers.F.walks) b.fillRect(X(q.x0), Y(q.y0), (q.x1 - q.x0) * s, (q.y1 - q.y0) * s);
      b.fillStyle = "#4a4e53"; for (const q of layers.F.roads) b.fillRect(X(q.x0), Y(q.y0), (q.x1 - q.x0) * s, (q.y1 - q.y0) * s);
      mini.base = base; mini.dirty = false;
    }
    m.setTransform(1, 0, 0, 1, 0, 0); m.clearRect(0, 0, c.width, c.height); m.drawImage(mini.base, 0, 0);
    m.setTransform(dpr, 0, 0, dpr, 0, 0);
    m.fillStyle = "rgba(255,230,160,.55)"; for (const v of sim.vehicles) if (!v.ego) m.fillRect(X(v.x) - 0.6, Y(v.y) - 0.6, 1.2, 1.2);
    for (const jn of map.juncs) {
      const lE = sim.light(jn.id, jn.arms.E ? "E" : "W"), col = !lE ? "#d8d3c4" : lE.c === "G" ? "#2ee66b" : lE.c === "Y" ? "#ffc21a" : "#ff4a3d";
      m.fillStyle = col; m.beginPath(); m.arc(X(jn.x), Y(jn.y), 2.6, 0, 6.3); m.fill(); m.strokeStyle = "rgba(0,0,0,.5)"; m.lineWidth = 0.8; m.stroke();
    }
    if (env.ego) {
      const e = env.ego; m.strokeStyle = "rgba(34,200,185,.95)"; m.lineWidth = 2; m.beginPath();
      e.route.forEach((r, k) => { if (k < e.legs - 1) return; const p = sim.path(r.j, r.from, r.to); for (let i = k === e.legs - 1 ? Math.floor(e.s / p.ds) : 0; i < p.n; i += 6) m.lineTo(X(p.X[i]), Y(p.Y[i])); });
      m.stroke(); m.fillStyle = "#16b5a6"; m.beginPath(); m.arc(X(e.x), Y(e.y), 3.4, 0, 6.3); m.fill(); m.strokeStyle = "#fff"; m.lineWidth = 1.2; m.stroke();
    }
    m.strokeStyle = "rgba(255,255,255,.9)"; m.lineWidth = 1.2; m.strokeRect(X(cam.x - W / 2 / cam.z), Y(cam.y - H / 2 / cam.z), W / cam.z * s, H / cam.z * s);
  }

  /* ---------- inspector */
  const TYPE_NAME = { moto: "Xe máy", bike: "Xe đạp", car: "Ô tô con", taxi: "Taxi", bus: "Xe buýt", truck: "Xe tải" };
  const ARM_NAME = { E: "phía Đông", W: "phía Tây", N: "phía Bắc", S: "phía Nam" };
  const KIND = { straight: "đi thẳng", left: "rẽ trái", right: "rẽ phải" };
  const traits = [["aggr", "Liều lĩnh", (v) => v.aggr], ["patience", "Kiên nhẫn", (v) => v.patience], ["comply", "Tuân thủ luật", (v) => v.comply], ["speed", "Thích chạy nhanh", (v) => (v.speedPref - 0.8) / 0.45]];
  const egoBars = [["spd", "Tốc độ mong muốn", () => (lastAct[0] + 1) / 2], ["lat", "Vị trí trong làn", () => (lastAct[1] + 1) / 2]];
  function buildTraits(list) { $("iTraits").innerHTML = list.map(([k, n]) => `<div class="trait"><span>${n}</span><span class="bar"><b id="t_${k}"></b></span><span class="num" id="tn_${k}"></span></div>`).join(""); }
  function fillTraits(list, v) { for (const [k, , f] of list) { const x = Math.max(0, Math.min(1, f(v))); $("t_" + k).style.width = (x * 100).toFixed(0) + "%"; $("tn_" + k).textContent = Math.round(x * 100); } }
  function select(v) {
    sel = v; $("insp").hidden = !v; document.body.classList.toggle("inspecting", !!v);
    if (!v) { follow = false; $("iFollow").setAttribute("aria-pressed", "false"); return; }
    if (v.ego) { $("iName").textContent = "Xe tự lái · " + drivers[drv].short; buildTraits(egoBars); $("iTraitsLbl").textContent = "Lệnh đang ra"; $("iFollow").hidden = true; }
    else {
      const extra = v.type === "moto" ? (v.delivery ? " · giao hàng" : v.passenger ? " · chở người" : v.cargo === "gas" ? " · chở bình gas" : v.cargo ? " · chở hàng" : "") : "";
      $("iName").textContent = TYPE_NAME[v.type] + extra; buildTraits(traits); fillTraits(traits, v); $("iTraitsLbl").textContent = "Tính cách (cố định từ lúc xuất hiện)"; $("iFollow").hidden = false;
    }
    updateInspector();
  }
  $("iClose").onclick = () => select(null);
  $("iFollow").onclick = () => { follow = !follow; $("iFollow").setAttribute("aria-pressed", String(follow)); if (follow && sel) { followEgo = false; $("avFollow").setAttribute("aria-pressed", "false"); animateCam(sel.x, sel.y, Math.max(cam.z, fitZoom() * 1.6)); } };
  function updateInspector() {
    if (!sel) return;
    $("iRoute").textContent = `Nút ${sel.path.junc + 1} · từ ${ARM_NAME[sel.path.from]} · ${KIND[sel.path.kind]}`;
    $("iSpeed").textContent = Math.round(sel.v * 3.6) + " km/h"; $("iWait").textContent = Math.round(sel.wait) + " s";
    const L = sel.path.signal ? sim.light(sel.path.junc, sel.path.from) : null;
    let cls = "go", txt = "Đang chạy";
    if (sel.ghostUntil > sim.t) { cls = "wait"; txt = "Đang chen lấn"; }
    else if ((sel.ego ? env.redLegs && env.redLegs.size : sel.ranRed) && sel.s < sel.path.sOut + 5) { cls = "stop"; txt = "Vượt đèn đỏ"; }
    else if (sel.v < 0.3 && sel.honk > 0) { cls = "wait"; txt = "Bấm còi"; }
    else if (sel.v < 0.3) { cls = "wait"; txt = sel.s < sel.path.sIn && L && L.c !== "G" ? "Chờ đèn" : !sel.path.signal && sel.s < sel.path.sIn + 2 && sel.s > sel.path.sStop - 4 ? "Nhường đường" : "Kẹt xe"; }
    else if (sel.braking) { cls = "wait"; txt = "Đang phanh"; }
    const st = $("iState"); st.className = "state " + cls; st.lastElementChild.textContent = txt;
    if (sel.ego) fillTraits(egoBars, sel);
  }

  /* ---------- the self-driving car */
  $("av").hidden = false;
  $("drvSeg").innerHTML = drivers.map((d, i) => `<button data-d="${i}" aria-pressed="${i === drv}">${d.short}</button>`).join("");
  $("drvSeg").querySelectorAll("button").forEach((b) => b.onclick = () => setDriver(+b.dataset.d));
  $("avFollow").onclick = () => { followEgo = !followEgo; $("avFollow").setAttribute("aria-pressed", String(followEgo)); if (followEgo) { follow = false; $("iFollow").setAttribute("aria-pressed", "false"); if (env.ego) animateCam(env.ego.x, env.ego.y, Math.max(cam.z, fitZoom())); } };
  $("avSee").onclick = () => { seeing = !seeing; $("avSee").setAttribute("aria-pressed", String(seeing)); };
  function setDriver(i) {
    drv = i; $("drvSeg").querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.d === i)));
    $("avDesc").textContent = drivers[i].desc;
    tally = { ok: 0, crash: 0, red: 0, timeout: 0, hit: 0, n: 0, tOk: 0 };
    const ev = AV && AV.evals ? AV.evals[drivers[i].id] : null;
    $("avEval").innerHTML = ev ? `Chấm trên ${ev.episodes} lượt cố định: qua được <b>${pct(ev.ok)}</b> · va chạm <b>${pct(ev.crash)}</b> · vượt đỏ <b>${pct(ev.red)}</b>` : "";
    if (env && env.ego) { env.removeEgo(); curObs = env.reset(); }
    if (sel && sel.ego) select(env.ego);
    updateTally(); drawCurve();
  }
  const pct = (x) => Math.round(x * 100) + "%";
  function updateTally() {
    for (const k of ["ok", "crash", "red", "timeout"]) $("ta_" + k).textContent = tally[k];
    $("avEp").textContent = tally.n + 1;
    $("avTime").textContent = tally.ok ? (tally.tOk / tally.ok).toFixed(1).replace(".", ",") + " s" : "–";
  }
  let toastT = null;
  function toast(text, kind) { const el = $("toast"); el.className = "toast " + kind; el.lastElementChild.textContent = text; el.hidden = false; clearTimeout(toastT); toastT = setTimeout(() => { el.hidden = true; }, 2400); }
  function onEpisodeEnd(info) {
    tally.n++; if (info.result !== "hit") tally[info.result === "red" ? "red" : info.result]++; else tally.hit++;
    if (info.red && info.result !== "red") tally.red++;
    const t = info.time.toFixed(1).replace(".", ",");
    if (info.result === "ok") { tally.tOk += info.time; toast(`Đi hết lộ trình ${info.legs} nút giao trong ${t} s${info.red ? " · nhưng có vượt đỏ" : ""}`, info.red ? "wait" : "go"); }
    else if (info.result === "crash" || info.result === "hit") {
      const u = env.crashWith; crashMark = { x: env.ego.x, y: env.ego.y, t: performance.now() };
      toast(info.result === "crash" ? `Va chạm với ${u ? TYPE_NAME[u.type].toLowerCase() : "xe khác"} sau ${t} s` : `Bị ${u ? TYPE_NAME[u.type].toLowerCase() : "xe khác"} tông vào (không phải lỗi xe tự lái)`, info.result === "crash" ? "stop" : "wait");
    }
    else if (info.result === "red") toast("Vượt đèn đỏ, kết thúc lượt", "stop");
    else toast("Kẹt quá 2 phút, bỏ lượt này", "wait");
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
    if (curveHover != null) { const r = AV.log.reduce((b, q) => Math.abs(X(q.steps) - curveHover) < Math.abs(X(b.steps) - curveHover) ? q : b, AV.log[0]); $("curveTip").textContent = `${Math.round(r.steps / 1000)}k bước · qua được ${pct(r.ok)} · va chạm ${pct(r.crash)}`; }
    else $("curveTip").textContent = "Rê chuột lên biểu đồ để xem từng mốc";
  }
  if (curve) { curve.addEventListener("pointermove", (e) => { curveHover = e.clientX - curve.getBoundingClientRect().left; drawCurve(); }); curve.addEventListener("pointerleave", () => { curveHover = null; drawCurve(); }); }

  /* ---------- one tick = one decision of the car (2 simulation steps) */
  const stepLen = DT * ENV.REPEAT;
  function tick() {
    const snap = new Map(); for (const v of sim.vehicles) snap.set(v, [v.x, v.y, v.h]);
    lastAct = drivers[drv].act(env);
    const r = env.step(lastAct); curObs = r.obs;
    if (r.done) { const old = env.ego; onEpisodeEnd(r.info); curObs = env.reset(); if (sel === old) select(env.ego); }
    for (const v of sim.vehicles) { const s = snap.get(v); if (s) { v.px = s[0]; v.py = s[1]; v.ph = s[2]; } else { v.px = v.x; v.py = v.y; v.ph = v.h; } }
  }

  /* ---------- HUD */
  const MIX_ORDER = ["moto", "car", "taxi", "bus", "truck", "bike"];
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
  function nearestJunction(x, y) { let best = map.juncs[0], bd = 1e9; for (const j of map.juncs) { const d = (j.x - x) ** 2 + (j.y - y) ** 2; if (d < bd) { bd = d; best = j; } } return best; }
  function updateHud() {
    const s = sim.summary();
    $("sTotal").firstChild.nodeValue = s.total;
    $("sFlow").firstChild.nodeValue = s.flowPerHour.toLocaleString("vi-VN");
    $("sSpeed").firstChild.nodeValue = s.meanSpeed.toFixed(1).replace(".", ",");
    $("sRed").textContent = s.redRunsPerMin; $("sNear").textContent = s.nearMissPerMin;
    const nh = sim.stats.honks - lastHonks; lastHonks = sim.stats.honks; honkLog.push([sim.t, nh]); while (honkLog.length && honkLog[0][0] < sim.t - 60) honkLog.shift();
    $("sHonk").textContent = honkLog.reduce((a, b) => a + b[1], 0);
    for (const t of MIX_ORDER) { const c = s.counts[t] || 0; $("m_" + t).style.width = (s.total ? c / s.total * 100 : 0).toFixed(1) + "%"; $("mn_" + t).textContent = c; }
    // the junction closest to the middle of the screen
    const jn = nearestJunction(cam.x, cam.y);
    $("phaseJ").textContent = `Nút ${jn.id + 1}`;
    if (jn.type !== "signal") { $("phaseName").textContent = "Không đèn"; $("phaseSub").textContent = "Xe tự nhường nhau khi vào nút"; $("lampMain").style.color = "var(--ink-3)"; $("phaseLeft").style.color = "var(--ink-3)"; $("phaseLeft").textContent = "–"; }
    else {
      const ax = jn.arms.E ? "E" : "W", ay = jn.arms.N ? "N" : "S", lE = sim.light(jn.id, ax), lN = sim.light(jn.id, ay);
      let name, sub, c, left;
      if (lE.c !== "R") { name = "Đông – Tây"; sub = lE.c === "G" ? "Hướng Đông – Tây đang xanh" : "Đông – Tây sắp đỏ"; c = lE.c; left = lE.left; }
      else if (lN.c !== "R") { name = "Bắc – Nam"; sub = lN.c === "G" ? "Hướng Bắc – Nam đang xanh" : "Bắc – Nam sắp đỏ"; c = lN.c; left = lN.left; }
      else { name = "Tất cả đỏ"; sub = "Giải phóng nút giao"; c = "R"; left = Math.min(lE.left, lN.left); }
      $("phaseName").textContent = name; $("phaseSub").textContent = sub;
      const col = c === "G" ? "var(--go)" : c === "Y" ? "var(--wait)" : "var(--stop)";
      $("lampMain").style.color = col; $("phaseLeft").style.color = col; $("phaseLeft").textContent = String(Math.ceil(left)).padStart(2, "0");
    }
    if (env.ego) {
      const e = env.ego;
      $("avRoute").textContent = `Nút ${e.legs}/${e.route.length} · ${KIND[e.path.kind]}`;
      $("avSpeed").textContent = Math.round(e.v * 3.6) + " km/h";
      $("avRet").textContent = (env.ep.ret >= 0 ? "+" : "") + env.ep.ret.toFixed(1).replace(".", ",");
    }
    updateInspector(); drawSpark();
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
    const a = paused ? 1 : Math.min(1, acc / stepLen);
    const tgt = follow && sel ? sel : followEgo && env.ego ? env.ego : null;
    if (tgt && !camAnim) { const k = 1 - Math.exp(-real * 4); cam.x += (tgt.x - cam.x) * k; cam.y += (tgt.y - cam.y) * k; }
    clampCam();

    g.setTransform(1, 0, 0, 1, 0, 0); g.fillStyle = "#b7b0a2"; g.fillRect(0, 0, cv.width, cv.height);
    const z = cam.z * dpr; g.setTransform(z, 0, 0, z, cv.width / 2 - cam.x * z, cv.height / 2 - cam.y * z);
    g.imageSmoothingEnabled = true; g.imageSmoothingQuality = "high";
    const vx0 = cam.x - W / 2 / cam.z, vx1 = cam.x + W / 2 / cam.z, vy0 = cam.y - H / 2 / cam.z, vy1 = cam.y + H / 2 / cam.z;
    layers.draw(g, "ground", vx0, vy0, vx1, vy1, 2);
    const vis = [], pos = new Map(), m = 12;
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
    for (const [v, x, y, h] of vis) drawShadow(g, v, x, y, h);
    vis.sort((p, q) => (p[0].T.l > 3 ? 1 : 0) - (q[0].T.l > 3 ? 1 : 0));
    for (const [v, x, y, h] of vis) drawVehicle(g, v, x, y, h, sim.t);
    for (const [v, x, y] of vis) if (v.honk > 0) drawHonk(g, v, x, y, sim.t);
    if (ego && seeing && env.seen) {
      const [ex, ey] = pos.get(ego);
      g.save(); g.strokeStyle = "rgba(34,200,185,.5)"; g.lineWidth = 0.08; g.setLineDash([0.6, 0.5]); g.beginPath(); g.arc(ex, ey, ENV.RANGE, 0, 6.3); g.stroke(); g.setLineDash([]);
      for (const u of env.seen) {
        const q = pos.get(u); if (!q) continue;
        const hot = Math.max(0, Math.min(1, 1 - sim.boxGap(ego, u) / 4));
        g.strokeStyle = `rgba(${Math.round(34 + 221 * hot)},${Math.round(200 - 130 * hot)},${Math.round(185 - 120 * hot)},${0.35 + 0.5 * hot})`; g.lineWidth = 0.06 + 0.1 * hot;
        g.beginPath(); g.moveTo(ex, ey); g.lineTo(q[0], q[1]); g.stroke(); g.beginPath(); g.arc(q[0], q[1], Math.max(u.T.l, u.T.w) / 2 + 0.35, 0, 6.3); g.stroke();
      }
      g.restore();
    }
    if (crashMark) { const k = (now - crashMark.t) / 1600; if (k > 1) crashMark = null; else { g.strokeStyle = `rgba(255,70,55,${1 - k})`; g.lineWidth = 0.25; for (const r0 of [1.2, 2.4]) { g.beginPath(); g.arc(crashMark.x, crashMark.y, r0 + k * 3, 0, 6.3); g.stroke(); } } }
    if (sel) { const it = vis.find((q) => q[0] === sel); if (it) { const pulse = 0.5 + 0.5 * Math.sin(now / 220); g.strokeStyle = sel.ego ? `rgba(34,200,185,${0.6 + 0.4 * pulse})` : `rgba(255,214,70,${0.6 + 0.4 * pulse})`; g.lineWidth = 0.16; g.beginPath(); g.arc(it[1], it[2], sel.T.l / 2 + 0.9 + pulse * 0.2, 0, 6.3); g.stroke(); } }
    layers.draw(g, "over", vx0, vy0, vx1, vy1, 0);
    drawSignals(g, sim, layers.heads, vx0 - 2, vy0 - 2, vx1 + 2, vy1 + 2);

    hudT -= real; if (hudT <= 0) { hudT = 0.25; updateHud(); }
    miniT -= real; if (miniT <= 0) { miniT = 0.2; drawMini(); }
    requestAnimationFrame(frame);
  }
  setDensity(+$("density").value);
  newCity(citySeed);
  setDriver(0);
  cam.z = fitZoom();
  updateHud(); drawCurve();
  requestAnimationFrame(frame);
  window.__nta = { sim: () => sim, env: () => env, cam, select, setSpeed, setPaused, setDriver, newCity };
})();
