/* ======================================================================
   LOTUS: top-down renderer (Canvas 2D).
   Static layers are painted once: ground (asphalt, paint, sidewalks,
   houses) and an overhead layer (tree canopies, poles, tangled wires,
   umbrellas) that is drawn above the traffic.
   ====================================================================== */
"use strict";
let PPM = 14;   // pixels per metre in the pre-rendered layers (set per device)
const mkCanvas = (w, h) => { const c = document.createElement("canvas"); c.width = w; c.height = h; return c; };
const pick = (r, arr) => arr[Math.floor(r() * arr.length)];
const shade = (hex, f) => { const n = parseInt(hex.slice(1), 16); let r = n >> 16 & 255, g = n >> 8 & 255, b = n & 255; r = Math.max(0, Math.min(255, Math.round(r * f))); g = Math.max(0, Math.min(255, Math.round(g * f))); b = Math.max(0, Math.min(255, Math.round(b * f))); return `rgb(${r},${g},${b})`; };

/* ---------------- palettes */
const MOTO_COLORS = ["#1c1c1e", "#1c1c1e", "#b3141c", "#e9e6e0", "#e9e6e0", "#20509f", "#8b9097", "#2e6a3b", "#d6a01a", "#58307f", "#6b1d22"];
const SHIRTS = ["#f2f2ef", "#2a2a2c", "#7a7f86", "#3a5f9e", "#b8323a", "#e2c23d", "#e79ab6", "#9fd0ef", "#c8b6e6", "#2f7a55", "#d9772e", "#f5d0dc", "#a9d9c8"];
const HELMETS = ["#f4f4f1", "#c9252c", "#2a62b8", "#f1c42f", "#232325", "#ef8fb6", "#8f959c", "#ef7f2a", "#7bc4e8"];
const CAR_COLORS = ["#f1f0ec", "#f1f0ec", "#1f2124", "#a7adb4", "#7b2128", "#24426e", "#d8d3c8", "#3b4a3f"];
const TAXI_COLORS = ["#f4f4f0", "#27a35a", "#2c78c9", "#f0f0ea"];
const AWNINGS = [["#2f8f5b", "#e9e6dc"], ["#2b5fa8", "#e9e6dc"], ["#c63b33", "#efe7da"], ["#e69a28", "#f3ead8"], ["#7a3f8f", "#e9e1ee"], ["#d64f8a", "#f7e3ec"]];
const TON = ["#7d91a6", "#a4523f", "#9aa0a5", "#4e7095", "#6e7f5c", "#8c4f3c", "#b7bcc0", "#5f8aa8"];

/* ---------------- vehicles (drawn in metres, centred, facing +x) */
function drawMotoBody(g, o) {
  // tyres
  g.fillStyle = "#18181a"; g.fillRect(-0.96, -0.08, 0.36, 0.16); g.fillRect(0.6, -0.08, 0.36, 0.16);
  // body and seat
  g.fillStyle = o.body; roundRect(g, -0.64, -0.18, 1.3, 0.36, 0.15); g.fill();
  g.fillStyle = "rgba(255,255,255,.18)"; roundRect(g, -0.5, -0.1, 1.0, 0.07, 0.03); g.fill();
  g.fillStyle = "#232326"; roundRect(g, -0.5, -0.14, 0.62, 0.28, 0.1); g.fill();
  // front fairing, headlight, handlebar with mirrors
  g.fillStyle = shade(o.body, 1.15); roundRect(g, 0.46, -0.16, 0.28, 0.32, 0.1); g.fill();
  g.fillStyle = "#f6f1d8"; g.fillRect(0.7, -0.06, 0.06, 0.12);
  g.strokeStyle = "#2a2a2c"; g.lineWidth = 0.05; g.beginPath(); g.moveTo(0.42, -0.34); g.lineTo(0.42, 0.34); g.stroke();
  g.fillStyle = "#3a3a3c"; g.beginPath(); g.arc(0.45, -0.37, 0.05, 0, 6.3); g.arc(0.45, 0.37, 0.05, 0, 6.3); g.fill();
}
function drawRider(g, x, shirt, helmet, hat) {
  g.fillStyle = shirt;
  g.beginPath(); g.ellipse(x, 0, 0.2, 0.29, 0, 0, 6.3); g.fill();
  g.strokeStyle = shirt; g.lineWidth = 0.1; g.lineCap = "round";
  if (x > -0.2) { g.beginPath(); g.moveTo(x + 0.05, -0.24); g.lineTo(0.4, -0.31); g.moveTo(x + 0.05, 0.24); g.lineTo(0.4, 0.31); g.stroke(); }
  if (hat) {   // nón lá: a straw cone seen from above
    g.fillStyle = "#e7d6a3"; g.beginPath(); g.arc(x + 0.02, 0, 0.27, 0, 6.3); g.fill();
    g.strokeStyle = "rgba(120,95,40,.45)"; g.lineWidth = 0.02; g.beginPath(); g.arc(x + 0.02, 0, 0.18, 0, 6.3); g.stroke(); g.beginPath(); g.arc(x + 0.02, 0, 0.09, 0, 6.3); g.stroke();
  } else {
    g.fillStyle = helmet; g.beginPath(); g.arc(x + 0.03, 0, 0.145, 0, 6.3); g.fill();
    g.fillStyle = "rgba(255,255,255,.35)"; g.beginPath(); g.arc(x - 0.01, -0.04, 0.05, 0, 6.3); g.fill();
    g.fillStyle = "rgba(20,25,30,.55)"; g.beginPath(); g.arc(x + 0.03, 0, 0.145, -0.9, 0.9); g.lineTo(x + 0.03, 0); g.fill();
  }
}
function drawMoto(g, v, parked) {
  const r = v.look;
  drawMotoBody(g, { body: r.body });
  if (v.cargo === "box") { g.fillStyle = "#b8925d"; g.fillRect(-0.98, -0.26, 0.42, 0.52); g.strokeStyle = "rgba(80,55,25,.6)"; g.lineWidth = 0.025; g.strokeRect(-0.98, -0.26, 0.42, 0.52); g.beginPath(); g.moveTo(-0.77, -0.26); g.lineTo(-0.77, 0.26); g.stroke(); }
  if (v.cargo === "gas") { g.fillStyle = "#2c6fb8"; for (const y of [-0.17, 0.17]) { g.beginPath(); g.arc(-0.78, y, 0.16, 0, 6.3); g.fill(); } g.fillStyle = "#cfd6dc"; for (const y of [-0.17, 0.17]) { g.beginPath(); g.arc(-0.78, y, 0.05, 0, 6.3); g.fill(); } }
  if (v.delivery) { g.fillStyle = r.box; roundRect(g, -0.98, -0.24, 0.44, 0.48, 0.05); g.fill(); g.fillStyle = "rgba(255,255,255,.25)"; g.fillRect(-0.94, -0.2, 0.36, 0.08); }
  if (parked) return;
  if (v.passenger) drawRider(g, -0.44, r.shirt2, r.helmet2, r.hat2);
  drawRider(g, -0.08, v.delivery ? r.jacket : r.shirt, r.helmet, false);
}
function drawBike(g, v) {
  g.fillStyle = "#18181a"; g.fillRect(-0.85, -0.04, 0.4, 0.08); g.fillRect(0.45, -0.04, 0.4, 0.08);
  g.strokeStyle = v.look.body; g.lineWidth = 0.07; g.beginPath(); g.moveTo(-0.6, 0); g.lineTo(0.55, 0); g.stroke();
  g.strokeStyle = "#2a2a2c"; g.lineWidth = 0.04; g.beginPath(); g.moveTo(0.42, -0.26); g.lineTo(0.42, 0.26); g.stroke();
  if (v.cargo) { g.fillStyle = "#9bbf4a"; g.beginPath(); g.arc(-0.7, 0, 0.3, 0, 6.3); g.fill(); g.fillStyle = "#c9a14c"; g.beginPath(); g.arc(-0.7, 0, 0.18, 0, 6.3); g.fill(); }
  drawRider(g, -0.1, v.look.shirt, v.look.helmet, v.look.hatRider);
}
function drawCar(g, v) {
  const L = v.T.l, W = v.T.w, c = v.look.body, taxi = v.type === "taxi";
  g.fillStyle = c; roundRect(g, -L / 2, -W / 2, L, W, 0.55); g.fill();
  g.fillStyle = "rgba(255,255,255,.14)"; roundRect(g, -L / 2 + 0.3, -W / 2 + 0.25, L - 0.6, W * 0.22, 0.2); g.fill();
  g.fillStyle = "rgba(0,0,0,.12)"; roundRect(g, -L / 2 + 0.3, W / 2 - 0.4, L - 0.6, 0.18, 0.1); g.fill();
  // glass: windshield, roof, rear window
  const glass = "#1c2731";
  g.fillStyle = glass; g.beginPath(); g.moveTo(L * 0.12, -W / 2 + 0.2); g.lineTo(L * 0.27, -W / 2 + 0.3); g.lineTo(L * 0.27, W / 2 - 0.3); g.lineTo(L * 0.12, W / 2 - 0.2); g.closePath(); g.fill();
  g.fillStyle = "rgba(160,200,230,.25)"; g.beginPath(); g.moveTo(L * 0.14, -W / 2 + 0.3); g.lineTo(L * 0.22, -W / 2 + 0.36); g.lineTo(L * 0.18, 0); g.lineTo(L * 0.14, 0); g.closePath(); g.fill();
  g.fillStyle = shade(c, 1.06); roundRect(g, -L * 0.25, -W / 2 + 0.22, L * 0.37, W - 0.44, 0.25); g.fill();
  g.fillStyle = glass; g.beginPath(); g.moveTo(-L * 0.25, -W / 2 + 0.24); g.lineTo(-L * 0.36, -W / 2 + 0.32); g.lineTo(-L * 0.36, W / 2 - 0.32); g.lineTo(-L * 0.25, W / 2 - 0.24); g.closePath(); g.fill();
  if (taxi) { g.fillStyle = "#f6d23a"; roundRect(g, -L * 0.08, -0.25, 0.45, 0.5, 0.08); g.fill(); g.fillStyle = "rgba(0,0,0,.35)"; g.fillRect(-L * 0.08 + 0.12, -0.18, 0.2, 0.36); }
  // mirrors and lamps
  g.fillStyle = shade(c, 0.8); g.fillRect(L * 0.16, -W / 2 - 0.12, 0.16, 0.12); g.fillRect(L * 0.16, W / 2, 0.16, 0.12);
  g.fillStyle = "#fbf6dc"; g.fillRect(L / 2 - 0.12, -W / 2 + 0.15, 0.1, 0.32); g.fillRect(L / 2 - 0.12, W / 2 - 0.47, 0.1, 0.32);
  g.fillStyle = v.braking ? "#ff2a2a" : "#8e1d22"; g.fillRect(-L / 2 + 0.02, -W / 2 + 0.15, 0.1, 0.34); g.fillRect(-L / 2 + 0.02, W / 2 - 0.49, 0.1, 0.34);
}
function drawBus(g, v) {
  const L = v.T.l, W = v.T.w, c = v.look.body;
  g.fillStyle = c; roundRect(g, -L / 2, -W / 2, L, W, 0.35); g.fill();
  g.fillStyle = v.look.stripe; g.fillRect(-L / 2 + 0.2, -W / 2, L - 0.4, 0.22); g.fillRect(-L / 2 + 0.2, W / 2 - 0.22, L - 0.4, 0.22);
  g.fillStyle = "#1c2731"; g.fillRect(L / 2 - 0.55, -W / 2 + 0.15, 0.45, W - 0.3);
  g.fillStyle = "#d9dcdf"; roundRect(g, -1.4, -0.7, 2.6, 1.4, 0.2); g.fill();
  g.fillStyle = "#b5babf"; for (const x of [-0.8, 0.5]) { g.beginPath(); g.arc(x, 0, 0.42, 0, 6.3); g.fill(); }
  g.fillStyle = "rgba(0,0,0,.12)"; for (const x of [-3.6, 2.6]) g.fillRect(x, -0.45, 0.9, 0.9);
  g.fillStyle = v.braking ? "#ff2a2a" : "#8e1d22"; g.fillRect(-L / 2, -W / 2 + 0.15, 0.1, 0.4); g.fillRect(-L / 2, W / 2 - 0.55, 0.1, 0.4);
}
function drawTruck(g, v) {
  const L = v.T.l, W = v.T.w;
  g.fillStyle = v.look.cargo; roundRect(g, -L / 2, -W / 2, L - 1.9, W, 0.15); g.fill();
  g.strokeStyle = "rgba(0,0,0,.18)"; g.lineWidth = 0.05; for (let x = -L / 2 + 0.6; x < L / 2 - 2; x += 0.7) { g.beginPath(); g.moveTo(x, -W / 2 + 0.1); g.lineTo(x, W / 2 - 0.1); g.stroke(); }
  g.fillStyle = v.look.body; roundRect(g, L / 2 - 1.8, -W / 2 + 0.05, 1.8, W - 0.1, 0.35); g.fill();
  g.fillStyle = "#1c2731"; g.fillRect(L / 2 - 0.75, -W / 2 + 0.25, 0.4, W - 0.5);
  g.fillStyle = v.braking ? "#ff2a2a" : "#8e1d22"; g.fillRect(-L / 2, -W / 2 + 0.15, 0.1, 0.35); g.fillRect(-L / 2, W / 2 - 0.5, 0.1, 0.35);
}
function roundRect(g, x, y, w, h, r) { g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r); g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath(); }
function lookFor(v) {
  if (v.look) return v.look;
  const r = rngFrom(v.id * 131 + 7), ty = v.type;
  const L = { body: pick(r, ty === "taxi" ? TAXI_COLORS : ty === "car" ? CAR_COLORS : MOTO_COLORS), shirt: pick(r, SHIRTS), shirt2: pick(r, SHIRTS), helmet: pick(r, HELMETS), helmet2: pick(r, HELMETS), hat2: r() < 0.3, hatRider: r() < 0.5,
    jacket: pick(r, ["#2aa356", "#2aa356", "#e8702a", "#d63a3a", "#1f8fd6"]), box: pick(r, ["#2aa356", "#e8702a", "#d63a3a", "#1f8fd6", "#2b2b2b"]) };
  if (v.ego) L.body = "#f5f7f6";
  if (ty === "bus") { L.body = pick(r, ["#f3efe4", "#e9e2c9", "#2f7f57"]); L.stripe = pick(r, ["#2f7f57", "#c9302c", "#1f5fa8"]); }
  if (ty === "truck") { L.body = pick(r, ["#2c5fa3", "#d8d6cf", "#c23b2f", "#3d7a46"]); L.cargo = pick(r, ["#2e6c8f", "#7a8a4a", "#b9a27a", "#8d3f2e"]); }
  return v.look = L;
}
function drawVehicle(g, v, x, y, h, t) {
  lookFor(v);
  g.save(); g.translate(x, y); g.rotate(h);
  if (v.ego) {   // a soft teal halo so the learning car is easy to find in the swarm
    g.fillStyle = "rgba(25,190,175,.22)"; roundRect(g, -v.T.l / 2 - 0.6, -v.T.w / 2 - 0.6, v.T.l + 1.2, v.T.w + 1.2, 1.1); g.fill();
    g.strokeStyle = "rgba(25,190,175,.9)"; g.lineWidth = 0.08; roundRect(g, -v.T.l / 2 - 0.6, -v.T.w / 2 - 0.6, v.T.l + 1.2, v.T.w + 1.2, 1.1); g.stroke();
  }
  if (v.type === "moto") drawMoto(g, v, false);
  else if (v.type === "bike") drawBike(g, v);
  else if (v.type === "bus") drawBus(g, v);
  else if (v.type === "truck") drawTruck(g, v);
  else drawCar(g, v);
  if (v.ego) drawEgoKit(g, v);
  const L = v.T.l;
  if ((v.type === "moto" || v.type === "bike") && v.braking) { g.fillStyle = "#ff3030"; g.fillRect(-L / 2 - 0.06, -0.06, 0.08, 0.12); }
  if (v.signal && v.s > v.path.sIn - 25 && v.s < v.path.sOut && Math.floor(t * 2.6) % 2 === 0) {
    g.fillStyle = "#ffb11c"; const side = v.path.kind === "left" ? -1 : 1, w = v.T.w / 2;
    g.beginPath(); g.arc(L / 2 - 0.1, side * w, 0.11, 0, 6.3); g.arc(-L / 2 + 0.1, side * w, 0.11, 0, 6.3); g.fill();
  }
  g.restore();
}
function drawEgoKit(g, v) {
  const L = v.T.l, W = v.T.w;
  g.fillStyle = "#16b5a6"; g.fillRect(-L * 0.25, -W / 2 + 0.22, L * 0.37, 0.12); g.fillRect(-L * 0.25, W / 2 - 0.34, L * 0.37, 0.12);
  // roof lidar puck
  g.fillStyle = "rgba(0,0,0,.25)"; g.beginPath(); g.arc(-0.2, 0.06, 0.36, 0, 6.3); g.fill();
  g.fillStyle = "#24292d"; g.beginPath(); g.arc(-0.25, 0, 0.34, 0, 6.3); g.fill();
  g.strokeStyle = "#36e0cf"; g.lineWidth = 0.06; g.beginPath(); g.arc(-0.25, 0, 0.25, 0, 6.3); g.stroke();
  g.fillStyle = "#5a6066"; g.beginPath(); g.arc(-0.25, 0, 0.12, 0, 6.3); g.fill();
  // corner sensors
  g.fillStyle = "#24292d"; for (const [sx, sy] of [[L / 2 - 0.35, -W / 2 + 0.1], [L / 2 - 0.35, W / 2 - 0.1], [-L / 2 + 0.35, -W / 2 + 0.1], [-L / 2 + 0.35, W / 2 - 0.1]]) { g.beginPath(); g.arc(sx, sy, 0.1, 0, 6.3); g.fill(); }
}
function drawShadow(g, v, x, y, h) {
  g.save(); g.translate(x + 0.22, y + 0.3); g.rotate(h);
  g.fillStyle = "rgba(10,14,18,.26)";
  if (v.type === "moto" || v.type === "bike") { g.beginPath(); g.ellipse(0, 0, v.T.l / 2 + 0.05, 0.42, 0, 0, 6.3); g.fill(); }
  else { roundRect(g, -v.T.l / 2, -v.T.w / 2, v.T.l, v.T.w, 0.5); g.fill(); }
  g.restore();
}


/* ---------------- roofs, awnings, tree canopies */
function drawRoof(g, b, r) {
  const { x, y, w, h } = b, front = b.front || null;
  const along = front === "n" || front === "s" ? "y" : "x";   // ridges run front to back
  if (b.style === "ton") {
    g.fillStyle = b.c; g.fillRect(x, y, w, h);
    g.lineWidth = 0.06;
    if (along === "y") for (let k = x + 0.12; k < x + w; k += 0.24) { g.strokeStyle = (Math.round((k - x) / 0.24) % 2) ? "rgba(255,255,255,.16)" : "rgba(0,0,0,.14)"; g.beginPath(); g.moveTo(k, y); g.lineTo(k, y + h); g.stroke(); }
    else for (let k = y + 0.12; k < y + h; k += 0.24) { g.strokeStyle = (Math.round((k - y) / 0.24) % 2) ? "rgba(255,255,255,.16)" : "rgba(0,0,0,.14)"; g.beginPath(); g.moveTo(x, k); g.lineTo(x + w, k); g.stroke(); }
    if (r() < 0.4) { g.fillStyle = "rgba(140,70,35,.25)"; g.beginPath(); g.ellipse(x + r() * w, y + r() * h, 0.6 + r(), 0.4 + r(), r() * 3, 0, 6.3); g.fill(); }
    g.fillStyle = "rgba(0,0,0,.12)"; if (along === "y") g.fillRect(x, y + h * 0.5 - 0.05, w, 0.1); else g.fillRect(x + w * 0.5 - 0.05, y, 0.1, h);
  } else if (b.style === "tile") {
    g.fillStyle = "#b5623f"; g.fillRect(x, y, w, h);
    g.fillStyle = "rgba(0,0,0,.1)"; if (along === "y") g.fillRect(x + w / 2, y, w / 2, h); else g.fillRect(x, y + h / 2, w, h / 2);
    g.strokeStyle = "rgba(70,30,15,.25)"; g.lineWidth = 0.04;
    if (along === "y") for (let k = y + 0.25; k < y + h; k += 0.3) { g.beginPath(); g.moveTo(x, k); g.lineTo(x + w, k); g.stroke(); }
    else for (let k = x + 0.25; k < x + w; k += 0.3) { g.beginPath(); g.moveTo(k, y); g.lineTo(k, y + h); g.stroke(); }
    g.strokeStyle = "#7c3a22"; g.lineWidth = 0.14; g.beginPath(); if (along === "y") { g.moveTo(x + w / 2, y); g.lineTo(x + w / 2, y + h); } else { g.moveTo(x, y + h / 2); g.lineTo(x + w, y + h / 2); } g.stroke();
  } else {
    // flat concrete roof: parapet, stains, water tank (inox), solar heater, AC units, plants
    g.fillStyle = shade("#cdc9c0", 0.92 + r() * 0.12); g.fillRect(x, y, w, h);
    g.strokeStyle = "rgba(255,255,255,.35)"; g.lineWidth = 0.14; g.strokeRect(x + 0.08, y + 0.08, w - 0.16, h - 0.16);
    for (let k = 0; k < 4; k++) { g.fillStyle = `rgba(60,55,45,${0.05 + r() * 0.07})`; g.beginPath(); g.ellipse(x + r() * w, y + r() * h, 0.4 + r() * 1.2, 0.3 + r() * 0.8, r() * 3, 0, 6.3); g.fill(); }
    const tanks = 1 + (r() < 0.35 ? 1 : 0);
    for (let k = 0; k < tanks; k++) {
      const tx = x + 0.9 + r() * Math.max(0.1, w - 1.8), ty = y + 0.9 + r() * Math.max(0.1, h - 1.8), tr = 0.5 + r() * 0.2;
      g.fillStyle = "rgba(0,0,0,.25)"; g.beginPath(); g.arc(tx + 0.25, ty + 0.3, tr, 0, 6.3); g.fill();
      if (r() < 0.8) { const gr = g.createRadialGradient(tx - tr * 0.3, ty - tr * 0.3, tr * 0.1, tx, ty, tr); gr.addColorStop(0, "#f7f9fa"); gr.addColorStop(0.6, "#c9d0d6"); gr.addColorStop(1, "#8f99a1"); g.fillStyle = gr; }
      else g.fillStyle = "#2f6fb5";
      g.beginPath(); g.arc(tx, ty, tr, 0, 6.3); g.fill(); g.strokeStyle = "rgba(60,70,78,.6)"; g.lineWidth = 0.04; g.beginPath(); g.arc(tx, ty, tr * 0.4, 0, 6.3); g.stroke();
    }
    if (r() < 0.45) { const sx = x + 0.4 + r() * Math.max(0.1, w - 2), sy = y + 0.4 + r() * Math.max(0.1, h - 1.6); g.fillStyle = "#24365a"; g.fillRect(sx, sy, 1.5, 1.0); g.strokeStyle = "rgba(180,200,230,.35)"; g.lineWidth = 0.03; for (let k = 0.1; k < 1.5; k += 0.12) { g.beginPath(); g.moveTo(sx + k, sy); g.lineTo(sx + k, sy + 1); g.stroke(); } }
    const ac = Math.floor(r() * 3); for (let k = 0; k < ac; k++) { const ax = x + 0.3 + r() * Math.max(0.1, w - 1), ay = y + 0.3 + r() * Math.max(0.1, h - 0.8); g.fillStyle = "#e8e8e4"; g.fillRect(ax, ay, 0.7, 0.45); g.strokeStyle = "rgba(0,0,0,.35)"; g.lineWidth = 0.03; g.beginPath(); g.arc(ax + 0.35, ay + 0.22, 0.16, 0, 6.3); g.stroke(); }
    if (r() < 0.5) for (let k = 0; k < 3 + r() * 5; k++) { const px_ = x + 0.4 + r() * (w - 0.8), py_ = y + 0.4 + r() * (h - 0.8); g.fillStyle = pick(r, ["#4f8a3c", "#6aa34a", "#3b6e2f"]); g.beginPath(); g.arc(px_, py_, 0.2 + r() * 0.25, 0, 6.3); g.fill(); }
    if (r() < 0.25) { const ly = y + h * (0.3 + r() * 0.4); g.strokeStyle = "rgba(60,60,60,.5)"; g.lineWidth = 0.02; g.beginPath(); g.moveTo(x + 0.3, ly); g.lineTo(x + w - 0.3, ly); g.stroke(); for (let k = x + 0.5; k < x + w - 0.5; k += 0.35) { g.fillStyle = pick(r, SHIRTS); g.fillRect(k, ly, 0.25, 0.3); } }
  }
  g.strokeStyle = "rgba(20,20,22,.35)"; g.lineWidth = 0.05; g.strokeRect(x, y, w, h);
}
function drawAwning(g, b, r) {
  const [c1, c2] = pick(r, AWNINGS), depth = 1.1 + r() * 0.7, f = b.front;
  const horiz = f === "n" || f === "s";
  const ax = horiz ? b.x : f === "w" ? b.x - depth : b.x + b.w, ay = horiz ? (f === "n" ? b.y - depth : b.y + b.h) : b.y;
  const aw = horiz ? b.w : depth, ah = horiz ? depth : b.h;
  g.fillStyle = "rgba(0,0,0,.18)"; g.fillRect(ax + 0.35, ay + 0.45, aw, ah);
  const n = Math.max(2, Math.round((horiz ? aw : ah) / 0.5));
  for (let k = 0; k < n; k++) { g.fillStyle = k % 2 ? c2 : c1; if (horiz) g.fillRect(ax + k * aw / n, ay, aw / n + 0.01, ah); else g.fillRect(ax, ay + k * ah / n, aw, ah / n + 0.01); }
  g.fillStyle = "rgba(0,0,0,.12)"; if (horiz) g.fillRect(ax, f === "n" ? ay : ay + ah - 0.12, aw, 0.12); else g.fillRect(f === "w" ? ax : ax + aw - 0.12, ay, 0.12, ah);
}
function drawCanopy(o, t, r) {
  const n = 6 + Math.floor(r() * 4);
  const base = pick(r, [[58, 104, 50], [66, 112, 52], [52, 96, 48], [74, 118, 58]]);
  for (let k = 0; k < n; k++) {
    const a = r() * 6.28, d = r() * t.r * 0.55, cr = t.r * (0.45 + r() * 0.35), cx = t.x + Math.cos(a) * d, cy = t.y + Math.sin(a) * d;
    const gr = o.createRadialGradient(cx - cr * 0.35, cy - cr * 0.35, cr * 0.1, cx, cy, cr);
    gr.addColorStop(0, `rgba(${base[0] + 40},${base[1] + 45},${base[2] + 25},.97)`); gr.addColorStop(0.7, `rgba(${base[0]},${base[1]},${base[2]},.96)`); gr.addColorStop(1, `rgba(${base[0] - 22},${base[1] - 30},${base[2] - 20},.94)`);
    o.fillStyle = gr; o.beginPath(); o.arc(cx, cy, cr, 0, 6.3); o.fill();
  }
  for (let k = 0; k < 40; k++) { const a = r() * 6.28, d = r() * t.r * 0.9; o.fillStyle = r() < 0.5 ? "rgba(140,190,90,.35)" : "rgba(20,45,18,.3)"; o.beginPath(); o.arc(t.x + Math.cos(a) * d, t.y + Math.sin(a) * d, 0.12 + r() * 0.15, 0, 6.3); o.fill(); }
}
/* ---------------- the city: features are laid out once in world metres; tiles are painted on demand */
function drawShadowStatic(g, l, w) { g.fillStyle = "rgba(10,14,18,.2)"; g.beginPath(); g.ellipse(0.15, 0.2, l / 2, w * 0.6, 0, 0, 6.3); g.fill(); }
const SUN = [0.62, 0.78];   // shadows fall to the south-east

function cityFeatures(map, seed = 11) {
  const r = rngFrom(seed * 131 + 7), sw = map.sidewalk, R = map.cornerR, B = map.bounds;
  const F = { roads: [], walks: [], fillets: [], curbs: [], arcs: [], marks: [], strips: [], houses: [], trees: [], poles: [], props: [], umbrellas: [], wires: [], patches: [], heads: [] };
  const rect = (x0, y0, x1, y1, extra) => Object.assign({ x0: Math.min(x0, x1), y0: Math.min(y0, y1), x1: Math.max(x0, x1), y1: Math.max(y0, y1) }, extra || {});
  const toEdge = (jn, a) => { const d = a.dir; return d[0] > 0 ? B.x1 - jn.x + 2 : d[0] < 0 ? jn.x - B.x0 + 2 : d[1] > 0 ? B.y1 - jn.y + 2 : jn.y - B.y0 + 2; };
  for (const jn of map.juncs) {
    const hx = jn.hwY, hy = jn.hwX;   // half-size of the junction box along x and along y
    F.roads.push(rect(jn.x - hx, jn.y - hy, jn.x + hx, jn.y + hy)); F.walks.push(rect(jn.x - hx - sw, jn.y - hy - sw, jn.x + hx + sw, jn.y + hy + sw));
    for (const a of Object.values(jn.arms)) {
      const d = a.dir, len = a.to ? a.len : toEdge(jn, a), n = [-d[1], d[0]];
      const p0 = [jn.x + d[0] * a.box, jn.y + d[1] * a.box], p1 = [jn.x + d[0] * len, jn.y + d[1] * len];
      F.roads.push(rect(p0[0] - n[0] * a.hw, p0[1] - n[1] * a.hw, p1[0] + n[0] * a.hw, p1[1] + n[1] * a.hw));
      F.walks.push(rect(p0[0] - n[0] * (a.hw + sw), p0[1] - n[1] * (a.hw + sw), p1[0] + n[0] * (a.hw + sw), p1[1] + n[1] * (a.hw + sw)));
      // curbs and sidewalk strips on both sides; a side starts after the corner fillet if the perpendicular arm exists
      for (const s of [1, -1]) {
        const out = [n[0] * s, n[1] * s], perp = Object.values(jn.arms).find((q) => q.dir[0] === out[0] && q.dir[1] === out[1]);
        const u0 = a.box + (perp ? R : 0), c0 = [jn.x + d[0] * u0 + out[0] * a.hw, jn.y + d[1] * u0 + out[1] * a.hw], c1 = [jn.x + d[0] * len + out[0] * a.hw, jn.y + d[1] * len + out[1] * a.hw];
        F.curbs.push({ x0: c0[0], y0: c0[1], x1: c1[0], y1: c1[1] });
        F.strips.push({ jn, arm: a, d, out, u0: a.box + (perp ? R : 2), u1: len, hw: a.hw, cls: a.cls, axis: d[0] ? "x" : "y", toCity: !!a.to, perp: !!perp });
        if (!perp) {   // T-junction: the box side without an arm is a straight curb
          const e = [jn.x + out[0] * (out[0] ? hx : 0) + d[0] * 0, jn.y + out[1] * (out[1] ? hy : 0)];
          const half = out[0] ? hy : hx, t = [Math.abs(out[1]), Math.abs(out[0])];
          F.curbs.push({ x0: e[0] - t[0] * half, y0: e[1] - t[1] * half, x1: e[0] + t[0] * half, y1: e[1] + t[1] * half });
        }
      }
      // markings for traffic coming in on this arm
      F.marks.push({ jn, a, len });
      if (jn.type === "signal") {
        const app = [-n[0], -n[1]];   // the incoming half of the road (driver's right)
        const along = a.box + map.stop + 0.6, side = a.hw + 1.4;
        F.heads.push({ j: jn.id, arm: a.id, x: jn.x + d[0] * along + app[0] * side, y: jn.y + d[1] * along + app[1] * side, ang: Math.atan2(d[1], d[0]) });
        F.heads.push({ j: jn.id, arm: a.id, x: jn.x - d[0] * (a.box + 1.6) + app[0] * (a.hw + 1.4), y: jn.y - d[1] * (a.box + 1.6) + app[1] * (a.hw + 1.4), ang: Math.atan2(d[1], d[0]), far: true });
      }
    }
    // corner fillets between perpendicular arms
    for (const [ax, ay] of [["E", "N"], ["E", "S"], ["W", "N"], ["W", "S"]]) {
      if (!jn.arms[ax] || !jn.arms[ay]) continue;
      const sx = ax === "E" ? 1 : -1, sy = ay === "S" ? 1 : -1, px = jn.x + sx * hx, py = jn.y + sy * hy;
      F.fillets.push({ px, py, cx: px + sx * R, cy: py + sy * R, sx, sy });
      F.arcs.push({ cx: px + sx * R, cy: py + sy * R, sx, sy });
    }
  }
  // asphalt patches and manholes
  for (const rd of F.roads) {
    const area = (rd.x1 - rd.x0) * (rd.y1 - rd.y0), k = Math.round(area / 260);
    for (let i = 0; i < k; i++) { const w = 1 + r() * 3, h = 0.8 + r() * 2.2; F.patches.push({ x: rd.x0 + r() * Math.max(0, rd.x1 - rd.x0 - w), y: rd.y0 + r() * Math.max(0, rd.y1 - rd.y0 - h), w, h, kind: r() < 0.18 ? "hole" : r() < 0.5 ? "dark" : "light" }); }
  }
  // houses along every sidewalk; rows along x-axis roads own the corners
  const occ = new Map(), cell = 10;
  const occKey = (x, y) => Math.floor(x / cell) * 4096 + Math.floor(y / cell);
  const addOcc = (h) => { for (let x = Math.floor(h.x / cell); x <= Math.floor((h.x + h.w) / cell); x++) for (let y = Math.floor(h.y / cell); y <= Math.floor((h.y + h.h) / cell); y++) { const k = x * 4096 + y; (occ.get(k) || occ.set(k, []).get(k)).push(h); } };
  const free = (x, y, w, h, pad = 0.3) => {
    for (const q of F.walks) if (x < q.x1 + pad && x + w > q.x0 - pad && y < q.y1 + pad && y + h > q.y0 - pad) return false;
    for (const f of F.fillets) {   // inside a rounded corner only the land within the inner arc is free
      const fx0 = Math.min(f.px, f.cx), fy0 = Math.min(f.py, f.cy), ix0 = Math.max(x, fx0), iy0 = Math.max(y, fy0), ix1 = Math.min(x + w, fx0 + R), iy1 = Math.min(y + h, fy0 + R);
      if (ix0 >= ix1 || iy0 >= iy1) continue;
      for (const [qx, qy] of [[ix0, iy0], [ix1, iy0], [ix0, iy1], [ix1, iy1]]) if (Math.hypot(qx - f.cx, qy - f.cy) > R - sw - 0.2) return false;
    }
    for (let gx = Math.floor(x / cell); gx <= Math.floor((x + w) / cell); gx++) for (let gy = Math.floor(y / cell); gy <= Math.floor((y + h) / cell); gy++) for (const q of occ.get(gx * 4096 + gy) || []) if (x < q.x + q.w + 0.05 && x + w > q.x - 0.05 && y < q.y + q.h + 0.05 && y + h > q.y - 0.05) return false;
    return x > B.x0 - 30 && x + w < B.x1 + 30 && y > B.y0 - 30 && y + h < B.y1 + 30;
  };
  const mkHouse = (x, y, w, h, front, face) => ({ x, y, w, h, front, fl: 2 + Math.floor(r() * 5), style: r() < 0.45 ? "ton" : r() < 0.82 ? "flat" : "tile", c: pick(r, TON), awning: !!front && r() < 0.55, seed: Math.floor(r() * 1e9), face });
  const frontOf = (out) => out[1] > 0 ? "n" : out[1] < 0 ? "s" : out[0] > 0 ? "w" : "e";   // the house faces back toward the road
  for (const pass of ["x", "y"]) for (const st of F.strips) {
    if (st.axis !== pass) continue;
    const jn = st.jn, d = st.d, out = st.out, base = st.hw + sw;
    let u = pass === "x" ? jn.hwY + sw + 0.6 : jn.hwX + sw + 0.6;
    while (u < st.u1 - 0.5) {
      const w = 3.6 + r() * 1.9, depth = 12 + r() * 7;
      const ua = u, ub = Math.min(st.u1, u + w - 0.08);
      const p0 = [jn.x + d[0] * ua + out[0] * base, jn.y + d[1] * ua + out[1] * base], p1 = [jn.x + d[0] * ub + out[0] * (base + depth), jn.y + d[1] * ub + out[1] * (base + depth)];
      const x = Math.min(p0[0], p1[0]), y = Math.min(p0[1], p1[1]), hw_ = Math.abs(p1[0] - p0[0]), hh = Math.abs(p1[1] - p0[1]);
      if (ub - ua > 2.5 && free(x, y, hw_, hh, 0.05)) { const h = mkHouse(x, y, hw_, hh, frontOf(out), pass); F.houses.push(h); addOcc(h); }
      u += w;
    }
  }
  // the rest of every block: back houses on a jittered grid, leaving alleys
  for (let gx = B.x0 - 20; gx < B.x1 + 20; gx += 9 + r() * 3) for (let gy = B.y0 - 20; gy < B.y1 + 20; gy += 8 + r() * 4) {
    if (r() < 0.08) continue;
    const w = 6 + r() * 3.5, h = 5.5 + r() * 4, x = gx + r() * 1.2, y = gy + r() * 1.2;
    if (free(x, y, w, h, 1.0)) { const hs = mkHouse(x, y, w, h, null, "back"); F.houses.push(hs); addOcc(hs); }
  }
  // sidewalk life: trees, poles, parked motorbikes, tea stalls, carts
  const sp = (st, u, depth) => [st.jn.x + st.d[0] * u + st.out[0] * (st.hw + depth), st.jn.y + st.d[1] * u + st.out[1] * (st.hw + depth)];
  const ang = (v) => Math.atan2(v[1], v[0]);
  for (const st of F.strips) {
    const L = st.u1 - st.u0;
    if (L < 4) continue;
    for (let u = st.u0 + 2 + r() * 6; u < st.u1 - 2; u += 11 + r() * 7) { const [x, y] = sp(st, u, 0.9 + r() * 0.5); F.trees.push({ x, y, r: 2.1 + r() * 1.7, seed: Math.floor(r() * 1e9) }); }
    const line = [];
    for (let u = st.u0 + 4; u < st.u1 - 1; u += 24 + r() * 6) { const [x, y] = sp(st, u, 0.45); const p = { x, y, seed: Math.floor(r() * 1e9) }; F.poles.push(p); line.push(p); }
    for (let i = 0; i + 1 < line.length; i++) F.wires.push({ a: line[i], b: line[i + 1], n: 4 + Math.floor(r() * 4), seed: Math.floor(r() * 1e9) });
    if (line.length && r() < 0.35) {   // a bundle across the street
      const p = line[Math.floor(r() * line.length)], q = { x: p.x - st.out[0] * (2 * st.hw + 0.9), y: p.y - st.out[1] * (2 * st.hw + 0.9), seed: Math.floor(r() * 1e9) };
      F.poles.push(q); F.wires.push({ a: p, b: q, n: 9, seed: Math.floor(r() * 1e9), thick: true });
    }
    let u = st.u0 + 1 + r() * 4;
    while (u < st.u1 - 3) {
      const k = r();
      if (k < 0.42) {
        const n = 3 + Math.floor(r() * 9);
        for (let i = 0; i < n && u < st.u1 - 1; i++, u += 0.78) { const [x, y] = sp(st, u, sw - 1.2 + (r() - 0.5) * 0.15); F.props.push({ kind: "moto", x, y, a: ang(st.out) + (r() - 0.5) * 0.25 + 0.35, c: pick(r, MOTO_COLORS) }); }
        u += 1.5 + r() * 3;
      } else if (k < 0.6) {
        const [x, y] = sp(st, u + 1, sw * 0.55);
        F.props.push({ kind: "stall", x, y, a: ang(st.d), seed: Math.floor(r() * 1e9) });
        if (r() < 0.6) F.umbrellas.push({ x, y, r: 1.3 + r() * 0.3, c: pick(r, [["#e14b3b", "#f3eadb"], ["#2f7cc4", "#f1f1ea"], ["#f0b530", "#c9302c"], ["#3a9a5b", "#f2efe4"]]) });
        u += 4 + r() * 3;
      } else if (k < 0.68) { const [x, y] = sp(st, u, sw * 0.4); F.props.push({ kind: "cart", x, y, a: ang(st.d), c: pick(r, ["#e2582c", "#f2c230", "#5fa83a"]) }); u += 3; }
      else u += 2 + r() * 4;
    }
  }
  // bounding boxes for tile culling
  const bb = (o, x0, y0, x1, y1) => { o.bx0 = x0; o.by0 = y0; o.bx1 = x1; o.by1 = y1; return o; };
  for (const h of F.houses) { const off = h.fl * 0.55 + 2.5; bb(h, h.x - 2.5, h.y - 2.5, h.x + h.w + off, h.y + h.h + off); }
  for (const t of F.trees) bb(t, t.x - t.r - 0.5, t.y - t.r - 0.5, t.x + t.r * 1.05 + 1.6, t.y + t.r + 2);
  for (const p of F.props) bb(p, p.x - 1.6, p.y - 1.6, p.x + 1.6, p.y + 1.6);
  for (const u of F.umbrellas) bb(u, u.x - u.r - 0.1, u.y - u.r - 0.1, u.x + u.r + 0.6, u.y + u.r + 0.7);
  for (const p of F.poles) bb(p, p.x - 0.8, p.y - 0.8, p.x + 1, p.y + 1);
  for (const w of F.wires) bb(w, Math.min(w.a.x, w.b.x) - 2, Math.min(w.a.y, w.b.y) - 2, Math.max(w.a.x, w.b.x) + 2, Math.max(w.a.y, w.b.y) + 2);
  for (const p of F.patches) bb(p, p.x, p.y, p.x + p.w, p.y + p.h);
  for (const m of F.marks) { const d = m.a.dir, n = [-d[1], d[0]], j = m.jn; const ex = [j.x + d[0] * (m.a.box - 1), j.y + d[1] * (m.a.box - 1)], fx = [j.x + d[0] * m.len, j.y + d[1] * m.len]; bb(m, Math.min(ex[0], fx[0]) - m.a.hw - 1, Math.min(ex[1], fx[1]) - m.a.hw - 1, Math.max(ex[0], fx[0]) + m.a.hw + 1, Math.max(ex[1], fx[1]) + m.a.hw + 1); }
  return F;
}

const hit = (o, x0, y0, x1, y1) => o.bx0 < x1 && o.bx1 > x0 && o.by0 < y1 && o.by1 > y0;
const rhit = (q, x0, y0, x1, y1) => q.x0 < x1 && q.x1 > x0 && q.y0 < y1 && q.y1 > y0;

/* paint one tile [x0, x0+S) x [y0, y0+S) of the ground layer and the overhead layer */
function paintTile(map, F, x0, y0, S, ppm, lite) {
  const px = Math.round(S * ppm), ground = mkCanvas(px, px), over = mkCanvas(px, px);
  const g = ground.getContext("2d"), o = over.getContext("2d");
  for (const c of [g, o]) c.setTransform(ppm, 0, 0, ppm, -x0 * ppm, -y0 * ppm);
  const x1 = x0 + S, y1 = y0 + S, sw = map.sidewalk, R = map.cornerR;
  const r = rngFrom(Math.floor(x0 * 73856093) ^ Math.floor(y0 * 19349663));
  // 1. land
  g.fillStyle = "#b7b0a2"; g.fillRect(x0, y0, S, S);
  for (let i = 0; i < (lite ? 0 : S * S / 14); i++) { g.fillStyle = `rgba(${r() < 0.5 ? "60,55,45" : "255,250,240"},${0.04 + r() * 0.05})`; g.fillRect(x0 + r() * S, y0 + r() * S, 0.3 + r() * 1.5, 0.3 + r() * 1.5); }
  // 2. sidewalks with tiles
  const walks = F.walks.filter((q) => rhit(q, x0 - 1, y0 - 1, x1 + 1, y1 + 1));
  g.fillStyle = "#cdc6b8"; for (const q of walks) g.fillRect(q.x0, q.y0, q.x1 - q.x0, q.y1 - q.y0);
  g.save(); g.beginPath(); for (const q of walks) g.rect(q.x0, q.y0, q.x1 - q.x0, q.y1 - q.y0); g.clip();
  if (!lite) { g.strokeStyle = "rgba(90,80,65,.09)"; g.lineWidth = 0.025; g.beginPath();
  for (let x = Math.ceil(x0 / 0.4) * 0.4; x <= x1; x += 0.4) { g.moveTo(x, y0); g.lineTo(x, y1); }
  for (let y = Math.ceil(y0 / 0.4) * 0.4; y <= y1; y += 0.4) { g.moveTo(x0, y); g.lineTo(x1, y); }
  g.stroke();
  for (let i = 0; i < S * S / 12; i++) { g.fillStyle = `rgba(${r() < 0.6 ? "80,70,55" : "255,255,250"},${0.05 + r() * 0.08})`; g.fillRect(x0 + Math.floor(r() * S / 0.4) * 0.4, y0 + Math.floor(r() * S / 0.4) * 0.4, 0.4, 0.4); } }
  g.restore();
  // 3. asphalt with grain and wear
  const roads = F.roads.filter((q) => rhit(q, x0 - 1, y0 - 1, x1 + 1, y1 + 1));
  g.fillStyle = "#45494e"; for (const q of roads) g.fillRect(q.x0, q.y0, q.x1 - q.x0, q.y1 - q.y0);
  for (const f of F.fillets) {
    if (!(Math.max(f.px, f.cx) > x0 && Math.min(f.px, f.cx) < x1 && Math.max(f.py, f.cy) > y0 && Math.min(f.py, f.cy) < y1)) continue;
    g.save(); g.beginPath(); g.rect(Math.min(f.px, f.cx), Math.min(f.py, f.cy), R, R); g.clip();
    g.fillStyle = "#45494e"; g.fillRect(Math.min(f.px, f.cx), Math.min(f.py, f.cy), R, R);
    g.fillStyle = "#cdc6b8"; g.beginPath(); g.arc(f.cx, f.cy, R, 0, 6.3); g.fill();
    g.fillStyle = "#b7b0a2"; g.beginPath(); g.arc(f.cx, f.cy, R - sw, 0, 6.3); g.fill();
    g.restore();
  }
  if (roads.length && !lite) {
    g.save(); g.beginPath(); for (const q of roads) g.rect(q.x0, q.y0, q.x1 - q.x0, q.y1 - q.y0); g.clip();
    for (let i = 0; i < S * S / 70; i++) { const x = x0 + r() * S, y = y0 + r() * S, rad = 4 + r() * 14; const gr = g.createRadialGradient(x, y, 0, x, y, rad); const dark = r() < 0.55; gr.addColorStop(0, dark ? "rgba(20,22,25,.10)" : "rgba(120,124,128,.07)"); gr.addColorStop(1, "rgba(0,0,0,0)"); g.fillStyle = gr; g.fillRect(x - rad, y - rad, rad * 2, rad * 2); }
    for (let i = 0; i < S * S * 0.9; i++) { const x = x0 + r() * S, y = y0 + r() * S; const l = r() < 0.5; g.fillStyle = l ? `rgba(200,200,195,${0.05 + r() * 0.08})` : `rgba(10,10,12,${0.06 + r() * 0.1})`; const s = 0.04 + r() * 0.08; g.fillRect(x, y, s, s); }
    for (const p of F.patches) {
      if (!hit(p, x0, y0, x1, y1)) continue;
      if (p.kind === "hole") { const cx = p.x + p.w / 2, cy = p.y + p.h / 2; g.fillStyle = "#34373b"; g.beginPath(); g.arc(cx, cy, 0.34, 0, 6.3); g.fill(); g.strokeStyle = "#5b5f64"; g.lineWidth = 0.05; g.beginPath(); g.arc(cx, cy, 0.3, 0, 6.3); g.stroke(); g.beginPath(); g.moveTo(cx - 0.2, cy); g.lineTo(cx + 0.2, cy); g.moveTo(cx, cy - 0.2); g.lineTo(cx, cy + 0.2); g.stroke(); }
      else { g.fillStyle = p.kind === "dark" ? "#3c4045" : "#4d5157"; g.fillRect(p.x, p.y, p.w, p.h); g.strokeStyle = "rgba(15,15,18,.4)"; g.lineWidth = 0.05; g.strokeRect(p.x, p.y, p.w, p.h); }
    }
    g.restore();
  }
  // 4. curbs
  g.lineCap = "butt";
  for (const pass of [["#ece8de", 0.28, 0], ["rgba(25,25,28,.35)", 0.1, -0.2]]) {
    g.strokeStyle = pass[0]; g.lineWidth = pass[1]; g.beginPath();
    for (const c of F.curbs) {
      if (Math.max(c.x0, c.x1) < x0 - 1 || Math.min(c.x0, c.x1) > x1 + 1 || Math.max(c.y0, c.y1) < y0 - 1 || Math.min(c.y0, c.y1) > y1 + 1) continue;
      const dx = c.x1 - c.x0, dy = c.y1 - c.y0, L = Math.hypot(dx, dy) || 1, nx = -dy / L, ny = dx / L;
      // offset the inner dark line toward the road: the road is on the side away from the sidewalk; approximate by the nearer junction
      const k = pass[2]; g.moveTo(c.x0 + nx * k * 0, c.y0 + ny * k * 0); g.lineTo(c.x1, c.y1);
    }
    for (const a of F.arcs) {
      if (a.cx < x0 - R - 1 || a.cx > x1 + R + 1 || a.cy < y0 - R - 1 || a.cy > y1 + R + 1) continue;
      const s = Math.atan2(-a.sy, 0), e = Math.atan2(0, -a.sx);
      const a0 = Math.min(s, e), a1 = Math.max(s, e), span = a1 - a0;
      const rr = R - pass[2];
      if (span <= Math.PI) { g.moveTo(a.cx + Math.cos(a0) * rr, a.cy + Math.sin(a0) * rr); g.arc(a.cx, a.cy, rr, a0, a1); }
      else { g.moveTo(a.cx + Math.cos(a1) * rr, a.cy + Math.sin(a1) * rr); g.arc(a.cx, a.cy, rr, a1, a0 + 2 * Math.PI); }
    }
    g.stroke();
  }
  // 5. road paint
  for (const m of F.marks) if (hit(m, x0, y0, x1, y1)) paintApproach(g, map, m);
  // 6. sidewalk life under the traffic
  for (const p of F.props) {
    if (!hit(p, x0, y0, x1, y1)) continue;
    g.save(); g.translate(p.x, p.y); g.rotate(p.a);
    if (p.kind === "moto") { drawShadowStatic(g, 1.9, 0.6); drawMoto(g, { look: { body: p.c }, cargo: null, delivery: false }, true); }
    else if (p.kind === "cart") { g.fillStyle = "rgba(0,0,0,.2)"; g.fillRect(-0.7, -0.35, 1.5, 0.85); g.fillStyle = "#d8cdb5"; g.fillRect(-0.75, -0.4, 1.5, 0.8); g.fillStyle = p.c; g.fillRect(-0.6, -0.3, 0.55, 0.6); g.fillStyle = "#3a3a3a"; g.fillRect(-0.85, -0.45, 0.12, 0.9); }
    else {
      const q = rngFrom(p.seed);
      g.fillStyle = "rgba(0,0,0,.18)"; g.fillRect(-0.55, -0.25, 1.3, 0.7);
      g.fillStyle = "#b98a55"; g.fillRect(-0.6, -0.3, 1.2, 0.6);
      const stool = pick(q, ["#2d63c2", "#d0312d", "#2d63c2", "#e8b021"]);
      for (const [sx, sy] of [[-0.9, -0.1], [0.9, 0.1], [-0.3, -0.75], [0.3, 0.75], [0.35, -0.75], [-0.35, 0.75]]) {
        if (q() < 0.2) continue;
        g.fillStyle = stool; g.fillRect(sx - 0.15, sy - 0.15, 0.3, 0.3);
        if (q() < 0.65) { g.fillStyle = pick(q, SHIRTS); g.beginPath(); g.ellipse(sx, sy, 0.2, 0.26, 0, 0, 6.3); g.fill(); g.fillStyle = "#2a1f18"; g.beginPath(); g.arc(sx, sy, 0.12, 0, 6.3); g.fill(); }
      }
      g.fillStyle = "#d9d3c4"; g.fillRect(-0.15, -0.12, 0.3, 0.24); g.fillStyle = "#c9302c"; g.beginPath(); g.arc(0.35, 0, 0.12, 0, 6.3); g.fill();
    }
    g.restore();
  }
  for (const t of F.trees) if (hit(t, x0, y0, x1, y1)) { g.fillStyle = "#6b5a46"; g.fillRect(t.x - 0.5, t.y - 0.5, 1, 1); g.fillStyle = "#4b3d2f"; g.beginPath(); g.arc(t.x, t.y, 0.18, 0, 6.3); g.fill(); }
  // 7. houses: shadows first (low to high), then roofs, then awnings
  const hs = F.houses.filter((h) => hit(h, x0, y0, x1, y1)).sort((a, b) => a.fl - b.fl);
  for (const h of hs) {
    const off = h.fl * 0.55; g.fillStyle = "rgba(15,18,22,.26)"; g.beginPath();
    g.moveTo(h.x, h.y); g.lineTo(h.x + h.w, h.y); g.lineTo(h.x + h.w + off * SUN[0], h.y + off * SUN[1]); g.lineTo(h.x + h.w + off * SUN[0], h.y + h.h + off * SUN[1]); g.lineTo(h.x + off * SUN[0], h.y + h.h + off * SUN[1]); g.lineTo(h.x, h.y + h.h); g.closePath(); g.fill();
    drawRoof(g, h, rngFrom(h.seed));
  }
  for (const h of hs) if (h.awning) drawAwning(g, h, rngFrom(h.seed + 1));
  for (const t of F.trees) if (hit(t, x0, y0, x1, y1)) { g.fillStyle = "rgba(15,22,14,.2)"; g.beginPath(); g.ellipse(t.x + 1.4, t.y + 1.8, t.r * 1.05, t.r * 0.95, 0, 0, 6.3); g.fill(); }
  // ---- overhead layer
  for (const u of F.umbrellas) {
    if (!hit(u, x0, y0, x1, y1)) continue;
    o.fillStyle = "rgba(0,0,0,.16)"; o.beginPath(); o.arc(u.x + 0.5, u.y + 0.6, u.r, 0, 6.3); o.fill();
    for (let k = 0; k < 8; k++) { o.fillStyle = u.c[k % 2]; o.beginPath(); o.moveTo(u.x, u.y); o.arc(u.x, u.y, u.r, k * Math.PI / 4, (k + 1) * Math.PI / 4); o.closePath(); o.fill(); }
    o.fillStyle = "#555"; o.beginPath(); o.arc(u.x, u.y, 0.08, 0, 6.3); o.fill();
  }
  for (const t of F.trees) if (hit(t, x0, y0, x1, y1)) drawCanopy(o, t, rngFrom(t.seed));
  o.lineCap = "round";
  for (const w of F.wires) {
    if (!hit(w, x0, y0, x1, y1)) continue;
    const q = rngFrom(w.seed), a = w.a, b = w.b, L = Math.hypot(b.x - a.x, b.y - a.y) || 1, mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2, nx = -(b.y - a.y) / L, ny = (b.x - a.x) / L;
    for (let k = 0; k < w.n; k++) { const s = w.thick ? (q() - 0.5) * 2.5 : (q() - 0.3) * 1.2; o.strokeStyle = `rgba(16,16,18,${0.33 + q() * 0.32})`; o.lineWidth = (w.thick ? 0.04 : 0.035) + q() * 0.05; o.beginPath(); o.moveTo(a.x + (q() - 0.5) * 0.3, a.y + (q() - 0.5) * 0.3); o.quadraticCurveTo(mx + nx * s, my + ny * s, b.x + (q() - 0.5) * 0.3, b.y + (q() - 0.5) * 0.3); o.stroke(); }
  }
  for (const p of F.poles) {
    if (!hit(p, x0, y0, x1, y1)) continue;
    const q = rngFrom(p.seed);
    o.fillStyle = "rgba(0,0,0,.25)"; o.fillRect(p.x - 0.1, p.y - 0.05, 0.42, 0.42);
    o.fillStyle = "#a7a49c"; o.fillRect(p.x - 0.18, p.y - 0.18, 0.36, 0.36); o.strokeStyle = "#5e5b55"; o.lineWidth = 0.04; o.strokeRect(p.x - 0.18, p.y - 0.18, 0.36, 0.36);
    for (let k = 0; k < 3; k++) { o.strokeStyle = "rgba(15,15,17,.55)"; o.lineWidth = 0.05; o.beginPath(); o.arc(p.x + (q() - 0.5) * 0.4, p.y + (q() - 0.5) * 0.4, 0.25 + q() * 0.25, 0, 6.3); o.stroke(); }
    if (q() < 0.25) { o.fillStyle = "#6f7378"; o.fillRect(p.x + 0.2, p.y - 0.35, 0.55, 0.7); o.strokeStyle = "#3a3d40"; o.strokeRect(p.x + 0.2, p.y - 0.35, 0.55, 0.7); }
  }
  return { ground, over };
}

/* paint for traffic coming in on arm a of junction jn: zebra, stop or give-way line, arrows, centre and lane lines up to the middle of the road */
function paintApproach(g, map, m) {
  const jn = m.jn, a = m.a, d = a.dir, n = [-d[1], d[0]], app = [-n[0], -n[1]], len = m.len;
  const white = "rgba(238,236,228,.88)", yellow = "rgba(232,190,58,.92)";
  const P = (u, v) => [jn.x + d[0] * u + n[0] * v, jn.y + d[1] * u + n[1] * v];
  const line = (u0, v0, u1, v1, w, col) => { const p = P(u0, v0), q = P(u1, v1); g.strokeStyle = col; g.lineWidth = w; g.beginPath(); g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]); g.stroke(); };
  const dash = (u0, u1, v, on, off, w, col) => { g.strokeStyle = col; g.lineWidth = w; g.beginPath(); for (let u = u0; u < u1; u += on + off) { const p = P(u, v), q = P(Math.min(u1, u + on), v); g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]); } g.stroke(); };
  const sd = a.box + map.stop, signal = jn.type === "signal";
  // centre line: double yellow on two-lane roads, single white line otherwise; solid near the junction, dashed further out
  if (a.lanes > 1) { for (const v of [-0.16, 0.16]) line(sd, v, len, v, 0.12, yellow); for (const v of [-a.hw / 2, a.hw / 2]) { line(sd, v, Math.min(len, sd + 16), v, 0.12, white); dash(sd + 16, len, v, 3, 5, 0.12, white); } }
  else if (a.cls === "sec") { line(sd, 0, Math.min(len, sd + 10), 0, 0.12, white); dash(sd + 10, len, 0, 2, 3, 0.12, white); }
  else dash(sd + 2, len, 0, 1.5, 3, 0.1, white);
  // zebra across the whole road
  const z0 = a.box + map.zebra[0], z1 = a.box + map.zebra[1];
  if (signal || a.cls !== "small") {
    g.fillStyle = white;
    for (let k = -a.hw + 0.5; k < a.hw - 0.3; k += 1.0) { const c = P((z0 + z1) / 2, k + 0.25); const w = d[0] ? z1 - z0 : 0.5, h = d[0] ? 0.5 : z1 - z0; g.fillRect(c[0] - w / 2, c[1] - h / 2, w, h); }
  }
  // stop line (signal) or dashed give-way line (no signal) across the incoming half
  if (signal) line(sd, 0, sd, -(a.hw - 0.2), 0.42, white);
  else { g.strokeStyle = white; g.lineWidth = 0.3; g.beginPath(); for (let v = -0.2; v > -(a.hw - 0.2); v -= 0.9) { const p = P(sd, v), q = P(sd, Math.max(-(a.hw - 0.2), v - 0.5)); g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]); } g.stroke(); }
  // lane arrows, only where the road is long enough
  if (len > sd + 12 && a.cls !== "small") {
    const lw = a.hw / a.lanes, hasL = !!jn.arms[LEFT_OF[a.id]], hasR = !!jn.arms[RIGHT_OF[a.id]], hasS = !!jn.arms[OPP[a.id]];
    for (let k = 0; k < a.lanes; k++) {
      const c = P(sd + 9, -lw * (k + 0.5));
      const allowL = hasL && (a.lanes === 1 || k === 0), allowR = hasR && (a.lanes === 1 || k === a.lanes - 1);
      drawArrow2(g, c[0], c[1], Math.atan2(-d[1], -d[0]), hasS, allowL, allowR, white);
    }
  }
}
function drawArrow2(g, x, y, ang, S, Lt, Rt, col) {
  g.save(); g.translate(x, y); g.rotate(ang); g.fillStyle = col;
  g.fillRect(-2.2, -0.11, S ? 3.0 : 1.9, 0.22);
  if (S) { g.beginPath(); g.moveTo(1.6, 0); g.lineTo(0.7, -0.45); g.lineTo(0.7, 0.45); g.closePath(); g.fill(); }
  const branch = (side) => { g.save(); g.translate(-0.3, 0); g.fillRect(-0.11, side > 0 ? 0 : -1.1, 0.22, 1.1); g.beginPath(); g.moveTo(0, side * 1.6); g.lineTo(-0.42, side * 0.9); g.lineTo(0.42, side * 0.9); g.closePath(); g.fill(); g.restore(); };
  if (Lt) branch(-1);
  if (Rt) branch(1);
  g.restore();
}

/* lazy tile cache */
class CityLayers {
  constructor(map, seed = 11, ppm = 14, tile = 80, keep = 14) { this.map = map; this.F = cityFeatures(map, seed); this.ppm = ppm; this.S = tile; this.keep = keep; this.cache = new Map(); this.heads = this.F.heads; }
  key(i, j) { return i * 10000 + j; }
  get(i, j, allowBuild) {
    const k = this.key(i, j); let t = this.cache.get(k);
    if (t) { this.cache.delete(k); this.cache.set(k, t); return t; }
    if (!allowBuild) return null;
    t = paintTile(this.map, this.F, i * this.S, j * this.S, this.S, this.ppm);
    this.cache.set(k, t);
    while (this.cache.size > this.keep) this.cache.delete(this.cache.keys().next().value);
    return t;
  }
  /* one low-resolution picture of the whole city for zoomed-out views */
  overview() {
    if (!this.ov) { const B = this.map.bounds, S = Math.max(B.x1 - B.x0, B.y1 - B.y0) + 20; this.ov = paintTile(this.map, this.F, B.x0 - 10, B.y0 - 10, S, Math.min(3, 1600 / S), true); this.ovS = S; this.ovX = B.x0 - 10; this.ovY = B.y0 - 10; }
    return this.ov;
  }
  /* draw the visible tiles of one layer; builds at most `budget` new tiles per call */
  draw(g, layer, vx0, vy0, vx1, vy1, budget = 2) {
    const S = this.S, B = this.map.bounds;
    const i0 = Math.floor(Math.max(vx0, B.x0 - S) / S), i1 = Math.floor(Math.min(vx1, B.x1 + S) / S), j0 = Math.floor(Math.max(vy0, B.y0 - S) / S), j1 = Math.floor(Math.min(vy1, B.y1 + S) / S);
    if ((i1 - i0 + 1) * (j1 - j0 + 1) > this.keep - 2) {   // too many tiles on screen: use the overview picture
      const ov = this.overview(); g.drawImage(ov[layer], this.ovX, this.ovY, this.ovS, this.ovS); return 0;
    }
    let built = 0, missing = 0;
    for (let i = i0; i <= i1; i++) for (let j = j0; j <= j1; j++) {
      let t = this.get(i, j, false);
      if (!t && layer === "ground" && built < budget) { t = this.get(i, j, true); built++; }
      if (t) g.drawImage(t[layer], i * S, j * S, S + 0.02, S + 0.02);
      else if (layer === "ground") { missing++; g.fillStyle = "#b7b0a2"; g.fillRect(i * S, j * S, S, S); }
    }
    return missing;
  }
}

/* ---------------- signals with countdown, drawn every frame */
function drawSignals(g, sim, heads, vx0, vy0, vx1, vy1) {
  for (const s of heads) {
    if (s.x < vx0 || s.x > vx1 || s.y < vy0 || s.y > vy1) continue;
    const L = sim.light(s.j, s.arm); if (!L) continue;
    g.save(); g.translate(s.x, s.y); g.rotate(s.ang);
    g.fillStyle = "rgba(0,0,0,.3)"; roundRect(g, -0.2, -0.65, 0.65, 1.45, 0.12); g.fill();
    g.fillStyle = "#1d1f22"; roundRect(g, -0.32, -0.75, 0.62, 1.5, 0.12); g.fill();
    [["R", "#ff3b30", "#4a1714"], ["Y", "#ffc21a", "#4a3a10"], ["G", "#2ee66b", "#123f22"]].forEach(([c, on, off], k) => {
      const y = -0.45 + k * 0.45, lit = L.c === c;
      if (lit) { const gr = g.createRadialGradient(0, y, 0, 0, y, 0.6); gr.addColorStop(0, on + "aa"); gr.addColorStop(1, on + "00"); g.fillStyle = gr; g.beginPath(); g.arc(0, y, 0.6, 0, 6.3); g.fill(); }
      g.fillStyle = lit ? on : off; g.beginPath(); g.arc(0, y, 0.16, 0, 6.3); g.fill();
    });
    if (!s.far) {
      g.fillStyle = "#111214"; roundRect(g, -0.32, 0.85, 0.62, 0.62, 0.08); g.fill();
      g.save(); g.translate(0, 1.16); g.rotate(-s.ang);
      g.fillStyle = L.c === "G" ? "#39f07a" : L.c === "Y" ? "#ffc21a" : "#ff4a3d";
      g.scale(0.02, 0.02); g.font = "700 21px 'JetBrains Mono', ui-monospace, monospace"; g.textAlign = "center"; g.textBaseline = "middle";
      g.fillText(String(Math.ceil(L.left)).padStart(2, "0"), 0, 1);
      g.restore();
    }
    g.restore();
  }
}
function drawHonk(g, v, x, y, t) {
  const a = 1 - (0.6 - v.honk) / 0.6;
  g.save(); g.translate(x, y); g.rotate(v.h);
  g.strokeStyle = `rgba(255,214,90,${0.85 * a})`; g.lineWidth = 0.07;
  for (let k = 0; k < 3; k++) { g.beginPath(); g.arc(v.T.l / 2, 0, 0.35 + k * 0.25 + (1 - a) * 0.3, -0.6, 0.6); g.stroke(); }
  g.restore();
}

if (typeof module !== "undefined") module.exports = {};
