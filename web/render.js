/* ======================================================================
   LOTUS: top-down renderer (Canvas 2D).
   Static layers are painted in tiles on demand: ground (terrain, rice
   fields, sidewalks, asphalt, paint, rails, water, parked cars, houses) and
   an overhead layer (tree canopies, bamboo, poles, tangled wires, umbrellas,
   market tarps) drawn above the traffic. Signals, barriers, people,
   buffaloes and trains are drawn every frame.
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
  if (hat) {   // conical hat (nón lá): a straw cone seen from above
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
  else if (v.type === "tractor") drawTractor(g, v);
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

/* ---------------- the map: features laid out once in world metres; tiles are painted on demand */
const LAND = { urban: "#b7b0a2", industrial: "#aaa79f", rural: "#8fa564", forest: "#5d7a45", none: "#b3ad9c" };
const ASPHALT = { main: "#45494e", sec: "#45494e", small: "#4a4e52", avenue: "#43474c", highway: "#3e4247", alley: "#8f8b83", lane: "#97928a", rural: "#5d5f5c", pass: "#47494b" };
function bbOfPts(o, pts, pad = 0) { let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9; for (const p of pts) { if (p[0] < x0) x0 = p[0]; if (p[0] > x1) x1 = p[0]; if (p[1] < y0) y0 = p[1]; if (p[1] > y1) y1 = p[1]; } o.bx0 = x0 - pad; o.by0 = y0 - pad; o.bx1 = x1 + pad; o.by1 = y1 + pad; return o; }
function stripPoly(pts, dl, dr) { const a = offsetPoly(pts, dr), b = offsetPoly(pts, dl).reverse(); return a.concat(b); }
function slicePts(L, s0, s1) { const i0 = clampN(Math.floor(s0), 0, L.pts.length - 1), i1 = clampN(Math.ceil(s1), 0, L.pts.length - 1); return L.pts.slice(i0, i1 + 1); }
function convexHull(P) {
  const p = P.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]), cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], up = []; for (const q of p) { while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
  for (let i = p.length - 1; i >= 0; i--) { const q = p[i]; while (up.length >= 2 && cr(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
  return lo.slice(0, -1).concat(up.slice(0, -1));
}
const lineInter = (p, d, q, e) => { const den = d[0] * e[1] - d[1] * e[0]; if (Math.abs(den) < 1e-6) return null; const t = ((q[0] - p[0]) * e[1] - (q[1] - p[1]) * e[0]) / den; return [p[0] + d[0] * t, p[1] + d[1] * t]; };

function cityFeatures(map, seed = 11) {
  const r = rngFrom(seed * 131 + 7), B = map.bounds;
  const F = { walks: [], roads: [], islands: [], medians: [], lines: [], quads: [], arrows: [], rails: [], floods: [], cliffs: [], guards: [], fields: [], houses: [], trees: [], poles: [], props: [], umbrellas: [], wires: [], patches: [], heads: [], barriers: [], stops: [], statics: [],
    // real-map layers (OpenStreetMap): land use, water, flyover decks and their paint, tunnels, works, signs, labels
    land: [], water: [], decks: [], deckLines: [], tunnels: [], cons: [], signs: [], labels: [], viaducts: [] };
  const white = "rgba(238,236,228,.88)", yellow = "rgba(232,190,58,.92)";
  const nodeR = (n) => n.type === "roundabout" ? n.R + n.rw : 0;
  // ---- link surfaces: sidewalk strip, asphalt strip, median
  for (const L of map.links) {
    const nA = map.nodes[L.a], nB = L.b != null ? map.nodes[L.b] : null;
    const s0 = nodeR(nA) ? nodeR(nA) - 2 : 0, s1 = nB && nodeR(nB) ? L.len - nodeR(nB) + 2 : L.len;
    const pts = slicePts(L, s0, s1); if (pts.length < 2) continue;
    if (L.level > 0) { F.decks.push(bbOfPts({ poly: stripPoly(pts, -L.hw - 0.5, L.hw + 0.5), pts, hw: L.hw, c: ASPHALT[L.cls] || "#45494e", z: L.zDeck || 5 }, pts, L.hw + 8)); continue; }
    if (L.level < 0) { F.tunnels.push(bbOfPts({ pts, hw: L.hw }, pts, L.hw + 1)); continue; }
    if (L.sw) F.walks.push(bbOfPts({ poly: stripPoly(pts, -(L.hw + L.sw), L.hw + L.sw), enc: L.encroached || L.market }, pts, L.hw + L.sw));
    F.roads.push(bbOfPts({ poly: stripPoly(pts, -L.hw, L.hw), c: ASPHALT[L.cls] || "#45494e" }, pts, L.hw));
    if (L.med > 0) {   // a raised median, open at U-turns and junctions
      const a0 = L.armA ? L.armA.box + 1 : 0, a1 = L.len - (L.armB ? L.armB.box + 1 : 0);
      if (a1 - a0 > 4) { const mp = slicePts(L, a0, a1); F.medians.push(bbOfPts({ poly: stripPoly(mp, -L.med, L.med), pts: mp, w: L.med, trees: L.cls === "avenue" }, mp, L.med + 1)); }
    }
  }
  // ---- junction surfaces: rounded polygons between the arms
  for (const n of map.nodes) {
    const arms = n.order.map((id) => n.arms[id]);
    if (n.type === "roundabout") {
      F.walks.push(bbOfPts({ disk: [n.x, n.y, n.R + n.rw + 3.5] }, [[n.x, n.y]], n.R + n.rw + 4));
      F.roads.push(bbOfPts({ disk: [n.x, n.y, n.R + n.rw], c: "#45494e" }, [[n.x, n.y]], n.R + n.rw + 1));
      F.islands.push(bbOfPts({ x: n.x, y: n.y, R: n.R, seed: Math.floor(r() * 1e9) }, [[n.x, n.y]], n.R + 1));
      F.lines.push(bbOfPts({ circle: [n.x, n.y, n.R + n.rw * 0.5], w: 0.12, c: white, dash: [2, 3] }, [[n.x, n.y]], n.R + n.rw));
      continue;
    }
    if (arms.length < 3) {   // a bend: fill the outside of the corner; a straight pass-through needs nothing
      if (arms.length === 2 && Math.abs(Math.abs(wrapA(arms[0].ang - arms[1].ang)) - Math.PI) < 0.3) continue;
      const w = Math.min(...arms.map((a) => a.hw)); if (Math.abs(n.z) > 2.5 || arms.some((a) => a.level)) continue;
      F.roads.push(bbOfPts({ disk: [n.x, n.y, w], c: ASPHALT[arms[0].cls] || "#45494e" }, [[n.x, n.y]], w + 1)); continue;
    }
    const rf = arms.every((a) => a.kind === "alley" || a.kind === "rural") ? 1.5 : map.cornerR;
    for (const [off, list, isWalk] of [[(a) => a.hw + a.sw, F.walks, true], [(a) => a.hw, F.roads, false]]) {
      if (isWalk && !arms.some((a) => a.sw)) continue;
      const poly = [];
      for (let i = 0; i < arms.length; i++) {
        const a = arms[i], b = arms[(i + 1) % arms.length];
        const pa = armAt(a, Math.min(a.len - 1, a.box + rf)), na = rightNormal([pa.tx, pa.ty]), wa = off(a);
        const pb = armAt(b, Math.min(b.len - 1, b.box + rf)), nb = rightNormal([pb.tx, pb.ty]), wb = off(b);
        const La = [pa.x - na[0] * wa, pa.y - na[1] * wa], Ra = [pa.x + na[0] * wa, pa.y + na[1] * wa], Lb = [pb.x - nb[0] * wb, pb.y - nb[1] * wb];
        poly.push(La, Ra);
        const c = lineInter(Ra, [pa.tx, pa.ty], Lb, [pb.tx, pb.ty]);
        const ctl = c && Math.hypot(c[0] - n.x, c[1] - n.y) < n.r + 25 ? c : [n.x, n.y];
        for (let k = 1; k < 10; k++) { const t = k / 10, u = 1 - t; poly.push([u * u * Ra[0] + 2 * u * t * ctl[0] + t * t * Lb[0], u * u * Ra[1] + 2 * u * t * ctl[1] + t * t * Lb[1]]); }
      }
      if (n.z > 2.5) { if (!isWalk) F.decks.push(bbOfPts({ poly, c: ASPHALT[arms[0].cls] || "#45494e", z: n.z }, poly, 8)); }
      else if (n.z < -2.5) { /* inside a tunnel */ }
      else list.push(bbOfPts({ poly, c: isWalk ? null : (ASPHALT[arms[0].cls] || "#45494e") }, poly, 1));
    }
  }
  // ---- paint along every link: centre line, lane lines, motorbike lane, edge lines on country roads
  for (const L of map.links) {
    const a0 = (L.armA ? L.armA.box : 0) + map.stop, a1 = L.len - (L.armB ? L.armB.box + map.stop : 0);
    if (a1 - a0 < 3 || L.level < 0) continue;
    const pts = slicePts(L, a0, a1), push = (d, w, c, dash) => (L.level > 0 ? F.deckLines : F.lines).push(bbOfPts({ pts: offsetPoly(pts, d), w, c, dash }, pts, L.hw + 1));
    if (L.ow) {   // one-way: arrows along the carriageway, lane lines, no centre line
      for (let s = a0 + 8; s < a1 - 6; s += 38) { const p = linkPoint(L, s, 0); (L.level > 0 ? F.deckLines : F.arrows).push(bbOfPts({ x: p.x, y: p.y, ang: p.ang + (L.ow < 0 ? Math.PI : 0), S: true, ow: true }, [[p.x, p.y]], 3)); }
      if (L.kind !== "alley") { const lw = 2 * L.hw / Math.max(1, L.lanes); for (let k = 1; k < L.lanes; k++) push(-L.hw + lw * k, 0.12, white, [3, 5]); for (const sd of [1, -1]) push(sd * (L.hw - 0.25), 0.12, white, null); }
      continue;
    }
    if (L.kind === "alley") continue;
    if (L.med > 0) {
      const dirW = L.hw - L.med - L.bl, lw = dirW / L.lanes;
      for (const sd of [1, -1]) {
        for (let k = 1; k < L.lanes; k++) push(sd * (L.med + lw * k), 0.12, white, [3, 5]);
        if (L.bl) { push(sd * (L.hw - L.bl), 0.16, white, null); }
      }
    } else if (L.shared) { if (L.kind === "rural") for (const sd of [1, -1]) push(sd * (L.hw - 0.25), 0.1, "rgba(238,236,228,.5)", [1.2, 2.5]); }
    else if (L.lanes > 1) { for (const v of [-0.16, 0.16]) push(v, 0.12, yellow, null); for (const sd of [1, -1]) push(sd * L.hw / 2, 0.12, white, [3, 5]); }
    else if (L.kind === "pass") { push(0, 0.13, yellow, [4, 4]); for (const sd of [1, -1]) push(sd * (L.hw - 0.2), 0.12, white, null); }
    else if (L.cls === "sec") push(0, 0.12, white, [2, 3]);
    else push(0, 0.1, white, [1.5, 3]);
    // pass: rock cut on the uphill side, guard rail on the downhill side, chevrons on hairpins
    if (L.kind === "pass") {
      for (const sd of [1, -1]) {
        let run = [], kind = null;
        const flush = () => { if (run.length > 3) (kind === "cliff" ? F.cliffs : F.guards).push(bbOfPts({ pts: run, side: sd }, run, 3)); run = []; };
        for (let i = 2; i < L.pts.length - 2; i += 2) {
          const p = L.pts[i], q = L.pts[i + 1], tl = Math.hypot(q[0] - p[0], q[1] - p[1]) || 1, nn = rightNormal([(q[0] - p[0]) / tl, (q[1] - p[1]) / tl]);
          const off = sd * (L.hw + 2.5), h = map.H(p[0] + nn[0] * off, p[1] + nn[1] * off) - L.z[i];
          const k = h > 1.2 ? "cliff" : h < -1.2 ? "guard" : null;
          if (k !== kind) { flush(); kind = k; }
          if (k) run.push([p[0] + nn[0] * sd * (L.hw + (k === "cliff" ? 0.9 : 0.45)), p[1] + nn[1] * sd * (L.hw + (k === "cliff" ? 0.9 : 0.45))]);
        }
        flush();
      }
    }
    // zones: water over the road, a bus bay
    for (const zn of L.zones) {
      if (zn.type === "flood") { const fp = slicePts(L, zn.s0 - 6, zn.s1 + 6); F.floods.push(bbOfPts({ poly: stripPoly(fp, -(L.hw + 1.5), L.hw + 1.5), pts: fp, w: L.hw, seed: Math.floor(r() * 1e9), depth: zn.depth }, fp, L.hw + 3)); }
      if (zn.type === "bus") {
        const sd = zn.dir >= 0 ? 1 : -1, bp = slicePts(L, zn.s0, zn.s1);
        F.quads.push(bbOfPts({ poly: stripPoly(bp, sd > 0 ? L.hw - 3.0 : -L.hw + 0.2, sd > 0 ? L.hw - 0.2 : -L.hw + 3.0), c: "rgba(232,190,58,.28)", edge: yellow }, bp, L.hw));
        const m = bp[Math.floor(bp.length / 2)], m2 = bp[Math.min(bp.length - 1, Math.floor(bp.length / 2) + 1)], tl = Math.hypot(m2[0] - m[0], m2[1] - m[1]) || 1, nn = rightNormal([(m2[0] - m[0]) / tl, (m2[1] - m[1]) / tl]);
        F.stops.push(bbOfPts({ x: m[0] + nn[0] * sd * (L.hw + 2.2), y: m[1] + nn[1] * sd * (L.hw + 2.2), ang: Math.atan2(m2[1] - m[1], m2[0] - m[0]) }, [m], 6));
      }
    }
  }
  // ---- paint at each approach: zebra, stop line or give-way line, arrows; signal heads; barriers
  for (const n of map.nodes) for (const id of n.order) {
    const a = n.arms[id]; if (a.len < a.box + 6 || n.order.length < 2 || Math.abs(n.z) > 2.5 || a.level) continue;
    if (a.ow === 1) { if (n.order.length > 2) { const p = armAt(a, a.box + 2.5), nn = rightNormal([p.tx, p.ty]); F.signs.push(bbOfPts({ x: p.x + nn[0] * (a.hw + 0.9), y: p.y + nn[1] * (a.hw + 0.9), k: "noentry" }, [[p.x, p.y]], a.hw + 3)); } continue; }   // a one-way street seen from its far end: no entry
    const at = (s, v) => { const p = armAt(a, s), nn = rightNormal([p.tx, p.ty]); return [p.x + nn[0] * v, p.y + nn[1] * v]; };
    const quad = (s0, s1, v0, v1, c) => { const P = [at(s0, v0), at(s1, v0), at(s1, v1), at(s0, v1)]; F.quads.push(bbOfPts({ poly: P, c }, P, 0.5)); };
    const sd = a.box + map.stop, z0 = a.box + map.zebra[0], z1 = a.box + map.zebra[1], inc = (v) => -v;   // incoming half is the arm's left
    const signal = n.type === "signal", rail = n.type === "rail";
    const minorIn = n.type === "yield" || (n.type === "priority" && !a.major) || n.type === "roundabout";
    if ((signal || (a.cls !== "small" && a.kind === "urban")) && n.type !== "roundabout" && n.order.length > 2) for (let v = -a.hw + 0.5; v < a.hw - 0.3; v += 1.0) quad(z0, z1, v, v + 0.5, white);
    if (a.ow === -1 && (signal || rail)) quad(sd - 0.2, sd + 0.2, -a.hw + 0.2, a.hw - 0.2, white);
    else if (signal || rail) quad(sd - 0.2, sd + 0.2, inc(0.1), inc(a.hw - (a.med || 0) - 0.2) , white);
    else if (minorIn && a.kind !== "alley") for (let v = 0.3; v < a.hw - 0.3; v += 0.9) quad(sd - 0.15, sd + 0.15, inc(v), inc(Math.min(a.hw - 0.2, v + 0.5)), white);
    if (a.len > sd + 14 && (a.lanes > 1 || a.cls === "sec" || a.cls === "main") && !a.shared && n.order.length > 2) {
      const dirW = a.hw - (a.med || 0) - (a.bl || 0), lw = dirW / a.lanes;
      const ks = n.order.map((bid) => turnKind(map, n, a, n.arms[bid]));
      const hasS = ks.includes("straight"), hasL = ks.includes("left"), hasR = ks.includes("right");
      for (let k = 0; k < a.lanes; k++) {
        const c = at(sd + 9, inc((a.med || 0) + lw * (k + 0.5))), p = armAt(a, sd + 9);
        F.arrows.push(bbOfPts({ x: c[0], y: c[1], ang: Math.atan2(-p.ty, -p.tx), S: hasS, L: hasL && (a.lanes === 1 || k === 0), R: hasR && (a.lanes === 1 || k === a.lanes - 1) }, [c], 3));
      }
    }
    // turn restrictions at this approach: a no-left / no-right / no-U-turn sign at the stop line
    if (n.ban) for (const bid of n.order) if (n.ban.has(id + ">" + bid)) { const k = turnKind(map, n, a, n.arms[bid]); const p = at(sd + 2 + F.signs.length % 2, inc(a.hw + 1.6)); F.signs.push(bbOfPts({ x: p[0], y: p[1], k: "no_" + k }, [p], 2)); }
    if (signal) { const p = at(sd + 0.6, inc(a.ow === -1 ? -a.hw - 1.2 : a.hw + 1.2)); F.heads.push(bbOfPts({ j: n.id, arm: id, x: p[0], y: p[1], ang: a.ang }, [p], 2)); }
    if (rail) { const p = at(a.box - 0.3, inc(a.hw + 0.8)), q = at(a.box - 0.3, inc(-0.2)); F.barriers.push(bbOfPts({ j: n.id, arm: id, x: p[0], y: p[1], ex: q[0], ey: q[1] }, [p, q], 2)); }
  }
  // ---- rails
  for (const R of map.rails) {
    if (R.metro) {   // metro: viaduct sections (drawn overhead), nothing for the tunnel sections
      let run = []; const flush = () => { if (run.length > 1) F.viaducts.push(bbOfPts({ pts: run }, run, 12)); run = []; };
      R.pts.forEach((p, i) => { if ((R.z[i] || 0) > 2) run.push(p); else flush(); }); flush();
      continue;
    }
    if (!R.tunnel) F.rails.push(bbOfPts({ pts: R.pts }, R.pts, 3));
  }
  // ---- statics painted into the ground (they never move)
  for (const s of map.statics) F.statics.push(bbOfPts(Object.assign({}, s), [[s.x, s.y]], 3));
  // ---- buildings (oriented boxes)
  for (const h of map.buildings) {
    const c = Math.cos(h.ang), s = Math.sin(h.ang), corners = [[-h.w / 2, -h.d / 2], [h.w / 2, -h.d / 2], [h.w / 2, h.d / 2], [-h.w / 2, h.d / 2]].map(([u, v]) => [h.x + u * c - v * s, h.y + u * s + v * c]);
    const off = h.fl * 0.55, sh = convexHull(corners.concat(corners.map((p) => [p[0] + off * SUN[0], p[1] + off * SUN[1]])));
    F.houses.push(bbOfPts({ b: h, corners, shadow: sh }, sh, 2.5));
  }
  // ---- real map (OpenStreetMap): footprints, land use, water, works, signs, street and place names
  if (map.osm) {
    const TONE = ["#c8beb0", "#bfb3a2", "#d4cabb", "#b9ae9e", "#c9a98a", "#b8452e", "#9aa6ad", "#a8b0b5", "#d2c7b0", "#8e9aa3"];
    for (const b of map.osm.buildings) {
      const fl = Math.max(1, Math.round((b.h || 10) / 3.5)), off = fl * 0.55, sh = convexHull(b.p.concat(b.p.map((p) => [p[0] + off * SUN[0], p[1] + off * SUN[1]])));
      let hsh = 0; for (const p of b.p) hsh = (hsh * 31 + Math.round(p[0] * 7 + p[1] * 13)) | 0;
      F.houses.push(bbOfPts({ b: { fl, osm: true, poly: b.p, c: TONE[Math.abs(hsh) % TONE.length], big: b.p.length > 6 || fl > 5 }, shadow: sh }, sh, 2.5));
    }
    for (const a of map.osm.areas) F.land.push(bbOfPts({ poly: a.p, k: a.k }, a.p, 1));
    for (const w of map.osm.water) { if (w.p) F.water.push(bbOfPts({ poly: w.p }, w.p, 1)); else F.water.push(bbOfPts({ pts: w.l, w: w.w }, w.l, w.w / 2 + 1)); }
    for (const c of map.osm.cons) F.cons.push(bbOfPts({ pts: c.p }, c.p, 4));
  }
  for (const sg of map.signs || []) {
    if (sg.k === "bus") {   // a real bus stop: snap it to the curb of the nearest street, shelter facing the road
      let best = null; for (const L of map.links) { if (L.kind === "alley" || L.level) continue; for (let i = 0; i < L.pts.length; i += 2) { const d = (L.pts[i][0] - sg.x) ** 2 + (L.pts[i][1] - sg.y) ** 2; if (!best || d < best.d) best = { d, L, i }; } }
      if (!best || best.d > 900) continue;
      const L = best.L, i = Math.min(L.pts.length - 2, best.i), p = L.pts[i], q = L.pts[i + 1], tl = Math.hypot(q[0] - p[0], q[1] - p[1]) || 1, t = [(q[0] - p[0]) / tl, (q[1] - p[1]) / tl], nn = rightNormal(t);
      const sd = (sg.x - p[0]) * nn[0] + (sg.y - p[1]) * nn[1] >= 0 ? 1 : -1, off = L.hw + Math.max(1.4, (L.sw || 0) * 0.6);
      F.stops.push(bbOfPts({ x: p[0] + nn[0] * sd * off, y: p[1] + nn[1] * sd * off, ang: Math.atan2(t[1], t[0]) }, [[p[0], p[1]]], 6));
    }
    else if (sg.k === "stop" || sg.k === "yield" || sg.k === "sign") F.signs.push(bbOfPts({ x: sg.x, y: sg.y, k: sg.k, v: sg.v }, [[sg.x, sg.y]], 2));
  }
  // labels: one per street name every ~160 m, plus junction and place names (drawn live, so they stay sharp)
  if (map.preset === "hcm") {
    const seen = [];
    for (const L of map.links) {
      if (!L.name || L.len < 45 || L.level < 0) continue;
      const p = linkPoint(L, L.len / 2, 0);
      if (seen.some((q) => q.name === L.name && Math.hypot(q.x - p.x, q.y - p.y) < 160)) continue;
      let ang = p.ang; if (ang > Math.PI / 2) ang -= Math.PI; if (ang < -Math.PI / 2) ang += Math.PI;
      const q = { x: p.x, y: p.y, ang, name: L.name, major: ["main", "avenue", "highway", "sec"].includes(L.cls), z: L.level > 0 ? 1 : 0 }; seen.push(q); F.labels.push(q);
    }
    for (const lb of map.labels || []) F.labels.push({ x: lb.x, y: lb.y, ang: 0, name: lb.name, place: true, kind: lb.kind });
  }
  // ---- rice fields, villages' yards, forest
  for (const ar of map.areas) {
    if (ar.kind === "rural") for (let x = ar.x0; x < ar.x1; x += 26 + r() * 10) for (let y = ar.y0; y < ar.y1; y += 18 + r() * 8) {
      const w = 24 + r() * 10, h = 16 + r() * 8; F.fields.push(bbOfPts({ x, y, w, h, c: pick(r, ["#7fa24f", "#8bb156", "#97b95e", "#a9b55a", "#6f9a47", "#b8b968"]), rows: r() < 0.7, ang: (r() - 0.5) * 0.08 }, [[x, y], [x + w, y + h]], 1));
    }
  }
  // ---- trees: along town sidewalks and avenue medians, bamboo and trees along country roads, forest on the hills
  const free = (x, y, rad) => { const o = map.occ; for (const [dx, dy] of [[0, 0], [rad, 0], [-rad, 0], [0, rad], [0, -rad]]) { const k = o.cellOf(x + dx, y + dy); if (k < 0 || o.C[k] !== OCC.FREE && o.C[k] !== OCC.WALK) return false; } return true; };
  const freeLand = (x, y, rad) => { const o = map.occ; for (const [dx, dy] of [[0, 0], [rad, 0], [-rad, 0], [0, rad], [0, -rad]]) { const k = o.cellOf(x + dx, y + dy); if (k < 0 || o.C[k] !== OCC.FREE) return false; } return true; };
  for (const L of map.links) {
    const s0 = (L.armA ? L.armA.box : 0) + 8, s1 = L.len - (L.armB ? L.armB.box : 0) - 8;
    for (const sd of [1, -1]) {
      if (L.kind === "urban" && L.sw > 2) {
        for (let s = s0 + r() * 6; s < s1; s += 11 + r() * 8) { const p = linkPoint(L, s, sd * (L.hw + 0.9 + r() * 0.4)); if (free(p.x, p.y, 0.3)) F.trees.push({ x: p.x, y: p.y, r: 2.1 + r() * 1.7, seed: Math.floor(r() * 1e9) }); }
        const line = [];
        for (let s = s0 + 2; s < s1; s += 24 + r() * 8) { const p = linkPoint(L, s, sd * (L.hw + 0.45)); const q = { x: p.x, y: p.y, seed: Math.floor(r() * 1e9) }; F.poles.push(q); line.push(q); }
        for (let i = 0; i + 1 < line.length; i++) F.wires.push({ a: line[i], b: line[i + 1], n: 4 + Math.floor(r() * 4), seed: Math.floor(r() * 1e9) });
        if (line.length && r() < 0.35) { const p = line[Math.floor(r() * line.length)], q0 = linkPoint(L, s0 + r() * Math.max(1, s1 - s0), -sd * (L.hw + 0.45)), q = { x: q0.x, y: q0.y, seed: Math.floor(r() * 1e9) }; F.poles.push(q); F.wires.push({ a: p, b: q, n: 9, seed: Math.floor(r() * 1e9), thick: true }); }
        // sidewalk life: parked motorbikes, tea stalls, carts (denser where the sidewalk is taken)
        const dens = L.encroached || L.market ? 0.75 : 0.42;
        for (let s = s0 + r() * 4; s < s1 - 3;) {
          const k = r();
          if (k < dens * 0.7) { const n = 3 + Math.floor(r() * 8); for (let i = 0; i < n && s < s1 - 1; i++, s += 0.78) { const p = linkPoint(L, s, sd * (L.hw + L.sw - 1.0)); F.props.push({ kind: "moto", x: p.x, y: p.y, a: p.ang + sd * (Math.PI / 2 - 0.35) + (r() - 0.5) * 0.25, c: pick(r, MOTO_COLORS) }); } s += 1.5 + r() * 3; }
          else if (k < dens) { const p = linkPoint(L, s + 1, sd * (L.hw + L.sw * 0.55)); F.props.push({ kind: "stall", x: p.x, y: p.y, a: p.ang, seed: Math.floor(r() * 1e9) }); if (r() < 0.6) F.umbrellas.push({ x: p.x, y: p.y, r: 1.3 + r() * 0.3, c: pick(r, [["#e14b3b", "#f3eadb"], ["#2f7cc4", "#f1f1ea"], ["#f0b530", "#c9302c"], ["#3a9a5b", "#f2efe4"]]) }); s += 4 + r() * 3; }
          else s += 2 + r() * 4;
        }
      } else if (L.kind === "rural") {
        for (let s = s0; s < s1; s += 9 + r() * 22) { const p = linkPoint(L, s, sd * (L.hw + 2.2 + r() * 2.5)); if (freeLand(p.x, p.y, 1)) F.trees.push({ x: p.x, y: p.y, r: 1.6 + r() * 1.6, seed: Math.floor(r() * 1e9), bamboo: r() < 0.45 }); }
        if (r() < 0.6) { const line = []; for (let s = s0; s < s1; s += 35 + r() * 10) { const p = linkPoint(L, s, sd * (L.hw + 1.2)); line.push({ x: p.x, y: p.y, seed: Math.floor(r() * 1e9) }); } F.poles.push(...line); for (let i = 0; i + 1 < line.length; i++) F.wires.push({ a: line[i], b: line[i + 1], n: 2, seed: Math.floor(r() * 1e9) }); }
      } else if (L.kind === "highway") {
        for (let s = s0; s < s1; s += 18 + r() * 10) { const p = linkPoint(L, s, sd * (L.hw + 2.5)); if (freeLand(p.x, p.y, 1)) F.trees.push({ x: p.x, y: p.y, r: 2 + r() * 1.5, seed: Math.floor(r() * 1e9) }); }
      }
    }
  }
  for (const m of F.medians) if (m.trees) for (let i = 4; i < m.pts.length - 4; i += 9 + Math.floor(r() * 5)) F.trees.push({ x: m.pts[i][0], y: m.pts[i][1], r: 1.6 + r() * 0.8, seed: Math.floor(r() * 1e9) });
  for (const ar of map.areas) if (ar.kind === "forest") for (let k = 0; k < (ar.x1 - ar.x0) * (ar.y1 - ar.y0) / 70; k++) { const x = ar.x0 + r() * (ar.x1 - ar.x0), y = ar.y0 + r() * (ar.y1 - ar.y0); if (freeLand(x, y, 2.5)) F.trees.push({ x, y, r: 2.2 + r() * 2.2, seed: Math.floor(r() * 1e9), forest: true }); }
  for (const v of map.villages) for (let k = 0; k < 22; k++) { const a = r() * 6.28, d = 10 + r() * (v.r + 15), x = v.x + Math.cos(a) * d, y = v.y + Math.sin(a) * d; if (freeLand(x, y, 1.5)) F.trees.push({ x, y, r: 1.8 + r() * 2, seed: Math.floor(r() * 1e9), bamboo: r() < 0.3 }); }
  // asphalt wear on town roads
  for (const L of map.links) if ((L.kind === "urban" || L.kind === "highway") && !L.level) for (let s = 6; s < L.len - 6; s += 14 + r() * 26) { const p = linkPoint(L, s, (r() - 0.5) * 2 * (L.hw - 1)); F.patches.push({ x: p.x, y: p.y, w: 1 + r() * 2.4, h: 0.8 + r() * 1.8, a: p.ang, kind: r() < 0.18 ? "hole" : r() < 0.5 ? "dark" : "light" }); }
  const bb = (o, x0, y0, x1, y1) => { o.bx0 = x0; o.by0 = y0; o.bx1 = x1; o.by1 = y1; return o; };
  for (const t of F.trees) bb(t, t.x - t.r - 0.5, t.y - t.r - 0.5, t.x + t.r * 1.05 + 1.6, t.y + t.r + 2);
  for (const p of F.props) bb(p, p.x - 1.6, p.y - 1.6, p.x + 1.6, p.y + 1.6);
  for (const u of F.umbrellas) bb(u, u.x - u.r - 0.1, u.y - u.r - 0.1, u.x + u.r + 0.6, u.y + u.r + 0.7);
  for (const p of F.poles) bb(p, p.x - 0.8, p.y - 0.8, p.x + 1, p.y + 1);
  for (const w of F.wires) bb(w, Math.min(w.a.x, w.b.x) - 2, Math.min(w.a.y, w.b.y) - 2, Math.max(w.a.x, w.b.x) + 2, Math.max(w.a.y, w.b.y) + 2);
  for (const p of F.patches) bb(p, p.x - 2, p.y - 2, p.x + 2, p.y + 2);
  F.labels.sort((a, b) => (b.place ? 2 : b.major ? 1 : 0) - (a.place ? 2 : a.major ? 1 : 0));
  return F;
}

const hit = (o, x0, y0, x1, y1) => o.bx0 < x1 && o.bx1 > x0 && o.by0 < y1 && o.by1 > y0;
function fillShape(g, o) { g.beginPath(); if (o.disk) g.arc(o.disk[0], o.disk[1], o.disk[2], 0, 6.3); else { g.moveTo(o.poly[0][0], o.poly[0][1]); for (let i = 1; i < o.poly.length; i++) g.lineTo(o.poly[i][0], o.poly[i][1]); g.closePath(); } }
function strokePts(g, pts) { g.beginPath(); g.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length; i++) g.lineTo(pts[i][0], pts[i][1]); }

/* paint one tile [x0, x0+S) x [y0, y0+S) of the ground layer and the overhead layer */
function paintTile(map, F, x0, y0, S, ppm, lite) {
  const px = Math.round(S * ppm), ground = mkCanvas(px, px), over = mkCanvas(px, px);
  const g = ground.getContext("2d"), o = over.getContext("2d");
  for (const c of [g, o]) c.setTransform(ppm, 0, 0, ppm, -x0 * ppm, -y0 * ppm);
  const x1 = x0 + S, y1 = y0 + S, H = (o2) => hit(o2, x0 - 1, y0 - 1, x1 + 1, y1 + 1);
  const r = rngFrom(Math.floor(x0 * 73856093) ^ Math.floor(y0 * 19349663));
  // 1. land by area, then rice fields, then light and shade from the terrain
  g.fillStyle = map.osm ? LAND.urban : LAND.none; g.fillRect(x0, y0, S, S);
  const LANDC = { park: "#8fae6a", farm: "#a3b46d", forest: "#5d7a45", industrial: "#aaa79f", airport: "#b9c0a8" };
  for (const a of F.land) if (H(a)) { g.fillStyle = LANDC[a.k] || LAND.urban; fillShape(g, a); g.fill(); }
  for (const w of F.water) if (H(w)) {
    g.fillStyle = "#6f9fb8"; g.strokeStyle = "#6f9fb8"; g.lineCap = "round"; g.lineJoin = "round";
    if (w.poly) { fillShape(g, w); g.fill(); } else { g.lineWidth = w.w; strokePts(g, w.pts); g.stroke(); }
    if (!lite) { g.strokeStyle = "rgba(230,240,245,.18)"; g.lineWidth = 0.15; if (w.pts) { for (const d of [-w.w * 0.3, w.w * 0.15]) { strokePts(g, offsetPoly(w.pts, d)); g.stroke(); } } }
  }
  for (const ar of map.areas) if (ar.x1 > x0 && ar.x0 < x1 && ar.y1 > y0 && ar.y0 < y1) { g.fillStyle = LAND[ar.kind] || LAND.none; g.fillRect(ar.x0, ar.y0, ar.x1 - ar.x0, ar.y1 - ar.y0); }
  for (const f of F.fields) {
    if (!H(f)) continue;
    g.save(); g.translate(f.x + f.w / 2, f.y + f.h / 2); g.rotate(f.ang);
    g.fillStyle = f.c; g.fillRect(-f.w / 2, -f.h / 2, f.w, f.h);
    if (f.rows && !lite) { g.strokeStyle = "rgba(40,70,20,.16)"; g.lineWidth = 0.18; g.beginPath(); for (let k = -f.w / 2 + 0.6; k < f.w / 2; k += 0.9) { g.moveTo(k, -f.h / 2 + 0.5); g.lineTo(k, f.h / 2 - 0.5); } g.stroke(); }
    g.strokeStyle = "#b9ac82"; g.lineWidth = 0.55; g.strokeRect(-f.w / 2, -f.h / 2, f.w, f.h);
    g.restore();
  }
  if (!lite) for (let i = 0; i < S * S / 14; i++) { g.fillStyle = `rgba(${r() < 0.5 ? "60,55,45" : "255,250,240"},${0.04 + r() * 0.05})`; g.fillRect(x0 + r() * S, y0 + r() * S, 0.3 + r() * 1.5, 0.3 + r() * 1.5); }
  if (map.hills.length) {   // hillshade (light from the north-west) and contour lines every 5 m
    const st = lite ? 4 : 2;
    for (let y = y0; y < y1; y += st) for (let x = x0; x < x1; x += st) {
      const hx = map.H(x + st, y) - map.H(x - st, y), hy = map.H(x, y + st) - map.H(x, y - st), sh = (hx * 0.7 + hy * 0.7) / (2 * st);
      if (Math.abs(sh) > 0.004) { g.fillStyle = sh > 0 ? `rgba(20,30,15,${Math.min(0.32, sh * 1.6)})` : `rgba(255,250,225,${Math.min(0.18, -sh * 1.2)})`; g.fillRect(x, y, st, st); }
      const z = map.H(x, y), zr = map.H(x + st, y), zd = map.H(x, y + st);
      if (z > 2 && (Math.floor(z / 5) !== Math.floor(zr / 5) || Math.floor(z / 5) !== Math.floor(zd / 5))) { g.fillStyle = "rgba(60,50,30,.18)"; g.fillRect(x, y, st * 0.6, st * 0.6); }
    }
  }
  // 2. sidewalks
  g.fillStyle = "#cdc6b8"; for (const w of F.walks) if (H(w)) { fillShape(g, w); g.fill(); }
  if (!lite) { g.save(); g.beginPath(); for (const w of F.walks) if (H(w)) { if (w.disk) g.arc(w.disk[0], w.disk[1], w.disk[2], 0, 6.3); else { g.moveTo(w.poly[0][0], w.poly[0][1]); for (const p of w.poly) g.lineTo(p[0], p[1]); g.closePath(); } } g.clip();
    g.strokeStyle = "rgba(90,80,65,.09)"; g.lineWidth = 0.025; g.beginPath(); for (let x = Math.ceil(x0 / 0.4) * 0.4; x <= x1; x += 0.4) { g.moveTo(x, y0); g.lineTo(x, y1); } for (let y = Math.ceil(y0 / 0.4) * 0.4; y <= y1; y += 0.4) { g.moveTo(x0, y); g.lineTo(x1, y); } g.stroke(); g.restore(); }
  // 3. asphalt, medians, roundabout islands, wear
  for (const rd of F.roads) if (H(rd)) { g.fillStyle = rd.c; fillShape(g, rd); g.fill(); }
  if (!lite) {
    g.save(); g.beginPath(); for (const rd of F.roads) if (H(rd)) { if (rd.disk) g.arc(rd.disk[0], rd.disk[1], rd.disk[2], 0, 6.3); else { g.moveTo(rd.poly[0][0], rd.poly[0][1]); for (const p of rd.poly) g.lineTo(p[0], p[1]); g.closePath(); } } g.clip();
    for (let i = 0; i < S * S / 70; i++) { const x = x0 + r() * S, y = y0 + r() * S, rad = 4 + r() * 14; const gr = g.createRadialGradient(x, y, 0, x, y, rad); const dark = r() < 0.55; gr.addColorStop(0, dark ? "rgba(20,22,25,.10)" : "rgba(120,124,128,.07)"); gr.addColorStop(1, "rgba(0,0,0,0)"); g.fillStyle = gr; g.fillRect(x - rad, y - rad, rad * 2, rad * 2); }
    for (let i = 0; i < S * S * 0.9; i++) { const x = x0 + r() * S, y = y0 + r() * S; const l = r() < 0.5; g.fillStyle = l ? `rgba(200,200,195,${0.05 + r() * 0.08})` : `rgba(10,10,12,${0.06 + r() * 0.1})`; const s = 0.04 + r() * 0.08; g.fillRect(x, y, s, s); }
    for (const p of F.patches) { if (!hit(p, x0, y0, x1, y1)) continue; g.save(); g.translate(p.x, p.y); g.rotate(p.a); if (p.kind === "hole") { g.fillStyle = "#34373b"; g.beginPath(); g.arc(0, 0, 0.34, 0, 6.3); g.fill(); g.strokeStyle = "#5b5f64"; g.lineWidth = 0.05; g.beginPath(); g.arc(0, 0, 0.3, 0, 6.3); g.stroke(); } else { g.fillStyle = p.kind === "dark" ? "#3c4045" : "#4d5157"; g.fillRect(-p.w / 2, -p.h / 2, p.w, p.h); g.strokeStyle = "rgba(15,15,18,.4)"; g.lineWidth = 0.05; g.strokeRect(-p.w / 2, -p.h / 2, p.w, p.h); } g.restore(); }
    g.restore();
  }
  for (const m of F.medians) if (H(m)) { g.fillStyle = "#d9d4c8"; fillShape(g, m); g.fill(); g.fillStyle = m.trees ? "#6f8f4a" : "#7c9a55"; fillShape(g, { poly: stripPoly(m.pts, -m.w + 0.25, m.w - 0.25) }); g.fill(); }
  for (const is of F.islands) if (H(is)) {
    g.fillStyle = "#ece8de"; g.beginPath(); g.arc(is.x, is.y, is.R, 0, 6.3); g.fill();
    g.fillStyle = "#6f9a4a"; g.beginPath(); g.arc(is.x, is.y, is.R - 0.6, 0, 6.3); g.fill();
    const q = rngFrom(is.seed); for (let k = 0; k < 40; k++) { const a = q() * 6.28, d = (0.3 + q() * 0.6) * is.R; g.fillStyle = pick(q, ["#e2405a", "#f2c230", "#f07a2a", "#fafafa", "#c94bb0"]); g.beginPath(); g.arc(is.x + Math.cos(a) * d, is.y + Math.sin(a) * d, 0.25 + q() * 0.25, 0, 6.3); g.fill(); }
    g.fillStyle = "#cbbf9f"; g.beginPath(); g.arc(is.x, is.y, Math.min(3.5, is.R * 0.3), 0, 6.3); g.fill(); g.fillStyle = "#9c8f70"; g.beginPath(); g.arc(is.x, is.y, Math.min(1.6, is.R * 0.14), 0, 6.3); g.fill();
  }
  // 4. water over flooded stretches
  for (const f of F.floods) if (H(f)) {
    g.save(); fillShape(g, f); g.clip();
    const gr = g.createLinearGradient(f.pts[0][0], f.pts[0][1], f.pts[f.pts.length - 1][0], f.pts[f.pts.length - 1][1]);
    gr.addColorStop(0, "rgba(120,104,70,0)"); gr.addColorStop(0.18, "rgba(120,104,70,.72)"); gr.addColorStop(0.82, "rgba(120,104,70,.72)"); gr.addColorStop(1, "rgba(120,104,70,0)");
    g.fillStyle = gr; fillShape(g, f); g.fill();
    const q = rngFrom(f.seed); g.strokeStyle = "rgba(230,225,200,.35)"; g.lineWidth = 0.06;
    for (let k = 0; k < 60; k++) { const p = f.pts[Math.floor(q() * f.pts.length)], rr = 0.4 + q() * 1.4; g.beginPath(); g.arc(p[0] + (q() - 0.5) * f.w * 2, p[1] + (q() - 0.5) * f.w * 2, rr, q() * 6, q() * 6 + 2); g.stroke(); }
    g.restore();
  }
  // 5. road paint
  g.lineCap = "butt";
  for (const l of F.lines) if (H(l)) { g.strokeStyle = l.c; g.lineWidth = l.w; g.setLineDash(l.dash || []); if (l.circle) { g.beginPath(); g.arc(l.circle[0], l.circle[1], l.circle[2], 0, 6.3); } else strokePts(g, l.pts); g.stroke(); }
  g.setLineDash([]);
  for (const qd of F.quads) if (H(qd)) { g.fillStyle = qd.c; fillShape(g, qd); g.fill(); if (qd.edge) { g.strokeStyle = qd.edge; g.lineWidth = 0.15; fillShape(g, qd); g.stroke(); } }
  for (const a of F.arrows) if (H(a)) drawArrow2(g, a.x, a.y, a.ang, a.S, a.L, a.R, "rgba(238,236,228,.88)");
  // works in progress and tunnels (portals and a faint outline of the tube underground)
  for (const c of F.cons) if (H(c)) { g.strokeStyle = "rgba(230,140,40,.75)"; g.lineWidth = 2.2; g.setLineDash([2, 2]); strokePts(g, c.pts); g.stroke(); g.setLineDash([]); }
  for (const t of F.tunnels) if (H(t)) { g.strokeStyle = "rgba(40,44,48,.35)"; g.lineWidth = t.hw * 2; g.setLineDash([3, 3]); strokePts(g, t.pts); g.stroke(); g.setLineDash([]); }
  // 6. rail tracks (over the road at level crossings)
  for (const R of F.rails) if (H(R)) {
    g.strokeStyle = "#8f8576"; g.lineWidth = 3.2; g.lineCap = "round"; strokePts(g, R.pts); g.stroke();
    g.strokeStyle = "#5b4636"; g.lineWidth = 2.5; g.setLineDash([0.22, 0.42]); strokePts(g, R.pts); g.stroke(); g.setLineDash([]);
    g.strokeStyle = "#b8bcc0"; g.lineWidth = 0.12; for (const d of [-0.72, 0.72]) { strokePts(g, offsetPoly(R.pts, d)); g.stroke(); }
  }
  // 7. mountain road edges: rock cut and guard rail
  for (const c of F.cliffs) if (H(c)) { g.strokeStyle = "#7d776c"; g.lineWidth = 1.6; g.lineCap = "round"; strokePts(g, c.pts); g.stroke(); g.strokeStyle = "rgba(40,36,30,.35)"; g.lineWidth = 0.08; for (let i = 0; i < c.pts.length; i += 2) { const p = c.pts[i]; g.beginPath(); g.moveTo(p[0] - 0.5, p[1] - 0.3); g.lineTo(p[0] + 0.4, p[1] + 0.5); g.stroke(); } }
  for (const c of F.guards) if (H(c)) { g.strokeStyle = "#e8e8e2"; g.lineWidth = 0.22; strokePts(g, c.pts); g.stroke(); g.strokeStyle = "#c33"; g.lineWidth = 0.22; g.setLineDash([0.4, 1.6]); strokePts(g, c.pts); g.stroke(); g.setLineDash([]); }
  // 8. sidewalk life, statics (parked cars, stalls, motorbikes) under the traffic
  for (const p of F.props) {
    if (!hit(p, x0, y0, x1, y1)) continue;
    g.save(); g.translate(p.x, p.y); g.rotate(p.a);
    if (p.kind === "moto") { drawShadowStatic(g, 1.9, 0.6); drawMoto(g, { look: { body: p.c }, cargo: null, delivery: false }, true); }
    else drawTeaStall(g, p.seed);
    g.restore();
  }
  for (const s of F.statics) if (H(s)) drawStatic(g, s);
  for (const t of F.trees) if (hit(t, x0, y0, x1, y1) && !t.forest) { g.fillStyle = "#6b5a46"; g.fillRect(t.x - 0.5, t.y - 0.5, 1, 1); g.fillStyle = "#4b3d2f"; g.beginPath(); g.arc(t.x, t.y, 0.18, 0, 6.3); g.fill(); }
  for (const b of F.stops) if (H(b)) { g.save(); g.translate(b.x, b.y); g.rotate(b.ang); g.fillStyle = "rgba(0,0,0,.25)"; g.fillRect(-2.6, -0.4, 5.6, 1.6); g.fillStyle = "#2f6fb5"; g.fillRect(-2.8, -0.7, 5.6, 1.4); g.fillStyle = "#e9eef2"; g.fillRect(-2.6, -0.5, 5.2, 0.35); g.restore(); }
  // 9. houses: shadows (low to high), roofs, awnings
  const hs = F.houses.filter((h) => hit(h, x0, y0, x1, y1)).sort((a, b) => a.b.fl - b.b.fl);
  for (const h of hs) { g.fillStyle = "rgba(15,18,22,.26)"; fillShape(g, { poly: h.shadow }); g.fill(); }
  for (const h of hs) {
    const b = h.b;
    if (b.osm) {
      g.fillStyle = b.c; fillShape(g, { poly: b.poly }); g.fill();
      g.strokeStyle = "rgba(40,36,30,.35)"; g.lineWidth = 0.15; g.stroke();
      if (b.big) { g.strokeStyle = "rgba(255,255,255,.22)"; g.lineWidth = 0.6; fillShape(g, { poly: b.poly }); g.save(); g.clip(); g.stroke(); g.restore(); }
      continue;
    }
    g.save(); g.translate(b.x, b.y); g.rotate(b.ang);
    if (b.kind === "village") { g.fillStyle = "rgba(205,190,150,.6)"; g.fillRect(-b.w / 2 - 2.5, -b.d / 2 - 2.5, b.w + 5, b.d + 5); }
    const lb = { x: -b.w / 2, y: -b.d / 2, w: b.w, h: b.d, front: b.front === "right" ? "n" : b.front === "left" ? "s" : null, style: b.style, c: b.c };
    drawRoof(g, lb, rngFrom(b.seed));
    if (b.awning && lb.front) drawAwning(g, lb, rngFrom(b.seed + 1));
    g.restore();
  }
  for (const t of F.trees) if (hit(t, x0, y0, x1, y1)) { g.fillStyle = "rgba(15,22,14,.2)"; g.beginPath(); g.ellipse(t.x + 1.4, t.y + 1.8, t.r * 1.05, t.r * 0.95, 0, 0, 6.3); g.fill(); }
  // ---- overhead layer: flyover decks (with their shadow on the road below), metro viaducts, signs
  for (const d of F.decks) if (H(d)) { o.save(); o.translate(SUN[0] * 1.8, SUN[1] * 1.8); o.fillStyle = "rgba(15,18,22,.28)"; fillShape(o, d); o.fill(); o.restore(); }
  for (const d of F.decks) if (H(d)) {
    o.fillStyle = "#b9b4aa"; fillShape(o, d); o.fill();
    if (d.pts) { o.fillStyle = d.c; fillShape(o, { poly: stripPoly(d.pts, -d.hw + 0.05, d.hw - 0.05) }); o.fill(); o.strokeStyle = "#e3dfd4"; o.lineWidth = 0.35; for (const sd of [-1, 1]) { strokePts(o, offsetPoly(d.pts, sd * (d.hw + 0.3))); o.stroke(); } }
    else { o.fillStyle = d.c; fillShape(o, d); o.fill(); }
  }
  o.lineCap = "butt";
  for (const l of F.deckLines) if (H(l)) { if (l.ow) { drawArrow2(o, l.x, l.y, l.ang, true, false, false, "rgba(238,236,228,.85)"); continue; } o.strokeStyle = l.c; o.lineWidth = l.w; o.setLineDash(l.dash || []); strokePts(o, l.pts); o.stroke(); }
  o.setLineDash([]);
  for (const v of F.viaducts) if (H(v)) {
    o.save(); o.translate(SUN[0] * 2.4, SUN[1] * 2.4); o.strokeStyle = "rgba(15,18,22,.25)"; o.lineWidth = 9; o.lineCap = "round"; strokePts(o, v.pts); o.stroke(); o.restore();
    o.strokeStyle = "#cfcac0"; o.lineWidth = 9; o.lineCap = "butt"; strokePts(o, v.pts); o.stroke();
    o.strokeStyle = "#8f8a80"; o.lineWidth = 0.3; for (const d of [-4.3, 4.3]) { strokePts(o, offsetPoly(v.pts, d)); o.stroke(); }
    o.strokeStyle = "#6b655c"; o.lineWidth = 2.6; o.setLineDash([0.2, 0.45]); for (const d of [-2, 2]) { strokePts(o, offsetPoly(v.pts, d)); o.stroke(); } o.setLineDash([]);
  }
  for (const sg of F.signs) if (H(sg)) drawSign(o, sg);
  for (const u of F.umbrellas) {
    if (!hit(u, x0, y0, x1, y1)) continue;
    o.fillStyle = "rgba(0,0,0,.16)"; o.beginPath(); o.arc(u.x + 0.5, u.y + 0.6, u.r, 0, 6.3); o.fill();
    for (let k = 0; k < 8; k++) { o.fillStyle = u.c[k % 2]; o.beginPath(); o.moveTo(u.x, u.y); o.arc(u.x, u.y, u.r, k * Math.PI / 4, (k + 1) * Math.PI / 4); o.closePath(); o.fill(); }
    o.fillStyle = "#555"; o.beginPath(); o.arc(u.x, u.y, 0.08, 0, 6.3); o.fill();
  }
  for (const s of F.statics) if (s.kind === "stall" && H(s)) { const q = rngFrom(s.seed); o.save(); o.translate(s.x, s.y); o.rotate(s.h); o.fillStyle = "rgba(0,0,0,.15)"; o.fillRect(-s.l / 2 + 0.3, -s.w / 2 - 0.4, s.l + 0.2, s.w + 1.4); o.fillStyle = pick(q, ["#2f7cc4", "#e14b3b", "#f0b530", "#3a9a5b", "#8c4fb0"]); o.fillRect(-s.l / 2 - 0.2, -s.w / 2 - 0.6, s.l + 0.4, s.w + 1.0); o.fillStyle = "rgba(255,255,255,.18)"; for (let k = -s.l / 2; k < s.l / 2; k += 0.6) o.fillRect(k, -s.w / 2 - 0.6, 0.3, s.w + 1.0); o.restore(); }
  for (const t of F.trees) if (hit(t, x0, y0, x1, y1)) (t.bamboo ? drawBamboo : drawCanopy)(o, t, rngFrom(t.seed));
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
/* a road sign seen from above: a disc (or triangle) on a short pole, big enough to read when zoomed in */
function drawSign(g, sg) {
  g.save(); g.translate(sg.x, sg.y);
  g.fillStyle = "rgba(0,0,0,.25)"; g.beginPath(); g.arc(0.25, 0.3, 0.62, 0, 6.3); g.fill();
  g.fillStyle = "#6c7075"; g.fillRect(-0.05, 0, 0.1, 0.7);
  const ring = (fill, edge) => { g.fillStyle = edge; g.beginPath(); g.arc(0, 0, 0.6, 0, 6.3); g.fill(); g.fillStyle = fill; g.beginPath(); g.arc(0, 0, 0.46, 0, 6.3); g.fill(); };
  if (sg.k === "noentry") { g.fillStyle = "#d62b25"; g.beginPath(); g.arc(0, 0, 0.6, 0, 6.3); g.fill(); g.fillStyle = "#fff"; g.fillRect(-0.4, -0.1, 0.8, 0.2); }
  else if (sg.k === "stop") { g.fillStyle = "#d62b25"; g.beginPath(); for (let k = 0; k < 8; k++) { const a = Math.PI / 8 + k * Math.PI / 4; g.lineTo(Math.cos(a) * 0.62, Math.sin(a) * 0.62); } g.closePath(); g.fill(); g.fillStyle = "#fff"; g.fillRect(-0.38, -0.06, 0.76, 0.12); }
  else if (sg.k === "yield") { g.fillStyle = "#d62b25"; g.beginPath(); g.moveTo(-0.62, -0.42); g.lineTo(0.62, -0.42); g.lineTo(0, 0.62); g.closePath(); g.fill(); g.fillStyle = "#fff"; g.beginPath(); g.moveTo(-0.36, -0.26); g.lineTo(0.36, -0.26); g.lineTo(0, 0.34); g.closePath(); g.fill(); }
  else if (sg.k && sg.k.startsWith("no_")) {
    ring("#fff", "#d62b25"); g.strokeStyle = "#1d1f22"; g.lineWidth = 0.1; g.lineCap = "round"; g.beginPath();
    if (sg.k === "no_left") { g.moveTo(0.1, 0.3); g.lineTo(0.1, -0.05); g.lineTo(-0.25, -0.05); }
    else if (sg.k === "no_right") { g.moveTo(-0.1, 0.3); g.lineTo(-0.1, -0.05); g.lineTo(0.25, -0.05); }
    else if (sg.k === "no_uturn") { g.moveTo(0.15, 0.3); g.lineTo(0.15, -0.1); g.arc(0, -0.1, 0.15, 0, Math.PI, true); g.lineTo(-0.15, 0.15); }
    else { g.moveTo(0, 0.3); g.lineTo(0, -0.3); }
    g.stroke(); g.strokeStyle = "#d62b25"; g.lineWidth = 0.12; g.beginPath(); g.moveTo(-0.33, -0.33); g.lineTo(0.33, 0.33); g.stroke();
  } else { ring("#f4f2ea", "#2f6fb5"); g.fillStyle = "#2f6fb5"; g.beginPath(); g.arc(0, 0, 0.14, 0, 6.3); g.fill(); }
  g.restore();
}
function drawTeaStall(g, seed) {
  const q = rngFrom(seed);
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
/* static obstacles in the road: parked cars, market stalls (their tarps go in the overhead layer), parked or stalled motorbikes */
function drawStatic(g, s) {
  g.save(); g.translate(s.x, s.y); g.rotate(s.h);
  if (s.kind === "parked") { const v = { id: s.seed % 100000, type: "car", T: { l: s.l, w: s.w }, look: null, braking: false }; g.fillStyle = "rgba(10,14,18,.24)"; roundRect(g, -s.l / 2 + 0.2, -s.w / 2 + 0.25, s.l, s.w, 0.5); g.fill(); drawCar(g, Object.assign(v, { look: lookFor(v) })); }
  else if (s.kind === "stall") { const q = rngFrom(s.seed); g.fillStyle = "#8a6a45"; g.fillRect(-s.l / 2, -s.w / 2, s.l, s.w); for (let k = 0; k < 8; k++) { g.fillStyle = pick(q, ["#e2582c", "#f2c230", "#5fa83a", "#c9302c", "#7a3f8f", "#e8e2cf"]); g.beginPath(); g.arc(-s.l / 2 + 0.3 + q() * (s.l - 0.6), -s.w / 2 + 0.2 + q() * (s.w - 0.4), 0.18 + q() * 0.12, 0, 6.3); g.fill(); } }
  else { drawShadowStatic(g, 1.9, 0.6); drawMoto(g, { look: { body: MOTO_COLORS[s.seed % MOTO_COLORS.length] }, cargo: null, delivery: false }, true); if (s.kind === "stalled") { g.strokeStyle = "rgba(230,225,200,.5)"; g.lineWidth = 0.05; for (const rr of [1.2, 1.6]) { g.beginPath(); g.ellipse(0, 0, rr, rr * 0.6, 0, 0, 6.3); g.stroke(); } } }
  g.restore();
}
function drawBamboo(o, t, r) {
  for (let k = 0; k < 14; k++) { const a = r() * 6.28, d = r() * t.r * 0.7; o.fillStyle = r() < 0.5 ? "rgba(110,150,60,.9)" : "rgba(80,120,45,.9)"; o.beginPath(); o.ellipse(t.x + Math.cos(a) * d, t.y + Math.sin(a) * d, t.r * 0.45, t.r * 0.18, a, 0, 6.3); o.fill(); }
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
  constructor(map, seed = 11, ppm = 14, tile = 80, keep = 14) { this.map = map; this.F = cityFeatures(map, seed); this.ppm = ppm; this.S = tile; this.keep = keep; this.cache = new Map(); this.heads = this.F.heads; this.barriers = this.F.barriers; }
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
  /* one low-resolution picture of the whole map for zoomed-out views */
  overview() {
    if (!this.ov) { const B = this.map.bounds, S = Math.max(B.x1 - B.x0, B.y1 - B.y0) + 20; this.ov = paintTile(this.map, this.F, B.x0 - 10, B.y0 - 10, S, Math.min(3, 2200 / S), true); this.ovS = S; this.ovX = B.x0 - 10; this.ovY = B.y0 - 10; }
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

/* ---------------- signals with countdown, level-crossing barriers, drawn every frame */
function drawSignals(g, sim, layers, vx0, vy0, vx1, vy1) {
  for (const s of layers.heads) {
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
    g.fillStyle = "#111214"; roundRect(g, -0.32, 0.85, 0.62, 0.62, 0.08); g.fill();
    g.save(); g.translate(0, 1.16); g.rotate(-s.ang);
    g.fillStyle = L.c === "G" ? "#39f07a" : L.c === "Y" ? "#ffc21a" : "#ff4a3d";
    g.scale(0.02, 0.02); g.font = "700 21px 'JetBrains Mono', ui-monospace, monospace"; g.textAlign = "center"; g.textBaseline = "middle";
    g.fillText(String(Math.ceil(L.left)).padStart(2, "0"), 0, 1);
    g.restore(); g.restore();
  }
  for (const b of layers.barriers) {
    if (b.x < vx0 - 10 || b.x > vx1 + 10 || b.y < vy0 - 10 || b.y > vy1 + 10) continue;
    const L = sim.light(b.j, b.arm); if (!L) continue;
    const closed = L.c === "R" ? 1 : L.c === "Y" ? Math.max(0, 1 - L.left / 6) * 0.15 : 0, ang = Math.atan2(b.ey - b.y, b.ex - b.x), len = Math.hypot(b.ex - b.x, b.ey - b.y);
    g.save(); g.translate(b.x, b.y);
    g.fillStyle = "#2a2c30"; g.fillRect(-0.3, -0.3, 0.6, 0.6);
    const flash = L.c !== "G" && Math.floor(sim.t * 2.5) % 2 === 0;
    g.fillStyle = flash ? "#ff3b30" : "#4a1714"; g.beginPath(); g.arc(0, 0, 0.22, 0, 6.3); g.fill();
    if (flash) { const gr = g.createRadialGradient(0, 0, 0, 0, 0, 1.4); gr.addColorStop(0, "rgba(255,59,48,.55)"); gr.addColorStop(1, "rgba(255,59,48,0)"); g.fillStyle = gr; g.beginPath(); g.arc(0, 0, 1.4, 0, 6.3); g.fill(); }
    // the boom swings down across the road; seen from above it shortens while raised
    const vis = 0.12 + 0.88 * closed;
    g.rotate(ang); g.fillStyle = "rgba(0,0,0,.25)"; g.fillRect(0.2, 0.15, len * vis, 0.22);
    for (let k = 0; k < len * vis; k += 0.6) { g.fillStyle = Math.floor(k / 0.6) % 2 ? "#f4f2ea" : "#d63a2f"; g.fillRect(k, -0.11, Math.min(0.6, len * vis - k), 0.22); }
    g.restore();
  }
}
/* ---------------- people, buffaloes, trains (drawn every frame) */
// how much larger than life a pedestrian is drawn at the current zoom (also used for the rings of "Show what it sees")
function pedScale(g) {
  const M = g.getTransform(), ppm = Math.hypot(M.a, M.b) / ((typeof devicePixelRatio !== "undefined" && devicePixelRatio) || 1);
  return clampN(20 / Math.max(1e-3, 0.55 * ppm), 1.3, 3.5);
}
function drawAgent(g, a, x, y, h, t) {
  if (a.static) return;
  g.save(); g.translate(x, y); g.rotate(h);
  if (a.kind === "ped") {
    // a person is only ~0.5 m across, a few pixels at street zoom: draw a little larger than life when zoomed in close, and enlarge
    // (up to 3.5x) when zoomed out so a person stays at least ~20 screen pixels across the shoulders
    const k = pedScale(g); g.scale(k, k);
    const q = rngFrom(a.seed), shirt = pick(q, SHIRTS), hat = q() < 0.3, hair = pick(q, ["#1d1714", "#2a211c", "#3b2d24"]);
    const sw = a.v > 0.2 ? Math.sin(t * 7 + a.seed) * 0.14 : 0;
    g.fillStyle = "rgba(10,14,18,.28)"; g.beginPath(); g.ellipse(0.08, 0.1, 0.3, 0.36, 0, 0, 6.3); g.fill();
    // dark outline under everything, so the figure reads on the light sidewalk and on asphalt alike
    g.fillStyle = "rgba(18,20,24,.7)";
    g.beginPath(); g.ellipse(sw, -0.1, 0.14, 0.085, 0, 0, 6.3); g.fill(); g.beginPath(); g.ellipse(-sw, 0.1, 0.14, 0.085, 0, 0, 6.3); g.fill();
    g.beginPath(); g.ellipse(0, 0, 0.175, 0.3, 0, 0, 6.3); g.fill();
    // legs striding along the heading, arms swinging opposite, shoulders
    g.fillStyle = "#34363a"; g.beginPath(); g.ellipse(sw, -0.1, 0.11, 0.055, 0, 0, 6.3); g.fill(); g.beginPath(); g.ellipse(-sw, 0.1, 0.11, 0.055, 0, 0, 6.3); g.fill();
    g.fillStyle = shirt;
    g.beginPath(); g.ellipse(0, 0, 0.14, 0.265, 0, 0, 6.3); g.fill();
    g.fillStyle = "#c99a77"; g.beginPath(); g.arc(-sw * 0.9, -0.25, 0.045, 0, 6.3); g.fill(); g.beginPath(); g.arc(sw * 0.9, 0.25, 0.045, 0, 6.3); g.fill();   // hands
    if (hat) {   // conical hat (non la), smaller than the shoulders so the shirt still shows
      g.fillStyle = "#ead9a3"; g.beginPath(); g.arc(0.02, 0, 0.165, 0, 6.3); g.fill();
      g.strokeStyle = "rgba(120,95,40,.6)"; g.lineWidth = 0.02; g.beginPath(); g.arc(0.02, 0, 0.1, 0, 6.3); g.stroke();
      g.fillStyle = "#f7eecb"; g.beginPath(); g.arc(0.02, 0, 0.035, 0, 6.3); g.fill();
    } else { g.fillStyle = "#c99a77"; g.beginPath(); g.arc(0.03, 0, 0.1, 0, 6.3); g.fill(); g.fillStyle = hair; g.beginPath(); g.arc(0.01, 0, 0.1, 1.75, 4.55); g.fill(); }
  } else if (a.kind === "buffalo") {
    g.fillStyle = "rgba(10,14,18,.25)"; g.beginPath(); g.ellipse(0.15, 0.2, 1.2, 0.55, 0, 0, 6.3); g.fill();
    g.fillStyle = "#3d3a38"; g.beginPath(); g.ellipse(-0.1, 0, 1.0, 0.5, 0, 0, 6.3); g.fill();
    g.fillStyle = "#4b4744"; g.beginPath(); g.ellipse(0.95, 0, 0.32, 0.26, 0, 0, 6.3); g.fill();
    g.strokeStyle = "#cfc6b4"; g.lineWidth = 0.08; g.beginPath(); g.arc(0.9, 0, 0.45, -1.2, -0.4); g.stroke(); g.beginPath(); g.arc(0.9, 0, 0.45, 0.4, 1.2); g.stroke();
    g.strokeStyle = "#2b2826"; g.lineWidth = 0.05; g.beginPath(); g.moveTo(-1.05, 0); g.lineTo(-1.35, Math.sin(t * 3 + a.seed) * 0.15); g.stroke();
  } else if (a.kind === "train") {
    const L = a.T.l, W = a.T.w;
    g.fillStyle = "rgba(10,14,18,.3)"; g.fillRect(-L / 2 + 0.3, -W / 2 + 0.4, L, W);
    g.fillStyle = a.loco ? "#2f5d9a" : "#3a6fae"; roundRect(g, -L / 2, -W / 2, L, W, 0.4); g.fill();
    g.fillStyle = "#e9e4d6"; g.fillRect(-L / 2 + 0.3, -W / 2 + 0.25, L - 0.6, 0.35); g.fillRect(-L / 2 + 0.3, W / 2 - 0.6, L - 0.6, 0.35);
    g.fillStyle = "#c9302c"; g.fillRect(-L / 2 + 0.3, -0.12, L - 0.6, 0.24);
    if (a.loco) { g.fillStyle = "#f2c230"; g.fillRect(L / 2 - 1.2, -W / 2, 1.2, W); g.fillStyle = "#1c2731"; g.fillRect(L / 2 - 1.0, -W / 2 + 0.3, 0.5, W - 0.6); }
    else { g.fillStyle = "rgba(0,0,0,.15)"; for (let k = -L / 2 + 1; k < L / 2 - 1; k += 1.6) g.fillRect(k, -W / 2 + 0.7, 0.9, W - 1.4); }
  }
  g.restore();
}
function drawTractor(g, v) {
  g.fillStyle = "#1c1c1e"; g.fillRect(-1.4, -0.9, 0.9, 0.32); g.fillRect(-1.4, 0.58, 0.9, 0.32); g.fillRect(0.9, -0.55, 0.5, 0.2); g.fillRect(0.9, 0.35, 0.5, 0.2);
  g.fillStyle = "#9b7a4c"; g.fillRect(-1.7, -0.75, 1.9, 1.5);
  g.strokeStyle = "rgba(60,40,20,.5)"; g.lineWidth = 0.05; for (let k = -1.6; k < 0.2; k += 0.3) { g.beginPath(); g.moveTo(k, -0.75); g.lineTo(k, 0.75); g.stroke(); }
  g.fillStyle = "#2f7f57"; roundRect(g, 0.25, -0.45, 1.3, 0.9, 0.15); g.fill();
  g.fillStyle = "#3a3a3c"; g.fillRect(0.6, -0.08, 0.8, 0.16);
  drawRider(g, 0.4, v.look.shirt, v.look.helmet, true);
}

function drawHonk(g, v, x, y, t) {
  const a = 1 - (0.6 - v.honk) / 0.6;
  g.save(); g.translate(x, y); g.rotate(v.h);
  g.strokeStyle = `rgba(255,214,90,${0.85 * a})`; g.lineWidth = 0.07;
  for (let k = 0; k < 3; k++) { g.beginPath(); g.arc(v.T.l / 2, 0, 0.35 + k * 0.25 + (1 - a) * 0.3, -0.6, 0.6); g.stroke(); }
  g.restore();
}

if (typeof module !== "undefined") module.exports = {};
