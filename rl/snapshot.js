// Trusted local checkpoints only. Preserve aliases, prototypes, typed arrays, Maps,
// Sets and seeded RNGs. Raster editing closures are build-time only; cellOf is
// reattached from its data after restore. Never deserialize untrusted input.
const v8 = require('v8');
function encode(root) {
  const seen = new Map(), nodes = [];
  function visit(x) {
    if (x === undefined) return { u: 1 };
    if (x === null || (typeof x !== 'object' && typeof x !== 'function')) return { v: x };
    if (seen.has(x)) return { r: seen.get(x) };
    const id = nodes.length; seen.set(x, id); nodes.push(null);
    let n;
    if (typeof x === 'function') {
      if (!x.st || typeof x.st.a !== 'number') throw Error('Unsupported checkpoint function');
      n = { type: 'rng', state: x.st.a };
    } else if (ArrayBuffer.isView(x)) n = { type: 'typed', name: x.constructor.name, data: Buffer.from(x.buffer, x.byteOffset, x.byteLength) };
    else if (x instanceof Map) n = { type: 'map', data: [...x].map(([k, v]) => [visit(k), visit(v)]) };
    else if (x instanceof Set) n = { type: 'set', data: [...x].map(visit) };
    else {
      n = { type: Array.isArray(x) ? 'array' : 'object', name: x.constructor && x.constructor.name, data: [] };
      for (const k of Object.keys(x)) {
        if (['H', 'cellOf', 'stampPoly', 'fillPoly', 'fits', 'mark'].includes(k) && typeof x[k] === 'function') continue;
        n.data.push([k, visit(x[k])]);
      }
    }
    nodes[id] = n; return { r: id };
  }
  const entry = visit(root);
  return v8.serialize({ version: 1, entry, nodes }).toString('base64');
}
function decode(data) {
  const { version, entry, nodes } = v8.deserialize(Buffer.from(data, 'base64'));
  if (version !== 1) throw Error('Unsupported worker snapshot');
  const classes = { EgoEnv, Sim, Vehicle, Agent, NpcBrain, LookaheadPlanner };
  const values = nodes.map(n => {
    if (n.type === 'rng') { const r = rngFrom(0); r.st.a = n.state; return r; }
    if (n.type === 'typed') { const bytes = Uint8Array.from(n.data); return new global[n.name](bytes.buffer); }
    if (n.type === 'map') return new Map();
    if (n.type === 'set') return new Set();
    if (n.type === 'array') return [];
    return Object.create(classes[n.name] ? classes[n.name].prototype : Object.prototype);
  });
  const read = t => 'r' in t ? values[t.r] : 'u' in t ? undefined : t.v;
  nodes.forEach((n, i) => {
    const x = values[i];
    if (n.type === 'map') for (const [k, v] of n.data) x.set(read(k), read(v));
    else if (n.type === 'set') for (const v of n.data) x.add(read(v));
    else if (n.type === 'object' || n.type === 'array') for (const [k, v] of n.data) x[k] = read(v);
    if (x instanceof Vehicle && TYPES[x.type]) x.T = TYPES[x.type];
    if (x && Array.isArray(x.hills) && Array.isArray(x.links)) {
      x.H = (px, py) => x.hills.reduce((z, h) => z + h.h * Math.exp(-((px - h.x) ** 2 + (py - h.y) ** 2) / (2 * h.r * h.r)), 0);
    }
    if (x && x.G instanceof Float32Array && x.O instanceof Float32Array && x.nx) {
      x.cellOf = (px, py) => { const ix = Math.round((px - x.x0) / x.res), iy = Math.round((py - x.y0) / x.res); return ix >= 0 && iy >= 0 && ix < x.nx && iy < x.ny ? iy * x.nx + ix : -1; };
    }
  });
  return read(entry);
}
module.exports = { encode, decode };
