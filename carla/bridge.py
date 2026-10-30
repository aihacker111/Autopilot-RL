"""
LOTUS, Phase 4: bridge between the 2D core and CARLA.

The 2D core (bridge_worker.js) moves every road user, runs the signals and labels each moment with the
teacher's action. CARLA only renders: road users are spawned without physics and placed every tick
where the 2D core says, signals are set to the 2D core's colours, and three cameras on the car are read.

  # collect teacher-driven data (DAgger round 0)
  python carla/bridge.py collect --data runs/phase4/data --episodes 40 --teacher runs/coevolve/ft_g3/final.json
  # collect with the student driving part of the time (DAgger round k)
  python carla/bridge.py collect --data runs/phase4/data --episodes 40 --student runs/phase4/student_r0.pt --beta 0.5 --teacher ...
  # closed-loop evaluation of the student
  python carla/bridge.py drive --student runs/phase4/student_r3.pt --episodes 30 --teacher ...
  # everything without CARLA (synthetic images), to test the pipeline
  python carla/bridge.py collect --dry-run --data /tmp/p4 --episodes 2 --teacher idm

Requirements: CARLA 0.9.13-0.9.16 server running (./CarlaUE4.sh -RenderOffScreen), `pip install carla==<server version>`.
"""
import argparse, json, math, os, queue, random, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
CAMS = [("left", -55.0), ("front", 0.0), ("right", 55.0)]
W, H, FOV = 384, 216, 70
BEV = 24
SIZE = {"moto": (1.9, 0.72), "bike": (1.7, 0.6), "car": (4.5, 1.8), "taxi": (4.4, 1.75), "bus": (10.5, 2.5), "truck": (7.0, 2.3)}


class Sim:
    """the 2D core running in a Node process"""
    def __init__(self, a):
        self.p = subprocess.Popen(["node", os.path.join(HERE, "bridge_worker.js")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        genome = json.load(open(a.genome)) if a.genome else None
        self.info = self.call({"cmd": "init", "mapSeed": a.map_seed, "nx": a.nx, "ny": a.ny, "density": a.density, "genome": genome, "teacher": a.teacher, "seed": a.seed})
    def call(self, m): self.p.stdin.write(json.dumps(m) + "\n"); self.p.stdin.flush(); return json.loads(self.p.stdout.readline())
    def reset(self): return self.call({"cmd": "reset"})
    def step(self, action): return self.call({"cmd": "step", "action": action})
    def close(self):
        try: self.p.stdin.write('{"cmd":"close"}\n'); self.p.stdin.flush()
        except Exception: pass


def bev_grid(st):
    """privileged occupancy around the car: 24 m ahead x 24 m wide, 1 m cells (target of an auxiliary loss)"""
    g = np.zeros((BEV, BEV), np.uint8); e = st["ego"]; c, s = math.cos(e["h"]), math.sin(e["h"])
    for _, typ, x, y, h, _ in st["actors"]:
        dx, dy = x - e["x"], y - e["y"]; fx, fy = dx * c + dy * s, -dx * s + dy * c
        if not (-1 < fx < BEV + 1 and -BEV / 2 - 1 < fy < BEV / 2 + 1): continue
        L, Wd = SIZE.get(typ, (2, 1)); ch, sh = math.cos(h - e["h"]), math.sin(h - e["h"])
        for u in np.linspace(-L / 2, L / 2, max(2, int(L * 2))):
            for v in np.linspace(-Wd / 2, Wd / 2, 2):
                px, py = fx + u * ch - v * sh, fy + u * sh + v * ch
                i, j = int(px), int(py + BEV / 2)
                if 0 <= i < BEV and 0 <= j < BEV: g[BEV - 1 - i, j] = 255
    return g


# ---------------------------------------------------------------- renderers
class DryRenderer:
    """synthetic camera images drawn from the 2D state (no CARLA): tests the whole pipeline end to end"""
    def __init__(self, sim_info, a): self.juncs = sim_info["juncs"]
    def update(self, st):
        from PIL import Image, ImageDraw
        e = st["ego"]; out = []
        light = {"0": (40, 220, 90), "1": (240, 190, 30), "2": (230, 50, 40), "3": (90, 90, 90)}[str(st["aux"]["light"])]
        for _, yaw in CAMS:
            im = Image.new("RGB", (W, H), (120, 150, 190)); d = ImageDraw.Draw(im)
            d.rectangle([0, H // 2, W, H], fill=(70, 72, 76))
            h0 = e["h"] + math.radians(yaw)
            for _, typ, x, y, h, brk in st["actors"]:
                dx, dy = x - e["x"], y - e["y"]; fx, fy = dx * math.cos(h0) + dy * math.sin(h0), -dx * math.sin(h0) + dy * math.cos(h0)
                if fx < 1 or fx > 40 or abs(fy) > fx * 0.8: continue
                sx = W / 2 + fy / fx * (W / 2) / math.tan(math.radians(FOV / 2)); size = 260 / fx * (SIZE.get(typ, (2, 1))[1] + 0.6)
                col = (200, 40, 40) if brk else ((230, 230, 230) if typ in ("car", "taxi") else (40, 60, 160))
                d.rectangle([sx - size / 2, H / 2 - size * 0.2, sx + size / 2, H / 2 + size * 0.6], fill=col)
            if yaw == 0: d.rectangle([W - 30, 6, W - 8, 28], fill=light)
            out.append(np.asarray(im))
        return np.stack(out)
    def close(self): pass


class CarlaRenderer:
    def __init__(self, sim_info, a):
        import carla
        self.carla = carla
        xodr = open(a.xodr).read()
        self.client = carla.Client(a.host, a.port); self.client.set_timeout(120.0)
        params = carla.OpendriveGenerationParameters(vertex_distance=2.0, max_road_length=500.0, wall_height=0.0, additional_width=0.6, smooth_junctions=True, enable_mesh_visibility=True)
        self.world = self.client.generate_opendrive_world(xodr, params)
        s = self.world.get_settings(); s.synchronous_mode = True; s.fixed_delta_seconds = sim_info["dt"]; s.no_rendering_mode = False
        self.world.apply_settings(s)
        self.lib = self.world.get_blueprint_library(); self.rng = random.Random(a.seed)
        self.models = {k: [b for b in v if self.lib.filter(b)] for k, v in {
            "moto": ["vehicle.yamaha.yzf", "vehicle.kawasaki.ninja", "vehicle.vespa.zx125", "vehicle.harley-davidson.low_rider"],
            "bike": ["vehicle.diamondback.century", "vehicle.gazelle.omafiets", "vehicle.bh.crossbike"],
            "car": ["vehicle.toyota.prius", "vehicle.mini.cooper_s", "vehicle.nissan.micra", "vehicle.seat.leon", "vehicle.citroen.c3", "vehicle.audi.a2"],
            "taxi": ["vehicle.ford.crown", "vehicle.toyota.prius"], "bus": ["vehicle.mitsubishi.fusorosa"],
            "truck": ["vehicle.carlamotors.carlacola", "vehicle.mercedes.sprinter", "vehicle.volkswagen.t2"]}.items()}
        self.actors, self.ego, self.cams, self.queues = {}, None, [], []
        self.juncs = sim_info["juncs"]; self.tl = {}
        self.weathers = [carla.WeatherParameters.ClearNoon, carla.WeatherParameters.CloudyNoon, carla.WeatherParameters.WetNoon, carla.WeatherParameters.ClearSunset, carla.WeatherParameters.SoftRainNoon]
        self.world.set_weather(self.rng.choice(self.weathers))
        self._map_lights()

    def _map_lights(self):
        """match CARLA's generated traffic lights to (junction, arm) of the 2D core, and freeze them"""
        for tl in self.world.get_actors().filter("traffic.traffic_light"):
            try: wps = tl.get_stop_waypoints(); loc = wps[0].transform.location if wps else tl.get_location()
            except Exception: loc = tl.get_location()
            j = min(self.juncs, key=lambda q: (q["x"] - loc.x) ** 2 + (q["y"] - loc.y) ** 2)
            dx, dy = loc.x - j["x"], loc.y - j["y"]
            arm = ("E" if dx > 0 else "W") if abs(dx) > abs(dy) else ("S" if dy > 0 else "N")
            self.tl.setdefault((j["id"], arm), []).append(tl); tl.freeze(True)
        if not self.tl: print("warning: CARLA generated no traffic lights for this map; the student will not see signals (use --v2i-light)", flush=True)

    def _tf(self, x, y, h, z=0.05):
        c = self.carla; return c.Transform(c.Location(x=x, y=y, z=z), c.Rotation(yaw=math.degrees(h)))

    def _spawn(self, model, k):
        c = self.carla; bp = self.lib.find(model)
        if bp.has_attribute("color"): bp.set_attribute("color", self.rng.choice(bp.get_attribute("color").recommended_values))
        act = self.world.try_spawn_actor(bp, c.Transform(c.Location(x=0, y=0, z=300 + 3 * (k % 100)), c.Rotation()))
        if act: act.set_simulate_physics(False)
        return act

    def _ensure_ego(self, e):
        if self.ego: return
        c = self.carla
        self.ego = self._spawn(next(m for m in ["vehicle.lincoln.mkz_2020", "vehicle.tesla.model3", "vehicle.toyota.prius"] if self.lib.filter(m)), 0)
        cam = self.lib.find("sensor.camera.rgb"); cam.set_attribute("image_size_x", str(W)); cam.set_attribute("image_size_y", str(H)); cam.set_attribute("fov", str(FOV))
        for name, yaw in CAMS:
            s = self.world.spawn_actor(cam, c.Transform(c.Location(x=1.4, z=1.55), c.Rotation(yaw=yaw)), attach_to=self.ego)
            q = queue.Queue(); s.listen(q.put); self.cams.append(s); self.queues.append(q)

    def update(self, st):
        c = self.carla; e = st["ego"]; self._ensure_ego(e)
        seen, batch = set(), [c.command.ApplyTransform(self.ego.id, self._tf(e["x"], e["y"], e["h"]))]
        for aid, typ, x, y, h, brk in st["actors"]:
            seen.add(aid); act = self.actors.get(aid)
            if act is None:
                models = self.models.get(typ) or self.models["car"]
                act = self._spawn(self.rng.choice(models), aid) if models else None
                if act is None: continue
                self.actors[aid] = act
            batch.append(c.command.ApplyTransform(act.id, self._tf(x, y, h)))
        gone = [k for k in self.actors if k not in seen]
        if gone: self.client.apply_batch([c.command.DestroyActor(self.actors.pop(k).id) for k in gone])
        self.client.apply_batch(batch)
        state = {"G": c.TrafficLightState.Green, "Y": c.TrafficLightState.Yellow, "R": c.TrafficLightState.Red}
        lit = set()
        for j, arm, col in st["lights"]:
            for tl in self.tl.get((j, arm), []): tl.set_state(state[col]); lit.add(tl.id)
        for tls in self.tl.values():
            for tl in tls:
                if tl.id not in lit: tl.set_state(c.TrafficLightState.Off)
        frame = self.world.tick()
        imgs = []
        for q in self.queues:
            im = q.get(timeout=10.0)
            while im.frame < frame: im = q.get(timeout=10.0)
            a = np.frombuffer(im.raw_data, np.uint8).reshape(H, W, 4)[:, :, :3][:, :, ::-1]
            imgs.append(a.copy())
        return np.stack(imgs)

    def new_episode(self): self.world.set_weather(self.rng.choice(self.weathers))

    def close(self):
        for s in self.cams: s.stop()
        acts = [s.id for s in self.cams] + [a.id for a in self.actors.values()] + ([self.ego.id] if self.ego else [])
        self.client.apply_batch([self.carla.command.DestroyActor(i) for i in acts])
        s = self.world.get_settings(); s.synchronous_mode = False; self.world.apply_settings(s)


# ---------------------------------------------------------------- collection and driving
def save_episode(ep_dir, rows):
    from PIL import Image
    os.makedirs(ep_dir, exist_ok=True)
    for i, r in enumerate(rows):
        for (name, _), img in zip(CAMS, r["imgs"]): Image.fromarray(img).save(os.path.join(ep_dir, f"{i:05d}_{name}.jpg"), quality=90)
    from student import nav_vec
    np.savez_compressed(os.path.join(ep_dir, "meta.npz"), label=np.array([r["label"] for r in rows], np.float32), executed=np.array([r["exec"] for r in rows], np.float32),
                        nav=np.stack([nav_vec(r["nav"]) for r in rows]), light=np.array([r["light"] for r in rows], np.int64), bev=np.stack([r["bev"] for r in rows]),
                        teacher_drove=np.array([r["teacher"] for r in rows], bool))


def run(a):
    if not a.dry_run and not a.xodr:   # export the same city for CARLA
        a.xodr = os.path.join(a.out_dir, f"city{a.map_seed}.xodr")
        os.makedirs(a.out_dir, exist_ok=True)
        subprocess.run(["node", os.path.join(HERE, "export_xodr.js"), "--seed", str(a.map_seed), "--nx", str(a.nx), "--ny", str(a.ny), "--out", a.xodr], check=True)
    sim = Sim(a)
    ren = (DryRenderer if a.dry_run else CarlaRenderer)(sim.info, a)
    student = None
    if a.student:
        from student import StudentDriver
        student = StudentDriver(a.student, device=a.device)
    rng = random.Random(a.seed); results = []
    existing = len([d for d in os.listdir(a.data) if d.startswith("ep_")]) if a.mode == "collect" and os.path.isdir(a.data) else 0
    st = sim.reset(); rows, ep = [], 0; t0 = time.time()
    try:
        while ep < a.episodes:
            imgs = ren.update(st)
            teacher_turn = student is None or (a.mode == "collect" and rng.random() < a.beta)
            nav = dict(st["nav"])
            act = None if teacher_turn else student.act(imgs, nav)
            if a.mode == "collect": rows.append({"imgs": imgs, "label": st["label"], "exec": st["label"] if teacher_turn else act, "nav": nav, "light": st["aux"]["light"], "bev": bev_grid(st), "teacher": teacher_turn})
            st = sim.step(act)
            if st.get("done"):
                results.append(st["info"]); ep += 1
                if a.mode == "collect": save_episode(os.path.join(a.data, f"ep_{existing + ep:05d}"), rows)
                rows = []
                if student: student.reset()
                if hasattr(ren, "new_episode"): ren.new_episode()
                print(json.dumps({"episode": ep, "result": st["info"]["result"], "time": round(st["info"]["time"], 1), "min": round((time.time() - t0) / 60, 1)}), flush=True)
    finally:
        ren.close(); sim.close()
    m = lambda k: round(sum(r["result"] == k for r in results) / max(1, len(results)), 3)
    summ = {"mode": a.mode, "episodes": len(results), "ok": m("ok"), "crash": m("crash"), "red": round(sum(bool(r["red"]) for r in results) / max(1, len(results)), 3), "timeout": m("timeout"), "hit": m("hit")}
    print(json.dumps(summ))
    if a.result: json.dump(summ, open(a.result, "w"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["collect", "drive"])
    p.add_argument("--data", default=os.path.join(HERE, "..", "rl", "runs", "phase4", "data"))
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--teacher", default="planner", help="policy JSON from Phase 2/3, 'planner' or 'idm'")
    p.add_argument("--student", default=None, help="student checkpoint (student.py train)")
    p.add_argument("--beta", type=float, default=0.5, help="collect: probability that the teacher drives a given decision")
    p.add_argument("--map-seed", type=int, default=7); p.add_argument("--nx", type=int, default=3); p.add_argument("--ny", type=int, default=3)
    p.add_argument("--density", type=float, default=0.8); p.add_argument("--genome", default=None, help="Phase 3 scenario genome JSON")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--xodr", default=None); p.add_argument("--out-dir", default=os.path.join(HERE, "..", "rl", "runs", "phase4"))
    p.add_argument("--host", default="127.0.0.1"); p.add_argument("--port", type=int, default=2000)
    p.add_argument("--device", default="cuda"); p.add_argument("--dry-run", action="store_true"); p.add_argument("--result", default=None)
    a = p.parse_args()
    if a.mode == "collect": os.makedirs(a.data, exist_ok=True)
    run(a)
