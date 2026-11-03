"""
LOTUS training monitor: a small web server that shows the training while it runs.

Started automatically by pipeline.py (port 8765). It can also run on its own, for example
next to a training script started by hand, or after the run to look at the results:

  python monitor/server.py                       # watches rl/runs on port 8765
  python monitor/server.py --runs rl/runs_smoke --port 9000

Pages:   /       dashboard (stage, learning curves, scenario archive, server load, log)
         /sim    the simulator, driving with the newest checkpoints (reload to update)
API:     /api/status   everything the dashboard shows, as JSON
         /api/log?n=   the last n lines of pipeline.log

Python standard library only. Read-only: it never writes into the run folder.
"""
import argparse, glob, json, os, re, shutil, socket, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "web"))


def load(path, default=None):
    """JSON that may be missing or half-written (the trainer writes while we read)"""
    for _ in range(2):
        try:
            with open(path, encoding="utf-8") as f: return json.load(f)
        except FileNotFoundError: return default
        except (ValueError, OSError): time.sleep(0.05)
    return default


def tail(path, n):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2); size = f.tell(); f.seek(max(0, size - 200 * n))
            lines = f.read().decode("utf-8", "replace").splitlines()
        return [re.sub(r"\x1b\[[0-9;]*m", "", l) for l in lines[-n:]]
    except OSError: return []


def num(p, pat, d=0):
    m = re.findall(pat, p); return int(m[0]) if m else d


# ---------------------------------------------------------------- server load
class Host:
    def __init__(self, path):
        self.path, self.prev, self.gpu, self.gpu_t = path, None, None, 0

    def cpu(self):
        try:
            v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
            idle, total = v[3] + v[4], sum(v)
            pct = None
            if self.prev: di, dt = idle - self.prev[0], total - self.prev[1]; pct = round(100 * (1 - di / dt), 1) if dt else None
            self.prev = (idle, total); return pct
        except OSError: return None

    def gpus(self):
        if time.time() - self.gpu_t < 3: return self.gpu
        self.gpu_t = time.time()
        if not shutil.which("nvidia-smi"): self.gpu = []; return self.gpu
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=4).stdout
            self.gpu = [dict(zip(("name", "util", "mem_used", "mem_total", "temp"), [p.strip() for p in l.split(",")])) for l in out.strip().splitlines() if l.strip()]
            for g in self.gpu:
                for k in ("util", "mem_used", "mem_total", "temp"):
                    try: g[k] = float(g[k])
                    except ValueError: g[k] = None
        except (OSError, subprocess.SubprocessError): self.gpu = []
        return self.gpu

    def snapshot(self):
        mem = {}
        try:
            for l in open("/proc/meminfo"):
                k, v = l.split(":"); mem[k] = int(v.split()[0]) * 1024
        except OSError: pass
        try: du = shutil.disk_usage(self.path if os.path.exists(self.path) else ROOT)
        except OSError: du = None
        nodes = 0
        for p in glob.glob("/proc/[0-9]*/cmdline"):
            try:
                c = open(p, "rb").read().split(b"\0")[0]
                if c.endswith(b"node"): nodes += 1
            except OSError: pass
        return {"cpu_pct": self.cpu(), "cores": os.cpu_count(), "load": os.getloadavg() if hasattr(os, "getloadavg") else None,
                "mem_used": mem.get("MemTotal", 0) - mem.get("MemAvailable", 0) if mem else None, "mem_total": mem.get("MemTotal"),
                "disk_free": du.free if du else None, "disk_total": du.total if du else None, "gpus": self.gpus(), "node_workers": nodes,
                "hostname": socket.gethostname()}


# ---------------------------------------------------------------- what the run folder says
def ppo_runs(runs):
    """every PPO learning curve: phase 2 fine-tuning, phase 3 generations, SB3 runs"""
    out = []
    cands = [("finetune_v2", "Phase 2 · PPO")]
    cands += [(os.path.relpath(d, runs), f"Phase 3 · gen {num(d, r'ft_g(\d+)') + 1}") for d in sorted(glob.glob(os.path.join(runs, "coevolve", "ft_g*")), key=lambda d: num(d, r"ft_g(\d+)"))]
    cands += [(os.path.basename(d), "SB3 · " + os.path.basename(d)[4:].upper()) for d in sorted(glob.glob(os.path.join(runs, "sb3_*")))]
    for rel, name in cands:
        d = os.path.join(runs, rel); rows = load(os.path.join(d, "log.json"))
        if not rows: continue
        st = load(os.path.join(d, "status.json"), {})
        out.append({"id": rel, "name": name, "rows": rows, "total": st.get("total"), "done": os.path.exists(os.path.join(d, "final.json")),
                    "updated": os.path.getmtime(os.path.join(d, "log.json"))})
    return out


def checkpoints(runs):
    res = []
    for f in glob.glob(os.path.join(runs, "**", "*.json"), recursive=True):
        b = os.path.basename(f)
        rel = os.path.relpath(f, runs)
        if not re.match(r"(student_r\d+|step_\d+|final|latest)\.json$", b) or {"data", "tmp"} & set(rel.split(os.sep)): continue
        res.append({"path": rel, "mtime": os.path.getmtime(f)})
    return sorted(res, key=lambda r: r["mtime"])[-12:]


def status(runs, host):
    p2, co = os.path.join(runs, "teacher_v2"), os.path.join(runs, "coevolve")
    ev = lambda n: load(os.path.join(runs, "eval", n))
    return {
        "now": time.time(), "runs": os.path.abspath(runs),
        "pipeline": load(os.path.join(runs, "pipeline_status.json"), {}),
        "summary": load(os.path.join(runs, "pipeline_summary.json")),
        "host": host.snapshot(),
        "eval": {"idm": ev("idm.json"), "planner": ev("planner.json"), "phase2": ev("phase2_final.json")},
        "distill": {"log": load(os.path.join(p2, "distill_log.json"), []), "status": load(os.path.join(p2, "status.json"))},
        "ppo": ppo_runs(runs),
        "coevolve": {"log": load(os.path.join(co, "coevolve_log.json"), []), "status": load(os.path.join(co, "status.json")),
                     "progress": (load(os.path.join(co, "progress.json"), {}) or {}).get("rows", []),
                     "archive": load(os.path.join(co, "archive_live.json")), "result": load(os.path.join(co, "result.json"))},
        "phase4": {"log": load(os.path.join(runs, "phase4", "phase4_log.json"), []), "result": load(os.path.join(runs, "phase4", "result.json"))},
        "npc": {"status": load(os.path.join(runs, "npc", "status.json")), "report": load(os.path.join(runs, "npc", "report.json"))},
        "wm": {"status": load(os.path.join(runs, "tdmpc2", "status.json")), "result": load(os.path.join(runs, "tdmpc2", "eval_final.json"))},
        "checkpoints": checkpoints(runs),
        "log_mtime": os.path.getmtime(os.path.join(runs, "pipeline.log")) if os.path.exists(os.path.join(runs, "pipeline.log")) else None,
    }


# ---------------------------------------------------------------- live simulator page
LIVE_JS = """<script>
(function(){var sig=null;function chk(){fetch('/api/ckpt').then(function(r){return r.json()}).then(function(j){
if(sig===null){sig=j.sig;return}if(j.sig!==sig){var b=document.getElementById('lotusLive');b.hidden=false;
if(live){sig=j.sig;b.textContent='New checkpoint · loading';setTimeout(function(){location.reload()},2500)}}}).catch(function(){})}
var live=/[?&]driver=latest/.test(location.search);setInterval(chk,live?10000:20000);chk();})();
</script><style>#lotusLive{position:fixed;left:50%;bottom:86px;transform:translateX(-50%);z-index:99;background:#0f9f92;color:#fff;border:0;border-radius:999px;
padding:10px 18px;font:600 14px system-ui;box-shadow:0 6px 20px rgba(0,0,0,.25);cursor:pointer}</style>"""
LIVE_BTN = '<button id="lotusLive" hidden onclick="location.reload()">New checkpoint · reload</button>'


class Sim:
    def __init__(self, runs): self.runs, self.sig, self.html, self.lock = runs, None, None, threading.Lock()

    def dirs(self):
        return ([os.path.join(self.runs, "teacher_v2"), os.path.join(self.runs, "finetune_v2"), os.path.join(self.runs, "tdmpc2")] +
                sorted(glob.glob(os.path.join(self.runs, "coevolve", "ft_g*")), key=lambda d: num(d, r"ft_g(\d+)")))

    def signature(self):
        fs = []
        for d in self.dirs():
            if os.path.basename(d) == "tdmpc2": continue   # swapped in live by the page (/api/policy), no reload needed
            for f in glob.glob(os.path.join(d, "*.json")):
                if re.match(r"(student_r\d+|step_\d+|final|latest|policy)\.json$", os.path.basename(f)): fs.append((f, os.path.getmtime(f)))
        return str(hash(tuple(sorted(fs))))

    def page(self):
        with self.lock:
            sig = self.signature()
            if sig != self.sig:
                from build import build_page, standalone
                frag, _ = build_page(self.dirs())
                self.html, self.sig = standalone(frag + LIVE_BTN, LIVE_JS).encode("utf-8"), sig
            return self.html


# ---------------------------------------------------------------- HTTP
def make_handler(runs, host, sim):
    dash = os.path.join(HERE, "dashboard.html")

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass   # keep the training log clean

        def send(self, body, ctype="application/json; charset=utf-8", code=200):
            if isinstance(body, str): body = body.encode("utf-8")
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path); q = parse_qs(u.query)
            try:
                if u.path in ("/", "/index.html"): self.send(open(dash, encoding="utf-8").read(), "text/html; charset=utf-8")
                elif u.path == "/api/status": self.send(json.dumps(status(runs, host)))
                elif u.path == "/api/log": self.send(json.dumps(tail(os.path.join(runs, "pipeline.log"), min(2000, int(q.get("n", ["200"])[0])))))
                elif u.path == "/api/ckpt": self.send(json.dumps({"sig": sim.signature()}))
                elif u.path in ("/sim", "/drive"): self.send(sim.page(), "text/html; charset=utf-8")   # /drive opens on the test drive
                elif u.path == "/api/drive":   # the test drive of rl/drive_test.py: the journey, or one stretch (?id=)
                    d = os.path.join(runs, "drive")
                    if "id" in q:
                        f = os.path.join(d, f"seg_{int(q['id'][0]):03d}.json")
                        if os.path.isfile(f): self.send(open(f, "rb").read())
                        else: self.send(json.dumps({"error": "stretch not found"}), code=404)
                    else: self.send(json.dumps(load(os.path.join(d, "journey.json")) or {"segments": []}))
                elif u.path == "/api/episodes":   # recorded TD-MPC2 training episodes, replayed by the /sim page
                    d = os.path.join(runs, "tdmpc2", "episodes")
                    if "id" in q:
                        f = os.path.join(d, f"ep_{int(q['id'][0]):06d}.json")
                        if os.path.isfile(f): self.send(open(f, "rb").read())
                        else: self.send(json.dumps({"error": "episode not found"}), code=404)
                    else: self.send(json.dumps({"episodes": load(os.path.join(d, "index.json")) or []}))
                elif u.path == "/api/policy":   # the newest TD-MPC2 snapshot: the /sim page swaps it in while it drives
                    f = os.path.join(runs, "tdmpc2", "policy.json")
                    if "meta" in q: self.send(json.dumps({"v": os.path.getmtime(f) if os.path.isfile(f) else None}))
                    elif os.path.isfile(f): self.send(open(f, "rb").read())
                    else: self.send(json.dumps({"error": "no TD-MPC2 snapshot yet"}), code=404)
                elif u.path.startswith("/hcm/"):   # the Ho Chi Minh City map data (web/hcm), fetched by the /sim page
                    base = os.path.realpath(os.path.join(ROOT, "web", "hcm")); f = os.path.realpath(os.path.join(base, u.path[5:]))
                    if f.startswith(base + os.sep) and os.path.isfile(f): self.send(open(f, "rb").read())
                    else: self.send(json.dumps({"error": "no map data: build it with data/build_hcm.js"}), code=404)
                else: self.send(json.dumps({"error": "not found"}), code=404)
            except (BrokenPipeError, ConnectionResetError): pass
            except Exception as e:   # never let a bad file take the monitor down
                try: self.send(json.dumps({"error": repr(e)}), code=500)
                except Exception: pass
    return H


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try: s.connect(("10.255.255.255", 1)); return s.getsockname()[0]   # no packet is sent; this only asks the OS which interface it would use
    except OSError: return None
    finally: s.close()


def start(runs, port=8765, bind="0.0.0.0", quiet=False):
    """starts the monitor in a background thread; returns (server, url) or (None, None) if no port was free"""
    host, sim = Host(runs), Sim(runs)
    srv = None
    for p in range(port, port + 20):
        try: srv = ThreadingHTTPServer((bind, p), make_handler(runs, host, sim)); break
        except OSError: continue
    if srv is None:
        print(f"[monitor] no free port in {port}-{port + 19}; continuing without the dashboard", flush=True); return None, None
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    p = srv.server_address[1]; ip = lan_ip() if bind in ("0.0.0.0", "") else bind
    url = f"http://{ip or 'localhost'}:{p}"
    if not quiet:
        user = os.environ.get("USER") or "user"
        lines = ["LOTUS training monitor is running",
                 f"  Open in a browser:      {url}",
                 f"  Simulator, live:        {url}/sim",
                 "  If that address is not reachable (firewall, cloud server), on your own computer run:",
                 f"    ssh -L {p}:localhost:{p} {user}@{ip or '<server>'}",
                 f"  then open http://localhost:{p}"]
        w = max(len(l) for l in lines) + 2
        print("\n┌" + "─" * w + "┐\n" + "\n".join("│ " + l.ljust(w - 1) + "│" for l in lines) + "\n└" + "─" * w + "┘\n", flush=True)
    return srv, url


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "rl", "runs"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--bind", default="0.0.0.0", help="127.0.0.1 to allow only an SSH tunnel")
    a = ap.parse_args()
    srv, _ = start(a.runs, a.port, a.bind)
    if srv:
        try:
            while True: time.sleep(3600)
        except KeyboardInterrupt: print("\n[monitor] stopped")
