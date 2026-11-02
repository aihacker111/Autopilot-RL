"""
Builds web/lotus.html: the simulator page with the learned drivers embedded.

  python web/build.py                 # learned drivers from rl/runs/{teacher_v2,finetune_v2}
  AVRUN=dir1,dir2 python web/build.py # other run folders
  python web/build.py test.html       # also write a standalone local test copy

The training monitor (monitor/server.py) calls build_page() to serve the same page live,
with the newest checkpoints, while training runs.
"""
import json, os, sys, glob, re
HERE = os.path.dirname(os.path.abspath(__file__))
rd = lambda p: open(os.path.join(HERE, p), encoding="utf-8").read()
num = lambda p, pat: int(re.findall(pat, p)[0])
DEFAULT_RUNS = [os.path.join(HERE, '..', 'rl', 'runs', d) for d in ('teacher_v2', 'finetune_v2', 'tdmpc2')]
NPC_RUN = os.path.join(HERE, '..', 'rl', 'runs', 'npc', 'npc.json')


def label(run, base, spec):
    """short button label, full name and description for one checkpoint file"""
    rname, steps = os.path.basename(os.path.normpath(run)), spec.get('steps', 0)
    if spec.get('type') == 'tdmpc':
        return 'TD-MPC2', 'world model (TD-MPC2)', 'Policy prior of the TD-MPC2 world model. In training it plans with its learned model of the world; here it drives with the prior alone.'
    if base.startswith('student'):
        return 'Student', 'distilled student', 'Neural network distilled from the look-ahead teacher (DAgger).'
    gen = re.findall(r'ft_g(\d+)', rname)
    if base == 'latest.json':
        return 'Live', 'network in training', f"Latest snapshot of the network being trained ({steps // 1000}k steps)."
    if gen:
        return f"Gen {int(gen[0]) + 1}", f"co-evolved, gen {int(gen[0]) + 1}", f"The car after co-evolution generation {int(gen[0]) + 1}: fine-tuned among the hard scenarios the population found."
    tag = f"{num(base, r'step_(\d+)') // 1000}k" if base.startswith('step') else 'final'
    return 'PPO ' + tag, 'PPO ' + tag, f"The student fine-tuned further with PPO, {steps // 1000}k steps."


def collect(runs, keep=4):
    """checkpoints from the run folders, oldest first; keeps the last `keep`"""
    ck, log = [], None
    for run in runs:
        if not os.path.isdir(run): continue
        files = sorted(glob.glob(os.path.join(run, 'student_r*.json')), key=lambda p: num(p, r'student_r(\d+)'))[-1:]   # the last student only
        files += sorted([p for p in glob.glob(os.path.join(run, 'step_*.json')) if num(p, r'step_(\d+)') > 0], key=lambda p: num(p, r'step_(\d+)'))
        if os.path.exists(os.path.join(run, 'policy.json')): files.append(os.path.join(run, 'policy.json'))   # TD-MPC2 policy prior
        fin, lat = os.path.join(run, 'final.json'), os.path.join(run, 'latest.json')
        if os.path.exists(fin): files.append(fin)
        if os.path.exists(lat) and (not os.path.exists(fin) or os.path.getmtime(lat) > os.path.getmtime(fin)): files.append(lat)
        for f in files:
            try: spec = json.load(open(f))
            except (OSError, ValueError): continue   # being written right now
            if spec.get('type') != 'set' and len(spec.get('obsMean', '')) < 1000: continue   # skip old single-junction checkpoints
            if spec.get('type') in ('set', 'tdmpc') and spec.get('egoDim') != 43: continue   # an older observation layout (before v3)
            spec['short'], spec['name'], d = label(run, os.path.basename(f), spec)   # English labels, whatever name the checkpoint carries
            spec['desc'] = spec.get('desc') or d
            ck.append(spec)
        if os.path.exists(os.path.join(run, 'log.json')):
            try: L = json.load(open(os.path.join(run, 'log.json')))
            except (OSError, ValueError): L = None
            if L and 'ok' in L[0]: log = [{k: r[k] for k in ('steps', 'ok', 'crash', 'red')} for r in L]
    return ({'ckpts': ck[-keep:], 'log': log or []} if ck else None), len(ck)


def build_page(runs=None):
    av, n = collect(runs or DEFAULT_RUNS)
    out = (rd('template.html').replace('/*WORLD*/', rd('world.js') + '\n' + rd('osm.js') + '\n' + rd('npc.js')).replace('/*SIM*/', rd('sim.js')).replace('/*RENDER*/', rd('render.js'))
           .replace('/*ENV*/', rd('env.js')).replace('/*PLANNER*/', rd('planner.js'))
           .replace('/*AVDATA*/', ('const AV_DATA = ' + json.dumps(av, separators=(',', ':')) + ';' if av else '') + (('const NPC_DATA = ' + open(NPC_RUN).read() + ';') if os.path.exists(NPC_RUN) else ''))
           .replace('/*UI*/', rd('ui.js')))
    return out, n


def favicon():
    p = os.path.join(HERE, '..', 'docs', 'logo', 'lotus-icon.svg')
    if not os.path.exists(p): return ''
    import base64
    return '<link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,' + base64.b64encode(open(p, 'rb').read()).decode() + '">'


def standalone(fragment, extra_head=''):
    """wrap the page fragment in a full HTML document (for a plain web server or a local file)"""
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><title>LOTUS Simulator</title>' + favicon() + '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
            '<style>:root{color-scheme:light dark}body{margin:0;font:14px system-ui}[hidden]{display:none!important}</style>' + extra_head + '</head><body>' + fragment + '</body></html>')


if __name__ == '__main__':
    runs = os.environ.get('AVRUN', '').split(',') if os.environ.get('AVRUN') else DEFAULT_RUNS
    out, n = build_page(runs)
    open(os.path.join(HERE, 'lotus.html'), 'w', encoding="utf-8").write(out)
    if len(sys.argv) > 1:   # local test copy: wrap in a skeleton and block the font CDN
        open(sys.argv[1], 'w', encoding="utf-8").write(standalone(out.replace('https://fonts.googleapis.com/css2', 'http://invalid.localhost/css2')))
    print(len(out) // 1024, 'KB', f'{n} learned drivers' if n else 'teacher and baseline only')
