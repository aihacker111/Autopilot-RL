import json, os, sys, glob, re
HERE = os.path.dirname(os.path.abspath(__file__))
rd = lambda p: open(os.path.join(HERE, p)).read()
# Learned drivers to show on the page: AVRUN is a comma-separated list of run folders
# (default: the distillation folder, then the PPO fine-tuning folder). Each contributes
# student_r*.json (distilled students), step_*.json and final.json (fine-tuned checkpoints).
RUNS = (os.environ.get('AVRUN') or ','.join(os.path.join(HERE, '..', 'rl', 'runs', d) for d in ('teacher_v2', 'finetune_v2'))).split(',')
num = lambda p, pat: int(re.findall(pat, p)[0])
ck, log = [], None
for run in RUNS:
    if not os.path.isdir(run): continue
    files = sorted(glob.glob(os.path.join(run, 'student_r*.json')), key=lambda p: num(p, r'student_r(\d+)'))[-1:]   # the last student only
    files += sorted([p for p in glob.glob(os.path.join(run, 'step_*.json')) if num(p, r'step_(\d+)') > 0], key=lambda p: num(p, r'step_(\d+)'))
    if os.path.exists(os.path.join(run, 'final.json')): files.append(os.path.join(run, 'final.json'))
    for f in files:
        spec = json.load(open(f))
        if spec.get('type') != 'set' and len(spec.get('obsMean', '')) < 1000: continue   # skip old single-junction checkpoints
        base = os.path.basename(f)
        spec['short'] = 'Học sinh' if base.startswith('student') else ('PPO ' + (f"{num(base, r'step_(\d+)') // 1000}k" if base.startswith('step') else 'cuối'))
        spec['desc'] = spec.get('desc') or ('Mạng nơ-ron chưng cất từ người thầy nhìn trước (DAgger).' if base.startswith('student') else f"Học sinh được tinh chỉnh thêm bằng PPO, {spec.get('steps', 0) // 1000}k bước.")
        ck.append(spec)
    if os.path.exists(os.path.join(run, 'log.json')):
        L = json.load(open(os.path.join(run, 'log.json')))
        if L and 'ok' in L[0]: log = [{k: r[k] for k in ('steps', 'ok', 'crash', 'red')} for r in L]
av = {'ckpts': ck[-4:], 'log': log or []} if ck else None

out = (rd('template.html').replace('/*SIM*/', rd('sim.js')).replace('/*RENDER*/', rd('render.js'))
       .replace('/*ENV*/', rd('env.js')).replace('/*PLANNER*/', rd('planner.js'))
       .replace('/*AVDATA*/', 'const AV_DATA = ' + json.dumps(av, separators=(',', ':')) + ';' if av else '')
       .replace('/*UI*/', rd('ui.js')))
open(os.path.join(HERE, 'lotus.html'), 'w').write(out)
if len(sys.argv) > 1:   # local test copy: wrap in a skeleton and block the font CDN
    test = out.replace('https://fonts.googleapis.com/css2', 'http://invalid.localhost/css2')
    open(sys.argv[1], 'w').write('<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><style>:root{color-scheme:light}body{margin:0;font:14px system-ui}[hidden]{display:none!important}</style></head><body>' + test + '</body></html>')
print(len(out) // 1024, 'KB', f'{len(ck)} learned drivers' if ck else 'teacher and baseline only')
