import json
import socket
import urllib.request
from pathlib import Path
from v5_agent import Agent
from v5_train import load_config, fingerprint
from v5_eval import ROOT
from v5 import TrainingMonitor, compatible_sources


def test_monitor_serves_hcm_and_latest_prior_without_reloading_page(tmp_path, capsys):
    cfg = load_config(ROOT / 'configs/v5_smoke.yaml')
    agent = Agent(cfg, 'cpu')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    out = tmp_path / 'custom_run_name'
    with TrainingMonitor(out, port, '127.0.0.1', generated=False) as monitor:
        assert monitor.server is not None
        agent.export(out / 'policy.json')
        (out / 'status.json').write_text(json.dumps({'phase':'train','steps':0,'total':48,'t':123}))
        (out / 'metrics.jsonl').write_text(json.dumps({'steps':8,'ok':0})+'\n')
        url = f'http://127.0.0.1:{port}'
        def fetch(path):
            with urllib.request.urlopen(url + path, timeout=10) as response: return response.read()
        html = fetch('/sim?driver=latest&map=hcm')
        assert b'tdmpc_v5' in html and b'RecurrentPolicy' in html
        initial_signature = fetch('/api/ckpt')
        assert json.loads(fetch('/api/policy'))['steps'] == 0
        agent.steps = 24; agent.export(out / 'policy.json')
        assert json.loads(fetch('/api/policy'))['steps'] == 24
        assert fetch('/api/ckpt') == initial_signature  # live swap, no forced page reload
        index = json.loads(fetch('/hcm/index.json')); key = next(iter(index['tiles']))
        assert 'J' in json.loads(fetch('/hcm/t/' + key + '.json'))
        status = json.loads(fetch('/api/status'))
        assert status['wm']['run'] == 'custom_run_name' and status['wm']['log'][0]['steps'] == 8
        assert status['wm']['status']['total'] == 48
        print('monitor log test')
        assert 'monitor log test' in '\n'.join(json.loads(fetch('/api/log')))
    output = capsys.readouterr().out
    assert f'{url}/sim?driver=latest&map=hcm' in output
    assert monitor.server.fileno() == -1


def test_monitor_migration_is_pinned_not_a_training_source_exemption():
    current = fingerprint()
    migrations = json.loads((ROOT / 'monitor/v5_source_compat.json').read_text())
    old, new = migrations['rl/v5_train.py'][0]
    assert current['rl/v5_train.py'] == new
    previous = dict(current)
    for path, pairs in migrations.items(): previous[path] = pairs[0][0]
    assert compatible_sources(previous, current)
    assert not compatible_sources(previous, dict(current, **{'rl/v5_train.py': 'unreviewed-edit'}))
    assert not compatible_sources(previous, dict(current, **{'rl/v5_agent.py': 'different-controller'}))
