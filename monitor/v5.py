"""Observability for v5. Does not change controller, replay or simulator RNGs."""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def compatible_sources(saved, current):
    """Allow only a pinned, reviewed observability-only source migration.

    An arbitrary edit to training code still fails strict resume validation.
    The map contains exact old/new hashes, not a blanket file-name exemption.
    """
    if saved == current: return True
    if set(saved) != set(current): return False
    migrations = json.loads((HERE / 'v5_source_compat.json').read_text())
    return all(old == current[path] or [old, current[path]] in migrations.get(path, []) for path, old in saved.items())


class ConsoleLog:
    def __init__(self, terminal, file): self.terminal, self.file = terminal, file
    def write(self, text):
        self.terminal.write(text); self.file.write(text); return len(text)
    def flush(self): self.terminal.flush(); self.file.flush()
    def isatty(self): return self.terminal.isatty()


class TrainingMonitor:
    def __init__(self, out, port=8765, bind='0.0.0.0', generated=False):
        self.out, self.port, self.bind, self.generated = Path(out).resolve(), port, bind, generated
        self.server = self.url = None
        self.saved = self.file = None

    def __enter__(self):
        self.out.mkdir(parents=True, exist_ok=True)
        self.file = (self.out / 'train.log').open('a', buffering=1, encoding='utf-8')
        self.saved = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = ConsoleLog(sys.stdout, self.file), ConsoleLog(sys.stderr, self.file)
        if self.port:
            from server import start
            self.server, self.url = start(str(self.out.parent), self.port, self.bind, quiet=True, wm=str(self.out))
            if self.server:
                suffix = '/sim?driver=latest' + ('' if self.generated else '&map=hcm')
                local = f'http://127.0.0.1:{self.server.server_address[1]}'
                print(f'\nLOTUS v5 training monitor\n  Dashboard: {local}/\n  Simulator (latest policy prior): {local}{suffix}', flush=True)
                if self.url != local:
                    print(f'  Network dashboard: {self.url}/\n  Network simulator: {self.url}{suffix}', flush=True)
                print('  On a remote server: forward the monitor port with SSH, then open the localhost link.\n', flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.server:
            self.server.shutdown(); self.server.server_close()
        if self.saved: sys.stdout, sys.stderr = self.saved
        if self.file: self.file.close()

    def keep_open(self):
        if not self.server: return
        print(f'Training ended. Monitor remains at {self.url}/ · Ctrl+C to exit', flush=True)
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt: pass
