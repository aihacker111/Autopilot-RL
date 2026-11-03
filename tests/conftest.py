import os
from pathlib import Path
import sys
import shutil
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'rl'))
os.environ['LOTUS_NODE'] = shutil.which('node') or '/opt/homebrew/bin/node'
