"""Train the corrected LOTUS v6 controller (separate from existing v5 runs)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'rl'))
from v6.train import main

if __name__ == '__main__': main()
