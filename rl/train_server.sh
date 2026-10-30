#!/usr/bin/env bash
# Train the self-driving car on a server and keep it running after you log out.
#   bash train_server.sh            # 5 million steps, one worker per CPU core
#   STEPS=10000000 bash train_server.sh
set -e
PY=${PY:-$(command -v python3 || command -v python)}
cd "$(dirname "$0")"
command -v node >/dev/null || { echo "Need Node.js 18+ (e.g. conda install -c conda-forge nodejs, or nvm install 20)"; exit 1; }
STEPS=${STEPS:-5000000}
CORES=$($PY -c "import os; print(os.cpu_count())")
WORKERS=${WORKERS:-$(( CORES > 2 ? CORES - 2 : 1 ))}
OUT=${OUT:-runs/teacher}
mkdir -p "$OUT"
echo "steps=$STEPS workers=$WORKERS (cores=$CORES) out=$OUT"
nohup "$PY" train.py --steps "$STEPS" --workers "$WORKERS" --envs 4 \
  --milestones 120000,400000,1000000,2000000 --out "$OUT" > "$OUT/train.log" 2>&1 &
echo "PID $!  ·  follow with: tail -f $OUT/train.log"
