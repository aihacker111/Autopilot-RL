#!/usr/bin/env bash
# Phase 2 (v2) on a server: look-ahead teacher -> DAgger distillation -> PPO fine-tuning -> evaluation.
#   bash run_phase2.sh                    # defaults below; override with env vars, e.g. ROUNDS=5 DECISIONS=40000 bash run_phase2.sh
set -e
cd "$(dirname "$0")"
PY=${PY:-$(command -v python3 || command -v python)}
CORES=$($PY -c "import os; print(os.cpu_count())")
WORKERS=${WORKERS:-$(( CORES > 2 ? CORES - 2 : 1 ))}
ROUNDS=${ROUNDS:-4}; DECISIONS=${DECISIONS:-25000}; FT_STEPS=${FT_STEPS:-3000000}; EVAL_N=${EVAL_N:-240}
mkdir -p runs/eval
echo "== 0. baseline and teacher on the fixed suite (workers=$WORKERS)"
node eval.js --driver idm --n $EVAL_N --out runs/eval/idm.json
for s in $(seq 0 $((WORKERS-1))); do node eval.js --driver planner --n $EVAL_N --shard $s --shards $WORKERS --out runs/eval/planner_s$s.json > /dev/null & done; wait
echo "== 1. DAgger distillation"
$PY distill.py --rounds $ROUNDS --decisions $DECISIONS --workers $WORKERS --eval-n $EVAL_N --out runs/teacher_v2
LAST=$(ls runs/teacher_v2/student_r*.pt | sort -V | tail -1)
echo "== 2. PPO fine-tuning from $LAST"
$PY finetune.py --init "$LAST" --steps $FT_STEPS --workers $WORKERS --out runs/finetune_v2
echo "== 3. evaluate the fine-tuned policy"
node eval.js --driver runs/finetune_v2/final.json --n $EVAL_N --out runs/eval/finetune.json
echo "== done. Build the page: cd ../web && python build.py"
