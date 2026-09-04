#!/usr/bin/env bash
# Multi-model replication of Step 6 (C1) -- see docs/WORKLOG.md entry 35's priority list.
# Runs the SAME C1 experiment on 3 subject models, sequentially (one at a time, so each gets
# the full GPU and there's no memory contention). Each run writes its own output prefix so
# nothing overwrites the original Llama-2-13b result already analysed in nb/03.
#
#   bash src/run_c1_multimodel.sh
#
# Progress: tail -f outputs/c1_multimodel.log
# Per-model raw output: outputs/c1_<model>.log
# Per-model results: data/c1_<model>_summary.json (+ _behavioural.json, _ceiling.json)
# Per-model figure: outputs/figure1_c1_<model>_paired_scatter.png

set -uo pipefail  # no -e: one model failing should not stop the others

PY=/opt/conda/envs/talktuner-gpu/bin/python
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT/outputs/c1_multimodel.log"
cd "$ROOT"

declare -A MODELS=(
    [llama13b]="NousResearch/Llama-2-13b-chat-hf"
    [qwen7b]="Qwen/Qwen2.5-7B-Instruct"
    [qwen14b]="Qwen/Qwen2.5-14B-Instruct"
)
ORDER=(llama13b qwen7b qwen14b)   # llama13b first: already have a result to sanity-check against

{
    echo "=================================================================="
    echo "Multi-model C1 replication started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "Models, in order: ${ORDER[*]}"
    echo "=================================================================="
} > "$LOG"

for key in "${ORDER[@]}"; do
    model="${MODELS[$key]}"
    {
        echo
        echo "### [$key] $model -- starting $(date -u '+%H:%M:%S UTC')"
    } >> "$LOG"

    "$PY" -u src/c1_total_effect.py --model "$model" --out-prefix "c1_$key" \
        > "outputs/c1_${key}.log" 2>&1
    status=$?

    if [ $status -eq 0 ]; then
        d_z=$($PY -c "import json; print('%.3f' % json.load(open('data/c1_${key}_summary.json'))['behavioural']['d_z'])" 2>/dev/null)
        echo "### [$key] done $(date -u '+%H:%M:%S UTC') -- behavioural d_z = ${d_z:-?}" >> "$LOG"
    else
        echo "### [$key] FAILED (exit $status) -- see outputs/c1_${key}.log" >> "$LOG"
    fi
done

echo "" >> "$LOG"
echo "ALL MODELS DONE: $(date -u '+%Y-%m-%d %H:%M:%S UTC')" >> "$LOG"
echo "Summary:" >> "$LOG"
for key in "${ORDER[@]}"; do
    "$PY" -c "
import json
try:
    s = json.load(open('data/c1_${key}_summary.json'))
    b, c = s['behavioural'], s['ceiling']
    print('  ${key}: behavioural d_z=%+.3f [%.3f,%.3f]  ceiling d_z=%+.3f [%.3f,%.3f]' %
          (b['d_z'], *b['d_z_ci'], c['d_z'], *c['d_z_ci']))
except Exception as e:
    print('  ${key}: no result (%s)' % e)
" >> "$LOG" 2>&1
done
echo "Full log: $LOG" >> "$LOG"
