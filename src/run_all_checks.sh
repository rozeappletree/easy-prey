#!/usr/bin/env bash
# Regenerate every file in outputs/. Pure CPU (plus one nvidia-smi query) -- safe to re-run at
# any point, and it consumes none of the GPU budget.
#
#   bash src/run_all_checks.sh
#
# Each output file gets a header recording the command and the time, so a stale artifact is
# obvious. These files are the evidence behind docs/WORKLOG.md.

set -euo pipefail

PY=/opt/conda/envs/talktuner-gpu/bin/python
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/outputs"
mkdir -p "$OUT"
cd "$ROOT"

run() {
    local name="$1"; shift
    {
        echo "# $name"
        echo "# command: $*"
        echo "# generated: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
        echo "# regenerate: bash src/run_all_checks.sh"
        echo
    } > "$OUT/$name"
    "$@" >> "$OUT/$name" 2>&1
    echo "  outputs/$name"
}

echo "Regenerating outputs/ ..."
run env_report.txt          "$PY" src/env_report.py
run toy_run_stats.txt       "$PY" src/toy_run_stats.py
run build_items.txt         "$PY" src/build_items.py
run prefix_qc_selfcheck.txt "$PY" src/prefix_qc.py
run test_scoring.txt        "$PY" src/test_scoring.py
run persona_check.txt       "$PY" src/check_persona_templates.py data/persona_templates_for_T3_human_augmented.json
echo "Done."
