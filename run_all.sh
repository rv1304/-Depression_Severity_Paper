#!/usr/bin/env bash
# Full benchmark. caffeinate keeps macOS awake; logs go to results/logs/.
#   NB first (fast), then RF and LR in parallel on 4 cores each, then the
#   duplicate-leakage sensitivity run, then merge everything into tables/figures/LaTeX.
set -u
cd "$(dirname "$0")"
PY=.venv/bin/python
LOG=results/logs
mkdir -p "$LOG"
run() { caffeinate -i "$PY" -u -W ignore -m src.funnel "$@"; }

echo "start $(date)"                                          | tee -a "$LOG/all.log"
run --models nb                         > "$LOG/nb.log" 2>&1; echo "nb exit $? $(date)" | tee -a "$LOG/all.log"
run --models rf --n-jobs 4              > "$LOG/rf.log" 2>&1 &  RF=$!
run --models lr --n-jobs 4              > "$LOG/lr.log" 2>&1 &  LR=$!
wait $RF; echo "rf exit $? $(date)" | tee -a "$LOG/all.log"
wait $LR; echo "lr exit $? $(date)" | tee -a "$LOG/all.log"
run --models nb lr --keep-duplicates --out results_with_duplicates > "$LOG/sens.log" 2>&1
echo "sens exit $? $(date)" | tee -a "$LOG/all.log"
"$PY" -W ignore -m src.report > "$LOG/report.log" 2>&1; echo "report exit $? $(date)" | tee -a "$LOG/all.log"
echo "done $(date)" | tee -a "$LOG/all.log"
