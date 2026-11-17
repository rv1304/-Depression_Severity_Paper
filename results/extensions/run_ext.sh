#!/usr/bin/env bash
# Revision experiments. Chain A: ablation -> seeds. Chain B: symptoms -> leakage -> (wait for ablation) -> noise.
cd "$(dirname "$0")/../.."
PY=.venv/bin/python; LOG=results/extensions/logs
run() { caffeinate -i "$PY" -u -W ignore -m src.extensions "$@"; }
( run --parts ablation --n-jobs 4 > $LOG/ablation.log 2>&1; echo "ablation exit $?" >> $LOG/all.log
  run --parts seeds --n-jobs 3 > $LOG/seeds.log 2>&1; echo "seeds exit $?" >> $LOG/all.log ) &
( run --parts symptoms > $LOG/symptoms.log 2>&1; echo "symptoms exit $?" >> $LOG/all.log
  run --parts leakage --n-jobs 3 > $LOG/leakage.log 2>&1; echo "leakage exit $?" >> $LOG/all.log
  while [ ! -f results/extensions/ablation.json ]; do sleep 20; done
  run --parts noise --n-jobs 3 > $LOG/noise.log 2>&1; echo "noise exit $?" >> $LOG/all.log ) &
wait
run --parts report > $LOG/report.log 2>&1; echo "report exit $?" >> $LOG/all.log
echo "done $(date)" >> $LOG/all.log
