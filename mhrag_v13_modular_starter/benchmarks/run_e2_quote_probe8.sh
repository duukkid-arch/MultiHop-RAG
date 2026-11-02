#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
ACT="${1:-preflight}"
OUT="logs/v13_e2_quote_probe8_01"
LOG="logs/v13_e2_quote_probe8_01.stdout.log"
SESSION="v13_e2_quote_probe8"
PYTHON="$(command -v python)"
if [[ "$ACT" == report ]];then
  "$PYTHON" benchmarks/run_e2_quote_probe8.py --report;exit 0
fi
[[ "$ACT" =~ ^(preflight|start|resume)$ ]] || { echo 'Usage: bash benchmarks/run_e2_quote_probe8.sh [preflight|start|report|resume]';exit 2; }
"$PYTHON" -m unittest discover -s tests -p test_e2_quote_probe.py -q
"$PYTHON" benchmarks/run_e2_quote_probe8.py
for p in 7866 8002;do
  (echo > /dev/tcp/127.0.0.1/"$p") 2>/dev/null || { echo "Service $p unavailable" >&2;exit 2; }
done
if [[ "$ACT" == preflight ]];then exit 0;fi
command -v screen >/dev/null || { echo 'screen unavailable' >&2;exit 2; }
if screen -ls 2>/dev/null | grep -Eq "[.]${SESSION}([[:space:]]|$)";then
  echo 'ERROR: experiment already active, do not restart';exit 2
fi
if [[ "$ACT" == start && -e "$OUT" ]];then echo 'ERROR: results already exist';exit 2;fi
if [[ "$ACT" == resume && ! -f "$OUT/manifest.json" ]];then echo 'ERROR: no previous manifest';exit 2;fi
RESUME=""; [[ "$ACT" == resume ]] && RESUME='--resume'
printf -v cmd 'cd %q; set -o pipefail; %q benchmarks/run_e2_quote_probe8.py --execute %s 2>&1 | tee -a %q; exit ${PIPESTATUS[0]}' "$ROOT" "$PYTHON" "$RESUME" "$LOG"
screen -dmS "$SESSION" bash -lc "$cmd"
sleep 1
if ! screen -ls 2>/dev/null | grep -Eq "[.]${SESSION}([[:space:]]|$)";then
  if [[ -f "$OUT/summary.json" ]];then echo 'Completed; run report';exit 0;fi
  echo 'Session ended; check log:' "$LOG";tail -n 30 "$LOG" || true;exit 2
fi
echo 'SCREEN_STARTED:' "$SESSION"
echo 'Progress: tail -n 12' "$LOG"
echo 'Report: bash benchmarks/run_e2_quote_probe8.sh report'
echo 'Resume only after screen exits and time_cap: bash benchmarks/run_e2_quote_probe8.sh resume'
