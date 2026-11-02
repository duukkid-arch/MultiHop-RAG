#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
ACTION="${1:-preflight}"
RUN=logs/v13_e2v2_dev24_01
LOG=logs/v13_e2v2_dev24_01.stdout.log
SESSION=v13_e2v2_dev24
PY="$(command -v python)"
case "$ACTION" in
  report)
    [[ -f "$RUN/summary.json" ]] || { echo "No summary yet: $RUN";exit 2; }
    "$PY" -m json.tool "$RUN/summary.json"
    exit 0;;
  preflight|start|resume);;
  *) echo 'Usage: bash benchmarks/run_dev24_e2v2.sh [preflight|start|resume|report]' >&2;exit 2;;
esac
for f in ../mhrag_gpu_chat_musa/data/knowledge.sqlite study_aliases.reviewed.json benchmarks/dev24_e2v2_NO_GOLD.json; do
 [[ -f "$f" ]] || { echo "Missing prerequisite: $f" >&2;exit 2; }
done
"$PY" -W ignore::ResourceWarning -m unittest discover -s tests -q
"$PY" benchmarks/run_dev24_e2v2.py > logs/v13_e2v2_dev24_preflight.json
"$PY" - <<'PY'
import json,socket
x=json.load(open('logs/v13_e2v2_dev24_preflight.json'))
assert x['cases']==24 and x['tasks']==96 and set(x['arms'])=={'F','E1','E2R','E3R'}
print('PREFLIGHT OK: 24 historical DEV cases × 4 independent arms = 96 tasks; NOT medical gold')
for port in (7866,8002):
    with socket.create_connection(('127.0.0.1',port),timeout=5):pass
print('7866/8002 reachable')
PY
if [[ "$ACTION" == preflight ]];then exit 0;fi
command -v screen >/dev/null || { echo 'screen not installed' >&2;exit 2; }
if screen -ls 2>/dev/null | grep -Eq "[.]${SESSION}([[:space:]]|$)";then
 echo "ERROR: $SESSION already active" >&2;exit 2
fi
if [[ "$ACTION" == start && -e "$RUN" ]];then
 echo "ERROR: $RUN exists; use resume if interrupted" >&2;exit 2
fi
if [[ "$ACTION" == resume && ! -f "$RUN/manifest.json" ]];then
 echo 'ERROR: no manifest to resume' >&2;exit 2
fi
RESUME=''
[[ "$ACTION" == resume ]] && RESUME='--resume'
printf -v cmd 'cd %q; set -o pipefail; %q benchmarks/run_dev24_e2v2.py --execute --max-hours 2 %s 2>&1 | tee -a %q; exit ${PIPESTATUS[0]}' "$ROOT" "$PY" "$RESUME" "$LOG"
screen -dmS "$SESSION" bash -lc "$cmd"
sleep 1
if ! screen -ls 2>/dev/null | grep -Eq "[.]${SESSION}([[:space:]]|$)";then
 if [[ -f "$RUN/summary.json" ]];then echo 'Already completed; use report';exit 0;fi
 echo 'Session ended quickly; inspect log:' >&2;tail -n 25 "$LOG" || true;exit 2
fi
echo "SCREEN_STARTED: $SESSION"
echo "Progress: tail -n 12 $LOG"
echo 'When complete: bash benchmarks/run_dev24_e2v2.sh report'
echo 'Resume only after screen exits and experiment stops before 96 tasks'
