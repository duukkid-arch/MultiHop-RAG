#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
ACTION="${1:-preflight}"
PYTHON="$(command -v python || true)"
[[ -n "$PYTHON" ]] || { echo 'ERROR: python missing' >&2;exit 2; }
DB="../mhrag_gpu_chat_musa/data/knowledge.sqlite"
ALIASES="study_aliases.reviewed.json"
CASES="benchmarks/dev91_questions_NO_GOLD.json"
OUT="logs/v13_ablation_dev91_01"
LOG="logs/v13_ablation_dev91_01.stdout.log"
SESSION="v13_ablation_dev91"
mkdir -p logs
if [[ "$ACTION" == 'report' ]];then
  [[ -f "$OUT/summary.json" ]] || { echo "No summary yet: $OUT" >&2;exit 2; }
  "$PYTHON" -m json.tool "$OUT/summary.json"
  if [[ -f "$OUT/manifest.json" && ! -e logs/v13_ablation_dev91_01_review ]]; then
    "$PYTHON" benchmarks/v13_ablations_review.py --run "$OUT" --out logs/v13_ablation_dev91_01_review || true
  fi
  exit 0
fi
[[ "$ACTION" =~ ^(preflight|start|resume)$ ]] || {
  echo 'Usage: bash benchmarks/run_ablations_dev91.sh [preflight|start|resume|report]' >&2;exit 2; }
[[ -f "$DB" && -f "$ALIASES" && -f "$CASES" ]] || { echo 'Missing DB/aliases/DEV data' >&2;exit 2; }
"$PYTHON" -W ignore::ResourceWarning -m unittest discover -s tests -q
"$PYTHON" benchmarks/v13_ablations_runner.py --db "$DB" --aliases "$ALIASES" --cases "$CASES" --out "$OUT" > logs/v13_ablation_dev91_preflight.json
"$PYTHON" - <<'PY'
import json,socket
j=json.load(open('logs/v13_ablation_dev91_preflight.json',encoding='utf-8'))
assert j['dry_run'] and j['cases']==91 and j['jobs']==364 and set(j['arms'])=={'F','E1','E2','E3'}
print('PREFLIGHT OK: 91 historical DEV cases × 4 arms = 364 jobs, no gold')
for port in (7866,8002):
    with socket.create_connection(('127.0.0.1',port),timeout=5):pass
print('SERVICES: 7866/8002 TCP reachable (not clinical validity)')
PY
if [[ "$ACTION" == preflight ]];then exit 0;fi
command -v screen >/dev/null || { echo 'screen not installed' >&2;exit 2; }
if screen -ls 2>/dev/null | grep -Eq "[.]${SESSION}([[:space:]]|$)";then
 echo "ERROR: $SESSION already active" >&2;exit 2
fi
if [[ "$ACTION" == start && -e "$OUT" ]];then
 echo "ERROR: $OUT exists; use resume if interrupted" >&2;exit 2
fi
if [[ "$ACTION" == resume && ! -f "$OUT/manifest.json" ]];then
 echo 'ERROR: missing previous manifest' >&2;exit 2
fi
RESUME=""
[[ "$ACTION" == resume ]] && RESUME="--resume"
printf -v cmd 'cd %q; set -o pipefail; %q benchmarks/v13_ablations_runner.py --db %q --aliases %q --cases %q --out %q --max-hours 2 --execute %s 2>&1 | tee -a %q; exit ${PIPESTATUS[0]}' \
  "$ROOT" "$PYTHON" "$DB" "$ALIASES" "$CASES" "$OUT" "$RESUME" "$LOG"
screen -dmS "$SESSION" bash -lc "$cmd"
sleep 1
if ! screen -ls 2>/dev/null | grep -Eq "[.]${SESSION}([[:space:]]|$)";then
 if [[ -f "$OUT/summary.json" ]];then echo 'Completed quickly. Run report';exit 0;fi
 echo 'Screen exited unexpectedly. Inspect log' >&2;tail -n 25 "$LOG" 2>/dev/null || true;exit 2
fi
echo "SCREEN_STARTED: $SESSION"
echo "Progress: tail -n 20 $ROOT/$LOG"
echo 'Report: bash benchmarks/run_ablations_dev91.sh report'
echo 'If time cap reached: bash benchmarks/run_ablations_dev91.sh resume'
echo 'Do not modify source/DB or start any other GPU evaluation before this run completes.'
