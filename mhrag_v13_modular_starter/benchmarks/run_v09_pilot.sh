#!/usr/bin/env bash
# v0.9 independent DEV-only F/G paired experiment under screen, no modifications to v0.8.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
ACTION="${1:-start}"
DB="../mhrag_gpu_chat_musa/data/knowledge.sqlite"
ALIAS="study_aliases.reviewed.json"
OUT="logs/v13_pilot_16_v09_01"
LOG="logs/v13_pilot_16_v09_01.stdout.log"
SESSION="v13_pilot_v09"
PYTHON="$(command -v python || true)"
[[ -n "$PYTHON" ]] || { echo 'ERROR: python unavailable' >&2; exit 2; }
mkdir -p logs
if [[ "$ACTION" == 'report' ]]; then
  [[ -f "$OUT/summary.json" ]] || { echo "ERROR: no summary at $OUT/summary.json" >&2; exit 2; }
  "$PYTHON" -m json.tool "$OUT/summary.json"
  "$PYTHON" - <<'PY'
import json
p='logs/v13_pilot_16_v09_01/summary.json'
d=json.load(open(p,encoding='utf-8'))
g=d.get('arms',{}).get('G',{})
print('\n===== v0.9 TASK COVERAGE (NOT CLINICAL ACCURACY) =====')
print('State:',d.get('state'),'Paired:',d.get('paired_cases_completed'))
print('Runtime errors:',d.get('transport_or_runtime_errors'))
print('Task coverage statuses:',g.get('task_coverage_statuses'))
print('Full provisional:',g.get('provisional_answer_count'))
print('Partial:',g.get('partial_count'),'Abstained:',g.get('abstention_count'))
print('Medical accuracy:',d.get('medical_accuracy'))
PY
  if [[ -f "$OUT/manifest.json" && ! -e logs/v13_pilot_16_v09_review ]]; then
    "$PYTHON" benchmarks/v13_pilot_review.py --run "$OUT" --out logs/v13_pilot_16_v09_review
  fi
  exit 0
fi
[[ "$ACTION" == 'start' || "$ACTION" == 'resume' || "$ACTION" == 'preflight' ]] || {
  echo 'Usage: bash benchmarks/run_v09_pilot.sh [preflight|start|resume|report]' >&2; exit 2;
}
[[ -f "$DB" && -f "$ALIAS" ]] || { echo "ERROR: Missing SQLite/aliases under $ROOT" >&2; exit 2; }
"$PYTHON" -W ignore::ResourceWarning -m unittest discover -s tests -q
"$PYTHON" benchmarks/v13_pilot_runner_v09.py --db "$DB" --aliases "$ALIAS" --out "$OUT" > logs/v13_pilot_v09_preflight.json
"$PYTHON" - <<'PY'
import json,socket
p=json.load(open('logs/v13_pilot_v09_preflight.json',encoding='utf-8'))
assert p.get('dry_run') and p.get('jobs')==32 and p.get('cases')==16, 'Wrong DEV pilot plan'
print('PREFLIGHT OK: 16 DEV cases / 32 paired calls; not 2417-call formal test')
for port in (7866,8002):
  with socket.create_connection(('127.0.0.1',port),timeout=4):pass
print('EXISTING SERVICES: 7866 and 8002 TCP reachable (not a clinical validation)')
PY
if [[ "$ACTION" == 'preflight' ]];then exit 0;fi
command -v screen >/dev/null || { echo 'ERROR: screen missing' >&2; exit 2; }
if screen -ls 2>/dev/null | grep -Eq "[.]${SESSION}([[:space:]]|$)";then
  echo "ERROR: $SESSION session already active. Do not launch twice." >&2;exit 2
fi
if [[ "$ACTION" == 'start' && -e "$OUT/manifest.json" ]];then
  echo "ERROR: old output at $OUT. Use resume or new directory." >&2;exit 2
fi
if [[ "$ACTION" == 'resume' && ! -e "$OUT/manifest.json" ]];then
  echo "ERROR: No previous manifest under $OUT" >&2;exit 2
fi
printf -v cmd 'cd %q; set -o pipefail; %q benchmarks/v13_pilot_runner_v09.py --db %q --aliases %q --out %q --mode qwen --draft-strategy text-first --max-draft-calls 6 --max-hours 1.5 --execute %s 2>&1 | tee -a %q; exit ${PIPESTATUS[0]}' \
  "$ROOT" "$PYTHON" "$DB" "$ALIAS" "$OUT" "$([[ "$ACTION" == 'resume' ]] && echo '--resume' || echo '')" "$LOG"
screen -dmS "$SESSION" bash -lc "$cmd"
sleep 1
if ! screen -ls 2>/dev/null | grep -Eq "[.]${SESSION}([[:space:]]|$)";then
  if [[ -f "$OUT/summary.json" ]];then echo 'COMPLETED QUICKLY: run report';exit 0;fi
  echo 'WARNING: Screen session exited; inspect log.' >&2
  tail -n 25 "$LOG" 2>/dev/null || true;exit 2
fi
echo "SCREEN_STARTED: $SESSION"
echo "Progress: tail -n 20 $ROOT/$LOG"
echo "Summary: bash benchmarks/run_v09_pilot.sh report"
echo "Resume if interrupted: bash benchmarks/run_v09_pilot.sh resume"
echo 'Do not modify code/SQLite or launch another evaluation while this runs.'
