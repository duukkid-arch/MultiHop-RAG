#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
ACTION="${1:-preflight}"
PYTHON="$(command -v python || true)"
[[ -n "$PYTHON" ]] || { echo 'ERROR: python not found' >&2;exit 2; }
DB="../mhrag_gpu_chat_musa/data/knowledge.sqlite"
ALIASES="study_aliases.reviewed.json"
CASES="benchmarks/dev16_e2_repair_NO_GOLD.json"
OUT="logs/v13_e2_repair16_01"
LOG="logs/v13_e2_repair16_01.stdout.log"
SESSION="v13_e2_repair16"
mkdir -p logs
if [[ "$ACTION" == report ]];then
  [[ -f "$OUT/summary.json" ]] || { echo 'No summary yet; check progress:' >&2; echo "tail -n 15 $LOG";exit 2; }
  "$PYTHON" -m json.tool "$OUT/summary.json"
  if "$PYTHON" -c "import json; s=json.load(open('$OUT/summary.json'));exit(0 if s['state']=='complete' and s['total_runtime_errors']==0 else 1)";then
    if [[ ! -e "${OUT}_review" ]];then
      "$PYTHON" benchmarks/run_e2_repair16.py --out "$OUT" --db "$DB" --aliases "$ALIASES" --cases "$CASES" --export-review
    fi
  fi
  exit 0
fi
[[ "$ACTION" =~ ^(start|preflight|resume)$ ]] || { echo 'Usage: bash benchmarks/run_e2_repair16.sh [preflight|start|report|resume]' >&2;exit 2; }
[[ -f "$DB" && -f "$ALIASES" && -f "$CASES" ]] || { echo 'ERROR: missing DB/aliases/cases' >&2;exit 2; }
"$PYTHON" -W ignore::ResourceWarning -m unittest discover -s tests -q
"$PYTHON" benchmarks/run_e2_repair16.py --db "$DB" --aliases "$ALIASES" --cases "$CASES" --out "$OUT" > logs/v13_e2_repair16_preflight.json
"$PYTHON" - <<'PY'
import json,socket
j=json.load(open('logs/v13_e2_repair16_preflight.json',encoding='utf-8'))
assert j['dry_run'] and j['cases']==16 and j['jobs']==32 and j['arms']==['E2_legacy','E2_repaired'] and j['e3_is_posthoc']
for port in (7866,8002):
    with socket.create_connection(('127.0.0.1',port),timeout=5):pass
print('PREFLIGHT OK: 16 historical DEV questions × 2 E2 variants = 32 jobs; E3 posthoc, no additional Qwen calls')
PY
if [[ "$ACTION" == preflight ]];then exit 0;fi
command -v screen >/dev/null || { echo 'ERROR: screen missing' >&2;exit 2; }
if screen -ls 2>/dev/null | grep -Eq "[.]${SESSION}([[:space:]]|$)";then echo 'ERROR: repair experiment already active' >&2;exit 2;fi
if [[ "$ACTION" == start && -e "$OUT" ]];then echo 'ERROR: results exist, cannot restart over same data' >&2;exit 2;fi
if [[ "$ACTION" == resume && ! -f "$OUT/manifest.json" ]];then echo 'ERROR: missing frozen manifest' >&2;exit 2;fi
RESUME='';[[ "$ACTION" == resume ]] && RESUME='--resume'
printf -v cmd 'cd %q; set -o pipefail; %q benchmarks/run_e2_repair16.py --db %q --aliases %q --cases %q --out %q --max-hours 1 --execute %s 2>&1 | tee -a %q; exit ${PIPESTATUS[0]}' \
 "$ROOT" "$PYTHON" "$DB" "$ALIASES" "$CASES" "$OUT" "$RESUME" "$LOG"
screen -dmS "$SESSION" bash -lc "$cmd"
sleep 1
if ! screen -ls 2>/dev/null | grep -Eq "[.]${SESSION}([[:space:]]|$)";then
 if [[ -f "$OUT/summary.json" ]];then echo 'Completed quickly; run report';exit 0;fi
 echo 'ERROR: session exited. See:' "$LOG" >&2;tail -n 25 "$LOG" || true;exit 2
fi
echo 'SCREEN_STARTED:' "$SESSION"
echo 'Progress: tail -n 12' "$ROOT/$LOG"
echo 'Report: bash benchmarks/run_e2_repair16.sh report'
echo 'Resume only if stopped or time_cap: bash benchmarks/run_e2_repair16.sh resume'
