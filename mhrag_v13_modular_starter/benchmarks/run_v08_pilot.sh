#!/usr/bin/env bash
# Safe, one-entry DEV-only pilot under screen; no daemon/service restart.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
ACTION="${1:-start}"
DB="../mhrag_gpu_chat_musa/data/knowledge.sqlite"
ALIAS="study_aliases.reviewed.json"
OUT="logs/v13_pilot_16_v08_01"
LOG="logs/v13_pilot_16_v08_01.stdout.log"
SESSION="v13_pilot_v08"
PYTHON="$(command -v python || true)"
[[ -n "$PYTHON" ]] || { echo 'ERROR: python 不存在，请先激活 agentic4090_env' >&2; exit 2; }
mkdir -p logs
if [[ "$ACTION" == 'report' ]]; then
  [[ -f "$OUT/summary.json" ]] || { echo "尚未生成汇总: $OUT/summary.json"; exit 2; }
  "$PYTHON" -m json.tool "$OUT/summary.json"
  "$PYTHON" - <<'PY'
import json
from pathlib import Path
p=Path('logs/v13_pilot_16_v08_01/summary.json')
d=json.loads(p.read_text(encoding='utf-8'))
g=d.get('arms',{}).get('G',{})
print('\n===== 人工验收门槛 =====')
print('配对题数:',d.get('paired_cases_completed'))
print('程序错误:',d.get('transport_or_runtime_errors'))
print('G 有论断的题数:',g.get('cases_with_accepted_claim'))
print('G 弃答数(原始计数):',g.get('abstention_count'))
print('医学准确率:',d.get('medical_accuracy'))
print('Gate:',d.get('gate'))
print('注意：自动引用/数值检查不是医学事实准确率。')
PY
  if [[ ! -f "$OUT/manifest.json" ]]; then exit 2; fi
  if [[ ! -e "logs/v13_pilot_16_v08_review" ]]; then
    "$PYTHON" benchmarks/v13_pilot_review.py --run "$OUT" --out logs/v13_pilot_16_v08_review
  else
    echo '人工审核文件夹已存在，保留旧版，不覆盖。'
  fi
  exit 0
fi
[[ "$ACTION" == 'start' || "$ACTION" == 'resume' || "$ACTION" == 'preflight' ]] || {
  echo '用法: bash benchmarks/run_v08_pilot.sh [preflight|start|resume|report]' >&2;exit 2;
}
[[ -f "$DB" && -f "$ALIAS" ]] || {
  echo "ERROR: SQLite/别名文件缺失，确认当前项目为: $ROOT" >&2;exit 2;
}
"$PYTHON" -W ignore::ResourceWarning -m unittest discover -s tests -q
"$PYTHON" benchmarks/v13_pilot_runner.py --db "$DB" --aliases "$ALIAS" --out "$OUT" > logs/v13_pilot_v08_preflight.json
"$PYTHON" - <<'PY'
import json,socket
p='logs/v13_pilot_v08_preflight.json'
d=json.load(open(p,encoding='utf-8'))
if not(d.get('dry_run') and d.get('jobs')==32 and d.get('cases')==16):
    raise SystemExit('STOP: DEV pilot dry run incorrect')
print('PREFLIGHT: 16 DEV cases, 32 paired jobs; not 2417-job experiment')
for port in (7866,8002):
    try:
        with socket.create_connection(('127.0.0.1',port),timeout=3):pass
    except OSError as e:
        raise SystemExit(f'STOP: existing service 127.0.0.1:{port} unreachable: {e}')
print('SERVICES: 7866 / 8002 TCP reachable (does not validate answer quality)')
PY
if [[ "$ACTION" == 'preflight' ]];then echo '只做预检，不启动 GPU 实验。';exit 0;fi
command -v screen >/dev/null || {
  echo 'ERROR: 未发现 screen。请先运行 apt-get update && apt-get install -y screen，然后重试。' >&2;exit 2;
}
if screen -ls 2>/dev/null | grep -Eq "[.]${SESSION}([[:space:]]|$)";then
  echo "ERROR: $SESSION 已在运行，不要重复启动" >&2;exit 2
fi
if [[ "$ACTION" == 'start' && -e "$OUT/manifest.json" ]];then
  echo "ERROR: 目录 $OUT 已包含实验记录；用 resume 或新实验目录，不要覆盖。" >&2;exit 2
fi
if [[ "$ACTION" == 'resume' && ! -e "$OUT/manifest.json" ]];then
  echo "ERROR: $OUT 没有历史实验，不可 resume" >&2;exit 2
fi
printf -v runner 'cd %q; set -o pipefail; %q benchmarks/v13_pilot_runner.py --db %q --aliases %q --out %q --mode qwen --draft-strategy text-first --max-draft-calls 6 --max-hours 1.5 --execute %s 2>&1 | tee -a %q; exit ${PIPESTATUS[0]}' \
  "$ROOT" "$PYTHON" "$DB" "$ALIAS" "$OUT" "$([[ "$ACTION" == 'resume' ]] && echo '--resume' || echo '')" "$LOG"
screen -dmS "$SESSION" bash -lc "$runner"
sleep 1
if ! screen -ls 2>/dev/null | grep -Eq "[.]${SESSION}([[:space:]]|$)"; then
  echo "WARNING: screen 已退出（可能运行失败）。检查日志：$ROOT/$LOG" >&2
  tail -n 30 "$LOG" 2>/dev/null || true
  exit 2
fi
echo "SCREEN_STARTED: $SESSION"
echo "看进度: tail -n 20 $ROOT/$LOG"
echo "进入会话: screen -r $SESSION (离开按 Ctrl+A，再按 D)"
echo '跑完后: bash benchmarks/run_v08_pilot.sh report'
echo '不要启动第二个同名实验；不要修改运行中的源码/数据库。'
