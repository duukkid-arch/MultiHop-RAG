#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
DATA="${1:-/root/autodl-tmp/medhop_benchmark_private/medhop_original}"
mkdir -p results
python -m pip install -r requirements.txt
for MODE in single agentic; do
  echo "===== Full validation: ${MODE} ====="
  python medhop_pipeline.py --data "$DATA" --split validation --mode "$MODE" --out "results/${MODE}_validation.jsonl" 2>&1 | tee "results/${MODE}_validation.log"
done
python - <<'PY'
import json
from pathlib import Path
p=Path('results')
for mode in ['single','agentic']:
    j=json.loads((p/f'{mode}_validation_metrics.json').read_text())
    print(mode, json.dumps(j, ensure_ascii=False))
PY
echo 'Finished: results/*_validation_metrics.json, results/*_validation.jsonl'
