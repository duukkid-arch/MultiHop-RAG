#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/agentic_4090_v12_4/MultiHop-RAG-main
source /root/autodl-tmp/agentic4090_env/bin/activate
SCRIPT=${SCRIPT:-/root/autodl-tmp/mhrag_release_night_kit/benchmarks/night_v124.py}
BANK=${BANK:-/root/autodl-tmp/mhrag_release_night_kit/benchmarks/v121_frozen_cases.json}
exec python -u "$SCRIPT" --cases "$BANK" --db "$PWD/mhrag_gpu_chat_musa/data/knowledge.sqlite" \
  --url http://127.0.0.1:7866 --pilot --arms rag,agentic \
  --out results/v124_pilot --run-label v12.4_pilot --hours 1 --timeout 240
