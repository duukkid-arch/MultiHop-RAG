#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/MultiHop-RAG
R=results/v6_focus_750_musa
if pgrep -af '[p]ython .*mhrag_v6_musa/run.py' | grep -q .; then
  echo 'There is already a v6 runner; refusing to start another one.';exit 1
fi
python mhrag_gpu_chat_musa/tools/verify_checkpoint.py --root "$R"
echo 'Check that any BAD trailing checkpoint was repaired before continuing.'
if [ "${1:-}" != '--yes' ]; then
  echo 'To resume after reviewing output: bash mhrag_gpu_chat_musa/tools/resume_v6.sh --yes';exit 0
fi
nohup python -u mhrag_v6_musa/run.py --root "$R" >> "$R/launcher.log" 2>&1 </dev/null &
echo "Resumed v6 PID=$!"
