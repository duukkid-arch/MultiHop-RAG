#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv_musa_llm"
echo "Creating isolated LLM Python package overlay at $VENV"
python -m venv --system-site-packages "$VENV"
# Pin Transformers separately from the working BGE+CE main process.
# --no-deps is intentional to avoid replacing torch / torch_musa / numpy in the base MUSA stack.
"$VENV/bin/python" -m pip install --no-deps \
  'transformers==4.44.2' \
  'tokenizers==0.19.1' \
  'huggingface_hub==0.24.7' \
  'safetensors==0.4.5'
echo "Isolated runtime ready. Existing main conda environment was not modified."
"$VENV/bin/python" -m mhrag_local_llm_musa.preflight
