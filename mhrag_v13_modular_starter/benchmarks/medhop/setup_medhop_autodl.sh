#!/usr/bin/env bash
# Reproducible MedHop preparation, no modifications to original 116-PDF database.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${MEDHOP_WORKDIR:-/root/autodl-tmp/medhop_benchmark_private}"
mkdir -p "$WORK"

if [[ -s "$WORK/prepared_v1/manifest.json" ]]; then
  echo "ALREADY_PREPARED (no overwrite): $WORK/prepared_v1"
  exit 0
fi

if [[ -s "$WORK/medhop_original/train.json" && -s "$WORK/medhop_original/dev.json" ]]; then
  echo "REUSING_EXISTING_SOURCE: $WORK/medhop_original"
else
  # Isolated dependency install: do not mutate GPU environment.
  VENV="$WORK/.medhop_convert_venv"
  if [[ ! -x "$VENV/bin/python" ]]; then
    python -m venv "$VENV"
  fi
  if ! "$VENV/bin/python" -c 'import pyarrow' 2>/dev/null; then
    echo "Installing pyarrow into isolated MedHop venv (not the GPU environment)..."
    "$VENV/bin/python" -m pip install --disable-pip-version-check 'pyarrow>=16,<24'
  fi
  "$VENV/bin/python" "$HERE/download_medhop_hf.py" --workdir "$WORK"
fi

python "$HERE/prepare_medhop.py" --data-dir "$WORK/medhop_original" --out "$WORK/prepared_v1"
echo "READY_NO_GPU: $WORK/prepared_v1"
echo "NEXT: F/A/C/E adapters still require scoped-evidence GPU interface verification."
