#!/bin/bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: bash src/scripts/run_build_teacher_proposals.sh <output_teacher_proposals_jsonl> <input_raw_jsonl1> [input_raw_jsonl2 ...]"
  exit 1
fi

OUT_JSONL="$1"
shift 1

VENV_PATH="/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate"
if [ -f "$VENV_PATH" ]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH"
fi

python3 src/scripts/build_teacher_proposals_jsonl.py \
  --output_jsonl "$OUT_JSONL" \
  --input_jsonl "$@"

