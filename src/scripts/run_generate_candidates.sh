#!/bin/bash
# ==============================================================================
# run_generate_candidates.sh
# Phase B1: AR-conditioned candidate generator runner
# ==============================================================================
# Usage:
#   bash src/scripts/run_generate_candidates.sh \
#     data/SSTK/10K_local/filtered_sstk_100.parquet \
#     data/SSTK/10K_local/feats_c2.jsonl \
#     data/SSTK/10K_local/feats_c3.jsonl \
#     data/SSTK/10K_local/candidates_ar.jsonl \
#     --max_images 0 --max_candidates_per_ar 240
#
# Positional args:
#   (1) INPUT_PARQUET
#   (2) FEATS_C2_JSONL
#   (3) FEATS_C3_JSONL
#   (4) OUTPUT_JSONL
#
# Local-only options (handled by this script):
#   --server_mode 0|1   (default: 0; 0 means local and venv activation)
#   --venv_path PATH    (default: /media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate)
#
# Other options are passed through to src/generate_candidates.py.
# ==============================================================================

set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: bash src/scripts/run_generate_candidates.sh <input_parquet> <feats_c2_jsonl> <feats_c3_jsonl> <output_jsonl> [options...]"
  exit 1
fi

INPUT_PARQUET="$1"
FEATS_C2_JSONL="$2"
FEATS_C3_JSONL="$3"
OUTPUT_JSONL="$4"
shift 4

SERVER_MODE=0
VENV_PATH="/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate"
PASS_ARGS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --server_mode)
      SERVER_MODE="$2"
      shift 2
      ;;
    --venv_path)
      VENV_PATH="$2"
      shift 2
      ;;
    *)
      PASS_ARGS+=("$1")
      shift
      ;;
  esac
done

if [ "$SERVER_MODE" -ne 1 ]; then
  if [ -f "$VENV_PATH" ]; then
    # shellcheck disable=SC1090
    source "$VENV_PATH"
  else
    echo "Warning: venv not found at $VENV_PATH, using system python."
  fi
fi

SRC_DIR=$(cd "$(dirname "$0")/.." && pwd)
PY_SCRIPT="${SRC_DIR}/generate_candidates.py"

echo "=============================================="
echo " Candidate Generator Runner (Phase B1)"
echo "=============================================="
echo "  Input Parquet : $INPUT_PARQUET"
echo "  Feats C2      : $FEATS_C2_JSONL"
echo "  Feats C3      : $FEATS_C3_JSONL"
echo "  Output JSONL  : $OUTPUT_JSONL"
echo "  Server Mode   : $SERVER_MODE"
if [ "${#PASS_ARGS[@]}" -gt 0 ]; then
  echo "  Extra Args    : ${PASS_ARGS[*]}"
fi
echo "=============================================="

python3 "$PY_SCRIPT" \
  --input_parquet "$INPUT_PARQUET" \
  --feats_c2_jsonl "$FEATS_C2_JSONL" \
  --feats_c3_jsonl "$FEATS_C3_JSONL" \
  --output_jsonl "$OUTPUT_JSONL" \
  "${PASS_ARGS[@]}"

echo "=============================================="
echo " Done."
echo "=============================================="
