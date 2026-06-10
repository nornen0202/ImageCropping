#!/bin/bash
# ==============================================================================
# run_visualize_vlm_teacher.sh
# Stage-10 VLM Teacher label visualization runner
# ==============================================================================
# Usage:
#   bash src/scripts/run_visualize_vlm_teacher.sh <vlm_labels_jsonl> <parquet> <tar_dir> <out_dir> [options...]
#
# Examples:
#   bash src/scripts/run_visualize_vlm_teacher.sh \
#     data/SSTK/10K_local/artifacts/vlm_teacher/labels/crop_label_v1_rerun1_public_e2e.jsonl \
#     data/SSTK/10K_local/filtered_sstk_100.parquet \
#     /sstk/20230916/sstk_100 \
#     data/SSTK/10K_local/artifacts/vlm_teacher/visualizations/vlm_teacher_rerun1_public_e2e \
#     --teacher_scores_jsonl data/SSTK/10K_local/artifacts/teacher/scores/teacher_scores_ar_rerun1_public_e2e.jsonl \
#     --image_dir data/SSTK/10K_local/images \
#     --image_ids_file data/SSTK/10K_local/artifacts/reports/rerun1_public_e2e/sample_ids_supercat12.txt \
#     --num_samples 0 \
#     --server_mode 1

set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: bash src/scripts/run_visualize_vlm_teacher.sh <vlm_labels_jsonl> <parquet> <tar_dir> <out_dir> [options...]"
  exit 1
fi

VLM_LABELS_JSONL="$1"
PARQUET="$2"
TAR_DIR="$3"
OUT_DIR="$4"
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
    echo "[warn] venv not found at $VENV_PATH, using current python."
  fi
fi

SRC_DIR=$(cd "$(dirname "$0")/.." && pwd)
PY_SCRIPT="${SRC_DIR}/visualize_vlm_teacher_labels.py"

echo "=============================================="
echo " Visualize VLM Teacher Labels Runner"
echo "=============================================="
echo "  VLM Labels : $VLM_LABELS_JSONL"
echo "  Parquet    : $PARQUET"
echo "  Tar Dir    : $TAR_DIR"
echo "  Out Dir    : $OUT_DIR"
echo "  ServerMode : $SERVER_MODE"
if [ "${#PASS_ARGS[@]}" -gt 0 ]; then
  echo "  Extra Args : ${PASS_ARGS[*]}"
fi
echo "=============================================="

python "$PY_SCRIPT" \
  --vlm_labels_jsonl "$VLM_LABELS_JSONL" \
  --parquet "$PARQUET" \
  --tar_dir "$TAR_DIR" \
  --out_dir "$OUT_DIR" \
  "${PASS_ARGS[@]}"

echo "=============================================="
echo " Done."
echo "=============================================="

