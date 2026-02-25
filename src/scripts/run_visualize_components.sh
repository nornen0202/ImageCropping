#!/bin/bash
# ==============================================================================
# run_visualize_components.sh
# C2/C3/C5 컴포넌트 및 통합(한 장에 overlay) 시각화 실행 스크립트
# ==============================================================================
# USAGE
#   bash src/scripts/run_visualize_components.sh <parquet> <tar_dir> <out_dir> [options...]
#
# Positional args:
#   (1) PARQUET   : filtered parquet 경로
#   (2) TAR_DIR   : 원본 이미지 tar 디렉토리
#   (3) OUT_DIR   : 시각화 출력 디렉토리
#
# Options:
#   --merged_jsonl PATH   : 병합 피처 jsonl (c2/c3/c5 동시 로드)
#   --c2_jsonl PATH       : c2 jsonl
#   --c3_jsonl PATH       : c3 jsonl
#   --c5_jsonl PATH       : c5 jsonl
#   --num_samples N       : 샘플 수
#   --draw_c2 0|1         : c2 개별 이미지 저장 여부
#   --draw_c3 0|1         : c3 개별 이미지 저장 여부
#   --draw_c5 0|1         : c5 개별 이미지 저장 여부
#   --draw_combined 0|1   : c2+c3+c5 통합 overlay 저장 여부
#   --max_masks_per_image N
#   --kp_score_thr FLOAT
#   --server_mode 0|1     : 0=로컬(venv 활성화), 1=서버
#   --venv_path PATH      : 로컬 venv activate 경로

: <<'USAGE'
# Examples:
# [로컬: merged jsonl 기반, 통합 overlay 활성화]
   bash src/scripts/run_visualize_components.sh \
     data/SSTK/10K_local/filtered_sstk_100.parquet \
     /media/jyju25/T7_4TB_JY/Projects_26/Dataset/SSTK/20230916/sstk_100 \
     data/SSTK/10K_local/visualizations_v2 \
     --merged_jsonl data/SSTK/10K_local/feats_c2c3c5_v2_strict_enriched.jsonl \
     --draw_combined 1 --num_samples 200 --server_mode 0
#
# [서버: 분리 jsonl 기반]
   bash src/scripts/run_visualize_components.sh \
     data/SSTK/10K/filtered_sstk_100.parquet \
     /sstk/20230916/sstk_100 \
     data/SSTK/10K/visualizations_v2 \
     --c2_jsonl data/SSTK/10K/feats_c2.jsonl \
     --c3_jsonl data/SSTK/10K/feats_c3_v2_strict_enriched.jsonl \
     --c5_jsonl data/SSTK/10K/feats_c5.jsonl \
     --draw_combined 1 --num_samples 300 --server_mode 1
# ==============================================================================
USAGE

set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: bash src/scripts/run_visualize_components.sh <parquet> <tar_dir> <out_dir> [options...]"
  exit 1
fi

PARQUET="$1"
TAR_DIR="$2"
OUT_DIR="$3"
shift 3

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
PY_SCRIPT="${SRC_DIR}/visualize_components.py"

echo "=============================================="
echo " Visualize Components Runner"
echo "=============================================="
echo "  Parquet   : $PARQUET"
echo "  Tar Dir   : $TAR_DIR"
echo "  Out Dir   : $OUT_DIR"
echo "  ServerMode: $SERVER_MODE"
if [ "${#PASS_ARGS[@]}" -gt 0 ]; then
  echo "  Extra Args: ${PASS_ARGS[*]}"
fi
echo "=============================================="

python3 "$PY_SCRIPT" \
  --parquet "$PARQUET" \
  --tar_dir "$TAR_DIR" \
  --out_dir "$OUT_DIR" \
  "${PASS_ARGS[@]}"

echo "=============================================="
echo " Done."
echo "=============================================="
