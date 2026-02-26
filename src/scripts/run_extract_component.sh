#!/bin/bash
# ==============================================================================
# run_extract_component.sh
# 특징 추출 파이프라인 통합 실행 스크립트
#
# GPU 수에 따라 자동으로 모드를 선택:
#   GPU 1장  →  extract_features_single.py  (No-Ray, 낮은 오버헤드)
#   GPU 2장+ →  extract_features_pipeline.py (Ray Multi-GPU)
# --mode 옵션으로 강제 지정도 가능.
#
# ==============================================================================
# USAGE
# ==============================================================================
#
# [가이드 출력 — 10K 전체에서 feats_c2c3c5_v2_strict_enriched.jsonl 만들기]
# bash src/scripts/run_extract_component.sh --guide_10k
#
# [로컬 — C1만 추출]
# bash src/scripts/run_extract_component.sh \
#     data/SSTK/10K_local/filtered_sstk_100.parquet sstk_100 \
#     data/SSTK/10K_local/feats_c1.jsonl \
#     --component c1 --server_mode 0
#
# [서버 — C2+C3만, quality_first]
# bash src/scripts/run_extract_component.sh \
#     data/SSTK/10K/filtered_sstk_100.parquet sstk_100 \
#     data/SSTK/10K/feats_c2c3.jsonl \
#     --component c2 c3 --priority quality_first --server_mode 1
#
# [서버 — 전체, 멀티 GPU 강제]
# bash src/scripts/run_extract_component.sh \
#     data/SSTK/10K/filtered_sstk_100.parquet sstk_100 \
#     data/SSTK/10K/feats_all.jsonl \
#     --component all --mode multi --server_mode 1
#
# 2>&1 | tee src/scripts/log_extract_component.log
#
# ==============================================================================
# 인자 설명:
#   (1) INPUT_PARQUET : 필터링된 parquet 파일 경로
#   (2) BUCKET        : 버킷명 (예: sstk_100)
#   (3) OUTPUT_JSONL  : 결과 저장 경로
#   --component       : 실행할 컴포넌트 목록 (c1 c2 c3 c4 c5 all) [기본: all]
#   --priority        : high_efficiency | quality_first  [기본: high_efficiency]
#   --mode            : auto | single | multi            [기본: auto]
#   --num_workers     : Ray worker 수 (multi 모드; 기본: GPU 수 자동 감지)
#   --batch_size      : 배치 크기 [기본: 16]
#   --server_mode     : 0=로컬(venv 활성화+로컬 경로), 1=서버 [기본: 1]
#   --tar_dir         : tar 파일 디렉토리 (미지정 시 server_mode에서 자동 설정)
#   --weights_dir     : C3 가중치 디렉토리 (기본: PROJECT_ROOT/weights)
# ==============================================================================

: <<'USAGE'
# [가이드 출력 — 10K 전체에서 feats_c2c3c5_v2_strict_enriched.jsonl 만들기]
bash src/scripts/run_extract_component.sh --guide_10k

# [서버 — C1만 추출]
bash src/scripts/run_extract_component.sh \
   data/SSTK/10K_local/filtered_sstk_100.parquet sstk_100 \
   data/SSTK/10K_local/feats_c1.jsonl \
   --component c1 --priority quality_first --server_mode 1

# [서버 — C2만 추출]
bash src/scripts/run_extract_component.sh \
   data/SSTK/10K_local/filtered_sstk_100.parquet sstk_100 \
   data/SSTK/10K_local/feats_c2.jsonl \
   --component c2 --priority quality_first --server_mode 1

# [서버 — C3만 추출]
bash src/scripts/run_extract_component.sh \
   data/SSTK/10K_local/filtered_sstk_100.parquet sstk_100 \
   data/SSTK/10K_local/feats_c3.jsonl \
   --component c3 --priority quality_first --server_mode 1

# [로컬 — C5(수평선/롤/대칭)만 추출]
bash src/scripts/run_extract_component.sh \
   data/SSTK/10K_local/filtered_sstk_100.parquet sstk_100 \
   data/SSTK/10K_local/feats_c5.jsonl \
   --component c5 --priority high_efficiency --server_mode 0

# [로컬 — C3만 추출]
bash src/scripts/run_extract_component.sh \
   data/SSTK/10K_local/filtered_sstk_100.parquet sstk_100 \
   data/SSTK/10K_local/feats_c3.jsonl \
   --component c3 --priority quality_first --server_mode 0

# [서버 — C1+C2+C3만, quality_first]
bash src/scripts/run_extract_component.sh \
   data/SSTK/10K_local/filtered_sstk_100.parquet sstk_100 \
   data/SSTK/10K_local/feats_c1c2c3.jsonl \
   --component c1 c2 c3 --priority quality_first --server_mode 1

# [서버 — C1+C2+C3만, quality_first]
bash src/scripts/run_extract_component.sh \
   data/SSTK/10K/filtered_sstk_100.parquet sstk_100 \
   data/SSTK/10K/feats_c1c2c3.jsonl \
   --component c1 c2 c3 --priority quality_first --server_mode 1

# [서버 — 10K 전체에서 C2+C3(strict)+C5 추출 후 병합 피처 생성]
# 1) C2
bash src/scripts/run_extract_component.sh \
   data/SSTK/10K/filtered_sstk_100.parquet sstk_100 \
   data/SSTK/10K/feats_c2.jsonl \
   --component c2 --priority quality_first --server_mode 1

# 2) C3 (사람 오검출 억제 strict 모드: 기본값 1)
C3_PERSON_VERIFY_STRICT=1 bash src/scripts/run_extract_component.sh \
   data/SSTK/10K/filtered_sstk_100.parquet sstk_100 \
   data/SSTK/10K/feats_c3_v2_strict.jsonl \
   --component c3 --priority quality_first --server_mode 1

# 3) C3 enrich (face/gaze proxy 보강)
python src/scripts/enrich_c3_pose_jsonl.py \
   --input_c3_jsonl data/SSTK/10K/feats_c3_v2_strict.jsonl \
   --input_parquet data/SSTK/10K/filtered_sstk_100.parquet \
   --use_actual_image_size 1 \
   --tar_dir /sstk/20230916/sstk_100 \
   --output_jsonl data/SSTK/10K/feats_c3_v2_strict_enriched.jsonl

# 4) C5 (horizon/roll/symmetry)
bash src/scripts/run_extract_component.sh \
   data/SSTK/10K/filtered_sstk_100.parquet sstk_100 \
   data/SSTK/10K/feats_c5.jsonl \
   --component c5 --priority high_efficiency --server_mode 1

# 5) 병합 피처 생성 (최종)
python src/scripts/merge_feature_jsonl.py \
   --input_parquet data/SSTK/10K/filtered_sstk_100.parquet \
   --inputs data/SSTK/10K/feats_c2.jsonl \
            data/SSTK/10K/feats_c3_v2_strict_enriched.jsonl \
            data/SSTK/10K/feats_c5.jsonl \
   --output_jsonl data/SSTK/10K/feats_c2c3c5_v2_strict_enriched.jsonl
USAGE

set -euo pipefail

if [ "${1:-}" = "--guide_10k" ]; then
cat <<'GUIDE'
==============================================
 10K Full Guide (C2 + C3 strict + C5 + merge)
==============================================
# export CUDA_VISIBLE_DEVICES=0

# (A) C2 추출
bash src/scripts/run_extract_component.sh \
  data/SSTK/10K/filtered_sstk_100.parquet sstk_100 \
  data/SSTK/10K/feats_c2.jsonl \
  --component c2 --priority quality_first --server_mode 1 \
   2>&1  | tee src/scripts/logs/run_extract_component_10K_c2.log

# (B) C3 strict 추출
C3_PERSON_VERIFY_STRICT=1 bash src/scripts/run_extract_component.sh \
  data/SSTK/10K/filtered_sstk_100.parquet sstk_100 \
  data/SSTK/10K/feats_c3_v2_strict.jsonl \
  --component c3 --priority quality_first --server_mode 1 \
   2>&1  | tee src/scripts/logs/run_extract_component_10K_c3_strict.log

# (C) C3 enrich
python src/scripts/enrich_c3_pose_jsonl.py \
  --input_c3_jsonl data/SSTK/10K/feats_c3_v2_strict.jsonl \
  --input_parquet data/SSTK/10K/filtered_sstk_100.parquet \
  --use_actual_image_size 1 \
  --tar_dir /sstk/20230916/sstk_100 \
  --output_jsonl data/SSTK/10K/feats_c3_v2_strict_enriched.jsonl \
   2>&1  | tee src/scripts/logs/enrich_c3_pose_jsonl_10K.log

# (D) C5 추출
bash src/scripts/run_extract_component.sh \
  data/SSTK/10K/filtered_sstk_100.parquet sstk_100 \
  data/SSTK/10K/feats_c5.jsonl \
  --component c5 --priority quality_first --server_mode 1 \
   2>&1  | tee src/scripts/logs/run_extract_component_10K_c5.log

# (E) 최종 병합
python src/scripts/merge_feature_jsonl.py \
  --input_parquet data/SSTK/10K/filtered_sstk_100.parquet \
  --inputs data/SSTK/10K/feats_c2.jsonl \
           data/SSTK/10K/feats_c3_v2_strict_enriched.jsonl \
           data/SSTK/10K/feats_c5.jsonl \
  --output_jsonl data/SSTK/10K/feats_c2c3c5_v2_strict_enriched.jsonl \
   2>&1  | tee src/scripts/logs/merge_feature_jsonl_10K.log

==============================================
GUIDE
exit 0
fi

export HUGGINGFACEHUB_API_TOKEN=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
export HF_TOKEN="$HUGGINGFACEHUB_API_TOKEN"

# ── 위치 인자 ─────────────────────────────────────────────────────────────────
if [ "$#" -lt 3 ]; then
    echo "Usage: bash run_extract_component.sh <parquet> <bucket> <output> [options...]"
    exit 1
fi

INPUT_PARQUET="$1"
BUCKET="$2"
OUTPUT_JSONL="$3"
shift 3

# ── 옵션 기본값 ───────────────────────────────────────────────────────────────
COMPONENT="all"        # 'all' or space-separated list passed as single string
PRIORITY="high_efficiency"
MODE="auto"            # auto | single | multi
NUM_WORKERS=""
BATCH_SIZE=16
SERVER_MODE=1
TAR_DIR=""
WEIGHTS_DIR=""
C4_LANG="en"

# ── 옵션 파싱 ─────────────────────────────────────────────────────────────────
while [ "$#" -gt 0 ]; do
    case "$1" in
        --component)
            # 다음 인자가 '-'로 시작하지 않는 동안 컴포넌트 목록 수집
            COMPONENT=""
            shift
            while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do
                COMPONENT="$COMPONENT $1"
                shift
            done
            COMPONENT="${COMPONENT# }"   # 앞 공백 제거
            ;;
        --priority)    PRIORITY="$2";    shift 2 ;;
        --mode)        MODE="$2";        shift 2 ;;
        --num_workers) NUM_WORKERS="$2"; shift 2 ;;
        --batch_size)  BATCH_SIZE="$2";  shift 2 ;;
        --server_mode) SERVER_MODE="$2"; shift 2 ;;
        --tar_dir)     TAR_DIR="$2";     shift 2 ;;
        --weights_dir) WEIGHTS_DIR="$2"; shift 2 ;;
        --c4_lang)     C4_LANG="$2";     shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── 경로 계산 ─────────────────────────────────────────────────────────────────
SRC_DIR=$(cd "$(dirname "$0")/.." && pwd)
PROJECT_ROOT=$(cd "${SRC_DIR}/.." && pwd)

# TAR_DIR 기본값: server_mode에 따라 분기
if [ -z "$TAR_DIR" ]; then
    if [ "$SERVER_MODE" -ne 1 ]; then
        TAR_DIR="/media/jyju25/T7_4TB_JY/Projects_26/Dataset/SSTK/20230916/sstk_100"
    else
        TAR_DIR="/sstk/20230916/sstk_100"
    fi
fi

# WEIGHTS_DIR 기본값
if [ -z "$WEIGHTS_DIR" ]; then
    WEIGHTS_DIR="${PROJECT_ROOT}/weights"
fi

# ── 가상환경 활성화 (로컬 모드만) ────────────────────────────────────────────
if [ "$SERVER_MODE" -ne 1 ]; then
    VENV="/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate"
    if [ -f "$VENV" ]; then
        # shellcheck disable=SC1090
        source "$VENV"
    else
        echo "Warning: venv not found at $VENV, using system python."
    fi
fi

# ── PYTHONPATH (extract_features/ + third_party 포함) ────────────────────────
export PYTHONPATH="${SRC_DIR}/extract_features:${PROJECT_ROOT}/third_party/efficientvit:${PROJECT_ROOT}/third_party/sam2:${PYTHONPATH:-}"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export OMP_NUM_THREADS=8
export RAY_IGNORE_UNHANDLED_ERRORS=1

# ── GPU 수 감지 ───────────────────────────────────────────────────────────────
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "Warning: No GPUs detected. Forcing single mode (CPU)."
    NUM_GPUS=0
    MODE="single"
fi

# ── 모드 결정 (auto) ──────────────────────────────────────────────────────────
if [ "$MODE" = "auto" ]; then
    if [ "$NUM_GPUS" -le 1 ]; then
        MODE="single"
    else
        MODE="multi"
    fi
fi

echo "=============================================="
echo " Feature Extraction Component Runner"
echo "=============================================="
echo "  Input    : $INPUT_PARQUET"
echo "  Bucket   : $BUCKET"
echo "  Output   : $OUTPUT_JSONL"
echo "  Component: $COMPONENT"
echo "  Priority : $PRIORITY"
echo "  Mode     : $MODE  (Detected GPUs: $NUM_GPUS)"
echo "  Tar Dir  : $TAR_DIR"
echo "  WeightsDir: $WEIGHTS_DIR"
echo "=============================================="

# ── 실행 ─────────────────────────────────────────────────────────────────────
if [ "$MODE" = "single" ]; then
    echo "[MODE] Single-GPU (No Ray)"
    python3 "${SRC_DIR}/extract_features_single.py" \
        --input_parquet  "$INPUT_PARQUET" \
        --bucket         "$BUCKET" \
        --tar_dir        "$TAR_DIR" \
        --output_jsonl   "$OUTPUT_JSONL" \
        --component      $COMPONENT \
        --priority       "$PRIORITY" \
        --batch_size     "$BATCH_SIZE" \
        --weights_dir    "$WEIGHTS_DIR" \
        --c4_lang        "$C4_LANG"
else
    echo "[MODE] Multi-GPU (Ray)"
    WORKERS_ARG=""
    if [ -n "$NUM_WORKERS" ]; then
        WORKERS_ARG="--num_workers $NUM_WORKERS"
    fi
    python3 "${SRC_DIR}/extract_features_pipeline.py" \
        --input_parquet  "$INPUT_PARQUET" \
        --bucket         "$BUCKET" \
        --tar_dir        "$TAR_DIR" \
        --output_jsonl   "$OUTPUT_JSONL" \
        --component      $COMPONENT \
        --priority       "$PRIORITY" \
        --batch_size     "$BATCH_SIZE" \
        --weights_dir    "$WEIGHTS_DIR" \
        --c4_lang        "$C4_LANG" \
        $WORKERS_ARG
fi

echo "=============================================="
echo " Done."
echo "=============================================="
