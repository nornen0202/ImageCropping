#!/bin/bash
# ==============================================================================
# run_extract_component.sh
# 특징 추출 파이프라인 통합 실행 스크립트
#
# GPU 수에 따라 자동으로 모드를 선택:
#   GPU 1장  →  extract_features_single.py (single)
#   GPU 2장+ →  extract_features_single.py 샤딩 병렬 (multi, No-Ray)
#              (각 shard는 CUDA_VISIBLE_DEVICES를 1개 GPU로 제한하고
#               C4/C5/C6 디바이스를 shard-local 0번으로 자동 정규화)
# Ray는 --mode ray로 명시적으로 요청한 경우에만 사용.
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
# [서버 — 전체, 멀티 GPU 강제(기본: No-Ray 샤딩)]
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
#   --component       : 실행할 컴포넌트 목록 (c1 c2 c3 c4 c5 c6 all) [기본: all]
#   --priority        : high_efficiency | quality_first  [기본: high_efficiency]
#   --mode            : auto | single | multi | ray      [기본: auto]
#                       * multi = No-Ray 멀티 GPU 샤딩
#                       * ray   = 레거시 Ray 모드(명시적 요청 시)
#   --num_workers     : 멀티 모드 worker 수(기본: GPU 수 자동 감지)
#   --gpu_ids         : 사용할 GPU id CSV (예: 0,1,2,3)
#   --batch_size      : 배치 크기 [기본: 16]
#   --server_mode     : 0=로컬(venv 활성화+로컬 경로), 1=서버 [기본: 1]
#   --venv_path       : 로컬 모드에서 사용할 가상환경 activate 경로
#   --tar_dir         : tar 파일 디렉토리 (미지정 시 server_mode에서 자동 설정)
#   --image_dir       : optional 로컬 이미지 디렉토리(<image_id>.<ext>) 우선 로드
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

# [로컬 — C6(gaze/headpose)만 추출]
bash src/scripts/run_extract_component.sh \
   data/SSTK/10K_local/filtered_sstk_100.parquet sstk_100 \
   data/SSTK/10K_local/feats_c6.jsonl \
   --component c6 --priority quality_first --server_mode 0

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
# 0) 권장: C2+C3(strict)+C5를 1-pass로 추출(가장 효율적)
C3_PERSON_VERIFY_STRICT=1 bash src/scripts/run_extract_component.sh \
   data/SSTK/10K/filtered_sstk_100.parquet sstk_100 \
   data/SSTK/10K/feats_c2c3c5_v2_strict_raw.jsonl \
   --component c2 c3 c5 --priority quality_first --server_mode 1

# 1) C3 enrich (face/gaze proxy 보강) -> 최종 병합 피처
python3 src/scripts/enrich_c3_pose_jsonl.py \
   --input_c3_jsonl data/SSTK/10K/feats_c2c3c5_v2_strict_raw.jsonl \
   --input_parquet data/SSTK/10K/filtered_sstk_100.parquet \
   --use_actual_image_size 1 \
   --tar_dir /sstk/20230916/sstk_100 \
   --output_jsonl data/SSTK/10K/feats_c2c3c5_v2_strict_enriched.jsonl

# [레거시 분리 모드 예시]
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
python3 src/scripts/enrich_c3_pose_jsonl.py \
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
python3 src/scripts/merge_feature_jsonl.py \
   --input_parquet data/SSTK/10K/filtered_sstk_100.parquet \
   --inputs data/SSTK/10K/feats_c2.jsonl \
            data/SSTK/10K/feats_c3_v2_strict_enriched.jsonl \
            data/SSTK/10K/feats_c5.jsonl \
   --output_jsonl data/SSTK/10K/feats_c2c3c5_v2_strict_enriched.jsonl
USAGE

set -euo pipefail

if [ "${1:-}" = "--guide_10k" ]; then
cat <<'GUIDE'
===========================================================
 10K Full Guide (권장: C2+C3 strict+C5 1-pass + C3 enrich)
===========================================================
# export CUDA_VISIBLE_DEVICES=0

# (A) 1-pass precompute: C2 + C3(strict) + C5
C3_PERSON_VERIFY_STRICT=1 bash src/scripts/run_extract_component.sh \
  data/SSTK/10K/filtered_sstk_100.parquet sstk_100 \
  data/SSTK/10K/feats_c2c3c5_v2_strict_raw.jsonl \
  --component c2 c3 c5 --priority quality_first --server_mode 1 \
   2>&1  | tee src/scripts/logs/run_extract_component_10K_c2c3c5_unified.log

# (B) C3 enrich(face/gaze proxy 포함) -> 최종 병합 피처
python3 src/scripts/enrich_c3_pose_jsonl.py \
  --input_c3_jsonl data/SSTK/10K/feats_c2c3c5_v2_strict_raw.jsonl \
  --input_parquet data/SSTK/10K/filtered_sstk_100.parquet \
  --use_actual_image_size 1 \
  --tar_dir /sstk/20230916/sstk_100 \
  --output_jsonl data/SSTK/10K/feats_c2c3c5_v2_strict_enriched.jsonl \
   2>&1  | tee src/scripts/logs/enrich_c3_pose_jsonl_10K.log

# (참고) 레거시 분리 모드가 필요한 경우에만 C2/C3/C5를 별도 실행 후 merge_feature_jsonl.py 사용

===========================================================
GUIDE
exit 0
fi

# Optional HuggingFace auth propagation (for private/gated resources).
# Public weights should still work without tokens.
if [ -n "${HUGGINGFACEHUB_API_TOKEN:-}" ]; then
  export HF_TOKEN="${HF_TOKEN:-$HUGGINGFACEHUB_API_TOKEN}"
elif [ -n "${HF_TOKEN:-}" ]; then
  export HUGGINGFACEHUB_API_TOKEN="${HUGGINGFACEHUB_API_TOKEN:-$HF_TOKEN}"
fi

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
MODE="auto"            # auto | single | multi | ray
NUM_WORKERS=""
GPU_IDS=""
BATCH_SIZE=16
SERVER_MODE=1
TAR_DIR=""
IMAGE_DIR=""
WEIGHTS_DIR=""
C4_LANG="en"
VENV_PATH="/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate"

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
        --gpu_ids)     GPU_IDS="$2";     shift 2 ;;
        --batch_size)  BATCH_SIZE="$2";  shift 2 ;;
        --server_mode) SERVER_MODE="$2"; shift 2 ;;
        --venv_path)   VENV_PATH="$2";  shift 2 ;;
        --tar_dir)     TAR_DIR="$2";     shift 2 ;;
        --image_dir)   IMAGE_DIR="$2";   shift 2 ;;
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
    if [ -f "$VENV_PATH" ]; then
        # shellcheck disable=SC1090
        source "$VENV_PATH"
    else
        echo "Warning: venv not found at $VENV_PATH, using system python."
    fi
fi

# ── PYTHONPATH (extract_features/ + third_party 포함) ────────────────────────
export PYTHONPATH="${SRC_DIR}/extract_features:${PROJECT_ROOT}/third_party/efficientvit:${PROJECT_ROOT}/third_party/sam2:${PROJECT_ROOT}/third_party/scalelsd:${PROJECT_ROOT}/third_party/gazelle:${PYTHONPATH:-}"
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

if [ "$MODE" != "single" ] && [ "$MODE" != "multi" ] && [ "$MODE" != "ray" ]; then
    echo "Error: --mode must be one of auto|single|multi|ray"
    exit 1
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
echo "  Image Dir: ${IMAGE_DIR:-<none>}"
echo "  WeightsDir: $WEIGHTS_DIR"
echo "=============================================="

IMAGE_DIR_ARGS=()
if [ -n "$IMAGE_DIR" ]; then
    IMAGE_DIR_ARGS=(--image_dir "$IMAGE_DIR")
fi

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
        --c4_lang        "$C4_LANG" \
        "${IMAGE_DIR_ARGS[@]}"
elif [ "$MODE" = "multi" ]; then
    echo "[MODE] Multi-GPU (No Ray, sharded single-process workers)"

    resolve_gpu_ids() {
        local csv="$GPU_IDS"
        if [ -z "$csv" ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
            csv="$CUDA_VISIBLE_DEVICES"
        fi
        if [ -z "$csv" ]; then
            local n="$NUM_GPUS"
            local arr=()
            local i
            for ((i=0; i<n; i++)); do
                arr+=("$i")
            done
            (IFS=,; echo "${arr[*]}")
            return
        fi
        # normalize spaces
        csv="${csv// /}"
        echo "$csv"
    }
    # shard 프로세스는 CUDA_VISIBLE_DEVICES로 단일 GPU만 보이도록 제한한다.
    # 모델별 디바이스 env가 절대 인덱스(gpu:3/cuda:3)로 들어오더라도 shard-local
    # 인덱스(0)로 강제 정규화하여 C4/C5/C6가 올바른 GPU를 사용하도록 한다.
    normalize_paddle_gpu_dev() {
        local raw="$1"
        local s="${raw// /}"
        if [[ "$s" == gpu:* ]] || [ "$s" = "gpu" ]; then
            echo "gpu:0"
            return
        fi
        echo "$s"
    }

    normalize_torch_cuda_dev() {
        local raw="$1"
        local s="${raw// /}"
        if [[ "$s" == cuda:* ]] || [ "$s" = "cuda" ]; then
            echo "cuda:0"
            return
        fi
        echo "$s"
    }

    GPU_CSV="$(resolve_gpu_ids)"
    IFS=',' read -r -a GPU_ARR <<< "$GPU_CSV"
    WORKERS="${#GPU_ARR[@]}"
    if [ -n "$NUM_WORKERS" ]; then
        WORKERS="$NUM_WORKERS"
    fi
    if [ "$WORKERS" -lt 1 ]; then
        echo "Error: invalid worker count: $WORKERS"
        exit 1
    fi
    if [ "$WORKERS" -gt "${#GPU_ARR[@]}" ]; then
        echo "Warning: requested workers($WORKERS) > available gpu_ids(${#GPU_ARR[@]}). clipping."
        WORKERS="${#GPU_ARR[@]}"
    fi
    if [ "$WORKERS" -le 1 ]; then
        echo "Warning: effective workers <= 1. falling back to single mode."
        python3 "${SRC_DIR}/extract_features_single.py" \
            --input_parquet  "$INPUT_PARQUET" \
            --bucket         "$BUCKET" \
            --tar_dir        "$TAR_DIR" \
            --output_jsonl   "$OUTPUT_JSONL" \
            --component      $COMPONENT \
            --priority       "$PRIORITY" \
            --batch_size     "$BATCH_SIZE" \
            --weights_dir    "$WEIGHTS_DIR" \
            --c4_lang        "$C4_LANG" \
            "${IMAGE_DIR_ARGS[@]}"
    else
        TMP_DIR="${OUTPUT_JSONL}.shards.$$"
        mkdir -p "$TMP_DIR"
        PIDS=()
        SHARD_FILES=()
        C4_DEVICE_SHARD="$(normalize_paddle_gpu_dev "${C4_PPOCR_DEVICE:-gpu:0}")"
        C5_DEVICE_SHARD="$(normalize_torch_cuda_dev "${C5_SCALELSD_DEVICE:-cuda}")"
        C6_DEVICE_SHARD="$(normalize_torch_cuda_dev "${C6_GAZELLE_DEVICE:-cuda}")"
        echo "[multi] gpu_ids=${GPU_CSV} workers=${WORKERS}"
        echo "[multi] shard-local devices: C4_PPOCR_DEVICE=${C4_DEVICE_SHARD}, C5_SCALELSD_DEVICE=${C5_DEVICE_SHARD}, C6_GAZELLE_DEVICE=${C6_DEVICE_SHARD}"

        i=0
        while [ "$i" -lt "$WORKERS" ]; do
            GPU_ID="${GPU_ARR[$i]}"
            SHARD_OUT="${TMP_DIR}/part_${i}.jsonl"
            SHARD_LOG="${TMP_DIR}/part_${i}.log"
            SHARD_FILES+=("$SHARD_OUT")
            echo "[multi] launch shard=$i/$WORKERS gpu=$GPU_ID -> $SHARD_OUT"
            CUDA_VISIBLE_DEVICES="$GPU_ID" \
            C4_PPOCR_DEVICE="$C4_DEVICE_SHARD" \
            C5_SCALELSD_DEVICE="$C5_DEVICE_SHARD" \
            C6_GAZELLE_DEVICE="$C6_DEVICE_SHARD" \
            python3 "${SRC_DIR}/extract_features_single.py" \
                --input_parquet  "$INPUT_PARQUET" \
                --bucket         "$BUCKET" \
                --tar_dir        "$TAR_DIR" \
                --output_jsonl   "$SHARD_OUT" \
                --component      $COMPONENT \
                --priority       "$PRIORITY" \
                --batch_size     "$BATCH_SIZE" \
                --weights_dir    "$WEIGHTS_DIR" \
                --c4_lang        "$C4_LANG" \
                "${IMAGE_DIR_ARGS[@]}" \
                --num_shards     "$WORKERS" \
                --shard_index    "$i" \
                >"$SHARD_LOG" 2>&1 &
            PIDS+=("$!")
            i=$((i + 1))
        done

        FAIL=0
        for pid in "${PIDS[@]}"; do
            if ! wait "$pid"; then
                FAIL=1
            fi
        done
        if [ "$FAIL" -ne 0 ]; then
            echo "Error: one or more shard workers failed. logs under: $TMP_DIR"
            exit 1
        fi

        : > "$OUTPUT_JSONL"
        for f in "${SHARD_FILES[@]}"; do
            if [ -f "$f" ]; then
                cat "$f" >> "$OUTPUT_JSONL"
            fi
        done
        echo "[multi] merged output -> $OUTPUT_JSONL"
        echo "[multi] shard logs -> $TMP_DIR"
    fi
else
    echo "[MODE] Multi-GPU (Ray, explicit)"
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
        "${IMAGE_DIR_ARGS[@]}" \
        $WORKERS_ARG
fi

echo "=============================================="
echo " Done."
echo "=============================================="
