#!/bin/bash
# ==============================================================================
# run_phaseA_to_teacher_e2e.sh
# End-to-end pipeline:
#   Phase A Filter -> Phase B Perception Precompute(C1,C2,C3,C5, enrich, merge)
#   -> Candidate Generator -> Teacher Scorer(+QA/+Viz)
# ------------------------------------------------------------------------------
# OCR(C4)는 기본적으로 제외합니다.
# ==============================================================================

: <<'USAGE'
Usage
-----
# 로컬 기본 실행(10K_local, proxy expensive)
bash src/scripts/run_phaseA_to_teacher_e2e.sh

# 서버 기본 실행
bash src/scripts/run_phaseA_to_teacher_e2e.sh --server_mode 1 --data_dir data/SSTK/10K

# 필터는 건너뛰고(이미 parquet 있음) 나머지만 실행
bash src/scripts/run_phaseA_to_teacher_e2e.sh --run_filter 0 --skip_existing 1

# C1/C2/C3/C5를 한 번에 추출 (기본: unified, run_c1=1일 때 C1 포함)
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --precompute_mode unified \
  --run_filter 0

# Teacher real-expensive 활성화(C1 포함 자동 실행)
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --use_real_expensive 1 \
  --align_device cuda \
  --aesthetic_device cuda \
  --exp_batch_size 12

# 스모크 테스트(앞 200장)
bash src/scripts/run_phaseA_to_teacher_e2e.sh --max_images 200 --run_tag smoke200

Core options
------------
--data_dir PATH                 output root (default: data/SSTK/10K_local)
--bucket NAME                   SSTK bucket (default: sstk_100)
--server_mode 0|1               0=local, 1=server (default: 0)
--tar_dir PATH                  TAR root (default: server_mode별 자동)
--run_filter 0|1                Phase A filter 실행 여부 (default: 1)
--curated_pool_size INT         filter curated pool size (default: 10000)
--top_percentile FLOAT          filter top percentile per category (default: 0.2)
--export_curated_images 0|1     filter 후 curated 이미지를 로컬 dir로 추출 (default: 0)
--curated_image_dir PATH        curated 이미지 디렉토리 (default: <data_dir>/images)
--curated_image_skip_existing 0|1  이미지 추출 시 기존 파일 skip (default: 1)
--prefer_curated_images 0|1     후속 단계에서 curated image dir 우선 사용 (default: 1)
--skip_existing 0|1             output 파일이 있으면 단계 skip (default: 1)
--run_tag TAG                   candidates/teacher 출력 suffix (default: "")
--max_images INT                0=all, >0=앞에서 n장(candidate/teacher) (default: 0)
--teacher_proposals_jsonl CSV   candidate 단계 teacher proposal jsonl(쉼표로 다중 경로)
--precompute_mode MODE          unified|split (default: unified)
--extract_gpu_ids CSV           precompute에서 사용할 GPU 목록 (예: 0,1,2,3)

--use_real_expensive 0|1        teacher expensive real model 사용 (default: 0)
--run_qa 0|1                    QA report 생성 여부 (default: 1)
--run_viz 0|1                   teacher viz 생성 여부 (default: 1)
--teacher_multi_gpu -1|0|1      -1=auto(use_real_expensive && multi-gpu면 on), default -1
--teacher_gpu_ids CSV           teacher multi-gpu에서 사용할 GPU 목록
--teacher_num_workers INT       teacher multi-gpu shard worker 수

Advanced stage toggles
----------------------
--run_c1 0|1|-1                 -1=auto(use_real_expensive=1이면 1) (default: -1)
--run_c2 0|1                    (default: 1)
--run_c3 0|1                    (default: 1)
--run_c3_enrich 0|1             (default: 1)
--run_c5 0|1                    (default: 1)
--run_merge 0|1                 (default: 1)
--run_candidates 0|1            (default: 1)
--run_teacher 0|1               (default: 1)
USAGE

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$PROJECT_ROOT"

# ------------------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------------------
BUCKET="sstk_100"
DATA_DIR="data/SSTK/10K_local"
SERVER_MODE=0
VENV_PATH="/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate"
LOCAL_TAR_DIR="/media/jyju25/T7_4TB_JY/Projects_26/Dataset/SSTK/20230916/sstk_100"
SERVER_TAR_DIR="/sstk/20230916/sstk_100"
TAR_DIR=""
SDP_DIR=""
TRAIN_DIR=""
LOG_DIR=""
RUN_TAG=""
SKIP_EXISTING=1
PRECOMPUTE_MODE="unified"

# Filter
RUN_FILTER=1
CURATED_POOL_SIZE=10000
TOP_PERCENTILE=0.2
EXPORT_CURATED_IMAGES=0
CURATED_IMAGE_DIR=""
CURATED_IMAGE_SKIP_EXISTING=1
PREFER_CURATED_IMAGES=1

# Extract/merge
RUN_C1=-1
RUN_C2=1
RUN_C3=1
RUN_C3_ENRICH=1
RUN_C5=1
RUN_MERGE=1
EXTRACT_MODE="auto"
EXTRACT_PRIORITY="quality_first"
C5_PRIORITY="quality_first"
BATCH_SIZE=16
NUM_WORKERS=""
EXTRACT_GPU_IDS=""
C3_PERSON_VERIFY_STRICT=1

# Candidate
RUN_CANDIDATES=1
USE_ACTUAL_IMAGE_SIZE=1
STRICT_ACTUAL_SIZE=1
ACTUAL_SIZE_CACHE_JSON=""
MAX_IMAGES=0
TEACHER_PROPOSALS_JSONL=""
TEACHER_NMS_IOU=0.95
TEACHER_MAX_SEEDS_PER_TEACHER=1
TEACHER_PREFER_EXPAND=1
TEACHER_JITTER_SHIFT_FRACS="0.03"
TEACHER_JITTER_SCALES="0.92,1.0,1.08"

# Teacher
RUN_TEACHER=1
USE_REAL_EXPENSIVE=0
RUN_QA=1
RUN_VIZ=1
NUM_VIZ=120
TARGET_AR="all"
DECISION_FILTER="all"
CHEAP_TOP_M=30
TOP_K=5
TAU_DIV=0.75
ALIGN_MODEL_NAME=""
ALIGN_PRETRAINED=""
ALIGN_DEVICE="auto"
AESTHETIC_DEVICE="auto"
EXP_BATCH_SIZE=24
AESTHETIC_MLP_PATH="weights/improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth"
AESTHETIC_MLP_URL="https://raw.githubusercontent.com/christophschuhmann/improved-aesthetic-predictor/main/sac+logos+ava1-l14-linearMSE.pth"
TEACHER_MULTI_GPU=-1
TEACHER_GPU_IDS=""
TEACHER_NUM_WORKERS=""

# ------------------------------------------------------------------------------
# Option parse
# ------------------------------------------------------------------------------
while [ "$#" -gt 0 ]; do
  case "$1" in
    --bucket) BUCKET="$2"; shift 2 ;;
    --data_dir) DATA_DIR="$2"; shift 2 ;;
    --server_mode) SERVER_MODE="$2"; shift 2 ;;
    --venv_path) VENV_PATH="$2"; shift 2 ;;
    --tar_dir) TAR_DIR="$2"; shift 2 ;;
    --sdp_dir) SDP_DIR="$2"; shift 2 ;;
    --train_dir) TRAIN_DIR="$2"; shift 2 ;;
    --log_dir) LOG_DIR="$2"; shift 2 ;;
    --run_tag) RUN_TAG="$2"; shift 2 ;;
    --skip_existing) SKIP_EXISTING="$2"; shift 2 ;;
    --precompute_mode) PRECOMPUTE_MODE="$2"; shift 2 ;;
    --extract_gpu_ids) EXTRACT_GPU_IDS="$2"; shift 2 ;;

    --run_filter) RUN_FILTER="$2"; shift 2 ;;
    --curated_pool_size) CURATED_POOL_SIZE="$2"; shift 2 ;;
    --top_percentile) TOP_PERCENTILE="$2"; shift 2 ;;
    --export_curated_images) EXPORT_CURATED_IMAGES="$2"; shift 2 ;;
    --curated_image_dir) CURATED_IMAGE_DIR="$2"; shift 2 ;;
    --curated_image_skip_existing) CURATED_IMAGE_SKIP_EXISTING="$2"; shift 2 ;;
    --prefer_curated_images) PREFER_CURATED_IMAGES="$2"; shift 2 ;;

    --run_c1) RUN_C1="$2"; shift 2 ;;
    --run_c2) RUN_C2="$2"; shift 2 ;;
    --run_c3) RUN_C3="$2"; shift 2 ;;
    --run_c3_enrich) RUN_C3_ENRICH="$2"; shift 2 ;;
    --run_c5) RUN_C5="$2"; shift 2 ;;
    --run_merge) RUN_MERGE="$2"; shift 2 ;;
    --extract_mode) EXTRACT_MODE="$2"; shift 2 ;;
    --extract_priority) EXTRACT_PRIORITY="$2"; shift 2 ;;
    --c5_priority) C5_PRIORITY="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --num_workers) NUM_WORKERS="$2"; shift 2 ;;
    --c3_person_verify_strict) C3_PERSON_VERIFY_STRICT="$2"; shift 2 ;;

    --run_candidates) RUN_CANDIDATES="$2"; shift 2 ;;
    --use_actual_image_size) USE_ACTUAL_IMAGE_SIZE="$2"; shift 2 ;;
    --strict_actual_size) STRICT_ACTUAL_SIZE="$2"; shift 2 ;;
    --actual_size_cache_json) ACTUAL_SIZE_CACHE_JSON="$2"; shift 2 ;;
    --max_images) MAX_IMAGES="$2"; shift 2 ;;
    --teacher_proposals_jsonl) TEACHER_PROPOSALS_JSONL="$2"; shift 2 ;;
    --teacher_nms_iou) TEACHER_NMS_IOU="$2"; shift 2 ;;
    --teacher_max_seeds_per_teacher) TEACHER_MAX_SEEDS_PER_TEACHER="$2"; shift 2 ;;
    --teacher_prefer_expand) TEACHER_PREFER_EXPAND="$2"; shift 2 ;;
    --teacher_jitter_shift_fracs) TEACHER_JITTER_SHIFT_FRACS="$2"; shift 2 ;;
    --teacher_jitter_scales) TEACHER_JITTER_SCALES="$2"; shift 2 ;;

    --run_teacher) RUN_TEACHER="$2"; shift 2 ;;
    --use_real_expensive) USE_REAL_EXPENSIVE="$2"; shift 2 ;;
    --run_qa) RUN_QA="$2"; shift 2 ;;
    --run_viz) RUN_VIZ="$2"; shift 2 ;;
    --num_viz) NUM_VIZ="$2"; shift 2 ;;
    --target_ar) TARGET_AR="$2"; shift 2 ;;
    --decision_filter) DECISION_FILTER="$2"; shift 2 ;;
    --cheap_top_m) CHEAP_TOP_M="$2"; shift 2 ;;
    --top_k) TOP_K="$2"; shift 2 ;;
    --tau_div) TAU_DIV="$2"; shift 2 ;;
    --align_model_name) ALIGN_MODEL_NAME="$2"; shift 2 ;;
    --align_pretrained) ALIGN_PRETRAINED="$2"; shift 2 ;;
    --align_device) ALIGN_DEVICE="$2"; shift 2 ;;
    --aesthetic_device) AESTHETIC_DEVICE="$2"; shift 2 ;;
    --exp_batch_size) EXP_BATCH_SIZE="$2"; shift 2 ;;
    --aesthetic_mlp_path) AESTHETIC_MLP_PATH="$2"; shift 2 ;;
    --aesthetic_mlp_url) AESTHETIC_MLP_URL="$2"; shift 2 ;;
    --teacher_multi_gpu) TEACHER_MULTI_GPU="$2"; shift 2 ;;
    --teacher_gpu_ids) TEACHER_GPU_IDS="$2"; shift 2 ;;
    --teacher_num_workers) TEACHER_NUM_WORKERS="$2"; shift 2 ;;

    -h|--help)
      sed -n '1,220p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [ "$PRECOMPUTE_MODE" != "unified" ] && [ "$PRECOMPUTE_MODE" != "split" ]; then
  echo "[error] --precompute_mode must be one of: unified, split"
  exit 1
fi

if [ -z "$TAR_DIR" ]; then
  if [ "$SERVER_MODE" -eq 1 ]; then
    TAR_DIR="$SERVER_TAR_DIR"
  else
    TAR_DIR="$LOCAL_TAR_DIR"
  fi
fi

if [ "$RUN_C1" -lt 0 ]; then
  if [ "$USE_REAL_EXPENSIVE" -eq 1 ]; then
    RUN_C1=1
  else
    RUN_C1=0
  fi
fi

if [ "$TEACHER_MULTI_GPU" -gt 1 ]; then
  echo "[warn] --teacher_multi_gpu expects -1|0|1. got=${TEACHER_MULTI_GPU}, treating as 1"
  TEACHER_MULTI_GPU=1
fi

if [ "$TEACHER_MULTI_GPU" -lt 0 ]; then
  GPU_CNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)
  if [ "$USE_REAL_EXPENSIVE" -eq 1 ] && [ "$GPU_CNT" -gt 1 ]; then
    TEACHER_MULTI_GPU=1
  else
    TEACHER_MULTI_GPU=0
  fi
fi

mkdir -p "$DATA_DIR"

if [ -z "$CURATED_IMAGE_DIR" ]; then
  CURATED_IMAGE_DIR="${DATA_DIR}/images"
fi

if [ -z "$LOG_DIR" ]; then
  if [ -n "$RUN_TAG" ]; then
    LOG_DIR="${DATA_DIR}/logs/e2e_${RUN_TAG}"
  else
    LOG_DIR="${DATA_DIR}/logs/e2e_default"
  fi
fi
mkdir -p "$LOG_DIR"

if [ "$SERVER_MODE" -ne 1 ]; then
  if [ -f "$VENV_PATH" ]; then
    # shellcheck disable=SC1090
    source "$VENV_PATH"
  else
    echo "[warn] venv not found: $VENV_PATH (using current python)"
  fi
fi

if [ -n "$RUN_TAG" ]; then
  SUFFIX="_${RUN_TAG}"
else
  SUFFIX=""
fi

FILTERED_PARQUET="${DATA_DIR}/filtered_${BUCKET}.parquet"
FEATS_C1="${DATA_DIR}/feats_c1.jsonl"
FEATS_C2="${DATA_DIR}/feats_c2.jsonl"
FEATS_C3="${DATA_DIR}/feats_c3_v2_strict.jsonl"
FEATS_C3_ENRICHED="${DATA_DIR}/feats_c3_v2_strict_enriched.jsonl"
FEATS_C5="${DATA_DIR}/feats_c5.jsonl"
FEATS_C2C3C5_RAW="${DATA_DIR}/feats_c2c3c5_v2_strict_raw.jsonl"
MERGED_FEATS="${DATA_DIR}/feats_c2c3c5_v2_strict_enriched.jsonl"

CANDIDATES_JSONL="${DATA_DIR}/candidates_ar${SUFFIX}.jsonl"
TEACHER_JSONL="${DATA_DIR}/teacher_scores_ar${SUFFIX}.jsonl"
TEACHER_OVERVIEW_JSON="${DATA_DIR}/teacher_scores_overview${SUFFIX}.json"
TEACHER_OVERVIEW_CSV="${DATA_DIR}/teacher_scores_overview_by_ar${SUFFIX}.csv"
TEACHER_QA_JSON="${DATA_DIR}/teacher_scores_qa_report${SUFFIX}.json"
TEACHER_QA_CSV="${DATA_DIR}/teacher_scores_qa_report_by_ar${SUFFIX}.csv"
TEACHER_VIZ_DIR="${DATA_DIR}/visualizations/teacher_scorer${SUFFIX}"

if [ -z "$ACTUAL_SIZE_CACHE_JSON" ]; then
  ACTUAL_SIZE_CACHE_JSON="${DATA_DIR}/actual_image_size_map.json"
fi

EFFECTIVE_IMAGE_DIR=""
if [ "$PREFER_CURATED_IMAGES" -eq 1 ] && [ -d "$CURATED_IMAGE_DIR" ]; then
  EFFECTIVE_IMAGE_DIR="$CURATED_IMAGE_DIR"
fi

run_with_log() {
  local name="$1"; shift
  local log_path="${LOG_DIR}/${name}.log"
  echo "[run] ${name}"
  "$@" 2>&1 | tee "$log_path"
}

run_with_log_env() {
  local name="$1"; shift
  local log_path="${LOG_DIR}/${name}.log"
  echo "[run] ${name}"
  env "$@" 2>&1 | tee "$log_path"
}

should_skip_file() {
  local f="$1"
  if [ "$SKIP_EXISTING" -eq 1 ] && [ -f "$f" ]; then
    echo "[skip] exists: $f"
    return 0
  fi
  return 1
}

filter_extra_args=()
if [ "$EXPORT_CURATED_IMAGES" -eq 1 ]; then
  filter_extra_args+=("$CURATED_IMAGE_DIR" "$CURATED_IMAGE_SKIP_EXISTING")
fi

jsonl_has_c1_embeddings() {
  local f="$1"
  if [ ! -f "$f" ]; then
    return 1
  fi
  python - "$f" <<'PY'
import json
import sys
path = sys.argv[1]
count = 0
with open(path, "r", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if isinstance(d.get("c1_img_embed"), list) and d.get("c1_img_embed") and \
           isinstance(d.get("c1_txt_embed"), list) and d.get("c1_txt_embed"):
            count += 1
            if count >= 1:
                sys.exit(0)
sys.exit(1)
PY
}

extract_common_args=(
  --priority "$EXTRACT_PRIORITY"
  --mode "$EXTRACT_MODE"
  --batch_size "$BATCH_SIZE"
  --server_mode "$SERVER_MODE"
  --tar_dir "$TAR_DIR"
)
if [ -n "$NUM_WORKERS" ]; then
  extract_common_args+=(--num_workers "$NUM_WORKERS")
fi
if [ -n "$EXTRACT_GPU_IDS" ]; then
  extract_common_args+=(--gpu_ids "$EXTRACT_GPU_IDS")
fi

echo "========================================================"
echo " E2E Config"
echo "========================================================"
echo " bucket              : $BUCKET"
echo " data_dir            : $DATA_DIR"
echo " server_mode         : $SERVER_MODE"
echo " tar_dir             : $TAR_DIR"
echo " run_tag             : ${RUN_TAG:-<none>}"
echo " skip_existing       : $SKIP_EXISTING"
echo " precompute_mode     : $PRECOMPUTE_MODE"
echo " run_filter          : $RUN_FILTER"
echo " export_curated_img  : $EXPORT_CURATED_IMAGES (dir=$CURATED_IMAGE_DIR, skip_existing=$CURATED_IMAGE_SKIP_EXISTING)"
echo " prefer_curated_img  : $PREFER_CURATED_IMAGES (effective=${EFFECTIVE_IMAGE_DIR:-<none>})"
echo " extract_mode        : $EXTRACT_MODE (gpu_ids=${EXTRACT_GPU_IDS:-auto}, workers=${NUM_WORKERS:-auto})"
echo " run_c1/c2/c3/c5    : $RUN_C1/$RUN_C2/$RUN_C3/$RUN_C5"
echo " run_c3_enrich/merge : $RUN_C3_ENRICH/$RUN_MERGE"
echo " run_candidates      : $RUN_CANDIDATES"
echo " teacher proposals   : ${TEACHER_PROPOSALS_JSONL:-<none>}"
echo " run_teacher         : $RUN_TEACHER (real_expensive=$USE_REAL_EXPENSIVE)"
echo " teacher_multi_gpu   : $TEACHER_MULTI_GPU (gpu_ids=${TEACHER_GPU_IDS:-auto}, workers=${TEACHER_NUM_WORKERS:-auto})"
echo "========================================================"

# ------------------------------------------------------------------------------
# 1) Phase A Filter
# ------------------------------------------------------------------------------
if [ "$RUN_FILTER" -eq 1 ]; then
  if ! should_skip_file "$FILTERED_PARQUET"; then
    run_with_log_env "01_filter" \
      SDP_DIR="$SDP_DIR" \
      TRAIN_DIR="$TRAIN_DIR" \
      TAR_DIR="$TAR_DIR" \
      bash src/scripts/run_filter.sh \
        "$BUCKET" \
        "$FILTERED_PARQUET" \
        "$CURATED_POOL_SIZE" \
        "$TOP_PERCENTILE" \
        "$SERVER_MODE" \
        "${filter_extra_args[@]}"
  fi
fi

if [ ! -f "$FILTERED_PARQUET" ]; then
  echo "[error] filtered parquet not found: $FILTERED_PARQUET"
  exit 1
fi

if [ "$PREFER_CURATED_IMAGES" -eq 1 ] && [ -d "$CURATED_IMAGE_DIR" ]; then
  EFFECTIVE_IMAGE_DIR="$CURATED_IMAGE_DIR"
else
  EFFECTIVE_IMAGE_DIR=""
fi
if [ -n "$EFFECTIVE_IMAGE_DIR" ]; then
  extract_common_args+=(--image_dir "$EFFECTIVE_IMAGE_DIR")
  echo "[info] using curated image cache dir: $EFFECTIVE_IMAGE_DIR"
fi
image_dir_args=()
if [ -n "$EFFECTIVE_IMAGE_DIR" ]; then
  image_dir_args=(--image_dir "$EFFECTIVE_IMAGE_DIR")
fi

# ------------------------------------------------------------------------------
# 2) Phase B Perception Precompute
# ------------------------------------------------------------------------------
if [ "$PRECOMPUTE_MODE" = "unified" ]; then
  # Quality-first path: single pass for C1/C2/C3/C5 to avoid repeated tar traversal.
  if [ "$RUN_C1" -eq 1 ] || [ "$RUN_C2" -eq 1 ] || [ "$RUN_C3" -eq 1 ] || [ "$RUN_C5" -eq 1 ]; then
    if ! should_skip_file "$FEATS_C2C3C5_RAW"; then
      comp_args=()
      if [ "$RUN_C1" -eq 1 ]; then
        comp_args+=("c1")
      fi
      if [ "$RUN_C2" -eq 1 ]; then
        comp_args+=("c2")
      fi
      if [ "$RUN_C3" -eq 1 ]; then
        comp_args+=("c3")
      fi
      if [ "$RUN_C5" -eq 1 ]; then
        comp_args+=("c5")
      fi
      if [ "${#comp_args[@]}" -gt 0 ]; then
        run_with_log_env "02_extract_precompute_unified" \
          C3_PERSON_VERIFY_STRICT="$C3_PERSON_VERIFY_STRICT" \
          bash src/scripts/run_extract_component.sh \
            "$FILTERED_PARQUET" "$BUCKET" "$FEATS_C2C3C5_RAW" \
            --component "${comp_args[@]}" \
            "${extract_common_args[@]}"
      fi
    fi
  fi

  if [ "$RUN_C3_ENRICH" -eq 1 ]; then
    if [ ! -f "$FEATS_C2C3C5_RAW" ]; then
      echo "[error] unified precompute jsonl not found for enrich: $FEATS_C2C3C5_RAW"
      exit 1
    fi
    if ! should_skip_file "$MERGED_FEATS"; then
      run_with_log "03_enrich_c3_unified" \
        python src/scripts/enrich_c3_pose_jsonl.py \
          --input_c3_jsonl "$FEATS_C2C3C5_RAW" \
          --input_parquet "$FILTERED_PARQUET" \
          --use_actual_image_size "$USE_ACTUAL_IMAGE_SIZE" \
          --tar_dir "$TAR_DIR" \
          "${image_dir_args[@]}" \
          --actual_size_cache_json "$ACTUAL_SIZE_CACHE_JSON" \
          --output_jsonl "$MERGED_FEATS"
    fi
  fi

  # Compatibility: keep legacy names mapped to unified output.
  FEATS_C2="$MERGED_FEATS"
  FEATS_C3_ENRICHED="$MERGED_FEATS"
  FEATS_C5="$MERGED_FEATS"
  if [ "$RUN_C1" -eq 1 ]; then
    # Prefer unified raw jsonl as C1 source. It is generated in the same pass and
    # is not affected by stale enriched-file reuse when skip_existing=1.
    C1_CANDIDATE="$FEATS_C2C3C5_RAW"
    if [ -f "$C1_CANDIDATE" ] && jsonl_has_c1_embeddings "$C1_CANDIDATE"; then
      FEATS_C1="$C1_CANDIDATE"
    elif [ -f "$FEATS_C1" ] && jsonl_has_c1_embeddings "$FEATS_C1"; then
      echo "[warn] unified raw has no c1 embeddings. falling back to legacy c1 file: $FEATS_C1"
    else
      echo "[warn] c1 embeddings missing in unified raw/legacy c1. extracting standalone c1..."
      run_with_log "02b_extract_c1_fallback" \
        bash src/scripts/run_extract_component.sh \
          "$FILTERED_PARQUET" "$BUCKET" "$FEATS_C1" \
          --component c1 \
          "${extract_common_args[@]}"
      if ! jsonl_has_c1_embeddings "$FEATS_C1"; then
        echo "[error] failed to build valid c1 embeddings: $FEATS_C1"
        exit 1
      fi
    fi
  fi
else
  if [ "$RUN_C1" -eq 1 ]; then
    if ! should_skip_file "$FEATS_C1"; then
      run_with_log "02_extract_c1" \
        bash src/scripts/run_extract_component.sh \
          "$FILTERED_PARQUET" "$BUCKET" "$FEATS_C1" \
          --component c1 \
          "${extract_common_args[@]}"
    fi
  fi

  if [ "$RUN_C2" -eq 1 ]; then
    if ! should_skip_file "$FEATS_C2"; then
      run_with_log "03_extract_c2" \
        bash src/scripts/run_extract_component.sh \
          "$FILTERED_PARQUET" "$BUCKET" "$FEATS_C2" \
          --component c2 \
          "${extract_common_args[@]}"
    fi
  fi

  if [ "$RUN_C3" -eq 1 ]; then
    if ! should_skip_file "$FEATS_C3"; then
      run_with_log_env "04_extract_c3_strict" \
        C3_PERSON_VERIFY_STRICT="$C3_PERSON_VERIFY_STRICT" \
        bash src/scripts/run_extract_component.sh \
          "$FILTERED_PARQUET" "$BUCKET" "$FEATS_C3" \
          --component c3 \
          "${extract_common_args[@]}"
    fi
  fi

  if [ "$RUN_C3_ENRICH" -eq 1 ]; then
    if [ ! -f "$FEATS_C3" ]; then
      echo "[error] c3 jsonl not found for enrich: $FEATS_C3"
      exit 1
    fi
    if ! should_skip_file "$FEATS_C3_ENRICHED"; then
      run_with_log "05_enrich_c3" \
        python src/scripts/enrich_c3_pose_jsonl.py \
          --input_c3_jsonl "$FEATS_C3" \
          --input_parquet "$FILTERED_PARQUET" \
          --use_actual_image_size "$USE_ACTUAL_IMAGE_SIZE" \
          --tar_dir "$TAR_DIR" \
          "${image_dir_args[@]}" \
          --actual_size_cache_json "$ACTUAL_SIZE_CACHE_JSON" \
          --output_jsonl "$FEATS_C3_ENRICHED"
    fi
  fi

  if [ "$RUN_C5" -eq 1 ]; then
    if ! should_skip_file "$FEATS_C5"; then
      c5_args=(
        --priority "$C5_PRIORITY"
        --mode "$EXTRACT_MODE"
        --batch_size "$BATCH_SIZE"
        --server_mode "$SERVER_MODE"
        --tar_dir "$TAR_DIR"
      )
      if [ -n "$EFFECTIVE_IMAGE_DIR" ]; then
        c5_args+=(--image_dir "$EFFECTIVE_IMAGE_DIR")
      fi
      if [ -n "$NUM_WORKERS" ]; then
        c5_args+=(--num_workers "$NUM_WORKERS")
      fi
      if [ -n "$EXTRACT_GPU_IDS" ]; then
        c5_args+=(--gpu_ids "$EXTRACT_GPU_IDS")
      fi
      run_with_log "06_extract_c5" \
        bash src/scripts/run_extract_component.sh \
          "$FILTERED_PARQUET" "$BUCKET" "$FEATS_C5" \
          --component c5 \
          "${c5_args[@]}"
    fi
  fi

  if [ "$RUN_MERGE" -eq 1 ]; then
    if [ ! -f "$FEATS_C2" ]; then
      echo "[error] c2 jsonl not found: $FEATS_C2"
      exit 1
    fi
    if [ ! -f "$FEATS_C3_ENRICHED" ]; then
      echo "[error] c3 enriched jsonl not found: $FEATS_C3_ENRICHED"
      exit 1
    fi
    if [ ! -f "$FEATS_C5" ]; then
      echo "[error] c5 jsonl not found: $FEATS_C5"
      exit 1
    fi
    if ! should_skip_file "$MERGED_FEATS"; then
      run_with_log "07_merge_features" \
        python src/scripts/merge_feature_jsonl.py \
          --input_parquet "$FILTERED_PARQUET" \
          --inputs "$FEATS_C2" "$FEATS_C3_ENRICHED" "$FEATS_C5" \
          --output_jsonl "$MERGED_FEATS"
    fi
  fi
fi

# ------------------------------------------------------------------------------
# 3) Candidate Generator
# ------------------------------------------------------------------------------
if [ "$RUN_CANDIDATES" -eq 1 ]; then
  if [ ! -f "$FEATS_C2" ] || [ ! -f "$FEATS_C3_ENRICHED" ]; then
    echo "[error] candidate generation requires c2 + c3_enriched jsonl"
    exit 1
  fi
  if ! should_skip_file "$CANDIDATES_JSONL"; then
    cand_extra_args=()
    if [ -n "$TEACHER_PROPOSALS_JSONL" ]; then
      IFS_OLD="$IFS"
      IFS=',' read -r -a teacher_paths <<< "$TEACHER_PROPOSALS_JSONL"
      IFS="$IFS_OLD"
      if [ "${#teacher_paths[@]}" -gt 0 ]; then
        valid_teacher_paths=()
        for tp in "${teacher_paths[@]}"; do
          if [ -n "$tp" ]; then
            valid_teacher_paths+=("$tp")
          fi
        done
        if [ "${#valid_teacher_paths[@]}" -eq 0 ]; then
          valid_teacher_paths=()
        else
          cand_extra_args+=(--teacher_proposals_jsonl)
          for tp in "${valid_teacher_paths[@]}"; do
            cand_extra_args+=("$tp")
          done
        fi
      fi
      if [ "${#cand_extra_args[@]}" -gt 0 ]; then
        cand_extra_args+=(
          --teacher_nms_iou "$TEACHER_NMS_IOU"
          --teacher_max_seeds_per_teacher "$TEACHER_MAX_SEEDS_PER_TEACHER"
          --teacher_prefer_expand "$TEACHER_PREFER_EXPAND"
          --teacher_jitter_shift_fracs
        )
        IFS_OLD="$IFS"
        IFS=',' read -r -a tshift_arr <<< "$TEACHER_JITTER_SHIFT_FRACS"
        IFS="$IFS_OLD"
        for v in "${tshift_arr[@]}"; do
          if [ -n "$v" ]; then
            cand_extra_args+=("$v")
          fi
        done
        cand_extra_args+=(--teacher_jitter_scales)
        IFS_OLD="$IFS"
        IFS=',' read -r -a tscale_arr <<< "$TEACHER_JITTER_SCALES"
        IFS="$IFS_OLD"
        for v in "${tscale_arr[@]}"; do
          if [ -n "$v" ]; then
            cand_extra_args+=("$v")
          fi
        done
      fi
    fi
    run_with_log "08_generate_candidates" \
      bash src/scripts/run_generate_candidates.sh \
        "$FILTERED_PARQUET" \
        "$FEATS_C2" \
        "$FEATS_C3_ENRICHED" \
        "$CANDIDATES_JSONL" \
        --server_mode "$SERVER_MODE" \
        --tar_dir "$TAR_DIR" \
        "${image_dir_args[@]}" \
        --use_actual_image_size "$USE_ACTUAL_IMAGE_SIZE" \
        --strict_actual_size "$STRICT_ACTUAL_SIZE" \
        --actual_size_cache_json "$ACTUAL_SIZE_CACHE_JSON" \
        --max_images "$MAX_IMAGES" \
        "${cand_extra_args[@]}"
  fi
fi

# ------------------------------------------------------------------------------
# 4) Teacher Scorer
# ------------------------------------------------------------------------------
if [ "$RUN_TEACHER" -eq 1 ]; then
  if [ ! -f "$CANDIDATES_JSONL" ]; then
    echo "[error] candidates jsonl not found: $CANDIDATES_JSONL"
    exit 1
  fi
  if [ ! -f "$MERGED_FEATS" ]; then
    echo "[error] merged features jsonl not found: $MERGED_FEATS"
    exit 1
  fi
  if [ "$USE_REAL_EXPENSIVE" -eq 1 ]; then
    if [ ! -f "$FEATS_C1" ]; then
      echo "[error] use_real_expensive=1 requires c1 jsonl: $FEATS_C1"
      exit 1
    fi
    if ! jsonl_has_c1_embeddings "$FEATS_C1"; then
      echo "[error] c1 jsonl exists but has no valid c1 embeddings: $FEATS_C1"
      echo "        hint: rerun with --run_c1 1 and --skip_existing 0"
      exit 1
    fi
  fi
  if ! should_skip_file "$TEACHER_JSONL"; then
    run_with_log "09_teacher_scorer" \
      bash src/scripts/run_teacher_scorer.sh \
        --server_mode "$SERVER_MODE" \
        --venv_path "$VENV_PATH" \
        --candidates_jsonl "$CANDIDATES_JSONL" \
        --features_jsonl "$MERGED_FEATS" \
        --c1_jsonl "$FEATS_C1" \
        --parquet "$FILTERED_PARQUET" \
        --tar_dir "$TAR_DIR" \
        "${image_dir_args[@]}" \
        --output_jsonl "$TEACHER_JSONL" \
        --output_overview_json "$TEACHER_OVERVIEW_JSON" \
        --output_overview_csv "$TEACHER_OVERVIEW_CSV" \
        --qa_out_json "$TEACHER_QA_JSON" \
        --qa_out_csv "$TEACHER_QA_CSV" \
        --viz_out_dir "$TEACHER_VIZ_DIR" \
        --use_real_expensive "$USE_REAL_EXPENSIVE" \
        --cheap_top_m "$CHEAP_TOP_M" \
        --top_k "$TOP_K" \
        --tau_div "$TAU_DIV" \
        --align_model_name "$ALIGN_MODEL_NAME" \
        --align_pretrained "$ALIGN_PRETRAINED" \
        --align_device "$ALIGN_DEVICE" \
        --aesthetic_device "$AESTHETIC_DEVICE" \
        --exp_batch_size "$EXP_BATCH_SIZE" \
        --aesthetic_mlp_path "$AESTHETIC_MLP_PATH" \
        --aesthetic_mlp_url "$AESTHETIC_MLP_URL" \
        --max_images "$MAX_IMAGES" \
        --run_qa "$RUN_QA" \
        --run_viz "$RUN_VIZ" \
        --num_viz "$NUM_VIZ" \
        --target_ar "$TARGET_AR" \
        --decision_filter "$DECISION_FILTER" \
        --multi_gpu "$TEACHER_MULTI_GPU" \
        --gpu_ids "$TEACHER_GPU_IDS" \
        --num_workers "$TEACHER_NUM_WORKERS"
  fi
fi

echo "========================================================"
echo " Done"
echo "========================================================"
echo " filtered parquet : $FILTERED_PARQUET"
if [ "$PRECOMPUTE_MODE" = "unified" ]; then
  echo " precompute raw    : $FEATS_C2C3C5_RAW"
fi
echo " feats c1          : $FEATS_C1"
echo " feats c2/c3e/c5  : $FEATS_C2 | $FEATS_C3_ENRICHED | $FEATS_C5"
echo " merged feats     : $MERGED_FEATS"
echo " candidates       : $CANDIDATES_JSONL"
echo " teacher jsonl    : $TEACHER_JSONL"
echo " teacher overview : $TEACHER_OVERVIEW_JSON"
echo " teacher QA       : $TEACHER_QA_JSON"
echo " teacher viz dir  : $TEACHER_VIZ_DIR"
echo " logs             : $LOG_DIR"
