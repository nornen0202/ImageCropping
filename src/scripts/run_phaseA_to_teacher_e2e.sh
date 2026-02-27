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

# Teacher 산출물 자동 복구/검증(기본 on)
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --run_filter 0 --run_candidates 0 --run_teacher 1 \
  --teacher_auto_repair 1 --teacher_auto_repair_strict 1

# 스모크 테스트(앞 200장)
bash src/scripts/run_phaseA_to_teacher_e2e.sh --max_images 200 --run_tag smoke200

# precompute 컴포넌트 시각화 동시 생성
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --run_component_viz 1 \
  --component_viz_num_samples 120 \
  --run_tag with_comp_viz

# 공개 Teacher proposal 주입(5.5 설치/추론/변환 자동 포함)
bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode 1 \
  --data_dir data/SSTK/10K \
  --run_filter 0 \
  --enable_public_teacher_proposals 1 \
  --public_teacher_setup 1 \
  --public_teacher_download_weights 1 \
  --public_teachers gaic,cacnet,cgs \
  --public_gaic_weight_path weights/public_cropping_teachers/gaic/shufflenet_0.682_0.641_0.607_0.566_0.858_0.825_0.805_0.778_0.850_0.872.pth \
  --run_tag public_seeded

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
--cand_num_workers INT          candidate 생성 멀티프로세스 worker 수 (0=auto, 1=single)
--cand_mp_chunksize INT         candidate 멀티프로세스 map chunksize (default: 64)
--cand_mp_start_method STR      candidate mp 시작 방식(auto|fork|forkserver|spawn)
--run_component_viz 0|1         precompute(C2/C3/C5) 시각화 자동 생성 여부 (default: 0)
--component_viz_num_samples INT precompute 시각화 샘플 수 (default: 120)
--component_viz_out_dir PATH    precompute 시각화 출력 경로 (default: <data_dir>/artifacts/precompute/visualizations/components<suffix>)
--component_viz_image_ids CSV   precompute 시각화 대상 image_id CSV(명시 시 우선)
--component_viz_image_ids_file PATH precompute 시각화 대상 image_id 파일(한 줄 1개)
--teacher_proposals_jsonl CSV   candidate 단계 teacher proposal jsonl(쉼표로 다중 경로)
--enable_public_teacher_proposals 0|1   공개 teacher(5.5) setup+infer+build 자동 수행 (default: 0)
--public_teacher_setup 0|1      공개 teacher setup 수행 여부 (default: 1)
--public_teacher_download_weights 0|1    setup 시 가중치 다운로드 (default: 1)
--public_teachers CSV           추론 teacher 목록 (default: gaic,cacnet,cgs)
--public_teacher_max_images INT 공개 teacher 추론 이미지 수 (-1이면 --max_images 상속, default: -1)
--public_teacher_raw_jsonl PATH 공개 teacher raw 결과 경로 (default: <data_dir>/artifacts/public_teachers/raw/teacher_raw_public<suffix>.jsonl)
--public_teacher_proposals_jsonl PATH    변환된 proposals 경로 (default: <data_dir>/artifacts/public_teachers/proposals/teacher_proposals_public<suffix>.jsonl)
--public_teacher_device DEVICE 공개 teacher 추론 장치 (auto|cuda|cpu, default: auto)
--public_gaic_weight_path PATH GAIC 가중치 경로 (default: shufflenet .pth)
--public_infer_skip_on_oom 0|1 공개 teacher 추론 중 OOM skip (default: 1)
--public_infer_fallback_cpu_on_oom 0|1 GPU OOM 시 CPU 재시도 (default: 1)
--public_infer_fallback_cpu_max_images INT CPU fallback 샘플 수 (default: 3)
--public_infer_skip_if_fallback_failed 0|1 CPU fallback 실패 시 전체 진행 지속 (default: 1)
--public_infer_multi_gpu -1|0|1 공개 teacher 추론 multi-gpu (default: -1, auto)
--public_infer_gpu_ids CSV 공개 teacher multi-gpu 대상 GPU 목록 (예: 0,1)
--public_infer_num_workers INT 공개 teacher shard worker 수 (default: gpu 개수)
--precompute_mode MODE          unified|split (default: unified)
--extract_gpu_ids CSV           precompute에서 사용할 GPU 목록 (예: 0,1,2,3)

--use_real_expensive 0|1        teacher expensive real model 사용 (default: 0)
--run_qa 0|1                    QA report 생성 여부 (default: 1)
--run_viz 0|1                   teacher viz 생성 여부 (default: 1)
--teacher_multi_gpu -1|0|1      -1=auto(use_real_expensive && multi-gpu면 on), default -1
--teacher_gpu_ids CSV           teacher multi-gpu에서 사용할 GPU 목록
--teacher_num_workers INT       teacher multi-gpu shard worker 수
--teacher_auto_repair 0|1       teacher shard 병합/검증 자동 복구 실행 (default: 1)
--teacher_auto_repair_strict 0|1 expected rows 불일치 시 shard promote 금지 (default: 1)
--expensive_eval_top_m INT      expensive stage에서 AR별 평가 상한(0=cheap_top_m 전체)
--exp_preprocess_workers INT    expensive clip preprocess thread 수(0=auto)
--exp_pin_memory 0|1            expensive batch H2D pin_memory 사용 여부 (default: 1)

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
SCRIPT_VERSION="2026-02-27.3"

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
CAND_NUM_WORKERS=0
CAND_MP_CHUNKSIZE=64
CAND_MP_START_METHOD="auto"
RUN_COMPONENT_VIZ=0
COMPONENT_VIZ_NUM_SAMPLES=120
COMPONENT_VIZ_OUT_DIR=""
COMPONENT_VIZ_IMAGE_IDS=""
COMPONENT_VIZ_IMAGE_IDS_FILE=""
TEACHER_PROPOSALS_JSONL=""
TEACHER_NMS_IOU=0.95
TEACHER_MAX_SEEDS_PER_TEACHER=1
TEACHER_PREFER_EXPAND=1
TEACHER_JITTER_SHIFT_FRACS="0.03"
TEACHER_JITTER_SCALES="0.92,1.0,1.08"
ENABLE_PUBLIC_TEACHER_PROPOSALS=0
PUBLIC_TEACHER_SETUP=1
PUBLIC_TEACHER_DOWNLOAD_WEIGHTS=1
PUBLIC_TEACHERS="gaic,cacnet,cgs"
PUBLIC_TEACHER_MAX_IMAGES=-1
PUBLIC_TEACHER_DEVICE="auto"
PUBLIC_TEACHER_ROOT_DIR="third_party/public_cropping_teachers"
PUBLIC_TEACHER_WEIGHTS_DIR="weights/public_cropping_teachers"
PUBLIC_GAIC_WEIGHT_PATH="weights/public_cropping_teachers/gaic/shufflenet_0.682_0.641_0.607_0.566_0.858_0.825_0.805_0.778_0.850_0.872.pth"
PUBLIC_TEACHER_RAW_JSONL=""
PUBLIC_TEACHER_PROPOSALS_JSONL=""
PUBLIC_INFER_SKIP_ON_OOM=1
PUBLIC_INFER_FALLBACK_CPU_ON_OOM=1
PUBLIC_INFER_FALLBACK_CPU_MAX_IMAGES=3
PUBLIC_INFER_SKIP_IF_FALLBACK_FAILED=1
PUBLIC_INFER_MULTI_GPU=-1
PUBLIC_INFER_GPU_IDS=""
PUBLIC_INFER_NUM_WORKERS=""

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
EXPENSIVE_EVAL_TOP_M=0
EXP_PREPROCESS_WORKERS=0
EXP_PIN_MEMORY=1
AESTHETIC_MLP_PATH="weights/improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth"
AESTHETIC_MLP_URL="https://raw.githubusercontent.com/christophschuhmann/improved-aesthetic-predictor/main/sac+logos+ava1-l14-linearMSE.pth"
TEACHER_MULTI_GPU=-1
TEACHER_GPU_IDS=""
TEACHER_NUM_WORKERS=""
TEACHER_AUTO_REPAIR=1
TEACHER_AUTO_REPAIR_STRICT=1

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
    --run_component_viz) RUN_COMPONENT_VIZ="$2"; shift 2 ;;
    --component_viz_num_samples) COMPONENT_VIZ_NUM_SAMPLES="$2"; shift 2 ;;
    --component_viz_out_dir) COMPONENT_VIZ_OUT_DIR="$2"; shift 2 ;;
    --component_viz_image_ids) COMPONENT_VIZ_IMAGE_IDS="$2"; shift 2 ;;
    --component_viz_image_ids_file) COMPONENT_VIZ_IMAGE_IDS_FILE="$2"; shift 2 ;;
    --use_actual_image_size) USE_ACTUAL_IMAGE_SIZE="$2"; shift 2 ;;
    --strict_actual_size) STRICT_ACTUAL_SIZE="$2"; shift 2 ;;
    --actual_size_cache_json) ACTUAL_SIZE_CACHE_JSON="$2"; shift 2 ;;
    --max_images) MAX_IMAGES="$2"; shift 2 ;;
    --cand_num_workers) CAND_NUM_WORKERS="$2"; shift 2 ;;
    --cand_mp_chunksize) CAND_MP_CHUNKSIZE="$2"; shift 2 ;;
    --cand_mp_start_method) CAND_MP_START_METHOD="$2"; shift 2 ;;
    --teacher_proposals_jsonl) TEACHER_PROPOSALS_JSONL="$2"; shift 2 ;;
    --teacher_nms_iou) TEACHER_NMS_IOU="$2"; shift 2 ;;
    --teacher_max_seeds_per_teacher) TEACHER_MAX_SEEDS_PER_TEACHER="$2"; shift 2 ;;
    --teacher_prefer_expand) TEACHER_PREFER_EXPAND="$2"; shift 2 ;;
    --teacher_jitter_shift_fracs) TEACHER_JITTER_SHIFT_FRACS="$2"; shift 2 ;;
    --teacher_jitter_scales) TEACHER_JITTER_SCALES="$2"; shift 2 ;;
    --enable_public_teacher_proposals) ENABLE_PUBLIC_TEACHER_PROPOSALS="$2"; shift 2 ;;
    --public_teacher_setup) PUBLIC_TEACHER_SETUP="$2"; shift 2 ;;
    --public_teacher_download_weights) PUBLIC_TEACHER_DOWNLOAD_WEIGHTS="$2"; shift 2 ;;
    --public_teachers) PUBLIC_TEACHERS="$2"; shift 2 ;;
    --public_teacher_max_images) PUBLIC_TEACHER_MAX_IMAGES="$2"; shift 2 ;;
    --public_teacher_device) PUBLIC_TEACHER_DEVICE="$2"; shift 2 ;;
    --public_teacher_root_dir) PUBLIC_TEACHER_ROOT_DIR="$2"; shift 2 ;;
    --public_teacher_weights_dir) PUBLIC_TEACHER_WEIGHTS_DIR="$2"; shift 2 ;;
    --public_gaic_weight_path) PUBLIC_GAIC_WEIGHT_PATH="$2"; shift 2 ;;
    --public_teacher_raw_jsonl) PUBLIC_TEACHER_RAW_JSONL="$2"; shift 2 ;;
    --public_teacher_proposals_jsonl) PUBLIC_TEACHER_PROPOSALS_JSONL="$2"; shift 2 ;;
    --public_infer_skip_on_oom) PUBLIC_INFER_SKIP_ON_OOM="$2"; shift 2 ;;
    --public_infer_fallback_cpu_on_oom) PUBLIC_INFER_FALLBACK_CPU_ON_OOM="$2"; shift 2 ;;
    --public_infer_fallback_cpu_max_images) PUBLIC_INFER_FALLBACK_CPU_MAX_IMAGES="$2"; shift 2 ;;
    --public_infer_skip_if_fallback_failed) PUBLIC_INFER_SKIP_IF_FALLBACK_FAILED="$2"; shift 2 ;;
    --public_infer_multi_gpu) PUBLIC_INFER_MULTI_GPU="$2"; shift 2 ;;
    --public_infer_gpu_ids) PUBLIC_INFER_GPU_IDS="$2"; shift 2 ;;
    --public_infer_num_workers) PUBLIC_INFER_NUM_WORKERS="$2"; shift 2 ;;

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
    --expensive_eval_top_m) EXPENSIVE_EVAL_TOP_M="$2"; shift 2 ;;
    --exp_preprocess_workers) EXP_PREPROCESS_WORKERS="$2"; shift 2 ;;
    --exp_pin_memory) EXP_PIN_MEMORY="$2"; shift 2 ;;
    --aesthetic_mlp_path) AESTHETIC_MLP_PATH="$2"; shift 2 ;;
    --aesthetic_mlp_url) AESTHETIC_MLP_URL="$2"; shift 2 ;;
    --teacher_multi_gpu) TEACHER_MULTI_GPU="$2"; shift 2 ;;
    --teacher_gpu_ids) TEACHER_GPU_IDS="$2"; shift 2 ;;
    --teacher_num_workers) TEACHER_NUM_WORKERS="$2"; shift 2 ;;
    --teacher_auto_repair) TEACHER_AUTO_REPAIR="$2"; shift 2 ;;
    --teacher_auto_repair_strict) TEACHER_AUTO_REPAIR_STRICT="$2"; shift 2 ;;

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
ARTIFACTS_DIR="${DATA_DIR}/artifacts"
PRECOMPUTE_DIR="${ARTIFACTS_DIR}/precompute"
PRECOMPUTE_VIZ_BASE_DIR="${PRECOMPUTE_DIR}/visualizations"
CANDIDATES_DIR="${ARTIFACTS_DIR}/candidates"
PUBLIC_TEACHERS_DIR="${ARTIFACTS_DIR}/public_teachers"
PUBLIC_RAW_DIR="${PUBLIC_TEACHERS_DIR}/raw"
PUBLIC_PROPOSALS_DIR="${PUBLIC_TEACHERS_DIR}/proposals"
TEACHER_DIR="${ARTIFACTS_DIR}/teacher"
TEACHER_SCORES_DIR="${TEACHER_DIR}/scores"
TEACHER_OVERVIEW_DIR="${TEACHER_DIR}/overview"
TEACHER_QA_DIR="${TEACHER_DIR}/qa"
TEACHER_VIZ_BASE_DIR="${TEACHER_DIR}/visualizations"
CACHE_DIR="${DATA_DIR}/cache"

FEATS_C1="${PRECOMPUTE_DIR}/feats_c1.jsonl"
FEATS_C2="${PRECOMPUTE_DIR}/feats_c2.jsonl"
FEATS_C3="${PRECOMPUTE_DIR}/feats_c3_v2_strict.jsonl"
FEATS_C3_ENRICHED="${PRECOMPUTE_DIR}/feats_c3_v2_strict_enriched.jsonl"
FEATS_C5="${PRECOMPUTE_DIR}/feats_c5.jsonl"
FEATS_C2C3C5_RAW="${PRECOMPUTE_DIR}/feats_c2c3c5_v2_strict_raw.jsonl"
MERGED_FEATS="${PRECOMPUTE_DIR}/feats_c2c3c5_v2_strict_enriched.jsonl"

CANDIDATES_JSONL="${CANDIDATES_DIR}/candidates_ar${SUFFIX}.jsonl"
TEACHER_JSONL="${TEACHER_SCORES_DIR}/teacher_scores_ar${SUFFIX}.jsonl"
TEACHER_OVERVIEW_JSON="${TEACHER_OVERVIEW_DIR}/teacher_scores_overview${SUFFIX}.json"
TEACHER_OVERVIEW_CSV="${TEACHER_OVERVIEW_DIR}/teacher_scores_overview_by_ar${SUFFIX}.csv"
TEACHER_QA_JSON="${TEACHER_QA_DIR}/teacher_scores_qa_report${SUFFIX}.json"
TEACHER_QA_CSV="${TEACHER_QA_DIR}/teacher_scores_qa_report_by_ar${SUFFIX}.csv"
TEACHER_VIZ_DIR="${TEACHER_VIZ_BASE_DIR}/teacher_scorer${SUFFIX}"

if [ -z "$PUBLIC_TEACHER_RAW_JSONL" ]; then
  PUBLIC_TEACHER_RAW_JSONL="${PUBLIC_RAW_DIR}/teacher_raw_public${SUFFIX}.jsonl"
fi
if [ -z "$PUBLIC_TEACHER_PROPOSALS_JSONL" ]; then
  PUBLIC_TEACHER_PROPOSALS_JSONL="${PUBLIC_PROPOSALS_DIR}/teacher_proposals_public${SUFFIX}.jsonl"
fi
if [ "$PUBLIC_TEACHER_MAX_IMAGES" -lt 0 ]; then
  PUBLIC_TEACHER_MAX_IMAGES="$MAX_IMAGES"
fi

if [ -z "$ACTUAL_SIZE_CACHE_JSON" ]; then
  ACTUAL_SIZE_CACHE_JSON="${CACHE_DIR}/actual_image_size_map.json"
fi

if [ -z "$COMPONENT_VIZ_OUT_DIR" ]; then
  COMPONENT_VIZ_OUT_DIR="${PRECOMPUTE_VIZ_BASE_DIR}/components${SUFFIX}"
fi

mkdir -p \
  "$PRECOMPUTE_DIR" \
  "$PRECOMPUTE_VIZ_BASE_DIR" \
  "$CANDIDATES_DIR" \
  "$PUBLIC_RAW_DIR" \
  "$PUBLIC_PROPOSALS_DIR" \
  "$TEACHER_SCORES_DIR" \
  "$TEACHER_OVERVIEW_DIR" \
  "$TEACHER_QA_DIR" \
  "$TEACHER_VIZ_BASE_DIR" \
  "$CACHE_DIR"

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

latest_glob_match() {
  local pattern="$1"
  local found
  found=$(ls -1t $pattern 2>/dev/null | head -n 1 || true)
  echo "$found"
}

promote_file_if_missing() {
  local dst="$1"
  shift || true
  if [ -f "$dst" ]; then
    return 0
  fi
  local src
  for src in "$@"; do
    if [ -z "$src" ] || [ ! -f "$src" ]; then
      continue
    fi
    mkdir -p "$(dirname "$dst")"
    cp -f "$src" "$dst"
    echo "[promote] restored $(basename "$dst") from $src -> $dst"
    return 0
  done
  return 1
}

promote_from_legacy_or_cleanup() {
  local dst="$1"
  local base
  base="$(basename "$dst")"
  local in_root="${DATA_DIR}/${base}"
  local in_temp="${DATA_DIR}/Temp/${base}"
  local in_cleanup
  in_cleanup="$(latest_glob_match "${DATA_DIR}/Temp/cleanup_*/${base}")"
  promote_file_if_missing "$dst" "$in_root" "$in_temp" "$in_cleanup" || true
  return 0
}

csv_append_unique() {
  local csv="$1"
  local item="$2"
  if [ -z "$item" ]; then
    echo "$csv"
    return
  fi
  if [ -z "$csv" ]; then
    echo "$item"
    return
  fi
  local IFS_OLD="$IFS"
  local found=0
  IFS=',' read -r -a _arr <<< "$csv"
  IFS="$IFS_OLD"
  local x
  for x in "${_arr[@]}"; do
    if [ "$x" = "$item" ]; then
      found=1
      break
    fi
  done
  if [ "$found" -eq 1 ]; then
    echo "$csv"
  else
    echo "${csv},${item}"
  fi
}

filter_extra_args=()
if [ "$EXPORT_CURATED_IMAGES" -eq 1 ]; then
  filter_extra_args+=("$CURATED_IMAGE_DIR" "$CURATED_IMAGE_SKIP_EXISTING")
fi

# Promote previously generated artifacts back to canonical paths.
# This prevents hard failures when old cleanup/layout reorg moved files to Temp.
promote_from_legacy_or_cleanup "$FEATS_C1"
promote_from_legacy_or_cleanup "$FEATS_C2"
promote_from_legacy_or_cleanup "$FEATS_C3"
promote_from_legacy_or_cleanup "$FEATS_C3_ENRICHED"
promote_from_legacy_or_cleanup "$FEATS_C5"
promote_from_legacy_or_cleanup "$FEATS_C2C3C5_RAW"
promote_from_legacy_or_cleanup "$MERGED_FEATS"
promote_from_legacy_or_cleanup "$CANDIDATES_JSONL"
promote_from_legacy_or_cleanup "$PUBLIC_TEACHER_RAW_JSONL"
promote_from_legacy_or_cleanup "$PUBLIC_TEACHER_PROPOSALS_JSONL"

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
echo " script_version      : $SCRIPT_VERSION"
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
echo " run_component_viz   : $RUN_COMPONENT_VIZ (out=$COMPONENT_VIZ_OUT_DIR, num_samples=$COMPONENT_VIZ_NUM_SAMPLES)"
echo " public proposals    : enable=$ENABLE_PUBLIC_TEACHER_PROPOSALS setup=$PUBLIC_TEACHER_SETUP teachers=$PUBLIC_TEACHERS max_images=$PUBLIC_TEACHER_MAX_IMAGES"
echo " public raw/proposal : $PUBLIC_TEACHER_RAW_JSONL | $PUBLIC_TEACHER_PROPOSALS_JSONL"
echo " public infer multi  : multi_gpu=$PUBLIC_INFER_MULTI_GPU gpu_ids=${PUBLIC_INFER_GPU_IDS:-auto} workers=${PUBLIC_INFER_NUM_WORKERS:-auto}"
echo " run_candidates      : $RUN_CANDIDATES"
echo " candidate_mp       : workers=$CAND_NUM_WORKERS chunksize=$CAND_MP_CHUNKSIZE start=$CAND_MP_START_METHOD"
echo " teacher proposals   : ${TEACHER_PROPOSALS_JSONL:-<none>}"
echo " run_teacher         : $RUN_TEACHER (real_expensive=$USE_REAL_EXPENSIVE)"
echo " teacher_multi_gpu   : $TEACHER_MULTI_GPU (gpu_ids=${TEACHER_GPU_IDS:-auto}, workers=${TEACHER_NUM_WORKERS:-auto})"
echo " teacher_auto_repair : $TEACHER_AUTO_REPAIR (strict=$TEACHER_AUTO_REPAIR_STRICT)"
echo " teacher_accel       : exp_batch=$EXP_BATCH_SIZE exp_eval_top_m=$EXPENSIVE_EVAL_TOP_M preprocess_workers=$EXP_PREPROCESS_WORKERS pin_memory=$EXP_PIN_MEMORY"
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
# 2.5) Precompute Visualization (optional)
# ------------------------------------------------------------------------------
if [ "$RUN_COMPONENT_VIZ" -eq 1 ]; then
  if [ ! -f "$MERGED_FEATS" ]; then
    echo "[error] component visualization requires merged features jsonl: $MERGED_FEATS"
    exit 1
  fi
  if [ "$SKIP_EXISTING" -eq 1 ] && [ -f "${COMPONENT_VIZ_OUT_DIR}/viz_overview.json" ]; then
    echo "[skip] exists: ${COMPONENT_VIZ_OUT_DIR}/viz_overview.json"
  else
    comp_viz_args=(
      --merged_jsonl "$MERGED_FEATS"
      --num_samples "$COMPONENT_VIZ_NUM_SAMPLES"
      --draw_c2 1
      --draw_c3 1
      --draw_c5 1
      --draw_combined 1
      --server_mode "$SERVER_MODE"
      --venv_path "$VENV_PATH"
      --oom_cpu_fallback 1
      --oom_fallback_num_samples 3
      --skip_on_oom_fail 1
    )
    if [ -n "$EFFECTIVE_IMAGE_DIR" ]; then
      comp_viz_args+=(--image_dir "$EFFECTIVE_IMAGE_DIR")
    fi
    if [ -n "$COMPONENT_VIZ_IMAGE_IDS" ]; then
      comp_viz_args+=(--image_ids "$COMPONENT_VIZ_IMAGE_IDS")
    fi
    if [ -n "$COMPONENT_VIZ_IMAGE_IDS_FILE" ]; then
      comp_viz_args+=(--image_ids_file "$COMPONENT_VIZ_IMAGE_IDS_FILE")
    fi
    run_with_log "07b_visualize_components" \
      bash src/scripts/run_visualize_components.sh \
        "$FILTERED_PARQUET" \
        "$TAR_DIR" \
        "$COMPONENT_VIZ_OUT_DIR" \
        "${comp_viz_args[@]}"
  fi
fi

# ------------------------------------------------------------------------------
# 3) Public Teacher Proposals (v1.9 8.2.0a, optional)
# ------------------------------------------------------------------------------
if [ "$ENABLE_PUBLIC_TEACHER_PROPOSALS" -eq 1 ]; then
  need_public_infer=1
  if [ "$SKIP_EXISTING" -eq 1 ] && [ -f "$PUBLIC_TEACHER_RAW_JSONL" ]; then
    need_public_infer=0
    echo "[skip] exists: $PUBLIC_TEACHER_RAW_JSONL"
  fi

  need_public_build=1
  if [ "$SKIP_EXISTING" -eq 1 ] && [ -f "$PUBLIC_TEACHER_PROPOSALS_JSONL" ]; then
    need_public_build=0
    echo "[skip] exists: $PUBLIC_TEACHER_PROPOSALS_JSONL"
  fi

  if [ "$need_public_infer" -eq 1 ] || [ "$need_public_build" -eq 1 ]; then
    if [ "$PUBLIC_TEACHER_SETUP" -eq 1 ]; then
      run_with_log "08a_public_teacher_setup" \
        bash src/scripts/run_setup_public_cropping_teachers.sh \
          --teacher_root_dir "$PUBLIC_TEACHER_ROOT_DIR" \
          --weights_dir "$PUBLIC_TEACHER_WEIGHTS_DIR" \
          --download_weights "$PUBLIC_TEACHER_DOWNLOAD_WEIGHTS"
    fi
  fi

  if [ "$need_public_infer" -eq 1 ]; then
    public_teacher_args=()
    IFS_OLD="$IFS"
    IFS=',' read -r -a public_teachers_arr <<< "$PUBLIC_TEACHERS"
    IFS="$IFS_OLD"
    filtered_public_teachers=()
    for t in "${public_teachers_arr[@]}"; do
      if [ -n "$t" ]; then
        filtered_public_teachers+=("$t")
      fi
    done
    if [ "${#filtered_public_teachers[@]}" -gt 0 ]; then
      public_teacher_args+=(--teachers)
      for t in "${filtered_public_teachers[@]}"; do
        public_teacher_args+=("$t")
      done
    fi
    if [ -n "$PUBLIC_GAIC_WEIGHT_PATH" ]; then
      public_teacher_args+=(--gaic_weight_path "$PUBLIC_GAIC_WEIGHT_PATH")
    fi
    public_teacher_args+=(
      --teacher_root_dir "$PUBLIC_TEACHER_ROOT_DIR"
      --weights_dir "$PUBLIC_TEACHER_WEIGHTS_DIR"
      --device "$PUBLIC_TEACHER_DEVICE"
      --max_images "$PUBLIC_TEACHER_MAX_IMAGES"
      --run_setup 0
      --skip_on_oom "$PUBLIC_INFER_SKIP_ON_OOM"
      --prefer_curated_images "$PREFER_CURATED_IMAGES"
      --curated_image_dir "$CURATED_IMAGE_DIR"
      --fallback_cpu_on_oom "$PUBLIC_INFER_FALLBACK_CPU_ON_OOM"
      --fallback_cpu_max_images "$PUBLIC_INFER_FALLBACK_CPU_MAX_IMAGES"
      --skip_if_fallback_failed "$PUBLIC_INFER_SKIP_IF_FALLBACK_FAILED"
      --multi_gpu "$PUBLIC_INFER_MULTI_GPU"
    )
    if [ -n "$PUBLIC_INFER_GPU_IDS" ]; then
      public_teacher_args+=(--gpu_ids "$PUBLIC_INFER_GPU_IDS")
    fi
    if [ -n "$PUBLIC_INFER_NUM_WORKERS" ]; then
      public_teacher_args+=(--num_workers "$PUBLIC_INFER_NUM_WORKERS")
    fi
    run_with_log "08b_public_teacher_infer" \
      bash src/scripts/run_infer_public_cropping_teachers.sh \
        "$FILTERED_PARQUET" \
        "$TAR_DIR" \
        "$PUBLIC_TEACHER_RAW_JSONL" \
        "${public_teacher_args[@]}"
  fi

  if [ ! -f "$PUBLIC_TEACHER_RAW_JSONL" ]; then
    echo "[error] public teacher raw jsonl not found: $PUBLIC_TEACHER_RAW_JSONL"
    exit 1
  fi

  if [ "$need_public_build" -eq 1 ]; then
    run_with_log "08c_public_teacher_build_proposals" \
      bash src/scripts/run_build_teacher_proposals.sh \
        "$PUBLIC_TEACHER_PROPOSALS_JSONL" \
        "$PUBLIC_TEACHER_RAW_JSONL"
  fi

  if [ ! -f "$PUBLIC_TEACHER_PROPOSALS_JSONL" ]; then
    echo "[error] public teacher proposals jsonl not found: $PUBLIC_TEACHER_PROPOSALS_JSONL"
    exit 1
  fi

  TEACHER_PROPOSALS_JSONL="$(csv_append_unique "$TEACHER_PROPOSALS_JSONL" "$PUBLIC_TEACHER_PROPOSALS_JSONL")"
  echo "[info] teacher proposal injection paths: $TEACHER_PROPOSALS_JSONL"
fi

# ------------------------------------------------------------------------------
# 4) Candidate Generator
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
    run_with_log "09_generate_candidates" \
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
        --num_workers "$CAND_NUM_WORKERS" \
        --mp_chunksize "$CAND_MP_CHUNKSIZE" \
        --mp_start_method "$CAND_MP_START_METHOD" \
        "${cand_extra_args[@]}"
  fi
fi

# ------------------------------------------------------------------------------
# 5) Teacher Scorer
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
    run_with_log "10_teacher_scorer" \
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
        --expensive_eval_top_m "$EXPENSIVE_EVAL_TOP_M" \
        --exp_preprocess_workers "$EXP_PREPROCESS_WORKERS" \
        --exp_pin_memory "$EXP_PIN_MEMORY" \
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

  if [ "$TEACHER_AUTO_REPAIR" -eq 1 ]; then
    run_with_log "10b_teacher_repair_outputs" \
      python src/scripts/repair_teacher_outputs.py \
        --teacher_scores_jsonl "$TEACHER_JSONL" \
        --overview_json "$TEACHER_OVERVIEW_JSON" \
        --overview_csv "$TEACHER_OVERVIEW_CSV" \
        --qa_json "$TEACHER_QA_JSON" \
        --qa_csv "$TEACHER_QA_CSV" \
        --candidates_jsonl "$CANDIDATES_JSONL" \
        --max_images "$MAX_IMAGES" \
        --prefer_real_expensive "$USE_REAL_EXPENSIVE" \
        --strict_expected_match "$TEACHER_AUTO_REPAIR_STRICT" \
        --verbose 1
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
if [ "$ENABLE_PUBLIC_TEACHER_PROPOSALS" -eq 1 ]; then
  echo " public raw       : $PUBLIC_TEACHER_RAW_JSONL"
  echo " public proposals : $PUBLIC_TEACHER_PROPOSALS_JSONL"
fi
if [ "$RUN_COMPONENT_VIZ" -eq 1 ]; then
  echo " component viz    : $COMPONENT_VIZ_OUT_DIR"
fi
echo " candidates       : $CANDIDATES_JSONL"
echo " teacher jsonl    : $TEACHER_JSONL"
echo " teacher overview : $TEACHER_OVERVIEW_JSON"
echo " teacher QA       : $TEACHER_QA_JSON"
echo " teacher viz dir  : $TEACHER_VIZ_DIR"
echo " logs             : $LOG_DIR"
