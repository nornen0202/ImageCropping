#!/bin/bash
# ==============================================================================
# run_gaic_to_teacher_e2e.sh
# GAIC image-only end-to-end wrapper:
#   prepare flat curated pool + pseudo parquet
#   -> reuse SSTK e2e runner with GAIC-safe defaults
#   -> validate outputs
# ------------------------------------------------------------------------------
# Key GAIC policies:
#   - all images are treated as curated_pool (no Phase-A filtering/sampling)
#   - metadata-dependent C1/real-expensive path is disabled by default
#   - Stage-10 labels default to heuristic backend for offline robustness
# ==============================================================================

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$PROJECT_ROOT"

DATA_DIR="data/GAIC/All"
BUCKET="gaic_all"
IMAGE_ROOT="data/Publics/GAIC/images"
FLAT_IMAGE_DIR=""
RUN_TAG="gaic_e2e"
SERVER_MODE=0
VENV_PATH="/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate"

PSEUDO_TAR_CHUNK_SIZE=256
LINK_MODE="symlink"
PREPARE_MAX_IMAGES=0
SKIP_EXISTING=1
SKIP_EXISTING_LINKS=1
PREPARE_ONLY=0
SKIP_VALIDATION=0
RUN_VLM_TEACHER=1
VLM_BACKEND="heuristic"
VLM_FALLBACK_BACKEND="none"
RUN_TRAINING_LABELS=1
SAFE_LEFTOVER_POLICY="ignore"
RUN_DETAILED_REPORT=0
RUN_C6=0
USE_REAL_EXPENSIVE=0
RUN_C1=0
PRECOMPUTE_MODE="unified"
GAIC_REFERENCE_JSON=""
GAIC_GENERATE_CAPTIONS=-1
GAIC_CAPTION_PRESET=""
GAIC_CAPTION_BACKEND=""
GAIC_CAPTION_MODEL_ID=""
GAIC_CAPTION_DEVICE="auto"
GAIC_CAPTION_DTYPE="auto"
GAIC_CAPTION_BATCH_SIZE=4
GAIC_CAPTION_MAX_NEW_TOKENS=64
GAIC_CAPTION_NUM_BEAMS=3
GAIC_CAPTION_PROMPT=""
GAIC_CAPTION_JSONL=""
GAIC_CAPTION_SUMMARY_JSON=""
GAIC_CAPTION_SKIP_EXISTING=1

PASSTHROUGH_ARGS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --data_dir) DATA_DIR="$2"; shift 2 ;;
    --bucket) BUCKET="$2"; shift 2 ;;
    --image_root) IMAGE_ROOT="$2"; shift 2 ;;
    --flat_image_dir) FLAT_IMAGE_DIR="$2"; shift 2 ;;
    --run_tag) RUN_TAG="$2"; shift 2 ;;
    --server_mode) SERVER_MODE="$2"; shift 2 ;;
    --venv_path) VENV_PATH="$2"; shift 2 ;;
    --pseudo_tar_chunk_size) PSEUDO_TAR_CHUNK_SIZE="$2"; shift 2 ;;
    --link_mode) LINK_MODE="$2"; shift 2 ;;
    --prepare_max_images) PREPARE_MAX_IMAGES="$2"; shift 2 ;;
    --skip_existing) SKIP_EXISTING="$2"; shift 2 ;;
    --skip_existing_links) SKIP_EXISTING_LINKS="$2"; shift 2 ;;
    --prepare_only) PREPARE_ONLY="$2"; shift 2 ;;
    --skip_validation) SKIP_VALIDATION="$2"; shift 2 ;;
    --run_vlm_teacher) RUN_VLM_TEACHER="$2"; shift 2 ;;
    --vlm_backend) VLM_BACKEND="$2"; shift 2 ;;
    --vlm_fallback_backend) VLM_FALLBACK_BACKEND="$2"; shift 2 ;;
    --run_training_labels) RUN_TRAINING_LABELS="$2"; shift 2 ;;
    --safe_leftover_policy) SAFE_LEFTOVER_POLICY="$2"; shift 2 ;;
    --run_detailed_report) RUN_DETAILED_REPORT="$2"; shift 2 ;;
    --run_c6) RUN_C6="$2"; shift 2 ;;
    --use_real_expensive) USE_REAL_EXPENSIVE="$2"; shift 2 ;;
    --run_c1) RUN_C1="$2"; shift 2 ;;
    --precompute_mode) PRECOMPUTE_MODE="$2"; shift 2 ;;
    --gaic_reference_json) GAIC_REFERENCE_JSON="$2"; shift 2 ;;
    --gaic_generate_captions) GAIC_GENERATE_CAPTIONS="$2"; shift 2 ;;
    --gaic_caption_preset) GAIC_CAPTION_PRESET="$2"; shift 2 ;;
    --gaic_caption_backend) GAIC_CAPTION_BACKEND="$2"; shift 2 ;;
    --gaic_caption_model_id) GAIC_CAPTION_MODEL_ID="$2"; shift 2 ;;
    --gaic_caption_device) GAIC_CAPTION_DEVICE="$2"; shift 2 ;;
    --gaic_caption_dtype) GAIC_CAPTION_DTYPE="$2"; shift 2 ;;
    --gaic_caption_batch_size) GAIC_CAPTION_BATCH_SIZE="$2"; shift 2 ;;
    --gaic_caption_max_new_tokens) GAIC_CAPTION_MAX_NEW_TOKENS="$2"; shift 2 ;;
    --gaic_caption_num_beams) GAIC_CAPTION_NUM_BEAMS="$2"; shift 2 ;;
    --gaic_caption_prompt) GAIC_CAPTION_PROMPT="$2"; shift 2 ;;
    --gaic_caption_jsonl) GAIC_CAPTION_JSONL="$2"; shift 2 ;;
    --gaic_caption_summary_json) GAIC_CAPTION_SUMMARY_JSON="$2"; shift 2 ;;
    --gaic_caption_skip_existing) GAIC_CAPTION_SKIP_EXISTING="$2"; shift 2 ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      if [ "$#" -ge 2 ] && [[ "${2:-}" != --* ]]; then
        PASSTHROUGH_ARGS+=("$2")
        shift 2
      else
        shift 1
      fi
      ;;
  esac
done

mkdir -p "$DATA_DIR"
if [ -z "$FLAT_IMAGE_DIR" ]; then
  FLAT_IMAGE_DIR="${DATA_DIR}/images"
fi
if [ -z "$GAIC_REFERENCE_JSON" ]; then
  GAIC_REFERENCE_JSON="${DATA_DIR}/gaic_reference_available.json"
fi
if [ "$USE_REAL_EXPENSIVE" -eq 1 ] && [ "$RUN_C1" -ne 1 ]; then
  RUN_C1=1
fi
if [ "$GAIC_GENERATE_CAPTIONS" -lt 0 ]; then
  if [ "$RUN_C1" -eq 1 ]; then
    GAIC_GENERATE_CAPTIONS=1
  else
    GAIC_GENERATE_CAPTIONS=0
  fi
fi

if [ -n "$RUN_TAG" ]; then
  SUFFIX="_${RUN_TAG}"
  TRAINING_LABELS_SUBDIR="$RUN_TAG"
else
  SUFFIX=""
  TRAINING_LABELS_SUBDIR="latest"
fi

PREP_SUMMARY_JSON="${DATA_DIR}/gaic_prepare_summary.json"
FILTERED_PARQUET="${DATA_DIR}/filtered_${BUCKET}.parquet"
ARTIFACTS_DIR="${DATA_DIR}/artifacts"
PRECOMPUTE_DIR="${ARTIFACTS_DIR}/precompute"
CANDIDATES_DIR="${ARTIFACTS_DIR}/candidates"
TEACHER_DIR="${ARTIFACTS_DIR}/teacher"
VLM_DIR="${ARTIFACTS_DIR}/vlm_teacher"
TRAINING_DIR="${ARTIFACTS_DIR}/training_labels/${TRAINING_LABELS_SUBDIR}"
VALIDATION_DIR="${ARTIFACTS_DIR}/validation"
METADATA_DIR="${ARTIFACTS_DIR}/metadata"
VALIDATION_SUMMARY_JSON="${VALIDATION_DIR}/gaic_e2e_validation${SUFFIX}.json"

FEATS_RAW_JSONL="${PRECOMPUTE_DIR}/feats_c2c3c5_v2_strict_raw.jsonl"
ROUTED_JSONL="${PRECOMPUTE_DIR}/feats_c2c3c5_v2_strict_enriched_routed.jsonl"
CANDIDATES_JSONL="${CANDIDATES_DIR}/candidates_ar${SUFFIX}.jsonl"
TEACHER_JSONL="${TEACHER_DIR}/scores/teacher_scores_ar${SUFFIX}.jsonl"
VLM_LABELS_JSONL="${VLM_DIR}/labels/crop_label_v1${SUFFIX}.jsonl"
VLM_META_JSONL="${VLM_DIR}/meta/meta_norm_v1${SUFFIX}.jsonl"
VLM_SUMMARY_JSON="${VLM_DIR}/summary/vlm_teacher_summary${SUFFIX}.json"
TRAINING_VALIDATION_JSON="${TRAINING_DIR}/validation_summary.json"
TRAINING_GAIC_LIKE_SUMMARY_JSON="${TRAINING_DIR}/coco/gaic_like_conversion_summary.json"
if [ -z "$GAIC_CAPTION_JSONL" ]; then
  GAIC_CAPTION_JSONL="${METADATA_DIR}/gaic_captions${SUFFIX}.jsonl"
fi
if [ -z "$GAIC_CAPTION_SUMMARY_JSON" ]; then
  GAIC_CAPTION_SUMMARY_JSON="${METADATA_DIR}/gaic_captions${SUFFIX}_summary.json"
fi
if [ -z "$GAIC_CAPTION_PRESET" ]; then
  if [ "$SERVER_MODE" -eq 1 ]; then
    GAIC_CAPTION_PRESET="server_quality"
  else
    GAIC_CAPTION_PRESET="local_efficient"
  fi
fi

if [ -f "$VENV_PATH" ]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH"
  echo "[info] activated venv: $VENV_PATH"
elif [ "$SERVER_MODE" -ne 1 ]; then
  echo "[warn] venv not found: $VENV_PATH (using current python)"
fi

echo "========================================================"
echo " GAIC e2e wrapper"
echo "========================================================"
echo " data_dir              : $DATA_DIR"
echo " image_root            : $IMAGE_ROOT"
echo " flat_image_dir        : $FLAT_IMAGE_DIR"
echo " bucket                : $BUCKET"
echo " run_tag               : $RUN_TAG"
echo " prepare_max_images    : $PREPARE_MAX_IMAGES"
echo " pseudo_tar_chunk_size : $PSEUDO_TAR_CHUNK_SIZE"
echo " link_mode             : $LINK_MODE"
echo " skip_existing         : $SKIP_EXISTING"
echo " run_c1                : $RUN_C1"
echo " use_real_expensive    : $USE_REAL_EXPENSIVE"
echo " gaic_generate_caps    : $GAIC_GENERATE_CAPTIONS (preset=$GAIC_CAPTION_PRESET backend=${GAIC_CAPTION_BACKEND:-auto})"
echo " run_c6                : $RUN_C6"
echo " run_vlm_teacher       : $RUN_VLM_TEACHER (backend=$VLM_BACKEND fallback=$VLM_FALLBACK_BACKEND)"
echo " run_training_labels   : $RUN_TRAINING_LABELS (policy=$SAFE_LEFTOVER_POLICY)"
echo " run_detailed_report   : $RUN_DETAILED_REPORT"
echo " gaic_reference_json   : $GAIC_REFERENCE_JSON"
echo "========================================================"

if [ "$GAIC_GENERATE_CAPTIONS" -eq 1 ]; then
  mkdir -p "$METADATA_DIR"
  CAPTION_ARGS=(
    --image_dir "$IMAGE_ROOT"
    --output_jsonl "$GAIC_CAPTION_JSONL"
    --summary_json "$GAIC_CAPTION_SUMMARY_JSON"
    --preset "$GAIC_CAPTION_PRESET"
    --device "$GAIC_CAPTION_DEVICE"
    --dtype "$GAIC_CAPTION_DTYPE"
    --batch_size "$GAIC_CAPTION_BATCH_SIZE"
    --max_new_tokens "$GAIC_CAPTION_MAX_NEW_TOKENS"
    --num_beams "$GAIC_CAPTION_NUM_BEAMS"
    --skip_existing "$GAIC_CAPTION_SKIP_EXISTING"
    --max_images "$PREPARE_MAX_IMAGES"
  )
  if [ -n "$GAIC_CAPTION_BACKEND" ]; then
    CAPTION_ARGS+=(--backend "$GAIC_CAPTION_BACKEND")
  fi
  if [ -n "$GAIC_CAPTION_MODEL_ID" ]; then
    CAPTION_ARGS+=(--model_id "$GAIC_CAPTION_MODEL_ID")
  fi
  if [ -n "$GAIC_CAPTION_PROMPT" ]; then
    CAPTION_ARGS+=(--caption_prompt "$GAIC_CAPTION_PROMPT")
  fi
  python3 src/scripts/generate_gaic_captions.py "${CAPTION_ARGS[@]}"
fi

PREPARE_CAPTION_ARGS=()
if [ -f "$GAIC_CAPTION_JSONL" ]; then
  PREPARE_CAPTION_ARGS+=(--caption_jsonl "$GAIC_CAPTION_JSONL")
fi

python3 src/scripts/prepare_gaic_curated_dataset.py \
  --image_root "$IMAGE_ROOT" \
  --output_parquet "$FILTERED_PARQUET" \
  --flat_image_dir "$FLAT_IMAGE_DIR" \
  --summary_json "$PREP_SUMMARY_JSON" \
  --reference_json_out "$GAIC_REFERENCE_JSON" \
  --annotation_jsons \
    data/Publics/GAIC/annotations_json/instances_train.json \
    data/Publics/GAIC/annotations_json/instances_test.json \
  --bucket "$BUCKET" \
  --pseudo_tar_chunk_size "$PSEUDO_TAR_CHUNK_SIZE" \
  --link_mode "$LINK_MODE" \
  --max_images "$PREPARE_MAX_IMAGES" \
  --skip_existing_links "$SKIP_EXISTING_LINKS" \
  "${PREPARE_CAPTION_ARGS[@]}"

if [ "$PREPARE_ONLY" -eq 1 ]; then
  echo "[done] prepare_only=1 -> prepared parquet and flat image dir only."
  exit 0
fi

bash src/scripts/run_phaseA_to_teacher_e2e.sh \
  --server_mode "$SERVER_MODE" \
  --venv_path "$VENV_PATH" \
  --data_dir "$DATA_DIR" \
  --bucket "$BUCKET" \
  --run_tag "$RUN_TAG" \
  --run_filter 0 \
  --skip_existing "$SKIP_EXISTING" \
  --precompute_mode "$PRECOMPUTE_MODE" \
  --prefer_curated_images 1 \
  --curated_image_dir "$FLAT_IMAGE_DIR" \
  --run_c1 "$RUN_C1" \
  --use_real_expensive "$USE_REAL_EXPENSIVE" \
  --run_c6 "$RUN_C6" \
  --run_vlm_teacher "$RUN_VLM_TEACHER" \
  --vlm_backend "$VLM_BACKEND" \
  --vlm_fallback_backend "$VLM_FALLBACK_BACKEND" \
  --run_training_labels "$RUN_TRAINING_LABELS" \
  --safe_leftover_policy "$SAFE_LEFTOVER_POLICY" \
  --run_detailed_report "$RUN_DETAILED_REPORT" \
  --gaic_reference_json "$GAIC_REFERENCE_JSON" \
  "${PASSTHROUGH_ARGS[@]}"

if [ "$SKIP_VALIDATION" -eq 1 ]; then
  echo "[done] skip_validation=1 -> pipeline run finished without validator."
  exit 0
fi

VALIDATE_ARGS=(
  --manifest_parquet "$FILTERED_PARQUET"
  --prepare_summary_json "$PREP_SUMMARY_JSON"
  --precompute_jsonl "$FEATS_RAW_JSONL"
  --routed_jsonl "$ROUTED_JSONL"
  --candidates_jsonl "$CANDIDATES_JSONL"
  --teacher_jsonl "$TEACHER_JSONL"
  --summary_json "$VALIDATION_SUMMARY_JSON"
)

if [ "$RUN_VLM_TEACHER" -eq 1 ]; then
  VALIDATE_ARGS+=(
    --vlm_labels_jsonl "$VLM_LABELS_JSONL"
    --vlm_meta_jsonl "$VLM_META_JSONL"
    --vlm_summary_json "$VLM_SUMMARY_JSON"
  )
fi

if [ "$RUN_TRAINING_LABELS" -eq 1 ]; then
  VALIDATE_ARGS+=(
    --training_validation_json "$TRAINING_VALIDATION_JSON"
    --training_gaic_like_summary_json "$TRAINING_GAIC_LIKE_SUMMARY_JSON"
  )
fi

python3 src/scripts/validate_gaic_e2e_outputs.py "${VALIDATE_ARGS[@]}"

echo "[done] validation summary: $VALIDATION_SUMMARY_JSON"
