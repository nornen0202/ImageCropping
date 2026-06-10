#!/bin/bash
# ==============================================================================
# run_test_images_to_teacher_e2e.sh
# Test-images end-to-end wrapper:
#   prepare flat curated pool + minimal reference json
#   -> reuse SSTK e2e runner with test-image defaults
#   -> validate outputs
# ------------------------------------------------------------------------------
# Key policies:
#   - all images under data/test_images are treated as curated_pool
#   - no GT/benchmark path is executed
#   - filter is skipped, VLM teacher is disabled
#   - all other major stages default to enabled
# ==============================================================================

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$PROJECT_ROOT"

DATA_DIR="data/TestImages/All"
BUCKET="test_images_all"
IMAGE_ROOT="data/test_images"
FLAT_IMAGE_DIR=""
RUN_TAG="test_images_e2e"
SERVER_MODE=0
VENV_PATH="/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate"
PYTHON_BIN="/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python"

PSEUDO_TAR_CHUNK_SIZE=256
LINK_MODE="symlink"
PREPARE_MAX_IMAGES=0
PREPARE_NUM_WORKERS=0
SKIP_EXISTING=1
SKIP_EXISTING_LINKS=1
PREPARE_ONLY=0
SKIP_VALIDATION=0
GPU_IDS=""
GPU_WORKERS=0
RUN_VLM_TEACHER=0
VLM_BACKEND="heuristic"
VLM_FALLBACK_BACKEND="none"
RUN_TRAINING_LABELS=1
RUN_MULTIMODE_TRAINING_LABELS=0
SAFE_LEFTOVER_POLICY="ignore"
SCORE_PROFILE="single_stage2"
SCORE_PROFILE_OVERRIDES_JSON=""
RUN_TRAINING_LABEL_DEBUG_VIZ=0
TRAINING_LABEL_DEBUG_VIZ_SAMPLE_SIZE=50
TRAINING_LABEL_DEBUG_VIZ_SEED=42
TRAINING_LABEL_DEBUG_VIZ_OUT_DIR=""
AUTO_LEFTOVER_VARIANTS=0
LEFTOVER_VARIANT_POLICIES="keep_negative,promote_soft_positive"
RUN_DETAILED_REPORT=1
RUN_C1=1
RUN_C1_EXPLICIT=0
RUN_C2=1
RUN_C3=1
RUN_C3_ENRICH=1
RUN_C4=1
RUN_C5=1
RUN_C6=1
RUN_MERGE=1
RUN_SUBJECT_ROUTING=1
RUN_CANDIDATES=1
RUN_TEACHER=1
USE_REAL_EXPENSIVE=1
PRECOMPUTE_MODE="unified"
RUN_C7_SALIENCY=1
C7_SALIENCY_PRIORITY="quality_first"
C7_SALIENCY_WEIGHTS_DIR=""
C7_SALIENCY_DEVICE="auto"
REFERENCE_JSON=""
TRAINING_DIR_OVERRIDE=""
MULTIMODE_TRAINING_DIR_OVERRIDE=""
MULTIMODE_TARGET_ARS="FREE,1:1,9:16,16:9,3:4,4:3"
MULTIMODE_MAX_IMAGES=0
MULTIMODE_WRITE_DEBUG_VIZ=0
MULTIMODE_DEBUG_VIZ_LIMIT=16
MULTIMODE_FEATURES_JSONL_OVERRIDE=""
MULTIMODE_CANDIDATES_JSONL_OVERRIDE=""
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
GAIC_CAPTION_MULTI_GPU=-1
GAIC_CAPTION_GPU_IDS=""
GAIC_CAPTION_NUM_WORKERS=0

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
    --prepare_num_workers) PREPARE_NUM_WORKERS="$2"; shift 2 ;;
    --skip_existing) SKIP_EXISTING="$2"; shift 2 ;;
    --skip_existing_links) SKIP_EXISTING_LINKS="$2"; shift 2 ;;
    --prepare_only) PREPARE_ONLY="$2"; shift 2 ;;
    --skip_validation) SKIP_VALIDATION="$2"; shift 2 ;;
    --gpu_ids) GPU_IDS="$2"; shift 2 ;;
    --gpu_workers) GPU_WORKERS="$2"; shift 2 ;;
    --run_vlm_teacher) RUN_VLM_TEACHER="$2"; shift 2 ;;
    --vlm_backend) VLM_BACKEND="$2"; shift 2 ;;
    --vlm_fallback_backend) VLM_FALLBACK_BACKEND="$2"; shift 2 ;;
    --run_training_labels) RUN_TRAINING_LABELS="$2"; shift 2 ;;
    --run_multimode_training_labels) RUN_MULTIMODE_TRAINING_LABELS="$2"; shift 2 ;;
    --safe_leftover_policy) SAFE_LEFTOVER_POLICY="$2"; shift 2 ;;
    --score_profile) SCORE_PROFILE="$2"; shift 2 ;;
    --score_profile_overrides_json) SCORE_PROFILE_OVERRIDES_JSON="$2"; shift 2 ;;
    --run_training_label_debug_viz) RUN_TRAINING_LABEL_DEBUG_VIZ="$2"; shift 2 ;;
    --training_label_debug_viz_sample_size) TRAINING_LABEL_DEBUG_VIZ_SAMPLE_SIZE="$2"; shift 2 ;;
    --training_label_debug_viz_seed) TRAINING_LABEL_DEBUG_VIZ_SEED="$2"; shift 2 ;;
    --training_label_debug_viz_out_dir) TRAINING_LABEL_DEBUG_VIZ_OUT_DIR="$2"; shift 2 ;;
    --auto_leftover_variants) AUTO_LEFTOVER_VARIANTS="$2"; shift 2 ;;
    --leftover_variant_policies) LEFTOVER_VARIANT_POLICIES="$2"; shift 2 ;;
    --run_detailed_report) RUN_DETAILED_REPORT="$2"; shift 2 ;;
    --run_c1) RUN_C1="$2"; RUN_C1_EXPLICIT=1; shift 2 ;;
    --run_c2) RUN_C2="$2"; shift 2 ;;
    --run_c3) RUN_C3="$2"; shift 2 ;;
    --run_c3_enrich) RUN_C3_ENRICH="$2"; shift 2 ;;
    --run_c4) RUN_C4="$2"; shift 2 ;;
    --run_c5) RUN_C5="$2"; shift 2 ;;
    --run_c6) RUN_C6="$2"; shift 2 ;;
    --run_merge) RUN_MERGE="$2"; shift 2 ;;
    --run_subject_routing) RUN_SUBJECT_ROUTING="$2"; shift 2 ;;
    --run_candidates) RUN_CANDIDATES="$2"; shift 2 ;;
    --run_teacher) RUN_TEACHER="$2"; shift 2 ;;
    --use_real_expensive) USE_REAL_EXPENSIVE="$2"; shift 2 ;;
    --precompute_mode) PRECOMPUTE_MODE="$2"; shift 2 ;;
    --run_c7_saliency) RUN_C7_SALIENCY="$2"; shift 2 ;;
    --c7_saliency_priority) C7_SALIENCY_PRIORITY="$2"; shift 2 ;;
    --c7_saliency_weights_dir) C7_SALIENCY_WEIGHTS_DIR="$2"; shift 2 ;;
    --c7_saliency_device) C7_SALIENCY_DEVICE="$2"; shift 2 ;;
    --reference_json) REFERENCE_JSON="$2"; shift 2 ;;
    --training_labels_dir) TRAINING_DIR_OVERRIDE="$2"; shift 2 ;;
    --multimode_training_labels_dir) MULTIMODE_TRAINING_DIR_OVERRIDE="$2"; shift 2 ;;
    --multimode_target_ars) MULTIMODE_TARGET_ARS="$2"; shift 2 ;;
    --multimode_max_images) MULTIMODE_MAX_IMAGES="$2"; shift 2 ;;
    --multimode_write_debug_viz) MULTIMODE_WRITE_DEBUG_VIZ="$2"; shift 2 ;;
    --multimode_debug_viz_limit) MULTIMODE_DEBUG_VIZ_LIMIT="$2"; shift 2 ;;
    --multimode_features_jsonl_override) MULTIMODE_FEATURES_JSONL_OVERRIDE="$2"; shift 2 ;;
    --multimode_candidates_jsonl_override) MULTIMODE_CANDIDATES_JSONL_OVERRIDE="$2"; shift 2 ;;
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
    --gaic_caption_multi_gpu) GAIC_CAPTION_MULTI_GPU="$2"; shift 2 ;;
    --gaic_caption_gpu_ids) GAIC_CAPTION_GPU_IDS="$2"; shift 2 ;;
    --gaic_caption_num_workers) GAIC_CAPTION_NUM_WORKERS="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,260p' "$0"
      exit 0
      ;;
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

normalize_csv_ids() {
  local s="${1:-}"
  s=$(echo "$s" | tr -d ' ')
  s=$(echo "$s" | sed -E 's/^,+//; s/,+$//; s/,+/,/g')
  echo "$s"
}

count_csv_ids() {
  local s
  s=$(normalize_csv_ids "${1:-}")
  if [ -z "$s" ]; then
    echo 0
    return
  fi
  awk -F',' '{print NF}' <<< "$s"
}

detect_gpu_ids() {
  local csv="${CUDA_VISIBLE_DEVICES:-}"
  csv=$(normalize_csv_ids "$csv")
  if [ -n "$csv" ]; then
    echo "$csv"
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo ""
    return
  fi
  nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | awk '{print $1}' | paste -sd, -
}

detect_cpu_cores() {
  local detected
  detected=$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)
  if [ -z "$detected" ] || [ "$detected" -le 0 ] 2>/dev/null; then
    detected=$(nproc 2>/dev/null || true)
  fi
  if [ -z "$detected" ] || [ "$detected" -le 0 ] 2>/dev/null; then
    detected=1
  fi
  echo "$detected"
}

resolve_cpu_workers() {
  local requested="$1"
  local cpu_cores="$2"
  if [ -n "$requested" ] && [ "$requested" -gt 0 ] 2>/dev/null; then
    echo "$requested"
    return
  fi
  echo "$cpu_cores"
}

resolve_gpu_workers() {
  local gpu_ids="$1"
  local requested="$2"
  local gpu_count
  gpu_count=$(count_csv_ids "$gpu_ids")
  if [ "$gpu_count" -le 0 ]; then
    echo ""
    return
  fi
  if [ -n "$requested" ] && [ "$requested" -gt 0 ] 2>/dev/null; then
    if [ "$requested" -lt "$gpu_count" ]; then
      echo "$requested"
    else
      echo "$gpu_count"
    fi
    return
  fi
  echo "$gpu_count"
}

mkdir -p "$DATA_DIR"
if [ -z "$FLAT_IMAGE_DIR" ]; then
  FLAT_IMAGE_DIR="${DATA_DIR}/images"
fi
if [ -z "$REFERENCE_JSON" ]; then
  REFERENCE_JSON="${DATA_DIR}/test_images_reference.json"
fi
if [ "$SERVER_MODE" -eq 1 ]; then
  VENV_PATH=""
  PYTHON_BIN="python3"
fi
if [ "$USE_REAL_EXPENSIVE" -eq 1 ] && [ "$RUN_C1_EXPLICIT" -eq 0 ] && [ "$RUN_C1" -ne 1 ]; then
  RUN_C1=1
fi
if [ "$GAIC_GENERATE_CAPTIONS" -lt 0 ]; then
  if [ "$RUN_C1" -eq 1 ]; then
    GAIC_GENERATE_CAPTIONS=1
  else
    GAIC_GENERATE_CAPTIONS=0
  fi
fi

CPU_CORES=$(detect_cpu_cores)
EFFECTIVE_PREPARE_NUM_WORKERS=$(resolve_cpu_workers "$PREPARE_NUM_WORKERS" "$CPU_CORES")
EFFECTIVE_GPU_IDS=$(normalize_csv_ids "$GPU_IDS")
if [ -z "$EFFECTIVE_GPU_IDS" ]; then
  EFFECTIVE_GPU_IDS=$(normalize_csv_ids "$(detect_gpu_ids)")
fi
EFFECTIVE_GPU_WORKERS=$(resolve_gpu_workers "$EFFECTIVE_GPU_IDS" "$GPU_WORKERS")
EFFECTIVE_CAPTION_GPU_IDS=$(normalize_csv_ids "$GAIC_CAPTION_GPU_IDS")
if [ -z "$EFFECTIVE_CAPTION_GPU_IDS" ]; then
  EFFECTIVE_CAPTION_GPU_IDS="$EFFECTIVE_GPU_IDS"
fi
EFFECTIVE_CAPTION_GPU_WORKERS=$(resolve_gpu_workers "$EFFECTIVE_CAPTION_GPU_IDS" "$GAIC_CAPTION_NUM_WORKERS")
EFFECTIVE_CAPTION_GPU_COUNT=$(count_csv_ids "$EFFECTIVE_CAPTION_GPU_IDS")
if [ "$GAIC_CAPTION_MULTI_GPU" -lt 0 ]; then
  if [ "$EFFECTIVE_CAPTION_GPU_COUNT" -gt 1 ] && [[ "$GAIC_CAPTION_DEVICE" != cpu* ]]; then
    GAIC_CAPTION_MULTI_GPU=1
  else
    GAIC_CAPTION_MULTI_GPU=0
  fi
fi

if [ -n "$RUN_TAG" ]; then
  SUFFIX="_${RUN_TAG}"
  case "$SAFE_LEFTOVER_POLICY" in
    keep_negative) TRAINING_LABELS_SUBDIR="${RUN_TAG}_leftover_keepneg_monotonic" ;;
    ignore) TRAINING_LABELS_SUBDIR="${RUN_TAG}_leftover_ignore_monotonic" ;;
    promote_soft_positive) TRAINING_LABELS_SUBDIR="${RUN_TAG}_leftover_softpos_monotonic" ;;
    *)
      echo "[error] unsupported safe_leftover_policy: $SAFE_LEFTOVER_POLICY"
      exit 1
      ;;
  esac
else
  SUFFIX=""
  case "$SAFE_LEFTOVER_POLICY" in
    keep_negative) TRAINING_LABELS_SUBDIR="latest_leftover_keepneg_monotonic" ;;
    ignore) TRAINING_LABELS_SUBDIR="latest_leftover_ignore_monotonic" ;;
    promote_soft_positive) TRAINING_LABELS_SUBDIR="latest_leftover_softpos_monotonic" ;;
    *)
      echo "[error] unsupported safe_leftover_policy: $SAFE_LEFTOVER_POLICY"
      exit 1
      ;;
  esac
fi

PREP_SUMMARY_JSON="${DATA_DIR}/test_images_prepare_summary.json"
FILTERED_PARQUET="${DATA_DIR}/filtered_${BUCKET}.parquet"
ARTIFACTS_DIR="${DATA_DIR}/artifacts"
PRECOMPUTE_DIR="${ARTIFACTS_DIR}/precompute"
CANDIDATES_DIR="${ARTIFACTS_DIR}/candidates"
TEACHER_DIR="${ARTIFACTS_DIR}/teacher"
VLM_DIR="${ARTIFACTS_DIR}/vlm_teacher"
if [ -n "$TRAINING_DIR_OVERRIDE" ]; then
  TRAINING_DIR="$TRAINING_DIR_OVERRIDE"
else
  TRAINING_DIR="${ARTIFACTS_DIR}/training_labels/${TRAINING_LABELS_SUBDIR}"
fi
if [ -n "$MULTIMODE_TRAINING_DIR_OVERRIDE" ]; then
  MULTIMODE_TRAINING_DIR="$MULTIMODE_TRAINING_DIR_OVERRIDE"
elif [ -n "$RUN_TAG" ]; then
  MULTIMODE_TRAINING_DIR="${ARTIFACTS_DIR}/training_labels_multimode/${RUN_TAG}_multimode_v1"
else
  MULTIMODE_TRAINING_DIR="${ARTIFACTS_DIR}/training_labels_multimode/latest_multimode_v1"
fi
VALIDATION_DIR="${ARTIFACTS_DIR}/validation"
METADATA_DIR="${ARTIFACTS_DIR}/metadata"
VALIDATION_SUMMARY_JSON="${VALIDATION_DIR}/test_images_e2e_validation${SUFFIX}.json"

FEATS_RAW_JSONL="${PRECOMPUTE_DIR}/feats_c2c3c5_v2_strict_raw.jsonl"
MERGED_JSONL="${PRECOMPUTE_DIR}/feats_c2c3c5_v2_strict_enriched.jsonl"
ROUTED_JSONL="${PRECOMPUTE_DIR}/feats_c2c3c5_v2_strict_enriched_routed.jsonl"
ROUTED_C7_JSONL="${PRECOMPUTE_DIR}/feats_c2c3c5_v2_strict_enriched_routed_c7_saliency.jsonl"
CANDIDATES_JSONL="${CANDIDATES_DIR}/candidates_ar${SUFFIX}.jsonl"
TEACHER_JSONL="${TEACHER_DIR}/scores/teacher_scores_ar${SUFFIX}.jsonl"
VLM_LABELS_JSONL="${VLM_DIR}/labels/crop_label_v1${SUFFIX}.jsonl"
VLM_META_JSONL="${VLM_DIR}/meta/meta_norm_v1${SUFFIX}.jsonl"
VLM_SUMMARY_JSON="${VLM_DIR}/summary/vlm_teacher_summary${SUFFIX}.json"
TRAINING_VALIDATION_JSON="${TRAINING_DIR}/validation_summary.json"
TRAINING_DETR_CANONICAL_JSON="${TRAINING_DIR}/train_conditional_detr_canonical.jsonl"
TRAINING_DETR_BATCH_JSON="${TRAINING_DIR}/train_conditional_detr_batch.jsonl"
TRAINING_GAIC_LIKE_SUMMARY_JSON="${TRAINING_DIR}/label_json/gaic_like_conversion_summary.json"
TRAINING_LABEL_DEBUG_VIZ_OUT_DIR=${TRAINING_LABEL_DEBUG_VIZ_OUT_DIR:-"${TRAINING_DIR}/debug_visualizations_balanced${TRAINING_LABEL_DEBUG_VIZ_SAMPLE_SIZE}_bottomneg"}
TRAINING_LABEL_DEBUG_VIZ_SUMMARY_JSON="${TRAINING_LABEL_DEBUG_VIZ_OUT_DIR}/summary/summary.json"
MULTIMODE_SUMMARY_JSON="${MULTIMODE_TRAINING_DIR}/summary.json"
MULTIMODE_QUERY_STATUS_JSONL="${MULTIMODE_TRAINING_DIR}/mode_query_status.jsonl"
MULTIMODE_LABEL_JSON="${MULTIMODE_TRAINING_DIR}/label_json/multimode_labels_full.json"
MULTIMODE_VALIDATION_JSON="${MULTIMODE_TRAINING_DIR}/validation_summary.json"
if [ -z "$GAIC_CAPTION_JSONL" ]; then
  GAIC_CAPTION_JSONL="${METADATA_DIR}/test_images_captions${SUFFIX}.jsonl"
fi
if [ -z "$GAIC_CAPTION_SUMMARY_JSON" ]; then
  GAIC_CAPTION_SUMMARY_JSON="${METADATA_DIR}/test_images_captions${SUFFIX}_summary.json"
fi
if [ -z "$GAIC_CAPTION_PRESET" ]; then
  if [ "$SERVER_MODE" -eq 1 ]; then
    GAIC_CAPTION_PRESET="server_quality"
  else
    GAIC_CAPTION_PRESET="local_efficient"
  fi
fi

if [ -n "$VENV_PATH" ] && [ -f "$VENV_PATH" ]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH"
  echo "[info] activated venv: $VENV_PATH"
  _derived_python_bin="$(dirname "$VENV_PATH")/python"
  if [ -x "$_derived_python_bin" ]; then
    PYTHON_BIN="$_derived_python_bin"
  fi
elif [ "$SERVER_MODE" -ne 1 ]; then
  echo "[warn] venv not found: $VENV_PATH (using current python)"
else
  echo "[info] server_mode=1 -> using python3 without venv activation"
fi

echo "========================================================"
echo " Test-images e2e wrapper"
echo "========================================================"
echo " data_dir              : $DATA_DIR"
echo " image_root            : $IMAGE_ROOT"
echo " flat_image_dir        : $FLAT_IMAGE_DIR"
echo " bucket                : $BUCKET"
echo " run_tag               : $RUN_TAG"
echo " prepare_max_images    : $PREPARE_MAX_IMAGES"
echo " prepare_num_workers   : $EFFECTIVE_PREPARE_NUM_WORKERS (cpu_cores=$CPU_CORES)"
echo " pseudo_tar_chunk_size : $PSEUDO_TAR_CHUNK_SIZE"
echo " link_mode             : $LINK_MODE"
echo " skip_existing         : $SKIP_EXISTING"
echo " shared_gpu_ids        : ${EFFECTIVE_GPU_IDS:-auto} (workers=${EFFECTIVE_GPU_WORKERS:-auto})"
echo " run_c1                : $RUN_C1"
echo " run_c2                : $RUN_C2"
echo " run_c3                : $RUN_C3"
echo " run_c3_enrich         : $RUN_C3_ENRICH"
echo " run_c4                : $RUN_C4"
echo " run_c5                : $RUN_C5"
echo " run_c6                : $RUN_C6"
echo " run_merge             : $RUN_MERGE"
echo " run_subject_routing   : $RUN_SUBJECT_ROUTING"
echo " run_candidates        : $RUN_CANDIDATES"
echo " run_teacher           : $RUN_TEACHER"
echo " use_real_expensive    : $USE_REAL_EXPENSIVE"
echo " run_c7_saliency       : $RUN_C7_SALIENCY (priority=$C7_SALIENCY_PRIORITY device=$C7_SALIENCY_DEVICE)"
echo " generate_captions     : $GAIC_GENERATE_CAPTIONS (preset=$GAIC_CAPTION_PRESET backend=${GAIC_CAPTION_BACKEND:-auto} multi_gpu=$GAIC_CAPTION_MULTI_GPU gpu_ids=${EFFECTIVE_CAPTION_GPU_IDS:-auto} workers=${EFFECTIVE_CAPTION_GPU_WORKERS:-auto})"
echo " run_vlm_teacher       : $RUN_VLM_TEACHER (backend=$VLM_BACKEND fallback=$VLM_FALLBACK_BACKEND)"
echo " run_training_labels   : $RUN_TRAINING_LABELS (policy=$SAFE_LEFTOVER_POLICY)"
echo " run_multimode_labels  : $RUN_MULTIMODE_TRAINING_LABELS (dir=$MULTIMODE_TRAINING_DIR)"
echo " multimode target ARs  : $MULTIMODE_TARGET_ARS (max_images=$MULTIMODE_MAX_IMAGES debug_viz=$MULTIMODE_WRITE_DEBUG_VIZ limit=$MULTIMODE_DEBUG_VIZ_LIMIT)"
echo " score_profile         : $SCORE_PROFILE (overrides=${SCORE_PROFILE_OVERRIDES_JSON:-<none>})"
echo " training_debug_viz    : $RUN_TRAINING_LABEL_DEBUG_VIZ (out=$TRAINING_LABEL_DEBUG_VIZ_OUT_DIR sample_size=$TRAINING_LABEL_DEBUG_VIZ_SAMPLE_SIZE seed=$TRAINING_LABEL_DEBUG_VIZ_SEED)"
echo " auto_leftover_variants: $AUTO_LEFTOVER_VARIANTS (policies=$LEFTOVER_VARIANT_POLICIES)"
echo " run_detailed_report   : $RUN_DETAILED_REPORT"
echo " reference_json        : $REFERENCE_JSON"
echo "========================================================"

leftover_policy_suffix() {
  case "$1" in
    keep_negative) printf "keepneg" ;;
    ignore) printf "ignore" ;;
    promote_soft_positive) printf "softpos" ;;
    *)
      echo "[error] unsupported leftover policy: $1" >&2
      return 1
      ;;
  esac
}

current_routed_jsonl() {
  local routed="$ROUTED_JSONL"
  if [ "$RUN_C7_SALIENCY" -eq 1 ] && [ -f "$ROUTED_C7_JSONL" ]; then
    routed="$ROUTED_C7_JSONL"
  fi
  printf "%s" "$routed"
}

build_test_images_like_export() {
  local training_dir="$1"
  local batch_jsonl="${training_dir}/train_conditional_detr_batch.jsonl"
  local label_json_dir="${training_dir}/label_json"
  local out_json="${label_json_dir}/gaic_like_labels_full.json"
  local out_summary_json="${label_json_dir}/gaic_like_conversion_summary.json"
  local out_guide_md="${label_json_dir}/GAIC_LIKE_LABEL_FORMAT_KO.md"
  "$PYTHON_BIN" src/scripts/convert_sstk_detr_batch_to_gaic_like.py \
    --batch_jsonl "$batch_jsonl" \
    --gaic_reference_json "$REFERENCE_JSON" \
    --out_json "$out_json" \
    --out_summary_json "$out_summary_json" \
    --progress 1 \
    --out_guide_md "$out_guide_md"
}

validate_training_outputs() {
  local training_dir="$1"
  local validation_json="$2"
  local validate_args=(
    --manifest_parquet "$FILTERED_PARQUET"
    --prepare_summary_json "$PREP_SUMMARY_JSON"
    --precompute_jsonl "$FEATS_RAW_JSONL"
    --routed_jsonl "$(current_routed_jsonl)"
    --candidates_jsonl "$CANDIDATES_JSONL"
    --teacher_jsonl "$TEACHER_JSONL"
    --training_validation_json "${training_dir}/validation_summary.json"
    --training_gaic_like_summary_json "${training_dir}/label_json/gaic_like_conversion_summary.json"
    --summary_json "$validation_json"
  )
  if [ "$RUN_VLM_TEACHER" -eq 1 ]; then
    validate_args+=(
      --vlm_labels_jsonl "$VLM_LABELS_JSONL"
      --vlm_meta_jsonl "$VLM_META_JSONL"
      --vlm_summary_json "$VLM_SUMMARY_JSON"
    )
  fi
  "$PYTHON_BIN" src/scripts/validate_gaic_e2e_outputs.py --progress 1 "${validate_args[@]}"
}

build_training_variant() {
  local policy="$1"
  if [ "$policy" = "$SAFE_LEFTOVER_POLICY" ]; then
    echo "[skip] leftover variant policy matches main output: ${policy}"
    return 0
  fi
  local suffix
  suffix=$(leftover_policy_suffix "$policy")
  local variant_dir="${ARTIFACTS_DIR}/training_labels/${RUN_TAG}_leftover_${suffix}_monotonic"
  local variant_validation_json="${VALIDATION_DIR}/test_images_e2e_validation_${RUN_TAG}_leftover_${suffix}_monotonic.json"
  if [ "$SKIP_EXISTING" -eq 1 ] \
    && [ -f "${variant_dir}/validation_summary.json" ] \
    && [ -f "${variant_dir}/label_json/gaic_like_conversion_summary.json" ] \
    && [ -f "$variant_validation_json" ]; then
    echo "[skip] leftover variant already complete: ${variant_dir}"
    return 0
  fi
  echo "[run] build leftover variant: policy=${policy} out_dir=${variant_dir}"
  "$PYTHON_BIN" src/scripts/build_finalscore_training_data.py \
    --teacher_scores_jsonl "$TEACHER_JSONL" \
    --out_dir "$variant_dir" \
    --image_root "$FLAT_IMAGE_DIR" \
    --safe_leftover_policy "$policy" \
    --score_profile "$SCORE_PROFILE" \
    --score_profile_overrides_json "$SCORE_PROFILE_OVERRIDES_JSON" \
    --progress 1 \
    --strict_validation 1 \
    --report_examples 8
  "$PYTHON_BIN" src/scripts/export_sstk_detr_labels_to_annotation_json.py \
    --canonical_jsonl "${variant_dir}/train_conditional_detr_canonical.jsonl" \
    --batch_jsonl "${variant_dir}/train_conditional_detr_batch.jsonl" \
    --progress 1 \
    --out_dir "${variant_dir}/label_json"
  build_test_images_like_export "$variant_dir"
  validate_training_outputs "$variant_dir" "$variant_validation_json"
}

build_all_leftover_variants() {
  local raw_policy
  local policy
  IFS=',' read -r -a _policies <<< "$LEFTOVER_VARIANT_POLICIES"
  for raw_policy in "${_policies[@]}"; do
    policy=$(printf "%s" "$raw_policy" | tr -d '[:space:]')
    if [ -z "$policy" ]; then
      continue
    fi
    build_training_variant "$policy"
  done
}

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
    --progress 1
    --multi_gpu "$GAIC_CAPTION_MULTI_GPU"
  )
  if [ -n "$EFFECTIVE_CAPTION_GPU_IDS" ]; then
    CAPTION_ARGS+=(--gpu_ids "$EFFECTIVE_CAPTION_GPU_IDS")
  fi
  if [ -n "$EFFECTIVE_CAPTION_GPU_WORKERS" ]; then
    CAPTION_ARGS+=(--num_workers "$EFFECTIVE_CAPTION_GPU_WORKERS")
  fi
  if [ -n "$GAIC_CAPTION_BACKEND" ]; then
    CAPTION_ARGS+=(--backend "$GAIC_CAPTION_BACKEND")
  fi
  if [ -n "$GAIC_CAPTION_MODEL_ID" ]; then
    CAPTION_ARGS+=(--model_id "$GAIC_CAPTION_MODEL_ID")
  fi
  if [ -n "$GAIC_CAPTION_PROMPT" ]; then
    CAPTION_ARGS+=(--caption_prompt "$GAIC_CAPTION_PROMPT")
  fi
  "$PYTHON_BIN" src/scripts/generate_gaic_captions.py "${CAPTION_ARGS[@]}"
fi

PREPARE_CAPTION_ARGS=()
if [ -f "$GAIC_CAPTION_JSONL" ]; then
  PREPARE_CAPTION_ARGS+=(--caption_jsonl "$GAIC_CAPTION_JSONL")
fi

"$PYTHON_BIN" src/scripts/prepare_gaic_curated_dataset.py \
  --image_root "$IMAGE_ROOT" \
  --output_parquet "$FILTERED_PARQUET" \
  --flat_image_dir "$FLAT_IMAGE_DIR" \
  --summary_json "$PREP_SUMMARY_JSON" \
  --reference_json_out "$REFERENCE_JSON" \
  --bucket "$BUCKET" \
  --pseudo_tar_chunk_size "$PSEUDO_TAR_CHUNK_SIZE" \
  --link_mode "$LINK_MODE" \
  --max_images "$PREPARE_MAX_IMAGES" \
  --progress 1 \
  --num_workers "$EFFECTIVE_PREPARE_NUM_WORKERS" \
  --skip_existing_links "$SKIP_EXISTING_LINKS" \
  --dataset_name "TestImages" \
  "${PREPARE_CAPTION_ARGS[@]}"

if [ "$PREPARE_ONLY" -eq 1 ]; then
  echo "[done] prepare_only=1 -> prepared parquet and flat image dir only."
  exit 0
fi

C7_EXTRA_ARGS=()
if [ -n "$C7_SALIENCY_WEIGHTS_DIR" ]; then
  C7_EXTRA_ARGS+=(--c7_saliency_weights_dir "$C7_SALIENCY_WEIGHTS_DIR")
fi
PHASEA_SHARED_ACCEL_ARGS=()
if [ -n "$EFFECTIVE_GPU_IDS" ]; then
  PHASEA_SHARED_ACCEL_ARGS+=(--gpu_ids "$EFFECTIVE_GPU_IDS")
fi
if [ -n "$EFFECTIVE_GPU_WORKERS" ]; then
  PHASEA_SHARED_ACCEL_ARGS+=(--gpu_workers "$EFFECTIVE_GPU_WORKERS")
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
  --run_c2 "$RUN_C2" \
  --run_c3 "$RUN_C3" \
  --run_c3_enrich "$RUN_C3_ENRICH" \
  --run_c4 "$RUN_C4" \
  --run_c5 "$RUN_C5" \
  --run_c6 "$RUN_C6" \
  --run_merge "$RUN_MERGE" \
  --run_subject_routing "$RUN_SUBJECT_ROUTING" \
  --run_candidates "$RUN_CANDIDATES" \
  --run_teacher "$RUN_TEACHER" \
  --run_c7_saliency "$RUN_C7_SALIENCY" \
  --c7_saliency_priority "$C7_SALIENCY_PRIORITY" \
  --c7_saliency_device "$C7_SALIENCY_DEVICE" \
  --use_real_expensive "$USE_REAL_EXPENSIVE" \
  --run_vlm_teacher "$RUN_VLM_TEACHER" \
  --vlm_backend "$VLM_BACKEND" \
  --vlm_fallback_backend "$VLM_FALLBACK_BACKEND" \
  --run_training_labels "$RUN_TRAINING_LABELS" \
  --training_labels_dir "$TRAINING_DIR" \
  --run_multimode_training_labels "$RUN_MULTIMODE_TRAINING_LABELS" \
  --multimode_training_labels_dir "$MULTIMODE_TRAINING_DIR" \
  --multimode_target_ars "$MULTIMODE_TARGET_ARS" \
  --multimode_max_images "$MULTIMODE_MAX_IMAGES" \
  --multimode_write_debug_viz "$MULTIMODE_WRITE_DEBUG_VIZ" \
  --multimode_debug_viz_limit "$MULTIMODE_DEBUG_VIZ_LIMIT" \
  --multimode_features_jsonl_override "$MULTIMODE_FEATURES_JSONL_OVERRIDE" \
  --multimode_candidates_jsonl_override "$MULTIMODE_CANDIDATES_JSONL_OVERRIDE" \
  --safe_leftover_policy "$SAFE_LEFTOVER_POLICY" \
  --score_profile "$SCORE_PROFILE" \
  --score_profile_overrides_json "$SCORE_PROFILE_OVERRIDES_JSON" \
  --run_training_label_debug_viz "$RUN_TRAINING_LABEL_DEBUG_VIZ" \
  --training_label_debug_viz_sample_size "$TRAINING_LABEL_DEBUG_VIZ_SAMPLE_SIZE" \
  --training_label_debug_viz_seed "$TRAINING_LABEL_DEBUG_VIZ_SEED" \
  --training_label_debug_viz_out_dir "$TRAINING_LABEL_DEBUG_VIZ_OUT_DIR" \
  --run_detailed_report "$RUN_DETAILED_REPORT" \
  --gaic_reference_json "$REFERENCE_JSON" \
  "${PHASEA_SHARED_ACCEL_ARGS[@]}" \
  "${C7_EXTRA_ARGS[@]}" \
  "${PASSTHROUGH_ARGS[@]}"

if [ -n "$MULTIMODE_FEATURES_JSONL_OVERRIDE" ] || [ -n "$MULTIMODE_CANDIDATES_JSONL_OVERRIDE" ]; then
  echo "[info] multimode stage uses overrides: feats=${MULTIMODE_FEATURES_JSONL_OVERRIDE:-<default>} candidates=${MULTIMODE_CANDIDATES_JSONL_OVERRIDE:-<default>}"
fi

if [ "$RUN_TRAINING_LABELS" -eq 1 ]; then
  if [ ! -f "$TRAINING_DETR_BATCH_JSON" ] || [ ! -f "$TRAINING_DETR_CANONICAL_JSON" ]; then
    echo "[error] expected training label jsonl missing: $TRAINING_DETR_BATCH_JSON | $TRAINING_DETR_CANONICAL_JSON"
    exit 1
  fi
  echo "[run] refresh GAIC-like export for test images"
  build_test_images_like_export "$TRAINING_DIR"
  if [ "$AUTO_LEFTOVER_VARIANTS" -eq 1 ] && [ -n "$RUN_TAG" ]; then
    build_all_leftover_variants
  fi
fi

if [ "$RUN_MULTIMODE_TRAINING_LABELS" -eq 1 ]; then
  if [ ! -f "$MULTIMODE_SUMMARY_JSON" ] || [ ! -f "$MULTIMODE_LABEL_JSON" ] || [ ! -f "$MULTIMODE_VALIDATION_JSON" ]; then
    echo "[error] expected multimode outputs missing: $MULTIMODE_SUMMARY_JSON | $MULTIMODE_LABEL_JSON | $MULTIMODE_VALIDATION_JSON"
    exit 1
  fi
fi

if [ "$SKIP_VALIDATION" -eq 1 ]; then
  echo "[done] skip_validation=1 -> pipeline run finished without validator."
  exit 0
fi

VALIDATE_ARGS=(
  --manifest_parquet "$FILTERED_PARQUET"
  --prepare_summary_json "$PREP_SUMMARY_JSON"
  --precompute_jsonl "$FEATS_RAW_JSONL"
  --routed_jsonl "$(current_routed_jsonl)"
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

"$PYTHON_BIN" src/scripts/validate_gaic_e2e_outputs.py --progress 1 "${VALIDATE_ARGS[@]}"

if [ "$RUN_TRAINING_LABELS" -eq 1 ] && [ "$AUTO_LEFTOVER_VARIANTS" -eq 1 ] && [ -n "$RUN_TAG" ]; then
  echo "[done] leftover variants:"
  for raw_policy in ${LEFTOVER_VARIANT_POLICIES//,/ }; do
    policy=$(printf "%s" "$raw_policy" | tr -d '[:space:]')
    if [ -z "$policy" ] || [ "$policy" = "$SAFE_LEFTOVER_POLICY" ]; then
      continue
    fi
    suffix=$(leftover_policy_suffix "$policy")
    echo "  - ${ARTIFACTS_DIR}/training_labels/${RUN_TAG}_leftover_${suffix}_monotonic"
    echo "    validation: ${VALIDATION_DIR}/test_images_e2e_validation_${RUN_TAG}_leftover_${suffix}_monotonic.json"
  done
fi

if [ "$RUN_TRAINING_LABELS" -eq 1 ] && [ "$RUN_TRAINING_LABEL_DEBUG_VIZ" -eq 1 ]; then
  echo "[done] training-label debug viz: ${TRAINING_LABEL_DEBUG_VIZ_OUT_DIR}"
  echo "       summary: ${TRAINING_LABEL_DEBUG_VIZ_SUMMARY_JSON}"
fi

echo "[done] validation summary: $VALIDATION_SUMMARY_JSON"
if [ "$RUN_MULTIMODE_TRAINING_LABELS" -eq 1 ]; then
  echo "[done] multimode labels: $MULTIMODE_TRAINING_DIR"
  echo "       multimode validation: $MULTIMODE_VALIDATION_JSON"
fi
