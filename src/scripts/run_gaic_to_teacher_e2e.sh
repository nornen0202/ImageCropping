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
#   - optional C7 saliency + GAIC benchmark/A-B steps can be chained after e2e
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
AUTO_LEFTOVER_VARIANTS=1
LEFTOVER_VARIANT_POLICIES="keep_negative,ignore,promote_soft_positive"
RUN_DETAILED_REPORT=0
RUN_C6=0
USE_REAL_EXPENSIVE=0
RUN_C1=0
PRECOMPUTE_MODE="unified"
RUN_C7_SALIENCY=0
C7_SALIENCY_PRIORITY="quality_first"
C7_SALIENCY_WEIGHTS_DIR=""
C7_SALIENCY_DEVICE="auto"
GAIC_REFERENCE_JSON=""
GAIC_TRAIN_REFERENCE_JSON="data/Publics/GAIC/annotations_json/instances_train.json"
GAIC_TEST_REFERENCE_JSON="data/Publics/GAIC/annotations_json/instances_test.json"
TRAINING_DIR_OVERRIDE=""
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
RUN_GAIC_BENCHMARK_EVAL=0
GAIC_BENCHMARK_SAMPLE_COUNT=8
GAIC_BENCHMARK_MAX_IMAGES=0
RUN_GAIC_SUBJECT_REGION_AB=0
GAIC_SUBJECT_AB_RUN_TAG=""
GAIC_SUBJECT_AB_SAMPLE_COUNT=8
GAIC_SUBJECT_AB_MAX_IMAGES=0
GAIC_SUBJECT_AB_SKIP_EXISTING=1
GAIC_SUBJECT_AB_BASELINE_CANDIDATES_JSONL=""
GAIC_SUBJECT_AB_BASELINE_BENCHMARK_SUMMARY=""
GAIC_SUBJECT_AB_SALIENCY_JSONL_OVERRIDE=""

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
    --auto_leftover_variants) AUTO_LEFTOVER_VARIANTS="$2"; shift 2 ;;
    --leftover_variant_policies) LEFTOVER_VARIANT_POLICIES="$2"; shift 2 ;;
    --run_detailed_report) RUN_DETAILED_REPORT="$2"; shift 2 ;;
    --run_c6) RUN_C6="$2"; shift 2 ;;
    --use_real_expensive) USE_REAL_EXPENSIVE="$2"; shift 2 ;;
    --run_c1) RUN_C1="$2"; shift 2 ;;
    --precompute_mode) PRECOMPUTE_MODE="$2"; shift 2 ;;
    --run_c7_saliency) RUN_C7_SALIENCY="$2"; shift 2 ;;
    --c7_saliency_priority) C7_SALIENCY_PRIORITY="$2"; shift 2 ;;
    --c7_saliency_weights_dir) C7_SALIENCY_WEIGHTS_DIR="$2"; shift 2 ;;
    --c7_saliency_device) C7_SALIENCY_DEVICE="$2"; shift 2 ;;
    --gaic_reference_json) GAIC_REFERENCE_JSON="$2"; shift 2 ;;
    --gaic_train_reference_json) GAIC_TRAIN_REFERENCE_JSON="$2"; shift 2 ;;
    --gaic_test_reference_json) GAIC_TEST_REFERENCE_JSON="$2"; shift 2 ;;
    --training_labels_dir) TRAINING_DIR_OVERRIDE="$2"; shift 2 ;;
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
    --run_gaic_benchmark_eval) RUN_GAIC_BENCHMARK_EVAL="$2"; shift 2 ;;
    --gaic_benchmark_sample_count) GAIC_BENCHMARK_SAMPLE_COUNT="$2"; shift 2 ;;
    --gaic_benchmark_max_images) GAIC_BENCHMARK_MAX_IMAGES="$2"; shift 2 ;;
    --run_gaic_subject_region_ab) RUN_GAIC_SUBJECT_REGION_AB="$2"; shift 2 ;;
    --gaic_subject_ab_run_tag) GAIC_SUBJECT_AB_RUN_TAG="$2"; shift 2 ;;
    --gaic_subject_ab_sample_count) GAIC_SUBJECT_AB_SAMPLE_COUNT="$2"; shift 2 ;;
    --gaic_subject_ab_max_images) GAIC_SUBJECT_AB_MAX_IMAGES="$2"; shift 2 ;;
    --gaic_subject_ab_skip_existing) GAIC_SUBJECT_AB_SKIP_EXISTING="$2"; shift 2 ;;
    --gaic_subject_ab_baseline_candidates_jsonl) GAIC_SUBJECT_AB_BASELINE_CANDIDATES_JSONL="$2"; shift 2 ;;
    --gaic_subject_ab_baseline_benchmark_summary) GAIC_SUBJECT_AB_BASELINE_BENCHMARK_SUMMARY="$2"; shift 2 ;;
    --gaic_subject_ab_saliency_jsonl_override) GAIC_SUBJECT_AB_SALIENCY_JSONL_OVERRIDE="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,240p' "$0"
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
if [ -n "$TRAINING_DIR_OVERRIDE" ]; then
  TRAINING_DIR="$TRAINING_DIR_OVERRIDE"
else
  TRAINING_DIR="${ARTIFACTS_DIR}/training_labels/${TRAINING_LABELS_SUBDIR}"
fi
VALIDATION_DIR="${ARTIFACTS_DIR}/validation"
METADATA_DIR="${ARTIFACTS_DIR}/metadata"
VALIDATION_SUMMARY_JSON="${VALIDATION_DIR}/gaic_e2e_validation${SUFFIX}.json"

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
TRAINING_GAIC_LIKE_SUMMARY_JSON="${TRAINING_DIR}/coco/gaic_like_conversion_summary.json"
TRAINING_GAIC_LIKE_JSON="${TRAINING_DIR}/coco/instances_conditional_detr_batch_gaic_like.json"
TRAINING_GAIC_LIKE_TRAIN_JSON="${TRAINING_DIR}/coco/instances_conditional_detr_batch_gaic_like_train.json"
TRAINING_GAIC_LIKE_TEST_JSON="${TRAINING_DIR}/coco/instances_conditional_detr_batch_gaic_like_test.json"
TRAINING_GAIC_LIKE_UNASSIGNED_JSON="${TRAINING_DIR}/coco/instances_conditional_detr_batch_gaic_like_unassigned.json"
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
if [ -z "$GAIC_SUBJECT_AB_RUN_TAG" ]; then
  if [ -n "$RUN_TAG" ]; then
    GAIC_SUBJECT_AB_RUN_TAG="${RUN_TAG}_saliency_v1"
  else
    GAIC_SUBJECT_AB_RUN_TAG="gaic_subject_ab_saliency_v1"
  fi
fi
if [ -z "$GAIC_SUBJECT_AB_BASELINE_CANDIDATES_JSONL" ]; then
  GAIC_SUBJECT_AB_BASELINE_CANDIDATES_JSONL="${DATA_DIR}/artifacts/candidates/candidates_ar_gaic_260320_r0.jsonl"
fi
if [ -z "$GAIC_SUBJECT_AB_BASELINE_BENCHMARK_SUMMARY" ]; then
  GAIC_SUBJECT_AB_BASELINE_BENCHMARK_SUMMARY="${DATA_DIR}/artifacts/reports/gaic_benchmark_eval_gaic_260320_r0/benchmark_summary.json"
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
echo " run_c7_saliency       : $RUN_C7_SALIENCY (priority=$C7_SALIENCY_PRIORITY device=$C7_SALIENCY_DEVICE)"
echo " gaic_generate_caps    : $GAIC_GENERATE_CAPTIONS (preset=$GAIC_CAPTION_PRESET backend=${GAIC_CAPTION_BACKEND:-auto})"
echo " run_c6                : $RUN_C6"
echo " run_vlm_teacher       : $RUN_VLM_TEACHER (backend=$VLM_BACKEND fallback=$VLM_FALLBACK_BACKEND)"
echo " run_training_labels   : $RUN_TRAINING_LABELS (policy=$SAFE_LEFTOVER_POLICY)"
echo " auto_leftover_variants: $AUTO_LEFTOVER_VARIANTS (policies=$LEFTOVER_VARIANT_POLICIES)"
echo " run_detailed_report   : $RUN_DETAILED_REPORT"
echo " run_gaic_benchmark    : $RUN_GAIC_BENCHMARK_EVAL (sample_count=$GAIC_BENCHMARK_SAMPLE_COUNT max_images=$GAIC_BENCHMARK_MAX_IMAGES)"
echo " run_gaic_subject_ab   : $RUN_GAIC_SUBJECT_REGION_AB (run_tag=$GAIC_SUBJECT_AB_RUN_TAG sample_count=$GAIC_SUBJECT_AB_SAMPLE_COUNT max_images=$GAIC_SUBJECT_AB_MAX_IMAGES)"
echo " gaic_reference_json   : $GAIC_REFERENCE_JSON"
echo " gaic_train_ref_json   : $GAIC_TRAIN_REFERENCE_JSON"
echo " gaic_test_ref_json    : $GAIC_TEST_REFERENCE_JSON"
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

build_gaic_like_export() {
  local training_dir="$1"
  local batch_jsonl="${training_dir}/train_conditional_detr_batch.jsonl"
  local coco_dir="${training_dir}/coco"
  local out_json="${coco_dir}/instances_conditional_detr_batch_gaic_like.json"
  local out_summary_json="${coco_dir}/gaic_like_conversion_summary.json"
  local out_guide_md="${coco_dir}/GAIC_INSTANCES_TRAIN_FORMAT_KO.md"
  local out_train_json="${coco_dir}/instances_conditional_detr_batch_gaic_like_train.json"
  local out_test_json="${coco_dir}/instances_conditional_detr_batch_gaic_like_test.json"
  local out_unassigned_json="${coco_dir}/instances_conditional_detr_batch_gaic_like_unassigned.json"
  local split_args=()
  if [ -f "$GAIC_TRAIN_REFERENCE_JSON" ]; then
    split_args+=(--gaic_train_reference_json "$GAIC_TRAIN_REFERENCE_JSON")
  fi
  if [ -f "$GAIC_TEST_REFERENCE_JSON" ]; then
    split_args+=(--gaic_test_reference_json "$GAIC_TEST_REFERENCE_JSON")
  fi
  python3 src/scripts/convert_sstk_detr_batch_to_gaic_like.py \
    --batch_jsonl "$batch_jsonl" \
    --gaic_reference_json "$GAIC_REFERENCE_JSON" \
    --out_json "$out_json" \
    --out_summary_json "$out_summary_json" \
    --out_guide_md "$out_guide_md" \
    --out_train_json "$out_train_json" \
    --out_test_json "$out_test_json" \
    --out_unassigned_json "$out_unassigned_json" \
    "${split_args[@]}"
}

validate_training_outputs() {
  local training_dir="$1"
  local validation_json="$2"
  local validate_args=(
    --manifest_parquet "$FILTERED_PARQUET"
    --prepare_summary_json "$PREP_SUMMARY_JSON"
    --precompute_jsonl "$FEATS_RAW_JSONL"
    --routed_jsonl "$ROUTED_JSONL"
    --candidates_jsonl "$CANDIDATES_JSONL"
    --teacher_jsonl "$TEACHER_JSONL"
    --training_validation_json "${training_dir}/validation_summary.json"
    --training_gaic_like_summary_json "${training_dir}/coco/gaic_like_conversion_summary.json"
    --summary_json "$validation_json"
  )
  if [ "$RUN_VLM_TEACHER" -eq 1 ]; then
    validate_args+=(
      --vlm_labels_jsonl "$VLM_LABELS_JSONL"
      --vlm_meta_jsonl "$VLM_META_JSONL"
      --vlm_summary_json "$VLM_SUMMARY_JSON"
    )
  fi
  python3 src/scripts/validate_gaic_e2e_outputs.py "${validate_args[@]}"
}

build_training_variant() {
  local policy="$1"
  local suffix
  suffix=$(leftover_policy_suffix "$policy")
  local variant_dir="${ARTIFACTS_DIR}/training_labels/${RUN_TAG}_leftover_${suffix}"
  local variant_validation_json="${VALIDATION_DIR}/gaic_e2e_validation_${RUN_TAG}_leftover_${suffix}.json"
  if [ "$SKIP_EXISTING" -eq 1 ] \
    && [ -f "${variant_dir}/validation_summary.json" ] \
    && [ -f "${variant_dir}/coco/gaic_like_conversion_summary.json" ] \
    && [ -f "$variant_validation_json" ]; then
    echo "[skip] leftover variant already complete: ${variant_dir}"
    return 0
  fi
  echo "[run] build leftover variant: policy=${policy} out_dir=${variant_dir}"
  python3 src/scripts/build_finalscore_training_data.py \
    --teacher_scores_jsonl "$TEACHER_JSONL" \
    --out_dir "$variant_dir" \
    --image_root "$FLAT_IMAGE_DIR" \
    --safe_leftover_policy "$policy" \
    --strict_validation 1 \
    --report_examples 8
  python3 src/scripts/convert_sstk_detr_labels_to_coco.py \
    --canonical_jsonl "${variant_dir}/train_conditional_detr_canonical.jsonl" \
    --batch_jsonl "${variant_dir}/train_conditional_detr_batch.jsonl" \
    --out_dir "${variant_dir}/coco"
  build_gaic_like_export "$variant_dir"
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

C7_EXTRA_ARGS=()
if [ -n "$C7_SALIENCY_WEIGHTS_DIR" ]; then
  C7_EXTRA_ARGS+=(--c7_saliency_weights_dir "$C7_SALIENCY_WEIGHTS_DIR")
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
  --run_c7_saliency "$RUN_C7_SALIENCY" \
  --c7_saliency_priority "$C7_SALIENCY_PRIORITY" \
  --c7_saliency_device "$C7_SALIENCY_DEVICE" \
  --use_real_expensive "$USE_REAL_EXPENSIVE" \
  --run_c6 "$RUN_C6" \
  --run_vlm_teacher "$RUN_VLM_TEACHER" \
  --vlm_backend "$VLM_BACKEND" \
  --vlm_fallback_backend "$VLM_FALLBACK_BACKEND" \
  --run_training_labels "$RUN_TRAINING_LABELS" \
  --safe_leftover_policy "$SAFE_LEFTOVER_POLICY" \
  --run_detailed_report "$RUN_DETAILED_REPORT" \
  --gaic_reference_json "$GAIC_REFERENCE_JSON" \
  "${C7_EXTRA_ARGS[@]}" \
  "${PASSTHROUGH_ARGS[@]}"

if [ "$RUN_TRAINING_LABELS" -eq 1 ]; then
  if [ ! -f "$TRAINING_DETR_BATCH_JSON" ] || [ ! -f "$TRAINING_DETR_CANONICAL_JSON" ]; then
    echo "[error] expected training label jsonl missing: $TRAINING_DETR_BATCH_JSON | $TRAINING_DETR_CANONICAL_JSON"
    exit 1
  fi
  echo "[run] refresh GAIC-like export with official split outputs"
  build_gaic_like_export "$TRAINING_DIR"
  if [ "$AUTO_LEFTOVER_VARIANTS" -eq 1 ] && [ -n "$RUN_TAG" ]; then
    build_all_leftover_variants
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

BENCHMARK_FEATURES_JSONL="$ROUTED_JSONL"
if [ ! -f "$BENCHMARK_FEATURES_JSONL" ]; then
  BENCHMARK_FEATURES_JSONL="$MERGED_JSONL"
fi
if [ ! -f "$BENCHMARK_FEATURES_JSONL" ]; then
  BENCHMARK_FEATURES_JSONL="$FEATS_RAW_JSONL"
fi
if [ "$RUN_C7_SALIENCY" -eq 1 ] && [ -f "$ROUTED_C7_JSONL" ]; then
  BENCHMARK_FEATURES_JSONL="$ROUTED_C7_JSONL"
fi

if [ "$RUN_GAIC_BENCHMARK_EVAL" -eq 1 ]; then
  if [ ! -d "$TRAINING_DIR" ]; then
    echo "[error] gaic benchmark eval requires training labels dir: $TRAINING_DIR"
    exit 1
  fi
  BENCHMARK_REPORT_DIR="${ARTIFACTS_DIR}/reports/gaic_benchmark_eval_${RUN_TAG}"
  echo "[run] GAIC benchmark eval -> ${BENCHMARK_REPORT_DIR}"
  python3 src/scripts/run_gaic_benchmark_eval.py \
    --candidates_jsonl "$CANDIDATES_JSONL" \
    --features_jsonl "$BENCHMARK_FEATURES_JSONL" \
    --teacher_jsonl "$TEACHER_JSONL" \
    --training_label_dir "$TRAINING_DIR" \
    --gaic_train_json "$GAIC_TRAIN_REFERENCE_JSON" \
    --gaic_test_json "$GAIC_TEST_REFERENCE_JSON" \
    --image_dir "$FLAT_IMAGE_DIR" \
    --output_dir "$BENCHMARK_REPORT_DIR" \
    --sample_count "$GAIC_BENCHMARK_SAMPLE_COUNT" \
    --max_images "$GAIC_BENCHMARK_MAX_IMAGES"
fi

if [ "$RUN_GAIC_SUBJECT_REGION_AB" -eq 1 ]; then
  if [ ! -f "$GAIC_SUBJECT_AB_BASELINE_CANDIDATES_JSONL" ]; then
    echo "[error] gaic subject-region A/B requires baseline candidates: $GAIC_SUBJECT_AB_BASELINE_CANDIDATES_JSONL"
    exit 1
  fi
  if [ ! -f "$GAIC_SUBJECT_AB_BASELINE_BENCHMARK_SUMMARY" ]; then
    echo "[error] gaic subject-region A/B requires baseline benchmark summary: $GAIC_SUBJECT_AB_BASELINE_BENCHMARK_SUMMARY"
    exit 1
  fi
  AB_ARGS=(
    --filtered_parquet "$FILTERED_PARQUET"
    --input_features_jsonl "$MERGED_JSONL"
    --image_dir "$FLAT_IMAGE_DIR"
    --actual_size_cache_json "${DATA_DIR}/cache/actual_image_size_map.json"
    --gaic_train_json "$GAIC_TRAIN_REFERENCE_JSON"
    --gaic_test_json "$GAIC_TEST_REFERENCE_JSON"
    --baseline_candidates_jsonl "$GAIC_SUBJECT_AB_BASELINE_CANDIDATES_JSONL"
    --baseline_benchmark_summary "$GAIC_SUBJECT_AB_BASELINE_BENCHMARK_SUMMARY"
    --saliency_priority "$C7_SALIENCY_PRIORITY"
    --run_tag "$GAIC_SUBJECT_AB_RUN_TAG"
    --sample_count "$GAIC_SUBJECT_AB_SAMPLE_COUNT"
    --max_images "$GAIC_SUBJECT_AB_MAX_IMAGES"
    --skip_existing "$GAIC_SUBJECT_AB_SKIP_EXISTING"
  )
  if [ -n "$GAIC_SUBJECT_AB_SALIENCY_JSONL_OVERRIDE" ]; then
    AB_ARGS+=(--saliency_jsonl_override "$GAIC_SUBJECT_AB_SALIENCY_JSONL_OVERRIDE")
  fi
  echo "[run] GAIC saliency/subject-region A/B -> ${GAIC_SUBJECT_AB_RUN_TAG}"
  python3 src/scripts/run_gaic_subject_region_ab.py "${AB_ARGS[@]}"
fi

if [ "$RUN_TRAINING_LABELS" -eq 1 ] && [ "$AUTO_LEFTOVER_VARIANTS" -eq 1 ] && [ -n "$RUN_TAG" ]; then
  echo "[done] leftover variants:"
  for raw_policy in ${LEFTOVER_VARIANT_POLICIES//,/ }; do
    policy=$(printf "%s" "$raw_policy" | tr -d '[:space:]')
    if [ -z "$policy" ]; then
      continue
    fi
    suffix=$(leftover_policy_suffix "$policy")
    echo "  - ${ARTIFACTS_DIR}/training_labels/${RUN_TAG}_leftover_${suffix}"
    echo "    validation: ${VALIDATION_DIR}/gaic_e2e_validation_${RUN_TAG}_leftover_${suffix}.json"
  done
fi

echo "[done] validation summary: $VALIDATION_SUMMARY_JSON"
