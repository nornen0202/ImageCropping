#!/bin/bash

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$PROJECT_ROOT"

RUN_TAG="${RUN_TAG:-gaic_full_260406_v1}"
DATA_DIR="${DATA_DIR:-data/GAIC/All}"
IMAGE_ROOT="${IMAGE_ROOT:-${DATA_DIR}/images}"
GAIC_REF_JSON="${GAIC_REF_JSON:-${DATA_DIR}/gaic_reference_available.json}"
GAIC_TRAIN_JSON="${GAIC_TRAIN_JSON:-data/Publics/GAIC/annotations_json/instances_train.json}"
GAIC_TEST_JSON="${GAIC_TEST_JSON:-data/Publics/GAIC/annotations_json/instances_test.json}"
SAFE_LEFTOVER_POLICY="${SAFE_LEFTOVER_POLICY:-ignore}"
TRAINING_DEBUG_VIZ_SAMPLE_SIZE="${TRAINING_DEBUG_VIZ_SAMPLE_SIZE:-50}"
TRAINING_DEBUG_VIZ_SEED="${TRAINING_DEBUG_VIZ_SEED:-42}"
GAIC_BENCHMARK_SAMPLE_COUNT="${GAIC_BENCHMARK_SAMPLE_COUNT:-8}"
GAIC_BENCHMARK_MAX_IMAGES="${GAIC_BENCHMARK_MAX_IMAGES:-0}"
GAIC_SUBJECT_AB_RUN_TAG="${GAIC_SUBJECT_AB_RUN_TAG:-gaic_eval_${RUN_TAG}}"
GAIC_SUBJECT_AB_SAMPLE_COUNT="${GAIC_SUBJECT_AB_SAMPLE_COUNT:-8}"
GAIC_SUBJECT_AB_MAX_IMAGES="${GAIC_SUBJECT_AB_MAX_IMAGES:-0}"
GAIC_SUBJECT_AB_SKIP_EXISTING="${GAIC_SUBJECT_AB_SKIP_EXISTING:-1}"
SALIENCY_PRIORITY="${SALIENCY_PRIORITY:-quality_first}"
VENV_PATH="${VENV_PATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ -n "$VENV_PATH" ] && [ -f "$VENV_PATH" ]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH"
  echo "[info] activated venv: $VENV_PATH"
fi

ARTIFACTS_DIR="${DATA_DIR}/artifacts"
case "$SAFE_LEFTOVER_POLICY" in
  keep_negative) MAIN_POLICY_SUFFIX="keepneg" ;;
  ignore) MAIN_POLICY_SUFFIX="ignore" ;;
  promote_soft_positive) MAIN_POLICY_SUFFIX="softpos" ;;
  *)
    echo "[error] unsupported SAFE_LEFTOVER_POLICY: $SAFE_LEFTOVER_POLICY"
    exit 1
    ;;
esac

TRAINING_DIR="${ARTIFACTS_DIR}/training_labels/${RUN_TAG}_leftover_${MAIN_POLICY_SUFFIX}_monotonic"
TRAINING_DEBUG_VIZ_OUT_DIR="${TRAINING_DIR}/debug_visualizations_balanced${TRAINING_DEBUG_VIZ_SAMPLE_SIZE}_bottomneg"
VALIDATION_JSON="${ARTIFACTS_DIR}/validation/gaic_e2e_validation_${RUN_TAG}.json"
BENCHMARK_DIR="${ARTIFACTS_DIR}/reports/gaic_benchmark_eval_${RUN_TAG}"
SUBJECT_AB_BASELINE_CANDIDATES_JSONL="${SUBJECT_AB_BASELINE_CANDIDATES_JSONL:-${DATA_DIR}/artifacts/candidates/candidates_ar_gaic_260320_r0.jsonl}"
SUBJECT_AB_BASELINE_BENCHMARK_SUMMARY="${SUBJECT_AB_BASELINE_BENCHMARK_SUMMARY:-${DATA_DIR}/artifacts/reports/gaic_benchmark_eval_gaic_260320_r0/benchmark_summary.json}"
ACTUAL_SIZE_CACHE_JSON="${ACTUAL_SIZE_CACHE_JSON:-${DATA_DIR}/cache/actual_image_size_map.json}"

PRECOMPUTE_RAW_JSONL="${ARTIFACTS_DIR}/precompute/feats_c2c3c5_v2_strict_raw.jsonl"
PRECOMPUTE_ROUTED_JSONL="${ARTIFACTS_DIR}/precompute/feats_c2c3c5_v2_strict_enriched_routed.jsonl"
PRECOMPUTE_ROUTED_C7_JSONL="${ARTIFACTS_DIR}/precompute/feats_c2c3c5_v2_strict_enriched_routed_c7_saliency.jsonl"
PRECOMPUTE_MERGED_JSONL="${ARTIFACTS_DIR}/precompute/feats_c2c3c5_v2_strict_enriched.jsonl"
CANDIDATES_JSONL="${ARTIFACTS_DIR}/candidates/candidates_ar_${RUN_TAG}.jsonl"
TEACHER_JSONL="${ARTIFACTS_DIR}/teacher/scores/teacher_scores_ar_${RUN_TAG}.jsonl"
FILTERED_PARQUET="${DATA_DIR}/filtered_gaic_all.parquet"
PREPARE_SUMMARY_JSON="${DATA_DIR}/gaic_prepare_summary.json"

if [ -f "$PRECOMPUTE_ROUTED_C7_JSONL" ]; then
  BENCHMARK_FEATURES_JSONL="$PRECOMPUTE_ROUTED_C7_JSONL"
else
  BENCHMARK_FEATURES_JSONL="$PRECOMPUTE_ROUTED_JSONL"
fi

main_canonical_jsonl() {
  printf "%s/train_conditional_detr_canonical.jsonl" "$TRAINING_DIR"
}

main_batch_jsonl() {
  printf "%s/train_conditional_detr_batch.jsonl" "$TRAINING_DIR"
}

main_label_json_dir() {
  printf "%s/label_json" "$TRAINING_DIR"
}

ensure_file() {
  local path="$1"
  if [ ! -e "$path" ]; then
    echo "[error] missing required path: $path"
    exit 1
  fi
}

run_cmd() {
  local name="$1"
  shift
  echo "[run] $name"
  "$@"
}

build_gaic_like_export() {
  local training_dir="$1"
  run_cmd "convert_sstk_detr_batch_to_gaic_like:$(basename "$training_dir")" \
    "$PYTHON_BIN" src/scripts/convert_sstk_detr_batch_to_gaic_like.py \
      --batch_jsonl "${training_dir}/train_conditional_detr_batch.jsonl" \
      --gaic_reference_json "$GAIC_REF_JSON" \
      --gaic_train_reference_json "$GAIC_TRAIN_JSON" \
      --gaic_test_reference_json "$GAIC_TEST_JSON" \
      --out_json "${training_dir}/label_json/gaic_like_labels_full.json" \
      --out_summary_json "${training_dir}/label_json/gaic_like_conversion_summary.json" \
      --out_guide_md "${training_dir}/label_json/GAIC_LIKE_LABEL_FORMAT_KO.md" \
      --out_train_json "${training_dir}/label_json/gaic_like_labels_train.json" \
      --out_test_json "${training_dir}/label_json/gaic_like_labels_test.json" \
      --out_unassigned_json "${training_dir}/label_json/gaic_like_labels_unassigned.json"
}

validate_training_variant() {
  local training_dir="$1"
  local validation_json="$2"
  run_cmd "validate_gaic_e2e_outputs:$(basename "$training_dir")" \
    "$PYTHON_BIN" src/scripts/validate_gaic_e2e_outputs.py \
      --manifest_parquet "$FILTERED_PARQUET" \
      --prepare_summary_json "$PREPARE_SUMMARY_JSON" \
      --precompute_jsonl "$PRECOMPUTE_RAW_JSONL" \
      --routed_jsonl "$BENCHMARK_FEATURES_JSONL" \
      --candidates_jsonl "$CANDIDATES_JSONL" \
      --teacher_jsonl "$TEACHER_JSONL" \
      --training_validation_json "${training_dir}/validation_summary.json" \
      --training_gaic_like_summary_json "${training_dir}/label_json/gaic_like_conversion_summary.json" \
      --summary_json "$validation_json"
}

build_training_variant() {
  local policy="$1"
  local suffix="$2"
  local training_dir="${ARTIFACTS_DIR}/training_labels/${RUN_TAG}_leftover_${suffix}_monotonic"
  local validation_json="${ARTIFACTS_DIR}/validation/gaic_e2e_validation_${RUN_TAG}_leftover_${suffix}_monotonic.json"

  run_cmd "build_finalscore_training_data:${suffix}" \
    "$PYTHON_BIN" src/scripts/build_finalscore_training_data.py \
      --teacher_scores_jsonl "$TEACHER_JSONL" \
      --out_dir "$training_dir" \
      --image_root "$IMAGE_ROOT" \
      --safe_leftover_policy "$policy" \
      --progress 1 \
      --strict_validation 1 \
      --report_examples 8

  run_cmd "export_sstk_detr_labels_to_annotation_json:${suffix}" \
    "$PYTHON_BIN" src/scripts/export_sstk_detr_labels_to_annotation_json.py \
      --canonical_jsonl "${training_dir}/train_conditional_detr_canonical.jsonl" \
      --batch_jsonl "${training_dir}/train_conditional_detr_batch.jsonl" \
      --out_dir "${training_dir}/label_json"

  build_gaic_like_export "$training_dir"
  validate_training_variant "$training_dir" "$validation_json"
}

echo "========================================================"
echo " Resume GAIC downstream"
echo "========================================================"
echo " run_tag                 : $RUN_TAG"
echo " data_dir                : $DATA_DIR"
echo " image_root              : $IMAGE_ROOT"
echo " teacher_jsonl           : $TEACHER_JSONL"
echo " training_dir            : $TRAINING_DIR"
echo " benchmark_dir           : $BENCHMARK_DIR"
echo " subject_ab_run_tag      : $GAIC_SUBJECT_AB_RUN_TAG"
echo " python_bin              : $PYTHON_BIN"
echo "========================================================"

ensure_file "$TEACHER_JSONL"
ensure_file "$CANDIDATES_JSONL"
ensure_file "$FILTERED_PARQUET"
ensure_file "$PREPARE_SUMMARY_JSON"
ensure_file "$PRECOMPUTE_RAW_JSONL"
ensure_file "$BENCHMARK_FEATURES_JSONL"
ensure_file "$GAIC_REF_JSON"
ensure_file "$GAIC_TRAIN_JSON"
ensure_file "$GAIC_TEST_JSON"
ensure_file "$SUBJECT_AB_BASELINE_CANDIDATES_JSONL"
ensure_file "$SUBJECT_AB_BASELINE_BENCHMARK_SUMMARY"

run_cmd "build_finalscore_training_data:main" \
  "$PYTHON_BIN" src/scripts/build_finalscore_training_data.py \
    --teacher_scores_jsonl "$TEACHER_JSONL" \
    --out_dir "$TRAINING_DIR" \
    --image_root "$IMAGE_ROOT" \
    --safe_leftover_policy "$SAFE_LEFTOVER_POLICY" \
    --progress 1 \
    --strict_validation 1 \
    --report_examples 8

run_cmd "export_sstk_detr_labels_to_annotation_json:main" \
  "$PYTHON_BIN" src/scripts/export_sstk_detr_labels_to_annotation_json.py \
    --canonical_jsonl "$(main_canonical_jsonl)" \
    --batch_jsonl "$(main_batch_jsonl)" \
    --out_dir "$(main_label_json_dir)"

build_gaic_like_export "$TRAINING_DIR"

build_training_variant "keep_negative" "keepneg"
build_training_variant "promote_soft_positive" "softpos"

run_cmd "build_gaic_training_label_debug_viz:main" \
  "$PYTHON_BIN" src/scripts/build_gaic_training_label_debug_viz.py \
    --label_json "${TRAINING_DIR}/label_json/gaic_like_labels_full.json" \
    --batch_jsonl "${TRAINING_DIR}/train_conditional_detr_batch.jsonl" \
    --gaic_gt_train_json "$GAIC_TRAIN_JSON" \
    --gaic_gt_test_json "$GAIC_TEST_JSON" \
    --image_root "$IMAGE_ROOT" \
    --subject_mode_vocab "${TRAINING_DIR}/subject_mode_vocab.json" \
    --sample_size "$TRAINING_DEBUG_VIZ_SAMPLE_SIZE" \
    --seed "$TRAINING_DEBUG_VIZ_SEED" \
    --out_dir "$TRAINING_DEBUG_VIZ_OUT_DIR"

validate_training_variant "$TRAINING_DIR" "$VALIDATION_JSON"

run_cmd "run_gaic_benchmark_eval" \
  "$PYTHON_BIN" src/scripts/run_gaic_benchmark_eval.py \
    --candidates_jsonl "$CANDIDATES_JSONL" \
    --features_jsonl "$BENCHMARK_FEATURES_JSONL" \
    --teacher_jsonl "$TEACHER_JSONL" \
    --training_label_dir "$TRAINING_DIR" \
    --gaic_train_json "$GAIC_TRAIN_JSON" \
    --gaic_test_json "$GAIC_TEST_JSON" \
    --image_dir "$IMAGE_ROOT" \
    --output_dir "$BENCHMARK_DIR" \
    --sample_count "$GAIC_BENCHMARK_SAMPLE_COUNT" \
    --max_images "$GAIC_BENCHMARK_MAX_IMAGES"

run_cmd "run_gaic_subject_region_ab" \
  "$PYTHON_BIN" src/scripts/run_gaic_subject_region_ab.py \
    --filtered_parquet "$FILTERED_PARQUET" \
    --input_features_jsonl "$PRECOMPUTE_MERGED_JSONL" \
    --image_dir "$IMAGE_ROOT" \
    --actual_size_cache_json "$ACTUAL_SIZE_CACHE_JSON" \
    --gaic_train_json "$GAIC_TRAIN_JSON" \
    --gaic_test_json "$GAIC_TEST_JSON" \
    --baseline_candidates_jsonl "$SUBJECT_AB_BASELINE_CANDIDATES_JSONL" \
    --baseline_benchmark_summary "$SUBJECT_AB_BASELINE_BENCHMARK_SUMMARY" \
    --saliency_priority "$SALIENCY_PRIORITY" \
    --run_tag "$GAIC_SUBJECT_AB_RUN_TAG" \
    --sample_count "$GAIC_SUBJECT_AB_SAMPLE_COUNT" \
    --max_images "$GAIC_SUBJECT_AB_MAX_IMAGES" \
    --skip_existing "$GAIC_SUBJECT_AB_SKIP_EXISTING"

echo "========================================================"
echo " Downstream Resume Done"
echo "========================================================"
echo " main training dir : $TRAINING_DIR"
echo " validation json   : $VALIDATION_JSON"
echo " benchmark dir     : $BENCHMARK_DIR"
echo " subject ab tag    : $GAIC_SUBJECT_AB_RUN_TAG"
