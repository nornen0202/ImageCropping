#!/bin/bash
# ============================================================================
# Public benchmark E2E runner: stage -> real precompute -> candidates -> teacher
# -> FCDB/CPC/GNMC Mode S/SC eval.
#
# This mirrors the GAIC runner shape in run_gaic_to_teacher_e2e.sh while using
# staged public benchmark images and task keymaps.
# ============================================================================

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$PROJECT_ROOT"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
DATA_ROOT="data/Publics"
OUTPUT_DIR="artifacts/public_benchmark_eval/public_to_teacher_e2e_${RUN_ID}"
DATASETS=(fcdb cpc gnmc)
SPLIT="all"
MAX_TASKS_PER_DATASET=20
MAX_PAIRWISE_PER_TASK=40
MIN_PAIRWISE_GAP=0.0
LINK_MODE="symlink"
SERVER_MODE=0
VENV_PATH="/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate"
PYTHON_BIN="/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python"

PRECOMPUTE_COMPONENTS=(c1 c2 c3 c5 c6)
PRECOMPUTE_STRATEGY="split"
PRECOMPUTE_PRIORITY="quality_first"
PRECOMPUTE_MODE="single"
PRECOMPUTE_BATCH_SIZE=4
RUN_C7=1
C7_PRIORITY="quality_first"
C7_MAX_IMAGES=0

RUN_BENCHMARK_TEACHER=1
RUN_GENERATED_CANDIDATES=1
RUN_GENERATED_TEACHER=1
RUN_EVAL=1
USE_REAL_EXPENSIVE=1
BENCHMARK_TEACHER_MAX_IMAGES=3
GENERATED_TEACHER_MAX_IMAGES=0
EXP_BATCH_SIZE=8
EXPENSIVE_EVAL_TOP_M=16
TARGET_AR_FILTER_MODE="hard"
SAVE_PUBLIC_TEACHER_REF_EVAL=1

CANDIDATE_MAX_IMAGES=0
MAX_CANDIDATES_PER_AR=160
CANDIDATE_NUM_WORKERS=1
AR_LIST=(FREE 1:1 4:3 3:4 16:9 2:1)
EXPORT_CANDIDATE_GROUP="utility_pool"
EXPORT_SCORE_FIELD="scores.crop_utility_raw"
EVAL_INJECT_GT=0
EVAL_FAILURE_GALLERY_MAX=0
EVAL_PAIRWISE_FAILURE_MAX_PER_TASK=20
GAIC_SUMMARY_JSON=""
GAIC_REPORT_MD=""
BASELINE_SUMMARY_JSON=""
GATE_CONFIG_JSON=""

usage() {
  sed -n '1,80p' "$0"
  cat <<'USAGE'

Options:
  --output_dir PATH
  --data_root PATH
  --datasets fcdb cpc gnmc
  --max_tasks_per_dataset N      0 means all
  --max_pairwise_per_task N
  --link_mode symlink|copy|hardlink|none
  --server_mode 0|1
  --python_bin PATH
  --precompute_components c1 c2 c3 c5 c6
  --precompute_strategy split|unified
  --precompute_mode single|multi|ray|auto
  --precompute_batch_size N
  --run_c7 0|1
  --c7_max_images N
  --use_real_expensive 0|1
  --benchmark_teacher_max_images N
  --generated_teacher_max_images N
  --candidate_max_images N
  --max_candidates_per_ar N
  --candidate_num_workers N
  --target_ar_filter_mode none|hard
  --eval_inject_gt 0|1
  --gaic_summary_json PATH
  --gaic_report_md PATH
  --baseline_summary_json PATH
  --gate_config_json PATH
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --data_root) DATA_ROOT="$2"; shift 2 ;;
    --datasets)
      DATASETS=()
      shift
      while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do
        DATASETS+=("$1")
        shift
      done
      ;;
    --split) SPLIT="$2"; shift 2 ;;
    --max_tasks_per_dataset) MAX_TASKS_PER_DATASET="$2"; shift 2 ;;
    --max_pairwise_per_task) MAX_PAIRWISE_PER_TASK="$2"; shift 2 ;;
    --min_pairwise_gap) MIN_PAIRWISE_GAP="$2"; shift 2 ;;
    --link_mode) LINK_MODE="$2"; shift 2 ;;
    --server_mode) SERVER_MODE="$2"; shift 2 ;;
    --venv_path) VENV_PATH="$2"; shift 2 ;;
    --python_bin) PYTHON_BIN="$2"; shift 2 ;;
    --precompute_components)
      PRECOMPUTE_COMPONENTS=()
      shift
      while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do
        PRECOMPUTE_COMPONENTS+=("$1")
        shift
      done
      ;;
    --precompute_priority) PRECOMPUTE_PRIORITY="$2"; shift 2 ;;
    --precompute_strategy) PRECOMPUTE_STRATEGY="$2"; shift 2 ;;
    --precompute_mode) PRECOMPUTE_MODE="$2"; shift 2 ;;
    --precompute_batch_size) PRECOMPUTE_BATCH_SIZE="$2"; shift 2 ;;
    --run_c7) RUN_C7="$2"; shift 2 ;;
    --c7_priority) C7_PRIORITY="$2"; shift 2 ;;
    --c7_max_images) C7_MAX_IMAGES="$2"; shift 2 ;;
    --run_benchmark_teacher) RUN_BENCHMARK_TEACHER="$2"; shift 2 ;;
    --run_generated_candidates) RUN_GENERATED_CANDIDATES="$2"; shift 2 ;;
    --run_generated_teacher) RUN_GENERATED_TEACHER="$2"; shift 2 ;;
    --run_eval) RUN_EVAL="$2"; shift 2 ;;
    --use_real_expensive) USE_REAL_EXPENSIVE="$2"; shift 2 ;;
    --benchmark_teacher_max_images) BENCHMARK_TEACHER_MAX_IMAGES="$2"; shift 2 ;;
    --generated_teacher_max_images) GENERATED_TEACHER_MAX_IMAGES="$2"; shift 2 ;;
    --exp_batch_size) EXP_BATCH_SIZE="$2"; shift 2 ;;
    --expensive_eval_top_m) EXPENSIVE_EVAL_TOP_M="$2"; shift 2 ;;
    --target_ar_filter_mode) TARGET_AR_FILTER_MODE="$2"; shift 2 ;;
    --save_public_teacher_ref_eval) SAVE_PUBLIC_TEACHER_REF_EVAL="$2"; shift 2 ;;
    --candidate_max_images) CANDIDATE_MAX_IMAGES="$2"; shift 2 ;;
    --max_candidates_per_ar) MAX_CANDIDATES_PER_AR="$2"; shift 2 ;;
    --candidate_num_workers) CANDIDATE_NUM_WORKERS="$2"; shift 2 ;;
    --ar_list)
      AR_LIST=()
      shift
      while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do
        AR_LIST+=("$1")
        shift
      done
      ;;
    --export_candidate_group) EXPORT_CANDIDATE_GROUP="$2"; shift 2 ;;
    --export_score_field) EXPORT_SCORE_FIELD="$2"; shift 2 ;;
    --eval_inject_gt) EVAL_INJECT_GT="$2"; shift 2 ;;
    --eval_failure_gallery_max) EVAL_FAILURE_GALLERY_MAX="$2"; shift 2 ;;
    --eval_pairwise_failure_max_per_task) EVAL_PAIRWISE_FAILURE_MAX_PER_TASK="$2"; shift 2 ;;
    --gaic_summary_json) GAIC_SUMMARY_JSON="$2"; shift 2 ;;
    --gaic_report_md) GAIC_REPORT_MD="$2"; shift 2 ;;
    --baseline_summary_json) BASELINE_SUMMARY_JSON="$2"; shift 2 ;;
    --gate_config_json) GATE_CONFIG_JSON="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [ "${#DATASETS[@]}" -eq 0 ]; then
  echo "[error] --datasets must include at least one dataset"
  exit 1
fi
if [ "${#PRECOMPUTE_COMPONENTS[@]}" -eq 0 ]; then
  echo "[error] --precompute_components must include at least one component"
  exit 1
fi

if [ "$SERVER_MODE" -ne 1 ]; then
  PYTHON_BIN="${PYTHON_BIN:-/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python}"
  if [ -f "$VENV_PATH" ]; then
    # shellcheck disable=SC1090
    source "$VENV_PATH"
  fi
else
  PYTHON_BIN="/usr/local/bin/python3"
fi

mkdir -p "$OUTPUT_DIR"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

STAGE_DIR="$OUTPUT_DIR/stage"
PRECOMPUTE_DIR="$OUTPUT_DIR/precompute"
CAND_DIR="$OUTPUT_DIR/candidates"
TEACHER_DIR="$OUTPUT_DIR/teacher"
PRED_DIR="$OUTPUT_DIR/predictions"
EVAL_DIR="$OUTPUT_DIR/eval"
mkdir -p "$PRECOMPUTE_DIR" "$CAND_DIR" "$TEACHER_DIR" "$PRED_DIR" "$EVAL_DIR"

PARQUET="$STAGE_DIR/sstk_public_images.parquet"
IMAGE_DIR="$STAGE_DIR/images"
KEYMAP_JSONL="$STAGE_DIR/benchmark_keymap.jsonl"
BENCHMARK_CANDIDATES_JSONL="$STAGE_DIR/benchmark_candidates_for_teacher.jsonl"
FEATS_RAW="$PRECOMPUTE_DIR/features_c1c2c3c5c6_raw.jsonl"
FEATS_ENRICHED="$PRECOMPUTE_DIR/features_c1c2c3c5c6_enriched.jsonl"
FEATS_ROUTED="$PRECOMPUTE_DIR/features_c1c2c3c5c6_routed.jsonl"
FEATS_FINAL="$PRECOMPUTE_DIR/features_c1c2c3c5c6c7_final.jsonl"
GENERATED_CANDIDATES_JSONL="$CAND_DIR/generated_candidates.jsonl"
BENCHMARK_TEACHER_JSONL="$TEACHER_DIR/benchmark_teacher_scores.jsonl"
GENERATED_TEACHER_JSONL="$TEACHER_DIR/generated_teacher_scores.jsonl"
PREDICTIONS_JSONL="$PRED_DIR/generated_teacher_predictions.jsonl"

run_step() {
  local name="$1"
  shift
  echo
  echo "===== [$name] $*"
  "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
}

echo "[config] output_dir=$OUTPUT_DIR"
echo "[config] data_root=$DATA_ROOT datasets=${DATASETS[*]} split=$SPLIT max_tasks_per_dataset=$MAX_TASKS_PER_DATASET"
echo "[config] precompute_components=${PRECOMPUTE_COMPONENTS[*]} strategy=$PRECOMPUTE_STRATEGY run_c7=$RUN_C7 use_real_expensive=$USE_REAL_EXPENSIVE"
echo "[config] target_ar_filter_mode=$TARGET_AR_FILTER_MODE python_bin=$PYTHON_BIN"

component_requested() {
  local needle="$1"
  local comp
  for comp in "${PRECOMPUTE_COMPONENTS[@]}"; do
    if [ "$comp" = "all" ] || [ "$comp" = "$needle" ]; then
      return 0
    fi
  done
  return 1
}

run_step stage \
  "$PYTHON_BIN" src/scripts/stage_public_benchmark_inputs.py \
    --data_root "$DATA_ROOT" \
    --datasets "${DATASETS[@]}" \
    --split "$SPLIT" \
    --output_dir "$STAGE_DIR" \
    --max_tasks_per_dataset "$MAX_TASKS_PER_DATASET" \
    --max_pairwise_per_task "$MAX_PAIRWISE_PER_TASK" \
    --min_pairwise_gap "$MIN_PAIRWISE_GAP" \
    --link_mode "$LINK_MODE"

if [ "$PRECOMPUTE_STRATEGY" = "unified" ]; then
  run_step precompute_real \
    bash src/scripts/run_extract_component.sh \
      "$PARQUET" public_benchmark "$FEATS_RAW" \
      --component "${PRECOMPUTE_COMPONENTS[@]}" \
      --priority "$PRECOMPUTE_PRIORITY" \
      --mode "$PRECOMPUTE_MODE" \
      --batch_size "$PRECOMPUTE_BATCH_SIZE" \
      --server_mode "$SERVER_MODE" \
      --python_bin "$PYTHON_BIN" \
      --image_dir "$IMAGE_DIR"
elif [ "$PRECOMPUTE_STRATEGY" = "split" ]; then
  MERGE_INPUTS=()
  if component_requested c1; then
    FEATS_C1="$PRECOMPUTE_DIR/features_c1.jsonl"
    run_step precompute_c1 \
      bash src/scripts/run_extract_component.sh \
        "$PARQUET" public_benchmark "$FEATS_C1" \
        --component c1 \
        --priority "$PRECOMPUTE_PRIORITY" \
        --mode "$PRECOMPUTE_MODE" \
        --batch_size "$PRECOMPUTE_BATCH_SIZE" \
        --server_mode "$SERVER_MODE" \
        --python_bin "$PYTHON_BIN" \
        --image_dir "$IMAGE_DIR"
    MERGE_INPUTS+=("$FEATS_C1")
  fi
  if component_requested c2; then
    FEATS_C2="$PRECOMPUTE_DIR/features_c2.jsonl"
    run_step precompute_c2 \
      bash src/scripts/run_extract_component.sh \
        "$PARQUET" public_benchmark "$FEATS_C2" \
        --component c2 \
        --priority "$PRECOMPUTE_PRIORITY" \
        --mode "$PRECOMPUTE_MODE" \
        --batch_size "$PRECOMPUTE_BATCH_SIZE" \
        --server_mode "$SERVER_MODE" \
        --python_bin "$PYTHON_BIN" \
        --image_dir "$IMAGE_DIR"
    MERGE_INPUTS+=("$FEATS_C2")
  fi
  if component_requested c3 && component_requested c6; then
    FEATS_C3C6="$PRECOMPUTE_DIR/features_c3c6.jsonl"
    run_step precompute_c3c6 \
      bash src/scripts/run_extract_component.sh \
        "$PARQUET" public_benchmark "$FEATS_C3C6" \
        --component c3 c6 \
        --priority "$PRECOMPUTE_PRIORITY" \
        --mode "$PRECOMPUTE_MODE" \
        --batch_size "$PRECOMPUTE_BATCH_SIZE" \
        --server_mode "$SERVER_MODE" \
        --python_bin "$PYTHON_BIN" \
        --image_dir "$IMAGE_DIR"
    MERGE_INPUTS+=("$FEATS_C3C6")
  else
    if component_requested c3; then
      FEATS_C3="$PRECOMPUTE_DIR/features_c3.jsonl"
      run_step precompute_c3 \
        bash src/scripts/run_extract_component.sh \
          "$PARQUET" public_benchmark "$FEATS_C3" \
          --component c3 \
          --priority "$PRECOMPUTE_PRIORITY" \
          --mode "$PRECOMPUTE_MODE" \
          --batch_size "$PRECOMPUTE_BATCH_SIZE" \
          --server_mode "$SERVER_MODE" \
          --python_bin "$PYTHON_BIN" \
          --image_dir "$IMAGE_DIR"
      MERGE_INPUTS+=("$FEATS_C3")
    fi
    if component_requested c6; then
      FEATS_C6="$PRECOMPUTE_DIR/features_c6.jsonl"
      run_step precompute_c6 \
        bash src/scripts/run_extract_component.sh \
          "$PARQUET" public_benchmark "$FEATS_C6" \
          --component c6 \
          --priority "$PRECOMPUTE_PRIORITY" \
          --mode "$PRECOMPUTE_MODE" \
          --batch_size "$PRECOMPUTE_BATCH_SIZE" \
          --server_mode "$SERVER_MODE" \
          --python_bin "$PYTHON_BIN" \
          --image_dir "$IMAGE_DIR"
      MERGE_INPUTS+=("$FEATS_C6")
    fi
  fi
  if component_requested c5; then
    FEATS_C5="$PRECOMPUTE_DIR/features_c5.jsonl"
    run_step precompute_c5 \
      bash src/scripts/run_extract_component.sh \
        "$PARQUET" public_benchmark "$FEATS_C5" \
        --component c5 \
        --priority "$PRECOMPUTE_PRIORITY" \
        --mode "$PRECOMPUTE_MODE" \
        --batch_size "$PRECOMPUTE_BATCH_SIZE" \
        --server_mode "$SERVER_MODE" \
        --python_bin "$PYTHON_BIN" \
        --image_dir "$IMAGE_DIR"
    MERGE_INPUTS+=("$FEATS_C5")
  fi
  run_step merge_precompute \
    "$PYTHON_BIN" src/scripts/merge_feature_jsonl.py \
      --input_parquet "$PARQUET" \
      --inputs "${MERGE_INPUTS[@]}" \
      --output_jsonl "$FEATS_RAW" \
      --progress 1
else
  echo "[error] unsupported --precompute_strategy: $PRECOMPUTE_STRATEGY"
  exit 1
fi

run_step enrich_c3 \
  "$PYTHON_BIN" src/scripts/enrich_c3_pose_jsonl.py \
    --input_c3_jsonl "$FEATS_RAW" \
    --input_parquet "$PARQUET" \
    --output_jsonl "$FEATS_ENRICHED" \
    --use_actual_image_size 1 \
    --image_dir "$IMAGE_DIR" \
    --actual_size_cache_json "$PRECOMPUTE_DIR/actual_image_size_map.json"

run_step route_subject_mode \
  "$PYTHON_BIN" src/scripts/enrich_subject_mode_jsonl.py \
    --input_feats_jsonl "$FEATS_ENRICHED" \
    --input_filtered_parquet "$PARQUET" \
    --output_jsonl "$FEATS_ROUTED" \
    --progress 1

if [ "$RUN_C7" -ne 0 ]; then
  run_step augment_c7 \
    "$PYTHON_BIN" src/scripts/augment_saliency_subject_features.py \
      --input_jsonl "$FEATS_ROUTED" \
      --output_jsonl "$FEATS_FINAL" \
      --image_dir "$IMAGE_DIR" \
      --priority "$C7_PRIORITY" \
      --max_images "$C7_MAX_IMAGES" \
      --progress 1
else
  cp "$FEATS_ROUTED" "$FEATS_FINAL"
fi

if [ "$RUN_BENCHMARK_TEACHER" -ne 0 ]; then
  run_step teacher_benchmark \
    bash src/scripts/run_teacher_scorer.sh \
      --candidates_jsonl "$BENCHMARK_CANDIDATES_JSONL" \
      --features_jsonl "$FEATS_FINAL" \
      --c1_jsonl "$FEATS_RAW" \
      --parquet "$PARQUET" \
      --image_dir "$IMAGE_DIR" \
      --server_mode "$SERVER_MODE" \
      --python_bin "$PYTHON_BIN" \
      --output_jsonl "$BENCHMARK_TEACHER_JSONL" \
      --output_overview_json "$TEACHER_DIR/benchmark_teacher_overview.json" \
      --output_overview_csv "$TEACHER_DIR/benchmark_teacher_overview_by_ar.csv" \
      --use_real_expensive "$USE_REAL_EXPENSIVE" \
      --target_ar_filter_mode "$TARGET_AR_FILTER_MODE" \
      --exp_batch_size "$EXP_BATCH_SIZE" \
      --expensive_eval_top_m "$EXPENSIVE_EVAL_TOP_M" \
      --save_public_teacher_ref_eval "$SAVE_PUBLIC_TEACHER_REF_EVAL" \
      --max_images "$BENCHMARK_TEACHER_MAX_IMAGES" \
      --run_qa 0 \
      --run_viz 0
fi

if [ "$RUN_GENERATED_CANDIDATES" -ne 0 ]; then
  run_step generate_candidates \
    bash src/scripts/run_generate_candidates.sh \
      "$PARQUET" "$FEATS_FINAL" "$FEATS_FINAL" "$GENERATED_CANDIDATES_JSONL" \
      --server_mode "$SERVER_MODE" \
      --python_bin "$PYTHON_BIN" \
      --image_dir "$IMAGE_DIR" \
      --use_actual_image_size 1 \
      --strict_actual_size 1 \
      --ar_list "${AR_LIST[@]}" \
      --max_images "$CANDIDATE_MAX_IMAGES" \
      --max_candidates_per_ar "$MAX_CANDIDATES_PER_AR" \
      --num_workers "$CANDIDATE_NUM_WORKERS" \
      --save_visualization 0 \
      --save_overview 1 \
      --overview_json "$CAND_DIR/generated_candidates_overview.json" \
      --overview_ar_csv "$CAND_DIR/generated_candidates_overview_by_ar.csv" \
      --overview_source_csv "$CAND_DIR/generated_candidates_overview_source.csv" \
      --overview_subject_source_csv "$CAND_DIR/generated_candidates_overview_subject_source.csv"
fi

if [ "$RUN_GENERATED_TEACHER" -ne 0 ]; then
  run_step teacher_generated \
    bash src/scripts/run_teacher_scorer.sh \
      --candidates_jsonl "$GENERATED_CANDIDATES_JSONL" \
      --features_jsonl "$FEATS_FINAL" \
      --c1_jsonl "$FEATS_RAW" \
      --parquet "$PARQUET" \
      --image_dir "$IMAGE_DIR" \
      --server_mode "$SERVER_MODE" \
      --python_bin "$PYTHON_BIN" \
      --output_jsonl "$GENERATED_TEACHER_JSONL" \
      --output_overview_json "$TEACHER_DIR/generated_teacher_overview.json" \
      --output_overview_csv "$TEACHER_DIR/generated_teacher_overview_by_ar.csv" \
      --use_real_expensive "$USE_REAL_EXPENSIVE" \
      --target_ar_filter_mode "$TARGET_AR_FILTER_MODE" \
      --exp_batch_size "$EXP_BATCH_SIZE" \
      --expensive_eval_top_m "$EXPENSIVE_EVAL_TOP_M" \
      --save_public_teacher_ref_eval "$SAVE_PUBLIC_TEACHER_REF_EVAL" \
      --max_images "$GENERATED_TEACHER_MAX_IMAGES" \
      --run_qa 0 \
      --run_viz 0
fi

if [ "$RUN_EVAL" -ne 0 ]; then
  run_step export_predictions \
    "$PYTHON_BIN" src/scripts/export_public_benchmark_predictions.py \
      --teacher_scores_jsonl "$GENERATED_TEACHER_JSONL" \
      --keymap_jsonl "$KEYMAP_JSONL" \
      --output_jsonl "$PREDICTIONS_JSONL" \
      --candidate_group "$EXPORT_CANDIDATE_GROUP" \
      --score_field "$EXPORT_SCORE_FIELD" \
      --summary_json "$PRED_DIR/generated_teacher_predictions_summary.json"

  COMMON_EVAL_ARGS=(
    "$PYTHON_BIN" src/scripts/run_combined_benchmark_eval.py
    --data_root "$DATA_ROOT"
    --datasets "${DATASETS[@]}"
    --split "$SPLIT"
    --predictions_jsonl "$PREDICTIONS_JSONL"
    --max_tasks_per_dataset "$MAX_TASKS_PER_DATASET"
    --max_pairwise_per_task "$MAX_PAIRWISE_PER_TASK"
    --min_pairwise_gap "$MIN_PAIRWISE_GAP"
    --score_field score
    --inject_gt "$EVAL_INJECT_GT"
    --target_ar_filter_mode "$TARGET_AR_FILTER_MODE"
    --pairwise_failure_max_per_task "$EVAL_PAIRWISE_FAILURE_MAX_PER_TASK"
    --failure_gallery_max "$EVAL_FAILURE_GALLERY_MAX"
  )
  if [ -n "$GAIC_SUMMARY_JSON" ]; then
    COMMON_EVAL_ARGS+=(--gaic_summary_json "$GAIC_SUMMARY_JSON")
  fi
  if [ -n "$GAIC_REPORT_MD" ]; then
    COMMON_EVAL_ARGS+=(--gaic_report_md "$GAIC_REPORT_MD")
  fi
  if [ -n "$BASELINE_SUMMARY_JSON" ]; then
    COMMON_EVAL_ARGS+=(--baseline_summary_json "$BASELINE_SUMMARY_JSON")
  fi
  if [ -n "$GATE_CONFIG_JSON" ]; then
    COMMON_EVAL_ARGS+=(--gate_config_json "$GATE_CONFIG_JSON")
  fi

  run_step eval_mode_s \
    "${COMMON_EVAL_ARGS[@]}" \
      --mode S \
      --output_dir "$EVAL_DIR/mode_S"

  run_step eval_mode_sc \
    "${COMMON_EVAL_ARGS[@]}" \
      --mode SC \
      --output_dir "$EVAL_DIR/mode_SC"
fi

SUMMARY_JSON="$OUTPUT_DIR/public_to_teacher_e2e_summary.json"
"$PYTHON_BIN" - "$SUMMARY_JSON" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
root = out.parent
payload = {
    "output_dir": str(root.resolve()),
    "artifacts": {
        "stage_summary": str((root / "stage" / "staging_summary.json").resolve()),
        "features_raw": str((root / "precompute" / "features_c1c2c3c5c6_raw.jsonl").resolve()),
        "features_final": str((root / "precompute" / "features_c1c2c3c5c6c7_final.jsonl").resolve()),
        "generated_candidates": str((root / "candidates" / "generated_candidates.jsonl").resolve()),
        "benchmark_teacher_scores": str((root / "teacher" / "benchmark_teacher_scores.jsonl").resolve()),
        "generated_teacher_scores": str((root / "teacher" / "generated_teacher_scores.jsonl").resolve()),
        "predictions": str((root / "predictions" / "generated_teacher_predictions.jsonl").resolve()),
        "eval_mode_s": str((root / "eval" / "mode_S" / "combined_benchmark_summary.json").resolve()),
        "eval_mode_sc": str((root / "eval" / "mode_SC" / "combined_benchmark_summary.json").resolve()),
        "logs": str((root / "logs").resolve()),
    },
}
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
PY

echo "[done] Public benchmark E2E outputs under: $OUTPUT_DIR"
