#!/bin/bash
# ==============================================================================
# run_gaic_v2_splits_to_teacher_labels.sh
# GAIC v2 official train/val/test split wrapper for SSTK teacher label generation.
#
# This script is designed to run inside the same environment as
# run_gaic_to_teacher_e2e.sh. GPU-heavy full-option runs should be launched as an
# MLP GPU workload command per AGENTS.md; use --dry_run 1 locally to inspect the
# exact commands.
# ==============================================================================

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$PROJECT_ROOT"

PUBLIC_ROOT="data/Publics/GAIC_v2"
OUTPUT_ROOT="data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP"
RUN_TAG_SUFFIX="qf_c1c6capexp_c7exp_labels_v1"
SPLITS="test train val"
SERVER_MODE=1
DRY_RUN=0
SKIP_EXISTING=1
PREPARE_MAX_IMAGES=0
PREPARE_NUM_WORKERS=0
LINK_MODE="symlink"
RUN_TEST_BENCHMARK_EVAL=1
RUN_BENCHMARK_EVAL_ALL_SPLITS=0
RUN_TRAINING_LABELS=1
RUN_MULTIMODE_TRAINING_LABELS=1
MULTIMODE_TARGET_ARS="FREE,1:1,9:16,16:9,3:4,4:3"
MULTIMODE_MAX_IMAGES=0
RUN_VLM_TEACHER=1
VLM_BACKEND="heuristic"
VLM_FALLBACK_BACKEND="none"
SCORE_PROFILE="single_stage2"
SCORE_PROFILE_OVERRIDES_JSON=""
SAFE_LEFTOVER_POLICY="ignore"
GPU_IDS=""
GPU_WORKERS=0
ALIGN_DEVICE="cuda"
AESTHETIC_DEVICE="cuda"
EXP_BATCH_SIZE=24
EXP_PREPROCESS_WORKERS=0
EXP_PIN_MEMORY=1
C7_SALIENCY_DEVICE="cuda"
C7_SALIENCY_PRIORITY="quality_first"
GAIC_CAPTION_PRESET="server_quality"
GAIC_CAPTION_BACKEND=""
GAIC_CAPTION_MODEL_ID=""
GAIC_CAPTION_DEVICE="cuda"
GAIC_CAPTION_DTYPE="auto"
GAIC_CAPTION_BATCH_SIZE=8
GAIC_CAPTION_NUM_WORKERS=0
GAIC_CAPTION_MULTI_GPU=-1

while [ "$#" -gt 0 ]; do
  case "$1" in
    --public_root) PUBLIC_ROOT="$2"; shift 2 ;;
    --output_root) OUTPUT_ROOT="$2"; shift 2 ;;
    --run_tag_suffix) RUN_TAG_SUFFIX="$2"; shift 2 ;;
    --splits) SPLITS="$2"; shift 2 ;;
    --server_mode) SERVER_MODE="$2"; shift 2 ;;
    --dry_run) DRY_RUN="$2"; shift 2 ;;
    --skip_existing) SKIP_EXISTING="$2"; shift 2 ;;
    --prepare_max_images) PREPARE_MAX_IMAGES="$2"; shift 2 ;;
    --prepare_num_workers) PREPARE_NUM_WORKERS="$2"; shift 2 ;;
    --link_mode) LINK_MODE="$2"; shift 2 ;;
    --run_test_benchmark_eval) RUN_TEST_BENCHMARK_EVAL="$2"; shift 2 ;;
    --run_benchmark_eval_all_splits) RUN_BENCHMARK_EVAL_ALL_SPLITS="$2"; shift 2 ;;
    --run_training_labels) RUN_TRAINING_LABELS="$2"; shift 2 ;;
    --run_multimode_training_labels) RUN_MULTIMODE_TRAINING_LABELS="$2"; shift 2 ;;
    --multimode_target_ars) MULTIMODE_TARGET_ARS="$2"; shift 2 ;;
    --multimode_max_images) MULTIMODE_MAX_IMAGES="$2"; shift 2 ;;
    --run_vlm_teacher) RUN_VLM_TEACHER="$2"; shift 2 ;;
    --vlm_backend) VLM_BACKEND="$2"; shift 2 ;;
    --vlm_fallback_backend) VLM_FALLBACK_BACKEND="$2"; shift 2 ;;
    --score_profile) SCORE_PROFILE="$2"; shift 2 ;;
    --score_profile_overrides_json) SCORE_PROFILE_OVERRIDES_JSON="$2"; shift 2 ;;
    --safe_leftover_policy) SAFE_LEFTOVER_POLICY="$2"; shift 2 ;;
    --gpu_ids) GPU_IDS="$2"; shift 2 ;;
    --gpu_workers) GPU_WORKERS="$2"; shift 2 ;;
    --align_device) ALIGN_DEVICE="$2"; shift 2 ;;
    --aesthetic_device) AESTHETIC_DEVICE="$2"; shift 2 ;;
    --exp_batch_size) EXP_BATCH_SIZE="$2"; shift 2 ;;
    --exp_preprocess_workers) EXP_PREPROCESS_WORKERS="$2"; shift 2 ;;
    --exp_pin_memory) EXP_PIN_MEMORY="$2"; shift 2 ;;
    --c7_saliency_device) C7_SALIENCY_DEVICE="$2"; shift 2 ;;
    --c7_saliency_priority) C7_SALIENCY_PRIORITY="$2"; shift 2 ;;
    --gaic_caption_preset) GAIC_CAPTION_PRESET="$2"; shift 2 ;;
    --gaic_caption_backend) GAIC_CAPTION_BACKEND="$2"; shift 2 ;;
    --gaic_caption_model_id) GAIC_CAPTION_MODEL_ID="$2"; shift 2 ;;
    --gaic_caption_device) GAIC_CAPTION_DEVICE="$2"; shift 2 ;;
    --gaic_caption_dtype) GAIC_CAPTION_DTYPE="$2"; shift 2 ;;
    --gaic_caption_batch_size) GAIC_CAPTION_BATCH_SIZE="$2"; shift 2 ;;
    --gaic_caption_num_workers) GAIC_CAPTION_NUM_WORKERS="$2"; shift 2 ;;
    --gaic_caption_multi_gpu) GAIC_CAPTION_MULTI_GPU="$2"; shift 2 ;;
    -h|--help)
      sed -n '1,180p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

TRAIN_JSON="${PUBLIC_ROOT}/annotations_json/instances_train.json"
VAL_JSON="${PUBLIC_ROOT}/annotations_json/instances_val.json"
TEST_JSON="${PUBLIC_ROOT}/annotations_json/instances_test.json"

require_path() {
  if [ ! -e "$1" ]; then
    echo "[error] required path not found: $1"
    exit 1
  fi
}

print_command() {
  printf '[cmd]'
  printf ' %q' "$@"
  printf '\n'
}

run_or_print() {
  print_command "$@"
  if [ "$DRY_RUN" -eq 0 ]; then
    "$@"
  fi
}

split_title() {
  case "$1" in
    train) printf "Train2636" ;;
    val) printf "Val200" ;;
    test) printf "Test500" ;;
    *)
      echo "[error] unsupported split: $1" >&2
      return 1
      ;;
  esac
}

split_ref_json() {
  case "$1" in
    train) printf "%s" "$TRAIN_JSON" ;;
    val) printf "%s" "$VAL_JSON" ;;
    test) printf "%s" "$TEST_JSON" ;;
  esac
}

require_path "$PUBLIC_ROOT"
require_path "$TRAIN_JSON"
require_path "$VAL_JSON"
require_path "$TEST_JSON"

echo "========================================================"
echo " GAIC v2 official split teacher-label wrapper"
echo "========================================================"
echo " public_root       : $PUBLIC_ROOT"
echo " output_root       : $OUTPUT_ROOT"
echo " splits            : $SPLITS"
echo " run_tag_suffix    : $RUN_TAG_SUFFIX"
echo " server_mode       : $SERVER_MODE"
echo " dry_run           : $DRY_RUN"
echo " full-option       : C1=1 C4=0 C6=1 C7=1 captions=1 real_expensive=1"
echo " labels            : finalscore=$RUN_TRAINING_LABELS multimode=$RUN_MULTIMODE_TRAINING_LABELS"
echo " benchmark_on_test : $RUN_TEST_BENCHMARK_EVAL"
echo " benchmark_all     : $RUN_BENCHMARK_EVAL_ALL_SPLITS"
echo "========================================================"

for split in $SPLITS; do
  case "$split" in
    train|val|test) ;;
    *)
      echo "[error] unsupported split in --splits: $split"
      exit 1
      ;;
  esac

  image_root="${PUBLIC_ROOT}/images/${split}"
  split_json=$(split_ref_json "$split")
  title=$(split_title "$split")
  data_dir="${OUTPUT_ROOT}/${title}"
  bucket="gaic_v2_${split}"
  run_tag="gaic_v2_${split}_${RUN_TAG_SUFFIX}"
  run_benchmark=0
  if [ "$RUN_BENCHMARK_EVAL_ALL_SPLITS" -eq 1 ]; then
    run_benchmark=1
  fi
  if [ "$split" = "test" ] && [ "$RUN_TEST_BENCHMARK_EVAL" -eq 1 ]; then
    run_benchmark=1
  fi

  require_path "$image_root"
  require_path "$split_json"

  cmd=(
    bash src/scripts/run_gaic_to_teacher_e2e.sh
    --server_mode "$SERVER_MODE"
    --data_dir "$data_dir"
    --bucket "$bucket"
    --image_root "$image_root"
    --run_tag "$run_tag"
    --prepare_max_images "$PREPARE_MAX_IMAGES"
    --prepare_num_workers "$PREPARE_NUM_WORKERS"
    --link_mode "$LINK_MODE"
    --skip_existing "$SKIP_EXISTING"
    --skip_existing_links "$SKIP_EXISTING"
    --precompute_mode unified
    --run_c1 1
    --run_c4 0
    --run_c6 1
    --run_c7_saliency 1
    --c7_saliency_priority "$C7_SALIENCY_PRIORITY"
    --c7_saliency_device "$C7_SALIENCY_DEVICE"
    --gaic_generate_captions 1
    --gaic_caption_preset "$GAIC_CAPTION_PRESET"
    --gaic_caption_device "$GAIC_CAPTION_DEVICE"
    --gaic_caption_dtype "$GAIC_CAPTION_DTYPE"
    --gaic_caption_batch_size "$GAIC_CAPTION_BATCH_SIZE"
    --gaic_caption_num_workers "$GAIC_CAPTION_NUM_WORKERS"
    --gaic_caption_multi_gpu "$GAIC_CAPTION_MULTI_GPU"
    --use_real_expensive 1
    --align_device "$ALIGN_DEVICE"
    --aesthetic_device "$AESTHETIC_DEVICE"
    --exp_batch_size "$EXP_BATCH_SIZE"
    --exp_preprocess_workers "$EXP_PREPROCESS_WORKERS"
    --exp_pin_memory "$EXP_PIN_MEMORY"
    --run_vlm_teacher "$RUN_VLM_TEACHER"
    --vlm_backend "$VLM_BACKEND"
    --vlm_fallback_backend "$VLM_FALLBACK_BACKEND"
    --run_training_labels "$RUN_TRAINING_LABELS"
    --run_multimode_training_labels "$RUN_MULTIMODE_TRAINING_LABELS"
    --multimode_target_ars "$MULTIMODE_TARGET_ARS"
    --multimode_max_images "$MULTIMODE_MAX_IMAGES"
    --safe_leftover_policy "$SAFE_LEFTOVER_POLICY"
    --score_profile "$SCORE_PROFILE"
    --score_profile_overrides_json "$SCORE_PROFILE_OVERRIDES_JSON"
    --gaic_reference_json "${data_dir}/gaic_reference_available.json"
    --gaic_train_reference_json "$TRAIN_JSON"
    --gaic_val_reference_json "$VAL_JSON"
    --gaic_test_reference_json "$TEST_JSON"
    --run_gaic_benchmark_eval "$run_benchmark"
    --gaic_benchmark_sample_count 8
    --gaic_benchmark_max_images 0
  )
  if [ -n "$GPU_IDS" ]; then
    cmd+=(--gpu_ids "$GPU_IDS")
  fi
  if [ -n "$GAIC_CAPTION_BACKEND" ]; then
    cmd+=(--gaic_caption_backend "$GAIC_CAPTION_BACKEND")
  fi
  if [ -n "$GAIC_CAPTION_MODEL_ID" ]; then
    cmd+=(--gaic_caption_model_id "$GAIC_CAPTION_MODEL_ID")
  fi
  if [ "$GPU_WORKERS" -gt 0 ] 2>/dev/null; then
    cmd+=(--gpu_workers "$GPU_WORKERS")
  fi

  echo "[split] $split -> $data_dir"
  run_or_print "${cmd[@]}"
done
