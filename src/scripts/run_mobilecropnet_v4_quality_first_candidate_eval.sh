#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/group-volume/users/jaden.ju/Sources/ImageCropping}"
EVAL_KIND="${EVAL_KIND:-final_bundle}"

CHECKPOINT="${CHECKPOINT:-artifacts/mobilecropnet_v4/quality_first_20260428/embedded_route_expert_checkpoints/mcn-q-blendrepair-rankerpolicy-r1-ip134-20260428-swinroute-embedded.pt}"
METHOD_ID="${METHOD_ID:-mcn-q-blendrepair-rankerpolicy-swinembed-vthr030-20260428}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts/mobilecropnet_v4/quality_first_20260428/final_eval_bundle/${METHOD_ID}}"
VAL_JSONL="${VAL_JSONL:-data/SSTK/Full_10000/artifacts/training_labels_uctr_stage3_product_ar_260422_hashsplit_80_10_10_260422/val/train_conditional_detr_batch.jsonl}"
TEST_JSONL="${TEST_JSONL:-data/SSTK/Full_10000/artifacts/training_labels_uctr_stage3_product_ar_260422_hashsplit_80_10_10_260422/test/train_conditional_detr_batch.jsonl}"
PUBLIC_TASK_MANIFEST_JSONL="${PUBLIC_TASK_MANIFEST_JSONL:-artifacts/unified_public_benchmark_20260420/stage_full/benchmark_task_manifest.jsonl}"
GAIC_ANNOTATIONS_JSON="${GAIC_ANNOTATIONS_JSON:-data/Publics/GAIC_v2/annotations_json/instances_test.json}"
GAIC_IMAGE_ROOTS="${GAIC_IMAGE_ROOTS:-data/Publics/GAIC_v2/images/test data/Publics/GAIC/images/test}"

DEVICE="${DEVICE:-cuda}"
PUBLIC_BATCH_SIZE="${PUBLIC_BATCH_SIZE:-64}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-24}"
DIRECT_BATCH_SIZE="${DIRECT_BATCH_SIZE:-24}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DIRECT_SELECTION_POLICY="${DIRECT_SELECTION_POLICY:-proposal_topk_rerank}"
SUBJECT_BOX_TARGET_SOURCE="${SUBJECT_BOX_TARGET_SOURCE:-support_latent}"
SUBJECT_VALID_THRESHOLD="${SUBJECT_VALID_THRESHOLD:-0.3}"
SUBJECT_VALID_POLICY="${SUBJECT_VALID_POLICY:-route_and_conf}"
RUNTIME_SUBJECT_PRIOR_MODE="${RUNTIME_SUBJECT_PRIOR_MODE:-none}"
EXACT_TARGET_AR_POSTPROCESS="${EXACT_TARGET_AR_POSTPROCESS:-1}"
ROUTE_EXPERT_CHECKPOINT="${ROUTE_EXPERT_CHECKPOINT:-}"
ROUTE_EXPERT_CHECKPOINTS="${ROUTE_EXPERT_CHECKPOINTS:-}"
ROUTE_EXPERT_WEIGHTS="${ROUTE_EXPERT_WEIGHTS:-}"
LATENCY_BATCH_SIZE="${LATENCY_BATCH_SIZE:-1}"
LATENCY_WARMUP="${LATENCY_WARMUP:-30}"
LATENCY_ITERATIONS="${LATENCY_ITERATIONS:-120}"
LATENCY_CANDIDATE_COUNTS="${LATENCY_CANDIDATE_COUNTS:-auto,86}"
LATENCY_TARGET_AR="${LATENCY_TARGET_AR:-FREE}"
LATENCY_AMP="${LATENCY_AMP:-1}"
QUAL_MAX_PER_BUCKET="${QUAL_MAX_PER_BUCKET:-24}"
QUAL_MAX_TOTAL="${QUAL_MAX_TOTAL:-120}"
QUAL_RENDER_OVERLAYS="${QUAL_RENDER_OVERLAYS:-0}"

cd "${ROOT}"
mkdir -p "${OUTPUT_DIR}"

write_status() {
  local phase="$1"
  local state="$2"
  /usr/local/bin/python3 - <<PY
import json
import time
from pathlib import Path

payload = {
    "phase": ${phase@Q},
    "state": ${state@Q},
    "eval_kind": ${EVAL_KIND@Q},
    "checkpoint": ${CHECKPOINT@Q},
    "method_id": ${METHOD_ID@Q},
    "output_dir": ${OUTPUT_DIR@Q},
    "updated_at_unix": time.time(),
}
Path(${OUTPUT_DIR@Q}, "runner_status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

trap 'write_status failed failed' ERR

{
  date -u +"start_utc=%Y-%m-%dT%H:%M:%SZ"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  /usr/local/bin/python3 - <<'PY'
import torch
print("torch_version", torch.__version__)
print("cuda_available", torch.cuda.is_available())
assert torch.cuda.is_available()
print("device_name", torch.cuda.get_device_name(0))
PY
} | tee "${OUTPUT_DIR}/${EVAL_KIND}_environment.log"

write_status "${EVAL_KIND}" running

case "${EVAL_KIND}" in
  final_bundle)
    CHECKPOINT="${CHECKPOINT}" \
    VAL_JSONL="${VAL_JSONL}" \
    TEST_JSONL="${TEST_JSONL}" \
    PUBLIC_TASK_MANIFEST_JSONL="${PUBLIC_TASK_MANIFEST_JSONL}" \
    METHOD_ID="${METHOD_ID}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    DEVICE="${DEVICE}" \
    PUBLIC_BATCH_SIZE="${PUBLIC_BATCH_SIZE}" \
    EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE}" \
    DIRECT_BATCH_SIZE="${DIRECT_BATCH_SIZE}" \
    NUM_WORKERS="${NUM_WORKERS}" \
    SUBJECT_BOX_TARGET_SOURCE="${SUBJECT_BOX_TARGET_SOURCE}" \
    SUBJECT_VALID_THRESHOLD="${SUBJECT_VALID_THRESHOLD}" \
    SUBJECT_VALID_POLICY="${SUBJECT_VALID_POLICY}" \
    RUNTIME_SUBJECT_PRIOR_MODE="${RUNTIME_SUBJECT_PRIOR_MODE}" \
    EXACT_TARGET_AR_POSTPROCESS="${EXACT_TARGET_AR_POSTPROCESS}" \
      bash src/scripts/run_mobilecropnet_v4_shortlist_eval_bundle.sh
    ;;
  gaic_official)
    GAIC_OUT="${OUTPUT_DIR}/gaic_official_test"
    mkdir -p "${GAIC_OUT}"
    # shellcheck disable=SC2206
    GAIC_ROOT_ARRAY=(${GAIC_IMAGE_ROOTS})
    /usr/local/bin/python3 src/scripts/evaluate_mobilecropnet_v4_gaic_benchmark.py \
      --checkpoint "${CHECKPOINT}" \
      --annotations_json "${GAIC_ANNOTATIONS_JSON}" \
      --image_roots "${GAIC_ROOT_ARRAY[@]}" \
      --output_dir "${GAIC_OUT}" \
      --model_name "${METHOD_ID}" \
      --target_ar FREE \
      --device "${DEVICE}" \
      --progress_json "${GAIC_OUT}/progress.json" \
      2>&1 | tee "${GAIC_OUT}/run.log"
    ;;
  latency)
    LATENCY_OUT="${OUTPUT_DIR}/latency"
    mkdir -p "${LATENCY_OUT}"
    /usr/local/bin/python3 src/scripts/benchmark_mobilecropnet_v4_inference.py \
      --checkpoint "${METHOD_ID}=${CHECKPOINT}" \
      --output_json "${LATENCY_OUT}/latency.json" \
      --output_md "${LATENCY_OUT}/LATENCY_BENCHMARK.md" \
      --device "${DEVICE}" \
      --target_ar "${LATENCY_TARGET_AR}" \
      --candidate_counts "${LATENCY_CANDIDATE_COUNTS}" \
      --batch_size "${LATENCY_BATCH_SIZE}" \
      --warmup "${LATENCY_WARMUP}" \
      --iterations "${LATENCY_ITERATIONS}" \
      --amp "${LATENCY_AMP}" \
      --runtime_no_prior \
      2>&1 | tee "${LATENCY_OUT}/run.log"
    ;;
  qualitative)
    QUAL_OUT="${OUTPUT_DIR}/qualitative_test"
    DIRECT_PREDICTIONS_JSONL="${QUAL_DIRECT_PREDICTIONS_JSONL:-${OUTPUT_DIR}/direct_test_${DIRECT_SELECTION_POLICY}/predictions.jsonl}"
    mkdir -p "${QUAL_OUT}"
    QUAL_ARGS=(
      --direct_predictions_jsonl "${DIRECT_PREDICTIONS_JSONL}"
      --output_dir "${QUAL_OUT}"
      --checkpoint "${CHECKPOINT}"
      --subject_valid_threshold "${SUBJECT_VALID_THRESHOLD}"
      --max_per_bucket "${QUAL_MAX_PER_BUCKET}"
      --max_total "${QUAL_MAX_TOTAL}"
      --device "${DEVICE}"
    )
    if [[ -n "${ROUTE_EXPERT_CHECKPOINT}" ]]; then
      QUAL_ARGS+=(--route_expert_checkpoint "${ROUTE_EXPERT_CHECKPOINT}")
    fi
    if [[ -n "${ROUTE_EXPERT_CHECKPOINTS}" ]]; then
      read -r -a QUAL_ROUTE_EXPERT_CHECKPOINT_ARRAY <<< "${ROUTE_EXPERT_CHECKPOINTS}"
      QUAL_ARGS+=(--route_expert_checkpoints "${QUAL_ROUTE_EXPERT_CHECKPOINT_ARRAY[@]}")
    fi
    if [[ -n "${ROUTE_EXPERT_WEIGHTS}" ]]; then
      QUAL_ARGS+=(--route_expert_weights "${ROUTE_EXPERT_WEIGHTS}")
    fi
    if [[ "${QUAL_RENDER_OVERLAYS}" == "1" || "${QUAL_RENDER_OVERLAYS}" == "true" || "${QUAL_RENDER_OVERLAYS}" == "yes" ]]; then
      QUAL_ARGS+=(--render_overlays)
    fi
    /usr/local/bin/python3 src/scripts/build_mobilecropnet_v4_composite_qualitative_pack.py \
      "${QUAL_ARGS[@]}" \
      2>&1 | tee "${QUAL_OUT}/run.log"
    ;;
  release_full)
    export ROOT
    export CHECKPOINT METHOD_ID OUTPUT_DIR
    export VAL_JSONL TEST_JSONL PUBLIC_TASK_MANIFEST_JSONL
    export GAIC_ANNOTATIONS_JSON GAIC_IMAGE_ROOTS
    export DEVICE PUBLIC_BATCH_SIZE EVAL_BATCH_SIZE DIRECT_BATCH_SIZE NUM_WORKERS
    export DIRECT_SELECTION_POLICY SUBJECT_BOX_TARGET_SOURCE SUBJECT_VALID_THRESHOLD SUBJECT_VALID_POLICY
    export RUNTIME_SUBJECT_PRIOR_MODE EXACT_TARGET_AR_POSTPROCESS
    export ROUTE_EXPERT_CHECKPOINT ROUTE_EXPERT_CHECKPOINTS ROUTE_EXPERT_WEIGHTS
    export LATENCY_BATCH_SIZE LATENCY_WARMUP LATENCY_ITERATIONS LATENCY_CANDIDATE_COUNTS LATENCY_TARGET_AR LATENCY_AMP
    export QUAL_MAX_PER_BUCKET QUAL_MAX_TOTAL QUAL_RENDER_OVERLAYS
    export QUAL_DIRECT_PREDICTIONS_JSONL="${QUAL_DIRECT_PREDICTIONS_JSONL:-${OUTPUT_DIR}/direct_test_${DIRECT_SELECTION_POLICY}/predictions.jsonl}"
    EVAL_KIND=final_bundle bash "$0"
    EVAL_KIND=gaic_official bash "$0"
    EVAL_KIND=latency bash "$0"
    EVAL_KIND=qualitative bash "$0"
    ;;
  *)
    echo "unsupported EVAL_KIND=${EVAL_KIND}; expected final_bundle, gaic_official, latency, qualitative, or release_full" >&2
    exit 2
    ;;
esac

write_status "${EVAL_KIND}" completed
date -u +"end_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "${OUTPUT_DIR}/${EVAL_KIND}_environment.log"
