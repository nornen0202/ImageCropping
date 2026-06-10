#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/group-volume/users/jaden.ju/Sources/ImageCropping}"
CHECKPOINT="${CHECKPOINT:-artifacts/mobilecropnet_v4/quality_first_20260427/route_smokes/sstk_uctr_supporttarget/mcn-q-b448-intprior-auxfine-nomargin-r2-ip134-20260427/best.pt}"
ROUTE_EXPERT_CHECKPOINT="${ROUTE_EXPERT_CHECKPOINT:-artifacts/mobilecropnet_v4/quality_first_20260427/route_experts/swinv2b384_uctr_r1-20260427/best.pt}"
ROUTE_EXPERT_CHECKPOINTS="${ROUTE_EXPERT_CHECKPOINTS:-}"
ROUTE_EXPERT_WEIGHTS="${ROUTE_EXPERT_WEIGHTS:-}"
ROUTE_EXPERT_MODE="${ROUTE_EXPERT_MODE:-single}"
EVAL_JSONL="${EVAL_JSONL:-data/SSTK/Full_10000/artifacts/training_labels_uctr_stage3_product_ar_260422_hashsplit_80_10_10_260422/val/train_conditional_detr_batch.jsonl}"
RUN_TAG="${RUN_TAG:-composite_fullr2_swinroute_vthr030_routeand_r1-20260427}"
HEAD_OUT="${HEAD_OUT:-artifacts/mobilecropnet_v4/quality_first_20260427/head_audits/${RUN_TAG}}"
DIRECT_OUT="${DIRECT_OUT:-artifacts/mobilecropnet_v4/quality_first_20260427/direct_eval/${RUN_TAG}_sortfix}"
SUBJECT_VALID_THRESHOLD="${SUBJECT_VALID_THRESHOLD:-0.30}"
SUBJECT_VALID_POLICY="${SUBJECT_VALID_POLICY:-route_and_conf}"
SUBJECT_BOX_TARGET_SOURCE="${SUBJECT_BOX_TARGET_SOURCE:-support_latent}"
USE_DECISION_SOURCE_ACTION_GATE="${USE_DECISION_SOURCE_ACTION_GATE:-0}"
MAX_ROWS="${MAX_ROWS:-3200}"
AUDIT_BATCH_SIZE="${AUDIT_BATCH_SIZE:-32}"
DIRECT_BATCH_SIZE="${DIRECT_BATCH_SIZE:-24}"
NUM_WORKERS="${NUM_WORKERS:-4}"
STATUS_JSON="${STATUS_JSON:-artifacts/mobilecropnet_v4/quality_first_20260427/head_audits/${RUN_TAG}_launcher_status.json}"

cd "${ROOT}"
mkdir -p "$(dirname "${STATUS_JSON}")" "${HEAD_OUT}" "${DIRECT_OUT}"

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
    "checkpoint": ${CHECKPOINT@Q},
    "route_expert_checkpoint": ${ROUTE_EXPERT_CHECKPOINT@Q},
    "route_expert_checkpoints": ${ROUTE_EXPERT_CHECKPOINTS@Q},
    "route_expert_weights": ${ROUTE_EXPERT_WEIGHTS@Q},
    "route_expert_mode": ${ROUTE_EXPERT_MODE@Q},
    "eval_jsonl": ${EVAL_JSONL@Q},
    "head_out": ${HEAD_OUT@Q},
    "direct_out": ${DIRECT_OUT@Q},
    "subject_valid_threshold": ${SUBJECT_VALID_THRESHOLD@Q},
    "subject_valid_policy": ${SUBJECT_VALID_POLICY@Q},
    "updated_at_unix": time.time(),
}
Path(${STATUS_JSON@Q}).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

trap 'write_status failed failed' ERR

ROUTE_EXPERT_ARGS=()
if [[ "${ROUTE_EXPERT_MODE}" == "none" ]]; then
  ROUTE_EXPERT_ARGS=()
elif [[ -n "${ROUTE_EXPERT_CHECKPOINTS}" ]]; then
  ROUTE_EXPERT_ARGS+=(--route_expert_checkpoints)
  read -r -a ROUTE_EXPERT_PATH_ARRAY <<< "${ROUTE_EXPERT_CHECKPOINTS}"
  ROUTE_EXPERT_ARGS+=("${ROUTE_EXPERT_PATH_ARRAY[@]}")
else
  ROUTE_EXPERT_ARGS+=(--route_expert_checkpoint "${ROUTE_EXPERT_CHECKPOINT}")
fi
if [[ -n "${ROUTE_EXPERT_WEIGHTS}" ]]; then
  ROUTE_EXPERT_ARGS+=(--route_expert_weights "${ROUTE_EXPERT_WEIGHTS}")
fi

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
} | tee "${HEAD_OUT}/launcher_environment.log"

write_status py_compile running
/usr/local/bin/python3 -m py_compile \
  src/mobilecropnet_v4/eval_utils.py \
  src/scripts/audit_mobilecropnet_v4_release_heads.py \
  src/scripts/evaluate_mobilecropnet_v4_product_ar_direct.py

write_status audit running
/usr/local/bin/python3 src/scripts/audit_mobilecropnet_v4_release_heads.py \
  --checkpoint "${CHECKPOINT}" \
  --eval_jsonl "${EVAL_JSONL}" \
  --output_dir "${HEAD_OUT}" \
  --batch_size "${AUDIT_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --max_rows "${MAX_ROWS}" \
  --valid_conf_threshold "${SUBJECT_VALID_THRESHOLD}" \
  --subject_valid_policy "${SUBJECT_VALID_POLICY}" \
  --subject_box_target_source "${SUBJECT_BOX_TARGET_SOURCE}" \
  "${ROUTE_EXPERT_ARGS[@]}" \
  --device cuda \
  2>&1 | tee "${HEAD_OUT}/run.log"

write_status direct running
/usr/local/bin/python3 src/scripts/evaluate_mobilecropnet_v4_product_ar_direct.py \
  --checkpoint "${CHECKPOINT}" \
  --eval_jsonl "${EVAL_JSONL}" \
  --project_root . \
  --output_dir "${DIRECT_OUT}" \
  --selection_policy proposal_topk_rerank \
  --proposal_top_m 8 \
  --runtime_subject_prior_mode none \
  --exact_target_ar_postprocess \
  $(if [[ "${USE_DECISION_SOURCE_ACTION_GATE,,}" =~ ^(1|true|yes|on)$ ]]; then echo "--use_decision_source_action_gate"; fi) \
  --subject_valid_threshold "${SUBJECT_VALID_THRESHOLD}" \
  --subject_valid_policy "${SUBJECT_VALID_POLICY}" \
  --subject_box_target_source "${SUBJECT_BOX_TARGET_SOURCE}" \
  "${ROUTE_EXPERT_ARGS[@]}" \
  --batch_size "${DIRECT_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --max_rows "${MAX_ROWS}" \
  --device cuda \
  2>&1 | tee "${DIRECT_OUT}/run.log"

write_status completed completed
date -u +"end_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "${HEAD_OUT}/launcher_environment.log"
