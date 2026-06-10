#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${CHECKPOINT:?checkpoint required}"
VAL_JSONL="${VAL_JSONL:?val_jsonl required}"
TEST_JSONL="${TEST_JSONL:?test_jsonl required}"
METHOD_ID="${METHOD_ID:?method_id required}"
OUTPUT_DIR="${OUTPUT_DIR:?output_dir required}"

ROOT="${ROOT:-/group-volume/users/jaden.ju/Sources/ImageCropping}"
DEVICE="${DEVICE:-cuda}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
DIRECT_BATCH_SIZE="${DIRECT_BATCH_SIZE:-24}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DIRECT_SELECTION_POLICY="${DIRECT_SELECTION_POLICY:-proposal_topk_rerank}"
DIRECT_PROPOSAL_TOP_M="${DIRECT_PROPOSAL_TOP_M:-8}"
SUBJECT_VALID_THRESHOLD="${SUBJECT_VALID_THRESHOLD:-}"
SUBJECT_VALID_POLICY="${SUBJECT_VALID_POLICY:-confidence}"
SUBJECT_BOX_TARGET_SOURCE="${SUBJECT_BOX_TARGET_SOURCE:-}"
ROUTE_EXPERT_CHECKPOINT="${ROUTE_EXPERT_CHECKPOINT:-}"
ROUTE_EXPERT_CHECKPOINTS="${ROUTE_EXPERT_CHECKPOINTS:-}"
ROUTE_EXPERT_WEIGHTS="${ROUTE_EXPERT_WEIGHTS:-}"
RUNTIME_SUBJECT_PRIOR_MODE="${RUNTIME_SUBJECT_PRIOR_MODE:-none}"
EXACT_TARGET_AR_POSTPROCESS="${EXACT_TARGET_AR_POSTPROCESS:-1}"
STATUS_JSON="${OUTPUT_DIR}/residual_eval_status.json"

cd "${ROOT}"
mkdir -p "${OUTPUT_DIR}"

resolve_script() {
  local name="$1"
  if [[ -f "src/scripts/${name}" ]]; then
    printf '%s\n' "src/scripts/${name}"
    return 0
  fi
  if [[ -f "src/${name}" ]]; then
    printf '%s\n' "src/${name}"
    return 0
  fi
  echo "missing script: ${name}" >&2
  return 1
}

write_status() {
  local phase="$1"
  local state="$2"
  /usr/local/bin/python3 - <<PY
import json
import time
from pathlib import Path

root = Path(${OUTPUT_DIR@Q})
payload = {
    "phase": ${phase@Q},
    "state": ${state@Q},
    "checkpoint": ${CHECKPOINT@Q},
    "method_id": ${METHOD_ID@Q},
    "output_dir": ${OUTPUT_DIR@Q},
    "updated_at_unix": time.time(),
    "replay_eval_log": str(root / "replay_eval_test.log"),
    "head_analysis_log": str(root / "head_analysis.log"),
    "direct_val_log": str(root / "direct_val_${DIRECT_SELECTION_POLICY}.log"),
    "direct_test_log": str(root / "direct_test_${DIRECT_SELECTION_POLICY}.log"),
    "direct_top1_log": str(root / "direct_test_utility_top1.log"),
    "summary_json": str(root / "final_eval_bundle_summary.json"),
}
Path(${STATUS_JSON@Q}).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

trap 'write_status failed failed' ERR

EVAL_REPLAY_SCRIPT="$(resolve_script evaluate_mobilecropnet_v4.py)"
HEAD_ANALYSIS_SCRIPT="$(resolve_script analyze_mobilecropnet_v4_head_predictions.py)"
DIRECT_EVAL_SCRIPT="$(resolve_script evaluate_mobilecropnet_v4_product_ar_direct.py)"

EXTRA_REPLAY_ARGS=()
EXTRA_DIRECT_ARGS=()
if [[ -n "${SUBJECT_VALID_THRESHOLD}" ]]; then
  EXTRA_REPLAY_ARGS+=(--subject_valid_threshold "${SUBJECT_VALID_THRESHOLD}")
  EXTRA_DIRECT_ARGS+=(--subject_valid_threshold "${SUBJECT_VALID_THRESHOLD}")
fi
EXTRA_REPLAY_ARGS+=(--subject_valid_policy "${SUBJECT_VALID_POLICY}")
EXTRA_DIRECT_ARGS+=(--subject_valid_policy "${SUBJECT_VALID_POLICY}")
if [[ -n "${SUBJECT_BOX_TARGET_SOURCE}" ]]; then
  EXTRA_REPLAY_ARGS+=(--subject_box_target_source "${SUBJECT_BOX_TARGET_SOURCE}")
  EXTRA_DIRECT_ARGS+=(--subject_box_target_source "${SUBJECT_BOX_TARGET_SOURCE}")
fi
if [[ -n "${ROUTE_EXPERT_CHECKPOINT}" ]]; then
  EXTRA_REPLAY_ARGS+=(--route_expert_checkpoint "${ROUTE_EXPERT_CHECKPOINT}")
  EXTRA_DIRECT_ARGS+=(--route_expert_checkpoint "${ROUTE_EXPERT_CHECKPOINT}")
fi
if [[ -n "${ROUTE_EXPERT_CHECKPOINTS}" ]]; then
  read -r -a ROUTE_EXPERT_CHECKPOINT_ARRAY <<< "${ROUTE_EXPERT_CHECKPOINTS}"
  EXTRA_REPLAY_ARGS+=(--route_expert_checkpoints "${ROUTE_EXPERT_CHECKPOINT_ARRAY[@]}")
  EXTRA_DIRECT_ARGS+=(--route_expert_checkpoints "${ROUTE_EXPERT_CHECKPOINT_ARRAY[@]}")
fi
if [[ -n "${ROUTE_EXPERT_WEIGHTS}" ]]; then
  EXTRA_REPLAY_ARGS+=(--route_expert_weights "${ROUTE_EXPERT_WEIGHTS}")
  EXTRA_DIRECT_ARGS+=(--route_expert_weights "${ROUTE_EXPERT_WEIGHTS}")
fi
EXTRA_DIRECT_ARGS+=(--runtime_subject_prior_mode "${RUNTIME_SUBJECT_PRIOR_MODE}")
if [[ "${EXACT_TARGET_AR_POSTPROCESS}" == "1" || "${EXACT_TARGET_AR_POSTPROCESS}" == "true" || "${EXACT_TARGET_AR_POSTPROCESS}" == "yes" ]]; then
  EXTRA_DIRECT_ARGS+=(--exact_target_ar_postprocess)
else
  EXTRA_DIRECT_ARGS+=(--no-exact_target_ar_postprocess)
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
} | tee "${OUTPUT_DIR}/residual_eval_environment.log"

write_status py_compile running
/usr/local/bin/python3 -m py_compile \
  "${EVAL_REPLAY_SCRIPT}" \
  "${HEAD_ANALYSIS_SCRIPT}" \
  "${DIRECT_EVAL_SCRIPT}" \
  src/mobilecropnet_v4/eval_utils.py

write_status replay_eval_test running
/usr/local/bin/python3 "${EVAL_REPLAY_SCRIPT}" \
  --checkpoint "${CHECKPOINT}" \
  --eval_jsonl "${TEST_JSONL}" \
  --project_root . \
  --output_dir "${OUTPUT_DIR}/replay_eval_test" \
  --batch_size "${EVAL_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  "${EXTRA_REPLAY_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_DIR}/replay_eval_test.log"

write_status head_analysis running
/usr/local/bin/python3 "${HEAD_ANALYSIS_SCRIPT}" \
  --predictions_jsonl "${OUTPUT_DIR}/replay_eval_test/predictions.jsonl" \
  --eval_jsonl "${TEST_JSONL}" \
  --output_dir "${OUTPUT_DIR}/head_analysis" \
  2>&1 | tee "${OUTPUT_DIR}/head_analysis.log"

write_status direct_val running
/usr/local/bin/python3 "${DIRECT_EVAL_SCRIPT}" \
  --checkpoint "${CHECKPOINT}" \
  --eval_jsonl "${VAL_JSONL}" \
  --project_root . \
  --output_dir "${OUTPUT_DIR}/direct_val_${DIRECT_SELECTION_POLICY}" \
  --selection_policy "${DIRECT_SELECTION_POLICY}" \
  --proposal_top_m "${DIRECT_PROPOSAL_TOP_M}" \
  --batch_size "${DIRECT_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  "${EXTRA_DIRECT_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_DIR}/direct_val_${DIRECT_SELECTION_POLICY}.log"

write_status direct_test running
/usr/local/bin/python3 "${DIRECT_EVAL_SCRIPT}" \
  --checkpoint "${CHECKPOINT}" \
  --eval_jsonl "${TEST_JSONL}" \
  --project_root . \
  --output_dir "${OUTPUT_DIR}/direct_test_${DIRECT_SELECTION_POLICY}" \
  --selection_policy "${DIRECT_SELECTION_POLICY}" \
  --proposal_top_m "${DIRECT_PROPOSAL_TOP_M}" \
  --batch_size "${DIRECT_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  "${EXTRA_DIRECT_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_DIR}/direct_test_${DIRECT_SELECTION_POLICY}.log"

write_status direct_test_utility_top1 running
/usr/local/bin/python3 "${DIRECT_EVAL_SCRIPT}" \
  --checkpoint "${CHECKPOINT}" \
  --eval_jsonl "${TEST_JSONL}" \
  --project_root . \
  --output_dir "${OUTPUT_DIR}/direct_test_utility_top1" \
  --selection_policy utility_top1 \
  --proposal_top_m "${DIRECT_PROPOSAL_TOP_M}" \
  --batch_size "${DIRECT_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  "${EXTRA_DIRECT_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_DIR}/direct_test_utility_top1.log"

write_status summarize running
/usr/local/bin/python3 - <<PY
import json
from pathlib import Path

root = Path("${OUTPUT_DIR}")
def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

summary = {
    "checkpoint": "${CHECKPOINT}",
    "val_jsonl": "${VAL_JSONL}",
    "test_jsonl": "${TEST_JSONL}",
    "method_id": "${METHOD_ID}",
    "residual_only": True,
    "artifacts": {
        "public_export_summary": read_json(root / "public_benchmark" / "export_summary.json"),
        "public_eval_summary": read_json(root / "public_benchmark" / "eval" / "summary.json"),
        "replay_eval_test": read_json(root / "replay_eval_test" / "metrics.json"),
        "head_analysis": read_json(root / "head_analysis" / "head_analysis_summary.json"),
        "direct_val": read_json(root / "direct_val_${DIRECT_SELECTION_POLICY}" / "metrics.json"),
        "direct_test": read_json(root / "direct_test_${DIRECT_SELECTION_POLICY}" / "metrics.json"),
        "direct_test_utility_top1": read_json(root / "direct_test_utility_top1" / "metrics.json"),
    },
}
(root / "final_eval_bundle_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps({
    "method_id": summary["method_id"],
    "summary_json": str(root / "final_eval_bundle_summary.json"),
    "replay_eval_test": str(root / "replay_eval_test" / "metrics.json"),
    "direct_test": str(root / "direct_test_${DIRECT_SELECTION_POLICY}" / "metrics.json"),
}, ensure_ascii=False, indent=2))
PY

date -u +"end_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "${OUTPUT_DIR}/residual_eval_environment.log"
write_status completed completed
