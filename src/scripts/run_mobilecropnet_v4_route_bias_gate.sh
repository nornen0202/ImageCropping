#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/group-volume/users/jaden.ju/Sources/ImageCropping}"
TAG="${TAG:?TAG required}"
BASE="${BASE:-artifacts/mobilecropnet_v4/quality_first_20260428}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:?SOURCE_CHECKPOINT required}"
CALIBRATION_JSONL="${CALIBRATION_JSONL:-artifacts/mobilecropnet_v4/quality_first_20260428/blended_uctr80_public20_labels/train_blended_public_uctr.jsonl}"
EVAL_JSONL="${EVAL_JSONL:-data/SSTK/Full_10000/artifacts/training_labels_uctr_stage3_product_ar_260422_hashsplit_80_10_10_260422/val/train_conditional_detr_batch.jsonl}"
OUT_DIR="${OUT_DIR:-${BASE}/route_bias_calibration/${TAG}}"
HEAD_OUT="${HEAD_OUT:-${BASE}/head_audits/${TAG}}"
DIRECT_OUT="${DIRECT_OUT:-${BASE}/direct_eval/${TAG}_sortfix}"
QUAL_OUT="${QUAL_OUT:-${BASE}/qualitative/${TAG}_sortfix}"
STATUS_JSON="${STATUS_JSON:-${BASE}/gate_runs/${TAG}/status.json}"

SUBJECT_BOX_TARGET_SOURCE="${SUBJECT_BOX_TARGET_SOURCE:-support_latent}"
SUBJECT_VALID_THRESHOLD="${SUBJECT_VALID_THRESHOLD:-0.30}"
SUBJECT_VALID_POLICY="${SUBJECT_VALID_POLICY:-route_and_conf}"
BATCH_SIZE="${BATCH_SIZE:-24}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_CALIBRATION_ROWS="${MAX_CALIBRATION_ROWS:-12000}"
MAX_EVAL_ROWS="${MAX_EVAL_ROWS:-4200}"
MAX_GATE_ROWS="${MAX_GATE_ROWS:-3200}"
BIAS_SPAN="${BIAS_SPAN:-3.0}"
BIAS_STEP="${BIAS_STEP:-0.25}"
BIAS_ROUNDS="${BIAS_ROUNDS:-4}"
ROUTE_BALANCED_MIN="${ROUTE_BALANCED_MIN:-0.52}"
AUDIT_BATCH_SIZE="${AUDIT_BATCH_SIZE:-12}"
DIRECT_BATCH_SIZE="${DIRECT_BATCH_SIZE:-12}"

cd "${ROOT}"
mkdir -p "${OUT_DIR}" "$(dirname "${STATUS_JSON}")"

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
} | tee "${OUT_DIR}/environment.log"

/usr/local/bin/python3 - <<PY
import json
import time
from pathlib import Path

payload = {
    "state": "running",
    "phase": "route_bias_calibration",
    "run_name": ${TAG@Q},
    "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "source_checkpoint": ${SOURCE_CHECKPOINT@Q},
    "calibration_jsonl": ${CALIBRATION_JSONL@Q},
    "eval_jsonl": ${EVAL_JSONL@Q},
    "output_dir": ${OUT_DIR@Q},
}
Path(${STATUS_JSON@Q}).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
Path(${OUT_DIR@Q}, "route_bias_gate_launch_config.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

/usr/local/bin/python3 src/scripts/calibrate_mobilecropnet_v4_route_bias.py \
  --checkpoint "${SOURCE_CHECKPOINT}" \
  --calibration_jsonl "${CALIBRATION_JSONL}" \
  --eval_jsonl "${EVAL_JSONL}" \
  --output_dir "${OUT_DIR}" \
  --subject_box_target_source "${SUBJECT_BOX_TARGET_SOURCE}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --max_calibration_rows "${MAX_CALIBRATION_ROWS}" \
  --max_eval_rows "${MAX_EVAL_ROWS}" \
  --bias_span "${BIAS_SPAN}" \
  --bias_step "${BIAS_STEP}" \
  --rounds "${BIAS_ROUNDS}" \
  --device cuda \
  2>&1 | tee "${OUT_DIR}/route_bias_calibration.log"

/usr/local/bin/python3 - <<PY
import json
import time
from pathlib import Path

summary_path = Path(${OUT_DIR@Q}) / "route_bias_calibration_summary.json"
decision_path = Path(${OUT_DIR@Q}) / "route_bias_gate_decision.json"
status_path = Path(${STATUS_JSON@Q})
summary = json.loads(summary_path.read_text(encoding="utf-8"))
eval_before = float(summary["eval_before"]["balanced_accuracy"])
eval_after = float(summary["eval_after"]["balanced_accuracy"])
threshold = float(${ROUTE_BALANCED_MIN@Q})
passed = eval_after >= threshold
payload = {
    "state": "route_bias_gate_passed" if passed else "excluded_route_bias_gate",
    "phase": "route_bias_calibration",
    "run_name": ${TAG@Q},
    "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "source_checkpoint": ${SOURCE_CHECKPOINT@Q},
    "calibrated_checkpoint": str(Path(${OUT_DIR@Q}) / "route_bias_calibrated.pt"),
    "summary": str(summary_path),
    "route_balanced_before": eval_before,
    "route_balanced_after": eval_after,
    "route_balanced_min": threshold,
    "next_action": "run no-prior head/direct/qualitative gate" if passed else "exclude calibrated checkpoint; route remains below gate",
}
decision_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
Path(${OUT_DIR@Q}, ".route_bias_gate_pass").write_text("1\n" if passed else "0\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False))
PY

if [[ "$(tr -d '[:space:]' < "${OUT_DIR}/.route_bias_gate_pass")" != "1" ]]; then
  echo "route bias gate failed; skip no-prior composite gate" | tee -a "${OUT_DIR}/environment.log"
  exit 0
fi

RUN_TAG="${TAG}" \
CHECKPOINT="${OUT_DIR}/route_bias_calibrated.pt" \
ROUTE_EXPERT_MODE=none \
EVAL_JSONL="${EVAL_JSONL}" \
HEAD_OUT="${HEAD_OUT}" \
DIRECT_OUT="${DIRECT_OUT}" \
SUBJECT_VALID_THRESHOLD="${SUBJECT_VALID_THRESHOLD}" \
SUBJECT_VALID_POLICY="${SUBJECT_VALID_POLICY}" \
SUBJECT_BOX_TARGET_SOURCE="${SUBJECT_BOX_TARGET_SOURCE}" \
MAX_ROWS="${MAX_GATE_ROWS}" \
AUDIT_BATCH_SIZE="${AUDIT_BATCH_SIZE}" \
DIRECT_BATCH_SIZE="${DIRECT_BATCH_SIZE}" \
NUM_WORKERS="${NUM_WORKERS}" \
STATUS_JSON="${STATUS_JSON}" \
bash src/scripts/run_mobilecropnet_v4_composite_routeaware_gate.sh

/usr/local/bin/python3 src/scripts/build_mobilecropnet_v4_composite_qualitative_pack.py \
  --direct_predictions_jsonl "${DIRECT_OUT}/predictions.jsonl" \
  --output_dir "${QUAL_OUT}" \
  --max_per_bucket 24 \
  --max_total 120

date -u +"end_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "${OUT_DIR}/environment.log"
