#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:?run dir required}"
VAL_JSONL="${2:?val jsonl required}"
TEST_JSONL="${3:?test jsonl required}"

ROOT="${ROOT:-/group-volume/users/jaden.ju/Sources/ImageCropping}"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python3}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-24}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DIRECT_SELECTION_POLICY="${DIRECT_SELECTION_POLICY:-proposal_topk_rerank}"
DIRECT_PROPOSAL_TOP_M="${DIRECT_PROPOSAL_TOP_M:-8}"
VIZ_SAMPLE_SIZE="${VIZ_SAMPLE_SIZE:-64}"
VIZ_MAX_SIDE="${VIZ_MAX_SIDE:-1500}"

cd "${ROOT}"

CHECKPOINT="${CHECKPOINT_PATH:-${RUN_DIR}/best.pt}"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "missing checkpoint: ${CHECKPOINT}" >&2
  exit 2
fi
OUTPUT_TAG="${OUTPUT_TAG:-}"
OUTPUT_SUFFIX=""
if [[ -n "${OUTPUT_TAG}" ]]; then
  OUTPUT_SUFFIX="_${OUTPUT_TAG}"
fi
DIRECT_VAL_DIR="${RUN_DIR}/direct_val_${DIRECT_SELECTION_POLICY}${OUTPUT_SUFFIX}"
DIRECT_TEST_DIR="${RUN_DIR}/direct_test_${DIRECT_SELECTION_POLICY}${OUTPUT_SUFFIX}"
DIRECT_TEST_UTILITY_DIR="${RUN_DIR}/direct_test_utility_top1${OUTPUT_SUFFIX}"
DIRECT_VIZ_DIR="${RUN_DIR}/direct_viz_test_${DIRECT_SELECTION_POLICY}${OUTPUT_SUFFIX}"

{
  date -u +"post_eval_start_utc=%Y-%m-%dT%H:%M:%SZ"
  echo "run_dir=${RUN_DIR}"
  echo "checkpoint=${CHECKPOINT}"
  echo "val_jsonl=${VAL_JSONL}"
  echo "test_jsonl=${TEST_JSONL}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  fi
} | tee "${RUN_DIR}/subject_box_post_eval.log"

"${PYTHON_BIN}" src/scripts/evaluate_mobilecropnet_v4_product_ar_direct.py \
  --checkpoint "${CHECKPOINT}" \
  --eval_jsonl "${VAL_JSONL}" \
  --project_root . \
  --output_dir "${DIRECT_VAL_DIR}" \
  --selection_policy "${DIRECT_SELECTION_POLICY}" \
  --proposal_top_m "${DIRECT_PROPOSAL_TOP_M}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  2>&1 | tee -a "${RUN_DIR}/subject_box_post_eval.log"

"${PYTHON_BIN}" src/scripts/analyze_mobilecropnet_v4_subject_box_predictions.py \
  --predictions_jsonl "${DIRECT_VAL_DIR}/predictions.jsonl" \
  --output_dir "${DIRECT_VAL_DIR}/subject_box_calibration" \
  2>&1 | tee -a "${RUN_DIR}/subject_box_post_eval.log"

"${PYTHON_BIN}" src/scripts/evaluate_mobilecropnet_v4_product_ar_direct.py \
  --checkpoint "${CHECKPOINT}" \
  --eval_jsonl "${TEST_JSONL}" \
  --project_root . \
  --output_dir "${DIRECT_TEST_DIR}" \
  --selection_policy "${DIRECT_SELECTION_POLICY}" \
  --proposal_top_m "${DIRECT_PROPOSAL_TOP_M}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  2>&1 | tee -a "${RUN_DIR}/subject_box_post_eval.log"

"${PYTHON_BIN}" src/scripts/analyze_mobilecropnet_v4_subject_box_predictions.py \
  --predictions_jsonl "${DIRECT_TEST_DIR}/predictions.jsonl" \
  --output_dir "${DIRECT_TEST_DIR}/subject_box_calibration" \
  2>&1 | tee -a "${RUN_DIR}/subject_box_post_eval.log"

"${PYTHON_BIN}" src/scripts/evaluate_mobilecropnet_v4_product_ar_direct.py \
  --checkpoint "${CHECKPOINT}" \
  --eval_jsonl "${TEST_JSONL}" \
  --project_root . \
  --output_dir "${DIRECT_TEST_UTILITY_DIR}" \
  --selection_policy utility_top1 \
  --proposal_top_m "${DIRECT_PROPOSAL_TOP_M}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  2>&1 | tee -a "${RUN_DIR}/subject_box_post_eval.log"

"${PYTHON_BIN}" src/scripts/analyze_mobilecropnet_v4_subject_box_predictions.py \
  --predictions_jsonl "${DIRECT_TEST_UTILITY_DIR}/predictions.jsonl" \
  --output_dir "${DIRECT_TEST_UTILITY_DIR}/subject_box_calibration" \
  2>&1 | tee -a "${RUN_DIR}/subject_box_post_eval.log"

"${PYTHON_BIN}" src/scripts/visualize_mobilecropnet_v4_predictions.py \
  --predictions_jsonl "${DIRECT_TEST_DIR}/predictions.jsonl" \
  --output_dir "${DIRECT_VIZ_DIR}" \
  --sample_size "${VIZ_SAMPLE_SIZE}" \
  --max_side "${VIZ_MAX_SIDE}" \
  2>&1 | tee -a "${RUN_DIR}/subject_box_post_eval.log"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

run = Path("${RUN_DIR}")
def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

summary = {
    "run_dir": "${RUN_DIR}",
    "checkpoint": "${CHECKPOINT}",
    "output_tag": "${OUTPUT_TAG}",
    "direct_val": read_json(Path("${DIRECT_VAL_DIR}") / "metrics.json"),
    "direct_val_subject_box_calibration": read_json(Path("${DIRECT_VAL_DIR}") / "subject_box_calibration" / "subject_box_confidence_calibration.json"),
    "direct_test": read_json(Path("${DIRECT_TEST_DIR}") / "metrics.json"),
    "direct_test_subject_box_calibration": read_json(Path("${DIRECT_TEST_DIR}") / "subject_box_calibration" / "subject_box_confidence_calibration.json"),
    "direct_test_utility_top1": read_json(Path("${DIRECT_TEST_UTILITY_DIR}") / "metrics.json"),
    "direct_test_utility_top1_subject_box_calibration": read_json(Path("${DIRECT_TEST_UTILITY_DIR}") / "subject_box_calibration" / "subject_box_confidence_calibration.json"),
    "direct_viz_manifest": read_json(Path("${DIRECT_VIZ_DIR}") / "visualization_manifest.json"),
}
output_tag = "${OUTPUT_TAG}"
summary_name = "subject_box_post_eval_summary.json" if not output_tag else f"subject_box_post_eval_summary_{output_tag}.json"
(run / summary_name).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({
    "run_dir": summary["run_dir"],
    "direct_test_metrics": str(Path("${DIRECT_TEST_DIR}") / "metrics.json"),
    "direct_viz": str(Path("${DIRECT_VIZ_DIR}") / "contact_sheet.png"),
}, ensure_ascii=False, indent=2))
PY

date -u +"post_eval_end_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "${RUN_DIR}/subject_box_post_eval.log"
