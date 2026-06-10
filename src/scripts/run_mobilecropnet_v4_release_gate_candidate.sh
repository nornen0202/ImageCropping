#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:?variant required}"
PROFILE="${2:?profile required}"
RUN_NAME="${3:?run name required}"

ROOT="${ROOT:-/group-volume/users/jaden.ju/Sources/ImageCropping}"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python3}"
OUT_BASE="${OUT_BASE:-artifacts/mobilecropnet_v4/release_gate_v1_20260424}"
TRAIN_LABEL_DIR="${TRAIN_LABEL_DIR:?TRAIN_LABEL_DIR required}"
VAL_LABEL_DIR="${VAL_LABEL_DIR:?VAL_LABEL_DIR required}"
TEST_LABEL_DIR="${TEST_LABEL_DIR:?TEST_LABEL_DIR required}"
PUBLIC_TASK_MANIFEST_JSONL="${PUBLIC_TASK_MANIFEST_JSONL:-artifacts/unified_public_benchmark_20260420/stage_full/benchmark_task_manifest.jsonl}"

FULL_RUN_BASE="${OUT_BASE}/full_runs/sstk_uctr"
RUN_DIR="${FULL_RUN_BASE}/${RUN_NAME}"
BUNDLE_DIR="${OUT_BASE}/final_bundle/${RUN_NAME}"
STATUS_JSON="${OUT_BASE}/status/${RUN_NAME}.json"

cd "${ROOT}"
mkdir -p "$(dirname "${STATUS_JSON}")" "${RUN_DIR}" "${BUNDLE_DIR}"

write_status() {
  local phase="$1"
  local state="$2"
  "${PYTHON_BIN}" - <<PY
import json
import time
from pathlib import Path

payload = {
    "phase": ${phase@Q},
    "state": ${state@Q},
    "last_update_time_unix": time.time(),
    "variant": ${VARIANT@Q},
    "profile": ${PROFILE@Q},
    "run_name": ${RUN_NAME@Q},
    "run_dir": ${RUN_DIR@Q},
    "bundle_dir": ${BUNDLE_DIR@Q},
    "checkpoint": str(Path(${RUN_DIR@Q}) / "best.pt"),
    "train_status": str(Path(${RUN_DIR@Q}) / "status.json"),
    "bundle_status": str(Path(${BUNDLE_DIR@Q}) / "status.json"),
    "summary_json": str(Path(${OUT_BASE@Q}) / "release_gate_candidate_summary.json"),
}
Path(${STATUS_JSON@Q}).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

trap 'write_status failed failed' ERR

{
  date -u +"start_utc=%Y-%m-%dT%H:%M:%SZ"
  nvidia-smi --query-gpu=name,memory.total,driver_version,utilization.gpu,memory.used --format=csv,noheader
  "${PYTHON_BIN}" -m py_compile \
    src/mobilecropnet_v4/model.py \
    src/mobilecropnet_v4/eval_utils.py \
    src/scripts/train_mobilecropnet_v4.py
} | tee "${RUN_DIR}/release_gate_environment.log"

write_status train_eval running
OUT_BASE="${FULL_RUN_BASE}" \
PYTHON_BIN="${PYTHON_BIN}" \
ROOT="${ROOT}" \
bash src/scripts/run_mobilecropnet_v4_single_profile_variant.sh \
  "${VARIANT}" \
  "${PROFILE}" \
  "${RUN_NAME}" \
  2>&1 | tee "${RUN_DIR}/release_gate_train_eval.log"

write_status final_bundle running
CHECKPOINT="${RUN_DIR}/best.pt" \
VAL_JSONL="${VAL_LABEL_DIR}/train_conditional_detr_batch.jsonl" \
TEST_JSONL="${TEST_LABEL_DIR}/train_conditional_detr_batch.jsonl" \
PUBLIC_TASK_MANIFEST_JSONL="${PUBLIC_TASK_MANIFEST_JSONL}" \
METHOD_ID="sstk_uctr__${VARIANT}__${PROFILE}" \
OUTPUT_DIR="${BUNDLE_DIR}" \
ROOT="${ROOT}" \
DEVICE=cuda \
PUBLIC_BATCH_SIZE="${PUBLIC_BATCH_SIZE:-64}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}" \
DIRECT_BATCH_SIZE="${DIRECT_BATCH_SIZE:-24}" \
NUM_WORKERS="${NUM_WORKERS:-4}" \
DIRECT_SELECTION_POLICY=proposal_topk_rerank \
DIRECT_PROPOSAL_TOP_M=8 \
bash src/scripts/run_mobilecropnet_v4_shortlist_eval_bundle.sh \
  2>&1 | tee "${BUNDLE_DIR}/release_gate_final_bundle.log"

write_status summarize running
"${PYTHON_BIN}" - "${OUT_BASE}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
rows = []
for status_path in sorted((root / "status").glob("*.json")):
    status = json.loads(status_path.read_text(encoding="utf-8"))
    run_dir = Path(status["run_dir"])
    bundle_dir = Path(status["bundle_dir"])
    def read_json(path: Path):
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    rows.append(
        {
            **status,
            "run_summary": read_json(run_dir / "run_summary.json"),
            "final_eval_bundle_summary": read_json(bundle_dir / "final_eval_bundle_summary.json"),
        }
    )
(root / "release_gate_candidate_summary.json").write_text(json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"row_count": len(rows), "root": str(root)}, ensure_ascii=False, indent=2))
PY

date -u +"end_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "${RUN_DIR}/release_gate_environment.log"
write_status completed completed
