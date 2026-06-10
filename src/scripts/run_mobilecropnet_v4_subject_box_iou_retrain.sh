#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:?variant required}"
PROFILE="${2:?profile required}"
RUN_NAME="${3:?run name required}"

ROOT="${ROOT:-/group-volume/users/jaden.ju/Sources/ImageCropping}"
OUT_BASE="${OUT_BASE:?OUT_BASE required}"
TRAIN_LABEL_DIR="${TRAIN_LABEL_DIR:?TRAIN_LABEL_DIR required}"
VAL_LABEL_DIR="${VAL_LABEL_DIR:?VAL_LABEL_DIR required}"
TEST_LABEL_DIR="${TEST_LABEL_DIR:?TEST_LABEL_DIR required}"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python3}"

OUT_DIR="${OUT_BASE}/${RUN_NAME}"
STATUS_JSON="${OUT_DIR}/status.json"

cd "${ROOT}"
mkdir -p "${OUT_DIR}"

write_status() {
  local phase="$1"
  local state="$2"
  "${PYTHON_BIN}" - "$STATUS_JSON" "$phase" "$state" "$RUN_NAME" "$PROFILE" "$VARIANT" "$OUT_DIR" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
phase = sys.argv[2]
state = sys.argv[3]
run_name = sys.argv[4]
profile = sys.argv[5]
variant = sys.argv[6]
out_dir = sys.argv[7]

payload = {}
if path.exists():
    payload = json.loads(path.read_text(encoding="utf-8"))
payload.update(
    {
        "run_name": run_name,
        "profile": profile,
        "variant": variant,
        "out_dir": out_dir,
        "phase": phase,
        "state": state,
        "last_update_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False))
PY
}

write_status "bootstrap" "running"

{
  date -u +"bootstrap_start_utc=%Y-%m-%dT%H:%M:%SZ"
  nvidia-smi --query-gpu=name,memory.total,driver_version,utilization.gpu,memory.used --format=csv,noheader
  "${PYTHON_BIN}" -m py_compile \
    src/mobilecropnet_v4/model.py \
    src/mobilecropnet_v4/eval_utils.py \
    src/scripts/train_mobilecropnet_v4.py \
    src/scripts/build_mobilecropnet_v4_subject_box_report.py
} | tee "${OUT_DIR}/bootstrap.log"

write_status "train_eval_bundle" "running"

bash src/scripts/run_mobilecropnet_v4_single_profile_variant.sh \
  "${VARIANT}" \
  "${PROFILE}" \
  "${RUN_NAME}" \
  2>&1 | tee "${OUT_DIR}/bundle.log"

write_status "post_eval_best" "running"

bash src/scripts/run_mobilecropnet_v4_subject_box_post_eval.sh \
  "${OUT_DIR}" \
  "${VAL_LABEL_DIR}/train_conditional_detr_batch.jsonl" \
  "${TEST_LABEL_DIR}/train_conditional_detr_batch.jsonl" \
  2>&1 | tee "${OUT_DIR}/post_eval_best.log"

if [[ -f "${OUT_DIR}/subject_box_best.pt" ]]; then
  write_status "post_eval_subject_box_best" "running"
  CHECKPOINT_PATH="${OUT_DIR}/subject_box_best.pt" \
  OUTPUT_TAG="subject_box_best" \
  bash src/scripts/run_mobilecropnet_v4_subject_box_post_eval.sh \
    "${OUT_DIR}" \
    "${VAL_LABEL_DIR}/train_conditional_detr_batch.jsonl" \
    "${TEST_LABEL_DIR}/train_conditional_detr_batch.jsonl" \
    2>&1 | tee "${OUT_DIR}/post_eval_subject_box_best.log"
fi

write_status "finalize" "running"

"${PYTHON_BIN}" - "${OUT_DIR}" <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

run_dir = Path(sys.argv[1])

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

summary = {
    "run_dir": str(run_dir),
    "completed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "run_summary": read_json(run_dir / "run_summary.json"),
    "post_eval_best": read_json(run_dir / "subject_box_post_eval_summary.json"),
    "post_eval_subject_box_best": read_json(run_dir / "subject_box_post_eval_summary_subject_box_best.json"),
}
(run_dir / "subject_box_iou_retrain_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps({
    "run_dir": summary["run_dir"],
    "best_selection_score": (((summary["run_summary"] or {}).get("best_selection_score"))),
    "has_subject_box_best_post_eval": summary["post_eval_subject_box_best"] is not None,
}, ensure_ascii=False, indent=2))
PY

write_status "completed" "succeeded"
