#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:?run name required}"
WEIGHTS="${WEIGHTS:-src/scripts/yolov8n.pt}"
ROOT="${ROOT:-/group-volume/users/jaden.ju/Sources/ImageCropping}"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python3}"
OUT_BASE="${OUT_BASE:-artifacts/mobilecropnet_v4/quality_first_20260427/internal_detector_priors}"
DATASET_DIR="${DATASET_DIR:-${OUT_BASE}/yolo_subject_dataset_uctr_remote_smoke}"
TRAIN_JSONL="${TRAIN_JSONL:-data/SSTK/Full_10000/artifacts/training_labels_uctr_stage3_product_ar_260422_hashsplit_80_10_10_260422/train/train_conditional_detr_batch.jsonl}"
VAL_JSONL="${VAL_JSONL:-data/SSTK/Full_10000/artifacts/training_labels_uctr_stage3_product_ar_260422_hashsplit_80_10_10_260422/val/train_conditional_detr_batch.jsonl}"
MAX_TRAIN_IMAGES="${MAX_TRAIN_IMAGES:-5000}"
MAX_VAL_IMAGES="${MAX_VAL_IMAGES:-1200}"
EPOCHS="${EPOCHS:-24}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-48}"
WORKERS="${WORKERS:-4}"
PATIENCE="${PATIENCE:-8}"
DEVICE="${DEVICE:-0}"
CONF="${CONF:-0.05}"
COPY_MODE="${COPY_MODE:-symlink}"
YOLO_CLASS_MODE="${YOLO_CLASS_MODE:-one}"
YOLO_INCLUDE_NEGATIVE_IMAGES="${YOLO_INCLUDE_NEGATIVE_IMAGES:-0}"

RUN_DIR="${OUT_BASE}/yolo_train_runs/${RUN_NAME}"
EVAL_DIR="${OUT_BASE}/${RUN_NAME}_eval"
STATUS_JSON="${OUT_BASE}/${RUN_NAME}_status.json"
TRAIN_LOG="${RUN_DIR}/train.log"
SUMMARY_JSON="${OUT_BASE}/${RUN_NAME}_summary.json"

cd "${ROOT}"
mkdir -p "${RUN_DIR}" "${EVAL_DIR}"

write_status() {
  local state="$1"
  local phase="$2"
  "${PYTHON_BIN}" - <<PY
import json
import time
from pathlib import Path

payload = {
    "state": ${state@Q},
    "phase": ${phase@Q},
    "run_name": ${RUN_NAME@Q},
    "weights": ${WEIGHTS@Q},
    "dataset_dir": ${DATASET_DIR@Q},
    "run_dir": ${RUN_DIR@Q},
    "eval_dir": ${EVAL_DIR@Q},
    "train_log": ${TRAIN_LOG@Q},
    "summary_json": ${SUMMARY_JSON@Q},
    "last_update_time_unix": time.time(),
}
Path(${STATUS_JSON@Q}).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

trap 'write_status failed failed' ERR

write_status running prepare
PIP_CONFIG_FILE=/dev/null "${PYTHON_BIN}" -m pip install --user --index-url https://bart.sec.samsung.net/artifactory/api/pypi/pypi-remote/simple ultralytics==8.4.14

if [[ ! -f "${DATASET_DIR}/dataset_summary.json" ]]; then
  write_status running build_dataset
  declare -a DATASET_EXTRA_ARGS=()
  if [[ "${YOLO_INCLUDE_NEGATIVE_IMAGES}" == "1" || "${YOLO_INCLUDE_NEGATIVE_IMAGES}" == "true" ]]; then
    DATASET_EXTRA_ARGS+=(--include_negative_images)
  fi
  "${PYTHON_BIN}" src/scripts/build_yolo_subject_dataset.py \
    --train_jsonl "${TRAIN_JSONL}" \
    --val_jsonl "${VAL_JSONL}" \
    --project_root . \
    --output_dir "${DATASET_DIR}" \
    --max_train_images "${MAX_TRAIN_IMAGES}" \
    --max_val_images "${MAX_VAL_IMAGES}" \
    --copy_mode "${COPY_MODE}" \
    --class_mode "${YOLO_CLASS_MODE}" \
    "${DATASET_EXTRA_ARGS[@]}"
fi

write_status running train
export YOLO_CONFIG_DIR="${OUT_BASE}/.ultralytics_${RUN_NAME}"
mkdir -p "${YOLO_CONFIG_DIR}"
"${PYTHON_BIN}" - <<PY 2>&1 | tee "${TRAIN_LOG}"
from pathlib import Path
from ultralytics import YOLO

root = Path(${OUT_BASE@Q}).resolve()
model = YOLO(${WEIGHTS@Q})
model.train(
    data=str(Path(${DATASET_DIR@Q}) / "subject.yaml"),
    epochs=int(${EPOCHS@Q}),
    imgsz=int(${IMGSZ@Q}),
    batch=int(${BATCH@Q}),
    device=${DEVICE@Q},
    workers=int(${WORKERS@Q}),
    project=str(root / "yolo_train_runs"),
    name=${RUN_NAME@Q},
    exist_ok=True,
    patience=int(${PATIENCE@Q}),
    verbose=True,
)
PY

BEST="${RUN_DIR}/weights/best.pt"
if [[ ! -f "${BEST}" ]]; then
  echo "missing best checkpoint: ${BEST}" >&2
  exit 2
fi

write_status running eval
"${PYTHON_BIN}" src/scripts/evaluate_yolo_subject_prior_smoke.py \
  --eval_jsonl "${VAL_JSONL}" \
  --project_root . \
  --weights "${BEST}" \
  --output_dir "${EVAL_DIR}" \
  --max_unique_images "${MAX_VAL_IMAGES}" \
  --imgsz "${IMGSZ}" \
  --conf "${CONF}" \
  --device "${DEVICE}" \
  --progress_log_interval 100

write_status completed completed
"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

summary = {
    "state": "completed",
    "run_name": ${RUN_NAME@Q},
    "weights": ${WEIGHTS@Q},
    "best_checkpoint": ${BEST@Q},
    "dataset_summary": str(Path(${DATASET_DIR@Q}) / "dataset_summary.json"),
    "train_results": str(Path(${RUN_DIR@Q}) / "results.csv"),
    "eval_summary": str(Path(${EVAL_DIR@Q}) / "summary.json"),
    "status_json": ${STATUS_JSON@Q},
}
Path(${SUMMARY_JSON@Q}).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
