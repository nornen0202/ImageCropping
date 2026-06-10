#!/usr/bin/env bash
set -euo pipefail

MATRIX_NAME="${1:?matrix name required}"
PROFILE="${2:?profile required}"

ROOT="${ROOT:-$(pwd)}"
cd "${ROOT}"

TRAIN_LABEL_DIR="${TRAIN_LABEL_DIR:?train label dir required}"
VAL_LABEL_DIR="${VAL_LABEL_DIR:?val label dir required}"
TEST_LABEL_DIR="${TEST_LABEL_DIR:?test label dir required}"
OUT_BASE="${OUT_BASE:-artifacts/mobilecropnet_v4/runtime_no_prior_rerun_20260424}"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python3}"
VARIANTS="${VARIANTS:-baseline_current calibration_v1 calibration_v2}"

MATRIX_DIR="${OUT_BASE}/${MATRIX_NAME}_matrix"
MATRIX_LOG="${MATRIX_DIR}/run.log"
MATRIX_STATUS="${MATRIX_DIR}/matrix_status.json"
mkdir -p "${MATRIX_DIR}"

read -r -a VARIANT_LIST <<< "${VARIANTS}"

write_matrix_status() {
  local phase="$1"
  local state="$2"
  local current_variant="$3"
  local current_index="$4"
  "${PYTHON_BIN}" - <<PY
import json
import time
from pathlib import Path

matrix_name = ${MATRIX_NAME@Q}
output_base = ${OUT_BASE@Q}
variant_list = ${VARIANTS@Q}.split()
payload = {
    "matrix_name": matrix_name,
    "profile": ${PROFILE@Q},
    "phase": ${phase@Q},
    "state": ${state@Q},
    "last_update_time_unix": time.time(),
    "train_label_dir": ${TRAIN_LABEL_DIR@Q},
    "val_label_dir": ${VAL_LABEL_DIR@Q},
    "test_label_dir": ${TEST_LABEL_DIR@Q},
    "output_base": output_base,
    "matrix_dir": ${MATRIX_DIR@Q},
    "matrix_log": ${MATRIX_LOG@Q},
    "variant_count": len(variant_list),
    "variants": variant_list,
    "current_variant": ${current_variant@Q},
    "current_index": int(${current_index@Q}),
    "run_dirs": {
        variant: str(Path(output_base) / f"{matrix_name}_{variant}")
        for variant in variant_list
    },
}
Path(${MATRIX_STATUS@Q}).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

trap 'write_matrix_status failed failed "" -1' ERR

{
  date -u +"matrix_start_utc=%Y-%m-%dT%H:%M:%SZ"
  echo "matrix_name=${MATRIX_NAME}"
  echo "profile=${PROFILE}"
  echo "variants=${VARIANTS}"
  echo "train_label_dir=${TRAIN_LABEL_DIR}"
  echo "val_label_dir=${VAL_LABEL_DIR}"
  echo "test_label_dir=${TEST_LABEL_DIR}"
} | tee -a "${MATRIX_LOG}"

for idx in "${!VARIANT_LIST[@]}"; do
  variant="${VARIANT_LIST[$idx]}"
  run_name="${MATRIX_NAME}_${variant}"
  write_matrix_status "run" "running" "${variant}" "${idx}"
  {
    echo "===== variant=${variant} run_name=${run_name} start_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ") ====="
    TRAIN_LABEL_DIR="${TRAIN_LABEL_DIR}" \
    VAL_LABEL_DIR="${VAL_LABEL_DIR}" \
    TEST_LABEL_DIR="${TEST_LABEL_DIR}" \
    OUT_BASE="${OUT_BASE}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    bash src/scripts/run_mobilecropnet_v4_runtime_no_prior_variant.sh "${variant}" "${PROFILE}" "${run_name}"
    echo "===== variant=${variant} run_name=${run_name} end_utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ") ====="
  } 2>&1 | tee -a "${MATRIX_LOG}"
done

date -u +"matrix_end_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "${MATRIX_LOG}"
write_matrix_status "completed" "completed" "" "${#VARIANT_LIST[@]}"
