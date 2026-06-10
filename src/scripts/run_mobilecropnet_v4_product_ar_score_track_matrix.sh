#!/usr/bin/env bash
set -euo pipefail

TRACK_NAME="${1:?track name required}"
OUT_BASE="${2:?output base required}"
TRAIN_LABEL_DIR="${3:?train label dir required}"
VAL_LABEL_DIR="${4:?val label dir required}"
TEST_LABEL_DIR="${5:?test label dir required}"

ROOT="${ROOT:-/group-volume/users/jaden.ju/Sources/ImageCropping}"
cd "${ROOT}"

PROFILES_CSV="${PROFILES_CSV:-q24_288_w075,turbo_256,balanced_288,hq_320,rank_320,hybrid_384,plus_384}"
IFS=',' read -r -a PROFILES <<< "${PROFILES_CSV}"

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
STATUS_JSONL="${OUT_BASE}/matrix_status_${RUN_STAMP}.jsonl"
SUMMARY_DIR="${OUT_BASE}/summary_${RUN_STAMP}"
mkdir -p "${OUT_BASE}" "${SUMMARY_DIR}"

echo "track=${TRACK_NAME}"
echo "out_base=${OUT_BASE}"
echo "train_label_dir=${TRAIN_LABEL_DIR}"
echo "val_label_dir=${VAL_LABEL_DIR}"
echo "test_label_dir=${TEST_LABEL_DIR}"
echo "profiles=${PROFILES_CSV}"

RUN_DIRS=()
for PROFILE in "${PROFILES[@]}"; do
  PROFILE_SAFE="${PROFILE//_/-}"
  RUN_NAME="mcn-${TRACK_NAME}-${PROFILE_SAFE}-${RUN_STAMP}"
  RUN_DIR="${OUT_BASE}/${RUN_NAME}"
  RUN_DIRS+=("${RUN_DIR}")
  echo "[matrix] start ${RUN_NAME}"
  set +e
  TRAIN_LABEL_DIR="${TRAIN_LABEL_DIR}" \
  VAL_LABEL_DIR="${VAL_LABEL_DIR}" \
  TEST_LABEL_DIR="${TEST_LABEL_DIR}" \
  OUT_BASE="${OUT_BASE}" \
  EPOCHS="${EPOCHS:-8}" \
  BATCH_SIZE="${BATCH_SIZE:-16}" \
  LR="${LR:-0.0003}" \
  WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}" \
  SEED="${SEED:-20260420}" \
  SELECTION_METRIC="${SELECTION_METRIC:-sstk_balanced_topreturn}" \
  TOP_RETURN_WEIGHT="${TOP_RETURN_WEIGHT:-0.6}" \
  LISTWISE_WEIGHT="${LISTWISE_WEIGHT:-0.7}" \
  PAIRWISE_WEIGHT="${PAIRWISE_WEIGHT:-0.5}" \
  EXPLICIT_PAIRWISE_WEIGHT="${EXPLICIT_PAIRWISE_WEIGHT:-0.25}" \
  TOP1_RISK_WEIGHT="${TOP1_RISK_WEIGHT:-0.15}" \
  TEACHER_DISTILL_WEIGHT="${TEACHER_DISTILL_WEIGHT:-0.20}" \
  TOPK_COVERAGE_WEIGHT="${TOPK_COVERAGE_WEIGHT:-0.10}" \
  PROPOSAL_WEIGHT="${PROPOSAL_WEIGHT:-0.7}" \
  MAX_PAIRWISE_PAIRS="${MAX_PAIRWISE_PAIRS:-72}" \
  EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}" \
  bash src/scripts/run_mobilecropnet_v4_sstk_product_experiment.sh "${PROFILE}" "${RUN_NAME}"
  STATUS=$?
  set -e
  /usr/local/bin/python3 - "${STATUS_JSONL}" "${RUN_NAME}" "${PROFILE}" "${RUN_DIR}" "${STATUS}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
row = {
    "run_name": sys.argv[2],
    "profile": sys.argv[3],
    "run_dir": sys.argv[4],
    "status_code": int(sys.argv[5]),
    "status": "succeeded" if int(sys.argv[5]) == 0 else "failed",
}
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print(json.dumps(row, ensure_ascii=False))
PY
  if [[ "${STATUS}" != "0" ]]; then
    echo "[matrix] ${RUN_NAME} failed with status ${STATUS}; continuing to remaining profiles"
  fi
done

SUMMARY_INPUTS=()
for RUN_DIR in "${RUN_DIRS[@]}"; do
  if [[ -d "${RUN_DIR}" ]]; then
    SUMMARY_INPUTS+=("${RUN_DIR}")
  fi
done

if [[ "${#SUMMARY_INPUTS[@]}" -gt 0 ]]; then
  /usr/local/bin/python3 src/scripts/summarize_mobilecropnet_v4_sstk_product_runs.py \
    --run_dirs "${SUMMARY_INPUTS[@]}" \
    --output_dir "${SUMMARY_DIR}"
fi

/usr/local/bin/python3 - "${STATUS_JSONL}" "${SUMMARY_DIR}/matrix_status.json" "${TRACK_NAME}" "${RUN_STAMP}" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
rows = []
if src.exists():
    with src.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
payload = {
    "track": sys.argv[3],
    "run_stamp": sys.argv[4],
    "run_count": len(rows),
    "succeeded": sum(1 for row in rows if row.get("status") == "succeeded"),
    "failed": sum(1 for row in rows if row.get("status") == "failed"),
    "runs": rows,
}
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
raise SystemExit(0 if payload["failed"] == 0 else 1)
PY
