#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:?profile required}"
RUN_NAME="${2:?run name required}"

ROOT="${ROOT:-$(pwd)}"
cd "${ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python3}"
TRAIN_LABEL_DIR="${TRAIN_LABEL_DIR:?train label dir required}"
VAL_LABEL_DIR="${VAL_LABEL_DIR:?val label dir required}"
TEST_LABEL_DIR="${TEST_LABEL_DIR:?test label dir required}"
OUT_BASE="${OUT_BASE:-artifacts/mobilecropnet_v4/runtime_no_prior_20260424}"
OUT_DIR="${OUT_BASE}/${RUN_NAME}"
STATUS_JSON="${OUT_DIR}/status.json"

EPOCHS="${EPOCHS:-6}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
SEED="${SEED:-20260424}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-0}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
IMAGE_TENSOR_CACHE_SIZE="${IMAGE_TENSOR_CACHE_SIZE:-128}"
AMP="${AMP:-1}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"

mkdir -p "${OUT_DIR}"

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
    "output_dir": ${OUT_DIR@Q},
    "run_name": ${RUN_NAME@Q},
    "profile": ${PROFILE@Q},
    "environment_log": str(Path(${OUT_DIR@Q}) / "environment.log"),
    "train_log": str(Path(${OUT_DIR@Q}) / "train.log"),
    "eval_log": str(Path(${OUT_DIR@Q}) / "eval_test.log"),
    "direct_val_log": str(Path(${OUT_DIR@Q}) / "direct_val.log"),
    "direct_test_log": str(Path(${OUT_DIR@Q}) / "direct_test.log"),
    "compare_log": str(Path(${OUT_DIR@Q}) / "compare_test.log"),
    "viz_log": str(Path(${OUT_DIR@Q}) / "viz_test.log"),
    "paper_pack_log": str(Path(${OUT_DIR@Q}) / "paper_pack.log"),
    "run_summary": str(Path(${OUT_DIR@Q}) / "run_summary.json"),
    "train_status": str(Path(${OUT_DIR@Q}) / "train_status.json"),
    "best_checkpoint": str(Path(${OUT_DIR@Q}) / "best.pt"),
    "latest_checkpoint": str(Path(${OUT_DIR@Q}) / "last.pt"),
}
Path(${STATUS_JSON@Q}).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

trap 'write_status failed failed' ERR

declare -a DATALOADER_ARGS
DATALOADER_ARGS=(--num_workers "${NUM_WORKERS}" --image_tensor_cache_size "${IMAGE_TENSOR_CACHE_SIZE}")
if [[ "${NUM_WORKERS}" -gt 0 && "${PERSISTENT_WORKERS}" == "1" ]]; then
  DATALOADER_ARGS+=(--persistent_workers)
fi
if [[ "${NUM_WORKERS}" -gt 0 ]]; then
  DATALOADER_ARGS+=(--prefetch_factor "${PREFETCH_FACTOR}")
fi

declare -a AMP_ARGS
AMP_ARGS=()
if [[ "${AMP}" == "1" ]]; then
  AMP_ARGS+=(--amp)
fi

declare -a PAIRWISE_ARGS
PAIRWISE_ARGS=()
if [[ -f "${TRAIN_LABEL_DIR}/train_pairwise.jsonl" ]]; then
  PAIRWISE_ARGS+=(--pairwise_jsonl "${TRAIN_LABEL_DIR}/train_pairwise.jsonl")
fi
if [[ -f "${TRAIN_LABEL_DIR}/train_listwise.jsonl" ]]; then
  PAIRWISE_ARGS+=(--listwise_jsonl "${TRAIN_LABEL_DIR}/train_listwise.jsonl")
fi
if [[ -f "${VAL_LABEL_DIR}/train_pairwise.jsonl" ]]; then
  PAIRWISE_ARGS+=(--val_pairwise_jsonl "${VAL_LABEL_DIR}/train_pairwise.jsonl")
fi
if [[ -f "${VAL_LABEL_DIR}/train_listwise.jsonl" ]]; then
  PAIRWISE_ARGS+=(--val_listwise_jsonl "${VAL_LABEL_DIR}/train_listwise.jsonl")
fi

{
  date -u +"start_utc=%Y-%m-%dT%H:%M:%SZ"
  echo "profile=${PROFILE}"
  echo "run_name=${RUN_NAME}"
  echo "train_label_dir=${TRAIN_LABEL_DIR}"
  echo "val_label_dir=${VAL_LABEL_DIR}"
  echo "test_label_dir=${TEST_LABEL_DIR}"
} | tee "${OUT_DIR}/environment.log"

write_status train running
"${PYTHON_BIN}" src/scripts/train_mobilecropnet_v4.py \
  --train_jsonl "${TRAIN_LABEL_DIR}/train_conditional_detr_batch.jsonl" \
  --val_jsonl "${VAL_LABEL_DIR}/train_conditional_detr_batch.jsonl" \
  --project_root . \
  --output_dir "${OUT_DIR}" \
  --model_profile "${PROFILE}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --selection_metric deploy_align_topreturn \
  --top_return_weight 0.8 \
  --listwise_weight 0.6 \
  --pairwise_weight 0.5 \
  --explicit_pairwise_weight 0.2 \
  --top1_risk_weight 0.12 \
  --generated_proposal_align_weight 0.15 \
  --generated_proposal_score_weight 1.0 \
  --generated_proposal_listwise_weight 0.5 \
  --generated_proposal_positive_weight 0.25 \
  --generated_proposal_risk_weight 0.15 \
  --policy_score_head \
  --policy_score_weight 0.15 \
  --decision_weight 0.18 \
  --action_consistency_weight 0.25 \
  --action_consistency_margin 0.15 \
  --route_balanced_ce \
  --detail_score_weight 0.08 \
  --checklist_class_weight 0.04 \
  --why_tag_weight 0.02 \
  "${PAIRWISE_ARGS[@]}" \
  "${DATALOADER_ARGS[@]}" \
  "${AMP_ARGS[@]}" \
  --device cuda \
  --seed "${SEED}" \
  ${EXTRA_TRAIN_ARGS} \
  2>&1 | tee "${OUT_DIR}/train.log"

write_status eval_test running
"${PYTHON_BIN}" src/scripts/evaluate_mobilecropnet_v4.py \
  --checkpoint "${OUT_DIR}/best.pt" \
  --eval_jsonl "${TEST_LABEL_DIR}/train_conditional_detr_batch.jsonl" \
  --project_root . \
  --output_dir "${OUT_DIR}/eval_test" \
  --batch_size 16 \
  --num_workers 2 \
  --device cuda \
  2>&1 | tee "${OUT_DIR}/eval_test.log"

write_status direct_val running
"${PYTHON_BIN}" src/scripts/evaluate_mobilecropnet_v4_product_ar_direct.py \
  --checkpoint "${OUT_DIR}/best.pt" \
  --eval_jsonl "${VAL_LABEL_DIR}/train_conditional_detr_batch.jsonl" \
  --project_root . \
  --output_dir "${OUT_DIR}/direct_val_proposal_topk_rerank" \
  --batch_size 16 \
  --num_workers 2 \
  --selection_policy proposal_topk_rerank \
  --runtime_subject_prior_mode none \
  --exact_target_ar_postprocess \
  --device cuda \
  2>&1 | tee "${OUT_DIR}/direct_val.log"

write_status direct_test running
"${PYTHON_BIN}" src/scripts/evaluate_mobilecropnet_v4_product_ar_direct.py \
  --checkpoint "${OUT_DIR}/best.pt" \
  --eval_jsonl "${TEST_LABEL_DIR}/train_conditional_detr_batch.jsonl" \
  --project_root . \
  --output_dir "${OUT_DIR}/direct_test_proposal_topk_rerank" \
  --batch_size 16 \
  --num_workers 2 \
  --selection_policy proposal_topk_rerank \
  --runtime_subject_prior_mode none \
  --exact_target_ar_postprocess \
  --device cuda \
  2>&1 | tee "${OUT_DIR}/direct_test.log"

write_status compare_test running
"${PYTHON_BIN}" src/scripts/compare_mobilecropnet_v4_methods.py \
  --predictions_jsonl "${OUT_DIR}/eval_test/predictions.jsonl" \
  --output_dir "${OUT_DIR}/compare_test" \
  2>&1 | tee "${OUT_DIR}/compare_test.log"

write_status viz_test running
"${PYTHON_BIN}" src/scripts/visualize_mobilecropnet_v4_predictions.py \
  --predictions_jsonl "${OUT_DIR}/eval_test/predictions.jsonl" \
  --output_dir "${OUT_DIR}/viz_test" \
  --sample_size 48 \
  --max_side 1500 \
  2>&1 | tee "${OUT_DIR}/viz_test.log"

write_status paper_pack running
"${PYTHON_BIN}" src/scripts/build_mobilecropnet_v4_report.py \
  --run_dir "${OUT_DIR}" \
  --eval_dir "${OUT_DIR}/eval_test" \
  --comparison_dir "${OUT_DIR}/compare_test" \
  --viz_dir "${OUT_DIR}/viz_test" \
  --output_dir "${OUT_DIR}/paper_pack" \
  2>&1 | tee "${OUT_DIR}/paper_pack.log" || true

write_status summarize running
"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path

run = Path("${OUT_DIR}")

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

history = read_json(run / "metrics.json") or []
best_row = max(history, key=lambda row: row.get("selection_score", float("-inf"))) if history else {}
summary = {
    "run_name": "${RUN_NAME}",
    "profile": "${PROFILE}",
    "output_dir": str(run),
    "best_epoch": best_row.get("epoch"),
    "best_selection_score": best_row.get("selection_score"),
    "train_config": read_json(run / "config.json"),
    "dataset_summary": read_json(run / "dataset_summary.json"),
    "train_history": history,
    "replay_test_metrics": read_json(run / "eval_test" / "metrics.json"),
    "direct_val_metrics": read_json(run / "direct_val_proposal_topk_rerank" / "metrics.json"),
    "direct_test_metrics": read_json(run / "direct_test_proposal_topk_rerank" / "metrics.json"),
    "comparison_metrics": read_json(run / "compare_test" / "comparison_metrics.json"),
    "visualization_manifest": read_json(run / "viz_test" / "visualization_manifest.json"),
}
(run / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({
    "run_name": summary["run_name"],
    "profile": summary["profile"],
    "best_epoch": summary["best_epoch"],
    "best_selection_score": summary["best_selection_score"],
    "direct_test_metrics": (summary["direct_test_metrics"] or {}).get("metrics", {}),
}, ensure_ascii=False, indent=2))
PY

date -u +"end_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "${OUT_DIR}/environment.log"
write_status completed completed
