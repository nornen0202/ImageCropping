#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:?profile required}"
RUN_NAME="${2:?run name required}"

ROOT="${ROOT:-/group-volume/users/jaden.ju/Sources/ImageCropping}"
cd "${ROOT}"

TRAIN_LABEL_DIR="${TRAIN_LABEL_DIR:-data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Train2636/artifacts/training_labels/gaic_v2_train_qf_c1c6capexp_c7exp_largecap_v2_260416_leftover_ignore_monotonic}"
VAL_LABEL_DIR="${VAL_LABEL_DIR:-data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Val200/artifacts/training_labels/gaic_v2_val_qf_c1c6capexp_c7exp_largecap_v2_260416_leftover_ignore_monotonic}"
TEST_LABEL_DIR="${TEST_LABEL_DIR:-data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Test500/artifacts/training_labels/gaic_v2_test_qf_c1c6capexp_c7exp_largecap_v2_260416_leftover_ignore_monotonic}"
TEST_TEACHER_ROWS="${TEST_TEACHER_ROWS:-data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Test500/artifacts/reports/gaic_benchmark_eval_gaic_v2_test_qf_c1c6capexp_c7exp_largecap_v2_260416/candidate_eval_rows.jsonl}"
OUT_BASE="${OUT_BASE:-artifacts/mobilecropnet_v4/sstk_product_topreturn}"
OUT_DIR="${OUT_BASE}/${RUN_NAME}"
STATUS_JSON="${OUT_DIR}/status.json"

EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
SEED="${SEED:-20260417}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-0}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
PRECOMPUTE_SAMPLE_TENSORS="${PRECOMPUTE_SAMPLE_TENSORS:-0}"
IMAGE_TENSOR_CACHE_SIZE="${IMAGE_TENSOR_CACHE_SIZE:-128}"
SELECTION_METRIC="${SELECTION_METRIC:-sstk_topreturn}"
TOP_RETURN_WEIGHT="${TOP_RETURN_WEIGHT:-0.8}"
TOP_RETURN_SCORE_MARGIN="${TOP_RETURN_SCORE_MARGIN:-0.03}"
TOP_RETURN_LOGIT_MARGIN="${TOP_RETURN_LOGIT_MARGIN:-0.25}"
TOP1_RISK_WEIGHT="${TOP1_RISK_WEIGHT:-0.12}"
LISTWISE_WEIGHT="${LISTWISE_WEIGHT:-0.6}"
PAIRWISE_WEIGHT="${PAIRWISE_WEIGHT:-0.5}"
EXPLICIT_PAIRWISE_WEIGHT="${EXPLICIT_PAIRWISE_WEIGHT:-0.20}"
TEACHER_DISTILL_WEIGHT="${TEACHER_DISTILL_WEIGHT:-0.0}"
TEACHER_DISTILL_TEMPERATURE="${TEACHER_DISTILL_TEMPERATURE:-0.12}"
TOPK_COVERAGE_WEIGHT="${TOPK_COVERAGE_WEIGHT:-0.0}"
TOPK_COVERAGE_K="${TOPK_COVERAGE_K:-4}"
TOPK_COVERAGE_TEMPERATURE="${TOPK_COVERAGE_TEMPERATURE:-0.12}"
CHECKLIST_CLASS_WEIGHT="${CHECKLIST_CLASS_WEIGHT:-0.02}"
CHECKLIST_APPLICABILITY_WEIGHT="${CHECKLIST_APPLICABILITY_WEIGHT:-0.02}"
DETAIL_SCORE_WEIGHT="${DETAIL_SCORE_WEIGHT:-0.02}"
WHY_TAG_WEIGHT="${WHY_TAG_WEIGHT:-0.01}"
PROPOSAL_WEIGHT="${PROPOSAL_WEIGHT:-0.6}"
MAX_PAIRWISE_PAIRS="${MAX_PAIRWISE_PAIRS:-48}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
AMP="${AMP:-1}"

declare -a DATALOADER_ARGS
DATALOADER_ARGS=(--num_workers "${NUM_WORKERS}" --image_tensor_cache_size "${IMAGE_TENSOR_CACHE_SIZE}")
if [[ "${NUM_WORKERS}" -gt 0 && "${PERSISTENT_WORKERS}" == "1" ]]; then
  DATALOADER_ARGS+=(--persistent_workers)
fi
if [[ "${NUM_WORKERS}" -gt 0 ]]; then
  DATALOADER_ARGS+=(--prefetch_factor "${PREFETCH_FACTOR}")
fi
if [[ "${PRECOMPUTE_SAMPLE_TENSORS}" == "1" ]]; then
  DATALOADER_ARGS+=(--precompute_sample_tensors)
fi
declare -a AMP_ARGS
AMP_ARGS=()
if [[ "${AMP}" == "1" ]]; then
  AMP_ARGS+=(--amp)
fi

mkdir -p "${OUT_DIR}"

write_status() {
  local phase="$1"
  local state="$2"
  /usr/local/bin/python3 - <<PY
import json
from pathlib import Path

payload = {
    "phase": ${phase@Q},
    "state": ${state@Q},
    "output_dir": ${OUT_DIR@Q},
    "run_name": ${RUN_NAME@Q},
    "profile": ${PROFILE@Q},
    "train_log": str(Path(${OUT_DIR@Q}) / "train.log"),
    "eval_log": str(Path(${OUT_DIR@Q}) / "eval_test.log"),
    "compare_log": str(Path(${OUT_DIR@Q}) / "compare_test.log"),
    "viz_log": str(Path(${OUT_DIR@Q}) / "viz_test.log"),
    "gaic_log": str(Path(${OUT_DIR@Q}) / "gaic_official_test.log"),
    "run_summary": str(Path(${OUT_DIR@Q}) / "run_summary.json"),
    "train_status": str(Path(${OUT_DIR@Q}) / "train_status.json"),
}
Path(${STATUS_JSON@Q}).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

trap 'write_status failed failed' ERR

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

write_status setup running

write_status train running
/usr/local/bin/python3 src/scripts/train_mobilecropnet_v4.py \
  --train_jsonl "${TRAIN_LABEL_DIR}/train_conditional_detr_batch.jsonl" \
  --val_jsonl "${VAL_LABEL_DIR}/train_conditional_detr_batch.jsonl" \
  --project_root . \
  --output_dir "${OUT_DIR}" \
  --model_profile "${PROFILE}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  "${DATALOADER_ARGS[@]}" \
  --pairwise_jsonl "${TRAIN_LABEL_DIR}/train_pairwise.jsonl" \
  --listwise_jsonl "${TRAIN_LABEL_DIR}/train_listwise.jsonl" \
  --val_pairwise_jsonl "${VAL_LABEL_DIR}/train_pairwise.jsonl" \
  --val_listwise_jsonl "${VAL_LABEL_DIR}/train_listwise.jsonl" \
  --max_pairwise_pairs "${MAX_PAIRWISE_PAIRS}" \
  --score_target_mode crop_utility_prob \
  --listwise_weight "${LISTWISE_WEIGHT}" \
  --pairwise_weight "${PAIRWISE_WEIGHT}" \
  --explicit_pairwise_weight "${EXPLICIT_PAIRWISE_WEIGHT}" \
  --top1_risk_weight "${TOP1_RISK_WEIGHT}" \
  --top_return_weight "${TOP_RETURN_WEIGHT}" \
  --top_return_score_margin "${TOP_RETURN_SCORE_MARGIN}" \
  --top_return_logit_margin "${TOP_RETURN_LOGIT_MARGIN}" \
  --teacher_distill_weight "${TEACHER_DISTILL_WEIGHT}" \
  --teacher_distill_temperature "${TEACHER_DISTILL_TEMPERATURE}" \
  --topk_coverage_weight "${TOPK_COVERAGE_WEIGHT}" \
  --topk_coverage_k "${TOPK_COVERAGE_K}" \
  --topk_coverage_temperature "${TOPK_COVERAGE_TEMPERATURE}" \
  --selection_metric "${SELECTION_METRIC}" \
  --checklist_class_weight "${CHECKLIST_CLASS_WEIGHT}" \
  --checklist_applicability_weight "${CHECKLIST_APPLICABILITY_WEIGHT}" \
  --detail_score_weight "${DETAIL_SCORE_WEIGHT}" \
  --why_tag_weight "${WHY_TAG_WEIGHT}" \
  --proposal_weight "${PROPOSAL_WEIGHT}" \
  "${AMP_ARGS[@]}" \
  --device cuda \
  --gpu_usage_sample_interval 20 \
  --seed "${SEED}" \
  ${EXTRA_TRAIN_ARGS} \
  2>&1 | tee "${OUT_DIR}/train.log"

write_status eval_test running
/usr/local/bin/python3 src/scripts/evaluate_mobilecropnet_v4.py \
  --checkpoint "${OUT_DIR}/best.pt" \
  --eval_jsonl "${TEST_LABEL_DIR}/train_conditional_detr_batch.jsonl" \
  --project_root . \
  --output_dir "${OUT_DIR}/eval_test" \
  --batch_size 32 \
  --num_workers 4 \
  --device cuda \
  2>&1 | tee "${OUT_DIR}/eval_test.log"

write_status compare_test running
/usr/local/bin/python3 src/scripts/compare_mobilecropnet_v4_methods.py \
  --predictions_jsonl "${OUT_DIR}/eval_test/predictions.jsonl" \
  --output_dir "${OUT_DIR}/compare_test" \
  2>&1 | tee "${OUT_DIR}/compare_test.log"

write_status viz_test running
/usr/local/bin/python3 src/scripts/visualize_mobilecropnet_v4_predictions.py \
  --predictions_jsonl "${OUT_DIR}/eval_test/predictions.jsonl" \
  --output_dir "${OUT_DIR}/viz_test" \
  --sample_size 48 \
  --max_side 1500 \
  2>&1 | tee "${OUT_DIR}/viz_test.log"

write_status gaic_official_test running
/usr/local/bin/python3 src/scripts/evaluate_mobilecropnet_v4_gaic_benchmark.py \
  --checkpoint "${OUT_DIR}/best.pt" \
  --annotations_json data/Publics/GAIC_v2/annotations_json/instances_test.json \
  --image_roots data/Publics/GAIC_v2/images/test data/Publics/GAIC/images/test \
  --output_dir "${OUT_DIR}/gaic_official_test" \
  --model_name "mcn_v4_${PROFILE}_${RUN_NAME}" \
  --target_ar FREE \
  --teacher_candidate_eval_jsonl "${TEST_TEACHER_ROWS}" \
  --teacher_name "sstk_teacher_fullopt_largecap_v2_raw" \
  --teacher_protocol Gc \
  --teacher_score_field crop_utility_raw \
  --device cuda \
  --save_per_image \
  2>&1 | tee "${OUT_DIR}/gaic_official_test.log"

write_status summarize running
/usr/local/bin/python3 - <<PY
import json
from pathlib import Path
run = Path("${OUT_DIR}")
def read_json(path):
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
    "comparison_metrics": read_json(run / "compare_test" / "comparison_metrics.json"),
    "gaic_official_metrics": read_json(run / "gaic_official_test" / "metrics.json"),
    "visualization_manifest": read_json(run / "viz_test" / "visualization_manifest.json"),
}
(run / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({
    "run_name": summary["run_name"],
    "profile": summary["profile"],
    "best_epoch": summary["best_epoch"],
    "best_selection_score": summary["best_selection_score"],
    "replay_test": (summary["replay_test_metrics"] or {}).get("metrics", {}),
    "gaic_methods": list((summary["gaic_official_metrics"] or {}).get("methods", {}).keys()),
}, ensure_ascii=False, indent=2))
PY

date -u +"end_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "${OUT_DIR}/environment.log"
write_status completed completed
