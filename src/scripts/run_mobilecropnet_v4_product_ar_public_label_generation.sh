#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/group-volume/users/jaden.ju/Sources/ImageCropping}"
cd "${ROOT}"

SPLIT_ROOT="${SPLIT_ROOT:-data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP}"
CAND_ROOT="${CAND_ROOT:-artifacts/mobilecropnet_v4/product_ar_score_labels_260420/candidate_rows}"
OUT_ROOT="${OUT_ROOT:-artifacts/mobilecropnet_v4/product_ar_score_labels_260420/public_direct}"
LABEL_NAME="${LABEL_NAME:-training_labels_public_score_product_ar_v1}"
SCORE_SOURCE_NAME="${SCORE_SOURCE_NAME:-gaic_public_score_product_ar_v1}"

mkdir -p "${OUT_ROOT}/scores"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
/usr/local/bin/python3 - <<'PY'
import torch
print("torch_version", torch.__version__)
print("cuda_available", torch.cuda.is_available())
assert torch.cuda.is_available()
print("device_name", torch.cuda.get_device_name(0))
PY

/usr/local/bin/python3 src/scripts/score_mobilecropnet_v4_product_ar_candidates_with_public_cropper.py \
  --candidate_rows_jsonl "${CAND_ROOT}/train_product_ar_candidates.jsonl" \
  --output_jsonl "${OUT_ROOT}/scores/train_public_product_ar_scores.jsonl" \
  --summary_json "${OUT_ROOT}/scores/train_public_product_ar_scores_summary.json" \
  --method gaic \
  --device cuda

/usr/local/bin/python3 src/scripts/score_mobilecropnet_v4_product_ar_candidates_with_public_cropper.py \
  --candidate_rows_jsonl "${CAND_ROOT}/val_product_ar_candidates.jsonl" \
  --output_jsonl "${OUT_ROOT}/scores/val_public_product_ar_scores.jsonl" \
  --summary_json "${OUT_ROOT}/scores/val_public_product_ar_scores_summary.json" \
  --method gaic \
  --device cuda

/usr/local/bin/python3 src/scripts/score_mobilecropnet_v4_product_ar_candidates_with_public_cropper.py \
  --candidate_rows_jsonl "${CAND_ROOT}/test_product_ar_candidates.jsonl" \
  --output_jsonl "${OUT_ROOT}/scores/test_public_product_ar_scores.jsonl" \
  --summary_json "${OUT_ROOT}/scores/test_public_product_ar_scores_summary.json" \
  --method gaic \
  --device cuda

TRAIN_OUT="${SPLIT_ROOT}/Train2636/artifacts/${LABEL_NAME}"
VAL_OUT="${SPLIT_ROOT}/Val200/artifacts/${LABEL_NAME}"
TEST_OUT="${SPLIT_ROOT}/Test500/artifacts/${LABEL_NAME}"
mkdir -p "${TRAIN_OUT}" "${VAL_OUT}" "${TEST_OUT}"

/usr/local/bin/python3 src/scripts/build_mobilecropnet_v4_product_ar_score_labels.py \
  --input_jsonl "${SPLIT_ROOT}/Train2636/artifacts/training_labels/gaic_v2_train_qf_c1c6capexp_c7exp_largecap_v2_260416_leftover_ignore_monotonic/train_conditional_detr_batch.jsonl" \
  --scored_candidates_jsonl "${OUT_ROOT}/scores/train_public_product_ar_scores.jsonl" \
  --output_jsonl "${TRAIN_OUT}/train_conditional_detr_batch.jsonl" \
  --summary_json "${TRAIN_OUT}/conversion_summary.json" \
  --output_pairwise_jsonl "${TRAIN_OUT}/train_pairwise.jsonl" \
  --output_listwise_jsonl "${TRAIN_OUT}/train_listwise.jsonl" \
  --score_source_name "${SCORE_SOURCE_NAME}"

/usr/local/bin/python3 src/scripts/build_mobilecropnet_v4_product_ar_score_labels.py \
  --input_jsonl "${SPLIT_ROOT}/Val200/artifacts/training_labels/gaic_v2_val_qf_c1c6capexp_c7exp_largecap_v2_260416_leftover_ignore_monotonic/train_conditional_detr_batch.jsonl" \
  --scored_candidates_jsonl "${OUT_ROOT}/scores/val_public_product_ar_scores.jsonl" \
  --output_jsonl "${VAL_OUT}/train_conditional_detr_batch.jsonl" \
  --summary_json "${VAL_OUT}/conversion_summary.json" \
  --output_pairwise_jsonl "${VAL_OUT}/train_pairwise.jsonl" \
  --output_listwise_jsonl "${VAL_OUT}/train_listwise.jsonl" \
  --score_source_name "${SCORE_SOURCE_NAME}"

/usr/local/bin/python3 src/scripts/build_mobilecropnet_v4_product_ar_score_labels.py \
  --input_jsonl "${SPLIT_ROOT}/Test500/artifacts/training_labels/gaic_v2_test_qf_c1c6capexp_c7exp_largecap_v2_260416_leftover_ignore_monotonic/train_conditional_detr_batch.jsonl" \
  --scored_candidates_jsonl "${OUT_ROOT}/scores/test_public_product_ar_scores.jsonl" \
  --output_jsonl "${TEST_OUT}/train_conditional_detr_batch.jsonl" \
  --summary_json "${TEST_OUT}/conversion_summary.json" \
  --output_pairwise_jsonl "${TEST_OUT}/train_pairwise.jsonl" \
  --output_listwise_jsonl "${TEST_OUT}/train_listwise.jsonl" \
  --score_source_name "${SCORE_SOURCE_NAME}"

/usr/local/bin/python3 - <<'PY'
import json
from pathlib import Path
paths = [
    Path("data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Train2636/artifacts/training_labels_public_score_product_ar_v1/conversion_summary.json"),
    Path("data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Val200/artifacts/training_labels_public_score_product_ar_v1/conversion_summary.json"),
    Path("data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Test500/artifacts/training_labels_public_score_product_ar_v1/conversion_summary.json"),
]
for path in paths:
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(path)
    print(json.dumps(payload, ensure_ascii=False)[:1600])
PY
