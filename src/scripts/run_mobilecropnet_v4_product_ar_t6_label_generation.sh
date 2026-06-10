#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/group-volume/users/jaden.ju/Sources/ImageCropping}"
cd "${ROOT}"

SPLIT_ROOT="${SPLIT_ROOT:-data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP}"
CAND_ROOT="${CAND_ROOT:-artifacts/mobilecropnet_v4/product_ar_score_labels_260420/candidate_rows}"
OUT_ROOT="${OUT_ROOT:-artifacts/mobilecropnet_v4/product_ar_score_labels_260420/t6_direct}"
LABEL_NAME="${LABEL_NAME:-training_labels_t6_score_product_ar_v1}"
SCORE_SOURCE_NAME="${SCORE_SOURCE_NAME:-t6_score_product_ar_v1}"
T6_BUNDLE="${T6_BUNDLE:-artifacts/mobilecropnet_v4/teacher_improvement/t6_deep_crop_ranker_260417/full_bundle_v2/Gc/selected_deep_crop_ranker_bundle.pt}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-${OUT_ROOT}/feature_cache}"
FEATURE_CACHE_KEY="${FEATURE_CACHE_KEY:-product_ar_qf_candidates}"
TIMM_MODEL_NAME="${TIMM_MODEL_NAME:-mobilenetv4_conv_small.e3600_r256_in1k}"
TIMM_WEIGHTS="${TIMM_WEIGHTS:-weights/hf/timm/mobilenetv4_conv_small.e3600_r256_in1k/model.safetensors}"

mkdir -p "${OUT_ROOT}/scores" "${FEATURE_CACHE_DIR}"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
/usr/local/bin/python3 - <<'PY'
import torch
print("torch_version", torch.__version__)
print("cuda_available", torch.cuda.is_available())
assert torch.cuda.is_available()
print("device_name", torch.cuda.get_device_name(0))
PY

/usr/local/bin/python3 src/scripts/cache_gaic_candidate_timm_roi_features.py \
  --candidate_eval_jsonl "${CAND_ROOT}/train_product_ar_candidates.jsonl" \
  --feature_cache_dir "${FEATURE_CACHE_DIR}" \
  --feature_cache_key "${FEATURE_CACHE_KEY}" \
  --split_name train \
  --summary_json "${OUT_ROOT}/train_feature_cache_summary.json" \
  --timm_model_name "${TIMM_MODEL_NAME}" \
  --timm_weights "${TIMM_WEIGHTS}" \
  --device cuda

/usr/local/bin/python3 src/scripts/cache_gaic_candidate_timm_roi_features.py \
  --candidate_eval_jsonl "${CAND_ROOT}/val_product_ar_candidates.jsonl" \
  --feature_cache_dir "${FEATURE_CACHE_DIR}" \
  --feature_cache_key "${FEATURE_CACHE_KEY}" \
  --split_name val \
  --summary_json "${OUT_ROOT}/val_feature_cache_summary.json" \
  --timm_model_name "${TIMM_MODEL_NAME}" \
  --timm_weights "${TIMM_WEIGHTS}" \
  --device cuda

/usr/local/bin/python3 src/scripts/cache_gaic_candidate_timm_roi_features.py \
  --candidate_eval_jsonl "${CAND_ROOT}/test_product_ar_candidates.jsonl" \
  --feature_cache_dir "${FEATURE_CACHE_DIR}" \
  --feature_cache_key "${FEATURE_CACHE_KEY}" \
  --split_name test \
  --summary_json "${OUT_ROOT}/test_feature_cache_summary.json" \
  --timm_model_name "${TIMM_MODEL_NAME}" \
  --timm_weights "${TIMM_WEIGHTS}" \
  --device cuda

/usr/local/bin/python3 src/scripts/export_t6_fixed_ar_candidate_scores.py \
  --candidate_eval_jsonl "${CAND_ROOT}/train_product_ar_candidates.jsonl" \
  --bundle "${T6_BUNDLE}" \
  --feature_cache_dir "${FEATURE_CACHE_DIR}" \
  --feature_cache_key "${FEATURE_CACHE_KEY}" \
  --split_name train \
  --output_jsonl "${OUT_ROOT}/scores/train_t6_product_ar_scores.jsonl" \
  --summary_json "${OUT_ROOT}/scores/train_t6_product_ar_scores_summary.json" \
  --group_keys image_id,target_ar \
  --score_source t6_deep_crop_ranker_product_ar \
  --device cuda

/usr/local/bin/python3 src/scripts/export_t6_fixed_ar_candidate_scores.py \
  --candidate_eval_jsonl "${CAND_ROOT}/val_product_ar_candidates.jsonl" \
  --bundle "${T6_BUNDLE}" \
  --feature_cache_dir "${FEATURE_CACHE_DIR}" \
  --feature_cache_key "${FEATURE_CACHE_KEY}" \
  --split_name val \
  --output_jsonl "${OUT_ROOT}/scores/val_t6_product_ar_scores.jsonl" \
  --summary_json "${OUT_ROOT}/scores/val_t6_product_ar_scores_summary.json" \
  --group_keys image_id,target_ar \
  --score_source t6_deep_crop_ranker_product_ar \
  --device cuda

/usr/local/bin/python3 src/scripts/export_t6_fixed_ar_candidate_scores.py \
  --candidate_eval_jsonl "${CAND_ROOT}/test_product_ar_candidates.jsonl" \
  --bundle "${T6_BUNDLE}" \
  --feature_cache_dir "${FEATURE_CACHE_DIR}" \
  --feature_cache_key "${FEATURE_CACHE_KEY}" \
  --split_name test \
  --output_jsonl "${OUT_ROOT}/scores/test_t6_product_ar_scores.jsonl" \
  --summary_json "${OUT_ROOT}/scores/test_t6_product_ar_scores_summary.json" \
  --group_keys image_id,target_ar \
  --score_source t6_deep_crop_ranker_product_ar \
  --device cuda

TRAIN_OUT="${SPLIT_ROOT}/Train2636/artifacts/${LABEL_NAME}"
VAL_OUT="${SPLIT_ROOT}/Val200/artifacts/${LABEL_NAME}"
TEST_OUT="${SPLIT_ROOT}/Test500/artifacts/${LABEL_NAME}"
mkdir -p "${TRAIN_OUT}" "${VAL_OUT}" "${TEST_OUT}"

/usr/local/bin/python3 src/scripts/build_mobilecropnet_v4_product_ar_score_labels.py \
  --input_jsonl "${SPLIT_ROOT}/Train2636/artifacts/training_labels/gaic_v2_train_qf_c1c6capexp_c7exp_largecap_v2_260416_leftover_ignore_monotonic/train_conditional_detr_batch.jsonl" \
  --scored_candidates_jsonl "${OUT_ROOT}/scores/train_t6_product_ar_scores.jsonl" \
  --output_jsonl "${TRAIN_OUT}/train_conditional_detr_batch.jsonl" \
  --summary_json "${TRAIN_OUT}/conversion_summary.json" \
  --output_pairwise_jsonl "${TRAIN_OUT}/train_pairwise.jsonl" \
  --output_listwise_jsonl "${TRAIN_OUT}/train_listwise.jsonl" \
  --score_source_name "${SCORE_SOURCE_NAME}"

/usr/local/bin/python3 src/scripts/build_mobilecropnet_v4_product_ar_score_labels.py \
  --input_jsonl "${SPLIT_ROOT}/Val200/artifacts/training_labels/gaic_v2_val_qf_c1c6capexp_c7exp_largecap_v2_260416_leftover_ignore_monotonic/train_conditional_detr_batch.jsonl" \
  --scored_candidates_jsonl "${OUT_ROOT}/scores/val_t6_product_ar_scores.jsonl" \
  --output_jsonl "${VAL_OUT}/train_conditional_detr_batch.jsonl" \
  --summary_json "${VAL_OUT}/conversion_summary.json" \
  --output_pairwise_jsonl "${VAL_OUT}/train_pairwise.jsonl" \
  --output_listwise_jsonl "${VAL_OUT}/train_listwise.jsonl" \
  --score_source_name "${SCORE_SOURCE_NAME}"

/usr/local/bin/python3 src/scripts/build_mobilecropnet_v4_product_ar_score_labels.py \
  --input_jsonl "${SPLIT_ROOT}/Test500/artifacts/training_labels/gaic_v2_test_qf_c1c6capexp_c7exp_largecap_v2_260416_leftover_ignore_monotonic/train_conditional_detr_batch.jsonl" \
  --scored_candidates_jsonl "${OUT_ROOT}/scores/test_t6_product_ar_scores.jsonl" \
  --output_jsonl "${TEST_OUT}/train_conditional_detr_batch.jsonl" \
  --summary_json "${TEST_OUT}/conversion_summary.json" \
  --output_pairwise_jsonl "${TEST_OUT}/train_pairwise.jsonl" \
  --output_listwise_jsonl "${TEST_OUT}/train_listwise.jsonl" \
  --score_source_name "${SCORE_SOURCE_NAME}"

/usr/local/bin/python3 - <<'PY'
import json
from pathlib import Path
paths = [
    Path("data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Train2636/artifacts/training_labels_t6_score_product_ar_v1/conversion_summary.json"),
    Path("data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Val200/artifacts/training_labels_t6_score_product_ar_v1/conversion_summary.json"),
    Path("data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Test500/artifacts/training_labels_t6_score_product_ar_v1/conversion_summary.json"),
]
for path in paths:
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(path)
    print(json.dumps(payload, ensure_ascii=False)[:1600])
PY
