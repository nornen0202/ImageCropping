#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:?profile required}"
RUN_NAME="${2:?run name required}"

ROOT="${ROOT:-/group-volume/users/jaden.ju/Sources/ImageCropping}"
cd "${ROOT}"

TRAIN_LABEL_DIR="${TRAIN_LABEL_DIR:?train label dir required}"
VAL_LABEL_DIR="${VAL_LABEL_DIR:?val label dir required}"
TEST_LABEL_DIR="${TEST_LABEL_DIR:?test label dir required}"
OUT_BASE="${OUT_BASE:-artifacts/mobilecropnet_v4/product_ar_deploy_align}"

GENERATED_ALIGN_WEIGHT="${GENERATED_ALIGN_WEIGHT:-0.30}"
GENERATED_ALIGN_SCORE_WEIGHT="${GENERATED_ALIGN_SCORE_WEIGHT:-1.0}"
GENERATED_ALIGN_LISTWISE_WEIGHT="${GENERATED_ALIGN_LISTWISE_WEIGHT:-0.5}"
GENERATED_ALIGN_POSITIVE_WEIGHT="${GENERATED_ALIGN_POSITIVE_WEIGHT:-0.3}"
GENERATED_ALIGN_RISK_WEIGHT="${GENERATED_ALIGN_RISK_WEIGHT:-0.2}"
GENERATED_ALIGN_MATCH_IOU="${GENERATED_ALIGN_MATCH_IOU:-0.35}"
DIRECT_SELECTION_POLICY="${DIRECT_SELECTION_POLICY:-proposal_topk_rerank}"
DIRECT_PROPOSAL_TOP_M="${DIRECT_PROPOSAL_TOP_M:-8}"
SELECTION_METRIC="${SELECTION_METRIC:-deploy_align_topreturn}"

EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
RUN_DIR="${OUT_BASE}/${RUN_NAME}"

mkdir -p "${RUN_DIR}"

TRAIN_LABEL_DIR="${TRAIN_LABEL_DIR}" \
VAL_LABEL_DIR="${VAL_LABEL_DIR}" \
TEST_LABEL_DIR="${TEST_LABEL_DIR}" \
OUT_BASE="${OUT_BASE}" \
SELECTION_METRIC="${SELECTION_METRIC}" \
EXTRA_TRAIN_ARGS="\
  --generated_proposal_align_weight ${GENERATED_ALIGN_WEIGHT} \
  --generated_proposal_score_weight ${GENERATED_ALIGN_SCORE_WEIGHT} \
  --generated_proposal_listwise_weight ${GENERATED_ALIGN_LISTWISE_WEIGHT} \
  --generated_proposal_positive_weight ${GENERATED_ALIGN_POSITIVE_WEIGHT} \
  --generated_proposal_risk_weight ${GENERATED_ALIGN_RISK_WEIGHT} \
  --generated_proposal_match_iou ${GENERATED_ALIGN_MATCH_IOU} \
  ${EXTRA_TRAIN_ARGS}" \
bash src/scripts/run_mobilecropnet_v4_sstk_product_experiment.sh "${PROFILE}" "${RUN_NAME}"

/usr/local/bin/python3 src/scripts/evaluate_mobilecropnet_v4_product_ar_direct.py \
  --checkpoint "${RUN_DIR}/best.pt" \
  --eval_jsonl "${VAL_LABEL_DIR}/train_conditional_detr_batch.jsonl" \
  --project_root . \
  --output_dir "${RUN_DIR}/direct_val_${DIRECT_SELECTION_POLICY}" \
  --selection_policy "${DIRECT_SELECTION_POLICY}" \
  --proposal_top_m "${DIRECT_PROPOSAL_TOP_M}" \
  --batch_size 24 \
  --num_workers 4 \
  --device cuda \
  2>&1 | tee "${RUN_DIR}/direct_val_${DIRECT_SELECTION_POLICY}.log"

/usr/local/bin/python3 src/scripts/evaluate_mobilecropnet_v4_product_ar_direct.py \
  --checkpoint "${RUN_DIR}/best.pt" \
  --eval_jsonl "${TEST_LABEL_DIR}/train_conditional_detr_batch.jsonl" \
  --project_root . \
  --output_dir "${RUN_DIR}/direct_test_${DIRECT_SELECTION_POLICY}" \
  --selection_policy "${DIRECT_SELECTION_POLICY}" \
  --proposal_top_m "${DIRECT_PROPOSAL_TOP_M}" \
  --batch_size 24 \
  --num_workers 4 \
  --device cuda \
  2>&1 | tee "${RUN_DIR}/direct_test_${DIRECT_SELECTION_POLICY}.log"

/usr/local/bin/python3 src/scripts/evaluate_mobilecropnet_v4_product_ar_direct.py \
  --checkpoint "${RUN_DIR}/best.pt" \
  --eval_jsonl "${TEST_LABEL_DIR}/train_conditional_detr_batch.jsonl" \
  --project_root . \
  --output_dir "${RUN_DIR}/direct_test_utility_top1" \
  --selection_policy utility_top1 \
  --proposal_top_m "${DIRECT_PROPOSAL_TOP_M}" \
  --batch_size 24 \
  --num_workers 4 \
  --device cuda \
  2>&1 | tee "${RUN_DIR}/direct_test_utility_top1.log"

/usr/local/bin/python3 src/scripts/visualize_mobilecropnet_v4_public_ar_comparison.py \
  --checkpoint "${RUN_DIR}/best.pt" \
  --output_dir "${RUN_DIR}/deploy_viz_gaic_test" \
  --annotations_json data/Publics/GAIC_v2/annotations_json/instances_test.json \
  --image_roots data/Publics/GAIC_v2/images/test data/Publics/GAIC/images/test \
  --label_jsonl "${TEST_LABEL_DIR}/train_conditional_detr_batch.jsonl" \
  --selection_policy "${DIRECT_SELECTION_POLICY}" \
  --proposal_top_m "${DIRECT_PROPOSAL_TOP_M}" \
  --max_images 24 \
  --device cuda \
  2>&1 | tee "${RUN_DIR}/deploy_viz_gaic_test.log"

/usr/local/bin/python3 src/scripts/visualize_mobilecropnet_v4_public_ar_comparison.py \
  --checkpoint "${RUN_DIR}/best.pt" \
  --output_dir "${RUN_DIR}/deploy_viz_test_images" \
  --image_glob 'data/test_images/*' \
  --selection_policy "${DIRECT_SELECTION_POLICY}" \
  --proposal_top_m "${DIRECT_PROPOSAL_TOP_M}" \
  --max_images 24 \
  --device cuda \
  2>&1 | tee "${RUN_DIR}/deploy_viz_test_images.log"
