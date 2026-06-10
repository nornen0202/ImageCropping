#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/group-volume/users/jaden.ju/Sources/ImageCropping}"
RUN_NAME="${RUN_NAME:?RUN_NAME required}"
VARIANT="${VARIANT:-ranker}"
OUT_BASE="${OUT_BASE:-artifacts/mobilecropnet_v4/quality_first_20260427/public_utility_tune}"
OUT_DIR="${OUT_DIR:-${OUT_BASE}/${RUN_NAME}}"
TRAIN_JSONL="${TRAIN_JSONL:-data/SSTK/Full_10000/artifacts/training_labels_public_ensemble_best_product_ar_260422_hashsplit_80_10_10_260422/train/train_conditional_detr_batch.jsonl}"
VAL_JSONL="${VAL_JSONL:-data/SSTK/Full_10000/artifacts/training_labels_public_ensemble_best_product_ar_260422_hashsplit_80_10_10_260422/val/train_conditional_detr_batch.jsonl}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-artifacts/mobilecropnet_v4/quality_first_20260427/route_smokes/sstk_uctr_supporttarget/mcn-q-b448-intprior-auxfine-nomargin-r2-ip134-20260427/best.pt}"
EPOCHS="${EPOCHS:-2}"
LR="${LR:-4e-5}"
INPUT_SIZE="${INPUT_SIZE:-448}"
CANDIDATE_K="${CANDIDATE_K:-64}"
PROPOSAL_Q="${PROPOSAL_Q:-64}"
WIDTH_MULT="${WIDTH_MULT:-1.0}"
TOKEN_DIM="${TOKEN_DIM:-384}"
BACKBONE_NAME="${BACKBONE_NAME:-convnextv2_base.fcmae_ft_in22k_in1k_384}"
RANKER_TYPE="${RANKER_TYPE:-set_transformer}"
RANKER_DEPTH="${RANKER_DEPTH:-4}"
SUBJECT_BOX_SPATIAL_MULTISCALE="${SUBJECT_BOX_SPATIAL_MULTISCALE:-1}"
SUBJECT_BOX_SPATIAL_OUTPUT="${SUBJECT_BOX_SPATIAL_OUTPUT:-spatial}"
SUBJECT_BOX_SPATIAL_BOX_MODE="${SUBJECT_BOX_SPATIAL_BOX_MODE:-mask_moment_regress}"
ROUTE_HEAD_DEPTH="${ROUTE_HEAD_DEPTH:-3}"
ROUTE_HEAD_HIDDEN_MULT="${ROUTE_HEAD_HIDDEN_MULT:-0.75}"
ROUTE_DECODE_MODE="${ROUTE_DECODE_MODE:-hierarchical}"
ROUTE_USE_SUBJECT_SPATIAL_TOKEN="${ROUTE_USE_SUBJECT_SPATIAL_TOKEN:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-}"
MAX_VAL_ROWS="${MAX_VAL_ROWS:-}"
LIMIT_TRAIN_STEPS="${LIMIT_TRAIN_STEPS:-}"
LIMIT_VAL_STEPS="${LIMIT_VAL_STEPS:-}"
DECISION_WEIGHT="${DECISION_WEIGHT:-}"
ACTION_CONSISTENCY_WEIGHT="${ACTION_CONSISTENCY_WEIGHT:-}"
DECISION_UTILITY_ALIGN_WEIGHT="${DECISION_UTILITY_ALIGN_WEIGHT:-}"
POLICY_SCORE_WEIGHT="${POLICY_SCORE_WEIGHT:-}"
DELTA_WEIGHT="${DELTA_WEIGHT:-}"

case "${VARIANT}" in
  ranker)
    TRAINABLE_PREFIXES="candidate_encoder,relation_mlp,relation_norm,set_ranker,utility_head,positive_head,risk_head"
    ;;
  headonly)
    TRAINABLE_PREFIXES="utility_head,positive_head,risk_head"
    ;;
  ranker_detail)
    TRAINABLE_PREFIXES="candidate_encoder,relation_mlp,relation_norm,set_ranker,utility_head,positive_head,risk_head,macro_head,detail_score_head"
    ;;
  ranker_policy)
    TRAINABLE_PREFIXES="candidate_encoder,relation_mlp,relation_norm,set_ranker,utility_head,positive_head,risk_head,policy_head,decision_head,delta_head,policy_score_predictor"
    DECISION_WEIGHT="${DECISION_WEIGHT:-0.2}"
    ACTION_CONSISTENCY_WEIGHT="${ACTION_CONSISTENCY_WEIGHT:-0.25}"
    DECISION_UTILITY_ALIGN_WEIGHT="${DECISION_UTILITY_ALIGN_WEIGHT:-0.2}"
    POLICY_SCORE_WEIGHT="${POLICY_SCORE_WEIGHT:-0.15}"
    DELTA_WEIGHT="${DELTA_WEIGHT:-0.1}"
    ;;
  full_ranker)
    TRAINABLE_PREFIXES=""
    ;;
  full_policy)
    TRAINABLE_PREFIXES=""
    DECISION_WEIGHT="${DECISION_WEIGHT:-0.2}"
    ACTION_CONSISTENCY_WEIGHT="${ACTION_CONSISTENCY_WEIGHT:-0.25}"
    DECISION_UTILITY_ALIGN_WEIGHT="${DECISION_UTILITY_ALIGN_WEIGHT:-0.2}"
    POLICY_SCORE_WEIGHT="${POLICY_SCORE_WEIGHT:-0.15}"
    DELTA_WEIGHT="${DELTA_WEIGHT:-0.1}"
    ;;
  *)
    echo "unknown VARIANT: ${VARIANT}" >&2
    exit 2
    ;;
esac

DECISION_WEIGHT="${DECISION_WEIGHT:-0.0}"
ACTION_CONSISTENCY_WEIGHT="${ACTION_CONSISTENCY_WEIGHT:-0.0}"
DECISION_UTILITY_ALIGN_WEIGHT="${DECISION_UTILITY_ALIGN_WEIGHT:-0.0}"
POLICY_SCORE_WEIGHT="${POLICY_SCORE_WEIGHT:-0.0}"
DELTA_WEIGHT="${DELTA_WEIGHT:-0.0}"

cd "${ROOT}"
mkdir -p "${OUT_DIR}"

EXTRA_LIMIT_ARGS=()
if [[ -n "${MAX_TRAIN_ROWS}" ]]; then
  EXTRA_LIMIT_ARGS+=(--max_train_rows "${MAX_TRAIN_ROWS}")
fi
if [[ -n "${MAX_VAL_ROWS}" ]]; then
  EXTRA_LIMIT_ARGS+=(--max_val_rows "${MAX_VAL_ROWS}")
fi
if [[ -n "${LIMIT_TRAIN_STEPS}" ]]; then
  EXTRA_LIMIT_ARGS+=(--limit_train_steps "${LIMIT_TRAIN_STEPS}")
fi
if [[ -n "${LIMIT_VAL_STEPS}" ]]; then
  EXTRA_LIMIT_ARGS+=(--limit_val_steps "${LIMIT_VAL_STEPS}")
fi
EXTRA_MODEL_ARGS=()
case "${SUBJECT_BOX_SPATIAL_MULTISCALE,,}" in
  1|true|yes|on)
    EXTRA_MODEL_ARGS+=(--subject_box_spatial_multiscale)
    ;;
esac
case "${ROUTE_USE_SUBJECT_SPATIAL_TOKEN,,}" in
  1|true|yes|on)
    EXTRA_MODEL_ARGS+=(--route_use_subject_spatial_token)
    ;;
esac
INIT_ARGS=()
if [[ -n "${INIT_CHECKPOINT}" ]]; then
  INIT_ARGS+=(--init_checkpoint "${INIT_CHECKPOINT}")
fi
TRAINABLE_ARGS=()
if [[ -n "${TRAINABLE_PREFIXES}" ]]; then
  TRAINABLE_ARGS+=(--trainable_module_prefixes "${TRAINABLE_PREFIXES}")
fi

/usr/local/bin/python3 - <<PY
import json
from pathlib import Path

payload = {
    "run_name": ${RUN_NAME@Q},
    "variant": ${VARIANT@Q},
    "trainable_prefixes": ${TRAINABLE_PREFIXES@Q},
    "init_checkpoint": ${INIT_CHECKPOINT@Q},
    "train_jsonl": ${TRAIN_JSONL@Q},
    "val_jsonl": ${VAL_JSONL@Q},
    "out_dir": ${OUT_DIR@Q},
    "epochs": ${EPOCHS@Q},
    "lr": ${LR@Q},
    "input_size": ${INPUT_SIZE@Q},
    "candidate_k": ${CANDIDATE_K@Q},
    "proposal_q": ${PROPOSAL_Q@Q},
    "width_mult": ${WIDTH_MULT@Q},
    "token_dim": ${TOKEN_DIM@Q},
    "backbone_name": ${BACKBONE_NAME@Q},
    "ranker_type": ${RANKER_TYPE@Q},
    "ranker_depth": ${RANKER_DEPTH@Q},
    "subject_box_spatial_multiscale": ${SUBJECT_BOX_SPATIAL_MULTISCALE@Q},
    "subject_box_spatial_output": ${SUBJECT_BOX_SPATIAL_OUTPUT@Q},
    "subject_box_spatial_box_mode": ${SUBJECT_BOX_SPATIAL_BOX_MODE@Q},
    "route_head_depth": ${ROUTE_HEAD_DEPTH@Q},
    "route_head_hidden_mult": ${ROUTE_HEAD_HIDDEN_MULT@Q},
    "route_decode_mode": ${ROUTE_DECODE_MODE@Q},
    "route_use_subject_spatial_token": ${ROUTE_USE_SUBJECT_SPATIAL_TOKEN@Q},
    "decision_weight": ${DECISION_WEIGHT@Q},
    "action_consistency_weight": ${ACTION_CONSISTENCY_WEIGHT@Q},
    "decision_utility_align_weight": ${DECISION_UTILITY_ALIGN_WEIGHT@Q},
    "policy_score_weight": ${POLICY_SCORE_WEIGHT@Q},
    "delta_weight": ${DELTA_WEIGHT@Q},
    "batch_size": ${BATCH_SIZE@Q},
    "num_workers": ${NUM_WORKERS@Q},
}
Path(${OUT_DIR@Q}, "public_utility_tune_launch_config.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

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

/usr/local/bin/python3 src/scripts/train_mobilecropnet_v4.py \
  --train_jsonl "${TRAIN_JSONL}" \
  --val_jsonl "${VAL_JSONL}" \
  --project_root "${ROOT}" \
  --output_dir "${OUT_DIR}" \
  --model_profile none \
  --input_size "${INPUT_SIZE}" \
  --candidate_k "${CANDIDATE_K}" \
  --proposal_q "${PROPOSAL_Q}" \
  --width_mult "${WIDTH_MULT}" \
  --token_dim "${TOKEN_DIM}" \
  --backbone_name "${BACKBONE_NAME}" \
  --backbone_pretrained \
  --ranker_type "${RANKER_TYPE}" \
  --ranker_depth "${RANKER_DEPTH}" \
  --use_subject_prior \
  --route_use_subject_prior \
  --route_use_candidate_context \
  --route_use_subject_box_features \
  --route_aux_heads \
  --route_head_depth "${ROUTE_HEAD_DEPTH}" \
  --route_head_hidden_mult "${ROUTE_HEAD_HIDDEN_MULT}" \
  --route_decode_mode "${ROUTE_DECODE_MODE}" \
  --policy_use_subject_prior \
  --proposal_use_subject_prior \
  --subject_box_head \
  --subject_box_spatial_head \
  --subject_box_spatial_output "${SUBJECT_BOX_SPATIAL_OUTPUT}" \
  --subject_box_spatial_box_mode "${SUBJECT_BOX_SPATIAL_BOX_MODE}" \
  --subject_box_coord_space content \
  "${EXTRA_MODEL_ARGS[@]}" \
  --use_pred_subject_box_as_prior \
  --detach_pred_subject_prior \
  --policy_score_head \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --min_lr 1e-6 \
  --warmup_epochs 0 \
  --weight_decay 1e-4 \
  --num_workers "${NUM_WORKERS}" \
  --amp \
  "${INIT_ARGS[@]}" \
  "${TRAINABLE_ARGS[@]}" \
  --score_target_mode hybrid_rank \
  --subject_box_target_source support_latent \
  --score_weight 1.0 \
  --listwise_weight 1.0 \
  --pairwise_weight 0.8 \
  --explicit_pairwise_weight 0.4 \
  --positive_weight 0.5 \
  --risk_weight 0.25 \
  --top1_risk_weight 0.25 \
  --top_return_weight 0.8 \
  --teacher_distill_weight 0.0 \
  --macro_weight 0.0 \
  --checklist_class_weight 0.0 \
  --checklist_applicability_weight 0.0 \
  --detail_score_weight 0.0 \
  --why_tag_weight 0.0 \
  --route_weight 0.0 \
  --decision_weight "${DECISION_WEIGHT}" \
  --action_consistency_weight "${ACTION_CONSISTENCY_WEIGHT}" \
  --decision_utility_align_weight "${DECISION_UTILITY_ALIGN_WEIGHT}" \
  --policy_score_weight "${POLICY_SCORE_WEIGHT}" \
  --delta_weight "${DELTA_WEIGHT}" \
  --proposal_weight 0.0 \
  --proposal_subject_weight 0.0 \
  --subject_proposal_align_weight 0.0 \
  --subject_box_weight 0.0 \
  --subject_box_valid_weight 0.0 \
  --generated_proposal_align_weight 0.0 \
  --selection_metric sstk_balanced_topreturn \
  --progress_log_interval 25 \
  --device cuda \
  "${EXTRA_LIMIT_ARGS[@]}" \
  2>&1 | tee "${OUT_DIR}/train.log"

date -u +"end_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "${OUT_DIR}/environment.log"
