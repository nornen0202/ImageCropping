#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:?variant required}"
RUN_NAME="${2:?run name required}"

ROOT="${ROOT:-/group-volume/users/jaden.ju/Sources/ImageCropping}"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python3}"
OUT_BASE="${OUT_BASE:-artifacts/mobilecropnet_v4/release_gate_v2_20260424/route_smokes/sstk_public}"
RUN_DIR="${OUT_BASE}/${RUN_NAME}"
STATUS_JSON="${RUN_DIR}/route_smoke_status.json"

TRAIN_JSONL="${TRAIN_JSONL:-data/SSTK/Full_10000/artifacts/training_labels_public_ensemble_best_product_ar_260422_hashsplit_80_10_10_260422/train/train_conditional_detr_batch.jsonl}"
VAL_JSONL="${VAL_JSONL:-data/SSTK/Full_10000/artifacts/training_labels_public_ensemble_best_product_ar_260422_hashsplit_80_10_10_260422/val/train_conditional_detr_batch.jsonl}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-artifacts/mobilecropnet_v4/subject_box_iou_patch_20260424_full/sstk_uctr_subjectprior/mcn-sstk-uctr-subjboxpatch-r320-20260424/best.pt}"
BALANCED_ROUTE_INIT_CHECKPOINT="${BALANCED_ROUTE_INIT_CHECKPOINT:-artifacts/mobilecropnet_v4/route_gate_recover_20260424/full_runs/sstk_public/mcn-sstk-public-routefix-balanced-288-20260424/best.pt}"

cd "${ROOT}"
mkdir -p "${RUN_DIR}"

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
    "variant": ${VARIANT@Q},
    "run_name": ${RUN_NAME@Q},
    "run_dir": ${RUN_DIR@Q},
    "train_log": str(Path(${RUN_DIR@Q}) / "train.log"),
    "config_path": str(Path(${RUN_DIR@Q}) / "config.json"),
    "metrics_path": str(Path(${RUN_DIR@Q}) / "metrics.json"),
    "summary_path": str(Path(${RUN_DIR@Q}) / "summary.json"),
    "train_status": str(Path(${RUN_DIR@Q}) / "train_status.json"),
    "best_checkpoint": str(Path(${RUN_DIR@Q}) / "best.pt"),
    "latest_checkpoint": str(Path(${RUN_DIR@Q}) / "last.pt"),
}
Path(${STATUS_JSON@Q}).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

trap 'write_status failed failed' ERR

PROFILE="${PROFILE:-rank_320}"
EPOCHS="${EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-24}"
LR="${LR:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
SEED="${SEED:-20260427}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-12000}"
MAX_VAL_ROWS="${MAX_VAL_ROWS:-2400}"
LIMIT_TRAIN_STEPS="${LIMIT_TRAIN_STEPS:-500}"
LIMIT_VAL_STEPS="${LIMIT_VAL_STEPS:-100}"
ROUTE_SAMPLER_MAX_WEIGHT="${ROUTE_SAMPLER_MAX_WEIGHT:-4.0}"
ROUTE_WEIGHT="${ROUTE_WEIGHT:-8.0}"
SELECTION_METRIC="${SELECTION_METRIC:-route_only}"
SCORE_TARGET_MODE="${SCORE_TARGET_MODE:-}"
SUBJECT_BOX_TARGET_SOURCE="${SUBJECT_BOX_TARGET_SOURCE:-support_latent}"

declare -a VARIANT_ARGS
declare -a INIT_ARGS=()
declare -a EXTRA_TRAIN_ARGS=()
if [[ -n "${EXTRA_TRAIN_ARGS_TEXT:-}" ]]; then
  read -r -a EXTRA_TRAIN_ARGS <<< "${EXTRA_TRAIN_ARGS_TEXT}"
fi
declare -a WEIGHT_ARGS=(
  --score_weight 0.0
  --listwise_weight 0.0
  --pairwise_weight 0.0
  --explicit_pairwise_weight 0.0
  --positive_weight 0.0
  --risk_weight 0.0
  --top1_risk_weight 0.0
  --top_return_weight 0.0
  --macro_weight 0.0
  --checklist_class_weight 0.0
  --checklist_applicability_weight 0.0
  --detail_score_weight 0.0
  --why_tag_weight 0.0
  --decision_weight 0.0
  --policy_score_weight 0.0
  --delta_weight 0.0
  --proposal_weight 0.0
  --proposal_subject_weight 0.0
  --subject_proposal_align_weight 0.0
  --subject_box_weight 0.0
  --subject_box_valid_weight 0.0
  --generated_proposal_align_weight 0.0
)
case "${VARIANT}" in
  ctx_freezesubj_lr1e4)
    EPOCHS="${EPOCHS:-5}"
    LR="${LR:-0.0001}"
    VARIANT_ARGS=(
      --use_subject_prior
      --route_use_subject_prior
      --route_use_candidate_context
      --route_use_subject_box_features
      --proposal_use_subject_prior
      --subject_box_head
      --subject_box_coord_space content
      --subject_box_refine_with_proposals
      --use_pred_subject_box_as_prior
      --detach_pred_subject_prior
      --route_balanced_ce
      --train_route_balanced_sampler
      --train_route_balanced_sampler_max_weight "${ROUTE_SAMPLER_MAX_WEIGHT}"
      --route_weight "${ROUTE_WEIGHT}"
      --freeze_subject_box_stack
      --init_checkpoint "${INIT_CHECKPOINT}"
      --trainable_module_prefixes candidate_encoder,relation_mlp,relation_norm,set_ranker,utility_head,subject_prior_encoder,route_head
    )
    ;;
  ctx_full_lr2e5)
    LR="${LR:-0.00002}"
    EPOCHS="${EPOCHS:-6}"
    VARIANT_ARGS=(
      --use_subject_prior
      --route_use_subject_prior
      --route_use_candidate_context
      --route_use_subject_box_features
      --proposal_use_subject_prior
      --subject_box_head
      --subject_box_coord_space content
      --subject_box_refine_with_proposals
      --use_pred_subject_box_as_prior
      --detach_pred_subject_prior
      --route_balanced_ce
      --train_route_balanced_sampler
      --train_route_balanced_sampler_max_weight "${ROUTE_SAMPLER_MAX_WEIGHT}"
      --route_weight "${ROUTE_WEIGHT}"
      --init_checkpoint "${INIT_CHECKPOINT}"
      --trainable_module_prefixes backbone,ar_embed,cond_proj,candidate_encoder,relation_mlp,relation_norm,set_ranker,utility_head,subject_prior_encoder,route_head
    )
    ;;
  ctx_focal_margin_lr1e4)
    EPOCHS="${EPOCHS:-5}"
    LR="${LR:-0.0001}"
    ROUTE_SAMPLER_MAX_WEIGHT="${ROUTE_SAMPLER_MAX_WEIGHT:-6.0}"
    VARIANT_ARGS=(
      --use_subject_prior
      --route_use_subject_prior
      --route_use_candidate_context
      --route_use_subject_box_features
      --proposal_use_subject_prior
      --subject_box_head
      --subject_box_coord_space content
      --subject_box_refine_with_proposals
      --use_pred_subject_box_as_prior
      --detach_pred_subject_prior
      --route_balanced_ce
      --route_focal_gamma 1.5
      --train_route_balanced_sampler
      --train_route_balanced_sampler_max_weight "${ROUTE_SAMPLER_MAX_WEIGHT}"
      --route_weight "${ROUTE_WEIGHT}"
      --route_object_single_margin_weight 0.12
      --route_object_single_margin 0.18
      --init_checkpoint "${INIT_CHECKPOINT}"
      --trainable_module_prefixes subject_prior_encoder,route_head
    )
    ;;
  ctx_focal_freezesubj_lr8e5)
    EPOCHS="${EPOCHS:-5}"
    LR="${LR:-0.00008}"
    ROUTE_SAMPLER_MAX_WEIGHT="${ROUTE_SAMPLER_MAX_WEIGHT:-6.0}"
    VARIANT_ARGS=(
      --use_subject_prior
      --route_use_subject_prior
      --route_use_candidate_context
      --route_use_subject_box_features
      --proposal_use_subject_prior
      --subject_box_head
      --subject_box_coord_space content
      --subject_box_refine_with_proposals
      --use_pred_subject_box_as_prior
      --detach_pred_subject_prior
      --route_balanced_ce
      --route_focal_gamma 1.25
      --train_route_balanced_sampler
      --train_route_balanced_sampler_max_weight "${ROUTE_SAMPLER_MAX_WEIGHT}"
      --route_weight "${ROUTE_WEIGHT}"
      --freeze_subject_box_stack
      --init_checkpoint "${INIT_CHECKPOINT}"
      --trainable_module_prefixes candidate_encoder,relation_mlp,relation_norm,set_ranker,utility_head,subject_prior_encoder,route_head
    )
    ;;
  b288_routeinit_subj_joint_lr8e5)
    PROFILE="balanced_288"
    EPOCHS="${EPOCHS:-6}"
    BATCH_SIZE="28"
    LR="${LR:-0.00008}"
    MAX_TRAIN_ROWS="16000"
    MAX_VAL_ROWS="3200"
    LIMIT_TRAIN_STEPS="650"
    LIMIT_VAL_STEPS="120"
    ROUTE_SAMPLER_MAX_WEIGHT="6.0"
    SELECTION_METRIC="route_balanced"
    WEIGHT_ARGS=(
      --score_weight 0.15
      --listwise_weight 0.0
      --pairwise_weight 0.0
      --explicit_pairwise_weight 0.0
      --positive_weight 0.05
      --risk_weight 0.0
      --top1_risk_weight 0.0
      --top_return_weight 0.0
      --macro_weight 0.0
      --checklist_class_weight 0.0
      --checklist_applicability_weight 0.0
      --detail_score_weight 0.0
      --why_tag_weight 0.0
      --decision_weight 0.0
      --policy_score_weight 0.0
      --delta_weight 0.0
      --proposal_weight 0.35
      --proposal_subject_weight 0.18
      --subject_proposal_align_weight 0.15
      --subject_box_weight 0.55
      --subject_box_valid_weight 1.2
      --subject_box_l1_weight 2.0
      --subject_box_iou_weight 1.0
      --subject_box_center_weight 0.5
      --subject_box_size_weight 0.25
      --subject_box_aspect_weight 0.15
      --subject_box_ciou_weight 0.5
      --subject_box_valid_balanced_bce
      --generated_proposal_align_weight 0.0
    )
    VARIANT_ARGS=(
      --use_subject_prior
      --route_use_subject_prior
      --route_use_candidate_context
      --route_use_subject_box_features
      --proposal_use_subject_prior
      --subject_box_head
      --subject_box_coord_space content
      --subject_box_refine_with_proposals
      --use_pred_subject_box_as_prior
      --detach_pred_subject_prior
      --pred_subject_prior_start_epoch 1
      --pred_subject_prior_warmup_epochs 3
      --route_balanced_ce
      --train_route_balanced_sampler
      --train_route_balanced_sampler_max_weight "${ROUTE_SAMPLER_MAX_WEIGHT}"
      --route_weight "${ROUTE_WEIGHT}"
      --init_checkpoint "${BALANCED_ROUTE_INIT_CHECKPOINT}"
      --trainable_module_prefixes candidate_encoder,relation_mlp,relation_norm,set_ranker,utility_head,subject_prior_encoder,route_head,subject_box_head,subject_valid_head,subject_proposal_head,subject_refine_proj,subject_refine_box_head,subject_refine_valid_head
    )
    ;;
  b288_routeinit_subj_full_lr4e5)
    PROFILE="balanced_288"
    EPOCHS="${EPOCHS:-6}"
    BATCH_SIZE="24"
    LR="${LR:-0.00004}"
    MAX_TRAIN_ROWS="16000"
    MAX_VAL_ROWS="3200"
    LIMIT_TRAIN_STEPS="650"
    LIMIT_VAL_STEPS="120"
    ROUTE_SAMPLER_MAX_WEIGHT="6.0"
    SELECTION_METRIC="release_gate"
    WEIGHT_ARGS=(
      --score_weight 0.35
      --listwise_weight 0.0
      --pairwise_weight 0.0
      --explicit_pairwise_weight 0.0
      --positive_weight 0.08
      --risk_weight 0.0
      --top1_risk_weight 0.0
      --top_return_weight 0.0
      --macro_weight 0.0
      --checklist_class_weight 0.0
      --checklist_applicability_weight 0.0
      --detail_score_weight 0.0
      --why_tag_weight 0.0
      --decision_weight 0.0
      --policy_score_weight 0.0
      --delta_weight 0.0
      --proposal_weight 0.45
      --proposal_subject_weight 0.20
      --subject_proposal_align_weight 0.18
      --subject_box_weight 0.70
      --subject_box_valid_weight 1.5
      --subject_box_l1_weight 2.0
      --subject_box_iou_weight 1.0
      --subject_box_center_weight 0.5
      --subject_box_size_weight 0.25
      --subject_box_aspect_weight 0.15
      --subject_box_ciou_weight 0.5
      --subject_box_valid_balanced_bce
      --generated_proposal_align_weight 0.0
    )
    VARIANT_ARGS=(
      --use_subject_prior
      --route_use_subject_prior
      --route_use_candidate_context
      --route_use_subject_box_features
      --proposal_use_subject_prior
      --subject_box_head
      --subject_box_coord_space content
      --subject_box_refine_with_proposals
      --use_pred_subject_box_as_prior
      --detach_pred_subject_prior
      --pred_subject_prior_start_epoch 1
      --pred_subject_prior_warmup_epochs 3
      --route_balanced_ce
      --train_route_balanced_sampler
      --train_route_balanced_sampler_max_weight "${ROUTE_SAMPLER_MAX_WEIGHT}"
      --route_weight "${ROUTE_WEIGHT}"
      --init_checkpoint "${BALANCED_ROUTE_INIT_CHECKPOINT}"
      --trainable_module_prefixes backbone,ar_embed,cond_proj,candidate_encoder,relation_mlp,relation_norm,set_ranker,utility_head,subject_prior_encoder,route_head,subject_box_head,subject_valid_head,subject_proposal_head,subject_refine_proj,subject_refine_box_head,subject_refine_valid_head
    )
    ;;
  b288_public_subjpatch_align_lr6e5)
    PROFILE="balanced_288"
    EPOCHS="${EPOCHS:-7}"
    BATCH_SIZE="24"
    LR="${LR:-0.00006}"
    MAX_TRAIN_ROWS="18000"
    MAX_VAL_ROWS="3600"
    LIMIT_TRAIN_STEPS="750"
    LIMIT_VAL_STEPS="140"
    ROUTE_SAMPLER_MAX_WEIGHT="7.0"
    ROUTE_WEIGHT="${ROUTE_WEIGHT:-9.0}"
    SELECTION_METRIC="release_gate"
    SCORE_TARGET_MODE="${SCORE_TARGET_MODE:-hybrid_rank}"
    WEIGHT_ARGS=(
      --score_weight 0.42
      --listwise_weight 0.08
      --pairwise_weight 0.0
      --explicit_pairwise_weight 0.0
      --positive_weight 0.10
      --risk_weight 0.08
      --top1_risk_weight 0.04
      --top_return_weight 0.0
      --macro_weight 0.0
      --checklist_class_weight 0.0
      --checklist_applicability_weight 0.0
      --detail_score_weight 0.0
      --why_tag_weight 0.0
      --decision_weight 0.0
      --policy_score_weight 0.0
      --delta_weight 0.0
      --proposal_weight 0.55
      --proposal_subject_weight 0.22
      --subject_proposal_align_weight 0.22
      --subject_box_weight 0.90
      --subject_box_valid_weight 1.4
      --subject_box_l1_weight 2.2
      --subject_box_iou_weight 1.2
      --subject_box_center_weight 0.55
      --subject_box_size_weight 0.30
      --subject_box_aspect_weight 0.18
      --subject_box_ciou_weight 0.7
      --subject_box_valid_balanced_bce
      --subject_box_valid_negative_scale 0.75
      --generated_proposal_align_weight 0.35
      --generated_proposal_score_weight 1.0
      --generated_proposal_listwise_weight 0.6
      --generated_proposal_positive_weight 0.45
      --generated_proposal_risk_weight 0.25
      --generated_proposal_match_iou 0.30
      --teacher_distill_weight 0.08
      --teacher_distill_temperature 0.16
    )
    VARIANT_ARGS=(
      --use_subject_prior
      --route_use_subject_prior
      --route_use_candidate_context
      --route_use_subject_box_features
      --proposal_use_subject_prior
      --subject_box_head
      --subject_box_coord_space content
      --subject_box_refine_with_proposals
      --use_pred_subject_box_as_prior
      --detach_pred_subject_prior
      --pred_subject_prior_start_epoch 1
      --pred_subject_prior_warmup_epochs 3
      --route_balanced_ce
      --route_focal_gamma 1.0
      --train_route_balanced_sampler
      --train_route_balanced_sampler_max_weight "${ROUTE_SAMPLER_MAX_WEIGHT}"
      --route_weight "${ROUTE_WEIGHT}"
      --init_checkpoint "${INIT_CHECKPOINT}"
      --trainable_module_prefixes ar_embed,cond_proj,candidate_encoder,relation_mlp,relation_norm,set_ranker,utility_head,proposal_queries,proposal_head,proposal_score_head,proposal_positive_head,proposal_risk_head,subject_prior_encoder,route_head,subject_box_head,subject_valid_head,subject_proposal_head,subject_refine_proj,subject_refine_box_head,subject_refine_valid_head
    )
    ;;
  b288_public_routefix_align_lr5e5)
    PROFILE="balanced_288"
    EPOCHS="${EPOCHS:-7}"
    BATCH_SIZE="26"
    LR="${LR:-0.00005}"
    MAX_TRAIN_ROWS="18000"
    MAX_VAL_ROWS="3600"
    LIMIT_TRAIN_STEPS="750"
    LIMIT_VAL_STEPS="140"
    ROUTE_SAMPLER_MAX_WEIGHT="7.0"
    ROUTE_WEIGHT="${ROUTE_WEIGHT:-9.0}"
    SELECTION_METRIC="release_gate"
    SCORE_TARGET_MODE="${SCORE_TARGET_MODE:-hybrid_rank}"
    WEIGHT_ARGS=(
      --score_weight 0.45
      --listwise_weight 0.08
      --pairwise_weight 0.0
      --explicit_pairwise_weight 0.0
      --positive_weight 0.10
      --risk_weight 0.08
      --top1_risk_weight 0.04
      --top_return_weight 0.0
      --macro_weight 0.0
      --checklist_class_weight 0.0
      --checklist_applicability_weight 0.0
      --detail_score_weight 0.0
      --why_tag_weight 0.0
      --decision_weight 0.0
      --policy_score_weight 0.0
      --delta_weight 0.0
      --proposal_weight 0.60
      --proposal_subject_weight 0.22
      --subject_proposal_align_weight 0.24
      --subject_box_weight 0.80
      --subject_box_valid_weight 1.3
      --subject_box_l1_weight 2.0
      --subject_box_iou_weight 1.1
      --subject_box_center_weight 0.50
      --subject_box_size_weight 0.28
      --subject_box_aspect_weight 0.18
      --subject_box_ciou_weight 0.6
      --subject_box_valid_balanced_bce
      --subject_box_valid_negative_scale 0.85
      --generated_proposal_align_weight 0.45
      --generated_proposal_score_weight 1.0
      --generated_proposal_listwise_weight 0.65
      --generated_proposal_positive_weight 0.50
      --generated_proposal_risk_weight 0.25
      --generated_proposal_match_iou 0.30
      --teacher_distill_weight 0.08
      --teacher_distill_temperature 0.16
    )
    VARIANT_ARGS=(
      --use_subject_prior
      --route_use_subject_prior
      --route_use_candidate_context
      --route_use_subject_box_features
      --proposal_use_subject_prior
      --subject_box_head
      --subject_box_coord_space content
      --subject_box_refine_with_proposals
      --use_pred_subject_box_as_prior
      --detach_pred_subject_prior
      --pred_subject_prior_start_epoch 1
      --pred_subject_prior_warmup_epochs 3
      --route_balanced_ce
      --route_focal_gamma 1.0
      --train_route_balanced_sampler
      --train_route_balanced_sampler_max_weight "${ROUTE_SAMPLER_MAX_WEIGHT}"
      --route_weight "${ROUTE_WEIGHT}"
      --init_checkpoint "${BALANCED_ROUTE_INIT_CHECKPOINT}"
      --trainable_module_prefixes ar_embed,cond_proj,candidate_encoder,relation_mlp,relation_norm,set_ranker,utility_head,proposal_queries,proposal_head,proposal_score_head,proposal_positive_head,proposal_risk_head,subject_prior_encoder,route_head,subject_box_head,subject_valid_head,subject_proposal_head,subject_refine_proj,subject_refine_box_head,subject_refine_valid_head
    )
    ;;
  quality_cnv2b448_joint_lr5e5)
    PROFILE="quality_cnv2b_448"
    EPOCHS="${EPOCHS:-5}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    LR="${LR:-0.00005}"
    MAX_TRAIN_ROWS="14000"
    MAX_VAL_ROWS="3000"
    LIMIT_TRAIN_STEPS="520"
    LIMIT_VAL_STEPS="120"
    ROUTE_SAMPLER_MAX_WEIGHT="${ROUTE_SAMPLER_MAX_WEIGHT:-8.0}"
    ROUTE_WEIGHT="${ROUTE_WEIGHT:-7.5}"
    SELECTION_METRIC="release_gate_joint"
    SCORE_TARGET_MODE="${SCORE_TARGET_MODE:-hybrid_rank}"
    WEIGHT_ARGS=(
      --score_weight 0.48
      --listwise_weight 0.10
      --pairwise_weight 0.0
      --explicit_pairwise_weight 0.0
      --positive_weight 0.12
      --risk_weight 0.08
      --top1_risk_weight 0.05
      --top_return_weight 0.04
      --macro_weight 0.02
      --checklist_class_weight 0.03
      --checklist_applicability_weight 0.02
      --detail_score_weight 0.04
      --why_tag_weight 0.02
      --decision_weight 0.12
      --policy_score_weight 0.12
      --delta_weight 0.04
      --decision_utility_align_weight 0.10
      --action_consistency_weight 0.24
      --proposal_weight 0.65
      --proposal_subject_weight 0.25
      --subject_proposal_align_weight 0.26
      --subject_box_weight 0.95
      --subject_box_valid_weight 1.6
      --subject_box_l1_weight 2.2
      --subject_box_iou_weight 1.2
      --subject_box_center_weight 0.55
      --subject_box_size_weight 0.30
      --subject_box_aspect_weight 0.18
      --subject_box_ciou_weight 0.7
      --subject_box_valid_balanced_bce
      --subject_box_valid_negative_scale 0.85
      --generated_proposal_align_weight 0.50
      --generated_proposal_score_weight 1.0
      --generated_proposal_listwise_weight 0.70
      --generated_proposal_positive_weight 0.55
      --generated_proposal_risk_weight 0.28
      --generated_proposal_positive_utility_weight 0.45
      --generated_proposal_positive_margin_weight 0.25
      --generated_proposal_positive_margin 0.25
      --generated_proposal_match_iou 0.30
    )
    VARIANT_ARGS=(
      --use_subject_prior
      --route_use_subject_prior
      --route_use_candidate_context
      --route_use_subject_box_features
      --policy_use_subject_prior
      --proposal_use_subject_prior
      --subject_box_head
      --subject_box_coord_space content
      --subject_box_refine_with_proposals
      --use_pred_subject_box_as_prior
      --detach_pred_subject_prior
      --pred_subject_prior_start_epoch 2
      --pred_subject_prior_warmup_epochs 2
      --route_balanced_ce
      --route_focal_gamma 1.0
      --train_route_balanced_sampler
      --train_route_balanced_sampler_max_weight "${ROUTE_SAMPLER_MAX_WEIGHT}"
      --route_weight "${ROUTE_WEIGHT}"
      --route_object_single_margin_weight 0.12
      --route_object_single_margin 0.18
      --policy_score_head
    )
    ;;
  quality_cnv2l512_joint_lr3e5)
    PROFILE="quality_cnv2l_512"
    EPOCHS="${EPOCHS:-4}"
    BATCH_SIZE="4"
    LR="${LR:-0.00003}"
    MAX_TRAIN_ROWS="10000"
    MAX_VAL_ROWS="2400"
    LIMIT_TRAIN_STEPS="420"
    LIMIT_VAL_STEPS="100"
    ROUTE_SAMPLER_MAX_WEIGHT="${ROUTE_SAMPLER_MAX_WEIGHT:-8.0}"
    ROUTE_WEIGHT="${ROUTE_WEIGHT:-8.0}"
    SELECTION_METRIC="release_gate_joint"
    SCORE_TARGET_MODE="${SCORE_TARGET_MODE:-hybrid_rank}"
    WEIGHT_ARGS=(
      --score_weight 0.46
      --listwise_weight 0.08
      --pairwise_weight 0.0
      --explicit_pairwise_weight 0.0
      --positive_weight 0.12
      --risk_weight 0.08
      --top1_risk_weight 0.05
      --top_return_weight 0.03
      --macro_weight 0.02
      --checklist_class_weight 0.03
      --checklist_applicability_weight 0.02
      --detail_score_weight 0.04
      --why_tag_weight 0.02
      --decision_weight 0.12
      --policy_score_weight 0.12
      --delta_weight 0.04
      --decision_utility_align_weight 0.10
      --action_consistency_weight 0.24
      --proposal_weight 0.65
      --proposal_subject_weight 0.25
      --subject_proposal_align_weight 0.26
      --subject_box_weight 0.95
      --subject_box_valid_weight 1.6
      --subject_box_l1_weight 2.2
      --subject_box_iou_weight 1.2
      --subject_box_center_weight 0.55
      --subject_box_size_weight 0.30
      --subject_box_aspect_weight 0.18
      --subject_box_ciou_weight 0.7
      --subject_box_valid_balanced_bce
      --subject_box_valid_negative_scale 0.85
      --generated_proposal_align_weight 0.50
      --generated_proposal_score_weight 1.0
      --generated_proposal_listwise_weight 0.70
      --generated_proposal_positive_weight 0.55
      --generated_proposal_risk_weight 0.28
      --generated_proposal_positive_utility_weight 0.45
      --generated_proposal_positive_margin_weight 0.25
      --generated_proposal_positive_margin 0.25
      --generated_proposal_match_iou 0.30
    )
    VARIANT_ARGS=(
      --use_subject_prior
      --route_use_subject_prior
      --route_use_candidate_context
      --route_use_subject_box_features
      --policy_use_subject_prior
      --proposal_use_subject_prior
      --subject_box_head
      --subject_box_coord_space content
      --subject_box_refine_with_proposals
      --use_pred_subject_box_as_prior
      --detach_pred_subject_prior
      --pred_subject_prior_start_epoch 2
      --pred_subject_prior_warmup_epochs 2
      --route_balanced_ce
      --route_focal_gamma 1.0
      --train_route_balanced_sampler
      --train_route_balanced_sampler_max_weight "${ROUTE_SAMPLER_MAX_WEIGHT}"
      --route_weight "${ROUTE_WEIGHT}"
      --route_object_single_margin_weight 0.12
      --route_object_single_margin 0.18
      --policy_score_head
    )
    ;;
  quality_cnv2b448_genrank_lr3e5)
    PROFILE="quality_cnv2b_448"
    EPOCHS="${EPOCHS:-5}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    LR="${LR:-0.00003}"
    MAX_TRAIN_ROWS="16000"
    MAX_VAL_ROWS="3200"
    LIMIT_TRAIN_STEPS="620"
    LIMIT_VAL_STEPS="140"
    ROUTE_SAMPLER_MAX_WEIGHT="${ROUTE_SAMPLER_MAX_WEIGHT:-8.0}"
    ROUTE_WEIGHT="${ROUTE_WEIGHT:-8.5}"
    SELECTION_METRIC="release_gate_joint"
    SCORE_TARGET_MODE="${SCORE_TARGET_MODE:-hybrid_rank}"
    SUBJECT_BOX_VALID_WEIGHT="${SUBJECT_BOX_VALID_WEIGHT:-1.9}"
    SUBJECT_BOX_VALID_NEGATIVE_SCALE="${SUBJECT_BOX_VALID_NEGATIVE_SCALE:-1.60}"
    GENERATED_POSITIVE_UTILITY_WEIGHT="${GENERATED_POSITIVE_UTILITY_WEIGHT:-1.20}"
    GENERATED_POSITIVE_MARGIN_WEIGHT="${GENERATED_POSITIVE_MARGIN_WEIGHT:-0.70}"
    if [[ -n "${QUALITY_INIT_CHECKPOINT:-}" ]]; then
      INIT_ARGS=(--init_checkpoint "${QUALITY_INIT_CHECKPOINT}")
    fi
    WEIGHT_ARGS=(
      --score_weight 0.42
      --listwise_weight 0.08
      --pairwise_weight 0.0
      --explicit_pairwise_weight 0.0
      --positive_weight 0.12
      --risk_weight 0.08
      --top1_risk_weight 0.05
      --top_return_weight 0.05
      --macro_weight 0.02
      --checklist_class_weight 0.03
      --checklist_applicability_weight 0.02
      --detail_score_weight 0.04
      --why_tag_weight 0.02
      --decision_weight 0.12
      --policy_score_weight 0.12
      --delta_weight 0.04
      --decision_utility_align_weight 0.10
      --action_consistency_weight 0.24
      --proposal_weight 0.70
      --proposal_subject_weight 0.30
      --subject_proposal_align_weight 0.30
      --subject_box_weight 1.05
      --subject_box_valid_weight "${SUBJECT_BOX_VALID_WEIGHT}"
      --subject_box_l1_weight 2.2
      --subject_box_iou_weight 1.2
      --subject_box_center_weight 0.55
      --subject_box_size_weight 0.30
      --subject_box_aspect_weight 0.18
      --subject_box_ciou_weight 0.7
      --subject_box_valid_balanced_bce
      --subject_box_valid_negative_scale "${SUBJECT_BOX_VALID_NEGATIVE_SCALE}"
      --generated_proposal_align_weight 0.95
      --generated_proposal_score_weight 0.8
      --generated_proposal_listwise_weight 0.35
      --generated_proposal_positive_weight 0.65
      --generated_proposal_risk_weight 0.28
      --generated_proposal_positive_utility_weight "${GENERATED_POSITIVE_UTILITY_WEIGHT}"
      --generated_proposal_positive_margin_weight "${GENERATED_POSITIVE_MARGIN_WEIGHT}"
      --generated_proposal_positive_margin 0.25
      --generated_proposal_match_iou 0.25
    )
    VARIANT_ARGS=(
      --use_subject_prior
      --route_use_subject_prior
      --route_use_candidate_context
      --route_use_subject_box_features
      --policy_use_subject_prior
      --proposal_use_subject_prior
      --subject_box_head
      --subject_box_coord_space content
      --subject_box_refine_with_proposals
      --use_pred_subject_box_as_prior
      --detach_pred_subject_prior
      --pred_subject_prior_start_epoch 1
      --pred_subject_prior_warmup_epochs 3
      --route_balanced_ce
      --route_focal_gamma 1.15
      --train_route_balanced_sampler
      --train_route_balanced_sampler_max_weight "${ROUTE_SAMPLER_MAX_WEIGHT}"
      --route_weight "${ROUTE_WEIGHT}"
      --route_object_single_margin_weight 0.14
      --route_object_single_margin 0.18
      --policy_score_head
    )
    ;;
  quality_cnv2b448_spatial_genrank_lr2e5)
    PROFILE="${QUALITY_PROFILE:-quality_cnv2b_448}"
    EPOCHS="${EPOCHS:-5}"
    BATCH_SIZE="${BATCH_SIZE:-8}"
    LR="${LR:-0.00002}"
    MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-18000}"
    MAX_VAL_ROWS="${MAX_VAL_ROWS:-3600}"
    LIMIT_TRAIN_STEPS="${LIMIT_TRAIN_STEPS:-700}"
    LIMIT_VAL_STEPS="${LIMIT_VAL_STEPS:-160}"
    ROUTE_SAMPLER_MAX_WEIGHT="${ROUTE_SAMPLER_MAX_WEIGHT:-8.0}"
    ROUTE_WEIGHT="${ROUTE_WEIGHT:-9.0}"
    SELECTION_METRIC="release_gate_joint"
    SCORE_TARGET_MODE="${SCORE_TARGET_MODE:-hybrid_rank}"
    SUBJECT_BOX_VALID_WEIGHT="${SUBJECT_BOX_VALID_WEIGHT:-2.2}"
    SUBJECT_BOX_VALID_NEGATIVE_SCALE="${SUBJECT_BOX_VALID_NEGATIVE_SCALE:-2.20}"
    SUBJECT_BOX_SPATIAL_AUX_WEIGHT="${SUBJECT_BOX_SPATIAL_AUX_WEIGHT:-0.45}"
    SUBJECT_BOX_SPATIAL_HEATMAP_WEIGHT="${SUBJECT_BOX_SPATIAL_HEATMAP_WEIGHT:-0.35}"
    SUBJECT_BOX_SPATIAL_MASK_WEIGHT="${SUBJECT_BOX_SPATIAL_MASK_WEIGHT:-0.0}"
    SUBJECT_BOX_SPATIAL_MIX_BIAS="${SUBJECT_BOX_SPATIAL_MIX_BIAS:--1.0}"
    SUBJECT_BOX_SPATIAL_MULTISCALE="${SUBJECT_BOX_SPATIAL_MULTISCALE:-0}"
    SUBJECT_BOX_SPATIAL_OUTPUT="${SUBJECT_BOX_SPATIAL_OUTPUT:-mix}"
    SUBJECT_BOX_SPATIAL_BOX_MODE="${SUBJECT_BOX_SPATIAL_BOX_MODE:-regress}"
    SUBJECT_BOX_SPATIAL_EXTENT_SCALE="${SUBJECT_BOX_SPATIAL_EXTENT_SCALE:-1.0}"
    SUBJECT_BOX_REFINE_WITH_PROPOSALS="${SUBJECT_BOX_REFINE_WITH_PROPOSALS:-1}"
    GENERATED_POSITIVE_UTILITY_WEIGHT="${GENERATED_POSITIVE_UTILITY_WEIGHT:-1.00}"
    GENERATED_POSITIVE_MARGIN_WEIGHT="${GENERATED_POSITIVE_MARGIN_WEIGHT:-0.60}"
    ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT="${ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT:-0.14}"
    ROUTE_OBJECT_SINGLE_MARGIN="${ROUTE_OBJECT_SINGLE_MARGIN:-0.18}"
    ROUTE_HEAD_DEPTH="${ROUTE_HEAD_DEPTH:-1}"
    ROUTE_HEAD_HIDDEN_MULT="${ROUTE_HEAD_HIDDEN_MULT:-0.5}"
    ROUTE_HEAD_DROPOUT="${ROUTE_HEAD_DROPOUT:-0.0}"
    ROUTE_AUX_HEADS="${ROUTE_AUX_HEADS:-0}"
    ROUTE_DECODE_MODE="${ROUTE_DECODE_MODE:-fine}"
    ROUTE_DECODE_KIND_WEIGHT="${ROUTE_DECODE_KIND_WEIGHT:-1.0}"
    ROUTE_DECODE_CARDINALITY_WEIGHT="${ROUTE_DECODE_CARDINALITY_WEIGHT:-1.0}"
    ROUTE_HIERARCHY_WEIGHT="${ROUTE_HIERARCHY_WEIGHT:-0.0}"
    ROUTE_CARDINALITY_WEIGHT="${ROUTE_CARDINALITY_WEIGHT:-0.0}"
    if [[ -n "${QUALITY_INIT_CHECKPOINT:-}" ]]; then
      INIT_ARGS=(--init_checkpoint "${QUALITY_INIT_CHECKPOINT}")
    fi
    WEIGHT_ARGS=(
      --score_weight 0.40
      --listwise_weight 0.08
      --pairwise_weight 0.0
      --explicit_pairwise_weight 0.0
      --positive_weight 0.12
      --risk_weight 0.08
      --top1_risk_weight 0.05
      --top_return_weight 0.05
      --macro_weight 0.02
      --checklist_class_weight 0.03
      --checklist_applicability_weight 0.02
      --detail_score_weight 0.04
      --why_tag_weight 0.02
      --decision_weight 0.12
      --policy_score_weight 0.12
      --delta_weight 0.04
      --decision_utility_align_weight 0.10
      --action_consistency_weight 0.24
      --proposal_weight 0.66
      --proposal_subject_weight 0.32
      --subject_proposal_align_weight 0.32
      --subject_box_weight 1.15
      --subject_box_valid_weight "${SUBJECT_BOX_VALID_WEIGHT}"
      --subject_box_l1_weight 2.2
      --subject_box_iou_weight 1.3
      --subject_box_center_weight 0.65
      --subject_box_size_weight 0.36
      --subject_box_aspect_weight 0.18
      --subject_box_ciou_weight 0.8
      --subject_box_spatial_aux_weight "${SUBJECT_BOX_SPATIAL_AUX_WEIGHT}"
      --subject_box_spatial_heatmap_weight "${SUBJECT_BOX_SPATIAL_HEATMAP_WEIGHT}"
      --subject_box_spatial_mask_weight "${SUBJECT_BOX_SPATIAL_MASK_WEIGHT}"
      --subject_box_valid_balanced_bce
      --subject_box_valid_negative_scale "${SUBJECT_BOX_VALID_NEGATIVE_SCALE}"
      --generated_proposal_align_weight 0.85
      --generated_proposal_score_weight 0.8
      --generated_proposal_listwise_weight 0.35
      --generated_proposal_positive_weight 0.65
      --generated_proposal_risk_weight 0.28
      --generated_proposal_positive_utility_weight "${GENERATED_POSITIVE_UTILITY_WEIGHT}"
      --generated_proposal_positive_margin_weight "${GENERATED_POSITIVE_MARGIN_WEIGHT}"
      --generated_proposal_positive_margin 0.25
      --generated_proposal_match_iou 0.25
    )
    VARIANT_ARGS=(
      --use_subject_prior
      --route_use_subject_prior
      --route_use_candidate_context
      --route_use_subject_box_features
      --policy_use_subject_prior
      --proposal_use_subject_prior
      --subject_box_head
      --subject_box_spatial_head
      $(if [[ "${SUBJECT_BOX_SPATIAL_MULTISCALE}" == "1" || "${SUBJECT_BOX_SPATIAL_MULTISCALE}" == "true" ]]; then printf '%s' "--subject_box_spatial_multiscale"; fi)
      --subject_box_spatial_mix_bias "${SUBJECT_BOX_SPATIAL_MIX_BIAS}"
      --subject_box_spatial_output "${SUBJECT_BOX_SPATIAL_OUTPUT}"
      --subject_box_spatial_box_mode "${SUBJECT_BOX_SPATIAL_BOX_MODE}"
      --subject_box_spatial_extent_scale "${SUBJECT_BOX_SPATIAL_EXTENT_SCALE}"
      --subject_box_coord_space content
      $(if [[ "${SUBJECT_BOX_REFINE_WITH_PROPOSALS}" == "1" || "${SUBJECT_BOX_REFINE_WITH_PROPOSALS}" == "true" ]]; then printf '%s' "--subject_box_refine_with_proposals"; fi)
      --use_pred_subject_box_as_prior
      --detach_pred_subject_prior
      --pred_subject_prior_start_epoch 1
      --pred_subject_prior_warmup_epochs 3
      --route_balanced_ce
      --route_focal_gamma 1.15
      --train_route_balanced_sampler
      --train_route_balanced_sampler_max_weight "${ROUTE_SAMPLER_MAX_WEIGHT}"
      --route_weight "${ROUTE_WEIGHT}"
      --route_head_depth "${ROUTE_HEAD_DEPTH}"
      --route_head_hidden_mult "${ROUTE_HEAD_HIDDEN_MULT}"
      --route_head_dropout "${ROUTE_HEAD_DROPOUT}"
      $(if [[ "${ROUTE_AUX_HEADS}" == "1" || "${ROUTE_AUX_HEADS}" == "true" ]]; then printf '%s' "--route_aux_heads"; fi)
      --route_decode_mode "${ROUTE_DECODE_MODE}"
      --route_decode_kind_weight "${ROUTE_DECODE_KIND_WEIGHT}"
      --route_decode_cardinality_weight "${ROUTE_DECODE_CARDINALITY_WEIGHT}"
      --route_hierarchy_weight "${ROUTE_HIERARCHY_WEIGHT}"
      --route_cardinality_weight "${ROUTE_CARDINALITY_WEIGHT}"
      --route_object_single_margin_weight "${ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT}"
      --route_object_single_margin "${ROUTE_OBJECT_SINGLE_MARGIN}"
      --policy_score_head
    )
    ;;
  quality_cnv2b448_subject_only_lr2e5)
    PROFILE="quality_cnv2b_448"
    EPOCHS="${EPOCHS:-5}"
    BATCH_SIZE="${SUBJECT_BATCH_SIZE:-6}"
    LR="${LR:-0.00002}"
    MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-22000}"
    MAX_VAL_ROWS="${MAX_VAL_ROWS:-4200}"
    LIMIT_TRAIN_STEPS="${LIMIT_TRAIN_STEPS:-800}"
    LIMIT_VAL_STEPS="${LIMIT_VAL_STEPS:-180}"
    SELECTION_METRIC="subject_box"
    SUBJECT_BOX_VALID_WEIGHT="${SUBJECT_BOX_VALID_WEIGHT:-1.8}"
    SUBJECT_BOX_VALID_NEGATIVE_SCALE="${SUBJECT_BOX_VALID_NEGATIVE_SCALE:-1.4}"
    SUBJECT_BOX_SPATIAL_AUX_WEIGHT="${SUBJECT_BOX_SPATIAL_AUX_WEIGHT:-1.2}"
    SUBJECT_BOX_SPATIAL_HEATMAP_WEIGHT="${SUBJECT_BOX_SPATIAL_HEATMAP_WEIGHT:-0.75}"
    SUBJECT_BOX_SPATIAL_MASK_WEIGHT="${SUBJECT_BOX_SPATIAL_MASK_WEIGHT:-0.0}"
    SUBJECT_BOX_SPATIAL_MIX_BIAS="${SUBJECT_BOX_SPATIAL_MIX_BIAS:-0.0}"
    WEIGHT_ARGS=(
      --score_weight 0.0
      --listwise_weight 0.0
      --pairwise_weight 0.0
      --explicit_pairwise_weight 0.0
      --positive_weight 0.0
      --risk_weight 0.0
      --top1_risk_weight 0.0
      --top_return_weight 0.0
      --macro_weight 0.0
      --checklist_class_weight 0.0
      --checklist_applicability_weight 0.0
      --detail_score_weight 0.0
      --why_tag_weight 0.0
      --route_weight 0.0
      --decision_weight 0.0
      --policy_score_weight 0.0
      --delta_weight 0.0
      --proposal_weight "${SUBJECT_ONLY_PROPOSAL_WEIGHT:-0.0}"
      --proposal_subject_weight "${SUBJECT_ONLY_PROPOSAL_SUBJECT_WEIGHT:-0.0}"
      --subject_proposal_align_weight "${SUBJECT_ONLY_SUBJECT_PROPOSAL_ALIGN_WEIGHT:-0.0}"
      --subject_box_weight 1.0
      --subject_box_valid_weight "${SUBJECT_BOX_VALID_WEIGHT}"
      --subject_box_l1_weight 2.8
      --subject_box_iou_weight 1.8
      --subject_box_center_weight 0.8
      --subject_box_size_weight 0.45
      --subject_box_aspect_weight 0.22
      --subject_box_ciou_weight 1.0
      --subject_box_spatial_aux_weight "${SUBJECT_BOX_SPATIAL_AUX_WEIGHT}"
      --subject_box_spatial_heatmap_weight "${SUBJECT_BOX_SPATIAL_HEATMAP_WEIGHT}"
      --subject_box_spatial_mask_weight "${SUBJECT_BOX_SPATIAL_MASK_WEIGHT}"
      --subject_box_valid_balanced_bce
      --subject_box_valid_negative_scale "${SUBJECT_BOX_VALID_NEGATIVE_SCALE}"
      --generated_proposal_align_weight 0.0
    )
    VARIANT_ARGS=(
      --subject_box_head
      --subject_box_spatial_head
      --subject_box_spatial_multiscale
      --subject_box_spatial_mix_bias "${SUBJECT_BOX_SPATIAL_MIX_BIAS}"
      --subject_box_coord_space content
      $(if [[ "${SUBJECT_BOX_REFINE_WITH_PROPOSALS:-0}" == "1" || "${SUBJECT_BOX_REFINE_WITH_PROPOSALS:-0}" == "true" ]]; then printf '%s' "--subject_box_refine_with_proposals"; fi)
    )
    ;;
  quality_cnv2l512_subject_only_lr8e6)
    PROFILE="quality_cnv2l_512"
    EPOCHS="${EPOCHS:-5}"
    BATCH_SIZE="${SUBJECT_BATCH_SIZE:-2}"
    LR="${LR:-0.000008}"
    MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-18000}"
    MAX_VAL_ROWS="${MAX_VAL_ROWS:-3600}"
    LIMIT_TRAIN_STEPS="${LIMIT_TRAIN_STEPS:-650}"
    LIMIT_VAL_STEPS="${LIMIT_VAL_STEPS:-150}"
    SELECTION_METRIC="subject_box"
    SUBJECT_BOX_VALID_WEIGHT="${SUBJECT_BOX_VALID_WEIGHT:-1.8}"
    SUBJECT_BOX_VALID_NEGATIVE_SCALE="${SUBJECT_BOX_VALID_NEGATIVE_SCALE:-1.4}"
    SUBJECT_BOX_SPATIAL_AUX_WEIGHT="${SUBJECT_BOX_SPATIAL_AUX_WEIGHT:-1.2}"
    SUBJECT_BOX_SPATIAL_HEATMAP_WEIGHT="${SUBJECT_BOX_SPATIAL_HEATMAP_WEIGHT:-0.75}"
    SUBJECT_BOX_SPATIAL_MASK_WEIGHT="${SUBJECT_BOX_SPATIAL_MASK_WEIGHT:-0.0}"
    SUBJECT_BOX_SPATIAL_MIX_BIAS="${SUBJECT_BOX_SPATIAL_MIX_BIAS:-0.0}"
    WEIGHT_ARGS=(
      --score_weight 0.0
      --listwise_weight 0.0
      --pairwise_weight 0.0
      --explicit_pairwise_weight 0.0
      --positive_weight 0.0
      --risk_weight 0.0
      --top1_risk_weight 0.0
      --top_return_weight 0.0
      --macro_weight 0.0
      --checklist_class_weight 0.0
      --checklist_applicability_weight 0.0
      --detail_score_weight 0.0
      --why_tag_weight 0.0
      --route_weight 0.0
      --decision_weight 0.0
      --policy_score_weight 0.0
      --delta_weight 0.0
      --proposal_weight "${SUBJECT_ONLY_PROPOSAL_WEIGHT:-0.0}"
      --proposal_subject_weight "${SUBJECT_ONLY_PROPOSAL_SUBJECT_WEIGHT:-0.0}"
      --subject_proposal_align_weight "${SUBJECT_ONLY_SUBJECT_PROPOSAL_ALIGN_WEIGHT:-0.0}"
      --subject_box_weight 1.0
      --subject_box_valid_weight "${SUBJECT_BOX_VALID_WEIGHT}"
      --subject_box_l1_weight 2.8
      --subject_box_iou_weight 1.8
      --subject_box_center_weight 0.8
      --subject_box_size_weight 0.45
      --subject_box_aspect_weight 0.22
      --subject_box_ciou_weight 1.0
      --subject_box_spatial_aux_weight "${SUBJECT_BOX_SPATIAL_AUX_WEIGHT}"
      --subject_box_spatial_heatmap_weight "${SUBJECT_BOX_SPATIAL_HEATMAP_WEIGHT}"
      --subject_box_spatial_mask_weight "${SUBJECT_BOX_SPATIAL_MASK_WEIGHT}"
      --subject_box_valid_balanced_bce
      --subject_box_valid_negative_scale "${SUBJECT_BOX_VALID_NEGATIVE_SCALE}"
      --generated_proposal_align_weight 0.0
    )
    VARIANT_ARGS=(
      --subject_box_head
      --subject_box_spatial_head
      --subject_box_spatial_multiscale
      --subject_box_spatial_mix_bias "${SUBJECT_BOX_SPATIAL_MIX_BIAS}"
      --subject_box_coord_space content
      $(if [[ "${SUBJECT_BOX_REFINE_WITH_PROPOSALS:-0}" == "1" || "${SUBJECT_BOX_REFINE_WITH_PROPOSALS:-0}" == "true" ]]; then printf '%s' "--subject_box_refine_with_proposals"; fi)
    )
    ;;
  quality_eva02b448_subject_only_lr1e5)
    PROFILE="quality_eva02b_subject_448"
    EPOCHS="${EPOCHS:-5}"
    BATCH_SIZE="${SUBJECT_BATCH_SIZE:-4}"
    LR="${LR:-0.00001}"
    MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-18000}"
    MAX_VAL_ROWS="${MAX_VAL_ROWS:-3600}"
    LIMIT_TRAIN_STEPS="${LIMIT_TRAIN_STEPS:-700}"
    LIMIT_VAL_STEPS="${LIMIT_VAL_STEPS:-160}"
    SELECTION_METRIC="subject_box"
    SUBJECT_BOX_VALID_WEIGHT="${SUBJECT_BOX_VALID_WEIGHT:-1.8}"
    SUBJECT_BOX_VALID_NEGATIVE_SCALE="${SUBJECT_BOX_VALID_NEGATIVE_SCALE:-1.4}"
    SUBJECT_BOX_SPATIAL_AUX_WEIGHT="${SUBJECT_BOX_SPATIAL_AUX_WEIGHT:-1.2}"
    SUBJECT_BOX_SPATIAL_HEATMAP_WEIGHT="${SUBJECT_BOX_SPATIAL_HEATMAP_WEIGHT:-0.75}"
    SUBJECT_BOX_SPATIAL_MASK_WEIGHT="${SUBJECT_BOX_SPATIAL_MASK_WEIGHT:-0.0}"
    SUBJECT_BOX_SPATIAL_MIX_BIAS="${SUBJECT_BOX_SPATIAL_MIX_BIAS:-0.0}"
    WEIGHT_ARGS=(
      --score_weight 0.0
      --listwise_weight 0.0
      --pairwise_weight 0.0
      --explicit_pairwise_weight 0.0
      --positive_weight 0.0
      --risk_weight 0.0
      --top1_risk_weight 0.0
      --top_return_weight 0.0
      --macro_weight 0.0
      --checklist_class_weight 0.0
      --checklist_applicability_weight 0.0
      --detail_score_weight 0.0
      --why_tag_weight 0.0
      --route_weight 0.0
      --decision_weight 0.0
      --policy_score_weight 0.0
      --delta_weight 0.0
      --proposal_weight "${SUBJECT_ONLY_PROPOSAL_WEIGHT:-0.0}"
      --proposal_subject_weight "${SUBJECT_ONLY_PROPOSAL_SUBJECT_WEIGHT:-0.0}"
      --subject_proposal_align_weight "${SUBJECT_ONLY_SUBJECT_PROPOSAL_ALIGN_WEIGHT:-0.0}"
      --subject_box_weight 1.0
      --subject_box_valid_weight "${SUBJECT_BOX_VALID_WEIGHT}"
      --subject_box_l1_weight 2.8
      --subject_box_iou_weight 1.8
      --subject_box_center_weight 0.8
      --subject_box_size_weight 0.45
      --subject_box_aspect_weight 0.22
      --subject_box_ciou_weight 1.0
      --subject_box_spatial_aux_weight "${SUBJECT_BOX_SPATIAL_AUX_WEIGHT}"
      --subject_box_spatial_heatmap_weight "${SUBJECT_BOX_SPATIAL_HEATMAP_WEIGHT}"
      --subject_box_spatial_mask_weight "${SUBJECT_BOX_SPATIAL_MASK_WEIGHT}"
      --subject_box_valid_balanced_bce
      --subject_box_valid_negative_scale "${SUBJECT_BOX_VALID_NEGATIVE_SCALE}"
      --generated_proposal_align_weight 0.0
    )
    VARIANT_ARGS=(
      --subject_box_head
      --subject_box_spatial_head
      --subject_box_spatial_multiscale
      --subject_box_spatial_mix_bias "${SUBJECT_BOX_SPATIAL_MIX_BIAS}"
      --subject_box_coord_space content
      $(if [[ "${SUBJECT_BOX_REFINE_WITH_PROPOSALS:-0}" == "1" || "${SUBJECT_BOX_REFINE_WITH_PROPOSALS:-0}" == "true" ]]; then printf '%s' "--subject_box_refine_with_proposals"; fi)
    )
    ;;
  quality_swinv2b384_subject_only_lr8e6)
    PROFILE="quality_swinv2b_subject_384"
    EPOCHS="${EPOCHS:-5}"
    BATCH_SIZE="${SUBJECT_BATCH_SIZE:-4}"
    LR="${LR:-0.000008}"
    MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-18000}"
    MAX_VAL_ROWS="${MAX_VAL_ROWS:-3600}"
    LIMIT_TRAIN_STEPS="${LIMIT_TRAIN_STEPS:-650}"
    LIMIT_VAL_STEPS="${LIMIT_VAL_STEPS:-150}"
    SELECTION_METRIC="subject_box"
    SUBJECT_BOX_VALID_WEIGHT="${SUBJECT_BOX_VALID_WEIGHT:-1.8}"
    SUBJECT_BOX_VALID_NEGATIVE_SCALE="${SUBJECT_BOX_VALID_NEGATIVE_SCALE:-1.4}"
    SUBJECT_BOX_SPATIAL_AUX_WEIGHT="${SUBJECT_BOX_SPATIAL_AUX_WEIGHT:-1.2}"
    SUBJECT_BOX_SPATIAL_HEATMAP_WEIGHT="${SUBJECT_BOX_SPATIAL_HEATMAP_WEIGHT:-0.75}"
    SUBJECT_BOX_SPATIAL_MASK_WEIGHT="${SUBJECT_BOX_SPATIAL_MASK_WEIGHT:-0.0}"
    SUBJECT_BOX_SPATIAL_MIX_BIAS="${SUBJECT_BOX_SPATIAL_MIX_BIAS:-0.0}"
    WEIGHT_ARGS=(
      --score_weight 0.0
      --listwise_weight 0.0
      --pairwise_weight 0.0
      --explicit_pairwise_weight 0.0
      --positive_weight 0.0
      --risk_weight 0.0
      --top1_risk_weight 0.0
      --top_return_weight 0.0
      --macro_weight 0.0
      --checklist_class_weight 0.0
      --checklist_applicability_weight 0.0
      --detail_score_weight 0.0
      --why_tag_weight 0.0
      --route_weight 0.0
      --decision_weight 0.0
      --policy_score_weight 0.0
      --delta_weight 0.0
      --proposal_weight "${SUBJECT_ONLY_PROPOSAL_WEIGHT:-0.0}"
      --proposal_subject_weight "${SUBJECT_ONLY_PROPOSAL_SUBJECT_WEIGHT:-0.0}"
      --subject_proposal_align_weight "${SUBJECT_ONLY_SUBJECT_PROPOSAL_ALIGN_WEIGHT:-0.0}"
      --subject_box_weight 1.0
      --subject_box_valid_weight "${SUBJECT_BOX_VALID_WEIGHT}"
      --subject_box_l1_weight 2.8
      --subject_box_iou_weight 1.8
      --subject_box_center_weight 0.8
      --subject_box_size_weight 0.45
      --subject_box_aspect_weight 0.22
      --subject_box_ciou_weight 1.0
      --subject_box_spatial_aux_weight "${SUBJECT_BOX_SPATIAL_AUX_WEIGHT}"
      --subject_box_spatial_heatmap_weight "${SUBJECT_BOX_SPATIAL_HEATMAP_WEIGHT}"
      --subject_box_spatial_mask_weight "${SUBJECT_BOX_SPATIAL_MASK_WEIGHT}"
      --subject_box_valid_balanced_bce
      --subject_box_valid_negative_scale "${SUBJECT_BOX_VALID_NEGATIVE_SCALE}"
      --generated_proposal_align_weight 0.0
    )
    VARIANT_ARGS=(
      --subject_box_head
      --subject_box_spatial_head
      --subject_box_spatial_multiscale
      --subject_box_spatial_mix_bias "${SUBJECT_BOX_SPATIAL_MIX_BIAS}"
      --subject_box_coord_space content
      $(if [[ "${SUBJECT_BOX_REFINE_WITH_PROPOSALS:-0}" == "1" || "${SUBJECT_BOX_REFINE_WITH_PROPOSALS:-0}" == "true" ]]; then printf '%s' "--subject_box_refine_with_proposals"; fi)
    )
    ;;
  quality_swinv2b384_spatial_genrank_lr1e5)
    PROFILE="quality_swinv2b_384"
    EPOCHS="${EPOCHS:-5}"
    BATCH_SIZE="${BATCH_SIZE:-6}"
    LR="${LR:-0.00001}"
    MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-18000}"
    MAX_VAL_ROWS="${MAX_VAL_ROWS:-3600}"
    LIMIT_TRAIN_STEPS="${LIMIT_TRAIN_STEPS:-700}"
    LIMIT_VAL_STEPS="${LIMIT_VAL_STEPS:-160}"
    ROUTE_SAMPLER_MAX_WEIGHT="${ROUTE_SAMPLER_MAX_WEIGHT:-8.0}"
    ROUTE_WEIGHT="${ROUTE_WEIGHT:-9.0}"
    SELECTION_METRIC="release_gate_joint"
    SCORE_TARGET_MODE="${SCORE_TARGET_MODE:-hybrid_rank}"
    SUBJECT_BOX_VALID_WEIGHT="${SUBJECT_BOX_VALID_WEIGHT:-2.2}"
    SUBJECT_BOX_VALID_NEGATIVE_SCALE="${SUBJECT_BOX_VALID_NEGATIVE_SCALE:-2.0}"
    SUBJECT_BOX_SPATIAL_AUX_WEIGHT="${SUBJECT_BOX_SPATIAL_AUX_WEIGHT:-0.55}"
    SUBJECT_BOX_SPATIAL_HEATMAP_WEIGHT="${SUBJECT_BOX_SPATIAL_HEATMAP_WEIGHT:-0.35}"
    SUBJECT_BOX_SPATIAL_MASK_WEIGHT="${SUBJECT_BOX_SPATIAL_MASK_WEIGHT:-0.0}"
    SUBJECT_BOX_SPATIAL_MIX_BIAS="${SUBJECT_BOX_SPATIAL_MIX_BIAS:--0.5}"
    SUBJECT_BOX_SPATIAL_OUTPUT="${SUBJECT_BOX_SPATIAL_OUTPUT:-mix}"
    SUBJECT_BOX_SPATIAL_BOX_MODE="${SUBJECT_BOX_SPATIAL_BOX_MODE:-regress}"
    SUBJECT_BOX_SPATIAL_EXTENT_SCALE="${SUBJECT_BOX_SPATIAL_EXTENT_SCALE:-1.0}"
    SUBJECT_BOX_REFINE_WITH_PROPOSALS="${SUBJECT_BOX_REFINE_WITH_PROPOSALS:-0}"
    GENERATED_POSITIVE_UTILITY_WEIGHT="${GENERATED_POSITIVE_UTILITY_WEIGHT:-0.95}"
    GENERATED_POSITIVE_MARGIN_WEIGHT="${GENERATED_POSITIVE_MARGIN_WEIGHT:-0.55}"
    ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT="${ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT:-0.12}"
    ROUTE_OBJECT_SINGLE_MARGIN="${ROUTE_OBJECT_SINGLE_MARGIN:-0.18}"
    ROUTE_HEAD_DEPTH="${ROUTE_HEAD_DEPTH:-1}"
    ROUTE_HEAD_HIDDEN_MULT="${ROUTE_HEAD_HIDDEN_MULT:-0.5}"
    ROUTE_HEAD_DROPOUT="${ROUTE_HEAD_DROPOUT:-0.0}"
    ROUTE_AUX_HEADS="${ROUTE_AUX_HEADS:-0}"
    ROUTE_DECODE_MODE="${ROUTE_DECODE_MODE:-fine}"
    ROUTE_DECODE_KIND_WEIGHT="${ROUTE_DECODE_KIND_WEIGHT:-1.0}"
    ROUTE_DECODE_CARDINALITY_WEIGHT="${ROUTE_DECODE_CARDINALITY_WEIGHT:-1.0}"
    ROUTE_HIERARCHY_WEIGHT="${ROUTE_HIERARCHY_WEIGHT:-0.0}"
    ROUTE_CARDINALITY_WEIGHT="${ROUTE_CARDINALITY_WEIGHT:-0.0}"
    if [[ -n "${QUALITY_INIT_CHECKPOINT:-}" ]]; then
      INIT_ARGS=(--init_checkpoint "${QUALITY_INIT_CHECKPOINT}")
    fi
    WEIGHT_ARGS=(
      --score_weight 0.40
      --listwise_weight 0.08
      --pairwise_weight 0.0
      --explicit_pairwise_weight 0.0
      --positive_weight 0.12
      --risk_weight 0.08
      --top1_risk_weight 0.05
      --top_return_weight 0.05
      --macro_weight 0.02
      --checklist_class_weight 0.03
      --checklist_applicability_weight 0.02
      --detail_score_weight 0.04
      --why_tag_weight 0.02
      --decision_weight 0.12
      --policy_score_weight 0.12
      --delta_weight 0.04
      --decision_utility_align_weight 0.10
      --action_consistency_weight 0.24
      --proposal_weight 0.66
      --proposal_subject_weight 0.32
      --subject_proposal_align_weight 0.32
      --subject_box_weight 1.15
      --subject_box_valid_weight "${SUBJECT_BOX_VALID_WEIGHT}"
      --subject_box_l1_weight 2.2
      --subject_box_iou_weight 1.3
      --subject_box_center_weight 0.65
      --subject_box_size_weight 0.36
      --subject_box_aspect_weight 0.18
      --subject_box_ciou_weight 0.8
      --subject_box_spatial_aux_weight "${SUBJECT_BOX_SPATIAL_AUX_WEIGHT}"
      --subject_box_spatial_heatmap_weight "${SUBJECT_BOX_SPATIAL_HEATMAP_WEIGHT}"
      --subject_box_spatial_mask_weight "${SUBJECT_BOX_SPATIAL_MASK_WEIGHT}"
      --subject_box_valid_balanced_bce
      --subject_box_valid_negative_scale "${SUBJECT_BOX_VALID_NEGATIVE_SCALE}"
      --generated_proposal_align_weight 0.85
      --generated_proposal_score_weight 0.8
      --generated_proposal_listwise_weight 0.35
      --generated_proposal_positive_weight 0.65
      --generated_proposal_risk_weight 0.28
      --generated_proposal_positive_utility_weight "${GENERATED_POSITIVE_UTILITY_WEIGHT}"
      --generated_proposal_positive_margin_weight "${GENERATED_POSITIVE_MARGIN_WEIGHT}"
      --generated_proposal_positive_margin 0.25
      --generated_proposal_match_iou 0.25
    )
    VARIANT_ARGS=(
      --use_subject_prior
      --route_use_subject_prior
      --route_use_candidate_context
      --route_use_subject_box_features
      --route_use_subject_spatial_token
      --policy_use_subject_prior
      --proposal_use_subject_prior
      --subject_box_head
      --subject_box_spatial_head
      --subject_box_spatial_multiscale
      --subject_box_spatial_mix_bias "${SUBJECT_BOX_SPATIAL_MIX_BIAS}"
      --subject_box_spatial_output "${SUBJECT_BOX_SPATIAL_OUTPUT}"
      --subject_box_spatial_box_mode "${SUBJECT_BOX_SPATIAL_BOX_MODE}"
      --subject_box_spatial_extent_scale "${SUBJECT_BOX_SPATIAL_EXTENT_SCALE}"
      --subject_box_coord_space content
      $(if [[ "${SUBJECT_BOX_REFINE_WITH_PROPOSALS}" == "1" || "${SUBJECT_BOX_REFINE_WITH_PROPOSALS}" == "true" ]]; then printf '%s' "--subject_box_refine_with_proposals"; fi)
      --use_pred_subject_box_as_prior
      --detach_pred_subject_prior
      --pred_subject_prior_start_epoch 1
      --pred_subject_prior_warmup_epochs 3
      --route_balanced_ce
      --route_focal_gamma 1.15
      --train_route_balanced_sampler
      --train_route_balanced_sampler_max_weight "${ROUTE_SAMPLER_MAX_WEIGHT}"
      --route_weight "${ROUTE_WEIGHT}"
      --route_head_depth "${ROUTE_HEAD_DEPTH}"
      --route_head_hidden_mult "${ROUTE_HEAD_HIDDEN_MULT}"
      --route_head_dropout "${ROUTE_HEAD_DROPOUT}"
      $(if [[ "${ROUTE_AUX_HEADS}" == "1" || "${ROUTE_AUX_HEADS}" == "true" ]]; then printf '%s' "--route_aux_heads"; fi)
      --route_decode_mode "${ROUTE_DECODE_MODE}"
      --route_decode_kind_weight "${ROUTE_DECODE_KIND_WEIGHT}"
      --route_decode_cardinality_weight "${ROUTE_DECODE_CARDINALITY_WEIGHT}"
      --route_hierarchy_weight "${ROUTE_HIERARCHY_WEIGHT}"
      --route_cardinality_weight "${ROUTE_CARDINALITY_WEIGHT}"
      --route_object_single_margin_weight "${ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT}"
      --route_object_single_margin "${ROUTE_OBJECT_SINGLE_MARGIN}"
      --policy_score_head
    )
    ;;
  quality_eva02b448_spatial_genrank_lr8e6)
    PROFILE="quality_eva02b_448"
    EPOCHS="${EPOCHS:-5}"
    BATCH_SIZE="${BATCH_SIZE:-4}"
    LR="${LR:-0.000008}"
    MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-18000}"
    MAX_VAL_ROWS="${MAX_VAL_ROWS:-3600}"
    LIMIT_TRAIN_STEPS="${LIMIT_TRAIN_STEPS:-650}"
    LIMIT_VAL_STEPS="${LIMIT_VAL_STEPS:-150}"
    ROUTE_SAMPLER_MAX_WEIGHT="${ROUTE_SAMPLER_MAX_WEIGHT:-8.0}"
    ROUTE_WEIGHT="${ROUTE_WEIGHT:-8.5}"
    SELECTION_METRIC="release_gate_joint"
    SCORE_TARGET_MODE="${SCORE_TARGET_MODE:-hybrid_rank}"
    SUBJECT_BOX_VALID_WEIGHT="${SUBJECT_BOX_VALID_WEIGHT:-2.2}"
    SUBJECT_BOX_VALID_NEGATIVE_SCALE="${SUBJECT_BOX_VALID_NEGATIVE_SCALE:-2.0}"
    SUBJECT_BOX_SPATIAL_AUX_WEIGHT="${SUBJECT_BOX_SPATIAL_AUX_WEIGHT:-0.55}"
    SUBJECT_BOX_SPATIAL_HEATMAP_WEIGHT="${SUBJECT_BOX_SPATIAL_HEATMAP_WEIGHT:-0.35}"
    SUBJECT_BOX_SPATIAL_MASK_WEIGHT="${SUBJECT_BOX_SPATIAL_MASK_WEIGHT:-0.0}"
    SUBJECT_BOX_SPATIAL_MIX_BIAS="${SUBJECT_BOX_SPATIAL_MIX_BIAS:--0.5}"
    SUBJECT_BOX_SPATIAL_OUTPUT="${SUBJECT_BOX_SPATIAL_OUTPUT:-mix}"
    SUBJECT_BOX_SPATIAL_BOX_MODE="${SUBJECT_BOX_SPATIAL_BOX_MODE:-regress}"
    SUBJECT_BOX_SPATIAL_EXTENT_SCALE="${SUBJECT_BOX_SPATIAL_EXTENT_SCALE:-1.0}"
    SUBJECT_BOX_REFINE_WITH_PROPOSALS="${SUBJECT_BOX_REFINE_WITH_PROPOSALS:-0}"
    GENERATED_POSITIVE_UTILITY_WEIGHT="${GENERATED_POSITIVE_UTILITY_WEIGHT:-0.95}"
    GENERATED_POSITIVE_MARGIN_WEIGHT="${GENERATED_POSITIVE_MARGIN_WEIGHT:-0.55}"
    ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT="${ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT:-0.12}"
    ROUTE_OBJECT_SINGLE_MARGIN="${ROUTE_OBJECT_SINGLE_MARGIN:-0.18}"
    ROUTE_HEAD_DEPTH="${ROUTE_HEAD_DEPTH:-1}"
    ROUTE_HEAD_HIDDEN_MULT="${ROUTE_HEAD_HIDDEN_MULT:-0.5}"
    ROUTE_HEAD_DROPOUT="${ROUTE_HEAD_DROPOUT:-0.0}"
    ROUTE_AUX_HEADS="${ROUTE_AUX_HEADS:-0}"
    ROUTE_DECODE_MODE="${ROUTE_DECODE_MODE:-fine}"
    ROUTE_DECODE_KIND_WEIGHT="${ROUTE_DECODE_KIND_WEIGHT:-1.0}"
    ROUTE_DECODE_CARDINALITY_WEIGHT="${ROUTE_DECODE_CARDINALITY_WEIGHT:-1.0}"
    ROUTE_HIERARCHY_WEIGHT="${ROUTE_HIERARCHY_WEIGHT:-0.0}"
    ROUTE_CARDINALITY_WEIGHT="${ROUTE_CARDINALITY_WEIGHT:-0.0}"
    if [[ -n "${QUALITY_INIT_CHECKPOINT:-}" ]]; then
      INIT_ARGS=(--init_checkpoint "${QUALITY_INIT_CHECKPOINT}")
    fi
    WEIGHT_ARGS=(
      --score_weight 0.40
      --listwise_weight 0.08
      --pairwise_weight 0.0
      --explicit_pairwise_weight 0.0
      --positive_weight 0.12
      --risk_weight 0.08
      --top1_risk_weight 0.05
      --top_return_weight 0.05
      --macro_weight 0.02
      --checklist_class_weight 0.03
      --checklist_applicability_weight 0.02
      --detail_score_weight 0.04
      --why_tag_weight 0.02
      --decision_weight 0.12
      --policy_score_weight 0.12
      --delta_weight 0.04
      --decision_utility_align_weight 0.10
      --action_consistency_weight 0.24
      --proposal_weight 0.66
      --proposal_subject_weight 0.32
      --subject_proposal_align_weight 0.32
      --subject_box_weight 1.15
      --subject_box_valid_weight "${SUBJECT_BOX_VALID_WEIGHT}"
      --subject_box_l1_weight 2.2
      --subject_box_iou_weight 1.3
      --subject_box_center_weight 0.65
      --subject_box_size_weight 0.36
      --subject_box_aspect_weight 0.18
      --subject_box_ciou_weight 0.8
      --subject_box_spatial_aux_weight "${SUBJECT_BOX_SPATIAL_AUX_WEIGHT}"
      --subject_box_spatial_heatmap_weight "${SUBJECT_BOX_SPATIAL_HEATMAP_WEIGHT}"
      --subject_box_spatial_mask_weight "${SUBJECT_BOX_SPATIAL_MASK_WEIGHT}"
      --subject_box_valid_balanced_bce
      --subject_box_valid_negative_scale "${SUBJECT_BOX_VALID_NEGATIVE_SCALE}"
      --generated_proposal_align_weight 0.85
      --generated_proposal_score_weight 0.8
      --generated_proposal_listwise_weight 0.35
      --generated_proposal_positive_weight 0.65
      --generated_proposal_risk_weight 0.28
      --generated_proposal_positive_utility_weight "${GENERATED_POSITIVE_UTILITY_WEIGHT}"
      --generated_proposal_positive_margin_weight "${GENERATED_POSITIVE_MARGIN_WEIGHT}"
      --generated_proposal_positive_margin 0.25
      --generated_proposal_match_iou 0.25
    )
    VARIANT_ARGS=(
      --use_subject_prior
      --route_use_subject_prior
      --route_use_candidate_context
      --route_use_subject_box_features
      --route_use_subject_spatial_token
      --policy_use_subject_prior
      --proposal_use_subject_prior
      --subject_box_head
      --subject_box_spatial_head
      --subject_box_spatial_multiscale
      --subject_box_spatial_mix_bias "${SUBJECT_BOX_SPATIAL_MIX_BIAS}"
      --subject_box_spatial_output "${SUBJECT_BOX_SPATIAL_OUTPUT}"
      --subject_box_spatial_box_mode "${SUBJECT_BOX_SPATIAL_BOX_MODE}"
      --subject_box_spatial_extent_scale "${SUBJECT_BOX_SPATIAL_EXTENT_SCALE}"
      --subject_box_coord_space content
      $(if [[ "${SUBJECT_BOX_REFINE_WITH_PROPOSALS}" == "1" || "${SUBJECT_BOX_REFINE_WITH_PROPOSALS}" == "true" ]]; then printf '%s' "--subject_box_refine_with_proposals"; fi)
      --use_pred_subject_box_as_prior
      --detach_pred_subject_prior
      --pred_subject_prior_start_epoch 1
      --pred_subject_prior_warmup_epochs 3
      --route_balanced_ce
      --route_focal_gamma 1.15
      --train_route_balanced_sampler
      --train_route_balanced_sampler_max_weight "${ROUTE_SAMPLER_MAX_WEIGHT}"
      --route_weight "${ROUTE_WEIGHT}"
      --route_head_depth "${ROUTE_HEAD_DEPTH}"
      --route_head_hidden_mult "${ROUTE_HEAD_HIDDEN_MULT}"
      --route_head_dropout "${ROUTE_HEAD_DROPOUT}"
      $(if [[ "${ROUTE_AUX_HEADS}" == "1" || "${ROUTE_AUX_HEADS}" == "true" ]]; then printf '%s' "--route_aux_heads"; fi)
      --route_decode_mode "${ROUTE_DECODE_MODE}"
      --route_decode_kind_weight "${ROUTE_DECODE_KIND_WEIGHT}"
      --route_decode_cardinality_weight "${ROUTE_DECODE_CARDINALITY_WEIGHT}"
      --route_hierarchy_weight "${ROUTE_HIERARCHY_WEIGHT}"
      --route_cardinality_weight "${ROUTE_CARDINALITY_WEIGHT}"
      --route_object_single_margin_weight "${ROUTE_OBJECT_SINGLE_MARGIN_WEIGHT}"
      --route_object_single_margin "${ROUTE_OBJECT_SINGLE_MARGIN}"
      --policy_score_head
    )
    ;;
  quality_cnv2l512_full_toprepair_lr8e6)
    PROFILE="quality_cnv2l_512"
    EPOCHS="${QUALITY_EPOCHS:-4}"
    BATCH_SIZE="${QUALITY_BATCH_SIZE:-4}"
    LR="${QUALITY_LR:-0.000008}"
    WEIGHT_DECAY="${QUALITY_WEIGHT_DECAY:-0.0001}"
    MAX_TRAIN_ROWS="${QUALITY_MAX_TRAIN_ROWS:-24000}"
    MAX_VAL_ROWS="${QUALITY_MAX_VAL_ROWS:-4200}"
    LIMIT_TRAIN_STEPS="${QUALITY_LIMIT_TRAIN_STEPS:-480}"
    LIMIT_VAL_STEPS="${QUALITY_LIMIT_VAL_STEPS:-170}"
    ROUTE_SAMPLER_MAX_WEIGHT="${QUALITY_ROUTE_SAMPLER_MAX_WEIGHT:-12.0}"
    SELECTION_METRIC="release_gate_joint"
    SCORE_TARGET_MODE="${SCORE_TARGET_MODE:-hybrid_rank}"
    SUBJECT_BOX_SPATIAL_OUTPUT="${SUBJECT_BOX_SPATIAL_OUTPUT:-spatial}"
    SUBJECT_BOX_SPATIAL_BOX_MODE="${SUBJECT_BOX_SPATIAL_BOX_MODE:-mask_moment_regress}"
    EXTRA_TRAIN_ARGS+=(
      --min_lr 1e-6
      --warmup_epochs 1
      --train_route_balanced_sampler_power 0.7
      --route_expert_distill_checkpoints
      artifacts/mobilecropnet_v4/quality_first_20260427/route_experts/swinv2b384_uctr_r1-20260427/best.pt
      artifacts/mobilecropnet_v4/quality_first_20260427/route_experts/cnv2l512_uctr_r2-20260427/best.pt
      artifacts/mobilecropnet_v4/quality_first_20260427/route_experts/eva02l448_m38m_uctr_r1-20260427/best.pt
      --route_expert_distill_weights 0.9,1.25,1.0
      --route_expert_distill_weight 4.4
      --route_expert_distill_temperature 2.4
      --route_expert_distill_hard_weight 0.45
    )
    WEIGHT_ARGS=(
      --score_weight 0.70
      --listwise_weight 0.85
      --pairwise_weight 0.55
      --explicit_pairwise_weight 0.3
      --positive_weight 0.42
      --risk_weight 0.28
      --top1_risk_weight 0.24
      --top_return_weight 0.30
      --top_return_score_margin 0.035
      --top_return_logit_margin 0.36
      --macro_weight 0.14
      --checklist_class_weight 0.12
      --checklist_applicability_weight 0.07
      --detail_score_weight 0.06
      --why_tag_weight 0.07
      --decision_weight 0.32
      --decision_crop_rebalance
      --decision_crop_rebalance_max_weight 5.0
      --action_consistency_weight 0.22
      --decision_utility_align_weight 0.22
      --policy_score_weight 0.20
      --delta_weight 0.1
      --proposal_weight 1.0
      --proposal_subject_weight 0.36
      --subject_proposal_align_weight 0.36
      --subject_box_weight 1.25
      --subject_box_valid_weight 1.35
      --subject_box_l1_weight 2.4
      --subject_box_iou_weight 1.7
      --subject_box_center_weight 0.28
      --subject_box_size_weight 0.24
      --subject_box_aspect_weight 0.12
      --subject_box_ciou_weight 0.35
      --subject_box_spatial_aux_weight 0.40
      --subject_box_spatial_heatmap_weight 0.25
      --subject_box_spatial_mask_weight 0.30
      --subject_box_valid_balanced_bce
      --subject_box_valid_negative_scale 1.45
      --generated_proposal_align_weight 0.90
      --generated_proposal_score_weight 1.0
      --generated_proposal_listwise_weight 0.75
      --generated_proposal_positive_weight 1.10
      --generated_proposal_risk_weight 0.30
      --generated_proposal_positive_utility_weight 0.25
      --generated_proposal_positive_margin_weight 1.35
      --generated_proposal_positive_margin 0.28
      --generated_proposal_match_iou 0.35
    )
    VARIANT_ARGS=(
      --route_use_candidate_context
      --route_use_subject_box_features
      --route_use_subject_spatial_token
      --route_aux_heads
      --route_decode_mode hierarchical
      --route_head_depth 3
      --route_head_hidden_mult 1.0
      --route_head_dropout 0.08
      --route_balanced_ce
      --route_focal_gamma 1.5
      --route_hierarchy_weight 0.65
      --route_cardinality_weight 0.28
      --route_weight 0.95
      --train_route_balanced_sampler
      --train_route_balanced_sampler_max_weight "${ROUTE_SAMPLER_MAX_WEIGHT}"
      --subject_box_head
      --subject_box_spatial_head
      --subject_box_spatial_multiscale
      --subject_box_spatial_output "${SUBJECT_BOX_SPATIAL_OUTPUT}"
      --subject_box_spatial_box_mode "${SUBJECT_BOX_SPATIAL_BOX_MODE}"
      --subject_box_refine_with_proposals
      --use_pred_subject_box_as_prior
      --pred_subject_prior_start_epoch 1
      --pred_subject_prior_warmup_epochs 2
      --policy_score_head
    )
    ;;
  quality_cnv2h512_full_toprepair_lr5e6)
    PROFILE="quality_cnv2h_512"
    EPOCHS="${CNV2H_EPOCHS:-3}"
    BATCH_SIZE="${CNV2H_BATCH_SIZE:-2}"
    LR="${CNV2H_LR:-0.000005}"
    WEIGHT_DECAY="${CNV2H_WEIGHT_DECAY:-0.0001}"
    MAX_TRAIN_ROWS="${CNV2H_MAX_TRAIN_ROWS:-16000}"
    MAX_VAL_ROWS="${CNV2H_MAX_VAL_ROWS:-3600}"
    LIMIT_TRAIN_STEPS="${CNV2H_LIMIT_TRAIN_STEPS:-420}"
    LIMIT_VAL_STEPS="${CNV2H_LIMIT_VAL_STEPS:-150}"
    ROUTE_SAMPLER_MAX_WEIGHT="${CNV2H_ROUTE_SAMPLER_MAX_WEIGHT:-14.0}"
    SELECTION_METRIC="release_gate_joint"
    SCORE_TARGET_MODE="${SCORE_TARGET_MODE:-hybrid_rank}"
    SUBJECT_BOX_SPATIAL_OUTPUT="${SUBJECT_BOX_SPATIAL_OUTPUT:-spatial}"
    SUBJECT_BOX_SPATIAL_BOX_MODE="${SUBJECT_BOX_SPATIAL_BOX_MODE:-mask_moment_regress}"
    EXTRA_TRAIN_ARGS+=(
      --min_lr 8e-7
      --warmup_epochs 1
      --train_route_balanced_sampler_power 0.75
      --route_expert_distill_checkpoints
      artifacts/mobilecropnet_v4/quality_first_20260427/route_experts/swinv2b384_uctr_r1-20260427/best.pt
      --route_expert_distill_weights 1.0
      --route_expert_distill_weight 4.0
      --route_expert_distill_temperature 2.4
      --route_expert_distill_hard_weight 0.55
    )
    WEIGHT_ARGS=(
      --score_weight 0.68
      --listwise_weight 0.82
      --pairwise_weight 0.52
      --explicit_pairwise_weight 0.28
      --positive_weight 0.44
      --risk_weight 0.30
      --top1_risk_weight 0.26
      --top_return_weight 0.34
      --top_return_score_margin 0.04
      --top_return_logit_margin 0.40
      --macro_weight 0.14
      --checklist_class_weight 0.12
      --checklist_applicability_weight 0.07
      --detail_score_weight 0.06
      --why_tag_weight 0.07
      --decision_weight 0.34
      --decision_crop_rebalance
      --decision_crop_rebalance_max_weight 5.0
      --action_consistency_weight 0.24
      --decision_utility_align_weight 0.24
      --policy_score_weight 0.22
      --delta_weight 0.1
      --proposal_weight 1.05
      --proposal_subject_weight 0.40
      --subject_proposal_align_weight 0.40
      --subject_box_weight 1.35
      --subject_box_valid_weight 1.55
      --subject_box_l1_weight 2.6
      --subject_box_iou_weight 1.9
      --subject_box_center_weight 0.32
      --subject_box_size_weight 0.28
      --subject_box_aspect_weight 0.14
      --subject_box_ciou_weight 0.42
      --subject_box_spatial_aux_weight 0.46
      --subject_box_spatial_heatmap_weight 0.28
      --subject_box_spatial_mask_weight 0.34
      --subject_box_valid_balanced_bce
      --subject_box_valid_negative_scale 1.65
      --generated_proposal_align_weight 0.95
      --generated_proposal_score_weight 1.05
      --generated_proposal_listwise_weight 0.80
      --generated_proposal_positive_weight 1.20
      --generated_proposal_risk_weight 0.34
      --generated_proposal_positive_utility_weight 0.30
      --generated_proposal_positive_margin_weight 1.50
      --generated_proposal_positive_margin 0.30
      --generated_proposal_match_iou 0.35
    )
    VARIANT_ARGS=(
      --route_use_candidate_context
      --route_use_subject_box_features
      --route_use_subject_spatial_token
      --route_aux_heads
      --route_decode_mode hierarchical
      --route_head_depth 3
      --route_head_hidden_mult 1.0
      --route_head_dropout 0.08
      --route_balanced_ce
      --route_focal_gamma 1.6
      --route_hierarchy_weight 0.70
      --route_cardinality_weight 0.30
      --route_weight 1.05
      --train_route_balanced_sampler
      --train_route_balanced_sampler_max_weight "${ROUTE_SAMPLER_MAX_WEIGHT}"
      --subject_box_head
      --subject_box_spatial_head
      --subject_box_spatial_multiscale
      --subject_box_spatial_output "${SUBJECT_BOX_SPATIAL_OUTPUT}"
      --subject_box_spatial_box_mode "${SUBJECT_BOX_SPATIAL_BOX_MODE}"
      --subject_box_refine_with_proposals
      --use_pred_subject_box_as_prior
      --pred_subject_prior_start_epoch 1
      --pred_subject_prior_warmup_epochs 2
      --policy_score_head
    )
    ;;
  quality_dinov2b518_subject_only_lr5e6)
    PROFILE="quality_dinov2b_subject_518"
    EPOCHS="${EPOCHS:-5}"
    BATCH_SIZE="${SUBJECT_BATCH_SIZE:-3}"
    LR="${LR:-0.000005}"
    MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-22000}"
    MAX_VAL_ROWS="${MAX_VAL_ROWS:-4200}"
    LIMIT_TRAIN_STEPS="${LIMIT_TRAIN_STEPS:-800}"
    LIMIT_VAL_STEPS="${LIMIT_VAL_STEPS:-180}"
    SELECTION_METRIC="subject_box"
    SUBJECT_BOX_VALID_WEIGHT="${SUBJECT_BOX_VALID_WEIGHT:-1.8}"
    SUBJECT_BOX_VALID_NEGATIVE_SCALE="${SUBJECT_BOX_VALID_NEGATIVE_SCALE:-1.4}"
    SUBJECT_BOX_SPATIAL_AUX_WEIGHT="${SUBJECT_BOX_SPATIAL_AUX_WEIGHT:-1.2}"
    SUBJECT_BOX_SPATIAL_HEATMAP_WEIGHT="${SUBJECT_BOX_SPATIAL_HEATMAP_WEIGHT:-0.75}"
    SUBJECT_BOX_SPATIAL_MASK_WEIGHT="${SUBJECT_BOX_SPATIAL_MASK_WEIGHT:-0.25}"
    SUBJECT_BOX_SPATIAL_MIX_BIAS="${SUBJECT_BOX_SPATIAL_MIX_BIAS:-0.0}"
    WEIGHT_ARGS=(
      --score_weight 0.0
      --listwise_weight 0.0
      --pairwise_weight 0.0
      --explicit_pairwise_weight 0.0
      --positive_weight 0.0
      --risk_weight 0.0
      --top1_risk_weight 0.0
      --top_return_weight 0.0
      --macro_weight 0.0
      --checklist_class_weight 0.0
      --checklist_applicability_weight 0.0
      --detail_score_weight 0.0
      --why_tag_weight 0.0
      --route_weight 0.0
      --decision_weight 0.0
      --policy_score_weight 0.0
      --delta_weight 0.0
      --proposal_weight "${SUBJECT_ONLY_PROPOSAL_WEIGHT:-0.0}"
      --proposal_subject_weight "${SUBJECT_ONLY_PROPOSAL_SUBJECT_WEIGHT:-0.0}"
      --subject_proposal_align_weight "${SUBJECT_ONLY_SUBJECT_PROPOSAL_ALIGN_WEIGHT:-0.0}"
      --subject_box_weight 1.0
      --subject_box_valid_weight "${SUBJECT_BOX_VALID_WEIGHT}"
      --subject_box_l1_weight 2.8
      --subject_box_iou_weight 1.8
      --subject_box_center_weight 0.8
      --subject_box_size_weight 0.45
      --subject_box_aspect_weight 0.22
      --subject_box_ciou_weight 1.0
      --subject_box_spatial_aux_weight "${SUBJECT_BOX_SPATIAL_AUX_WEIGHT}"
      --subject_box_spatial_heatmap_weight "${SUBJECT_BOX_SPATIAL_HEATMAP_WEIGHT}"
      --subject_box_spatial_mask_weight "${SUBJECT_BOX_SPATIAL_MASK_WEIGHT}"
      --subject_box_valid_balanced_bce
      --subject_box_valid_negative_scale "${SUBJECT_BOX_VALID_NEGATIVE_SCALE}"
      --generated_proposal_align_weight 0.0
    )
    VARIANT_ARGS=(
      --subject_box_head
      --subject_box_spatial_head
      --subject_box_spatial_multiscale
      --subject_box_spatial_mix_bias "${SUBJECT_BOX_SPATIAL_MIX_BIAS}"
      --subject_box_coord_space content
      $(if [[ "${SUBJECT_BOX_REFINE_WITH_PROPOSALS:-0}" == "1" || "${SUBJECT_BOX_REFINE_WITH_PROPOSALS:-0}" == "true" ]]; then printf '%s' "--subject_box_refine_with_proposals"; fi)
    )
    ;;
  quality_dinov2l518_subject_only_lr3e6)
    PROFILE="quality_dinov2l_subject_518"
    EPOCHS="${EPOCHS:-4}"
    BATCH_SIZE="${SUBJECT_BATCH_SIZE:-1}"
    LR="${LR:-0.000003}"
    MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-18000}"
    MAX_VAL_ROWS="${MAX_VAL_ROWS:-3600}"
    LIMIT_TRAIN_STEPS="${LIMIT_TRAIN_STEPS:-650}"
    LIMIT_VAL_STEPS="${LIMIT_VAL_STEPS:-150}"
    SELECTION_METRIC="subject_box"
    SUBJECT_BOX_VALID_WEIGHT="${SUBJECT_BOX_VALID_WEIGHT:-1.8}"
    SUBJECT_BOX_VALID_NEGATIVE_SCALE="${SUBJECT_BOX_VALID_NEGATIVE_SCALE:-1.4}"
    SUBJECT_BOX_SPATIAL_AUX_WEIGHT="${SUBJECT_BOX_SPATIAL_AUX_WEIGHT:-1.2}"
    SUBJECT_BOX_SPATIAL_HEATMAP_WEIGHT="${SUBJECT_BOX_SPATIAL_HEATMAP_WEIGHT:-0.75}"
    SUBJECT_BOX_SPATIAL_MASK_WEIGHT="${SUBJECT_BOX_SPATIAL_MASK_WEIGHT:-0.25}"
    SUBJECT_BOX_SPATIAL_MIX_BIAS="${SUBJECT_BOX_SPATIAL_MIX_BIAS:-0.0}"
    WEIGHT_ARGS=(
      --score_weight 0.0
      --listwise_weight 0.0
      --pairwise_weight 0.0
      --explicit_pairwise_weight 0.0
      --positive_weight 0.0
      --risk_weight 0.0
      --top1_risk_weight 0.0
      --top_return_weight 0.0
      --macro_weight 0.0
      --checklist_class_weight 0.0
      --checklist_applicability_weight 0.0
      --detail_score_weight 0.0
      --why_tag_weight 0.0
      --route_weight 0.0
      --decision_weight 0.0
      --policy_score_weight 0.0
      --delta_weight 0.0
      --proposal_weight "${SUBJECT_ONLY_PROPOSAL_WEIGHT:-0.0}"
      --proposal_subject_weight "${SUBJECT_ONLY_PROPOSAL_SUBJECT_WEIGHT:-0.0}"
      --subject_proposal_align_weight "${SUBJECT_ONLY_SUBJECT_PROPOSAL_ALIGN_WEIGHT:-0.0}"
      --subject_box_weight 1.0
      --subject_box_valid_weight "${SUBJECT_BOX_VALID_WEIGHT}"
      --subject_box_l1_weight 2.8
      --subject_box_iou_weight 1.8
      --subject_box_center_weight 0.8
      --subject_box_size_weight 0.45
      --subject_box_aspect_weight 0.22
      --subject_box_ciou_weight 1.0
      --subject_box_spatial_aux_weight "${SUBJECT_BOX_SPATIAL_AUX_WEIGHT}"
      --subject_box_spatial_heatmap_weight "${SUBJECT_BOX_SPATIAL_HEATMAP_WEIGHT}"
      --subject_box_spatial_mask_weight "${SUBJECT_BOX_SPATIAL_MASK_WEIGHT}"
      --subject_box_valid_balanced_bce
      --subject_box_valid_negative_scale "${SUBJECT_BOX_VALID_NEGATIVE_SCALE}"
      --generated_proposal_align_weight 0.0
    )
    VARIANT_ARGS=(
      --subject_box_head
      --subject_box_spatial_head
      --subject_box_spatial_multiscale
      --subject_box_spatial_mix_bias "${SUBJECT_BOX_SPATIAL_MIX_BIAS}"
      --subject_box_coord_space content
      $(if [[ "${SUBJECT_BOX_REFINE_WITH_PROPOSALS:-0}" == "1" || "${SUBJECT_BOX_REFINE_WITH_PROPOSALS:-0}" == "true" ]]; then printf '%s' "--subject_box_refine_with_proposals"; fi)
    )
    ;;
  *)
    echo "unsupported route smoke variant: ${VARIANT}" >&2
    exit 2
    ;;
esac

SCORE_TARGET_MODE="${SCORE_TARGET_MODE:-default}"

write_status environment running
{
  date -u +"start_utc=%Y-%m-%dT%H:%M:%SZ"
  nvidia-smi --query-gpu=name,memory.total,driver_version,utilization.gpu,memory.used --format=csv,noheader
  "${PYTHON_BIN}" - <<'PY'
import torch
print("torch_version", torch.__version__)
print("cuda_available", torch.cuda.is_available())
assert torch.cuda.is_available()
print("device_name", torch.cuda.get_device_name(0))
PY
  "${PYTHON_BIN}" -m py_compile src/mobilecropnet_v4/model.py src/scripts/train_mobilecropnet_v4.py
} | tee "${RUN_DIR}/environment.log"

write_status train running
"${PYTHON_BIN}" src/scripts/train_mobilecropnet_v4.py \
  --train_jsonl "${TRAIN_JSONL}" \
  --val_jsonl "${VAL_JSONL}" \
  --project_root . \
  --output_dir "${RUN_DIR}" \
  --model_profile "${PROFILE}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --num_workers "${NUM_WORKERS}" \
  --image_tensor_cache_size 128 \
  --selection_metric "${SELECTION_METRIC}" \
  --score_target_mode "${SCORE_TARGET_MODE}" \
  --subject_box_target_source "${SUBJECT_BOX_TARGET_SOURCE}" \
  "${WEIGHT_ARGS[@]}" \
  "${INIT_ARGS[@]}" \
  "${EXTRA_TRAIN_ARGS[@]}" \
  --max_train_rows "${MAX_TRAIN_ROWS}" \
  --max_val_rows "${MAX_VAL_ROWS}" \
  --limit_train_steps "${LIMIT_TRAIN_STEPS}" \
  --limit_val_steps "${LIMIT_VAL_STEPS}" \
  --amp \
  --device cuda \
  --gpu_usage_sample_interval 20 \
  --progress_log_interval 50 \
  --seed "${SEED}" \
  "${VARIANT_ARGS[@]}" \
  2>&1 | tee "${RUN_DIR}/train.log"

write_status summarize running
"${PYTHON_BIN}" - "${RUN_DIR}" <<'PY'
import json
import sys
from pathlib import Path

run = Path(sys.argv[1])
metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8")) if (run / "metrics.json").exists() else []
best = None
for row in metrics:
    val = row.get("val") if isinstance(row.get("val"), dict) else {}
    score = row.get("selection_score")
    if score is None:
        score = val.get("route_acc")
    if score is None:
        continue
    best_score = best.get("selection_score") if isinstance(best, dict) else None
    if best_score is None and isinstance(best, dict):
        best_score = (best.get("val") or {}).get("route_acc", -1)
    if best is None or float(score) > float(best_score):
        best = row
summary = json.loads((run / "summary.json").read_text(encoding="utf-8")) if (run / "summary.json").exists() else {}
payload = {
    "run_name": run.name,
    "run_dir": str(run),
    "state": summary.get("state"),
    "best_epoch": (best or {}).get("epoch"),
    "best_val": (best or {}).get("val"),
    "summary": summary,
}
(run / "route_smoke_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY

date -u +"end_utc=%Y-%m-%dT%H:%M:%SZ" | tee -a "${RUN_DIR}/environment.log"
write_status completed completed
