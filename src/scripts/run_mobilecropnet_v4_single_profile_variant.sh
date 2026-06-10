#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:?variant required}"
PROFILE="${2:?profile required}"
RUN_NAME="${3:?run name required}"

ROOT="${ROOT:-/group-volume/users/jaden.ju/Sources/ImageCropping}"
cd "${ROOT}"

declare -a VARIANT_ARGS
case "${VARIANT}" in
  baseline_current)
    VARIANT_ARGS=()
    ;;
  subjectprior_route_proposal_v1)
    VARIANT_ARGS=(
      --use_subject_prior
      --route_use_subject_prior
      --proposal_use_subject_prior
      --route_balanced_ce
      --proposal_subject_weight 0.25
    )
    ;;
  subjectprior_route_recover_v1)
    VARIANT_ARGS=(
      --use_subject_prior
      --route_use_subject_prior
      --route_use_candidate_context
      --route_use_subject_box_features
      --proposal_use_subject_prior
      --route_balanced_ce
      --route_weight 0.14
      --route_object_single_margin_weight 0.08
      --route_object_single_margin 0.18
      --proposal_subject_weight 0.25
    )
    ;;
  subjectprior_route_recover_v2)
    VARIANT_ARGS=(
      --use_subject_prior
      --route_use_subject_prior
      --route_use_candidate_context
      --proposal_use_subject_prior
      --route_balanced_ce
      --train_route_balanced_sampler
      --route_weight 0.16
      --route_object_single_margin_weight 0.08
      --route_object_single_margin 0.18
      --proposal_subject_weight 0.25
    )
    ;;
  subject_box_prior_v1)
    VARIANT_ARGS=(
      --use_subject_prior
      --route_use_subject_prior
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
      --proposal_subject_weight 0.15
      --subject_proposal_align_weight 0.12
      --subject_box_weight 0.35
      --subject_box_valid_weight 1.0
      --subject_box_l1_weight 2.0
      --subject_box_iou_weight 1.0
      --subject_box_center_weight 0.5
      --subject_box_size_weight 0.25
      --subject_box_aspect_weight 0.15
      --subject_box_ciou_weight 0.5
      --policy_score_head
      --policy_score_weight 0.15
      --decision_weight 0.10
      --detail_score_weight 0.08
      --checklist_class_weight 0.04
      --why_tag_weight 0.02
    )
    ;;
  subject_box_prior_v2_balanced_valid)
    VARIANT_ARGS=(
      --use_subject_prior
      --route_use_subject_prior
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
      --proposal_subject_weight 0.15
      --subject_proposal_align_weight 0.12
      --subject_box_weight 0.35
      --subject_box_valid_weight 1.5
      --subject_box_l1_weight 2.0
      --subject_box_iou_weight 1.0
      --subject_box_center_weight 0.5
      --subject_box_size_weight 0.25
      --subject_box_aspect_weight 0.15
      --subject_box_ciou_weight 0.5
      --subject_box_valid_balanced_bce
      --policy_score_head
      --policy_score_weight 0.15
      --decision_weight 0.10
      --detail_score_weight 0.08
      --checklist_class_weight 0.04
      --why_tag_weight 0.02
    )
    ;;
  policy_score_v1)
    VARIANT_ARGS=(
      --policy_score_head
      --policy_score_weight 0.25
      --decision_weight 0.10
      --detail_score_weight 0.08
      --checklist_class_weight 0.04
    )
    ;;
  hybrid_subject_policy_v1)
    VARIANT_ARGS=(
      --use_subject_prior
      --route_use_subject_prior
      --policy_use_subject_prior
      --proposal_use_subject_prior
      --route_balanced_ce
      --proposal_subject_weight 0.25
      --policy_score_head
      --policy_score_weight 0.25
      --decision_weight 0.08
      --detail_score_weight 0.08
      --checklist_class_weight 0.04
      --why_tag_weight 0.02
    )
    ;;
  deploy_no_prior_executor_v1)
    VARIANT_ARGS=(
      --selection_metric deploy_align_topreturn
      --route_balanced_ce
      --policy_score_head
      --policy_score_weight 0.15
      --decision_weight 0.18
      --action_consistency_weight 0.25
      --action_consistency_margin 0.15
      --generated_proposal_align_weight 0.15
      --generated_proposal_score_weight 1.0
      --generated_proposal_listwise_weight 0.5
      --generated_proposal_positive_weight 0.25
      --generated_proposal_risk_weight 0.15
      --detail_score_weight 0.08
      --checklist_class_weight 0.04
      --why_tag_weight 0.02
    )
    ;;
  release_gate_v1)
    VARIANT_ARGS=(
      --selection_metric deploy_align_topreturn
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
      --pred_subject_prior_warmup_epochs 2
      --route_balanced_ce
      --train_route_balanced_sampler
      --route_weight 0.18
      --route_object_single_margin_weight 0.10
      --route_object_single_margin 0.18
      --proposal_subject_weight 0.20
      --subject_proposal_align_weight 0.15
      --subject_box_weight 0.45
      --subject_box_valid_weight 1.5
      --subject_box_l1_weight 2.0
      --subject_box_iou_weight 1.0
      --subject_box_center_weight 0.5
      --subject_box_size_weight 0.25
      --subject_box_aspect_weight 0.15
      --subject_box_ciou_weight 0.5
      --subject_box_valid_balanced_bce
      --policy_score_head
      --policy_score_weight 0.25
      --decision_weight 0.10
      --decision_utility_align_weight 0.12
      --decision_utility_align_temperature 0.35
      --action_consistency_weight 0.25
      --action_consistency_margin 0.15
      --generated_proposal_align_weight 0.15
      --generated_proposal_score_weight 1.0
      --generated_proposal_listwise_weight 0.5
      --generated_proposal_positive_weight 0.25
      --generated_proposal_risk_weight 0.15
      --detail_score_weight 0.08
      --checklist_class_weight 0.04
      --why_tag_weight 0.02
    )
    ;;
  release_gate_v2)
    VARIANT_ARGS=(
      --selection_metric release_gate
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
      --train_route_balanced_sampler
      --train_route_balanced_sampler_max_weight 8.0
      --route_weight 0.32
      --route_object_single_margin_weight 0.14
      --route_object_single_margin 0.18
      --proposal_subject_weight 0.18
      --subject_proposal_align_weight 0.12
      --subject_box_weight 0.40
      --subject_box_valid_weight 2.0
      --subject_box_valid_negative_scale 2.5
      --subject_box_l1_weight 2.0
      --subject_box_iou_weight 1.0
      --subject_box_center_weight 0.5
      --subject_box_size_weight 0.25
      --subject_box_aspect_weight 0.15
      --subject_box_ciou_weight 0.5
      --subject_box_valid_balanced_bce
      --policy_score_head
      --policy_score_weight 0.20
      --decision_weight 0.12
      --decision_utility_align_weight 0.10
      --decision_utility_align_temperature 0.35
      --action_consistency_weight 0.22
      --action_consistency_margin 0.15
      --generated_proposal_align_weight 0.10
      --generated_proposal_score_weight 1.0
      --generated_proposal_listwise_weight 0.5
      --generated_proposal_positive_weight 0.25
      --generated_proposal_risk_weight 0.15
      --detail_score_weight 0.08
      --checklist_class_weight 0.04
      --why_tag_weight 0.02
    )
    ;;
  release_gate_v2_nopriorroute)
    VARIANT_ARGS=(
      --selection_metric release_gate
      --use_subject_prior
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
      --train_route_balanced_sampler
      --train_route_balanced_sampler_max_weight 8.0
      --route_weight 0.36
      --route_object_single_margin_weight 0.16
      --route_object_single_margin 0.18
      --proposal_subject_weight 0.18
      --subject_proposal_align_weight 0.12
      --subject_box_weight 0.40
      --subject_box_valid_weight 2.0
      --subject_box_valid_negative_scale 2.5
      --subject_box_l1_weight 2.0
      --subject_box_iou_weight 1.0
      --subject_box_center_weight 0.5
      --subject_box_size_weight 0.25
      --subject_box_aspect_weight 0.15
      --subject_box_ciou_weight 0.5
      --subject_box_valid_balanced_bce
      --policy_score_head
      --policy_score_weight 0.20
      --decision_weight 0.12
      --decision_utility_align_weight 0.10
      --decision_utility_align_temperature 0.35
      --action_consistency_weight 0.22
      --action_consistency_margin 0.15
      --generated_proposal_align_weight 0.10
      --generated_proposal_score_weight 1.0
      --generated_proposal_listwise_weight 0.5
      --generated_proposal_positive_weight 0.25
      --generated_proposal_risk_weight 0.15
      --detail_score_weight 0.08
      --checklist_class_weight 0.04
      --why_tag_weight 0.02
    )
    ;;
  release_gate_v2_validcal)
    VARIANT_ARGS=(
      --selection_metric release_gate
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
      --pred_subject_prior_start_epoch 3
      --pred_subject_prior_warmup_epochs 2
      --route_balanced_ce
      --train_route_balanced_sampler
      --train_route_balanced_sampler_max_weight 8.0
      --route_weight 0.30
      --route_object_single_margin_weight 0.14
      --route_object_single_margin 0.18
      --proposal_subject_weight 0.16
      --subject_proposal_align_weight 0.12
      --subject_box_weight 0.35
      --subject_box_valid_weight 2.5
      --subject_box_valid_negative_scale 4.0
      --subject_box_l1_weight 2.0
      --subject_box_iou_weight 1.0
      --subject_box_center_weight 0.5
      --subject_box_size_weight 0.25
      --subject_box_aspect_weight 0.15
      --subject_box_ciou_weight 0.5
      --subject_box_valid_balanced_bce
      --policy_score_head
      --policy_score_weight 0.20
      --decision_weight 0.12
      --decision_utility_align_weight 0.10
      --decision_utility_align_temperature 0.35
      --action_consistency_weight 0.22
      --action_consistency_margin 0.15
      --generated_proposal_align_weight 0.08
      --generated_proposal_score_weight 1.0
      --generated_proposal_listwise_weight 0.5
      --generated_proposal_positive_weight 0.25
      --generated_proposal_risk_weight 0.15
      --detail_score_weight 0.08
      --checklist_class_weight 0.04
      --why_tag_weight 0.02
    )
    ;;
  *)
    echo "unsupported variant: ${VARIANT}" >&2
    exit 2
    ;;
esac

EXTRA_TRAIN_ARGS_COMBINED="${EXTRA_TRAIN_ARGS:-}"
if [[ "${#VARIANT_ARGS[@]}" -gt 0 ]]; then
  if [[ -n "${EXTRA_TRAIN_ARGS_COMBINED}" ]]; then
    EXTRA_TRAIN_ARGS_COMBINED+=" "
  fi
  EXTRA_TRAIN_ARGS_COMBINED+="${VARIANT_ARGS[*]}"
fi
if [[ -n "${EXTRA_TRAIN_ARGS_POST:-}" ]]; then
  if [[ -n "${EXTRA_TRAIN_ARGS_COMBINED}" ]]; then
    EXTRA_TRAIN_ARGS_COMBINED+=" "
  fi
  EXTRA_TRAIN_ARGS_COMBINED+="${EXTRA_TRAIN_ARGS_POST}"
fi

echo "variant=${VARIANT}"
echo "profile=${PROFILE}"
echo "run_name=${RUN_NAME}"
echo "extra_train_args=${EXTRA_TRAIN_ARGS_COMBINED}"

EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS_COMBINED}" \
bash src/scripts/run_mobilecropnet_v4_sstk_product_experiment.sh \
  "${PROFILE}" \
  "${RUN_NAME}"
