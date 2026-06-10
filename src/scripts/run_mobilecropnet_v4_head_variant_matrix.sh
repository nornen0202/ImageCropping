#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:?variant required}"
TRACK_NAME="${2:?track name required}"
OUT_BASE="${3:?output base required}"
TRAIN_LABEL_DIR="${4:?train label dir required}"
VAL_LABEL_DIR="${5:?val label dir required}"
TEST_LABEL_DIR="${6:?test label dir required}"

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

echo "variant=${VARIANT}"
echo "track_name=${TRACK_NAME}"
echo "out_base=${OUT_BASE}"
echo "extra_train_args=${EXTRA_TRAIN_ARGS_COMBINED}"

EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS_COMBINED}" \
bash src/scripts/run_mobilecropnet_v4_product_ar_score_track_matrix.sh \
  "${TRACK_NAME}" \
  "${OUT_BASE}" \
  "${TRAIN_LABEL_DIR}" \
  "${VAL_LABEL_DIR}" \
  "${TEST_LABEL_DIR}"
