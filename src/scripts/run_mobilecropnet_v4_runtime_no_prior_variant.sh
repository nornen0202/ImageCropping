#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:?variant required}"
PROFILE="${2:?profile required}"
RUN_NAME="${3:?run name required}"

ROOT="${ROOT:-$(pwd)}"
cd "${ROOT}"

declare -a VARIANT_ARGS
case "${VARIANT}" in
  baseline_current|deploy_no_prior_executor_v1)
    VARIANT_ARGS=()
    ;;
  calibration_v1|deploy_no_prior_executor_calibration_v1)
    VARIANT_ARGS=(
      --decision_weight 0.10
      --policy_score_weight 0.25
      --decision_utility_align_weight 0.12
      --decision_utility_align_temperature 0.35
    )
    ;;
  calibration_v2|deploy_no_prior_executor_calibration_v2)
    VARIANT_ARGS=(
      --decision_weight 0.10
      --policy_score_weight 0.25
      --decision_utility_align_weight 0.12
      --decision_utility_align_temperature 0.35
      --decision_crop_rebalance
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
echo "profile=${PROFILE}"
echo "run_name=${RUN_NAME}"
echo "extra_train_args=${EXTRA_TRAIN_ARGS_COMBINED}"

EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS_COMBINED}" \
bash src/scripts/run_mobilecropnet_v4_runtime_no_prior_experiment.sh \
  "${PROFILE}" \
  "${RUN_NAME}"
