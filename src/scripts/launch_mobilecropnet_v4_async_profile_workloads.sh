#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:?variant required}"
TRACK_NAME="${2:?track name required}"
OUT_BASE="${3:?output base required}"
TRAIN_LABEL_DIR="${4:?train label dir required}"
VAL_LABEL_DIR="${5:?val label dir required}"
TEST_LABEL_DIR="${6:?test label dir required}"

ROOT="${ROOT:-/media/jyju25/T7_4TB_JY/Projects_26/Sources/ImageCropping}"
REMOTE_ROOT="${REMOTE_ROOT:-/group-volume/users/jaden.ju/Sources/ImageCropping}"
GROUP_ID="${GROUP_ID:-SR-AR-Interactive-Media-Exp}"
EXPERIMENT_ID="${EXPERIMENT_ID:-918}"
IMAGE_NAME="${IMAGE_NAME:-sr-ar-interactive-media-exp/jy-cropping-260415-tfs4.57.6-rsync}"
CORE_ID="${CORE_ID:-2}"
PROFILES_CSV="${PROFILES_CSV:-q24_288_w075,turbo_256,balanced_288,hq_320,rank_320,hybrid_384,plus_384}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d-%H%M%S)}"
EXPECTED_RUNTIME_DEFAULT="${EXPECTED_RUNTIME_DEFAULT:-12h}"
EXPECTED_RUNTIME_Q24_288_W075="${EXPECTED_RUNTIME_Q24_288_W075:-${EXPECTED_RUNTIME_DEFAULT}}"
EXPECTED_RUNTIME_TURBO_256="${EXPECTED_RUNTIME_TURBO_256:-${EXPECTED_RUNTIME_DEFAULT}}"
EXPECTED_RUNTIME_BALANCED_288="${EXPECTED_RUNTIME_BALANCED_288:-${EXPECTED_RUNTIME_DEFAULT}}"
EXPECTED_RUNTIME_HQ_320="${EXPECTED_RUNTIME_HQ_320:-${EXPECTED_RUNTIME_DEFAULT}}"
EXPECTED_RUNTIME_RANK_320="${EXPECTED_RUNTIME_RANK_320:-${EXPECTED_RUNTIME_DEFAULT}}"
EXPECTED_RUNTIME_HYBRID_384="${EXPECTED_RUNTIME_HYBRID_384:-${EXPECTED_RUNTIME_DEFAULT}}"
EXPECTED_RUNTIME_PLUS_384="${EXPECTED_RUNTIME_PLUS_384:-${EXPECTED_RUNTIME_DEFAULT}}"
MANIFEST_PATH="${MANIFEST_PATH:-${OUT_BASE}/async_launch_manifest_${RUN_STAMP}.tsv}"
ALLOW_SPOT="${ALLOW_SPOT:-1}"

mkdir -p "${OUT_BASE}"

IFS=',' read -r -a PROFILES <<< "${PROFILES_CSV}"

runtime_for_profile() {
  local profile="$1"
  case "${profile}" in
    q24_288_w075) echo "${EXPECTED_RUNTIME_Q24_288_W075}" ;;
    turbo_256) echo "${EXPECTED_RUNTIME_TURBO_256}" ;;
    balanced_288) echo "${EXPECTED_RUNTIME_BALANCED_288}" ;;
    hq_320) echo "${EXPECTED_RUNTIME_HQ_320}" ;;
    rank_320) echo "${EXPECTED_RUNTIME_RANK_320}" ;;
    hybrid_384) echo "${EXPECTED_RUNTIME_HYBRID_384}" ;;
    plus_384) echo "${EXPECTED_RUNTIME_PLUS_384}" ;;
    *) echo "${EXPECTED_RUNTIME_DEFAULT}" ;;
  esac
}

printf "run_name\tprofile\tvariant\ttrack_name\texpected_runtime\tout_base\ttrain_label_dir\tval_label_dir\ttest_label_dir\trun_id\n" > "${MANIFEST_PATH}"

echo "variant=${VARIANT}"
echo "track_name=${TRACK_NAME}"
echo "out_base=${OUT_BASE}"
echo "profiles=${PROFILES_CSV}"
echo "manifest_path=${MANIFEST_PATH}"

for PROFILE in "${PROFILES[@]}"; do
  PROFILE_SAFE="${PROFILE//_/-}"
  RUN_NAME="mcn-${TRACK_NAME}-${PROFILE_SAFE}-${RUN_STAMP}"
  EXPECTED_RUNTIME="$(runtime_for_profile "${PROFILE}")"
  EXEC_COMMAND="cd ${REMOTE_ROOT} && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader && TRAIN_LABEL_DIR=${TRAIN_LABEL_DIR} VAL_LABEL_DIR=${VAL_LABEL_DIR} TEST_LABEL_DIR=${TEST_LABEL_DIR} OUT_BASE=${OUT_BASE} EPOCHS=${EPOCHS:-8} BATCH_SIZE=${BATCH_SIZE:-16} LR=${LR:-0.0003} WEIGHT_DECAY=${WEIGHT_DECAY:-0.0001} SEED=${SEED:-20260420} SELECTION_METRIC=${SELECTION_METRIC:-sstk_balanced_topreturn} TOP_RETURN_WEIGHT=${TOP_RETURN_WEIGHT:-0.6} LISTWISE_WEIGHT=${LISTWISE_WEIGHT:-0.7} PAIRWISE_WEIGHT=${PAIRWISE_WEIGHT:-0.5} EXPLICIT_PAIRWISE_WEIGHT=${EXPLICIT_PAIRWISE_WEIGHT:-0.25} TOP1_RISK_WEIGHT=${TOP1_RISK_WEIGHT:-0.15} TEACHER_DISTILL_WEIGHT=${TEACHER_DISTILL_WEIGHT:-0.20} TOPK_COVERAGE_WEIGHT=${TOPK_COVERAGE_WEIGHT:-0.10} PROPOSAL_WEIGHT=${PROPOSAL_WEIGHT:-0.7} MAX_PAIRWISE_PAIRS=${MAX_PAIRWISE_PAIRS:-72} NUM_WORKERS=${NUM_WORKERS:-2} PERSISTENT_WORKERS=${PERSISTENT_WORKERS:-0} PREFETCH_FACTOR=${PREFETCH_FACTOR:-2} PRECOMPUTE_SAMPLE_TENSORS=${PRECOMPUTE_SAMPLE_TENSORS:-0} IMAGE_TENSOR_CACHE_SIZE=${IMAGE_TENSOR_CACHE_SIZE:-128} AMP=${AMP:-1} EXTRA_TRAIN_ARGS='${EXTRA_TRAIN_ARGS:-}' bash src/scripts/run_mobilecropnet_v4_single_profile_variant.sh ${VARIANT} ${PROFILE} ${RUN_NAME}"
  echo "[launch] ${RUN_NAME} expected_runtime=${EXPECTED_RUNTIME}"
  TMP_OUT="$(mktemp)"
  set +e
  if [[ "${ALLOW_SPOT}" == "1" ]]; then
    space --region n6 mlp create run "${RUN_NAME}" \
      -c "${CORE_ID}" \
      --core-count=1 \
      --experiment-id="${EXPERIMENT_ID}" \
      --image="${IMAGE_NAME}" \
      --exec-command="${EXEC_COMMAND}" \
      --group-id="${GROUP_ID}" \
      --allow-spot \
      --expected-run-time="${EXPECTED_RUNTIME}" | tee "${TMP_OUT}"
  else
    space --region n6 mlp create run "${RUN_NAME}" \
      -c "${CORE_ID}" \
      --core-count=1 \
      --experiment-id="${EXPERIMENT_ID}" \
      --image="${IMAGE_NAME}" \
      --exec-command="${EXEC_COMMAND}" \
      --group-id="${GROUP_ID}" \
      --group-pool-id="${GROUP_POOL_ID:?GROUP_POOL_ID required when ALLOW_SPOT=0}" \
      --quota-id="${QUOTA_ID:?QUOTA_ID required when ALLOW_SPOT=0}" \
      --expected-run-time="${EXPECTED_RUNTIME}" | tee "${TMP_OUT}"
  fi
  STATUS=$?
  set -e
  RUN_ID="$(awk -F'|' '/Run ID/{gsub(/ /, "", $3); print $3}' "${TMP_OUT}" | tail -n 1)"
  rm -f "${TMP_OUT}"
  if [[ "${STATUS}" != "0" ]]; then
    echo "[launch] failed ${RUN_NAME}" >&2
    exit "${STATUS}"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${RUN_NAME}" "${PROFILE}" "${VARIANT}" "${TRACK_NAME}" "${EXPECTED_RUNTIME}" "${OUT_BASE}" "${TRAIN_LABEL_DIR}" "${VAL_LABEL_DIR}" "${TEST_LABEL_DIR}" "${RUN_ID}" \
    >> "${MANIFEST_PATH}"
done

echo "[launch] manifest written to ${MANIFEST_PATH}"
