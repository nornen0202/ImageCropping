#!/usr/bin/env bash
set -euo pipefail

METHOD="${1:?method required: t1 | uctr_stage3 | public_ensemble_best}"
DATASET="${2:?dataset required: gaic_v2_official | sstk_full_10000}"

ROOT="${ROOT:-$(pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python}"

cd "${ROOT}"

DEVICE="${DEVICE:-auto}"
if [[ "${DEVICE}" == "auto" ]]; then
  DEVICE="$("${PYTHON_BIN}" - <<'PY'
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
PY
)"
fi

MAX_ROWS="${MAX_ROWS:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_CANDIDATES="${MAX_CANDIDATES:-128}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
CROP_SIZE="${CROP_SIZE:-160}"
HIGH_SCORE_THRESHOLD="${HIGH_SCORE_THRESHOLD:-0.85}"
LOW_SCORE_THRESHOLD="${LOW_SCORE_THRESHOLD:-0.20}"
CONTRADICTION_THRESHOLD="${CONTRADICTION_THRESHOLD:-0.35}"
FALLBACK_POLICY="${FALLBACK_POLICY:-sstk_positive}"
HARD_REJECT_POLICY="${HARD_REJECT_POLICY:-cap_and_demote}"
UNSAFE_SCORE_CAP="${UNSAFE_SCORE_CAP:-0.05}"
PAIRWISE_SCORE_MARGIN="${PAIRWISE_SCORE_MARGIN:-0.03}"
MAX_PAIRS_PER_IMAGE="${MAX_PAIRS_PER_IMAGE:-48}"
LISTWISE_TEMPERATURE="${LISTWISE_TEMPERATURE:-0.12}"
REUSE_EXISTING="${REUSE_EXISTING:-0}"
SPLIT_SSTK_AFTER_BUILD="${SPLIT_SSTK_AFTER_BUILD:-0}"
SPLIT_SEED="${SPLIT_SEED:-20260422}"
SPLIT_TRAIN_FRACTION="${SPLIT_TRAIN_FRACTION:-0.8}"
SPLIT_VAL_FRACTION="${SPLIT_VAL_FRACTION:-0.1}"
SPLIT_TEST_FRACTION="${SPLIT_TEST_FRACTION:-0.1}"

UCTR_CHECKPOINT="${UCTR_CHECKPOINT:-artifacts/universal_crop_teacher_h_20260421/stage3_gate128_gaicselect/train_uctr_h_stage3_gate128/universal_crop_teacher_h_best.pt}"
UCTR_METHOD_NAME="${UCTR_METHOD_NAME:-universal_crop_teacher_h_stage3_gate128}"

PUBLIC_NORMALIZATION="${PUBLIC_NORMALIZATION:-rank_pct}"
PUBLIC_PRIMARY_WEIGHT="${PUBLIC_PRIMARY_WEIGHT:-0.65}"
PUBLIC_SECONDARY_WEIGHT="${PUBLIC_SECONDARY_WEIGHT:-0.35}"

case "${METHOD}" in
  t1)
    LABEL_DIR_NAME="${LABEL_DIR_NAME:-training_labels_t1_product_ar_260422}"
    SCORE_SOURCE_NAME="${SCORE_SOURCE_NAME:-sstk_teacher_t1_product_ar}"
    ;;
  uctr_stage3)
    LABEL_DIR_NAME="${LABEL_DIR_NAME:-training_labels_uctr_stage3_product_ar_260422}"
    SCORE_SOURCE_NAME="${SCORE_SOURCE_NAME:-uctr_stage3_gate128_product_ar}"
    ;;
  public_ensemble_best)
    LABEL_DIR_NAME="${LABEL_DIR_NAME:-training_labels_public_ensemble_best_product_ar_260422}"
    SCORE_SOURCE_NAME="${SCORE_SOURCE_NAME:-public_cropper_ensemble_best}"
    ;;
  *)
    echo "unsupported method: ${METHOD}" >&2
    exit 2
    ;;
esac

if [[ "${DATASET}" == "sstk_full_10000" ]]; then
  SPLIT_OUTPUT_ROOT="${SPLIT_OUTPUT_ROOT:-data/SSTK/Full_10000/artifacts/${LABEL_DIR_NAME}_hashsplit_80_10_10_260422}"
else
  SPLIT_OUTPUT_ROOT="${SPLIT_OUTPUT_ROOT:-}"
fi

OUT_ROOT="${OUT_ROOT:-artifacts/mobilecropnet_v4/product_ar_label_factory_20260422/${DATASET}/${METHOD}}"
mkdir -p "${OUT_ROOT}/candidate_rows" "${OUT_ROOT}/scores"

SUMMARY_INDEX="${OUT_ROOT}/generated_summaries.txt"
: > "${SUMMARY_INDEX}"

declare -a SPLITS=()
declare -a BATCH_JSONLS=()
declare -a CAND_PREFIXES=()
declare -a OUTPUT_LABEL_DIRS=()
declare -a DATASET_NAMES=()
declare -a FALLBACK_CAND_JSONLS=()
declare -a FALLBACK_BUILD_JSONLS=()

case "${DATASET}" in
  gaic_v2_official)
    SPLITS=(train val test)
    CAND_PREFIXES=(train val test)
    DATASET_NAMES=(gaic gaic gaic)
    BATCH_JSONLS=(
      "data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Train2636/artifacts/training_labels/gaic_v2_train_qf_c1c6capexp_c7exp_largecap_v2_260416_leftover_ignore_monotonic/train_conditional_detr_batch.jsonl"
      "data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Val200/artifacts/training_labels/gaic_v2_val_qf_c1c6capexp_c7exp_largecap_v2_260416_leftover_ignore_monotonic/train_conditional_detr_batch.jsonl"
      "data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Test500/artifacts/training_labels/gaic_v2_test_qf_c1c6capexp_c7exp_largecap_v2_260416_leftover_ignore_monotonic/train_conditional_detr_batch.jsonl"
    )
    OUTPUT_LABEL_DIRS=(
      "data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Train2636/artifacts/${LABEL_DIR_NAME}"
      "data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Val200/artifacts/${LABEL_DIR_NAME}"
      "data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Test500/artifacts/${LABEL_DIR_NAME}"
    )
    FALLBACK_CAND_JSONLS=(
      "artifacts/mobilecropnet_v4/product_ar_score_labels_260420/candidate_rows/train_product_ar_candidates.jsonl"
      "artifacts/mobilecropnet_v4/product_ar_score_labels_260420/candidate_rows/val_product_ar_candidates.jsonl"
      "artifacts/mobilecropnet_v4/product_ar_score_labels_260420/candidate_rows/test_product_ar_candidates.jsonl"
    )
    FALLBACK_BUILD_JSONLS=(
      "data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Train2636/artifacts/training_labels_public_score_product_ar_v1/train_conditional_detr_batch.jsonl"
      "data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Val200/artifacts/training_labels_public_score_product_ar_v1/train_conditional_detr_batch.jsonl"
      "data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Test500/artifacts/training_labels_public_score_product_ar_v1/train_conditional_detr_batch.jsonl"
    )
    ;;
  sstk_full_10000)
    SPLITS=(train)
    CAND_PREFIXES=(train)
    DATASET_NAMES=(sstk)
    BATCH_JSONLS=(
      "data/SSTK/Full_10000/artifacts/training_labels/260413_v1_multimode_leftover_ignore_monotonic/train_conditional_detr_batch.jsonl"
    )
    OUTPUT_LABEL_DIRS=(
      "data/SSTK/Full_10000/artifacts/${LABEL_DIR_NAME}"
    )
    FALLBACK_CAND_JSONLS=("")
    FALLBACK_BUILD_JSONLS=("")
    ;;
  *)
    echo "unsupported dataset: ${DATASET}" >&2
    exit 2
    ;;
esac

for idx in "${!SPLITS[@]}"; do
  SPLIT="${SPLITS[$idx]}"
  CAND_PREFIX="${CAND_PREFIXES[$idx]}"
  DATASET_NAME="${DATASET_NAMES[$idx]}"
  INPUT_BATCH_JSONL="${BATCH_JSONLS[$idx]}"
  OUTPUT_LABEL_DIR="${OUTPUT_LABEL_DIRS[$idx]}"
  FALLBACK_CAND_JSONL="${FALLBACK_CAND_JSONLS[$idx]}"
  FALLBACK_BUILD_JSONL="${FALLBACK_BUILD_JSONLS[$idx]}"

  mkdir -p "${OUTPUT_LABEL_DIR}"

  CAND_JSONL="${OUT_ROOT}/candidate_rows/${CAND_PREFIX}_product_ar_candidates.jsonl"
  CAND_SUMMARY="${OUT_ROOT}/candidate_rows/${CAND_PREFIX}_product_ar_candidates_summary.json"
  BUILD_INPUT_JSONL="${INPUT_BATCH_JSONL}"

  if [[ "${REUSE_EXISTING}" == "1" && -f "${CAND_JSONL}" && -f "${CAND_SUMMARY}" ]]; then
    echo "[label-gen] reuse candidate rows ${CAND_JSONL}"
  elif [[ -f "${INPUT_BATCH_JSONL}" ]]; then
    "${PYTHON_BIN}" src/scripts/export_mobilecropnet_v4_product_ar_candidates.py \
      --input_jsonl "${INPUT_BATCH_JSONL}" \
      --output_jsonl "${CAND_JSONL}" \
      --summary_json "${CAND_SUMMARY}" \
      --include_ignored_candidates \
      --include_overflow_candidates \
      --max_rows "${MAX_ROWS}"
  elif [[ -n "${FALLBACK_CAND_JSONL}" && -f "${FALLBACK_CAND_JSONL}" ]]; then
      "${PYTHON_BIN}" - "${FALLBACK_CAND_JSONL}" "${CAND_JSONL}" "${CAND_SUMMARY}" "${INPUT_BATCH_JSONL}" "${MAX_ROWS}" <<'PY'
import json
import sys
from pathlib import Path

fallback = Path(sys.argv[1])
dst = Path(sys.argv[2])
summary = Path(sys.argv[3])
missing_batch = sys.argv[4]
max_groups = int(sys.argv[5])
rows = 0
groups = set()
images = set()
kept_groups = set()
dst.parent.mkdir(parents=True, exist_ok=True)
with fallback.open("r", encoding="utf-8") as src_handle, dst.open("w", encoding="utf-8") as dst_handle:
    for line in src_handle:
        if not line.strip():
            continue
        row = json.loads(line)
        group_id = str(row.get("score_group_id", ""))
        if max_groups > 0 and group_id not in kept_groups and len(kept_groups) >= max_groups:
            continue
        kept_groups.add(group_id)
        dst_handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        rows += 1
        groups.add(group_id)
        images.add(str(row.get("image_id", "")))
payload = {
    "status": "ok",
    "fallback_used": True,
    "missing_input_batch_jsonl": missing_batch,
    "fallback_candidate_rows_jsonl": str(fallback),
    "output_jsonl": str(dst),
    "row_count": rows,
    "group_count": len(groups),
    "image_count": len(images),
}
summary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
    BUILD_INPUT_JSONL="${FALLBACK_BUILD_JSONL}"
  else
    echo "missing candidate source for ${DATASET} ${SPLIT}: ${INPUT_BATCH_JSONL}" >&2
    exit 1
  fi
  printf '%s\n' "${CAND_SUMMARY}" >> "${SUMMARY_INDEX}"

  if [[ ! -f "${BUILD_INPUT_JSONL}" ]]; then
    echo "missing build input jsonl for ${DATASET} ${SPLIT}: ${BUILD_INPUT_JSONL}" >&2
    exit 1
  fi

  case "${METHOD}" in
    t1)
      T1_SUMMARY="${OUTPUT_LABEL_DIR}/conversion_summary.json"
      "${PYTHON_BIN}" src/scripts/build_mobilecropnet_v4_product_ar_t1_labels.py \
        --candidate_rows_jsonl "${CAND_JSONL}" \
        --output_jsonl "${OUTPUT_LABEL_DIR}/train_conditional_detr_batch.jsonl" \
        --summary_json "${T1_SUMMARY}" \
        --output_pairwise_jsonl "${OUTPUT_LABEL_DIR}/train_pairwise.jsonl" \
        --output_listwise_jsonl "${OUTPUT_LABEL_DIR}/train_listwise.jsonl" \
        --score_source_name "${SCORE_SOURCE_NAME}" \
        --pairwise_score_margin "${PAIRWISE_SCORE_MARGIN}" \
        --max_pairs_per_image "${MAX_PAIRS_PER_IMAGE}" \
        --listwise_temperature "${LISTWISE_TEMPERATURE}" \
        --max_rows "${MAX_ROWS}"
      printf '%s\n' "${T1_SUMMARY}" >> "${SUMMARY_INDEX}"
      ;;
    uctr_stage3)
      SCORE_JSONL="${OUT_ROOT}/scores/${CAND_PREFIX}_uctr_stage3_scores.jsonl"
      SCORE_SUMMARY="${OUT_ROOT}/scores/${CAND_PREFIX}_uctr_stage3_scores_summary.json"
      BUILD_SUMMARY="${OUTPUT_LABEL_DIR}/conversion_summary.json"

      if [[ "${REUSE_EXISTING}" == "1" && -f "${SCORE_JSONL}" && -f "${SCORE_SUMMARY}" ]]; then
        echo "[label-gen] reuse uctr scores ${SCORE_JSONL}"
      else
        "${PYTHON_BIN}" src/scripts/score_mobilecropnet_v4_product_ar_candidates_with_uctr.py \
          --candidate_rows_jsonl "${CAND_JSONL}" \
          --checkpoint "${UCTR_CHECKPOINT}" \
          --output_jsonl "${SCORE_JSONL}" \
          --summary_json "${SCORE_SUMMARY}" \
          --dataset_name "${DATASET_NAME}" \
          --method_name "${UCTR_METHOD_NAME}" \
          --score_source "${SCORE_SOURCE_NAME}" \
          --high_score_threshold "${HIGH_SCORE_THRESHOLD}" \
          --low_score_threshold "${LOW_SCORE_THRESHOLD}" \
          --contradiction_threshold "${CONTRADICTION_THRESHOLD}" \
          --max_rows 0 \
          --batch_size "${BATCH_SIZE}" \
          --num_workers "${NUM_WORKERS}" \
          --max_candidates "${MAX_CANDIDATES}" \
          --image_size "${IMAGE_SIZE}" \
          --crop_size "${CROP_SIZE}" \
          --device "${DEVICE}"
      fi
      printf '%s\n' "${SCORE_SUMMARY}" >> "${SUMMARY_INDEX}"

      "${PYTHON_BIN}" src/scripts/build_mobilecropnet_v4_product_ar_score_labels.py \
        --input_jsonl "${BUILD_INPUT_JSONL}" \
        --scored_candidates_jsonl "${SCORE_JSONL}" \
        --output_jsonl "${OUTPUT_LABEL_DIR}/train_conditional_detr_batch.jsonl" \
        --summary_json "${BUILD_SUMMARY}" \
        --output_pairwise_jsonl "${OUTPUT_LABEL_DIR}/train_pairwise.jsonl" \
        --output_listwise_jsonl "${OUTPUT_LABEL_DIR}/train_listwise.jsonl" \
        --score_source_name "${SCORE_SOURCE_NAME}" \
        --fallback_policy "${FALLBACK_POLICY}" \
        --hard_reject_policy "${HARD_REJECT_POLICY}" \
        --unsafe_score_cap "${UNSAFE_SCORE_CAP}" \
        --max_rows "${MAX_ROWS}" \
        --max_pairs_per_image "${MAX_PAIRS_PER_IMAGE}" \
        --pairwise_score_margin "${PAIRWISE_SCORE_MARGIN}" \
        --listwise_temperature "${LISTWISE_TEMPERATURE}"
      printf '%s\n' "${BUILD_SUMMARY}" >> "${SUMMARY_INDEX}"
      ;;
    public_ensemble_best)
      GAIC_SCORE_JSONL="${OUT_ROOT}/scores/${CAND_PREFIX}_public_gaic_scores.jsonl"
      GAIC_SCORE_SUMMARY="${OUT_ROOT}/scores/${CAND_PREFIX}_public_gaic_scores_summary.json"
      CGS_SCORE_JSONL="${OUT_ROOT}/scores/${CAND_PREFIX}_public_cgs_scores.jsonl"
      CGS_SCORE_SUMMARY="${OUT_ROOT}/scores/${CAND_PREFIX}_public_cgs_scores_summary.json"
      FUSED_SCORE_JSONL="${OUT_ROOT}/scores/${CAND_PREFIX}_public_ensemble_scores.jsonl"
      FUSED_SCORE_SUMMARY="${OUT_ROOT}/scores/${CAND_PREFIX}_public_ensemble_scores_summary.json"
      BUILD_SUMMARY="${OUTPUT_LABEL_DIR}/conversion_summary.json"

      if [[ "${REUSE_EXISTING}" == "1" && -f "${GAIC_SCORE_JSONL}" && -f "${GAIC_SCORE_SUMMARY}" ]]; then
        echo "[label-gen] reuse public gaic scores ${GAIC_SCORE_JSONL}"
      else
        "${PYTHON_BIN}" src/scripts/score_mobilecropnet_v4_product_ar_candidates_with_public_cropper.py \
          --candidate_rows_jsonl "${CAND_JSONL}" \
          --output_jsonl "${GAIC_SCORE_JSONL}" \
          --summary_json "${GAIC_SCORE_SUMMARY}" \
          --method gaic \
          --score_source public_cropper_gaic \
          --high_score_threshold "${HIGH_SCORE_THRESHOLD}" \
          --low_score_threshold "${LOW_SCORE_THRESHOLD}" \
          --contradiction_threshold "${CONTRADICTION_THRESHOLD}" \
          --max_rows 0 \
          --device "${DEVICE}"
      fi
      printf '%s\n' "${GAIC_SCORE_SUMMARY}" >> "${SUMMARY_INDEX}"

      if [[ "${REUSE_EXISTING}" == "1" && -f "${CGS_SCORE_JSONL}" && -f "${CGS_SCORE_SUMMARY}" ]]; then
        echo "[label-gen] reuse public cgs scores ${CGS_SCORE_JSONL}"
      else
        "${PYTHON_BIN}" src/scripts/score_mobilecropnet_v4_product_ar_candidates_with_public_cropper.py \
          --candidate_rows_jsonl "${CAND_JSONL}" \
          --output_jsonl "${CGS_SCORE_JSONL}" \
          --summary_json "${CGS_SCORE_SUMMARY}" \
          --method cgs \
          --score_source public_cropper_cgs \
          --high_score_threshold "${HIGH_SCORE_THRESHOLD}" \
          --low_score_threshold "${LOW_SCORE_THRESHOLD}" \
          --contradiction_threshold "${CONTRADICTION_THRESHOLD}" \
          --max_rows 0 \
          --device "${DEVICE}"
      fi
      printf '%s\n' "${CGS_SCORE_SUMMARY}" >> "${SUMMARY_INDEX}"

      if [[ "${REUSE_EXISTING}" == "1" && -f "${FUSED_SCORE_JSONL}" && -f "${FUSED_SCORE_SUMMARY}" ]]; then
        echo "[label-gen] reuse fused public scores ${FUSED_SCORE_JSONL}"
      else
        "${PYTHON_BIN}" src/scripts/fuse_mobilecropnet_v4_product_ar_public_scores.py \
          --primary_scores_jsonl "${GAIC_SCORE_JSONL}" \
          --secondary_scores_jsonl "${CGS_SCORE_JSONL}" \
          --output_jsonl "${FUSED_SCORE_JSONL}" \
          --summary_json "${FUSED_SCORE_SUMMARY}" \
          --method_name public_cropper_ensemble_best \
          --score_source "${SCORE_SOURCE_NAME}" \
          --normalization "${PUBLIC_NORMALIZATION}" \
          --primary_weight "${PUBLIC_PRIMARY_WEIGHT}" \
          --secondary_weight "${PUBLIC_SECONDARY_WEIGHT}" \
          --high_score_threshold "${HIGH_SCORE_THRESHOLD}" \
          --low_score_threshold "${LOW_SCORE_THRESHOLD}" \
          --contradiction_threshold "${CONTRADICTION_THRESHOLD}" \
          --max_rows 0
      fi
      printf '%s\n' "${FUSED_SCORE_SUMMARY}" >> "${SUMMARY_INDEX}"

      "${PYTHON_BIN}" src/scripts/build_mobilecropnet_v4_product_ar_score_labels.py \
        --input_jsonl "${BUILD_INPUT_JSONL}" \
        --scored_candidates_jsonl "${FUSED_SCORE_JSONL}" \
        --output_jsonl "${OUTPUT_LABEL_DIR}/train_conditional_detr_batch.jsonl" \
        --summary_json "${BUILD_SUMMARY}" \
        --output_pairwise_jsonl "${OUTPUT_LABEL_DIR}/train_pairwise.jsonl" \
        --output_listwise_jsonl "${OUTPUT_LABEL_DIR}/train_listwise.jsonl" \
        --score_source_name "${SCORE_SOURCE_NAME}" \
        --fallback_policy "${FALLBACK_POLICY}" \
        --hard_reject_policy "${HARD_REJECT_POLICY}" \
        --unsafe_score_cap "${UNSAFE_SCORE_CAP}" \
        --max_rows "${MAX_ROWS}" \
        --max_pairs_per_image "${MAX_PAIRS_PER_IMAGE}" \
        --pairwise_score_margin "${PAIRWISE_SCORE_MARGIN}" \
        --listwise_temperature "${LISTWISE_TEMPERATURE}"
      printf '%s\n' "${BUILD_SUMMARY}" >> "${SUMMARY_INDEX}"
      ;;
  esac

  echo "[label-gen] method=${METHOD} dataset=${DATASET} split=${SPLIT} output=${OUTPUT_LABEL_DIR}"
done

if [[ "${DATASET}" == "sstk_full_10000" && "${SPLIT_SSTK_AFTER_BUILD}" == "1" ]]; then
  INPUT_SPLIT_DIR="${OUTPUT_LABEL_DIRS[0]}"
  "${PYTHON_BIN}" src/scripts/split_mobilecropnet_v4_label_dir_by_image.py \
    --input_label_dir "${INPUT_SPLIT_DIR}" \
    --output_root "${SPLIT_OUTPUT_ROOT}" \
    --seed "${SPLIT_SEED}" \
    --train_fraction "${SPLIT_TRAIN_FRACTION}" \
    --val_fraction "${SPLIT_VAL_FRACTION}" \
    --test_fraction "${SPLIT_TEST_FRACTION}"
  printf '%s\n' "${SPLIT_OUTPUT_ROOT}/split_manifest.json" >> "${SUMMARY_INDEX}"
  echo "[label-gen] dataset=${DATASET} split_policy=image_hash train=${SPLIT_TRAIN_FRACTION} val=${SPLIT_VAL_FRACTION} test=${SPLIT_TEST_FRACTION} output=${SPLIT_OUTPUT_ROOT}"
fi

"${PYTHON_BIN}" - "${METHOD}" "${DATASET}" "${OUT_ROOT}" "${SUMMARY_INDEX}" <<'PY'
import json
import sys
from pathlib import Path

method = sys.argv[1]
dataset = sys.argv[2]
out_root = Path(sys.argv[3])
summary_index = Path(sys.argv[4])
summary_paths = [Path(line.strip()) for line in summary_index.read_text(encoding="utf-8").splitlines() if line.strip()]
payload = {
    "status": "ok",
    "method": method,
    "dataset": dataset,
    "out_root": str(out_root),
    "summaries": [],
}
for path in summary_paths:
    row = {"path": str(path)}
    if path.exists():
      try:
        row["payload"] = json.loads(path.read_text(encoding="utf-8"))
      except Exception as exc:
        row["error"] = str(exc)
    else:
      row["missing"] = True
    payload["summaries"].append(row)
manifest = out_root / "label_generation_manifest.json"
manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"status": "ok", "manifest": str(manifest), "summary_count": len(summary_paths)}, ensure_ascii=False, indent=2))
PY
