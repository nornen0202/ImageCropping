#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/group-volume/users/jaden.ju/Sources/ImageCropping}"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python3}"
OUT_ROOT="${OUT_ROOT:-artifacts/mobilecropnet_v4/public_score_sstk_explain_260420}"
RAW_LABEL_ROOT="${RAW_LABEL_ROOT:-${OUT_ROOT}/labels}"
LABEL_DIR_NAME="${LABEL_DIR_NAME:-training_labels_public_score_sstk_explain_distill_v2}"
METHOD="${METHOD:-gaic}"
MAX_CANDIDATES_PER_IMAGE="${MAX_CANDIDATES_PER_IMAGE:-128}"
SCORE_TARGET_POLICY="${SCORE_TARGET_POLICY:-blend_rank_zscore}"
RAW_SCORE_TEMPERATURE="${RAW_SCORE_TEMPERATURE:-1.0}"
RAW_SCORE_BLEND_WEIGHT="${RAW_SCORE_BLEND_WEIGHT:-0.35}"
LISTWISE_TEMPERATURE="${LISTWISE_TEMPERATURE:-0.10}"
PAIRWISE_SCORE_MARGIN="${PAIRWISE_SCORE_MARGIN:-0.02}"
MAX_PAIRS_PER_IMAGE="${MAX_PAIRS_PER_IMAGE:-160}"

cd "${ROOT}"

convert_split() {
  local split_root="$1"
  local split_name="$2"
  local img_dir="$3"
  local out_dir="${split_root}/artifacts/${LABEL_DIR_NAME}"
  mkdir -p "${out_dir}"

  "${PYTHON_BIN}" src/scripts/convert_t6_score_labels_to_mobilecropnet_v4_batch.py \
    --t6_labels_jsonl "${RAW_LABEL_ROOT}/${split_name}_public_${METHOD}_score_labels.jsonl" \
    --output_jsonl "${out_dir}/train_conditional_detr_batch.jsonl" \
    --summary_json "${out_dir}/conversion_summary.json" \
    --project_root . \
    --image_root "${img_dir}" \
    --image_root "data/Publics/GAIC/images/${split_name}" \
    --split "${split_name}" \
    --max_candidates_per_image "${MAX_CANDIDATES_PER_IMAGE}" \
    --positive_top_k 1 \
    --positive_selection_policy best_safe \
    --unsafe_group_policy skip \
    --score_adjustment_policy safe_rescale \
    --unsafe_score_cap 0.05 \
    --unsafe_candidate_weight 0.85 \
    --score_target_policy "${SCORE_TARGET_POLICY}" \
    --raw_score_temperature "${RAW_SCORE_TEMPERATURE}" \
    --raw_score_blend_weight "${RAW_SCORE_BLEND_WEIGHT}" \
    --label_source_name "${METHOD}_public_score_sstk_explain_distill_v2" \
    --candidate_source "${METHOD}_public_score_official" \
    --score_semantics_name "${METHOD}_public_score_blend_rank_zscore" \
    --output_pairwise_jsonl "${out_dir}/train_pairwise.jsonl" \
    --output_listwise_jsonl "${out_dir}/train_listwise.jsonl" \
    --max_pairs_per_image "${MAX_PAIRS_PER_IMAGE}" \
    --pairwise_score_margin "${PAIRWISE_SCORE_MARGIN}" \
    --listwise_temperature "${LISTWISE_TEMPERATURE}"
}

convert_split \
  "data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Train2636" \
  "train" \
  "data/Publics/GAIC_v2/images/train"

convert_split \
  "data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Val200" \
  "val" \
  "data/Publics/GAIC_v2/images/val"

convert_split \
  "data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Test500" \
  "test" \
  "data/Publics/GAIC_v2/images/test"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
payload = {
    "label_dir_name": "${LABEL_DIR_NAME}",
    "method": "${METHOD}",
    "max_candidates_per_image": int("${MAX_CANDIDATES_PER_IMAGE}"),
    "score_target_policy": "${SCORE_TARGET_POLICY}",
    "raw_score_temperature": float("${RAW_SCORE_TEMPERATURE}"),
    "raw_score_blend_weight": float("${RAW_SCORE_BLEND_WEIGHT}"),
    "listwise_temperature": float("${LISTWISE_TEMPERATURE}"),
    "pairwise_score_margin": float("${PAIRWISE_SCORE_MARGIN}"),
    "max_pairs_per_image": int("${MAX_PAIRS_PER_IMAGE}"),
    "splits": {},
}
roots = {
    "train": "data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Train2636",
    "val": "data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Val200",
    "test": "data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Test500",
}
for split, root in roots.items():
    path = Path(root) / "artifacts" / "${LABEL_DIR_NAME}" / "conversion_summary.json"
    payload["splits"][split] = json.loads(path.read_text(encoding="utf-8"))
out = Path("${OUT_ROOT}") / "public_score_distill_v2_label_convert_summary.json"
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
