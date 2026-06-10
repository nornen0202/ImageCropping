#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/group-volume/users/jaden.ju/Sources/ImageCropping}"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python3}"
OUT_ROOT="${OUT_ROOT:-artifacts/mobilecropnet_v4/public_score_sstk_explain_260420}"
LABEL_DIR_NAME="${LABEL_DIR_NAME:-training_labels_public_score_sstk_explain_safe_v1}"
METHOD="${METHOD:-gaic}"
PROTOCOL="${PROTOCOL:-Gc}"
MAX_IMAGES="${MAX_IMAGES:-0}"

cd "${ROOT}"

run_split() {
  local split_root="$1"
  local split_name="$2"
  local ann_json="$3"
  local img_dir="$4"
  local report_dir="$5"
  local label_prefix="$6"

  local max_args=()
  if [[ "${MAX_IMAGES}" != "0" ]]; then
    max_args=(--max_images "${MAX_IMAGES}")
  fi

  local raw_dir="${OUT_ROOT}/labels"
  local out_dir="${split_root}/artifacts/${LABEL_DIR_NAME}"
  mkdir -p "${raw_dir}" "${out_dir}"

  "${PYTHON_BIN}" src/scripts/export_gaic_public_cropper_score_labels.py \
    --annotations_json "${ann_json}" \
    --candidate_eval_jsonl "${report_dir}/candidate_eval_rows.jsonl" \
    --image_roots "${img_dir}" "data/Publics/GAIC/images/${split_name}" \
    --method "${METHOD}" \
    --protocol "${PROTOCOL}" \
    --split_name "${split_name}" \
    --output_jsonl "${raw_dir}/${split_name}_public_${METHOD}_score_labels.jsonl" \
    --summary_json "${raw_dir}/${split_name}_public_${METHOD}_score_labels_summary.json" \
    --score_source "${METHOD}_public_cropper" \
    --device cuda \
    "${max_args[@]}"

  "${PYTHON_BIN}" src/scripts/convert_t6_score_labels_to_mobilecropnet_v4_batch.py \
    --t6_labels_jsonl "${raw_dir}/${split_name}_public_${METHOD}_score_labels.jsonl" \
    --output_jsonl "${out_dir}/train_conditional_detr_batch.jsonl" \
    --summary_json "${out_dir}/conversion_summary.json" \
    --project_root . \
    --image_root "${img_dir}" \
    --image_root "data/Publics/GAIC/images/${split_name}" \
    --split "${split_name}" \
    --max_candidates_per_image 96 \
    --positive_top_k 1 \
    --positive_selection_policy best_safe \
    --unsafe_group_policy skip \
    --score_adjustment_policy safe_rescale \
    --unsafe_score_cap 0.05 \
    --unsafe_candidate_weight 0.85 \
    --label_source_name "${METHOD}_public_score_sstk_explain_safe_v1" \
    --candidate_source "${METHOD}_public_score_official" \
    --score_semantics_name "${METHOD}_public_score_rank_pct" \
    --output_pairwise_jsonl "${out_dir}/train_pairwise.jsonl" \
    --output_listwise_jsonl "${out_dir}/train_listwise.jsonl"

  if [[ "${split_name}" != "train" ]]; then
    "${PYTHON_BIN}" src/scripts/visualize_public_score_label_tiers.py \
      --labels_jsonl "${raw_dir}/${split_name}_public_${METHOD}_score_labels.jsonl" \
      --annotations_json "${ann_json}" \
      --image_roots "${img_dir}" "data/Publics/GAIC/images/${split_name}" \
      --output_dir "${OUT_ROOT}/tier_viz/${split_name}" \
      --max_per_tier 32 \
      --max_side 1500
  fi

  "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
summary = {
    "split": "${split_name}",
    "raw_label_summary": json.loads(Path("${raw_dir}/${split_name}_public_${METHOD}_score_labels_summary.json").read_text()),
    "conversion_summary": json.loads(Path("${out_dir}/conversion_summary.json").read_text()),
}
Path("${out_dir}/public_score_export_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
print(json.dumps({
    "split": summary["split"],
    "raw_rows": summary["raw_label_summary"].get("written_rows"),
    "converted_images": summary["conversion_summary"].get("written_rows"),
    "tiers": summary["raw_label_summary"].get("label_summary", {}).get("tiers", {}),
    "converted_buckets": summary["conversion_summary"].get("bucket_counts", {}),
}, ensure_ascii=False, indent=2))
PY
}

run_split \
  "data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Train2636" \
  "train" \
  "data/Publics/GAIC_v2/annotations_json/instances_train.json" \
  "data/Publics/GAIC_v2/images/train" \
  "data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Train2636/artifacts/reports/gaic_benchmark_eval_gaic_v2_train_qf_c1c6capexp_c7exp_largecap_v2_260416" \
  "gaic_v2_train"

run_split \
  "data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Val200" \
  "val" \
  "data/Publics/GAIC_v2/annotations_json/instances_val.json" \
  "data/Publics/GAIC_v2/images/val" \
  "data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Val200/artifacts/reports/gaic_benchmark_eval_gaic_v2_val_qf_c1c6capexp_c7exp_largecap_v2_260416" \
  "gaic_v2_val"

run_split \
  "data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Test500" \
  "test" \
  "data/Publics/GAIC_v2/annotations_json/instances_test.json" \
  "data/Publics/GAIC_v2/images/test" \
  "data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Test500/artifacts/reports/gaic_benchmark_eval_gaic_v2_test_qf_c1c6capexp_c7exp_largecap_v2_260416" \
  "gaic_v2_test"

"${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
root = Path("${OUT_ROOT}")
payload = {"out_root": str(root), "label_dir_name": "${LABEL_DIR_NAME}", "method": "${METHOD}", "splits": {}}
for split_name, split_root in {
    "train": "data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Train2636",
    "val": "data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Val200",
    "test": "data/GAIC_v2/OfficialSplits_SSTK_QF_C1C6CAPEXP_C7_EXP/Test500",
}.items():
    summary_path = Path(split_root) / "artifacts" / "${LABEL_DIR_NAME}" / "public_score_export_summary.json"
    payload["splits"][split_name] = json.loads(summary_path.read_text()) if summary_path.exists() else {"missing": str(summary_path)}
(root / "public_score_label_export_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
