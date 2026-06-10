#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from routing.routing_v2_v16_modes import IMAGE_ROUTE_NO_PLACEMENT_NAMES, V16_MODE_SPECS  # noqa: E402


EXPECTED_IMAGE_ROUTES = set(IMAGE_ROUTE_NO_PLACEMENT_NAMES)
EXPECTED_V16_MODES = {spec.name for spec in V16_MODE_SPECS}


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _animal_pose_head_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for row in _iter_jsonl(path):
        image_id = str(row.get("image_id") or "").strip()
        groups = row.get("visible_groups") if isinstance(row.get("visible_groups"), dict) else {}
        if image_id and str(row.get("species_hint") or "").lower() in {"dog", "cat", "dogcat_mixed"} and float(groups.get("head") or 0.0) >= 0.45:
            ids.add(image_id)
    return ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate routing_v2 precompute-native v17 outputs.")
    parser.add_argument("--training_label_dir", type=Path, required=True)
    parser.add_argument("--multimode_dir", type=Path, required=True)
    parser.add_argument("--animal_pose_jsonl", type=Path, required=True)
    parser.add_argument("--candidate_completion_overview", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    errors: list[str] = []
    warnings: list[str] = []
    tl_dir = args.training_label_dir
    mm_dir = args.multimode_dir
    summary = _load_json(mm_dir / "summary.json")
    candidate_summary = _load_json(args.candidate_completion_overview)
    if summary.get("image_count") != 10000:
        errors.append(f"image_count_not_10000:{summary.get('image_count')}")
    if summary.get("candidate_bank_empty_ar_counts"):
        errors.append(f"candidate_bank_empty_ar_counts:{summary.get('candidate_bank_empty_ar_counts')}")
    if int(summary.get("missing_candidate_image_count") or 0) != 0:
        errors.append(f"missing_candidate_image_count:{summary.get('missing_candidate_image_count')}")
    if candidate_summary.get("empty_after_by_ar"):
        errors.append(f"candidate_completion_empty_after:{candidate_summary.get('empty_after_by_ar')}")

    route_counts = Counter()
    context_counts = Counter()
    training_rows = 0
    seen_images: set[str] = set()
    for split in ("full", "train", "val"):
        path = tl_dir / f"routing_v2_v17_precompute_native_{split}.jsonl"
        if not path.exists():
            errors.append(f"missing_training_jsonl:{path}")
            continue
        split_rows = 0
        for row in _iter_jsonl(path):
            split_rows += 1
            if split == "full":
                training_rows += 1
                image_id = str(row.get("image_id") or "")
                seen_images.add(image_id)
                route = str(row.get("image_route_name_no_placement") or "")
                route_counts[route] += 1
                routing = row.get("routing_v2_simple") if isinstance(row.get("routing_v2_simple"), dict) else {}
                context_counts[str(routing.get("context_intent") or "")] += 1
                if route not in EXPECTED_IMAGE_ROUTES:
                    errors.append(f"invalid_image_route:{image_id}:{route}")
                if "environmental_portrait" in route or "food" in route:
                    errors.append(f"forbidden_image_route:{image_id}:{route}")
                if route.endswith("_center") or route.endswith("_rot"):
                    errors.append(f"image_route_contains_placement:{image_id}:{route}")
                if row.get("positive_vote_summary") is not None:
                    errors.append(f"image_row_has_positive_vote_summary:{image_id}")
        if split == "full" and split_rows != summary.get("image_count"):
            errors.append(f"full_training_row_count_mismatch:{split_rows}!={summary.get('image_count')}")

    pet_pose_ids = _animal_pose_head_ids(args.animal_pose_jsonl)
    mode_counts = Counter()
    positive_counts = Counter()
    negative_reasons = Counter()
    bad_full_body = 0
    bad_pet_pose = 0
    image_id_by_int: dict[int, str] = {}
    payload = _load_json(mm_dir / "label_json/multimode_labels_full.json")
    for image in payload.get("images", []):
        image_id_by_int[int(image["id"])] = str(image.get("source_image_id") or "")
    categories = payload.get("categories") if isinstance(payload.get("categories"), list) else []
    category_names = {str(cat.get("name")) for cat in categories if isinstance(cat, dict)}
    if category_names != EXPECTED_V16_MODES:
        errors.append(f"v16_category_names_mismatch:missing={sorted(EXPECTED_V16_MODES-category_names)} extra={sorted(category_names-EXPECTED_V16_MODES)}")
    for ann in payload.get("annotations", []):
        mode = str(ann.get("mode_name") or "")
        mode_counts[mode] += 1
        if mode not in EXPECTED_V16_MODES:
            errors.append(f"invalid_mode:{ann.get('id')}:{mode}")
        if "environmental_portrait" in mode or "food" in mode:
            errors.append(f"forbidden_mode:{ann.get('id')}:{mode}")
        attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
        routing = attrs.get("routing_v2_simple") if isinstance(attrs.get("routing_v2_simple"), dict) else {}
        if routing.get("placement_intent") not in {"center", "rot"}:
            errors.append(f"crop_mode_missing_placement:{ann.get('id')}:{mode}")
        if int(ann.get("gt_flag") or 0) == 1:
            positive_counts[mode] += 1
            if mode.startswith("person_single_full_body") and attrs.get("crop_shot_inferred") != "full_body":
                bad_full_body += 1
            if mode.startswith("pet_dogcat"):
                source_id = image_id_by_int.get(int(ann.get("image_id") or -1), "")
                if source_id not in pet_pose_ids:
                    bad_pet_pose += 1
        else:
            negative_reasons[str(attrs.get("negative_reason") or "")] += 1
    if bad_full_body:
        errors.append(f"positive_full_body_crop_shot_mismatch:{bad_full_body}")
    if bad_pet_pose:
        errors.append(f"positive_pet_without_pose_head:{bad_pet_pose}")

    out = {
        "state": "passed" if not errors else "failed",
        "errors": errors[:200],
        "warning_count": len(warnings),
        "warnings": warnings[:200],
        "training_rows_full": training_rows,
        "unique_training_images": len(seen_images),
        "annotation_count": len(payload.get("annotations", [])),
        "positive_annotation_count": sum(positive_counts.values()),
        "image_route_counts": dict(route_counts),
        "context_counts": dict(context_counts),
        "mode_counts": dict(mode_counts),
        "positive_mode_counts": dict(positive_counts),
        "negative_reason_top": dict(negative_reasons.most_common(20)),
        "candidate_completion": {
            "completed_row_count": candidate_summary.get("completed_row_count"),
            "completed_candidate_count": candidate_summary.get("completed_candidate_count"),
            "empty_after_by_ar": candidate_summary.get("empty_after_by_ar"),
        },
    }
    _write_json(args.output_json, out)
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
