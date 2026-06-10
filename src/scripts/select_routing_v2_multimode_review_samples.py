#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_PHASE_ROOT = Path("data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529")
DEFAULT_RUN_TAG = "260609_v16_routing_v2_mode_catalog_nofood_petstrict_shotstrict_nms_v2"
DEFAULT_RUN_DIR = DEFAULT_PHASE_ROOT / "artifacts/training_labels_multimode" / DEFAULT_RUN_TAG


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select routing_v2-aware image ids for multimode crop review panels.")
    parser.add_argument("--label_json", type=Path, default=DEFAULT_RUN_DIR / "label_json/multimode_labels_full.json")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_RUN_DIR / "crop_review_routing_v2_v16_nofood_petstrict_shotstrict/unified_selection")
    parser.add_argument("--per_bucket", type=int, default=8)
    parser.add_argument("--max_images", type=int, default=160)
    parser.add_argument("--include_ar_mode_buckets", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include_co_primary_person_pet", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--co_primary_min_positive", type=int, default=2)
    parser.add_argument("--force_source_image_ids", default="", help="Comma-separated source image ids that must be included first.")
    return parser


def _source_id(image: dict[str, Any]) -> str:
    return str(image.get("source_image_id") or Path(str(image.get("file_name") or "")).stem)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def _ann_score(ann: dict[str, Any]) -> float:
    for key in ("score_mode", "score", "final_score"):
        if ann.get(key) is not None:
            return _safe_float(ann.get(key))
    attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
    scores = attrs.get("checklist_scores") if isinstance(attrs.get("checklist_scores"), dict) else {}
    return _safe_float(scores.get("final_score"))


def _routing(image: dict[str, Any]) -> dict[str, Any]:
    value = image.get("routing_v2_simple")
    return value if isinstance(value, dict) else {}


def _ann_routing(ann: dict[str, Any]) -> dict[str, Any]:
    attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
    value = attrs.get("routing_v2_simple")
    return value if isinstance(value, dict) else {}


def _ann_target_ar(ann: dict[str, Any]) -> str:
    attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
    return str(attrs.get("target_ar") or ann.get("target_ar") or "FREE")


def _row_key(row: dict[str, Any]) -> tuple[float, int, str]:
    return (-float(row.get("routing_confidence") or 0.0), -int(row.get("positive_count") or 0), str(row.get("source_image_id") or ""))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.label_json.read_text(encoding="utf-8"))
    images = payload.get("images") if isinstance(payload.get("images"), list) else []
    anns = payload.get("annotations") if isinstance(payload.get("annotations"), list) else []

    positive_count_by_image: Counter[int] = Counter()
    best_score_by_image: dict[int, float] = defaultdict(float)
    positive_mode_count_by_image: dict[int, Counter[str]] = defaultdict(Counter)
    positive_route_count_by_image: dict[int, Counter[str]] = defaultdict(Counter)
    positive_ar_mode_count_by_image: dict[int, Counter[str]] = defaultdict(Counter)
    for ann in anns:
        if not isinstance(ann, dict) or int(ann.get("gt_flag") or 0) != 1:
            continue
        image_id = int(ann.get("image_id") or -1)
        mode = str(ann.get("mode_name") or "unknown")
        target_ar = _ann_target_ar(ann)
        routing = _ann_routing(ann)
        route_family = str(routing.get("route_family_v2") or "missing")
        positive_count_by_image[image_id] += 1
        best_score_by_image[image_id] = max(best_score_by_image.get(image_id, 0.0), _ann_score(ann))
        positive_mode_count_by_image[image_id][mode] += 1
        positive_route_count_by_image[image_id][route_family] += 1
        positive_ar_mode_count_by_image[image_id][f"{mode}|{target_ar}"] += 1

    bucket_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_rows: list[dict[str, Any]] = []
    for image in images:
        if not isinstance(image, dict):
            continue
        image_id = int(image.get("id") or -1)
        routing = _routing(image)
        family = str(routing.get("route_family_v2") or "missing")
        shot = str(routing.get("person_shot_type") or "na")
        context = str(routing.get("context_intent") or "missing")
        feasible = "feasible" if bool(routing.get("mode_feasible", True)) else "infeasible"
        image_route = str(image.get("image_route_name_no_placement") or family)
        positive_routes = positive_route_count_by_image.get(image_id, Counter())
        positive_ar_modes = positive_ar_mode_count_by_image.get(image_id, Counter())
        row = {
            "image_id": image_id,
            "source_image_id": _source_id(image),
            "image_route_name_no_placement": image_route,
            "route_family_v2": family,
            "person_shot_type": shot,
            "context_intent": context,
            "mode_feasible": feasible,
            "routing_confidence": _safe_float(routing.get("routing_confidence")),
            "positive_count": int(positive_count_by_image.get(image_id, 0)),
            "best_score": round(float(best_score_by_image.get(image_id, 0.0)), 6),
            "positive_mode_counts": dict(positive_mode_count_by_image.get(image_id, Counter())),
            "positive_route_counts": dict(positive_routes),
            "positive_ar_mode_counts": dict(positive_ar_modes),
        }
        all_rows.append(row)
        bucket_rows[f"image_route:{image_route}"].append(row)
        bucket_rows[f"route_family:{family}"].append(row)
        if family == "person_single":
            bucket_rows[f"person_shot:{shot}"].append(row)
        if context in {"tight_subject", "environmental"}:
            bucket_rows[f"context:{context}"].append(row)
        if family == "pet_dogcat":
            bucket_rows[f"pet_context:{context}"].append(row)
        if feasible == "infeasible":
            bucket_rows[f"feasible:{feasible}"].append(row)
        if args.include_ar_mode_buckets:
            for ar_mode in positive_ar_modes:
                bucket_rows[f"positive_ar_mode:{ar_mode}"].append(row)
        if bool(args.include_co_primary_person_pet):
            min_positive = max(1, int(args.co_primary_min_positive))
            if int(positive_routes.get("person_single", 0)) >= min_positive and int(positive_routes.get("pet_dogcat", 0)) >= min_positive:
                bucket_rows["co_primary:person_single+pet_dogcat"].append(row)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_by_id: dict[str, dict[str, Any]] = {}
    bucket_summary: dict[str, Any] = {}

    def add_selected(row: dict[str, Any], reason: str) -> bool:
        source_id = str(row["source_image_id"])
        if source_id in selected_by_id:
            reasons = selected_by_id[source_id].setdefault("selection_reasons", [])
            if reason not in reasons:
                reasons.append(reason)
            return False
        row = dict(row)
        row["primary_selection_bucket"] = reason
        row["selection_reasons"] = [reason]
        selected.append(row)
        selected_ids.add(source_id)
        selected_by_id[source_id] = row
        return True

    force_ids = {value.strip() for value in str(args.force_source_image_ids).split(",") if value.strip()}
    if force_ids:
        forced = []
        for row in sorted(all_rows, key=_row_key):
            source_id = str(row["source_image_id"])
            if source_id not in force_ids:
                continue
            if add_selected(row, "forced:source_image_ids"):
                forced.append(selected_by_id[source_id])
        bucket_summary["forced:source_image_ids"] = {
            "candidate_count": len(force_ids),
            "picked_count": len(forced),
            "picked_ids": [row["source_image_id"] for row in forced],
        }
    for bucket in sorted(bucket_rows):
        rows = sorted(bucket_rows[bucket], key=_row_key)
        picked = []
        for row in rows:
            source_id = str(row["source_image_id"])
            if source_id in selected_ids:
                add_selected(row, bucket)
                continue
            if add_selected(row, bucket):
                picked.append(selected_by_id[source_id])
            if len(picked) >= max(1, int(args.per_bucket)):
                break
            if 0 < int(args.max_images) <= len(selected):
                break
        bucket_summary[bucket] = {
            "candidate_count": len(rows),
            "picked_count": len(picked),
            "picked_ids": [row["source_image_id"] for row in picked],
        }
        if 0 < int(args.max_images) <= len(selected):
            break

    if 0 < int(args.max_images) and len(selected) < int(args.max_images):
        for row in sorted(all_rows, key=_row_key):
            source_id = str(row["source_image_id"])
            if source_id in selected_ids:
                continue
            add_selected(row, "fill:high_confidence")
            if len(selected) >= int(args.max_images):
                break

    ids_path = output_dir / "routing_v2_review_selected_image_ids.txt"
    ids_path.write_text("\n".join(str(row["source_image_id"]) for row in selected) + "\n", encoding="utf-8")
    csv_path = output_dir / "routing_v2_review_selection_manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "source_image_id",
            "primary_selection_bucket",
            "selection_reasons_json",
            "image_route_name_no_placement",
            "route_family_v2",
            "person_shot_type",
            "context_intent",
            "mode_feasible",
            "routing_confidence",
            "positive_count",
            "best_score",
            "positive_mode_counts_json",
            "positive_route_counts_json",
            "positive_ar_mode_counts_json",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            out = dict(row)
            out["positive_mode_counts_json"] = json.dumps(out.pop("positive_mode_counts"), ensure_ascii=False, sort_keys=True)
            out["positive_route_counts_json"] = json.dumps(out.pop("positive_route_counts"), ensure_ascii=False, sort_keys=True)
            out["positive_ar_mode_counts_json"] = json.dumps(out.pop("positive_ar_mode_counts"), ensure_ascii=False, sort_keys=True)
            out["selection_reasons_json"] = json.dumps(out.pop("selection_reasons", []), ensure_ascii=False)
            out.pop("image_id", None)
            writer.writerow(out)

    image_route_counts = Counter(str(row["image_route_name_no_placement"]) for row in selected)
    route_counts = Counter(str(row["route_family_v2"]) for row in selected)
    shot_counts = Counter(str(row["person_shot_type"]) for row in selected if row["person_shot_type"] != "na")
    context_counts = Counter(str(row["context_intent"]) for row in selected)
    summary = {
        "state": "completed",
        "label_json": str(args.label_json),
        "selected_count": len(selected),
        "per_bucket": int(args.per_bucket),
        "max_images": int(args.max_images),
        "include_ar_mode_buckets": bool(args.include_ar_mode_buckets),
        "include_co_primary_person_pet": bool(args.include_co_primary_person_pet),
        "force_source_image_ids": sorted(force_ids),
        "ids_path": str(ids_path),
        "csv_path": str(csv_path),
        "image_route_counts": dict(sorted(image_route_counts.items())),
        "route_family_counts": dict(sorted(route_counts.items())),
        "person_shot_counts": dict(sorted(shot_counts.items())),
        "context_counts": dict(sorted(context_counts.items())),
        "bucket_summary": bucket_summary,
    }
    summary_path = output_dir / "routing_v2_review_selection_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
