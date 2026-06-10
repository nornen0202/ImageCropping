#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from routing.routing_v2_v16_modes import (  # noqa: E402
    IMAGE_ROUTE_NO_PLACEMENT_NAMES,
    V16_MODE_SPECS,
    image_route_no_placement_from_routing,
)


EXPECTED_IMAGE_ROUTES = list(IMAGE_ROUTE_NO_PLACEMENT_NAMES)
EXPECTED_V16_MODES = [spec.name for spec in V16_MODE_SPECS]
FORBIDDEN_PRIMARY_TARGETS = {
    "v16_mode_name",
    "v16_mode_id",
    "placement_intent",
    "placement_intent_id",
    "flat_route_class",
    "flat_route_class_id",
}
PRIMARY_TARGET_KEYS = {
    "primary_target",
    "primary_label",
    "image_level_target",
    "default_target",
    "classification_target",
    "label_source_default",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate SSTK routing_v2/v16 taxonomy and artifact contract.")
    parser.add_argument("--training-label-dir", type=Path, required=True)
    parser.add_argument("--multimode-dir", type=Path, required=True)
    parser.add_argument("--animal-pose-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--max-json-config-bytes", type=int, default=2_000_000)
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _record_error(errors: list[str], code: str, detail: str) -> None:
    errors.append(f"{code}: {detail}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _catalog_names_from_v16(payload: dict[str, Any]) -> list[str]:
    specs = payload.get("mode_specs")
    if isinstance(specs, list):
        return [str(item.get("name")) for item in specs if isinstance(item, dict)]
    categories = payload.get("categories")
    if isinstance(categories, list):
        return [str(item.get("name")) for item in categories if isinstance(item, dict)]
    return []


def _validate_catalogs(training_label_dir: Path, multimode_dir: Path, errors: list[str]) -> dict[str, Any]:
    image_catalog_path = training_label_dir / "image_route_no_placement_catalog.json"
    v16_catalog_path = training_label_dir / "v16_mode_catalog.json"
    multimode_image_catalog_path = multimode_dir / "image_route_no_placement_catalog.json"
    multimode_v16_catalog_path = multimode_dir / "v16_mode_catalog.json"
    for path in (image_catalog_path, v16_catalog_path, multimode_image_catalog_path, multimode_v16_catalog_path):
        if not path.is_file():
            _record_error(errors, "missing_catalog", str(path))

    image_payload = _load_json(image_catalog_path) if image_catalog_path.is_file() else {}
    v16_payload = _load_json(v16_catalog_path) if v16_catalog_path.is_file() else {}
    route_names = [str(x) for x in image_payload.get("route_names", [])]
    v16_names = _catalog_names_from_v16(v16_payload)
    if route_names != EXPECTED_IMAGE_ROUTES:
        _record_error(errors, "image_route_catalog_mismatch", f"actual={route_names}")
    if v16_names != EXPECTED_V16_MODES:
        _record_error(errors, "v16_mode_catalog_mismatch", f"actual={v16_names}")
    forbidden_names = [name for name in route_names + v16_names if "environmental_portrait" in name]
    if forbidden_names:
        _record_error(errors, "environmental_portrait_class_reintroduced", ",".join(forbidden_names))
    target_policy = str(image_payload.get("target_policy") or "")
    if "excludes placement_intent" not in target_policy:
        _record_error(errors, "image_route_target_policy_missing_no_placement", target_policy)
    return {
        "image_route_catalog": str(image_catalog_path),
        "v16_mode_catalog": str(v16_catalog_path),
        "image_route_names": route_names,
        "v16_mode_names": v16_names,
        "image_route_count": len(route_names),
        "v16_mode_count": len(v16_names),
    }


def _find_primary_target_violations(path: Path, payload: Any, errors: list[str]) -> list[dict[str, str]]:
    violations: list[dict[str, str]] = []

    def walk(obj: Any, parent_key: str = "") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                key_text = str(key)
                if key_text in PRIMARY_TARGET_KEYS and str(value) in FORBIDDEN_PRIMARY_TARGETS:
                    item = {"path": str(path), "key": key_text, "value": str(value)}
                    violations.append(item)
                    _record_error(errors, "forbidden_primary_target", json.dumps(item, sort_keys=True))
                walk(value, key_text)
        elif isinstance(obj, list):
            for item in obj:
                walk(item, parent_key)

    walk(payload)
    return violations


def _scan_small_json_configs(root: Path, errors: list[str], max_bytes: int) -> list[dict[str, str]]:
    violations: list[dict[str, str]] = []
    for path in sorted(root.rglob("*.json")):
        if path.name in {"multimode_labels_full.json", "multimode_labels_train.json", "multimode_labels_val.json"}:
            continue
        try:
            if path.stat().st_size > int(max_bytes):
                continue
            payload = _load_json(path)
        except Exception:
            continue
        violations.extend(_find_primary_target_violations(path, payload, errors))
    return violations


def _validate_split_rows(training_label_dir: Path, errors: list[str]) -> dict[str, Any]:
    row_paths = {
        "train": training_label_dir / "routing_v2_v16_train.jsonl",
        "val": training_label_dir / "routing_v2_v16_val.jsonl",
    }
    for path in row_paths.values():
        if not path.is_file():
            _record_error(errors, "missing_split_jsonl", str(path))

    split_ids: dict[str, set[str]] = {"train": set(), "val": set()}
    route_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    missing_fields: Counter[str] = Counter()
    invalid_route_rows = 0
    invalid_mode_rows = 0
    environmental_rows = 0
    positive_vote_override_rows = 0
    row_counts: Counter[str] = Counter()
    required_fields = [
        "image_id",
        "image_route_name_no_placement",
        "image_route_id_no_placement",
        "v16_mode_name",
        "v16_mode_id",
        "routing_v2_simple",
        "hierarchical_targets",
        "split",
    ]

    for split, path in row_paths.items():
        if not path.is_file():
            continue
        for row in _iter_jsonl(path):
            row_counts[split] += 1
            image_id = str(row.get("image_id") or "")
            if image_id:
                split_ids[split].add(image_id)
            if str(row.get("split") or split) != split:
                _record_error(errors, "split_field_mismatch", f"{path}:{image_id}:{row.get('split')}")
            for field in required_fields:
                if field not in row or row.get(field) in (None, ""):
                    missing_fields[field] += 1
            route_name = str(row.get("image_route_name_no_placement") or "")
            mode_name = str(row.get("v16_mode_name") or "")
            route_counts[route_name] += 1
            mode_counts[mode_name] += 1
            if route_name not in EXPECTED_IMAGE_ROUTES:
                invalid_route_rows += 1
            if mode_name not in EXPECTED_V16_MODES:
                invalid_mode_rows += 1
            if "environmental_portrait" in route_name or "environmental_portrait" in mode_name:
                environmental_rows += 1
            routing = row.get("routing_v2_simple") if isinstance(row.get("routing_v2_simple"), dict) else {}
            derived_route = image_route_no_placement_from_routing(routing)
            vote = row.get("positive_vote_summary") if isinstance(row.get("positive_vote_summary"), dict) else {}
            vote_family = str(vote.get("route_family_v2") or "")
            if vote_family and vote_family != str(routing.get("route_family_v2") or "") and route_name != derived_route:
                positive_vote_override_rows += 1

    overlap = split_ids["train"] & split_ids["val"]
    if overlap:
        _record_error(errors, "train_val_overlap", str(len(overlap)))
    if invalid_route_rows:
        _record_error(errors, "invalid_image_route_rows", str(invalid_route_rows))
    if invalid_mode_rows:
        _record_error(errors, "invalid_v16_mode_rows", str(invalid_mode_rows))
    if environmental_rows:
        _record_error(errors, "environmental_portrait_rows", str(environmental_rows))
    if positive_vote_override_rows:
        _record_error(errors, "positive_vote_override_rows", str(positive_vote_override_rows))
    for field, count in sorted(missing_fields.items()):
        if count:
            _record_error(errors, "missing_required_field", f"{field}={count}")

    return {
        "row_counts": dict(row_counts),
        "train_unique_images": len(split_ids["train"]),
        "val_unique_images": len(split_ids["val"]),
        "train_val_overlap": len(overlap),
        "missing_required_fields": dict(missing_fields),
        "route_counts": dict(sorted(route_counts.items())),
        "mode_counts": dict(sorted(mode_counts.items())),
        "invalid_image_route_rows": invalid_route_rows,
        "invalid_v16_mode_rows": invalid_mode_rows,
        "environmental_portrait_rows": environmental_rows,
        "positive_vote_override_rows": positive_vote_override_rows,
    }


def _query_status_path(multimode_dir: Path) -> Path | None:
    for name in ("mode_query_status_v16.jsonl", "mode_query_status.jsonl"):
        path = multimode_dir / name
        if path.is_file():
            return path
    return None


def _validate_multimode(multimode_dir: Path, errors: list[str]) -> dict[str, Any]:
    summary_path = multimode_dir / "summary.json"
    if not summary_path.is_file():
        _record_error(errors, "missing_multimode_summary", str(summary_path))
        summary = {}
    else:
        summary = _load_json(summary_path)
    query_path = _query_status_path(multimode_dir)
    if query_path is None:
        _record_error(errors, "missing_mode_query_status", str(multimode_dir))
        return {"summary_json": str(summary_path), "query_status_jsonl": None}

    mode_counts: Counter[str] = Counter()
    no_positive_reasons: Counter[str] = Counter()
    missing_no_positive_reason = 0
    query_route_counts: Counter[str] = Counter()
    rows = 0
    missing_query_fields: Counter[str] = Counter()
    required_fields = ["query_id", "target_ar", "v16_mode_name", "query_route_name_no_placement", "positive_exists", "no_positive_reason"]
    for row in _iter_jsonl(query_path):
        rows += 1
        for field in required_fields:
            if field not in row:
                missing_query_fields[field] += 1
        mode_name = str(row.get("v16_mode_name") or row.get("mode_name") or "")
        route_name = str(row.get("query_route_name_no_placement") or "")
        mode_counts[mode_name] += 1
        query_route_counts[route_name] += 1
        if mode_name not in EXPECTED_V16_MODES:
            _record_error(errors, "invalid_query_v16_mode", mode_name)
        if route_name and route_name not in EXPECTED_IMAGE_ROUTES:
            _record_error(errors, "invalid_query_image_route", route_name)
        if int(row.get("positive_exists") or 0) == 0:
            reason = str(row.get("no_positive_reason") or "")
            if not reason:
                missing_no_positive_reason += 1
            else:
                no_positive_reasons[reason] += 1
    for field, count in sorted(missing_query_fields.items()):
        if count:
            _record_error(errors, "missing_query_status_field", f"{field}={count}")
    if missing_no_positive_reason:
        _record_error(errors, "missing_no_positive_reason", str(missing_no_positive_reason))
    missing_modes = [name for name in EXPECTED_V16_MODES if mode_counts.get(name, 0) == 0]
    if missing_modes:
        _record_error(errors, "missing_v16_modes_in_query_status", ",".join(missing_modes))

    return {
        "summary_json": str(summary_path),
        "query_status_jsonl": str(query_path),
        "summary_state": summary.get("state"),
        "summary_schema_version": summary.get("schema_version"),
        "query_rows": rows,
        "mode_counts": dict(sorted(mode_counts.items())),
        "query_route_counts": dict(sorted(query_route_counts.items())),
        "missing_no_positive_reason": missing_no_positive_reason,
        "no_positive_reasons": dict(no_positive_reasons.most_common(25)),
        "missing_query_status_fields": dict(missing_query_fields),
    }


def _validate_animal_pose(animal_pose_dir: Path | None, errors: list[str]) -> dict[str, Any]:
    if animal_pose_dir is None:
        return {"provided": False}
    summary_path = animal_pose_dir / "summary.json"
    if not summary_path.is_file():
        _record_error(errors, "missing_animal_pose_summary", str(summary_path))
        return {"provided": True, "summary_json": str(summary_path), "exists": False}
    summary = _load_json(summary_path)
    if float(summary.get("det_nms_iou", -1.0)) != 0.65:
        _record_error(errors, "animal_pose_det_nms_iou_mismatch", str(summary.get("det_nms_iou")))
    if float(summary.get("det_nms_containment", -1.0)) != 0.85:
        _record_error(errors, "animal_pose_det_nms_containment_mismatch", str(summary.get("det_nms_containment")))
    if "raw_instances_seen" not in summary:
        _record_error(errors, "animal_pose_missing_raw_instances_seen", str(summary_path))
    if "nms_suppressed_instances" not in summary:
        _record_error(errors, "animal_pose_missing_nms_suppressed_instances", str(summary_path))
    if int(summary.get("nms_suppressed_instances") or 0) <= 0:
        _record_error(errors, "animal_pose_no_nms_suppression_observed", str(summary.get("nms_suppressed_instances")))
    return {
        "provided": True,
        "summary_json": str(summary_path),
        "state": summary.get("state"),
        "raw_instances_seen": summary.get("raw_instances_seen"),
        "nms_suppressed_instances": summary.get("nms_suppressed_instances"),
        "det_nms_iou": summary.get("det_nms_iou"),
        "det_nms_containment": summary.get("det_nms_containment"),
        "rows_written": summary.get("rows_written"),
        "pose_counter": summary.get("pose_counter"),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start = time.time()
    errors: list[str] = []
    warnings: list[str] = []
    training_label_dir = args.training_label_dir
    multimode_dir = args.multimode_dir
    for path in (training_label_dir, multimode_dir):
        if not path.is_dir():
            _record_error(errors, "missing_directory", str(path))

    catalogs = _validate_catalogs(training_label_dir, multimode_dir, errors) if training_label_dir.is_dir() and multimode_dir.is_dir() else {}
    rows = _validate_split_rows(training_label_dir, errors) if training_label_dir.is_dir() else {}
    primary_target_violations = []
    if training_label_dir.is_dir():
        primary_target_violations.extend(_scan_small_json_configs(training_label_dir, errors, int(args.max_json_config_bytes)))
    if multimode_dir.is_dir():
        primary_target_violations.extend(_scan_small_json_configs(multimode_dir, errors, int(args.max_json_config_bytes)))
    multimode = _validate_multimode(multimode_dir, errors) if multimode_dir.is_dir() else {}
    animal_pose = _validate_animal_pose(args.animal_pose_dir, errors)

    out_dir = args.output_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    route_distribution_json = out_dir / "route_distributions.json"
    route_distribution_csv = out_dir / "route_distributions.csv"
    distributions = {
        "image_route_counts": rows.get("route_counts", {}),
        "v16_mode_counts": rows.get("mode_counts", {}),
        "query_route_counts": multimode.get("query_route_counts", {}),
        "query_v16_mode_counts": multimode.get("mode_counts", {}),
    }
    route_distribution_json.write_text(json.dumps(distributions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_rows = []
    for group, counts in distributions.items():
        if isinstance(counts, dict):
            for name, count in counts.items():
                csv_rows.append({"group": group, "name": name, "count": count})
    _write_csv(route_distribution_csv, csv_rows)

    payload = {
        "ok": not errors,
        "generated_at_unix": time.time(),
        "elapsed_sec": round(time.time() - start, 3),
        "training_label_dir": str(training_label_dir),
        "multimode_dir": str(multimode_dir),
        "catalogs": catalogs,
        "split_rows": rows,
        "primary_target_violations": primary_target_violations,
        "multimode": multimode,
        "animal_pose": animal_pose,
        "route_distribution_json": str(route_distribution_json),
        "route_distribution_csv": str(route_distribution_csv),
        "errors": errors,
        "warnings": warnings,
    }
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
