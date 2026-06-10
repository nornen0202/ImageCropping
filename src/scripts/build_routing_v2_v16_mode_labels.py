#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from routing.routing_v2_v16_modes import (  # noqa: E402
    IMAGE_ROUTE_NO_PLACEMENT_SCHEMA_VERSION,
    V16_SCHEMA_VERSION,
    image_route_no_placement_catalog_payload,
    image_route_no_placement_from_routing,
    image_route_no_placement_id_from_routing,
    v16_categories,
    v16_mode_catalog_payload,
    v16_mode_from_routing,
)


DEFAULT_PHASE_ROOT = Path("data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529")
DEFAULT_SOURCE_TAG = "260609_routing_v2_simple_v8_nofood_petstrict_shotstrict_nms_v2"
DEFAULT_RUN_TAG = "260609_v16_routing_v2_mode_catalog_nofood_petstrict_shotstrict_nms_v2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Promote routing_v2_simple labels to the v16 route/shot/placement mode catalog.")
    parser.add_argument("--phase_root", type=Path, default=DEFAULT_PHASE_ROOT)
    parser.add_argument("--source_run_tag", default=DEFAULT_SOURCE_TAG)
    parser.add_argument("--run_tag", default=DEFAULT_RUN_TAG)
    return parser


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _source_image_id(image: dict[str, Any]) -> str:
    return str(image.get("source_image_id") or Path(str(image.get("file_name") or "")).stem)


def _mode_from_routing_container(container: dict[str, Any]) -> tuple[str, int]:
    routing = container.get("routing_v2_simple") if isinstance(container.get("routing_v2_simple"), dict) else {}
    spec = v16_mode_from_routing(routing)
    return spec.name, int(spec.category_id)


def _image_route_no_placement_fields(container: dict[str, Any]) -> dict[str, Any]:
    routing = container.get("routing_v2_simple") if isinstance(container.get("routing_v2_simple"), dict) else {}
    return {
        "image_route_schema_version": IMAGE_ROUTE_NO_PLACEMENT_SCHEMA_VERSION,
        "image_route_name_no_placement": image_route_no_placement_from_routing(routing),
        "image_route_id_no_placement": image_route_no_placement_id_from_routing(routing),
    }


def _augment_multimode_split(source_path: Path, output_path: Path) -> dict[str, Any]:
    payload = _read_json(source_path)
    images = payload.get("images") if isinstance(payload.get("images"), list) else []
    anns = payload.get("annotations") if isinstance(payload.get("annotations"), list) else []
    id_to_source = {int(image.get("id")): _source_image_id(image) for image in images if image.get("id") is not None}
    category_counts: Counter[str] = Counter()
    image_route_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    source_mode_counts: Counter[str] = Counter()

    for image in images:
        mode_name, mode_id = _mode_from_routing_container(image)
        image["v16_mode_name"] = mode_name
        image["v16_mode_id"] = mode_id
        image.update(_image_route_no_placement_fields(image))
        image_route_counts[str(image["image_route_name_no_placement"])] += 1
        routing = image.get("routing_v2_simple") if isinstance(image.get("routing_v2_simple"), dict) else {}
        image["routing_v2_mode_catalog_version"] = V16_SCHEMA_VERSION
        image["v16_mode_key"] = {
            "route_family_v2": routing.get("route_family_v2"),
            "person_shot_type": routing.get("person_shot_type"),
            "placement_intent": routing.get("placement_intent"),
        }

    for ann in anns:
        attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
        if not isinstance(attrs, dict):
            attrs = {}
        routing = attrs.get("routing_v2_simple") if isinstance(attrs.get("routing_v2_simple"), dict) else {}
        spec = v16_mode_from_routing(routing)
        old_mode_name = str(ann.get("mode_name") or "")
        source_mode_counts[old_mode_name] += 1
        ann["source_v14_mode_name"] = old_mode_name
        ann["source_v14_category_id"] = ann.get("category_id")
        ann["mode_name"] = spec.name
        ann["category_id"] = int(spec.category_id)
        target_ar = str(attrs.get("target_ar") or "")
        entity_id = str(ann.get("entity_id") or "")
        source_image_id = id_to_source.get(int(ann.get("image_id") or -1), "")
        ann["source_query_id"] = ann.get("query_id")
        if source_image_id and target_ar and entity_id:
            ann["query_id"] = f"{source_image_id}::{target_ar}::{spec.name}::{entity_id}"
        attrs["routing_v2_mode_catalog_version"] = V16_SCHEMA_VERSION
        attrs["v16_mode_name"] = spec.name
        attrs["v16_mode_id"] = int(spec.category_id)
        attrs["source_v14_mode_name"] = old_mode_name
        attrs["v16_mode_key"] = {
            "route_family_v2": spec.route_family_v2,
            "person_shot_type": spec.person_shot_type,
            "placement_intent": spec.placement_intent,
        }
        ann["attributes"] = attrs
        category_counts[spec.name] += 1
        route_counts[spec.route_family_v2] += 1

    payload["categories"] = v16_categories()
    payload["routing_v2_mode_catalog_version"] = V16_SCHEMA_VERSION
    _write_json(output_path, payload)
    return {
        "images": len(images),
        "annotations": len(anns),
        "image_route_no_placement_counts": dict(sorted(image_route_counts.items())),
        "mode_counts": dict(sorted(category_counts.items())),
        "route_counts": dict(sorted(route_counts.items())),
        "source_v14_mode_counts": dict(sorted(source_mode_counts.items())),
    }


def _augment_training_split(source_path: Path, output_path: Path) -> dict[str, Any]:
    rows = _iter_jsonl(source_path)
    counts: Counter[str] = Counter()
    image_route_counts: Counter[str] = Counter()
    for row in rows:
        mode_name, mode_id = _mode_from_routing_container(row)
        row["routing_v2_mode_catalog_version"] = V16_SCHEMA_VERSION
        row["v16_mode_name"] = mode_name
        row["v16_mode_id"] = mode_id
        row.update(_image_route_no_placement_fields(row))
        counts[mode_name] += 1
        image_route_counts[str(row["image_route_name_no_placement"])] += 1
    _write_jsonl(output_path, rows)
    return {
        "rows": len(rows),
        "mode_counts": dict(sorted(counts.items())),
        "image_route_no_placement_counts": dict(sorted(image_route_counts.items())),
    }


def _augment_query_status(source_path: Path, output_path: Path) -> dict[str, Any]:
    rows = _iter_jsonl(source_path)
    counts: Counter[str] = Counter()
    for row in rows:
        routing = row.get("routing_v2_simple") if isinstance(row.get("routing_v2_simple"), dict) else {}
        spec = v16_mode_from_routing(routing)
        row["source_v14_mode_name"] = row.get("mode_name")
        row["mode_name"] = spec.name
        row["routing_v2_mode_catalog_version"] = V16_SCHEMA_VERSION
        row["v16_mode_name"] = spec.name
        row["v16_mode_id"] = int(spec.category_id)
        row["query_route_name_no_placement"] = image_route_no_placement_from_routing(routing)
        row["query_route_id_no_placement"] = image_route_no_placement_id_from_routing(routing)
        counts[spec.name] += 1
    _write_jsonl(output_path, rows)
    return {"rows": len(rows), "mode_counts": dict(sorted(counts.items()))}


def _write_mode_stats(path: Path, counts: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode_name", "count"])
        for key, value in sorted(counts.items()):
            writer.writerow([key, int(value)])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    start = time.time()
    phase_root: Path = args.phase_root
    source_mm = phase_root / "artifacts/training_labels_multimode" / args.source_run_tag
    source_tl = phase_root / "artifacts/training_labels" / args.source_run_tag
    out_mm = phase_root / "artifacts/training_labels_multimode" / args.run_tag
    out_tl = phase_root / "artifacts/training_labels" / args.run_tag
    label_out = out_mm / "label_json"
    out_mm.mkdir(parents=True, exist_ok=True)
    out_tl.mkdir(parents=True, exist_ok=True)
    status_path = out_mm / "status.json"
    _write_json(status_path, {"state": "running", "phase": "start", "run_tag": args.run_tag})
    _write_json(out_mm / "v16_mode_catalog.json", v16_mode_catalog_payload())
    _write_json(out_tl / "v16_mode_catalog.json", v16_mode_catalog_payload())
    _write_json(out_mm / "image_route_no_placement_catalog.json", image_route_no_placement_catalog_payload())
    _write_json(out_tl / "image_route_no_placement_catalog.json", image_route_no_placement_catalog_payload())

    split_summaries: dict[str, Any] = {}
    for split in ("full", "train", "val"):
        _write_json(status_path, {"state": "running", "phase": f"multimode_{split}"})
        split_summaries[f"multimode_{split}"] = _augment_multimode_split(
            source_mm / "label_json" / f"multimode_labels_{split}.json",
            label_out / f"multimode_labels_{split}.json",
        )
        target_ar_source = source_mm / "label_json" / f"multimode_labels_target_ar_only_{split}.json"
        if target_ar_source.exists():
            shutil.copy2(target_ar_source, label_out / f"multimode_labels_target_ar_only_{split}.json")
        _write_json(status_path, {"state": "running", "phase": f"training_{split}"})
        split_summaries[f"training_{split}"] = _augment_training_split(
            source_tl / f"routing_v2_simple_{split}.jsonl",
            out_tl / f"routing_v2_v16_{split}.jsonl",
        )

    query_summary = _augment_query_status(
        source_mm / "mode_query_status_routing_v2_simple.jsonl",
        out_mm / "mode_query_status_v16.jsonl",
    )
    split_summaries["mode_query_status"] = query_summary
    for sidecar_name in ("routing_v2_simple_vocabs.json", "routing_v2_simple_counts.csv", "categories.json", "dataset_stats.json", "validation_summary.json"):
        source = source_mm / sidecar_name
        if source.exists():
            shutil.copy2(source, out_mm / sidecar_name)
    if (source_mm / "splits").exists():
        shutil.copytree(source_mm / "splits", out_mm / "splits", dirs_exist_ok=True)

    full_modes = split_summaries.get("multimode_full", {}).get("mode_counts", {})
    _write_mode_stats(out_mm / "v16_mode_stats_full.csv", full_modes)
    summary = {
        "state": "completed",
        "phase": "completed",
        "run_tag": args.run_tag,
        "source_run_tag": args.source_run_tag,
        "schema_version": V16_SCHEMA_VERSION,
        "image_route_schema_version": IMAGE_ROUTE_NO_PLACEMENT_SCHEMA_VERSION,
        "mode_count": len(v16_categories()),
        "image_route_no_placement_count": len(image_route_no_placement_catalog_payload()["route_names"]),
        "multimode_output_dir": str(out_mm),
        "training_labels_output_dir": str(out_tl),
        "split_summaries": split_summaries,
        "outputs": {
            "multimode_label_json": str(label_out),
            "training_labels_jsonl": str(out_tl),
            "mode_query_status": str(out_mm / "mode_query_status_v16.jsonl"),
            "v16_mode_catalog": str(out_mm / "v16_mode_catalog.json"),
            "image_route_no_placement_catalog": str(out_mm / "image_route_no_placement_catalog.json"),
        },
        "elapsed_sec": round(time.time() - start, 3),
    }
    _write_json(out_mm / "summary.json", summary)
    _write_json(out_tl / "summary.json", summary)
    _write_json(status_path, summary)
    _write_json(out_tl / "status.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
