#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from label_artifacts.paths import LABEL_JSON_DIRNAME, MULTIMODE_LABEL_FILENAMES
from multimode.explainability import CHECKLIST_CLASS_KEYS, SCHEMA_VERSION


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _label_path(run_dir: Path, split: str, target_ar_only: bool) -> Path:
    return run_dir / LABEL_JSON_DIRNAME / MULTIMODE_LABEL_FILENAMES[(split, target_ar_only)]


def _regular_split_summary(dataset: Mapping[str, Any]) -> dict[str, Any]:
    images = dataset.get("images") if isinstance(dataset.get("images"), list) else []
    annotations = dataset.get("annotations") if isinstance(dataset.get("annotations"), list) else []
    mode_counts: Counter[str] = Counter()
    ar_counts: Counter[str] = Counter()
    label_counts: dict[str, Counter[str]] = {key: Counter() for key in CHECKLIST_CLASS_KEYS}
    why_tag_counts: Counter[str] = Counter()
    missing_schema = 0
    missing_labels = 0
    missing_scores = 0
    missing_applicable = 0
    missing_teacher_aliases = 0
    missing_score_components = 0
    missing_final_score = 0
    final_score_mismatch = 0
    unexpected_missing_label_keys = 0
    unexpected_missing_applicable_keys = 0
    for annotation in annotations:
        if not isinstance(annotation, Mapping):
            continue
        attrs = annotation.get("attributes") if isinstance(annotation.get("attributes"), Mapping) else {}
        mode_counts[str(annotation.get("mode_name") or attrs.get("mode_name") or "")] += 1
        ar_counts[str(attrs.get("target_ar") or annotation.get("target_ar") or "FREE")] += 1
        if attrs.get("explainability_schema_version") != SCHEMA_VERSION:
            missing_schema += 1
        labels = attrs.get("checklist_labels") if isinstance(attrs.get("checklist_labels"), Mapping) else None
        scores = attrs.get("checklist_scores") if isinstance(attrs.get("checklist_scores"), Mapping) else None
        applicable = attrs.get("checklist_applicable") if isinstance(attrs.get("checklist_applicable"), Mapping) else None
        if labels is None:
            missing_labels += 1
        else:
            for key in CHECKLIST_CLASS_KEYS:
                if key not in labels:
                    unexpected_missing_label_keys += 1
                label_counts[key][str(labels.get(key, "na") or "na")] += 1
        if scores is None:
            missing_scores += 1
        else:
            if "final_score" not in scores:
                missing_final_score += 1
            else:
                try:
                    if abs(float(scores["final_score"]) - float(annotation.get("score_mode", 0.0))) > 1e-6:
                        final_score_mismatch += 1
                except (TypeError, ValueError):
                    final_score_mismatch += 1
        if applicable is None:
            missing_applicable += 1
        else:
            for key in CHECKLIST_CLASS_KEYS:
                if key not in applicable:
                    unexpected_missing_applicable_keys += 1
        if not isinstance(attrs.get("score_components"), Mapping):
            missing_score_components += 1
        if (
            not isinstance(attrs.get("teacher_checklist_labels"), Mapping)
            or not isinstance(attrs.get("teacher_checklist_scores"), Mapping)
            or not isinstance(attrs.get("teacher_why_tags"), list)
        ):
            missing_teacher_aliases += 1
        for tag in attrs.get("why_tags") or []:
            why_tag_counts[str(tag)] += 1
    annotation_count = len(annotations)
    passed = (
        missing_schema == 0
        and missing_labels == 0
        and missing_scores == 0
        and missing_applicable == 0
        and missing_teacher_aliases == 0
        and missing_final_score == 0
        and final_score_mismatch == 0
        and unexpected_missing_label_keys == 0
        and unexpected_missing_applicable_keys == 0
    )
    return {
        "passed": passed,
        "image_count": len(images),
        "annotation_count": annotation_count,
        "schema_version": SCHEMA_VERSION,
        "schema_coverage_rate": round((annotation_count - missing_schema) / float(annotation_count), 6) if annotation_count else 0.0,
        "missing_schema_count": missing_schema,
        "missing_checklist_labels_count": missing_labels,
        "missing_checklist_scores_count": missing_scores,
        "missing_checklist_applicable_count": missing_applicable,
        "missing_teacher_aliases_count": missing_teacher_aliases,
        "missing_final_score_count": missing_final_score,
        "final_score_mismatch_count": final_score_mismatch,
        "missing_score_components_count": missing_score_components,
        "unexpected_missing_label_key_count": unexpected_missing_label_keys,
        "unexpected_missing_applicable_key_count": unexpected_missing_applicable_keys,
        "mode_counts": dict(sorted(mode_counts.items())),
        "target_ar_counts": dict(sorted(ar_counts.items())),
        "checklist_label_counts": {key: dict(counter) for key, counter in label_counts.items()},
        "why_tag_counts": dict(sorted(why_tag_counts.items())),
    }


def _target_ar_only_split_summary(dataset: Mapping[str, Any]) -> dict[str, Any]:
    images = dataset.get("images") if isinstance(dataset.get("images"), list) else []
    annotations = dataset.get("annotations") if isinstance(dataset.get("annotations"), list) else []
    invalid_attributes = 0
    missing_target_ar = 0
    target_ar_counts: Counter[str] = Counter()
    for annotation in annotations:
        if not isinstance(annotation, Mapping):
            continue
        attrs = annotation.get("attributes") if isinstance(annotation.get("attributes"), Mapping) else {}
        if set(attrs.keys()) != {"target_ar"}:
            invalid_attributes += 1
        target_ar = attrs.get("target_ar")
        if not target_ar:
            missing_target_ar += 1
        target_ar_counts[str(target_ar or "")] += 1
    return {
        "passed": invalid_attributes == 0 and missing_target_ar == 0,
        "image_count": len(images),
        "annotation_count": len(annotations),
        "invalid_attributes_count": invalid_attributes,
        "missing_target_ar_count": missing_target_ar,
        "target_ar_counts": dict(sorted(target_ar_counts.items())),
    }


def validate_run(run_dir: Path, splits: Sequence[str]) -> dict[str, Any]:
    run_dir = Path(run_dir)
    summary: dict[str, Any] = {
        "schema_version": "multimode_explainability_validation_v1",
        "run_dir": str(run_dir),
        "validated_at": _utc_timestamp(),
        "splits": {},
    }
    passed = True
    for split in splits:
        label_path = _label_path(run_dir, split, False)
        target_path = _label_path(run_dir, split, True)
        regular = _regular_split_summary(_read_json(label_path))
        target = _target_ar_only_split_summary(_read_json(target_path))
        count_match = regular["image_count"] == target["image_count"] and regular["annotation_count"] == target["annotation_count"]
        split_passed = bool(regular["passed"] and target["passed"] and count_match)
        passed = bool(passed and split_passed)
        summary["splits"][split] = {
            "passed": split_passed,
            "regular_label_json": str(label_path),
            "target_ar_only_label_json": str(target_path),
            "regular": regular,
            "target_ar_only": target,
            "regular_target_ar_only_count_match": count_match,
        }
    summary["passed"] = passed
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate multimode explainability checklist labels in existing label JSONs.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--splits", default="full,train,val")
    parser.add_argument("--output_json", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    splits = [chunk.strip() for chunk in str(args.splits).split(",") if chunk.strip()]
    summary = validate_run(run_dir, splits)
    out_path = Path(args.output_json) if args.output_json else run_dir / "explainability_validation_summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": summary["passed"], "output_json": str(out_path)}, ensure_ascii=False, indent=2))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
