from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PERSON_MODES = {"single_person_center", "single_person_rot", "face"}
OBJECT_MODES = {
    "object_single_center",
    "object_single_rot",
    "object_multi_center",
    "object_multi_rot",
}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _attr(ann: Dict[str, Any]) -> Dict[str, Any]:
    value = ann.get("attributes")
    return value if isinstance(value, dict) else {}


def _components(ann: Dict[str, Any]) -> Dict[str, Any]:
    value = _attr(ann).get("score_components")
    return value if isinstance(value, dict) else {}


def _target_ar(ann: Dict[str, Any]) -> str:
    attrs = _attr(ann)
    return str(attrs.get("target_ar") or ann.get("target_ar") or "")


def _candidate_source(ann: Dict[str, Any]) -> str:
    return str(_attr(ann).get("candidate_source") or "")


def _source_maps(label_payload: Dict[str, Any]) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, int]]:
    by_id: Dict[int, Dict[str, Any]] = {}
    source_to_id: Dict[str, int] = {}
    for row in label_payload.get("images", []):
        if not isinstance(row, dict):
            continue
        image_id = _safe_int(row.get("id"), -1)
        if image_id < 0:
            continue
        by_id[image_id] = row
        source_id = str(row.get("source_image_id") or Path(str(row.get("file_name") or "")).stem)
        if source_id:
            source_to_id[source_id] = image_id
    return by_id, source_to_id


def _source_id_for_ann(ann: Dict[str, Any], image_by_id: Dict[int, Dict[str, Any]]) -> str:
    image = image_by_id.get(_safe_int(ann.get("image_id"), -1), {})
    return str(image.get("source_image_id") or Path(str(image.get("file_name") or "")).stem)


def _file_name_for_ann(ann: Dict[str, Any], image_by_id: Dict[int, Dict[str, Any]]) -> str:
    image = image_by_id.get(_safe_int(ann.get("image_id"), -1), {})
    return str(image.get("file_name") or "")


def _risk_signals(ann: Dict[str, Any]) -> List[str]:
    mode = str(ann.get("mode_name") or "")
    attrs = _attr(ann)
    comps = _components(ann)
    signals: List[str] = []
    source_side_relaxed = _safe_int(comps.get("portrait_source_side_margin_relaxed"), 0) == 1

    hard_rejects = attrs.get("hard_reject_reasons")
    if isinstance(hard_rejects, list) and hard_rejects:
        signals.append("positive_with_hard_reject_reason")

    if mode in PERSON_MODES:
        if _safe_int(attrs.get("person_atom_trusted"), 1) != 1:
            signals.append("untrusted_person_atom_positive")
        if _safe_int(attrs.get("person_rank"), 0) >= 1:
            if _safe_int(attrs.get("secondary_person_exception"), 0) == 1:
                pass
            else:
                signals.append("secondary_person_positive")
        if mode == "face":
            if _safe_int(attrs.get("face_query_valid"), 1) != 1:
                signals.append("face_query_invalid_positive")
            if _safe_float(attrs.get("face_score"), 1.0) < 0.55:
                signals.append("face_low_score_lt_0_55")

    if mode == "single_person_center":
        if _safe_float(comps.get("q_place"), 1.0) < 0.42:
            signals.append("center_q_place_lt_0_42")
        if comps.get("portrait_q_x") is not None and _safe_float(comps.get("portrait_q_x"), 1.0) < 0.35:
            signals.append("center_q_x_lt_0_35")
        if (
            comps.get("portrait_control_center_deviation") is not None
            and _safe_float(comps.get("portrait_control_center_deviation"), 0.0) > 0.20
        ):
            signals.append("center_control_dev_gt_0_20")
        if (
            min(_safe_float(comps.get("subject_margin_left"), 1.0), _safe_float(comps.get("subject_margin_right"), 1.0)) < 0.025
            and not source_side_relaxed
        ):
            signals.append("center_side_margin_tight")
        if _safe_float(comps.get("subject_outside"), 0.0) > 0.01:
            signals.append("center_subject_outside")

    if mode == "single_person_rot":
        if _safe_float(comps.get("q_place"), 1.0) < 0.45:
            signals.append("rot_q_place_lt_0_45")
        if comps.get("portrait_q_x") is not None and _safe_float(comps.get("portrait_q_x"), 1.0) < 0.35:
            signals.append("rot_q_x_lt_0_35")
        if (
            comps.get("portrait_control_thirds_deviation") is not None
            and _safe_float(comps.get("portrait_control_thirds_deviation"), 0.0) > 0.16
        ):
            signals.append("rot_control_thirds_dev_gt_0_16")
        if (
            min(_safe_float(comps.get("subject_margin_left"), 1.0), _safe_float(comps.get("subject_margin_right"), 1.0)) < 0.035
            and not source_side_relaxed
        ):
            signals.append("rot_side_margin_tight")
        if _safe_float(comps.get("subject_outside"), 0.0) > 0.01:
            signals.append("rot_subject_outside")

    if mode == "group_center":
        if _safe_float(comps.get("q_place"), 1.0) < 0.35:
            signals.append("group_center_q_place_lt_0_35")
        if (
            min(_safe_float(comps.get("subject_margin_left"), 1.0), _safe_float(comps.get("subject_margin_right"), 1.0)) < 0.025
            and not source_side_relaxed
        ):
            signals.append("group_center_side_margin_tight")
        if _safe_float(comps.get("subject_outside"), 0.0) > 0.01:
            signals.append("group_center_subject_outside")

    if mode == "group_rot":
        if _safe_float(comps.get("q_place"), 1.0) < 0.40:
            signals.append("group_rot_q_place_lt_0_40")
        if (
            comps.get("portrait_control_thirds_deviation") is not None
            and _safe_float(comps.get("portrait_control_thirds_deviation"), 0.0) > 0.16
        ):
            signals.append("group_rot_control_thirds_dev_gt_0_16")
        if (
            min(_safe_float(comps.get("subject_margin_left"), 1.0), _safe_float(comps.get("subject_margin_right"), 1.0)) < 0.025
            and not source_side_relaxed
        ):
            signals.append("group_rot_side_margin_tight")
        if _safe_float(comps.get("subject_outside"), 0.0) > 0.01:
            signals.append("group_rot_subject_outside")

    if mode == "landscape":
        attrs_entity = str(attrs.get("atom_entity_type") or attrs.get("entity_type") or "")
        if attrs_entity != "scene" and not _candidate_source(ann).startswith("teacher:"):
            signals.append("landscape_non_scene_non_teacher")

    if mode in OBJECT_MODES:
        saliency_overlap = _safe_float(comps.get("saliency_crop_overlap"), 1.0)
        saliency_area_ratio = _safe_float(comps.get("saliency_fg_area_ratio"), 1.0)
        if saliency_area_ratio < 0.85 and saliency_overlap < 0.35:
            signals.append("object_low_saliency_overlap")

    return signals


def _compact_annotation(ann: Dict[str, Any]) -> Dict[str, Any]:
    attrs = _attr(ann)
    return {
        "mode_name": ann.get("mode_name"),
        "target_ar": _target_ar(ann),
        "entity_id": ann.get("entity_id"),
        "score_mode": ann.get("score_mode"),
        "bbox": ann.get("bbox"),
        "candidate_source": attrs.get("candidate_source"),
        "person_rank": attrs.get("person_rank"),
        "person_atom_trusted": attrs.get("person_atom_trusted"),
        "face_score": attrs.get("face_score"),
    }


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    fieldnames = [
        "source_image_id",
        "file_name",
        "mode_name",
        "target_ar",
        "entity_id",
        "score_mode",
        "bbox",
        "candidate_source",
        "source_route_mode",
        "risk_signal",
        "q_place",
        "portrait_q_x",
        "portrait_control_center_deviation",
        "portrait_control_thirds_deviation",
        "subject_margin_left",
        "subject_margin_right",
        "subject_outside",
        "saliency_crop_overlap",
        "saliency_fg_area_ratio",
        "person_rank",
        "person_atom_trusted",
        "face_score",
    ]
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
            count += 1
    return count


def _markdown_table(counter: Dict[str, int], image_counter: Dict[str, int]) -> str:
    if not counter:
        return "| signal | annotations | images |\n|---|---:|---:|\n| none | 0 | 0 |"
    lines = ["| signal | annotations | images |", "|---|---:|---:|"]
    for key, value in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {key} | {value:,} | {image_counter.get(key, 0):,} |")
    return "\n".join(lines)


def _write_report(
    path: Path,
    *,
    label_json: Path,
    summary: Dict[str, Any],
    risk_counts: Dict[str, int],
    risk_image_counts: Dict[str, int],
    cited_examples: Dict[str, List[Dict[str, Any]]],
    csv_path: Path,
) -> None:
    mode_counts = summary.get("positive_count_by_mode") or {}
    mode_lines = ["| mode | positives |", "|---|---:|"]
    for mode, count in sorted(mode_counts.items()):
        mode_lines.append(f"| {mode} | {int(count):,} |")

    cited_lines: List[str] = []
    for source_id, rows in cited_examples.items():
        cited_lines.append(f"### {source_id}")
        if not rows:
            cited_lines.append("- positive annotation 없음")
            continue
        for row in rows[:24]:
            cited_lines.append(f"- `{row.get('mode_name')}` `{row.get('target_ar')}` `{row.get('entity_id')}` score={row.get('score_mode')} source={row.get('candidate_source')} bbox={row.get('bbox')}")

    text = "\n".join(
        [
            "# Multimode Label Quality Audit",
            "",
            f"- label: `{label_json}`",
            f"- source images: {int(summary.get('source_image_count', 0)):,}",
            f"- annotations: {int(summary.get('annotation_count', 0)):,}",
            f"- positive annotations: {int(summary.get('positive_annotation_count', 0)):,}",
            f"- approved secondary exceptions: {int(summary.get('approved_secondary_exception_annotation_count', 0)):,} annotations / {int(summary.get('approved_secondary_exception_image_count', 0)):,} images",
            f"- suspicious csv: `{csv_path}`",
            "",
            "## Mode Positive Count",
            "\n".join(mode_lines),
            "",
            "## Residual Risk Signals",
            _markdown_table(risk_counts, risk_image_counts),
            "",
            "## Cited Examples",
            "\n".join(cited_lines),
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit generated multimode label positives for quality risk signals.")
    parser.add_argument("--label_json", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--summary_json", default="")
    parser.add_argument("--prefix", default="quality")
    parser.add_argument("--focus_image_ids", default="", help="Optional text file with source image ids to summarize separately.")
    parser.add_argument(
        "--cited_image_ids",
        default="bigstock_image_108250919,sstk_image_400750522,pond5_image_84875481",
        help="Comma-separated source image ids to include in the report.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    label_json = Path(args.label_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = _load_json(label_json)
    image_by_id, source_to_id = _source_maps(payload)

    positive_by_mode: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()
    risk_image_sets: Dict[str, set[str]] = defaultdict(set)
    risk_rows: List[Dict[str, Any]] = []
    focus_ids: set[str] = set()
    if args.focus_image_ids:
        focus_ids = {line.strip() for line in Path(args.focus_image_ids).read_text(encoding="utf-8").splitlines() if line.strip()}
    focus_risk_counts: Counter[str] = Counter()
    focus_risk_image_sets: Dict[str, set[str]] = defaultdict(set)
    approved_secondary_count = 0
    approved_secondary_image_set: set[str] = set()

    for ann in payload.get("annotations", []):
        if not isinstance(ann, dict) or _safe_int(ann.get("gt_flag"), 0) != 1:
            continue
        mode = str(ann.get("mode_name") or "")
        positive_by_mode[mode] += 1
        source_id = _source_id_for_ann(ann, image_by_id)
        file_name = _file_name_for_ann(ann, image_by_id)
        attrs = _attr(ann)
        comps = _components(ann)
        if (
            mode in PERSON_MODES
            and _safe_int(attrs.get("person_rank"), 0) >= 1
            and _safe_int(attrs.get("secondary_person_exception"), 0) == 1
        ):
            approved_secondary_count += 1
            approved_secondary_image_set.add(source_id)
        for signal in _risk_signals(ann):
            risk_counts[signal] += 1
            risk_image_sets[signal].add(source_id)
            if source_id in focus_ids:
                focus_risk_counts[signal] += 1
                focus_risk_image_sets[signal].add(source_id)
            risk_rows.append(
                {
                    "source_image_id": source_id,
                    "file_name": file_name,
                    "mode_name": mode,
                    "target_ar": _target_ar(ann),
                    "entity_id": ann.get("entity_id"),
                    "score_mode": ann.get("score_mode"),
                    "bbox": json.dumps(ann.get("bbox"), ensure_ascii=False),
                    "candidate_source": attrs.get("candidate_source"),
                    "source_route_mode": ann.get("source_route_mode"),
                    "risk_signal": signal,
                    "q_place": comps.get("q_place"),
                    "portrait_q_x": comps.get("portrait_q_x"),
                    "portrait_control_center_deviation": comps.get("portrait_control_center_deviation"),
                    "portrait_control_thirds_deviation": comps.get("portrait_control_thirds_deviation"),
                    "subject_margin_left": comps.get("subject_margin_left"),
                    "subject_margin_right": comps.get("subject_margin_right"),
                    "subject_outside": comps.get("subject_outside"),
                    "saliency_crop_overlap": comps.get("saliency_crop_overlap"),
                    "saliency_fg_area_ratio": comps.get("saliency_fg_area_ratio"),
                    "person_rank": attrs.get("person_rank"),
                    "person_atom_trusted": attrs.get("person_atom_trusted"),
                    "face_score": attrs.get("face_score"),
                }
            )

    cited_examples: Dict[str, List[Dict[str, Any]]] = {}
    cited_ids = [item.strip() for item in str(args.cited_image_ids).split(",") if item.strip()]
    cited_image_ids = {source_to_id.get(source_id): source_id for source_id in cited_ids if source_to_id.get(source_id) is not None}
    for source_id in cited_ids:
        cited_examples[source_id] = []
    for ann in payload.get("annotations", []):
        if not isinstance(ann, dict) or _safe_int(ann.get("gt_flag"), 0) != 1:
            continue
        source_id = cited_image_ids.get(_safe_int(ann.get("image_id"), -1))
        if source_id:
            cited_examples.setdefault(source_id, []).append(_compact_annotation(ann))
    for rows in cited_examples.values():
        rows.sort(key=lambda row: (str(row.get("target_ar")), str(row.get("mode_name")), -_safe_float(row.get("score_mode"))))

    summary = {
        "label_json": str(label_json),
        "source_image_count": len(image_by_id),
        "annotation_count": len(payload.get("annotations", [])),
        "positive_annotation_count": sum(positive_by_mode.values()),
        "positive_count_by_mode": dict(sorted(positive_by_mode.items())),
        "risk_annotation_counts": dict(sorted(risk_counts.items())),
        "risk_image_counts": {key: len(value) for key, value in sorted(risk_image_sets.items())},
        "approved_secondary_exception_annotation_count": approved_secondary_count,
        "approved_secondary_exception_image_count": len(approved_secondary_image_set),
        "focus_image_count": len(focus_ids),
        "focus_risk_annotation_counts": dict(sorted(focus_risk_counts.items())),
        "focus_risk_image_counts": {key: len(value) for key, value in sorted(focus_risk_image_sets.items())},
        "cited_examples": cited_examples,
    }

    if args.summary_json:
        summary["generator_summary_json"] = str(Path(args.summary_json))
        generator_summary = _load_json(Path(args.summary_json))
        summary["generator_positive_annotation_count"] = generator_summary.get("positive_annotation_count")
        summary["generator_query_count"] = generator_summary.get("query_count")

    summary_path = out_dir / f"{args.prefix}_quality_audit_summary.json"
    csv_path = out_dir / f"{args.prefix}_suspicious_positive_annotations.csv"
    report_path = out_dir / f"{args.prefix}_quality_audit_report_ko.md"
    summary["output_paths"] = {
        "summary_json": str(summary_path),
        "suspicious_csv": str(csv_path),
        "report_md": str(report_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(csv_path, risk_rows)
    _write_report(
        report_path,
        label_json=label_json,
        summary=summary,
        risk_counts=summary["risk_annotation_counts"],
        risk_image_counts=summary["risk_image_counts"],
        cited_examples=cited_examples,
        csv_path=csv_path,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
