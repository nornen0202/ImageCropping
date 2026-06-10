from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


PERSON_CENTER_MODES = {"single_person_center", "group_center"}
PERSON_ROT_MODES = {"single_person_rot", "group_rot"}
OBJECT_MODES = {
    "object_single_center",
    "object_single_rot",
    "object_multi_center",
    "object_multi_rot",
}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
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


def _attrs(ann: Dict[str, Any]) -> Dict[str, Any]:
    value = ann.get("attributes")
    return value if isinstance(value, dict) else {}


def _components(ann: Dict[str, Any]) -> Dict[str, Any]:
    value = _attrs(ann).get("score_components")
    return value if isinstance(value, dict) else {}


def _target_ar(ann: Dict[str, Any]) -> str:
    attrs = _attrs(ann)
    return str(attrs.get("target_ar") or ann.get("target_ar") or "")


def _source_image_maps(payload: Dict[str, Any]) -> Dict[int, str]:
    by_id: Dict[int, str] = {}
    for row in payload.get("images", []):
        if not isinstance(row, dict):
            continue
        image_id = _safe_int(row.get("id"), -1)
        if image_id < 0:
            continue
        by_id[image_id] = str(row.get("source_image_id") or Path(str(row.get("file_name") or "")).stem)
    return by_id


def _source_id(ann: Dict[str, Any], image_by_id: Dict[int, str]) -> str:
    return image_by_id.get(_safe_int(ann.get("image_id"), -1), "")


def _has(comps: Dict[str, Any], key: str) -> bool:
    return _safe_float(comps.get(key), None) is not None


def _thirds_like(center_dev: Optional[float], thirds_dev: Optional[float]) -> bool:
    if center_dev is None or thirds_dev is None:
        return False
    return center_dev >= 0.10 and thirds_dev <= 0.075


def _center_like(center_dev: Optional[float], thirds_dev: Optional[float]) -> bool:
    if center_dev is None or thirds_dev is None:
        return False
    return center_dev <= 0.07 and thirds_dev >= 0.12


def _strict_intent_flags(ann: Dict[str, Any]) -> List[str]:
    mode = str(ann.get("mode_name") or "")
    attrs = _attrs(ann)
    comps = _components(ann)
    flags: List[str] = []
    q_place = _safe_float(comps.get("q_place"), None)
    q_x = _safe_float(comps.get("portrait_q_x"), None)
    center_dev = _safe_float(comps.get("portrait_control_center_deviation"), None)
    thirds_dev = _safe_float(comps.get("portrait_control_thirds_deviation"), None)
    anchor_rx = _safe_float(comps.get("anchor_rx"), None)
    leakage = _safe_float(comps.get("face_body_leakage"), None)
    saliency_overlap = _safe_float(comps.get("saliency_crop_overlap"), None)

    if mode == "single_person_center":
        strict_exception = _safe_int(comps.get("portrait_strict_center_exception"), 0) == 1
        if q_place is not None and q_place < 0.56 and not strict_exception:
            flags.append("single_center_q_place_strict_miss")
        if q_x is not None and q_x < 0.52 and not strict_exception:
            flags.append("single_center_q_x_strict_miss")
        if center_dev is not None and center_dev > 0.125 and not strict_exception:
            flags.append("single_center_far_from_center")
        if _thirds_like(center_dev, thirds_dev):
            flags.append("single_center_rot_like")

    if mode == "single_person_rot":
        if q_place is not None and q_place < 0.54:
            flags.append("single_rot_q_place_strict_miss")
        if thirds_dev is not None and thirds_dev > 0.105:
            flags.append("single_rot_thirds_strict_miss")
        if _center_like(center_dev, thirds_dev):
            flags.append("single_rot_center_like")

    if mode == "group_center":
        if q_place is not None and q_place < 0.50:
            flags.append("group_center_q_place_strict_miss")
        if center_dev is not None and center_dev > 0.16:
            flags.append("group_center_far_from_center")

    if mode == "group_rot":
        if q_place is not None and q_place < 0.50:
            flags.append("group_rot_q_place_strict_miss")
        if thirds_dev is not None and thirds_dev > 0.105:
            flags.append("group_rot_thirds_strict_miss")
        if _center_like(center_dev, thirds_dev):
            flags.append("group_rot_center_like")

    if mode == "face":
        if q_place is not None and q_place < 0.55:
            flags.append("face_q_place_strict_miss")
        if anchor_rx is not None and abs(anchor_rx - 0.5) > 0.22:
            flags.append("face_anchor_x_far")
        if leakage is not None and leakage > 0.25:
            flags.append("face_body_leakage_high")

    if mode == "landscape" and _safe_int(comps.get("landscape_subject_safe_active"), 0) == 1:
        min_core = _safe_float(comps.get("landscape_min_core_recall"), None)
        min_q_subj = _safe_float(comps.get("landscape_min_q_subj"), None)
        core_recall = _safe_float(comps.get("core_recall"), None)
        q_subj = _safe_float(comps.get("q_subj"), None)
        if min_core is not None and core_recall is not None and core_recall < min_core:
            flags.append("landscape_subject_core_recall_miss")
        if min_q_subj is not None and q_subj is not None and q_subj < min_q_subj:
            flags.append("landscape_subject_q_subj_miss")

    if mode in OBJECT_MODES:
        if saliency_overlap is not None and saliency_overlap < 0.45:
            flags.append("object_saliency_overlap_strict_miss")
        if mode.endswith("_center") and q_place is not None and q_place < 0.50:
            flags.append("object_center_q_place_miss")
        if mode.endswith("_rot") and _center_like(center_dev, thirds_dev):
            flags.append("object_rot_center_like")

    required_metric_keys = []
    if mode == "single_person_center":
        required_metric_keys = ["q_place", "portrait_q_x", "portrait_control_center_deviation"]
    elif mode == "group_center":
        required_metric_keys = ["q_place", "portrait_control_center_deviation"]
    elif mode in PERSON_ROT_MODES:
        required_metric_keys = ["q_place", "portrait_control_center_deviation", "portrait_control_thirds_deviation"]
    elif mode == "face":
        required_metric_keys = ["q_place", "anchor_rx", "face_body_leakage"]
    elif mode == "landscape":
        required_metric_keys = ["q_teach", "q_subj", "core_recall"]
    elif mode in OBJECT_MODES:
        required_metric_keys = ["q_place", "saliency_crop_overlap"]
    for key in required_metric_keys:
        if not _has(comps, key):
            flags.append(f"missing_metric_{key}")
    return flags


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    fieldnames = [
        "source_image_id",
        "mode_name",
        "target_ar",
        "entity_id",
        "score_mode",
        "bbox",
        "candidate_source",
        "intent_flag",
        "q_place",
        "q_subj",
        "q_teach",
        "core_recall",
        "portrait_q_x",
        "portrait_control_center_deviation",
        "portrait_control_thirds_deviation",
        "anchor_rx",
        "face_body_leakage",
        "saliency_crop_overlap",
    ]
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
            count += 1
    return count


def _markdown_counts(title: str, counts: Dict[str, int], image_counts: Dict[str, int]) -> str:
    lines = [f"## {title}", "", "| flag | annotations | images |", "|---|---:|---:|"]
    if not counts:
        lines.append("| none | 0 | 0 |")
    else:
        for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"| {key} | {value:,} | {image_counts.get(key, 0):,} |")
    return "\n".join(lines)


def _write_report(path: Path, summary: Dict[str, Any]) -> None:
    mode_lines = ["| mode | positives |", "|---|---:|"]
    for mode, count in sorted(summary["positive_count_by_mode"].items()):
        mode_lines.append(f"| {mode} | {int(count):,} |")
    text = "\n".join(
        [
            "# Multimode Intent Alignment Audit",
            "",
            f"- label: `{summary['label_json']}`",
            f"- source images: {summary['source_image_count']:,}",
            f"- positive annotations audited: {summary['positive_annotation_count']:,}",
            f"- suspicious rows: {summary['suspicious_annotation_count']:,}",
            f"- suspicious csv: `{summary['output_paths']['suspicious_csv']}`",
            "",
            "## Positive Count",
            "",
            "\n".join(mode_lines),
            "",
            _markdown_counts(
                "Strict Intent Flags",
                summary["intent_flag_annotation_counts"],
                summary["intent_flag_image_counts"],
            ),
            "",
        ]
    )
    path.write_text(text, encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit multimode positives for strict mode-intent alignment.")
    parser.add_argument("--label_json", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--prefix", default="intent")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    label_json = Path(args.label_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = _load_json(label_json)
    image_by_id = _source_image_maps(payload)

    positive_counts: Counter[str] = Counter()
    flag_counts: Counter[str] = Counter()
    flag_image_sets: Dict[str, set[str]] = defaultdict(set)
    rows: List[Dict[str, Any]] = []
    positive_total = 0
    for ann in payload.get("annotations", []):
        if not isinstance(ann, dict) or _safe_int(ann.get("gt_flag"), 0) != 1:
            continue
        positive_total += 1
        mode = str(ann.get("mode_name") or "")
        positive_counts[mode] += 1
        source_id = _source_id(ann, image_by_id)
        attrs = _attrs(ann)
        comps = _components(ann)
        for flag in _strict_intent_flags(ann):
            flag_counts[flag] += 1
            flag_image_sets[flag].add(source_id)
            rows.append(
                {
                    "source_image_id": source_id,
                    "mode_name": mode,
                    "target_ar": _target_ar(ann),
                    "entity_id": ann.get("entity_id"),
                    "score_mode": ann.get("score_mode"),
                    "bbox": json.dumps(ann.get("bbox"), ensure_ascii=False),
                    "candidate_source": attrs.get("candidate_source"),
                    "intent_flag": flag,
                    "q_place": comps.get("q_place"),
                    "q_subj": comps.get("q_subj"),
                    "q_teach": comps.get("q_teach"),
                    "core_recall": comps.get("core_recall"),
                    "portrait_q_x": comps.get("portrait_q_x"),
                    "portrait_control_center_deviation": comps.get("portrait_control_center_deviation"),
                    "portrait_control_thirds_deviation": comps.get("portrait_control_thirds_deviation"),
                    "anchor_rx": comps.get("anchor_rx"),
                    "face_body_leakage": comps.get("face_body_leakage"),
                    "saliency_crop_overlap": comps.get("saliency_crop_overlap"),
                }
            )

    summary_path = out_dir / f"{args.prefix}_intent_alignment_summary.json"
    csv_path = out_dir / f"{args.prefix}_intent_alignment_suspicious.csv"
    report_path = out_dir / f"{args.prefix}_intent_alignment_report_ko.md"
    summary = {
        "label_json": str(label_json),
        "source_image_count": len(image_by_id),
        "positive_annotation_count": positive_total,
        "positive_count_by_mode": dict(sorted(positive_counts.items())),
        "suspicious_annotation_count": len(rows),
        "intent_flag_annotation_counts": dict(sorted(flag_counts.items())),
        "intent_flag_image_counts": {key: len(value) for key, value in sorted(flag_image_sets.items())},
        "output_paths": {
            "summary_json": str(summary_path),
            "suspicious_csv": str(csv_path),
            "report_md": str(report_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(csv_path, rows)
    _write_report(report_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
