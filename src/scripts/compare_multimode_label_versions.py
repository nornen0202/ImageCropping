from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


NON_LANDSCAPE_MODES = {
    "single_person_center",
    "single_person_rot",
    "group_center",
    "group_rot",
    "face",
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


def _source_id(image_row: Dict[str, Any]) -> str:
    return str(image_row.get("source_image_id") or Path(str(image_row.get("file_name") or "")).stem)


def _bbox_xywh_to_xyxy(bbox: Sequence[Any]) -> Tuple[float, float, float, float]:
    x, y, w, h = [_safe_float(value) for value in list(bbox)[:4]]
    return x, y, x + w, y + h


def _iou(a: Sequence[Any], b: Sequence[Any]) -> float:
    ax1, ay1, ax2, ay2 = _bbox_xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = _bbox_xywh_to_xyxy(b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def _ann_target_ar(ann: Dict[str, Any]) -> str:
    attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
    return str(attrs.get("target_ar") or "")


def _index_labels(path: Path) -> Dict[str, Any]:
    payload = _load_json(path)
    category_names = {
        int(row.get("id", -1)): str(row.get("name") or row.get("mode_name") or row.get("id"))
        for row in payload.get("categories", [])
        if isinstance(row, dict)
    }
    images = {
        int(row.get("id", -1)): row
        for row in payload.get("images", [])
        if isinstance(row, dict)
    }
    source_ids = {_source_id(row) for row in images.values()}
    anns_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ann in payload.get("annotations", []):
        if not isinstance(ann, dict) or int(ann.get("gt_flag", 0)) != 1:
            continue
        image_row = images.get(int(ann.get("image_id", -1)))
        if not image_row:
            continue
        source_id = _source_id(image_row)
        mode_name = str(ann.get("mode_name") or category_names.get(int(ann.get("category_id", -1)), "unknown"))
        item = dict(ann)
        item["mode_name"] = mode_name
        anns_by_source[source_id].append(item)
    return {
        "payload": payload,
        "source_ids": source_ids,
        "anns_by_source": anns_by_source,
        "category_names": category_names,
    }


def _mode_counts(anns: Iterable[Dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for ann in anns:
        counts[str(ann.get("mode_name") or "unknown")] += 1
    return counts


def _target_ar_counts(anns: Iterable[Dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for ann in anns:
        target_ar = _ann_target_ar(ann)
        if target_ar:
            counts[target_ar] += 1
    return counts


def _mode_target_ar_counts(anns: Iterable[Dict[str, Any]]) -> Counter[Tuple[str, str]]:
    counts: Counter[Tuple[str, str]] = Counter()
    for ann in anns:
        target_ar = _ann_target_ar(ann)
        if target_ar:
            counts[(str(ann.get("mode_name") or "unknown"), target_ar)] += 1
    return counts


def _landscape_by_ar(anns: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for ann in anns:
        if str(ann.get("mode_name") or "") != "landscape":
            continue
        target_ar = _ann_target_ar(ann)
        prev = out.get(target_ar)
        if prev is None or _safe_float(ann.get("score_mode"), 0.0) > _safe_float(prev.get("score_mode"), 0.0):
            out[target_ar] = ann
    return out


def _new_gaic_landscape_count(anns: Iterable[Dict[str, Any]]) -> int:
    count = 0
    for ann in anns:
        if str(ann.get("mode_name") or "") != "landscape":
            continue
        attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
        tier = str(attrs.get("landscape_candidate_selected_tier") or "")
        comps = attrs.get("score_components") if isinstance(attrs.get("score_components"), dict) else {}
        if "gaic" in tier or comps.get("landscape_teacher_score_scope") == "gaic":
            count += 1
    return count


def _read_ids(path: Optional[Path]) -> List[str]:
    if path is None or not path.is_file():
        return []
    out: List[str] = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#") or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _read_mode_stats(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if path is None or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {str(row.get("mode_name") or row.get("mode") or ""): row for row in csv.DictReader(handle)}


def _mode_delta_rows(
    *,
    old_stats: Dict[str, Dict[str, str]],
    new_stats: Dict[str, Dict[str, str]],
    old_counts: Counter[str],
    new_counts: Counter[str],
) -> List[Dict[str, Any]]:
    modes = sorted((set(old_counts) | set(new_counts) | set(old_stats) | set(new_stats)) - {""})
    rows: List[Dict[str, Any]] = []
    for mode in modes:
        old_row = old_stats.get(mode, {})
        new_row = new_stats.get(mode, {})
        old_positive = int(old_counts.get(mode, 0))
        new_positive = int(new_counts.get(mode, 0))
        rows.append(
            {
                "mode_name": mode,
                "old_query_count": old_row.get("query_count", ""),
                "new_query_count": new_row.get("query_count", ""),
                "query_delta": _safe_int(new_row.get("query_count")) - _safe_int(old_row.get("query_count")),
                "old_positive": old_positive,
                "new_positive": new_positive,
                "positive_delta": new_positive - old_positive,
                "old_negative": old_row.get("negative_annotation_count", ""),
                "new_negative": new_row.get("negative_annotation_count", ""),
                "negative_delta": _safe_int(new_row.get("negative_annotation_count")) - _safe_int(old_row.get("negative_annotation_count")),
                "old_positive_image_count": old_row.get("positive_image_count", ""),
                "new_positive_image_count": new_row.get("positive_image_count", ""),
            }
        )
    rows.sort(key=lambda row: (row["mode_name"] != "landscape", row["mode_name"]))
    return rows


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _dump_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def compare_versions(
    *,
    old_label_json: Path,
    new_label_json: Path,
    out_dir: Path,
    old_name: str,
    new_name: str,
    old_mode_stats: Optional[Path],
    new_mode_stats: Optional[Path],
    must_include_ids: Optional[Path],
    visual_ids: Optional[Path],
    selected_limit: int,
    landscape_iou_threshold: float,
) -> Dict[str, Any]:
    old = _index_labels(old_label_json)
    new = _index_labels(new_label_json)
    must_include = _read_ids(must_include_ids)
    all_ids = sorted(set(old["source_ids"]) | set(new["source_ids"]) | set(old["anns_by_source"]) | set(new["anns_by_source"]) | set(must_include))

    old_global_modes: Counter[str] = Counter()
    new_global_modes: Counter[str] = Counter()
    old_global_targets: Counter[str] = Counter()
    new_global_targets: Counter[str] = Counter()
    old_mode_targets: Counter[Tuple[str, str]] = Counter()
    new_mode_targets: Counter[Tuple[str, str]] = Counter()
    image_rows: List[Dict[str, Any]] = []

    for source_id in all_ids:
        old_anns = list(old["anns_by_source"].get(source_id, []))
        new_anns = list(new["anns_by_source"].get(source_id, []))
        old_counts = _mode_counts(old_anns)
        new_counts = _mode_counts(new_anns)
        old_global_modes.update(old_counts)
        new_global_modes.update(new_counts)
        old_targets = _target_ar_counts(old_anns)
        new_targets = _target_ar_counts(new_anns)
        old_global_targets.update(old_targets)
        new_global_targets.update(new_targets)
        old_mode_targets.update(_mode_target_ar_counts(old_anns))
        new_mode_targets.update(_mode_target_ar_counts(new_anns))

        old_non_landscape = sum(old_counts.get(mode, 0) for mode in NON_LANDSCAPE_MODES)
        new_non_landscape = sum(new_counts.get(mode, 0) for mode in NON_LANDSCAPE_MODES)
        removed_by_mode = {
            mode: max(0, old_counts.get(mode, 0) - new_counts.get(mode, 0))
            for mode in sorted(NON_LANDSCAPE_MODES)
        }
        added_by_mode = {
            mode: max(0, new_counts.get(mode, 0) - old_counts.get(mode, 0))
            for mode in sorted(NON_LANDSCAPE_MODES)
        }
        old_landscape = _landscape_by_ar(old_anns)
        new_landscape = _landscape_by_ar(new_anns)
        changed_landscape = 0
        iou_values: List[float] = []
        for target_ar, old_ann in old_landscape.items():
            new_ann = new_landscape.get(target_ar)
            if not new_ann:
                continue
            iou = _iou(old_ann.get("bbox") or [0, 0, 0, 0], new_ann.get("bbox") or [0, 0, 0, 0])
            if iou < landscape_iou_threshold:
                changed_landscape += 1
                iou_values.append(iou)
        landscape_removed = len(set(old_landscape) - set(new_landscape))
        landscape_added = len(set(new_landscape) - set(old_landscape))
        new_gaic_landscape = _new_gaic_landscape_count(new_anns)
        removed_non_landscape = max(0, old_non_landscape - new_non_landscape)
        added_non_landscape = max(0, new_non_landscape - old_non_landscape)
        total_delta_abs = abs(len(old_anns) - len(new_anns))
        score = (
            4.0 * removed_non_landscape
            + 1.5 * changed_landscape
            + 1.0 * landscape_removed
            + 0.2 * total_delta_abs
            + 0.15 * new_gaic_landscape
            + 0.75 * sum(removed_by_mode.values())
        )
        if source_id in must_include:
            score += 1000.0
        image_rows.append(
            {
                "source_image_id": source_id,
                "selection_score": round(score, 4),
                "must_include": int(source_id in must_include),
                "old_positive_total": len(old_anns),
                "new_positive_total": len(new_anns),
                "positive_delta": len(new_anns) - len(old_anns),
                "old_non_landscape_positive": int(old_non_landscape),
                "new_non_landscape_positive": int(new_non_landscape),
                "removed_non_landscape_positive": int(removed_non_landscape),
                "added_non_landscape_positive": int(added_non_landscape),
                "changed_landscape_ar_count": int(changed_landscape),
                "landscape_removed_ar_count": int(landscape_removed),
                "landscape_added_ar_count": int(landscape_added),
                "changed_landscape_iou_min": "" if not iou_values else round(min(iou_values), 4),
                "new_gaic_landscape_positive": int(new_gaic_landscape),
                "removed_face": removed_by_mode.get("face", 0),
                "removed_single_person_center": removed_by_mode.get("single_person_center", 0),
                "removed_single_person_rot": removed_by_mode.get("single_person_rot", 0),
                "removed_group_center": removed_by_mode.get("group_center", 0),
                "removed_group_rot": removed_by_mode.get("group_rot", 0),
                "removed_object_single_center": removed_by_mode.get("object_single_center", 0),
                "removed_object_single_rot": removed_by_mode.get("object_single_rot", 0),
                "removed_object_multi_center": removed_by_mode.get("object_multi_center", 0),
                "removed_object_multi_rot": removed_by_mode.get("object_multi_rot", 0),
                "old_mode_counts_json": json.dumps(dict(old_counts), ensure_ascii=False, sort_keys=True),
                "new_mode_counts_json": json.dumps(dict(new_counts), ensure_ascii=False, sort_keys=True),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    image_rows_sorted = sorted(image_rows, key=lambda row: (-float(row["selection_score"]), row["source_image_id"]))
    rows_by_id = {str(row["source_image_id"]): row for row in image_rows}
    visual_id_values = _read_ids(visual_ids)
    if visual_id_values:
        selected_rows = [rows_by_id[source_id] for source_id in visual_id_values if source_id in rows_by_id]
    else:
        selected_rows = image_rows_sorted if selected_limit <= 0 else image_rows_sorted[:selected_limit]
    selected_ids_path = out_dir / f"{old_name}_{new_name}_selected_visual_ids.txt"
    selected_ids_path.write_text("\n".join(str(row["source_image_id"]) for row in selected_rows) + "\n", encoding="utf-8")

    image_delta_csv = out_dir / f"{old_name}_{new_name}_image_delta_full.csv"
    image_fields = [
        "source_image_id",
        "selection_score",
        "must_include",
        "old_positive_total",
        "new_positive_total",
        "positive_delta",
        "old_non_landscape_positive",
        "new_non_landscape_positive",
        "removed_non_landscape_positive",
        "added_non_landscape_positive",
        "changed_landscape_ar_count",
        "landscape_removed_ar_count",
        "landscape_added_ar_count",
        "changed_landscape_iou_min",
        "new_gaic_landscape_positive",
        "removed_face",
        "removed_single_person_center",
        "removed_single_person_rot",
        "removed_group_center",
        "removed_group_rot",
        "removed_object_single_center",
        "removed_object_single_rot",
        "removed_object_multi_center",
        "removed_object_multi_rot",
        "old_mode_counts_json",
        "new_mode_counts_json",
    ]
    _dump_csv(image_delta_csv, image_rows_sorted, image_fields)

    selected_csv = out_dir / f"{old_name}_{new_name}_selected_visual_samples.csv"
    selected_json = out_dir / f"{old_name}_{new_name}_selected_visual_samples.json"
    _dump_csv(selected_csv, selected_rows, image_fields)
    selected_json.write_text(json.dumps({"selected_count": len(selected_rows), "rows": selected_rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    mode_rows = _mode_delta_rows(
        old_stats=_read_mode_stats(old_mode_stats),
        new_stats=_read_mode_stats(new_mode_stats),
        old_counts=old_global_modes,
        new_counts=new_global_modes,
    )
    mode_delta_csv = out_dir / f"{old_name}_{new_name}_mode_delta.csv"
    _dump_csv(
        mode_delta_csv,
        mode_rows,
        [
            "mode_name",
            "old_query_count",
            "new_query_count",
            "query_delta",
            "old_positive",
            "new_positive",
            "positive_delta",
            "old_negative",
            "new_negative",
            "negative_delta",
            "old_positive_image_count",
            "new_positive_image_count",
        ],
    )

    target_rows = []
    for target_ar in sorted(set(old_global_targets) | set(new_global_targets)):
        target_rows.append(
            {
                "target_ar": target_ar,
                "old_positive": int(old_global_targets.get(target_ar, 0)),
                "new_positive": int(new_global_targets.get(target_ar, 0)),
                "positive_delta": int(new_global_targets.get(target_ar, 0) - old_global_targets.get(target_ar, 0)),
            }
        )
    target_delta_csv = out_dir / f"{old_name}_{new_name}_target_ar_delta.csv"
    _dump_csv(target_delta_csv, target_rows, ["target_ar", "old_positive", "new_positive", "positive_delta"])

    mode_target_rows = []
    for mode, target_ar in sorted(set(old_mode_targets) | set(new_mode_targets)):
        mode_target_rows.append(
            {
                "mode_name": mode,
                "target_ar": target_ar,
                "old_positive": int(old_mode_targets.get((mode, target_ar), 0)),
                "new_positive": int(new_mode_targets.get((mode, target_ar), 0)),
                "positive_delta": int(new_mode_targets.get((mode, target_ar), 0) - old_mode_targets.get((mode, target_ar), 0)),
            }
        )
    mode_target_delta_csv = out_dir / f"{old_name}_{new_name}_mode_target_ar_delta.csv"
    _dump_csv(mode_target_delta_csv, mode_target_rows, ["mode_name", "target_ar", "old_positive", "new_positive", "positive_delta"])

    old_total = sum(old_global_modes.values())
    new_total = sum(new_global_modes.values())
    summary = {
        "old_name": old_name,
        "new_name": new_name,
        "old_label_json": str(old_label_json),
        "new_label_json": str(new_label_json),
        "old_source_image_count": len(old["source_ids"]),
        "new_source_image_count": len(new["source_ids"]),
        "shared_source_image_count": len(set(old["source_ids"]) & set(new["source_ids"])),
        "old_positive_annotation_count": int(old_total),
        "new_positive_annotation_count": int(new_total),
        "positive_annotation_delta": int(new_total - old_total),
        "old_non_landscape_positive": int(sum(old_global_modes.get(mode, 0) for mode in NON_LANDSCAPE_MODES)),
        "new_non_landscape_positive": int(sum(new_global_modes.get(mode, 0) for mode in NON_LANDSCAPE_MODES)),
        "old_landscape_positive": int(old_global_modes.get("landscape", 0)),
        "new_landscape_positive": int(new_global_modes.get("landscape", 0)),
        "image_delta_csv": str(image_delta_csv),
        "mode_delta_csv": str(mode_delta_csv),
        "target_ar_delta_csv": str(target_delta_csv),
        "mode_target_ar_delta_csv": str(mode_target_delta_csv),
        "selected_visual_ids": str(selected_ids_path),
        "selected_visual_samples_json": str(selected_json),
        "selected_visual_samples_csv": str(selected_csv),
        "selected_visual_count": len(selected_rows),
        "selected_visual_source": str(visual_ids) if visual_id_values else "selection_score",
        "landscape_iou_threshold": landscape_iou_threshold,
        "top_selected_examples": selected_rows[:20],
    }
    summary_json = out_dir / f"{old_name}_{new_name}_full_delta_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two multimode label versions and select visualization samples.")
    parser.add_argument("--old_label_json", required=True)
    parser.add_argument("--new_label_json", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--old_name", default="old")
    parser.add_argument("--new_name", default="new")
    parser.add_argument("--old_mode_stats", default="")
    parser.add_argument("--new_mode_stats", default="")
    parser.add_argument("--must_include_ids", default="")
    parser.add_argument("--visual_ids", default="", help="Optional fixed source-image-id list for visualization rows.")
    parser.add_argument("--selected_limit", type=int, default=500)
    parser.add_argument("--landscape_iou_threshold", type=float, default=0.78)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = compare_versions(
        old_label_json=Path(args.old_label_json),
        new_label_json=Path(args.new_label_json),
        out_dir=Path(args.out_dir),
        old_name=str(args.old_name),
        new_name=str(args.new_name),
        old_mode_stats=Path(args.old_mode_stats) if args.old_mode_stats else None,
        new_mode_stats=Path(args.new_mode_stats) if args.new_mode_stats else None,
        must_include_ids=Path(args.must_include_ids) if args.must_include_ids else None,
        visual_ids=Path(args.visual_ids) if args.visual_ids else None,
        selected_limit=int(args.selected_limit),
        landscape_iou_threshold=float(args.landscape_iou_threshold),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
