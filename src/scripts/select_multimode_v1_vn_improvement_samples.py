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
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


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


def _index_labels(path: Path) -> Dict[str, Any]:
    payload = _load_json(path)
    images = {
        int(row.get("id", -1)): row
        for row in payload.get("images", [])
        if isinstance(row, dict)
    }
    anns_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ann in payload.get("annotations", []):
        if not isinstance(ann, dict) or int(ann.get("gt_flag", 0)) != 1:
            continue
        image_row = images.get(int(ann.get("image_id", -1)))
        if not image_row:
            continue
        source_id = _source_id(image_row)
        anns_by_source[source_id].append(ann)
    return {"payload": payload, "anns_by_source": anns_by_source}


def _mode_counts(anns: Iterable[Dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for ann in anns:
        counts[str(ann.get("mode_name") or "unknown")] += 1
    return counts


def _landscape_by_ar(anns: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for ann in anns:
        if str(ann.get("mode_name") or "") != "landscape":
            continue
        attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
        target_ar = str(attrs.get("target_ar") or "")
        prev = out.get(target_ar)
        if prev is None or _safe_float(ann.get("score_mode"), 0.0) > _safe_float(prev.get("score_mode"), 0.0):
            out[target_ar] = ann
    return out


def _read_suspicious_csv(path: Optional[Path]) -> Counter[str]:
    counts: Counter[str] = Counter()
    if path is None or not path.is_file():
        return counts
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            source_id = str(row.get("source_image_id") or "").strip()
            if source_id:
                counts[source_id] += 1
    return counts


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


def select_samples(
    *,
    v1_label_json: Path,
    new_label_json: Path,
    v1_suspicious_csv: Optional[Path],
    must_include_ids: Optional[Path],
    limit: int,
    require_v1_positive: bool = False,
) -> List[Dict[str, Any]]:
    v1 = _index_labels(v1_label_json)
    new = _index_labels(new_label_json)
    suspicious = _read_suspicious_csv(v1_suspicious_csv)
    must_include = _read_ids(must_include_ids)
    all_ids = sorted(set(v1["anns_by_source"]) | set(new["anns_by_source"]) | set(must_include))
    rows: List[Dict[str, Any]] = []
    for source_id in all_ids:
        v1_anns = list(v1["anns_by_source"].get(source_id, []))
        new_anns = list(new["anns_by_source"].get(source_id, []))
        if require_v1_positive and not v1_anns:
            continue
        v1_counts = _mode_counts(v1_anns)
        new_counts = _mode_counts(new_anns)
        v1_non_landscape = sum(v1_counts.get(mode, 0) for mode in NON_LANDSCAPE_MODES)
        new_non_landscape = sum(new_counts.get(mode, 0) for mode in NON_LANDSCAPE_MODES)
        removed_non_landscape = max(0, v1_non_landscape - new_non_landscape)
        v1_landscape = _landscape_by_ar(v1_anns)
        new_landscape = _landscape_by_ar(new_anns)
        changed_landscape = 0
        low_iou_values: List[float] = []
        for target_ar, v1_ann in v1_landscape.items():
            new_ann = new_landscape.get(target_ar)
            if not new_ann:
                continue
            iou = _iou(v1_ann.get("bbox") or [0, 0, 0, 0], new_ann.get("bbox") or [0, 0, 0, 0])
            if iou < 0.78:
                changed_landscape += 1
                low_iou_values.append(iou)
        gaic_landscape = _new_gaic_landscape_count(new_anns)
        risk_count = suspicious.get(source_id, 0)
        total_delta = abs(len(v1_anns) - len(new_anns))
        score = (
            8.0 * min(risk_count, 8)
            + 3.0 * removed_non_landscape
            + 1.5 * changed_landscape
            + 0.25 * total_delta
            + 0.15 * gaic_landscape
        )
        if source_id in must_include:
            score += 1000.0
        if score <= 0:
            continue
        rows.append(
            {
                "source_image_id": source_id,
                "selection_score": round(score, 4),
                "must_include": int(source_id in must_include),
                "v1_suspicious_positive_count": int(risk_count),
                "v1_positive_total": len(v1_anns),
                "new_positive_total": len(new_anns),
                "v1_non_landscape_positive": int(v1_non_landscape),
                "new_non_landscape_positive": int(new_non_landscape),
                "removed_non_landscape_positive": int(removed_non_landscape),
                "changed_landscape_ar_count": int(changed_landscape),
                "changed_landscape_iou_min": "" if not low_iou_values else round(min(low_iou_values), 4),
                "new_gaic_landscape_positive": int(gaic_landscape),
                "v1_mode_counts": dict(v1_counts),
                "new_mode_counts": dict(new_counts),
            }
        )
    rows.sort(key=lambda row: (-int(row["must_include"]), -float(row["selection_score"]), row["source_image_id"]))
    if limit > 0:
        rows = rows[:limit]
    return rows


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select v1 vs newer multimode label comparison samples.")
    parser.add_argument("--v1_label_json", required=True)
    parser.add_argument("--new_label_json", required=True)
    parser.add_argument("--out_ids", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--v1_suspicious_csv", default="")
    parser.add_argument("--must_include_ids", default="")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument(
        "--require_v1_positive",
        type=int,
        default=0,
        help="When set to 1, only select source images with at least one v1 positive label.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rows = select_samples(
        v1_label_json=Path(args.v1_label_json),
        new_label_json=Path(args.new_label_json),
        v1_suspicious_csv=Path(args.v1_suspicious_csv) if args.v1_suspicious_csv else None,
        must_include_ids=Path(args.must_include_ids) if args.must_include_ids else None,
        limit=int(args.limit),
        require_v1_positive=bool(int(args.require_v1_positive)),
    )
    out_ids = Path(args.out_ids)
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)
    out_ids.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_ids.write_text("\n".join(row["source_image_id"] for row in rows) + "\n", encoding="utf-8")
    out_json.write_text(json.dumps({"selected_count": len(rows), "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    fieldnames = [
        "source_image_id",
        "selection_score",
        "must_include",
        "v1_suspicious_positive_count",
        "v1_positive_total",
        "new_positive_total",
        "v1_non_landscape_positive",
        "new_non_landscape_positive",
        "removed_non_landscape_positive",
        "changed_landscape_ar_count",
        "changed_landscape_iou_min",
        "new_gaic_landscape_positive",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(json.dumps({"selected_count": len(rows), "out_ids": str(out_ids), "out_json": str(out_json), "out_csv": str(out_csv)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
