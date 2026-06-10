from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _box_area(box: Any) -> float:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return 0.0
    x1, y1, x2, y2 = [_safe_float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _read_ids(path: Optional[Path]) -> set[str]:
    if path is None or not path.is_file():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def _landscape_subject_safe_active(row: Dict[str, Any]) -> bool:
    attrs = row.get("attributes") if isinstance(row.get("attributes"), dict) else {}
    if _safe_int(attrs.get("landscape_subject_safe_enabled"), 1) == 0:
        return False
    route_bucket = str(attrs.get("route_bucket") or "").strip().lower()
    route_family = str(row.get("route_family") or "").strip().lower()
    route_mode = str(row.get("route_mode") or "").strip().lower()
    core_box = attrs.get("scene_guidance_bbox_norm_xyxy")
    if not isinstance(core_box, (list, tuple)) or len(core_box) != 4:
        core_box = attrs.get("scene_effective_bbox_norm_xyxy")
    core_area = _box_area(core_box)
    return (
        core_area >= 0.025
        and (
            route_bucket in {"human", "object"}
            or route_family in {"human", "person", "object", "animal"}
            or route_mode.startswith("portrait")
            or route_mode.startswith("object")
        )
    )


def collect_ids(
    *,
    query_status_jsonl: Path,
    out_txt: Path,
    summary_json: Path,
    audit_affected_ids: Optional[Path],
) -> Dict[str, Any]:
    audit_ids = _read_ids(audit_affected_ids)
    inactive_ids: set[str] = set()
    inactive_query_count = 0
    disabled_query_count = 0
    landscape_query_count = 0
    with query_status_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("mode_name") != "landscape":
                continue
            landscape_query_count += 1
            source_id = str(row.get("source_image_id") or "")
            attrs = row.get("attributes") if isinstance(row.get("attributes"), dict) else {}
            if _safe_int(attrs.get("landscape_subject_safe_enabled"), 1) == 0:
                disabled_query_count += 1
            if not _landscape_subject_safe_active(row):
                inactive_query_count += 1
                if source_id:
                    inactive_ids.add(source_id)
    union_ids = sorted(audit_ids | inactive_ids)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(union_ids) + ("\n" if union_ids else ""), encoding="utf-8")
    summary = {
        "query_status_jsonl": str(query_status_jsonl),
        "audit_affected_ids": str(audit_affected_ids) if audit_affected_ids else "",
        "landscape_query_count": landscape_query_count,
        "subjectsafe_inactive_query_count": inactive_query_count,
        "subjectsafe_disabled_query_count": disabled_query_count,
        "subjectsafe_inactive_image_count": len(inactive_ids),
        "audit_affected_image_count": len(audit_ids),
        "replacement_image_count": len(union_ids),
        "out_txt": str(out_txt),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect v13 replacement ids for landscape subject-safe inactive cases.")
    parser.add_argument("--query_status_jsonl", required=True)
    parser.add_argument("--out_txt", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--audit_affected_ids", default="")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = collect_ids(
        query_status_jsonl=Path(args.query_status_jsonl),
        out_txt=Path(args.out_txt),
        summary_json=Path(args.summary_json),
        audit_affected_ids=Path(args.audit_affected_ids) if args.audit_affected_ids else None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
