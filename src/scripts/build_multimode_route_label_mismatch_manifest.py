from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


PERSON_MODES = {"single_person_center", "single_person_rot"}
FACE_MODES = {"face"}
GROUP_MODES = {"group_center", "group_rot"}
OBJECT_MODES = {"object_single_center", "object_single_rot", "object_multi_center", "object_multi_rot"}
LANDSCAPE_MODES = {"landscape"}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _source_maps(label_payload: Dict[str, Any]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for image in label_payload.get("images", []):
        if not isinstance(image, dict):
            continue
        image_id = _safe_int(image.get("id"), -1)
        if image_id < 0:
            continue
        source_id = str(image.get("source_image_id") or Path(str(image.get("file_name") or "")).stem)
        out[image_id] = source_id
    return out


def _positive_mode_counts(label_payload: Dict[str, Any]) -> Dict[str, Counter[str]]:
    image_id_to_source = _source_maps(label_payload)
    counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for ann in label_payload.get("annotations", []):
        if not isinstance(ann, dict) or _safe_int(ann.get("gt_flag"), 0) != 1:
            continue
        source_id = image_id_to_source.get(_safe_int(ann.get("image_id"), -1), "")
        if not source_id:
            continue
        counts[source_id][str(ann.get("mode_name") or "unknown")] += 1
    return counts


def _route_by_source(query_status_jsonl: Path) -> Dict[str, str]:
    route_votes: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in _iter_jsonl(query_status_jsonl):
        source_id = str(row.get("source_image_id") or "")
        route_mode = str(row.get("route_mode") or "")
        if source_id and route_mode:
            route_votes[source_id][route_mode] += 1
    return {source_id: votes.most_common(1)[0][0] for source_id, votes in route_votes.items() if votes}


def _flags(route_mode: str, counts: Counter[str]) -> List[str]:
    route_norm = str(route_mode or "").strip().lower()
    person_count = sum(counts.get(mode, 0) for mode in PERSON_MODES)
    face_count = sum(counts.get(mode, 0) for mode in FACE_MODES)
    group_count = sum(counts.get(mode, 0) for mode in GROUP_MODES)
    object_count = sum(counts.get(mode, 0) for mode in OBJECT_MODES)
    landscape_count = sum(counts.get(mode, 0) for mode in LANDSCAPE_MODES)
    out: List[str] = []
    if route_norm == "portrait_single":
        if person_count == 0 and face_count == 0:
            out.append("portrait_single_without_person_or_face_positive")
        if group_count > 0:
            out.append("portrait_single_with_group_positive")
    elif route_norm == "portrait_group":
        if group_count == 0 and person_count == 0 and face_count == 0:
            out.append("portrait_group_without_human_positive")
    elif route_norm in {"object_single", "object_multi"}:
        if object_count == 0:
            out.append("object_route_without_object_positive")
    elif route_norm in {"scene_general", "background_texture_copyspace"}:
        if landscape_count == 0:
            out.append("scene_route_without_landscape_positive")
    return out


def build_manifest(*, label_json: Path, query_status_jsonl: Path, out_json: Path, out_csv: Path) -> Dict[str, Any]:
    label_payload = _load_json(label_json)
    positive_counts = _positive_mode_counts(label_payload)
    routes = _route_by_source(query_status_jsonl)
    rows: List[Dict[str, Any]] = []
    flag_counts: Counter[str] = Counter()
    all_source_ids = sorted(set(routes) | set(positive_counts))
    for source_id in all_source_ids:
        counts = positive_counts.get(source_id, Counter())
        route_mode = routes.get(source_id, "")
        flags = _flags(route_mode, counts)
        for flag in flags:
            flag_counts[flag] += 1
        rows.append(
            {
                "source_image_id": source_id,
                "route_mode": route_mode,
                "flags": "|".join(flags),
                "positive_total": int(sum(counts.values())),
                "landscape": int(sum(counts.get(mode, 0) for mode in LANDSCAPE_MODES)),
                "single_person": int(sum(counts.get(mode, 0) for mode in PERSON_MODES)),
                "face": int(sum(counts.get(mode, 0) for mode in FACE_MODES)),
                "group": int(sum(counts.get(mode, 0) for mode in GROUP_MODES)),
                "object": int(sum(counts.get(mode, 0) for mode in OBJECT_MODES)),
            }
        )
    flagged_rows = [row for row in rows if row["flags"]]
    summary = {
        "label_json": str(label_json),
        "query_status_jsonl": str(query_status_jsonl),
        "source_image_count": len(rows),
        "flagged_image_count": len(flagged_rows),
        "flag_counts": dict(flag_counts),
        "csv_path": str(out_csv),
        "flagged_examples": flagged_rows[:50],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "source_image_id",
            "route_mode",
            "flags",
            "positive_total",
            "landscape",
            "single_person",
            "face",
            "group",
            "object",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build route-label mismatch manifest for multimode labels.")
    parser.add_argument("--label_json", required=True)
    parser.add_argument("--query_status_jsonl", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--out_csv", required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = build_manifest(
        label_json=Path(args.label_json),
        query_status_jsonl=Path(args.query_status_jsonl),
        out_json=Path(args.out_json),
        out_csv=Path(args.out_csv),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
