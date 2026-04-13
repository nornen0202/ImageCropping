from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from scripts.progress_utils import count_nonempty_lines, progress_log


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_mode_rows(summary: Dict[str, Any], query_status_jsonl: Path) -> Dict[str, Any]:
    mode_counter = Counter()
    positive_counter = Counter()
    with query_status_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            mode_name = str(row.get("mode_name", ""))
            mode_counter[mode_name] += 1
            if int(row.get("positive_exists", 0)) == 1:
                positive_counter[mode_name] += 1
    mode_summary = summary.get("mode_summary") if isinstance(summary.get("mode_summary"), dict) else {}
    mismatches = []
    for mode_name, payload in sorted(mode_summary.items()):
        if not isinstance(payload, dict):
            continue
        expected_queries = int(payload.get("query_count", 0))
        expected_positives = int(payload.get("positive_query_count", 0))
        if mode_counter.get(mode_name, 0) != expected_queries:
            mismatches.append(f"{mode_name}:query_count")
        if positive_counter.get(mode_name, 0) != expected_positives:
            mismatches.append(f"{mode_name}:positive_query_count")
    return {
        "mode_query_counts": dict(mode_counter),
        "mode_positive_counts": dict(positive_counter),
        "mismatches": mismatches,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate multimode training label outputs.")
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--coco_json", required=True)
    parser.add_argument("--query_status_jsonl", required=True)
    parser.add_argument("--out_json", default="")
    parser.add_argument("--progress", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary_json = Path(args.summary_json)
    coco_json = Path(args.coco_json)
    query_status_jsonl = Path(args.query_status_jsonl)
    for path in (summary_json, coco_json, query_status_jsonl):
        if not path.is_file():
            raise FileNotFoundError(f"required file missing: {path}")
    summary = _load_json(summary_json)
    coco = _load_json(coco_json)
    query_row_count = count_nonempty_lines(query_status_jsonl)
    if int(summary.get("query_count", -1)) != query_row_count:
        raise ValueError(
            f"query_count mismatch summary={summary.get('query_count')} query_status_rows={query_row_count}"
        )
    images = coco.get("images") if isinstance(coco.get("images"), list) else []
    annotations = coco.get("annotations") if isinstance(coco.get("annotations"), list) else []
    categories = coco.get("categories") if isinstance(coco.get("categories"), list) else []
    if int(summary.get("coco_image_count", -1)) != len(images):
        raise ValueError("coco image count mismatch")
    if int(summary.get("annotation_count", -1)) != len(annotations):
        raise ValueError("annotation count mismatch")
    if len(categories) <= 0:
        raise ValueError("categories is empty")
    mode_validation = _validate_mode_rows(summary, query_status_jsonl)
    if mode_validation["mismatches"]:
        raise ValueError("mode summary mismatch: " + ",".join(mode_validation["mismatches"]))
    payload = {
        "ok": True,
        "summary_json": str(summary_json),
        "coco_json": str(coco_json),
        "query_status_jsonl": str(query_status_jsonl),
        "query_count": query_row_count,
        "coco_image_count": len(images),
        "annotation_count": len(annotations),
        "category_count": len(categories),
        "mode_query_counts": mode_validation["mode_query_counts"],
        "mode_positive_counts": mode_validation["mode_positive_counts"],
    }
    progress_log(
        f"multimode validation ok images={len(images)} annotations={len(annotations)} queries={query_row_count}",
        enabled=bool(args.progress),
    )
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
