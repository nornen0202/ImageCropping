#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MobileCropNet v4 public benchmark shard predictions를 병합하고 검증한다.")
    parser.add_argument("--shard_dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--output_jsonl", required=True, type=Path)
    parser.add_argument("--summary_json", required=True, type=Path)
    parser.add_argument("--expected_tasks", type=int, default=0)
    parser.add_argument("--allow_duplicate_same_payload", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid json: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row is not object")
            yield row


def prediction_key(row: dict[str, Any]) -> tuple[str, str, str]:
    target_ar = row.get("target_ar")
    if target_ar is None:
        target_ar_key = ""
    else:
        target_ar_key = str(target_ar)
    return (str(row.get("dataset", "")).lower(), str(row.get("image_id", "")), target_ar_key)


def main() -> int:
    args = parse_args()
    rows_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    shard_summaries: list[dict[str, Any]] = []
    duplicate_count = 0
    input_row_count = 0
    missing_shards: list[str] = []
    for shard_dir in args.shard_dirs:
        pred_path = shard_dir / "predictions.jsonl"
        summary_path = shard_dir / "export_summary.json"
        status_path = shard_dir / "status.json"
        shard_summary: dict[str, Any] = {
            "shard_dir": str(shard_dir),
            "predictions_jsonl": str(pred_path),
            "summary_json": str(summary_path),
            "status_json": str(status_path),
            "exists": pred_path.exists(),
            "row_count": 0,
        }
        if status_path.exists():
            shard_summary["status"] = read_json(status_path)
        if summary_path.exists():
            shard_summary["export_summary"] = read_json(summary_path)
        if not pred_path.exists():
            missing_shards.append(str(shard_dir))
            shard_summaries.append(shard_summary)
            continue
        for row in read_jsonl(pred_path):
            input_row_count += 1
            shard_summary["row_count"] += 1
            key = prediction_key(row)
            prev = rows_by_key.get(key)
            if prev is not None:
                duplicate_count += 1
                if not args.allow_duplicate_same_payload or prev != row:
                    raise ValueError(f"duplicate prediction key with different payload: {key}")
                continue
            rows_by_key[key] = row
        shard_summaries.append(shard_summary)

    if missing_shards:
        raise SystemExit(f"missing shard prediction files: {missing_shards}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for key in sorted(rows_by_key.keys()):
            handle.write(json.dumps(rows_by_key[key], ensure_ascii=False, sort_keys=True) + "\n")

    merged_count = len(rows_by_key)
    expected = int(args.expected_tasks)
    complete = expected <= 0 or merged_count == expected
    summary = {
        "state": "completed" if complete else "incomplete",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_jsonl": str(args.output_jsonl),
        "input_row_count": int(input_row_count),
        "merged_row_count": int(merged_count),
        "duplicate_count": int(duplicate_count),
        "expected_tasks": expected,
        "complete": bool(complete),
        "shards": shard_summaries,
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not complete:
        raise SystemExit(f"merged rows {merged_count} != expected_tasks {expected}")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
