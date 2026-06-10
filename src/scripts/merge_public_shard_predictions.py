#!/usr/bin/env python3
"""Merge public benchmark prediction shards into one JSONL with a summary."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards_base", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--shard_count", type=int, default=4)
    parser.add_argument("--prediction_name", default="predictions.jsonl")
    parser.add_argument("--summary_name", default="export_summary.json")
    parser.add_argument("--output_name", default="predictions.jsonl")
    parser.add_argument("--summary_json", type=Path, default=None)
    parser.add_argument("--allow_incomplete", action="store_true")
    return parser


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def line_count(path: Path) -> int:
    count = 0
    with path.open("rb") as handle:
        for _ in handle:
            count += 1
    return count


def main() -> None:
    args = build_parser().parse_args()
    start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    summary_path = args.summary_json or (args.output_dir / "merge_summary.json")

    shard_payloads: list[dict[str, Any]] = []
    missing: list[str] = []
    prediction_rows = 0
    with output_path.open("wb") as out_handle:
        for shard_idx in range(int(args.shard_count)):
            shard_dir = args.shards_base / f"shard{shard_idx:02d}"
            pred_path = shard_dir / args.prediction_name
            shard_summary_path = shard_dir / args.summary_name
            if not pred_path.exists() or not shard_summary_path.exists():
                missing.append(str(shard_dir))
                if args.allow_incomplete:
                    continue
                raise SystemExit(f"missing complete shard artifacts: {shard_dir}")
            rows = line_count(pred_path)
            prediction_rows += rows
            shard_summary = read_json(shard_summary_path)
            shard_payloads.append(
                {
                    "shard_idx": shard_idx,
                    "prediction_path": str(pred_path),
                    "summary_path": str(shard_summary_path),
                    "prediction_rows": rows,
                    "summary": shard_summary,
                }
            )
            with pred_path.open("rb") as in_handle:
                shutil.copyfileobj(in_handle, out_handle, length=1024 * 1024)

    payload = {
        "state": "completed" if not missing else "incomplete",
        "shards_base": str(args.shards_base),
        "output_dir": str(args.output_dir),
        "output_path": str(output_path),
        "shard_count": int(args.shard_count),
        "completed_shards": len(shard_payloads),
        "missing_shards": missing,
        "prediction_rows": int(prediction_rows),
        "duration_sec": round(time.time() - start, 3),
        "shards": shard_payloads,
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    if missing and not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
