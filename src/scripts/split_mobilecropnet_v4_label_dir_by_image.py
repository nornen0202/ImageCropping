#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


JSONL_FILENAMES = (
    "train_conditional_detr_batch.jsonl",
    "train_pairwise.jsonl",
    "train_listwise.jsonl",
    "train_decision.jsonl",
    "train_checklist.jsonl",
    "train_regression.jsonl",
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def stable_unit_float(key: str, seed: int) -> float:
    digest = hashlib.sha1(f"{seed}:{key}".encode("utf-8", errors="ignore")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def assign_split(image_id: str, *, seed: int, train_fraction: float, val_fraction: float, test_fraction: float) -> str:
    if train_fraction <= 0.0 or val_fraction < 0.0 or test_fraction < 0.0:
        raise ValueError("invalid split fractions")
    total = float(train_fraction + val_fraction + test_fraction)
    if total <= 0.0:
        raise ValueError("split fractions must sum to > 0")
    train_boundary = float(train_fraction) / total
    val_boundary = float(train_fraction + val_fraction) / total
    value = stable_unit_float(str(image_id), int(seed))
    if value < train_boundary:
        return "train"
    if value < val_boundary:
        return "val"
    return "test"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a MobileCropNet v4 label directory into deterministic train/val/test subsets by image_id.")
    parser.add_argument("--input_label_dir", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260422)
    parser.add_argument("--train_fraction", type=float, default=0.8)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--strict", action="store_true", help="Fail if a known JSONL file is missing.")
    return parser.parse_args(argv)


def split_rows(
    rows: Iterable[dict[str, Any]],
    *,
    seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> tuple[dict[str, list[dict[str, Any]]], Counter[str]]:
    out = {"train": [], "val": [], "test": []}
    counters: Counter[str] = Counter()
    for row in rows:
        image_id = str(row.get("image_id", ""))
        split_name = assign_split(
            image_id,
            seed=int(seed),
            train_fraction=float(train_fraction),
            val_fraction=float(val_fraction),
            test_fraction=float(test_fraction),
        )
        out[split_name].append(row)
        counters[f"{split_name}::rows"] += 1
        counters[f"{split_name}::images::{image_id}"] += 1
    return out, counters


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary: dict[str, Any] = {
        "status": "ok",
        "input_label_dir": str(args.input_label_dir),
        "output_root": str(args.output_root),
        "seed": int(args.seed),
        "fractions": {
            "train": float(args.train_fraction),
            "val": float(args.val_fraction),
            "test": float(args.test_fraction),
        },
        "files": {},
    }
    totals: Counter[str] = Counter()

    for filename in JSONL_FILENAMES:
        src = args.input_label_dir / filename
        if not src.exists():
            if args.strict:
                raise FileNotFoundError(src)
            summary["files"][filename] = {"status": "missing"}
            continue

        split_rows_by_name, counters = split_rows(
            iter_jsonl(src),
            seed=int(args.seed),
            train_fraction=float(args.train_fraction),
            val_fraction=float(args.val_fraction),
            test_fraction=float(args.test_fraction),
        )
        file_summary: dict[str, Any] = {"status": "ok", "source": str(src)}
        for split_name, rows in split_rows_by_name.items():
            dst = args.output_root / split_name / filename
            written = write_jsonl(dst, rows)
            image_count = len({str(row.get("image_id", "")) for row in rows})
            file_summary[split_name] = {
                "path": str(dst),
                "row_count": int(written),
                "image_count": int(image_count),
            }
            totals[f"{split_name}::rows"] += int(written)
            totals[f"{split_name}::images"] += int(image_count)
        summary["files"][filename] = file_summary

        for key, value in counters.items():
            if key.endswith("::rows"):
                totals[f"{filename}::{key}"] += int(value)

    summary["totals"] = dict(sorted(totals.items()))
    write_json(args.output_root / "split_manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
