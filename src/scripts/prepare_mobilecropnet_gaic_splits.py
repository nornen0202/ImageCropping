#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create image-group train/dev splits for MobileCropNet GAIC-like annotation label JSON.")
    parser.add_argument("--source_json", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--val_ratio", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=20260415)
    parser.add_argument("--prefix", default="mobilecropnet_gaic")
    return parser


def _group_key(image: dict[str, Any]) -> str:
    return str(image.get("sstk_image_id", image.get("file_name", image.get("id"))))


def _target_ar(image: dict[str, Any]) -> str:
    return str(image.get("target_ar", "FREE"))


def _subset_payload(payload: dict[str, Any], image_ids: set[int]) -> dict[str, Any]:
    images = [image for image in payload.get("images", []) if int(image["id"]) in image_ids]
    annotations = [ann for ann in payload.get("annotations", []) if int(ann["image_id"]) in image_ids]
    out = {key: value for key, value in payload.items() if key not in {"images", "annotations"}}
    out["images"] = images
    out["annotations"] = annotations
    return out


def _summary(payload: dict[str, Any]) -> dict[str, Any]:
    pos = 0
    neg = 0
    ar_counts: dict[str, int] = defaultdict(int)
    groups: set[str] = set()
    for image in payload.get("images", []):
        ar_counts[_target_ar(image)] += 1
        groups.add(_group_key(image))
    for ann in payload.get("annotations", []):
        if int(ann.get("gt_flag", 0)) == 1:
            pos += 1
        else:
            neg += 1
    return {
        "image_count": len(payload.get("images", [])),
        "group_count": len(groups),
        "annotation_count": len(payload.get("annotations", [])),
        "positive_annotation_count": pos,
        "negative_annotation_count": neg,
        "target_ar_counts": dict(sorted(ar_counts.items())),
    }


def _split_group_keys(payload: dict[str, Any], *, val_ratio: float, seed: int) -> tuple[set[str], set[str]]:
    groups: list[str] = []
    seen: set[str] = set()
    for image in payload.get("images", []):
        key = _group_key(image)
        if key not in seen:
            seen.add(key)
            groups.append(key)
    rng = random.Random(seed)
    ordered = list(groups)
    rng.shuffle(ordered)
    val_count = max(1, round(len(ordered) * max(0.0, min(0.9, val_ratio))))
    val_groups = set(ordered[:val_count])
    train_groups = set(ordered[val_count:])
    if not train_groups or not val_groups:
        raise ValueError("split produced empty train or val group set")
    return train_groups, val_groups


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(args.source_json.read_text(encoding="utf-8"))
    train_groups, val_groups = _split_group_keys(payload, val_ratio=args.val_ratio, seed=args.seed)
    train_ids = {int(image["id"]) for image in payload.get("images", []) if _group_key(image) in train_groups}
    val_ids = {int(image["id"]) for image in payload.get("images", []) if _group_key(image) in val_groups}
    train_payload = _subset_payload(payload, train_ids)
    val_payload = _subset_payload(payload, val_ids)
    train_path = args.output_dir / f"{args.prefix}_train_dev.json"
    val_path = args.output_dir / f"{args.prefix}_val_dev.json"
    train_path.write_text(json.dumps(train_payload, ensure_ascii=False), encoding="utf-8")
    val_path.write_text(json.dumps(val_payload, ensure_ascii=False), encoding="utf-8")
    summary = {
        "source_json": str(args.source_json),
        "train_json": str(train_path),
        "val_json": str(val_path),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "train": _summary(train_payload),
        "val": _summary(val_payload),
        "group_overlap_count": len(train_groups & val_groups),
    }
    (args.output_dir / f"{args.prefix}_split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
