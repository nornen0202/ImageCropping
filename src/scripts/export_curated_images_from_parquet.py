#!/usr/bin/env python3
"""
Export curated images from filtered parquet + SSTK tar directory.

Input parquet must contain:
  - image_id
  - tar_name
"""
from __future__ import annotations

import argparse
import os
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set

import pandas as pd
from tqdm import tqdm


def resolve_tar_path(tar_dir: str, bucket: str, tar_name: str) -> str:
    candidates = [
        os.path.join(tar_dir, tar_name),
        os.path.join(tar_dir, bucket, tar_name) if bucket else "",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export curated images from parquet")
    p.add_argument("--input_parquet", required=True, help="filtered parquet path")
    p.add_argument("--tar_dir", required=True, help="SSTK tar root")
    p.add_argument("--output_dir", required=True, help="image output directory")
    p.add_argument("--bucket", default="sstk_100", help="bucket name for nested tar layout fallback")
    p.add_argument("--skip_existing", type=int, default=1, help="1=skip existing files")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input_parquet, columns=["image_id", "tar_name"])
    if df.empty:
        print("[export] empty parquet rows. nothing to do.")
        return

    by_tar: Dict[str, Set[str]] = defaultdict(set)
    for _, row in df.iterrows():
        by_tar[str(row["tar_name"])].add(str(row["image_id"]))

    skip_existing = bool(int(args.skip_existing))
    saved = 0
    skipped = 0
    missing = 0

    for tar_name, id_set in tqdm(by_tar.items(), total=len(by_tar), desc="export-images"):
        tar_path = resolve_tar_path(args.tar_dir, args.bucket, tar_name)
        if not tar_path:
            missing += len(id_set)
            continue

        targets = set(id_set)
        try:
            with tarfile.open(tar_path, "r|") as tf:
                for member in tf:
                    if not member.isfile():
                        continue
                    base = os.path.basename(member.name)
                    stem, ext = os.path.splitext(base)
                    if stem not in targets:
                        continue
                    ext = ext.lower() if ext else ".jpg"
                    out_path = out_dir / f"{stem}{ext}"
                    if skip_existing and out_path.exists():
                        skipped += 1
                    else:
                        fobj = tf.extractfile(member)
                        if fobj is None:
                            continue
                        out_path.write_bytes(fobj.read())
                        saved += 1
                    targets.remove(stem)
                    if not targets:
                        break
            if targets:
                missing += len(targets)
        except Exception as e:
            print(f"[export] failed tar={tar_name}: {e}")
            missing += len(targets)

    print(
        f"[export] done. saved={saved} skipped={skipped} missing={missing} "
        f"target={len(df)} output_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
