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
import multiprocessing as mp
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd
from tqdm import tqdm

DEFAULT_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


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
    p.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="number of worker processes (0=auto-balanced, -1=all CPU cores, 1=single process)",
    )
    p.add_argument(
        "--auto_workers_cap",
        type=int,
        default=8,
        help="auto mode max workers cap (only used when --num_workers=0)",
    )
    return p.parse_args()


def build_tar_jobs(df: pd.DataFrame) -> List[Tuple[str, List[str]]]:
    by_tar: Dict[str, Set[str]] = defaultdict(set)
    for tar_name, image_id in zip(df["tar_name"].astype(str), df["image_id"].astype(str)):
        by_tar[tar_name].add(image_id)
    return [(tar_name, sorted(list(id_set))) for tar_name, id_set in by_tar.items()]


def resolve_workers(num_workers: int, auto_workers_cap: int, num_jobs: int) -> int:
    if num_jobs <= 0:
        return 1
    cpu_cnt = max(1, int(os.cpu_count() or 1))
    if num_workers == -1:
        w = cpu_cnt
    elif num_workers <= 0:
        cap = max(1, int(auto_workers_cap))
        w = min(cpu_cnt, cap)
    else:
        w = int(num_workers)
    return max(1, min(w, num_jobs))


def _find_existing_image(out_dir: Path, stem: str) -> Path | None:
    for ext in DEFAULT_IMAGE_EXTS:
        p = out_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def export_one_tar(job: Tuple[str, List[str], str, str, str, int]) -> Tuple[int, int, int, str, str]:
    tar_name, image_ids, tar_dir, bucket, output_dir, skip_existing_int = job
    skip_existing = bool(int(skip_existing_int))
    out_dir = Path(output_dir)
    id_set = set(image_ids)
    targets = set(id_set)
    saved = 0
    skipped = 0
    missing = 0

    tar_path = resolve_tar_path(tar_dir, bucket, tar_name)
    if not tar_path:
        return 0, 0, len(id_set), tar_name, "tar_not_found"

    try:
        with tarfile.open(tar_path, "r|") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                base = os.path.basename(member.name)
                stem, ext = os.path.splitext(base)
                if stem not in targets:
                    continue
                ext = ext.lower()
                if ext not in DEFAULT_IMAGE_EXTS:
                    # SSTK tar member order is often: .desc/.id/.jpg/.tags.
                    # Only export real image payloads.
                    continue
                out_path = out_dir / f"{stem}{ext}"
                if skip_existing and _find_existing_image(out_dir, stem) is not None:
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
        return saved, skipped, missing, tar_name, ""
    except Exception as e:
        missing += len(targets)
        return saved, skipped, missing, tar_name, str(e)


def run_export_jobs(
    jobs: List[Tuple[str, List[str], str, str, str, int]],
    workers: int,
) -> Iterable[Tuple[int, int, int, str, str]]:
    if workers <= 1:
        for job in jobs:
            yield export_one_tar(job)
        return

    with mp.Pool(processes=workers) as pool:
        for res in pool.imap_unordered(export_one_tar, jobs, chunksize=1):
            yield res


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input_parquet, columns=["image_id", "tar_name"])
    if df.empty:
        print("[export] empty parquet rows. nothing to do.")
        return

    tar_jobs = build_tar_jobs(df)
    workers = resolve_workers(args.num_workers, args.auto_workers_cap, len(tar_jobs))
    print(
        f"[export] jobs={len(tar_jobs)} workers={workers} "
        f"(num_workers={args.num_workers}, auto_cap={args.auto_workers_cap})"
    )

    skip_existing_int = int(args.skip_existing)
    skip_existing = bool(skip_existing_int)
    saved = 0
    skipped = 0
    missing = 0
    err_count = 0

    jobs = [
        (tar_name, image_ids, args.tar_dir, args.bucket, str(out_dir), skip_existing_int)
        for tar_name, image_ids in tar_jobs
    ]
    for _saved, _skipped, _missing, _tar_name, _err in tqdm(
        run_export_jobs(jobs, workers),
        total=len(jobs),
        desc="export-images",
    ):
        saved += int(_saved)
        skipped += int(_skipped)
        missing += int(_missing)
        if _err:
            err_count += 1
            if err_count <= 20:
                print(f"[export] failed tar={_tar_name}: {_err}")

    if err_count > 20:
        print(f"[export] ... and {err_count - 20} more tar errors")

    print(
        f"[export] done. saved={saved} skipped={skipped} missing={missing} "
        f"target={len(df)} output_dir={out_dir} workers={workers} "
        f"skip_existing={int(skip_existing)} tar_errors={err_count}"
    )


if __name__ == "__main__":
    main()
