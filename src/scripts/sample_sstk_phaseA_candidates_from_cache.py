#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import shutil
import tarfile
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


DEFAULT_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
DEFAULT_COLUMNS = [
    "image_id",
    "width",
    "height",
    "aesthetic_score_center",
    "aesthetic_score_pad",
    "sstk_type",
    "tags",
    "tar_name",
    "aes_score",
    "super_cat",
    "rarest_freq",
    "sampling_weight",
]


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not math.isfinite(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class StatusWriter:
    def __init__(self, path: Path | None, *, started_at: str, argv: list[str] | None = None) -> None:
        self.path = path
        self.payload: dict[str, Any] = {
            "started_at": started_at,
            "updated_at": started_at,
            "pid": os.getpid(),
            "argv": argv or [],
        }

    def update(self, **kwargs: Any) -> None:
        self.payload.update({k: _json_safe(v) for k, v in kwargs.items()})
        self.payload["updated_at"] = _now()
        if self.path is not None:
            _write_json(self.path, self.payload)


def _required_columns(path: Path, requested_columns: list[str]) -> list[str]:
    schema_names = set(pq.ParquetFile(path).schema_arrow.names)
    missing = [c for c in ["image_id", "aes_score", "super_cat", "tar_name"] if c not in schema_names]
    if missing:
        raise ValueError(f"mapped cache parquet is missing required columns: {missing}")
    return [c for c in requested_columns if c in schema_names]


def _iter_batch_dfs(path: Path, columns: list[str], batch_size: int):
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        yield batch.to_pandas()


def compute_category_thresholds(
    parquet_path: Path,
    *,
    top_percentile: float,
    batch_size: int,
    status: StatusWriter | None = None,
) -> tuple[dict[str, float], dict[str, int]]:
    if not (0.0 < top_percentile <= 1.0):
        raise ValueError("--top_percentile must be in (0, 1]")

    chunks_by_cat: dict[str, list[np.ndarray]] = defaultdict(list)
    counts: Counter[str] = Counter()
    rows_seen = 0
    batches_seen = 0
    for df in _iter_batch_dfs(parquet_path, ["super_cat", "aes_score"], batch_size=batch_size):
        batches_seen += 1
        rows_seen += len(df)
        df = df[["super_cat", "aes_score"]].dropna()
        if len(df) > 0:
            df["aes_score"] = pd.to_numeric(df["aes_score"], errors="coerce")
            df = df[np.isfinite(df["aes_score"].to_numpy(dtype=float, copy=False))]
            for cat, group in df.groupby("super_cat", sort=False):
                cat_key = str(cat)
                values = group["aes_score"].to_numpy(dtype=np.float64, copy=True)
                if len(values) == 0:
                    continue
                chunks_by_cat[cat_key].append(values)
                counts[cat_key] += int(len(values))
        if status is not None and (batches_seen == 1 or batches_seen % 10 == 0):
            status.update(
                phase="threshold_pass",
                batches_seen=batches_seen,
                rows_seen=rows_seen,
                categories_seen=len(counts),
            )

    q = 1.0 - float(top_percentile)
    thresholds: dict[str, float] = {}
    for cat, chunks in chunks_by_cat.items():
        values = np.concatenate(chunks)
        thresholds[cat] = float(np.quantile(values, q))
    return thresholds, {str(k): int(v) for k, v in counts.items()}


def allocate_category_targets(category_counts: dict[str, int], *, target_size: int) -> dict[str, int]:
    cats = {str(k): int(v) for k, v in category_counts.items() if int(v) > 0}
    if not cats or target_size <= 0:
        return {}

    remaining = int(target_size)
    allocation: dict[str, int] = {}
    cats_sorted = sorted(cats.items(), key=lambda kv: (kv[1], kv[0]))
    cap = 0
    n_cats = len(cats_sorted)
    for i, (cat, count) in enumerate(cats_sorted):
        alloc = remaining // max(1, n_cats - i)
        if count <= alloc:
            allocation[cat] = int(count)
            remaining -= int(count)
        else:
            cap = int(alloc)
            break
    if cap == 0 and remaining > 0:
        cap = max(1, remaining // max(1, n_cats - len(allocation)))
    for cat, count in cats_sorted:
        if cat in allocation:
            continue
        allocation[cat] = min(int(count), int(cap))

    allocated = sum(allocation.values())
    if allocated < target_size:
        for cat, count in sorted(cats.items(), key=lambda kv: (-kv[1], kv[0])):
            if allocated >= target_size:
                break
            add = min(int(count) - allocation.get(cat, 0), target_size - allocated)
            if add > 0:
                allocation[cat] = allocation.get(cat, 0) + add
                allocated += add
    return allocation


def _record_from_row(row: pd.Series, columns: list[str]) -> dict[str, Any]:
    return {col: _json_safe(row[col]) for col in columns if col in row.index}


def sample_candidates(
    parquet_path: Path,
    *,
    output_parquet: Path,
    target_size: int,
    top_percentile: float,
    random_state: int,
    batch_size: int,
    status: StatusWriter | None = None,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    requested_columns = columns or DEFAULT_COLUMNS
    read_columns = _required_columns(parquet_path, requested_columns)
    thresholds, total_counts = compute_category_thresholds(
        parquet_path,
        top_percentile=top_percentile,
        batch_size=batch_size,
        status=status,
    )
    estimated_filtered_counts = {
        cat: max(1, int(math.ceil(count * float(top_percentile)))) for cat, count in total_counts.items()
    }
    allocation = allocate_category_targets(estimated_filtered_counts, target_size=int(target_size))
    if status is not None:
        status.update(
            phase="reservoir_pass",
            category_count=len(total_counts),
            target_size=int(target_size),
            allocation=allocation,
        )

    rng = np.random.default_rng(int(random_state))
    heaps: dict[str, list[tuple[float, int, dict[str, Any]]]] = {cat: [] for cat in allocation}
    eligible_counts: Counter[str] = Counter()
    rows_seen = 0
    batches_seen = 0
    seq = 0
    for df in _iter_batch_dfs(parquet_path, read_columns, batch_size=batch_size):
        batches_seen += 1
        rows_seen += len(df)
        df["aes_score"] = pd.to_numeric(df["aes_score"], errors="coerce")
        df = df.dropna(subset=["image_id", "tar_name", "super_cat", "aes_score"])
        if len(df) == 0:
            continue

        for cat, group in df.groupby("super_cat", sort=False):
            cat_key = str(cat)
            cap = allocation.get(cat_key, 0)
            threshold = thresholds.get(cat_key)
            if cap <= 0 or threshold is None:
                continue
            eligible = group[group["aes_score"] >= threshold]
            if len(eligible) == 0:
                continue
            eligible_counts[cat_key] += int(len(eligible))
            if "sampling_weight" in eligible.columns:
                weights = pd.to_numeric(eligible["sampling_weight"], errors="coerce").to_numpy(dtype=np.float64)
                weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
            else:
                weights = np.ones(len(eligible), dtype=np.float64)
            keys = np.power(rng.random(len(eligible)), 1.0 / weights)
            if len(keys) > cap:
                take_idx = np.argpartition(keys, -cap)[-cap:]
            else:
                take_idx = np.arange(len(keys))

            heap = heaps[cat_key]
            eligible_take = eligible.iloc[take_idx]
            for key, (_, row) in zip(keys[take_idx], eligible_take.iterrows()):
                seq += 1
                item = (float(key), seq, _record_from_row(row, read_columns))
                if len(heap) < cap:
                    heapq.heappush(heap, item)
                elif item[0] > heap[0][0]:
                    heapq.heapreplace(heap, item)

        if status is not None and (batches_seen == 1 or batches_seen % 10 == 0):
            status.update(
                phase="reservoir_pass",
                batches_seen=batches_seen,
                rows_seen=rows_seen,
                sampled_so_far=sum(len(h) for h in heaps.values()),
                eligible_counts={str(k): int(v) for k, v in eligible_counts.items()},
            )

    records: list[dict[str, Any]] = []
    for cat in sorted(heaps):
        records.extend([item[2] for item in sorted(heaps[cat], reverse=True)])
    sampled = pd.DataFrame(records, columns=read_columns)
    if len(sampled) > target_size:
        sampled = sampled.sample(n=int(target_size), random_state=int(random_state), weights=sampled.get("sampling_weight"))
    sampled = sampled.reset_index(drop=True)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_parquet(output_parquet, index=False)

    summary = {
        "mapped_cache_parquet": str(parquet_path),
        "output_parquet": str(output_parquet),
        "target_size": int(target_size),
        "selected_rows": int(len(sampled)),
        "top_percentile": float(top_percentile),
        "random_state": int(random_state),
        "batch_size": int(batch_size),
        "read_columns": read_columns,
        "thresholds": {str(k): float(v) for k, v in thresholds.items()},
        "total_category_counts": {str(k): int(v) for k, v in total_counts.items()},
        "estimated_filtered_category_counts": {str(k): int(v) for k, v in estimated_filtered_counts.items()},
        "observed_eligible_category_counts": {str(k): int(v) for k, v in eligible_counts.items()},
        "allocation_targets": {str(k): int(v) for k, v in allocation.items()},
        "selected_category_counts": {
            str(k): int(v) for k, v in sampled.get("super_cat", pd.Series(dtype=str)).value_counts().to_dict().items()
        },
        "selection_complete": bool(len(sampled) == int(target_size)),
    }
    return sampled, summary


def _collect_existing_image_stems(image_dir: Path) -> set[str]:
    stems: set[str] = set()
    try:
        with os.scandir(image_dir) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                stem, ext = os.path.splitext(entry.name)
                if stem and ext.lower() in DEFAULT_IMAGE_EXTS:
                    stems.add(stem)
    except FileNotFoundError:
        return set()
    return stems


def _resolve_tar_path(tar_dir: Path, bucket: str, tar_name: str) -> Path | None:
    candidates = [tar_dir / tar_name]
    if bucket:
        candidates.append(tar_dir / bucket / tar_name)
    for path in candidates:
        if path.exists():
            return path
    return None


def _export_from_single_tar(job: tuple[str, list[str], str, str, str]) -> dict[str, Any]:
    tar_name, image_ids, tar_dir_s, bucket, image_dir_s = job
    targets = set(image_ids)
    tar_path = _resolve_tar_path(Path(tar_dir_s), bucket, tar_name)
    if tar_path is None:
        return {"saved": 0, "missing": len(targets), "tar_name": tar_name, "error": "tar_not_found"}

    image_dir = Path(image_dir_s)
    saved = 0
    try:
        with tarfile.open(tar_path, "r|") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                stem, ext = os.path.splitext(os.path.basename(member.name))
                if stem not in targets or ext.lower() not in DEFAULT_IMAGE_EXTS:
                    continue
                fobj = tf.extractfile(member)
                if fobj is None:
                    continue
                out_path = image_dir / f"{stem}{ext.lower()}"
                with open(out_path, "wb") as wf:
                    shutil.copyfileobj(fobj, wf, length=1024 * 1024)
                saved += 1
                targets.remove(stem)
                if not targets:
                    break
    except Exception as exc:
        return {"saved": saved, "missing": len(targets), "tar_name": tar_name, "error": str(exc)}
    return {"saved": saved, "missing": len(targets), "tar_name": tar_name, "error": ""}


def export_images(
    df: pd.DataFrame,
    *,
    image_dir: Path,
    tar_dir: Path,
    bucket: str,
    workers: int,
    chunksize: int,
    skip_existing: bool,
    status: StatusWriter | None = None,
) -> dict[str, Any]:
    image_dir.mkdir(parents=True, exist_ok=True)
    by_tar: dict[str, set[str]] = defaultdict(set)
    for tar_name, image_id in zip(df["tar_name"].astype(str), df["image_id"].astype(str)):
        by_tar[tar_name].add(Path(image_id).stem)

    existing = _collect_existing_image_stems(image_dir) if skip_existing else set()
    skipped = 0
    jobs: list[tuple[str, list[str], str, str, str]] = []
    for tar_name, ids in by_tar.items():
        pending = set(ids)
        if existing:
            already_saved = pending & existing
            skipped += len(already_saved)
            pending -= already_saved
        if pending:
            jobs.append((tar_name, sorted(pending), str(tar_dir), bucket, str(image_dir)))

    workers = max(1, min(int(workers), len(jobs) if jobs else 1))
    chunksize = max(1, int(chunksize))
    saved = 0
    missing = 0
    failed_jobs = 0
    errors: list[dict[str, Any]] = []
    iterator = None
    pool = None
    try:
        if jobs and workers > 1:
            pool = Pool(processes=workers)
            iterator = pool.imap_unordered(_export_from_single_tar, jobs, chunksize=chunksize)
        else:
            iterator = map(_export_from_single_tar, jobs)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(jobs), desc="Export Phase A candidates")
        for idx, result in enumerate(iterator, start=1):
            saved += int(result.get("saved", 0))
            missing += int(result.get("missing", 0))
            if result.get("error"):
                failed_jobs += 1
                if len(errors) < 100:
                    errors.append(result)
            if status is not None and (idx == 1 or idx % 20 == 0 or idx == len(jobs)):
                status.update(
                    phase="export_images",
                    export_jobs_done=idx,
                    export_jobs_total=len(jobs),
                    saved=saved,
                    skipped=skipped,
                    missing=missing,
                    failed_jobs=failed_jobs,
                )
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return {
        "image_dir": str(image_dir),
        "tar_dir": str(tar_dir),
        "bucket": bucket,
        "jobs": int(len(jobs)),
        "workers": int(workers),
        "saved": int(saved),
        "skipped_existing": int(skipped),
        "missing": int(missing),
        "failed_jobs": int(failed_jobs),
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream-sample Phase A SSTK candidates from a full mapped cache without loading all rows."
    )
    parser.add_argument("--mapped_cache_parquet", required=True)
    parser.add_argument("--output_parquet", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--tar_dir", required=True)
    parser.add_argument("--bucket", default="sstk_100")
    parser.add_argument("--candidate_size", type=int, default=40000)
    parser.add_argument("--top_percentile", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=500000)
    parser.add_argument("--export_workers", type=int, default=8)
    parser.add_argument("--export_chunksize", type=int, default=1)
    parser.add_argument("--skip_existing_images", type=int, default=1)
    parser.add_argument("--skip_export_images", type=int, default=0)
    parser.add_argument("--status_json", default="")
    parser.add_argument("--summary_json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started_at = _now()
    status = StatusWriter(
        Path(args.status_json) if str(args.status_json).strip() else None,
        started_at=started_at,
        argv=list(os.sys.argv),
    )
    mapped_cache = Path(args.mapped_cache_parquet)
    output_parquet = Path(args.output_parquet)
    image_dir = Path(args.image_dir)
    summary_json = Path(args.summary_json)

    try:
        status.update(
            state="running",
            phase="start",
            mapped_cache_parquet=str(mapped_cache),
            output_parquet=str(output_parquet),
            image_dir=str(image_dir),
        )
        sampled, summary = sample_candidates(
            mapped_cache,
            output_parquet=output_parquet,
            target_size=int(args.candidate_size),
            top_percentile=float(args.top_percentile),
            random_state=int(args.random_state),
            batch_size=int(args.batch_size),
            status=status,
        )
        export_summary: dict[str, Any] = {"skipped": bool(int(args.skip_export_images))}
        if not int(args.skip_export_images):
            status.update(phase="export_images", selected_rows=int(len(sampled)))
            export_summary = export_images(
                sampled,
                image_dir=image_dir,
                tar_dir=Path(args.tar_dir),
                bucket=str(args.bucket),
                workers=int(args.export_workers),
                chunksize=int(args.export_chunksize),
                skip_existing=bool(int(args.skip_existing_images)),
                status=status,
            )
        summary["export"] = export_summary
        summary["started_at"] = started_at
        summary["finished_at"] = _now()
        summary["status_json"] = str(args.status_json)
        _write_json(summary_json, summary)
        status.update(
            state="complete",
            phase="complete",
            summary_json=str(summary_json),
            selected_rows=int(len(sampled)),
            export=export_summary,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        if len(sampled) != int(args.candidate_size):
            raise SystemExit(1)
        if not int(args.skip_export_images):
            expected_images = int(len(sampled))
            available = int(export_summary.get("saved", 0)) + int(export_summary.get("skipped_existing", 0))
            if available < expected_images or int(export_summary.get("failed_jobs", 0)) > 0:
                raise SystemExit(1)
    except Exception as exc:
        status.update(state="failed", phase="failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
