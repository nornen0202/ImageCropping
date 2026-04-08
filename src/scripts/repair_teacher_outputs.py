#!/usr/bin/env python
"""
Auto-repair utility for teacher scorer outputs.

Purpose
-------
- Detect stale/inconsistent final teacher jsonl compared to multi-GPU shard outputs.
- Promote (merge) the best shard set when it is clearly better than final jsonl.
- Rebuild overview/QA outputs to keep artifacts consistent.

This script is intended to be called from `run_phaseA_to_teacher_e2e.sh` after
teacher stage execution.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from progress_utils import ProgressTracker, count_nonempty_lines, progress_log


@dataclass
class JsonlStats:
    path: Path
    rows: int
    expensive_real_rows: int
    parse_error_rows: int
    mtime: float


@dataclass
class ShardDirStats:
    shard_dir: Path
    shard_files: List[Path]
    total_rows: int
    expensive_real_rows: int
    parse_error_rows: int
    mtime: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repair/verify teacher scorer outputs")
    p.add_argument("--teacher_scores_jsonl", required=True)
    p.add_argument("--overview_json", required=True)
    p.add_argument("--overview_csv", required=True)
    p.add_argument("--qa_json", required=True)
    p.add_argument("--qa_csv", required=True)
    p.add_argument("--candidates_jsonl", default="")
    p.add_argument("--max_images", type=int, default=0, help="same max_images used in teacher run (0=all)")
    p.add_argument("--prefer_real_expensive", type=int, default=1, help="1=prefer expensive_real_applied rows")
    p.add_argument("--strict_expected_match", type=int, default=1, help="1=if expected rows known, require exact match")
    p.add_argument("--dry_run", type=int, default=0)
    p.add_argument("--verbose", type=int, default=1)
    p.add_argument("--progress", type=int, default=1)
    p.add_argument("--progress_every", type=int, default=1000)
    p.add_argument("--progress_min_seconds", type=float, default=10.0)
    return p.parse_args()


def as_bool(v: int) -> bool:
    return int(v) != 0


def info(msg: str, *, verbose: bool = True) -> None:
    if verbose:
        print(msg)


def count_lines(path: Path) -> int:
    c = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c += 1
    return c


def analyze_jsonl(
    path: Path,
    *,
    progress: bool = False,
    progress_every: int = 1000,
    progress_min_seconds: float = 10.0,
) -> JsonlStats:
    rows = 0
    exp_rows = 0
    parse_error_rows = 0
    if path.exists():
        total = count_nonempty_lines(path) if progress else None
        tracker = ProgressTracker(
            f"repair_teacher_outputs:analyze_jsonl:{path.name}",
            total=total,
            unit="rows",
            every=progress_every,
            min_seconds=progress_min_seconds,
            enabled=progress,
        )
        with path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    parse_error_rows += 1
                    continue
                rows += 1
                if bool(d.get("teacher_scorer", {}).get("expensive_real_applied", False)):
                    exp_rows += 1
                tracker.update(idx)
        tracker.finish(rows, extra=f"parse_errors={parse_error_rows}")
        mtime = path.stat().st_mtime
    else:
        mtime = 0.0
    return JsonlStats(
        path=path,
        rows=rows,
        expensive_real_rows=exp_rows,
        parse_error_rows=parse_error_rows,
        mtime=mtime,
    )


def shard_index_from_name(name: str) -> int:
    m = re.search(r"teacher_scores_shard_(\d+)\.jsonl$", name)
    return int(m.group(1)) if m else 10**9


def analyze_shard_dir(
    shard_dir: Path,
    *,
    progress: bool = False,
    progress_every: int = 1000,
    progress_min_seconds: float = 10.0,
) -> Optional[ShardDirStats]:
    if not shard_dir.exists() or not shard_dir.is_dir():
        return None
    files = sorted(
        [p for p in shard_dir.glob("teacher_scores_shard_*.jsonl") if p.is_file()],
        key=lambda p: shard_index_from_name(p.name),
    )
    if not files:
        return None
    total_rows = 0
    total_exp = 0
    total_parse_errors = 0
    latest_mtime = 0.0
    tracker = ProgressTracker(
        f"repair_teacher_outputs:analyze_shard_dir:{shard_dir.name}",
        total=len(files),
        unit="files",
        every=1,
        min_seconds=1.0,
        enabled=progress,
    )
    for idx, fp in enumerate(files, start=1):
        st = analyze_jsonl(
            fp,
            progress=progress,
            progress_every=progress_every,
            progress_min_seconds=progress_min_seconds,
        )
        total_rows += st.rows
        total_exp += st.expensive_real_rows
        total_parse_errors += st.parse_error_rows
        latest_mtime = max(latest_mtime, st.mtime)
        tracker.update(idx, extra=f"path={fp.name} rows={st.rows}")
    tracker.finish(len(files), extra=f"rows={total_rows}")
    return ShardDirStats(
        shard_dir=shard_dir,
        shard_files=files,
        total_rows=total_rows,
        expensive_real_rows=total_exp,
        parse_error_rows=total_parse_errors,
        mtime=latest_mtime,
    )


def expected_rows_from_candidates(candidates_jsonl: Path, max_images: int) -> Optional[int]:
    if not candidates_jsonl.exists():
        return None
    n = count_lines(candidates_jsonl)
    if max_images > 0:
        n = min(n, int(max_images))
    return n


def score_candidate(
    *,
    rows: int,
    exp_rows: int,
    parse_error_rows: int,
    mtime: float,
    expected_rows: Optional[int],
    prefer_real: bool,
) -> Tuple[int, int, int, int, int, float]:
    if expected_rows is None:
        expected_match = 0
        dist_score = rows
    else:
        expected_match = 1 if rows == expected_rows else 0
        dist_score = -abs(rows - expected_rows)
    parse_clean = 1 if parse_error_rows == 0 else 0
    real_score = 1 if (prefer_real and exp_rows > 0) else 0
    return (parse_clean, expected_match, dist_score, real_score, rows, mtime)


def choose_best_shard(
    *,
    final_stats: JsonlStats,
    shard_stats: Sequence[ShardDirStats],
    expected_rows: Optional[int],
    prefer_real: bool,
    strict_expected_match: bool,
) -> Optional[ShardDirStats]:
    if not shard_stats:
        return None

    # If strict expected rows are available, keep only exact-match shard dirs.
    candidates = list(shard_stats)
    if strict_expected_match and expected_rows is not None:
        exact = [s for s in candidates if s.total_rows == expected_rows]
        if exact:
            candidates = exact
        else:
            return None

    best = max(
        candidates,
        key=lambda s: score_candidate(
            rows=s.total_rows,
            exp_rows=s.expensive_real_rows,
            parse_error_rows=s.parse_error_rows,
            mtime=s.mtime,
            expected_rows=expected_rows,
            prefer_real=prefer_real,
        ),
    )
    return best


def should_promote(
    *,
    final_stats: JsonlStats,
    best_shard: ShardDirStats,
    expected_rows: Optional[int],
    prefer_real: bool,
) -> bool:
    # If final does not exist/empty, promote.
    if final_stats.rows <= 0:
        return best_shard.total_rows > 0

    # Prefer parse-clean outputs over corrupted ones.
    if final_stats.parse_error_rows > 0 and best_shard.parse_error_rows == 0:
        return True
    if final_stats.parse_error_rows == 0 and best_shard.parse_error_rows > 0:
        return False

    # Prefer expected-row exact match when available.
    if expected_rows is not None:
        final_match = final_stats.rows == expected_rows
        shard_match = best_shard.total_rows == expected_rows
        if shard_match and not final_match:
            return True
        if final_match and not shard_match:
            return False

    # If shard has more rows, it is likely more complete.
    if best_shard.total_rows > final_stats.rows:
        return True
    if best_shard.total_rows < final_stats.rows:
        return False

    # Same rows: if real expensive is expected and shard has real rows while final does not.
    if prefer_real and final_stats.expensive_real_rows <= 0 and best_shard.expensive_real_rows > 0:
        return True

    # No strict benefit.
    return False


def merge_shards_to_output(
    shard_files: Sequence[Path],
    out_jsonl: Path,
    *,
    progress: bool = False,
    progress_every: int = 1000,
    progress_min_seconds: float = 10.0,
) -> None:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    tracker = ProgressTracker(
        "repair_teacher_outputs:merge_shards",
        total=len(shard_files),
        unit="files",
        every=1,
        min_seconds=1.0,
        enabled=progress,
    )
    with out_jsonl.open("w", encoding="utf-8") as fout:
        for idx, fp in enumerate(shard_files, start=1):
            with fp.open("r", encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line if line.endswith("\n") else line + "\n")
            tracker.update(idx, extra=f"path={fp.name}")
    tracker.finish(len(shard_files), extra=f"output={out_jsonl.name}")


def ensure_integrity(
    *,
    stats: JsonlStats,
    expected_rows: Optional[int],
    context: str,
) -> None:
    problems: List[str] = []
    if stats.parse_error_rows > 0:
        problems.append(f"parse_error_rows={stats.parse_error_rows}")
    if expected_rows is not None and stats.rows != expected_rows:
        problems.append(f"rows={stats.rows} expected_rows={expected_rows}")
    if problems:
        joined = ", ".join(problems)
        raise SystemExit(f"[repair] integrity check failed for {context}: {joined}")


def run_py(script: Path, args: Sequence[str]) -> None:
    cmd = [sys.executable, str(script), *args]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    verbose = as_bool(args.verbose)
    progress_enabled = bool(int(args.progress))
    progress_every = max(1, int(args.progress_every))
    progress_min_seconds = max(0.0, float(args.progress_min_seconds))
    prefer_real = as_bool(args.prefer_real_expensive)
    strict_expected = as_bool(args.strict_expected_match)
    dry_run = as_bool(args.dry_run)

    out_jsonl = Path(args.teacher_scores_jsonl)
    overview_json = Path(args.overview_json)
    overview_csv = Path(args.overview_csv)
    qa_json = Path(args.qa_json)
    qa_csv = Path(args.qa_csv)
    candidates_jsonl = Path(args.candidates_jsonl) if str(args.candidates_jsonl).strip() else Path("")

    progress_log(
        f"repair_teacher_outputs: start | teacher_scores_jsonl={out_jsonl}",
        enabled=progress_enabled,
    )
    final_stats = analyze_jsonl(
        out_jsonl,
        progress=progress_enabled,
        progress_every=progress_every,
        progress_min_seconds=progress_min_seconds,
    )
    expected_rows = None
    if candidates_jsonl:
        expected_rows = expected_rows_from_candidates(candidates_jsonl, int(args.max_images))

    shard_dirs = sorted(
        [p for p in out_jsonl.parent.glob(f"{out_jsonl.name}.shards.*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
    )
    shard_stats: List[ShardDirStats] = []
    for d in shard_dirs:
        st = analyze_shard_dir(
            d,
            progress=progress_enabled,
            progress_every=progress_every,
            progress_min_seconds=progress_min_seconds,
        )
        if st is not None:
            shard_stats.append(st)

    info(
        "[repair] final rows={} expensive_real_rows={} parse_error_rows={} expected_rows={} shard_dirs={}".format(
            final_stats.rows,
            final_stats.expensive_real_rows,
            final_stats.parse_error_rows,
            expected_rows if expected_rows is not None else "<unknown>",
            len(shard_stats),
        ),
        verbose=verbose,
    )

    best_shard = choose_best_shard(
        final_stats=final_stats,
        shard_stats=shard_stats,
        expected_rows=expected_rows,
        prefer_real=prefer_real,
        strict_expected_match=strict_expected,
    )
    if best_shard is None:
        info("[repair] no promotable shard dir found (or strict expected match not satisfied).", verbose=verbose)
        # Still ensure overview/QA exist for current final.
        if out_jsonl.exists() and out_jsonl.stat().st_size > 0:
            script_dir = Path(__file__).resolve().parent
            if not dry_run:
                run_py(
                    script_dir / "rebuild_teacher_overview.py",
                    [
                        "--teacher_scores_jsonl",
                        str(out_jsonl),
                        "--output_json",
                        str(overview_json),
                        "--output_by_ar_csv",
                        str(overview_csv),
                    ],
                )
                run_py(
                    script_dir / "qa_teacher_report.py",
                    [
                        "--teacher_scores_jsonl",
                        str(out_jsonl),
                        "--output_json",
                        str(qa_json),
                        "--output_by_ar_csv",
                        str(qa_csv),
                    ],
                )
        return

    info(
        "[repair] best shard dir={} rows={} expensive_real_rows={} parse_error_rows={}".format(
            best_shard.shard_dir,
            best_shard.total_rows,
            best_shard.expensive_real_rows,
            best_shard.parse_error_rows,
        ),
        verbose=verbose,
    )

    promote = should_promote(
        final_stats=final_stats,
        best_shard=best_shard,
        expected_rows=expected_rows,
        prefer_real=prefer_real,
    )

    if promote:
        info(f"[repair] promoting shards -> {out_jsonl}", verbose=verbose)
        if not dry_run:
            merge_shards_to_output(
                best_shard.shard_files,
                out_jsonl,
                progress=progress_enabled,
                progress_every=progress_every,
                progress_min_seconds=progress_min_seconds,
            )
            script_dir = Path(__file__).resolve().parent
            run_py(
                script_dir / "rebuild_teacher_overview.py",
                [
                    "--teacher_scores_jsonl",
                    str(out_jsonl),
                    "--output_json",
                    str(overview_json),
                    "--output_by_ar_csv",
                    str(overview_csv),
                ],
            )
            run_py(
                script_dir / "qa_teacher_report.py",
                [
                    "--teacher_scores_jsonl",
                    str(out_jsonl),
                    "--output_json",
                    str(qa_json),
                    "--output_by_ar_csv",
                    str(qa_csv),
                ],
            )
        info("[repair] promote+rebuild done.", verbose=verbose)
    else:
        info("[repair] final output already preferred. rebuilding overview/QA for consistency.", verbose=verbose)
        script_dir = Path(__file__).resolve().parent
        if not dry_run and out_jsonl.exists():
            run_py(
                script_dir / "rebuild_teacher_overview.py",
                [
                    "--teacher_scores_jsonl",
                    str(out_jsonl),
                    "--output_json",
                    str(overview_json),
                    "--output_by_ar_csv",
                    str(overview_csv),
                ],
            )
            run_py(
                script_dir / "qa_teacher_report.py",
                [
                    "--teacher_scores_jsonl",
                    str(out_jsonl),
                    "--output_json",
                    str(qa_json),
                    "--output_by_ar_csv",
                    str(qa_csv),
                ],
            )

    final_after = analyze_jsonl(
        out_jsonl,
        progress=progress_enabled,
        progress_every=progress_every,
        progress_min_seconds=progress_min_seconds,
    )
    info(
        "[repair] final(after) rows={} expensive_real_rows={} parse_error_rows={}".format(
            final_after.rows,
            final_after.expensive_real_rows,
            final_after.parse_error_rows,
        ),
        verbose=verbose,
    )
    progress_log(
        f"repair_teacher_outputs: finished | rows={final_after.rows} | expensive_real_rows={final_after.expensive_real_rows}",
        enabled=progress_enabled,
    )
    ensure_integrity(stats=final_after, expected_rows=expected_rows, context=str(out_jsonl))


if __name__ == "__main__":
    main()
