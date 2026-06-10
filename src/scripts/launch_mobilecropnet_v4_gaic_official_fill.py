#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="누락된 MobileCropNet v4 GAIC official 평가를 병렬 보강한다.")
    parser.add_argument("--manifest_json", type=Path, required=True)
    parser.add_argument("--indices", nargs="+", type=int, required=True, help="1-based manifest row indices.")
    parser.add_argument("--status_json", type=Path, required=True)
    parser.add_argument("--project_root", type=Path, default=Path("."))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_parallel", type=int, default=8)
    parser.add_argument("--threads_per_job", type=int, default=8)
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    return parser


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")[:160]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _public_job(entry: dict[str, Any], index: int, root: Path, args: argparse.Namespace) -> dict[str, Any]:
    out_dir = root / entry["output_dir"] / "gaic_official_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = out_dir / "metrics.json"
    progress = out_dir / "progress.json"
    log = out_dir / "run.log"
    model_name = _safe_name(f"mcn_v4_{entry.get('profile')}_{entry.get('run_name')}")
    cmd = [
        "/usr/local/bin/python3",
        "src/scripts/evaluate_mobilecropnet_v4_gaic_benchmark.py",
        "--checkpoint",
        entry["checkpoint"],
        "--output_dir",
        str(out_dir.relative_to(root)),
        "--model_name",
        model_name,
        "--target_ar",
        "FREE",
        "--device",
        str(args.device),
        "--progress_json",
        str(progress.relative_to(root)),
        "--progress_every",
        str(args.progress_every),
    ]
    state = "pending"
    if metrics.exists() and not args.force:
        state = "skipped_existing"
    return {
        "index": index,
        "track": entry.get("track"),
        "variant": entry.get("variant"),
        "profile": entry.get("profile"),
        "run_name": entry.get("run_name"),
        "checkpoint": entry.get("checkpoint"),
        "output_dir": str(out_dir),
        "metrics_json": str(metrics),
        "progress_json": str(progress),
        "log": str(log),
        "model_name": model_name,
        "cmd": cmd,
        "state": state,
    }


def _summarize_jobs(jobs: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for job in jobs:
        state = str(job.get("state") or "unknown")
        counts[state] = counts.get(state, 0) + 1
    return counts


def main() -> int:
    args = build_parser().parse_args()
    root = args.project_root.resolve()
    payload = _read_json(root / args.manifest_json if not args.manifest_json.is_absolute() else args.manifest_json)
    entries = payload.get("entries", payload if isinstance(payload, list) else [])
    jobs = [_public_job(entries[index - 1], index, root, args) for index in args.indices]

    started_at = time.time()
    _write_json(args.status_json, {"state": "running", "started_at_unix": started_at, "counts": _summarize_jobs(jobs), "jobs": jobs})

    env_base = os.environ.copy()
    thread_count = str(max(1, int(args.threads_per_job)))
    env_base.update(
        {
            "OMP_NUM_THREADS": thread_count,
            "MKL_NUM_THREADS": thread_count,
            "OPENBLAS_NUM_THREADS": thread_count,
            "NUMEXPR_NUM_THREADS": thread_count,
        }
    )
    active: list[tuple[dict[str, Any], subprocess.Popen[Any], Any]] = []
    pending = [job for job in jobs if job["state"] == "pending"]
    completed = 0
    failed = 0

    while pending or active:
        while pending and len(active) < max(1, int(args.max_parallel)):
            job = pending.pop(0)
            log_fh = Path(job["log"]).open("w", encoding="utf-8")
            proc = subprocess.Popen(job["cmd"], cwd=root, env=env_base, stdout=log_fh, stderr=subprocess.STDOUT)
            job["pid"] = proc.pid
            job["state"] = "running"
            job["started_at_unix"] = time.time()
            active.append((job, proc, log_fh))

        next_active: list[tuple[dict[str, Any], subprocess.Popen[Any], Any]] = []
        for job, proc, log_fh in active:
            rc = proc.poll()
            if rc is None:
                next_active.append((job, proc, log_fh))
                continue
            log_fh.close()
            job["returncode"] = rc
            job["completed_at_unix"] = time.time()
            job["state"] = "completed" if rc == 0 else "failed"
            completed += int(rc == 0)
            failed += int(rc != 0)
        active = next_active
        _write_json(
            args.status_json,
            {
                "state": "running" if active or pending else ("completed" if failed == 0 else "failed"),
                "started_at_unix": started_at,
                "updated_at_unix": time.time(),
                "completed_count": completed,
                "failed_count": failed,
                "counts": _summarize_jobs(jobs),
                "jobs": jobs,
            },
        )
        if active or pending:
            time.sleep(5.0)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
