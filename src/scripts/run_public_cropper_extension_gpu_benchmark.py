#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_log(log_path: Path, text: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        if text and not text.endswith("\n"):
            handle.write("\n")


def _run_phase(
    *,
    name: str,
    command: list[str],
    project_root: Path,
    run_dir: Path,
    status: dict[str, Any],
) -> None:
    phase_log = run_dir / "logs" / f"{name}.log"
    status["phase"] = name
    status["phase_started_at"] = _utc_now()
    status["last_update_at"] = status["phase_started_at"]
    status.setdefault("phases", {})[name] = {
        "status": "running",
        "started_at": status["phase_started_at"],
        "command": " ".join(shlex.quote(part) for part in command),
        "log": str(phase_log),
    }
    _write_json(run_dir / "status.json", status)
    _append_log(phase_log, f"$ {' '.join(shlex.quote(part) for part in command)}\n")

    started = time.time()
    with phase_log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=project_root,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            elapsed = time.time() - started
            phase_payload = status["phases"][name]
            phase_payload["elapsed_sec"] = round(elapsed, 3)
            phase_payload["log_size_bytes"] = phase_log.stat().st_size if phase_log.exists() else 0
            phase_payload["last_heartbeat_at"] = _utc_now()
            status["last_update_at"] = phase_payload["last_heartbeat_at"]
            _write_json(run_dir / "status.json", status)
            time.sleep(30.0)
    elapsed = time.time() - started

    phase_payload = status["phases"][name]
    phase_payload["status"] = "succeeded" if process.returncode == 0 else "failed"
    phase_payload["returncode"] = process.returncode
    phase_payload["ended_at"] = _utc_now()
    phase_payload["elapsed_sec"] = elapsed
    status["last_update_at"] = phase_payload["ended_at"]
    _write_json(run_dir / "status.json", status)
    if process.returncode != 0:
        raise RuntimeError(f"Phase {name} failed with exit code {process.returncode}; see {phase_log}")


def _build_summary(run_dir: Path, status: dict[str, Any]) -> dict[str, Any]:
    prediction_dir = run_dir / "public_benchmark_predictions"
    gaic_dir = run_dir / "gaic_v2_test500"
    prediction_summary = prediction_dir / "public_cropper_benchmark_predictions_summary.json"
    gaic_summary = gaic_dir / "metrics.json"
    summary = {
        "status": status.get("status"),
        "run_dir": str(run_dir),
        "status_json": str(run_dir / "status.json"),
        "prediction_summary_json": str(prediction_summary),
        "gaic_v2_metrics_json": str(gaic_summary),
        "prediction_summary_exists": prediction_summary.exists(),
        "gaic_v2_metrics_exists": gaic_summary.exists(),
        "completed_at": _utc_now(),
    }
    if prediction_summary.exists():
        summary["prediction_summary"] = json.loads(prediction_summary.read_text(encoding="utf-8"))
    if gaic_summary.exists():
        summary["gaic_v2_metrics"] = json.loads(gaic_summary.read_text(encoding="utf-8"))
    _write_json(run_dir / "gpu_benchmark_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPU-bound CACNet/S2CNet public cropper extension scoring.")
    parser.add_argument("--project_root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument(
        "--task_manifest_jsonl",
        type=Path,
        default=Path("artifacts/unified_public_benchmark_20260420/stage_full/benchmark_task_manifest.jsonl"),
    )
    parser.add_argument("--methods", nargs="+", default=["cacnet", "s2cnet"], choices=["cacnet", "s2cnet"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python_bin", default=sys.executable)
    parser.add_argument("--max_tasks", type=int, default=0)
    parser.add_argument("--max_gaic_images", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    status: dict[str, Any] = {
        "status": "running",
        "started_at": _utc_now(),
        "last_update_at": _utc_now(),
        "project_root": str(project_root),
        "run_dir": str(run_dir),
        "methods": args.methods,
        "device": args.device,
        "pid": os.getpid(),
    }
    _write_json(run_dir / "status.json", status)

    try:
        _run_phase(
            name="env_probe",
            command=[
                args.python_bin,
                "-c",
                (
                    "import json, subprocess, torch; "
                    "gpu=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,driver_version',"
                    "'--format=csv,noheader'],capture_output=True,text=True); "
                    "print(json.dumps({'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),"
                    "'device_count':torch.cuda.device_count(),'gpu':gpu.stdout.strip()},ensure_ascii=False))"
                ),
            ],
            project_root=project_root,
            run_dir=run_dir,
            status=status,
        )
        _run_phase(
            name="py_compile",
            command=[
                args.python_bin,
                "-m",
                "py_compile",
                "src/scripts/evaluate_public_croppers_gaic_v2.py",
                "src/scripts/export_public_cropper_benchmark_predictions.py",
            ],
            project_root=project_root,
            run_dir=run_dir,
            status=status,
        )

        export_command = [
            args.python_bin,
            "src/scripts/export_public_cropper_benchmark_predictions.py",
            "--task_manifest_jsonl",
            str(args.task_manifest_jsonl),
            "--output_dir",
            str(run_dir / "public_benchmark_predictions"),
            "--methods",
            *args.methods,
            "--device",
            args.device,
            "--group_by_image",
            "1",
            "--strict",
        ]
        if args.max_tasks > 0:
            export_command.extend(["--max_tasks", str(args.max_tasks)])
        _run_phase(
            name="export_public_benchmark_predictions",
            command=export_command,
            project_root=project_root,
            run_dir=run_dir,
            status=status,
        )

        gaic_command = [
            args.python_bin,
            "src/scripts/evaluate_public_croppers_gaic_v2.py",
            "--output_dir",
            str(run_dir / "gaic_v2_test500"),
            "--methods",
            *args.methods,
            "--device",
            args.device,
            "--strict",
        ]
        if args.max_gaic_images > 0:
            gaic_command.extend(["--max_images", str(args.max_gaic_images)])
        _run_phase(
            name="evaluate_gaic_v2_test500",
            command=gaic_command,
            project_root=project_root,
            run_dir=run_dir,
            status=status,
        )
        status["status"] = "succeeded"
    except Exception as exc:
        status["status"] = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["last_update_at"] = _utc_now()
        _write_json(run_dir / "status.json", status)
        _build_summary(run_dir, status)
        raise

    status["ended_at"] = _utc_now()
    status["last_update_at"] = status["ended_at"]
    _write_json(run_dir / "status.json", status)
    _build_summary(run_dir, status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
