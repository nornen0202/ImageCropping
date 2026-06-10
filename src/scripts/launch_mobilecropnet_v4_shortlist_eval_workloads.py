#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_GROUP_ID = "SR-AR-Interactive-Media-Exp"
DEFAULT_EXPERIMENT_ID = "918"
DEFAULT_IMAGE = "sr-ar-interactive-media-exp/jy-cropping-260415-tfs4.57.6-rsync"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch MobileCropNet v4 shortlist evaluation workloads.")
    parser.add_argument("--manifest_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--group_id", default=DEFAULT_GROUP_ID)
    parser.add_argument("--experiment_id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--core_id", default="2")
    parser.add_argument("--expected_run_time", default="4h")
    parser.add_argument("--max_pending_sec", type=int, default=600)
    parser.add_argument("--max_retries", type=int, default=3)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _slug(text: str) -> str:
    out = []
    for ch in str(text).lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("-")
    slug = "".join(out)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def _now_tag() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def _space(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["space", *args], check=check, text=True, capture_output=True)


def _create_run(entry: dict[str, Any], *, args: argparse.Namespace, attempt: int) -> tuple[str, str]:
    run_name = f"mcn-feval-{_slug(entry['track'])}-{_slug(entry['profile'])}-r{attempt}-{_now_tag()}"
    env_bits = [
        f"ROOT=/group-volume/users/jaden.ju/Sources/ImageCropping",
        f"CHECKPOINT='{entry['checkpoint']}'",
        f"VAL_JSONL='{entry['val_jsonl']}'",
        f"TEST_JSONL='{entry['test_jsonl']}'",
        f"PUBLIC_TASK_MANIFEST_JSONL='{entry['public_task_manifest_jsonl']}'",
        f"METHOD_ID='{entry['method_id']}'",
        f"OUTPUT_DIR='{entry['output_dir']}'",
        "DIRECT_SELECTION_POLICY='proposal_topk_rerank'",
        "DIRECT_PROPOSAL_TOP_M='8'",
    ]
    exec_command = "cd /group-volume/users/jaden.ju/Sources/ImageCropping && " + " ".join(env_bits) + " bash src/scripts/run_mobilecropnet_v4_shortlist_eval_bundle.sh"
    proc = _space(
        [
            "mlp",
            "create",
            "run",
            run_name,
            "-c",
            args.core_id,
            "--core-count=1",
            f"--experiment-id={args.experiment_id}",
            f"--image={args.image}",
            f"--exec-command={exec_command}",
            f"--group-id={args.group_id}",
            "--allow-spot",
            f"--expected-run-time={args.expected_run_time}",
        ]
    )
    run_id = proc.stdout.strip().splitlines()[-1].strip()
    return run_id, run_name


def _get_status(run_id: str) -> str:
    proc = _space(["mlp", "get", "run", run_id], check=False)
    text = proc.stdout
    for label in ("Current Phase", "Status"):
        for line in text.splitlines():
            if label in line:
                parts = [part.strip() for part in line.split(":", 1)]
                if len(parts) == 2 and parts[1]:
                    return parts[1]
    return ""


def _remove_run(run_id: str) -> None:
    _space(["mlp", "remove", "run", run_id], check=False)


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(args.manifest_json)
    launched: list[dict[str, Any]] = []

    for entry in manifest.get("entries", []):
        success = False
        attempts: list[dict[str, Any]] = []
        for attempt in range(1, int(args.max_retries) + 1):
            run_id, run_name = _create_run(entry, args=args, attempt=attempt)
            started = time.time()
            status = ""
            while time.time() - started < float(args.max_pending_sec):
                time.sleep(20)
                status = _get_status(run_id)
                if status and status not in {"Pending"}:
                    break
            final_status = status or _get_status(run_id)
            attempts.append(
                {
                    "attempt": attempt,
                    "run_id": run_id,
                    "run_name": run_name,
                    "status_after_pending_window": final_status,
                }
            )
            if final_status in {"Running", "Succeeded", "Failed", "Stopped", "RunContainerError"}:
                if final_status == "Pending":
                    _remove_run(run_id)
                    continue
                if final_status == "Running" or final_status == "Succeeded":
                    success = True
                    break
                _remove_run(run_id)
                continue
            if final_status == "Pending" or not final_status:
                _remove_run(run_id)
                continue
        launched.append(
            {
                "entry": entry,
                "success": success,
                "attempts": attempts,
            }
        )

    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()),
        "manifest_json": str(args.manifest_json),
        "group_id": args.group_id,
        "experiment_id": args.experiment_id,
        "image": args.image,
        "launches": launched,
    }
    (args.output_dir / "shortlist_eval_launches.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
