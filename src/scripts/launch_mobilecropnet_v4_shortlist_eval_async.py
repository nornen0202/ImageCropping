#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_GROUP_ID = "SR-AR-Interactive-Media-Exp"
DEFAULT_EXPERIMENT_ID = "918"
DEFAULT_IMAGE = "sr-ar-interactive-media-exp/jy-cropping-260415-tfs4.57.6-rsync"
DEFAULT_ROOT = "/group-volume/users/jaden.ju/Sources/ImageCropping"
ACTIVE_STATUSES = {"Pending", "Running"}
TERMINAL_STATUSES = {"Succeeded", "Failed", "Stopped", "RunContainerError"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch and maintain async shortlist final-eval workloads.")
    parser.add_argument("--manifest_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--group_id", default=DEFAULT_GROUP_ID)
    parser.add_argument("--experiment_id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--core_id", default="2")
    parser.add_argument("--expected_run_time", default="4h")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--max_pending_sec", type=int, default=600)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _slug(text: str) -> str:
    out = []
    for ch in str(text).lower():
        out.append(ch if ch.isalnum() else "-")
    slug = "".join(out)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def _now_tag() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def _space(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["space", *args], check=check, text=True, capture_output=True)


def _get_run_payload(run_id: str) -> dict[str, Any]:
    proc = _space(["mlp", "get", "run", run_id], check=False)
    payload: dict[str, Any] = {"status": "", "register_time": "", "start_time": "", "end_time": "", "raw": proc.stdout}
    for line in proc.stdout.splitlines():
        if "|" not in line:
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) != 2:
            continue
        key, value = parts
        key = key.lower().replace(" ", "_")
        if key == "status":
            payload["status"] = value
        elif key == "register_time":
            payload["register_time"] = value
        elif key == "start_time":
            payload["start_time"] = value
        elif key == "end_time":
            payload["end_time"] = value
        elif key == "container_name":
            payload["run_name"] = value
    return payload


def _extract_run_id(text: str) -> str:
    matches = re.findall(r"\b\d{6,}\b", text or "")
    return matches[0] if matches else ""


def _resolve_run_id_by_name(run_name: str) -> str:
    proc = _space(["mlp", "list", "run", f"--group-id={DEFAULT_GROUP_ID}"], check=False)
    for line in proc.stdout.splitlines():
        if str(run_name) not in line:
            continue
        resolved = _extract_run_id(line)
        if resolved:
            return resolved
    return ""


def _parse_iso8601_epoch(value: str) -> float | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return time.mktime(time.strptime(raw, "%Y-%m-%dT%H:%M:%S.000Z"))
    except ValueError:
        try:
            return time.mktime(time.strptime(raw, "%Y-%m-%dT%H:%M:%SZ"))
        except ValueError:
            return None


def _remove_run(run_id: str) -> None:
    _space(["mlp", "remove", "run", run_id], check=False)


def _entry_key(entry: dict[str, Any]) -> str:
    return f"{entry['track']}__{entry['profile']}"


def _build_exec_command(entry: dict[str, Any], root: str) -> str:
    env_bits = [
        f"ROOT='{root}'",
        f"CHECKPOINT='{entry['checkpoint']}'",
        f"VAL_JSONL='{entry['val_jsonl']}'",
        f"TEST_JSONL='{entry['test_jsonl']}'",
        f"PUBLIC_TASK_MANIFEST_JSONL='{entry['public_task_manifest_jsonl']}'",
        f"METHOD_ID='{entry['method_id']}'",
        f"OUTPUT_DIR='{entry['output_dir']}'",
        "DIRECT_SELECTION_POLICY='proposal_topk_rerank'",
        "DIRECT_PROPOSAL_TOP_M='8'",
    ]
    return f"cd {root} && " + " ".join(env_bits) + " bash src/scripts/run_mobilecropnet_v4_shortlist_eval_bundle.sh"


def _compact_track_slug(track: str) -> str:
    words = [word for word in _slug(track).split("-") if word]
    initials = "".join(word[0] for word in words[:6])
    return initials[:8] or "trk"


def _create_run(entry: dict[str, Any], *, args: argparse.Namespace, attempt: int) -> dict[str, Any]:
    run_name = f"mfe-{_compact_track_slug(entry['track'])}-{_slug(entry['profile'])[:18]}-r{attempt}-{_now_tag()}"
    exec_command = _build_exec_command(entry, args.root)
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
        ],
        check=False,
    )
    run_id = _extract_run_id(proc.stdout)
    if proc.returncode != 0 or not run_id:
        raise RuntimeError(
            "space mlp create run failed for {track}/{profile}: rc={rc} stdout={stdout!r} stderr={stderr!r}".format(
                track=entry["track"],
                profile=entry["profile"],
                rc=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )
        )
    return {
        "attempt": attempt,
        "run_id": run_id,
        "run_name": run_name,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()),
        "exec_command": exec_command,
        "status": "created",
    }


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(args.manifest_json)
    state_path = args.output_dir / "shortlist_eval_async_state.json"
    state = _read_json(state_path)
    state.setdefault("generated_at", time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()))
    state.setdefault("manifest_json", str(args.manifest_json))
    state.setdefault("entries", {})

    launched_count = 0
    pending_cutoff = float(args.max_pending_sec)
    now = time.time()
    for entry in manifest.get("entries", []):
        key = _entry_key(entry)
        row = state["entries"].setdefault(
            key,
            {
                "track": entry["track"],
                "profile": entry["profile"],
                "variant": entry.get("variant"),
                "run_dir": entry["run_dir"],
                "output_dir": entry["output_dir"],
                "attempts": [],
            },
        )
        summary_path = Path(entry["output_dir"]) / "final_eval_bundle_summary.json"
        if summary_path.exists():
            row["final_status"] = "completed"
            row["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(summary_path.stat().st_mtime))
            continue

        active_attempt: dict[str, Any] | None = None
        for attempt in row["attempts"]:
            if not str(attempt.get("run_id") or "").isdigit():
                resolved = _resolve_run_id_by_name(str(attempt.get("run_name") or ""))
                if resolved:
                    attempt["run_id"] = resolved
                else:
                    attempt["status"] = attempt.get("status") or "unresolved_run_id"
                    continue
            payload = _get_run_payload(str(attempt["run_id"]))
            attempt["status"] = payload.get("status") or attempt.get("status") or ""
            attempt["register_time"] = payload.get("register_time") or attempt.get("register_time") or ""
            attempt["start_time"] = payload.get("start_time") or attempt.get("start_time") or ""
            attempt["end_time"] = payload.get("end_time") or attempt.get("end_time") or ""
            if attempt["status"] == "Pending":
                register_epoch = _parse_iso8601_epoch(attempt.get("register_time", ""))
                if register_epoch is not None and now - register_epoch > pending_cutoff:
                    _remove_run(str(attempt["run_id"]))
                    attempt["status"] = "removed_stuck_pending"
                    continue
            if attempt["status"] in ACTIVE_STATUSES:
                active_attempt = attempt
                break

        if active_attempt is not None:
            row["final_status"] = active_attempt["status"]
            continue

        attempts_used = len(row["attempts"])
        if attempts_used >= int(args.max_retries):
            row["final_status"] = "retry_exhausted"
            continue

        try:
            new_attempt = _create_run(entry, args=args, attempt=attempts_used + 1)
        except Exception as exc:  # noqa: BLE001
            row["final_status"] = "launch_error"
            row["launch_error"] = str(exc)
            continue
        else:
            row["attempts"].append(new_attempt)
            row["final_status"] = "launched"
            launched_count += 1
        if args.limit is not None and launched_count >= int(args.limit):
            break

    state["last_updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
    _write_json(state_path, state)
    print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
