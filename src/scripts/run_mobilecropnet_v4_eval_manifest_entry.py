#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MobileCropNet v4 shortlist/evaluation manifest row 하나를 final bundle로 실행한다."
    )
    parser.add_argument("--manifest_json", type=Path, required=True)
    parser.add_argument("--entry_index", type=int, default=None, help="1-based manifest entry index")
    parser.add_argument("--run_name", type=str, default=None, help="manifest run_name으로 row 선택")
    parser.add_argument("--root", type=Path, default=Path("/group-volume/users/jaden.ju/Sources/ImageCropping"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--public_batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--direct_batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--direct_selection_policy", type=str, default="proposal_topk_rerank")
    parser.add_argument("--direct_proposal_top_m", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _select_entry(entries: list[dict[str, Any]], entry_index: int | None, run_name: str | None) -> dict[str, Any]:
    if entry_index is None and not run_name:
        raise SystemExit("Either --entry_index or --run_name is required.")
    if entry_index is not None:
        idx = entry_index - 1
        if idx < 0 or idx >= len(entries):
            raise SystemExit(f"--entry_index out of range: {entry_index} / {len(entries)}")
        return entries[idx]
    matches = [entry for entry in entries if entry.get("run_name") == run_name]
    if not matches:
        raise SystemExit(f"run_name not found in manifest: {run_name}")
    if len(matches) > 1:
        raise SystemExit(f"run_name is ambiguous in manifest: {run_name}")
    return matches[0]


def _require_file(root: Path, path_text: str, label: str) -> Path:
    path = Path(path_text)
    full = path if path.is_absolute() else root / path
    if not full.exists():
        raise SystemExit(f"Missing {label}: {full}")
    return full


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    manifest_path = args.manifest_json if args.manifest_json.is_absolute() else root / args.manifest_json
    manifest = _read_json(manifest_path)
    entries = manifest.get("entries") or []
    entry = _select_entry(entries, args.entry_index, args.run_name)

    output_dir = Path(entry["output_dir"])
    output_dir = output_dir if output_dir.is_absolute() else root / output_dir
    summary_path = output_dir / "final_eval_bundle_summary.json"
    if summary_path.exists() and not args.force:
        print(json.dumps({"state": "skipped", "reason": "summary_exists", "summary_json": str(summary_path)}, ensure_ascii=False))
        return 0

    checkpoint = _require_file(root, entry["checkpoint"], "checkpoint")
    val_jsonl = _require_file(root, entry["val_jsonl"], "val_jsonl")
    test_jsonl = _require_file(root, entry["test_jsonl"], "test_jsonl")
    public_manifest = _require_file(root, entry["public_task_manifest_jsonl"], "public_task_manifest_jsonl")
    bundle_script = root / "src" / "scripts" / "run_mobilecropnet_v4_shortlist_eval_bundle.sh"
    if not bundle_script.exists():
        alt = root / "src" / "run_mobilecropnet_v4_shortlist_eval_bundle.sh"
        bundle_script = alt if alt.exists() else bundle_script
    if not bundle_script.exists():
        raise SystemExit(f"Missing bundle script: {bundle_script}")

    output_dir.mkdir(parents=True, exist_ok=True)
    launcher_summary = {
        "state": "launching",
        "entry_index": args.entry_index,
        "track": entry.get("track"),
        "variant": entry.get("variant"),
        "profile": entry.get("profile"),
        "run_name": entry.get("run_name"),
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "method_id": entry.get("method_id"),
    }
    (output_dir / "launcher_summary.json").write_text(
        json.dumps(launcher_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(launcher_summary, ensure_ascii=False, indent=2), flush=True)

    env = os.environ.copy()
    env.update(
        {
            "ROOT": str(root),
            "CHECKPOINT": str(checkpoint),
            "VAL_JSONL": str(val_jsonl),
            "TEST_JSONL": str(test_jsonl),
            "PUBLIC_TASK_MANIFEST_JSONL": str(public_manifest),
            "METHOD_ID": str(entry["method_id"]),
            "OUTPUT_DIR": str(output_dir),
            "DEVICE": args.device,
            "PUBLIC_BATCH_SIZE": str(args.public_batch_size),
            "EVAL_BATCH_SIZE": str(args.eval_batch_size),
            "DIRECT_BATCH_SIZE": str(args.direct_batch_size),
            "NUM_WORKERS": str(args.num_workers),
            "DIRECT_SELECTION_POLICY": args.direct_selection_policy,
            "DIRECT_PROPOSAL_TOP_M": str(args.direct_proposal_top_m),
        }
    )
    cmd = ["bash", str(bundle_script)]
    return subprocess.call(cmd, cwd=str(root), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
