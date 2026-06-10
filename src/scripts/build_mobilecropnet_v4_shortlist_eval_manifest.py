#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build shortlist evaluation manifest for MobileCropNet v4 final candidate runs.")
    parser.add_argument("--interim_deployment_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
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


def _append_manual_track_entries(rows: list[dict[str, Any]], *, seen: set[tuple[str, str]]) -> None:
    by_track: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_track.setdefault(str(row.get("track") or ""), row)

    manual_entries = [
        {
            "track": "SSTK T1",
            "variant": "baseline_current",
            "profile": "hybrid_384",
            "run_name": "mcn-sstk-t1-baseasync-hybrid-384-20260423-023818",
            "run_dir": "artifacts/mobilecropnet_v4/head_variant_async_20260423/sstk_t1_baseline_spot/mcn-sstk-t1-baseasync-hybrid-384-20260423-023818",
        },
        {
            "track": "SSTK T1",
            "variant": "baseline_current",
            "profile": "plus_384",
            "run_name": "mcn-sstk-t1-baseasync-r3-plus-384-20260423-031502",
            "run_dir": "artifacts/mobilecropnet_v4/head_variant_async_20260423/sstk_t1_baseline_spot/mcn-sstk-t1-baseasync-r3-plus-384-20260423-031502",
        },
    ]
    for manual in manual_entries:
        key = (manual["track"], manual["profile"])
        if key in seen:
            continue
        template = by_track.get(manual["track"])
        if template is None:
            continue
        run_dir = Path(manual["run_dir"])
        entry = {
            "track": manual["track"],
            "variant": manual["variant"],
            "profile": manual["profile"],
            "run_name": manual["run_name"],
            "run_dir": str(run_dir),
            "checkpoint": str(run_dir / "best.pt"),
            "val_jsonl": str(template.get("val_jsonl") or ""),
            "test_jsonl": str(template.get("test_jsonl") or ""),
            "public_task_manifest_jsonl": "artifacts/unified_public_benchmark_20260420/stage_full/benchmark_task_manifest.jsonl",
            "output_dir": str(
                Path("artifacts/mobilecropnet_v4/shortlist_final_eval_20260423")
                / _slug(manual["track"])
                / _slug(manual["profile"])
            ),
            "method_id": f"{_slug(manual['track'])}__{_slug(manual['profile'])}",
        }
        rows.append(entry)
        seen.add(key)


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = _read_json(args.interim_deployment_json)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for row in payload.get("shortlist_rows", []):
        track = str(row.get("track") or "")
        profile = str(row.get("profile") or "")
        key = (track, profile)
        if key in seen:
            continue
        seen.add(key)
        run_dir = Path(str(row.get("run_dir") or ""))
        config = _read_json(run_dir / "config.json")
        eval_test = _read_json(run_dir / "eval_test" / "metrics.json")
        entry = {
            "track": track,
            "variant": str(row.get("variant") or ""),
            "profile": profile,
            "run_name": str(row.get("run_name") or ""),
            "run_dir": str(run_dir),
            "checkpoint": str(row.get("checkpoint") or (run_dir / "best.pt")),
            "val_jsonl": str(config.get("val_jsonl") or ""),
            "test_jsonl": str(eval_test.get("eval_jsonl") or ""),
            "public_task_manifest_jsonl": "artifacts/unified_public_benchmark_20260420/stage_full/benchmark_task_manifest.jsonl",
            "output_dir": str(
                Path("artifacts/mobilecropnet_v4/shortlist_final_eval_20260423")
                / _slug(track)
                / _slug(profile)
            ),
            "method_id": f"{_slug(track)}__{_slug(profile)}",
        }
        rows.append(entry)

    _append_manual_track_entries(rows, seen=seen)

    rows.sort(key=lambda item: (item["track"], item["profile"]))
    manifest_json = {"entry_count": len(rows), "entries": rows}
    (args.output_dir / "shortlist_eval_manifest.json").write_text(
        json.dumps(manifest_json, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "shortlist_eval_manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "track",
            "variant",
            "profile",
            "run_name",
            "run_dir",
            "checkpoint",
            "val_jsonl",
            "test_jsonl",
            "public_task_manifest_jsonl",
            "output_dir",
            "method_id",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(json.dumps(manifest_json, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
