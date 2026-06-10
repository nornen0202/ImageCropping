from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect async MobileCropNet v4 single-profile workload results.")
    parser.add_argument("--manifest_tsv", nargs="+", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--group_id", default="SR-AR-Interactive-Media-Exp")
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _space_run_status(group_id: str) -> dict[str, str]:
    proc = subprocess.run(
        ["space", "mlp", "list", "run", "--group-id", group_id],
        check=True,
        text=True,
        capture_output=True,
    )
    out: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "|" not in line or "RUN ID" in line or "---" in line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) < 6:
            continue
        run_id = parts[1]
        run_name = parts[2]
        status = parts[5]
        if run_name and status:
            out[run_name] = status
        if run_id and status:
            out[run_id] = status
    return out


def _load_manifests(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                item = dict(row)
                key = (item.get("run_name", ""), item.get("run_id", ""))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(item)
    return rows


def _classify_run(row: dict[str, str], run_status: dict[str, str]) -> dict[str, Any]:
    run_name = row["run_name"]
    out_base = Path(row["out_base"])
    run_dir = out_base / run_name
    run_summary = _read_json(run_dir / "run_summary.json")
    metrics = _read_json(run_dir / "metrics.json")
    dataset_summary = _read_json(run_dir / "dataset_summary.json")
    config = _read_json(run_dir / "config.json")
    status = "missing"
    if run_summary:
        status = "complete"
    elif (run_dir / "train.log").exists() or metrics:
        status = "partial"
    remote_status = run_status.get(run_name) or run_status.get(row.get("run_id", ""))
    return {
        "run_name": run_name,
        "run_id": row.get("run_id"),
        "profile": row["profile"],
        "variant": row["variant"],
        "track_name": row["track_name"],
        "expected_runtime": row["expected_runtime"],
        "run_dir": str(run_dir),
        "artifact_status": status,
        "remote_status": remote_status,
        "has_train_log": (run_dir / "train.log").exists(),
        "has_run_summary": bool(run_summary),
        "best_selection_score": run_summary.get("best_selection_score"),
        "best_epoch": run_summary.get("best_epoch"),
        "dataset_summary": dataset_summary,
        "config": config,
    }


def _preferred_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_profile: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_profile.setdefault(row["profile"], []).append(row)
    selected = []
    for profile, candidates in by_profile.items():
        candidates.sort(
            key=lambda row: (
                1 if row["artifact_status"] == "complete" else 0,
                1 if row.get("remote_status") == "Succeeded" else 0,
                row.get("best_selection_score") or float("-inf"),
                row["run_name"],
            ),
            reverse=True,
        )
        selected.append(candidates[0])
    selected.sort(key=lambda row: row["profile"])
    return selected


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Async Profile Workload Collection",
        "",
        "| profile | run_name | artifact_status | remote_status | best_epoch | best_selection_score |",
        "| --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in rows:
        score = row.get("best_selection_score")
        score_str = f"{float(score):.6f}" if score is not None else ""
        lines.append(
            f"| {row['profile']} | {row['run_name']} | {row['artifact_status']} | {row.get('remote_status') or ''} | "
            f"{row.get('best_epoch') or ''} | {score_str} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = _load_manifests(list(args.manifest_tsv))
    run_status = _space_run_status(args.group_id)
    collected = [_classify_run(row, run_status) for row in manifest_rows]
    selected = _preferred_rows(collected)

    payload = {
        "manifest_tsvs": [str(path) for path in args.manifest_tsv],
        "row_count": len(collected),
        "selected_profile_count": len(selected),
        "runs": collected,
        "selected_runs": selected,
    }
    (args.output_dir / "async_profile_collection.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(args.output_dir / "async_profile_collection.md", selected)

    complete_run_dirs = [Path(row["run_dir"]) for row in selected if row["artifact_status"] == "complete"]
    if complete_run_dirs:
        subprocess.run(
            [
                "/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/python",
                "src/scripts/summarize_mobilecropnet_v4_sstk_product_runs.py",
                "--run_dirs",
                *[str(path) for path in complete_run_dirs],
                "--output_dir",
                str(args.output_dir / "summary_complete"),
            ],
            check=True,
        )

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
