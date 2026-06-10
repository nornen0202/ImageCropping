#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from label_artifacts.paths import LABEL_JSON_DIRNAME, MULTIMODE_LABEL_FILENAMES
from multimode.explainability import (
    attach_explainability_to_attributes,
    explainability_policy_payload,
    summarize_explainability_annotations,
)
from multimode.coco_writer import build_target_ar_only_coco_dataset, write_json


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    row = dict(payload)
    row["last_update_time"] = _utc_timestamp()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_log(path: Path, event: str, payload: dict[str, Any] | None = None) -> None:
    row = dict(payload or {})
    row["event"] = event
    row["time"] = _utc_timestamp()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    tmp_size = tmp_path.stat().st_size
    os.replace(tmp_path, path)
    return {
        "label_json": str(path),
        "tmp_json": str(tmp_path),
        "size_bytes": path.stat().st_size,
        "tmp_size_bytes": tmp_size,
    }


def _annotation_target_ar(annotation: dict[str, Any]) -> str:
    attrs = annotation.get("attributes") if isinstance(annotation.get("attributes"), dict) else {}
    return str(attrs.get("target_ar") or annotation.get("target_ar") or "FREE")


def enrich_dataset(dataset: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    annotations = dataset.get("annotations")
    if not isinstance(annotations, list):
        raise ValueError("label JSON missing annotations[]")
    changed = 0
    missing_score_components = 0
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        attrs = annotation.get("attributes")
        if not isinstance(attrs, dict):
            attrs = {}
        if not isinstance(attrs.get("score_components"), dict):
            missing_score_components += 1
        enriched = attach_explainability_to_attributes(
            mode_name=str(annotation.get("mode_name") or attrs.get("mode_name") or ""),
            target_ar=_annotation_target_ar(annotation),
            score_mode=float(annotation.get("score_mode") or 0.0),
            attributes=attrs,
            hard_reject_reasons=attrs.get("hard_reject_reasons") or [],
        )
        annotation["attributes"] = enriched
        changed += 1
    summary = summarize_explainability_annotations(annotations)
    summary["changed_annotation_count"] = changed
    summary["missing_score_components_count"] = missing_score_components
    return dataset, summary


def _label_paths_for_run(run_dir: Path, splits: Sequence[str]) -> list[tuple[str, Path]]:
    label_dir = run_dir / LABEL_JSON_DIRNAME
    paths: list[tuple[str, Path]] = []
    for split in splits:
        key = (str(split), False)
        if key not in MULTIMODE_LABEL_FILENAMES:
            raise ValueError(f"unsupported split: {split}")
        paths.append((str(split), label_dir / MULTIMODE_LABEL_FILENAMES[key]))
    return paths


def process_run(
    run_dir: Path,
    *,
    splits: Sequence[str],
    refresh_target_ar_only: bool,
    status_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    started = _utc_timestamp()
    out_summary: dict[str, Any] = {
        "schema_version": "multimode_explainability_enrichment_v1",
        "run_dir": str(run_dir),
        "started_at": started,
        "refresh_target_ar_only": bool(refresh_target_ar_only),
        "files": {},
        "policy": explainability_policy_payload(),
    }
    _append_log(
        log_path,
        "started",
        {
            "run_dir": str(run_dir),
            "splits": list(splits),
            "refresh_target_ar_only": bool(refresh_target_ar_only),
            "pid": os.getpid(),
        },
    )
    _write_status(
        status_path,
        {
            "phase": "started",
            "run_dir": str(run_dir),
            "splits": list(splits),
            "pid": os.getpid(),
            "log_jsonl": str(log_path),
        },
    )
    for split, label_path in _label_paths_for_run(run_dir, splits):
        if not label_path.is_file():
            raise FileNotFoundError(f"label JSON not found: {label_path}")
        _append_log(log_path, "reading", {"split": split, "label_json": str(label_path), "size_bytes": label_path.stat().st_size})
        _write_status(
            status_path,
            {
                "phase": "reading",
                "split": split,
                "label_json": str(label_path),
                "pid": os.getpid(),
                "log_jsonl": str(log_path),
            },
        )
        dataset = _read_json(label_path)
        _write_status(
            status_path,
            {
                "phase": "enriching",
                "split": split,
                "label_json": str(label_path),
                "annotation_count": len(dataset.get("annotations", []) or []),
                "pid": os.getpid(),
                "log_jsonl": str(log_path),
            },
        )
        _append_log(log_path, "enriching", {"split": split, "annotation_count": len(dataset.get("annotations", []) or [])})
        dataset, file_summary = enrich_dataset(dataset)
        tmp_path = label_path.with_suffix(label_path.suffix + ".tmp")
        _write_status(
            status_path,
            {
                "phase": "writing",
                "split": split,
                "label_json": str(label_path),
                "tmp_json": str(tmp_path),
                "pid": os.getpid(),
                "log_jsonl": str(log_path),
            },
        )
        _append_log(log_path, "writing", {"split": split, "label_json": str(label_path), "tmp_json": str(tmp_path)})
        write_info = _atomic_write_json(label_path, dataset)
        _append_log(log_path, "wrote_label_json", {"split": split, **write_info})
        if refresh_target_ar_only:
            target_key = (split, True)
            target_path = run_dir / LABEL_JSON_DIRNAME / MULTIMODE_LABEL_FILENAMES[target_key]
            _write_status(
                status_path,
                {
                    "phase": "writing_target_ar_only",
                    "split": split,
                    "label_json": str(target_path),
                    "pid": os.getpid(),
                    "log_jsonl": str(log_path),
                },
            )
            _append_log(log_path, "writing_target_ar_only", {"split": split, "label_json": str(target_path)})
            write_json(target_path, build_target_ar_only_coco_dataset(dataset))
            file_summary["target_ar_only_refreshed"] = True
            file_summary["target_ar_only_label_json"] = str(target_path)
            file_summary["target_ar_only_size_bytes"] = target_path.stat().st_size
            _append_log(
                log_path,
                "wrote_target_ar_only",
                {"split": split, "label_json": str(target_path), "size_bytes": target_path.stat().st_size},
            )
        else:
            file_summary["target_ar_only_refreshed"] = False
        file_summary["label_json"] = str(label_path)
        file_summary["size_bytes"] = label_path.stat().st_size
        out_summary["files"][split] = file_summary
        _append_log(log_path, "completed_split", {"split": split, "label_json": str(label_path), "size_bytes": label_path.stat().st_size})
    out_summary["completed_at"] = _utc_timestamp()
    summary_path = run_dir / "explainability_enrichment_summary.json"
    policy_path = run_dir / "explainability_label_policy.json"
    summary_path.write_text(json.dumps(out_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    policy_path.write_text(json.dumps(explainability_policy_payload(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_status(
        status_path,
        {
            "phase": "completed",
            "run_dir": str(run_dir),
            "summary_json": str(summary_path),
            "policy_json": str(policy_path),
            "log_jsonl": str(log_path),
            "pid": os.getpid(),
        },
    )
    _append_log(log_path, "completed", {"run_dir": str(run_dir), "summary_json": str(summary_path), "policy_json": str(policy_path)})
    return out_summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add explainability checklist labels/scores to existing multimode label JSON files.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--splits", default="full,train,val", help="Comma-separated label splits to enrich. target_ar_only files are not enriched.")
    parser.add_argument("--refresh_target_ar_only", type=int, default=1, help="Regenerate target_ar_only JSON files after enrichment while keeping only target_ar attributes.")
    parser.add_argument("--status_json", default="")
    parser.add_argument("--log_jsonl", default="")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    splits = [chunk.strip() for chunk in str(args.splits).split(",") if chunk.strip()]
    status_path = Path(args.status_json) if args.status_json else run_dir / "explainability_enrichment_status.json"
    log_path = Path(args.log_jsonl) if args.log_jsonl else run_dir / "explainability_enrichment.log.jsonl"
    try:
        process_run(
            run_dir,
            splits=splits,
            refresh_target_ar_only=bool(int(args.refresh_target_ar_only)),
            status_path=status_path,
            log_path=log_path,
        )
    except Exception as exc:
        _append_log(
            log_path,
            "failed",
            {
                "run_dir": str(run_dir),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "pid": os.getpid(),
            },
        )
        _write_status(
            status_path,
            {
                "phase": "failed",
                "run_dir": str(run_dir),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "log_jsonl": str(log_path),
                "pid": os.getpid(),
            },
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
