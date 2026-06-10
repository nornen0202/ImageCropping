#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MobileCropNet v4 composite runtime 정성 리뷰 팩을 생성한다.")
    parser.add_argument("--direct_predictions_jsonl", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--route_expert_checkpoint", type=Path, default=None)
    parser.add_argument("--route_expert_checkpoints", type=Path, nargs="*", default=None)
    parser.add_argument("--route_expert_weights", default=None)
    parser.add_argument("--subject_valid_threshold", type=float, default=None)
    parser.add_argument("--max_per_bucket", type=int, default=6)
    parser.add_argument("--max_total", type=int, default=24)
    parser.add_argument("--render_overlays", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _slug(value: Any) -> str:
    out = []
    for ch in str(value or "").lower():
        out.append(ch if ch.isalnum() else "-")
    slug = "".join(out)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-") or "item"


def _failure_buckets(row: dict[str, Any]) -> list[str]:
    metrics = row.get("metrics") or {}
    buckets: list[str] = []
    if _f(metrics.get("final_positive_hit_iou_0_5")) < 0.5:
        buckets.append("final_not_positive")
    if _f(metrics.get("final_risk_hit_iou_0_5")) >= 0.5:
        buckets.append("risk_hit")
    if _f(metrics.get("proposal_target_ar_recall_at_5_iou_0_5")) < 0.5:
        buckets.append("proposal_top5_miss")
    if _f(metrics.get("subject_box_target_valid")) >= 0.5 and _f(metrics.get("subject_box_iou_to_teacher"), 1.0) < 0.5:
        buckets.append("subject_iou_low")
    if _f(metrics.get("subject_box_valid_acc"), 1.0) < 0.5:
        buckets.append("subject_valid_mismatch")
    if _f(metrics.get("final_target_ar_compatible"), 1.0) < 0.5:
        buckets.append("target_ar_incompatible")
    return buckets


def _severity(row: dict[str, Any]) -> float:
    metrics = row.get("metrics") or {}
    return (
        (1.0 - _f(metrics.get("final_best_iou_to_positive"))) * 3.0
        + _f(metrics.get("final_risk_hit_iou_0_5")) * 2.0
        + (1.0 - _f(metrics.get("proposal_target_ar_best_iou_at_5"))) * 1.0
        + (1.0 - _f(metrics.get("subject_box_iou_to_teacher"), 1.0)) * 0.75
    )


def _select(rows: list[dict[str, Any]], *, max_per_bucket: int, max_total: int) -> list[dict[str, Any]]:
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets = _failure_buckets(row)
        if not buckets:
            continue
        enriched = dict(row)
        enriched["failure_buckets"] = buckets
        enriched["severity"] = _severity(row)
        for bucket in buckets:
            by_bucket[bucket].append(enriched)
    for bucket_rows in by_bucket.values():
        bucket_rows.sort(key=lambda item: _f(item.get("severity")), reverse=True)

    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for bucket in sorted(by_bucket):
        added = 0
        for row in by_bucket[bucket]:
            key = (str(row.get("image_id") or ""), str(row.get("target_ar") or ""))
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
            added += 1
            if added >= max_per_bucket or len(selected) >= max_total:
                break
        if len(selected) >= max_total:
            break
    selected.sort(key=lambda item: _f(item.get("severity")), reverse=True)
    return selected[:max_total]


def _render_overlay(args: argparse.Namespace, sample: dict[str, Any], idx: int) -> tuple[str | None, str | None]:
    if not args.render_overlays or args.checkpoint is None:
        return None, None
    image_path = Path(str(sample.get("image_path") or ""))
    if not image_path.exists():
        return None, None
    target_ar = str(sample.get("target_ar") or "FREE")
    stem = f"{idx:02d}_{_slug(sample.get('image_id'))}_{_slug(target_ar)}"
    output_json = args.output_dir / "inference_json" / f"{stem}.json"
    output_png = args.output_dir / "overlays" / f"{stem}.png"
    cmd = [
        sys.executable,
        "src/scripts/infer_mobilecropnet_v4.py",
        "--checkpoint",
        str(args.checkpoint),
        "--image",
        str(image_path),
        "--target_ar",
        target_ar,
        "--output_json",
        str(output_json),
        "--output_png",
        str(output_png),
        "--selection_policy",
        "proposal_topk_rerank",
        "--proposal_top_m",
        "8",
        "--exact_target_ar_postprocess",
        "--device",
        str(args.device),
    ]
    if args.subject_valid_threshold is not None:
        cmd.extend(["--subject_valid_threshold", str(args.subject_valid_threshold)])
    if args.route_expert_checkpoint is not None:
        cmd.extend(["--route_expert_checkpoint", str(args.route_expert_checkpoint)])
    if args.route_expert_checkpoints:
        cmd.append("--route_expert_checkpoints")
        cmd.extend(str(path) for path in args.route_expert_checkpoints)
    if args.route_expert_weights:
        cmd.extend(["--route_expert_weights", str(args.route_expert_weights)])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    return str(output_json), str(output_png)


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# MobileCropNet v4 Composite 정성 리뷰 팩",
        "",
        f"- direct predictions: `{payload.get('direct_predictions_jsonl')}`",
        f"- 전체 rows: `{payload.get('row_count')}`",
        f"- 리뷰 샘플: `{payload.get('sample_count')}`",
        "",
        "## 실패 버킷 요약",
        "",
            "| 버킷 | 건수 | 비율 |",
        "| --- | ---: | ---: |",
    ]
    for key, row in sorted((payload.get("failure_summary") or {}).items()):
        lines.append(f"| `{key}` | {row['count']} | {row['rate']:.6f} |")
    lines.extend(
        [
            "",
            "## 리뷰 샘플",
            "",
            "| 버킷 | 심각도 | 이미지 | target_ar | final_hit | risk_hit | proposal_r5 | subject_iou | overlay |",
            "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for sample in payload.get("samples", []):
        metrics = sample.get("metrics") or {}
        overlay = sample.get("overlay_path")
        overlay_md = f"[overlay]({overlay})" if overlay else "-"
        lines.append(
            f"| {', '.join(sample.get('failure_buckets') or [])} | {_f(sample.get('severity')):.3f} | "
            f"`{sample.get('image_id')}` | `{sample.get('target_ar')}` | "
            f"{_f(metrics.get('final_positive_hit_iou_0_5')):.3f} | {_f(metrics.get('final_risk_hit_iou_0_5')):.3f} | "
            f"{_f(metrics.get('proposal_target_ar_recall_at_5_iou_0_5')):.3f} | {_f(metrics.get('subject_box_iou_to_teacher')):.3f} | {overlay_md} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(args.direct_predictions_jsonl)
    selected = _select(rows, max_per_bucket=int(args.max_per_bucket), max_total=int(args.max_total))
    failure_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        for bucket in _failure_buckets(row):
            failure_counts[bucket] += 1
    samples = []
    for idx, sample in enumerate(selected, start=1):
        out = {
            "image_id": sample.get("image_id"),
            "image_path": sample.get("image_path"),
            "target_ar": sample.get("target_ar"),
            "failure_buckets": sample.get("failure_buckets"),
            "severity": sample.get("severity"),
            "selected": sample.get("selected"),
            "subject_box": sample.get("subject_box"),
            "metrics": sample.get("metrics"),
        }
        inference_json, overlay_path = _render_overlay(args, sample, idx)
        out["inference_json"] = inference_json
        out["overlay_path"] = overlay_path
        samples.append(out)
    payload = {
        "direct_predictions_jsonl": str(args.direct_predictions_jsonl),
        "row_count": len(rows),
        "sample_count": len(samples),
        "failure_summary": {
            key: {"count": int(count), "rate": float(count / max(1, len(rows)))}
            for key, count in sorted(failure_counts.items(), key=lambda item: (-item[1], item[0]))
        },
        "samples": samples,
    }
    (args.output_dir / "qualitative_review_pack.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(args.output_dir / "QUALITATIVE_REVIEW_PACK.md", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
