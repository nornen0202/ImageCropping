#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="최종 shortlist artifact에서 MobileCropNet v4 정성 리뷰 팩을 생성한다.")
    parser.add_argument("--leaderboard_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--candidate_limit", type=int, default=3)
    parser.add_argument("--max_rows_per_candidate", type=int, default=40)
    parser.add_argument("--max_total_rows", type=int, default=100)
    parser.add_argument("--min_severity", type=float, default=0.0)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _f(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _slug(text: Any) -> str:
    out = []
    for ch in str(text or "").lower():
        out.append(ch if ch.isalnum() else "-")
    slug = "".join(out)
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug.strip("-")


def _row_rank_key(row: dict[str, Any]) -> tuple[int, float, float, float]:
    return (
        1 if row.get("final_gate_pass") else 0,
        _f(row.get("equal4_zscore")) or -1e9,
        _f(row.get("head_sanity_score")) or -1e9,
        _f(row.get("direct_alignment_score")) or -1e9,
    )


def _failure_buckets(row: dict[str, Any]) -> list[str]:
    buckets: list[str] = []
    if row.get("teacher_subject_mode") and row.get("pred_subject_mode") and row.get("teacher_subject_mode") != row.get("pred_subject_mode"):
        buckets.append("route_mismatch")
    if row.get("teacher_decision") and row.get("pred_decision") and row.get("teacher_decision") != row.get("pred_decision"):
        buckets.append("decision_mismatch")
    proposal_r5 = _f(row.get("proposal_recall_at_5_iou_0_5"))
    if proposal_r5 is not None and proposal_r5 < 0.5:
        buckets.append("low_proposal_recall")
    top1_iou = _f(row.get("top1_iou_to_best_positive"))
    if top1_iou is not None and top1_iou < 0.5:
        buckets.append("low_top1_iou")
    prop_subj = _f(row.get("proposal_to_subject_iou"))
    selected_subj = _f(row.get("selected_to_subject_iou"))
    if (prop_subj is not None and prop_subj < 0.5) or (selected_subj is not None and selected_subj < 0.5):
        buckets.append("low_subject_iou")
    explain = _f(row.get("explain_label_agreement_when_applicable"))
    if explain is not None and explain < 0.5:
        buckets.append("checklist_disagreement")
    risk_tags = {str(item.get("tag")) for item in (row.get("model_why_tags") or []) if isinstance(item, dict)}
    if risk_tags.intersection({"subject_poor", "headroom_violation", "context_loss", "tight_crop"}):
        buckets.append("model_rationale_warning")
    return buckets or ["high_severity"]


def _find_existing_overlay(run_dir: Path, image_id: Any, target_ar: Any) -> str | None:
    overlay_dir = run_dir / "viz_test" / "overlays"
    if not overlay_dir.exists():
        return None
    image_slug = _slug(image_id)
    ar_slug = _slug(target_ar or "FREE")
    matches = sorted(path for path in overlay_dir.glob("*.png") if image_slug in _slug(path.name) and ar_slug in _slug(path.name))
    if matches:
        return str(matches[0])
    matches = sorted(path for path in overlay_dir.glob("*.png") if image_slug in _slug(path.name))
    return str(matches[0]) if matches else None


def _select_failures(rows: list[dict[str, Any]], *, max_rows: int, min_severity: float) -> list[dict[str, Any]]:
    by_bucket: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        severity = _f(row.get("severity")) or 0.0
        if severity < min_severity:
            continue
        enriched = dict(row)
        enriched["failure_buckets"] = _failure_buckets(row)
        for bucket in enriched["failure_buckets"]:
            by_bucket.setdefault(bucket, []).append(enriched)
    for bucket_rows in by_bucket.values():
        bucket_rows.sort(key=lambda item: _f(item.get("severity")) or 0.0, reverse=True)

    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    while len(selected) < max_rows:
        added = False
        for bucket in sorted(by_bucket):
            if not by_bucket[bucket] or len(selected) >= max_rows:
                continue
            row = by_bucket[bucket].pop(0)
            key = (str(row.get("image_id") or ""), str(row.get("target_ar") or ""))
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
            added = True
        if not added:
            break
    selected.sort(key=lambda item: _f(item.get("severity")) or 0.0, reverse=True)
    return selected[:max_rows]


def _candidate_rows(payload: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    rows = [row for row in payload.get("rows", []) if isinstance(row, dict)]
    rows.sort(key=_row_rank_key, reverse=True)
    return rows[: max(0, int(limit))]


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    def fmt(value: Any, digits: int = 4) -> str:
        number = _f(value)
        return "-" if number is None else f"{number:.{digits}f}"

    lines = [
        "# MobileCropNet v4 정성 리뷰 팩",
        "",
        f"- 리더보드 JSON: `{payload.get('leaderboard_json')}`",
        f"- 후보 수: `{payload.get('candidate_count')}`",
        f"- 샘플 수: `{payload.get('sample_count')}`",
        "",
        "## 후보 요약",
        "",
        "| 트랙 | 프로파일 | equal4_z | head | direct | route_bal | route_collapse | 평가 경로 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for cand in payload.get("candidates", []):
        lines.append(
            f"| {cand.get('track')} | {cand.get('profile')} | {fmt(cand.get('equal4_zscore'))} | "
            f"{fmt(cand.get('head_sanity_score'))} | {fmt(cand.get('direct_alignment_score'))} | "
            f"{fmt(cand.get('route_balanced_accuracy'))} | {cand.get('route_collapse_flag')} | `{cand.get('eval_output_dir')}` |"
        )
    lines.extend(
        [
            "",
            "## 리뷰 샘플",
            "",
            "| 후보 | 실패 유형 | 심각도 | image | target_ar | route | decision | prop_r5 | subj_iou | top1_iou | 오버레이/원본 |",
            "| --- | --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for sample in payload.get("samples", []):
        link = sample.get("overlay_path") or sample.get("image_path") or ""
        link_text = "overlay" if sample.get("overlay_path") else "원본"
        link_md = f"[{link_text}]({link})" if link else "-"
        route = f"{sample.get('teacher_subject_mode') or '-'} -> {sample.get('pred_subject_mode') or '-'}"
        decision = f"{sample.get('teacher_decision') or '-'} -> {sample.get('pred_decision') or '-'}"
        lines.append(
            f"| {sample.get('candidate_id')} | {', '.join(sample.get('failure_buckets') or [])} | {fmt(sample.get('severity'), 2)} | "
            f"`{sample.get('image_id')}` | `{sample.get('target_ar')}` | {route} | {decision} | "
            f"{fmt(sample.get('proposal_recall_at_5_iou_0_5'))} | {fmt(sample.get('selected_to_subject_iou'))} | "
            f"{fmt(sample.get('top1_iou_to_best_positive'))} | {link_md} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    leaderboard = _read_json(args.leaderboard_json)
    candidates = _candidate_rows(leaderboard, args.candidate_limit)

    review_candidates: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    remaining_total = max(0, int(args.max_total_rows))
    for cand in candidates:
        eval_dir = Path(str(cand.get("eval_output_dir") or ""))
        run_dir = Path(str(cand.get("run_dir") or ""))
        failures_path = eval_dir / "head_analysis" / "head_analysis_failures.jsonl"
        selected = _select_failures(
            _read_jsonl(failures_path),
            max_rows=min(int(args.max_rows_per_candidate), remaining_total),
            min_severity=float(args.min_severity),
        )
        candidate_id = f"{_slug(cand.get('track'))}__{_slug(cand.get('profile'))}"
        review_candidates.append(
            {
                "candidate_id": candidate_id,
                "track": cand.get("track"),
                "profile": cand.get("profile"),
                "run_name": cand.get("run_name"),
                "run_dir": cand.get("run_dir"),
                "eval_output_dir": cand.get("eval_output_dir"),
                "equal4_zscore": cand.get("equal4_zscore"),
                "head_sanity_score": cand.get("head_sanity_score"),
                "direct_alignment_score": cand.get("direct_alignment_score"),
                "route_balanced_accuracy": cand.get("route_balanced_accuracy"),
                "route_collapse_flag": cand.get("route_collapse_flag"),
                "failure_source": str(failures_path),
                "selected_failure_count": len(selected),
            }
        )
        for row in selected:
            out = {
                "candidate_id": candidate_id,
                "track": cand.get("track"),
                "profile": cand.get("profile"),
                **row,
            }
            out["overlay_path"] = _find_existing_overlay(run_dir, out.get("image_id"), out.get("target_ar"))
            samples.append(out)
        remaining_total -= len(selected)
        if remaining_total <= 0:
            break

    payload = {
        "leaderboard_json": str(args.leaderboard_json),
        "candidate_count": len(review_candidates),
        "sample_count": len(samples),
        "candidates": review_candidates,
        "samples": samples,
    }
    (args.output_dir / "qualitative_review_pack.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(args.output_dir / "QUALITATIVE_REVIEW_PACK.md", payload)
    print(json.dumps({"output_dir": str(args.output_dir), "candidate_count": len(review_candidates), "sample_count": len(samples)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
