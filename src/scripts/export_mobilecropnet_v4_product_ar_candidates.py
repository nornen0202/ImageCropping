#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet_v4.data import target_ar_value


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except (TypeError, ValueError):
        return float(default)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(float(lo), min(float(hi), float(value)))


def clamp_box(box: Sequence[float] | None) -> list[float]:
    if not box or len(box) < 4:
        return [0.0, 0.0, 1.0, 1.0]
    x1, y1, x2, y2 = [clamp(safe_float(v), 0.0, 1.0) for v in box[:4]]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 <= x1:
        x2 = min(1.0, x1 + 1e-4)
    if y2 <= y1:
        y2 = min(1.0, y1 + 1e-4)
    return [x1, y1, x2, y2]


def box_ar(box: Sequence[float] | None) -> float:
    x1, y1, x2, y2 = clamp_box(box)
    return max(1e-6, x2 - x1) / max(1e-6, y2 - y1)


def score_value(candidate: dict[str, Any], key: str, default: float = 0.0) -> float:
    if candidate.get(key) is not None:
        return safe_float(candidate.get(key), default)
    targets = candidate.get("score_targets")
    if isinstance(targets, dict) and targets.get(key) is not None:
        return safe_float(targets.get(key), default)
    return float(default)


def first_score(candidate: dict[str, Any]) -> float:
    for key in ("crop_utility_prob", "score_prob", "score_rank_pct", "crop_utility_rank_pct"):
        value = score_value(candidate, key, None)  # type: ignore[arg-type]
        if value is not None:
            return clamp(float(value))
    return 0.0


def macro_value(candidate: dict[str, Any], *keys: str) -> float:
    for source_key in ("macro_targets", "macro_components"):
        source = candidate.get(source_key)
        if isinstance(source, dict):
            for key in keys:
                if source.get(key) is not None:
                    return safe_float(source.get(key), 0.0)
    return 0.0


def nested_value(candidate: dict[str, Any], source_key: str, *keys: str) -> float:
    source = candidate.get(source_key)
    if isinstance(source, dict):
        for key in keys:
            if source.get(key) is not None:
                return safe_float(source.get(key), 0.0)
    return 0.0


def resolve_path(path: str | Path, project_root: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return project_root / p


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def iter_batch_rows(path: Path, max_rows: int = 0) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if max_rows > 0 and idx >= max_rows:
                break
            if line.strip():
                yield json.loads(line)


def hard_reject_tags(candidate: dict[str, Any]) -> list[str]:
    tags = {str(v) for v in candidate.get("reject_tags") or [] if str(v)}
    if bool(candidate.get("is_hard_negative")):
        tags.add("sstk_hard_negative")
    if bool(candidate.get("is_unsafe_negative")):
        tags.add("sstk_unsafe_negative")
    if bool(candidate.get("is_overflow_candidate")):
        tags.add("sstk_overflow")
    safety = candidate.get("safety_penalty")
    if isinstance(safety, dict):
        if safe_float(safety.get("hard_total"), 0.0) > 0.0:
            tags.add("hard_safety_penalty")
    return sorted(tags)


def candidate_to_row(
    *,
    batch_row: dict[str, Any],
    candidate: dict[str, Any],
    role: str,
    candidate_index: int,
    project_root: Path,
) -> dict[str, Any]:
    target_ar = str(batch_row.get("target_ar", "FREE"))
    image_id = str(batch_row.get("image_id", ""))
    image_path = resolve_path(str(batch_row.get("image_path", "")), project_root)
    bbox = clamp_box(candidate.get("bbox_norm_xyxy"))
    expected_ar = target_ar_value(target_ar)
    candidate_ar = box_ar(bbox)
    ar_error = 0.0 if expected_ar is None else abs(math.log(max(candidate_ar, 1e-6) / max(float(expected_ar), 1e-6)))
    routing = batch_row.get("routing") if isinstance(batch_row.get("routing"), dict) else {}
    subject_overlay = routing.get("subject_support_overlay") if isinstance(routing.get("subject_support_overlay"), dict) else {}
    mode = str(routing.get("subject_mode") or candidate.get("subject_mode") or "")
    mode_bucket = str(routing.get("mode_bucket") or candidate.get("mode_bucket") or mode)
    reject_tags = hard_reject_tags(candidate)
    score_targets = candidate.get("score_targets") if isinstance(candidate.get("score_targets"), dict) else {}
    source_subject_prior_bbox = routing.get("subject_prior_bbox_norm_xyxy")
    if not isinstance(source_subject_prior_bbox, list):
        source_subject_prior_bbox = subject_overlay.get("bbox_norm_xyxy")
    return {
        "image_id": image_id,
        "score_group_id": f"{image_id}::{target_ar}",
        "target_ar": target_ar,
        "protocol": "ProductAR",
        "official_split": str(batch_row.get("label_generation", {}).get("split", "")) if isinstance(batch_row.get("label_generation"), dict) else "",
        "candidate_id": str(candidate.get("candidate_id") or f"{role}_{candidate_index:04d}"),
        "candidate_role": str(role),
        "candidate_index": int(candidate_index),
        "gt_flag": int(bool(role == "matching_target" or candidate.get("is_positive_candidate"))),
        "bbox_norm_xyxy": bbox,
        "mos": 0.0,
        "image_path": str(image_path),
        "subject_mode": mode,
        "mode_bucket": mode_bucket,
        "sstk_score_prob": clamp(score_value(candidate, "score_prob", first_score(candidate))),
        "sstk_crop_utility_prob": clamp(score_value(candidate, "crop_utility_prob", first_score(candidate))),
        "sstk_score_rank_pct": clamp(score_value(candidate, "score_rank_pct", score_value(candidate, "rank_pct", 0.0))),
        "sstk_crop_utility_rank_pct": clamp(score_value(candidate, "crop_utility_rank_pct", score_value(candidate, "rank_pct", 0.0))),
        "sstk_candidate_weight": clamp(safe_float(candidate.get("candidate_weight", candidate.get("training_weight", 1.0)), 1.0), 0.05, 1.0),
        "sstk_is_positive_candidate": bool(candidate.get("is_positive_candidate") or role == "matching_target"),
        "sstk_is_hard_negative": bool(candidate.get("is_hard_negative", False)),
        "sstk_is_unsafe_negative": bool(candidate.get("is_unsafe_negative", False)),
        "sstk_is_ignore_candidate": bool(candidate.get("is_ignore_candidate", False)),
        "sstk_is_overflow_candidate": bool(candidate.get("is_overflow_candidate", False)),
        "sstk_training_bucket": str(candidate.get("training_bucket", "")),
        "sstk_reject_tags": reject_tags,
        "sstk_hard_reject": bool(reject_tags),
        "target_ar_log_error": float(ar_error),
        "checklist_labels": candidate.get("checklist_labels", {}),
        "checklist_scores": candidate.get("checklist_scores", {}),
        "why_tags": list(candidate.get("why_tags") or []),
        "profile_A_macro": macro_value(candidate, "aesthetic_prob", "A_macro", "A", "aesthetic"),
        "profile_S_macro": macro_value(candidate, "subject_prob", "S_macro", "S", "subject"),
        "profile_C_macro": macro_value(candidate, "composition_prob", "C_macro", "C", "composition"),
        "profile_T_macro": macro_value(candidate, "technical_prob", "T_macro", "T", "technical"),
        "profile_A_active": 1.0,
        "profile_S_active": 1.0,
        "profile_C_active": 1.0,
        "profile_T_active": 1.0,
        "profile_area_log_prior": safe_float(score_targets.get("area_log_prior"), 0.0),
        "profile_safety_penalty_total": nested_value(candidate, "safety_penalty", "total", "soft_total"),
        "profile_safety_penalty_soft": nested_value(candidate, "safety_penalty", "soft_total", "soft"),
        "profile_safety_penalty_hard": nested_value(candidate, "safety_penalty", "hard_total", "hard"),
        "profile_component_aesthetic_norm": macro_value(candidate, "aesthetic_prob", "A_macro", "A", "aesthetic"),
        "profile_component_cosine_img_text": 0.0,
        "profile_component_cov": nested_value(candidate, "checklist_scores", "subject_coverage_ratio"),
        "profile_component_p_cut": nested_value(candidate, "safety_penalty", "subject_cut", "p_cut"),
        "profile_component_p_text": nested_value(candidate, "safety_penalty", "text_cut", "p_text"),
        "profile_component_p_ar_free": float(ar_error),
        "profile_component_r_comp": macro_value(candidate, "composition_prob", "C_macro", "C", "composition"),
        "profile_component_r_headroom": nested_value(candidate, "macro_components", "C_headroom"),
        "profile_component_r_lookroom": nested_value(candidate, "macro_components", "C_lookroom"),
        "profile_component_r_horizon": nested_value(candidate, "checklist_scores", "horizon_visible_ratio"),
        "profile_component_r_sym": nested_value(candidate, "checklist_scores", "symmetry_score"),
        "profile_component_r_context": nested_value(candidate, "macro_components", "C_context"),
        "profile_component_r_teach": first_score(candidate),
        "crop_utility_prob": clamp(score_value(candidate, "crop_utility_prob", first_score(candidate))),
        "score_prob": clamp(score_value(candidate, "score_prob", first_score(candidate))),
        "score_rank_pct": clamp(score_value(candidate, "score_rank_pct", score_value(candidate, "rank_pct", 0.0))),
        "crop_utility_rank_pct": clamp(score_value(candidate, "crop_utility_rank_pct", score_value(candidate, "rank_pct", 0.0))),
        "source_batch_schema_version": str(batch_row.get("schema_version", "")),
        "source_label_generation": batch_row.get("label_generation", {}),
        "source_decision_target": batch_row.get("decision_target", {}) if isinstance(batch_row.get("decision_target"), dict) else {},
        "source_baseline": batch_row.get("baseline", {}) if isinstance(batch_row.get("baseline"), dict) else {},
        "source_subject_mode": mode,
        "source_subject_mode_id": int(safe_float(routing.get("subject_mode_id"), 0)),
        "source_mode_bucket": mode_bucket,
        "source_policy_id": str(routing.get("policy_id") or ""),
        "source_router_rule_id": str(routing.get("router_rule_id") or routing.get("rule_id") or ""),
        "source_subject_prior_bbox_norm_xyxy": clamp_box(source_subject_prior_bbox) if isinstance(source_subject_prior_bbox, list) else None,
        "source_subject_reliability": clamp(
            safe_float(
                (routing.get("flags") if isinstance(routing.get("flags"), dict) else {}).get("subject_reliability"),
                safe_float(subject_overlay.get("subject_reliability"), 0.0),
            ),
            0.0,
            1.0,
        ),
    }


def iter_candidate_rows(
    batch_jsonl: Path,
    *,
    project_root: Path,
    include_ignored: bool,
    include_overflow: bool,
    max_rows: int,
) -> Iterable[dict[str, Any]]:
    for batch_row in iter_batch_rows(batch_jsonl, max_rows=max_rows):
        streams: list[tuple[str, list[dict[str, Any]]]] = [
            ("matching_target", list(batch_row.get("matching_targets") or [])),
            ("candidate_pool", list(batch_row.get("candidate_pool") or [])),
        ]
        if include_ignored:
            streams.append(("ignored_candidate", list(batch_row.get("ignored_candidates") or [])))
        if include_overflow:
            streams.append(("overflow_candidate", list(batch_row.get("overflow_candidates") or [])))
        idx = 0
        for role, candidates in streams:
            for candidate in candidates:
                yield candidate_to_row(
                    batch_row=batch_row,
                    candidate=candidate,
                    role=role,
                    candidate_index=idx,
                    project_root=project_root,
                )
                idx += 1


def summarize(path: Path) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    group_ids: set[str] = set()
    image_ids: set[str] = set()
    rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            counters[f"target_ar::{row.get('target_ar', '')}"] += 1
            counters[f"role::{row.get('candidate_role', '')}"] += 1
            if row.get("sstk_hard_reject"):
                counters["sstk_hard_reject"] += 1
            group_ids.add(str(row.get("score_group_id", "")))
            image_ids.add(str(row.get("image_id", "")))
    return {
        "row_count": rows,
        "image_count": len(image_ids),
        "group_count": len(group_ids),
        "counter": dict(sorted(counters.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flatten MobileCropNet v4 Product AR batch labels into candidate rows for external scoring.")
    parser.add_argument("--input_jsonl", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--include_ignored_candidates", action="store_true")
    parser.add_argument("--include_overflow_candidates", action="store_true")
    parser.add_argument("--max_rows", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    count = write_jsonl(
        args.output_jsonl,
        iter_candidate_rows(
            args.input_jsonl,
            project_root=args.project_root,
            include_ignored=bool(args.include_ignored_candidates),
            include_overflow=bool(args.include_overflow_candidates),
            max_rows=int(args.max_rows),
        ),
    )
    summary = {
        "status": "ok",
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "written_rows": int(count),
        "include_ignored_candidates": bool(args.include_ignored_candidates),
        "include_overflow_candidates": bool(args.include_overflow_candidates),
        **summarize(args.output_jsonl),
    }
    write_json(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
