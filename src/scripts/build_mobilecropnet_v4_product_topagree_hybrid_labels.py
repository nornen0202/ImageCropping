#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


DECISION_TO_ID = {"keep_full": 0, "minimal_crop": 1, "crop": 2}
RISK_KEYS = ("is_hard_negative", "is_unsafe_negative", "is_ignore_candidate", "is_overflow_candidate")
SCORE_COPY_KEYS = (
    "score_targets",
    "macro_targets",
    "macro_components",
    "composition_focus",
    "checklist_labels",
    "checklist_scores",
    "applicable_mask",
    "observed_mask",
    "derived_checklist_targets",
    "why_tags",
    "why_text_template",
    "external_score_teacher",
    "label_source_blend",
    "training_bucket",
    "candidate_weight",
    "score_prob",
    "crop_utility_prob",
    "score_rank_pct",
    "crop_utility_rank_pct",
    "rank_pct",
    "teacher_softmax_local",
    "crop_utility_softmax_local",
    "score_softmax_local",
    "pseudo_mos_1to5",
    "is_safe_high_score_leftover",
    "safe_leftover_policy_state",
    "safe_leftover_policy",
    *RISK_KEYS,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def row_key(row: dict[str, Any], fallback_index: int = 0) -> tuple[str, str, str]:
    return (
        str(row.get("image_id") or row.get("image_path") or fallback_index),
        str(row.get("target_ar") or row.get("target_ar_id") or "FREE"),
        str(row.get("schema_version") or ""),
    )


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except (TypeError, ValueError):
        return float(default)


def clip01(value: Any, default: float = 0.0) -> float:
    return max(0.0, min(1.0, safe_float(value, default)))


def candidate_id(candidate: dict[str, Any] | None) -> str:
    if not isinstance(candidate, dict):
        return ""
    return str(candidate.get("candidate_id") or candidate.get("target_id") or "").strip()


def clamp_box(box: Sequence[Any] | None) -> list[float]:
    if not box or len(box) < 4:
        return [0.0, 0.0, 1.0, 1.0]
    x1, y1, x2, y2 = [clip01(v) for v in box[:4]]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 <= x1:
        x2 = min(1.0, x1 + 1e-4)
    if y2 <= y1:
        y2 = min(1.0, y1 + 1e-4)
    return [x1, y1, x2, y2]


def candidate_box(candidate: dict[str, Any] | None) -> list[float]:
    if not isinstance(candidate, dict):
        return [0.0, 0.0, 1.0, 1.0]
    if isinstance(candidate.get("bbox_norm_xyxy"), list):
        return clamp_box(candidate.get("bbox_norm_xyxy"))
    if isinstance(candidate.get("bbox_cxcywh"), list) and len(candidate["bbox_cxcywh"]) >= 4:
        cx, cy, w, h = [safe_float(v) for v in candidate["bbox_cxcywh"][:4]]
        return clamp_box([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h])
    return [0.0, 0.0, 1.0, 1.0]


def box_area(box: Sequence[Any] | None) -> float:
    x1, y1, x2, y2 = clamp_box(box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(a: Sequence[Any] | None, b: Sequence[Any] | None) -> float:
    ax1, ay1, ax2, ay2 = clamp_box(a)
    bx1, by1, bx2, by2 = clamp_box(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area([ax1, ay1, ax2, ay2]) + box_area([bx1, by1, bx2, by2]) - inter
    return inter / max(1e-9, union)


def score(candidate: dict[str, Any] | None) -> float:
    if not isinstance(candidate, dict):
        return 0.0
    for key in ("crop_utility_prob", "score_prob", "crop_utility_rank_pct", "score_rank_pct", "rank_pct"):
        if candidate.get(key) is not None:
            return clip01(candidate.get(key))
        targets = candidate.get("score_targets")
        if isinstance(targets, dict) and targets.get(key) is not None:
            return clip01(targets.get(key))
    return 0.0


def decision_label(row: dict[str, Any]) -> str:
    decision = row.get("decision_target") if isinstance(row.get("decision_target"), dict) else {}
    raw = decision.get("decision_type", decision.get("decision_id", "crop"))
    if isinstance(raw, str) and raw in DECISION_TO_ID:
        return raw
    try:
        raw_id = int(raw)
    except (TypeError, ValueError):
        raw_id = 2
    for label, label_id in DECISION_TO_ID.items():
        if label_id == raw_id:
            return label
    return "crop"


def candidate_sources(row: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for field in ("matching_targets", "candidate_pool", "ignored_candidates", "overflow_candidates"):
        for candidate in row.get(field) or []:
            if isinstance(candidate, dict):
                out.append(candidate)
    return out


def candidate_map(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for candidate in candidate_sources(row):
        cid = candidate_id(candidate)
        if cid and cid not in out:
            out[cid] = candidate
    return out


def is_risky(candidate: dict[str, Any] | None) -> bool:
    return bool(isinstance(candidate, dict) and any(bool(candidate.get(key)) for key in RISK_KEYS))


def baseline_like(row: dict[str, Any], candidate: dict[str, Any] | None) -> bool:
    baseline = row.get("baseline") if isinstance(row.get("baseline"), dict) else None
    if not isinstance(candidate, dict):
        return False
    baseline_id = candidate_id(baseline)
    return bool(
        (baseline_id and candidate_id(candidate) == baseline_id)
        or (baseline is not None and box_iou(candidate_box(candidate), candidate_box(baseline)) >= 0.995)
    )


def top_blend_candidate(base_row: dict[str, Any], blend_row: dict[str, Any]) -> dict[str, Any] | None:
    candidates = candidate_sources(blend_row)
    safe = [candidate for candidate in candidates if not is_risky(candidate)]
    pool = safe or candidates
    non_base = [candidate for candidate in pool if not baseline_like(base_row, candidate)]
    pool = non_base or pool
    return max(pool, key=lambda candidate: (score(candidate), candidate_id(candidate)), default=None)


def copy_topagree_targets(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key in SCORE_COPY_KEYS:
        if key in src:
            dst[key] = deepcopy(src[key])


def update_positive_flags(row: dict[str, Any], positive_ids: set[str]) -> None:
    primary_id = next(iter(positive_ids), "")
    for field in ("candidate_pool", "ignored_candidates", "overflow_candidates"):
        for candidate in row.get(field) or []:
            if not isinstance(candidate, dict):
                continue
            cid = candidate_id(candidate)
            candidate["is_positive_candidate"] = bool(cid and cid == primary_id)
            candidate["is_soft_positive"] = bool(cid and cid in positive_ids and cid != primary_id)


def build_positive_targets(
    out_row: dict[str, Any],
    blend_row: dict[str, Any],
    *,
    top_id: str,
    soft_positive_margin: float,
    matching_top_k: int,
) -> list[dict[str, Any]]:
    by_id = candidate_map(out_row)
    blend_candidates = [
        candidate
        for candidate in candidate_sources(blend_row)
        if candidate_id(candidate) in by_id and not is_risky(candidate) and not baseline_like(out_row, candidate)
    ]
    max_score = score(by_id.get(top_id))
    ordered = sorted(
        blend_candidates,
        key=lambda candidate: (candidate_id(candidate) == top_id, score(candidate), candidate_id(candidate)),
        reverse=True,
    )
    positives: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in ordered:
        cid = candidate_id(candidate)
        if not cid or cid in seen:
            continue
        if cid != top_id and score(by_id.get(cid)) < max_score - max(0.0, soft_positive_margin):
            continue
        base_candidate = deepcopy(by_id[cid])
        base_candidate["is_positive_candidate"] = cid == top_id
        base_candidate["is_soft_positive"] = cid != top_id
        positives.append(base_candidate)
        seen.add(cid)
        if len(positives) >= max(1, int(matching_top_k)):
            break
    if not positives and top_id in by_id:
        top = deepcopy(by_id[top_id])
        top["is_positive_candidate"] = True
        top["is_soft_positive"] = False
        positives.append(top)
    return positives


def merge_crop_row(
    base_row: dict[str, Any],
    blend_row: dict[str, Any] | None,
    *,
    matching_top_k: int,
    soft_positive_margin: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out = deepcopy(base_row)
    stats = {
        "merged": False,
        "missing_blend": blend_row is None,
        "top_missing": False,
        "top_baseline_like": False,
        "top_not_in_base_pool": False,
        "winner_changed": False,
    }
    if blend_row is None:
        return out, stats
    top = top_blend_candidate(out, blend_row)
    if not isinstance(top, dict):
        stats["top_missing"] = True
        return out, stats
    if baseline_like(out, top):
        stats["top_baseline_like"] = True
        return out, stats
    top_id = candidate_id(top)
    by_id = candidate_map(out)
    if top_id not in by_id:
        stats["top_not_in_base_pool"] = True
        return out, stats

    blend_by_id = candidate_map(blend_row)
    for field in ("candidate_pool", "matching_targets", "ignored_candidates", "overflow_candidates"):
        for candidate in out.get(field) or []:
            if not isinstance(candidate, dict):
                continue
            cid = candidate_id(candidate)
            src = blend_by_id.get(cid)
            if src is not None and not baseline_like(out, candidate):
                copy_topagree_targets(candidate, src)

    positives = build_positive_targets(
        out,
        blend_row,
        top_id=top_id,
        soft_positive_margin=soft_positive_margin,
        matching_top_k=matching_top_k,
    )
    positive_ids = {candidate_id(candidate) for candidate in positives if candidate_id(candidate)}
    update_positive_flags(out, positive_ids)
    if positives:
        out["matching_targets"] = positives

    decision = dict(out.get("decision_target") if isinstance(out.get("decision_target"), dict) else {})
    previous_winner = str(decision.get("winner_post_gate_candidate_id") or "")
    top_candidate = candidate_map(out).get(top_id, top)
    decision["decision_type"] = "crop"
    decision["decision_id"] = DECISION_TO_ID["crop"]
    decision["winner_pre_gate_candidate_id"] = top_id
    decision["winner_post_gate_candidate_id"] = top_id
    decision["winner_pre_gate_bbox"] = candidate_box(top_candidate)
    decision["winner_post_gate_bbox"] = candidate_box(top_candidate)
    decision["chosen_score_rank"] = score(top_candidate)
    decision["chosen_crop_utility_raw"] = score(top_candidate)
    decision["external_score_source"] = "product_topagree_hybrid_public_uctr"
    decision["product_topagree_hybrid"] = {
        "schema_version": "mobilecropnet_v4_product_topagree_hybrid_v1",
        "previous_winner_post_gate_candidate_id": previous_winner,
        "topagree_winner_candidate_id": top_id,
        "winner_changed": previous_winner != top_id,
        "blend_row_source": (blend_row.get("label_source_blend") or {}).get("row_mode"),
    }
    out["decision_target"] = decision
    label_generation = dict(out.get("label_generation") if isinstance(out.get("label_generation"), dict) else {})
    label_generation["product_topagree_hybrid"] = {
        "enabled": True,
        "mode": "preserve_keep_full_minimal_crop_merge_crop_scores",
        "matching_top_k": max(1, int(matching_top_k)),
        "soft_positive_margin": float(soft_positive_margin),
    }
    out["label_generation"] = label_generation
    stats["merged"] = True
    stats["winner_changed"] = previous_winner != top_id
    return out, stats


def build_split(
    *,
    base_path: Path,
    blend_path: Path,
    output_path: Path,
    split_name: str,
    status_path: Path,
    matching_top_k: int,
    soft_positive_margin: float,
    status_every_rows: int,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        status_path,
        {
            "state": "running",
            "phase": split_name,
            "base_path": str(base_path),
            "blend_path": str(blend_path),
            "output_path": str(output_path),
            "rows": 0,
            "updated_at": utc_now(),
        },
    )
    blend_index: dict[tuple[str, str, str], dict[str, Any]] = {
        row_key(row, idx): row for idx, row in enumerate(read_jsonl(blend_path))
    }
    counts: Counter[str] = Counter()
    with output_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(read_jsonl(base_path)):
            counts["rows"] += 1
            key = row_key(row, idx)
            label = decision_label(row)
            counts[f"input_decision::{label}"] += 1
            if label != "crop":
                out = row
                counts["preserved_policy_rows"] += 1
                stats = {"merged": False}
            else:
                blend_row = blend_index.get(key)
                counts["crop_rows"] += 1
                counts["matched_blend_rows"] += int(blend_row is not None)
                out, stats = merge_crop_row(
                    row,
                    blend_row,
                    matching_top_k=matching_top_k,
                    soft_positive_margin=soft_positive_margin,
                )
            out_label = decision_label(out)
            counts[f"output_decision::{out_label}"] += 1
            for stat_key, value in stats.items():
                counts[f"stat::{stat_key}"] += int(bool(value))
            handle.write(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n")
            if status_every_rows > 0 and counts["rows"] % int(status_every_rows) == 0:
                write_json(
                    status_path,
                    {
                        "state": "running",
                        "phase": split_name,
                        "rows": counts["rows"],
                        "crop_rows": counts["crop_rows"],
                        "merged_crop_rows": counts["stat::merged"],
                        "preserved_policy_rows": counts["preserved_policy_rows"],
                        "updated_at": utc_now(),
                    },
                )
    rows = max(1, counts["rows"])
    crop_rows = max(1, counts["crop_rows"])
    return {
        "base_path": str(base_path),
        "blend_path": str(blend_path),
        "output_path": str(output_path),
        "rows": counts["rows"],
        "preserved_policy_rows": counts["preserved_policy_rows"],
        "preserved_policy_rate": counts["preserved_policy_rows"] / rows,
        "crop_rows": counts["crop_rows"],
        "matched_blend_rows": counts["matched_blend_rows"],
        "merged_crop_rows": counts["stat::merged"],
        "merged_crop_rate": counts["stat::merged"] / crop_rows,
        "winner_changed_rows": counts["stat::winner_changed"],
        "top_missing_rows": counts["stat::top_missing"],
        "top_baseline_like_rows": counts["stat::top_baseline_like"],
        "top_not_in_base_pool_rows": counts["stat::top_not_in_base_pool"],
        "input_decision_counts": {
            key.split("::", 1)[1]: value for key, value in counts.items() if key.startswith("input_decision::")
        },
        "output_decision_counts": {
            key.split("::", 1)[1]: value for key, value in counts.items() if key.startswith("output_decision::")
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build MobileCropNet v4 product/top-agree hybrid labels: preserve keep_full/minimal_crop "
            "rows from a product-consistent base set and merge public/UCTR top-agree crop score targets "
            "only into crop rows."
        )
    )
    parser.add_argument("--base_train_jsonl", required=True, type=Path)
    parser.add_argument("--base_val_jsonl", required=True, type=Path)
    parser.add_argument("--blend_train_jsonl", required=True, type=Path)
    parser.add_argument("--blend_val_jsonl", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--matching_top_k", type=int, default=2)
    parser.add_argument("--soft_positive_margin", type=float, default=0.08)
    parser.add_argument("--status_every_rows", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "status.json"
    train_path = args.output_dir / "train_product_topagree_hybrid.jsonl"
    val_path = args.output_dir / "val_product_topagree_hybrid.jsonl"
    train_summary = build_split(
        base_path=args.base_train_jsonl,
        blend_path=args.blend_train_jsonl,
        output_path=train_path,
        split_name="train",
        status_path=status_path,
        matching_top_k=max(1, int(args.matching_top_k)),
        soft_positive_margin=float(args.soft_positive_margin),
        status_every_rows=max(0, int(args.status_every_rows)),
    )
    val_summary = build_split(
        base_path=args.base_val_jsonl,
        blend_path=args.blend_val_jsonl,
        output_path=val_path,
        split_name="val",
        status_path=status_path,
        matching_top_k=max(1, int(args.matching_top_k)),
        soft_positive_margin=float(args.soft_positive_margin),
        status_every_rows=max(0, int(args.status_every_rows)),
    )
    summary = {
        "schema_version": "mobilecropnet_v4_product_topagree_hybrid_labels_v1",
        "purpose_ko": "Product-AR baseline/minimal 정책 타깃을 보존하면서 crop row의 ranking/checklist/why/risk supervision만 UCTR/public top-agree 라벨로 교체한다.",
        "matching_top_k": max(1, int(args.matching_top_k)),
        "soft_positive_margin": float(args.soft_positive_margin),
        "train": train_summary,
        "val": val_summary,
        "artifacts": {
            "train_jsonl": str(train_path),
            "val_jsonl": str(val_path),
            "summary_json": str(args.output_dir / "summary.json"),
            "status_json": str(status_path),
        },
        "updated_at": utc_now(),
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(
        status_path,
        {
            "state": "succeeded",
            "phase": "complete",
            "rows": {"train": train_summary["rows"], "val": val_summary["rows"]},
            "summary_json": str(args.output_dir / "summary.json"),
            "updated_at": utc_now(),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
