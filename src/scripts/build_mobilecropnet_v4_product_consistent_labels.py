#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


DECISION_TO_ID = {"keep_full": 0, "minimal_crop": 1, "crop": 2}
SCORE_KEYS = (
    "crop_utility_prob",
    "score_prob",
    "crop_utility_rank_pct",
    "score_rank_pct",
    "rank_pct",
    "teacher_softmax_local",
    "crop_utility_softmax_local",
    "score_softmax_local",
    "pseudo_mos_1to5",
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


def read_jsonl(path: Path, *, max_rows: int = 0) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if max_rows > 0 and idx >= max_rows:
                break
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def candidate_id(candidate: dict[str, Any] | None) -> str:
    if not isinstance(candidate, dict):
        return ""
    return str(candidate.get("candidate_id") or candidate.get("target_id") or "").strip()


def clamp_box(box: Sequence[Any] | None) -> list[float]:
    if not box or len(box) < 4:
        return [0.0, 0.0, 1.0, 1.0]
    vals = [clip01(v) for v in box[:4]]
    x1, y1, x2, y2 = vals
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


def box_area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = clamp_box(box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = clamp_box(a)
    bx1, by1, bx2, by2 = clamp_box(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area([ax1, ay1, ax2, ay2]) + box_area([bx1, by1, bx2, by2]) - inter
    return inter / max(1e-9, union)


def full_like(box: Sequence[float]) -> bool:
    x1, y1, x2, y2 = clamp_box(box)
    return x1 <= 0.015 and y1 <= 0.015 and x2 >= 0.985 and y2 >= 0.985


def is_baseline_like_candidate(
    source: str,
    candidate: dict[str, Any] | None,
    *,
    baseline: dict[str, Any] | None,
    baseline_id: str,
    baseline_box: Sequence[float],
) -> bool:
    if not isinstance(candidate, dict):
        return False
    return bool(
        source == "baseline"
        or (baseline_id and candidate_id(candidate) == baseline_id)
        or (baseline is not None and box_iou(candidate_box(candidate), baseline_box) >= 0.995)
    )


def score_value(candidate: dict[str, Any], key: str, default: float = 0.0) -> float:
    if candidate.get(key) is not None:
        return safe_float(candidate.get(key), default)
    targets = candidate.get("score_targets")
    if isinstance(targets, dict) and targets.get(key) is not None:
        return safe_float(targets.get(key), default)
    return float(default)


def best_score(candidate: dict[str, Any] | None) -> float:
    if not isinstance(candidate, dict):
        return 0.0
    values = []
    for key in ("crop_utility_prob", "score_prob", "crop_utility_rank_pct", "score_rank_pct", "rank_pct"):
        if candidate.get(key) is not None:
            values.append(clip01(candidate.get(key)))
        targets = candidate.get("score_targets")
        if isinstance(targets, dict) and targets.get(key) is not None:
            values.append(clip01(targets.get(key)))
    return max(values) if values else 0.0


def candidate_sources(row: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    baseline = row.get("baseline")
    if isinstance(baseline, dict):
        out.append(("baseline", baseline))
    for field in ("matching_targets", "candidate_pool", "ignored_candidates", "overflow_candidates"):
        for candidate in row.get(field) or []:
            if isinstance(candidate, dict):
                out.append((field, candidate))
    return out


def find_candidate(row: dict[str, Any], winner_id: str) -> tuple[str, dict[str, Any] | None]:
    if winner_id:
        for source, candidate in candidate_sources(row):
            if candidate_id(candidate) == winner_id:
                return source, candidate
    positives = [
        (source, candidate)
        for source, candidate in candidate_sources(row)
        if bool(candidate.get("is_positive_candidate")) or bool(candidate.get("is_soft_positive"))
    ]
    if positives:
        return max(positives, key=lambda item: (best_score(item[1]), candidate_id(item[1])))
    candidates = candidate_sources(row)
    if candidates:
        return max(candidates, key=lambda item: (best_score(item[1]), candidate_id(item[1])))
    return "", None


def copy_score_to_baseline(baseline: dict[str, Any], source: dict[str, Any], *, fallback_score: float) -> bool:
    changed = False
    score_targets = dict(baseline.get("score_targets") if isinstance(baseline.get("score_targets"), dict) else {})
    source_targets = source.get("score_targets") if isinstance(source.get("score_targets"), dict) else {}
    for key in SCORE_KEYS:
        value = source.get(key)
        if value is None and isinstance(source_targets, dict):
            value = source_targets.get(key)
        if value is None and key in {"crop_utility_prob", "score_prob", "crop_utility_rank_pct", "score_rank_pct", "rank_pct"}:
            value = fallback_score
        if value is None:
            continue
        score_targets[key] = value
        if key in {"crop_utility_prob", "score_prob", "crop_utility_rank_pct", "score_rank_pct", "rank_pct"}:
            baseline[key] = value
        changed = True
    if source.get("candidate_weight") is not None or source.get("training_weight") is not None:
        baseline["candidate_weight"] = max(
            safe_float(baseline.get("candidate_weight", baseline.get("training_weight", 0.0)), 0.0),
            safe_float(source.get("candidate_weight", source.get("training_weight", 0.0)), 0.0),
            0.75,
        )
        changed = True
    if source.get("external_score_teacher") is not None:
        baseline["external_score_teacher"] = source.get("external_score_teacher")
        changed = True
    if source.get("training_bucket") is not None:
        baseline["training_bucket"] = source.get("training_bucket")
        changed = True
    if score_targets:
        baseline["score_targets"] = score_targets
    return changed


def original_decision_label(row: dict[str, Any]) -> str:
    decision = row.get("decision_target") if isinstance(row.get("decision_target"), dict) else {}
    raw = decision.get("decision_type", decision.get("decision_id", "crop"))
    if isinstance(raw, str) and raw in DECISION_TO_ID:
        return raw
    try:
        idx = int(raw)
    except (TypeError, ValueError):
        idx = 2
    for label, label_idx in DECISION_TO_ID.items():
        if label_idx == idx:
            return label
    return "crop"


def source_margin_vs_opposite(
    row: dict[str, Any],
    *,
    new_label: str,
    winner: dict[str, Any] | None,
    baseline: dict[str, Any] | None,
    baseline_id: str,
    baseline_box: Sequence[float],
) -> float:
    winner_score = best_score(winner)
    winner_is_baseline_label = new_label in {"keep_full", "minimal_crop"}
    opposite_scores: list[float] = []
    for source, candidate in candidate_sources(row):
        is_base = is_baseline_like_candidate(
            source,
            candidate,
            baseline=baseline,
            baseline_id=baseline_id,
            baseline_box=baseline_box,
        )
        if winner_is_baseline_label and not is_base:
            opposite_scores.append(best_score(candidate))
        elif not winner_is_baseline_label and is_base:
            opposite_scores.append(best_score(candidate))
    return winner_score - (max(opposite_scores) if opposite_scores else 0.0)


def repair_row(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    out = json.loads(json.dumps(row, ensure_ascii=False))
    decision = dict(out.get("decision_target") if isinstance(out.get("decision_target"), dict) else {})
    baseline = out.get("baseline") if isinstance(out.get("baseline"), dict) else None
    baseline_id = candidate_id(baseline)
    baseline_box = candidate_box(baseline)
    winner_id = str(decision.get("winner_post_gate_candidate_id") or "").strip()
    source, winner = find_candidate(out, winner_id)
    if not winner_id:
        winner_id = candidate_id(winner)
    winner_box = candidate_box(winner)
    winner_is_baseline = is_baseline_like_candidate(
        source,
        winner,
        baseline=baseline,
        baseline_id=baseline_id,
        baseline_box=baseline_box,
    )
    old_label = original_decision_label(out)
    if winner_is_baseline:
        new_label = "keep_full" if str(out.get("target_ar", "FREE")).upper() == "FREE" and full_like(winner_box) else "minimal_crop"
    else:
        new_label = "crop"
    baseline_score_copied = False
    if winner_is_baseline and baseline is not None and isinstance(winner, dict):
        fallback = clip01(decision.get("chosen_score_rank"), best_score(winner))
        baseline_score_copied = copy_score_to_baseline(baseline, winner, fallback_score=fallback)
        out["baseline"] = baseline
    winner_for_margin = baseline if winner_is_baseline and baseline is not None else winner
    source_margin = source_margin_vs_opposite(
        out,
        new_label=new_label,
        winner=winner_for_margin,
        baseline=baseline,
        baseline_id=baseline_id,
        baseline_box=baseline_box,
    )
    decision["decision_type"] = new_label
    decision["decision_id"] = DECISION_TO_ID[new_label]
    decision["winner_post_gate_candidate_id"] = winner_id
    decision["winner_post_gate_bbox"] = winner_box
    decision["product_consistent_repair"] = {
        "schema_version": "mobilecropnet_v4_product_consistent_decision_repair_v1",
        "previous_decision_type": old_label,
        "repaired_decision_type": new_label,
        "winner_candidate_id": winner_id,
        "winner_source": source,
        "winner_is_baseline": winner_is_baseline,
        "baseline_candidate_id": baseline_id,
        "baseline_score_copied": baseline_score_copied,
        "source_margin_vs_opposite": source_margin,
    }
    out["decision_target"] = decision
    label_generation = dict(out.get("label_generation") if isinstance(out.get("label_generation"), dict) else {})
    label_generation["product_consistent_decision_repair"] = True
    out["label_generation"] = label_generation
    return out, {
        "old_label": old_label,
        "new_label": new_label,
        "changed": old_label != new_label,
        "winner_is_baseline": winner_is_baseline,
        "winner_missing": winner is None,
        "baseline_score_copied": baseline_score_copied,
        "crop_to_baseline": old_label == "crop" and new_label in {"keep_full", "minimal_crop"},
        "source_margin_vs_opposite": source_margin,
    }


def build_split(
    input_path: Path,
    output_path: Path,
    *,
    split_name: str,
    status_path: Path,
    max_rows: int = 0,
    status_every_rows: int = 1000,
    min_source_margin: float = 0.0,
    source_margin_action: str = "keep",
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    write_json(
        status_path,
        {
            "state": "running",
            "phase": split_name,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "rows": 0,
            "updated_at": utc_now(),
        },
    )
    with output_path.open("w", encoding="utf-8") as handle:
        for row in read_jsonl(input_path, max_rows=max_rows):
            repaired, stats = repair_row(row)
            source_margin = safe_float(stats.get("source_margin_vs_opposite"), 0.0)
            source_ambiguous = min_source_margin > 0.0 and source_margin < min_source_margin
            counts["source_margin_sum"] += source_margin
            counts["source_margin_count"] += 1
            counts["source_margin_negative"] += int(source_margin < 0.0)
            counts["source_margin_below_threshold"] += int(source_ambiguous)
            if source_ambiguous and source_margin_action == "drop":
                counts["dropped_source_ambiguous"] += 1
                counts[f"old_decision::{stats['old_label']}"] += 1
                counts[f"new_decision::{stats['new_label']}"] += 1
                continue
            handle.write(json.dumps(repaired, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts["rows"] += 1
            counts[f"old_decision::{stats['old_label']}"] += 1
            counts[f"new_decision::{stats['new_label']}"] += 1
            for key in ("changed", "winner_is_baseline", "winner_missing", "baseline_score_copied", "crop_to_baseline"):
                counts[key] += int(bool(stats[key]))
            if status_every_rows > 0 and counts["rows"] % int(status_every_rows) == 0:
                write_json(
                    status_path,
                    {
                        "state": "running",
                        "phase": split_name,
                        "input_path": str(input_path),
                        "output_path": str(output_path),
                        "rows": counts["rows"],
                        "dropped_source_ambiguous_rows": counts["dropped_source_ambiguous"],
                        "changed_decision_rows": counts["changed"],
                        "winner_is_baseline_rows": counts["winner_is_baseline"],
                        "updated_at": utc_now(),
                    },
                )
    rows = max(1, counts["rows"])
    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows": counts["rows"],
        "changed_decision_rows": counts["changed"],
        "changed_decision_rate": counts["changed"] / rows,
        "winner_is_baseline_rows": counts["winner_is_baseline"],
        "winner_is_baseline_rate": counts["winner_is_baseline"] / rows,
        "winner_missing_rows": counts["winner_missing"],
        "baseline_score_copied_rows": counts["baseline_score_copied"],
        "crop_to_baseline_rows": counts["crop_to_baseline"],
        "min_source_margin": min_source_margin,
        "source_margin_action": source_margin_action,
        "source_margin_mean": counts["source_margin_sum"] / max(1, counts["source_margin_count"]),
        "source_margin_negative_rows": counts["source_margin_negative"],
        "source_margin_below_threshold_rows": counts["source_margin_below_threshold"],
        "dropped_source_ambiguous_rows": counts["dropped_source_ambiguous"],
        "old_decision_counts": {key.split("::", 1)[1]: value for key, value in counts.items() if key.startswith("old_decision::")},
        "new_decision_counts": {key.split("::", 1)[1]: value for key, value in counts.items() if key.startswith("new_decision::")},
    }
    write_json(
        status_path,
        {
            "state": "running",
            "phase": f"{split_name}_complete",
            "input_path": str(input_path),
            "output_path": str(output_path),
            "rows": counts["rows"],
            "dropped_source_ambiguous_rows": counts["dropped_source_ambiguous"],
            "changed_decision_rows": counts["changed"],
            "winner_is_baseline_rows": counts["winner_is_baseline"],
            "updated_at": utc_now(),
        },
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair MobileCropNet v4 labels so decision/action source matches winner_post_gate_candidate_id.")
    parser.add_argument("--train_jsonl", required=True, type=Path)
    parser.add_argument("--val_jsonl", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--max_train_rows", type=int, default=0)
    parser.add_argument("--max_val_rows", type=int, default=0)
    parser.add_argument("--status_every_rows", type=int, default=1000)
    parser.add_argument("--min_source_margin", type=float, default=0.0)
    parser.add_argument("--source_margin_action", choices=("keep", "drop"), default="keep")
    parser.add_argument("--filter_train_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_out = args.output_dir / "train_product_consistent.jsonl"
    val_out = args.output_dir / "val_product_consistent.jsonl"
    status_path = args.output_dir / "status.json"
    train_summary = build_split(
        args.train_jsonl,
        train_out,
        split_name="train",
        status_path=status_path,
        max_rows=max(0, int(args.max_train_rows)),
        status_every_rows=max(0, int(args.status_every_rows)),
        min_source_margin=max(0.0, float(args.min_source_margin)),
        source_margin_action=str(args.source_margin_action),
    )
    val_margin_action = "keep" if bool(args.filter_train_only) else str(args.source_margin_action)
    val_min_source_margin = 0.0 if bool(args.filter_train_only) else max(0.0, float(args.min_source_margin))
    val_summary = build_split(
        args.val_jsonl,
        val_out,
        split_name="val",
        status_path=status_path,
        max_rows=max(0, int(args.max_val_rows)),
        status_every_rows=max(0, int(args.status_every_rows)),
        min_source_margin=val_min_source_margin,
        source_margin_action=val_margin_action,
    )
    summary = {
        "schema_version": "mobilecropnet_v4_product_consistent_labels_v1",
        "train": train_summary,
        "val": val_summary,
        "artifacts": {
            "train_jsonl": str(train_out),
            "val_jsonl": str(val_out),
            "summary_json": str(args.output_dir / "summary.json"),
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(
        status_path,
        {
            "state": "succeeded",
            "phase": "complete",
            "rows": {
                "train": train_summary["rows"],
                "val": val_summary["rows"],
            },
            "summary_json": str(args.output_dir / "summary.json"),
            "updated_at": utc_now(),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
