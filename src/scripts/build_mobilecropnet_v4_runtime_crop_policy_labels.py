#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DECISION_TO_ID = {"keep_full": 0, "minimal_crop": 1, "crop": 2}
BOX_EPS = 1.0e-5


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            text = line.strip()
            if not text:
                continue
            try:
                yield line_no, json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc


def _decision_label(row: dict[str, Any]) -> str:
    decision = row.get("decision_target") if isinstance(row.get("decision_target"), dict) else {}
    raw = decision.get("decision_type", decision.get("decision_id", "crop"))
    if isinstance(raw, int):
        for label, idx in DECISION_TO_ID.items():
            if idx == int(raw):
                return label
    return str(raw) if str(raw) in DECISION_TO_ID else "crop"


def _candidate_id(candidate: dict[str, Any] | None) -> str:
    if not isinstance(candidate, dict):
        return ""
    for key in ("candidate_id", "id", "proposal_id"):
        value = candidate.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def _candidate_box(candidate: dict[str, Any] | None) -> list[float] | None:
    if not isinstance(candidate, dict):
        return None
    value = candidate.get("bbox_norm_xyxy")
    if isinstance(value, list) and len(value) >= 4:
        return [float(v) for v in value[:4]]
    return None


def _same_box(a: list[float] | None, b: list[float] | None, *, eps: float = BOX_EPS) -> bool:
    if a is None or b is None:
        return False
    return all(abs(float(x) - float(y)) <= eps for x, y in zip(a[:4], b[:4]))


def _score(candidate: dict[str, Any] | None) -> float:
    if not isinstance(candidate, dict):
        return -1.0
    for key in ("score", "utility_score", "crop_utility_prob", "score_prob", "score_rank"):
        value = candidate.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    targets = candidate.get("score_targets") if isinstance(candidate.get("score_targets"), dict) else {}
    for key in ("crop_utility_prob", "score_prob"):
        value = targets.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return -1.0


def _non_base_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key in (
        "matching_targets",
        "candidate_pool",
        "candidates",
        "candidate_crops",
        "proposals",
        "generated_proposals",
    ):
        value = row.get(key)
        if not isinstance(value, list):
            continue
        for candidate in value:
            if not isinstance(candidate, dict):
                continue
            source = str(candidate.get("source", candidate.get("candidate_source", ""))).lower()
            cid = _candidate_id(candidate).lower()
            if bool(candidate.get("is_baseline", False)):
                continue
            if source.startswith("baseline") or cid.startswith("baseline"):
                continue
            out.append(candidate)
    return out


def _runtime_crop_candidates(row: dict[str, Any], candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    if not candidates:
        return [], 0
    non_baseline_like = [candidate for candidate in candidates if not _candidate_collides_with_baseline(row, candidate)]
    if non_baseline_like:
        return non_baseline_like, len(candidates) - len(non_baseline_like)
    return candidates, 0


def _baseline_signature(row: dict[str, Any]) -> tuple[str, list[float] | None]:
    baseline = row.get("baseline") if isinstance(row.get("baseline"), dict) else {}
    return _candidate_id(baseline), _candidate_box(baseline)


def _candidate_collides_with_baseline(row: dict[str, Any], candidate: dict[str, Any] | None) -> bool:
    if not isinstance(candidate, dict):
        return False
    baseline_id, baseline_box = _baseline_signature(row)
    candidate_id = _candidate_id(candidate)
    candidate_box = _candidate_box(candidate)
    if baseline_id and candidate_id and baseline_id == candidate_id:
        return True
    return _same_box(baseline_box, candidate_box)


def _append_runtime_crop_clone(row: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    clone = json.loads(json.dumps(candidate, ensure_ascii=False))
    original_id = _candidate_id(candidate)
    clone_id = f"{original_id or 'candidate'}__runtime_crop"
    existing_ids = {
        _candidate_id(item)
        for key in ("matching_targets", "candidate_pool", "candidates", "candidate_crops", "proposals", "generated_proposals")
        for item in (row.get(key) if isinstance(row.get(key), list) else [])
        if isinstance(item, dict)
    }
    suffix = 1
    unique_clone_id = clone_id
    while unique_clone_id in existing_ids:
        suffix += 1
        unique_clone_id = f"{clone_id}_{suffix}"
    clone["candidate_id"] = unique_clone_id
    clone["runtime_crop_policy_clone"] = {
        "schema_version": "mobilecropnet_v4_runtime_crop_policy_clone_v1",
        "reason": "winner_collided_with_baseline_dedup_key",
        "original_candidate_id": original_id,
    }
    clone.pop("is_baseline", None)
    matching_targets = row.get("matching_targets")
    if not isinstance(matching_targets, list):
        matching_targets = []
    matching_targets = [clone, *matching_targets]
    row["matching_targets"] = matching_targets
    return clone


def _source_margin(row: dict[str, Any]) -> float | None:
    decision = row.get("decision_target") if isinstance(row.get("decision_target"), dict) else {}
    for container_key in ("runtime_crop_policy_repair", "product_consistent_repair"):
        container = decision.get(container_key) if isinstance(decision.get(container_key), dict) else {}
        value = container.get("source_margin_vs_opposite")
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    for key in ("source_margin_vs_opposite", "decision_source_margin", "decision_delta_vs_baseline", "delta_vs_base"):
        value = decision.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


def _is_full_box(box: list[float] | None) -> bool:
    if box is None:
        return False
    return (
        abs(float(box[0])) <= BOX_EPS
        and abs(float(box[1])) <= BOX_EPS
        and abs(float(box[2]) - 1.0) <= BOX_EPS
        and abs(float(box[3]) - 1.0) <= BOX_EPS
    )


def _baseline_decision_label(row: dict[str, Any]) -> str:
    _baseline_id, baseline_box = _baseline_signature(row)
    return "keep_full" if _is_full_box(baseline_box) else "minimal_crop"


def _baseline_positive_clone(row: dict[str, Any]) -> dict[str, Any]:
    baseline = row.get("baseline") if isinstance(row.get("baseline"), dict) else {}
    clone = json.loads(json.dumps(baseline, ensure_ascii=False)) if baseline else {}
    base_id, base_box = _baseline_signature(row)
    if base_id:
        clone["candidate_id"] = base_id
    if base_box is not None:
        clone["bbox_norm_xyxy"] = base_box
        x1, y1, x2, y2 = base_box
        clone["bbox_cxcywh"] = [
            (float(x1) + float(x2)) / 2.0,
            (float(y1) + float(y2)) / 2.0,
            max(0.0, float(x2) - float(x1)),
            max(0.0, float(y2) - float(y1)),
        ]
    clone["is_baseline"] = True
    clone["risk_conservative_baseline_positive"] = True
    score_targets = dict(clone.get("score_targets") if isinstance(clone.get("score_targets"), dict) else {})
    for key in ("score_prob", "crop_utility_prob", "rank_pct", "crop_utility_rank_pct", "score_rank_pct"):
        score_targets[key] = 1.0
    score_targets.setdefault("score_raw_policy", clone.get("score_policy", 1.0))
    score_targets.setdefault("crop_utility_raw", clone.get("crop_utility_raw", clone.get("score_policy", 1.0)))
    clone["score_targets"] = score_targets
    return clone


def _demote_crop_to_baseline(
    row: dict[str, Any],
    *,
    old_label: str,
    source_margin: float | None,
    min_crop_source_margin: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out = json.loads(json.dumps(row, ensure_ascii=False))
    decision = dict(out.get("decision_target") if isinstance(out.get("decision_target"), dict) else {})
    new_label = _baseline_decision_label(out)
    baseline_candidate = _baseline_positive_clone(out)
    baseline_id = _candidate_id(baseline_candidate)
    baseline_box = _candidate_box(baseline_candidate)
    decision["decision_type"] = new_label
    decision["decision_id"] = DECISION_TO_ID[new_label]
    if baseline_id:
        decision["winner_post_gate_candidate_id"] = baseline_id
        decision["base_candidate_id"] = baseline_id
    if baseline_box is not None:
        decision["winner_post_gate_bbox"] = baseline_box
        decision["base_bbox"] = baseline_box
    decision["risk_conservative_source_repair"] = {
        "schema_version": "mobilecropnet_v4_risk_conservative_source_repair_v1",
        "previous_decision_type": old_label,
        "repaired_decision_type": new_label,
        "source_margin_vs_opposite": source_margin,
        "min_crop_source_margin": float(min_crop_source_margin),
        "reason": "crop_source_margin_below_threshold",
        "baseline_candidate_id": baseline_id,
    }
    out["decision_target"] = decision
    out["matching_targets"] = [baseline_candidate]
    label_generation = dict(out.get("label_generation") if isinstance(out.get("label_generation"), dict) else {})
    label_generation["risk_conservative_source_repair"] = True
    label_generation["risk_conservative_min_crop_source_margin"] = float(min_crop_source_margin)
    out["label_generation"] = label_generation
    return out, {
        "old_label": old_label,
        "new_label": new_label,
        "changed": True,
        "non_base_candidate_count": len(_non_base_candidates(out)),
        "winner_missing": False,
        "winner_cloned": False,
        "baseline_like_suppressed": 0,
        "preserved_baseline_decision": False,
        "risk_conservative_demoted_crop": True,
        "source_margin": source_margin,
    }


def repair_row(
    row: dict[str, Any],
    *,
    preserve_baseline_decisions: bool = False,
    min_crop_source_margin: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    out = json.loads(json.dumps(row, ensure_ascii=False))
    old_label = _decision_label(out)
    if preserve_baseline_decisions and old_label != "crop":
        return out, {
            "old_label": old_label,
            "new_label": old_label,
            "changed": False,
            "non_base_candidate_count": len(_non_base_candidates(out)),
            "winner_missing": False,
            "winner_cloned": False,
            "baseline_like_suppressed": 0,
            "preserved_baseline_decision": True,
            "risk_conservative_demoted_crop": False,
            "source_margin": _source_margin(out),
        }
    source_margin = _source_margin(out)
    if (
        min_crop_source_margin is not None
        and old_label == "crop"
        and (source_margin is None or float(source_margin) < float(min_crop_source_margin))
    ):
        return _demote_crop_to_baseline(
            out,
            old_label=old_label,
            source_margin=source_margin,
            min_crop_source_margin=float(min_crop_source_margin),
        )
    decision = dict(out.get("decision_target") if isinstance(out.get("decision_target"), dict) else {})
    non_base = _non_base_candidates(out)
    runtime_candidates, baseline_like_suppressed = _runtime_crop_candidates(out, non_base)
    winner = max(runtime_candidates, key=_score) if runtime_candidates else None
    cloned = False
    original_winner_id = _candidate_id(winner)
    if winner is not None and _candidate_collides_with_baseline(out, winner):
        winner = _append_runtime_crop_clone(out, winner)
        cloned = True
    decision["decision_type"] = "crop"
    decision["decision_id"] = DECISION_TO_ID["crop"]
    if winner is not None:
        cid = _candidate_id(winner)
        box = _candidate_box(winner)
        if cid:
            decision["winner_post_gate_candidate_id"] = cid
        if box is not None:
            decision["winner_post_gate_bbox"] = box
    decision["runtime_crop_policy_repair"] = {
        "schema_version": "mobilecropnet_v4_runtime_crop_policy_repair_v1",
        "previous_decision_type": old_label,
        "repaired_decision_type": "crop",
        "winner_source": "best_non_base_candidate" if winner is not None else "missing_non_base_fallback",
        "winner_candidate_id": _candidate_id(winner),
        "original_winner_candidate_id": original_winner_id,
        "winner_cloned_for_runtime_crop": cloned,
        "non_base_candidate_count": len(non_base),
        "baseline_like_candidate_suppressed_count": baseline_like_suppressed,
    }
    out["decision_target"] = decision
    label_generation = dict(out.get("label_generation") if isinstance(out.get("label_generation"), dict) else {})
    label_generation["runtime_crop_policy_repair"] = True
    out["label_generation"] = label_generation
    return out, {
        "old_label": old_label,
        "new_label": "crop",
        "changed": old_label != "crop",
        "non_base_candidate_count": len(non_base),
        "winner_missing": winner is None,
        "winner_cloned": cloned,
        "baseline_like_suppressed": baseline_like_suppressed,
        "preserved_baseline_decision": False,
        "risk_conservative_demoted_crop": False,
        "source_margin": source_margin,
    }


def convert_file(
    input_path: Path,
    output_path: Path,
    *,
    preserve_baseline_decisions: bool = False,
    min_crop_source_margin: float | None = None,
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    old_counts: Counter[str] = Counter()
    new_counts: Counter[str] = Counter()
    changed = 0
    winner_missing = 0
    winner_cloned = 0
    baseline_like_suppressed_rows = 0
    baseline_like_suppressed_total = 0
    preserved_baseline_decisions = 0
    risk_conservative_demoted_crop = 0
    row_count = 0
    non_base_hist: Counter[str] = Counter()
    source_margin_hist: Counter[str] = Counter()
    with output_path.open("w", encoding="utf-8") as out_f:
        for _line_no, row in _read_jsonl(input_path):
            repaired, stats = repair_row(
                row,
                preserve_baseline_decisions=preserve_baseline_decisions,
                min_crop_source_margin=min_crop_source_margin,
            )
            row_count += 1
            old_counts[str(stats["old_label"])] += 1
            new_counts[str(stats["new_label"])] += 1
            changed += int(bool(stats["changed"]))
            winner_missing += int(bool(stats["winner_missing"]))
            winner_cloned += int(bool(stats["winner_cloned"]))
            preserved_baseline_decisions += int(bool(stats.get("preserved_baseline_decision", False)))
            suppressed = int(stats["baseline_like_suppressed"])
            baseline_like_suppressed_total += suppressed
            baseline_like_suppressed_rows += int(suppressed > 0)
            risk_conservative_demoted_crop += int(bool(stats.get("risk_conservative_demoted_crop", False)))
            margin = stats.get("source_margin")
            if margin is None:
                source_margin_hist["missing"] += 1
            else:
                value = float(margin)
                bucket = "<0.05" if value < 0.05 else "<0.15" if value < 0.15 else "<0.30" if value < 0.30 else "<0.50" if value < 0.50 else ">=0.50"
                source_margin_hist[bucket] += 1
            non_base_count = int(stats["non_base_candidate_count"])
            bucket = "0" if non_base_count <= 0 else "1-8" if non_base_count <= 8 else "9+"
            non_base_hist[bucket] += 1
            out_f.write(json.dumps(repaired, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {
        "input": str(input_path),
        "output": str(output_path),
        "row_count": row_count,
        "old_decision_counts": dict(old_counts),
        "new_decision_counts": dict(new_counts),
        "changed_count": changed,
        "winner_missing_count": winner_missing,
        "winner_cloned_for_runtime_crop_count": winner_cloned,
        "preserved_baseline_decision_count": preserved_baseline_decisions,
        "risk_conservative_demoted_crop_count": risk_conservative_demoted_crop,
        "baseline_like_suppressed_row_count": baseline_like_suppressed_rows,
        "baseline_like_suppressed_candidate_count": baseline_like_suppressed_total,
        "non_base_candidate_count_buckets": dict(non_base_hist),
        "source_margin_buckets": dict(source_margin_hist),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_jsonl", type=Path, required=True)
    parser.add_argument("--val_jsonl", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_name", default="train_runtime_crop_policy.jsonl")
    parser.add_argument("--val_name", default="val_runtime_crop_policy.jsonl")
    parser.add_argument(
        "--preserve_baseline_decisions",
        action="store_true",
        help="Keep keep_full/minimal_crop rows unchanged and only repair crop rows with runtime crop clone ids.",
    )
    parser.add_argument(
        "--min_crop_source_margin",
        type=float,
        default=None,
        help=(
            "When set, crop rows whose source_margin_vs_opposite is below this threshold are demoted to the "
            "baseline/minimal decision. This creates a risk-conservative policy label set."
        ),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_summary = convert_file(
        args.train_jsonl,
        args.output_dir / args.train_name,
        preserve_baseline_decisions=bool(args.preserve_baseline_decisions),
        min_crop_source_margin=args.min_crop_source_margin,
    )
    val_summary = convert_file(
        args.val_jsonl,
        args.output_dir / args.val_name,
        preserve_baseline_decisions=bool(args.preserve_baseline_decisions),
        min_crop_source_margin=args.min_crop_source_margin,
    )
    manifest = {
        "schema_version": "mobilecropnet_v4_runtime_crop_policy_labels_v2",
        "purpose": "Align Product-AR runtime decision/action supervision to crop final action while keeping image+target_ar no-prior inputs. Baseline-colliding candidates are suppressed when a real crop alternative exists.",
        "preserve_baseline_decisions": bool(args.preserve_baseline_decisions),
        "min_crop_source_margin": args.min_crop_source_margin,
        "train": train_summary,
        "val": val_summary,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
