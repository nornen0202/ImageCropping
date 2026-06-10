from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _box_area(box: Any) -> float:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return 0.0
    x1, y1, x2, y2 = [_safe_float(v) for v in box]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_iou(a: Any, b: Any) -> float:
    if _box_area(a) <= 0.0 or _box_area(b) <= 0.0:
        return 0.0
    ax1, ay1, ax2, ay2 = [_safe_float(v) for v in a]
    bx1, by1, bx2, by2 = [_safe_float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / max(1e-8, _box_area(a) + _box_area(b) - inter)


def _source_id(row: Dict[str, Any]) -> str:
    return str(row.get("source_image_id") or row.get("image_id") or row.get("id") or "")


def _low_reliability_scene_subject(row: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    routing = row.get("routing") if isinstance(row.get("routing"), dict) else {}
    effective = routing.get("effective_subject_region") if isinstance(routing.get("effective_subject_region"), dict) else {}
    c7 = row.get("c7_saliency") if isinstance(row.get("c7_saliency"), dict) else {}
    saliency_semantic = row.get("saliency_semantic") if isinstance(row.get("saliency_semantic"), dict) else {}
    source = str(effective.get("source") or "").strip().lower()
    reasons = effective.get("reasons") if isinstance(effective.get("reasons"), list) else []
    severe = "detector_saliency_severe_disagreement" in source or "detector_saliency_severe_disagreement" in {
        str(reason) for reason in reasons
    }
    guidance = effective.get("guidance_envelope_bbox_norm_xyxy")
    effective_box = effective.get("effective_bbox_norm_xyxy")
    saliency_box = c7.get("foreground_bbox_norm_xyxy") or c7.get("top_component_bbox_norm_xyxy")
    state = str(effective.get("state") or "")
    reliability = _safe_float(effective.get("subject_reliability"), 1.0)
    agreement = _safe_float(effective.get("subject_agreement_iou"), 1.0)
    guidance_area = _box_area(guidance)
    saliency_area = _safe_float(c7.get("foreground_area_ratio"), _box_area(saliency_box))
    effective_area = _box_area(effective_box)
    residual_scene_mass = max(
        _safe_float(effective.get("semantic_residual_scene_mass"), 0.0),
        _safe_float(saliency_semantic.get("residual_scene_mass"), 0.0),
    )
    support_trust_tier = str(
        effective.get("support_trust_tier") or saliency_semantic.get("support_trust_tier") or ""
    ).strip().lower()
    soft_anchor = "scene_soft_anchor" in source
    missing_or_blank_saliency = not isinstance(saliency_box, (list, tuple)) or len(saliency_box) != 4 or saliency_area <= 0.005
    full_frame_saliency = saliency_area >= 0.70 or _box_area(saliency_box) >= 0.90
    guidance_saliency_iou = _box_iou(guidance, saliency_box)
    guidance_tiny_saliency_mismatch = (
        guidance_area >= 0.08
        and saliency_area <= 0.02
        and agreement <= 0.08
    )
    guidance_saliency_mismatch = (
        guidance_area >= 0.08
        and guidance_saliency_iou <= 0.10
        and agreement <= 0.08
    )
    legacy_flag = (
        state == "no_dominant_subject"
        and severe
        and reliability <= 0.25
        and agreement <= 0.05
    )
    expanded_flag = (
        state in {"no_dominant_subject", "distributed_attention"}
        and reliability <= 0.35
        and guidance_area > 0.0
        and (
            severe
            or (soft_anchor and (missing_or_blank_saliency or residual_scene_mass >= 0.85 or support_trust_tier == "low"))
            or (full_frame_saliency and (residual_scene_mass >= 0.70 or support_trust_tier == "low" or effective_area >= 0.80))
        )
    )
    multi_low_reliability_flag = (
        state == "multi_subject"
        and reliability <= 0.35
        and agreement <= 0.08
        and severe
    )
    raw_anchor_mismatch_flag = (
        state == "dominant_subject"
        and source == "raw_anchor"
        and reliability >= 0.70
        and (guidance_tiny_saliency_mismatch or guidance_saliency_mismatch)
    )
    flag = legacy_flag or expanded_flag or multi_low_reliability_flag or raw_anchor_mismatch_flag
    if state == "distributed_attention":
        suppression_reason = "low_reliability_distributed_scene_subject"
    elif soft_anchor and missing_or_blank_saliency:
        suppression_reason = "low_reliability_scene_soft_anchor_no_saliency"
    elif multi_low_reliability_flag:
        suppression_reason = "low_reliability_multi_subject_scene_disagreement"
    elif raw_anchor_mismatch_flag:
        suppression_reason = "raw_anchor_saliency_mismatch_scene_subject"
    else:
        suppression_reason = "low_reliability_no_dominant_scene_subject"
    return flag, {
        "effective_state": state,
        "effective_source": source,
        "subject_reliability": reliability,
        "subject_agreement_iou": agreement,
        "legacy_low_reliability_flag": int(bool(legacy_flag)),
        "expanded_low_reliability_flag": int(bool(expanded_flag)),
        "multi_low_reliability_flag": int(bool(multi_low_reliability_flag)),
        "raw_anchor_mismatch_flag": int(bool(raw_anchor_mismatch_flag)),
        "expected_suppression_reason": suppression_reason if flag else "",
        "guidance_area": guidance_area,
        "effective_area": effective_area,
        "residual_scene_mass": residual_scene_mass,
        "support_trust_tier": support_trust_tier,
        "guidance_bbox_norm_xyxy": guidance,
        "effective_bbox_norm_xyxy": effective_box,
        "saliency_fg_bbox_norm_xyxy": saliency_box,
        "guidance_saliency_iou": guidance_saliency_iou,
        "saliency_fg_area_ratio": _safe_float(c7.get("foreground_area_ratio"), 0.0),
    }


def _load_low_reliability_features(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source_id = _source_id(row)
            if not source_id:
                continue
            flag, info = _low_reliability_scene_subject(row)
            if flag:
                rows[source_id] = info
    return rows


def _iter_landscape_status(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("mode_name") == "landscape":
                yield row


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    fieldnames = [
        "source_image_id",
        "target_ar",
        "decision",
        "best_score",
        "positive_score",
        "no_positive_reason",
        "effective_state",
        "effective_source",
        "subject_reliability",
        "subject_agreement_iou",
        "legacy_low_reliability_flag",
        "expanded_low_reliability_flag",
        "multi_low_reliability_flag",
        "raw_anchor_mismatch_flag",
        "expected_suppression_reason",
        "guidance_area",
        "effective_area",
        "residual_scene_mass",
        "support_trust_tier",
        "guidance_saliency_iou",
        "saliency_fg_area_ratio",
        "guidance_bbox_norm_xyxy",
        "effective_bbox_norm_xyxy",
        "saliency_fg_bbox_norm_xyxy",
    ]
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
            count += 1
    return count


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit low-reliability scene pseudo-subject impact on landscape labels.")
    parser.add_argument("--features_jsonl", required=True)
    parser.add_argument("--query_status_jsonl", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--prefix", default="landscape_scene_subject")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lowrel = _load_low_reliability_features(Path(args.features_jsonl))
    rows: List[Dict[str, Any]] = []
    decision_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    expected_reason_counts: Counter[str] = Counter()
    legacy_feature_count = 0
    expanded_feature_count = 0
    image_decisions: Dict[str, Counter[str]] = defaultdict(Counter)
    total_landscape_queries = 0
    for row in _iter_landscape_status(Path(args.query_status_jsonl)):
        total_landscape_queries += 1
        source_id = str(row.get("source_image_id") or "")
        info = lowrel.get(source_id)
        if info is None:
            continue
        decision = str(row.get("decision") or "")
        decision_counts[decision] += 1
        image_decisions[source_id][decision] += 1
        if int(info.get("legacy_low_reliability_flag", 0)):
            legacy_feature_count += 1
        if int(info.get("expanded_low_reliability_flag", 0)):
            expanded_feature_count += 1
        expected_reason = str(info.get("expected_suppression_reason") or "")
        if expected_reason:
            expected_reason_counts[expected_reason] += 1
        reason = str(row.get("no_positive_reason") or "")
        if reason:
            for item in reason.split(","):
                reason_counts[item] += 1
        rows.append(
            {
                "source_image_id": source_id,
                "target_ar": row.get("target_ar"),
                "decision": decision,
                "best_score": row.get("best_score"),
                "positive_score": row.get("positive_score"),
                "no_positive_reason": reason,
                **info,
            }
        )
    csv_path = out_dir / f"{args.prefix}_query_rows.csv"
    ids_path = out_dir / f"{args.prefix}_affected_image_ids.txt"
    summary_path = out_dir / f"{args.prefix}_summary.json"
    report_path = out_dir / f"{args.prefix}_report_ko.md"
    _write_csv(csv_path, rows)
    ids_path.write_text(
        "\n".join(sorted(lowrel)) + ("\n" if lowrel else ""),
        encoding="utf-8",
    )
    positive_image_count = sum(1 for counter in image_decisions.values() if counter.get("positive", 0) > 0)
    summary = {
        "features_jsonl": str(args.features_jsonl),
        "query_status_jsonl": str(args.query_status_jsonl),
        "low_reliability_scene_image_count": len(lowrel),
        "legacy_low_reliability_feature_query_rows": legacy_feature_count,
        "expanded_low_reliability_feature_query_rows": expanded_feature_count,
        "landscape_query_count": total_landscape_queries,
        "affected_landscape_query_count": len(rows),
        "affected_positive_query_count": int(decision_counts.get("positive", 0)),
        "affected_positive_image_count": positive_image_count,
        "affected_decision_counts": dict(sorted(decision_counts.items())),
        "affected_no_positive_reason_counts": dict(sorted(reason_counts.items())),
        "expected_suppression_reason_counts": dict(sorted(expected_reason_counts.items())),
        "output_paths": {
            "summary_json": str(summary_path),
            "rows_csv": str(csv_path),
            "affected_image_ids": str(ids_path),
            "report_md": str(report_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = "\n".join(
        [
            "# Landscape Scene Subject Alignment Audit",
            "",
            f"- low-reliability scene image: {len(lowrel):,}",
            f"- affected landscape query: {len(rows):,}",
            f"- affected positive query: {int(decision_counts.get('positive', 0)):,}",
            f"- rows csv: `{csv_path}`",
            f"- affected image ids: `{ids_path}`",
            "",
            "## Expected Suppression Reasons",
            "",
            "| reason | count |",
            "|---|---:|",
            *[f"| {key} | {value:,} |" for key, value in expected_reason_counts.most_common()],
            "",
            "## No-Positive Reasons",
            "",
            "| reason | count |",
            "|---|---:|",
            *[f"| {key} | {value:,} |" for key, value in reason_counts.most_common()],
            "",
        ]
    )
    report_path.write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
