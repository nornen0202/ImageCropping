from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


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
    union = _box_area(a) + _box_area(b) - inter
    return inter / union if union > 1e-8 else 0.0


def _source_id(row: Dict[str, Any]) -> str:
    return str(row.get("source_image_id") or row.get("image_id") or row.get("id") or Path(str(row.get("file_name") or "")).stem)


def _scene_subject_flags(feature_row: Dict[str, Any]) -> Tuple[list[str], Dict[str, Any]]:
    routing = feature_row.get("routing") if isinstance(feature_row.get("routing"), dict) else {}
    effective = routing.get("effective_subject_region") if isinstance(routing.get("effective_subject_region"), dict) else {}
    c7 = feature_row.get("c7_saliency") if isinstance(feature_row.get("c7_saliency"), dict) else {}
    source = str(effective.get("source") or "").strip().lower()
    state = str(effective.get("state") or "").strip().lower()
    reasons = effective.get("reasons") if isinstance(effective.get("reasons"), list) else []
    severe = "detector_saliency_severe_disagreement" in source or "detector_saliency_severe_disagreement" in {
        str(reason) for reason in reasons
    }
    soft_anchor = "scene_soft_anchor" in source
    guidance = effective.get("guidance_envelope_bbox_norm_xyxy")
    effective_box = effective.get("effective_bbox_norm_xyxy")
    saliency = c7.get("foreground_bbox_norm_xyxy") or c7.get("top_component_bbox_norm_xyxy")
    guidance_area = _box_area(guidance)
    effective_area = _box_area(effective_box)
    saliency_area = _safe_float(c7.get("foreground_area_ratio"), _box_area(saliency))
    guidance_saliency_iou = _box_iou(guidance, saliency)
    reliability = _safe_float(effective.get("subject_reliability"), 1.0)
    agreement = _safe_float(effective.get("subject_agreement_iou"), 1.0)
    residual_scene_mass = _safe_float(effective.get("semantic_residual_scene_mass"), 0.0)
    trust = str(effective.get("support_trust_tier") or "").strip().lower()
    missing_or_blank_saliency = not isinstance(saliency, (list, tuple)) or len(saliency) != 4 or saliency_area <= 0.005
    full_frame_saliency = saliency_area >= 0.70 or _box_area(saliency) >= 0.90
    flags: list[str] = []
    if state in {"no_dominant_subject", "distributed_attention"} and reliability <= 0.35 and guidance_area > 0.0:
        if (
            severe
            or (soft_anchor and (missing_or_blank_saliency or residual_scene_mass >= 0.85 or trust == "low"))
            or (full_frame_saliency and (residual_scene_mass >= 0.70 or trust == "low" or effective_area >= 0.80))
        ):
            flags.append(f"low_reliability_{state}")
    if state == "multi_subject" and reliability <= 0.35 and agreement <= 0.08 and severe:
        flags.append("low_reliability_multi_subject_scene_disagreement")
    if (
        state == "dominant_subject"
        and source == "raw_anchor"
        and reliability >= 0.70
        and agreement <= 0.08
        and guidance_area >= 0.08
        and (saliency_area <= 0.02 or guidance_saliency_iou <= 0.10)
    ):
        flags.append("raw_anchor_saliency_mismatch_scene_subject")
    return flags, {
        "scene_subject_state": state,
        "scene_subject_source": source,
        "scene_subject_reliability": reliability,
        "scene_subject_agreement_iou": agreement,
        "scene_subject_guidance_area": guidance_area,
        "scene_subject_effective_area": effective_area,
        "scene_subject_saliency_area": saliency_area,
        "scene_subject_guidance_saliency_iou": guidance_saliency_iou,
        "scene_subject_residual_scene_mass": residual_scene_mass,
        "scene_subject_support_trust_tier": trust,
        "scene_guidance_bbox_norm_xyxy": guidance,
        "scene_saliency_fg_bbox_norm_xyxy": saliency,
    }


def _load_flagged_features(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source_id = _source_id(row)
            flags, metrics = _scene_subject_flags(row)
            if source_id and flags:
                out[source_id] = {**metrics, "residual_flags": flags}
    return out


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    fieldnames = [
        "source_image_id",
        "query_id",
        "target_ar",
        "score_mode",
        "gt_flag",
        "landscape_subject_safe_enabled",
        "landscape_subject_safe_active",
        "landscape_subject_safe_suppressed_reason",
        "residual_flags",
        "scene_subject_state",
        "scene_subject_source",
        "scene_subject_reliability",
        "scene_subject_agreement_iou",
        "scene_subject_guidance_area",
        "scene_subject_effective_area",
        "scene_subject_saliency_area",
        "scene_subject_guidance_saliency_iou",
        "scene_subject_residual_scene_mass",
        "scene_subject_support_trust_tier",
        "bbox",
        "scene_guidance_bbox_norm_xyxy",
        "scene_saliency_fg_bbox_norm_xyxy",
    ]
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
            count += 1
    return count


def audit(
    *,
    label_json: Path,
    features_jsonl: Path,
    out_dir: Path,
    prefix: str,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = _load_json(label_json)
    flagged_features = _load_flagged_features(features_jsonl)
    image_source = {int(row["id"]): _source_id(row) for row in payload.get("images", []) if isinstance(row, dict)}
    rows: list[Dict[str, Any]] = []
    positive_active_rows: list[Dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    flag_image_counts: defaultdict[str, set[str]] = defaultdict(set)
    for ann in payload.get("annotations", []):
        if not isinstance(ann, dict) or ann.get("mode_name") != "landscape":
            continue
        source_id = image_source.get(int(ann.get("image_id", -1)), "")
        metrics = flagged_features.get(source_id)
        if metrics is None:
            continue
        attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
        comps = attrs.get("score_components") if isinstance(attrs.get("score_components"), dict) else {}
        safe_enabled = _safe_int(attrs.get("landscape_subject_safe_enabled", comps.get("landscape_subject_safe_enabled", 1)), 1)
        safe_active = _safe_int(attrs.get("landscape_subject_safe_active", comps.get("landscape_subject_safe_active", 0)), 0)
        suppressed = str(
            attrs.get(
                "landscape_subject_safe_suppressed_reason",
                comps.get("landscape_subject_safe_suppressed_reason", ""),
            )
            or ""
        )
        if suppressed:
            reason_counts[suppressed] += 1
        row = {
            "source_image_id": source_id,
            "query_id": ann.get("query_id"),
            "target_ar": attrs.get("target_ar"),
            "score_mode": ann.get("score_mode"),
            "gt_flag": ann.get("gt_flag"),
            "landscape_subject_safe_enabled": safe_enabled,
            "landscape_subject_safe_active": safe_active,
            "landscape_subject_safe_suppressed_reason": suppressed,
            **metrics,
            "residual_flags": "|".join(metrics.get("residual_flags") or []),
            "bbox": ann.get("bbox"),
        }
        rows.append(row)
        for flag in metrics.get("residual_flags") or []:
            flag_image_counts[str(flag)].add(source_id)
        if int(ann.get("gt_flag", 0)) == 1 and safe_enabled == 1 and safe_active == 1 and not suppressed:
            positive_active_rows.append(row)
    csv_path = out_dir / f"{prefix}_residual_subjectsafe_rows.csv"
    active_csv_path = out_dir / f"{prefix}_residual_subjectsafe_positive_active_rows.csv"
    ids_path = out_dir / f"{prefix}_residual_subjectsafe_positive_active_image_ids.txt"
    summary_path = out_dir / f"{prefix}_residual_subjectsafe_summary.json"
    _write_csv(csv_path, rows)
    _write_csv(active_csv_path, positive_active_rows)
    active_ids = sorted({str(row.get("source_image_id") or "") for row in positive_active_rows if row.get("source_image_id")})
    ids_path.write_text("\n".join(active_ids) + ("\n" if active_ids else ""), encoding="utf-8")
    summary = {
        "label_json": str(label_json),
        "features_jsonl": str(features_jsonl),
        "flagged_feature_image_count": len(flagged_features),
        "flagged_landscape_annotation_count": len(rows),
        "positive_active_residual_annotation_count": len(positive_active_rows),
        "positive_active_residual_image_count": len(active_ids),
        "suppressed_reason_counts": dict(sorted(reason_counts.items())),
        "flagged_feature_image_counts_by_flag": {key: len(value) for key, value in sorted(flag_image_counts.items())},
        "output_paths": {
            "summary_json": str(summary_path),
            "all_rows_csv": str(csv_path),
            "positive_active_rows_csv": str(active_csv_path),
            "positive_active_image_ids": str(ids_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit residual suspicious landscape scene subjects that remain subject-safe active.")
    parser.add_argument("--label_json", required=True)
    parser.add_argument("--features_jsonl", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--prefix", default="landscape")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    audit(
        label_json=Path(args.label_json),
        features_jsonl=Path(args.features_jsonl),
        out_dir=Path(args.out_dir),
        prefix=str(args.prefix),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
