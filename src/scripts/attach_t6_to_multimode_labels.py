#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except (TypeError, ValueError):
        return float(default)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(float(lo), min(float(hi), float(value)))


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def bbox_xywh_to_norm_xyxy(bbox_xywh: Sequence[float], width: float, height: float) -> list[float]:
    x, y, w, h = [safe_float(v) for v in bbox_xywh[:4]]
    width = max(1.0, safe_float(width, 1.0))
    height = max(1.0, safe_float(height, 1.0))
    return [
        clamp(x / width),
        clamp(y / height),
        clamp((x + max(0.0, w)) / width),
        clamp((y + max(0.0, h)) / height),
    ]


def normalize_xyxy(box: Sequence[float]) -> list[float]:
    vals = [clamp(safe_float(v), 0.0, 1.0) for v in list(box)[:4]]
    if len(vals) < 4:
        vals = [0.0, 0.0, 1.0, 1.0]
    x1, y1, x2, y2 = vals
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def area_xyxy(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = normalize_xyxy(box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def overlap_metrics(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    ax1, ay1, ax2, ay2 = normalize_xyxy(a)
    bx1, by1, bx2, by2 = normalize_xyxy(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(1e-9, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-9, (bx2 - bx1) * (by2 - by1))
    union = max(1e-9, area_a + area_b - inter)
    acx, acy = 0.5 * (ax1 + ax2), 0.5 * (ay1 + ay2)
    bcx, bcy = 0.5 * (bx1 + bx2), 0.5 * (by1 + by2)
    center_dist = math.sqrt((acx - bcx) ** 2 + (acy - bcy) ** 2)
    center_score = clamp(1.0 - center_dist / 0.75)
    containment = inter / max(1e-9, min(area_a, area_b))
    iou = inter / union
    proxy = clamp(0.55 * iou + 0.25 * containment + 0.20 * center_score)
    return {
        "iou": iou,
        "containment": containment,
        "center_score": center_score,
        "center_dist": center_dist,
        "proxy": proxy,
    }


def t6_sort_key(row: dict[str, Any]) -> tuple[int, float, float]:
    return (
        1 if row.get("selected_by_ranker") else 0,
        safe_float(row.get("score_rank_pct_by_image"), 0.0),
        safe_float(row.get("ranker_score"), 0.0),
    )


def group_t6_labels(rows: Sequence[dict[str, Any]], top_k: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        image_id = str(row.get("image_id", "")).strip()
        if image_id:
            grouped[image_id].append(row)
    limit = max(1, int(top_k))
    return {key: sorted(vals, key=t6_sort_key, reverse=True)[:limit] for key, vals in grouped.items()}


def mode_strength(mode_name: str) -> float:
    mode = str(mode_name or "").lower()
    if mode == "landscape":
        return 1.0
    if mode.startswith("single_person"):
        return 0.65
    if mode.startswith("group"):
        return 0.55
    if mode.startswith("object"):
        return 0.50
    if mode == "face":
        return 0.25
    return 0.50


def ar_strength(target_ar: str) -> float:
    ar = str(target_ar or "").upper()
    if ar == "FREE":
        return 1.0
    if ar in {"1:1", "3:4", "4:3"}:
        return 0.55
    if ar in {"16:9", "9:16"}:
        return 0.35
    return 0.50


def compute_weight_and_tier(
    *,
    gt_flag: bool,
    mode_name: str,
    target_ar: str,
    t6_proxy: float,
    rank_pct: float,
    fatal: bool,
    contradiction: bool,
) -> tuple[float, str, bool]:
    strength = mode_strength(mode_name) * ar_strength(target_ar)
    quality = clamp(0.55 * rank_pct + 0.45 * t6_proxy)
    risky = bool(fatal or contradiction)
    if gt_flag:
        low_proxy_penalty = strength * clamp(0.45 - t6_proxy, 0.0, 0.45) * 0.80
        weight = 0.65 + 0.35 * quality - low_proxy_penalty
        if t6_proxy >= 0.55 and rank_pct >= 0.80 and not risky:
            weight += 0.10 * strength
            tier = "mode_positive_t6_aligned"
        elif quality >= 0.45 and not fatal:
            tier = "mode_positive_t6_neutral"
        else:
            tier = "mode_positive_t6_weak"
        if risky:
            weight *= 0.60 if fatal else 0.80
        return clamp(weight, 0.05, 1.0), tier, False

    conflict = bool(t6_proxy >= 0.55 and rank_pct >= 0.80 and not fatal)
    if conflict:
        weight = 0.20 + 0.25 * (1.0 - strength)
        tier = "mode_negative_t6_conflict"
    else:
        weight = 0.85
        tier = "mode_negative_clear"
    if fatal:
        weight = max(weight, 0.80)
    return clamp(weight, 0.05, 1.0), tier, conflict


def compact_t6_payload(row: dict[str, Any], metrics: dict[str, float]) -> dict[str, Any]:
    return {
        "available": True,
        "matched_candidate_id": str(row.get("candidate_id", "")),
        "gt_annotation_id": row.get("gt_annotation_id"),
        "mos": safe_float(row.get("mos"), 0.0),
        "ranker_method": str(row.get("ranker_method", "")),
        "ranker_score": safe_float(row.get("ranker_score"), 0.0),
        "score_rank_pct_by_image": clamp(safe_float(row.get("score_rank_pct_by_image"), 0.0)),
        "selected_by_ranker": bool(row.get("selected_by_ranker", False)),
        "label_quality_tier": str(row.get("label_quality_tier", "")),
        "training_weight": clamp(safe_float(row.get("training_weight"), 1.0), 0.05, 1.0),
        "explanation_score": safe_float(row.get("explanation_score"), 0.0),
        "explanation_confidence": safe_float(row.get("explanation_confidence"), 0.0),
        "fatal_flag": bool(row.get("fatal_flag", False)),
        "contradiction_flag": bool(row.get("contradiction_flag", False)),
        "warnings": list(row.get("warnings") or []),
        "bbox_norm_xyxy": normalize_xyxy(row.get("bbox_norm_xyxy") or [0.0, 0.0, 1.0, 1.0]),
        "match_iou": round(float(metrics["iou"]), 6),
        "match_containment": round(float(metrics["containment"]), 6),
        "match_center_score": round(float(metrics["center_score"]), 6),
        "match_center_dist": round(float(metrics["center_dist"]), 6),
        "proxy_score": round(float(metrics["proxy"]), 6),
    }


def attach_sidecar_to_coco(
    coco: dict[str, Any],
    *,
    t6_by_image: dict[str, list[dict[str, Any]]],
    policy: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    image_meta = {safe_int(img.get("id")): img for img in coco.get("images", [])}
    counters: Counter[str] = Counter()
    proxy_sum = 0.0
    proxy_count = 0
    out = json.loads(json.dumps(coco, ensure_ascii=False))
    for ann in out.get("annotations", []):
        image_id = safe_int(ann.get("image_id"))
        img = image_meta.get(image_id, {})
        source_image_id = str(img.get("source_image_id") or Path(str(img.get("file_name", ""))).stem)
        width = safe_float(img.get("width"), 1.0)
        height = safe_float(img.get("height"), 1.0)
        attrs = ann.setdefault("attributes", {})
        mode_name = str(ann.get("mode_name") or attrs.get("mode_name") or "")
        target_ar = str(attrs.get("target_ar") or "FREE")
        gt_flag = bool(safe_int(ann.get("gt_flag"), 0))
        ann_box = bbox_xywh_to_norm_xyxy(ann.get("bbox") or [0.0, 0.0, width, height], width, height)
        candidates = t6_by_image.get(source_image_id, [])
        if not candidates:
            sidecar = {"available": False, "reason": "missing_t6_labels", "source_image_id": source_image_id}
            suggested_weight, label_tier, conflict = (1.0, "missing_t6", False)
            rank_pct = 0.0
            proxy = 0.0
            counters["missing_t6"] += 1
        else:
            best_row: dict[str, Any] | None = None
            best_metrics: dict[str, float] | None = None
            for row in candidates:
                metrics = overlap_metrics(ann_box, row.get("bbox_norm_xyxy") or [0.0, 0.0, 1.0, 1.0])
                if best_metrics is None or metrics["proxy"] > best_metrics["proxy"]:
                    best_metrics = metrics
                    best_row = row
            assert best_row is not None and best_metrics is not None
            sidecar = compact_t6_payload(best_row, best_metrics)
            rank_pct = clamp(safe_float(best_row.get("score_rank_pct_by_image"), 0.0))
            proxy = clamp(best_metrics["proxy"])
            suggested_weight, label_tier, conflict = compute_weight_and_tier(
                gt_flag=gt_flag,
                mode_name=mode_name,
                target_ar=target_ar,
                t6_proxy=proxy,
                rank_pct=rank_pct,
                fatal=bool(best_row.get("fatal_flag", False)),
                contradiction=bool(best_row.get("contradiction_flag", False)),
            )
            counters["matched_t6"] += 1
            if proxy >= 0.55:
                counters["high_proxy"] += 1
            if gt_flag and proxy < 0.20:
                counters["low_proxy_positive"] += 1
            if conflict:
                counters["negative_conflict_with_t6"] += 1
            proxy_sum += proxy
            proxy_count += 1

        attrs["t6_teacher"] = sidecar
        attrs["t6_proxy_score"] = round(proxy, 6)
        attrs["t6_score_rank_pct_by_image"] = round(rank_pct, 6)
        attrs["label_tier_t6"] = label_tier
        attrs["suggested_training_weight_t6"] = round(suggested_weight, 6)
        attrs["training_weight_t6"] = round(suggested_weight if policy == "weighted_v1" else 1.0, 6)
        attrs["negative_conflict_with_t6"] = int(bool(conflict))
        counters[f"tier::{label_tier}"] += 1
        counters[f"target_ar::{target_ar}"] += 1
        counters[f"mode::{mode_name}"] += 1

    summary = {
        "schema_version": "multimode_t6_sidecar_v1",
        "policy": policy,
        "image_count": len(out.get("images", [])),
        "annotation_count": len(out.get("annotations", [])),
        "matched_t6_annotation_count": counters.get("matched_t6", 0),
        "missing_t6_annotation_count": counters.get("missing_t6", 0),
        "mean_t6_proxy_score": proxy_sum / proxy_count if proxy_count else 0.0,
        "counter": dict(sorted(counters.items())),
    }
    return out, summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_label = output_dir / "label_json" / "multimode_labels_t6_sidecar_full.json"
    output_target_ar = output_dir / "label_json" / "multimode_labels_target_ar_only_t6_sidecar_full.json"
    summary_path = output_dir / "summary_t6_sidecar.json"
    t6_rows = read_jsonl(Path(args.t6_labels_jsonl))
    t6_by_image = group_t6_labels(t6_rows, top_k=int(args.top_k))

    multimode_label_json = args.multimode_label_json
    coco = read_json(Path(multimode_label_json))
    attached, summary = attach_sidecar_to_coco(coco, t6_by_image=t6_by_image, policy=str(args.policy))
    write_json(output_label, attached)
    summary["t6_label_image_count"] = len(t6_by_image)
    summary["t6_labels_jsonl"] = str(args.t6_labels_jsonl)
    summary["multimode_label_json"] = str(multimode_label_json)
    summary["output_label_json"] = str(output_label)
    baseline_summary_path = Path(args.baseline_summary_json) if args.baseline_summary_json else None
    if baseline_summary_path and baseline_summary_path.exists():
        baseline_summary = read_json(baseline_summary_path)
        for key in (
            "query_count",
            "source_image_entry_count",
            "mode_summary",
            "target_ars",
            "categories",
            "negative_count",
            "positive_count",
            "config",
        ):
            if key in baseline_summary and key not in summary:
                summary[key] = baseline_summary[key]
        summary["baseline_summary_json"] = str(baseline_summary_path)

    target_ar_arg = args.target_ar_only_label_json
    target_ar_input = Path(target_ar_arg) if target_ar_arg else None
    if target_ar_input and target_ar_input.exists():
        target_coco = read_json(target_ar_input)
        target_attached, target_summary = attach_sidecar_to_coco(
            target_coco,
            t6_by_image=t6_by_image,
            policy=str(args.policy),
        )
        write_json(output_target_ar, target_attached)
        summary["target_ar_only"] = {
            "input_label_json": str(target_ar_input),
            "output_label_json": str(output_target_ar),
            "annotation_count": target_summary["annotation_count"],
            "matched_t6_annotation_count": target_summary["matched_t6_annotation_count"],
            "missing_t6_annotation_count": target_summary["missing_t6_annotation_count"],
            "mean_t6_proxy_score": target_summary["mean_t6_proxy_score"],
        }

    if args.query_status_jsonl:
        query_src = Path(args.query_status_jsonl)
        if query_src.exists():
            query_dst = output_dir / "mode_query_status_t6_sidecar.jsonl"
            query_dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(query_src, query_dst)
            summary["query_status_jsonl"] = str(query_dst)

    categories_src = Path(args.categories_json) if args.categories_json else None
    if categories_src and categories_src.exists():
        categories_dst = output_dir / "categories.json"
        categories_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(categories_src, categories_dst)
        summary["categories_json"] = str(categories_dst)

    write_json(summary_path, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach T6 score-only sidecar signals to multi-mode annotation label JSON.")
    parser.add_argument("--multimode_label_json", required=True)
    parser.add_argument("--target_ar_only_label_json", default="")
    parser.add_argument("--t6_labels_jsonl", required=True)
    parser.add_argument("--query_status_jsonl", default="")
    parser.add_argument("--categories_json", default="")
    parser.add_argument("--baseline_summary_json", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--policy", choices=["sidecar_only", "weighted_v1"], default="weighted_v1")
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
