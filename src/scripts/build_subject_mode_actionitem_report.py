#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not math.isfinite(x):
        return float(default)
    return x


def percentile(values: Sequence[float], q: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    q = max(0.0, min(1.0, float(q)))
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    alpha = pos - lo
    return (1.0 - alpha) * vals[lo] + alpha * vals[hi]


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    den = max(1e-8, den_x * den_y)
    return float(num / den)


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [safe_float(v) for v in a]
    bx1, by1, bx2, by2 = [safe_float(v) for v in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    ua = max(1e-8, (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return float(inter / ua)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_routed_map(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for rec in load_jsonl(path):
        image_id = str(rec.get("image_id", ""))
        if image_id:
            out[image_id] = rec
    return out


def build_top1_map(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for rec in load_jsonl(path):
        image_id = str(rec.get("image_id", ""))
        route_global = rec.get("route_global", {}) if isinstance(rec.get("route_global"), dict) else {}
        by_ar = rec.get("teacher_scorer", {}).get("results_by_ar", {})
        if not isinstance(by_ar, dict):
            continue
        for ar_text, ar_res in by_ar.items():
            if not isinstance(ar_res, dict):
                continue
            topk = ar_res.get("selected_topk", [])
            if not isinstance(topk, list) or not topk or not isinstance(topk[0], dict):
                continue
            routing: Dict[str, Any] = {}
            if isinstance(route_global, dict):
                routing.update(route_global)
            if isinstance(ar_res.get("routing"), dict):
                routing.update(ar_res.get("routing"))
            top1 = topk[0]
            out[(image_id, str(ar_text))] = {
                "image_id": image_id,
                "target_ar": str(ar_text),
                "bbox_norm_xyxy": top1.get("bbox_norm_xyxy"),
                "final_score": safe_float(top1.get("scores", {}).get("final", 0.0), 0.0),
                "checklist": top1.get("checklist", {}) if isinstance(top1.get("checklist"), dict) else {},
                "composition_checks": (
                    top1.get("composition_checks", {})
                    if isinstance(top1.get("composition_checks"), dict)
                    else {}
                ),
                "routing": routing or route_global,
            }
    return out


def horizon_state(exists_prob: float, conf: float, exists_thr: float, conf_thr: float) -> str:
    if exists_prob < exists_thr:
        return "none"
    if conf < conf_thr:
        return "weak"
    return "strong"


def choose_scene_samples(
    current_routed: Dict[str, Dict[str, Any]],
    baseline_routed: Dict[str, Dict[str, Any]],
    max_samples: int = 30,
) -> List[Dict[str, Any]]:
    by_subtype: Dict[str, List[str]] = defaultdict(list)
    for image_id, rec in current_routed.items():
        routing = rec.get("routing", {}) if isinstance(rec.get("routing"), dict) else {}
        if str(routing.get("subject_mode", "")) != "scene_general":
            continue
        subtype = str(routing.get("scene_subtype", "scene_general_unknown"))
        by_subtype[subtype].append(image_id)
    out: List[Dict[str, Any]] = []
    for subtype in sorted(by_subtype):
        for image_id in sorted(by_subtype[subtype])[: max(1, max_samples // max(1, len(by_subtype)))]:
            old_routing = baseline_routed.get(image_id, {}).get("routing", {})
            new_routing = current_routed.get(image_id, {}).get("routing", {})
            out.append(
                {
                    "image_id": image_id,
                    "old_mode": str((old_routing or {}).get("subject_mode", "missing")),
                    "new_mode": str((new_routing or {}).get("subject_mode", "missing")),
                    "scene_subtype": str((new_routing or {}).get("scene_subtype", "")),
                    "old_reasons": (old_routing or {}).get("subject_mode_reasons", []),
                    "new_reasons": (new_routing or {}).get("subject_mode_reasons", []),
                }
            )
            if len(out) >= max_samples:
                return out
    return out


def experiment_a(
    baseline_routed: Dict[str, Dict[str, Any]],
    current_routed: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    old_counts = Counter()
    new_counts = Counter()
    subtype_counts = Counter()
    for rec in baseline_routed.values():
        routing = rec.get("routing", {}) if isinstance(rec.get("routing"), dict) else {}
        old_counts[str(routing.get("subject_mode", "unknown"))] += 1
    for rec in current_routed.values():
        routing = rec.get("routing", {}) if isinstance(rec.get("routing"), dict) else {}
        new_counts[str(routing.get("subject_mode", "unknown"))] += 1
        if str(routing.get("subject_mode", "")) == "scene_general":
            subtype_counts[str(routing.get("scene_subtype", "scene_general_unknown"))] += 1
    old_scene_ids = {
        image_id
        for image_id, rec in baseline_routed.items()
        if str((rec.get("routing", {}) or {}).get("subject_mode", "")) == "scene_landscape"
    }
    current_scene_ids = {
        image_id
        for image_id, rec in current_routed.items()
        if str((rec.get("routing", {}) or {}).get("subject_mode", "")) == "scene_general"
    }
    return {
        "old_mode_counts": dict(old_counts),
        "new_mode_counts": dict(new_counts),
        "scene_old_count": len(old_scene_ids),
        "scene_new_count": len(current_scene_ids),
        "scene_overlap_count": len(old_scene_ids & current_scene_ids),
        "scene_subtype_counts": dict(subtype_counts),
        "scene_samples": choose_scene_samples(current_routed, baseline_routed, max_samples=30),
    }


def experiment_b(current_routed: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    combos: List[Dict[str, Any]] = []
    for exists_thr in (0.15, 0.20, 0.25):
        for conf_thr in (0.30, 0.35, 0.45):
            landscape_total = 0
            landscape_strong = 0
            non_scene_total = 0
            non_scene_fp = 0
            reflection_total = 0
            reflection_mid = 0
            for rec in current_routed.values():
                routing = rec.get("routing", {}) if isinstance(rec.get("routing"), dict) else {}
                c5 = rec.get("c5_geom", {}) if isinstance(rec.get("c5_geom"), dict) else {}
                hr = c5.get("horizon_roll", {}) if isinstance(c5.get("horizon_roll"), dict) else {}
                exists_prob = safe_float(hr.get("conf", c5.get("horizon_conf", 0.0)), 0.0)
                conf = safe_float(hr.get("conf", c5.get("horizon_conf", 0.0)), 0.0)
                y_norm = hr.get("horizon_y_norm")
                state = horizon_state(exists_prob, conf, exists_thr, conf_thr)
                mode = str(routing.get("subject_mode", ""))
                subtype = str(routing.get("scene_subtype", "scene_general_unknown"))
                if subtype == "scene_landscape_nature":
                    landscape_total += 1
                    if state == "strong":
                        landscape_strong += 1
                if mode != "scene_general":
                    non_scene_total += 1
                    if state == "strong":
                        non_scene_fp += 1
                if subtype == "scene_reflection_symmetry":
                    reflection_total += 1
                    if state == "strong" and y_norm is not None and abs(safe_float(y_norm, 0.0) - 0.5) <= 0.12:
                        reflection_mid += 1
            combos.append(
                {
                    "exists_thr": exists_thr,
                    "conf_thr": conf_thr,
                    "landscape_strong_rate": landscape_strong / max(1, landscape_total),
                    "non_scene_fp_rate": non_scene_fp / max(1, non_scene_total),
                    "reflection_mid_capture_rate": reflection_mid / max(1, reflection_total),
                }
            )
    combos.sort(key=lambda x: (-x["landscape_strong_rate"], x["non_scene_fp_rate"], -x["reflection_mid_capture_rate"]))
    return {
        "grid": combos,
        "recommended": combos[0] if combos else {},
    }


def collect_scale_context_records(top1_map: Dict[Tuple[str, str], Dict[str, Any]]) -> Tuple[List[float], List[float], Dict[str, Dict[str, float]]]:
    scale_vals: List[float] = []
    context_vals: List[float] = []
    mode_buckets: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: {"scale": [], "context": []})
    for row in top1_map.values():
        checklist = row.get("checklist", {}) if isinstance(row.get("checklist"), dict) else {}
        scale = checklist.get("subject_scale", {}) if isinstance(checklist.get("subject_scale"), dict) else {}
        context = checklist.get("context", {}) if isinstance(checklist.get("context"), dict) else {}
        if "value" not in scale or "value" not in context:
            continue
        scale_v = safe_float(scale.get("value", 0.0), 0.0)
        context_v = safe_float(context.get("value", 0.0), 0.0)
        mode = str((row.get("routing", {}) or {}).get("subject_mode", "unknown"))
        scale_vals.append(scale_v)
        context_vals.append(context_v)
        mode_buckets[mode]["scale"].append(scale_v)
        mode_buckets[mode]["context"].append(context_v)
    summary = {
        mode: {
            "subject_scale_mean": mean(vals["scale"]) if vals["scale"] else 0.0,
            "context_mean": mean(vals["context"]) if vals["context"] else 0.0,
            "count": len(vals["scale"]),
        }
        for mode, vals in sorted(mode_buckets.items())
    }
    return scale_vals, context_vals, summary


def experiment_c(
    baseline_top1: Dict[Tuple[str, str], Dict[str, Any]],
    current_top1: Dict[Tuple[str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    base_scale, base_context, base_by_mode = collect_scale_context_records(baseline_top1)
    cur_scale, cur_context, cur_by_mode = collect_scale_context_records(current_top1)
    return {
        "baseline_corr": pearson(base_scale, base_context),
        "current_corr": pearson(cur_scale, cur_context),
        "baseline_by_mode": base_by_mode,
        "current_by_mode": cur_by_mode,
    }


def experiment_d(current_routed: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    all_counts = Counter()
    bg_counts = Counter()
    gate_passed = 0
    total = 0
    for rec in current_routed.values():
        routing = rec.get("routing", {}) if isinstance(rec.get("routing"), dict) else {}
        copyspace = routing.get("copyspace", {}) if isinstance(routing.get("copyspace"), dict) else {}
        fired_by = str(copyspace.get("mode_fired_by", "none"))
        all_counts[fired_by] += 1
        total += 1
        if bool(copyspace.get("gate_passed", False)):
            gate_passed += 1
        if str(routing.get("subject_mode", "")) == "background_texture_copyspace":
            bg_counts[fired_by] += 1
    return {
        "all_mode_fired_by_counts": dict(all_counts),
        "copyspace_mode_mode_fired_by_counts": dict(bg_counts),
        "copyspace_gate_passed_rate": gate_passed / max(1, total),
    }


def experiment_e(
    baseline_top1: Dict[Tuple[str, str], Dict[str, Any]],
    current_top1: Dict[Tuple[str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    keys = [
        key
        for key, row in current_top1.items()
        if str((row.get("routing", {}) or {}).get("scene_subtype", "")) == "scene_landscape_nature" and key in baseline_top1
    ]
    bbox_ious: List[float] = []
    old_horizon_dist: List[float] = []
    new_horizon_dist: List[float] = []
    old_horizon_na = 0
    new_horizon_na = 0
    new_on_target = 0
    changed = 0
    roll_constancy_by_image: Dict[str, List[float]] = defaultdict(list)
    for key in keys:
        old_row = baseline_top1[key]
        new_row = current_top1[key]
        old_box = old_row.get("bbox_norm_xyxy")
        new_box = new_row.get("bbox_norm_xyxy")
        if isinstance(old_box, list) and isinstance(new_box, list) and len(old_box) == 4 and len(new_box) == 4:
            iou = iou_xyxy(old_box, new_box)
            bbox_ious.append(iou)
            if iou < 0.98:
                changed += 1
        old_h = old_row.get("checklist", {}).get("horizon", {})
        new_h = new_row.get("checklist", {}).get("horizon", {})
        old_label = str((old_h or {}).get("label", ""))
        new_label = str((new_h or {}).get("label", ""))
        if old_label == "horizon_na":
            old_horizon_na += 1
        if new_label == "horizon_na":
            new_horizon_na += 1
        if new_label == "horizon_on_target":
            new_on_target += 1
        if (old_h or {}).get("third_dist") is not None:
            old_horizon_dist.append(safe_float((old_h or {}).get("third_dist", 0.0), 0.0))
        if (new_h or {}).get("third_dist") is not None:
            new_horizon_dist.append(safe_float((new_h or {}).get("third_dist", 0.0), 0.0))
        roll_val = (new_row.get("composition_checks", {}).get("roll", {}) or {}).get("value")
        if roll_val is not None:
            roll_constancy_by_image[key[0]].append(safe_float(roll_val, 0.0))
    roll_spread = [
        max(vals) - min(vals)
        for vals in roll_constancy_by_image.values()
        if vals
    ]
    return {
        "task_count": len(keys),
        "bbox_changed_rate": changed / max(1, len(keys)),
        "bbox_iou_p50": percentile(bbox_ious, 0.50),
        "bbox_iou_p90": percentile(bbox_ious, 0.90),
        "baseline_horizon_na_rate": old_horizon_na / max(1, len(keys)),
        "current_horizon_na_rate": new_horizon_na / max(1, len(keys)),
        "current_horizon_on_target_rate": new_on_target / max(1, len(keys)),
        "baseline_horizon_dist_mean": mean(old_horizon_dist) if old_horizon_dist else 0.0,
        "current_horizon_dist_mean": mean(new_horizon_dist) if new_horizon_dist else 0.0,
        "roll_spread_p50": percentile(roll_spread, 0.50),
        "roll_spread_p90": percentile(roll_spread, 0.90),
    }


def build_markdown(report: Dict[str, Any], baseline_tag: str, current_tag: str) -> str:
    a = report["experiment_a"]
    b = report["experiment_b"]
    c = report["experiment_c"]
    d = report["experiment_d"]
    e = report["experiment_e"]
    lines = [
        f"# SubjectMode / Horizon / Context / Copyspace ActionItem 실험 리포트",
        "",
        f"- baseline: `{baseline_tag}`",
        f"- current: `{current_tag}`",
        "",
        "## A. mode rename + subtype sanity",
        f"- old `scene_landscape`: {a['scene_old_count']}장",
        f"- new `scene_general`: {a['scene_new_count']}장",
        f"- overlap: {a['scene_overlap_count']}장",
        f"- scene_subtype 분포: `{json.dumps(a['scene_subtype_counts'], ensure_ascii=False)}`",
        "",
        "## B. horizon threshold sweep",
        f"- 추천 조합: exists={b['recommended'].get('exists_thr', 'na')}, conf={b['recommended'].get('conf_thr', 'na')}",
        f"- landscape strong rate: {b['recommended'].get('landscape_strong_rate', 0.0):.4f}",
        f"- non-scene FP rate: {b['recommended'].get('non_scene_fp_rate', 0.0):.4f}",
        f"- reflection 1/2 capture rate: {b['recommended'].get('reflection_mid_capture_rate', 0.0):.4f}",
        "",
        "## C. subject_scale / context 분리",
        f"- baseline corr: {c['baseline_corr']:.4f}",
        f"- current corr: {c['current_corr']:.4f}",
        "",
        "## D. copyspace fired_by",
        f"- 전체 fired_by: `{json.dumps(d['all_mode_fired_by_counts'], ensure_ascii=False)}`",
        f"- copyspace mode fired_by: `{json.dumps(d['copyspace_mode_mode_fired_by_counts'], ensure_ascii=False)}`",
        f"- gate_passed rate: {d['copyspace_gate_passed_rate']:.4f}",
        "",
        "## E. scene_landscape_nature top1 비교",
        f"- task 수: {e['task_count']}",
        f"- bbox changed rate: {e['bbox_changed_rate']:.4f}",
        f"- old/new horizon_na rate: {e['baseline_horizon_na_rate']:.4f} -> {e['current_horizon_na_rate']:.4f}",
        f"- current horizon_on_target rate: {e['current_horizon_on_target_rate']:.4f}",
        f"- old/new horizon dist mean: {e['baseline_horizon_dist_mean']:.4f} -> {e['current_horizon_dist_mean']:.4f}",
        f"- roll spread p50/p90 across AR tasks: {e['roll_spread_p50']:.4f} / {e['roll_spread_p90']:.4f}",
        "",
        "## 결론",
        "- `scene_landscape`는 `scene_general + scene_subtype`로 분해되어 scene 해석이 더 구체화되었는지 확인한다.",
        "- horizon은 subtype-aware threshold와 target set으로 재해석했고, roll은 AR 간 상수에 가까운 image-level QA 힌트인지 확인한다.",
        "- subject_scale/context는 baseline 대비 상관이 낮아졌는지 확인한다.",
        "- copyspace는 `mode_fired_by`와 `gate_passed`로 발화 근거를 추적한다.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build action-item experiment report for subject-mode v2 rerun.")
    p.add_argument("--baseline_routed_jsonl", required=True)
    p.add_argument("--current_routed_jsonl", required=True)
    p.add_argument("--baseline_teacher_jsonl", required=True)
    p.add_argument("--current_teacher_jsonl", required=True)
    p.add_argument("--baseline_tag", default="baseline")
    p.add_argument("--current_tag", default="current")
    p.add_argument("--output_json", required=True)
    p.add_argument("--output_md", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    baseline_routed = load_routed_map(Path(args.baseline_routed_jsonl))
    current_routed = load_routed_map(Path(args.current_routed_jsonl))
    baseline_top1 = build_top1_map(Path(args.baseline_teacher_jsonl))
    current_top1 = build_top1_map(Path(args.current_teacher_jsonl))

    report = {
        "baseline_tag": args.baseline_tag,
        "current_tag": args.current_tag,
        "experiment_a": experiment_a(baseline_routed, current_routed),
        "experiment_b": experiment_b(current_routed),
        "experiment_c": experiment_c(baseline_top1, current_top1),
        "experiment_d": experiment_d(current_routed),
        "experiment_e": experiment_e(baseline_top1, current_top1),
    }
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(build_markdown(report, args.baseline_tag, args.current_tag), encoding="utf-8")
    print(f"[done] output_json={out_json}")
    print(f"[done] output_md={out_md}")


if __name__ == "__main__":
    main()
