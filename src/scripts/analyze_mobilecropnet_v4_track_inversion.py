#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable


SPLITS = ("train", "val", "test")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="UCTR teacher가 public cropper보다 강한데 SSTK 학생 모델은 public 트랙이 강한 현상을 분석한다."
    )
    parser.add_argument("--public_root", type=Path, required=True)
    parser.add_argument("--uctr_root", type=Path, required=True)
    parser.add_argument("--leaderboard_json", type=Path, required=True)
    parser.add_argument("--teacher_leaderboard_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_rows_per_split", type=int, default=0)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _f(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _fmt(value: Any, digits: int = 6) -> str:
    val = _f(value)
    if val is None:
        return "-"
    return f"{val:.{digits}f}"


def _safe_mean(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else None


def _quantiles(values: list[float]) -> dict[str, float | None]:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return {"mean": None, "median": None, "p10": None, "p90": None, "min": None, "max": None}
    def q(prob: float) -> float:
        idx = min(len(vals) - 1, max(0, int(round(prob * (len(vals) - 1)))))
        return vals[idx]
    return {
        "mean": sum(vals) / len(vals),
        "median": median(vals),
        "p10": q(0.10),
        "p90": q(0.90),
        "min": vals[0],
        "max": vals[-1],
    }


def _bbox_area(bbox: Any) -> float | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    vals = [_f(v) for v in bbox]
    if any(v is None for v in vals):
        return None
    x1, y1, x2, y2 = [float(v) for v in vals]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_ar(bbox: Any) -> float | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    vals = [_f(v) for v in bbox]
    if any(v is None for v in vals):
        return None
    x1, y1, x2, y2 = [float(v) for v in vals]
    h = max(0.0, y2 - y1)
    if h <= 1e-9:
        return None
    return max(0.0, x2 - x1) / h


def _bbox_center_dist(a: Any, b: Any) -> float | None:
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != 4 or len(b) != 4:
        return None
    av = [_f(v) for v in a]
    bv = [_f(v) for v in b]
    if any(v is None for v in av + bv):
        return None
    ax1, ay1, ax2, ay2 = [float(v) for v in av]
    bx1, by1, bx2, by2 = [float(v) for v in bv]
    acx, acy = (ax1 + ax2) / 2.0, (ay1 + ay2) / 2.0
    bcx, bcy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
    return math.sqrt((acx - bcx) ** 2 + (acy - bcy) ** 2)


def _bbox_iou(a: Any, b: Any) -> float | None:
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != 4 or len(b) != 4:
        return None
    av = [_f(v) for v in a]
    bv = [_f(v) for v in b]
    if any(v is None for v in av + bv):
        return None
    ax1, ay1, ax2, ay2 = [float(v) for v in av]
    bx1, by1, bx2, by2 = [float(v) for v in bv]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 1e-9 else None


def _iter_jsonl(path: Path, max_rows: int = 0):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, 1):
            if max_rows and idx > max_rows:
                break
            if line.strip():
                yield json.loads(line)


def _candidate_score(candidate: dict[str, Any], key: str) -> float | None:
    val = _f(candidate.get(key))
    if val is not None:
        return val
    score_targets = candidate.get("score_targets") if isinstance(candidate.get("score_targets"), dict) else {}
    return _f(score_targets.get(key))


def _all_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key in ("matching_targets", "candidate_pool", "ignored_candidates", "overflow_candidates"):
        values = row.get(key)
        if isinstance(values, list):
            out.extend([item for item in values if isinstance(item, dict)])
    return out


def _positive_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [cand for cand in _all_candidates(row) if cand.get("is_positive_candidate") is True]


def _top_positive(row: dict[str, Any]) -> dict[str, Any] | None:
    positives = _positive_candidates(row)
    if not positives:
        return None
    return max(positives, key=lambda item: _candidate_score(item, "crop_utility_prob") or -1e9)


def _track_summary(root: Path, max_rows_per_split: int = 0) -> dict[str, Any]:
    split_out: dict[str, Any] = {}
    for split in SPLITS:
        path = root / split / "train_conditional_detr_batch.jsonl"
        counters: dict[str, Counter[str]] = defaultdict(Counter)
        vals: dict[str, list[float]] = defaultdict(list)
        rows = 0
        top_positive_index: dict[str, dict[str, Any]] = {}
        for row in _iter_jsonl(path, max_rows_per_split) or []:
            rows += 1
            counters["target_ar"][str(row.get("target_ar") or "UNKNOWN")] += 1
            routing = row.get("routing") if isinstance(row.get("routing"), dict) else {}
            counters["subject_mode"][str(routing.get("subject_mode") or "UNKNOWN")] += 1
            counters["policy_id"][str(routing.get("policy_id") or "UNKNOWN")] += 1
            decision = row.get("decision_target") if isinstance(row.get("decision_target"), dict) else {}
            counters["decision_type"][str(decision.get("decision_type") or "UNKNOWN")] += 1
            teacher_meta = row.get("teacher_meta") if isinstance(row.get("teacher_meta"), dict) else {}
            for key in ("teacher_confidence", "target_reliability", "route_conf", "candidate_count", "selectable_candidate_count"):
                val = _f(teacher_meta.get(key))
                if val is not None:
                    vals[f"teacher_meta.{key}"].append(val)
            subject_bbox = routing.get("subject_prior_bbox_norm_xyxy")
            area = _bbox_area(subject_bbox)
            if area is not None:
                vals["subject_prior.area"].append(area)
                vals["subject_prior.ar"].append(_bbox_ar(subject_bbox) or 0.0)
            vals["subject_prior.valid"].append(1.0 if area is not None and area > 1e-6 else 0.0)

            matching = row.get("matching_targets") if isinstance(row.get("matching_targets"), list) else []
            pool = row.get("candidate_pool") if isinstance(row.get("candidate_pool"), list) else []
            ignored = row.get("ignored_candidates") if isinstance(row.get("ignored_candidates"), list) else []
            overflow = row.get("overflow_candidates") if isinstance(row.get("overflow_candidates"), list) else []
            vals["row.matching_count"].append(float(len(matching)))
            vals["row.candidate_pool_count"].append(float(len(pool)))
            vals["row.ignored_count"].append(float(len(ignored)))
            vals["row.overflow_count"].append(float(len(overflow)))

            all_candidates = _all_candidates(row)
            positives = [cand for cand in all_candidates if cand.get("is_positive_candidate") is True]
            vals["row.all_candidate_count"].append(float(len(all_candidates)))
            vals["row.positive_count"].append(float(len(positives)))
            vals["row.hard_negative_count"].append(float(sum(cand.get("is_hard_negative") is True for cand in all_candidates)))
            vals["row.unsafe_negative_count"].append(float(sum(cand.get("is_unsafe_negative") is True for cand in all_candidates)))
            if all_candidates:
                scores = sorted(
                    [_candidate_score(cand, "crop_utility_prob") for cand in all_candidates if _candidate_score(cand, "crop_utility_prob") is not None],
                    reverse=True,
                )
                if len(scores) >= 2:
                    vals["row.top2_crop_utility_margin"].append(float(scores[0] - scores[1]))
            for cand in all_candidates:
                source = ((cand.get("external_score_teacher") or {}).get("score_source") if isinstance(cand.get("external_score_teacher"), dict) else None)
                counters["candidate_score_source"][str(source or "UNKNOWN")] += 1
                for key in ("score_prob", "crop_utility_prob", "score_rank_pct", "crop_utility_rank_pct", "score_raw_policy", "target_ar_log_error"):
                    val = _candidate_score(cand, key)
                    if val is not None:
                        vals[f"candidate.{key}"].append(val)
                bbox = cand.get("bbox_norm_xyxy")
                cand_area = _bbox_area(bbox)
                if cand_area is not None:
                    vals["candidate.area"].append(cand_area)
                cand_ar = _bbox_ar(bbox)
                if cand_ar is not None:
                    vals["candidate.ar"].append(cand_ar)
                safety = cand.get("safety_penalty") if isinstance(cand.get("safety_penalty"), dict) else {}
                for key in ("total", "soft_total", "hard_total"):
                    val = _f(safety.get(key))
                    if val is not None:
                        vals[f"candidate.safety_{key}"].append(val)

            top_pos = _top_positive(row)
            if top_pos:
                key = f"{row.get('image_id')}::{row.get('target_ar')}"
                top_positive_index[key] = {
                    "bbox_norm_xyxy": top_pos.get("bbox_norm_xyxy"),
                    "score_prob": _candidate_score(top_pos, "score_prob"),
                    "crop_utility_prob": _candidate_score(top_pos, "crop_utility_prob"),
                    "score_raw_policy": _candidate_score(top_pos, "score_raw_policy"),
                    "target_ar_log_error": _candidate_score(top_pos, "target_ar_log_error"),
                    "area": _bbox_area(top_pos.get("bbox_norm_xyxy")),
                    "ar": _bbox_ar(top_pos.get("bbox_norm_xyxy")),
                    "subject_mode": routing.get("subject_mode"),
                }
                for key_name in ("score_prob", "crop_utility_prob", "score_rank_pct", "crop_utility_rank_pct", "score_raw_policy", "target_ar_log_error"):
                    val = _candidate_score(top_pos, key_name)
                    if val is not None:
                        vals[f"positive.{key_name}"].append(val)
                top_area = _bbox_area(top_pos.get("bbox_norm_xyxy"))
                if top_area is not None:
                    vals["positive.area"].append(top_area)
                top_ar = _bbox_ar(top_pos.get("bbox_norm_xyxy"))
                if top_ar is not None:
                    vals["positive.ar"].append(top_ar)
                safety = top_pos.get("safety_penalty") if isinstance(top_pos.get("safety_penalty"), dict) else {}
                for key_name in ("total", "soft_total", "hard_total"):
                    val = _f(safety.get(key_name))
                    if val is not None:
                        vals[f"positive.safety_{key_name}"].append(val)

        split_out[split] = {
            "path": str(path),
            "exists": path.exists(),
            "row_count": rows,
            "counters": {key: dict(counter.most_common()) for key, counter in counters.items()},
            "stats": {key: _quantiles(values) for key, values in sorted(vals.items())},
            "top_positive_index": top_positive_index,
        }
    return split_out


def _paired_summary(public_summary: dict[str, Any], uctr_summary: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split in SPLITS:
        pub = public_summary.get(split, {}).get("top_positive_index", {})
        uctr = uctr_summary.get(split, {}).get("top_positive_index", {})
        common = sorted(set(pub) & set(uctr))
        vals: dict[str, list[float]] = defaultdict(list)
        for key in common:
            p = pub[key]
            u = uctr[key]
            iou = _bbox_iou(p.get("bbox_norm_xyxy"), u.get("bbox_norm_xyxy"))
            if iou is not None:
                vals["top_positive_bbox_iou"].append(iou)
            dist = _bbox_center_dist(p.get("bbox_norm_xyxy"), u.get("bbox_norm_xyxy"))
            if dist is not None:
                vals["top_positive_center_dist"].append(dist)
            for metric in ("score_prob", "crop_utility_prob", "score_raw_policy", "target_ar_log_error", "area", "ar"):
                pv = _f(p.get(metric))
                uv = _f(u.get(metric))
                if pv is not None and uv is not None:
                    vals[f"uctr_minus_public.{metric}"].append(uv - pv)
        out[split] = {
            "common_top_positive_rows": len(common),
            "public_top_positive_rows": len(pub),
            "uctr_top_positive_rows": len(uctr),
            "stats": {key: _quantiles(values) for key, values in sorted(vals.items())},
        }
    return out


def _teacher_rows(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    out: dict[str, Any] = {}
    for row in payload.get("rows", []):
        method = row.get("method")
        if method in {"universal_crop_teacher_h_stage3_gate128", "public_cropper_ensemble_best"}:
            out[method] = row
    return out


def _student_profile_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(path)
    rows = payload.get("rows", [])
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        track = row.get("track")
        variant = row.get("variant")
        profile = row.get("profile")
        if variant != "subjectprior_route_proposal_v1":
            continue
        if track == "SSTK public":
            out[f"public::{profile}"] = row
        if track == "SSTK UCTR subjectprior":
            out[f"uctr::{profile}"] = row
    return out


def _metric(stats: dict[str, Any], key: str, field: str = "mean") -> float | None:
    return _f((stats.get(key) or {}).get(field))


def _profile_table(rows: dict[str, dict[str, Any]]) -> str:
    profiles = sorted({key.split("::", 1)[1] for key in rows})
    lines = [
        "| profile | public equal4 | UCTR equal4 | public z | UCTR z | public routeBal | UCTR routeBal | public hit@0.5 | UCTR hit@0.5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in profiles:
        pub = rows.get(f"public::{profile}", {})
        uctr = rows.get(f"uctr::{profile}", {})
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                profile,
                _fmt(pub.get("equal4_raw_mean")),
                _fmt(uctr.get("equal4_raw_mean")),
                _fmt(pub.get("equal4_zscore")),
                _fmt(uctr.get("equal4_zscore")),
                _fmt(pub.get("route_balanced_accuracy")),
                _fmt(uctr.get("route_balanced_accuracy")),
                _fmt(pub.get("direct_test_final_positive_hit_iou_0_5")),
                _fmt(uctr.get("direct_test_final_positive_hit_iou_0_5")),
            )
        )
    return "\n".join(lines)


def _split_table(summary: dict[str, Any], track_label: str) -> str:
    lines = [
        f"### {track_label}",
        "",
        "| split | rows | pos/row | top2 margin | selectable | subject prior valid | subject prior area | positive area | candidate unsafe/row |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        item = summary.get(split, {})
        stats = item.get("stats", {})
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                split,
                item.get("row_count", 0),
                _fmt(_metric(stats, "row.positive_count"), 4),
                _fmt(_metric(stats, "row.top2_crop_utility_margin"), 6),
                _fmt(_metric(stats, "teacher_meta.selectable_candidate_count"), 3),
                _fmt(_metric(stats, "subject_prior.valid"), 4),
                _fmt(_metric(stats, "subject_prior.area"), 6),
                _fmt(_metric(stats, "positive.area"), 6),
                _fmt(_metric(stats, "row.unsafe_negative_count"), 4),
            )
        )
    return "\n".join(lines)


def _paired_table(summary: dict[str, Any]) -> str:
    lines = [
        "| split | paired rows | bbox IoU | center dist | UCTR-public area | UCTR-public crop utility | UCTR-public AR error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        item = summary.get(split, {})
        stats = item.get("stats", {})
        lines.append(
            "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                split,
                item.get("common_top_positive_rows", 0),
                _fmt(_metric(stats, "top_positive_bbox_iou")),
                _fmt(_metric(stats, "top_positive_center_dist")),
                _fmt(_metric(stats, "uctr_minus_public.area")),
                _fmt(_metric(stats, "uctr_minus_public.crop_utility_prob")),
                _fmt(_metric(stats, "uctr_minus_public.target_ar_log_error")),
            )
        )
    return "\n".join(lines)


def _build_markdown(payload: dict[str, Any]) -> str:
    teacher = payload["teacher_equal4"]
    uctr = teacher.get("universal_crop_teacher_h_stage3_gate128", {})
    public = teacher.get("public_cropper_ensemble_best", {})
    public_stats = payload["label_stats"]["public"]
    uctr_stats = payload["label_stats"]["uctr"]
    paired = payload["paired_positive"]
    lines = [
        "# MobileCropNet v4.0 SSTK UCTR/Public 역전 분석",
        "",
        f"- 생성 시각: `{payload['generated_at']}`",
        "- 목적: UCTR teacher는 equal-4 teacher benchmark에서 public cropper보다 강하지만, SSTK 학생 모델에서는 public 트랙이 더 강한 현상을 label distribution과 학생 평가 지표로 분해한다.",
        "",
        "## 1. Teacher 기준 재확인",
        "",
        "| teacher | equal4 raw | equal4 z | FCDB IoU | CPC weighted | GNMC IoU | GAIC primary | GAIC top1 MOS | GAIC SRCC | GAIC Accw4@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        "| `universal_crop_teacher_h_stage3_gate128` | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            _fmt(uctr.get("equal4_raw_mean")),
            _fmt(uctr.get("equal4_zscore")),
            _fmt(uctr.get("fcdb_iou_top1")),
            _fmt(uctr.get("cpc_weighted_pairwise")),
            _fmt(uctr.get("gnmc_iou_top1")),
            _fmt(uctr.get("gaic_primary")),
            _fmt(uctr.get("top1_mos")),
            _fmt(uctr.get("srcc")),
            _fmt(uctr.get("accw4_of_top10")),
        ),
        "| `public_cropper_ensemble_best` | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            _fmt(public.get("equal4_raw_mean")),
            _fmt(public.get("equal4_zscore")),
            _fmt(public.get("fcdb_iou_top1")),
            _fmt(public.get("cpc_weighted_pairwise")),
            _fmt(public.get("gnmc_iou_top1")),
            _fmt(public.get("gaic_primary")),
            _fmt(public.get("top1_mos")),
            _fmt(public.get("srcc")),
            _fmt(public.get("accw4_of_top10")),
        ),
        "",
        "해석: teacher 자체의 public/general geometry 성능은 UCTR이 public cropper보다 높다. 따라서 학생 모델의 public 우세는 teacher 성능 순위만으로 설명되지 않고, label 난이도, 목표 분포, head 학습 안정성, public benchmark와의 목적함수 정렬을 같이 봐야 한다.",
        "",
        "## 2. 학생 모델 프로파일 비교",
        "",
        _profile_table(payload["student_profiles"]),
        "",
        "## 3. Label Distribution 요약",
        "",
        _split_table(public_stats, "SSTK public label"),
        "",
        _split_table(uctr_stats, "SSTK UCTR label"),
        "",
        "## 4. 동일 이미지/target_AR positive crop 비교",
        "",
        _paired_table(paired),
        "",
        "## 5. 현재 결론",
        "",
        "- UCTR teacher는 equal-4 teacher benchmark에서 우세하지만, 현재 학생 모델은 `SSTK public / balanced_288`이 equal-4 raw/z와 public deployment quality에서 더 안정적이다.",
        "- UCTR 계열은 direct hit@0.5가 높게 나오는 profile이 있어 positive proposal alignment는 강하지만, route balanced accuracy와 subject bbox valid calibration이 release gate를 막고 있다.",
        "- 따라서 다음 개선은 UCTR를 폐기하는 방향이 아니라 `UCTR crop target + public/general geometry distillation + route-balanced sampling + subject valid calibration`을 묶는 bounded rerun으로 잡아야 한다.",
        "- 배포 후보 선정에서는 teacher equal-4 순위나 best_selection_score 단독 승자를 금지하고, subject bbox IoU/valid, route balanced accuracy, policy/action alignment, target-AR compatibility, qualitative failure taxonomy를 gate로 유지한다.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    public_summary = _track_summary(args.public_root, args.max_rows_per_split)
    uctr_summary = _track_summary(args.uctr_root, args.max_rows_per_split)
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "inputs": {
            "public_root": str(args.public_root),
            "uctr_root": str(args.uctr_root),
            "leaderboard_json": str(args.leaderboard_json),
            "teacher_leaderboard_json": str(args.teacher_leaderboard_json),
            "max_rows_per_split": args.max_rows_per_split,
        },
        "teacher_equal4": _teacher_rows(args.teacher_leaderboard_json),
        "student_profiles": _student_profile_rows(args.leaderboard_json),
        "label_stats": {
            "public": public_summary,
            "uctr": uctr_summary,
        },
        "paired_positive": _paired_summary(public_summary, uctr_summary),
    }

    json_payload = json.loads(json.dumps(payload, ensure_ascii=False))
    for track in ("public", "uctr"):
        for split in SPLITS:
            json_payload["label_stats"][track][split].pop("top_positive_index", None)

    (args.output_dir / "track_inversion_analysis.json").write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "TRACK_INVERSION_ANALYSIS_KO.md").write_text(
        _build_markdown(json_payload),
        encoding="utf-8",
    )
    print(json.dumps({"output_dir": str(args.output_dir), "summary_md": str(args.output_dir / "TRACK_INVERSION_ANALYSIS_KO.md")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
