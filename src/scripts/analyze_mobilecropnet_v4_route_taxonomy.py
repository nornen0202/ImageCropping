#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MobileCropNet v4 route taxonomy와 public/UCTR label 전이 차이를 분석한다.")
    parser.add_argument("--public_jsonl", type=Path, required=True)
    parser.add_argument("--uctr_jsonl", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--max_rows", type=int, default=0)
    return parser


def _iter_jsonl(path: Path, *, max_rows: int = 0):
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)
            count += 1
            if max_rows > 0 and count >= max_rows:
                break


def _f(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _fmt(value: Any, digits: int = 6) -> str:
    number = _f(value)
    return "-" if number is None else f"{number:.{digits}f}"


def _box_area(box: Any) -> float | None:
    if not isinstance(box, list) or len(box) != 4:
        return None
    vals = [_f(v) for v in box]
    if any(v is None for v in vals):
        return None
    x1, y1, x2, y2 = [float(v) for v in vals]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_iou(a: Any, b: Any) -> float | None:
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != 4 or len(b) != 4:
        return None
    av = [_f(v) for v in a]
    bv = [_f(v) for v in b]
    if any(v is None for v in av + bv):
        return None
    ax1, ay1, ax2, ay2 = [float(v) for v in av]
    bx1, by1, bx2, by2 = [float(v) for v in bv]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 1e-12 else None


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("image_id") or ""), str(row.get("target_ar") or "FREE")


def _route(row: dict[str, Any]) -> dict[str, Any]:
    routing = row.get("routing") if isinstance(row.get("routing"), dict) else {}
    return {
        "subject_mode": str(routing.get("subject_mode") or "unknown"),
        "policy_id": str(routing.get("policy_id") or ""),
        "router_rule_id": str(routing.get("router_rule_id") or routing.get("rule_id") or ""),
        "route_conf": _f(routing.get("route_conf")),
        "subject_prior_box": routing.get("subject_prior_bbox_norm_xyxy"),
        "flags": routing.get("flags") if isinstance(routing.get("flags"), dict) else {},
    }


def _positive(row: dict[str, Any]) -> dict[str, Any] | None:
    candidates = []
    for key in ("matching_targets", "candidate_pool"):
        values = row.get(key)
        if isinstance(values, list):
            candidates.extend([item for item in values if isinstance(item, dict)])
    positives = [item for item in candidates if bool(item.get("is_positive_candidate"))]
    if not positives:
        positives = [item for item in row.get("matching_targets", []) if isinstance(item, dict)]
    if not positives:
        return None
    return max(
        positives,
        key=lambda item: (
            _f(item.get("crop_utility_prob")) or -1.0,
            _f(item.get("score_policy")) or -1.0,
            _f(item.get("score_rank")) or -1.0,
        ),
    )


def _candidate_count(row: dict[str, Any]) -> int:
    count = 0
    for key in ("matching_targets", "candidate_pool", "ignored_candidates", "overflow_candidates"):
        values = row.get(key)
        if isinstance(values, list):
            count += len(values)
    return count


def _unsafe_count(row: dict[str, Any]) -> int:
    count = 0
    for key in ("matching_targets", "candidate_pool", "ignored_candidates", "overflow_candidates"):
        values = row.get(key)
        if not isinstance(values, list):
            continue
        for item in values:
            if isinstance(item, dict) and bool(item.get("is_unsafe_negative")):
                count += 1
    return count


def _bucket_summary(path: Path, label: str, *, max_rows: int = 0) -> dict[str, Any]:
    route_counts: Counter[str] = Counter()
    target_ar_counts: Counter[str] = Counter()
    mode_ar_counts: dict[str, Counter[str]] = defaultdict(Counter)
    policy_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    route_conf_by_mode: dict[str, list[float]] = defaultdict(list)
    subject_area_by_mode: dict[str, list[float]] = defaultdict(list)
    positive_area_by_mode: dict[str, list[float]] = defaultdict(list)
    positive_policy_by_mode: dict[str, list[float]] = defaultdict(list)
    positive_utility_by_mode: dict[str, list[float]] = defaultdict(list)
    candidate_count_by_mode: dict[str, list[float]] = defaultdict(list)
    unsafe_count_by_mode: dict[str, list[float]] = defaultdict(list)
    subject_valid_count = 0
    positive_missing = 0
    row_count = 0
    paired_positive: dict[tuple[str, str], dict[str, Any]] = {}

    for row in _iter_jsonl(path, max_rows=max_rows):
        row_count += 1
        route = _route(row)
        mode = route["subject_mode"]
        ar = str(row.get("target_ar") or "FREE")
        route_counts[mode] += 1
        target_ar_counts[ar] += 1
        mode_ar_counts[mode][ar] += 1
        if route["policy_id"]:
            policy_counts[route["policy_id"]] += 1
        if route["router_rule_id"]:
            rule_counts[route["router_rule_id"]] += 1
        decision = row.get("decision_target") if isinstance(row.get("decision_target"), dict) else {}
        decision_counts[str(decision.get("decision_type", decision.get("decision_id", "unknown")))] += 1
        if route["route_conf"] is not None:
            route_conf_by_mode[mode].append(float(route["route_conf"]))
        subject_area = _box_area(route.get("subject_prior_box"))
        if subject_area is not None:
            subject_valid_count += 1
            subject_area_by_mode[mode].append(subject_area)
        positive = _positive(row)
        if positive is None:
            positive_missing += 1
        else:
            pos_box = positive.get("bbox_norm_xyxy")
            pos_area = _box_area(pos_box)
            if pos_area is not None:
                positive_area_by_mode[mode].append(pos_area)
            policy = _f(positive.get("score_policy"))
            utility = _f(positive.get("crop_utility_prob"))
            if policy is not None:
                positive_policy_by_mode[mode].append(policy)
            if utility is not None:
                positive_utility_by_mode[mode].append(utility)
            paired_positive[_row_key(row)] = {
                "subject_mode": mode,
                "target_ar": ar,
                "bbox": pos_box,
                "area": pos_area,
                "score_policy": policy,
                "crop_utility_prob": utility,
            }
        candidate_count_by_mode[mode].append(float(_candidate_count(row)))
        unsafe_count_by_mode[mode].append(float(_unsafe_count(row)))

    by_mode = {}
    for mode in sorted(route_counts):
        by_mode[mode] = {
            "rows": route_counts[mode],
            "route_conf_mean": _mean(route_conf_by_mode.get(mode, [])),
            "subject_area_mean": _mean(subject_area_by_mode.get(mode, [])),
            "positive_area_mean": _mean(positive_area_by_mode.get(mode, [])),
            "positive_score_policy_mean": _mean(positive_policy_by_mode.get(mode, [])),
            "positive_crop_utility_mean": _mean(positive_utility_by_mode.get(mode, [])),
            "candidate_count_mean": _mean(candidate_count_by_mode.get(mode, [])),
            "unsafe_count_mean": _mean(unsafe_count_by_mode.get(mode, [])),
            "target_ar_counts": dict(mode_ar_counts[mode]),
        }

    return {
        "label": label,
        "rows": row_count,
        "subject_valid_rate": subject_valid_count / max(1, row_count),
        "positive_missing": positive_missing,
        "route_counts": dict(route_counts),
        "target_ar_counts": dict(target_ar_counts),
        "policy_counts": dict(policy_counts.most_common(20)),
        "router_rule_counts": dict(rule_counts.most_common(20)),
        "decision_counts": dict(decision_counts),
        "by_mode": by_mode,
        "paired_positive": paired_positive,
    }


def _paired_comparison(public: dict[str, Any], uctr: dict[str, Any]) -> dict[str, Any]:
    pub = public["paired_positive"]
    uct = uctr["paired_positive"]
    keys = sorted(set(pub) & set(uct))
    by_mode: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    overall: dict[str, list[float]] = defaultdict(list)
    for key in keys:
        a = pub[key]
        b = uct[key]
        mode = str(a.get("subject_mode") or b.get("subject_mode") or "unknown")
        iou = _box_iou(a.get("bbox"), b.get("bbox"))
        area_delta = None
        if a.get("area") is not None and b.get("area") is not None:
            area_delta = float(b["area"]) - float(a["area"])
        policy_delta = None
        if a.get("score_policy") is not None and b.get("score_policy") is not None:
            policy_delta = float(b["score_policy"]) - float(a["score_policy"])
        utility_delta = None
        if a.get("crop_utility_prob") is not None and b.get("crop_utility_prob") is not None:
            utility_delta = float(b["crop_utility_prob"]) - float(a["crop_utility_prob"])
        for name, value in (("bbox_iou", iou), ("area_delta_uctr_minus_public", area_delta), ("score_policy_delta", policy_delta), ("crop_utility_delta", utility_delta)):
            if value is not None:
                overall[name].append(float(value))
                by_mode[mode][name].append(float(value))
    return {
        "paired_rows": len(keys),
        "overall": {key: _mean(values) for key, values in overall.items()},
        "by_mode": {
            mode: {"rows": len(next(iter(values.values()))) if values else 0, **{key: _mean(vals) for key, vals in values.items()}}
            for mode, values in sorted(by_mode.items())
        },
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    public = payload["public"]
    uctr = payload["uctr"]
    paired = payload["paired"]
    lines = [
        "# MobileCropNet v4 경로 분류체계 및 라벨 전이 분석",
        "",
        f"- 생성 시각: `{payload['generated_at_utc']}`",
        f"- public 라벨 JSONL: `{payload['public_jsonl']}`",
        f"- UCTR JSONL: `{payload['uctr_jsonl']}`",
        "",
        "## 1. 전체 요약",
        "",
        "| 항목 | public | UCTR |",
        "| --- | ---: | ---: |",
        f"| rows | {public['rows']} | {uctr['rows']} |",
        f"| subject valid rate | {_fmt(public['subject_valid_rate'])} | {_fmt(uctr['subject_valid_rate'])} |",
        f"| positive missing | {public['positive_missing']} | {uctr['positive_missing']} |",
        f"| paired positive rows | {paired['paired_rows']} | {paired['paired_rows']} |",
        "",
        "## 2. 주체 모드 분포",
        "",
        "| 주체 모드 | public rows | UCTR rows | public positive 면적 | UCTR positive 면적 | public utility | UCTR utility |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    modes = sorted(set(public["by_mode"]) | set(uctr["by_mode"]))
    for mode in modes:
        p = public["by_mode"].get(mode, {})
        u = uctr["by_mode"].get(mode, {})
        lines.append(
            f"| `{mode}` | {p.get('rows', 0)} | {u.get('rows', 0)} | "
            f"{_fmt(p.get('positive_area_mean'))} | {_fmt(u.get('positive_area_mean'))} | "
            f"{_fmt(p.get('positive_crop_utility_mean'))} | {_fmt(u.get('positive_crop_utility_mean'))} |"
        )
    lines.extend(
        [
            "",
        "## 3. 동일 이미지/Target-AR Positive 비교",
            "",
            "| mode | paired | bbox IoU | UCTR-public area | UCTR-public score_policy | UCTR-public utility |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode, row in paired.get("by_mode", {}).items():
        lines.append(
            f"| `{mode}` | {row.get('rows', 0)} | {_fmt(row.get('bbox_iou'))} | "
            f"{_fmt(row.get('area_delta_uctr_minus_public'))} | {_fmt(row.get('score_policy_delta'))} | {_fmt(row.get('crop_utility_delta'))} |"
        )
    lines.extend(
        [
            "",
            "## 4. 현재 판단",
            "",
            "- public과 UCTR의 route taxonomy 분포는 동일 routing metadata를 공유하므로 큰 차이가 나지 않는 것이 정상이다.",
            "- 학생 성능 역전의 핵심은 route class 비율 자체보다 positive crop geometry와 score target의 전이 난이도에 있다.",
            "- UCTR positive가 같은 이미지/target-AR에서 더 큰 crop을 선택하고 public score/utility와 덜 맞는 경우, crop quality head는 public deployment benchmark에서 손해를 본다.",
            "- 다음 retrain은 public crop target을 유지하면서 subject-box, route-balanced sampling, generated proposal/action alignment를 보강하는 현재 v3 방향을 우선한다.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    public = _bucket_summary(args.public_jsonl, "public", max_rows=int(args.max_rows))
    uctr = _bucket_summary(args.uctr_jsonl, "uctr", max_rows=int(args.max_rows))
    paired = _paired_comparison(public, uctr)
    public_out = dict(public)
    uctr_out = dict(uctr)
    public_out.pop("paired_positive", None)
    uctr_out.pop("paired_positive", None)
    payload = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "public_jsonl": str(args.public_jsonl),
        "uctr_jsonl": str(args.uctr_jsonl),
        "max_rows": int(args.max_rows),
        "public": public_out,
        "uctr": uctr_out,
        "paired": paired,
    }
    (args.output_dir / "route_taxonomy_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(args.output_dir / "ROUTE_TAXONOMY_ANALYSIS_KO.md", payload)
    print(json.dumps({"state": "completed", "output_dir": str(args.output_dir), "paired_rows": paired["paired_rows"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
