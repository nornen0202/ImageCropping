#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Iterable


ARS = ("FREE", "16:9", "1:1", "3:4", "4:3", "9:16")
OBJECT_MODES = {"object_single_center", "object_single_rot", "object_multi_center", "object_multi_rot"}
PERSON_MODES = {"single_person_center", "single_person_rot", "group_center", "group_rot", "face"}
HUMAN_PERSON_MODES = {"single_person_center", "single_person_rot"}


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k)) for k in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _reason_counter(rows: Iterable[dict[str, Any]]) -> Counter[str]:
    out: Counter[str] = Counter()
    for row in rows:
        reason = str(row.get("no_positive_reason") or "").strip()
        if not reason:
            continue
        for part in reason.split(","):
            part = part.strip()
            if part:
                out[part] += 1
    return out


def _reason_text(counter: Counter[str], n: int = 8) -> str:
    if not counter:
        return "-"
    return ", ".join(f"{k}:{v}" for k, v in counter.most_common(n))


def _group_stats(values: list[int | float], total_images: int) -> dict[str, Any]:
    all_values = list(values)
    positive_values = [float(v) for v in all_values if float(v) > 0]
    return {
        "per_all_image_mean": round(mean(all_values), 6) if all_values else 0.0,
        "per_all_image_std": round(pstdev(all_values), 6) if len(all_values) > 1 else 0.0,
        "positive_image_count": len(positive_values),
        "positive_image_rate": round(len(positive_values) / float(max(1, total_images)), 6),
        "per_positive_image_mean": round(mean(positive_values), 6) if positive_values else 0.0,
        "per_positive_image_std": round(pstdev(positive_values), 6) if len(positive_values) > 1 else 0.0,
        "per_positive_image_min": min(positive_values) if positive_values else 0,
        "per_positive_image_max": max(positive_values) if positive_values else 0,
    }


def build_annotation_indexes(label_json: dict[str, Any]) -> dict[str, Any]:
    images = {int(im["id"]): im for im in label_json["images"]}
    source_to_image_id = {str(im["source_image_id"]): int(im["id"]) for im in label_json["images"]}
    category_by_id = {int(cat["id"]): str(cat["name"]) for cat in label_json.get("categories", [])}
    ann_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    ann_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for ann in label_json["annotations"]:
        image_id = int(ann["image_id"])
        ann_by_image[image_id].append(ann)
        ann_by_source[str(images[image_id]["source_image_id"])].append(ann)
    return {
        "images": images,
        "source_to_image_id": source_to_image_id,
        "category_by_id": category_by_id,
        "ann_by_image": ann_by_image,
        "ann_by_source": ann_by_source,
    }


def summarize_mode_ar(label_json: dict[str, Any], status_path: Path, out_dir: Path, *, prefix: str) -> dict[str, Any]:
    images = label_json["images"]
    total_images = len(images)
    category_by_id = {int(cat["id"]): str(cat["name"]) for cat in label_json.get("categories", [])}
    query_counts: Counter[tuple[str, str]] = Counter()
    query_positive: Counter[tuple[str, str]] = Counter()
    query_no_positive: Counter[tuple[str, str]] = Counter()
    reason_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    face_image_ar_positive: dict[str, set[str]] = defaultdict(set)
    face_image_ar_query: dict[str, set[str]] = defaultdict(set)
    face_no_positive_rows: list[dict[str, Any]] = []
    for row in _iter_jsonl(status_path):
        mode = str(row.get("mode_name") or "")
        ar = str(row.get("target_ar") or "")
        key = (mode, ar)
        query_counts[key] += 1
        if int(row.get("positive_exists") or 0) == 1:
            query_positive[key] += 1
            if mode == "face":
                face_image_ar_positive[str(row.get("source_image_id"))].add(ar)
        else:
            query_no_positive[key] += 1
            reason = str(row.get("no_positive_reason") or "").strip()
            for part in reason.split(","):
                part = part.strip()
                if part:
                    reason_counts[key][part] += 1
            if mode == "face":
                slim = {
                    "source_image_id": row.get("source_image_id"),
                    "target_ar": ar,
                    "query_id": row.get("query_id"),
                    "best_score": row.get("best_score"),
                    "candidate_count": row.get("candidate_count"),
                    "no_positive_reason": reason,
                    "route_mode": row.get("route_mode"),
                    "entity_id": row.get("entity_id"),
                }
                face_no_positive_rows.append(slim)
        if mode == "face":
            face_image_ar_query[str(row.get("source_image_id"))].add(ar)

    ann_counts: Counter[tuple[str, str]] = Counter()
    ann_pos: Counter[tuple[str, str]] = Counter()
    ann_neg: Counter[tuple[str, str]] = Counter()
    per_image_pos_counts: dict[tuple[str, str], Counter[int]] = defaultdict(Counter)
    per_image_all_pos = Counter()
    per_mode_image_pos: dict[str, Counter[int]] = defaultdict(Counter)
    per_ar_image_pos: dict[str, Counter[int]] = defaultdict(Counter)
    for ann in label_json["annotations"]:
        mode = str(ann.get("mode_name") or category_by_id.get(int(ann.get("category_id", -1)), ""))
        ar = str((ann.get("attributes") or {}).get("target_ar") or "")
        key = (mode, ar)
        image_id = int(ann["image_id"])
        ann_counts[key] += 1
        if int(ann.get("gt_flag") or 0) == 1:
            ann_pos[key] += 1
            per_image_pos_counts[key][image_id] += 1
            per_image_all_pos[image_id] += 1
            per_mode_image_pos[mode][image_id] += 1
            per_ar_image_pos[ar][image_id] += 1
        else:
            ann_neg[key] += 1

    all_keys = sorted(set(query_counts) | set(ann_counts))
    mode_ar_rows: list[dict[str, Any]] = []
    image_mode_ar_rows: list[dict[str, Any]] = []
    for mode, ar in all_keys:
        key = (mode, ar)
        counts_by_image = per_image_pos_counts[key]
        values = [counts_by_image.get(int(im["id"]), 0) for im in images]
        stats = _group_stats(values, total_images)
        row = {
            "mode_name": mode,
            "target_ar": ar,
            "query_count": query_counts[key],
            "positive_query_count": query_positive[key],
            "no_positive_query_count": query_no_positive[key],
            "positive_query_rate": round(query_positive[key] / float(max(1, query_counts[key])), 6),
            "annotation_count": ann_counts[key],
            "positive_annotation_count": ann_pos[key],
            "negative_annotation_count": ann_neg[key],
            "top_no_positive_reasons": _reason_text(reason_counts[key]),
        }
        row.update(stats)
        mode_ar_rows.append(row)
        image_mode_ar_rows.append(
            {
                "mode_name": mode,
                "target_ar": ar,
                **stats,
            }
        )

    mode_rows: list[dict[str, Any]] = []
    for mode, counts in sorted(per_mode_image_pos.items()):
        values = [counts.get(int(im["id"]), 0) for im in images]
        mode_rows.append({"mode_name": mode, **_group_stats(values, total_images)})
    ar_rows: list[dict[str, Any]] = []
    for ar, counts in sorted(per_ar_image_pos.items()):
        values = [counts.get(int(im["id"]), 0) for im in images]
        ar_rows.append({"target_ar": ar, **_group_stats(values, total_images)})
    all_values = [per_image_all_pos.get(int(im["id"]), 0) for im in images]
    all_image_stats = _group_stats(all_values, total_images)

    missing_face_rows: list[dict[str, Any]] = []
    for src, queried in sorted(face_image_ar_query.items()):
        positives = face_image_ar_positive.get(src, set())
        missing = [ar for ar in ARS if ar in queried and ar not in positives]
        if missing:
            missing_face_rows.append(
                {
                    "source_image_id": src,
                    "queried_ars": sorted(queried),
                    "positive_ars": sorted(positives),
                    "missing_positive_ars": missing,
                    "missing_count": len(missing),
                }
            )
    face_ar_missing_counts = Counter()
    for row in missing_face_rows:
        for ar in row["missing_positive_ars"]:
            face_ar_missing_counts[ar] += 1

    _write_csv(out_dir / f"{prefix}_mode_target_ar_enhanced_stats.csv", mode_ar_rows)
    _write_csv(out_dir / f"{prefix}_mode_image_positive_stats.csv", mode_rows)
    _write_csv(out_dir / f"{prefix}_target_ar_image_positive_stats.csv", ar_rows)
    _write_csv(out_dir / f"{prefix}_face_missing_positive_by_image.csv", missing_face_rows)
    _write_csv(out_dir / f"{prefix}_face_no_positive_rows.csv", face_no_positive_rows)
    summary = {
        "total_images": total_images,
        "total_annotations": len(label_json["annotations"]),
        "all_positive_annotations_per_image": all_image_stats,
        "face_ar_missing_image_counts": dict(face_ar_missing_counts),
        "face_ar_positive_query_rates": {
            ar: round(query_positive[("face", ar)] / float(max(1, query_counts[("face", ar)])), 6)
            for ar in ARS
            if ("face", ar) in query_counts
        },
        "face_top_no_positive_reasons_by_ar": {
            ar: dict(reason_counts[("face", ar)].most_common(12))
            for ar in ARS
            if ("face", ar) in query_counts
        },
        "outputs": {
            "mode_target_ar_enhanced_stats": str(out_dir / f"{prefix}_mode_target_ar_enhanced_stats.csv"),
            "mode_image_positive_stats": str(out_dir / f"{prefix}_mode_image_positive_stats.csv"),
            "target_ar_image_positive_stats": str(out_dir / f"{prefix}_target_ar_image_positive_stats.csv"),
            "face_missing_positive_by_image": str(out_dir / f"{prefix}_face_missing_positive_by_image.csv"),
            "face_no_positive_rows": str(out_dir / f"{prefix}_face_no_positive_rows.csv"),
        },
    }
    with (out_dir / f"{prefix}_stats_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def collect_status_rows(status_path: Path, source_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _iter_jsonl(status_path):
        src = str(row.get("source_image_id"))
        if src in source_ids:
            out[src].append(row)
    return out


def collect_ann_rows(label_json: dict[str, Any], source_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    idx = build_annotation_indexes(label_json)
    out: dict[str, list[dict[str, Any]]] = {}
    for src in source_ids:
        out[src] = list(idx["ann_by_source"].get(src, []))
    return out


def _flatten_status_row(row: dict[str, Any]) -> dict[str, Any]:
    attrs = row.get("attributes") or {}
    portrait = attrs.get("portrait_comp") or {}
    return {
        "source_image_id": row.get("source_image_id"),
        "target_ar": row.get("target_ar"),
        "mode_name": row.get("mode_name"),
        "entity_id": row.get("entity_id"),
        "decision": row.get("decision"),
        "positive_exists": row.get("positive_exists"),
        "positive_score": row.get("positive_score"),
        "best_score": row.get("best_score"),
        "candidate_count": row.get("candidate_count"),
        "negative_count": row.get("negative_count"),
        "no_positive_reason": row.get("no_positive_reason"),
        "route_mode": row.get("route_mode"),
        "person_rank": attrs.get("person_rank"),
        "face_query_valid": attrs.get("face_query_valid"),
        "direction_hint": portrait.get("direction_hint"),
        "shot_type": portrait.get("shot_type"),
    }


def _flatten_ann_row(ann: dict[str, Any]) -> dict[str, Any]:
    attrs = ann.get("attributes") or {}
    comps = attrs.get("score_components") or {}
    labels = attrs.get("checklist_labels") or {}
    subj = attrs.get("subject_debug") or {}
    return {
        "annotation_id": ann.get("id"),
        "image_id": ann.get("image_id"),
        "target_ar": attrs.get("target_ar"),
        "mode_name": ann.get("mode_name"),
        "gt_flag": ann.get("gt_flag"),
        "is_best": ann.get("is_best"),
        "bbox": ann.get("bbox"),
        "score_mode": ann.get("score_mode"),
        "candidate_id": attrs.get("candidate_id"),
        "candidate_source": attrs.get("candidate_source"),
        "hard_reject_reasons": ann.get("hard_reject_reasons") or attrs.get("hard_reject_reasons"),
        "negative_reason": attrs.get("negative_reason"),
        "q_place": comps.get("q_place"),
        "q_subj": comps.get("q_subj"),
        "q_scale": comps.get("q_scale"),
        "q_head": comps.get("q_head"),
        "q_look": comps.get("q_look"),
        "anchor_rx": comps.get("anchor_rx"),
        "anchor_ry": comps.get("anchor_ry"),
        "center_dist": (attrs.get("checklist_scores") or {}).get("center_dist"),
        "center_label": labels.get("center_dist"),
        "headroom": labels.get("headroom"),
        "lookroom": labels.get("lookroom"),
        "face_recall": comps.get("face_recall"),
        "face_body_leakage": comps.get("face_body_leakage"),
        "portrait_control_center_deviation": comps.get("portrait_control_center_deviation"),
        "core_bbox_norm_xyxy": subj.get("core_bbox_norm_xyxy"),
        "face_bbox_norm_xyxy": subj.get("face_bbox_norm_xyxy"),
    }


def write_specific_sample_audits(
    *,
    phase_label: dict[str, Any],
    phase_status: Path,
    full_label: dict[str, Any],
    full_status: Path,
    out_dir: Path,
) -> dict[str, Any]:
    phase_sources = {"bigstock_image_41702452"}
    full_sources = {"sstk_image_1197510367", "pond5_image_241706455"}
    phase_status_rows = collect_status_rows(phase_status, phase_sources)
    full_status_rows = collect_status_rows(full_status, full_sources)
    phase_ann_rows = collect_ann_rows(phase_label, phase_sources)
    full_ann_rows = collect_ann_rows(full_label, full_sources)

    for src, rows in phase_status_rows.items():
        _write_csv(out_dir / f"sample_{src}_phase_status.csv", [_flatten_status_row(r) for r in rows])
    for src, rows in phase_ann_rows.items():
        _write_csv(out_dir / f"sample_{src}_phase_annotations.csv", [_flatten_ann_row(a) for a in rows])
    for src, rows in full_status_rows.items():
        _write_csv(out_dir / f"sample_{src}_full_status.csv", [_flatten_status_row(r) for r in rows])
    for src, rows in full_ann_rows.items():
        _write_csv(out_dir / f"sample_{src}_full_annotations.csv", [_flatten_ann_row(a) for a in rows])

    person_rows = [r for r in full_status_rows.get("sstk_image_1197510367", []) if str(r.get("mode_name")) in PERSON_MODES]
    person_no_pos = [r for r in person_rows if int(r.get("positive_exists") or 0) == 0]
    pond_face_anns = [
        _flatten_ann_row(a)
        for a in full_ann_rows.get("pond5_image_241706455", [])
        if str(a.get("mode_name")) == "face"
    ]
    pond_face_1_3 = [r for r in pond_face_anns if str(r.get("target_ar")) in {"1:1", "3:4"}]
    sample_summary = {
        "bigstock_image_41702452_phase_face_status": [
            _flatten_status_row(r)
            for r in phase_status_rows.get("bigstock_image_41702452", [])
            if str(r.get("mode_name")) == "face"
        ],
        "sstk_image_1197510367_person_no_positive_reason_counts": dict(_reason_counter(person_no_pos)),
        "sstk_image_1197510367_person_status_count": len(person_rows),
        "pond5_image_241706455_face_1_1_3_4_annotations": pond_face_1_3,
    }
    with (out_dir / "specific_sample_audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(sample_summary, f, ensure_ascii=False, indent=2)
    return sample_summary


def make_no_object_variant(run_dir: Path, out_dir: Path) -> dict[str, Any]:
    src_label_dir = run_dir / "label_json"
    dst_label_dir = out_dir / "label_json_no_object_modes"
    dst_label_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"variant_policy": "Remove annotations whose mode_name is object_*; keep non-object categories reindexed to contiguous ids."}
    for stem in [
        "multimode_labels_full",
        "multimode_labels_train",
        "multimode_labels_val",
        "multimode_labels_target_ar_only_full",
        "multimode_labels_target_ar_only_train",
        "multimode_labels_target_ar_only_val",
    ]:
        src = src_label_dir / f"{stem}.json"
        if not src.exists():
            continue
        data = _read_json(src)
        categories = data.get("categories", [])
        kept_categories = [cat for cat in categories if str(cat.get("name")) not in OBJECT_MODES]
        old_to_new = {int(cat["id"]): idx for idx, cat in enumerate(kept_categories)}
        new_categories = [
            {"id": old_to_new[int(cat["id"])], "name": str(cat["name"]), "supercategory": cat.get("supercategory", "crop")}
            for cat in kept_categories
        ]
        old_cat_name = {int(cat["id"]): str(cat["name"]) for cat in categories}
        new_annotations = []
        removed = 0
        for ann in data.get("annotations", []):
            mode = str(ann.get("mode_name") or old_cat_name.get(int(ann.get("category_id", -1)), ""))
            if mode in OBJECT_MODES:
                removed += 1
                continue
            out_ann = dict(ann)
            if int(out_ann.get("category_id", -1)) in old_to_new:
                out_ann["category_id"] = old_to_new[int(out_ann["category_id"])]
            new_annotations.append(out_ann)
        out = dict(data)
        out["categories"] = new_categories
        out["annotations"] = new_annotations
        dst = dst_label_dir / f"{stem}.json"
        with dst.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
        summary[stem] = {
            "images": len(out.get("images", [])),
            "annotations_before": len(data.get("annotations", [])),
            "annotations_after": len(new_annotations),
            "removed_annotations": removed,
            "categories_after": new_categories,
            "path": str(dst),
        }
    with (dst_label_dir / "no_object_modes_variant_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def write_report(
    *,
    out_dir: Path,
    phase_summary: dict[str, Any],
    full_summary: dict[str, Any],
    sample_summary: dict[str, Any],
    no_object_summary: dict[str, Any],
) -> None:
    face_rates = phase_summary["face_ar_positive_query_rates"]
    face_reasons = phase_summary["face_top_no_positive_reasons_by_ar"]
    specific_face = sample_summary["bigstock_image_41702452_phase_face_status"]
    sstk_reasons = sample_summary["sstk_image_1197510367_person_no_positive_reason_counts"]
    pond_face = sample_summary["pond5_image_241706455_face_1_1_3_4_annotations"]
    object_full = no_object_summary.get("multimode_labels_full", {})
    lines: list[str] = []
    lines.append("# review_jy_260604 후속 분석 및 작업 결과")
    lines.append("")
    lines.append("이 문서는 `review_jy_260604.md`의 항목을 기준으로 PhaseA v14와 Full v14 산출물을 재분석하고, 생성 가능한 보완 산출물을 실제로 만든 결과다.")
    lines.append("")
    lines.append("## 1. 수행 요약")
    lines.append("")
    lines.append("- PhaseA v14 전체 `10,000` image / `167,387` annotation과 Full v14 전체 label/status를 전수 집계했다.")
    lines.append("- face mode의 target AR별 positive 누락 원인을 `mode_query_status.jsonl` 기준으로 집계했다.")
    lines.append("- `bigstock_image_41702452`, `sstk_image_1197510367`, `pond5_image_241706455`를 별도 CSV/JSON으로 분해했다.")
    lines.append("- `SSTK_MultiMode_Headroom_Lookroom_Explainability_Guide_KO_2026-06-04.md`의 gaze anchor 문제는 같은 이미지 기반 부록 asset으로 보완했고, 기존 문제 섹션에서 참조할 수 있게 정리했다.")
    lines.append("- object mode annotation을 제거한 별도 label JSON variant를 생성했다.")
    lines.append("")
    lines.append("## 2. Face mode AR 누락 전수 분석")
    lines.append("")
    lines.append("| target AR | positive query rate | top no-positive reasons |")
    lines.append("|---|---:|---|")
    for ar in ARS:
        if ar in face_rates:
            reasons = ", ".join(f"{k}:{v}" for k, v in list(face_reasons.get(ar, {}).items())[:6])
            lines.append(f"| `{ar}` | {face_rates[ar]:.6f} | {reasons or '-'} |")
    lines.append("")
    lines.append("결론: 3:4와 9:16이 시각화에서 자주 빠지는 것은 visualizer 문제가 아니라 label factory의 positive query rate가 낮기 때문이다. 특히 PhaseA v14 face positive query rate는 3:4 `0.252979`, 9:16 `0.074756`으로 낮다. 주요 원인은 `body_leakage_strict`, `face_center_x_miss_strict`, `face_too_loose`, `head_subject_coverage`, `head_top_cut` 계열이다. 세로 AR은 face bbox를 보존하면서 body leakage와 center strict gate를 동시에 만족하기 어렵고, 9:16은 후보 crop 폭/위치 제약 때문에 더 자주 탈락한다.")
    lines.append("")
    lines.append("`review_bigstock_image_41702452`의 face status는 아래 파일에 저장했다.")
    lines.append("")
    lines.append("- `sample_bigstock_image_41702452_phase_status.csv`")
    lines.append("- `sample_bigstock_image_41702452_phase_annotations.csv`")
    lines.append("")
    if specific_face:
        lines.append("해당 이미지의 face query 요약:")
        lines.append("")
        lines.append("| AR | decision | positive | best score | reason |")
        lines.append("|---|---|---:|---:|---|")
        for row in specific_face:
            lines.append(f"| `{row.get('target_ar')}` | `{row.get('decision')}` | {row.get('positive_exists')} | {row.get('best_score')} | `{row.get('no_positive_reason') or ''}` |")
        lines.append("")
    lines.append("## 3. 종횡비별 누락/탈락 분석")
    lines.append("")
    lines.append("전체 mode/AR별 query, annotation, image-level 평균/표준편차는 아래 파일로 저장했다.")
    lines.append("")
    lines.append("- `phaseA_mode_target_ar_enhanced_stats.csv`")
    lines.append("- `phaseA_mode_image_positive_stats.csv`")
    lines.append("- `phaseA_target_ar_image_positive_stats.csv`")
    lines.append("- `phaseA_face_missing_positive_by_image.csv`")
    lines.append("- `full_mode_target_ar_enhanced_stats.csv`")
    lines.append("")
    lines.append(f"PhaseA 전체 positive annotation의 image당 평균은 `{phase_summary['all_positive_annotations_per_image']['per_all_image_mean']}`개, 표준편차는 `{phase_summary['all_positive_annotations_per_image']['per_all_image_std']}`다. Full v14는 평균 `{full_summary['all_positive_annotations_per_image']['per_all_image_mean']}`개, 표준편차 `{full_summary['all_positive_annotations_per_image']['per_all_image_std']}`다.")
    lines.append("")
    lines.append("## 4. `sstk_image_1197510367` person crop 누락")
    lines.append("")
    lines.append("Full v14 status를 확인한 결과 person/face 계열 query는 존재하지만 일부 query가 positive로 승격되지 않았다. no-positive reason counter는 아래와 같다.")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(sstk_reasons, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("상세 row는 `sample_sstk_image_1197510367_full_status.csv`, annotation row는 `sample_sstk_image_1197510367_full_annotations.csv`에 저장했다. 위 counter의 `body_leakage/body_leakage_strict`는 face 9:16 query에서 온 값이고, single-person center/rot query 12개는 모두 `support_grounding_miss`와 `center_support_grounding_miss` 또는 `rot_support_grounding_miss` 조합으로 탈락했다. 따라서 현재 원인은 candidate 미생성이 아니라 v14 strict intent/safety gate, 특히 full-shot support grounding 기준으로 positive 승격이 막힌 것으로 보는 것이 맞다. v3처럼 생성될 법해 보이는 이유는 v3가 이러한 hard gate를 덜 엄격하게 적용했기 때문이다.")
    lines.append("")
    lines.append("보완안: upper-body/half-body portrait는 full-body support grounding을 요구하면 과도하게 탈락할 수 있으므로, `portrait_shot_type in {headshot, bust, half, upper_body}`일 때 support grounding gate를 완화하고, 대신 face/torso coverage와 headroom/lookroom/joint cut 중심으로 검증하는 shot-type policy를 추가한다.")
    lines.append("")
    lines.append("## 5. `pond5_image_241706455` face 1:1/3:4 center 의도 분석")
    lines.append("")
    lines.append("Face mode는 center family에 속하며, 코드상 `face_center_placement_miss`, `face_center_x_miss`, strict variant가 적용된다. 해당 이미지의 face 1:1/3:4 annotation은 아래와 같다.")
    lines.append("")
    lines.append("| ann | AR | gt | score | anchor_rx | q_place | center label | face leakage | candidate |")
    lines.append("|---:|---|---:|---:|---:|---:|---|---:|---|")
    for row in pond_face:
        lines.append(
            f"| {row.get('annotation_id')} | `{row.get('target_ar')}` | {row.get('gt_flag')} | {row.get('score_mode')} | {row.get('anchor_rx')} | {row.get('q_place')} | `{row.get('center_label')}` | {row.get('face_body_leakage')} | `{row.get('candidate_source')}` |"
        )
    lines.append("")
    lines.append("결론: 이 샘플은 candidate 미생성보다는 center score와 body leakage gate 사이의 trade-off가 드러난 사례다. 1:1/3:4 모두 `center_comp_strong` 후보가 존재하지만 해당 후보는 `face_body_leakage`가 더 높아 negative로 밀렸고, 최종 positive는 leakage가 더 낮은 대신 `anchor_rx`가 0.62~0.65 수준인 `center_comp_weak` 후보가 선택됐다. 따라서 rot처럼 보이는 face crop을 줄이려면 단순히 candidate를 늘리는 것보다 face mode positive에 대한 최소 center 조건을 강화하되, head/shoulder face crop에서 허용 가능한 body leakage 상한을 재보정해야 한다. 보완안은 `center_strength`/`anchor_rx` tie-breaker, center-strong 후보가 relaxed leakage cap 안에 있으면 우선 선택하는 정책, face mode visual review의 center line overlay 추가다.")
    lines.append("")
    lines.append("## 6. Headroom/Lookroom guide gaze anchor 보완")
    lines.append("")
    lines.append("기존 guide의 일부 예시는 group aggregate `portrait_comp.eye_point_norm_xy`가 실제 얼굴이 아닌 그룹 중심/중간점처럼 보이는 문제가 있었다. 보완으로 `sstk_image_789389968` 하나를 기준으로 실제 positive, 실제 negative, synthetic threshold crop을 비교하는 부록 A를 추가했다.")
    lines.append("")
    lines.append("- 문서: `Implement_Docs/SSTK_MultiMode_Headroom_Lookroom_Explainability_Guide_KO_2026-06-04.md`")
    lines.append("- same-image contact sheet: `Implement_Docs/assets_phasea_v14_headroom_lookroom_guide/phasea_v14_headlook_02_same_image_variant_contact_sheet.jpg`")
    lines.append("")
    lines.append("## 7. Group headroom 필요성")
    lines.append("")
    lines.append("사진학 관점에서 group portrait도 headroom이 필요하다. 다만 단일 인물처럼 한 얼굴 기준이 아니라 group envelope의 상단, 주요 얼굴선, shared breathing room을 기준으로 약하게 반영해야 한다. 현재 구현도 group mode에서 `member_face_cut`, `member_joint_cut`, `q_head` utility 가중치, bottom/side margin gate를 사용하므로 headroom을 완전히 무시하지 않는다. 보완 방향은 group headroom을 hard binary가 아니라 group face row envelope 기반 score로 유지하고, extreme top cut만 hard gate로 두는 것이다.")
    lines.append("")
    lines.append("참고 자료: Icon Photography School의 headroom/lead room 설명, Adobe Research의 composition/content preservation crop 모델, subject-aware cropping 연구는 모두 crop quality가 subject preservation과 composition을 함께 봐야 한다는 점을 뒷받침한다.")
    lines.append("")
    lines.append("## 8. Single person crop 내 다른 사람 처리")
    lines.append("")
    lines.append("현재 single person mode에는 이미 다음 hard reject/penalty가 있다.")
    lines.append("")
    lines.append("- `secondary_person_without_exception`, `secondary_person_crop_dominance_miss`, `secondary_person_core_recall_miss`, `secondary_person_intrusion`, `secondary_person_too_loose`")
    lines.append("- `joint_cutoff`, `other_person_intrusion`")
    lines.append("- utility penalty: `-0.25 * intrusion_ratio`, `-0.18 * joint_cut_score`")
    lines.append("")
    lines.append("따라서 다른 사람이 crop 안에 들어오는 경우 완전히 무시되는 구조는 아니다. 다만 visible secondary의 joint cut을 별도 person instance별로 세분화하지는 않으므로, 후속으로 `secondary_joint_cut_score`, `secondary_face_cut_score`, `secondary_partial_person_count`를 debug component와 why-tag로 추가하는 것이 좋다.")
    lines.append("")
    lines.append("## 9. Object 제거 label JSON variant")
    lines.append("")
    lines.append(f"object mode 제거 variant를 생성했다. PhaseA full split 기준 annotation은 `{object_full.get('annotations_before')}`개에서 `{object_full.get('annotations_after')}`개로 줄었고, 제거된 annotation은 `{object_full.get('removed_annotations')}`개다.")
    lines.append("")
    lines.append("- variant dir: `label_json_no_object_modes/`")
    lines.append("- summary: `label_json_no_object_modes/no_object_modes_variant_summary.json`")
    lines.append("")
    lines.append("## 10. 향후 연구/개발")
    lines.append("")
    lines.append("### 10.1 Object mode 세분화")
    lines.append("")
    lines.append("object mode는 단일 `object_single/object_multi`보다 음식, 동물/pet, 제품, vehicle, indoor object, texture/copy-space object로 나누는 편이 맞다. 음식은 angle/plate/context/negative space가 중요하고, 동물은 얼굴/눈/몸통 보존과 gaze/lead room이 중요하며, 제품은 clean boundary와 배경/context ratio가 중요하다. 단일 foreground saliency score만으로 이들을 모두 설명하기 어렵다.")
    lines.append("")
    lines.append("근거 자료:")
    lines.append("")
    lines.append("- Adobe Research, Automatic Image Cropping using Visual Composition, Boundary Simplicity and Content Preservation Models: https://research.adobe.com/publication/automatic-image-cropping-using-visual-composition-boundary-simplicity-and-content-preservation-models/")
    lines.append("- ProCrop, Learning Aesthetic Image Cropping from Professional Compositions: https://arxiv.org/abs/2505.22490")
    lines.append("- Learning Subject-Aware Cropping by Outpainting Professional Photos: https://arxiv.org/abs/2312.12080")
    lines.append("- A2-RL, Aesthetics Aware Reinforcement Learning for Image Cropping: https://arxiv.org/abs/1709.04595")
    lines.append("- Food photography composition factors: https://cir.nii.ac.jp/crid/1872553968014864768")
    lines.append("")
    lines.append("### 10.2 Person mode 세분화")
    lines.append("")
    lines.append("single person은 `face/headshot`, `upper_body`, `half_body`, `full_body`, `environmental portrait`로 나눠야 한다. 현재 v14에서 support grounding/margin gate가 upper-body에도 강하게 들어가면 충분히 좋은 상반신 crop이 탈락할 수 있다. 후속 v15 후보 정책은 shot type별 utility와 hard gate를 분리하는 방향이 적절하다.")
    lines.append("")
    lines.append("## 11. 생성 파일")
    lines.append("")
    for path in sorted(out_dir.iterdir()):
        if path.is_file():
            lines.append(f"- `{path.name}`")
    lines.append("- `label_json_no_object_modes/`")
    lines.append("")
    (out_dir / "review_jy_260604_followup_report_KO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and fulfill review_jy_260604 multimode v14 follow-up tasks.")
    parser.add_argument("--phase_run_dir", type=Path, default=Path("data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529/artifacts/training_labels_multimode/260601_v14_photo_primary_fullsstk_phaseA_scene_subjectsafe_residual_patch_split9010"))
    parser.add_argument("--full_run_dir", type=Path, default=Path("data/SSTK/Full_10000/artifacts/training_labels_multimode/260601_v14_multimode_full10k_scene_subjectsafe_residual_patch_split9010"))
    parser.add_argument("--output_dir", type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.output_dir or args.phase_run_dir / "crop_review_v14_photo_primary/review_jy_260604_followup"
    out_dir.mkdir(parents=True, exist_ok=True)
    phase_label = _read_json(args.phase_run_dir / "label_json/multimode_labels_full.json")
    full_label = _read_json(args.full_run_dir / "label_json/multimode_labels_full.json")
    phase_summary = summarize_mode_ar(phase_label, args.phase_run_dir / "mode_query_status.jsonl", out_dir, prefix="phaseA")
    full_summary = summarize_mode_ar(full_label, args.full_run_dir / "mode_query_status.jsonl", out_dir, prefix="full")
    sample_summary = write_specific_sample_audits(
        phase_label=phase_label,
        phase_status=args.phase_run_dir / "mode_query_status.jsonl",
        full_label=full_label,
        full_status=args.full_run_dir / "mode_query_status.jsonl",
        out_dir=out_dir,
    )
    no_object_summary = make_no_object_variant(args.phase_run_dir, out_dir)
    write_report(
        out_dir=out_dir,
        phase_summary=phase_summary,
        full_summary=full_summary,
        sample_summary=sample_summary,
        no_object_summary=no_object_summary,
    )
    status = {
        "output_dir": str(out_dir),
        "report": str(out_dir / "review_jy_260604_followup_report_KO.md"),
        "phase_summary": str(out_dir / "phaseA_stats_summary.json"),
        "full_summary": str(out_dir / "full_stats_summary.json"),
        "no_object_variant": str(out_dir / "label_json_no_object_modes"),
    }
    with (out_dir / "run_status.json").open("w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
