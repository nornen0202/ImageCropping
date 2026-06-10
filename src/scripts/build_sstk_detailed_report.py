#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image
try:
    from progress_utils import ProgressTracker, progress_log
except ModuleNotFoundError:
    from scripts.progress_utils import ProgressTracker, progress_log


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build detailed SSTK report package with self-contained assets.")
    p.add_argument("--run_tag", required=True)
    p.add_argument("--data_root", required=True, help="e.g. data/SSTK/Test_100")
    p.add_argument("--report_dir", required=True)
    p.add_argument("--teacher_viz_fallback_dir", default="", help="optional staging dir with by_ar/<ar_tok>/<image_id>.jpg")
    p.add_argument("--teacher_scores_jsonl", default="", help="override teacher JSONL; compact downstream JSONL is preferred when present")
    p.add_argument("--routed_feats_jsonl", default="", help="override routed features JSONL")
    p.add_argument("--candidates_jsonl", default="", help="override candidates JSONL used by drill-down reports")
    p.add_argument("--training_labels_dir", default="", help="override FinalScore training labels directory")
    p.add_argument("--num_workers", type=int, default=0, help="CPU workers for drill-down report generation (0=all cores, 1=single).")
    p.add_argument("--progress", type=int, default=1)
    p.add_argument("--progress_every", type=int, default=250)
    p.add_argument("--progress_min_seconds", type=float, default=10.0)
    return p.parse_args()


def resolve_num_workers(requested_workers: int, num_items: int) -> int:
    if num_items <= 1:
        return 1
    req = int(requested_workers)
    if req == 1:
        return 1
    if req <= 0:
        req = max(1, int(os.cpu_count() or 1))
    return max(1, min(req, num_items))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def ar_token(ar: str) -> str:
    return str(ar).replace(":", "x")


def fmt(v: Any, digits: int = 4) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return str(v)
    return f"{safe_float(v):.{digits}f}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dataset_label_from_data_root(data_root: Path) -> str:
    parts = [part for part in data_root.as_posix().split("/") if part]
    if "data" in parts:
        idx = parts.index("data")
        tail = parts[idx + 1 :]
        if tail:
            return "/".join(tail)
    return data_root.as_posix()


def score_rank_val(row: Dict[str, Any]) -> float:
    scores = row.get("scores", {}) if isinstance(row.get("scores"), dict) else {}
    return safe_float(scores.get("rank", scores.get("final", -1e9)), -1e9)


def score_policy_val(row: Dict[str, Any]) -> float:
    scores = row.get("scores", {}) if isinstance(row.get("scores"), dict) else {}
    return safe_float(scores.get("policy", scores.get("final", -1e9)), -1e9)


def score_legacy_val(row: Dict[str, Any]) -> float:
    scores = row.get("scores", {}) if isinstance(row.get("scores"), dict) else {}
    return safe_float(scores.get("final_legacy", scores.get("expensive", -1e9)), -1e9)


def iou_xyxy(box_a: Any, box_b: Any) -> Optional[float]:
    if not (isinstance(box_a, list) and isinstance(box_b, list) and len(box_a) == 4 and len(box_b) == 4):
        return None
    ax1, ay1, ax2, ay2 = [safe_float(v) for v in box_a]
    bx1, by1, bx2, by2 = [safe_float(v) for v in box_b]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return None
    return inter / denom


def load_teacher_rows(
    path: Path,
    *,
    progress: bool = False,
    progress_every: int = 500,
    progress_min_seconds: float = 10.0,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    tracker = ProgressTracker(
        f"build_sstk_detailed_report:load_teacher_rows:{path.name}",
        unit="rows",
        every=progress_every,
        min_seconds=progress_min_seconds,
        enabled=progress,
    )
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["_row_index"] = idx
            rows.append(rec)
            tracker.update(idx)
    tracker.finish(len(rows), extra=f"path={path.name}")
    return rows


def build_routed_map(
    path: Path,
    *,
    progress: bool = False,
    progress_every: int = 500,
    progress_min_seconds: float = 10.0,
) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    tracker = ProgressTracker(
        f"build_sstk_detailed_report:build_routed_map:{path.name}",
        unit="rows",
        every=progress_every,
        min_seconds=progress_min_seconds,
        enabled=progress,
    )
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            image_id = str(rec.get("image_id", ""))
            rec["_line_index"] = idx
            out[image_id] = rec
            tracker.update(idx, extra=f"mapped={len(out)}")
    tracker.finish(len(out), extra=f"path={path.name}")
    return out


def first_existing_path(paths: Sequence[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def resolve_teacher_jsonl(args: argparse.Namespace, artifacts: Path, run_tag: str) -> Path:
    if str(args.teacher_scores_jsonl).strip():
        return Path(args.teacher_scores_jsonl)
    compact = artifacts / "teacher" / "scores" / f"teacher_scores_ar_{run_tag}_downstream_compact.jsonl"
    full = artifacts / "teacher" / "scores" / f"teacher_scores_ar_{run_tag}.jsonl"
    return compact if compact.exists() else full


def resolve_routed_feats_jsonl(args: argparse.Namespace, artifacts: Path) -> Path:
    if str(args.routed_feats_jsonl).strip():
        return Path(args.routed_feats_jsonl)
    precompute = artifacts / "precompute"
    return first_existing_path(
        [
            precompute / "feats_c2c3c5_v2_strict_enriched_routed_final.jsonl",
            precompute / "feats_c2c3c5_v2_strict_enriched_routed_c7_saliency.jsonl",
            precompute / "feats_c2c3c5_v2_strict_enriched_routed.jsonl",
        ]
    )


def resolve_candidates_jsonl(args: argparse.Namespace, artifacts: Path, run_tag: str) -> Path:
    if str(args.candidates_jsonl).strip():
        return Path(args.candidates_jsonl)
    return artifacts / "candidates" / f"candidates_ar_{run_tag}.jsonl"


def resolve_training_labels_dir(args: argparse.Namespace, artifacts: Path, run_tag: str) -> Path:
    if str(args.training_labels_dir).strip():
        return Path(args.training_labels_dir)
    base = artifacts / "training_labels"
    return first_existing_path(
        [
            base / f"{run_tag}_leftover_ignore_monotonic",
            base / run_tag,
        ]
    )


def load_jsonl_map_for_ids(
    path: Path,
    image_ids: Sequence[str],
    *,
    progress: bool = False,
    progress_every: int = 500,
    progress_min_seconds: float = 10.0,
) -> Dict[str, Dict[str, Any]]:
    wanted = {str(image_id) for image_id in image_ids if str(image_id)}
    out: Dict[str, Dict[str, Any]] = {}
    if not wanted:
        return out
    tracker = ProgressTracker(
        f"build_sstk_detailed_report:load_jsonl_map_for_ids:{path.name}",
        unit="rows",
        every=progress_every,
        min_seconds=progress_min_seconds,
        enabled=progress,
    )
    scanned = 0
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            scanned = idx
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            image_id = str(rec.get("image_id", ""))
            if image_id in wanted:
                out[image_id] = rec
            if len(out) >= len(wanted):
                tracker.update(idx, extra=f"matched={len(out)}/{len(wanted)}")
                break
            tracker.update(idx, extra=f"matched={len(out)}/{len(wanted)}")
    tracker.finish(scanned, extra=f"matched={len(out)}/{len(wanted)} path={path.name}")
    return out


def build_paths(data_root: str, run_tag: str, image_id: str, target_ar: Optional[str] = None) -> Dict[str, str]:
    base = f"{data_root}/artifacts"
    out = {
        "source_teacher_jsonl": f"{base}/teacher/scores/teacher_scores_ar_{run_tag}.jsonl",
        "source_routed_feats_jsonl": f"{base}/precompute/feats_c2c3c5_v2_strict_enriched_routed.jsonl",
        "original_image": f"{base}/precompute/visualizations/components_{run_tag}/original/{image_id}.jpg",
        "precompute_combined": f"{base}/precompute/visualizations/components_{run_tag}/combined_all/{image_id}.jpg",
        "precompute_c2": f"{base}/precompute/visualizations/components_{run_tag}/c2_seg/{image_id}.jpg",
        "precompute_c3": f"{base}/precompute/visualizations/components_{run_tag}/c3_pose/{image_id}.jpg",
        "precompute_c4": f"{base}/precompute/visualizations/components_{run_tag}/c4_ocr/{image_id}.jpg",
        "precompute_c5": f"{base}/precompute/visualizations/components_{run_tag}/c5_geom/{image_id}.jpg",
        "precompute_c6": f"{base}/precompute/visualizations/components_{run_tag}/c6_gaze/{image_id}.jpg",
    }
    if target_ar is not None:
        out["teacher_visualization"] = f"{base}/teacher/visualizations/teacher_scorer_{run_tag}/by_ar/{ar_token(target_ar)}/{image_id}.jpg"
    return out


def build_flat_examples(
    teacher_rows: Sequence[Dict[str, Any]],
    routed_map: Dict[str, Dict[str, Any]],
    data_root: str,
    run_tag: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in teacher_rows:
        image_id = str(row.get("image_id", ""))
        routed = routed_map.get(image_id, {})
        route_global = row.get("route_global", {}) if isinstance(row.get("route_global"), dict) else {}
        pipeline_debug = row.get("pipeline_debug", {}) if isinstance(row.get("pipeline_debug"), dict) else {}
        by_ar = row.get("teacher_scorer", {}).get("results_by_ar", {})
        if not isinstance(by_ar, dict):
            continue
        for target_ar, ar_res in by_ar.items():
            if not isinstance(ar_res, dict):
                continue
            topk = ar_res.get("selected_topk", [])
            if not isinstance(topk, list) or not topk:
                continue
            top1 = topk[0]
            checklist = top1.get("checklist", {}) if isinstance(top1.get("checklist"), dict) else {}
            checklist_labels = top1.get("checklist_labels", {}) if isinstance(top1.get("checklist_labels"), dict) else {}
            entry = {
                "record_index": int(row.get("_row_index", 0)),
                "image_id": image_id,
                "target_ar": str(target_ar),
                "decision_type": str(ar_res.get("decision", {}).get("decision_type", "")),
                "candidate_id": str(top1.get("candidate_id", "")),
                "source": str(top1.get("source", "")),
                "bbox_norm_xyxy": top1.get("bbox_norm_xyxy"),
                "area_ratio": safe_float(top1.get("area_ratio", 0.0)),
                "scores": top1.get("scores", {}),
                "macro_scores": top1.get("macro_scores", {}),
                "flags": top1.get("flags", {}),
                "checklist": checklist,
                "checklist_labels": checklist_labels,
                "why_tags": top1.get("why_tags", []),
                "why_tags_numeric": top1.get("why_tags_numeric", []),
                "why_text_template": top1.get("why_text_template", ""),
                "composition_checks": top1.get("composition_checks", {}),
                "delta_improve": safe_float(ar_res.get("decision", {}).get("delta_improve", 0.0)),
                "tau_improve": safe_float(ar_res.get("decision", {}).get("tau_improve", 0.0)),
                "subject_mode": str(route_global.get("subject_mode", "")),
                "scene_subtype": str(route_global.get("scene_subtype", routed.get("routing", {}).get("scene_subtype", ""))) if isinstance(routed.get("routing"), dict) else str(route_global.get("scene_subtype", "")),
                "policy_id": str(route_global.get("policy_id", "")),
                "subject_mode_conf": safe_float(route_global.get("subject_mode_conf", 0.0)),
                "routing_evidence": {
                    "routed_line_index": routed.get("_line_index", -1),
                    "subject_mode": route_global.get("subject_mode"),
                    "subject_mode_conf": route_global.get("subject_mode_conf"),
                    "subject_mode_reasons": route_global.get("subject_mode_reasons", []),
                    "subject_mode_flags": route_global.get("flags", {}),
                    "subject_mode_conflict": route_global.get("subject_mode_conflict"),
                    "policy_id": route_global.get("policy_id"),
                    "router_rule_id": route_global.get("router_rule_id"),
                    "router_signals": route_global.get("router_signals", {}),
                    "subject_set": route_global.get("subject_set", {}),
                    "c2_seg_count": len(routed.get("c2_seg", [])) if isinstance(routed.get("c2_seg"), list) else 0,
                    "c3_pose_count": len(routed.get("c3_pose", [])) if isinstance(routed.get("c3_pose"), list) else 0,
                    "c4_num_boxes": safe_float((route_global.get("router_signals") or {}).get("ocr_text_boxes", 0), 0.0),
                },
                "paths": build_paths(data_root=data_root, run_tag=run_tag, image_id=image_id, target_ar=str(target_ar)),
                "pipeline_debug": pipeline_debug,
            }
            out.append(entry)
    return out


def pick_best(rows: Sequence[Dict[str, Any]], predicate: Callable[[Dict[str, Any]], bool]) -> Optional[Dict[str, Any]]:
    matches = [row for row in rows if predicate(row)]
    if not matches:
        return None
    matches.sort(
        key=lambda r: (
            score_policy_val(r),
            score_rank_val(r),
            safe_float(r.get("delta_improve", 0.0)),
            -safe_float(r.get("scores", {}).get("components", {}).get("p_cut", 0.0)),
        ),
        reverse=True,
    )
    return matches[0]


def copy_external_asset(
    *,
    report_dir: Path,
    project_root: Path,
    project_rel_path: str,
    mappings: List[Dict[str, str]],
    teacher_viz_fallback_dir: Optional[Path] = None,
) -> Tuple[str, bool]:
    rel = str(project_rel_path).replace("\\", "/").lstrip("/")
    src = project_root / rel
    if (not src.exists()) and teacher_viz_fallback_dir and "/teacher/visualizations/" in rel:
        try:
            suffix = rel.split("/teacher/visualizations/teacher_scorer_", 1)[1].split("/", 1)[1]
            alt = teacher_viz_fallback_dir / suffix
            if alt.exists():
                src = alt
        except Exception:
            pass
    rewritten = f"assets/self_contained/external_links/{rel}"
    dest = report_dir / rewritten
    if not src.exists():
        return rewritten, False
    ensure_dir(dest.parent)
    shutil.copy2(src, dest)
    logical_link = str(Path(report_dir.relative_to(project_root)) / "../../..")  # unused, keep manifest parity via original_link below
    mappings.append(
        {
            "original_link": str(Path("..") / ".." / ".." / Path(rel).relative_to(Path("data/SSTK/Test_100/artifacts/reports")).parent) if False else f"../../{Path(rel).relative_to(Path('data/SSTK/Test_100/artifacts'))}".replace("\\", "/"),
            "rewritten_link": rewritten,
            "src": str(src),
            "dest": str(dest),
        }
    )
    return rewritten, True


def copy_external_by_artifact_rel(
    *,
    report_dir: Path,
    project_root: Path,
    data_root: Path,
    artifact_rel: str,
    mappings: List[Dict[str, str]],
    teacher_viz_fallback_dir: Optional[Path] = None,
) -> Tuple[str, bool]:
    project_rel = str((data_root / "artifacts" / artifact_rel)).replace("\\", "/")
    src = project_root / project_rel
    if (not src.exists()) and teacher_viz_fallback_dir and artifact_rel.startswith("teacher/visualizations/"):
        sub = artifact_rel.split(f"teacher/visualizations/teacher_scorer_", 1)[1].split("/", 1)[1]
        alt = teacher_viz_fallback_dir / sub
        if alt.exists():
            src = alt
    rewritten = f"assets/self_contained/external_links/{project_rel}"
    dest = report_dir / rewritten
    if not src.exists():
        return rewritten, False
    ensure_dir(dest.parent)
    shutil.copy2(src, dest)
    mappings.append(
        {
            "original_link": f"../../{artifact_rel}".replace("\\", "/"),
            "rewritten_link": rewritten,
            "src": str(src),
            "dest": str(dest),
        }
    )
    return rewritten, True


def build_teacher_rep_assets(
    examples: Sequence[Dict[str, Any]],
    report_dir: Path,
    run_tag: str,
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    rep_dir = report_dir / "assets" / "evidence" / "teacher_rep"
    ensure_dir(rep_dir)
    spec: List[Tuple[str, Callable[[Dict[str, Any]], bool]]] = [
        ("decision_keep_full", lambda r: r.get("decision_type") == "keep_full"),
        ("decision_minimal_crop", lambda r: r.get("decision_type") == "minimal_crop"),
        ("decision_crop", lambda r: r.get("decision_type") == "crop"),
        ("subject_coverage_excellent", lambda r: r.get("checklist_labels", {}).get("subject_coverage") == "excellent"),
        ("subject_coverage_poor", lambda r: r.get("checklist_labels", {}).get("subject_coverage") == "poor"),
        ("face_cut", lambda r: r.get("checklist_labels", {}).get("face_cut") == "face_cut"),
        ("joint_cut_severe", lambda r: r.get("checklist_labels", {}).get("joint_cut") == "joint_cut_severe"),
        ("teacher_aligned", lambda r: r.get("checklist_labels", {}).get("teacher_consensus") == "teacher_aligned"),
        ("teacher_mismatch", lambda r: r.get("checklist_labels", {}).get("teacher_consensus") == "teacher_mismatch"),
        ("headroom_tight", lambda r: r.get("checklist_labels", {}).get("headroom") == "headroom_tight"),
        ("lookroom_insufficient", lambda r: r.get("checklist_labels", {}).get("lookroom") == "lookroom_insufficient"),
        ("tight_crop", lambda r: r.get("checklist_labels", {}).get("crop_tightness") == "tight_crop"),
        ("wide_crop", lambda r: r.get("checklist_labels", {}).get("crop_tightness") == "wide_crop"),
    ]
    out: Dict[str, Dict[str, Any]] = {}
    ids: List[str] = []
    for key, pred in spec:
        row = pick_best(examples, pred)
        if row is None:
            continue
        out[key] = row
        ids.append(str(row.get("image_id", "")))
        write_json(rep_dir / f"{key}.json", row)
    write_json(report_dir / "assets" / "evidence" / f"teacher_representative_examples_{run_tag}.json", out)
    return out, ids


def build_checklist_assets(
    examples: Sequence[Dict[str, Any]],
    report_dir: Path,
    run_tag: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    checklist_dir = report_dir / "assets" / "evidence" / "checklist"
    ensure_dir(checklist_dir)
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in examples:
        labels = row.get("checklist_labels", {})
        if not isinstance(labels, dict):
            continue
        for metric, label in labels.items():
            grouped[(str(metric), str(label))].append(row)
    rows_out: List[Dict[str, Any]] = []
    ids: List[str] = []
    for metric, label in sorted(grouped):
        members = grouped[(metric, label)]
        members.sort(
            key=lambda r: (
                score_policy_val(r),
                score_rank_val(r),
                safe_float(r.get("delta_improve", 0.0)),
            ),
            reverse=True,
        )
        rep = members[0]
        item = {
            "metric": metric,
            "label": label,
            "count": len(members),
            "image_id": rep.get("image_id"),
            "target_ar": rep.get("target_ar"),
            "candidate_id": rep.get("candidate_id"),
            "decision_type": rep.get("decision_type"),
            "policy_score": score_policy_val(rep),
            "rank_score": score_rank_val(rep),
            "legacy_score": score_legacy_val(rep),
            "delta_improve": safe_float(rep.get("delta_improve", 0.0)),
            "precompute_combined": rep.get("paths", {}).get("precompute_combined"),
            "teacher_visualization": rep.get("paths", {}).get("teacher_visualization"),
        }
        rows_out.append(item)
        ids.append(str(rep.get("image_id", "")))
        write_json(checklist_dir / f"{metric}__{label}.json", rep)

    analytics_dir = report_dir / "assets" / "analytics"
    ensure_dir(analytics_dir)
    write_json(analytics_dir / f"checklist_label_examples_{run_tag}.json", rows_out)
    with (analytics_dir / f"checklist_label_examples_{run_tag}.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "label",
                "count",
                "image_id",
                "target_ar",
                "candidate_id",
                "decision_type",
                "policy_score",
                "rank_score",
                "legacy_score",
                "delta_improve",
                "precompute_combined",
                "teacher_visualization",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)
    md_lines = [f"# checklist_label_examples_{run_tag}", "", "|metric|label|count|example(image_id/ar)|decision|final_score|delta|", "|---|---:|---:|---|---|---:|---:|"]
    for row in rows_out:
        md_lines.append(
            f"|{row['metric']}|{row['label']}|{row['count']}|{row['image_id']} / {row['target_ar']}|{row['decision_type']}|"
            f"{safe_float(row['policy_score']):.4f}/{safe_float(row['rank_score']):.4f}/{safe_float(row['legacy_score']):.4f}|{safe_float(row['delta_improve']):.4f}|"
        )
    (analytics_dir / f"checklist_label_examples_{run_tag}.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return rows_out, ids


def build_rank_comparison_assets(
    teacher_rows: Sequence[Dict[str, Any]],
    report_dir: Path,
    run_tag: str,
    data_root: str,
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]], List[str]]:
    evidence_dir = report_dir / "assets" / "evidence" / "rank_compare"
    analytics_dir = report_dir / "assets" / "analytics"
    ensure_dir(evidence_dir)
    ensure_dir(analytics_dir)

    face_rank = {"no_face_cut": 0, "face_cut": 1}
    joint_rank = {"no_joint_cut": 0, "joint_cut_mild": 1, "joint_cut_severe": 2}
    coverage_rank = {"poor": 0, "marginal": 1, "good": 2, "excellent": 3}

    rank_rows: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    rep_by_rank: Dict[int, Dict[str, Any]] = {}
    rep_ids: List[str] = []

    for teacher_row in teacher_rows:
        image_id = str(teacher_row.get("image_id", ""))
        by_ar = teacher_row.get("teacher_scorer", {}).get("results_by_ar", {})
        if not isinstance(by_ar, dict):
            continue
        for target_ar, ar_res in by_ar.items():
            if not isinstance(ar_res, dict):
                continue
            topk = ar_res.get("selected_topk", [])
            if not isinstance(topk, list) or not topk:
                continue
            rank1 = topk[0]
            rank1_scores = rank1.get("scores", {}) if isinstance(rank1.get("scores"), dict) else {}
            rank1_labels = rank1.get("checklist_labels", {}) if isinstance(rank1.get("checklist_labels"), dict) else {}
            rank1_final = safe_float(rank1_scores.get("final", 0.0))
            rank1_bbox = rank1.get("bbox_norm_xyxy")
            for rank_idx, cand in enumerate(topk, start=1):
                cand_scores = cand.get("scores", {}) if isinstance(cand.get("scores"), dict) else {}
                cand_labels = cand.get("checklist_labels", {}) if isinstance(cand.get("checklist_labels"), dict) else {}
                entry = {
                    "image_id": image_id,
                    "target_ar": str(target_ar),
                    "rank": rank_idx,
                    "candidate_id": str(cand.get("candidate_id", "")),
                    "source": str(cand.get("source", "")),
                    "bbox_norm_xyxy": cand.get("bbox_norm_xyxy"),
                    "final_score": safe_float(cand_scores.get("final", 0.0)),
                    "cheap_score": safe_float(cand_scores.get("cheap", 0.0)),
                    "expensive_score": safe_float(cand_scores.get("expensive", 0.0)),
                    "delta_vs_rank1": safe_float(cand_scores.get("final", 0.0)) - rank1_final,
                    "iou_vs_rank1": iou_xyxy(rank1_bbox, cand.get("bbox_norm_xyxy")),
                    "face_cut_label": str(cand_labels.get("face_cut", "")),
                    "joint_cut_label": str(cand_labels.get("joint_cut", "")),
                    "subject_coverage_label": str(cand_labels.get("subject_coverage", "")),
                    "is_face_cut_worse_than_rank1": face_rank.get(str(cand_labels.get("face_cut", "")), -1) > face_rank.get(str(rank1_labels.get("face_cut", "")), -1),
                    "is_joint_cut_worse_than_rank1": joint_rank.get(str(cand_labels.get("joint_cut", "")), -1) > joint_rank.get(str(rank1_labels.get("joint_cut", "")), -1),
                    "is_coverage_worse_than_rank1": coverage_rank.get(str(cand_labels.get("subject_coverage", "")), -1) < coverage_rank.get(str(rank1_labels.get("subject_coverage", "")), -1),
                    "paths": build_paths(data_root=data_root, run_tag=run_tag, image_id=image_id, target_ar=str(target_ar)),
                    "why_tags": cand.get("why_tags", []),
                    "checklist_labels": cand_labels,
                }
                rank_rows[rank_idx].append(entry)

    summary_rows: List[Dict[str, Any]] = []
    for rank in range(2, 6):
        rows = rank_rows.get(rank, [])
        if not rows:
            continue
        rows.sort(key=lambda r: (r["final_score"], r["iou_vs_rank1"] if r["iou_vs_rank1"] is not None else -1.0), reverse=True)
        rep = rows[0]
        rep_by_rank[rank] = rep
        rep_ids.append(str(rep.get("image_id", "")))
        write_json(evidence_dir / f"rank_{rank}.json", rep)

        iou_vals = [safe_float(r["iou_vs_rank1"]) for r in rows if r.get("iou_vs_rank1") is not None]
        source_counts = Counter(str(r.get("source", "")) for r in rows)
        item = {
            "rank": rank,
            "count": len(rows),
            "avg_final_score": sum(r["final_score"] for r in rows) / len(rows),
            "avg_cheap_score": sum(r["cheap_score"] for r in rows) / len(rows),
            "avg_expensive_score": sum(r["expensive_score"] for r in rows) / len(rows),
            "avg_delta_vs_rank1": sum(r["delta_vs_rank1"] for r in rows) / len(rows),
            "avg_iou_vs_rank1": (sum(iou_vals) / len(iou_vals)) if iou_vals else None,
            "face_cut_worse_than_rank1_rate": sum(1 for r in rows if r["is_face_cut_worse_than_rank1"]) / len(rows),
            "joint_cut_worse_than_rank1_rate": sum(1 for r in rows if r["is_joint_cut_worse_than_rank1"]) / len(rows),
            "coverage_worse_than_rank1_rate": sum(1 for r in rows if r["is_coverage_worse_than_rank1"]) / len(rows),
            "top_sources": [{"source": k, "count": v} for k, v in source_counts.most_common(5)],
            "representative": {
                "image_id": rep["image_id"],
                "target_ar": rep["target_ar"],
                "candidate_id": rep["candidate_id"],
                "source": rep["source"],
                "final_score": rep["final_score"],
                "delta_vs_rank1": rep["delta_vs_rank1"],
                "iou_vs_rank1": rep["iou_vs_rank1"],
            },
        }
        summary_rows.append(item)

    write_json(analytics_dir / f"topk_rank_comparison_{run_tag}.json", summary_rows)
    return summary_rows, rep_by_rank, rep_ids


def build_public_teacher_comparison_assets(
    teacher_rows: Sequence[Dict[str, Any]],
    public_raw_path: Path,
    report_dir: Path,
    run_tag: str,
    data_root: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], List[str]]:
    evidence_dir = report_dir / "assets" / "evidence" / "public_teachers"
    analytics_dir = report_dir / "assets" / "analytics"
    ensure_dir(evidence_dir)
    ensure_dir(analytics_dir)

    public_raw: Dict[Tuple[str, str], Dict[str, Any]] = {}
    raw_counts: Counter[str] = Counter()
    raw_num_props: Dict[str, List[int]] = defaultdict(list)
    if public_raw_path.exists():
        with public_raw_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                image_id = str(rec.get("image_id", ""))
                teacher_id = str(rec.get("teacher_id", ""))
                public_raw[(image_id, teacher_id)] = rec
                raw_counts[teacher_id] += 1
                raw_num_props[teacher_id].append(int(rec.get("num_proposals", 0)))

    per_teacher_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for teacher_row in teacher_rows:
        image_id = str(teacher_row.get("image_id", ""))
        by_ar = teacher_row.get("teacher_scorer", {}).get("results_by_ar", {})
        if not isinstance(by_ar, dict):
            continue
        for target_ar, ar_res in by_ar.items():
            if not isinstance(ar_res, dict):
                continue
            topk = ar_res.get("selected_topk", [])
            if not isinstance(topk, list) or not topk:
                continue
            top1 = topk[0]
            top1_box = top1.get("bbox_norm_xyxy")
            teacher_seen_top5: set[str] = set()
            for cand in topk:
                source = str(cand.get("source", ""))
                if source.startswith("teacher:"):
                    teacher_seen_top5.add(source.split(":", 1)[1])
            for teacher_id in ["gaic", "cacnet", "cgs"]:
                raw_rec = public_raw.get((image_id, teacher_id), {})
                proposals = raw_rec.get("proposals", []) if isinstance(raw_rec, dict) else []
                best_iou = None
                best_score = None
                for prop in proposals:
                    bbox = prop.get("bbox_norm_xyxy")
                    iou = iou_xyxy(top1_box, bbox)
                    if iou is not None and (best_iou is None or iou > best_iou):
                        best_iou = iou
                        best_score = safe_float(prop.get("score", 0.0))
                top1_from_teacher = str(top1.get("source", "")) == f"teacher:{teacher_id}"
                consensus = ar_res.get("proposal_injection", {}).get("teacher_consensus", {})
                per_teacher_rows[teacher_id].append(
                    {
                        "image_id": image_id,
                        "target_ar": str(target_ar),
                        "teacher_id": teacher_id,
                        "teacher_in_top5": teacher_id in teacher_seen_top5,
                        "teacher_top1_selected": top1_from_teacher,
                        "selected_top1_final_score": safe_float(top1.get("scores", {}).get("final", 0.0)),
                        "best_iou_to_selected_top1": best_iou,
                        "best_raw_teacher_score": best_score,
                        "candidate_top1_iou_to_teacher_seed": safe_float(ar_res.get("proposal_injection", {}).get("candidate_top1_iou_to_teacher_seed", 0.0)),
                        "teacher_consensus_available": bool(consensus.get("available", False)),
                        "teacher_consensus_flag": bool(consensus.get("consensus", False)),
                        "teacher_consensus_baseline_iou": safe_float(consensus.get("baseline_iou", 0.0)),
                        "teacher_consensus_num_boxes": int(consensus.get("num_boxes", 0)) if consensus else 0,
                        "paths": build_paths(data_root=data_root, run_tag=run_tag, image_id=image_id, target_ar=str(target_ar)),
                    }
                )

    summary_rows: List[Dict[str, Any]] = []
    rep_by_teacher: Dict[str, Dict[str, Any]] = {}
    rep_ids: List[str] = []
    for teacher_id in ["gaic", "cacnet", "cgs"]:
        rows = per_teacher_rows.get(teacher_id, [])
        if not rows:
            continue
        ious = [safe_float(r["best_iou_to_selected_top1"]) for r in rows if r.get("best_iou_to_selected_top1") is not None]
        top1_rows = [r for r in rows if r["teacher_top1_selected"]]
        rep_pool = top1_rows or sorted(rows, key=lambda r: (safe_float(r.get("best_iou_to_selected_top1", -1.0)), safe_float(r.get("selected_top1_final_score", -1e9))), reverse=True)
        rep = rep_pool[0]
        rep_by_teacher[teacher_id] = rep
        rep_ids.append(str(rep.get("image_id", "")))
        write_json(evidence_dir / f"{teacher_id}.json", rep)

        item = {
            "teacher_id": teacher_id,
            "raw_image_count": int(raw_counts.get(teacher_id, 0)),
            "avg_raw_num_proposals": (sum(raw_num_props.get(teacher_id, [])) / len(raw_num_props.get(teacher_id, []))) if raw_num_props.get(teacher_id) else 0.0,
            "tasks_total": len(rows),
            "top5_inclusion_rate": sum(1 for r in rows if r["teacher_in_top5"]) / len(rows),
            "top1_selected_rate": sum(1 for r in rows if r["teacher_top1_selected"]) / len(rows),
            "avg_top1_final_when_selected": (sum(r["selected_top1_final_score"] for r in top1_rows) / len(top1_rows)) if top1_rows else None,
            "avg_best_iou_to_selected_top1": (sum(ious) / len(ious)) if ious else None,
            "avg_candidate_top1_iou_to_teacher_seed": sum(r["candidate_top1_iou_to_teacher_seed"] for r in rows) / len(rows),
            "teacher_consensus_available_rate": sum(1 for r in rows if r["teacher_consensus_available"]) / len(rows),
            "teacher_consensus_true_rate": sum(1 for r in rows if r["teacher_consensus_flag"]) / len(rows),
            "representative": {
                "image_id": rep["image_id"],
                "target_ar": rep["target_ar"],
                "teacher_top1_selected": rep["teacher_top1_selected"],
                "best_iou_to_selected_top1": rep["best_iou_to_selected_top1"],
                "selected_top1_final_score": rep["selected_top1_final_score"],
            },
        }
        summary_rows.append(item)

    write_json(analytics_dir / f"public_teacher_comparison_{run_tag}.json", summary_rows)
    return summary_rows, rep_by_teacher, rep_ids


def unique_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def distribution_lines(counter: Counter[str]) -> List[str]:
    total = sum(counter.values())
    lines: List[str] = []
    for key, count in counter.most_common():
        rate = (count / total) if total else 0.0
        lines.append(f"|{key}|{count}|{fmt(rate)}|")
    return lines


def summarize_score_migration(teacher_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    task_total = 0
    chosen_vs_best_rank = 0
    chosen_vs_best_policy = 0
    best_rank_vs_best_legacy = 0
    top1_rank_vals: List[float] = []
    top1_policy_vals: List[float] = []
    top1_legacy_vals: List[float] = []
    top1_area_priors: List[float] = []
    macro_top1 = defaultdict(list)
    checklist_counters = {
        "subject_scale": Counter(),
        "context": Counter(),
        "copyspace": Counter(),
        "roll": Counter(),
        "horizon": Counter(),
    }
    for teacher_row in teacher_rows:
        by_ar = teacher_row.get("teacher_scorer", {}).get("results_by_ar", {})
        if not isinstance(by_ar, dict):
            continue
        for ar_res in by_ar.values():
            if not isinstance(ar_res, dict):
                continue
            scored = []
            for key in ["selected_topk", "hard_negatives", "also_considered_rejected"]:
                vals = ar_res.get(key, [])
                if isinstance(vals, list):
                    scored.extend([v for v in vals if isinstance(v, dict)])
            uniq_scored = {}
            for cand in scored:
                cid = str(cand.get("candidate_id", ""))
                if cid and cid not in uniq_scored:
                    uniq_scored[cid] = cand
            pool = list(uniq_scored.values())
            if not pool:
                continue
            task_total += 1
            best_rank = max(pool, key=score_rank_val)
            best_policy = max(pool, key=score_policy_val)
            best_legacy = max(pool, key=score_legacy_val)
            chosen = None
            chosen_id = str(ar_res.get("decision", {}).get("chosen_candidate_id", ""))
            if chosen_id:
                chosen = uniq_scored.get(chosen_id)
            if chosen is None:
                topk = ar_res.get("selected_topk", [])
                chosen = topk[0] if isinstance(topk, list) and topk else best_policy
            if str(chosen.get("candidate_id", "")) != str(best_rank.get("candidate_id", "")):
                chosen_vs_best_rank += 1
            if str(chosen.get("candidate_id", "")) != str(best_policy.get("candidate_id", "")):
                chosen_vs_best_policy += 1
            if str(best_rank.get("candidate_id", "")) != str(best_legacy.get("candidate_id", "")):
                best_rank_vs_best_legacy += 1
            top1_rank_vals.append(score_rank_val(chosen))
            top1_policy_vals.append(score_policy_val(chosen))
            top1_legacy_vals.append(score_legacy_val(chosen))
            top1_area_priors.append(safe_float((chosen.get("scores", {}) if isinstance(chosen.get("scores"), dict) else {}).get("area_log_prior", 0.0)))
            for key in ["subject_scale", "context", "copyspace", "roll", "horizon"]:
                checklist_counters[key][str((chosen.get("checklist_labels", {}) if isinstance(chosen.get("checklist_labels"), dict) else {}).get(key, "na"))] += 1
            macro = chosen.get("macro_scores", {}) if isinstance(chosen.get("macro_scores"), dict) else {}
            for mk in ["A_macro", "S_macro", "C_macro", "T_macro"]:
                if mk in macro:
                    macro_top1[mk].append(safe_float(macro.get(mk, 0.0)))

    return {
        "task_total": task_total,
        "chosen_vs_best_rank": chosen_vs_best_rank,
        "chosen_vs_best_policy": chosen_vs_best_policy,
        "best_rank_vs_best_legacy": best_rank_vs_best_legacy,
        "chosen_vs_best_rank_rate": (chosen_vs_best_rank / task_total) if task_total else 0.0,
        "chosen_vs_best_policy_rate": (chosen_vs_best_policy / task_total) if task_total else 0.0,
        "best_rank_vs_best_legacy_rate": (best_rank_vs_best_legacy / task_total) if task_total else 0.0,
        "mean_top1_rank": (sum(top1_rank_vals) / len(top1_rank_vals)) if top1_rank_vals else 0.0,
        "mean_top1_policy": (sum(top1_policy_vals) / len(top1_policy_vals)) if top1_policy_vals else 0.0,
        "mean_top1_legacy": (sum(top1_legacy_vals) / len(top1_legacy_vals)) if top1_legacy_vals else 0.0,
        "mean_top1_area_log_prior": (sum(top1_area_priors) / len(top1_area_priors)) if top1_area_priors else 0.0,
        "macro_mean_top1": {
            mk: ((sum(vals) / len(vals)) if vals else 0.0)
            for mk, vals in macro_top1.items()
        },
        "checklist_top1": {key: dict(counter) for key, counter in checklist_counters.items()},
    }


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def summarize_training_labels(
    training_dir: Path,
    run_tag: str,
    report_dir: Path,
    *,
    progress: bool = False,
    progress_every: int = 500,
    progress_min_seconds: float = 10.0,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    qa_path = training_dir / "qa_summary.json"
    listwise_path = training_dir / "train_listwise.jsonl"
    decision_path = training_dir / "train_decision.jsonl"
    if not qa_path.exists():
        return None, [], []
    qa = read_json(qa_path)
    decision_rows = load_jsonl(decision_path)
    decision_map = {
        (str(row.get("image_id", "")), str(row.get("target_ar", ""))): row
        for row in decision_rows
    }
    rank_vs_policy = 0
    rank_vs_legacy = 0
    chosen_vs_best_rank = 0
    examples: List[Dict[str, Any]] = []
    extra_ids: List[str] = []
    tracker = ProgressTracker(
        "build_sstk_detailed_report:summarize_training_labels",
        unit="rows",
        every=progress_every,
        min_seconds=progress_min_seconds,
        enabled=progress,
    )
    task_total = 0
    for idx, row in enumerate(iter_jsonl(listwise_path), start=1):
        task_total = idx
        cands = row.get("candidates", [])
        if not isinstance(cands, list) or not cands:
            tracker.update(idx)
            continue
        best_rank = max(cands, key=lambda c: safe_float(c.get("score_raw_rank", -1e9), -1e9))
        best_policy = max(cands, key=lambda c: safe_float(c.get("score_raw_policy", -1e9), -1e9))
        best_legacy = max(cands, key=lambda c: safe_float(c.get("score_raw_final_legacy", -1e9), -1e9))
        key = (str(row.get("image_id", "")), str(row.get("target_ar", "")))
        dec = decision_map.get(key, {})
        chosen_id = str(dec.get("winner_post_gate_candidate_id", dec.get("winner_pre_gate_candidate_id", "")))
        if str(best_rank.get("candidate_id", "")) != str(best_policy.get("candidate_id", "")):
            rank_vs_policy += 1
            examples.append(
                {
                    "kind": "rank_vs_policy",
                    "image_id": row.get("image_id"),
                    "target_ar": row.get("target_ar"),
                    "mode": row.get("mode"),
                    "decision_type": row.get("decision_type"),
                    "best_rank_candidate_id": best_rank.get("candidate_id"),
                    "best_policy_candidate_id": best_policy.get("candidate_id"),
                    "best_rank_score": safe_float(best_rank.get("score_raw_rank", 0.0)),
                    "best_policy_score": safe_float(best_policy.get("score_raw_policy", 0.0)),
                }
            )
            extra_ids.append(str(row.get("image_id", "")))
        if str(best_rank.get("candidate_id", "")) != str(best_legacy.get("candidate_id", "")):
            rank_vs_legacy += 1
            extra_ids.append(str(row.get("image_id", "")))
        if chosen_id and chosen_id != str(best_rank.get("candidate_id", "")):
            chosen_vs_best_rank += 1
            examples.append(
                {
                    "kind": "gate_vs_best_rank",
                    "image_id": row.get("image_id"),
                    "target_ar": row.get("target_ar"),
                    "mode": row.get("mode"),
                    "decision_type": row.get("decision_type"),
                    "chosen_candidate_id": chosen_id,
                    "best_rank_candidate_id": best_rank.get("candidate_id"),
                    "best_policy_candidate_id": best_policy.get("candidate_id"),
                    "best_rank_score": safe_float(best_rank.get("score_raw_rank", 0.0)),
                    "best_policy_score": safe_float(best_policy.get("score_raw_policy", 0.0)),
                    "delta_vs_base": dec.get("delta_vs_base"),
                    "tau_improve": dec.get("tau_improve"),
                }
            )
            extra_ids.append(str(row.get("image_id", "")))
        tracker.update(idx)

    examples = unique_keep_order(json.dumps(x, ensure_ascii=False, sort_keys=True) for x in examples)
    example_rows = [json.loads(x) for x in examples[:12]]
    write_json(report_dir / "assets" / "analytics" / f"training_label_examples_{run_tag}.json", example_rows)
    summary = {
        "qa": qa,
        "task_total": task_total,
        "rank_vs_policy_count": rank_vs_policy,
        "rank_vs_policy_rate": (rank_vs_policy / task_total) if task_total else 0.0,
        "rank_vs_legacy_count": rank_vs_legacy,
        "rank_vs_legacy_rate": (rank_vs_legacy / task_total) if task_total else 0.0,
        "chosen_vs_best_rank_count": chosen_vs_best_rank,
        "chosen_vs_best_rank_rate": (chosen_vs_best_rank / task_total) if task_total else 0.0,
        "paths": {
            "qa_summary": str(qa_path).replace("\\", "/"),
            "training_report": str((training_dir / "TRAINING_DATA_REPORT_KO.md")).replace("\\", "/"),
            "implementation_report": str((training_dir / "FINALSCORE_TRAINING_DATA_IMPLEMENTATION_REPORT_KO.md")).replace("\\", "/"),
        },
    }
    tracker.finish(task_total, extra=f"examples={len(example_rows)}")
    return summary, example_rows, extra_ids


def validate_single_image_report(report_dir: Path, analysis: Dict[str, Any]) -> Dict[str, Any]:
    issues: List[str] = []
    tile_w = 520
    tile_h = 1040
    for ar, ar_info in (analysis.get("ars", {}) if isinstance(analysis.get("ars"), dict) else {}).items():
        token = ar_token(ar)
        overlay = report_dir / ar_info.get("overlay_rel", "")
        panel = report_dir / ar_info.get("panel_rel", "")
        if not overlay.exists():
            issues.append(f"{ar}:missing_overlay")
        else:
            with Image.open(overlay) as img:
                if img.width <= 32 or img.height <= 80:
                    issues.append(f"{ar}:overlay_too_small")
        if not panel.exists():
            issues.append(f"{ar}:missing_panel")
        else:
            item_count = len(ar_info.get("topk", [])) + len(ar_info.get("teacher_refs", []))
            expected_h = math.ceil(max(1, item_count) / 2) * tile_h + (math.ceil(max(1, item_count) / 2) + 1) * 20 + 60
            with Image.open(panel) as img:
                if img.width != 1100 or img.height != expected_h:
                    issues.append(f"{ar}:panel_size_mismatch:{img.width}x{img.height}!={1100}x{expected_h}")
        crop_dir = report_dir / "assets" / "crops" / token
        crop_files = sorted(crop_dir.glob("*.jpg")) if crop_dir.exists() else []
        if len(crop_files) < len(ar_info.get("topk", [])):
            issues.append(f"{ar}:crop_count_lt_topk")
        for crop in crop_files:
            with Image.open(crop) as img:
                if img.width != tile_w or img.height != tile_h:
                    issues.append(f"{ar}:tile_size_mismatch:{crop.name}:{img.width}x{img.height}")
    return {
        "report_dir": str(report_dir),
        "issue_count": len(issues),
        "issues": issues,
    }


def generate_drilldown_reports(
    *,
    run_tag: str,
    data_root: Path,
    report_dir: Path,
    image_ids: Sequence[str],
    routed_map: Dict[str, Dict[str, Any]],
    teacher_rows: Sequence[Dict[str, Any]],
    teacher_jsonl: Path,
    candidates_jsonl: Path,
    features_jsonl: Path,
    ars: str = "FREE,1:1,16:9",
    num_workers: int = 1,
    progress: bool = False,
    progress_every: int = 10,
    progress_min_seconds: float = 10.0,
) -> Dict[str, Any]:
    teacher_map = {str(row.get("image_id", "")): row for row in teacher_rows}
    candidate_map = load_jsonl_map_for_ids(
        candidates_jsonl,
        image_ids,
        progress=progress,
        progress_every=max(1, progress_every),
        progress_min_seconds=progress_min_seconds,
    )
    cache_dir = report_dir / "assets" / "_drilldown_input_rows"
    ensure_dir(cache_dir)
    cached_teacher_rows: Dict[str, Path] = {}
    cached_candidate_rows: Dict[str, Path] = {}
    cached_feature_rows: Dict[str, Path] = {}
    for image_id in image_ids:
        image_id_str = str(image_id)
        if image_id_str in teacher_map:
            path = cache_dir / "teacher" / f"{image_id_str}.json"
            write_json(path, teacher_map[image_id_str])
            cached_teacher_rows[image_id_str] = path
        if image_id_str in candidate_map:
            path = cache_dir / "candidates" / f"{image_id_str}.json"
            write_json(path, candidate_map[image_id_str])
            cached_candidate_rows[image_id_str] = path
        if image_id_str in routed_map:
            path = cache_dir / "features" / f"{image_id_str}.json"
            write_json(path, routed_map[image_id_str])
            cached_feature_rows[image_id_str] = path
    index_lines = ["# Drill-down Report Index", ""]

    def build_one_report(image_id: str) -> Tuple[str, Dict[str, Any], str]:
        out_dir = report_dir / image_id
        attempts = 0
        last_qa: Optional[Dict[str, Any]] = None
        while attempts < 2:
            cmd = [
                sys.executable,
                "src/scripts/build_single_image_crop_report.py",
                "--run_tag",
                run_tag,
                "--data_root",
                str(data_root),
                "--image_id",
                image_id,
                "--out_dir",
                str(out_dir),
                "--ars",
                ars,
                "--teacher_scores_jsonl",
                str(teacher_jsonl),
                "--candidates_jsonl",
                str(candidates_jsonl),
                "--features_jsonl",
                str(features_jsonl),
            ]
            if image_id in cached_teacher_rows:
                cmd.extend(["--teacher_row_json", str(cached_teacher_rows[image_id])])
            if image_id in cached_candidate_rows:
                cmd.extend(["--candidate_row_json", str(cached_candidate_rows[image_id])])
            if image_id in cached_feature_rows:
                cmd.extend(["--feature_row_json", str(cached_feature_rows[image_id])])
            subprocess.run(
                cmd,
                check=True,
            )
            analysis_path = out_dir / "assets" / "analysis.json"
            if not analysis_path.exists():
                last_qa = {"report_dir": str(out_dir), "issue_count": 1, "issues": ["missing_analysis_json"]}
            else:
                analysis = read_json(analysis_path)
                last_qa = validate_single_image_report(out_dir, analysis)
            if last_qa["issue_count"] == 0:
                break
            attempts += 1
        if last_qa is None:
            last_qa = {"report_dir": str(out_dir), "issue_count": 1, "issues": ["unknown_failure"]}
        write_json(out_dir / "assets" / "layout_qa.json", last_qa)
        route = teacher_map.get(image_id, {}).get("route_global", {}) if isinstance(teacher_map.get(image_id, {}).get("route_global"), dict) else {}
        subject_mode = str(route.get("subject_mode", routed_map.get(image_id, {}).get("routing", {}).get("subject_mode", ""))) if isinstance(routed_map.get(image_id, {}).get("routing"), dict) else str(route.get("subject_mode", ""))
        return image_id, last_qa, subject_mode

    ordered_results: List[Tuple[str, Dict[str, Any], str]] = []
    tracker = ProgressTracker(
        "build_sstk_detailed_report:generate_drilldown_reports",
        total=len(image_ids),
        unit="images",
        every=max(1, progress_every),
        min_seconds=progress_min_seconds,
        enabled=progress,
    )
    if int(num_workers) <= 1:
        for idx, image_id in enumerate(image_ids, start=1):
            ordered_results.append(build_one_report(image_id))
            tracker.update(idx, extra=f"image_id={image_id}")
    else:
        result_map: Dict[str, Tuple[str, Dict[str, Any], str]] = {}
        with ThreadPoolExecutor(max_workers=int(num_workers)) as executor:
            future_map = {executor.submit(build_one_report, image_id): image_id for image_id in image_ids}
            completed = 0
            for future in as_completed(future_map):
                image_id = future_map[future]
                result_map[image_id] = future.result()
                completed += 1
                tracker.update(completed, extra=f"image_id={image_id}")
        for image_id in image_ids:
            ordered_results.append(result_map[image_id])

    qa_rows: List[Dict[str, Any]] = []
    for image_id, last_qa, subject_mode in ordered_results:
        qa_rows.append(last_qa)
        index_lines.append(f"- [{image_id}]({image_id}/REPORT_{image_id}_KO.md) `{subject_mode}`")
    index_path = report_dir / "DRILLDOWN_INDEX.md"
    index_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    qa_summary = {
        "total_reports": len(qa_rows),
        "reports_with_issues": sum(1 for row in qa_rows if row.get("issue_count", 0) > 0),
        "total_issue_count": sum(int(row.get("issue_count", 0)) for row in qa_rows),
        "rows": qa_rows,
    }
    write_json(report_dir / "assets" / "analytics" / "drilldown_layout_qa.json", qa_summary)
    tracker.finish(len(image_ids), extra=f"reports_with_issues={qa_summary['reports_with_issues']}")
    return qa_summary


def build_report_zip(report_dir: Path) -> Path:
    zip_path = report_dir / f"{report_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(report_dir.rglob("*")):
            if path == zip_path or path.is_dir():
                continue
            zf.write(path, path.relative_to(report_dir))
    return zip_path


def build_overview_zip(report_dir: Path) -> Path:
    zip_path = report_dir / f"{report_dir.name}_overview.zip"
    include_roots = [
        report_dir / "REPORT_DRAFT_KO.md",
        report_dir / "DRILLDOWN_INDEX.md",
        report_dir / "assets" / "analytics",
        report_dir / "assets" / "evidence",
        report_dir / "assets" / "self_contained" / "self_contained_manifest.json",
    ]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root in include_roots:
            if not root.exists():
                continue
            if root.is_file():
                zf.write(root, root.relative_to(report_dir))
                continue
            for path in sorted(root.rglob("*")):
                if path.is_dir():
                    continue
                zf.write(path, path.relative_to(report_dir))
    return zip_path


def build_report_text(
    *,
    run_tag: str,
    subject_count_examples: Dict[str, List[Dict[str, Any]]],
    subject_mode_examples: Dict[str, List[Dict[str, Any]]],
    teacher_rep: Dict[str, Dict[str, Any]],
    checklist_rows: List[Dict[str, Any]],
    rank_rows: List[Dict[str, Any]],
    rank_reps: Dict[int, Dict[str, Any]],
    public_teacher_rows: List[Dict[str, Any]],
    public_teacher_reps: Dict[str, Dict[str, Any]],
    summary: Dict[str, Any],
    scene_subtype_counts: Dict[str, int],
    qa: Dict[str, Any],
    overview: Dict[str, Any],
    cand_overview: Dict[str, Any],
    viz_overview: Dict[str, Any],
    score_migration_summary: Dict[str, Any],
    training_labels_summary: Optional[Dict[str, Any]],
    training_label_examples: Sequence[Dict[str, Any]],
    drilldown_summary: Optional[Dict[str, Any]],
    mappings: List[Dict[str, str]],
    report_dir: Path,
    project_root: Path,
    data_root: Path,
    teacher_viz_fallback_dir: Optional[Path],
) -> str:
    counts = summary["counts"]
    p0 = qa["global"]["p0_release_gates"]
    decision_counts = qa["global"]["decision_counts"]
    decision_rates = qa["global"]["decision_rates"]
    mode_qa = qa["mode_qa_summary"]
    global_qa = qa["global"]
    over_global = overview["summary"]["global"]
    missing_assets: List[str] = []
    mode_order = sorted(counts["subject_mode_counts"].keys())
    dataset_label = dataset_label_from_data_root(data_root)
    dataset_path = data_root.as_posix()
    cand_num_images = int(cand_overview.get("num_images", counts["num_images"]))
    real_applied_images = int(overview.get("expensive_stage", {}).get("real_applied_images", 0))
    ar_per_image = (float(counts["num_top1_ar_results"]) / float(counts["num_images"])) if counts["num_images"] else 0.0
    ar_per_image_text = str(int(round(ar_per_image))) if abs(ar_per_image - round(ar_per_image)) <= 1e-6 else f"{ar_per_image:.2f}"
    text_document_count = int(counts["subject_mode_counts"].get("text_document", 0))
    overview_config = overview.get("config", {}) if isinstance(overview.get("config"), dict) else {}
    use_real_expensive = int(safe_float(overview_config.get("use_real_expensive", 0), 0.0))
    public_teacher_prop_summary = ", ".join(
        f"{row['teacher_id']}≈{fmt(row['avg_raw_num_proposals'],2)}"
        for row in public_teacher_rows
    ) if public_teacher_rows else "public raw teacher outputs unavailable"
    public_teacher_top1_summary = ", ".join(
        f"{row['teacher_id']}={int(round(safe_float(row.get('top1_selected_rate', 0.0)) * int(row.get('tasks_total', 0))))}/{int(row.get('tasks_total', 0))}"
        for row in public_teacher_rows
    ) if public_teacher_rows else "집계 대상 없음"

    def sc_link(artifact_rel: str) -> str:
        rewritten, ok = copy_external_by_artifact_rel(
            report_dir=report_dir,
            project_root=project_root,
            data_root=data_root,
            artifact_rel=artifact_rel,
            mappings=mappings,
            teacher_viz_fallback_dir=teacher_viz_fallback_dir,
        )
        if not ok:
            missing_assets.append(artifact_rel)
        return rewritten

    mode_qa_lines: List[str] = []
    for mode in mode_order:
        row = mode_qa.get(mode)
        if not isinstance(row, dict):
            continue
        mode_qa_lines.append(
            f"|{mode}|{row['count']}|{fmt(row['num_person_mean'],2)}|{fmt(row['num_c2_instances_mean'],2)}|"
            f"{fmt(row['num_effective_subjects_mean'],2)}|{fmt(row['primary_subject_exists_rate'])}|"
            f"{fmt(row['face_cut_rate'])}|{fmt(row['joint_cut_rate'])}|{fmt(row['head_top_cut_rate'])}|"
            f"{fmt(row['portrait_route_without_human_rate'])}|{fmt(row['scene_horizon_na_rate'])}|"
        )

    subject_bucket_lines: List[str] = []
    bucket_label = {"ge2": ">=2", "eq1": "=1", "eq0": "=0"}
    for bucket in ["ge2", "eq1", "eq0"]:
        for row in subject_count_examples.get(bucket, [])[:3]:
            ss = row["subject_set"]
            rs = row.get("router_signals", {}) or row.get("routing", {}).get("router_signals", {})
            reasons = "; ".join(row.get("subject_mode_reasons", []))
            img_link = sc_link(f"precompute/visualizations/components_{run_tag}/combined_all/{row['image_id']}.jpg")
            subject_bucket_lines.append(
                f"|{bucket_label[bucket]}|{row['image_id']}|{row['subject_mode']} ({safe_float(row['subject_mode_conf']):.2f})|"
                f"{ss.get('num_person', 0)} / {ss.get('num_c2_instances', 0)} / {ss.get('num_effective_subjects', 0)}|"
                f"{reasons}; rule={row.get('router_rule_id', '')}; num_person_source={rs.get('num_person_source', 'none')}|"
                f"[combined]({img_link})|[json](assets/evidence/subject_count/{bucket}_{row['image_id']}.json)|"
            )

    mode_lines: List[str] = []
    for mode in mode_order:
        reps = subject_mode_examples.get(mode, [])
        if not reps:
            continue
        rep = reps[0]
        ss = rep["subject_set"]
        reasons = "; ".join(rep.get("subject_mode_reasons", []))
        img_link = sc_link(f"precompute/visualizations/components_{run_tag}/combined_all/{rep['image_id']}.jpg")
        mode_lines.append(
            f"|{mode}|{counts['subject_mode_counts'].get(mode, 0)}|{rep['image_id']}|"
            f"{ss.get('num_person', 0)} / {ss.get('num_c2_instances', 0)} / {ss.get('num_effective_subjects', 0)}|"
            f"{safe_float(rep['subject_mode_conf']):.2f}|{reasons}; rule={rep.get('router_rule_id', '')}|"
            f"[combined]({img_link})|[json](assets/evidence/subject_mode/{mode}_{rep['image_id']}.json)|"
        )

    scene_subtype_counter: Counter[str] = Counter(scene_subtype_counts)
    scene_subtype_lines = distribution_lines(scene_subtype_counter) if scene_subtype_counter else ["|na|0|0.0000|"]

    checklist_snapshot_lines: List[str] = []
    for key in ["subject_scale", "context", "copyspace", "roll", "horizon"]:
        counter = Counter(score_migration_summary.get("checklist_top1", {}).get(key, {}))
        for label, count in counter.most_common():
            rate = (count / score_migration_summary.get("task_total", 1)) if score_migration_summary.get("task_total", 0) else 0.0
            checklist_snapshot_lines.append(f"|{key}|{label}|{count}|{fmt(rate)}|")

    macro_mean = score_migration_summary.get("macro_mean_top1", {})
    macro_summary_lines = [
        f"|selected_top1 mean rank|{fmt(score_migration_summary.get('mean_top1_rank', 0.0))}|",
        f"|selected_top1 mean policy|{fmt(score_migration_summary.get('mean_top1_policy', 0.0))}|",
        f"|selected_top1 mean legacy|{fmt(score_migration_summary.get('mean_top1_legacy', 0.0))}|",
        f"|selected_top1 mean area_log_prior|{fmt(score_migration_summary.get('mean_top1_area_log_prior', 0.0))}|",
        f"|chosen != best(rank)|{score_migration_summary.get('chosen_vs_best_rank', 0)} / {score_migration_summary.get('task_total', 0)} ({fmt(score_migration_summary.get('chosen_vs_best_rank_rate', 0.0))})|",
        f"|chosen != best(policy)|{score_migration_summary.get('chosen_vs_best_policy', 0)} / {score_migration_summary.get('task_total', 0)} ({fmt(score_migration_summary.get('chosen_vs_best_policy_rate', 0.0))})|",
        f"|best(rank) != best(legacy)|{score_migration_summary.get('best_rank_vs_best_legacy', 0)} / {score_migration_summary.get('task_total', 0)} ({fmt(score_migration_summary.get('best_rank_vs_best_legacy_rate', 0.0))})|",
        f"|mean A/S/C/T|A={fmt(macro_mean.get('A_macro', 0.0))}, S={fmt(macro_mean.get('S_macro', 0.0))}, C={fmt(macro_mean.get('C_macro', 0.0))}, T={fmt(macro_mean.get('T_macro', 0.0))}|",
    ]

    training_label_lines: List[str] = []
    training_example_lines: List[str] = []
    training_report_link = ""
    training_impl_link = ""
    training_qa_link = ""
    if training_labels_summary:
        qa_counts = training_labels_summary["qa"]["counts"]
        training_label_lines.extend(
            [
                f"|pairwise/listwise/decision|{qa_counts['pairwise']} / {qa_counts['listwise']} / {qa_counts['decision']}|",
                f"|checklist/regression|{qa_counts['checklist']} / {qa_counts['regression']}|",
                f"|rank_vs_policy disagree|{training_labels_summary['rank_vs_policy_count']} / {training_labels_summary['task_total']} ({fmt(training_labels_summary['rank_vs_policy_rate'])})|",
                f"|rank_vs_legacy disagree|{training_labels_summary['rank_vs_legacy_count']} / {training_labels_summary['task_total']} ({fmt(training_labels_summary['rank_vs_legacy_rate'])})|",
                f"|chosen != best(rank)|{training_labels_summary['chosen_vs_best_rank_count']} / {training_labels_summary['task_total']} ({fmt(training_labels_summary['chosen_vs_best_rank_rate'])})|",
                f"|unsafe_negative_ratio(pairwise)|{fmt(training_labels_summary['qa']['pairwise']['unsafe_negative_ratio'])}|",
                f"|baseline_dominance_rate(decision)|{fmt(training_labels_summary['qa']['decision']['baseline_dominance_rate'])}|",
            ]
        )
        for row in training_label_examples[:8]:
            training_example_lines.append(
                f"|{row.get('kind','')}|{row.get('image_id','')} / {row.get('target_ar','')}|{row.get('mode','')}|{row.get('decision_type','')}|"
                f"{row.get('best_rank_candidate_id', row.get('chosen_candidate_id', ''))}|{row.get('best_policy_candidate_id', '')}|"
                f"{fmt(row.get('best_rank_score')) if row.get('best_rank_score') is not None else 'na'}|"
                f"{fmt(row.get('best_policy_score')) if row.get('best_policy_score') is not None else 'na'}|"
            )
        training_report_link = sc_link(f"training_labels/{run_tag}/TRAINING_DATA_REPORT_KO.md")
        training_impl_link = sc_link(f"training_labels/{run_tag}/FINALSCORE_TRAINING_DATA_IMPLEMENTATION_REPORT_KO.md")
        training_qa_link = sc_link(f"training_labels/{run_tag}/qa_summary.json")

    drilldown_summary_lines: List[str] = []
    if drilldown_summary:
        drilldown_summary_lines.extend(
            [
                f"|generated_reports|{drilldown_summary.get('total_reports', 0)}|",
                f"|reports_with_issues|{drilldown_summary.get('reports_with_issues', 0)}|",
                f"|total_issue_count|{drilldown_summary.get('total_issue_count', 0)}|",
            ]
        )

    rep_lines: List[str] = []
    ordered_rep_keys = [
        "decision_keep_full",
        "decision_minimal_crop",
        "decision_crop",
        "subject_coverage_excellent",
        "subject_coverage_poor",
        "face_cut",
        "joint_cut_severe",
        "teacher_aligned",
        "teacher_mismatch",
        "headroom_tight",
        "lookroom_insufficient",
        "tight_crop",
        "wide_crop",
    ]
    for key in ordered_rep_keys:
        row = teacher_rep.get(key)
        if not row:
            continue
        scores = row.get("scores", {})
        comps = scores.get("components", {}) if isinstance(scores, dict) else {}
        macro = row.get("macro_scores", {}) if isinstance(row.get("macro_scores"), dict) else {}
        labels = row.get("checklist_labels", {})
        core_labels = (
            f"cov={labels.get('subject_coverage')}, face={labels.get('face_cut')}, "
            f"joint={labels.get('joint_cut')}, cons={labels.get('teacher_consensus')}, tight={labels.get('crop_tightness')}"
        )
        why = ", ".join((row.get("why_tags") or [])[:5])
        teacher_viz = sc_link(f"teacher/visualizations/teacher_scorer_{run_tag}/by_ar/{ar_token(row['target_ar'])}/{row['image_id']}.jpg")
        rep_lines.append(
            f"|{key}|{row['image_id']} / {row['target_ar']}|{row['decision_type']}|"
            f"{row['candidate_id']} ({row['source']})|"
            f"{fmt(scores.get('cheap'))} / {fmt(scores.get('expensive'))} / {fmt(scores.get('rank'))} / {fmt(scores.get('policy'))} / {fmt(scores.get('final_legacy'))}|"
            f"{fmt(comps.get('cov', 0.0),3)}, {fmt(comps.get('p_cut', 0.0),3)}, {fmt(comps.get('r_teach', 0.0),3)}, "
            f"{fmt(comps.get('aesthetic_norm', 0.0),3)}, {fmt(comps.get('cosine_img_text', 0.0),3)}|"
            f"A={fmt(macro.get('A_macro'))}, S={fmt(macro.get('S_macro'))}, C={fmt(macro.get('C_macro'))}, T={fmt(macro.get('T_macro'))}|"
            f"{core_labels}|{why}|[teacher_viz]({teacher_viz})|[json](assets/evidence/teacher_rep/{key}.json)|"
        )

    checklist_lines: List[str] = []
    for row in checklist_rows:
        teacher_viz = sc_link(f"teacher/visualizations/teacher_scorer_{run_tag}/by_ar/{ar_token(row['target_ar'])}/{row['image_id']}.jpg")
        checklist_lines.append(
            f"|{row['metric']}|{row['label']}|{row['count']}|{row['image_id']} / {row['target_ar']}|{row['decision_type']}|"
            f"{fmt(row['policy_score'])}/{fmt(row['rank_score'])}/{fmt(row['legacy_score'])}|[img]({teacher_viz})|[json](assets/evidence/checklist/{row['metric']}__{row['label']}.json)|"
        )

    rank_lines: List[str] = []
    for row in rank_rows:
        rep = rank_reps.get(int(row["rank"]))
        teacher_viz = ""
        rep_json = ""
        if rep:
            teacher_viz = sc_link(f"teacher/visualizations/teacher_scorer_{run_tag}/by_ar/{ar_token(rep['target_ar'])}/{rep['image_id']}.jpg")
            rep_json = f"[json](assets/evidence/rank_compare/rank_{row['rank']}.json)"
        source_summary = ", ".join(f"{x['source']}:{x['count']}" for x in row.get("top_sources", [])[:3])
        rank_lines.append(
            f"|rank{row['rank']}|{row['count']}|{fmt(row['avg_final_score'])}|{fmt(row['avg_delta_vs_rank1'])}|"
            f"{fmt(row['avg_iou_vs_rank1']) if row['avg_iou_vs_rank1'] is not None else 'na'}|"
            f"{fmt(row['face_cut_worse_than_rank1_rate'])}|{fmt(row['joint_cut_worse_than_rank1_rate'])}|"
            f"{fmt(row['coverage_worse_than_rank1_rate'])}|{source_summary}|"
            f"{rep['image_id']} / {rep['target_ar'] if rep else ''}|"
            f"{f'[img]({teacher_viz})' if teacher_viz else ''}|{rep_json}|"
        )

    public_teacher_lines: List[str] = []
    for row in public_teacher_rows:
        rep = public_teacher_reps.get(str(row["teacher_id"]))
        teacher_viz = ""
        rep_json = ""
        rep_label = ""
        if rep:
            teacher_viz = sc_link(f"teacher/visualizations/teacher_scorer_{run_tag}/by_ar/{ar_token(rep['target_ar'])}/{rep['image_id']}.jpg")
            rep_json = f"[json](assets/evidence/public_teachers/{row['teacher_id']}.json)"
            rep_label = f"{rep['image_id']} / {rep['target_ar']}"
        public_teacher_lines.append(
            f"|{row['teacher_id']}|{row['raw_image_count']}|{fmt(row['avg_raw_num_proposals'],2)}|{row['tasks_total']}|"
            f"{fmt(row['top5_inclusion_rate'])}|{fmt(row['top1_selected_rate'])}|"
            f"{fmt(row['avg_top1_final_when_selected']) if row['avg_top1_final_when_selected'] is not None else 'na'}|"
            f"{fmt(row['avg_best_iou_to_selected_top1']) if row['avg_best_iou_to_selected_top1'] is not None else 'na'}|"
            f"{fmt(row['avg_candidate_top1_iou_to_teacher_seed'])}|{fmt(row['teacher_consensus_true_rate'])}|"
            f"{rep_label}|{f'[img]({teacher_viz})' if teacher_viz else ''}|{rep_json}|"
        )

    extra_notes = ""
    if missing_assets:
        missing_txt = ", ".join(sorted(set(missing_assets))[:20])
        extra_notes = f"- self-contained copy 시점에 누락된 외부 파일: `{missing_txt}`\n"

    text = f"""# {dataset_label} (RUN_TAG={run_tag}) 상세 분석 보고서

- 작성 기준: `{dataset_path}`의 `{run_tag}` 산출물
- 참고 설계 문서: `Implement_Docs/SSTK_v2_0_bundle/SSTK_Cropping_DataFactory_Reorganized_KO_v2_0.md`
- 리포트 포맷 레퍼런스: `data/SSTK/Test_100/artifacts/reports/260316_r2_detailed/REPORT_DRAFT_KO.md`

## 0) 입력 산출물/검증 요약

### 0.1 사용한 핵심 입력

- routed feats: `../../precompute/feats_c2c3c5_v2_strict_enriched_routed.jsonl`
- candidates: `../../candidates/candidates_ar_{run_tag}.jsonl`
- teacher scores: `../../teacher/scores/teacher_scores_ar_{run_tag}.jsonl`
- teacher overview: `../../teacher/overview/teacher_scores_overview_{run_tag}.json`
- teacher QA: `../../teacher/qa/teacher_scores_qa_report_{run_tag}.json`
- component viz: `../../precompute/visualizations/components_{run_tag}`
- teacher viz: `../../teacher/visualizations/teacher_scorer_{run_tag}`

### 0.2 정합성 체크

- 이미지 수: `{counts['num_images']}`
- AR 결과(top1 기준): `{counts['num_top1_ar_results']}` (평균 `{ar_per_image_text}` AR/image)
- Candidate overview: `num_images={cand_num_images}`, `proposal_injected_rate={fmt(cand_overview['proposal_injected_rate'])}`
- Teacher viz(보고서 샘플용): `tasks_selected={viz_overview['summary']['tasks_selected']}`, `rendered={viz_overview['summary']['rendered']}`, `missing={viz_overview['summary']['missing_images']}`
- real expensive 적용: `real_applied_images={real_applied_images}` / `{counts['num_images']}`

### 0.3 핵심 지표 스냅샷

|지표|값|근거|
|---|---:|---|
|candidate avg_total|{fmt(cand_overview['avg_total_candidates'],2)}|`../../candidates/candidates_ar_{run_tag}_overview.json`|
|proposal_injected_rate|{fmt(cand_overview['proposal_injected_rate'])}|same|
|keep_full/minimal_crop/crop|{fmt(decision_rates.get('keep_full', 0.0))} / {fmt(decision_rates.get('minimal_crop', 0.0))} / {fmt(decision_rates.get('crop', 0.0))}|`../../teacher/qa/teacher_scores_qa_report_{run_tag}.json`|
|face/head_top/joint cut rate|{fmt(p0['face_cut_rate'])} / {fmt(p0['head_top_cut_rate'])} / {fmt(p0['joint_cut_rate'])}|same|
|checklist_present_rate|{fmt(global_qa['explainability']['checklist_present_rate'])}|same|
|why_text_present_rate|{fmt(global_qa['explainability']['why_text_present_rate'])}|same|

### 0.4 해석 주의사항 및 P0 반영 메모

- 본 `{run_tag}` 런에서 `c4_ocr_meta.method = unavailable`은 오류가 아니라 **C4 OCR 미구현/비활성 상태**를 의미합니다. 따라서 텍스트 보존 관련 checklist가 `text_na` 위주로 나타나는 것은 현재 산출물 기준으로 정상입니다.
- 최신 P0 규칙에 따라 C4-backed text evidence가 없으면 `text_document` hard routing이 금지되며, 현재 `{dataset_label}` routed output의 `text_document` count는 `{text_document_count}`입니다.
- teacher visualization은 `subject bbox + C2 segmentation mask overlay`를 함께 표시하도록 렌더링합니다.
- 이번 `{run_tag}` 런의 expensive stage는 `use_real_expensive={use_real_expensive}` 설정으로 기록되며, 실제 적용 이미지는 `{real_applied_images}`개입니다.
- 서버 rerun 로그 기준 `save_public_teacher_ref_eval=1`로 수행됐고, main teacher jsonl에는 public teacher reference box의 exact eval이 저장됩니다. single-image drill-down 리포트는 이 exact eval을 우선 사용합니다.

### 0.5 P0 QA 상단 요약

|mode|count(AR task)|num_person mean|num_c2_instances mean|num_effective_subjects mean|primary_subject_exists_rate|face_cut|joint_cut|head_top_cut|portrait w/o human|scene horizon_na|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(mode_qa_lines)}

- global P0 KPI: `face_cut_rate={fmt(p0['face_cut_rate'])}`, `head_top_cut_rate={fmt(p0['head_top_cut_rate'])}`, `joint_cut_rate={fmt(p0['joint_cut_rate'])}`, `text_document_ocr_unavailable_rate={fmt(p0['text_document_ocr_unavailable_rate'])}`, `portrait_route_without_human_rate={fmt(p0['portrait_route_without_human_rate'])}`, `scene_horizon_na_rate={fmt(p0['scene_horizon_na_rate'])}`

## 1) 요구사항 1: Subject 2개 이상/1개/0개 샘플 + 분류 근거

### 1.1 분류 기준(명시)

- 본 보고서의 subject 개수 기준은 `routing.subject_set.num_person` (인물 수)입니다.
- 참고로 `routing.subject_set.num_c2_instances`(C2 전체 인스턴스 수), `routing.subject_set.num_effective_subjects`(정책상 유효 주피사체 수)도 같이 봐야 합니다.
- 분포(num_person): `>=2: {counts['subject_count_person'].get('ge2', 0)}`, `=1: {counts['subject_count_person'].get('eq1', 0)}`, `=0: {counts['subject_count_person'].get('eq0', 0)}`
- 보조 분포(num_c2_instances): `>=2: {counts['subject_count_instances'].get('ge2', 0)}`, `=1: {counts['subject_count_instances'].get('eq1', 0)}`, `=0: {counts['subject_count_instances'].get('eq0', 0)}`
- 보조 분포(num_effective_subjects): `>=2: {counts['subject_count_effective'].get('ge2', 0)}`, `=1: {counts['subject_count_effective'].get('eq1', 0)}`, `=0: {counts['subject_count_effective'].get('eq0', 0)}`

- 상세 근거 JSON: `assets/evidence/subject_count_examples_{run_tag}.json`

### 1.2 카테고리별 대표 샘플(각 3개)

|bucket|image_id|mode(conf)|num_person / num_c2_instances / num_effective_subjects|분류 근거(요약)|이미지|근거 JSON|
|---|---|---|---:|---|---|---|
{chr(10).join(subject_bucket_lines)}

### 1.3 카테고리별 대표 시각 샘플(요구사항 이미지 첨부)

- `>=2` 대표: `sstk_image_149449811`
  - 원본: ![]({sc_link(f'precompute/visualizations/components_{run_tag}/original/sstk_image_149449811.jpg')})
  - precompute combined: ![]({sc_link(f'precompute/visualizations/components_{run_tag}/combined_all/sstk_image_149449811.jpg')})
  - C3 pose: ![]({sc_link(f'precompute/visualizations/components_{run_tag}/c3_pose/sstk_image_149449811.jpg')})
  - C6 gaze: ![]({sc_link(f'precompute/visualizations/components_{run_tag}/c6_gaze/sstk_image_149449811.jpg')})
- `=1` 대표: `bigstock_image_188903926`
  - 원본: ![]({sc_link(f'precompute/visualizations/components_{run_tag}/original/bigstock_image_188903926.jpg')})
  - precompute combined: ![]({sc_link(f'precompute/visualizations/components_{run_tag}/combined_all/bigstock_image_188903926.jpg')})
  - C3 pose: ![]({sc_link(f'precompute/visualizations/components_{run_tag}/c3_pose/bigstock_image_188903926.jpg')})
  - C6 gaze: ![]({sc_link(f'precompute/visualizations/components_{run_tag}/c6_gaze/bigstock_image_188903926.jpg')})
- `=0` 대표: `bigstock_image_25120745`
  - 원본: ![]({sc_link(f'precompute/visualizations/components_{run_tag}/original/bigstock_image_25120745.jpg')})
  - precompute combined: ![]({sc_link(f'precompute/visualizations/components_{run_tag}/combined_all/bigstock_image_25120745.jpg')})
  - C3 pose: ![]({sc_link(f'precompute/visualizations/components_{run_tag}/c3_pose/bigstock_image_25120745.jpg')})
  - C6 gaze: ![]({sc_link(f'precompute/visualizations/components_{run_tag}/c6_gaze/bigstock_image_25120745.jpg')})

## 2) 요구사항 2: subject_mode별 샘플 + 분류 근거

- mode 분포 근거: `assets/analytics/summary_{run_tag}.json`
- mode 샘플 상세 근거(JSON): `assets/evidence/subject_mode_examples_{run_tag}.json`
- `text_document`는 P0 rerun 이후 `0건`입니다. OCR 미구현 상태에서는 hard routing으로 승격되지 않도록 막았습니다.

|subject_mode|count|대표 image_id|num_person / num_c2_instances / num_effective_subjects|mode_conf|분류 근거(reasons + rule)|대표 이미지|근거 JSON|
|---|---:|---|---:|---:|---|---|---|
{chr(10).join(mode_lines)}

## 2.1) subject-mode routing/checklist schema 보완 반영

- `scene_landscape` 단일 mode는 해체되고, 실제 산출물은 `scene_general + scene_subtype`로 기록됩니다.
- `subject_scale`와 `context`는 분리 라벨로 저장되며, `copyspace`와 `roll`도 별도 checklist label로 집계됩니다.
- `horizon`은 현재 detector 품질 문제로 전부 `horizon_na`지만, 상태 필드는 유지됩니다.

### 2.1.1 scene_general subtype 분해

|scene_subtype|count|rate|
|---|---:|---:|
{chr(10).join(scene_subtype_lines)}

### 2.1.2 selected top1 checklist snapshot

|metric|label|count|rate|
|---|---|---:|---:|
{chr(10).join(checklist_snapshot_lines)}

## 3) 요구사항 3: 대표 크롭 후보의 Teacher scoring + Why-tags/Checklist/Explainability

- 대표 후보 상세 근거(JSON): `assets/evidence/teacher_representative_examples_{run_tag}.json`
- 아래 표는 `selected_topk[0]` 기준이며, score component/체크리스트/설명 텍스트를 모두 포함합니다.

|대표케이스|image_id/ar|decision|candidate(source)|score(cheap/exp/rank/policy/legacy)|핵심 component(cov,p_cut,r_teach,A_norm,cos)|macro(A/S/C/T)|체크리스트 핵심 라벨|why_tags(상위)|이미지|근거 JSON|
|---|---|---|---|---|---|---|---|---|---|---|
{chr(10).join(rep_lines)}

## 4) 요구사항 4: checklist 각 항목/결과(label)별 대표 후보 + 이미지

- 전체 표 CSV: `assets/analytics/checklist_label_examples_{run_tag}.csv`
- 전체 표 MD: `assets/analytics/checklist_label_examples_{run_tag}.md`
- 아래는 metric-label 전체({len(checklist_rows)}개)에 대해 대표 예시를 1개씩 매핑한 결과입니다.

|metric|label|count|대표(image_id/ar)|decision|policy/rank/legacy|대표 이미지|근거 JSON|
|---|---|---:|---|---|---:|---|---|
{chr(10).join(checklist_lines)}

## 5) rank2~rank5 비교 리포트

- 집계 근거 JSON: `assets/analytics/topk_rank_comparison_{run_tag}.json`
- 비교 기준은 각 AR task의 `selected_topk[0]`을 기준으로 한 `selected_topk` 내부 순서 2~5의 평균 점수 차이, bbox IoU, structural risk 악화율입니다.
- 해석:
  - 여기서 `rank1~rank5`는 strict final-score rank가 아니라 `selected_topk`에 저장된 selection order입니다. force/diversity/fill 단계가 반영되므로 lower rank의 final score가 더 높아도 정상입니다.
  - `avg_delta_vs_rank1`은 selection order 1 대비 final score 차이입니다. 음수 절대값이 작을수록 대체안으로 가깝습니다.
  - `avg_iou_vs_rank1`은 rank1 bbox와의 평균 IoU입니다. 높을수록 비슷한 crop입니다.
  - `face/joint/coverage worse`는 rank1 대비 checklist label이 악화된 비율입니다.

|rank|count|avg_final|avg_delta_vs_rank1|avg_iou_vs_rank1|face worse|joint worse|coverage worse|top sources|대표(image_id/ar)|대표 이미지|근거 JSON|
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
{chr(10).join(rank_lines)}

- 요약:
  - rank2는 평균 IoU가 가장 높고(`{fmt(rank_rows[0]['avg_iou_vs_rank1']) if rank_rows else 'na'}`), rank1과 가장 가까운 대체안입니다.
  - rank가 내려갈수록 `avg_delta_vs_rank1`는 더 낮아지고 IoU도 감소합니다. 즉, score와 geometry 모두 rank1에서 멀어지는 방향입니다.
  - source 분포는 rank2부터 `teacher:jitter`, `grid`, `baseline_maxarea_slide` 비중이 커집니다. rank1의 보수 baseline 중심 선택과 대비됩니다.

## 6) public_teachers (gaic/cacnet/cgs) 비교 리포트

- 집계 근거 JSON: `assets/analytics/public_teacher_comparison_{run_tag}.json`
- `avg_best_iou_to_selected_top1`은 최종 top1 crop과 각 public teacher의 free-form raw proposal 간 최대 IoU 평균입니다.
- `avg_candidate_top1_iou_to_teacher_seed`는 pipeline의 proposal injection debug에 남은 teacher seed alignment 평균입니다.
- 본 섹션의 집계는 여전히 전체 run의 top1 정합/채택률 중심 요약입니다. per-image exact public teacher score는 main teacher jsonl의 `proposal_injection.public_teacher_ref_eval` 또는 single-image drill-down 리포트에서 확인하는 것이 맞습니다.

|teacher|raw images|avg raw proposals|tasks|top5 inclusion|top1 selected|avg final when selected|avg best IoU to final top1|avg seed IoU|consensus true rate|대표(image_id/ar)|대표 이미지|근거 JSON|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
{chr(10).join(public_teacher_lines)}

- 요약:
  - public teacher raw proposal 수 요약: `{public_teacher_prop_summary}`.
  - 최종 top1 채택률은 `jitter`를 제외하면 public teacher source가 낮습니다. 이 run 기준 top1 직접 채택 건수는 `{public_teacher_top1_summary}`입니다.
  - 그럼에도 teacher seed alignment 평균은 높게 유지됩니다. 즉, public teacher proposal은 직접 top1로 선택되기보다 baseline/grid/jitter 후보 생성과 consensus shaping에 더 크게 기여합니다.

## 6.1) FinalScore macro redesign 반영 스냅샷

- 참고 세부 보고서: [FINALSCORE_MACRO_MIGRATION_REPORT_KO.md](FINALSCORE_MACRO_MIGRATION_REPORT_KO.md)
- 이번 top-level 보고서는 `score_rank`, `score_policy`, `final_legacy`, `area_log_prior`, `macro_scores(A/S/C/T)`를 함께 반영합니다.

|metric|value|
|---|---|
{chr(10).join(macro_summary_lines)}

## 6.2) 학습용 라벨 생성 결과

- 별도 산출물:
  - training label report: {f"[TRAINING_DATA_REPORT_KO.md]({training_report_link})" if training_report_link else "`missing`"}
  - implementation report: {f"[FINALSCORE_TRAINING_DATA_IMPLEMENTATION_REPORT_KO.md]({training_impl_link})" if training_impl_link else "`missing`"}
  - qa summary: {f"[qa_summary.json]({training_qa_link})" if training_qa_link else "`missing`"}

|metric|value|
|---|---|
{chr(10).join(training_label_lines) if training_label_lines else "|training_labels|not_generated|"}

### 6.2.1 training-label 대표 케이스

|kind|image_id/ar|mode|decision|best_rank_candidate|best_policy_candidate|best_rank_score|best_policy_score|
|---|---|---|---|---|---|---:|---:|
{chr(10).join(training_example_lines) if training_example_lines else "|na|na|na|na|na|na|na|na|"}

## 7) Drill-down report package + 시각화 QA

- index: [DRILLDOWN_INDEX.md](DRILLDOWN_INDEX.md)
- layout QA summary: [drilldown_layout_qa.json](assets/analytics/drilldown_layout_qa.json)
- 각 single-image report는 `legacy -> rank -> policy` score ladder, `A/S/C/T` macro bars, `area_log_prior` 반영 테이블, `layout_qa.json`을 포함합니다.

|metric|value|
|---|---|
{chr(10).join(drilldown_summary_lines) if drilldown_summary_lines else "|drilldown|not_generated|"}

## 8) 추가 산출물/코드 수행 여부 (요청사항 대응)

요구사항을 충족하기 위해 아래 추가 산출물을 생성했습니다.

- summary: `assets/analytics/summary_{run_tag}.json`
- subject_count evidence: `assets/evidence/subject_count_examples_{run_tag}.json` + `assets/evidence/subject_count`
- subject_mode evidence: `assets/evidence/subject_mode_examples_{run_tag}.json` + `assets/evidence/subject_mode`
- representative teacher cases: `assets/evidence/teacher_representative_examples_{run_tag}.json` + `assets/evidence/teacher_rep`
- checklist per-label mapping: `assets/analytics/checklist_label_examples_{run_tag}.json` + `assets/evidence/checklist`
- rank2~rank5 comparison: `assets/analytics/topk_rank_comparison_{run_tag}.json` + `assets/evidence/rank_compare`
- public teacher comparison: `assets/analytics/public_teacher_comparison_{run_tag}.json` + `assets/evidence/public_teachers`
- training label examples: `assets/analytics/training_label_examples_{run_tag}.json`
- report image id list: `assets/analytics/report_example_image_ids_{run_tag}.txt`
- drilldown index: `DRILLDOWN_INDEX.md`
- drilldown layout qa: `assets/analytics/drilldown_layout_qa.json`
- self-contained bundle manifest: `assets/self_contained/self_contained_manifest.json`

또한 보고서 링크의 외부 이미지가 누락되지 않도록, 보고서가 참조하는 이미지들을 `assets/self_contained/external_links/...` 아래로 복사하고 링크를 내부 경로로 재작성했습니다.
{extra_notes}

## 9) 한계 및 후속 권장

- 현재 run에서는 C4 OCR이 의도적으로 `unavailable`(미구현/비활성)로 들어가 `text_keep_ratio`가 전부 `text_na`입니다. 이는 현 산출물 기준 정상 동작이며, 텍스트 보존 항목의 실측 검증은 C4 구현/활성화 이후 rerun이 필요합니다.
- 현재 run에서는 horizon checklist가 전부 `horizon_na`입니다(`scene_horizon_na_rate = {fmt(p0['scene_horizon_na_rate'])}`). C5 임계/규칙 조정 A/B를 별도 권장합니다.
- portrait 구조 리스크는 여전히 남아 있습니다. `face_cut_rate = {fmt(p0['face_cut_rate'])}`, `head_top_cut_rate = {fmt(p0['head_top_cut_rate'])}`, `joint_cut_rate = {fmt(p0['joint_cut_rate'])}`.
- rank 비교를 넣었지만 현재는 `selected_topk` 내부의 selection order만 본 것입니다. top5 밖 후보까지 포함한 full candidate margin 분석과 strict score-rank 분석은 별도 산출이 필요합니다.
- public teacher 비교는 free-form raw proposal과 최종 top1 간 정합 위주입니다. teacher별 AR-specific projection 품질 비교는 `teacher_proposals_public_{run_tag}.jsonl`을 함께 보는 후속 분석이 맞습니다.
"""
    return text


def main() -> None:
    args = parse_args()
    progress_enabled = bool(int(args.progress))
    progress_every = max(1, int(args.progress_every))
    progress_min_seconds = max(0.0, float(args.progress_min_seconds))
    project_root = Path.cwd()
    data_root = Path(args.data_root)
    report_dir = Path(args.report_dir)
    ensure_dir(report_dir)
    ensure_dir(report_dir / "assets")
    progress_log(
        f"build_sstk_detailed_report: start | run_tag={args.run_tag} | report_dir={report_dir}",
        enabled=progress_enabled,
    )

    run_tag = str(args.run_tag)
    artifacts = data_root / "artifacts"
    teacher_jsonl = resolve_teacher_jsonl(args, artifacts, run_tag)
    qa_json = artifacts / "teacher" / "qa" / f"teacher_scores_qa_report_{run_tag}.json"
    overview_json = artifacts / "teacher" / "overview" / f"teacher_scores_overview_{run_tag}.json"
    candidates_jsonl = resolve_candidates_jsonl(args, artifacts, run_tag)
    cand_overview_json = artifacts / "candidates" / f"candidates_ar_{run_tag}_overview.json"
    viz_overview_json = artifacts / "teacher" / "visualizations" / f"teacher_scorer_{run_tag}" / "viz_overview.json"
    routed_feats = resolve_routed_feats_jsonl(args, artifacts)
    training_labels_dir = resolve_training_labels_dir(args, artifacts, run_tag)
    progress_log(
        "build_sstk_detailed_report: inputs | "
        f"teacher_jsonl={teacher_jsonl} | routed_feats={routed_feats} | candidates_jsonl={candidates_jsonl} | "
        f"training_labels_dir={training_labels_dir}",
        enabled=progress_enabled,
    )

    summary_json = report_dir / "assets" / "analytics" / f"summary_{run_tag}.json"
    subject_count_json = report_dir / "assets" / "evidence" / f"subject_count_examples_{run_tag}.json"
    subject_mode_json = report_dir / "assets" / "evidence" / f"subject_mode_examples_{run_tag}.json"

    summary = read_json(summary_json)
    subject_count_examples = read_json(subject_count_json)
    subject_mode_examples = read_json(subject_mode_json)
    qa = read_json(qa_json)
    overview = read_json(overview_json)
    cand_overview = read_json(cand_overview_json)
    viz_overview = read_json(viz_overview_json)
    teacher_rows = load_teacher_rows(
        teacher_jsonl,
        progress=progress_enabled,
        progress_every=progress_every,
        progress_min_seconds=progress_min_seconds,
    )
    routed_map = build_routed_map(
        routed_feats,
        progress=progress_enabled,
        progress_every=progress_every,
        progress_min_seconds=progress_min_seconds,
    )
    progress_log("build_sstk_detailed_report: build_flat_examples", enabled=progress_enabled)
    flat_examples = build_flat_examples(teacher_rows, routed_map, str(data_root).replace("\\", "/"), run_tag)
    progress_log("build_sstk_detailed_report: summarize assets", enabled=progress_enabled)
    score_migration_summary = summarize_score_migration(teacher_rows)
    scene_subtype_counts: Counter[str] = Counter()
    for routed in routed_map.values():
        routing = routed.get("routing", {}) if isinstance(routed.get("routing"), dict) else {}
        if str(routing.get("subject_mode", "")) == "scene_general":
            scene_subtype_counts[str(routing.get("scene_subtype", "na"))] += 1

    teacher_rep, rep_ids = build_teacher_rep_assets(flat_examples, report_dir, run_tag)
    checklist_rows, checklist_ids = build_checklist_assets(flat_examples, report_dir, run_tag)
    progress_log("build_sstk_detailed_report: build rank/public teacher assets", enabled=progress_enabled)
    rank_rows, rank_reps, rank_ids = build_rank_comparison_assets(
        teacher_rows=teacher_rows,
        report_dir=report_dir,
        run_tag=run_tag,
        data_root=str(data_root).replace("\\", "/"),
    )
    public_teacher_rows, public_teacher_reps, public_teacher_ids = build_public_teacher_comparison_assets(
        teacher_rows=teacher_rows,
        public_raw_path=artifacts / "public_teachers" / "raw" / f"teacher_raw_public_{run_tag}.jsonl",
        report_dir=report_dir,
        run_tag=run_tag,
        data_root=str(data_root).replace("\\", "/"),
    )
    training_labels_summary, training_label_examples, training_extra_ids = summarize_training_labels(
        training_labels_dir,
        run_tag,
        report_dir,
        progress=progress_enabled,
        progress_every=progress_every,
        progress_min_seconds=progress_min_seconds,
    )

    subject_ids: List[str] = []
    for rows in subject_count_examples.values():
        subject_ids.extend([str(row.get("image_id", "")) for row in rows])
    for rows in subject_mode_examples.values():
        subject_ids.extend([str(row.get("image_id", "")) for row in rows])
    fallback_example_id = next(iter(routed_map.keys()), "")
    example_ids = unique_keep_order(subject_ids + rep_ids + checklist_ids + rank_ids + public_teacher_ids + training_extra_ids + ([fallback_example_id] if fallback_example_id else []))
    example_ids = example_ids[:64]
    analytics_dir = report_dir / "assets" / "analytics"
    ensure_dir(analytics_dir)
    (analytics_dir / f"report_example_image_ids_{run_tag}.txt").write_text("\n".join(example_ids) + "\n", encoding="utf-8")

    drilldown_summary = generate_drilldown_reports(
        run_tag=run_tag,
        data_root=data_root,
        report_dir=report_dir,
        image_ids=example_ids,
        routed_map=routed_map,
        teacher_rows=teacher_rows,
        teacher_jsonl=teacher_jsonl,
        candidates_jsonl=candidates_jsonl,
        features_jsonl=routed_feats,
        num_workers=resolve_num_workers(int(args.num_workers), len(example_ids)),
        progress=progress_enabled,
        progress_every=max(1, progress_every // 10),
        progress_min_seconds=progress_min_seconds,
    )

    mappings: List[Dict[str, str]] = []
    teacher_viz_fallback_dir = Path(args.teacher_viz_fallback_dir) if args.teacher_viz_fallback_dir else None
    report_text = build_report_text(
        run_tag=run_tag,
        subject_count_examples=subject_count_examples,
        subject_mode_examples=subject_mode_examples,
        teacher_rep=teacher_rep,
        checklist_rows=checklist_rows,
        rank_rows=rank_rows,
        rank_reps=rank_reps,
        public_teacher_rows=public_teacher_rows,
        public_teacher_reps=public_teacher_reps,
        summary=summary,
        scene_subtype_counts=dict(scene_subtype_counts),
        qa=qa,
        overview=overview,
        cand_overview=cand_overview,
        viz_overview=viz_overview,
        score_migration_summary=score_migration_summary,
        training_labels_summary=training_labels_summary,
        training_label_examples=training_label_examples,
        drilldown_summary=drilldown_summary,
        mappings=mappings,
        report_dir=report_dir,
        project_root=project_root,
        data_root=data_root,
        teacher_viz_fallback_dir=teacher_viz_fallback_dir,
    )
    report_path = report_dir / "REPORT_DRAFT_KO.md"
    report_path.write_text(report_text, encoding="utf-8")

    manifest = {
        "report": str(report_path),
        "package_dir": str(report_dir),
        "bundle_root": str(report_dir / "assets" / "self_contained"),
        "external_links_copied_count": len({m["rewritten_link"] for m in mappings}),
        "link_rewrite_count": len({m["rewritten_link"] for m in mappings}),
        "mappings": mappings,
    }
    ensure_dir(report_dir / "assets" / "self_contained")
    write_json(report_dir / "assets" / "self_contained" / "self_contained_manifest.json", manifest)
    zip_path = build_report_zip(report_dir)
    overview_zip_path = build_overview_zip(report_dir)
    progress_log(
        f"build_sstk_detailed_report: finished | example_ids={len(example_ids)} | drilldown_reports={drilldown_summary['total_reports']}",
        enabled=progress_enabled,
    )

    print(f"[done] report={report_path}")
    print(f"[done] drilldown_index={report_dir / 'DRILLDOWN_INDEX.md'}")
    print(f"[done] drilldown_layout_qa={analytics_dir / 'drilldown_layout_qa.json'}")
    print(f"[done] teacher_rep={report_dir / 'assets' / 'evidence' / f'teacher_representative_examples_{run_tag}.json'}")
    print(f"[done] checklist={report_dir / 'assets' / 'analytics' / f'checklist_label_examples_{run_tag}.json'}")
    print(f"[done] rank_compare={analytics_dir / f'topk_rank_comparison_{run_tag}.json'}")
    print(f"[done] public_teacher_compare={analytics_dir / f'public_teacher_comparison_{run_tag}.json'}")
    print(f"[done] training_label_examples={analytics_dir / f'training_label_examples_{run_tag}.json'}")
    print(f"[done] example_ids={analytics_dir / f'report_example_image_ids_{run_tag}.txt'}")
    print(f"[done] manifest={report_dir / 'assets' / 'self_contained' / 'self_contained_manifest.json'}")
    print(f"[done] zip={zip_path}")
    print(f"[done] overview_zip={overview_zip_path}")


if __name__ == "__main__":
    main()
