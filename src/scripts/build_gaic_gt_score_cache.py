#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from score_teacher import compute_candidate_scores
from debug_viz_detail_utils import extract_raw_candidate_detail
from score_profile_utils import apply_profile_to_cfg, build_profile_metadata
from scripts.build_finalscore_training_data import (
    apply_training_safe_policy_score,
    dedupe_candidates,
    rank_pct_desc,
    robust_z_scores,
    safe_dict,
    safe_float,
    safe_list,
    safe_policy_raw_score,
    sigmoid,
    softmax_local,
)
from scripts.run_gaic_benchmark_eval import (
    SampleExpensiveCacheManager,
    build_gt_candidates,
    build_scoring_context,
    load_gaic_annotations,
    load_scorer_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GAIC GT FREE-context training score cache for debug visualization.")
    parser.add_argument("--candidates_jsonl", required=True)
    parser.add_argument("--features_jsonl", required=True)
    parser.add_argument("--teacher_jsonl", required=True)
    parser.add_argument("--gaic_train_json", required=True)
    parser.add_argument("--gaic_test_json", required=True)
    parser.add_argument("--out_jsonl", required=True)
    parser.add_argument(
        "--image_ids_csv",
        default="",
        help="Optional CSV containing image_id column. Useful for selected debug-viz subsets.",
    )
    parser.add_argument("--target_ar", default="FREE", help="Currently only FREE is supported.")
    parser.add_argument("--softmax_tau", type=float, default=0.2)
    parser.add_argument(
        "--c1_jsonl",
        default="",
        help="Optional C1 embedding JSONL. When set with --image_root, GT boxes also receive real expensive-stage scores including A_macro.",
    )
    parser.add_argument(
        "--image_root",
        default="",
        help="Optional image root used together with --c1_jsonl to compute GT expensive-stage signals.",
    )
    parser.add_argument(
        "--sample_expensive_cache_jsonl",
        default="",
        help="Optional cache for GT expensive-stage signals. Useful for server-side precompute and local reuse.",
    )
    parser.add_argument("--score_profile", default="single_stage2")
    parser.add_argument("--score_profile_overrides_json", default="")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--align_device", default="auto")
    parser.add_argument("--aesthetic_device", default="auto")
    parser.add_argument("--exp_batch_size", type=int, default=24)
    parser.add_argument("--exp_preprocess_workers", type=int, default=0)
    parser.add_argument("--exp_pin_memory", type=int, default=1)
    return parser.parse_args()


def stable_shard_index(text: str, num_shards: int) -> int:
    n = max(1, int(num_shards))
    if n == 1:
        return 0
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return int(digest, 16) % n


def read_jsonl_selected(path: Path, selected_ids: Optional[Set[str]]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    pending = set(selected_ids) if selected_ids else None
    with path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"malformed jsonl: path={path} line={line_idx} col={exc.colno} char={exc.pos}"
                ) from exc
            image_id = str(row.get("image_id", "")).strip()
            if not image_id:
                continue
            if pending is not None and image_id not in pending:
                continue
            result[image_id] = row
            if pending is not None:
                pending.discard(image_id)
                if not pending:
                    break
    return result


def load_teacher_map_with_fallback(path: Path, selected_ids: Optional[Set[str]]) -> tuple[Dict[str, Dict[str, Any]], Path]:
    try:
        return read_jsonl_selected(path, selected_ids), path
    except RuntimeError:
        if path.name.endswith("_monotonic.jsonl"):
            raise
        fallback = path.with_name(f"{path.stem}_monotonic{path.suffix}")
        if not fallback.exists():
            raise
        return read_jsonl_selected(fallback, selected_ids), fallback


def load_image_ids_from_csv(path: Path) -> Set[str]:
    image_ids: Set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_id = str(row.get("image_id", "")).strip()
            if image_id:
                image_ids.add(image_id)
    return image_ids


def find_image_path(image_root: Path, image_id: str) -> Optional[Path]:
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        candidate = image_root / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def attach_training_normalization(
    candidates: Sequence[Dict[str, Any]],
    *,
    target_ar: str,
    softmax_tau: float,
    routing: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    prepared = [
        apply_training_safe_policy_score(candidate, target_ar=target_ar, routing=routing)
        for candidate in candidates
    ]
    raw_scores = [safe_policy_raw_score(candidate, target_ar=target_ar) for candidate in prepared]
    rank_pcts = rank_pct_desc(raw_scores)
    z_scores = robust_z_scores(raw_scores)
    softmax_scores = softmax_local(raw_scores, tau=softmax_tau)
    out: List[Dict[str, Any]] = []
    for idx, candidate in enumerate(prepared):
        row = copy.deepcopy(candidate)
        row["score_rank_pct"] = float(rank_pcts[idx])
        row["score_z_local"] = float(z_scores[idx])
        row["score_sigmoid_z_local"] = float(sigmoid(z_scores[idx]))
        row["score_softmax_local"] = float(softmax_scores[idx])
        row["score_policy_rank_pct"] = float(rank_pcts[idx])
        row["score_policy_z_local"] = float(z_scores[idx])
        row["score_policy_sigmoid_z_local"] = float(sigmoid(z_scores[idx]))
        row["score_policy_softmax_local"] = float(softmax_scores[idx])
        out.append(row)
    return out


def serialize_gt_row(row: Dict[str, Any], ann: Dict[str, Any]) -> Dict[str, Any]:
    scores = safe_dict(row.get("scores"))
    detail = extract_raw_candidate_detail(row)
    detail.update(
        {
        "ann_id": int(ann.get("id", 0)),
        "mos": round(safe_float(ann.get("score", 0.0), 0.0), 6),
        "bbox_norm_xyxy": [round(float(v), 6) for v in row.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0])],
        "score_rank": round(safe_float(scores.get("rank", row.get("score_rank", 0.0)), 0.0), 9),
        "score_policy": round(safe_float(scores.get("policy", row.get("score_policy", 0.0)), 0.0), 9),
        "score_policy_sigmoid_z_local": round(safe_float(row.get("score_policy_sigmoid_z_local", 0.0), 0.0), 9),
        "training_score_prob": round(safe_float(row.get("score_policy_sigmoid_z_local", 0.0), 0.0), 9),
        "score_rank_pct": round(safe_float(row.get("score_rank_pct", 0.0), 0.0), 9),
        "score_policy_z_local": round(safe_float(row.get("score_policy_z_local", 0.0), 0.0), 9),
        "hard_reject": bool(row.get("hard_reject", False)),
        "reject_tags": list(row.get("reject_tags", [])),
        "why_tags": list(row.get("why_tags", [])),
        }
    )
    return detail


def main() -> None:
    args = parse_args()
    target_ar = str(args.target_ar).strip().upper()
    if target_ar != "FREE":
        raise SystemExit("build_gaic_gt_score_cache.py currently supports only --target_ar FREE")
    if int(args.num_shards) < 1:
        raise SystemExit(f"--num_shards must be >= 1 (got {args.num_shards})")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
        raise SystemExit(
            f"--shard_index must satisfy 0 <= shard_index < num_shards "
            f"(got shard_index={args.shard_index}, num_shards={args.num_shards})"
        )

    selected_ids = load_image_ids_from_csv(Path(args.image_ids_csv)) if str(args.image_ids_csv).strip() else None
    candidates_map = read_jsonl_selected(Path(args.candidates_jsonl), selected_ids)
    features_map = read_jsonl_selected(Path(args.features_jsonl), selected_ids)
    teacher_map, resolved_teacher_jsonl = load_teacher_map_with_fallback(Path(args.teacher_jsonl), selected_ids)
    gt_images, gt_anns = load_gaic_annotations(Path(args.gaic_train_json), Path(args.gaic_test_json))
    cfg = load_scorer_config(resolved_teacher_jsonl)
    cfg, profile_spec = apply_profile_to_cfg(
        cfg,
        profile_name=str(args.score_profile),
        override_json_path=str(args.score_profile_overrides_json),
    )
    profile_metadata = build_profile_metadata(cfg, profile_spec)
    image_root = Path(args.image_root) if str(args.image_root).strip() else None
    c1_jsonl = Path(args.c1_jsonl) if str(args.c1_jsonl).strip() else None
    expensive_cache_path = (
        Path(args.sample_expensive_cache_jsonl)
        if str(args.sample_expensive_cache_jsonl).strip()
        else Path(args.out_jsonl).with_name(f"{Path(args.out_jsonl).stem}_expensive_cache.jsonl")
    )
    expensive_manager: Optional[SampleExpensiveCacheManager] = None
    expensive_images_used = 0
    expensive_images_unavailable = 0
    if c1_jsonl is not None and image_root is not None:
        expensive_manager = SampleExpensiveCacheManager(
            cfg=cfg,
            c1_jsonl=c1_jsonl,
            cache_path=expensive_cache_path,
            align_device=str(args.align_device),
            aesthetic_device=str(args.aesthetic_device),
            exp_batch_size=int(args.exp_batch_size),
            exp_preprocess_workers=int(args.exp_preprocess_workers),
            exp_pin_memory=bool(int(args.exp_pin_memory) > 0),
        )

    overlap_ids = sorted(set(candidates_map.keys()) & set(features_map.keys()) & set(teacher_map.keys()) & set(gt_images.keys()) & set(gt_anns.keys()))
    if int(args.num_shards) > 1:
        overlap_ids = [
            image_id
            for image_id in overlap_ids
            if stable_shard_index(str(image_id), int(args.num_shards)) == int(args.shard_index)
        ]
    out_rows: List[Dict[str, Any]] = []

    try:
        for image_id in overlap_ids:
            cand_rec = candidates_map[image_id]
            feat_rec = features_map[image_id]
            teacher_rec = teacher_map[image_id]
            ar_res = safe_dict(safe_dict(safe_dict(teacher_rec.get("teacher_scorer")).get("results_by_ar")).get("FREE"))
            if not ar_res:
                continue

            width = int(gt_images[image_id]["width"])
            height = int(gt_images[image_id]["height"])
            gt_ann_rows = list(gt_anns[image_id])
            gt_candidates = build_gt_candidates(gt_ann_rows, width=width, height=height)
            reference_candidates = dedupe_candidates(ar_res)

            context = build_scoring_context(cand_rec=cand_rec, feat_rec=feat_rec, cfg=cfg)
            scored_gt_candidates: List[Dict[str, Any]] = []
            for candidate in gt_candidates:
                scored_gt_candidates.append(
                    compute_candidate_scores(
                        candidate=copy.deepcopy(candidate),
                        target_ar=None,
                        image_ar=context["image_ar"],
                        subject_box=context["subject_box"],
                        subject_centroid=context["subject_centroid"],
                        c3_info=context["c3_info"],
                        c5_info=context["c5_info"],
                        c4_info=context["c4_info"],
                        route=context["route"],
                        teacher_ctx=context["teacher_ctx"],
                        cfg=cfg,
                    )
                )

            expensive_status = "disabled"
            if expensive_manager is not None and image_root is not None:
                image_path = find_image_path(image_root, image_id)
                if image_path is not None:
                    scored_gt_candidates, expensive_status = expensive_manager.apply(
                        image_id=image_id,
                        image_path=image_path,
                        rows=scored_gt_candidates,
                        route=context["route"],
                        target_ar_value=None,
                    )
                else:
                    expensive_status = "missing_image"
                if expensive_status in {"cache", "computed"}:
                    expensive_images_used += 1
                elif expensive_status != "disabled":
                    expensive_images_unavailable += 1

            combined = attach_training_normalization(
                list(reference_candidates) + list(scored_gt_candidates),
                target_ar="FREE",
                softmax_tau=float(args.softmax_tau),
                routing=safe_dict(ar_res.get("routing")),
            )
            gt_count = len(scored_gt_candidates)
            if gt_count <= 0:
                continue
            gt_scored_rows = combined[-gt_count:]
            serialized_gt_rows = [
                serialize_gt_row(row, ann)
                for row, ann in zip(gt_scored_rows, gt_ann_rows)
            ]
            out_rows.append(
                {
                    "image_id": image_id,
                    "target_ar": "FREE",
                    "scoring_context": "training_label_deduped_teacher_subset",
                    "reference_candidate_count": len(reference_candidates),
                    "gt_count": len(serialized_gt_rows),
                    "gt_scores_expensive_status": expensive_status,
                    "gt_scores": serialized_gt_rows,
                }
            )
    finally:
        if expensive_manager is not None:
            expensive_manager.flush()
            expensive_manager.close()

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "status": "ok",
        "out_jsonl": str(out_path),
        "target_ar": "FREE",
        "image_count": len(out_rows),
        "selected_image_count": len(selected_ids) if selected_ids is not None else len(overlap_ids),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "scoring_context": "training_label_deduped_teacher_subset",
        "teacher_jsonl_resolved": str(resolved_teacher_jsonl),
        "score_profile": profile_metadata,
        "gt_expensive_stage": {
            "enabled": bool(expensive_manager is not None),
            "c1_jsonl": str(c1_jsonl) if c1_jsonl is not None else "",
            "image_root": str(image_root) if image_root is not None else "",
            "sample_expensive_cache_jsonl": str(expensive_cache_path) if expensive_manager is not None else "",
            "images_with_real_or_cached_expensive": int(expensive_images_used),
            "images_without_expensive": int(expensive_images_unavailable),
        },
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
