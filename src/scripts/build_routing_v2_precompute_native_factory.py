#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from multimode.candidate_bank import build_global_candidate_bank, expand_query_candidates  # noqa: E402
from multimode.entity_atoms import build_entity_atoms  # noqa: E402
from multimode.mode_scorer import score_query_candidates  # noqa: E402
from multimode.query_builder import ModeQuery, build_mode_queries  # noqa: E402
from routing.routing_v2_simple import (  # noqa: E402
    ANIMAL_POSE_CONF_MIN,
    ANIMAL_POSE_HEAD_CONF_MIN,
    ANIMAL_POSE_SPECIES_CONF_MIN,
    CAT_CLASS_ID,
    DOG_CLASS_ID,
    derive_image_routing_v2_simple,
    flat_route_class_key,
    hierarchical_target_ids,
    probe_features_from_feature_row,
)
from routing.routing_v2_v16_modes import (  # noqa: E402
    IMAGE_ROUTE_NO_PLACEMENT_TO_ID,
    V16_MODE_BY_NAME,
    image_route_no_placement_from_routing,
    image_route_no_placement_id_from_routing,
    image_route_no_placement_catalog_payload,
    v16_categories,
    v16_mode_catalog_payload,
)


DEFAULT_PHASE_ROOT = Path("data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529")
DEFAULT_FEATURE_JSONL = DEFAULT_PHASE_ROOT / "artifacts/precompute/feats_c2c3c5_v2_strict_enriched_routed_final.jsonl"
DEFAULT_ANIMAL_POSE_JSONL = DEFAULT_PHASE_ROOT / "artifacts/precompute/animal_pose_260609_routing_v2_simple_gpu_full_ap10k_nms_v2/animal_pose.jsonl"
DEFAULT_CANDIDATES_JSONL = DEFAULT_PHASE_ROOT / "artifacts/candidates/candidates_ar_260529_v11_photo_primary_fullsstk_phaseA_strict_intent_subjectsafe_split9010.jsonl"
DEFAULT_RUN_TAG = "260609_v17_routing_v2_precompute_native_nofood_petstrict_cropstrict_nms_v3"
TARGET_ARS = ("FREE", "1:1", "4:3", "3:4", "16:9", "9:16")
SCHEMA_VERSION = "routing_v2_precompute_native_factory_v17_20260609"
DOGCAT_CLASS_IDS = {DOG_CLASS_ID, CAT_CLASS_ID}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build native routing_v2/v16 labels from perception precompute outputs.")
    parser.add_argument("--phase_root", type=Path, default=DEFAULT_PHASE_ROOT)
    parser.add_argument("--feature_jsonl", type=Path, default=DEFAULT_FEATURE_JSONL)
    parser.add_argument("--animal_pose_jsonl", type=Path, default=DEFAULT_ANIMAL_POSE_JSONL)
    parser.add_argument("--candidates_jsonl", type=Path, default=DEFAULT_CANDIDATES_JSONL)
    parser.add_argument("--image_root", type=Path, default=None)
    parser.add_argument("--run_tag", default=DEFAULT_RUN_TAG)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--max_candidates_per_query", type=int, default=0)
    parser.add_argument("--include_negative", type=int, default=1)
    parser.add_argument("--landscape_score_policy", default="teacher_subject_safe")
    parser.add_argument("--landscape_teacher_score_scope", default="gaic")
    parser.add_argument("--mode_intent_policy", default="strict_v11")
    parser.add_argument("--require_candidate_bank", type=int, default=1)
    return parser


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _load_features(path: Path, max_images: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("image_id") or "").strip():
                rows.append(row)
            if max_images > 0 and len(rows) >= max_images:
                break
    return rows


def _load_animal_pose_head_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            image_id = str(row.get("image_id") or "").strip()
            groups = row.get("visible_groups") if isinstance(row.get("visible_groups"), dict) else {}
            head = groups.get("head")
            head_conf = _safe_float(head, 1.0 if head is True else 0.0)
            species_conf = _safe_float(row.get("species_confidence"), 0.0)
            pose_conf = _safe_float(row.get("pose_confidence"), 0.0)
            species_hint = str(row.get("species_hint") or "").lower()
            if (
                image_id
                and species_hint in {"dog", "cat", "dogcat_mixed"}
                and species_conf >= ANIMAL_POSE_SPECIES_CONF_MIN
                and pose_conf >= ANIMAL_POSE_CONF_MIN
                and head_conf >= ANIMAL_POSE_HEAD_CONF_MIN
            ):
                ids.add(image_id)
    return ids


def _resolve_image_path(image_root: Path, image_id: str) -> Path:
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        path = image_root / f"{image_id}{suffix}"
        if path.exists():
            return path
    return image_root / f"{image_id}.jpg"


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return int(image.width), int(image.height)


def _split_for_image(image_id: str, val_ratio: float) -> str:
    digest = hashlib.md5(image_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / float(0xFFFFFFFF)
    return "val" if bucket < float(val_ratio) else "train"


def _clip_box(box: Sequence[Any]) -> list[float]:
    vals = [_safe_float(v) for v in list(box)[:4]]
    if len(vals) != 4:
        return [0.0, 0.0, 1.0, 1.0]
    x1, y1, x2, y2 = vals
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _box_area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def _inter_area(a: Sequence[float], b: Sequence[float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _recall(box: Sequence[float] | None, crop: Sequence[float]) -> float:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return 0.0
    denom = max(1.0e-8, _box_area([float(v) for v in box]))
    return _inter_area([float(v) for v in box], crop) / denom


def _kp_visible(kps: Sequence[Sequence[Any]], indices: Sequence[int], crop: Sequence[float], conf_thr: float = 0.15) -> bool:
    for idx in indices:
        if idx >= len(kps):
            continue
        kp = kps[idx]
        if not isinstance(kp, (list, tuple)) or len(kp) < 3:
            continue
        x = _safe_float(kp[0])
        y = _safe_float(kp[1])
        conf = _safe_float(kp[2])
        if conf >= conf_thr and crop[0] <= x <= crop[2] and crop[1] <= y <= crop[3]:
            return True
    return False


def _crop_shot_from_query_candidate(query: ModeQuery, candidate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    crop = _clip_box(candidate.get("bbox_norm_xyxy") or [0.0, 0.0, 1.0, 1.0])
    components = candidate.get("components") if isinstance(candidate.get("components"), dict) else {}
    if query.mode_name == "face":
        return "face_headshot", {
            "source": "native_face_query",
            "face_recall": _safe_float(components.get("face_recall"), _recall(query.face_bbox_norm_xyxy or query.anchor_bbox_norm_xyxy, crop)),
            "joint_visible_ratio": _safe_float(components.get("joint_visible_ratio"), 0.0),
        }
    kps = query.member_keypoints_norm[0] if query.member_keypoints_norm else []
    has_eye = _kp_visible(kps, (1, 2), crop)
    has_shoulder = _kp_visible(kps, (5, 6), crop)
    has_hip = _kp_visible(kps, (11, 12), crop)
    has_knee = _kp_visible(kps, (13, 14), crop)
    has_ankle = _kp_visible(kps, (15, 16), crop)
    pose_full = bool(has_shoulder and has_hip and has_knee and has_ankle)
    joint_visible = _safe_float(components.get("joint_visible_ratio"), 0.0)
    face_recall = _safe_float(components.get("face_recall"), _recall(query.face_bbox_norm_xyxy, crop))
    subject_ratio = _safe_float(components.get("subject_ratio"), _box_area(query.anchor_bbox_norm_xyxy) / max(1e-8, _box_area(crop)))
    bottom_relaxed = _safe_float(components.get("portrait_source_bottom_margin_relaxed"), 0.0) >= 0.5
    support_relaxed = _safe_float(components.get("portrait_full_body_support_grounding_relaxed"), 0.0) >= 0.5
    if pose_full and joint_visible >= 0.82 and not bottom_relaxed and not support_relaxed:
        shot = "full_body"
        source = "crop_pose_full_body"
    elif pose_full and joint_visible >= 0.82 and (bottom_relaxed or support_relaxed):
        shot = "upper_half_body"
        source = "crop_pose_full_body_demoted_relaxed_support"
    elif (has_shoulder or has_hip or has_knee) and not has_ankle:
        shot = "upper_half_body"
        source = "crop_pose_upper_half_body"
    elif face_recall >= 0.90 and subject_ratio >= 0.16 and not (has_hip or has_knee or has_ankle):
        shot = "face_headshot"
        source = "crop_face_only"
    else:
        shot = "upper_half_body"
        source = "crop_pose_fallback_upper_half_body"
    return shot, {
        "source": source,
        "has_eye": bool(has_eye),
        "has_shoulder": bool(has_shoulder),
        "has_hip": bool(has_hip),
        "has_knee": bool(has_knee),
        "has_ankle": bool(has_ankle),
        "joint_visible_ratio": round(joint_visible, 6),
        "face_recall": round(face_recall, 6),
        "subject_ratio": round(subject_ratio, 6),
        "bottom_relaxed": bool(bottom_relaxed),
        "support_relaxed": bool(support_relaxed),
    }


def _query_output_specs(query: ModeQuery, *, pet_pose_head_available: bool) -> list[tuple[str, str]]:
    mode = query.mode_name
    if mode == "face":
        return [("person_single_face_headshot_center", "face_headshot")]
    if mode == "single_person_center":
        return [
            ("person_single_upper_half_body_center", "upper_half_body"),
            ("person_single_full_body_center", "full_body"),
        ]
    if mode == "single_person_rot":
        return [
            ("person_single_upper_half_body_rot", "upper_half_body"),
            ("person_single_full_body_rot", "full_body"),
        ]
    if mode == "group_center":
        return [("person_group_center", "na")]
    if mode == "group_rot":
        return [("person_group_rot", "na")]
    if mode == "object_single_center":
        class_id = _safe_int(query.attributes.get("class_id", query.attributes.get("object_class_id")), -1)
        if class_id in DOGCAT_CLASS_IDS and pet_pose_head_available:
            return [("pet_dogcat_center", "na")]
        return []
    if mode == "object_single_rot":
        class_id = _safe_int(query.attributes.get("class_id", query.attributes.get("object_class_id")), -1)
        if class_id in DOGCAT_CLASS_IDS and pet_pose_head_available:
            return [("pet_dogcat_rot", "na")]
        return []
    if mode == "landscape":
        return [("scene_center", "na")]
    return []


def _routing_for_v16_mode(mode_name: str, *, score: float, feasible: bool, reason: str) -> dict[str, Any]:
    spec = V16_MODE_BY_NAME[mode_name]
    context = "balanced"
    if spec.route_family_v2 == "person_single" and spec.person_shot_type == "face_headshot" and spec.placement_intent == "center":
        context = "tight_subject"
    if spec.route_family_v2 == "pet_dogcat" and spec.placement_intent == "center":
        context = "tight_subject"
    conf = max(0.05, min(0.99, 0.62 + 0.33 * _safe_float(score, 0.0)))
    if not feasible:
        conf = min(conf, 0.50)
    return {
        "schema_version": "routing_v2_simple_v8_nofood_petstrict_shotstrict",
        "route_family_v2": spec.route_family_v2,
        "person_shot_type": spec.person_shot_type,
        "placement_intent": spec.placement_intent,
        "context_intent": context,
        "mode_feasible": bool(feasible),
        "routing_confidence": round(conf, 6),
        "support_grounding_policy": "active" if spec.person_shot_type == "full_body" else "animal_pose_required" if spec.route_family_v2 == "pet_dogcat" else "inactive" if spec.route_family_v2 == "person_single" else "weak" if spec.route_family_v2 == "person_group" else "not_applicable",
        "teacher_reliability": "high" if conf >= 0.75 else "medium" if conf >= 0.60 else "low",
        "reasons": [reason],
    }


def _best_candidate_for_output(
    query: ModeQuery,
    scored: Sequence[dict[str, Any]],
    *,
    final_mode: str,
    desired_shot: str,
) -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
    valid_rows: list[tuple[float, dict[str, Any], str, dict[str, Any]]] = []
    image_shot = str(query.attributes.get("image_person_shot_type") or "")
    for cand in scored:
        if bool(cand.get("hard_reject", False)):
            continue
        score = _safe_float(cand.get("score_mode"), 0.0)
        if score <= 0.0:
            continue
        if desired_shot == "full_body" and image_shot in {"face_headshot", "upper_half_body"}:
            continue
        inferred_shot, shot_debug = _crop_shot_from_query_candidate(query, cand)
        if desired_shot != "na" and inferred_shot != desired_shot:
            continue
        valid_rows.append((score, cand, inferred_shot, shot_debug))
    valid_rows.sort(key=lambda item: item[0], reverse=True)
    if valid_rows:
        _score, cand, inferred, debug = valid_rows[0]
        return dict(cand), inferred, debug
    best = next((dict(c) for c in scored if not bool(c.get("hard_reject", False))), None)
    if best is None and scored:
        best = dict(scored[0])
    inferred = "unknown"
    debug: dict[str, Any] = {"source": "no_matching_candidate"}
    if best is not None:
        inferred, debug = _crop_shot_from_query_candidate(query, best)
    return None, inferred, debug


def _norm_to_xywh_px(box: Sequence[Any], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = _clip_box(box)
    return [
        round(x1 * width, 3),
        round(y1 * height, 3),
        round(max(0.0, x2 - x1) * width, 3),
        round(max(0.0, y2 - y1) * height, 3),
    ]


def _ann_from_candidate(
    *,
    ann_id: int,
    image_id_int: int,
    width: int,
    height: int,
    final_mode: str,
    query: ModeQuery,
    candidate: dict[str, Any],
    gt_flag: int,
    negative_reason: str,
    crop_shot: str,
    crop_shot_debug: dict[str, Any],
) -> dict[str, Any]:
    spec = V16_MODE_BY_NAME[final_mode]
    score = _safe_float(candidate.get("score_mode"), 0.0)
    routing = _routing_for_v16_mode(
        final_mode,
        score=score,
        feasible=bool(gt_flag),
        reason=f"precompute_native:{query.mode_name}:{crop_shot_debug.get('source', '')}",
    )
    flat_key = flat_route_class_key(routing)
    bbox = _norm_to_xywh_px(candidate.get("bbox_norm_xyxy") or [0.0, 0.0, 1.0, 1.0], width, height)
    components = candidate.get("components") if isinstance(candidate.get("components"), dict) else {}
    attrs = {
        "schema_version": SCHEMA_VERSION,
        "target_ar": query.target_ar,
        "source_scorer_mode": query.mode_name,
        "source_query_id": query.query_id,
        "entity_id": query.entity_id,
        "entity_type": query.entity_type,
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "candidate_source": str(candidate.get("source") or ""),
        "score_components": components,
        "routing_v2_simple": routing,
        "flat_route_class": flat_key,
        "routing_v2_hierarchical_targets": hierarchical_target_ids(routing),
        "crop_shot_inferred": crop_shot,
        "crop_shot_debug": crop_shot_debug,
        "hard_reject_reasons": list(candidate.get("hard_reject_reasons") or []),
        "negative_reason": negative_reason,
        "query_has_positive": int(bool(gt_flag)),
        "native_factory": True,
    }
    return {
        "id": ann_id,
        "image_id": image_id_int,
        "category_id": int(spec.category_id),
        "mode_name": final_mode,
        "query_id": f"{query.query_id}::{final_mode}",
        "bbox": bbox,
        "area": round(bbox[2] * bbox[3], 3),
        "iscrowd": 0,
        "gt_flag": int(gt_flag),
        "is_best": int(gt_flag),
        "score_mode": round(score, 6),
        "target_ar": query.target_ar,
        "attributes": attrs,
    }


def _image_record(
    *,
    image_id_int: int,
    image_id: str,
    image_path: Path,
    width: int,
    height: int,
    routing: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": image_id_int,
        "file_name": image_path.name,
        "source_image_id": image_id,
        "width": int(width),
        "height": int(height),
        "routing_v2_simple": routing,
        "flat_route_class": flat_route_class_key(routing),
        "image_route_name_no_placement": image_route_no_placement_from_routing(routing),
        "image_route_id_no_placement": image_route_no_placement_id_from_routing(routing),
        "v16_mode_name": image_route_no_placement_from_routing(routing),
    }


def _training_label_row(
    *,
    image_id: str,
    image_path: Path,
    width: int,
    height: int,
    split: str,
    routing: dict[str, Any],
    feature_row: dict[str, Any],
) -> dict[str, Any]:
    route_name = image_route_no_placement_from_routing(routing)
    return {
        "image_id": image_id,
        "file_name": image_path.name,
        "image_path": str(image_path),
        "width": int(width),
        "height": int(height),
        "split": split,
        "schema_version": SCHEMA_VERSION,
        "routing_v2_simple": routing,
        "flat_route_class": flat_route_class_key(routing),
        "flat_route_class_id": -1,
        "hierarchical_targets": hierarchical_target_ids(routing),
        "routing_v2_hierarchical_targets": hierarchical_target_ids(routing),
        "image_route_name_no_placement": route_name,
        "image_route_id_no_placement": IMAGE_ROUTE_NO_PLACEMENT_TO_ID.get(route_name, 0),
        "v16_mode_name": route_name,
        "label_weight": _safe_float(routing.get("routing_confidence"), 0.0),
        "probe_features": probe_features_from_feature_row({**feature_row, "width": width, "height": height}),
    }


def _write_coco(path: Path, *, images: list[dict[str, Any]], annotations: list[dict[str, Any]]) -> None:
    payload = {
        "info": {
            "schema_version": SCHEMA_VERSION,
            "description": "routing_v2/v16 precompute-native multimode crop labels",
        },
        "images": images,
        "annotations": annotations,
        "categories": v16_categories(),
    }
    _write_json(path, payload)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    start = time.time()
    phase_root = args.phase_root
    image_root = args.image_root or phase_root / "images"
    run_tag = str(args.run_tag)
    mm_out_dir = phase_root / "artifacts/training_labels_multimode" / run_tag
    tl_out_dir = phase_root / "artifacts/training_labels" / run_tag
    label_dir = mm_out_dir / "label_json"
    status_path = mm_out_dir / "status.json"
    mm_out_dir.mkdir(parents=True, exist_ok=True)
    tl_out_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    _write_json(status_path, {"state": "running", "phase": "load_inputs", "run_tag": run_tag})

    features = _load_features(args.feature_jsonl, max_images=int(args.max_images))
    feature_by_id = {str(row.get("image_id") or "").strip(): row for row in features if str(row.get("image_id") or "").strip()}
    if int(args.require_candidate_bank) and not args.candidates_jsonl.exists():
        raise FileNotFoundError(f"required candidate bank not found: {args.candidates_jsonl}")
    animal_pose_head_ids = _load_animal_pose_head_ids(args.animal_pose_jsonl)
    images_full: list[dict[str, Any]] = []
    annotations_full: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    query_status_rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    seen_candidate_image_ids: set[str] = set()
    ann_id = 1

    with args.candidates_jsonl.open("r", encoding="utf-8") as candidate_handle:
        for candidate_lineno, line in enumerate(candidate_handle, start=1):
            if not line.strip():
                continue
            candidate_row = json.loads(line)
            image_id = str(candidate_row.get("image_id") or "").strip()
            feature_row = feature_by_id.get(image_id)
            if feature_row is None:
                continue
            seen_candidate_image_ids.add(image_id)
            idx = len(seen_candidate_image_ids)
            if idx % 500 == 0:
                _write_json(
                    status_path,
                    {
                        "state": "running",
                        "phase": "build_labels",
                        "processed": idx,
                        "total": len(features),
                        "candidate_lineno": candidate_lineno,
                        "annotations": len(annotations_full),
                    },
                )
            if not image_id:
                continue
            image_path = _resolve_image_path(image_root, image_id)
            if not image_path.exists():
                counters["missing_image"] += 1
                continue
            width, height = _image_size(image_path)
            feature_with_size = {**feature_row, "width": width, "height": height}
            pet_pose_head_available = image_id in animal_pose_head_ids if animal_pose_head_ids else None
            image_routing = derive_image_routing_v2_simple(
                feature_with_size,
                positive_vote_summary=None,
                pet_pose_head_available=pet_pose_head_available,
            )
            split = _split_for_image(image_id, float(args.val_ratio))
            image_row = _image_record(
                image_id_int=len(images_full) + 1,
                image_id=image_id,
                image_path=image_path,
                width=width,
                height=height,
                routing=image_routing,
            )
            images_full.append(image_row)
            training_rows.append(
                _training_label_row(
                    image_id=image_id,
                    image_path=image_path,
                    width=width,
                    height=height,
                    split=split,
                    routing=image_routing,
                    feature_row=feature_with_size,
                )
            )
            counters[f"image_route:{image_row['image_route_name_no_placement']}"] += 1

            atoms = build_entity_atoms(feature_with_size, width=width, height=height)
            image_ar = width / max(1.0, height)
            for target_ar in TARGET_ARS:
                base_candidates = build_global_candidate_bank(candidate_row, target_ar)
                if not base_candidates:
                    counters[f"candidate_bank_empty_ar:{target_ar}"] += 1
                base_candidate_count = len(base_candidates)
                base_teacher_count = sum(1 for cand in base_candidates if cand.get("teacher_provenance"))
                base_source_counts = Counter(str(cand.get("source") or "") for cand in base_candidates)
                base_queries = build_mode_queries(
                    image_id=image_id,
                    target_ar=target_ar,
                    atoms=atoms,
                    routing=feature_with_size.get("routing") if isinstance(feature_with_size.get("routing"), dict) else {},
                )
                for query in base_queries:
                    query.attributes["native_factory_source_mode"] = query.mode_name
                    # Carry dog/cat class id into object query attrs for pet gating.
                    for atom in atoms:
                        if atom.entity_id == query.entity_id and atom.meta.get("class_id") is not None:
                            query.attributes["class_id"] = int(_safe_int(atom.meta.get("class_id"), -1))
                            query.attributes["object_class_id"] = int(_safe_int(atom.meta.get("class_id"), -1))
                            break
                    outputs = _query_output_specs(query, pet_pose_head_available=bool(pet_pose_head_available))
                    if not outputs:
                        continue
                    if image_routing.get("route_family_v2") == "person_single":
                        query.attributes["image_person_shot_type"] = str(image_routing.get("person_shot_type") or "")
                    if base_candidates:
                        candidates = expand_query_candidates(base_candidates=base_candidates, query=query, image_ar=image_ar)
                        if int(args.max_candidates_per_query) > 0:
                            candidates = candidates[: int(args.max_candidates_per_query)]
                        scored = score_query_candidates(
                            query=query,
                            candidates=candidates,
                            feat_row=feature_with_size,
                            width=width,
                            height=height,
                            image_ar=image_ar,
                            landscape_score_policy=str(args.landscape_score_policy),
                            landscape_teacher_score_scope=str(args.landscape_teacher_score_scope),
                            mode_intent_policy=str(args.mode_intent_policy),
                        )
                    else:
                        candidates = []
                        scored = []
                    for final_mode, desired_shot in outputs:
                        positive, inferred_shot, shot_debug = _best_candidate_for_output(
                            query,
                            scored,
                            final_mode=final_mode,
                            desired_shot=desired_shot,
                        )
                        no_positive_reason = "" if positive is not None else (
                            "no_candidate_bank_for_ar" if not base_candidates else f"no_{desired_shot}_candidate"
                        )
                        status = {
                            "source_image_id": image_id,
                            "target_ar": target_ar,
                            "source_scorer_mode": query.mode_name,
                            "mode_name": final_mode,
                            "query_id": f"{query.query_id}::{final_mode}",
                            "entity_id": query.entity_id,
                            "base_candidate_count": base_candidate_count,
                            "base_teacher_candidate_count": base_teacher_count,
                            "base_source_top": dict(base_source_counts.most_common(8)),
                            "candidate_count": len(scored),
                            "positive_exists": int(positive is not None),
                            "positive_score": round(_safe_float(positive.get("score_mode")) if positive else 0.0, 6),
                            "best_score": round(_safe_float(scored[0].get("score_mode")) if scored else 0.0, 6),
                            "no_positive_reason": no_positive_reason,
                            "crop_shot_inferred_best": inferred_shot,
                        }
                        query_status_rows.append(status)
                        counters[f"query:{final_mode}"] += 1
                        if positive is not None:
                            annotations_full.append(
                                _ann_from_candidate(
                                    ann_id=ann_id,
                                    image_id_int=image_row["id"],
                                    width=width,
                                    height=height,
                                    final_mode=final_mode,
                                    query=query,
                                    candidate=positive,
                                    gt_flag=1,
                                    negative_reason="",
                                    crop_shot=inferred_shot,
                                    crop_shot_debug=shot_debug,
                                )
                            )
                            ann_id += 1
                            counters[f"positive:{final_mode}"] += 1
                        elif int(args.include_negative):
                            best = dict(scored[0]) if scored else {
                                "candidate_id": "",
                                "source": "none",
                                "bbox_norm_xyxy": [0.0, 0.0, 1.0, 1.0],
                                "score_mode": 0.0,
                                "hard_reject_reasons": [no_positive_reason or "no_candidate"],
                                "components": {},
                            }
                            annotations_full.append(
                                _ann_from_candidate(
                                    ann_id=ann_id,
                                    image_id_int=image_row["id"],
                                    width=width,
                                    height=height,
                                    final_mode=final_mode,
                                    query=query,
                                    candidate=best,
                                    gt_flag=0,
                                    negative_reason=status["no_positive_reason"],
                                    crop_shot=inferred_shot,
                                    crop_shot_debug=shot_debug,
                                )
                            )
                            ann_id += 1
                            counters[f"negative:{final_mode}"] += 1

            if len(seen_candidate_image_ids) >= len(feature_by_id):
                break

    missing_candidate_image_ids = sorted(set(feature_by_id) - seen_candidate_image_ids)
    if missing_candidate_image_ids and int(args.require_candidate_bank):
        raise RuntimeError(
            f"candidate bank is missing {len(missing_candidate_image_ids)} feature images; "
            f"first_missing={missing_candidate_image_ids[:5]}"
        )

    images_by_split = {
        "full": images_full,
        "train": [img for img in images_full if _split_for_image(str(img["source_image_id"]), float(args.val_ratio)) == "train"],
        "val": [img for img in images_full if _split_for_image(str(img["source_image_id"]), float(args.val_ratio)) == "val"],
    }
    image_id_to_split = {str(row["image_id"]): str(row["split"]) for row in training_rows}
    source_by_int = {int(img["id"]): str(img["source_image_id"]) for img in images_full}
    anns_by_split = {
        "full": annotations_full,
        "train": [ann for ann in annotations_full if image_id_to_split.get(source_by_int.get(int(ann["image_id"]), ""), "") == "train"],
        "val": [ann for ann in annotations_full if image_id_to_split.get(source_by_int.get(int(ann["image_id"]), ""), "") == "val"],
    }
    rows_by_split = {
        "full": training_rows,
        "train": [row for row in training_rows if row["split"] == "train"],
        "val": [row for row in training_rows if row["split"] == "val"],
    }

    for split in ("full", "train", "val"):
        _write_coco(label_dir / f"multimode_labels_{split}.json", images=images_by_split[split], annotations=anns_by_split[split])
        _write_jsonl(tl_out_dir / f"routing_v2_v17_precompute_native_{split}.jsonl", rows_by_split[split])

    _write_json(mm_out_dir / "categories.json", {"categories": v16_categories()})
    _write_json(mm_out_dir / "routing_v2_v16_mode_catalog.json", v16_mode_catalog_payload())
    _write_json(tl_out_dir / "routing_v2_image_route_no_placement_catalog.json", image_route_no_placement_catalog_payload())
    _write_jsonl(mm_out_dir / "mode_query_status_precompute_native.jsonl", query_status_rows)
    _write_csv(mm_out_dir / "routing_v2_precompute_native_counts.csv", [{"key": key, "count": int(value)} for key, value in sorted(counters.items())])

    positive_count = sum(1 for ann in annotations_full if int(ann.get("gt_flag") or 0) == 1)
    negative_count = len(annotations_full) - positive_count
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_tag": run_tag,
        "phase_root": str(phase_root),
        "feature_jsonl": str(args.feature_jsonl),
        "animal_pose_jsonl": str(args.animal_pose_jsonl),
        "candidates_jsonl": str(args.candidates_jsonl),
        "image_count": len(images_full),
        "annotation_count": len(annotations_full),
        "positive_annotation_count": positive_count,
        "negative_annotation_count": negative_count,
        "query_count": len(query_status_rows),
        "positive_query_count": sum(1 for row in query_status_rows if int(row["positive_exists"]) == 1),
        "target_ars": list(TARGET_ARS),
        "image_route_counts": {key.removeprefix("image_route:"): int(value) for key, value in sorted(counters.items()) if key.startswith("image_route:")},
        "positive_mode_counts": {key.removeprefix("positive:"): int(value) for key, value in sorted(counters.items()) if key.startswith("positive:")},
        "negative_mode_counts": {key.removeprefix("negative:"): int(value) for key, value in sorted(counters.items()) if key.startswith("negative:")},
        "candidate_bank_empty_ar_counts": {key.removeprefix("candidate_bank_empty_ar:"): int(value) for key, value in sorted(counters.items()) if key.startswith("candidate_bank_empty_ar:")},
        "missing_candidate_image_count": len(missing_candidate_image_ids),
        "missing_candidate_image_ids_sample": missing_candidate_image_ids[:20],
        "elapsed_sec": round(time.time() - start, 3),
        "output_paths": {
            "training_labels": str(tl_out_dir),
            "training_labels_multimode": str(mm_out_dir),
            "label_json": str(label_dir),
            "mode_query_status": str(mm_out_dir / "mode_query_status_precompute_native.jsonl"),
        },
        "notes": {
            "source_policy": "v14 label JSON is not used; perception precompute + image files + animal pose head ids are used.",
            "candidate_policy": "GAIC/CGS/CACNet enriched candidates_jsonl is required. Mode-local seeds may expand an existing bank, but an absent AR bank is not substituted inside the label factory.",
            "scene_policy": "scene queries are scored against the same candidate bank; no image-level positive-vote override is used.",
        },
    }
    _write_json(mm_out_dir / "summary.json", summary)
    _write_json(tl_out_dir / "summary.json", summary)
    _write_json(status_path, {"state": "completed", **summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
