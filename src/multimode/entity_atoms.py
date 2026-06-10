from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence

from portrait_composition import build_group_portrait_spec, build_single_portrait_spec
from saliency_semantic import ANIMAL_CLASS_IDS, PERSON_CLASS_IDS
from subject_region import box_area, box_center, norm_box_xyxy, union_boxes


@dataclass
class EntityAtom:
    entity_id: str
    entity_type: str
    family: str
    bbox_norm_xyxy: List[float]
    core_bbox_norm_xyxy: List[float]
    envelope_bbox_norm_xyxy: List[float]
    face_bbox_norm_xyxy: Optional[List[float]] = None
    head_bbox_norm_xyxy: Optional[List[float]] = None
    support_bbox_norm_xyxy: Optional[List[float]] = None
    member_boxes_norm_xyxy: List[List[float]] = field(default_factory=list)
    member_face_boxes_norm_xyxy: List[List[float]] = field(default_factory=list)
    member_keypoints_norm: List[List[List[float]]] = field(default_factory=list)
    head_y_norm: List[float] = field(default_factory=list)
    gaze_entries: List[Dict[str, Any]] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clip_box(box: Sequence[float]) -> List[float]:
    out = norm_box_xyxy(box, 1, 1)
    return [float(v) for v in out]


def _intersection_area(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    return max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))


def _box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    inter = _intersection_area(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 1e-8 else 0.0


def _box_recall(box: Sequence[float], cover: Sequence[float]) -> float:
    denom = max(1e-8, box_area(box))
    return _intersection_area(box, cover) / denom


def _dedupe_boxes(boxes: Sequence[Sequence[float]], iou_thr: float = 0.85) -> List[List[float]]:
    deduped: List[List[float]] = []
    for box in boxes:
        norm = _clip_box(box)
        if box_area(norm) <= 0.0:
            continue
        duplicated = False
        for existing in deduped:
            ix1 = max(norm[0], existing[0])
            iy1 = max(norm[1], existing[1])
            ix2 = min(norm[2], existing[2])
            iy2 = min(norm[3], existing[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            union = box_area(norm) + box_area(existing) - inter
            iou = inter / union if union > 1e-8 else 0.0
            if iou >= iou_thr:
                duplicated = True
                break
        if not duplicated:
            deduped.append(norm)
    return deduped


def _center_distance(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay = box_center(a)
    bx, by = box_center(b)
    return ((float(ax) - float(bx)) ** 2 + (float(ay) - float(by)) ** 2) ** 0.5


def _same_person_duplicate(a: EntityAtom, b: EntityAtom) -> bool:
    small, large = (a, b) if box_area(a.bbox_norm_xyxy) <= box_area(b.bbox_norm_xyxy) else (b, a)
    containment = _box_recall(small.bbox_norm_xyxy, large.bbox_norm_xyxy)
    if containment < 0.88:
        return False
    if small.face_bbox_norm_xyxy is not None and large.face_bbox_norm_xyxy is not None:
        face_iou = _box_iou(small.face_bbox_norm_xyxy, large.face_bbox_norm_xyxy)
        face_containment = max(
            _box_recall(small.face_bbox_norm_xyxy, large.face_bbox_norm_xyxy),
            _box_recall(large.face_bbox_norm_xyxy, small.face_bbox_norm_xyxy),
        )
        return face_iou >= 0.45 or face_containment >= 0.78
    size_ratio = box_area(small.bbox_norm_xyxy) / max(1e-8, box_area(large.bbox_norm_xyxy))
    return size_ratio >= 0.55 and _center_distance(small.bbox_norm_xyxy, large.bbox_norm_xyxy) <= 0.08


def _face_fallback_box(face_box: Sequence[float], person_box: Sequence[float]) -> Optional[List[float]]:
    face = _clip_box(face_box)
    person = _clip_box(person_box)
    face_area = box_area(face)
    person_area = box_area(person)
    if face_area <= 0.0 or person_area <= 0.0:
        return None
    if _box_recall(face, person) < 0.70:
        return None
    cx, cy = box_center(face)
    px1, py1, px2, py2 = person
    person_w = max(1e-6, px2 - px1)
    person_h = max(1e-6, py2 - py1)
    if not (px1 - 0.03 <= cx <= px2 + 0.03):
        return None
    if not (py1 - 0.03 <= cy <= py1 + 0.72 * person_h):
        return None
    fw = max(1e-6, face[2] - face[0])
    fh = max(1e-6, face[3] - face[1])
    max_area = min(0.095, max(0.018, 0.40 * person_area))
    scale = min(1.0, math.sqrt(max_area / max(face_area, 1e-8)))
    fw *= scale
    fh *= scale
    fw = min(fw, 0.72 * person_w)
    fh = min(fh, 0.46 * person_h)
    adjusted_area = fw * fh
    if adjusted_area > max_area:
        scale = math.sqrt(max_area / max(adjusted_area, 1e-8))
        fw *= scale
        fh *= scale
    x1 = cx - 0.5 * fw
    x2 = cx + 0.5 * fw
    y1 = cy - 0.5 * fh
    y2 = cy + 0.5 * fh
    if x1 < px1:
        x2 += px1 - x1
        x1 = px1
    if x2 > px2:
        x1 -= x2 - px2
        x2 = px2
    if y1 < py1:
        y2 += py1 - y1
        y1 = py1
    upper_limit = min(1.0, py1 + 0.74 * person_h)
    if y2 > upper_limit:
        y1 -= y2 - upper_limit
        y2 = upper_limit
    out = _clip_box([x1, y1, x2, y2])
    out_area = box_area(out)
    if not (0.00025 <= out_area <= 0.12):
        return None
    if out_area / max(1e-8, person_area) > 0.45:
        return None
    return out


def _dedupe_group_person_atoms(person_atoms: Sequence[EntityAtom]) -> List[EntityAtom]:
    ordered = sorted(person_atoms, key=lambda atom: box_area(atom.bbox_norm_xyxy), reverse=True)
    kept: List[EntityAtom] = []
    for atom in ordered:
        if any(_same_person_duplicate(atom, existing) for existing in kept):
            atom.meta["group_member_duplicate_suppressed"] = True
            continue
        atom.meta["group_member_duplicate_suppressed"] = False
        kept.append(atom)
    return kept


def _group_candidate_allowed(
    trusted_group_persons: Sequence[EntityAtom],
    person_atoms: Sequence[EntityAtom],
    routing: Dict[str, Any],
) -> tuple[bool, Dict[str, Any]]:
    subject_set = routing.get("subject_set") if isinstance(routing.get("subject_set"), dict) else {}
    router_signals = routing.get("router_signals") if isinstance(routing.get("router_signals"), dict) else {}
    route_mode_norm = str(routing.get("subject_mode", "") or "").strip().lower()
    routed_person_count = int(
        max(
            _safe_float(subject_set.get("num_person"), 0.0),
            _safe_float(router_signals.get("num_person"), 0.0),
        )
    )
    c2_raw_person_count = int(_safe_float(router_signals.get("num_person_c2_raw"), 0.0))
    c2_person_count = int(_safe_float(router_signals.get("num_person_c2"), 0.0))
    c3_supported_count = int(_safe_float(router_signals.get("num_person_c3_supported"), 0.0))
    semantic_human_count = int(_safe_float(router_signals.get("semantic_human_cluster_count"), 0.0))
    distinct_count = len(trusted_group_persons)
    distinct_face_count = sum(
        1
        for atom in trusted_group_persons
        if atom.face_bbox_norm_xyxy is not None and _safe_float(atom.meta.get("face_score"), 0.0) >= 0.72
    )
    multi_human_evidence = max(routed_person_count, c2_raw_person_count, c2_person_count, c3_supported_count, semantic_human_count)
    evidence_count = sum(
        1
        for passed in (
            routed_person_count >= 2,
            c2_raw_person_count >= 2,
            c2_person_count >= 2,
            c3_supported_count >= 2,
            semantic_human_count >= 2,
            distinct_face_count >= 2,
        )
        if passed
    )
    allow = False
    rule = "blocked"
    if distinct_count < 2:
        rule = "insufficient_distinct_members"
    elif route_mode_norm == "portrait_group":
        allow = evidence_count >= 1 or distinct_face_count >= 2
        rule = "portrait_group_multi_human_evidence" if allow else "portrait_group_missing_multi_human_evidence"
    elif route_mode_norm == "portrait_single":
        allow = multi_human_evidence >= 2 and evidence_count >= 2
        rule = "portrait_single_high_confidence_group_exception" if allow else "portrait_single_group_suppressed"
    else:
        allow = multi_human_evidence >= 2 or distinct_face_count >= 2
        rule = "generic_multi_human_evidence" if allow else "generic_group_missing_multi_human_evidence"
    return allow, {
        "group_gate_rule": rule,
        "group_gate_allow": bool(allow),
        "group_distinct_member_count": int(distinct_count),
        "group_raw_person_atom_count": int(len(person_atoms)),
        "group_distinct_face_count": int(distinct_face_count),
        "group_routed_person_count": int(routed_person_count),
        "group_c2_person_count": int(c2_person_count),
        "group_c2_raw_person_count": int(c2_raw_person_count),
        "group_c3_supported_count": int(c3_supported_count),
        "group_semantic_human_cluster_count": int(semantic_human_count),
        "group_multi_human_evidence_count": int(multi_human_evidence),
        "group_evidence_signal_count": int(evidence_count),
        "group_route_mode": route_mode_norm,
    }


def _group_fallback_candidate_allowed(
    trusted_group_persons: Sequence[EntityAtom],
    person_atoms: Sequence[EntityAtom],
    routing: Dict[str, Any],
    group_gate_meta: Dict[str, Any],
) -> tuple[bool, Dict[str, Any], List[EntityAtom]]:
    route_mode_norm = str(routing.get("subject_mode", "") or "").strip().lower()
    if route_mode_norm != "portrait_group":
        return False, {"group_fallback_rule": "non_group_route"}, []
    raw_trusted = [atom for atom in person_atoms if bool(atom.meta.get("person_atom_trusted", True))]
    if len(raw_trusted) < 2:
        return False, {"group_fallback_rule": "insufficient_raw_members"}, []
    if len(trusted_group_persons) >= 2 and bool(group_gate_meta.get("group_gate_allow", False)):
        return False, {"group_fallback_rule": "regular_group_available"}, []
    duplicate_suppressed = sum(1 for atom in raw_trusted if bool(atom.meta.get("group_member_duplicate_suppressed", False)))
    if duplicate_suppressed <= 0:
        return False, {"group_fallback_rule": "no_merged_duplicate_evidence"}, []

    subject_set = routing.get("subject_set") if isinstance(routing.get("subject_set"), dict) else {}
    router_signals = routing.get("router_signals") if isinstance(routing.get("router_signals"), dict) else {}
    raw_union_box = union_boxes([atom.bbox_norm_xyxy for atom in raw_trusted])
    raw_union_area = box_area(raw_union_box) if raw_union_box is not None else 0.0
    max_raw_area = max((box_area(atom.bbox_norm_xyxy) for atom in raw_trusted), default=0.0)
    raw_union_to_max = raw_union_area / max(1e-8, max_raw_area)
    multi_subject_flag = bool(subject_set.get("multi_subject", False))
    person_union_area = _safe_float(router_signals.get("person_union_area_ratio", subject_set.get("person_union_area_ratio", 0.0)), 0.0)
    face_count = sum(
        1
        for atom in raw_trusted
        if atom.face_bbox_norm_xyxy is not None and _safe_float(atom.meta.get("face_score"), 0.0) >= 0.70
    )
    allow = (
        raw_union_area >= 0.055
        and person_union_area >= 0.050
        and face_count >= 1
        and (multi_subject_flag or raw_union_to_max >= 1.04 or len(raw_trusted) >= 3)
    )
    return allow, {
        "group_fallback_rule": "portrait_group_merged_raw_persons" if allow else "portrait_group_merged_raw_threshold_miss",
        "group_fallback_atom": int(bool(allow)),
        "group_fallback_raw_trusted_count": int(len(raw_trusted)),
        "group_fallback_duplicate_suppressed_count": int(duplicate_suppressed),
        "group_fallback_raw_union_area_ratio": round(float(raw_union_area), 6),
        "group_fallback_raw_union_to_max_area": round(float(raw_union_to_max), 6),
        "group_fallback_person_union_area_ratio": round(float(person_union_area), 6),
        "group_fallback_multi_subject_flag": int(bool(multi_subject_flag)),
        "group_fallback_face_count": int(face_count),
    }, raw_trusted if allow else []


def _normalize_keypoints(keypoints: Any, width: int, height: int) -> List[List[float]]:
    out: List[List[float]] = []
    if not isinstance(keypoints, list):
        return out
    for kp in keypoints:
        if not isinstance(kp, (list, tuple)) or len(kp) < 3:
            continue
        out.append(
            [
                _safe_float(kp[0]) / max(1.0, float(width)),
                _safe_float(kp[1]) / max(1.0, float(height)),
                _safe_float(kp[2]),
            ]
        )
    return out


def _head_y_from_pose(pose: Dict[str, Any], width: int, height: int) -> Optional[float]:
    face = pose.get("face")
    if isinstance(face, dict):
        box = face.get("bbox_norm")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            box = face.get("bbox")
            if isinstance(box, (list, tuple)) and len(box) == 4:
                box = norm_box_xyxy(box, width, height)
        if isinstance(box, (list, tuple)) and len(box) == 4:
            fy1 = _safe_float(box[1])
            fy2 = _safe_float(box[3])
            fh = max(1e-6, fy2 - fy1)
            return max(0.0, fy1 - 0.22 * fh)
    keypoints = _normalize_keypoints(pose.get("keypoints"), width, height)
    top = []
    for idx in (0, 1, 2):
        if idx < len(keypoints) and keypoints[idx][2] >= 0.05:
            top.append(keypoints[idx][1])
    if top:
        return max(0.0, min(top) - 0.04)
    return None


def _gaze_entry_from_pose(pose: Dict[str, Any], bbox_norm_xyxy: Sequence[float], width: int, height: int) -> Optional[Dict[str, Any]]:
    headpose = pose.get("headpose_gaze")
    if not isinstance(headpose, dict):
        return None
    face = pose.get("face") if isinstance(pose.get("face"), dict) else {}
    face_box = face.get("bbox_norm")
    if not isinstance(face_box, (list, tuple)) or len(face_box) != 4:
        face_box = face.get("bbox")
        if isinstance(face_box, (list, tuple)) and len(face_box) == 4:
            face_box = norm_box_xyxy(face_box, width, height)
    if isinstance(face_box, (list, tuple)) and len(face_box) == 4:
        anchor_x = 0.5 * (_safe_float(face_box[0]) + _safe_float(face_box[2]))
        anchor_y = 0.5 * (_safe_float(face_box[1]) + _safe_float(face_box[3]))
    else:
        anchor_x = 0.5 * (_safe_float(bbox_norm_xyxy[0]) + _safe_float(bbox_norm_xyxy[2]))
        anchor_y = 0.5 * (_safe_float(bbox_norm_xyxy[1]) + _safe_float(bbox_norm_xyxy[3]))
    return {
        "gaze_dir": str(headpose.get("gaze_dir", "unknown")),
        "conf": _safe_float(headpose.get("conf", 0.0)),
        "anchor_x": anchor_x,
        "anchor_y": anchor_y,
        "yaw_proxy": _safe_float(headpose.get("yaw_proxy", 0.0)),
    }


def build_entity_atoms(feat_row: Dict[str, Any], *, width: int, height: int) -> List[EntityAtom]:
    atoms: List[EntityAtom] = []
    routing = feat_row.get("routing") if isinstance(feat_row.get("routing"), dict) else {}
    route_mode_norm = str(routing.get("subject_mode", "") or "").strip().lower()
    effective_subject_region = routing.get("effective_subject_region") if isinstance(routing.get("effective_subject_region"), dict) else {}
    saliency_semantic = routing.get("saliency_semantic_summary") if isinstance(routing.get("saliency_semantic_summary"), dict) else {}
    c3_pose = feat_row.get("c3_pose") if isinstance(feat_row.get("c3_pose"), list) else []
    c2_seg = feat_row.get("c2_seg") if isinstance(feat_row.get("c2_seg"), list) else []
    c2_det = feat_row.get("c2_det") if isinstance(feat_row.get("c2_det"), list) else []

    c2_person_boxes: List[List[float]] = []
    for inst in list(c2_seg) + list(c2_det):
        if not isinstance(inst, dict) or bool(inst.get("bg_like", False)):
            continue
        class_id = int(_safe_float(inst.get("class_id", -1), -1))
        if class_id not in PERSON_CLASS_IDS:
            continue
        box = inst.get("box") or inst.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        norm = norm_box_xyxy(box, width, height)
        if box_area(norm) > 0.0:
            c2_person_boxes.append([float(v) for v in norm])
    subject_set = routing.get("subject_set") if isinstance(routing.get("subject_set"), dict) else {}
    routed_person_count = int(_safe_float(subject_set.get("num_person"), 0.0))

    person_atoms: List[EntityAtom] = []
    for idx, pose in enumerate(c3_pose):
        if not isinstance(pose, dict):
            continue
        bbox = pose.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        bbox_norm = norm_box_xyxy(bbox, width, height)
        if box_area(bbox_norm) <= 0.0:
            continue
        face_bbox_norm = None
        face_score = None
        face = pose.get("face")
        if isinstance(face, dict):
            score_value = face.get("score", face.get("conf"))
            if score_value is not None:
                face_score = _safe_float(score_value, 0.0)
            face_bbox = face.get("bbox_norm")
            if not isinstance(face_bbox, (list, tuple)) or len(face_bbox) != 4:
                face_bbox = face.get("bbox")
                if isinstance(face_bbox, (list, tuple)) and len(face_bbox) == 4:
                    face_bbox = norm_box_xyxy(face_bbox, width, height)
            if isinstance(face_bbox, (list, tuple)) and len(face_bbox) == 4 and box_area(face_bbox) > 0.0:
                face_bbox_norm = [float(v) for v in face_bbox]
        head_y = _head_y_from_pose(pose, width, height)
        gaze_entry = _gaze_entry_from_pose(pose, bbox_norm, width, height)
        keypoints_norm = _normalize_keypoints(pose.get("keypoints"), width, height)
        portrait_comp = build_single_portrait_spec(
            bbox_norm_xyxy=bbox_norm,
            face_bbox_norm_xyxy=face_bbox_norm,
            keypoints_norm=keypoints_norm,
            gaze_entries=[gaze_entry] if gaze_entry is not None else [],
        )
        person_atoms.append(
            EntityAtom(
                entity_id=f"person_{idx:02d}",
                entity_type="person",
                family="human",
                bbox_norm_xyxy=[float(v) for v in bbox_norm],
                core_bbox_norm_xyxy=[float(v) for v in bbox_norm],
                envelope_bbox_norm_xyxy=[float(v) for v in bbox_norm],
                face_bbox_norm_xyxy=face_bbox_norm,
                head_bbox_norm_xyxy=(list(face_bbox_norm) if face_bbox_norm is not None else None),
                support_bbox_norm_xyxy=[float(v) for v in bbox_norm],
                member_boxes_norm_xyxy=[[float(v) for v in bbox_norm]],
                member_face_boxes_norm_xyxy=[face_bbox_norm] if face_bbox_norm is not None else [],
                member_keypoints_norm=[keypoints_norm] if keypoints_norm else [],
                head_y_norm=[head_y] if head_y is not None else [],
                gaze_entries=[gaze_entry] if gaze_entry is not None else [],
                meta={
                    "pose_index": idx,
                    "pose_score": _safe_float(pose.get("score"), 0.0),
                    "face_score": face_score,
                    "face_area_ratio": float(box_area(face_bbox_norm)) if face_bbox_norm is not None else 0.0,
                    "support_mass": None,
                    "portrait_comp": portrait_comp,
                },
            )
        )
    person_area_pairs = [
        (atom, box_area(atom.bbox_norm_xyxy))
        for atom in person_atoms
    ]
    total_person_area = sum(area for _, area in person_area_pairs)
    ranked_persons = sorted(person_area_pairs, key=lambda item: item[1], reverse=True)
    max_person_area = ranked_persons[0][1] if ranked_persons else 0.0
    for rank, (atom, area) in enumerate(ranked_persons):
        c2_iou = max((_box_iou(atom.bbox_norm_xyxy, box) for box in c2_person_boxes), default=0.0)
        c2_recall = max((_box_recall(atom.bbox_norm_xyxy, box) for box in c2_person_boxes), default=0.0)
        face_score = atom.meta.get("face_score")
        face_area = _safe_float(atom.meta.get("face_area_ratio"), 0.0)
        face_to_pose = face_area / max(1e-8, float(area))
        secondary_like = rank > 0 or (
            routed_person_count > 0 and int(_safe_float(atom.meta.get("pose_index"), 0)) >= routed_person_count
        )
        c2_supported = c2_iou >= 0.18 or c2_recall >= 0.55 or (not c2_person_boxes and area >= 0.12)
        face_conf_ok = face_score is None or _safe_float(face_score, 0.0) >= 0.55
        primary_ok = rank == 0 and area >= 0.015
        secondary_ok = (
            area >= 0.030
            and (float(area / total_person_area) if total_person_area > 1e-8 else 0.0) >= 0.10
            and c2_supported
            and face_conf_ok
        )
        person_atom_trusted = bool(primary_ok if not secondary_like else secondary_ok)
        face_query_valid = bool(
            person_atom_trusted
            and atom.face_bbox_norm_xyxy is not None
            and face_score is not None
            and _safe_float(face_score, 0.0) >= 0.55
            and 0.00025 <= face_area <= 0.12
            and face_to_pose <= 0.45
        )
        face_fallback_used = False
        original_face_bbox = list(atom.face_bbox_norm_xyxy) if atom.face_bbox_norm_xyxy is not None else None
        if (
            not face_query_valid
            and person_atom_trusted
            and atom.face_bbox_norm_xyxy is not None
            and face_score is not None
            and _safe_float(face_score, 0.0) >= 0.88
            and (face_area > 0.12 or face_to_pose > 0.45)
        ):
            fallback_box = _face_fallback_box(atom.face_bbox_norm_xyxy, atom.bbox_norm_xyxy)
            if fallback_box is not None:
                atom.face_bbox_norm_xyxy = fallback_box
                atom.head_bbox_norm_xyxy = list(fallback_box)
                atom.member_face_boxes_norm_xyxy = [list(fallback_box)]
                face_area = float(box_area(fallback_box))
                face_to_pose = face_area / max(1e-8, float(area))
                face_query_valid = bool(
                    0.00025 <= face_area <= 0.12
                    and face_to_pose <= 0.45
                    and _safe_float(face_score, 0.0) >= 0.55
                )
                face_fallback_used = bool(face_query_valid)
        reject_reasons: List[str] = []
        if not person_atom_trusted:
            if secondary_like:
                reject_reasons.append("secondary_person_atom_not_trusted")
            if not c2_supported:
                reject_reasons.append("low_c2_person_agreement")
            if not face_conf_ok:
                reject_reasons.append("low_face_score")
            if area < 0.030:
                reject_reasons.append("tiny_pose_area")
        if atom.face_bbox_norm_xyxy is not None and not face_query_valid:
            if face_score is None or _safe_float(face_score, 0.0) < 0.55:
                reject_reasons.append("face_query_low_confidence")
            if not (0.00025 <= face_area <= 0.12):
                reject_reasons.append("face_query_implausible_area")
            if face_to_pose > 0.45:
                reject_reasons.append("face_query_too_large_for_pose")
        if face_fallback_used:
            reject_reasons = [reason for reason in reject_reasons if reason not in {"face_query_implausible_area", "face_query_too_large_for_pose"}]
        atom.meta.update(
            {
                "area_ratio": float(area),
                "person_rank": rank,
                "person_area_share": float(area / total_person_area) if total_person_area > 1e-8 else 0.0,
                "person_size_rel": float(area / max_person_area) if max_person_area > 1e-8 else 0.0,
                "person_count": len(person_atoms),
                "routed_person_count": routed_person_count,
                "secondary_like": bool(secondary_like),
                "c2_person_iou": float(c2_iou),
                "c2_person_recall": float(c2_recall),
                "person_atom_trusted": bool(person_atom_trusted),
                "face_query_valid": bool(face_query_valid),
                "face_query_fallback_used": bool(face_fallback_used),
                "face_query_original_bbox_norm_xyxy": original_face_bbox if face_fallback_used else None,
                "face_to_pose_area_ratio": float(face_to_pose),
                "face_area_ratio": float(face_area),
                "atom_reject_reasons": reject_reasons,
            }
        )
    atoms.extend(person_atoms)

    trusted_group_persons = _dedupe_group_person_atoms(
        [atom for atom in person_atoms if bool(atom.meta.get("person_atom_trusted", True))]
    )
    group_allowed, group_gate_meta = _group_candidate_allowed(trusted_group_persons, person_atoms, routing)
    group_members_for_atom: List[EntityAtom] = []
    group_meta_extra: Dict[str, Any] = {}
    if len(trusted_group_persons) >= 2 and group_allowed:
        group_members_for_atom = list(trusted_group_persons)
        group_meta_extra = {"group_fallback_atom": 0}
    else:
        fallback_allowed, fallback_meta, fallback_members = _group_fallback_candidate_allowed(
            trusted_group_persons=trusted_group_persons,
            person_atoms=person_atoms,
            routing=routing,
            group_gate_meta=group_gate_meta,
        )
        if fallback_allowed:
            group_members_for_atom = list(fallback_members)
            group_meta_extra = dict(fallback_meta)
    if group_members_for_atom:
        union_box = union_boxes([atom.bbox_norm_xyxy for atom in group_members_for_atom])
        if union_box is not None and box_area(union_box) > 0.0:
            portrait_comp = build_group_portrait_spec(
                member_boxes_norm_xyxy=[atom.bbox_norm_xyxy for atom in group_members_for_atom],
                member_face_boxes_norm_xyxy=[box for atom in group_members_for_atom for box in atom.member_face_boxes_norm_xyxy],
                member_keypoints_norm=[kp for atom in group_members_for_atom for kp in atom.member_keypoints_norm],
                gaze_entries=[entry for atom in group_members_for_atom for entry in atom.gaze_entries],
            )
            atoms.append(
                EntityAtom(
                    entity_id="group_all",
                    entity_type="group",
                    family="human",
                    bbox_norm_xyxy=union_box,
                    core_bbox_norm_xyxy=union_box,
                    envelope_bbox_norm_xyxy=union_box,
                    support_bbox_norm_xyxy=union_box,
                    member_boxes_norm_xyxy=[atom.bbox_norm_xyxy for atom in group_members_for_atom],
                    member_face_boxes_norm_xyxy=[box for atom in group_members_for_atom for box in atom.member_face_boxes_norm_xyxy],
                    member_keypoints_norm=[kp for atom in group_members_for_atom for kp in atom.member_keypoints_norm],
                    head_y_norm=[val for atom in group_members_for_atom for val in atom.head_y_norm],
                    gaze_entries=[entry for atom in group_members_for_atom for entry in atom.gaze_entries],
                    meta={
                        "member_ids": [atom.entity_id for atom in group_members_for_atom],
                        "member_count": len(group_members_for_atom),
                        "raw_member_count": len(person_atoms),
                        "group_area_ratio": float(box_area(union_box)),
                        "portrait_comp": portrait_comp,
                        **group_gate_meta,
                        **group_meta_extra,
                    },
                )
            )

    object_atoms: List[EntityAtom] = []
    for idx, inst in enumerate(c2_seg):
        if not isinstance(inst, dict) or bool(inst.get("bg_like", False)):
            continue
        class_id = int(_safe_float(inst.get("class_id", -1), -1))
        if class_id < 0:
            continue
        if class_id in PERSON_CLASS_IDS:
            continue
        box = inst.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        bbox_norm = norm_box_xyxy(box, width, height)
        if box_area(bbox_norm) <= 0.0:
            continue
        importance = _safe_float(inst.get("importance_score", 0.0), 0.0)
        area_ratio = _safe_float(inst.get("area_ratio", box_area(bbox_norm)), box_area(bbox_norm))
        if importance < 0.12 and area_ratio < 0.01:
            continue
        if (
            area_ratio >= 0.78
            and bbox_norm[0] <= 0.02
            and bbox_norm[1] <= 0.02
            and bbox_norm[2] >= 0.98
            and bbox_norm[3] >= 0.98
        ):
            continue
        family = "animal" if class_id in ANIMAL_CLASS_IDS else "object"
        object_atoms.append(
            EntityAtom(
                entity_id=f"object_{len(object_atoms):02d}",
                entity_type="object",
                family=family,
                bbox_norm_xyxy=[float(v) for v in bbox_norm],
                core_bbox_norm_xyxy=[float(v) for v in bbox_norm],
                envelope_bbox_norm_xyxy=[float(v) for v in bbox_norm],
                support_bbox_norm_xyxy=[float(v) for v in bbox_norm],
                member_boxes_norm_xyxy=[[float(v) for v in bbox_norm]],
                meta={
                    "class_id": class_id,
                    "importance_score": importance,
                    "area_ratio": area_ratio,
                    "seg_index": idx,
                    "route_mode": route_mode_norm,
                },
            )
        )
    object_atoms = sorted(
        object_atoms,
        key=lambda atom: (
            _safe_float(atom.meta.get("importance_score", 0.0), 0.0),
            _safe_float(atom.meta.get("area_ratio", 0.0), 0.0),
        ),
        reverse=True,
    )
    deduped_object_atoms: List[EntityAtom] = []
    seen_boxes = _dedupe_boxes([atom.bbox_norm_xyxy for atom in object_atoms], iou_thr=0.88)
    for box in seen_boxes:
        chosen = next((atom for atom in object_atoms if atom.bbox_norm_xyxy == box), None)
        if chosen is not None:
            deduped_object_atoms.append(chosen)
    object_atoms = deduped_object_atoms[:3]
    object_area_pairs = [
        (atom, _safe_float(atom.meta.get("area_ratio", box_area(atom.bbox_norm_xyxy)), box_area(atom.bbox_norm_xyxy)))
        for atom in object_atoms
    ]
    total_object_area = sum(area for _, area in object_area_pairs)
    max_object_area = max((area for _, area in object_area_pairs), default=0.0)
    for rank, (atom, area) in enumerate(object_area_pairs):
        object_vs_person_ratio = float(area / max_person_area) if max_person_area > 1e-8 else None
        atom.meta.update(
            {
                "object_rank": rank,
                "object_area_share": float(area / total_object_area) if total_object_area > 1e-8 else 0.0,
                "object_size_rel": float(area / max_object_area) if max_object_area > 1e-8 else 0.0,
                "object_count": len(object_atoms),
                "max_person_area_ratio": float(max_person_area),
                "object_vs_person_ratio": object_vs_person_ratio,
            }
        )
    atoms.extend(object_atoms)

    if len(object_atoms) >= 2:
        union_box = union_boxes([atom.bbox_norm_xyxy for atom in object_atoms[:2]])
        if union_box is not None and box_area(union_box) > 0.0:
            top2_areas = [_safe_float(atom.meta.get("area_ratio", box_area(atom.bbox_norm_xyxy)), box_area(atom.bbox_norm_xyxy)) for atom in object_atoms[:2]]
            top2_area_sum = sum(top2_areas)
            top2_area_max = max(top2_areas) if top2_areas else 0.0
            top2_area_min = min(top2_areas) if top2_areas else 0.0
            atoms.append(
                EntityAtom(
                    entity_id="object_multi_top2",
                    entity_type="object_multi",
                    family="object",
                    bbox_norm_xyxy=union_box,
                    core_bbox_norm_xyxy=union_box,
                    envelope_bbox_norm_xyxy=union_box,
                    support_bbox_norm_xyxy=union_box,
                    member_boxes_norm_xyxy=[atom.bbox_norm_xyxy for atom in object_atoms[:2]],
                    meta={
                        "member_ids": [atom.entity_id for atom in object_atoms[:2]],
                        "member_count": 2,
                        "member_families": [atom.family for atom in object_atoms[:2]],
                        "animal_member_count": sum(1 for atom in object_atoms[:2] if atom.family == "animal"),
                        "top2_area_sum": float(top2_area_sum),
                        "top2_area_share": float(top2_area_sum / total_object_area) if total_object_area > 1e-8 else 0.0,
                        "member_balance": float(top2_area_min / top2_area_max) if top2_area_max > 1e-8 else 0.0,
                        "group_area_ratio": float(box_area(union_box)),
                        "max_person_area_ratio": float(max_person_area),
                        "top2_vs_person_ratio": float(top2_area_sum / max_person_area) if max_person_area > 1e-8 else None,
                    },
                )
            )

    scene_guidance_box = effective_subject_region.get("guidance_envelope_bbox_norm_xyxy")
    scene_effective_box = effective_subject_region.get("effective_bbox_norm_xyxy")
    scene_subject_state = str(effective_subject_region.get("state", "") or "").strip().lower()
    scene_subject_source = str(effective_subject_region.get("source", "") or "").strip().lower()
    scene_subject_reliability = _safe_float(effective_subject_region.get("subject_reliability"), 1.0)
    scene_subject_agreement_iou = _safe_float(effective_subject_region.get("subject_agreement_iou"), 1.0)
    scene_subject_reasons = effective_subject_region.get("reasons") if isinstance(effective_subject_region.get("reasons"), list) else []
    scene_guidance_box_norm = (
        [float(v) for v in norm_box_xyxy(scene_guidance_box, 1, 1)]
        if isinstance(scene_guidance_box, (list, tuple)) and len(scene_guidance_box) == 4
        else None
    )
    scene_effective_box_norm = (
        [float(v) for v in norm_box_xyxy(scene_effective_box, 1, 1)]
        if isinstance(scene_effective_box, (list, tuple)) and len(scene_effective_box) == 4
        else None
    )
    c7_saliency = feat_row.get("c7_saliency") if isinstance(feat_row.get("c7_saliency"), dict) else {}
    scene_saliency_fg_box = c7_saliency.get("foreground_bbox_norm_xyxy")
    if not isinstance(scene_saliency_fg_box, (list, tuple)) or len(scene_saliency_fg_box) != 4:
        scene_saliency_fg_box = None
    scene_saliency_fg_box_norm = (
        [float(v) for v in norm_box_xyxy(scene_saliency_fg_box, 1, 1)]
        if isinstance(scene_saliency_fg_box, (list, tuple)) and len(scene_saliency_fg_box) == 4
        else None
    )
    scene_guidance_area = box_area(scene_guidance_box_norm) if scene_guidance_box_norm is not None else 0.0
    scene_saliency_area = _safe_float(
        c7_saliency.get("foreground_area_ratio"),
        box_area(scene_saliency_fg_box_norm) if scene_saliency_fg_box_norm is not None else 0.0,
    )
    scene_effective_area = box_area(scene_effective_box_norm) if scene_effective_box_norm is not None else 0.0
    scene_residual_mass = max(
        _safe_float(effective_subject_region.get("semantic_residual_scene_mass"), 0.0),
        _safe_float(saliency_semantic.get("residual_scene_mass"), 0.0),
    )
    scene_support_trust = str(effective_subject_region.get("support_trust_tier") or saliency_semantic.get("support_trust_tier") or "").strip().lower()
    scene_severe_disagreement = (
        "detector_saliency_severe_disagreement" in scene_subject_source
        or "detector_saliency_severe_disagreement" in {str(reason) for reason in scene_subject_reasons}
    )
    scene_soft_anchor = "scene_soft_anchor" in scene_subject_source
    scene_missing_or_blank_saliency = scene_saliency_fg_box_norm is None or scene_saliency_area <= 0.005
    scene_full_frame_saliency = scene_saliency_area >= 0.70 or (
        scene_saliency_fg_box_norm is not None and box_area(scene_saliency_fg_box_norm) >= 0.90
    )
    scene_guidance_saliency_iou = (
        _box_iou(scene_guidance_box_norm, scene_saliency_fg_box_norm)
        if scene_guidance_box_norm is not None and scene_saliency_fg_box_norm is not None
        else 0.0
    )
    scene_guidance_tiny_saliency_mismatch = (
        scene_guidance_area >= 0.08
        and scene_saliency_area <= 0.02
        and scene_subject_agreement_iou <= 0.08
    )
    scene_guidance_saliency_mismatch = (
        scene_guidance_area >= 0.08
        and scene_guidance_saliency_iou <= 0.10
        and scene_subject_agreement_iou <= 0.08
    )
    scene_low_trust_guidance = (
        scene_subject_state in {"no_dominant_subject", "distributed_attention"}
        and scene_subject_reliability <= 0.35
        and scene_guidance_area > 0.0
        and (
            scene_severe_disagreement
            or (scene_soft_anchor and (scene_missing_or_blank_saliency or scene_residual_mass >= 0.85 or scene_support_trust == "low"))
            or (scene_full_frame_saliency and (scene_residual_mass >= 0.70 or scene_support_trust == "low" or scene_effective_area >= 0.80))
        )
    )
    scene_multi_low_reliability_disagreement = (
        scene_subject_state == "multi_subject"
        and scene_subject_reliability <= 0.35
        and scene_subject_agreement_iou <= 0.08
        and scene_severe_disagreement
    )
    scene_raw_anchor_saliency_mismatch = (
        scene_subject_state == "dominant_subject"
        and scene_subject_source == "raw_anchor"
        and scene_subject_reliability >= 0.70
        and (scene_guidance_tiny_saliency_mismatch or scene_guidance_saliency_mismatch)
    )
    scene_low_reliability_pseudo_subject = (
        scene_subject_state == "no_dominant_subject"
        and scene_severe_disagreement
        and scene_subject_reliability <= 0.25
        and scene_subject_agreement_iou <= 0.05
    ) or scene_low_trust_guidance or scene_multi_low_reliability_disagreement or scene_raw_anchor_saliency_mismatch
    scene_subject_core_source = "guidance_envelope"
    scene_subject_safe_enabled = True
    scene_subject_safe_suppressed_reason = ""
    if scene_low_reliability_pseudo_subject:
        scene_subject_box = [0.0, 0.0, 1.0, 1.0]
        scene_subject_core_source = "full_frame_low_reliability_scene"
        scene_subject_safe_enabled = False
        if scene_subject_state == "distributed_attention":
            scene_subject_safe_suppressed_reason = "low_reliability_distributed_scene_subject"
        elif scene_soft_anchor and scene_missing_or_blank_saliency:
            scene_subject_safe_suppressed_reason = "low_reliability_scene_soft_anchor_no_saliency"
        elif scene_multi_low_reliability_disagreement:
            scene_subject_safe_suppressed_reason = "low_reliability_multi_subject_scene_disagreement"
        elif scene_raw_anchor_saliency_mismatch:
            scene_subject_safe_suppressed_reason = "raw_anchor_saliency_mismatch_scene_subject"
        else:
            scene_subject_safe_suppressed_reason = "low_reliability_no_dominant_scene_subject"
    else:
        scene_subject_box = scene_guidance_box
        if not isinstance(scene_subject_box, (list, tuple)) or len(scene_subject_box) != 4:
            scene_subject_box = scene_effective_box
            scene_subject_core_source = "effective_subject_region"
        if not isinstance(scene_subject_box, (list, tuple)) or len(scene_subject_box) != 4:
            scene_subject_box = [0.0, 0.0, 1.0, 1.0]
            scene_subject_core_source = "full_frame_fallback"
            scene_subject_safe_enabled = False
            scene_subject_safe_suppressed_reason = "missing_scene_subject_region"
    scene_subject_box = [float(v) for v in norm_box_xyxy(scene_subject_box, 1, 1)]
    scene_full_box = [0.0, 0.0, 1.0, 1.0]
    scene_route_hint = str(routing.get("subject_mode", "")).strip().lower()
    scene_like = scene_route_hint in {"scene_general", "background_texture_copyspace"} or (
        bool(saliency_semantic.get("copyspace_intent", False))
        or _safe_float(saliency_semantic.get("residual_scene_mass", 0.0), 0.0) >= 0.20
        or not person_atoms
    )
    atoms.append(
        EntityAtom(
            entity_id="scene_main",
            entity_type="scene",
            family="scene",
            bbox_norm_xyxy=scene_full_box,
            core_bbox_norm_xyxy=scene_subject_box,
            envelope_bbox_norm_xyxy=scene_full_box,
            support_bbox_norm_xyxy=scene_subject_box,
            meta={
                "scene_subtype": routing.get("scene_subtype"),
                "copyspace_side": (routing.get("copyspace", {}) or {}).get("side", "none"),
                "copyspace_intent": bool((routing.get("copyspace", {}) or {}).get("gate_passed", False)),
                "scene_like_hint": bool(scene_like),
                "subject_mode": routing.get("subject_mode"),
                "scene_subject_state": scene_subject_state,
                "scene_subject_source": scene_subject_source,
                "scene_subject_reliability": float(scene_subject_reliability),
                "scene_subject_agreement_iou": float(scene_subject_agreement_iou),
                "scene_subject_core_source": scene_subject_core_source,
                "scene_low_reliability_pseudo_subject": bool(scene_low_reliability_pseudo_subject),
                "scene_subject_guidance_area": float(scene_guidance_area),
                "scene_subject_residual_scene_mass": float(scene_residual_mass),
                "scene_subject_support_trust_tier": scene_support_trust,
                "scene_subject_saliency_area": float(scene_saliency_area),
                "scene_subject_guidance_saliency_iou": float(scene_guidance_saliency_iou),
                "landscape_subject_safe_enabled": bool(scene_subject_safe_enabled),
                "landscape_subject_safe_suppressed_reason": scene_subject_safe_suppressed_reason,
                "scene_guidance_bbox_norm_xyxy": scene_guidance_box_norm,
                "scene_effective_bbox_norm_xyxy": scene_effective_box_norm,
                "scene_saliency_fg_bbox_norm_xyxy": scene_saliency_fg_box_norm,
            },
        )
    )
    return atoms
