from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from .entity_atoms import EntityAtom
from .mode_catalog import MODE_SPEC_BY_NAME, ModeSpec


@dataclass
class ModeQuery:
    query_id: str
    mode_name: str
    mode_id: int
    entity_id: str
    entity_type: str
    target_ar: str
    route_mode: str
    route_family: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    core_bbox_norm_xyxy: List[float] = field(default_factory=list)
    envelope_bbox_norm_xyxy: List[float] = field(default_factory=list)
    secondary_core_bbox_norm_xyxy: List[float] | None = None
    anchor_bbox_norm_xyxy: List[float] = field(default_factory=list)
    face_bbox_norm_xyxy: List[float] | None = None
    head_bbox_norm_xyxy: List[float] | None = None
    member_boxes_norm_xyxy: List[List[float]] = field(default_factory=list)
    member_face_boxes_norm_xyxy: List[List[float]] = field(default_factory=list)
    member_keypoints_norm: List[List[List[float]]] = field(default_factory=list)
    head_y_norm: List[float] = field(default_factory=list)
    gaze_entries: List[Dict[str, Any]] = field(default_factory=list)
    support_bbox_norm_xyxy: List[float] | None = None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(float(lo), min(float(hi), float(value)))


def _route_aligned(spec: ModeSpec, route_mode: str) -> bool:
    route_norm = str(route_mode or "").strip().lower()
    return route_norm in {alias.lower() for alias in spec.route_aliases}


def _route_bucket(route_mode: str, route_family: str) -> str:
    route_norm = str(route_mode or "").strip().lower()
    family_norm = str(route_family or "").strip().lower()
    if route_norm in {"portrait_single", "portrait_group"} or family_norm in {"human", "person"}:
        return "human"
    if route_norm in {"object_single", "object_multi"} or family_norm in {"object", "animal"}:
        return "object"
    if route_norm in {"scene_general", "background_texture_copyspace"} or family_norm == "scene":
        return "scene"
    if route_norm == "text_document":
        return "text"
    return "other"


def _secondary_person_exception(
    atom: EntityAtom,
    *,
    mode_name: str,
    route_mode: str,
    dominance: float,
    area_ratio: float,
    area_share: float,
) -> tuple[bool, Dict[str, Any]]:
    rank = int(_safe_float(atom.meta.get("person_rank"), 0))
    if rank <= 0:
        return True, {
            "secondary_person_policy": "primary",
            "secondary_person_exception": 0,
            "secondary_person_exception_rule": "primary_rank",
        }
    face_score = _safe_float(atom.meta.get("face_score"), 0.0)
    face_area_ratio = _safe_float(atom.meta.get("face_area_ratio"), 0.0)
    route_norm = str(route_mode or "").strip().lower()
    trusted = bool(atom.meta.get("person_atom_trusted", True))
    face_query_valid = bool(atom.meta.get("face_query_valid", True))
    size_rel = _safe_float(atom.meta.get("person_size_rel"), 0.0)
    if route_norm == "portrait_group" and trusted and rank <= 2:
        if mode_name == "face":
            allow = (
                face_query_valid
                and dominance >= 0.62
                and area_share >= 0.14
                and area_ratio >= 0.024
                and size_rel >= 0.32
                and face_score >= 0.80
                and face_area_ratio >= 0.006
            )
            return allow, {
                "secondary_person_policy": "primary_only_default_group_portrait_secondary_exception",
                "secondary_person_exception": int(bool(allow)),
                "secondary_person_exception_rule": (
                    "group_portrait_secondary_face_exception"
                    if allow
                    else "group_portrait_secondary_face_threshold_miss"
                ),
                "secondary_exception_min_dominance": 0.62,
                "secondary_exception_min_area_share": 0.14,
                "secondary_exception_min_area_ratio": 0.024,
                "secondary_exception_min_face_score": 0.80,
                "secondary_exception_min_face_area_ratio": 0.006,
                "secondary_exception_min_person_size_rel": 0.32,
            }
        allow = (
            dominance >= 0.64
            and area_share >= 0.16
            and area_ratio >= 0.030
            and size_rel >= 0.35
            and face_score >= 0.78
        )
        return allow, {
            "secondary_person_policy": "primary_only_default_group_portrait_secondary_exception",
            "secondary_person_exception": int(bool(allow)),
            "secondary_person_exception_rule": (
                "group_portrait_secondary_person_exception"
                if allow
                else "group_portrait_secondary_person_threshold_miss"
            ),
            "secondary_exception_min_dominance": 0.64,
            "secondary_exception_min_area_share": 0.16,
            "secondary_exception_min_area_ratio": 0.030,
            "secondary_exception_min_face_score": 0.78,
            "secondary_exception_min_face_area_ratio": None,
            "secondary_exception_min_person_size_rel": 0.35,
        }
    if mode_name == "face":
        allow = (
            rank == 1
            and dominance >= 0.90
            and area_share >= 0.45
            and face_score >= 0.94
            and face_area_ratio >= 0.020
        )
        rule = "dominant_secondary_face_exception"
    else:
        allow = (
            rank == 1
            and dominance >= 0.92
            and area_share >= 0.50
            and area_ratio >= 0.060
            and face_score >= 0.92
        )
        rule = "dominant_secondary_person_exception"
    return allow, {
        "secondary_person_policy": "primary_only_default_dominant_secondary_exception",
        "secondary_person_exception": int(bool(allow)),
        "secondary_person_exception_rule": rule if allow else "secondary_person_primary_only_default",
        "secondary_exception_min_dominance": 0.90 if mode_name == "face" else 0.92,
        "secondary_exception_min_area_share": 0.45 if mode_name == "face" else 0.50,
        "secondary_exception_min_area_ratio": None if mode_name == "face" else 0.060,
        "secondary_exception_min_face_score": 0.94 if mode_name == "face" else 0.92,
        "secondary_exception_min_face_area_ratio": 0.020 if mode_name == "face" else None,
    }


def _person_single_gate(atom: EntityAtom, spec: ModeSpec, route_mode: str, route_family: str) -> tuple[bool, Dict[str, Any]]:
    area_ratio = _safe_float(atom.meta.get("area_ratio"), 0.0)
    area_share = _safe_float(atom.meta.get("person_area_share"), 1.0)
    rank = int(_safe_float(atom.meta.get("person_rank"), 0))
    trusted = bool(atom.meta.get("person_atom_trusted", True))
    route_aligned = _route_aligned(spec, route_mode)
    route_bucket = _route_bucket(route_mode, route_family)
    rank_bonus = 1.0 if rank == 0 else 0.72 if rank == 1 else 0.50
    dominance = _clamp(
        0.45 * _clamp(area_ratio / 0.035)
        + 0.45 * _clamp(area_share / 0.40)
        + 0.10 * rank_bonus
    )
    secondary_allow, secondary_attrs = _secondary_person_exception(
        atom,
        mode_name=spec.name,
        route_mode=route_mode,
        dominance=dominance,
        area_ratio=area_ratio,
        area_share=area_share,
    )
    allow = True
    gate_rule = "aligned"
    gate_tau = 0.0
    if not trusted:
        gate_rule = "untrusted_person_atom"
        gate_tau = 1.0
        allow = False
    elif route_aligned and rank > 0:
        gate_rule = str(secondary_attrs["secondary_person_exception_rule"])
        gate_tau = _safe_float(secondary_attrs["secondary_exception_min_dominance"], 0.92)
        allow = secondary_allow
    elif not route_aligned:
        if str(route_mode or "").strip().lower() == "portrait_group":
            if rank > 0:
                gate_rule = str(secondary_attrs["secondary_person_exception_rule"])
                gate_tau = _safe_float(secondary_attrs["secondary_exception_min_dominance"], 0.92)
                allow = secondary_allow
            else:
                gate_rule = "portrait_group_prominent_person"
                gate_tau = 0.56
                allow = dominance >= gate_tau and area_share >= 0.16 and area_ratio >= 0.016
        elif route_bucket in {"object", "scene", "text"}:
            gate_rule = "nonhuman_prominent_person"
            gate_tau = 0.84
            allow = rank == 0 and dominance >= gate_tau and (area_ratio >= 0.024 or area_share >= 0.38)
        else:
            if rank > 0:
                gate_rule = str(secondary_attrs["secondary_person_exception_rule"])
                gate_tau = _safe_float(secondary_attrs["secondary_exception_min_dominance"], 0.92)
                allow = secondary_allow
            else:
                gate_rule = "fallback_prominent_person"
                gate_tau = 0.78
                allow = dominance >= gate_tau
    return allow, {
        "route_aligned": int(route_aligned),
        "route_bucket": route_bucket,
        "dominance_score": round(float(dominance), 6),
        "route_gate_tau": round(float(gate_tau), 6),
        "route_gate_rule": gate_rule,
        "person_atom_trusted": int(trusted),
        "person_rank": rank,
        "c2_person_iou": round(_safe_float(atom.meta.get("c2_person_iou"), 0.0), 6),
        "c2_person_recall": round(_safe_float(atom.meta.get("c2_person_recall"), 0.0), 6),
        "person_area_ratio": round(float(area_ratio), 6),
        "person_area_share": round(float(area_share), 6),
        "face_score": None if atom.meta.get("face_score") is None else round(_safe_float(atom.meta.get("face_score"), 0.0), 6),
        "atom_reject_reasons": list(atom.meta.get("atom_reject_reasons") or []),
        **secondary_attrs,
    }


def _object_single_gate(atom: EntityAtom, spec: ModeSpec, route_mode: str, route_family: str) -> tuple[bool, Dict[str, Any]]:
    area_ratio = _safe_float(atom.meta.get("area_ratio"), 0.0)
    importance = _safe_float(atom.meta.get("importance_score"), 0.0)
    area_share = _safe_float(atom.meta.get("object_area_share"), 1.0)
    size_rel = _safe_float(atom.meta.get("object_size_rel"), 1.0)
    rank = int(_safe_float(atom.meta.get("object_rank"), 0))
    route_aligned = _route_aligned(spec, route_mode)
    route_bucket = _route_bucket(route_mode, route_family)
    rank_bonus = 1.0 if rank == 0 else 0.60 if rank == 1 else 0.35
    dominance = _clamp(
        0.35 * _clamp(area_ratio / 0.03)
        + 0.30 * _clamp(importance / 1.0)
        + 0.25 * _clamp(area_share / 0.65)
        + 0.10 * rank_bonus
    )
    allow = True
    gate_rule = "aligned"
    gate_tau = 0.0
    route_norm = str(route_mode or "").strip().lower()
    if not route_aligned:
        if route_norm == "object_multi":
            gate_rule = "object_multi_dominant_top1"
            gate_tau = 0.80
            allow = dominance >= gate_tau and area_share >= 0.55 and area_ratio >= 0.022 and importance >= 0.95
        elif route_bucket == "scene":
            gate_rule = "scene_dominant_object"
            gate_tau = 0.86
            allow = dominance >= gate_tau and area_share >= 0.60 and area_ratio >= 0.020 and importance >= 0.90
        elif route_bucket in {"human", "text"}:
            object_vs_person = _safe_float(atom.meta.get("object_vs_person_ratio"), 0.0)
            if atom.family == "animal":
                gate_rule = "human_route_animal_companion_object"
                gate_tau = 0.72
                allow = (
                    dominance >= gate_tau
                    and area_share >= 0.35
                    and area_ratio >= 0.035
                    and importance >= 0.75
                    and (object_vs_person >= 0.12 or area_ratio >= 0.045)
                )
            else:
                gate_rule = "human_scene_override_object"
                gate_tau = 0.90
                allow = (
                    dominance >= gate_tau
                    and area_share >= 0.70
                    and area_ratio >= 0.030
                    and importance >= 0.95
                    and (object_vs_person >= 0.25 or area_ratio >= 0.12)
                )
        else:
            gate_rule = "fallback_dominant_object"
            gate_tau = 0.84
            allow = dominance >= gate_tau and area_share >= 0.58 and area_ratio >= 0.022
    return allow, {
        "route_aligned": int(route_aligned),
        "route_bucket": route_bucket,
        "dominance_score": round(float(dominance), 6),
        "route_gate_tau": round(float(gate_tau), 6),
        "route_gate_rule": gate_rule,
        "object_area_share": round(float(area_share), 6),
        "object_size_rel": round(float(size_rel), 6),
        "object_family": atom.family,
        "object_class_id": atom.meta.get("class_id"),
        "importance_score": round(float(importance), 6),
        "object_vs_person_ratio": (
            None
            if atom.meta.get("object_vs_person_ratio") is None
            else round(_safe_float(atom.meta.get("object_vs_person_ratio"), 0.0), 6)
        ),
    }


def _object_multi_gate(atom: EntityAtom, spec: ModeSpec, route_mode: str, route_family: str) -> tuple[bool, Dict[str, Any]]:
    top2_area_sum = _safe_float(atom.meta.get("top2_area_sum"), 0.0)
    top2_share = _safe_float(atom.meta.get("top2_area_share"), 1.0)
    member_balance = _safe_float(atom.meta.get("member_balance"), 0.0)
    route_aligned = _route_aligned(spec, route_mode)
    route_bucket = _route_bucket(route_mode, route_family)
    dominance = _clamp(
        0.45 * _clamp(top2_area_sum / 0.07)
        + 0.35 * _clamp(member_balance / 0.68)
        + 0.20 * _clamp(top2_share / 0.90)
    )
    allow = True
    gate_rule = "aligned"
    gate_tau = 0.0
    route_norm = str(route_mode or "").strip().lower()
    if not route_aligned:
        if route_norm == "object_single":
            gate_rule = "object_single_promote_to_multi"
            gate_tau = 0.82
            allow = dominance >= gate_tau and member_balance >= 0.62 and top2_area_sum >= 0.055
        elif route_bucket == "scene":
            gate_rule = "scene_object_pair"
            gate_tau = 0.82
            allow = dominance >= gate_tau and member_balance >= 0.58 and top2_area_sum >= 0.040 and top2_share >= 0.72
        elif route_bucket in {"human", "text"}:
            member_families = [str(value) for value in (atom.meta.get("member_families") or [])]
            animal_member_count = int(_safe_float(atom.meta.get("animal_member_count"), 0))
            top2_vs_person = _safe_float(atom.meta.get("top2_vs_person_ratio"), 0.0)
            if member_families and animal_member_count == len(member_families):
                gate_rule = "human_route_animal_pair_exception"
                gate_tau = 0.82
                allow = (
                    dominance >= gate_tau
                    and member_balance >= 0.58
                    and top2_area_sum >= 0.060
                    and (top2_vs_person >= 0.18 or top2_area_sum >= 0.10)
                )
            else:
                gate_rule = "human_route_object_pair_disabled"
                gate_tau = 1.01
                allow = False
        else:
            gate_rule = "fallback_object_pair"
            gate_tau = 0.84
            allow = dominance >= gate_tau
    return allow, {
        "route_aligned": int(route_aligned),
        "route_bucket": route_bucket,
        "dominance_score": round(float(dominance), 6),
        "route_gate_tau": round(float(gate_tau), 6),
        "route_gate_rule": gate_rule,
        "member_balance": round(float(member_balance), 6),
        "top2_area_sum": round(float(top2_area_sum), 6),
        "top2_area_share": round(float(top2_share), 6),
        "member_families": list(atom.meta.get("member_families") or []),
        "animal_member_count": int(_safe_float(atom.meta.get("animal_member_count"), 0)),
        "top2_vs_person_ratio": (
            None
            if atom.meta.get("top2_vs_person_ratio") is None
            else round(_safe_float(atom.meta.get("top2_vs_person_ratio"), 0.0), 6)
        ),
    }


def _query_gate_attributes(atom: EntityAtom, spec: ModeSpec, route_mode: str, route_family: str) -> tuple[bool, Dict[str, Any]]:
    if spec.name in {"single_person_center", "single_person_rot"}:
        return _person_single_gate(atom, spec, route_mode, route_family)
    if spec.name in {"object_single_center", "object_single_rot"}:
        return _object_single_gate(atom, spec, route_mode, route_family)
    if spec.name in {"object_multi_center", "object_multi_rot"}:
        return _object_multi_gate(atom, spec, route_mode, route_family)
    route_aligned = _route_aligned(spec, route_mode)
    return True, {
        "route_aligned": int(route_aligned),
        "route_bucket": _route_bucket(route_mode, route_family),
        "dominance_score": 1.0,
        "route_gate_tau": 0.0,
        "route_gate_rule": "always",
    }


def _group_atom_attributes(atom: EntityAtom) -> Dict[str, Any]:
    keys = (
        "member_ids",
        "member_count",
        "raw_member_count",
        "group_area_ratio",
        "group_gate_rule",
        "group_gate_allow",
        "group_distinct_member_count",
        "group_raw_person_atom_count",
        "group_distinct_face_count",
        "group_routed_person_count",
        "group_c2_person_count",
        "group_c2_raw_person_count",
        "group_c3_supported_count",
        "group_semantic_human_cluster_count",
        "group_multi_human_evidence_count",
        "group_evidence_signal_count",
        "group_fallback_atom",
        "group_fallback_rule",
        "group_fallback_raw_trusted_count",
        "group_fallback_duplicate_suppressed_count",
        "group_fallback_raw_union_area_ratio",
        "group_fallback_raw_union_to_max_area",
        "group_fallback_person_union_area_ratio",
        "group_fallback_multi_subject_flag",
        "group_fallback_face_count",
    )
    return {key: atom.meta.get(key) for key in keys if key in atom.meta}


def _scene_atom_attributes(atom: EntityAtom) -> Dict[str, Any]:
    keys = (
        "scene_subject_state",
        "scene_subject_source",
        "scene_subject_reliability",
        "scene_subject_agreement_iou",
        "scene_subject_core_source",
        "scene_low_reliability_pseudo_subject",
        "scene_subject_guidance_area",
        "scene_subject_residual_scene_mass",
        "scene_subject_support_trust_tier",
        "scene_subject_saliency_area",
        "scene_subject_guidance_saliency_iou",
        "landscape_subject_safe_enabled",
        "landscape_subject_safe_suppressed_reason",
        "scene_guidance_bbox_norm_xyxy",
        "scene_effective_bbox_norm_xyxy",
        "scene_saliency_fg_bbox_norm_xyxy",
    )
    return {key: atom.meta.get(key) for key in keys if key in atom.meta}


def _mode_query(
    *,
    image_id: str,
    target_ar: str,
    atom: EntityAtom,
    spec: ModeSpec,
    route_mode: str,
    route_family: str,
    extra_attributes: Dict[str, Any] | None = None,
) -> ModeQuery:
    query_id = f"{image_id}::{target_ar}::{spec.name}::{atom.entity_id}"
    attributes = {
        "atom_family": atom.family,
        "atom_entity_type": atom.entity_type,
    }
    portrait_comp = atom.meta.get("portrait_comp")
    if isinstance(portrait_comp, dict):
        attributes["portrait_comp"] = dict(portrait_comp)
    if extra_attributes:
        attributes.update(extra_attributes)
    return ModeQuery(
        query_id=query_id,
        mode_name=spec.name,
        mode_id=spec.category_id,
        entity_id=atom.entity_id,
        entity_type=atom.entity_type,
        target_ar=target_ar,
        route_mode=route_mode,
        route_family=route_family,
        attributes=attributes,
        core_bbox_norm_xyxy=list(atom.core_bbox_norm_xyxy),
        envelope_bbox_norm_xyxy=list(atom.envelope_bbox_norm_xyxy),
        anchor_bbox_norm_xyxy=list(atom.bbox_norm_xyxy),
        face_bbox_norm_xyxy=(list(atom.face_bbox_norm_xyxy) if atom.face_bbox_norm_xyxy is not None else None),
        head_bbox_norm_xyxy=(list(atom.head_bbox_norm_xyxy) if atom.head_bbox_norm_xyxy is not None else None),
        member_boxes_norm_xyxy=[list(box) for box in atom.member_boxes_norm_xyxy],
        member_face_boxes_norm_xyxy=[list(box) for box in atom.member_face_boxes_norm_xyxy],
        member_keypoints_norm=[[list(kp) for kp in kps] for kps in atom.member_keypoints_norm],
        head_y_norm=list(atom.head_y_norm),
        gaze_entries=[dict(entry) for entry in atom.gaze_entries],
        support_bbox_norm_xyxy=(list(atom.support_bbox_norm_xyxy) if atom.support_bbox_norm_xyxy is not None else None),
    )


def build_mode_queries(
    *,
    image_id: str,
    target_ar: str,
    atoms: Sequence[EntityAtom],
    routing: Dict[str, Any],
) -> List[ModeQuery]:
    route_mode = str(routing.get("subject_mode", "") or "")
    route_family = str(routing.get("subject_family", "") or "")
    queries: List[ModeQuery] = []
    for atom in atoms:
        if atom.entity_type == "person":
            single_center_allow, single_center_attrs = _query_gate_attributes(
                atom, MODE_SPEC_BY_NAME["single_person_center"], route_mode, route_family
            )
            if single_center_allow:
                queries.append(
                    _mode_query(
                        image_id=image_id,
                        target_ar=target_ar,
                        atom=atom,
                        spec=MODE_SPEC_BY_NAME["single_person_center"],
                        route_mode=route_mode,
                        route_family=route_family,
                        extra_attributes=single_center_attrs,
                    )
                )
            single_rot_allow, single_rot_attrs = _query_gate_attributes(
                atom, MODE_SPEC_BY_NAME["single_person_rot"], route_mode, route_family
            )
            if single_rot_allow:
                queries.append(
                    _mode_query(
                        image_id=image_id,
                        target_ar=target_ar,
                        atom=atom,
                        spec=MODE_SPEC_BY_NAME["single_person_rot"],
                        route_mode=route_mode,
                        route_family=route_family,
                        extra_attributes=single_rot_attrs,
                    )
                )
            if atom.face_bbox_norm_xyxy is not None and bool(atom.meta.get("face_query_valid", True)):
                _, face_attrs = _query_gate_attributes(atom, MODE_SPEC_BY_NAME["face"], route_mode, route_family)
                face_area_ratio = _safe_float(atom.meta.get("face_area_ratio"), 0.0)
                person_area_share = _safe_float(atom.meta.get("person_area_share"), 1.0)
                person_area_ratio = _safe_float(atom.meta.get("area_ratio"), 0.0)
                face_rank = int(_safe_float(atom.meta.get("person_rank"), 0))
                face_dominance = _clamp(
                    0.45 * _clamp(face_area_ratio / 0.030)
                    + 0.35 * _clamp(person_area_share / 0.40)
                    + 0.20 * (1.0 if face_rank == 0 else 0.65 if face_rank == 1 else 0.35)
                )
                face_secondary_allow, face_secondary_attrs = _secondary_person_exception(
                    atom,
                    mode_name="face",
                    route_mode=route_mode,
                    dominance=face_dominance,
                    area_ratio=person_area_ratio,
                    area_share=person_area_share,
                )
                if face_rank > 0 and not face_secondary_allow:
                    continue
                face_attrs = {
                    **face_attrs,
                    "person_atom_trusted": int(bool(atom.meta.get("person_atom_trusted", True))),
                    "person_rank": face_rank,
                    "face_query_valid": int(bool(atom.meta.get("face_query_valid", True))),
                    "face_query_fallback_used": int(bool(atom.meta.get("face_query_fallback_used", False))),
                    "face_query_original_bbox_norm_xyxy": atom.meta.get("face_query_original_bbox_norm_xyxy"),
                    "face_score": (
                        None
                        if atom.meta.get("face_score") is None
                        else round(_safe_float(atom.meta.get("face_score"), 0.0), 6)
                    ),
                    "face_area_ratio": round(face_area_ratio, 6),
                    "face_to_pose_area_ratio": round(_safe_float(atom.meta.get("face_to_pose_area_ratio"), 0.0), 6),
                    "person_area_ratio": round(person_area_ratio, 6),
                    "person_area_share": round(person_area_share, 6),
                    "dominance_score": round(float(face_dominance), 6),
                    "atom_reject_reasons": list(atom.meta.get("atom_reject_reasons") or []),
                    **face_secondary_attrs,
                }
                face_query = _mode_query(
                    image_id=image_id,
                    target_ar=target_ar,
                    atom=atom,
                    spec=MODE_SPEC_BY_NAME["face"],
                    route_mode=route_mode,
                    route_family=route_family,
                    extra_attributes=face_attrs,
                )
                face_subject_box = (
                    list(atom.head_bbox_norm_xyxy)
                    if atom.head_bbox_norm_xyxy is not None
                    else list(atom.face_bbox_norm_xyxy)
                )
                face_query.core_bbox_norm_xyxy = list(face_subject_box)
                face_query.envelope_bbox_norm_xyxy = list(face_subject_box)
                face_query.anchor_bbox_norm_xyxy = list(face_subject_box)
                face_query.support_bbox_norm_xyxy = list(face_subject_box)
                queries.append(face_query)
        elif atom.entity_type == "group":
            group_center_allow, group_center_attrs = _query_gate_attributes(
                atom, MODE_SPEC_BY_NAME["group_center"], route_mode, route_family
            )
            group_center_attrs = {**group_center_attrs, **_group_atom_attributes(atom)}
            if group_center_allow:
                queries.append(
                    _mode_query(
                        image_id=image_id,
                        target_ar=target_ar,
                        atom=atom,
                        spec=MODE_SPEC_BY_NAME["group_center"],
                        route_mode=route_mode,
                        route_family=route_family,
                        extra_attributes=group_center_attrs,
                    )
                )
            group_rot_allow, group_rot_attrs = _query_gate_attributes(
                atom, MODE_SPEC_BY_NAME["group_rot"], route_mode, route_family
            )
            group_rot_attrs = {**group_rot_attrs, **_group_atom_attributes(atom)}
            if group_rot_allow:
                queries.append(
                    _mode_query(
                        image_id=image_id,
                        target_ar=target_ar,
                        atom=atom,
                        spec=MODE_SPEC_BY_NAME["group_rot"],
                        route_mode=route_mode,
                        route_family=route_family,
                        extra_attributes=group_rot_attrs,
                    )
                )
        elif atom.entity_type == "object":
            obj_center_allow, obj_center_attrs = _query_gate_attributes(
                atom, MODE_SPEC_BY_NAME["object_single_center"], route_mode, route_family
            )
            if obj_center_allow:
                queries.append(
                    _mode_query(
                        image_id=image_id,
                        target_ar=target_ar,
                        atom=atom,
                        spec=MODE_SPEC_BY_NAME["object_single_center"],
                        route_mode=route_mode,
                        route_family=route_family,
                        extra_attributes={"object_family": atom.family, **obj_center_attrs},
                    )
                )
            obj_rot_allow, obj_rot_attrs = _query_gate_attributes(
                atom, MODE_SPEC_BY_NAME["object_single_rot"], route_mode, route_family
            )
            if obj_rot_allow:
                queries.append(
                    _mode_query(
                        image_id=image_id,
                        target_ar=target_ar,
                        atom=atom,
                        spec=MODE_SPEC_BY_NAME["object_single_rot"],
                        route_mode=route_mode,
                        route_family=route_family,
                        extra_attributes={"object_family": atom.family, **obj_rot_attrs},
                    )
                )
        elif atom.entity_type == "object_multi":
            objm_center_allow, objm_center_attrs = _query_gate_attributes(
                atom, MODE_SPEC_BY_NAME["object_multi_center"], route_mode, route_family
            )
            if objm_center_allow:
                queries.append(
                    _mode_query(
                        image_id=image_id,
                        target_ar=target_ar,
                        atom=atom,
                        spec=MODE_SPEC_BY_NAME["object_multi_center"],
                        route_mode=route_mode,
                        route_family=route_family,
                        extra_attributes=objm_center_attrs,
                    )
                )
            objm_rot_allow, objm_rot_attrs = _query_gate_attributes(
                atom, MODE_SPEC_BY_NAME["object_multi_rot"], route_mode, route_family
            )
            if objm_rot_allow:
                queries.append(
                    _mode_query(
                        image_id=image_id,
                        target_ar=target_ar,
                        atom=atom,
                        spec=MODE_SPEC_BY_NAME["object_multi_rot"],
                        route_mode=route_mode,
                        route_family=route_family,
                        extra_attributes=objm_rot_attrs,
                    )
                )
        elif atom.entity_type == "scene":
            _, scene_attrs = _query_gate_attributes(atom, MODE_SPEC_BY_NAME["landscape"], route_mode, route_family)
            queries.append(
                _mode_query(
                    image_id=image_id,
                    target_ar=target_ar,
                    atom=atom,
                    spec=MODE_SPEC_BY_NAME["landscape"],
                    route_mode=route_mode,
                    route_family=route_family,
                    extra_attributes={
                        "scene_subtype": atom.meta.get("scene_subtype"),
                        "copyspace_side": atom.meta.get("copyspace_side", "none"),
                        "copyspace_intent": int(bool(atom.meta.get("copyspace_intent", False))),
                        **_scene_atom_attributes(atom),
                        **scene_attrs,
                    },
                )
            )
    return queries
