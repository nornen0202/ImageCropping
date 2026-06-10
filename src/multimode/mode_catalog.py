from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass(frozen=True)
class ModeSpec:
    category_id: int
    name: str
    supercategory: str
    entity_type: str
    placement: str
    tau_pos: float
    subject_ratio_target: float
    subject_ratio_sigma: float
    context_target: float
    context_sigma: float
    route_aliases: Sequence[str]
    family_aliases: Sequence[str]


MODE_SPECS: List[ModeSpec] = [
    ModeSpec(
        category_id=0,
        name="landscape",
        supercategory="crop",
        entity_type="scene",
        placement="center",
        tau_pos=0.47,
        subject_ratio_target=0.16,
        subject_ratio_sigma=0.12,
        context_target=0.78,
        context_sigma=0.18,
        route_aliases=("scene_general", "background_texture_copyspace"),
        family_aliases=("scene",),
    ),
    ModeSpec(
        category_id=1,
        name="single_person_center",
        supercategory="crop",
        entity_type="person",
        placement="center",
        tau_pos=0.62,
        subject_ratio_target=0.24,
        subject_ratio_sigma=0.14,
        context_target=0.72,
        context_sigma=0.18,
        route_aliases=("portrait_single",),
        family_aliases=("human", "person"),
    ),
    ModeSpec(
        category_id=2,
        name="single_person_rot",
        supercategory="crop",
        entity_type="person",
        placement="rot",
        tau_pos=0.62,
        subject_ratio_target=0.22,
        subject_ratio_sigma=0.14,
        context_target=0.76,
        context_sigma=0.18,
        route_aliases=("portrait_single",),
        family_aliases=("human", "person"),
    ),
    ModeSpec(
        category_id=3,
        name="group_center",
        supercategory="crop",
        entity_type="group",
        placement="center",
        tau_pos=0.60,
        subject_ratio_target=0.36,
        subject_ratio_sigma=0.14,
        context_target=0.50,
        context_sigma=0.15,
        route_aliases=("portrait_group",),
        family_aliases=("human", "person"),
    ),
    ModeSpec(
        category_id=4,
        name="group_rot",
        supercategory="crop",
        entity_type="group",
        placement="rot",
        tau_pos=0.60,
        subject_ratio_target=0.33,
        subject_ratio_sigma=0.14,
        context_target=0.54,
        context_sigma=0.16,
        route_aliases=("portrait_group",),
        family_aliases=("human", "person"),
    ),
    ModeSpec(
        category_id=5,
        name="face",
        supercategory="crop",
        entity_type="face",
        placement="center",
        tau_pos=0.68,
        subject_ratio_target=0.24,
        subject_ratio_sigma=0.08,
        context_target=0.76,
        context_sigma=0.12,
        route_aliases=("portrait_single", "portrait_group"),
        family_aliases=("human", "person"),
    ),
    ModeSpec(
        category_id=6,
        name="object_single_center",
        supercategory="crop",
        entity_type="object",
        placement="center",
        tau_pos=0.58,
        subject_ratio_target=0.46,
        subject_ratio_sigma=0.15,
        context_target=0.50,
        context_sigma=0.16,
        route_aliases=("object_single",),
        family_aliases=("object", "animal"),
    ),
    ModeSpec(
        category_id=7,
        name="object_single_rot",
        supercategory="crop",
        entity_type="object",
        placement="rot",
        tau_pos=0.58,
        subject_ratio_target=0.42,
        subject_ratio_sigma=0.15,
        context_target=0.54,
        context_sigma=0.16,
        route_aliases=("object_single",),
        family_aliases=("object", "animal"),
    ),
    ModeSpec(
        category_id=8,
        name="object_multi_center",
        supercategory="crop",
        entity_type="object_multi",
        placement="center",
        tau_pos=0.57,
        subject_ratio_target=0.30,
        subject_ratio_sigma=0.14,
        context_target=0.56,
        context_sigma=0.18,
        route_aliases=("object_multi",),
        family_aliases=("object", "animal"),
    ),
    ModeSpec(
        category_id=9,
        name="object_multi_rot",
        supercategory="crop",
        entity_type="object_multi",
        placement="rot",
        tau_pos=0.57,
        subject_ratio_target=0.28,
        subject_ratio_sigma=0.14,
        context_target=0.60,
        context_sigma=0.18,
        route_aliases=("object_multi",),
        family_aliases=("object", "animal"),
    ),
]

MODE_SPEC_BY_NAME: Dict[str, ModeSpec] = {spec.name: spec for spec in MODE_SPECS}
BASE_CATEGORIES: List[Dict[str, object]] = [
    {
        "id": spec.category_id,
        "name": spec.name,
        "supercategory": spec.supercategory,
    }
    for spec in MODE_SPECS
]


def aligned_route_mode_names(route_mode: str) -> List[str]:
    route_norm = str(route_mode or "").strip().lower()
    return [spec.name for spec in MODE_SPECS if route_norm in {alias.lower() for alias in spec.route_aliases}]


def mode_alignment_bonus(mode_name: str, route_mode: str, route_family: str) -> float:
    spec = MODE_SPEC_BY_NAME[mode_name]
    route_norm = str(route_mode or "").strip().lower()
    family_norm = str(route_family or "").strip().lower()
    if route_norm in {alias.lower() for alias in spec.route_aliases}:
        return 0.03
    if family_norm in {alias.lower() for alias in spec.family_aliases}:
        return 0.01
    return 0.0


def is_person_mode(mode_name: str) -> bool:
    return mode_name in {"single_person_center", "single_person_rot", "group_center", "group_rot", "face"}


def is_object_mode(mode_name: str) -> bool:
    return mode_name.startswith("object_")
