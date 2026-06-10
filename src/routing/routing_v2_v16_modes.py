from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from routing.routing_v2_simple import PERSON_SHOT_TYPE, PLACEMENT_INTENT, ROUTE_FAMILY_V2


V16_SCHEMA_VERSION = "routing_v2_v16_mode_catalog"
IMAGE_ROUTE_NO_PLACEMENT_SCHEMA_VERSION = "routing_v2_image_route_no_placement_v1"


@dataclass(frozen=True)
class V16ModeSpec:
    category_id: int
    name: str
    route_family_v2: str
    person_shot_type: str | None
    placement_intent: str
    supercategory: str = "crop"


def _mode_name(route_family_v2: str, person_shot_type: str | None, placement_intent: str) -> str:
    if route_family_v2 == "person_single":
        return f"person_single_{person_shot_type}_{placement_intent}"
    return f"{route_family_v2}_{placement_intent}"


def _image_route_name(route_family_v2: str, person_shot_type: str | None) -> str:
    if route_family_v2 == "person_single":
        return f"person_single_{person_shot_type}"
    return route_family_v2


def build_v16_mode_specs() -> list[V16ModeSpec]:
    specs: list[V16ModeSpec] = []
    category_id = 0
    for route_family in ROUTE_FAMILY_V2:
        if route_family == "person_single":
            for shot in PERSON_SHOT_TYPE:
                for placement in PLACEMENT_INTENT:
                    specs.append(
                        V16ModeSpec(
                            category_id=category_id,
                            name=_mode_name(route_family, shot, placement),
                            route_family_v2=route_family,
                            person_shot_type=shot,
                            placement_intent=placement,
                        )
                    )
                    category_id += 1
        else:
            for placement in PLACEMENT_INTENT:
                specs.append(
                    V16ModeSpec(
                        category_id=category_id,
                        name=_mode_name(route_family, None, placement),
                        route_family_v2=route_family,
                        person_shot_type=None,
                        placement_intent=placement,
                    )
                )
                category_id += 1
    return specs


V16_MODE_SPECS = build_v16_mode_specs()
V16_MODE_BY_NAME = {spec.name: spec for spec in V16_MODE_SPECS}
V16_MODE_BY_KEY = {
    (spec.route_family_v2, spec.person_shot_type or "na", spec.placement_intent): spec
    for spec in V16_MODE_SPECS
}
IMAGE_ROUTE_NO_PLACEMENT_NAMES = [
    "scene",
    "person_single_face_headshot",
    "person_single_upper_half_body",
    "person_single_full_body",
    "person_group",
    "pet_dogcat",
]
IMAGE_ROUTE_NO_PLACEMENT_TO_ID = {name: idx for idx, name in enumerate(IMAGE_ROUTE_NO_PLACEMENT_NAMES)}


def v16_mode_from_routing(routing: dict[str, Any]) -> V16ModeSpec:
    route_family = str(routing.get("route_family_v2") or "scene")
    if route_family not in ROUTE_FAMILY_V2:
        route_family = "scene"
    shot = str(routing.get("person_shot_type") or "na")
    if route_family != "person_single":
        shot = "na"
    elif shot not in PERSON_SHOT_TYPE:
        shot = "upper_half_body"
    placement = str(routing.get("placement_intent") or "center")
    if placement not in PLACEMENT_INTENT:
        placement = "center"
    return V16_MODE_BY_KEY[(route_family, shot, placement)]


def image_route_no_placement_from_routing(routing: dict[str, Any]) -> str:
    route_family = str(routing.get("route_family_v2") or "scene")
    if route_family not in ROUTE_FAMILY_V2:
        route_family = "scene"
    shot: str | None = None
    if route_family == "person_single":
        shot = str(routing.get("person_shot_type") or "upper_half_body")
        if shot not in PERSON_SHOT_TYPE:
            shot = "upper_half_body"
    name = _image_route_name(route_family, shot)
    if name not in IMAGE_ROUTE_NO_PLACEMENT_TO_ID:
        return "scene"
    return name


def image_route_no_placement_id_from_routing(routing: dict[str, Any]) -> int:
    return IMAGE_ROUTE_NO_PLACEMENT_TO_ID[image_route_no_placement_from_routing(routing)]


def image_route_no_placement_catalog_payload() -> dict[str, Any]:
    return {
        "schema_version": IMAGE_ROUTE_NO_PLACEMENT_SCHEMA_VERSION,
        "target_policy": "image-level routing target excludes placement_intent",
        "route_count": len(IMAGE_ROUTE_NO_PLACEMENT_NAMES),
        "route_names": list(IMAGE_ROUTE_NO_PLACEMENT_NAMES),
        "route_to_id": dict(IMAGE_ROUTE_NO_PLACEMENT_TO_ID),
        "notes": {
            "included_in_image_route_name": ["route_family_v2", "person_shot_type when route_family_v2 == person_single"],
            "excluded_from_image_route_name": ["placement_intent", "context_intent", "mode_feasible"],
            "environmental_portrait_policy": "not a standalone image route or v16 mode class; encode as context_intent=environmental with body shot face_headshot/upper_half_body/full_body",
            "food_policy": "food/tableware evidence is retained only as audit/probe metadata and is folded into scene because food is not an active route class in this taxonomy.",
            "placement_intent_policy": "use only as query/crop intent or ablation/auxiliary metadata, not as the primary image-level class target",
        },
    }


def v16_categories() -> list[dict[str, Any]]:
    return [
        {
            "id": spec.category_id,
            "name": spec.name,
            "supercategory": spec.supercategory,
            "route_family_v2": spec.route_family_v2,
            "person_shot_type": spec.person_shot_type,
            "placement_intent": spec.placement_intent,
        }
        for spec in V16_MODE_SPECS
    ]


def v16_mode_catalog_payload() -> dict[str, Any]:
    return {
        "schema_version": V16_SCHEMA_VERSION,
        "mode_count": len(V16_MODE_SPECS),
        "mode_specs": [asdict(spec) for spec in V16_MODE_SPECS],
        "image_level_route_no_placement": image_route_no_placement_catalog_payload(),
        "notes": {
            "mode_key": "route_family_v2 + person_shot_type(applicable only for person_single) + placement_intent",
            "mode_key_policy": "v16 mode names are query/crop mode names; image-level route targets must use image_route_name_no_placement",
            "excluded_from_mode_name": ["context_intent", "mode_feasible"],
            "route_family_v2": list(ROUTE_FAMILY_V2),
            "person_shot_type": list(PERSON_SHOT_TYPE),
            "placement_intent": list(PLACEMENT_INTENT),
        },
    }
