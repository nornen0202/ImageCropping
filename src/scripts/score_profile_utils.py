from __future__ import annotations

import copy
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional


PROFILE_PRESETS: Dict[str, Dict[str, Any]] = {
    "current_refined": {
        "label": "Current Refined",
        "stage": 0,
        "description": "현재 refined monotonic scorer baseline.",
        "overrides": {},
    },
    "priority_stage1": {
        "label": "Priority Stage 1",
        "stage": 1,
        "description": "1순위 항목만 사용: S축 전체 + safety + C_place/C_headroom/C_lookroom/C_context.",
        "overrides": {
            "rank_weight_a": 0.0,
            "rank_weight_s": 1.20,
            "rank_weight_c": 0.95,
            "rank_weight_t": 0.0,
            "policy_area_weight_scale": 0.90,
            "c_macro_place_weight_scale": 1.50,
            "c_macro_comp_weight_scale": 0.0,
            "c_macro_headroom_weight_scale": 1.0,
            "c_macro_lookroom_weight_scale": 1.0,
            "c_macro_horizon_weight_scale": 0.0,
            "c_macro_sym_weight_scale": 0.0,
            "c_macro_context_weight_scale": 1.10,
            "c_macro_copyspace_weight_scale": 0.0,
        },
    },
    "priority_stage2": {
        "label": "Priority Stage 2",
        "stage": 2,
        "description": "Stage1 + 2순위 A/T 축 추가.",
        "overrides": {
            "rank_weight_a": 0.75,
            "rank_weight_s": 1.20,
            "rank_weight_c": 0.95,
            "rank_weight_t": 0.10,
            "policy_area_weight_scale": 0.90,
            "c_macro_place_weight_scale": 1.50,
            "c_macro_comp_weight_scale": 0.0,
            "c_macro_headroom_weight_scale": 1.0,
            "c_macro_lookroom_weight_scale": 1.0,
            "c_macro_horizon_weight_scale": 0.0,
            "c_macro_sym_weight_scale": 0.0,
            "c_macro_context_weight_scale": 1.10,
            "c_macro_copyspace_weight_scale": 0.0,
        },
    },
    "priority_stage3": {
        "label": "Priority Stage 3",
        "stage": 3,
        "description": "Stage2 + 3순위 C_comp/C_horizon_y/C_sym 추가.",
        "overrides": {
            "rank_weight_a": 0.75,
            "rank_weight_s": 1.15,
            "rank_weight_c": 1.00,
            "rank_weight_t": 0.10,
            "policy_area_weight_scale": 0.95,
            "c_macro_place_weight_scale": 1.50,
            "c_macro_comp_weight_scale": 0.55,
            "c_macro_headroom_weight_scale": 1.0,
            "c_macro_lookroom_weight_scale": 1.0,
            "c_macro_horizon_weight_scale": 0.60,
            "c_macro_sym_weight_scale": 0.50,
            "c_macro_context_weight_scale": 1.10,
            "c_macro_copyspace_weight_scale": 0.0,
        },
    },
    "priority_stage4": {
        "label": "Priority Stage 4",
        "stage": 4,
        "description": "Stage3 + 4순위 C_copyspace 추가. 우선순위 기반 full scorer.",
        "overrides": {
            "rank_weight_a": 0.75,
            "rank_weight_s": 1.15,
            "rank_weight_c": 1.00,
            "rank_weight_t": 0.10,
            "policy_area_weight_scale": 0.95,
            "c_macro_place_weight_scale": 1.50,
            "c_macro_comp_weight_scale": 0.55,
            "c_macro_headroom_weight_scale": 1.0,
            "c_macro_lookroom_weight_scale": 1.0,
            "c_macro_horizon_weight_scale": 0.60,
            "c_macro_sym_weight_scale": 0.50,
            "c_macro_context_weight_scale": 1.10,
            "c_macro_copyspace_weight_scale": 0.35,
        },
    },
}


def _load_override_json(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    json_path = Path(path)
    if not json_path.exists():
        raise FileNotFoundError(f"score profile override json not found: {json_path}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"score profile override json must be a JSON object: {json_path}")
    return payload


def cfg_to_dict(cfg: Any) -> Dict[str, Any]:
    if is_dataclass(cfg):
        return asdict(cfg)
    if hasattr(cfg, "__dict__"):
        return {key: copy.deepcopy(value) for key, value in vars(cfg).items() if not key.startswith("_")}
    raise TypeError("cfg_to_dict expects a dataclass or object with __dict__")


def resolve_profile_spec(
    *,
    profile_name: str,
    override_json_path: str = "",
) -> Dict[str, Any]:
    preset = PROFILE_PRESETS.get(str(profile_name).strip(), PROFILE_PRESETS["current_refined"])
    overrides = copy.deepcopy(preset.get("overrides", {}))
    overrides.update(_load_override_json(override_json_path))
    return {
        "profile_name": str(profile_name).strip() or "current_refined",
        "label": str(preset.get("label", profile_name)),
        "stage": int(preset.get("stage", 0)),
        "description": str(preset.get("description", "")),
        "overrides": overrides,
        "override_json_path": str(override_json_path or ""),
    }


def apply_profile_to_cfg(
    cfg: Any,
    *,
    profile_name: str,
    override_json_path: str = "",
) -> tuple[Any, Dict[str, Any]]:
    spec = resolve_profile_spec(profile_name=profile_name, override_json_path=override_json_path)
    out = copy.deepcopy(cfg)
    for key, value in spec["overrides"].items():
        if not hasattr(out, key):
            raise AttributeError(f"unknown scorer cfg override: {key}")
        setattr(out, key, value)
    return out, spec


def active_score_components(cfg: Any) -> Dict[str, Any]:
    out = {
        "macros": {
            "A_macro": float(getattr(cfg, "rank_weight_a", 0.0)) > 0.0,
            "S_macro": float(getattr(cfg, "rank_weight_s", 0.0)) > 0.0,
            "C_macro": float(getattr(cfg, "rank_weight_c", 0.0)) > 0.0,
            "T_macro": float(getattr(cfg, "rank_weight_t", 0.0)) > 0.0,
        },
        "components": {
            "A_aesthetic": float(getattr(cfg, "rank_weight_a", 0.0)) > 0.0 and float(getattr(cfg, "a_macro_aesthetic_weight", 0.0)) > 0.0,
            "A_align": float(getattr(cfg, "rank_weight_a", 0.0)) > 0.0 and float(getattr(cfg, "a_macro_align_weight", 0.0)) > 0.0,
            "S_cov": float(getattr(cfg, "rank_weight_s", 0.0)) > 0.0 and float(getattr(cfg, "s_macro_cov_weight_scale", 1.0)) > 0.0,
            "S_scale": float(getattr(cfg, "rank_weight_s", 0.0)) > 0.0 and float(getattr(cfg, "s_macro_scale_weight_scale", 1.0)) > 0.0,
            "S_support_structure": float(getattr(cfg, "rank_weight_s", 0.0)) > 0.0 and float(getattr(cfg, "s_macro_support_weight_scale", 1.0)) > 0.0,
            "S_border": float(getattr(cfg, "rank_weight_s", 0.0)) > 0.0 and float(getattr(cfg, "s_macro_border_weight_scale", 1.0)) > 0.0,
            "S_softcut_quality": float(getattr(cfg, "rank_weight_s", 0.0)) > 0.0 and float(getattr(cfg, "s_macro_softcut_weight_scale", 1.0)) > 0.0,
            "C_place": float(getattr(cfg, "rank_weight_c", 0.0)) > 0.0 and float(getattr(cfg, "c_macro_place_weight_scale", 0.0)) > 0.0,
            "C_comp": float(getattr(cfg, "rank_weight_c", 0.0)) > 0.0 and float(getattr(cfg, "c_macro_comp_weight_scale", 0.0)) > 0.0,
            "C_headroom": float(getattr(cfg, "rank_weight_c", 0.0)) > 0.0 and float(getattr(cfg, "c_macro_headroom_weight_scale", 1.0)) > 0.0,
            "C_lookroom": float(getattr(cfg, "rank_weight_c", 0.0)) > 0.0 and float(getattr(cfg, "c_macro_lookroom_weight_scale", 1.0)) > 0.0,
            "C_horizon_y": float(getattr(cfg, "rank_weight_c", 0.0)) > 0.0 and float(getattr(cfg, "c_macro_horizon_weight_scale", 1.0)) > 0.0,
            "C_sym": float(getattr(cfg, "rank_weight_c", 0.0)) > 0.0 and float(getattr(cfg, "c_macro_sym_weight_scale", 1.0)) > 0.0,
            "C_context": float(getattr(cfg, "rank_weight_c", 0.0)) > 0.0 and float(getattr(cfg, "c_macro_context_weight_scale", 1.0)) > 0.0,
            "C_copyspace": float(getattr(cfg, "rank_weight_c", 0.0)) > 0.0 and float(getattr(cfg, "c_macro_copyspace_weight_scale", 1.0)) > 0.0,
            "T_teacher": float(getattr(cfg, "rank_weight_t", 0.0)) > 0.0,
            "safety_bundle": True,
            "area_log_prior": float(getattr(cfg, "policy_area_weight_scale", 1.0)) > 0.0,
        },
    }
    return out


def build_profile_metadata(cfg: Any, spec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "profile_name": spec["profile_name"],
        "label": spec["label"],
        "stage": spec["stage"],
        "description": spec["description"],
        "override_json_path": spec.get("override_json_path", ""),
        "overrides": copy.deepcopy(spec.get("overrides", {})),
        "active_components": active_score_components(cfg),
        "scorer_cfg": cfg_to_dict(cfg),
    }
