#!/usr/bin/env python3
"""
VLM/MLLM Teacher Label Generator (Section 10)

Input:
  - teacher_scores jsonl (from score_teacher.py)
  - optional curated image dir (<image_id>.<ext>) for vision backends

Output:
  - crop_label_v1 jsonl (one row per image_id x target_ar)
  - optional meta_norm_v1 jsonl (one row per image)
  - summary json

Design goals:
  - Pluggable backend registry (default: qwen25_vl)
  - Strict post-validation (candidate-id/bbox consistency)
  - Retry + deterministic fallback (numeric scorer top-k)
  - OOM-safe behavior (optional CPU fallback / skip)
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import uuid

from tqdm import tqdm
from packaging.version import Version, InvalidVersion


AR_ORDER = ["FREE", "1:1", "9:16", "16:9", "3:4", "4:3"]
SUBJECT_STOP_TAG_KEYWORDS = (
    "background",
    "copy space",
    "copy-space",
    "negative space",
    "texture",
    "wallpaper",
    "backdrop",
    "pattern",
)

# Union of spec vocab + current numeric-scoring tags already used in pipeline
ALLOWED_WHY_TAGS = {
    "subject_preserved",
    "background_context_ok",
    "clutter_reduced",
    "avoid_face_cut",
    "avoid_person_cut",
    "avoid_object_cut",
    "copy_space_kept",
    "copy_space_preserved",
    "text_kept",
    "text_preserved",
    "rule_of_thirds",
    "phi_grid",
    "centered_subject",
    "center_comp",
    "symmetry",
    "leading_lines",
    "balanced_negative_space",
    "horizon_on_third",
    "ar_fits_well",
    "ar_choice_freeform",
    "ar_extreme_penalty",
    "ar_choice_portrait_focus",
    "ar_choice_context_wide",
    "ar_choice_symmetry_balance",
    "ar_choice_copyspace",
    "tight_crop",
    "wide_crop",
    "headroom_ok",
    "headroom_violation",
    "lookroom_ok",
    "lookroom_violation",
    "joint_cutoff",
    "face_cut",
    "roll_tilt",
    "context_preserved",
    "context_lost",
    "context_loss",
    "no_crop_needed",
    "crop_improves_comp",
    "balanced_crop",
    "diagonal",
    "triangle",
    "s_curve",
    "c_curve",
    "o_curve",
    "radial",
    "perspective",
    "pattern",
    "dense",
    "scatter",
}


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not math.isfinite(x):
        return float(default)
    return x


def safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def as_list(v: Any) -> List[Any]:
    if isinstance(v, list):
        return v
    if isinstance(v, tuple):
        return list(v)
    return []


def parse_bool01(v: Any, default: bool = False) -> bool:
    if v is None:
        return bool(default)
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def as_clean_str(v: Any) -> str:
    if isinstance(v, str):
        return v.strip()
    return ""


def is_template_placeholder_text(v: Any) -> bool:
    s = as_clean_str(v)
    if not s:
        return False
    s_l = s.lower()
    if re.fullmatch(r"<[^<>]{1,200}>", s):
        return True
    if s_l in {"crop|minimal_crop|keep_full"}:
        return True
    placeholder_needles = (
        "<candidate_id",
        "<allowed_tag>",
        "<1 sentence summary>",
        "<2~4 sentence rationale using selected candidates>",
        "<why this crop is selected>",
        "<why this candidate is rejected>",
    )
    return any(n in s_l for n in placeholder_needles)


def as_non_template_str(v: Any) -> str:
    s = as_clean_str(v)
    if is_template_placeholder_text(s):
        return ""
    return s


def _count_placeholder_strings(v: Any, depth: int = 0) -> int:
    if depth > 6:
        return 0
    if isinstance(v, str):
        return 1 if is_template_placeholder_text(v) else 0
    if isinstance(v, dict):
        cnt = 0
        for _, vv in list(v.items())[:64]:
            cnt += _count_placeholder_strings(vv, depth + 1)
        return cnt
    if isinstance(v, (list, tuple)):
        cnt = 0
        for vv in list(v)[:64]:
            cnt += _count_placeholder_strings(vv, depth + 1)
        return cnt
    return 0


def stable_shard_index(image_id: str, num_shards: int) -> int:
    n = max(1, int(num_shards))
    if n <= 1:
        return 0
    key = str(image_id).encode("utf-8", errors="ignore")
    hv = int(hashlib.md5(key).hexdigest(), 16)
    return int(hv % n)


def build_generation_kwargs(max_new_tokens: int, temperature: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "max_new_tokens": max(64, int(max_new_tokens)),
    }
    temp = float(temperature)
    if temp > 0.0:
        out["do_sample"] = True
        out["temperature"] = temp
    else:
        out["do_sample"] = False
    return out


def normalize_box(box: Any) -> Optional[List[float]]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(x) for x in box]
    except Exception:
        return None
    x1 = clamp(x1, 0.0, 1.0)
    y1 = clamp(y1, 0.0, 1.0)
    x2 = clamp(x2, 0.0, 1.0)
    y2 = clamp(y2, 0.0, 1.0)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if (x2 - x1) <= 1e-8 or (y2 - y1) <= 1e-8:
        return None
    return [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)]


def is_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    needles = [
        "out of memory",
        "cuda out of memory",
        "cublas status alloc failed",
        "cuda error: out of memory",
    ]
    return any(n in msg for n in needles)


def infer_category(tags: Sequence[str], route_flags: Dict[str, Any], has_human_evidence: bool) -> str:
    tset = {str(t).strip().lower() for t in tags if str(t).strip()}
    if has_human_evidence or any(x in tset for x in {"person", "people", "man", "woman", "portrait", "group"}):
        return "person"
    if parse_bool01(route_flags.get("is_product", False), False) or any(x in tset for x in {"product", "packshot"}):
        return "product"
    if parse_bool01(route_flags.get("is_landscape_scene", False), False):
        return "landscape"
    if any(x in tset for x in {"animal", "pet", "wildlife"}):
        return "animal"
    return "generic"


def infer_main_subject(tags: Sequence[str], category: str) -> Dict[str, Any]:
    cleaned = [str(t).strip() for t in tags if str(t).strip()]
    label = ""
    for t in cleaned:
        tl = t.lower()
        if any(k in tl for k in SUBJECT_STOP_TAG_KEYWORDS):
            continue
        label = t
        break
    if not label:
        label = str(category or "generic")

    if category == "person":
        stype = "person"
    elif category == "landscape":
        stype = "scene"
    else:
        stype = "object"

    filtered_out = [t for t in cleaned if any(k in t.lower() for k in SUBJECT_STOP_TAG_KEYWORDS)]
    return {
        "label": label,
        "type": stype,
        "confidence": 0.7,
        "evidence": {
            "tags_top": [str(x) for x in cleaned[:8]],
            "noun_phrases_top": [label],
            "notes": "rule_based_filtered_stop_tags" if filtered_out else "rule_based",
        },
    }


def build_meta_norm_v1(row: Dict[str, Any]) -> Dict[str, Any]:
    image_id = str(row.get("image_id", ""))
    tags = as_list(row.get("tags"))
    route_global = row.get("route_global", {}) if isinstance(row.get("route_global"), dict) else {}
    flags = route_global.get("flags", {}) if isinstance(route_global.get("flags"), dict) else {}

    has_human_evidence = parse_bool01(route_global.get("has_human_evidence", False), False)
    if not has_human_evidence:
        has_human_evidence = (safe_int(route_global.get("num_people", 0), 0) > 0) or (
            safe_int(route_global.get("c2_person_count", 0), 0) > 0
        )

    category = infer_category(tags, flags, has_human_evidence)
    main_subject = infer_main_subject(tags, category)

    copy_space = parse_bool01(flags.get("has_copy_space", False), False)
    portrait = category == "person"

    intents = []
    intents.append({"code": "avoid_face_cut", "priority": 5, "detail": "face_cut=false preferred"})
    if copy_space:
        intents.append({"code": "preserve_copy_space", "priority": 4, "detail": "keep negative space"})
    if parse_bool01(flags.get("is_landscape_scene", False), False):
        intents.append({"code": "horizon_level", "priority": 3, "detail": "roll tilt minimized"})

    return {
        "schema_version": "meta_norm_v1",
        "image_id": image_id,
        "language": "en",
        "category": category,
        "subcategory": str(route_global.get("portrait_category", "unknown") or "unknown"),
        "main_subject": main_subject,
        "secondary_subjects": [],
        "multi_subject": bool(safe_int(route_global.get("num_people", 0), 0) >= 2),
        "special_flags": {
            "copy_space": copy_space,
            "copy_space_side": "unknown",
            "portrait": portrait,
            "isolated": parse_bool01(flags.get("is_isolated_packshot", False), False),
            "panoramic": False,
            "close_up": str(route_global.get("shot_type", "unknown")) in {"headshot", "half"},
            "text_overlay_likely": False,
        },
        "intent": intents,
    }


def candidate_id_of(cand: Dict[str, Any]) -> str:
    return str(cand.get("candidate_id", "")).strip()


def dedupe_candidates(cands: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for c in cands:
        if not isinstance(c, dict):
            continue
        cid = candidate_id_of(c)
        if not cid or cid in seen:
            continue
        box = normalize_box(c.get("bbox_norm_xyxy"))
        if box is None:
            continue
        cc = dict(c)
        cc["bbox_norm_xyxy"] = box
        out.append(cc)
        seen.add(cid)
    return out


def _candidate_feature_view(c: Dict[str, Any]) -> Dict[str, Any]:
    comps = c.get("scores", {}).get("components", {}) if isinstance(c.get("scores"), dict) else {}
    flags = c.get("flags", {}) if isinstance(c.get("flags"), dict) else {}
    checks = c.get("composition_checks", {}) if isinstance(c.get("composition_checks"), dict) else {}
    return {
        "candidate_id": candidate_id_of(c),
        "bbox_norm_xyxy": normalize_box(c.get("bbox_norm_xyxy")) or [0.0, 0.0, 1.0, 1.0],
        "source": str(c.get("source", "unknown")),
        "must_keep": bool(c.get("must_keep", False)),
        "features": {
            "subj_coverage": safe_float(flags.get("subject_coverage", 0.0), 0.0),
            "subj_area": safe_float(flags.get("subject_area", 0.0), 0.0),
            "face_cut": bool(flags.get("face_cut", False)),
            "joint_cutoff_score": safe_float(flags.get("joint_cutoff_score", 0.0), 0.0),
            "horizon_y": safe_float(checks.get("horizon", {}).get("y", 0.0), 0.0)
            if isinstance(checks.get("horizon"), dict)
            else 0.0,
            "roll_deg": safe_float(checks.get("horizon", {}).get("roll_deg", 0.0), 0.0)
            if isinstance(checks.get("horizon"), dict)
            else 0.0,
            "horizon_conf": safe_float(checks.get("horizon", {}).get("conf", 0.0), 0.0)
            if isinstance(checks.get("horizon"), dict)
            else 0.0,
            "symmetry_score": safe_float(checks.get("symmetry", {}).get("value", comps.get("r_sym", 0.0)), 0.0)
            if isinstance(checks.get("symmetry"), dict)
            else safe_float(comps.get("r_sym", 0.0), 0.0),
            "text_keep_ratio": safe_float(checks.get("text", {}).get("text_keep_ratio", 0.0), 0.0)
            if isinstance(checks.get("text"), dict)
            else 0.0,
            "clip_img_txt": safe_float(comps.get("cosine_img_text", 0.0), 0.0),
            "aesthetic_pre": safe_float(comps.get("aesthetic_norm", 0.0), 0.0),
            "cheap": safe_float(c.get("scores", {}).get("cheap", 0.0), 0.0),
            "expensive": safe_float(c.get("scores", {}).get("expensive", 0.0), 0.0),
            "final": safe_float(c.get("scores", {}).get("final", 0.0), 0.0),
        },
        "why_tags": [str(x) for x in as_list(c.get("why_tags"))[:12]],
    }


def _target_ar_list(results_by_ar: Dict[str, Any], target_ar: str) -> List[str]:
    if target_ar == "all":
        ars = [ar for ar in AR_ORDER if ar in results_by_ar]
        for ar in sorted(results_by_ar.keys()):
            if ar not in ars:
                ars.append(ar)
        return ars
    out = []
    for ar in [x.strip() for x in target_ar.split(",") if x.strip()]:
        if ar in results_by_ar:
            out.append(ar)
    return out


def build_task(
    row: Dict[str, Any],
    target_ar: str,
    *,
    top_m: int,
    top_k: int,
    prompt_version: str,
) -> Optional[Dict[str, Any]]:
    scorer = row.get("teacher_scorer", {}) if isinstance(row.get("teacher_scorer"), dict) else {}
    results_by_ar = scorer.get("results_by_ar", {}) if isinstance(scorer.get("results_by_ar"), dict) else {}
    ar_res = results_by_ar.get(target_ar, {}) if isinstance(results_by_ar.get(target_ar), dict) else {}
    if not ar_res:
        return None

    pool: List[Dict[str, Any]] = []
    cheap_top = as_list(ar_res.get("cheap_top_m"))
    if cheap_top:
        pool.extend(cheap_top[: max(1, int(top_m))])
    pool.extend(as_list(ar_res.get("selected_topk")))
    pool.extend(as_list(ar_res.get("hard_negatives")))

    baseline = ar_res.get("baseline_candidate") if isinstance(ar_res.get("baseline_candidate"), dict) else None
    best = ar_res.get("best_candidate") if isinstance(ar_res.get("best_candidate"), dict) else None
    if baseline:
        pool.insert(0, baseline)
    if best:
        pool.insert(0, best)

    candidates = dedupe_candidates(pool)
    if not candidates:
        return None

    cand_map = {candidate_id_of(c): c for c in candidates}

    numeric_topk = []
    for c in as_list(ar_res.get("selected_topk")):
        cid = candidate_id_of(c)
        if cid and cid in cand_map:
            numeric_topk.append(cand_map[cid])
    if not numeric_topk:
        # fallback to highest final among candidate pool
        numeric_topk = sorted(
            candidates,
            key=lambda x: safe_float(x.get("scores", {}).get("final", -1e9), -1e9),
            reverse=True,
        )[: max(1, int(top_k))]

    hard_neg = []
    for c in as_list(ar_res.get("hard_negatives")):
        cid = candidate_id_of(c)
        if cid and cid in cand_map:
            hard_neg.append(cand_map[cid])

    task = {
        "schema_version": "teacher_ab_input_v1",
        "sample_id": f"{row.get('image_id', '')}|ar={target_ar}",
        "image_id": str(row.get("image_id", "")),
        "image": {
            "image_id": str(row.get("image_id", "")),
            "width": safe_int(row.get("width", 0), 0),
            "height": safe_int(row.get("height", 0), 0),
        },
        "target_ar": target_ar,
        "meta": {
            "tags": [str(x) for x in as_list(row.get("tags"))[:64]],
            "caption": "",
            "alt_text": "",
        },
        "meta_norm_v1": build_meta_norm_v1(row),
        "features": {
            "route_global": row.get("route_global", {}),
            "subject_prior": row.get("subject_prior", {}),
        },
        "candidates": [_candidate_feature_view(c) for c in candidates],
        "policy": {
            "topk": int(top_k),
            "topm": int(top_m),
            "prompt_version": str(prompt_version),
            "keep_policy": ar_res.get("keep_policy", {}),
        },
        "decision": ar_res.get("decision", {}),
        "baseline_candidate": _candidate_feature_view(baseline) if isinstance(baseline, dict) else None,
        "best_candidate": _candidate_feature_view(best) if isinstance(best, dict) else None,
        "numeric_topk": [_candidate_feature_view(c) for c in numeric_topk[: max(1, int(top_k))]],
        "hard_negatives": [_candidate_feature_view(c) for c in hard_neg[:5]],
        "composition_checks_top1": (
            numeric_topk[0].get("composition_checks", {})
            if numeric_topk and isinstance(numeric_topk[0], dict)
            else {}
        ),
    }
    return task


def sanitize_tags(tags: Any, fallback: Optional[List[str]] = None) -> List[str]:
    out: List[str] = []
    for t in as_list(tags):
        s = str(t).strip()
        if not s:
            continue
        if s in ALLOWED_WHY_TAGS and s not in out:
            out.append(s)
    if not out and fallback:
        for t in fallback:
            s = str(t).strip()
            if s and s in ALLOWED_WHY_TAGS and s not in out:
                out.append(s)
    if not out:
        out = ["crop_improves_comp"]
    return out[:8]


def _fallback_why_text(cand: Dict[str, Any]) -> str:
    feats = cand.get("features", {}) if isinstance(cand.get("features"), dict) else {}
    final_s = safe_float(feats.get("final", 0.0), 0.0)
    cov = safe_float(feats.get("subj_coverage", 0.0), 0.0)
    face_cut = bool(feats.get("face_cut", False))
    return (
        f"final={final_s:.3f}, subj_coverage={cov:.3f}, face_cut={str(face_cut).lower()} 기준으로 "
        "안정적인 후보로 선택했습니다."
    )


def build_fallback_output(task: Dict[str, Any], reason: str, backend_name: str) -> Dict[str, Any]:
    cands = as_list(task.get("numeric_topk"))
    k = max(1, safe_int(task.get("policy", {}).get("topk", 5), 5))
    selected = cands[:k]

    selected_rows = []
    for rank, c in enumerate(selected, start=1):
        selected_rows.append(
            {
                "rank": rank,
                "candidate_id": str(c.get("candidate_id", "")),
                "bbox_norm_xyxy": normalize_box(c.get("bbox_norm_xyxy")) or [0.0, 0.0, 1.0, 1.0],
                "why_tags": sanitize_tags(c.get("why_tags"), fallback=["crop_improves_comp"]),
                "why_text": _fallback_why_text(c),
            }
        )

    selected_ids = {x["candidate_id"] for x in selected_rows}
    also_considered = []
    for c in as_list(task.get("hard_negatives"))[:3]:
        cid = str(c.get("candidate_id", ""))
        if not cid or cid in selected_ids:
            continue
        also_considered.append(
            {
                "candidate_id": cid,
                "reject_tags": ["context_lost"],
                "reject_text": "Top-K 대비 우선순위가 낮아 제외되었습니다.",
            }
        )

    decision = task.get("decision", {}) if isinstance(task.get("decision"), dict) else {}
    baseline = task.get("baseline_candidate") if isinstance(task.get("baseline_candidate"), dict) else None
    baseline_id = str(baseline.get("candidate_id", "")) if baseline else ""

    top1_final = safe_float(selected[0].get("features", {}).get("final", 0.0), 0.0) if selected else 0.0
    top2_final = safe_float(selected[1].get("features", {}).get("final", top1_final), top1_final) if len(selected) > 1 else top1_final
    conf = clamp(0.5 + max(0.0, top1_final - top2_final), 0.0, 1.0)

    top1_id = selected_rows[0]["candidate_id"] if selected_rows else ""
    top1_src = ""
    if selected and isinstance(selected[0], dict):
        top1_src = str(selected[0].get("source", "unknown"))

    return {
        "schema_version": "crop_label_v1",
        "record_id": str(uuid.uuid4()),
        "sample_id": task.get("sample_id"),
        "image_id": task.get("image_id"),
        "target_ar": task.get("target_ar"),
        "decision_type": str(decision.get("decision_type", "crop")),
        "delta_improve_vs_baseline": safe_float(decision.get("delta_improve", 0.0), 0.0),
        "baseline_used_candidate_id": baseline_id,
        "selected_topk": selected_rows,
        "also_considered": also_considered,
        "composition_checks": task.get("composition_checks_top1", {}),
        "original_policy": {
            "original_is_candidate": bool(baseline_id),
            "kept_original": (bool(selected_rows) and selected_rows[0]["candidate_id"] == baseline_id),
            "why_not_original": "improvement_over_baseline" if baseline_id else "baseline_missing",
        },
        "explanations": {
            "short": (
                f"fallback 선택: top1={top1_id}({top1_src or 'unknown'}), "
                f"final={top1_final:.3f}, delta={safe_float(decision.get('delta_improve', 0.0), 0.0):.3f}"
            ),
            "long": (
                f"VLM 응답 실패({reason})로 numeric_topk 기반으로 재구성했습니다. "
                f"selected={len(selected_rows)}개, baseline={baseline_id or 'none'}, "
                f"top1-top2 margin={max(0.0, top1_final - top2_final):.3f}."
            ),
        },
        "teacher": {
            "backend": backend_name,
            "teacher_id": backend_name,
            "teacher_confidence": conf,
        },
        "validator": {
            "schema_ok": True,
            "numeric_consistency_ok": True,
            "notes": f"fallback:{reason}",
        },
    }


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    # Fast path: already a single JSON dict.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    def _scan_json_objects(seg_text: str, base_pos: int = 0) -> List[Tuple[Dict[str, Any], int]]:
        objs_local: List[Tuple[Dict[str, Any], int]] = []
        starts = [i for i, ch in enumerate(seg_text) if ch == "{"]
        for s in starts:
            depth = 0
            for i in range(s, len(seg_text)):
                ch = seg_text[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        seg = seg_text[s : i + 1]
                        try:
                            obj = json.loads(seg)
                        except Exception:
                            break
                        if isinstance(obj, dict):
                            objs_local.append((obj, base_pos + s))
                        break
        return objs_local

    # Prefer parsing the final assistant section first when present.
    assistant_marks = list(re.finditer(r"(?im)^assistant\s*$", text))
    if assistant_marks:
        tail_start = assistant_marks[-1].end()
        tail_text = text[tail_start:].strip()
        tail_objs = _scan_json_objects(tail_text, base_pos=tail_start)
    else:
        tail_objs = []

    # Collect every decodable JSON object and choose the one most likely
    # to be model output (not echoed input payload).
    objs: List[Tuple[Dict[str, Any], int]] = tail_objs if tail_objs else _scan_json_objects(text)
    if not objs:
        return None

    def _score_obj(o: Dict[str, Any]) -> int:
        keys = set(o.keys())
        score = 0
        if "selected_topk" in keys:
            score += 120
        if isinstance(o.get("selected_candidate_ids"), list):
            score += 118
        if isinstance(o.get("selected_ids"), list):
            score += 116
        if "selected_crops" in keys:
            score += 110
        if "selected_candidates" in keys:
            score += 100
        if "crop_labels" in keys:
            score += 90
        if "crop_label" in keys:
            score += 70
        if "decision_type" in keys:
            score += 20
        if "delta_improve_vs_baseline" in keys:
            score += 15
        if "explanations" in keys:
            score += 12
        if "also_considered" in keys:
            score += 10
        if "composition_checks" in keys:
            score += 6

        # Input payload echo pattern.
        if {"target_ar", "meta_norm", "candidates", "allowed_tags", "top_k"}.issubset(keys):
            score -= 220
        # Explicit schema template block inside input payload.
        if "required_output_schema" in keys:
            score -= 260
        if {"task", "candidate_id_list", "candidates", "allowed_tags", "top_k"}.issubset(keys):
            score -= 220
        if (
            "selected_candidate_ids" in keys
            and not isinstance(o.get("selected_candidate_ids"), list)
        ):
            score -= 180
        # Typical echoed sub-object of input meta_norm.
        if {"category", "main_subject", "intent", "special_flags"}.issubset(keys):
            score -= 80
        if {"code", "priority", "detail"}.issubset(keys):
            score -= 120
        if {"copy_space", "copy_space_side", "portrait", "isolated", "panoramic", "close_up", "text_overlay_likely"}.issubset(keys):
            score -= 120
        # Typical echoed candidate object.
        if (
            {"candidate_id", "bbox_norm_xyxy"}.issubset(keys)
            and "selected_topk" not in keys
            and "selected_candidate_ids" not in keys
            and "selected_crops" not in keys
        ):
            score -= 40
        # Composition-only object can still be useful when no picks are returned.
        if keys.issubset({"composition_checks", "checks", "scores"}):
            score -= 5

        # Penalize placeholder/template strings to avoid choosing required_output_schema.
        ph_cnt = _count_placeholder_strings(o)
        if ph_cnt > 0:
            score -= min(600, ph_cnt * 120)
        ex = o.get("explanations")
        if isinstance(ex, dict):
            if is_template_placeholder_text(ex.get("short")) or is_template_placeholder_text(ex.get("long")):
                score -= 240
        if isinstance(o.get("selected_topk"), list):
            first = o.get("selected_topk")[0] if o.get("selected_topk") else None
            if isinstance(first, dict) and is_template_placeholder_text(first.get("candidate_id")):
                score -= 240
        return score

    # Tie-breaker: prefer later position (assistant output is usually later).
    best_obj, _ = max(objs, key=lambda item: (_score_obj(item[0]), item[1]))
    return best_obj


def _has_model_selection_keys(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    if isinstance(obj.get("selected_candidate_ids"), list):
        return True
    if isinstance(obj.get("selected_ids"), list):
        return True
    if isinstance(obj.get("selected_topk"), list):
        return True
    if isinstance(obj.get("selected_crops"), list):
        return True
    if isinstance(obj.get("selected_candidates"), list):
        return True
    return False


def _extract_compact_selection_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Recover compact v2 selection when generation truncates after the ID list."""
    if not text:
        return None
    tail = text
    assistant_marks = list(re.finditer(r"(?im)^assistant\s*$", text))
    if assistant_marks:
        tail = text[assistant_marks[-1].end() :]

    def _extract_json_array(key: str) -> Optional[List[Any]]:
        pat = rf'"{re.escape(key)}"\s*:\s*(\[[^\]]*\])'
        m = re.search(pat, tail, flags=re.DOTALL)
        if not m:
            return None
        try:
            arr = json.loads(m.group(1))
        except Exception:
            return None
        return arr if isinstance(arr, list) else None

    selected_ids = _extract_json_array("selected_candidate_ids")
    if not selected_ids:
        selected_ids = _extract_json_array("selected_ids")
    clean_ids = [as_non_template_str(x) for x in selected_ids or []]
    clean_ids = [x for x in clean_ids if x]
    if not clean_ids:
        return None

    decision_type = "crop"
    m_dec = re.search(r'"decision_type"\s*:\s*"([^"]+)"', tail)
    if m_dec and m_dec.group(1) in {"crop", "minimal_crop", "keep_full"}:
        decision_type = m_dec.group(1)

    out: Dict[str, Any] = {
        "decision_type": decision_type,
        "selected_candidate_ids": clean_ids,
    }
    also_considered = _extract_json_array("also_considered_ids")
    if also_considered is not None:
        out["also_considered_ids"] = [as_non_template_str(x) for x in also_considered if as_non_template_str(x)]

    # Keep this intentionally shallow. Full JSON parsing handles why tags when
    # the generation completes; truncated recovery only needs candidate IDs.
    m_expl = re.search(r'"explanation"\s*:\s*"([^"]{1,240})', tail, flags=re.DOTALL)
    if m_expl:
        out["explanation"] = m_expl.group(1).strip()
    return out


def _bbox_iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    aa = normalize_box(a)
    bb = normalize_box(b)
    if aa is None or bb is None:
        return 0.0
    ax1, ay1, ax2, ay2 = aa
    bx1, by1, bx2, by2 = bb
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(1e-12, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1e-12, (bx2 - bx1) * (by2 - by1))
    union = max(1e-12, area_a + area_b - inter)
    return float(inter / union)


def _match_candidate_id_by_bbox(
    bbox: Any,
    cand_map: Dict[str, Dict[str, Any]],
    used_ids: set[str],
    min_iou: float = 0.75,
) -> Optional[str]:
    box = normalize_box(bbox)
    if box is None:
        return None
    best_id = None
    best_iou = -1.0
    for cid, cand in cand_map.items():
        if cid in used_ids:
            continue
        iou = _bbox_iou_xyxy(box, cand.get("bbox_norm_xyxy"))
        if iou > best_iou:
            best_iou = iou
            best_id = cid
    if best_id is None or best_iou < float(min_iou):
        return None
    return str(best_id)


def _extract_model_selected_items(
    parsed: Dict[str, Any],
    cand_map: Dict[str, Dict[str, Any]],
    k: int,
) -> Tuple[List[Dict[str, Any]], str]:
    raw_sel = as_list(parsed.get("selected_topk"))
    mode = "selected_topk"
    selected_id_items: List[Any] = []
    if not raw_sel:
        selected_id_items = as_list(parsed.get("selected_candidate_ids"))
        if not selected_id_items:
            selected_id_items = as_list(parsed.get("selected_ids"))
        if not selected_id_items:
            selected_id_items = as_list(parsed.get("candidate_ids"))
        if not selected_id_items:
            one_id = as_non_template_str(parsed.get("selected_candidate_id"))
            if one_id:
                selected_id_items = [one_id]
        if selected_id_items:
            mode = "selected_candidate_ids"
        else:
            alt = []
    if not raw_sel and not selected_id_items:
        alt = as_list(parsed.get("selected_crops"))
        if alt:
            raw_sel = alt
            mode = "selected_crops"
        else:
            alt = as_list(parsed.get("selected_candidates"))
            if alt:
                raw_sel = alt
                mode = "selected_candidates"
            else:
                alt = as_list(parsed.get("crop_labels"))
                if alt:
                    raw_sel = alt
                    mode = "crop_labels"
                else:
                    # weak format: single crop entry
                    if "crop_label" in parsed or "bbox_norm_xyxy" in parsed:
                        raw_sel = [parsed]
                        mode = "crop_label_single"
                    else:
                        alt = as_list(parsed.get("numeric_topk"))
                        if alt:
                            raw_sel = alt
                            mode = "numeric_topk_hint"

    out: List[Dict[str, Any]] = []
    used: set[str] = set()
    why_by_candidate = parsed.get("why_by_candidate")
    if not isinstance(why_by_candidate, dict):
        why_by_candidate = parsed.get("rationale_by_candidate")
    if not isinstance(why_by_candidate, dict):
        why_by_candidate = parsed.get("selected_why_tags")
    if not isinstance(why_by_candidate, dict):
        why_by_candidate = {}

    def append_candidate(cid: str, why_tags: Any = None, why_text: Any = None) -> None:
        cid = as_non_template_str(cid)
        if not cid or cid in used or cid not in cand_map:
            return
        why_entry = why_by_candidate.get(cid)
        if isinstance(why_entry, dict):
            if why_tags is None:
                why_tags = why_entry.get("why_tags")
            if why_text is None:
                why_text = why_entry.get("why_text") or why_entry.get("rationale")
        elif isinstance(why_entry, list) and why_tags is None:
            why_tags = why_entry
        elif isinstance(why_entry, str) and why_text is None:
            why_text = why_entry
        out.append(
            {
                "candidate_id": cid,
                "why_tags": as_list(why_tags),
                "why_text": why_text,
            }
        )
        used.add(cid)

    for it in selected_id_items:
        if isinstance(it, dict):
            cid = (
                as_non_template_str(it.get("candidate_id"))
                or as_non_template_str(it.get("id"))
                or as_non_template_str(it.get("candidate"))
            )
            append_candidate(
                cid,
                why_tags=it.get("why_tags"),
                why_text=it.get("why_text") or it.get("rationale"),
            )
        else:
            append_candidate(str(it))
        if len(out) >= max(1, int(k)):
            return out, mode

    for it in raw_sel:
        if not isinstance(it, dict):
            continue
        cid = as_non_template_str(it.get("candidate_id"))
        if (not cid) or (cid not in cand_map):
            cid = _match_candidate_id_by_bbox(it.get("bbox_norm_xyxy"), cand_map, used, min_iou=0.75) or ""
        append_candidate(
            cid,
            why_tags=(
                as_list(it.get("why_tags"))
                if as_list(it.get("why_tags"))
                else ([as_non_template_str(it.get("label"))] if as_non_template_str(it.get("label")) else [])
            ),
            why_text=it.get("why_text"),
        )
        if len(out) >= max(1, int(k)):
            break
    return out, mode


def _build_auto_explanations(
    task: Dict[str, Any],
    cand_map: Dict[str, Dict[str, Any]],
    selected_rows: List[Dict[str, Any]],
    decision_type: str,
    delta_improve: float,
    model_selected_n: int,
    selected_mode: str,
) -> Dict[str, str]:
    if not selected_rows:
        return {
            "short": "수치 기반 후보를 사용해 Top-K를 구성했습니다.",
            "long": "모델 출력이 불완전하여 numeric_topk를 기반으로 라벨을 구성했습니다.",
        }
    top1_id = str(selected_rows[0].get("candidate_id", ""))
    top1 = cand_map.get(top1_id, {}) if top1_id else {}
    feats = top1.get("features", {}) if isinstance(top1.get("features"), dict) else {}
    src = str(top1.get("source", "unknown")) if isinstance(top1, dict) else "unknown"
    final_v = safe_float(feats.get("final", 0.0), 0.0)
    cov_v = safe_float(feats.get("subj_coverage", 0.0), 0.0)
    face_cut = bool(feats.get("face_cut", False))
    k_sel = len(selected_rows)
    k_model = int(model_selected_n)

    dec = str(decision_type or "crop")
    short = (
        f"{dec} 선택: top1={top1_id}({src}), final={final_v:.3f}, "
        f"coverage={cov_v:.3f}, face_cut={str(face_cut).lower()}"
    )

    tau = safe_float(task.get("decision", {}).get("tau_improve", 0.0), 0.0) if isinstance(task.get("decision"), dict) else 0.0
    baseline_id = ""
    baseline = task.get("baseline_candidate")
    if isinstance(baseline, dict):
        baseline_id = str(baseline.get("candidate_id", ""))
    long = (
        f"selected_topk={k_sel}개 중 모델 직접 선택={k_model}개(mode={selected_mode}), "
        f"부족분은 numeric_topk로 보완했습니다. "
        f"delta={safe_float(delta_improve, 0.0):.3f}, tau={tau:.3f}, baseline={baseline_id or 'none'}."
    )
    return {"short": short, "long": long}


def _classify_parsed_object(parsed: Dict[str, Any]) -> str:
    if not isinstance(parsed, dict):
        return "non_dict"
    keys = set(parsed.keys())
    if "selected_topk" in keys:
        return "selected_topk"
    if isinstance(parsed.get("selected_candidate_ids"), list):
        return "selected_candidate_ids"
    if isinstance(parsed.get("selected_ids"), list):
        return "selected_ids"
    if "selected_crops" in keys:
        return "selected_crops"
    if "selected_candidates" in keys:
        return "selected_candidates"
    if "crop_labels" in keys:
        return "crop_labels"
    if "crop_label" in keys:
        return "crop_label"
    if "numeric_topk" in keys:
        return "numeric_topk"
    if {"target_ar", "meta_norm", "candidates", "allowed_tags", "top_k"}.issubset(keys):
        return "input_echo"
    if keys.issubset({"composition_checks", "checks", "scores"}):
        return "composition_only"
    return "other"


def normalize_backend_output(
    task: Dict[str, Any],
    parsed: Dict[str, Any],
    *,
    backend_name: str,
    model_id: str,
) -> Dict[str, Any]:
    cand_map = {str(c.get("candidate_id", "")): c for c in as_list(task.get("candidates"))}
    k = max(1, safe_int(task.get("policy", {}).get("topk", 5), 5))
    decision = task.get("decision", {}) if isinstance(task.get("decision"), dict) else {}

    selected: List[Dict[str, Any]] = []
    used = set()
    model_rows, selected_mode = _extract_model_selected_items(parsed, cand_map, k)
    for it in model_rows:
        cid = str(it.get("candidate_id", "")).strip()
        if not cid or cid in used or cid not in cand_map:
            continue
        base = cand_map[cid]
        selected.append(
            {
                "rank": len(selected) + 1,
                "candidate_id": cid,
                "bbox_norm_xyxy": normalize_box(base.get("bbox_norm_xyxy")) or [0.0, 0.0, 1.0, 1.0],
                "why_tags": sanitize_tags(it.get("why_tags"), fallback=as_list(base.get("why_tags"))),
                "why_text": as_non_template_str(it.get("why_text")) or _fallback_why_text(base),
            }
        )
        used.add(cid)
        if len(selected) >= k:
            break
    selected_from_model = len(selected)

    # Fill with numeric top-k when model output is partial
    for c in as_list(task.get("numeric_topk")):
        if len(selected) >= k:
            break
        cid = str(c.get("candidate_id", "")).strip()
        if not cid or cid in used or cid not in cand_map:
            continue
        selected.append(
            {
                "rank": len(selected) + 1,
                "candidate_id": cid,
                "bbox_norm_xyxy": normalize_box(c.get("bbox_norm_xyxy")) or [0.0, 0.0, 1.0, 1.0],
                "why_tags": sanitize_tags(c.get("why_tags"), fallback=["crop_improves_comp"]),
                "why_text": _fallback_why_text(c),
            }
        )
        used.add(cid)
        if len(selected) >= k:
            break

    if not selected:
        raise ValueError("selected_topk is empty after normalization")

    # also_considered
    raw_cons = as_list(parsed.get("also_considered"))
    if not raw_cons:
        raw_cons = as_list(parsed.get("also_considered_ids"))
    if not raw_cons:
        raw_cons = as_list(parsed.get("rejected_candidate_ids"))
    also_considered: List[Dict[str, Any]] = []
    selected_ids = {x["candidate_id"] for x in selected}
    for it in raw_cons:
        if isinstance(it, dict):
            cid = (
                as_non_template_str(it.get("candidate_id"))
                or as_non_template_str(it.get("id"))
                or as_non_template_str(it.get("candidate"))
            )
            reject_tags = it.get("reject_tags")
            reject_text = it.get("reject_text")
        else:
            cid = as_non_template_str(it)
            reject_tags = None
            reject_text = None
        if not cid or cid in selected_ids or cid not in cand_map:
            continue
        also_considered.append(
            {
                "candidate_id": cid,
                "reject_tags": sanitize_tags(reject_tags, fallback=["context_lost"]),
                "reject_text": as_non_template_str(reject_text) or "Top-K 대비 우선순위가 낮아 제외되었습니다.",
            }
        )
        if len(also_considered) >= 3:
            break

    if not also_considered:
        for c in as_list(task.get("hard_negatives")):
            cid = str(c.get("candidate_id", "")).strip()
            if not cid or cid in selected_ids:
                continue
            also_considered.append(
                {
                    "candidate_id": cid,
                    "reject_tags": ["context_lost"],
                    "reject_text": "Top-K 대비 우선순위가 낮아 제외되었습니다.",
                }
            )
            if len(also_considered) >= 3:
                break

    baseline = task.get("baseline_candidate") if isinstance(task.get("baseline_candidate"), dict) else None
    baseline_id = str(baseline.get("candidate_id", "")) if baseline else ""

    comp_checks = parsed.get("composition_checks") if isinstance(parsed.get("composition_checks"), dict) else {}
    if not comp_checks:
        comp_checks = task.get("composition_checks_top1", {}) if isinstance(task.get("composition_checks_top1"), dict) else {}

    top1_final = safe_float(cand_map[selected[0]["candidate_id"]].get("features", {}).get("final", 0.0), 0.0)
    top2_final = top1_final
    if len(selected) > 1:
        top2_final = safe_float(cand_map[selected[1]["candidate_id"]].get("features", {}).get("final", top1_final), top1_final)
    teacher_conf = clamp(0.5 + max(0.0, top1_final - top2_final), 0.0, 1.0)

    base_decision_type = as_non_template_str(decision.get("decision_type")) or "crop"
    parsed_decision_type = as_non_template_str(parsed.get("decision_type"))
    if parsed_decision_type in {"crop", "minimal_crop", "keep_full"}:
        decision_type = parsed_decision_type
    elif base_decision_type in {"crop", "minimal_crop", "keep_full"}:
        decision_type = base_decision_type
    else:
        decision_type = "crop"
    delta_improve = safe_float(parsed.get("delta_improve_vs_baseline", decision.get("delta_improve", 0.0)), 0.0)
    parsed_expl = parsed.get("explanations", {}) if isinstance(parsed.get("explanations"), dict) else {}
    out_short = as_non_template_str(parsed_expl.get("short"))
    out_long = as_non_template_str(parsed_expl.get("long"))
    single_expl = as_non_template_str(parsed.get("explanation"))
    if single_expl and not out_short:
        out_short = single_expl
    if single_expl and not out_long:
        out_long = single_expl
    if (not out_short) or (not out_long):
        auto_expl = _build_auto_explanations(
            task,
            cand_map,
            selected,
            decision_type=decision_type,
            delta_improve=delta_improve,
            model_selected_n=selected_from_model,
            selected_mode=selected_mode,
        )
        if not out_short:
            out_short = auto_expl["short"]
        if not out_long:
            out_long = auto_expl["long"]

    out = {
        "schema_version": "crop_label_v1",
        "record_id": str(uuid.uuid4()),
        "sample_id": task.get("sample_id"),
        "image_id": task.get("image_id"),
        "target_ar": task.get("target_ar"),
        "decision_type": decision_type,
        "delta_improve_vs_baseline": delta_improve,
        "baseline_used_candidate_id": (
            as_non_template_str(parsed.get("baseline_used_candidate_id"))
            if as_non_template_str(parsed.get("baseline_used_candidate_id")) in cand_map
            or as_non_template_str(parsed.get("baseline_used_candidate_id")) == baseline_id
            else baseline_id
        ),
        "selected_topk": selected,
        "also_considered": also_considered,
        "composition_checks": comp_checks,
        "original_policy": {
            "original_is_candidate": bool(baseline_id),
            "kept_original": (selected[0]["candidate_id"] == baseline_id) if baseline_id else False,
            "why_not_original": "selected_non_baseline" if baseline_id and selected[0]["candidate_id"] != baseline_id else "",
        },
        "explanations": {
            "short": out_short,
            "long": out_long,
        },
        "teacher": {
            "backend": backend_name,
            "teacher_id": model_id,
            "teacher_confidence": teacher_conf,
        },
        "validator": {
            "schema_ok": True,
            "numeric_consistency_ok": True,
            "notes": "",
            "model_selected_count": int(selected_from_model),
            "selected_mode": str(selected_mode),
        },
    }

    return out


def build_prompt(task: Dict[str, Any]) -> str:
    meta = task.get("meta_norm_v1", {}) if isinstance(task.get("meta_norm_v1"), dict) else {}
    route = task.get("features", {}).get("route_global", {}) if isinstance(task.get("features", {}), dict) else {}
    candidates = as_list(task.get("candidates"))

    # Keep prompt compact for token stability
    compact_candidates = []
    for c in candidates:
        if not isinstance(c, dict):
            continue
        feats = c.get("features", {}) if isinstance(c.get("features"), dict) else {}
        compact_candidates.append(
            {
                "candidate_id": c.get("candidate_id"),
                "bbox_norm_xyxy": c.get("bbox_norm_xyxy"),
                "source": c.get("source"),
                "must_keep": c.get("must_keep", False),
                "features": {
                    "subj_coverage": round(safe_float(feats.get("subj_coverage", 0.0), 0.0), 4),
                    "subj_area": round(safe_float(feats.get("subj_area", 0.0), 0.0), 4),
                    "face_cut": bool(feats.get("face_cut", False)),
                    "joint_cutoff_score": round(safe_float(feats.get("joint_cutoff_score", 0.0), 0.0), 4),
                    "horizon_y": round(safe_float(feats.get("horizon_y", 0.0), 0.0), 4),
                    "roll_deg": round(safe_float(feats.get("roll_deg", 0.0), 0.0), 3),
                    "symmetry_score": round(safe_float(feats.get("symmetry_score", 0.0), 0.0), 4),
                    "clip_img_txt": round(safe_float(feats.get("clip_img_txt", 0.0), 0.0), 4),
                    "aesthetic_pre": round(safe_float(feats.get("aesthetic_pre", 0.0), 0.0), 4),
                    "final": round(safe_float(feats.get("final", 0.0), 0.0), 4),
                },
                "why_tags": c.get("why_tags", []),
            }
        )

    policy = task.get("policy", {}) if isinstance(task.get("policy"), dict) else {}
    topk = safe_int(policy.get("topk", 5), 5)
    numeric_topk = as_list(task.get("numeric_topk"))
    numeric_topk_ids = [str(x.get("candidate_id", "")) for x in numeric_topk if isinstance(x, dict) and str(x.get("candidate_id", "")).strip()]
    candidate_id_list = [str(x.get("candidate_id", "")) for x in compact_candidates if str(x.get("candidate_id", "")).strip()]
    prompt_version = str(policy.get("prompt_version", "")).strip().lower()

    if "candidate_ids" in prompt_version or "compact" in prompt_version:
        compact_numeric = []
        for c in numeric_topk:
            if not isinstance(c, dict):
                continue
            feats = c.get("features", {}) if isinstance(c.get("features"), dict) else {}
            compact_numeric.append(
                {
                    "candidate_id": c.get("candidate_id"),
                    "final": round(safe_float(feats.get("final", 0.0), 0.0), 4),
                    "subj_coverage": round(safe_float(feats.get("subj_coverage", 0.0), 0.0), 4),
                    "face_cut": bool(feats.get("face_cut", False)),
                    "why_tags": c.get("why_tags", []),
                }
            )

        payload_v2 = {
            "task": "choose_crop_candidate_ids",
            "image_id": task.get("image_id"),
            "target_ar": task.get("target_ar"),
            "top_k": topk,
            "decision_hint": task.get("decision", {}),
            "meta": {
                "category": meta.get("category"),
                "main_subject": meta.get("main_subject", {}).get("label") if isinstance(meta.get("main_subject"), dict) else "",
                "intent": meta.get("intent", []),
                "shot_type_prior": route.get("shot_type", "unknown"),
                "portrait_category": route.get("portrait_category", "unknown"),
            },
            "candidate_id_list": candidate_id_list,
            "numeric_topk_ids": numeric_topk_ids,
            "numeric_topk_features": compact_numeric,
            "candidates": compact_candidates,
            "allowed_tags": sorted(ALLOWED_WHY_TAGS),
        }
        return (
            "You are a crop-selection judge. Return ONLY one valid JSON object.\n"
            "Do not include markdown, comments, or copied input JSON.\n"
            "Your main job is candidate ID selection, not schema assembly.\n"
            "Rules:\n"
            "1) selected_candidate_ids must contain only exact strings from candidate_id_list.\n"
            "2) Preserve order from best to worse. Return at most top_k IDs.\n"
            "3) Prefer candidates that preserve the main subject, avoid face/person/object cuts, and fit target_ar.\n"
            "4) Use numeric_topk_ids as a strong prior, but override when visual evidence or checklist features indicate a better crop.\n"
            "5) Use only allowed_tags. Use at most 3 tags per selected candidate.\n"
            "6) Do not output bbox fields.\n"
            "Required JSON shape:\n"
            '{"decision_type":"crop|minimal_crop|keep_full",'
            '"selected_candidate_ids":["exact_id"],'
            '"selected_why_tags":{"exact_id":["allowed_tag"]},'
            '"also_considered_ids":["exact_id"],'
            '"explanation":"one short sentence"}\n\n'
            f"Input JSON:\n{json.dumps(payload_v2, ensure_ascii=False)}"
        )

    payload = {
        "target_ar": task.get("target_ar"),
        "meta_norm": {
            "category": meta.get("category"),
            "main_subject": meta.get("main_subject", {}).get("label") if isinstance(meta.get("main_subject"), dict) else "",
            "intent": meta.get("intent", []),
            "special_flags": meta.get("special_flags", {}),
            "shot_type_prior": route.get("shot_type", "unknown"),
            "portrait_category": route.get("portrait_category", "unknown"),
        },
        "keep_policy": policy.get("keep_policy", {}),
        "baseline_candidate": task.get("baseline_candidate"),
        "numeric_topk": numeric_topk,
        "numeric_topk_ids": numeric_topk_ids,
        "candidate_id_list": candidate_id_list,
        "candidates": compact_candidates,
        "allowed_tags": sorted(ALLOWED_WHY_TAGS),
        "top_k": topk,
        "required_output_schema": {
            "decision_type": "crop|minimal_crop|keep_full",
            "delta_improve_vs_baseline": 0.0,
            "baseline_used_candidate_id": "<candidate_id>",
            "selected_topk": [
                {
                    "rank": 1,
                    "candidate_id": "<candidate_id_from_candidate_id_list>",
                    "why_tags": ["<allowed_tag>"],
                    "why_text": "<why this crop is selected>",
                }
            ],
            "also_considered": [
                {
                    "candidate_id": "<candidate_id_from_candidate_id_list>",
                    "reject_tags": ["<allowed_tag>"],
                    "reject_text": "<why this candidate is rejected>",
                }
            ],
            "composition_checks": {
                "headroom": {"pass": True},
                "lookroom": {"pass": True},
                "horizon": {"pass": True},
                "symmetry": {"pass": True},
                "context": {"pass": True},
                "cutoff": {"pass": True},
            },
            "explanations": {
                "short": "<1 sentence summary>",
                "long": "<2~4 sentence rationale using selected candidates>",
            },
        },
    }

    return (
        "You are a strict JSON generator for crop labeling.\n"
        "Rules:\n"
        "1) Select Top-K crops ONLY from provided candidates. Never invent candidate_id.\n"
        "2) bbox_norm_xyxy must match selected candidate.\n"
        "3) Use only allowed_tags for why_tags/reject_tags.\n"
        "4) Fill composition_checks with numeric consistency.\n"
        "5) Return a single JSON object only. No markdown.\n"
        "6) DO NOT echo/copy the Input JSON.\n"
        "7) Output keys must follow required_output_schema and MUST include explanations.short and explanations.long.\n\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=False)}"
    )


class BackendError(RuntimeError):
    pass


class BackendInitError(BackendError):
    pass


class BackendInferenceError(BackendError):
    pass


@dataclass
class BackendConfig:
    model_id: str
    device: str
    dtype: str
    max_new_tokens: int
    temperature: float


class BaseVLMBackend:
    backend_name = "base"

    def __init__(self, cfg: BackendConfig):
        self.cfg = cfg

    @property
    def model_id(self) -> str:
        return self.cfg.model_id

    def label(self, task: Dict[str, Any], image_path: Optional[Path]) -> Tuple[Dict[str, Any], str]:
        raise NotImplementedError

    def close(self) -> None:
        return None


class HeuristicBackend(BaseVLMBackend):
    backend_name = "heuristic"

    @property
    def model_id(self) -> str:
        return "heuristic"

    def label(self, task: Dict[str, Any], image_path: Optional[Path]) -> Tuple[Dict[str, Any], str]:
        del image_path
        out = build_fallback_output(task, reason="heuristic_backend", backend_name=self.backend_name)
        return out, json.dumps(out, ensure_ascii=False)


class Qwen25VLHFBackend(BaseVLMBackend):
    backend_name = "qwen25_vl"

    @staticmethod
    def _parse_version(v: str) -> Version:
        try:
            return Version(str(v))
        except InvalidVersion:
            return Version("0")

    @classmethod
    def _check_transformers_compat(cls, model_id: str, tf_version: str, transformers_mod: Any) -> None:
        model_l = str(model_id).strip().lower()
        tf_v = cls._parse_version(tf_version)

        # Qwen3-VL support check.
        if "qwen3-vl" in model_l or "qwen3_vl" in model_l:
            has_cls = hasattr(transformers_mod, "Qwen3VLForConditionalGeneration")
            if not has_cls:
                raise BackendInitError(
                    "Qwen3-VL requires a Transformers build exposing "
                    "Qwen3VLForConditionalGeneration. "
                    f"current={tf_version}, has_qwen3_vl_class={has_cls}. "
                    "Please upgrade, e.g.: "
                    "python -m pip install -U "
                    "\"git+https://github.com/huggingface/transformers\" "
                    "\"tokenizers>=0.21.0\" \"huggingface-hub>=0.26.0\""
                )

        # Qwen2.5-VL support check.
        if "qwen2.5-vl" in model_l or "qwen2_5_vl" in model_l:
            min_v = Version("4.49.0")
            has_cls = hasattr(transformers_mod, "Qwen2_5_VLForConditionalGeneration")
            if tf_v < min_v or not has_cls:
                raise BackendInitError(
                    "Qwen2.5-VL requires transformers>=4.49.0. "
                    f"current={tf_version}, has_qwen2_5_class={has_cls}. "
                    "Please upgrade, e.g.: "
                    "python -m pip install -U 'transformers>=4.49.0,<4.53.0' "
                    "'tokenizers>=0.21.0,<0.22.0' 'huggingface-hub>=0.26.0'"
                )

        # Qwen2-VL support check (for alternate model ids).
        if "qwen2-vl" in model_l and "qwen2.5-vl" not in model_l and "qwen2_5_vl" not in model_l:
            min_v = Version("4.45.0")
            has_cls = hasattr(transformers_mod, "Qwen2VLForConditionalGeneration")
            if tf_v < min_v or not has_cls:
                raise BackendInitError(
                    "Qwen2-VL requires transformers>=4.45.0. "
                    f"current={tf_version}, has_qwen2_vl_class={has_cls}."
                )

    @staticmethod
    def _sanitize_generation_config_for_greedy(model: Any, temperature: float) -> None:
        # Some checkpoints ship a non-null temperature in generation_config.
        # For deterministic decode (temperature<=0), this can produce noisy warnings.
        if safe_float(temperature, 0.0) > 0.0:
            return
        gen_cfg = getattr(model, "generation_config", None)
        if gen_cfg is None:
            return
        try:
            setattr(gen_cfg, "do_sample", False)
        except Exception:
            pass
        if hasattr(gen_cfg, "temperature"):
            try:
                setattr(gen_cfg, "temperature", None)
            except Exception:
                pass

    def __init__(self, cfg: BackendConfig):
        super().__init__(cfg)
        try:
            import torch
            import transformers
            from transformers import AutoProcessor
        except Exception as exc:
            raise BackendInitError(f"transformers backend unavailable: {exc}") from exc

        self._transformers = transformers
        self._torch = torch
        self._AutoProcessor = AutoProcessor
        auto_vision_model_cls = getattr(transformers, "AutoModelForVision2Seq", None)
        if auto_vision_model_cls is None:
            auto_vision_model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
        if auto_vision_model_cls is None:
            auto_vision_model_cls = getattr(transformers, "AutoModelForSeq2SeqLM", None)
        self._AutoModelForVision2Seq = auto_vision_model_cls

        self.device = str(cfg.device or "auto")
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._check_transformers_compat(cfg.model_id, getattr(transformers, "__version__", "0"), transformers)

        dtype_map = {
            "auto": None,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map.get(str(cfg.dtype).lower(), None)

        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
        }
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        model_l = str(cfg.model_id).strip().lower()
        model_cls = None
        if "qwen3-vl" in model_l or "qwen3_vl" in model_l:
            qwen3_cls = getattr(transformers, "Qwen3VLForConditionalGeneration", None)
            if qwen3_cls is not None:
                model_cls = qwen3_cls
        elif "qwen2.5-vl" in model_l or "qwen2_5_vl" in model_l:
            qwen25_cls = getattr(transformers, "Qwen2_5_VLForConditionalGeneration", None)
            if qwen25_cls is not None:
                model_cls = qwen25_cls
        elif "qwen2-vl" in model_l:
            qwen2vl_cls = getattr(transformers, "Qwen2VLForConditionalGeneration", None)
            if qwen2vl_cls is not None:
                model_cls = qwen2vl_cls
        if model_cls is None:
            model_cls = self._AutoModelForVision2Seq
        if model_cls is None:
            raise BackendInitError(
                "No compatible auto vision model class found in transformers. "
                "Expected one of AutoModelForVision2Seq / AutoModelForImageTextToText / AutoModelForSeq2SeqLM."
            )

        try:
            self.processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True)
            self.model = model_cls.from_pretrained(cfg.model_id, **model_kwargs)
            self.model.eval()
            if self.device.startswith("cuda"):
                self.model.to(self.device)
            else:
                self.model.to("cpu")
            self._sanitize_generation_config_for_greedy(self.model, cfg.temperature)
        except Exception as exc:
            raise BackendInitError(f"failed to load model_id={cfg.model_id}: {exc}") from exc

    def _load_image(self, image_path: Path):
        try:
            from PIL import Image
        except Exception as exc:
            raise BackendInferenceError(f"PIL import failed: {exc}") from exc
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as exc:
            raise BackendInferenceError(f"failed to open image: {image_path} ({exc})") from exc

    def label(self, task: Dict[str, Any], image_path: Optional[Path]) -> Tuple[Dict[str, Any], str]:
        if image_path is None:
            raise BackendInferenceError("qwen backend requires image_path")

        prompt = build_prompt(task)
        image = self._load_image(image_path)

        # Prefer chat template when available
        text_in = prompt
        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text_in = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text_in = prompt

        try:
            inputs = self.processor(text=[text_in], images=[image], return_tensors="pt")
        except Exception as exc:
            raise BackendInferenceError(f"processor encode failed: {exc}") from exc

        try:
            if hasattr(inputs, "to"):
                inputs = inputs.to(self.device)
            else:
                for k, v in list(inputs.items()):
                    if hasattr(v, "to"):
                        inputs[k] = v.to(self.device)
        except Exception as exc:
            raise BackendInferenceError(f"inputs.to(device={self.device}) failed: {exc}") from exc

        gen_kwargs = build_generation_kwargs(
            max_new_tokens=int(self.cfg.max_new_tokens),
            temperature=float(self.cfg.temperature),
        )

        try:
            with self._torch.inference_mode():
                generated = self.model.generate(**inputs, **gen_kwargs)
        except Exception as exc:
            raise BackendInferenceError(f"model.generate failed: {exc}") from exc

        try:
            input_len = 0
            if isinstance(inputs, dict) and "input_ids" in inputs:
                input_len = int(inputs["input_ids"].shape[-1])
            if input_len > 0:
                generated = generated[:, input_len:]
            raw_text = self.processor.batch_decode(generated, skip_special_tokens=True)[0]
        except Exception as exc:
            raise BackendInferenceError(f"decode failed: {exc}") from exc

        parsed = _extract_first_json_object(raw_text)
        compact_recovered = _extract_compact_selection_from_text(raw_text)
        if compact_recovered is not None and not _has_model_selection_keys(parsed):
            parsed = compact_recovered
        if parsed is None:
            raise BackendInferenceError("failed to parse json from model output")
        return parsed, raw_text


BACKEND_REGISTRY = {
    "heuristic": HeuristicBackend,
    "qwen25_vl": Qwen25VLHFBackend,
    "qwen25_vl_hf": Qwen25VLHFBackend,
}


def make_backend(backend_name: str, cfg: BackendConfig) -> BaseVLMBackend:
    name = str(backend_name).strip().lower()
    cls = BACKEND_REGISTRY.get(name)
    if cls is None:
        raise BackendInitError(f"unknown backend: {backend_name}")
    return cls(cfg)


def build_image_index(image_dir: Optional[Path]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if image_dir is None or (not image_dir.exists()):
        return out
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    for p in sorted(image_dir.glob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        stem = p.stem
        if stem not in out:
            out[stem] = p
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VLM/MLLM teacher label generator (Section 10)")
    p.add_argument("--teacher_scores_jsonl", required=True)
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--output_meta_jsonl", default="")
    p.add_argument("--summary_json", default="")

    p.add_argument("--backend", default="qwen25_vl", choices=sorted(BACKEND_REGISTRY.keys()))
    p.add_argument("--fallback_backend", default="heuristic", help="heuristic|none")
    p.add_argument("--model_id", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--device", default="auto", help="auto|cuda|cuda:0|cpu")
    p.add_argument("--dtype", default="auto", help="auto|float16|bfloat16|float32")
    p.add_argument("--max_new_tokens", type=int, default=768)
    p.add_argument("--temperature", type=float, default=0.0)

    p.add_argument("--image_dir", default="")
    p.add_argument("--target_ar", default="all", help="all or CSV(e.g., 1:1,16:9)")
    p.add_argument("--top_m", type=int, default=12, help="input candidates per AR for VLM")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--max_retries", type=int, default=2)
    p.add_argument("--prompt_version", default="crop_label_candidate_ids_v2")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--save_raw_response", type=int, default=0)
    p.add_argument("--debug_dir", default="")

    p.add_argument("--skip_on_oom", type=int, default=1)
    p.add_argument("--fallback_cpu_on_oom", type=int, default=1)
    p.add_argument("--fallback_cpu_max_images", type=int, default=3)
    p.add_argument("--skip_if_fallback_failed", type=int, default=1)
    p.add_argument("--strict_backend_init", type=int, default=1, help="1이면 primary backend init 실패 시 즉시 종료")
    p.add_argument("--num_shards", type=int, default=1, help="입력 image_id를 num_shards로 해시 샤딩")
    p.add_argument("--shard_index", type=int, default=0, help="현재 샤드 인덱스(0-based)")
    return p.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    random.seed(int(args.seed))

    in_path = Path(args.teacher_scores_jsonl)
    if not in_path.exists():
        raise FileNotFoundError(f"teacher_scores_jsonl not found: {in_path}")

    out_jsonl = Path(args.output_jsonl)
    ensure_parent(out_jsonl)

    out_meta = Path(args.output_meta_jsonl) if str(args.output_meta_jsonl).strip() else None
    if out_meta is not None:
        ensure_parent(out_meta)

    out_summary = Path(args.summary_json) if str(args.summary_json).strip() else None
    if out_summary is not None:
        ensure_parent(out_summary)

    debug_dir = Path(args.debug_dir) if str(args.debug_dir).strip() else None
    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)

    image_dir = Path(args.image_dir) if str(args.image_dir).strip() else None
    image_index = build_image_index(image_dir)

    backend_cfg = BackendConfig(
        model_id=str(args.model_id),
        device=str(args.device),
        dtype=str(args.dtype),
        max_new_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
    )

    backend_name = str(args.backend).strip().lower()
    fallback_name = str(args.fallback_backend).strip().lower()
    if fallback_name in {"", "none", "null", "off", "0"}:
        fallback_name = ""
    strict_backend_init = parse_bool01(args.strict_backend_init, True)

    backend: Optional[BaseVLMBackend] = None
    fallback_backend: Optional[BaseVLMBackend] = None

    summary_counter = Counter()
    per_ar_counter = Counter()

    cpu_fallback_active = False
    cpu_fallback_backend: Optional[BaseVLMBackend] = None
    cpu_fallback_seen_images: set[str] = set()
    cpu_fallback_limit = max(0, int(args.fallback_cpu_max_images))

    try:
        backend = make_backend(backend_name, backend_cfg)
        summary_counter["backend_init_ok"] += 1
    except Exception as exc:
        summary_counter["backend_init_fail"] += 1
        print(f"[warn] backend init failed ({backend_name}): {exc}")
        if strict_backend_init and backend_name != "heuristic":
            raise RuntimeError(
                f"primary backend init failed and strict_backend_init=1: backend={backend_name}, error={exc}"
            ) from exc
        if fallback_name:
            try:
                fallback_backend = make_backend(
                    fallback_name,
                    BackendConfig(
                        model_id=fallback_name,
                        device="cpu",
                        dtype="auto",
                        max_new_tokens=0,
                        temperature=0.0,
                    ),
                )
                summary_counter["fallback_backend_init_ok"] += 1
                backend = None
            except Exception as fexc:
                summary_counter["fallback_backend_init_fail"] += 1
                raise RuntimeError(f"Both backend and fallback backend failed: {exc} / {fexc}")
        else:
            raise

    if fallback_backend is None and fallback_name and fallback_name != backend_name:
        try:
            fallback_backend = make_backend(
                fallback_name,
                BackendConfig(
                    model_id=fallback_name,
                    device="cpu",
                    dtype="auto",
                    max_new_tokens=0,
                    temperature=0.0,
                ),
            )
            summary_counter["fallback_backend_init_ok"] += 1
        except Exception as exc:
            summary_counter["fallback_backend_init_fail"] += 1
            print(f"[warn] fallback backend init failed ({fallback_name}): {exc}")

    max_images = max(0, int(args.max_images))
    max_retries = max(0, int(args.max_retries))
    num_shards = max(1, int(args.num_shards))
    shard_index = int(args.shard_index)
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"invalid shard args: shard_index={shard_index}, num_shards={num_shards}")

    num_images = 0
    num_images_scanned = 0
    num_tasks = 0

    target_ar_req = str(args.target_ar).strip().lower()

    with in_path.open("r", encoding="utf-8") as f_in, out_jsonl.open("w", encoding="utf-8") as f_out:
        f_meta = out_meta.open("w", encoding="utf-8") if out_meta is not None else None
        try:
            for line in tqdm(f_in, desc="vlm_teacher", unit="img"):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    summary_counter["row_json_parse_fail"] += 1
                    continue
                image_id = str(row.get("image_id", "")).strip()
                if not image_id:
                    summary_counter["row_missing_image_id"] += 1
                    continue

                num_images_scanned += 1
                if max_images > 0 and num_images_scanned > max_images:
                    break
                if num_shards > 1 and stable_shard_index(image_id, num_shards) != shard_index:
                    summary_counter["row_not_in_shard"] += 1
                    continue
                num_images += 1

                meta_norm_v1 = build_meta_norm_v1(row)
                if f_meta is not None:
                    f_meta.write(json.dumps(meta_norm_v1, ensure_ascii=False) + "\n")

                scorer = row.get("teacher_scorer", {}) if isinstance(row.get("teacher_scorer"), dict) else {}
                results_by_ar = scorer.get("results_by_ar", {}) if isinstance(scorer.get("results_by_ar"), dict) else {}
                ars = _target_ar_list(results_by_ar, "all" if target_ar_req == "all" else str(args.target_ar))
                if not ars:
                    summary_counter["row_no_target_ar"] += 1
                    continue

                image_path = image_index.get(image_id)
                if image_path is None and backend is not None and backend_name.startswith("qwen"):
                    summary_counter["image_missing_for_qwen"] += 1

                for target_ar in ars:
                    task = build_task(
                        row,
                        target_ar,
                        top_m=max(1, int(args.top_m)),
                        top_k=max(1, int(args.top_k)),
                        prompt_version=str(args.prompt_version),
                    )
                    if task is None:
                        summary_counter["task_build_fail"] += 1
                        continue

                    num_tasks += 1
                    per_ar_counter[target_ar] += 1

                    parsed_obj: Optional[Dict[str, Any]] = None
                    raw_text: str = ""
                    used_backend_name = backend_name if backend is not None else (fallback_name or "none")
                    fallback_reason = ""
                    t0 = time.time()

                    if backend is not None:
                        # CPU fallback budget check (applies once qwen OOM triggered)
                        if cpu_fallback_active:
                            if image_id not in cpu_fallback_seen_images:
                                if len(cpu_fallback_seen_images) >= cpu_fallback_limit:
                                    fallback_reason = "cpu_fallback_limit_reached"
                                else:
                                    cpu_fallback_seen_images.add(image_id)

                        if not fallback_reason:
                            active_backend = cpu_fallback_backend if cpu_fallback_active and cpu_fallback_backend is not None else backend
                            used_backend_name = active_backend.backend_name

                            if active_backend.backend_name.startswith("qwen") and image_path is None:
                                fallback_reason = "missing_image"
                            else:
                                for attempt in range(max_retries + 1):
                                    try:
                                        parsed_obj, raw_text = active_backend.label(task, image_path)
                                        break
                                    except Exception as exc:
                                        summary_counter["backend_infer_fail"] += 1
                                        if is_oom_error(exc):
                                            summary_counter["backend_oom"] += 1
                                            # Try CPU fallback once
                                            if (
                                                parse_bool01(args.fallback_cpu_on_oom, True)
                                                and not cpu_fallback_active
                                                and active_backend.backend_name.startswith("qwen")
                                                and str(getattr(active_backend, "device", "")).startswith("cuda")
                                            ):
                                                try:
                                                    cpu_fallback_backend = make_backend(
                                                        backend_name,
                                                        BackendConfig(
                                                            model_id=str(args.model_id),
                                                            device="cpu",
                                                            dtype="float32",
                                                            max_new_tokens=int(args.max_new_tokens),
                                                            temperature=float(args.temperature),
                                                        ),
                                                    )
                                                    cpu_fallback_active = True
                                                    cpu_fallback_seen_images.add(image_id)
                                                    summary_counter["oom_switched_to_cpu"] += 1
                                                    # retry immediately on CPU backend
                                                    active_backend = cpu_fallback_backend
                                                    used_backend_name = active_backend.backend_name
                                                    continue
                                                except Exception as cpu_exc:
                                                    summary_counter["cpu_fallback_init_fail"] += 1
                                                    print(f"[warn] cpu fallback init failed: {cpu_exc}")
                                            if parse_bool01(args.skip_on_oom, True):
                                                fallback_reason = f"oom:{exc}"
                                                break
                                        # non-oom or retry exhaustion
                                        if attempt >= max_retries:
                                            fallback_reason = f"infer_fail:{exc}"
                                            break

                    if parsed_obj is not None:
                        parsed_profile = _classify_parsed_object(parsed_obj)
                        summary_counter[f"parsed_profile_{parsed_profile}"] += 1
                        if isinstance(parsed_obj.get("selected_topk"), list):
                            summary_counter["parsed_has_selected_topk"] += 1
                        if isinstance(parsed_obj.get("selected_candidate_ids"), list):
                            summary_counter["parsed_has_selected_candidate_ids"] += 1
                        if isinstance(parsed_obj.get("selected_ids"), list):
                            summary_counter["parsed_has_selected_ids"] += 1
                        if isinstance(parsed_obj.get("selected_crops"), list):
                            summary_counter["parsed_has_selected_crops"] += 1
                        pexp = parsed_obj.get("explanations")
                        if isinstance(pexp, dict):
                            summary_counter["parsed_has_explanations"] += 1
                            if str(pexp.get("short", "")).strip() and str(pexp.get("long", "")).strip():
                                summary_counter["parsed_has_explanations_full"] += 1
                        try:
                            normalized = normalize_backend_output(
                                task,
                                parsed_obj,
                                backend_name=used_backend_name,
                                model_id=(backend.model_id if backend is not None else used_backend_name),
                            )
                            summary_counter["normalized_ok"] += 1
                        except Exception as exc:
                            fallback_reason = f"normalize_fail:{exc}"
                            summary_counter["normalize_fail"] += 1
                            normalized = None
                    else:
                        normalized = None

                    if normalized is None:
                        if fallback_backend is not None:
                            try:
                                fb_obj, fb_raw = fallback_backend.label(task, image_path)
                                normalized = normalize_backend_output(
                                    task,
                                    fb_obj,
                                    backend_name=fallback_backend.backend_name,
                                    model_id=fallback_backend.model_id,
                                )
                                if not raw_text:
                                    raw_text = str(fb_raw or "")
                                note = normalized.get("validator", {}).get("notes", "")
                                normalized.setdefault("validator", {})["notes"] = (
                                    f"{note}; fallback_reason={fallback_reason}" if note else f"fallback_reason={fallback_reason}"
                                )
                                summary_counter["fallback_used"] += 1
                                used_backend_name = fallback_backend.backend_name
                            except Exception as exc:
                                summary_counter["fallback_fail"] += 1
                                if parse_bool01(args.skip_if_fallback_failed, True):
                                    summary_counter["task_skipped"] += 1
                                    continue
                                raise RuntimeError(f"fallback backend failed: {exc}") from exc
                        else:
                            if parse_bool01(args.skip_if_fallback_failed, True):
                                summary_counter["task_skipped"] += 1
                                continue
                            raise RuntimeError(f"task failed without fallback: {fallback_reason}")

                    latency_ms = int((time.time() - t0) * 1000.0)
                    normalized.setdefault("timing", {})
                    normalized["timing"]["latency_ms"] = latency_ms
                    normalized["timing"]["n_candidates_in"] = len(as_list(task.get("candidates")))
                    normalized.setdefault("input_refs", {})
                    normalized["input_refs"]["prompt_version"] = str(args.prompt_version)
                    normalized["input_refs"]["teacher_scores_source"] = str(in_path)
                    normalized["input_refs"]["image_path"] = str(image_path) if image_path is not None else ""
                    normalized.setdefault("meta_norm_v1", task.get("meta_norm_v1", {}))

                    f_out.write(json.dumps(normalized, ensure_ascii=False) + "\n")
                    summary_counter["task_written"] += 1

                    if parse_bool01(args.save_raw_response, False) and debug_dir is not None:
                        raw_dir = debug_dir / "raw_responses"
                        raw_dir.mkdir(parents=True, exist_ok=True)
                        safe_ar = target_ar.replace(":", "x")
                        raw_path = raw_dir / f"{image_id}__{safe_ar}.txt"
                        raw_path.write_text(raw_text or "", encoding="utf-8")

        finally:
            if f_meta is not None:
                f_meta.close()

    if backend is not None:
        backend.close()
    if cpu_fallback_backend is not None and cpu_fallback_backend is not backend:
        cpu_fallback_backend.close()
    if fallback_backend is not None:
        fallback_backend.close()

    summary = {
        "schema_version": "vlm_teacher_summary_v1",
        "input_teacher_scores_jsonl": str(in_path),
        "output_jsonl": str(out_jsonl),
        "output_meta_jsonl": str(out_meta) if out_meta is not None else "",
        "backend": backend_name,
        "fallback_backend": fallback_name or "none",
        "model_id": str(args.model_id),
        "device": str(args.device),
        "dtype": str(args.dtype),
        "target_ar": str(args.target_ar),
        "top_m": int(args.top_m),
        "top_k": int(args.top_k),
        "max_images": int(args.max_images),
        "num_images_seen": int(num_images),
        "num_images_scanned": int(num_images_scanned),
        "num_tasks_seen": int(num_tasks),
        "num_shards": int(num_shards),
        "shard_index": int(shard_index),
        "counts": dict(summary_counter),
        "per_ar_tasks": dict(per_ar_counter),
        "cpu_fallback_active": bool(cpu_fallback_active),
        "cpu_fallback_image_count": int(len(cpu_fallback_seen_images)),
    }

    if out_summary is not None:
        out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        "[done] vlm teacher labeler: "
        f"written={summary_counter.get('task_written', 0)} "
        f"fallback={summary_counter.get('fallback_used', 0)} "
        f"skipped={summary_counter.get('task_skipped', 0)}"
    )
    print(f"[done] output_jsonl={out_jsonl}")
    if out_meta is not None:
        print(f"[done] output_meta_jsonl={out_meta}")
    if out_summary is not None:
        print(f"[done] summary_json={out_summary}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"[error] {exc}")
        raise SystemExit(1)
