from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - optional dependency at runtime
    Image = None
    ImageDraw = None
    ImageFont = None


SEVERE_REJECT_TAGS = {
    "face_cut",
    "joint_cutoff",
    "head_top_cut",
    "text_cutoff",
    "lookroom_cut",
    "lookroom_violation",
}
SOFT_ISSUE_TOKENS = (
    "loose",
    "weak",
    "marginal",
    "partial",
    "off_target",
)
DECISION_ID_BY_TYPE = {
    "keep_full": 0,
    "minimal_crop": 1,
    "crop": 2,
}
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
BIN_TARGET_KEYS = ("face_cut", "joint_cut")
ORD_TARGET_KEYS = (
    "subject_coverage",
    "subject_scale",
    "headroom",
    "lookroom",
    "text_keep",
    "copyspace",
    "context",
)
REG_TARGET_KEYS = (
    "subject_coverage_ratio",
    "headroom_ratio",
    "lookroom_ratio",
    "text_keep_ratio",
    "horizon_y",
    "horizon_visible_ratio",
    "symmetry_score",
)
REG_APPLICABLE_KEY = {
    "subject_coverage_ratio": "subject_coverage",
    "headroom_ratio": "headroom",
    "lookroom_ratio": "lookroom",
    "text_keep_ratio": "text_keep",
    "horizon_y": "horizon_state",
    "horizon_visible_ratio": "horizon_state",
    "symmetry_score": "context",
}
ORDINAL_LABEL_MAPS: Dict[str, Dict[str, int]] = {
    "subject_coverage": {
        "poor": 0,
        "marginal": 1,
        "good": 2,
        "excellent": 3,
    },
    "subject_scale": {
        "too_loose": 0,
        "ideal_scale": 1,
        "too_tight": 2,
    },
    "headroom": {
        "headroom_tight": 0,
        "headroom_ok": 1,
        "headroom_loose": 2,
    },
    "lookroom": {
        "lookroom_tight": 0,
        "lookroom_adequate": 1,
        "lookroom_loose": 2,
    },
    "text_keep": {
        "text_cut": 0,
        "text_partial": 1,
        "text_preserved": 2,
    },
    "copyspace": {
        "copyspace_missing": 0,
        "copyspace_partial": 1,
        "copyspace_preserved": 2,
    },
    "context": {
        "context_lost": 0,
        "context_partial": 1,
        "context_preserved": 2,
    },
}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def maybe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def softmax_local(scores: Sequence[float], tau: float) -> List[float]:
    if not scores:
        return []
    tau_eff = max(1e-6, float(tau))
    scaled = [float(s) / tau_eff for s in scores]
    mx = max(scaled)
    exps = [math.exp(v - mx) for v in scaled]
    denom = sum(exps)
    if denom <= 0.0:
        return [1.0 / float(len(scores))] * len(scores)
    return [v / denom for v in exps]


def robust_z_scores(scores: Sequence[float], eps: float = 1e-6) -> List[float]:
    if not scores:
        return []
    seq = [float(s) for s in scores]
    vals = sorted(seq)
    n = len(vals)
    median = vals[n // 2] if n % 2 == 1 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    abs_dev = sorted(abs(v - median) for v in vals)
    mad = abs_dev[n // 2] if n % 2 == 1 else 0.5 * (abs_dev[n // 2 - 1] + abs_dev[n // 2])
    score_range = vals[-1] - vals[0] if vals else 0.0
    scale = max(1.4826 * mad, 0.05 * score_range, eps)
    return [clamp((s - median) / scale, -6.0, 6.0) for s in seq]


def rank_pct_desc(scores: Sequence[float]) -> List[float]:
    n = len(scores)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    order = sorted(range(n), key=lambda idx: float(scores[idx]), reverse=True)
    out = [0.0] * n
    denom = float(n - 1)
    for rank, idx in enumerate(order):
        out[idx] = 1.0 - (float(rank) / denom)
    return out


def entropy(values: Sequence[float]) -> float:
    ent = 0.0
    for value in values:
        v = float(value)
        if v <= 0.0:
            continue
        ent -= v * math.log(v)
    return ent


def norm_xyxy_to_xywh(bbox: Sequence[Any]) -> List[float]:
    x1 = clamp(safe_float(bbox[0]), 0.0, 1.0)
    y1 = clamp(safe_float(bbox[1]), 0.0, 1.0)
    x2 = clamp(safe_float(bbox[2]), 0.0, 1.0)
    y2 = clamp(safe_float(bbox[3]), 0.0, 1.0)
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def summarize_numeric(values: Sequence[Optional[float]]) -> Dict[str, float]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"count": 0, "min": 0.0, "p50": 0.0, "mean": 0.0, "p90": 0.0, "max": 0.0}
    vals.sort()

    def pct(p: float) -> float:
        idx = min(len(vals) - 1, max(0, int(round((len(vals) - 1) * p))))
        return float(vals[idx])

    return {
        "count": len(vals),
        "min": round(vals[0], 6),
        "p50": round(pct(0.50), 6),
        "mean": round(sum(vals) / float(len(vals)), 6),
        "p90": round(pct(0.90), 6),
        "max": round(vals[-1], 6),
    }


def safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_teacher_records(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def round_opt(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def normalize_label(label: Any) -> str:
    text = str(label or "").strip().lower()
    if not text:
        return "na"
    if text.endswith("_na"):
        return "na"
    return text


def normalize_bbox_xyxy(values: Any) -> List[float]:
    coords = list(values) if isinstance(values, (list, tuple)) else []
    if len(coords) < 4:
        coords = [0.0, 0.0, 1.0, 1.0]
    nums = []
    for idx, value in enumerate(coords[:4]):
        default = 0.0 if idx in {0, 1} else 1.0
        nums.append(clamp(safe_float(value, default), 0.0, 1.0))
    x1, y1, x2, y2 = nums
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)]


def normalize_bbox_or_none(values: Any) -> Optional[List[float]]:
    if not isinstance(values, (list, tuple)) or len(values) < 4:
        return None
    return normalize_bbox_xyxy(values)


def bbox_xyxy_to_cxcywh(values: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = [safe_float(v) for v in list(values)[:4]]
    return [
        round((x1 + x2) * 0.5, 6),
        round((y1 + y2) * 0.5, 6),
        round(max(0.0, x2 - x1), 6),
        round(max(0.0, y2 - y1), 6),
    ]


def bbox_is_valid(values: Any) -> bool:
    if not isinstance(values, (list, tuple)) or len(values) < 4:
        return False
    x1, y1, x2, y2 = [safe_float(v) for v in list(values)[:4]]
    return x2 > x1 and y2 > y1 and x1 >= 0.0 and y1 >= 0.0 and x2 <= 1.0 and y2 <= 1.0


def first_nonempty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def decision_id(decision_type: str) -> int:
    return DECISION_ID_BY_TYPE.get(str(decision_type or ""), -1)


def build_subject_mode_vocab(teacher_records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    modes: set[str] = set()
    for rec in teacher_records:
        results_by_ar = safe_dict(safe_dict(rec.get("teacher_scorer")).get("results_by_ar"))
        route_global = safe_dict(rec.get("route_global"))
        subject_prior = safe_dict(rec.get("subject_prior"))
        for ar_res in results_by_ar.values():
            routing = safe_dict(safe_dict(ar_res).get("routing"))
            mode = str(
                first_nonempty(
                    routing.get("subject_mode"),
                    route_global.get("subject_mode"),
                    subject_prior.get("subject_mode"),
                    "unknown",
                )
            ).strip()
            if mode:
                modes.add(mode)
    if not modes:
        modes.add("unknown")
    return {mode: idx for idx, mode in enumerate(sorted(modes))}


def resolve_default_image_root(teacher_scores_jsonl: Path, image_root_arg: Optional[str]) -> Optional[Path]:
    if image_root_arg:
        return Path(image_root_arg)
    try:
        candidate = teacher_scores_jsonl.parents[3] / "images"
    except IndexError:
        return None
    return candidate


def resolve_image_path(image_root: Optional[Path], image_id: str) -> Optional[Path]:
    if image_root is None:
        return None
    for ext in IMAGE_EXTENSIONS:
        path = image_root / f"{image_id}{ext}"
        if path.exists():
            return path
    matches = sorted(image_root.glob(f"{image_id}.*"))
    return matches[0] if matches else None


def dedupe_candidates(ar_res: Dict[str, Any]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for key in (
        "cheap_top_m",
        "selected_topk",
        "hard_negatives",
        "also_considered_rejected",
    ):
        for cand in safe_list(ar_res.get(key)):
            cid = str(safe_dict(cand).get("candidate_id", ""))
            if not cid or cid in seen:
                continue
            merged.append(copy.deepcopy(cand))
            seen.add(cid)
    for key in ("best_candidate", "baseline_candidate"):
        cand = safe_dict(ar_res.get(key))
        cid = str(cand.get("candidate_id", ""))
        if not cid or cid in seen:
            continue
        merged.append(copy.deepcopy(cand))
        seen.add(cid)
    return merged


def has_soft_issue(candidate: Dict[str, Any]) -> bool:
    labels = safe_dict(candidate.get("checklist_labels"))
    for label in labels.values():
        text = str(label).strip().lower()
        if any(token in text for token in SOFT_ISSUE_TOKENS):
            return True
    return False


def is_unsafe_candidate(candidate: Dict[str, Any]) -> bool:
    reject_tags = {str(tag) for tag in safe_list(candidate.get("reject_tags"))}
    if bool(candidate.get("hard_reject", False)):
        return True
    return len(reject_tags.intersection(SEVERE_REJECT_TAGS)) > 0


def annotate_group_candidates(
    image_id: str,
    target_ar: str,
    ar_res: Dict[str, Any],
    softmax_tau: float,
    hard_negative_rank_pct_max: float,
    near_margin_max: float,
) -> List[Dict[str, Any]]:
    candidates = dedupe_candidates(ar_res)
    if not candidates:
        return []
    selected_ids = [str(safe_dict(c).get("candidate_id", "")) for c in safe_list(ar_res.get("selected_topk"))]
    selected_id_set = {cid for cid in selected_ids if cid}
    top1_id = selected_ids[0] if selected_ids else str(safe_dict(ar_res.get("best_candidate")).get("candidate_id", ""))
    best_id = str(safe_dict(ar_res.get("best_candidate")).get("candidate_id", ""))
    baseline_id = str(safe_dict(ar_res.get("baseline_candidate")).get("candidate_id", ""))
    chosen_id = str(safe_dict(ar_res.get("decision")).get("chosen_candidate_id", ""))
    decision_type = str(safe_dict(ar_res.get("decision")).get("decision_type", ""))

    raw_scores = [safe_float(safe_dict(c.get("scores")).get("rank", safe_dict(c.get("scores")).get("final", 0.0))) for c in candidates]
    rank_pcts = rank_pct_desc(raw_scores)
    z_scores = robust_z_scores(raw_scores)
    softmax_scores = softmax_local(raw_scores, tau=softmax_tau)
    pseudo_mos = [1.0 + 4.0 * sigmoid(z) for z in z_scores]
    top1_score = max(raw_scores) if raw_scores else 0.0

    annotated: List[Dict[str, Any]] = []
    for idx, cand in enumerate(candidates):
        out = copy.deepcopy(cand)
        out["group_key"] = f"{image_id}::{target_ar}"
        out["image_id"] = image_id
        out["target_ar"] = target_ar
        out["score_rank_pct"] = float(rank_pcts[idx])
        out["score_z_local"] = float(z_scores[idx])
        out["score_softmax_local"] = float(softmax_scores[idx])
        out["pseudo_mos_1to5"] = float(pseudo_mos[idx])
        out["is_top1"] = str(out.get("candidate_id", "")) == top1_id
        out["is_selected_topk"] = str(out.get("candidate_id", "")) in selected_id_set
        out["is_best_candidate"] = str(out.get("candidate_id", "")) == best_id
        out["is_baseline_candidate"] = str(out.get("candidate_id", "")) == baseline_id
        out["is_chosen_candidate"] = str(out.get("candidate_id", "")) == chosen_id
        out["decision_type"] = decision_type
        out["mode"] = str(safe_dict(ar_res.get("routing")).get("subject_mode", ""))
        out["policy_id"] = str(safe_dict(ar_res.get("routing")).get("policy_id", ""))
        out["is_unsafe_negative"] = is_unsafe_candidate(out)
        out["is_soft_positive"] = bool(out["is_selected_topk"] and has_soft_issue(out) and not out["is_unsafe_negative"])
        out["is_positive_candidate"] = bool(
            out["is_top1"]
            or (out["is_selected_topk"] and not out["is_unsafe_negative"])
            or (
                decision_type in {"keep_full", "minimal_crop"}
                and out["is_baseline_candidate"]
                and not out["is_unsafe_negative"]
            )
        )
        margin_to_top1 = max(0.0, top1_score - raw_scores[idx])
        out["score_margin_to_top1"] = float(margin_to_top1)
        out["is_hard_negative"] = bool(
            out["is_unsafe_negative"] and out["score_rank_pct"] <= float(hard_negative_rank_pct_max)
        )
        out["is_near_negative"] = bool(
            (not out["is_positive_candidate"])
            and (not out["is_unsafe_negative"])
            and margin_to_top1 > 0.0
            and margin_to_top1 <= float(near_margin_max)
        )
        if out["is_top1"]:
            label_type = "top1"
        elif out["is_positive_candidate"] and out["is_baseline_candidate"]:
            label_type = "baseline_positive"
        elif out["is_positive_candidate"] and out["is_soft_positive"]:
            label_type = "soft_positive"
        elif out["is_positive_candidate"]:
            label_type = "topk_positive"
        elif out["is_hard_negative"]:
            label_type = "hard_negative"
        elif out["is_unsafe_negative"]:
            label_type = "unsafe_negative"
        elif out["is_near_negative"]:
            label_type = "near_negative"
        else:
            label_type = "negative"
        out["label_type"] = label_type
        annotated.append(out)

    annotated.sort(
        key=lambda cand: (
            safe_float(safe_dict(cand.get("scores")).get("rank", safe_dict(cand.get("scores")).get("final", 0.0))),
            safe_float(cand.get("score_softmax_local", 0.0)),
        ),
        reverse=True,
    )
    return annotated


def infer_mode_flags(routing: Dict[str, Any]) -> Dict[str, bool]:
    mode = str(routing.get("subject_mode", "")).strip().lower()
    flags = safe_dict(routing.get("flags"))
    return {
        "portrait": mode.startswith("portrait") or bool(flags.get("subject_mode_has_person")),
        "scene": mode.startswith("scene") or "landscape" in mode or "architecture" in mode or bool(flags.get("is_landscape_scene")),
        "copyspace": "copyspace" in mode or bool(flags.get("has_copy_space")) or bool(flags.get("subject_mode_has_copyspace_tag")),
        "text": "text" in mode or "document" in mode or bool(flags.get("subject_mode_has_text_heavy")),
    }


def raw_checklist_entry(candidate: Dict[str, Any], key: str) -> Dict[str, Any]:
    return safe_dict(safe_dict(candidate.get("checklist")).get(key))


def checklist_label_for(candidate: Dict[str, Any], raw_key: str) -> str:
    raw_labels = safe_dict(candidate.get("checklist_labels"))
    label = first_nonempty(raw_labels.get(raw_key), raw_checklist_entry(candidate, raw_key).get("label"))
    return normalize_label(label)


def build_checklist_labels(candidate: Dict[str, Any]) -> Dict[str, str]:
    horizon = raw_checklist_entry(candidate, "horizon")
    horizon_state = normalize_label(first_nonempty(horizon.get("state"), horizon.get("label")))
    if horizon_state in {"na", "none"}:
        horizon_state = "na"
    return {
        "subject_coverage": checklist_label_for(candidate, "subject_coverage"),
        "subject_scale": checklist_label_for(candidate, "subject_scale"),
        "headroom": checklist_label_for(candidate, "headroom"),
        "lookroom": checklist_label_for(candidate, "lookroom"),
        "face_cut": checklist_label_for(candidate, "face_cut"),
        "joint_cut": checklist_label_for(candidate, "joint_cut"),
        "text_keep": checklist_label_for(candidate, "text_keep_ratio"),
        "copyspace": checklist_label_for(candidate, "copyspace"),
        "horizon_state": horizon_state,
        "context": checklist_label_for(candidate, "context"),
    }


def build_checklist_scores(candidate: Dict[str, Any]) -> Dict[str, Optional[float]]:
    raw_check = safe_dict(candidate.get("checklist"))
    macro_components = safe_dict(candidate.get("macro_components"))
    score_components = safe_dict(safe_dict(candidate.get("scores")).get("components"))
    symmetry_score = first_nonempty(macro_components.get("C_sym"), score_components.get("r_sym"))
    return {
        "subject_coverage_ratio": round_opt(maybe_float(safe_dict(raw_check.get("subject_coverage")).get("value"))),
        "headroom_ratio": round_opt(maybe_float(safe_dict(raw_check.get("headroom")).get("value"))),
        "lookroom_ratio": round_opt(maybe_float(safe_dict(raw_check.get("lookroom")).get("value"))),
        "text_keep_ratio": round_opt(maybe_float(safe_dict(raw_check.get("text_keep_ratio")).get("value"))),
        "horizon_y": round_opt(maybe_float(safe_dict(raw_check.get("horizon")).get("value"))),
        "horizon_visible_ratio": round_opt(maybe_float(safe_dict(raw_check.get("horizon")).get("visible_ratio"))),
        "symmetry_score": round_opt(maybe_float(symmetry_score)),
    }


def build_applicable_mask(routing: Dict[str, Any], candidate: Dict[str, Any], labels: Dict[str, str]) -> Dict[str, int]:
    flags = infer_mode_flags(routing)
    text_keep_entry = raw_checklist_entry(candidate, "text_keep_ratio")
    horizon_entry = raw_checklist_entry(candidate, "horizon")
    return {
        "subject_coverage": 1,
        "subject_scale": 1,
        "headroom": int(flags["portrait"]),
        "lookroom": int(flags["portrait"]),
        "face_cut": int(flags["portrait"]),
        "joint_cut": int(flags["portrait"]),
        "text_keep": int(flags["text"] or bool(text_keep_entry.get("available"))),
        "copyspace": int(flags["copyspace"] or labels.get("copyspace") not in {"na", ""}),
        "horizon_state": int(
            flags["scene"]
            or safe_float(horizon_entry.get("visible_ratio", 0.0), 0.0) >= 0.25
            or labels.get("horizon_state") not in {"na", ""}
        ),
        "context": 1,
    }


def build_observed_mask(candidate: Dict[str, Any], labels: Dict[str, str], scores: Dict[str, Optional[float]]) -> Dict[str, int]:
    headroom = raw_checklist_entry(candidate, "headroom")
    lookroom = raw_checklist_entry(candidate, "lookroom")
    text_keep = raw_checklist_entry(candidate, "text_keep_ratio")
    horizon = raw_checklist_entry(candidate, "horizon")
    out = {
        "subject_coverage": int(scores.get("subject_coverage_ratio") is not None and labels.get("subject_coverage") != "na"),
        "subject_scale": int(labels.get("subject_scale") != "na"),
        "headroom": int(maybe_float(headroom.get("value")) is not None and labels.get("headroom") != "na"),
        "lookroom": int(maybe_float(lookroom.get("value")) is not None and labels.get("lookroom") != "na"),
        "face_cut": int(labels.get("face_cut") != "na"),
        "joint_cut": int(labels.get("joint_cut") != "na"),
        "text_keep": int(bool(text_keep.get("available")) and scores.get("text_keep_ratio") is not None and labels.get("text_keep") != "na"),
        "copyspace": int(labels.get("copyspace") != "na"),
        "horizon_state": int(
            labels.get("horizon_state") != "na"
            and maybe_float(horizon.get("value")) is not None
            and safe_float(horizon.get("visible_ratio", 0.0), 0.0) > 0.0
        ),
        "context": int(labels.get("context") != "na"),
        "subject_coverage_ratio": int(scores.get("subject_coverage_ratio") is not None),
        "headroom_ratio": int(scores.get("headroom_ratio") is not None),
        "lookroom_ratio": int(scores.get("lookroom_ratio") is not None),
        "text_keep_ratio": int(bool(text_keep.get("available")) and scores.get("text_keep_ratio") is not None),
        "horizon_y": int(scores.get("horizon_y") is not None and labels.get("horizon_state") != "na"),
        "horizon_visible_ratio": int(scores.get("horizon_visible_ratio") is not None and labels.get("horizon_state") != "na"),
        "symmetry_score": int(scores.get("symmetry_score") is not None),
    }
    return out


def binary_target_from_entry(field: str, candidate: Dict[str, Any], label: str) -> Optional[int]:
    value = maybe_float(raw_checklist_entry(candidate, field).get("value"))
    if value is not None:
        return 1 if value > 0.0 else 0
    normalized = normalize_label(label)
    if normalized == "na":
        return None
    if normalized.startswith("no_") or normalized.endswith("_ok") or "safe" in normalized:
        return 0
    return 1


def ordinal_target_from_label(field: str, candidate: Dict[str, Any], label: str) -> Optional[int]:
    normalized = normalize_label(label)
    if normalized == "na":
        return None
    mapping = ORDINAL_LABEL_MAPS.get(field, {})
    if normalized in mapping:
        return mapping[normalized]
    if field == "subject_scale":
        value = maybe_float(raw_checklist_entry(candidate, "subject_scale").get("value"))
        target_range = raw_checklist_entry(candidate, "subject_scale").get("target_range")
        if value is not None and isinstance(target_range, (list, tuple)) and len(target_range) >= 2:
            lo = safe_float(target_range[0], 0.0)
            hi = safe_float(target_range[1], lo)
            if value < lo:
                return 0
            if value <= hi:
                return 1
            return 2
    if field in {"headroom", "lookroom"}:
        if "tight" in normalized or "cut" in normalized:
            return 0
        if "ok" in normalized or "adequate" in normalized or "ideal" in normalized:
            return 1
        if "loose" in normalized:
            return 2
    if field == "context":
        if "lost" in normalized:
            return 0
        if "partial" in normalized:
            return 1
        if "preserved" in normalized:
            return 2
    if field == "copyspace":
        if "missing" in normalized:
            return 0
        if "partial" in normalized:
            return 1
        if "preserved" in normalized:
            return 2
    if field == "text_keep":
        if "cut" in normalized:
            return 0
        if "partial" in normalized:
            return 1
        if "preserved" in normalized:
            return 2
    return None


def derive_checklist_targets(
    candidate: Dict[str, Any],
    labels: Dict[str, str],
    scores: Dict[str, Optional[float]],
    applicable_mask: Dict[str, int],
    observed_mask: Dict[str, int],
) -> Dict[str, Dict[str, Any]]:
    bin_targets = {field: binary_target_from_entry(field, candidate, labels.get(field, "na")) for field in BIN_TARGET_KEYS}
    ord_targets = {field: ordinal_target_from_label(field, candidate, labels.get(field, "na")) for field in ORD_TARGET_KEYS}
    reg_targets = {field: scores.get(field) for field in REG_TARGET_KEYS}
    valid_mask: Dict[str, int] = {}
    for field, value in bin_targets.items():
        valid_mask[field] = int(bool(applicable_mask.get(field, 0)) and bool(observed_mask.get(field, 0)) and value is not None)
    for field, value in ord_targets.items():
        valid_mask[field] = int(bool(applicable_mask.get(field, 0)) and bool(observed_mask.get(field, 0)) and value is not None)
    for field, value in reg_targets.items():
        app_key = REG_APPLICABLE_KEY.get(field, "context")
        valid_mask[field] = int(bool(applicable_mask.get(app_key, 0)) and bool(observed_mask.get(field, 0)) and value is not None)
    return {
        "bin_targets": bin_targets,
        "ord_targets": ord_targets,
        "reg_targets": reg_targets,
        "valid_mask": valid_mask,
    }


def build_routing_record(
    teacher_record: Dict[str, Any],
    ar_res: Dict[str, Any],
    subject_mode_to_id: Dict[str, int],
) -> Dict[str, Any]:
    route_global = safe_dict(teacher_record.get("route_global"))
    subject_prior = safe_dict(teacher_record.get("subject_prior"))
    routing = safe_dict(ar_res.get("routing"))
    subject_mode = str(
        first_nonempty(
            routing.get("subject_mode"),
            route_global.get("subject_mode"),
            subject_prior.get("subject_mode"),
            "unknown",
        )
    )
    subject_mode_conf = safe_float(
        first_nonempty(
            routing.get("subject_mode_conf"),
            route_global.get("subject_mode_conf"),
            route_global.get("scene_conf"),
            0.0,
        ),
        0.0,
    )
    return {
        "subject_mode": subject_mode,
        "subject_mode_id": subject_mode_to_id.get(subject_mode, -1),
        "subject_mode_conf": round(subject_mode_conf, 6),
        "route_conf": round(subject_mode_conf, 6),
        "policy_id": str(first_nonempty(routing.get("policy_id"), route_global.get("policy_id"), subject_prior.get("policy_id"), "")),
        "scene_subtype": first_nonempty(routing.get("scene_subtype"), route_global.get("scene_subtype")),
        "shot_type": first_nonempty(routing.get("shot_type"), route_global.get("shot_type")),
        "subject_prior_bbox_norm_xyxy": normalize_bbox_or_none(subject_prior.get("bbox_norm_xyxy")),
        "subject_set": copy.deepcopy(first_nonempty(routing.get("subject_set"), route_global.get("subject_set"))),
        "router_rule_id": first_nonempty(routing.get("router_rule_id"), route_global.get("router_rule_id")),
        "flags": copy.deepcopy(first_nonempty(routing.get("flags"), route_global.get("flags"), {})),
    }


def build_baseline_record(ar_res: Dict[str, Any]) -> Dict[str, Any]:
    baseline = safe_dict(ar_res.get("baseline_candidate"))
    return {
        "candidate_id": str(baseline.get("candidate_id", "baseline_full")),
        "bbox_norm_xyxy": normalize_bbox_xyxy(baseline.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0])),
        "bbox_cxcywh": bbox_xyxy_to_cxcywh(normalize_bbox_xyxy(baseline.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0]))),
        "score_rank": round(safe_float(safe_dict(baseline.get("scores")).get("rank", 0.0)), 6),
        "score_policy": round(safe_float(safe_dict(baseline.get("scores")).get("policy", 0.0)), 6),
    }


def find_candidate(candidates: Sequence[Dict[str, Any]], candidate_id: str) -> Optional[Dict[str, Any]]:
    for candidate in candidates:
        if str(candidate.get("candidate_id", "")) == str(candidate_id):
            return candidate
    return None


def build_candidate_canonical_record(candidate: Dict[str, Any], routing: Dict[str, Any]) -> Dict[str, Any]:
    labels = build_checklist_labels(candidate)
    scores = build_checklist_scores(candidate)
    applicable_mask = build_applicable_mask(routing, candidate, labels)
    observed_mask = build_observed_mask(candidate, labels, scores)
    bbox = normalize_bbox_xyxy(candidate.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0]))
    raw_scores = safe_dict(candidate.get("scores"))
    macro_scores = safe_dict(candidate.get("macro_scores"))
    return {
        "candidate_id": str(candidate.get("candidate_id", "")),
        "source": str(candidate.get("source", "")),
        "label_type": str(candidate.get("label_type", "")),
        "is_positive_candidate": bool(candidate.get("is_positive_candidate", False)),
        "is_soft_positive": bool(candidate.get("is_soft_positive", False)),
        "is_hard_negative": bool(candidate.get("is_hard_negative", False)),
        "is_unsafe_negative": bool(candidate.get("is_unsafe_negative", False)),
        "bbox_norm_xyxy": bbox,
        "bbox_cxcywh": bbox_xyxy_to_cxcywh(bbox),
        "area_ratio": round(safe_float(candidate.get("area_ratio", 0.0)), 6),
        "score_targets": {
            "score_prob": round(sigmoid(safe_float(candidate.get("score_z_local", 0.0))), 6),
            "rank_pct": round(safe_float(candidate.get("score_rank_pct", 0.0)), 6),
            "z_local": round(safe_float(candidate.get("score_z_local", 0.0)), 6),
            "softmax_local": round(safe_float(candidate.get("score_softmax_local", 0.0)), 6),
            "pseudo_mos_1to5": round(safe_float(candidate.get("pseudo_mos_1to5", 1.0)), 6),
            "score_raw_rank": round(safe_float(raw_scores.get("rank", raw_scores.get("final", 0.0))), 6),
            "score_raw_policy": round(safe_float(raw_scores.get("policy", raw_scores.get("final", 0.0))), 6),
            "score_margin_to_top1": round(safe_float(candidate.get("score_margin_to_top1", 0.0)), 6),
        },
        "macro_targets": {
            "A_macro": round(safe_float(macro_scores.get("A_macro", 0.0)), 6),
            "S_macro": round(safe_float(macro_scores.get("S_macro", 0.0)), 6),
            "C_macro": round(safe_float(macro_scores.get("C_macro", 0.0)), 6),
            "T_macro": round(safe_float(macro_scores.get("T_macro", 0.0)), 6),
        },
        "checklist_labels": labels,
        "checklist_scores": scores,
        "applicable_mask": applicable_mask,
        "observed_mask": observed_mask,
        "why_tags": list(safe_list(candidate.get("why_tags"))),
        "reject_tags": list(safe_list(candidate.get("reject_tags"))),
        "provenance": {
            "source_lineage": copy.deepcopy(candidate.get("source_lineage")),
            "teacher_ids": copy.deepcopy(candidate.get("teacher_ids")),
            "teacher_stages": copy.deepcopy(candidate.get("teacher_stages")),
            "teacher_provenance": copy.deepcopy(candidate.get("teacher_provenance")),
        },
    }


def build_decision_record(
    image_id: str,
    target_ar: str,
    ar_res: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    decision = safe_dict(ar_res.get("decision"))
    baseline = safe_dict(ar_res.get("baseline_candidate"))
    best = safe_dict(ar_res.get("best_candidate"))
    chosen_id = str(decision.get("chosen_candidate_id", ""))
    chosen = find_candidate(candidates, chosen_id)
    if chosen is None and chosen_id == str(baseline.get("candidate_id", "")):
        chosen = baseline
    if chosen is None and chosen_id == str(best.get("candidate_id", "")):
        chosen = best
    decision_type = str(decision.get("decision_type", ""))
    return {
        "image_id": image_id,
        "target_ar": target_ar,
        "decision_type": decision_type,
        "decision_id": decision_id(decision_type),
        "delta_vs_base": round(safe_float(decision.get("delta_improve", 0.0)), 6),
        "tau_improve": round(safe_float(decision.get("tau_improve", 0.0)), 6),
        "mode": str(safe_dict(ar_res.get("routing")).get("subject_mode", "")),
        "policy_id": str(safe_dict(ar_res.get("routing")).get("policy_id", "")),
        "base_bbox": normalize_bbox_xyxy(baseline.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0])),
        "winner_pre_gate_bbox": normalize_bbox_xyxy(best.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0])),
        "winner_post_gate_bbox": normalize_bbox_xyxy(safe_dict(chosen).get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0])),
        "base_candidate_id": str(baseline.get("candidate_id", "")),
        "winner_pre_gate_candidate_id": str(best.get("candidate_id", "")),
        "winner_post_gate_candidate_id": chosen_id,
        "best_score_rank": round(safe_float(safe_dict(best.get("scores")).get("rank", 0.0)), 6),
        "best_score_policy": round(safe_float(safe_dict(best.get("scores")).get("policy", 0.0)), 6),
        "base_score_rank": round(safe_float(safe_dict(baseline.get("scores")).get("rank", 0.0)), 6),
        "base_score_policy": round(safe_float(safe_dict(baseline.get("scores")).get("policy", 0.0)), 6),
        "chosen_score_rank": round(safe_float(decision.get("chosen_score_rank", 0.0)), 6),
        "chosen_score_policy": round(safe_float(decision.get("chosen_score_policy", 0.0)), 6),
    }


def build_structured_decision(decision_row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "decision_type": decision_row["decision_type"],
        "decision_id": decision_row["decision_id"],
        "delta_vs_base": decision_row["delta_vs_base"],
        "tau_improve": decision_row["tau_improve"],
        "base_candidate_id": decision_row["base_candidate_id"],
        "winner_pre_gate_candidate_id": decision_row["winner_pre_gate_candidate_id"],
        "winner_post_gate_candidate_id": decision_row["winner_post_gate_candidate_id"],
        "base_bbox": decision_row["base_bbox"],
        "winner_pre_gate_bbox": decision_row["winner_pre_gate_bbox"],
        "winner_post_gate_bbox": decision_row["winner_post_gate_bbox"],
        "chosen_score_rank": decision_row["chosen_score_rank"],
        "chosen_score_policy": decision_row["chosen_score_policy"],
    }


def build_teacher_meta(routing: Dict[str, Any], annotated_candidates: Sequence[Dict[str, Any]], ar_res: Dict[str, Any]) -> Dict[str, Any]:
    best = safe_dict(ar_res.get("best_candidate"))
    consensus = maybe_float(raw_checklist_entry(best, "teacher_consensus").get("value"))
    best_softmax = 0.0
    if annotated_candidates:
        best_softmax = max(safe_float(safe_dict(candidate.get("score_targets")).get("softmax_local", 0.0)) for candidate in annotated_candidates)
    if consensus is None:
        consensus = best_softmax
    route_conf = safe_float(routing.get("route_conf", routing.get("subject_mode_conf", 0.0)), 0.0)
    reliability = 0.5 * clamp(consensus or 0.0, 0.0, 1.0) + 0.5 * clamp(route_conf, 0.0, 1.0)
    return {
        "teacher_confidence": round(clamp(consensus or 0.0, 0.0, 1.0), 6),
        "target_reliability": round(clamp(reliability, 0.0, 1.0), 6),
        "route_conf": round(clamp(route_conf, 0.0, 1.0), 6),
        "candidate_count": len(annotated_candidates),
    }


def build_matching_target(candidate_record: Dict[str, Any], target_index: int) -> Dict[str, Any]:
    derived_targets = derive_checklist_targets(
        candidate=source_candidate_from_record(candidate_record),
        labels=safe_dict(candidate_record.get("checklist_labels")),
        scores=safe_dict(candidate_record.get("checklist_scores")),
        applicable_mask=safe_dict(candidate_record.get("applicable_mask")),
        observed_mask=safe_dict(candidate_record.get("observed_mask")),
    )
    return {
        "target_id": f"pos_{target_index:03d}",
        "candidate_id": candidate_record["candidate_id"],
        "bbox_norm_xyxy": candidate_record["bbox_norm_xyxy"],
        "bbox_cxcywh": candidate_record["bbox_cxcywh"],
        "score_targets": candidate_record["score_targets"],
        "macro_targets": candidate_record["macro_targets"],
        "checklist_labels": candidate_record["checklist_labels"],
        "checklist_scores": candidate_record["checklist_scores"],
        "applicable_mask": candidate_record["applicable_mask"],
        "observed_mask": candidate_record["observed_mask"],
        "derived_checklist_targets": derived_targets,
        "why_tags": candidate_record["why_tags"],
    }


def source_candidate_from_record(candidate_record: Dict[str, Any]) -> Dict[str, Any]:
    raw = copy.deepcopy(candidate_record.get("_source_candidate"))
    return raw if isinstance(raw, dict) else {}


def build_candidate_pool_entry(candidate_record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_id": candidate_record["candidate_id"],
        "bbox_norm_xyxy": candidate_record["bbox_norm_xyxy"],
        "bbox_cxcywh": candidate_record["bbox_cxcywh"],
        "source": candidate_record["source"],
        "label_type": candidate_record["label_type"],
        "score_rank_pct": candidate_record["score_targets"]["rank_pct"],
        "score_prob": candidate_record["score_targets"]["score_prob"],
        "is_positive_candidate": candidate_record["is_positive_candidate"],
        "is_hard_negative": candidate_record["is_hard_negative"],
        "is_unsafe_negative": candidate_record["is_unsafe_negative"],
        "why_tags": candidate_record["why_tags"],
        "reject_tags": candidate_record["reject_tags"],
    }


def candidate_training_view(candidate: Dict[str, Any], routing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    candidate_copy = copy.deepcopy(candidate)
    canonical = build_candidate_canonical_record(candidate_copy, routing or {})
    return {
        "candidate_id": canonical["candidate_id"],
        "bbox": canonical["bbox_norm_xyxy"],
        "bbox_norm_xyxy": canonical["bbox_norm_xyxy"],
        "bbox_cxcywh": canonical["bbox_cxcywh"],
        "source": canonical["source"],
        "area_ratio": canonical["area_ratio"],
        "score_raw_final": canonical["score_targets"]["score_raw_rank"],
        "score_raw_rank": canonical["score_targets"]["score_raw_rank"],
        "score_raw_policy": canonical["score_targets"]["score_raw_policy"],
        "score_raw_final_legacy": round(safe_float(safe_dict(candidate.get("scores")).get("final_legacy", 0.0)), 6),
        "score_prob": canonical["score_targets"]["score_prob"],
        "score_rank_pct": canonical["score_targets"]["rank_pct"],
        "score_z_local": canonical["score_targets"]["z_local"],
        "score_softmax_local": canonical["score_targets"]["softmax_local"],
        "pseudo_mos_1to5": canonical["score_targets"]["pseudo_mos_1to5"],
        "label_type": canonical["label_type"],
        "is_positive_candidate": canonical["is_positive_candidate"],
        "is_soft_positive": canonical["is_soft_positive"],
        "is_hard_negative": canonical["is_hard_negative"],
        "is_unsafe_negative": canonical["is_unsafe_negative"],
        "macro_scores": canonical["macro_targets"],
        "checklist_labels": canonical["checklist_labels"],
        "checklist_scores": canonical["checklist_scores"],
        "applicable_mask": canonical["applicable_mask"],
        "observed_mask": canonical["observed_mask"],
        "why_tags": canonical["why_tags"],
        "reject_tags": canonical["reject_tags"],
    }


def build_pairwise_records(
    image_id: str,
    target_ar: str,
    ar_res: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    pair_margin_min: float,
    max_hard_pairs: int,
    max_near_pairs: int,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    decision = safe_dict(ar_res.get("decision"))
    mode = str(safe_dict(ar_res.get("routing")).get("subject_mode", ""))
    positives = [c for c in candidates if bool(c.get("is_positive_candidate", False))]
    hard_negatives = [c for c in candidates if str(c.get("label_type", "")) in {"hard_negative", "unsafe_negative"}]
    near_negatives = [c for c in candidates if str(c.get("label_type", "")) == "near_negative"]
    best = next((c for c in candidates if bool(c.get("is_best_candidate", False))), None)
    baseline = next((c for c in candidates if bool(c.get("is_baseline_candidate", False))), None)
    out: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str]] = set()

    def add_pair(a: Dict[str, Any], b: Dict[str, Any], pair_type: str) -> None:
        key = (str(a.get("candidate_id", "")), str(b.get("candidate_id", "")), pair_type)
        if not key[0] or not key[1] or key in seen:
            return
        score_a = safe_float(safe_dict(a.get("scores")).get("rank", safe_dict(a.get("scores")).get("final", 0.0)))
        score_b = safe_float(safe_dict(b.get("scores")).get("rank", safe_dict(b.get("scores")).get("final", 0.0)))
        margin = score_a - score_b
        if margin < float(pair_margin_min):
            return
        out.append(
            {
                "image_id": image_id,
                "target_ar": target_ar,
                "mode": mode,
                "decision_type": str(decision.get("decision_type", "")),
                "bbox_a": normalize_bbox_xyxy(a.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0])),
                "bbox_b": normalize_bbox_xyxy(b.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0])),
                "candidate_id_a": str(a.get("candidate_id", "")),
                "candidate_id_b": str(b.get("candidate_id", "")),
                "label": 1,
                "score_margin": round(margin, 6),
                "pair_type": pair_type,
                "score_rank_pct_a": round(safe_float(a.get("score_rank_pct", 0.0)), 6),
                "score_rank_pct_b": round(safe_float(b.get("score_rank_pct", 0.0)), 6),
                "label_type_a": str(a.get("label_type", "")),
                "label_type_b": str(b.get("label_type", "")),
                "source_a": str(a.get("source", "")),
                "source_b": str(b.get("source", "")),
            }
        )
        seen.add(key)

    top1 = positives[0] if positives else None
    if top1 is not None:
        for cand in hard_negatives[: max(0, int(max_hard_pairs))]:
            add_pair(top1, cand, "top1_vs_hard_negative")
        for cand in near_negatives[: max(0, int(max_near_pairs))]:
            add_pair(top1, cand, "top1_vs_near_negative")
    for cand in positives[1:3]:
        for neg in hard_negatives[:2]:
            add_pair(cand, neg, "topk_vs_hard_negative")
    if best is not None and baseline is not None and str(best.get("candidate_id", "")) != str(baseline.get("candidate_id", "")):
        add_pair(best, baseline, "baseline_vs_best")
    return out


def build_listwise_record(
    image_id: str,
    target_ar: str,
    ar_res: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    top_pos: int,
    neg: int,
    softmax_tau: float,
) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    routing = safe_dict(ar_res.get("routing"))
    mode = str(routing.get("subject_mode", ""))
    decision = safe_dict(ar_res.get("decision"))
    positives = [c for c in candidates if bool(c.get("is_positive_candidate", False))]
    hard_negs = [c for c in candidates if str(c.get("label_type", "")) in {"hard_negative", "unsafe_negative"}]
    other_negs = [c for c in candidates if str(c.get("label_type", "")) in {"near_negative", "negative"}]
    chosen: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def maybe_add(cand: Dict[str, Any]) -> None:
        cid = str(cand.get("candidate_id", ""))
        if not cid or cid in seen:
            return
        chosen.append(cand)
        seen.add(cid)

    for cand in positives[: max(1, int(top_pos))]:
        maybe_add(cand)
    baseline = next((c for c in candidates if bool(c.get("is_baseline_candidate", False))), None)
    best = next((c for c in candidates if bool(c.get("is_best_candidate", False))), None)
    if baseline is not None:
        maybe_add(baseline)
    if best is not None:
        maybe_add(best)
    for cand in hard_negs[: max(1, int(neg // 2))]:
        maybe_add(cand)
    for cand in other_negs[: max(0, int(neg))]:
        if len(chosen) >= int(top_pos) + int(neg):
            break
        maybe_add(cand)

    if not chosen:
        return None
    raw_scores = [safe_float(safe_dict(c.get("scores")).get("rank", safe_dict(c.get("scores")).get("final", 0.0))) for c in chosen]
    probs = softmax_local(raw_scores, tau=softmax_tau)
    candidate_rows = []
    for cand, prob in zip(chosen, probs):
        row = candidate_training_view(cand, routing=routing)
        row["score_softmax_local"] = round(float(prob), 6)
        candidate_rows.append(row)
    return {
        "image_id": image_id,
        "target_ar": target_ar,
        "decision_type": str(decision.get("decision_type", "")),
        "mode": mode,
        "policy_id": str(routing.get("policy_id", "")),
        "candidates": candidate_rows,
    }


def build_checklist_records(
    image_id: str,
    target_ar: str,
    ar_res: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    routing = safe_dict(ar_res.get("routing"))
    mode = str(routing.get("subject_mode", ""))
    decision_type = str(safe_dict(ar_res.get("decision")).get("decision_type", ""))
    out = []
    for cand in candidates:
        row = candidate_training_view(cand, routing=routing)
        row.update(
            {
                "image_id": image_id,
                "target_ar": target_ar,
                "mode": mode,
                "decision_type": decision_type,
                "policy_id": str(routing.get("policy_id", "")),
            }
        )
        out.append(row)
    return out


def build_regression_records(
    image_id: str,
    target_ar: str,
    ar_res: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    routing = safe_dict(ar_res.get("routing"))
    mode = str(routing.get("subject_mode", ""))
    decision_type = str(safe_dict(ar_res.get("decision")).get("decision_type", ""))
    rows = []
    for cand in candidates:
        row = candidate_training_view(cand, routing=routing)
        row.update(
            {
                "image_id": image_id,
                "target_ar": target_ar,
                "mode": mode,
                "decision_type": decision_type,
            }
        )
        rows.append(row)
    return rows


def build_conditional_detr_records(
    teacher_record: Dict[str, Any],
    image_id: str,
    target_ar: str,
    ar_res: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    subject_mode_to_id: Dict[str, int],
    image_root: Optional[Path],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    routing = build_routing_record(teacher_record, ar_res, subject_mode_to_id)
    baseline = build_baseline_record(ar_res)
    decision_row = build_decision_record(image_id, target_ar, ar_res, candidates)
    decision = build_structured_decision(decision_row)
    image_path = resolve_image_path(image_root, image_id)

    canonical_candidates: List[Dict[str, Any]] = []
    matching_targets: List[Dict[str, Any]] = []
    candidate_pool: List[Dict[str, Any]] = []

    for candidate in candidates:
        candidate_record = build_candidate_canonical_record(candidate, routing)
        candidate_record["_source_candidate"] = copy.deepcopy(candidate)
        canonical_candidates.append(candidate_record)
        if candidate_record["is_positive_candidate"] and not candidate_record["is_unsafe_negative"]:
            matching_targets.append(build_matching_target(candidate_record, len(matching_targets)))
        else:
            candidate_pool.append(build_candidate_pool_entry(candidate_record))

    if not matching_targets:
        fallback_id = str(first_nonempty(decision.get("winner_post_gate_candidate_id"), decision.get("base_candidate_id"), ""))
        fallback_candidate = next(
            (
                candidate_record
                for candidate_record in canonical_candidates
                if candidate_record["candidate_id"] == fallback_id and not candidate_record["is_unsafe_negative"]
            ),
            None,
        )
        if fallback_candidate is not None:
            matching_targets.append(build_matching_target(fallback_candidate, len(matching_targets)))
            candidate_pool = [row for row in candidate_pool if row["candidate_id"] != fallback_candidate["candidate_id"]]

    teacher_meta = build_teacher_meta(routing, canonical_candidates, ar_res)
    for candidate_record in canonical_candidates:
        candidate_record.pop("_source_candidate", None)

    canonical_record = {
        "schema_version": "sstk_detr_train_v2",
        "image_id": image_id,
        "image_path": str(image_path) if image_path is not None else None,
        "target_ar": target_ar,
        "routing": routing,
        "baseline": baseline,
        "decision": decision,
        "candidates": canonical_candidates,
        "teacher_meta": teacher_meta,
        "checklist_schema_version": "sstk_check_v2",
    }
    batch_record = {
        "schema_version": "sstk_conditional_detr_batch_v2",
        "image_id": image_id,
        "image_path": str(image_path) if image_path is not None else None,
        "target_ar": target_ar,
        "routing": routing,
        "baseline": baseline,
        "decision_target": decision,
        "matching_targets": matching_targets,
        "candidate_pool": candidate_pool,
        "teacher_meta": teacher_meta,
        "checklist_schema_version": "sstk_check_v2",
    }
    return canonical_record, batch_record


def build_training_datasets(
    teacher_records: Sequence[Dict[str, Any]],
    softmax_tau: float,
    listwise_top_pos: int,
    listwise_neg: int,
    pair_margin_min: float,
    hard_negative_rank_pct_max: float,
    near_margin_max: float,
    max_hard_pairs: int,
    max_near_pairs: int,
    subject_mode_to_id: Optional[Dict[str, int]] = None,
    image_root: Optional[Path] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    subject_mode_to_id = subject_mode_to_id or build_subject_mode_vocab(teacher_records)
    pairwise_rows: List[Dict[str, Any]] = []
    listwise_rows: List[Dict[str, Any]] = []
    decision_rows: List[Dict[str, Any]] = []
    checklist_rows: List[Dict[str, Any]] = []
    regression_rows: List[Dict[str, Any]] = []
    canonical_rows: List[Dict[str, Any]] = []
    batch_rows: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []

    for rec in teacher_records:
        image_id = str(rec.get("image_id", ""))
        results_by_ar = safe_dict(safe_dict(rec.get("teacher_scorer")).get("results_by_ar"))
        for target_ar, ar_res in results_by_ar.items():
            candidates = annotate_group_candidates(
                image_id=image_id,
                target_ar=str(target_ar),
                ar_res=safe_dict(ar_res),
                softmax_tau=softmax_tau,
                hard_negative_rank_pct_max=hard_negative_rank_pct_max,
                near_margin_max=near_margin_max,
            )
            if not candidates:
                continue
            ar_res_dict = safe_dict(ar_res)
            pairwise_rows.extend(
                build_pairwise_records(
                    image_id=image_id,
                    target_ar=str(target_ar),
                    ar_res=ar_res_dict,
                    candidates=candidates,
                    pair_margin_min=pair_margin_min,
                    max_hard_pairs=max_hard_pairs,
                    max_near_pairs=max_near_pairs,
                )
            )
            listwise = build_listwise_record(
                image_id=image_id,
                target_ar=str(target_ar),
                ar_res=ar_res_dict,
                candidates=candidates,
                top_pos=listwise_top_pos,
                neg=listwise_neg,
                softmax_tau=softmax_tau,
            )
            if listwise is not None:
                listwise_rows.append(listwise)
            decision_rows.append(
                build_decision_record(
                    image_id=image_id,
                    target_ar=str(target_ar),
                    ar_res=ar_res_dict,
                    candidates=candidates,
                )
            )
            checklist_rows.extend(build_checklist_records(image_id, str(target_ar), ar_res_dict, candidates))
            regression_rows.extend(build_regression_records(image_id, str(target_ar), ar_res_dict, candidates))
            canonical_record, batch_record = build_conditional_detr_records(
                teacher_record=rec,
                image_id=image_id,
                target_ar=str(target_ar),
                ar_res=ar_res_dict,
                candidates=candidates,
                subject_mode_to_id=subject_mode_to_id,
                image_root=image_root,
            )
            if batch_record["matching_targets"]:
                canonical_rows.append(canonical_record)
                batch_rows.append(batch_record)
            else:
                skipped_rows.append(
                    {
                        "image_id": image_id,
                        "target_ar": str(target_ar),
                        "skip_reason": "no_safe_positive_target",
                        "routing": canonical_record["routing"],
                        "decision": canonical_record["decision"],
                    }
                )

    return {
        "pairwise": pairwise_rows,
        "listwise": listwise_rows,
        "decision": decision_rows,
        "checklist": checklist_rows,
        "regression": regression_rows,
        "conditional_detr_canonical": canonical_rows,
        "conditional_detr_batch": batch_rows,
        "conditional_detr_skipped": skipped_rows,
    }


def counter_ratios(num: Counter, den: Counter) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key in sorted(den):
        denom = den[key]
        out[key] = round((float(num[key]) / float(denom)) if denom else 0.0, 6)
    return out


def build_qa_summary(datasets: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    pairwise_rows = datasets["pairwise"]
    listwise_rows = datasets["listwise"]
    decision_rows = datasets["decision"]
    checklist_rows = datasets["checklist"]
    regression_rows = datasets["regression"]
    canonical_rows = datasets["conditional_detr_canonical"]
    batch_rows = datasets["conditional_detr_batch"]
    skipped_rows = datasets.get("conditional_detr_skipped", [])

    pair_margins = [safe_float(row.get("score_margin", 0.0)) for row in pairwise_rows]
    pair_type_counts = Counter(str(row.get("pair_type", "")) for row in pairwise_rows)
    pair_mode_counts = Counter(str(row.get("mode", "")) for row in pairwise_rows)
    pair_ar_counts = Counter(str(row.get("target_ar", "")) for row in pairwise_rows)
    unsafe_ratio = 0.0
    if pairwise_rows:
        unsafe_ratio = sum(1 for row in pairwise_rows if "unsafe" in str(row.get("label_type_b", ""))) / float(len(pairwise_rows))

    list_counts = [len(row.get("candidates", [])) for row in listwise_rows]
    list_entropies = [entropy([safe_float(c.get("score_softmax_local", 0.0)) for c in row.get("candidates", [])]) for row in listwise_rows]
    list_top1_probs = []
    list_label_types = Counter()
    for row in listwise_rows:
        cand_probs = [safe_float(c.get("score_softmax_local", 0.0)) for c in row.get("candidates", [])]
        if cand_probs:
            list_top1_probs.append(max(cand_probs))
        for cand in row.get("candidates", []):
            list_label_types[str(cand.get("label_type", ""))] += 1

    decision_counts = Counter(str(row.get("decision_type", "")) for row in decision_rows)
    baseline_dominance_rate = 0.0
    if decision_rows:
        baseline_dominance_rate = sum(
            1 for row in decision_rows if str(row.get("winner_post_gate_candidate_id", "")) == str(row.get("base_candidate_id", ""))
        ) / float(len(decision_rows))

    regression_rank_pct = [safe_float(row.get("score_rank_pct", 0.0)) for row in regression_rows]
    regression_z = [safe_float(row.get("score_z_local", 0.0)) for row in regression_rows]
    regression_softmax = [safe_float(row.get("score_softmax_local", 0.0)) for row in regression_rows]
    regression_pmos = [safe_float(row.get("pseudo_mos_1to5", 1.0)) for row in regression_rows]

    checklist_label_types = Counter(str(row.get("label_type", "")) for row in checklist_rows)
    checklist_modes = Counter(str(row.get("mode", "")) for row in checklist_rows)

    canonical_mode_counts = Counter(str(safe_dict(row.get("routing")).get("subject_mode", "")) for row in canonical_rows)
    canonical_decision_counts = Counter(str(safe_dict(row.get("decision")).get("decision_type", "")) for row in canonical_rows)
    canonical_candidate_counts = [len(row.get("candidates", [])) for row in canonical_rows]
    matching_target_counts = [len(row.get("matching_targets", [])) for row in batch_rows]
    candidate_pool_counts = [len(row.get("candidate_pool", [])) for row in batch_rows]
    route_conf_values = [safe_float(safe_dict(row.get("routing")).get("route_conf", 0.0)) for row in canonical_rows]
    reliability_values = [safe_float(safe_dict(row.get("teacher_meta")).get("target_reliability", 0.0)) for row in canonical_rows]
    applicable_num = Counter()
    applicable_den = Counter()
    observed_num = Counter()
    observed_den = Counter()
    valid_num = Counter()
    valid_den = Counter()
    for row in batch_rows:
        for target in row.get("matching_targets", []):
            applicable = safe_dict(target.get("applicable_mask"))
            observed = safe_dict(target.get("observed_mask"))
            valid = safe_dict(safe_dict(target.get("derived_checklist_targets")).get("valid_mask"))
            for key, value in applicable.items():
                applicable_den[key] += 1
                applicable_num[key] += int(bool(value))
            for key, value in observed.items():
                observed_den[key] += 1
                observed_num[key] += int(bool(value))
            for key, value in valid.items():
                valid_den[key] += 1
                valid_num[key] += int(bool(value))

    return {
        "counts": {
            "pairwise": len(pairwise_rows),
            "listwise": len(listwise_rows),
            "decision": len(decision_rows),
            "checklist": len(checklist_rows),
            "regression": len(regression_rows),
            "conditional_detr_canonical": len(canonical_rows),
            "conditional_detr_batch": len(batch_rows),
            "conditional_detr_skipped": len(skipped_rows),
        },
        "pairwise": {
            "pair_count_by_mode": dict(pair_mode_counts),
            "pair_count_by_ar": dict(pair_ar_counts),
            "pair_type_counts": dict(pair_type_counts),
            "margin_distribution": summarize_numeric(pair_margins),
            "unsafe_negative_ratio": round(float(unsafe_ratio), 6),
        },
        "listwise": {
            "candidate_count_distribution": summarize_numeric(list_counts),
            "softmax_entropy_distribution": summarize_numeric(list_entropies),
            "top1_prob_distribution": summarize_numeric(list_top1_probs),
            "label_type_counts": dict(list_label_types),
        },
        "decision": {
            "decision_counts": dict(decision_counts),
            "baseline_dominance_rate": round(float(baseline_dominance_rate), 6),
        },
        "regression": {
            "rank_pct_distribution": summarize_numeric(regression_rank_pct),
            "z_local_distribution": summarize_numeric(regression_z),
            "softmax_local_distribution": summarize_numeric(regression_softmax),
            "pseudo_mos_1to5_distribution": summarize_numeric(regression_pmos),
        },
        "checklist": {
            "label_type_counts": dict(checklist_label_types),
            "mode_counts": dict(checklist_modes),
        },
        "conditional_detr": {
            "mode_counts": dict(canonical_mode_counts),
            "decision_counts": dict(canonical_decision_counts),
            "candidate_count_distribution": summarize_numeric(canonical_candidate_counts),
            "matching_target_count_distribution": summarize_numeric(matching_target_counts),
            "candidate_pool_count_distribution": summarize_numeric(candidate_pool_counts),
            "route_conf_distribution": summarize_numeric(route_conf_values),
            "target_reliability_distribution": summarize_numeric(reliability_values),
            "skipped_count": len(skipped_rows),
            "applicable_rate_by_key": counter_ratios(applicable_num, applicable_den),
            "observed_rate_by_key": counter_ratios(observed_num, observed_den),
            "valid_rate_by_key": counter_ratios(valid_num, valid_den),
        },
    }


def validate_datasets(datasets: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    canonical_rows = datasets["conditional_detr_canonical"]
    batch_rows = datasets["conditional_detr_batch"]
    skipped_rows = datasets.get("conditional_detr_skipped", [])
    errors: List[str] = []
    warnings: List[str] = []

    if len(canonical_rows) != len(batch_rows):
        errors.append(
            f"conditional_detr_canonical/batch row count mismatch: {len(canonical_rows)} vs {len(batch_rows)}"
        )

    for row in canonical_rows:
        image_id = str(row.get("image_id", ""))
        target_ar = str(row.get("target_ar", ""))
        routing = safe_dict(row.get("routing"))
        decision = safe_dict(row.get("decision"))
        baseline = safe_dict(row.get("baseline"))
        candidates = row.get("candidates", [])
        prefix = f"{image_id}::{target_ar}"
        if not str(routing.get("subject_mode", "")):
            errors.append(f"{prefix} missing routing.subject_mode")
        if decision_id(decision.get("decision_type", "")) < 0:
            errors.append(f"{prefix} invalid decision_type={decision.get('decision_type')}")
        if not bbox_is_valid(baseline.get("bbox_norm_xyxy")):
            errors.append(f"{prefix} invalid baseline bbox")
        if not candidates:
            errors.append(f"{prefix} missing canonical candidates")
        seen_candidate_ids: set[str] = set()
        for candidate in candidates:
            candidate_id = str(candidate.get("candidate_id", ""))
            if not candidate_id:
                errors.append(f"{prefix} candidate without id")
                continue
            if candidate_id in seen_candidate_ids:
                errors.append(f"{prefix} duplicate candidate_id={candidate_id}")
            seen_candidate_ids.add(candidate_id)
            if not bbox_is_valid(candidate.get("bbox_norm_xyxy")):
                errors.append(f"{prefix} invalid candidate bbox for {candidate_id}")

    for row in batch_rows:
        image_id = str(row.get("image_id", ""))
        target_ar = str(row.get("target_ar", ""))
        prefix = f"{image_id}::{target_ar}"
        targets = row.get("matching_targets", [])
        pool = row.get("candidate_pool", [])
        if not targets:
            errors.append(f"{prefix} has no matching_targets")
        seen_target_ids: set[str] = set()
        matched_candidate_ids: set[str] = set()
        for target in targets:
            target_id = str(target.get("target_id", ""))
            candidate_id = str(target.get("candidate_id", ""))
            if target_id in seen_target_ids:
                errors.append(f"{prefix} duplicate target_id={target_id}")
            seen_target_ids.add(target_id)
            matched_candidate_ids.add(candidate_id)
            if not bbox_is_valid(target.get("bbox_norm_xyxy")):
                errors.append(f"{prefix} invalid matching target bbox for {candidate_id}")
            derived = safe_dict(target.get("derived_checklist_targets"))
            valid_mask = safe_dict(derived.get("valid_mask"))
            for key, value in valid_mask.items():
                if value not in {0, 1}:
                    errors.append(f"{prefix} invalid valid_mask[{key}]={value}")
        for pool_candidate in pool:
            candidate_id = str(pool_candidate.get("candidate_id", ""))
            if candidate_id in matched_candidate_ids:
                errors.append(f"{prefix} candidate_id={candidate_id} appears in both matching_targets and candidate_pool")
            if not bbox_is_valid(pool_candidate.get("bbox_norm_xyxy")):
                errors.append(f"{prefix} invalid candidate_pool bbox for {candidate_id}")
        if row.get("image_path") is None:
            warnings.append(f"{prefix} image_path unresolved")

    summary = {
        "status": "ok" if not errors else "failed",
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors[:100],
        "warnings": warnings[:100],
        "counts": {
            "conditional_detr_canonical": len(canonical_rows),
            "conditional_detr_batch": len(batch_rows),
            "conditional_detr_skipped": len(skipped_rows),
        },
    }
    return summary


def compact_validation_summary(validation_summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": validation_summary.get("status"),
        "error_count": validation_summary.get("error_count"),
        "warning_count": validation_summary.get("warning_count"),
        "counts": validation_summary.get("counts"),
        "errors": validation_summary.get("errors", [])[:10],
        "warnings": validation_summary.get("warnings", [])[:10],
    }


def choose_report_examples(
    canonical_rows: Sequence[Dict[str, Any]],
    max_examples: int,
) -> List[Dict[str, Any]]:
    available = []
    for row in canonical_rows:
        image_path = row.get("image_path")
        if not image_path:
            continue
        if Path(str(image_path)).exists():
            available.append(row)
    selected: List[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    prioritized_contextual = sorted(
        (
            row
            for row in available
            if bool(safe_dict(safe_dict(row.get("routing")).get("flags")).get("contextual_tiny_human", False))
        ),
        key=lambda row: (
            0 if str(safe_dict(row.get("decision")).get("decision_type", "")) == "keep_full" else 1,
            str(safe_dict(row.get("routing")).get("subject_mode", "")),
            str(row.get("image_id", "")),
            str(row.get("target_ar", "")),
        ),
    )
    for row in prioritized_contextual[: min(max_examples, 2)]:
        group_key = f"{row.get('image_id')}::{row.get('target_ar')}"
        if group_key in seen_keys:
            continue
        selected.append(row)
        seen_keys.add(group_key)
    by_mode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in available:
        mode = str(safe_dict(row.get("routing")).get("subject_mode", "unknown"))
        by_mode[mode].append(row)
    mode_order = sorted(by_mode.keys(), key=lambda key: (-len(by_mode[key]), key))
    while len(selected) < max_examples:
        added = False
        for mode in mode_order:
            if not by_mode[mode]:
                continue
            row = by_mode[mode].pop(0)
            group_key = f"{row.get('image_id')}::{row.get('target_ar')}"
            if group_key in seen_keys:
                continue
            selected.append(row)
            seen_keys.add(group_key)
            added = True
            if len(selected) >= max_examples:
                break
        if not added:
            break
    return selected


def draw_norm_box(draw: Any, bbox: Sequence[float], width: int, height: int, color: str, label: str, line_width: int, font: Any) -> None:
    x1 = int(round(clamp(safe_float(bbox[0]), 0.0, 1.0) * width))
    y1 = int(round(clamp(safe_float(bbox[1]), 0.0, 1.0) * height))
    x2 = int(round(clamp(safe_float(bbox[2]), 0.0, 1.0) * width))
    y2 = int(round(clamp(safe_float(bbox[3]), 0.0, 1.0) * height))
    draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
    if label:
        text_bbox = draw.textbbox((0, 0), label, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        label_w = tw + 6
        label_h = th + 4
        box_w = max(0, x2 - x1)
        box_h = max(0, y2 - y1)

        # Default: place bbox label inside the box near bottom-left to avoid top-left metadata occlusion.
        tx = min(max(0, x1 + 1), max(0, width - label_w))
        ty = y2 - label_h - 1
        fits_inside_bottom = box_w >= label_w + 2 and box_h >= label_h + 2 and ty >= y1 + 1

        if not fits_inside_bottom:
            inside_top_ty = y1 + 1
            fits_inside_top = box_w >= label_w + 2 and box_h >= label_h + 2 and inside_top_ty + label_h <= y2 - 1
            if fits_inside_top:
                ty = inside_top_ty
            else:
                below_ty = y2 + 2
                above_ty = y1 - label_h - 2
                tx = min(max(0, x1), max(0, width - label_w))
                if below_ty + label_h <= height:
                    ty = below_ty
                elif above_ty >= 0:
                    ty = above_ty
                else:
                    ty = min(max(0, y1), max(0, height - label_h))

        draw.rectangle([tx, ty, tx + label_w, ty + label_h], fill=color)
        draw.text((tx + 3, ty + 2), label, fill="white", font=font)


def draw_text_block(draw: Any, x: int, y: int, lines: Sequence[str], font: Any, bg_color: str = "#111111", text_color: str = "white") -> None:
    if not lines:
        return
    widths: List[int] = []
    heights: List[int] = []
    for line in lines:
        text_bbox = draw.textbbox((0, 0), line, font=font)
        widths.append(text_bbox[2] - text_bbox[0])
        heights.append(text_bbox[3] - text_bbox[1])
    block_w = max(widths) + 10
    block_h = sum(heights) + max(0, len(lines) - 1) * 2 + 8
    draw.rectangle([x, y, x + block_w, y + block_h], fill=bg_color)
    cursor_y = y + 4
    for line, line_h in zip(lines, heights):
        draw.text((x + 5, cursor_y), line, fill=text_color, font=font)
        cursor_y += line_h + 2


def render_report_example(example_record: Dict[str, Any], example_path: Path) -> bool:
    if Image is None or ImageDraw is None or ImageFont is None:
        return False
    image_path = Path(str(example_record.get("image_path")))
    if not image_path.exists():
        return False
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    if max(width, height) > 1400:
        scale = 1400.0 / float(max(width, height))
        image = image.resize((int(round(width * scale)), int(round(height * scale))))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size
    line_width = max(2, int(round(max(width, height) / 240.0)))
    routing = safe_dict(example_record.get("routing"))
    decision = safe_dict(example_record.get("decision"))
    baseline = safe_dict(example_record.get("baseline"))
    candidates = safe_list(example_record.get("candidates"))
    candidate_by_id = {str(candidate.get("candidate_id", "")): candidate for candidate in candidates}
    subject_prior_bbox = routing.get("subject_prior_bbox_norm_xyxy")
    if bbox_is_valid(subject_prior_bbox):
        draw_norm_box(draw, subject_prior_bbox, width, height, "#f39c12", "subject prior", line_width, font)
    if bbox_is_valid(baseline.get("bbox_norm_xyxy")):
        draw_norm_box(draw, baseline["bbox_norm_xyxy"], width, height, "#1f77b4", "baseline", line_width, font)
    pre_gate_bbox = decision.get("winner_pre_gate_bbox")
    post_gate_bbox = decision.get("winner_post_gate_bbox")
    if bbox_is_valid(pre_gate_bbox) and pre_gate_bbox != post_gate_bbox:
        draw_norm_box(draw, pre_gate_bbox, width, height, "#f1c40f", "pre-gate best", line_width, font)
    if bbox_is_valid(post_gate_bbox):
        draw_norm_box(draw, post_gate_bbox, width, height, "#2ecc71", "chosen", line_width, font)
    hard_negative = next(
        (
            candidate
            for candidate in candidates
            if bool(candidate.get("is_hard_negative", False)) or bool(candidate.get("is_unsafe_negative", False))
        ),
        None,
    )
    if hard_negative is not None and bbox_is_valid(hard_negative.get("bbox_norm_xyxy")):
        draw_norm_box(draw, hard_negative["bbox_norm_xyxy"], width, height, "#e74c3c", "hard/unsafe neg", line_width, font)
    chosen_candidate = candidate_by_id.get(str(decision.get("winner_post_gate_candidate_id", "")))
    caption = " | ".join(
        [
            str(example_record.get("image_id", "")),
            str(example_record.get("target_ar", "")),
            str(routing.get("subject_mode", "")),
            str(decision.get("decision_type", "")),
        ]
    )
    text_bbox = draw.textbbox((0, 0), caption, font=font)
    tw = text_bbox[2] - text_bbox[0]
    th = text_bbox[3] - text_bbox[1]
    draw.rectangle([0, 0, min(width, tw + 10), th + 8], fill="#111111")
    draw.text((5, 4), caption, fill="white", font=font)
    if chosen_candidate is not None:
        why_tags = ", ".join(str(tag) for tag in chosen_candidate.get("why_tags", [])[:3])
        if why_tags:
            meta_bbox = draw.textbbox((0, 0), why_tags, font=font)
            mw = meta_bbox[2] - meta_bbox[0]
            mh = meta_bbox[3] - meta_bbox[1]
            draw.rectangle([0, th + 10, min(width, mw + 10), th + mh + 18], fill="#222222")
            draw.text((5, th + 14), why_tags, fill="white", font=font)
    example_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(example_path)
    return True


def build_gaic_like_annotation_preview(
    *,
    image_entry_id: int,
    image_path: str,
    routing: Dict[str, Any],
    decision_target: Dict[str, Any],
    target_ar: str,
    candidate: Dict[str, Any],
    gt_flag: int,
) -> Dict[str, Any]:
    image_path_obj = Path(image_path)
    with Image.open(image_path_obj) as image:
        width, height = int(image.width), int(image.height)
    bbox_norm_xyxy = candidate.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0])
    bbox_xywh = norm_xyxy_to_xywh(bbox_norm_xyxy)
    bbox_px = [
        round(safe_float(bbox_xywh[0]) * width, 3),
        round(safe_float(bbox_xywh[1]) * height, 3),
        round(safe_float(bbox_xywh[2]) * width, 3),
        round(safe_float(bbox_xywh[3]) * height, 3),
    ]
    annotation = {
        "area": round(max(0.0, bbox_px[2]) * max(0.0, bbox_px[3]), 3),
        "image_id": image_entry_id,
        "bbox": bbox_px,
        "category_id": 0,
        "score": safe_float(
            safe_dict(candidate.get("score_targets")).get("score_prob")
            if gt_flag == 1
            else candidate.get("score_prob", candidate.get("score_rank_pct", 0.0))
        ),
        "gt_flag": gt_flag,
        "iscrowd": 0,
        "subject_mode_id": int(routing.get("subject_mode_id", -1)) if routing.get("subject_mode_id") is not None else -1,
        "decision_id": int(decision_target.get("decision_id", -1)) if decision_target.get("decision_id") is not None else -1,
        "candidate_id": candidate.get("candidate_id"),
        "target_ar": target_ar,
        "label_type": "matching_target" if gt_flag == 1 else candidate.get("label_type"),
        "is_hard_negative": False if gt_flag == 1 else bool(candidate.get("is_hard_negative", False)),
        "is_unsafe_negative": False if gt_flag == 1 else bool(candidate.get("is_unsafe_negative", False)),
    }
    macro_targets = safe_dict(candidate.get("macro_targets"))
    if gt_flag == 1 and macro_targets:
        annotation["macro_targets"] = macro_targets
    return annotation


def render_candidate_focus_example(
    image_path: str,
    candidate: Dict[str, Any],
    output_path: Path,
    *,
    title: str,
    color: str,
    meta_lines: Sequence[str],
) -> bool:
    if Image is None or ImageDraw is None or ImageFont is None:
        return False
    image_path_obj = Path(str(image_path))
    if not image_path_obj.exists():
        return False
    image = Image.open(image_path_obj).convert("RGB")
    width, height = image.size
    if max(width, height) > 1400:
        scale = 1400.0 / float(max(width, height))
        image = image.resize((int(round(width * scale)), int(round(height * scale))))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size
    line_width = max(2, int(round(max(width, height) / 240.0)))
    bbox = candidate.get("bbox_norm_xyxy")
    if not bbox_is_valid(bbox):
        return False
    draw_norm_box(draw, bbox, width, height, color, title, line_width, font)
    draw_text_block(draw, 0, 0, list(meta_lines), font=font, bg_color="#111111", text_color="white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return True


def build_report_examples(
    canonical_rows: Sequence[Dict[str, Any]],
    batch_rows: Sequence[Dict[str, Any]],
    out_dir: Path,
    max_examples: int,
) -> List[Dict[str, Any]]:
    selected = choose_report_examples(canonical_rows, max_examples=max(0, int(max_examples)))
    batch_map = {(str(row.get("image_id", "")), str(row.get("target_ar", ""))): row for row in batch_rows}
    examples_dir = out_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    for old_path in examples_dir.glob("*.png"):
        old_path.unlink()
    candidate_examples_dir = examples_dir / "candidates"
    candidate_examples_dir.mkdir(parents=True, exist_ok=True)
    for old_path in candidate_examples_dir.glob("*.png"):
        old_path.unlink()
    examples: List[Dict[str, Any]] = []
    for idx, row in enumerate(selected, start=1):
        image_id = str(row.get("image_id", ""))
        target_ar = str(row.get("target_ar", ""))
        safe_ar = target_ar.replace(":", "x").replace("/", "_")
        file_name = f"{idx:02d}_{image_id}_{safe_ar}.png"
        output_path = examples_dir / file_name
        if not render_report_example(row, output_path):
            continue
        decision = safe_dict(row.get("decision"))
        routing = safe_dict(row.get("routing"))
        batch_record = safe_dict(batch_map.get((image_id, target_ar)))
        canonical_candidates = safe_list(row.get("candidates"))
        image_width = row.get("image_w")
        image_height = row.get("image_h")
        image_path_str = str(row.get("image_path", ""))
        if (image_width is None or image_height is None) and Path(image_path_str).exists():
            with Image.open(Path(image_path_str)) as image_obj:
                image_width = int(image_obj.width)
                image_height = int(image_obj.height)
        chosen_candidate = next(
            (candidate for candidate in canonical_candidates if str(candidate.get("candidate_id", "")) == str(decision.get("winner_post_gate_candidate_id", ""))),
            canonical_candidates[0] if canonical_candidates else {},
        )
        matching_targets = safe_list(batch_record.get("matching_targets"))
        first_target = safe_dict(matching_targets[0]) if matching_targets else {}
        candidate_pool = safe_list(batch_record.get("candidate_pool"))
        canonical_core = {
            "routing": {
                "subject_mode": routing.get("subject_mode"),
                "policy_id": routing.get("policy_id"),
                "subject_mode_conf": routing.get("subject_mode_conf"),
                "route_conf": routing.get("route_conf"),
                "shot_type": routing.get("shot_type"),
                "subject_prior_bbox_norm_xyxy": routing.get("subject_prior_bbox_norm_xyxy"),
                "flags": safe_dict(routing.get("flags")),
            },
            "decision": {
                "decision_type": decision.get("decision_type"),
                "delta_vs_base": decision.get("delta_vs_base"),
                "tau_improve": decision.get("tau_improve"),
                "winner_post_gate_candidate_id": decision.get("winner_post_gate_candidate_id"),
            },
            "chosen_candidate": {
                "candidate_id": safe_dict(chosen_candidate).get("candidate_id"),
                "label_type": safe_dict(chosen_candidate).get("label_type"),
                "score_targets": {
                    "score_prob": safe_dict(safe_dict(chosen_candidate).get("score_targets")).get("score_prob"),
                    "rank_pct": safe_dict(safe_dict(chosen_candidate).get("score_targets")).get("rank_pct"),
                    "softmax_local": safe_dict(safe_dict(chosen_candidate).get("score_targets")).get("softmax_local"),
                },
                "macro_targets": safe_dict(chosen_candidate).get("macro_targets"),
                "checklist_labels": safe_dict(chosen_candidate).get("checklist_labels"),
            },
        }
        batch_core = {
            "matching_target_count": len(matching_targets),
            "candidate_pool_count": len(candidate_pool),
            "first_matching_target": {
                "target_id": first_target.get("target_id"),
                "candidate_id": first_target.get("candidate_id"),
                "derived_checklist_targets": {
                    "bin_targets": safe_dict(safe_dict(first_target.get("derived_checklist_targets")).get("bin_targets")),
                    "ord_targets": safe_dict(safe_dict(first_target.get("derived_checklist_targets")).get("ord_targets")),
                    "reg_targets": safe_dict(safe_dict(first_target.get("derived_checklist_targets")).get("reg_targets")),
                    "valid_mask": safe_dict(safe_dict(first_target.get("derived_checklist_targets")).get("valid_mask")),
                },
            },
            "candidate_pool_head": [
                {
                    "candidate_id": pool_row.get("candidate_id"),
                    "label_type": pool_row.get("label_type"),
                    "is_hard_negative": pool_row.get("is_hard_negative"),
                    "is_unsafe_negative": pool_row.get("is_unsafe_negative"),
                }
                for pool_row in candidate_pool[:3]
            ],
        }
        decision_target = safe_dict(batch_record.get("decision_target"))
        gaic_like_preview = {
            "image": {
                "file_name": Path(image_path_str).name,
                "height": image_height,
                "width": image_width,
                "id": idx,
            },
            "positive_annotation": build_gaic_like_annotation_preview(
                image_entry_id=idx,
                image_path=image_path_str,
                routing=routing,
                decision_target=decision_target,
                target_ar=target_ar,
                candidate=first_target if first_target else safe_dict(chosen_candidate),
                gt_flag=1,
            )
            if first_target or chosen_candidate
            else {},
            "negative_annotations": [],
        }
        negative_preview_candidates: List[Dict[str, Any]] = []
        preferred_negative = next((pool for pool in candidate_pool if bool(pool.get("is_hard_negative")) or bool(pool.get("is_unsafe_negative"))), None)
        if preferred_negative is not None:
            negative_preview_candidates.append(preferred_negative)
        for pool in candidate_pool:
            if len(negative_preview_candidates) >= 2:
                break
            if preferred_negative is not None and str(pool.get("candidate_id", "")) == str(preferred_negative.get("candidate_id", "")):
                continue
            negative_preview_candidates.append(pool)
        gaic_like_preview["negative_annotations"] = [
            build_gaic_like_annotation_preview(
                image_entry_id=idx,
                image_path=str(row.get("image_path", "")),
                routing=routing,
                decision_target=decision_target,
                target_ar=target_ar,
                candidate=pool,
                gt_flag=0,
            )
            for pool in negative_preview_candidates
        ]
        candidate_visuals: List[Dict[str, Any]] = []
        visual_specs: List[Tuple[str, str, Dict[str, Any], str, List[str]]] = []
        positive_candidate = first_target if first_target else safe_dict(chosen_candidate)
        if positive_candidate:
            positive_score = safe_float(safe_dict(positive_candidate.get("score_targets")).get("score_prob", 0.0))
            positive_is_final_chosen = str(positive_candidate.get("candidate_id", "")) == str(decision_target.get("winner_post_gate_candidate_id", ""))
            positive_meta = [
                f"{image_id} | {target_ar} | matching_target_example",
                f"candidate_id={positive_candidate.get('candidate_id', '')}",
                f"score={positive_score:.6f} | gt_flag=1 | label_type=matching_target",
                f"decision={decision_target.get('decision_type')} | final_chosen={positive_is_final_chosen}",
            ]
            visual_specs.append(("positive", "matching target", positive_candidate, "#2ecc71", positive_meta))
        for neg_idx, pool in enumerate(negative_preview_candidates, start=1):
            neg_score = safe_float(pool.get("score_prob", pool.get("score_rank_pct", 0.0)))
            neg_meta = [
                f"{image_id} | {target_ar} | negative_{neg_idx}",
                f"candidate_id={pool.get('candidate_id', '')}",
                f"score={neg_score:.6f} | gt_flag=0 | label_type={pool.get('label_type', '')}",
                f"hard={bool(pool.get('is_hard_negative'))} | unsafe={bool(pool.get('is_unsafe_negative'))}",
            ]
            visual_specs.append((f"negative_{neg_idx}", f"negative {neg_idx}", pool, "#e74c3c" if bool(pool.get("is_hard_negative")) or bool(pool.get("is_unsafe_negative")) else "#f39c12", neg_meta))
        for visual_name, display_title, candidate_row, color, meta_lines in visual_specs:
            visual_path = candidate_examples_dir / f"{idx:02d}_{image_id}_{safe_ar}_{visual_name}.png"
            if render_candidate_focus_example(
                image_path_str,
                candidate_row,
                visual_path,
                title=display_title,
                color=color,
                meta_lines=meta_lines,
            ):
                candidate_visuals.append(
                    {
                        "label": visual_name,
                        "path": visual_path.relative_to(out_dir).as_posix(),
                    }
                )
        examples.append(
            {
                "image_id": image_id,
                "target_ar": target_ar,
                "subject_mode": routing.get("subject_mode"),
                "decision_type": decision.get("decision_type"),
                "delta_vs_base": decision.get("delta_vs_base"),
                "route_conf": routing.get("route_conf"),
                "target_reliability": safe_dict(row.get("teacher_meta")).get("target_reliability"),
                "path": output_path.relative_to(out_dir).as_posix(),
                "canonical_core": canonical_core,
                "batch_core": batch_core,
                "gaic_like_core": gaic_like_preview,
                "candidate_visuals": candidate_visuals,
            }
        )
    return examples


def json_block(payload: Any) -> str:
    return "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```"


def format_range(values: Sequence[Optional[float]]) -> str:
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return "n/a"
    return f"{min(nums):.6f} .. {max(nums):.6f}"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def infer_dataset_context(*paths: Path) -> Dict[str, str]:
    for path in paths:
        parts = [part for part in path.as_posix().split("/") if part]
        if "data" in parts and "artifacts" in parts:
            data_idx = parts.index("data")
            art_idx = parts.index("artifacts")
            dataset_parts = parts[data_idx + 1 : art_idx]
            if dataset_parts:
                dataset_label = "/".join(dataset_parts)
                dataset_short = dataset_parts[0]
                return {
                    "dataset_label": dataset_label,
                    "dataset_short": dataset_short,
                    "report_title": f"{dataset_short} Training Label Report",
                }
    fallback = paths[0].parent.name if paths else "Dataset"
    return {
        "dataset_label": fallback,
        "dataset_short": fallback,
        "report_title": "Training Label Report",
    }


def collect_label_guide_summary(
    canonical_rows: Sequence[Dict[str, Any]],
    batch_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    score_values: Dict[str, List[float]] = defaultdict(list)
    macro_values: Dict[str, List[float]] = defaultdict(list)
    checklist_score_values: Dict[str, List[float]] = defaultdict(list)
    checklist_label_values: Dict[str, set[str]] = defaultdict(set)
    derived_values: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    subject_mode_to_id: Dict[str, int] = {}
    policy_ids: set[str] = set()
    shot_types: set[str] = set()
    decision_type_to_id: Dict[str, int] = {}
    mode_to_policy: Dict[str, str] = {}
    why_tags: set[str] = set()
    reject_tags: set[str] = set()
    routing_flag_counts: Counter[str] = Counter()

    for row in canonical_rows:
        routing = safe_dict(row.get("routing"))
        decision = safe_dict(row.get("decision"))
        mode = str(routing.get("subject_mode", ""))
        if mode:
            subject_mode_to_id[mode] = int(routing.get("subject_mode_id", -1))
        policy = str(routing.get("policy_id", ""))
        if policy:
            policy_ids.add(policy)
        if mode and policy:
            mode_to_policy[mode] = policy
        shot_type = str(routing.get("shot_type", ""))
        if shot_type:
            shot_types.add(shot_type)
        for flag_name, flag_value in safe_dict(routing.get("flags")).items():
            if bool(flag_value):
                routing_flag_counts[str(flag_name)] += 1
        decision_type = str(decision.get("decision_type", ""))
        if decision_type:
            decision_type_to_id[decision_type] = int(decision.get("decision_id", -1))
        for candidate in safe_list(row.get("candidates")):
            for tag in safe_list(candidate.get("why_tags")):
                why_tags.add(str(tag))
            for tag in safe_list(candidate.get("reject_tags")):
                reject_tags.add(str(tag))
            for key, value in safe_dict(candidate.get("score_targets")).items():
                if isinstance(value, (int, float)):
                    score_values[key].append(float(value))
            for key, value in safe_dict(candidate.get("macro_targets")).items():
                if isinstance(value, (int, float)):
                    macro_values[key].append(float(value))
            for key, value in safe_dict(candidate.get("checklist_scores")).items():
                if isinstance(value, (int, float)):
                    checklist_score_values[key].append(float(value))
            for key, value in safe_dict(candidate.get("checklist_labels")).items():
                checklist_label_values[key].add(str(value))

    for row in batch_rows:
        for target in safe_list(row.get("matching_targets")):
            derived = safe_dict(target.get("derived_checklist_targets"))
            for group in ("bin_targets", "ord_targets", "reg_targets", "valid_mask"):
                for key, value in safe_dict(derived.get(group)).items():
                    if isinstance(value, (int, float)):
                        derived_values[(group, key)].append(float(value))

    return {
        "score_values": dict(score_values),
        "macro_values": dict(macro_values),
        "checklist_score_values": dict(checklist_score_values),
        "checklist_label_values": {key: sorted(values) for key, values in checklist_label_values.items()},
        "derived_values": dict(derived_values),
        "subject_mode_to_id": dict(sorted(subject_mode_to_id.items())),
        "policy_ids": sorted(policy_ids),
        "shot_types": sorted(shot_types),
        "decision_type_to_id": dict(sorted(decision_type_to_id.items())),
        "mode_to_policy": dict(sorted(mode_to_policy.items())),
        "why_tags": sorted(why_tags),
        "reject_tags": sorted(reject_tags),
        "routing_flag_counts": dict(sorted(routing_flag_counts.items())),
    }


def build_gaic_like_report_summary(batch_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    label_type_counts: Counter[str] = Counter()
    positive_count = 0
    negative_count = 0
    score_values: List[float] = []
    raw_image_ids: set[str] = set()
    for row in batch_rows:
        raw_image_ids.add(str(row.get("image_id", "")))
        for target in safe_list(row.get("matching_targets")):
            positive_count += 1
            label_type_counts["matching_target"] += 1
            score_values.append(safe_float(safe_dict(target.get("score_targets")).get("score_prob", 0.0)))
        for candidate in safe_list(row.get("candidate_pool")):
            negative_count += 1
            label_type_counts[str(candidate.get("label_type", ""))] += 1
            score_values.append(safe_float(candidate.get("score_prob", candidate.get("score_rank_pct", 0.0))))
    return {
        "image_rows": len(batch_rows),
        "unique_raw_images": len(raw_image_ids),
        "annotation_rows": positive_count + negative_count,
        "gt_flag_1_rows": positive_count,
        "gt_flag_0_rows": negative_count,
        "label_type_counts": dict(label_type_counts),
        "score_range": format_range(score_values),
    }


def build_label_guide_section(label_guide: Dict[str, Any]) -> str:
    dataset_label = str(label_guide.get("dataset_label") or "current dataset")
    run_label = str(label_guide.get("run_label") or "").strip()
    observed_scope = f"{dataset_label} / {run_label}" if run_label else dataset_label
    score_values = label_guide["score_values"]
    macro_values = label_guide["macro_values"]
    checklist_score_values = label_guide["checklist_score_values"]
    checklist_label_values = label_guide["checklist_label_values"]
    derived_values = label_guide["derived_values"]
    subject_mode_to_id = label_guide["subject_mode_to_id"]
    policy_ids = label_guide["policy_ids"]
    shot_types = label_guide["shot_types"]
    decision_type_to_id = label_guide["decision_type_to_id"]
    mode_to_policy = label_guide["mode_to_policy"]
    why_tags = label_guide["why_tags"]
    reject_tags = label_guide["reject_tags"]
    routing_flag_counts = label_guide["routing_flag_counts"]

    dataset_view_rows = [
        (
            "`*_canonical.jsonl`",
            "사람이 읽고 QA/재가공하기 좋은 정본 레코드",
            "`routing + baseline + decision + candidates[]` 전체 보존",
            "오프라인 분석, 데이터 감사, 재파생, 리포트 생성에 사용. 직접 Hungarian target으로 쓰기보다 원천 레코드 역할",
        ),
        (
            "`*_batch.jsonl`",
            "모델 학습 직전 dataloader 친화형 레코드",
            "`decision_target + matching_targets + candidate_pool + derived_checklist_targets`",
            "Conditional-DETR 학습 dataloader의 기본 입력으로 권장. Hungarian matching과 auxiliary loss 계산에 바로 연결",
        ),
    ]
    operational_rows = [
        (
            "matching_targets",
            "batch",
            "safe positive set",
            "각 `(image_id, target_ar)`에서 Hungarian matching에 실제로 투입되는 정답 집합",
        ),
        (
            "candidate_pool",
            "batch",
            "non-matching 후보 집합",
            "hard/unsafe/near/general negative를 보관. main DETR target은 아니고 negative mining, ranking, audit 용도",
        ),
        (
            "label_type",
            "canonical/batch",
            "candidate provenance class",
            "`top1`, `soft_positive`, `baseline_positive`, `near_negative`, `hard_negative`, `unsafe_negative` 등",
        ),
        (
            "why_tags",
            "canonical + matching_targets",
            "선정 이유 요약 태그",
            "teacher가 해당 crop을 좋게 본 이유의 힌트. loss target이라기보다 설명/분석/샘플 리포트용 side signal",
        ),
        (
            "reject_tags",
            "canonical + candidate_pool",
            "탈락/위험 사유 태그",
            "face_cut, joint_cutoff, head_top_cut 등. unsafe negative 구분과 audit에 중요",
        ),
        (
            "applicable_mask",
            "canonical + batch",
            "mode별 적용 가능 여부",
            "예: scene에서 face/headroom 항목은 0. item이 의미가 있는지 표시",
        ),
        (
            "observed_mask",
            "canonical + batch",
            "실측 신뢰 가능 여부",
            "항목이 개념적으로는 적용 가능해도 detector/OCR 관측이 약하면 0",
        ),
        (
            "valid_mask",
            "batch derived",
            "`applicable_mask ∧ observed_mask`",
            "실제 auxiliary loss에 넣을지 결정하는 최종 마스크",
        ),
    ]
    routing_rows = [
        (
            "subject_mode",
            ", ".join(subject_mode_to_id.keys()),
            "router가 image-level context를 요약한 상위 mode label",
            "portrait/scene/object/copyspace 계열의 조건부 학습 축",
        ),
        (
            "subject_mode_id",
            ", ".join(f"{mode}={idx}" for mode, idx in subject_mode_to_id.items()),
            "subject_mode의 정수 인덱스",
            "`subject_mode_vocab.json` 기준 deterministic id. 현재 구현은 관측 mode를 정렬한 뒤 0..N-1 부여",
        ),
        (
            "policy_id",
            ", ".join(policy_ids),
            "mode별 scoring/policy schedule id",
            "`subject_mode`보다 더 직접적으로 tau/lambda/keep policy를 고정하는 route key",
        ),
        (
            "shot_type",
            ", ".join(shot_types),
            "portrait 세부 촬영 타입",
            "태그 키워드 + human evidence 기반 휴리스틱. portrait route가 아니면 후단에서 `unknown` 강제",
        ),
        (
            "decision_type",
            ", ".join(decision_type_to_id.keys()),
            "최종 crop 정책 결정",
            "`keep_full`, `minimal_crop`, `crop`의 3값",
        ),
        (
            "decision_id",
            ", ".join(f"{name}={idx}" for name, idx in decision_type_to_id.items()),
            "decision_type의 정수 인덱스",
            "현재 고정 매핑은 `keep_full=0`, `minimal_crop=1`, `crop=2`",
        ),
    ]
    mode_policy_rows = [
        (mode, policy, f"{dataset_label} observed mapping")
        for mode, policy in mode_to_policy.items()
    ]

    score_rows = [
        ("score_prob", "`score_targets`", "sigmoid(z_local)로 만든 pseudo probability", "(0,1)", format_range(score_values.get("score_prob", []))),
        ("rank_pct", "`score_targets`", "그룹 내 내림차순 percentile", "[0,1]", format_range(score_values.get("rank_pct", []))),
        ("z_local", "`score_targets`", "그룹 내 robust z-score", "[-6,6] clip", format_range(score_values.get("z_local", []))),
        ("softmax_local", "`score_targets`", "그룹 내 local softmax, per-group 합=1", "[0,1]", format_range(score_values.get("softmax_local", []))),
        ("pseudo_mos_1to5", "`score_targets`", "z_local을 1~5로 매핑한 teacher-like score", "[1,5]", format_range(score_values.get("pseudo_mos_1to5", []))),
        ("score_raw_rank", "`score_targets`", "teacher rank/final raw score", "teacher-defined", format_range(score_values.get("score_raw_rank", []))),
        ("score_raw_policy", "`score_targets`", "policy-aware raw score", "teacher-defined", format_range(score_values.get("score_raw_policy", []))),
        ("score_margin_to_top1", "`score_targets`", "동일 `(image, AR)` top1과의 점수 차", "[0,+)", format_range(score_values.get("score_margin_to_top1", []))),
    ]
    macro_rows = [
        ("A_macro", "aesthetic/align 축 macro", "대체로 [0,1]", format_range(macro_values.get("A_macro", []))),
        ("S_macro", "subject preservation/safety 축", "대체로 [0,1]", format_range(macro_values.get("S_macro", []))),
        ("C_macro", "composition/context/copyspace 축", "대체로 [0,1]", format_range(macro_values.get("C_macro", []))),
        ("T_macro", "teacher consensus/teacher-alignment 축", "대체로 [0,1]", format_range(macro_values.get("T_macro", []))),
    ]
    checklist_score_rows = [
        ("subject_coverage_ratio", "주 피사체 포함 비율", "보통 [0,1]", format_range(checklist_score_values.get("subject_coverage_ratio", []))),
        ("headroom_ratio", "머리 상단 여백 비율", "mode-dependent", format_range(checklist_score_values.get("headroom_ratio", []))),
        ("lookroom_ratio", "시선/진행 방향 여백 비율", "mode-dependent", format_range(checklist_score_values.get("lookroom_ratio", []))),
        ("text_keep_ratio", "텍스트 유지 비율", "보통 [0,1]", format_range(checklist_score_values.get("text_keep_ratio", []))),
        ("horizon_y", "수평선 y 위치", "보통 [0,1] 기대이나 detector drift 가능", format_range(checklist_score_values.get("horizon_y", []))),
        ("horizon_visible_ratio", "수평선 관측 가시 비율", "[0,1]", format_range(checklist_score_values.get("horizon_visible_ratio", []))),
        ("symmetry_score", "대칭성/균형 점수", "teacher-derived", format_range(checklist_score_values.get("symmetry_score", []))),
    ]
    checklist_rows = [
        (
            "subject_coverage",
            ", ".join(checklist_label_values.get("subject_coverage", [])),
            "`subject_coverage_ratio`",
            "`excellent >= 0.95`, `good >= 0.85`, `marginal >= 0.70`, 그 미만 `poor`",
            f"`ord_targets.subject_coverage`는 class id이며 `0=poor, 1=marginal, 2=good, 3=excellent`; 현재 관측 범위 `{format_range(derived_values.get(('ord_targets', 'subject_coverage'), []))}`",
        ),
        (
            "subject_scale",
            ", ".join(checklist_label_values.get("subject_scale", [])),
            "subject scale value",
            "mode별 `target_range=[lo, hi]` 기준으로 `<lo: too_loose`, `lo~hi: ideal_scale`, `>hi: too_tight`",
            f"`ord_targets.subject_scale`는 class id이며 `0=too_loose, 1=ideal_scale, 2=too_tight`; quality score가 아니라 이산 상태 id, 현재 범위 `{format_range(derived_values.get(('ord_targets', 'subject_scale'), []))}`",
        ),
        (
            "headroom",
            ", ".join(checklist_label_values.get("headroom", [])),
            "`headroom_ratio`",
            "기본 range 예: `[0.03, 0.12]`; `<lo: tight`, `lo~hi: ok`, `>hi: loose`, 비적용/미관측은 `na`",
            f"`ord_targets.headroom`는 class id이며 `0=tight, 1=ok, 2=loose`; `valid_mask.headroom`은 `1=loss 사용, 0=ignore`, 현재 범위는 각각 `{format_range(derived_values.get(('ord_targets', 'headroom'), []))}`, `{format_range(derived_values.get(('valid_mask', 'headroom'), []))}`",
        ),
        (
            "lookroom",
            ", ".join(checklist_label_values.get("lookroom", [])),
            "`lookroom_ratio`",
            "기본 range 예: `[1.2, 2.5]`; `<lo: insufficient`, `lo~hi: adequate`, `>hi: excessive`, 비적용/미관측은 `na`",
            f"`ord_targets.lookroom`는 class id이며 `0=insufficient, 1=adequate, 2=excessive`; `valid_mask.lookroom`은 `1=loss 사용`, 현재 범위는 각각 `{format_range(derived_values.get(('ord_targets', 'lookroom'), []))}`, `{format_range(derived_values.get(('valid_mask', 'lookroom'), []))}`",
        ),
        (
            "face_cut",
            ", ".join(checklist_label_values.get("face_cut", [])),
            "face cut flag/value",
            "`no_face_cut -> 0`, `face_cut -> 1`",
            f"`bin_targets.face_cut`는 binary id이며 `0=no_face_cut, 1=face_cut`; 현재 관측 범위 `{format_range(derived_values.get(('bin_targets', 'face_cut'), []))}`",
        ),
        (
            "joint_cut",
            ", ".join(checklist_label_values.get("joint_cut", [])),
            "joint cut severity/value",
            "`no_joint_cut -> 0`, `joint_cut_mild/severe -> 1`로 binary collapse",
            f"`bin_targets.joint_cut`는 binary id이며 `0=no_joint_cut, 1=cut`; mild/severe는 canonical label에 남고 batch에선 binary로 collapse, 현재 범위 `{format_range(derived_values.get(('bin_targets', 'joint_cut'), []))}`",
        ),
        (
            "copyspace",
            ", ".join(checklist_label_values.get("copyspace", [])),
            "copyspace quality/side",
            "현재 subset에서는 `partial`, `preserved`만 관측; 전체 ontology는 `missing`까지 가능",
            f"`ord_targets.copyspace`는 class id이며 전체 ontology는 `0=missing, 1=partial, 2=preserved`; 현재 subset 관측 범위 `{format_range(derived_values.get(('ord_targets', 'copyspace'), []))}`",
        ),
        (
            "context",
            ", ".join(checklist_label_values.get("context", [])),
            "context preservation value",
            "현재 subset에서는 `poor/partial/preserved/excessive` 관측; batch ordinal은 coarse하게 lower/partial/preserved로 축약",
            f"`ord_targets.context`는 field-specific ordinal id로 사용되며 현재 batch에서는 주로 `1=partial 계열`, `2=preserved 계열`이 관측; 값 범위 `{format_range(derived_values.get(('ord_targets', 'context'), []))}`",
        ),
        (
            "horizon_state",
            ", ".join(checklist_label_values.get("horizon_state", [])),
            "`horizon_y`, `horizon_visible_ratio`, state",
            f"현재 {dataset_label} canonical positive들에서는 전부 `na`; scene routing이어도 `observed_mask`가 0일 수 있음",
            f"`reg_targets.horizon_y`/`horizon_visible_ratio`는 연속값이고 `valid_mask.horizon_*`는 `1=loss 사용, 0=ignore`; 현재 `valid_mask.horizon_y` 범위 `{format_range(derived_values.get(('valid_mask', 'horizon_y'), []))}`",
        ),
        (
            "text_keep",
            ", ".join(checklist_label_values.get("text_keep", [])),
            "`text_keep_ratio`",
            f"현재 {dataset_label} canonical positive들에서는 전부 `na`; 전체 ontology는 `text_cut/partial/preserved`를 가질 수 있음",
            f"`ord_targets.text_keep`는 전체 ontology에서 `0=cut, 1=partial, 2=preserved`; 현재 subset은 비관측이라 `valid_mask.text_keep` 범위 `{format_range(derived_values.get(('valid_mask', 'text_keep'), []))}`",
        ),
    ]

    return "\n\n".join(
        [
            "## 4. Train Label Field Guide",
            "",
            f"- 아래 표의 `empirical range`는 현재 `{observed_scope}` 산출물에서 실제 관측된 범위입니다. theoretical range와 다를 수 있습니다.",
            "- `canonical`은 사람이 읽기 좋은 원본형 라벨이고, `batch`는 dataloader에서 바로 쓰는 파생형입니다. 표의 마지막 열은 batch에서 실제로 어떻게 소비되는지를 요약합니다.",
            "",
            "### 4.0 Canonical vs Batch Usage",
            "",
            markdown_table(
                ["file/view", "primary role", "core payload", "recommended use"],
                [list(row) for row in dataset_view_rows],
            ),
            "",
            "- Conditional-DETR 학습 dataloader는 기본적으로 `train_conditional_detr_batch.jsonl`을 읽는 편이 맞습니다.",
            "- 이유는 Hungarian matching이 필요로 하는 `matching_targets`, policy supervision용 `decision_target`, auxiliary typed target과 `valid_mask`가 이미 정리되어 있기 때문입니다.",
            "- `train_conditional_detr_canonical.jsonl`은 dataloader가 직접 consume하기보다, batch view를 재구성하거나 디버깅/리포트/통계 재산출에 쓰는 원천 데이터로 보는 것이 안전합니다.",
            "- 예외적으로 end-to-end dataset class 안에서 canonical만 읽고 내부에서 batch를 다시 파생할 수도 있지만, 현재 저장 산출물 기준으로는 batch를 학습 입력, canonical을 reference/source-of-truth로 두는 구성이 가장 단순합니다.",
            "",
            "### 4.1 Routing / Policy / Decision Fields",
            "",
            markdown_table(
                [f"field", f"observed values ({dataset_label})", "semantic", "note"],
                [list(row) for row in routing_rows],
            ),
            "",
            "- `shot_type`은 미구현 필드가 아니라 `src/score_teacher.py::infer_shot_type(...)`에서 계산됩니다.",
            "- 먼저 `has_human_evidence=False`면 즉시 `unknown`입니다.",
            "- 태그 문자열에 `group/family/team/friends/couple/crowd/people`가 있으면 `group`, `full body/full length/standing/feet/legs` 계열이면 `full`, `waist up/upper body/half body/medium shot/torso` 계열이면 `half`, `headshot/close-up/portrait/face/beauty/selfie/profile picture` 계열이면 `headshot`입니다.",
            "- 어떤 키워드에도 걸리지 않으면 `unknown`입니다. 또한 최종 `subject_mode`가 portrait 계열이 아니면 후단 routing schedule에서 다시 `shot_type='unknown'`으로 강제됩니다.",
            "- portrait로 시작했더라도 human evidence 또는 person count가 부족해 non-portrait mode로 fallback 되면 `shot_type`도 `unknown`으로 리셋됩니다. 반대로 `portrait_group` route는 후단에서 `shot_type='group'`으로 고정됩니다.",
            "- `score_targets`는 완전히 mode-agnostic하지 않습니다. `score_raw_rank`와 `score_raw_policy`는 먼저 `subject_mode/policy_id`에 따라 달라지는 lambda, area prior, keep threshold, cut penalty의 영향을 받습니다.",
            "- 다만 `rank_pct`, `z_local`, `softmax_local`, `score_prob`는 이미 계산된 raw score를 동일 `(image_id, target_ar)` 그룹 안에서 다시 정규화한 값이므로, mode의 직접 효과라기보다 mode-conditioned raw score 분포의 간접 결과로 보는 편이 맞습니다.",
            "- 즉 `subject_mode`가 달라지면 분포 차이가 가장 직접적으로 나타나는 필드는 `score_raw_rank`, `score_raw_policy`, 그리고 decision의 `tau_improve`이며, 나머지 score target은 그 상대 순위/분산을 local normalization으로 압축한 값입니다.",
            "",
            "### 4.1c Subject Prior / Tiny-Human Guard",
            "",
            "- `subject prior`는 최종 crop 정답 박스가 아니라, router가 후단 candidate generation과 portrait safety 판단을 위해 남겨 두는 앵커 박스입니다.",
            "- 기본적으로는 person/subject union box를 우선 사용하지만, tiny human만 검출된 wide scene에서는 이 박스를 그대로 쓰면 과도한 portrait crop으로 붕괴할 수 있습니다.",
            "- 이를 막기 위해 `routing.flags.contextual_tiny_human=true`이면 router가 tiny human 검출을 portrait anchor로 보지 않고 `scene_general` 쪽으로 fallback합니다.",
            "- 이 경우 routing 단계에서 `union_box_xyxy`를 비우고, candidate generation에서는 stale person union을 재사용하지 않으며 scene fallback prior `[0.2, 0.2, 0.8, 0.8]`를 subject prior로 사용합니다.",
            f"- 현재 `{observed_scope}`에서 `contextual_tiny_human`이 관측된 row 수는 `{routing_flag_counts.get('contextual_tiny_human', 0)}`입니다.",
            "- 따라서 orange box가 항상 detector가 본 사람 bbox를 뜻하는 것은 아닙니다. scene fallback이 발동한 샘플에서는 '장면 중앙의 보수적 prior'를 뜻합니다.",
            "",
            "### 4.1a Observed Mode-Policy Mapping",
            "",
            markdown_table(
                ["subject_mode", "policy_id", "note"],
                [list(row) for row in mode_policy_rows],
            ),
            "",
            f"- 현재 `{dataset_label}`에 실제 관측된 mode들은 policy와 사실상 1:1로 매핑됩니다. 즉 이 subset 안에서는 서로 다른 `subject_mode`가 완전히 동일한 `policy_id`를 공유하는 경우는 없습니다.",
            "- 하지만 구현 레벨에서는 정책 묶음이 있습니다. 모든 `scene_*` mode는 `scene_v1` branch를 공유하고, `portrait_single/portrait_group`는 같은 portrait branch를 쓰되 group 전용 보정만 다르며, `object_single/object_multi`도 같은 object branch를 쓰되 multi일 때 coverage 가중이 더 강해집니다.",
            "- `other_ambiguous`와 person-signal 부족 fallback은 `generic_v1`로 수렴합니다. 따라서 데이터셋이 넓어지면 서로 다른 세부 mode가 동일 scoring policy family를 공유할 수 있습니다.",
            "",
            "### 4.1b Operational / Provenance Fields",
            "",
            markdown_table(
                ["field", "where", "semantic", "training use"],
                [list(row) for row in operational_rows],
            ),
            "",
            f"- `why_tags` observed examples in {dataset_label}: `{', '.join(why_tags[:20])}`",
            f"- `reject_tags` observed examples in {dataset_label}: `{', '.join(reject_tags[:20])}`",
            "",
            "### 4.2 Score Targets",
            "",
            markdown_table(
                ["field", "location", "semantic", "theoretical range", "empirical range"],
                [list(row) for row in score_rows],
            ),
            "",
            "### 4.3 Macro Targets",
            "",
            markdown_table(
                ["field", "semantic", "expected range", "empirical range"],
                [list(row) for row in macro_rows],
            ),
            "",
            "### 4.4 Checklist Scores",
            "",
            markdown_table(
                ["field", "semantic", "expected range", "empirical range"],
                [list(row) for row in checklist_score_rows],
            ),
            "",
            "### 4.5 Checklist Labels / Derived Targets",
            "",
            "- `batch view` 열의 숫자 범위는 연속 점수 범위가 아니라, 현재 데이터셋에서 실제 관측된 파생 텐서 값의 min/max입니다.",
            "- `ord_targets.*`는 공통 점수축이 아니라 field별 discrete class id입니다. 예를 들어 `headroom`의 `0/1/2`는 `tight/ok/loose`를 뜻하지, 0.0에서 2.0 사이의 연속 세기를 뜻하지 않습니다.",
            "- `bin_targets.*`는 binary id이고 보통 `0=문제 없음`, `1=문제 있음`으로 해석합니다.",
            "- `reg_targets.*`는 실제 연속 측정값입니다.",
            "- `valid_mask.*`는 값 자체의 quality가 아니라 loss on/off 스위치입니다. `1`이면 해당 항목을 학습 손실에 포함하고, `0`이면 비적용/미관측이어서 무시합니다.",
            "",
            markdown_table(
                ["field", f"observed labels ({dataset_label})", "score source", "labeling rule / threshold", "batch view"],
                [list(row) for row in checklist_rows],
            ),
            "",
        ]
    )


def build_markdown_report(
    teacher_scores_jsonl: Path,
    out_dir: Path,
    image_root: Optional[Path],
    qa_summary: Dict[str, Any],
    validation_summary: Dict[str, Any],
    examples: Sequence[Dict[str, Any]],
    label_guide: Dict[str, Any],
    gaic_like_summary: Dict[str, Any],
    dataset_context: Dict[str, str],
) -> str:
    counts = qa_summary["counts"]
    report_title = str(dataset_context.get("report_title") or "Training Label Report")
    lines = [
        f"# {report_title}",
        "",
        f"- input teacher scores: `{teacher_scores_jsonl}`",
        f"- output dir: `{out_dir}`",
        f"- image root: `{image_root}`" if image_root is not None else "- image root: `<unresolved>`",
        "",
        "## 1. 산출물 개요",
        "",
        f"- pairwise rows: `{counts['pairwise']}`",
        f"- listwise rows: `{counts['listwise']}`",
        f"- decision rows: `{counts['decision']}`",
        f"- checklist rows: `{counts['checklist']}`",
        f"- regression rows: `{counts['regression']}`",
        f"- conditional detr canonical rows: `{counts['conditional_detr_canonical']}`",
        f"- conditional detr batch rows: `{counts['conditional_detr_batch']}`",
        f"- conditional detr skipped rows: `{counts['conditional_detr_skipped']}`",
        "",
        "## 2. Conditional-DETR 요약",
        "",
        json_block(qa_summary["conditional_detr"]),
        "",
        "## 3. 검증 결과",
        "",
        json_block(compact_validation_summary(validation_summary)),
        "",
        build_label_guide_section(label_guide),
        "",
        "## 5. GAIC-like 변환본 요약",
        "",
        "- GAIC-like 변환본은 `train_conditional_detr_batch.jsonl`에서 `matching_targets`와 `candidate_pool`을 함께 COCO/GAIC 스타일 annotation으로 펼친 뷰입니다.",
        "- `matching_targets`는 `gt_flag=1`, `candidate_pool`은 `gt_flag=0`으로 저장되며, `score`는 positive/negative 모두 SSTK local score를 사용합니다.",
        "- 동일 report 예시 샘플 아래에도 각 샘플의 변환 결과 일부를 같이 붙여 두었습니다.",
        "",
        json_block(gaic_like_summary),
        "",
        "## 6. Legacy FinalScore QA",
        "",
        "- 여기의 `Legacy`는 이 QA가 기존 FinalScore/teacher-ranking 기반 산출물(`pairwise/listwise/decision/checklist/regression`)의 품질 점검 지표이기 때문입니다.",
        "- 즉 이 섹션은 Conditional-DETR의 직접 입력 스키마를 설명하는 부분이 아니라, 새 canonical/batch schema를 만들기 전에 존재하던 구세대 학습 뷰와 teacher score 분포를 계속 추적하기 위한 호환성용 모니터링입니다.",
        "- 운영상 의미는 여전히 크지만, DETR 학습에서 직접 consume하는 핵심 라벨은 위의 Conditional-DETR canonical/batch JSONL입니다.",
        "",
        json_block(
            {
                "pairwise": qa_summary["pairwise"],
                "listwise": qa_summary["listwise"],
                "decision": qa_summary["decision"],
                "checklist": qa_summary["checklist"],
                "regression": qa_summary["regression"],
            }
        ),
        "",
        "## 7. 예시 샘플",
        "",
        "- 색상 범례: `green=chosen`, `blue=baseline`, `yellow=pre-gate best`, `orange=subject prior`, `red=hard/unsafe negative`",
        "- `subject prior`는 router가 추정한 주 피사체 prior/union box입니다. 크롭 정답 박스가 아니라, mode/routing conditioning과 안전 여백 판단의 기준 앵커입니다.",
        "- `baseline`은 정책 비교 기준 박스입니다. 보통 full image 또는 매우 보수적인 기준 crop이며, `delta_vs_base`와 `keep_full / minimal_crop / crop` 판단의 출발점입니다.",
        "- `pre-gate best`는 순수 score ranking 상 가장 높았던 후보입니다. 아직 `tau_improve`, keep policy, safety gate를 통과하기 전이라 최종 teacher 선택과 다를 수 있습니다.",
        "- `chosen`은 gate 이후 최종 채택된 박스입니다. Conditional-DETR batch에서는 safe positive set 안에 있을 때만 `matching_targets`로 들어갑니다.",
        "- `hard/unsafe negative`는 얼굴/관절 컷, head-top 컷, severe reject 같은 이유로 학습 타깃에서 제외된 후보입니다. Hungarian matching target이 아니라 `candidate_pool`/negative mining 용도입니다.",
        "- 노란 박스가 없으면 `pre-gate best`와 `chosen`이 동일하다는 뜻이고, 파란 박스가 거의 보이지 않으면 `baseline`과 `chosen`이 거의 겹친 경우입니다.",
        "- 이미지 상단 캡션은 `image_id | target_ar | subject_mode | decision_type` 순서이며, 두 번째 검은 라벨이 보이면 최종 선택 crop의 대표 `why_tags`입니다.",
        "",
    ]
    if not examples:
        lines.append("- 생성된 예시 이미지가 없습니다.")
    for idx, example in enumerate(examples, start=1):
        lines.extend(
            [
                f"### {idx}. {example['image_id']} / {example['target_ar']}",
                "",
                f"- subject_mode: `{example['subject_mode']}`",
                f"- decision_type: `{example['decision_type']}`",
                f"- delta_vs_base: `{example['delta_vs_base']}`",
                f"- route_conf: `{example['route_conf']}`",
                f"- target_reliability: `{example['target_reliability']}`",
                "",
                "canonical 핵심 train labels:",
                "",
                json_block(example["canonical_core"]),
                "",
                "- `canonical_core.routing`은 이 샘플을 어떤 mode/policy로 해석했는지 보여 주는 image-level 조건부 정보입니다.",
                "- `canonical_core.routing.subject_prior_bbox_norm_xyxy`는 orange box 좌표입니다. `flags.contextual_tiny_human=true`이면 detector person box가 아니라 scene fallback prior일 수 있습니다.",
                "- `canonical_core.decision`은 최종 teacher decision과 `delta_vs_base`, `tau_improve`, winner id를 요약합니다.",
                "- `canonical_core.chosen_candidate`는 최종 선택 crop의 핵심 score/macro/checklist label만 추린 것입니다.",
                "",
                "batch 핵심 train labels:",
                "",
                json_block(example["batch_core"]),
                "",
                "- `matching_target_count`는 Hungarian matching에 실제로 들어가는 safe positive 개수입니다.",
                "- `candidate_pool_count`는 matching target으로 채택되지 않은 나머지 후보 수입니다.",
                "- `first_matching_target`는 batch 안 positive target 하나의 축약 예시입니다.",
                "- `bin_targets`는 binary class id입니다. 보통 `0=문제 없음`, `1=문제 있음`으로 읽습니다.",
                "- `ord_targets`는 연속 점수가 아니라 field별 discrete class id입니다. 예: `headroom`은 `0=tight, 1=ok, 2=loose`입니다.",
                "- `reg_targets`는 실제 연속형 측정값입니다.",
                "- `valid_mask`는 값의 크기가 아니라 loss on/off 스위치입니다. `1=해당 항목 학습`, `0=비적용/미관측으로 무시`입니다.",
                "- `candidate_pool_head`는 남은 후보 중 앞부분 몇 개를 보여 주는 미리보기이며, 여기의 `hard/unsafe` 정보는 negative mining이나 audit용입니다.",
                "",
                "GAIC-like 변환본 핵심 preview:",
                "",
                json_block(example["gaic_like_core"]),
                "",
                "- `image`는 GAIC-like `images[]` 행의 축약 예시입니다.",
                "- `positive_annotation`은 `matching_targets` 중 1개가 `gt_flag=1`로 변환된 모습입니다.",
                "- `negative_annotations`는 `candidate_pool`에서 추린 예시이며 `gt_flag=0`으로 들어갑니다.",
                "",
                f"![]({example['path']})",
                "",
                "candidate 별 시각화:",
                "",
                "- 각 candidate 시각화는 해당 후보 bbox 하나만 강조하며, 상단 검은 박스에는 `image_id | target_ar | positive/negative`, `candidate_id`, `score`, `gt_flag`, `label_type`, 필요 시 `hard/unsafe` 상태를 적었습니다.",
                "- 여기서 `positive` 시각화는 `final chosen`과 동의어가 아니라, `matching_targets` 안에 포함된 safe positive 예시 1개입니다. 따라서 `keep_full`/`minimal_crop` 샘플에서는 full-image baseline이 최종 선택이어도 crop 후보가 positive로 함께 남을 수 있습니다.",
                "",
            ]
        )
        for visual in example.get("candidate_visuals", []):
            lines.extend(
                [
                    f"- `{visual['label']}`",
                    f"![]({visual['path']})",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def checklist_target_schema() -> Dict[str, Any]:
    return {
        "schema_version": "sstk_check_v2",
        "binary_targets": list(BIN_TARGET_KEYS),
        "ordinal_targets": {key: value for key, value in ORDINAL_LABEL_MAPS.items()},
        "regression_targets": list(REG_TARGET_KEYS),
        "regression_applicable_key": dict(REG_APPLICABLE_KEY),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build training labels from teacher score jsonl")
    parser.add_argument("--teacher_scores_jsonl", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--image_root")
    parser.add_argument("--softmax_tau", type=float, default=0.2)
    parser.add_argument("--listwise_top_pos", type=int, default=5)
    parser.add_argument("--listwise_neg", type=int, default=15)
    parser.add_argument("--pair_margin_min", type=float, default=0.01)
    parser.add_argument("--hard_negative_rank_pct_max", type=float, default=0.2)
    parser.add_argument("--near_margin_max", type=float, default=0.05)
    parser.add_argument("--max_hard_pairs", type=int, default=4)
    parser.add_argument("--max_near_pairs", type=int, default=4)
    parser.add_argument("--report_examples", type=int, default=8)
    parser.add_argument("--strict_validation", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    teacher_scores_jsonl = Path(args.teacher_scores_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_context = infer_dataset_context(out_dir, teacher_scores_jsonl)

    teacher_records = load_teacher_records(teacher_scores_jsonl)
    subject_mode_vocab = build_subject_mode_vocab(teacher_records)
    image_root = resolve_default_image_root(teacher_scores_jsonl, args.image_root)
    datasets = build_training_datasets(
        teacher_records=teacher_records,
        softmax_tau=float(args.softmax_tau),
        listwise_top_pos=max(1, int(args.listwise_top_pos)),
        listwise_neg=max(1, int(args.listwise_neg)),
        pair_margin_min=max(0.0, float(args.pair_margin_min)),
        hard_negative_rank_pct_max=clamp(float(args.hard_negative_rank_pct_max), 0.0, 1.0),
        near_margin_max=max(0.0, float(args.near_margin_max)),
        max_hard_pairs=max(0, int(args.max_hard_pairs)),
        max_near_pairs=max(0, int(args.max_near_pairs)),
        subject_mode_to_id=subject_mode_vocab,
        image_root=image_root,
    )
    qa_summary = build_qa_summary(datasets)
    validation_summary = validate_datasets(datasets)
    label_guide = collect_label_guide_summary(
        datasets["conditional_detr_canonical"],
        datasets["conditional_detr_batch"],
    )
    label_guide["dataset_label"] = dataset_context["dataset_label"]
    label_guide["run_label"] = out_dir.name
    gaic_like_summary = build_gaic_like_report_summary(datasets["conditional_detr_batch"])
    examples = build_report_examples(
        datasets["conditional_detr_canonical"],
        datasets["conditional_detr_batch"],
        out_dir=out_dir,
        max_examples=args.report_examples,
    )

    counts = {
        "train_pairwise.jsonl": write_jsonl(out_dir / "train_pairwise.jsonl", datasets["pairwise"]),
        "train_listwise.jsonl": write_jsonl(out_dir / "train_listwise.jsonl", datasets["listwise"]),
        "train_decision.jsonl": write_jsonl(out_dir / "train_decision.jsonl", datasets["decision"]),
        "train_checklist.jsonl": write_jsonl(out_dir / "train_checklist.jsonl", datasets["checklist"]),
        "train_regression.jsonl": write_jsonl(out_dir / "train_regression.jsonl", datasets["regression"]),
        "train_conditional_detr_canonical.jsonl": write_jsonl(
            out_dir / "train_conditional_detr_canonical.jsonl", datasets["conditional_detr_canonical"]
        ),
        "train_conditional_detr_batch.jsonl": write_jsonl(
            out_dir / "train_conditional_detr_batch.jsonl", datasets["conditional_detr_batch"]
        ),
        "train_conditional_detr_skipped.jsonl": write_jsonl(
            out_dir / "train_conditional_detr_skipped.jsonl", datasets["conditional_detr_skipped"]
        ),
        "report_examples": len(examples),
    }

    (out_dir / "qa_summary.json").write_text(json.dumps(qa_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "validation_summary.json").write_text(
        json.dumps(validation_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "generation_counts.json").write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "subject_mode_vocab.json").write_text(
        json.dumps(subject_mode_vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "checklist_target_schema.json").write_text(
        json.dumps(checklist_target_schema(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_text = build_markdown_report(
        teacher_scores_jsonl=teacher_scores_jsonl,
        out_dir=out_dir,
        image_root=image_root,
        qa_summary=qa_summary,
        validation_summary=validation_summary,
        examples=examples,
        label_guide=label_guide,
        gaic_like_summary=gaic_like_summary,
        dataset_context=dataset_context,
    )
    (out_dir / "TRAINING_DATA_REPORT_KO.md").write_text(report_text, encoding="utf-8")

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "counts": counts,
                "qa_summary_path": str(out_dir / "qa_summary.json"),
                "validation_summary_path": str(out_dir / "validation_summary.json"),
                "status": validation_summary["status"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if int(args.strict_validation) and validation_summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
