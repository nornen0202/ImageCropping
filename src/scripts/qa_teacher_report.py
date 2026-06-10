#!/usr/bin/env python3
"""
Teacher scorer QA report generator.

Reads teacher_scores jsonl and builds global/per-AR QA statistics for routing and
top1 crop quality checks.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import pandas as pd


CHECKLIST_LABEL_KEYS = [
    "subject_coverage",
    "subject_scale",
    "face_cut",
    "joint_cut",
    "text_keep_ratio",
    "third_dist",
    "phi_dist",
    "center_dist",
    "headroom",
    "lookroom",
    "horizon",
    "context",
    "copyspace",
    "roll",
    "teacher_consensus",
    "ar",
    "crop_tightness",
]


def make_checklist_label_counter_map() -> Dict[str, Counter]:
    return {k: Counter() for k in CHECKLIST_LABEL_KEYS}


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not math.isfinite(x):
        return float(default)
    return x


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(float(v) for v in values)
    if len(arr) == 1:
        return arr[0]
    q = max(0.0, min(1.0, float(q)))
    pos = q * (len(arr) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return arr[lo]
    alpha = pos - lo
    return (1.0 - alpha) * arr[lo] + alpha * arr[hi]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate QA report from teacher_scores jsonl")
    p.add_argument("--teacher_scores_jsonl", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--output_by_ar_csv", default="")
    p.add_argument("--negative_score_thr", type=float, default=0.0)
    p.add_argument("--subject_coverage_fail_thr", type=float, default=0.9)
    p.add_argument("--copyspace_keep_thr", type=float, default=0.25)
    p.add_argument("--teacher_override_iou_thr", type=float, default=0.60)
    return p.parse_args()


def make_bucket() -> Dict[str, Any]:
    return {
        "count": 0,
        "decision": Counter(),
        "final_scores": [],
        "delta_improve": [],
        "num_input_candidates": [],
        "checklist_present_count": 0,
        "checklist_label_counts": make_checklist_label_counter_map(),
        "why_text_present_count": 0,
        "why_tag_count": [],
        "why_tag_vocab": Counter(),
        "subject_mode": Counter(),
        "policy_id": Counter(),
        "subject_mode_conf": [],
        "subject_mode_conflict_count": 0,
        "num_person": [],
        "c2_num_inst": [],
        "num_effective_subjects": [],
        "primary_subject_exists_count": 0,
        "c2_primary_bg_like_count": 0,
        "multi_subject_count": 0,
        "union_used_count": 0,
        "neg_final_count": 0,
        "face_cut_count": 0,
        "head_top_cut_count": 0,
        "joint_cut_count": 0,
        "subject_cov_fail_count": 0,
        "subject_cut_risk_count": 0,
        "text_fail_count": 0,
        "text_keep_ratio": [],
        "text_penalty": [],
        "text_available_count": 0,
        "copyspace_subset_count": 0,
        "copyspace_preserve_count": 0,
        "portrait_route_count": 0,
        "portrait_no_human_count": 0,
        "group_no_human_count": 0,
        "scene_mode_count": 0,
        "scene_horizon_na_count": 0,
        "needs_leveling_count": 0,
        "roll_abs": [],
        "scene_subtype": Counter(),
        "copyspace_mode_fired_by": Counter(),
        "text_document_ocr_unavailable_count": 0,
        "expensive_source": Counter(),
        "fallback_activated_count": 0,
        "fallback_mode": Counter(),
        "norm_size_source": Counter(),
        "router_rule_id": Counter(),
        "guard_no_person_for_portrait_count": 0,
        "guard_no_text_signal_count": 0,
        "guard_low_blank_ratio_copyspace_count": 0,
        "proposal_injected_count": 0,
        "teacher_seed_top1_count": 0,
        "teacher_seed_selected_any_count": 0,
        "proposal_teacher_seed_top1_count": 0,
        "proposal_teacher_seed_selected_any_count": 0,
        "teacher_seed_top1_risk_count": 0,
        "proposal_teacher_seed_top1_risk_count": 0,
        "teacher_consensus_available_count": 0,
        "teacher_consensus_count": 0,
        "teacher_disagreement_count": 0,
        "teacher_override_count": 0,
        "teacher_tau_boost_count": 0,
    }


def update_bucket(
    bucket: Dict[str, Any],
    *,
    rec: Dict[str, Any],
    ar_res: Dict[str, Any],
    route_global: Dict[str, Any],
    negative_score_thr: float,
    subject_coverage_fail_thr: float,
    copyspace_keep_thr: float,
    teacher_override_iou_thr: float,
) -> None:
    bucket["count"] += 1

    proposal_injected = bool(
        rec.get(
            "proposal_injected",
            (rec.get("candidate_meta", {}) or {}).get("proposal_injected", False),
        )
    )
    if proposal_injected:
        bucket["proposal_injected_count"] += 1

    proposal_info = ar_res.get("proposal_injection", {}) if isinstance(ar_res.get("proposal_injection"), dict) else {}
    teacher_cons = proposal_info.get("teacher_consensus", {}) if isinstance(proposal_info.get("teacher_consensus"), dict) else {}
    consensus_available = bool(teacher_cons.get("available", False))
    consensus = bool(teacher_cons.get("consensus", False))
    top1_iou_to_teacher = safe_float(proposal_info.get("candidate_top1_iou_to_teacher_seed", 0.0), 0.0)
    if consensus_available:
        bucket["teacher_consensus_available_count"] += 1
        if consensus:
            bucket["teacher_consensus_count"] += 1
        else:
            bucket["teacher_disagreement_count"] += 1
    if consensus and (top1_iou_to_teacher < float(teacher_override_iou_thr)):
        bucket["teacher_override_count"] += 1
    if consensus and bool(teacher_cons.get("tau_boost_applied", False)):
        bucket["teacher_tau_boost_count"] += 1

    decision = ar_res.get("decision", {}) if isinstance(ar_res.get("decision"), dict) else {}
    decision_type = str(decision.get("decision_type", "unknown"))
    delta = safe_float(decision.get("delta_improve", 0.0))
    bucket["decision"][decision_type] += 1
    bucket["delta_improve"].append(delta)
    bucket["num_input_candidates"].append(int(safe_float(ar_res.get("num_input_candidates", 0), 0.0)))

    topk = ar_res.get("selected_topk", [])
    if not isinstance(topk, list) or not topk:
        return
    top1 = topk[0] if isinstance(topk[0], dict) else {}

    why_tags = top1.get("why_tags", [])
    if isinstance(why_tags, list):
        bucket["why_tag_count"].append(int(len(why_tags)))
        for t in why_tags:
            tt = str(t).strip()
            if tt:
                bucket["why_tag_vocab"][tt] += 1
    why_text_template = str(top1.get("why_text_template", "")).strip()
    if why_text_template:
        bucket["why_text_present_count"] += 1

    checklist = top1.get("checklist", {})
    if isinstance(checklist, dict) and bool(checklist):
        bucket["checklist_present_count"] += 1
        label_counts = bucket.get("checklist_label_counts", {})
        if isinstance(label_counts, dict):
            for k in CHECKLIST_LABEL_KEYS:
                it = checklist.get(k, {})
                if not isinstance(it, dict):
                    continue
                label = str(it.get("label", "")).strip()
                if not label:
                    continue
                if not isinstance(label_counts.get(k), Counter):
                    label_counts[k] = Counter()
                label_counts[k][label] += 1

    top1_source = str(top1.get("source", ""))
    teacher_seed_top1 = top1_source.startswith("teacher:")
    if teacher_seed_top1:
        bucket["teacher_seed_top1_count"] += 1
        if proposal_injected:
            bucket["proposal_teacher_seed_top1_count"] += 1

    teacher_seed_selected_any = any(
        isinstance(c, dict) and str(c.get("source", "")).startswith("teacher:")
        for c in topk
    )
    if teacher_seed_selected_any:
        bucket["teacher_seed_selected_any_count"] += 1
        if proposal_injected:
            bucket["proposal_teacher_seed_selected_any_count"] += 1

    score_final = safe_float(top1.get("scores", {}).get("final", 0.0))
    bucket["final_scores"].append(score_final)
    if score_final < float(negative_score_thr):
        bucket["neg_final_count"] += 1

    flags = top1.get("flags", {}) if isinstance(top1.get("flags"), dict) else {}
    face_cut = bool(flags.get("face_cut", False))
    head_top_cut = bool(flags.get("head_top_cut", False))
    joint_cut = safe_float(flags.get("joint_cutoff_score", 0.0)) > 0.35
    subj_cov = safe_float(flags.get("subject_coverage", 1.0))
    subj_cov_fail = subj_cov < float(subject_coverage_fail_thr)

    if face_cut:
        bucket["face_cut_count"] += 1
    if head_top_cut:
        bucket["head_top_cut_count"] += 1
    if joint_cut:
        bucket["joint_cut_count"] += 1
    if subj_cov_fail:
        bucket["subject_cov_fail_count"] += 1
    if face_cut or head_top_cut or joint_cut or subj_cov_fail:
        bucket["subject_cut_risk_count"] += 1
    if teacher_seed_top1 and (face_cut or head_top_cut or joint_cut or subj_cov_fail):
        bucket["teacher_seed_top1_risk_count"] += 1
        if proposal_injected:
            bucket["proposal_teacher_seed_top1_risk_count"] += 1

    comps = top1.get("scores", {}).get("components", {})
    if isinstance(comps, dict):
        src = str(comps.get("expensive_source", "unknown"))
        bucket["expensive_source"][src] += 1
        bucket["text_penalty"].append(safe_float(comps.get("p_text", 0.0), 0.0))

    text_check = top1.get("composition_checks", {}).get("text", {})
    if isinstance(text_check, dict):
        if bool(text_check.get("available", False)):
            bucket["text_available_count"] += 1
        bucket["text_keep_ratio"].append(safe_float(text_check.get("text_keep_ratio", 1.0), 1.0))
        if not bool(text_check.get("pass", True)):
            bucket["text_fail_count"] += 1

    routing = {}
    if isinstance(route_global, dict):
        routing.update(route_global)
    if isinstance(ar_res.get("routing"), dict):
        routing.update(ar_res.get("routing"))
    flags_route = routing.get("flags", {}) if isinstance(routing.get("flags"), dict) else {}
    subject_mode = str(routing.get("subject_mode", route_global.get("subject_mode", "other_ambiguous")))
    policy_id = str(routing.get("policy_id", route_global.get("policy_id", "generic_v1")))
    subject_mode_conf = safe_float(routing.get("subject_mode_conf", route_global.get("subject_mode_conf", 0.0)))
    subject_mode_conflict = bool(routing.get("subject_mode_conflict", route_global.get("subject_mode_conflict", False)))
    subject_set = routing.get("subject_set", {}) if isinstance(routing.get("subject_set"), dict) else {}
    router_signals = routing.get("router_signals", {}) if isinstance(routing.get("router_signals"), dict) else {}
    rule_id = str(routing.get("router_rule_id", "")).strip()
    scene_subtype = str(routing.get("scene_subtype", ""))
    if rule_id:
        bucket["router_rule_id"][rule_id] += 1
    if scene_subtype and subject_mode.startswith("scene"):
        bucket["scene_subtype"][scene_subtype] += 1

    bucket["subject_mode"][subject_mode] += 1
    bucket["policy_id"][policy_id] += 1
    bucket["subject_mode_conf"].append(subject_mode_conf)
    if subject_mode_conflict:
        bucket["subject_mode_conflict_count"] += 1
    signal_num_person = int(
        safe_float(
            router_signals.get(
                "num_person",
                subject_set.get("num_person", route_global.get("num_people", 0)),
            ),
            0.0,
        )
    )
    c2_num_inst = int(
        safe_float(
            subject_set.get("num_c2_instances", subject_set.get("num_subject_inst", 0)),
            0.0,
        )
    )
    num_effective_subjects = int(
        safe_float(
            subject_set.get("num_effective_subjects", c2_num_inst),
            0.0,
        )
    )
    primary_subject_exists = bool(subject_set.get("primary_subject_exists", False))
    if not primary_subject_exists:
        primary_subject_exists = int(safe_float(subject_set.get("primary_idx", -1), -1.0)) >= 0
    bucket["num_person"].append(signal_num_person)
    bucket["c2_num_inst"].append(c2_num_inst)
    bucket["num_effective_subjects"].append(num_effective_subjects)
    if primary_subject_exists:
        bucket["primary_subject_exists_count"] += 1
    if bool(subject_set.get("c2_primary_bg_like", False)):
        bucket["c2_primary_bg_like_count"] += 1
    if bool(subject_set.get("multi_subject", False)):
        bucket["multi_subject_count"] += 1

    # Guard consistency diagnostics
    signal_has_text_hint = bool(router_signals.get("has_text_hint", False))
    signal_text_overlay = bool(router_signals.get("text_overlay_likely", False))
    signal_ocr_boxes = int(safe_float(router_signals.get("ocr_text_boxes", 0), 0.0))
    signal_ocr_available = bool(router_signals.get("ocr_available", False))
    signal_blank_ratio = safe_float(router_signals.get("blank_ratio_est", 0.0), 0.0)
    signal_blank_thr = safe_float(router_signals.get("blank_ratio_thr", 0.28), 0.28)
    signal_copy_tag = bool(router_signals.get("copyspace_tag_signal", router_signals.get("has_copyspace_tag", False)))
    signal_copy_fired_by = str(router_signals.get("copyspace_mode_fired_by", "none")).strip()
    if signal_copy_fired_by:
        bucket["copyspace_mode_fired_by"][signal_copy_fired_by] += 1

    if subject_mode.startswith("portrait") and signal_num_person <= 0:
        bucket["guard_no_person_for_portrait_count"] += 1
    if (
        subject_mode == "text_document"
        and signal_ocr_boxes <= 0
        and (not signal_text_overlay)
        and (not signal_has_text_hint)
    ):
        bucket["guard_no_text_signal_count"] += 1
    if subject_mode == "text_document" and (not signal_ocr_available):
        bucket["text_document_ocr_unavailable_count"] += 1
    if subject_mode == "background_texture_copyspace" and signal_copy_tag and signal_blank_ratio < signal_blank_thr:
        bucket["guard_low_blank_ratio_copyspace_count"] += 1

    subj_prior = rec.get("subject_prior", {}) if isinstance(rec.get("subject_prior"), dict) else {}
    if bool(subj_prior.get("union_used", False)):
        bucket["union_used_count"] += 1

    shot_type = str(routing.get("shot_type", "unknown"))
    norm_size_source = str(
        routing.get(
            "norm_size_source",
            route_global.get("norm_size_source", "unknown"),
        )
    )
    bucket["norm_size_source"][norm_size_source] += 1

    fallback = ar_res.get("fallback", {}) if isinstance(ar_res.get("fallback"), dict) else {}
    if bool(fallback.get("activated", False)):
        bucket["fallback_activated_count"] += 1
        bucket["fallback_mode"][str(fallback.get("mode", "unknown"))] += 1

    has_copyspace = bool(flags_route.get("has_copy_space", False))
    has_human_evidence = (
        signal_num_person > 0
        or int(safe_float(route_global.get("num_people", 0), 0.0)) > 0
    )

    if has_copyspace:
        bucket["copyspace_subset_count"] += 1
        copyspace = top1.get("composition_checks", {}).get("copyspace", {})
        if isinstance(copyspace, dict) and safe_float(copyspace.get("blank_ratio_keep", 0.0), 0.0) >= float(copyspace_keep_thr):
            bucket["copyspace_preserve_count"] += 1

    if shot_type in {"headshot", "half", "full", "group"}:
        bucket["portrait_route_count"] += 1
        if not has_human_evidence:
            bucket["portrait_no_human_count"] += 1
    if shot_type == "group" and not has_human_evidence:
        bucket["group_no_human_count"] += 1
    if subject_mode.startswith("scene"):
        bucket["scene_mode_count"] += 1
        horizon_check = checklist.get("horizon", {}) if isinstance(checklist, dict) else {}
        if str(horizon_check.get("label", "")).strip() in {"horizon_na", ""}:
            bucket["scene_horizon_na_count"] += 1
    roll_check = top1.get("composition_checks", {}).get("roll", {})
    if isinstance(roll_check, dict):
        roll_val = roll_check.get("value")
        if roll_val is not None:
            bucket["roll_abs"].append(abs(safe_float(roll_val, 0.0)))
        if bool(roll_check.get("needs_leveling", False)):
            bucket["needs_leveling_count"] += 1


def summarize_bucket(bucket: Dict[str, Any]) -> Dict[str, Any]:
    n = max(1, int(bucket["count"]))
    finals = [float(v) for v in bucket["final_scores"]]
    deltas = [float(v) for v in bucket["delta_improve"]]
    num_input = [int(v) for v in bucket["num_input_candidates"]]
    sm_conf = [float(v) for v in bucket["subject_mode_conf"]]
    num_person = [int(v) for v in bucket["num_person"]]
    c2_num_inst = [int(v) for v in bucket["c2_num_inst"]]
    num_effective_subjects = [int(v) for v in bucket["num_effective_subjects"]]
    proposal_n = int(bucket["proposal_injected_count"])
    consensus_n = int(bucket["teacher_consensus_count"])
    consensus_avail_n = int(bucket["teacher_consensus_available_count"])
    text_keep = [float(v) for v in bucket["text_keep_ratio"]]
    text_penalty = [float(v) for v in bucket["text_penalty"]]
    roll_abs = [float(v) for v in bucket.get("roll_abs", [])]
    why_tag_count = [int(v) for v in bucket.get("why_tag_count", [])]
    text_document_count = int(bucket["subject_mode"].get("text_document", 0))
    scene_mode_count = int(bucket["scene_mode_count"])
    checklist_label_counts_raw = (
        bucket.get("checklist_label_counts", {})
        if isinstance(bucket.get("checklist_label_counts"), dict)
        else {}
    )
    checklist_label_counts: Dict[str, Dict[str, int]] = {}
    for key, cnt in checklist_label_counts_raw.items():
        if isinstance(cnt, Counter):
            checklist_label_counts[str(key)] = dict(cnt)
        elif isinstance(cnt, dict):
            checklist_label_counts[str(key)] = {str(k): int(v) for k, v in cnt.items()}

    return {
        "count": int(bucket["count"]),
        "subject_mode_counts": dict(bucket["subject_mode"]),
        "scene_subtype_counts": dict(bucket.get("scene_subtype", Counter())),
        "policy_id_counts": dict(bucket["policy_id"]),
        "explainability": {
            "checklist_present_rate": float(bucket.get("checklist_present_count", 0)) / n,
            "why_text_present_rate": float(bucket.get("why_text_present_count", 0)) / n,
            "why_tag_count_mean": mean(why_tag_count) if why_tag_count else 0.0,
            "why_tag_count_p50": percentile([float(v) for v in why_tag_count], 0.50) if why_tag_count else 0.0,
            "why_tag_count_p90": percentile([float(v) for v in why_tag_count], 0.90) if why_tag_count else 0.0,
            "why_tag_vocab_top20": dict(bucket.get("why_tag_vocab", Counter()).most_common(20)),
            "checklist_label_counts": checklist_label_counts,
        },
        "subject_mode_conf": {
            "mean": mean(sm_conf) if sm_conf else 0.0,
            "p10": percentile(sm_conf, 0.10),
            "p50": percentile(sm_conf, 0.50),
            "p90": percentile(sm_conf, 0.90),
            "conflict_rate": float(bucket["subject_mode_conflict_count"]) / n,
        },
        "candidate_volume": {
            "num_input_mean": mean(num_input) if num_input else 0.0,
            "num_input_p95": percentile([float(v) for v in num_input], 0.95) if num_input else 0.0,
            "num_person_mean": mean(num_person) if num_person else 0.0,
            "num_person_p95": percentile([float(v) for v in num_person], 0.95) if num_person else 0.0,
            "c2_num_inst_mean": mean(c2_num_inst) if c2_num_inst else 0.0,
            "c2_num_inst_p95": percentile([float(v) for v in c2_num_inst], 0.95) if c2_num_inst else 0.0,
            "num_effective_subjects_mean": mean(num_effective_subjects) if num_effective_subjects else 0.0,
            "num_effective_subjects_p95": (
                percentile([float(v) for v in num_effective_subjects], 0.95) if num_effective_subjects else 0.0
            ),
        },
        "subject_mode_kpi": {
            "bg_selected_rate": float(bucket["c2_primary_bg_like_count"]) / n,
            "multi_subject_detect_rate": float(bucket["multi_subject_count"]) / n,
            "union_used_rate": float(bucket["union_used_count"]) / n,
            "primary_subject_exists_rate": float(bucket["primary_subject_exists_count"]) / n,
            "guard_no_person_for_portrait_rate": float(bucket["guard_no_person_for_portrait_count"]) / n,
            "guard_no_text_signal_rate": float(bucket["guard_no_text_signal_count"]) / n,
            "guard_low_blank_ratio_copyspace_rate": float(bucket["guard_low_blank_ratio_copyspace_count"]) / n,
        },
        "proposal_injection": {
            "proposal_injected_rate": float(proposal_n) / n,
            "teacher_seed_top1_rate_all": float(bucket["teacher_seed_top1_count"]) / n,
            "teacher_seed_selected_any_rate_all": float(bucket["teacher_seed_selected_any_count"]) / n,
            "teacher_seed_top1_risk_rate_all": float(bucket["teacher_seed_top1_risk_count"]) / n,
            "teacher_seed_top1_risk_rate_given_seed_top1": (
                float(bucket["teacher_seed_top1_risk_count"]) / max(1, int(bucket["teacher_seed_top1_count"]))
            ),
            "teacher_seed_top1_win_rate_given_proposal": (
                float(bucket["proposal_teacher_seed_top1_count"]) / max(1, proposal_n)
            ),
            "teacher_seed_selected_any_rate_given_proposal": (
                float(bucket["proposal_teacher_seed_selected_any_count"]) / max(1, proposal_n)
            ),
            "proposal_teacher_seed_top1_risk_rate_given_proposal": (
                float(bucket["proposal_teacher_seed_top1_risk_count"]) / max(1, proposal_n)
            ),
            "proposal_teacher_seed_top1_risk_rate_given_seed_top1": (
                float(bucket["proposal_teacher_seed_top1_risk_count"])
                / max(1, int(bucket["proposal_teacher_seed_top1_count"]))
            ),
            # Practical proxy for "proposal rescue": injected task where teacher-seed is selected top1.
            "proposal_rescue_rate_proxy": float(bucket["proposal_teacher_seed_top1_count"]) / max(1, proposal_n),
            "teacher_consensus_rate": float(bucket["teacher_consensus_count"]) / max(1, consensus_avail_n),
            "teacher_disagreement_rate": float(bucket["teacher_disagreement_count"]) / max(1, consensus_avail_n),
            "teacher_override_rate_given_consensus": float(bucket["teacher_override_count"]) / max(1, consensus_n),
            "teacher_override_rate_all": float(bucket["teacher_override_count"]) / n,
            "teacher_tau_boost_rate_given_consensus": float(bucket["teacher_tau_boost_count"]) / max(1, consensus_n),
            "teacher_consensus_available_rate": float(consensus_avail_n) / n,
        },
        "decision_counts": dict(bucket["decision"]),
        "decision_rates": {k: float(v) / n for k, v in bucket["decision"].items()},
        "final_score": {
            "mean": mean(finals) if finals else 0.0,
            "min": min(finals) if finals else 0.0,
            "max": max(finals) if finals else 0.0,
            "p50": percentile(finals, 0.50),
            "p90": percentile(finals, 0.90),
            "neg_rate": float(bucket["neg_final_count"]) / n,
        },
        "delta_improve": {
            "mean": mean(deltas) if deltas else 0.0,
            "p50": percentile(deltas, 0.50),
            "p90": percentile(deltas, 0.90),
        },
        "risk_rates": {
            "face_cut_rate": float(bucket["face_cut_count"]) / n,
            "head_top_cut_rate": float(bucket["head_top_cut_count"]) / n,
            "joint_cut_rate": float(bucket["joint_cut_count"]) / n,
            "subject_coverage_fail_rate": float(bucket["subject_cov_fail_count"]) / n,
            "subject_cut_risk_rate": float(bucket["subject_cut_risk_count"]) / n,
            "text_fail_rate": float(bucket["text_fail_count"]) / n,
            "text_available_rate": float(bucket["text_available_count"]) / n,
        },
        "text_quality": {
            "text_keep_ratio_mean": mean(text_keep) if text_keep else 1.0,
            "text_keep_ratio_p10": percentile(text_keep, 0.10),
            "text_keep_ratio_p50": percentile(text_keep, 0.50),
            "text_keep_ratio_p90": percentile(text_keep, 0.90),
            "text_penalty_mean": mean(text_penalty) if text_penalty else 0.0,
            "text_penalty_p90": percentile(text_penalty, 0.90),
        },
        "routing_consistency": {
            "portrait_route_rate": float(bucket["portrait_route_count"]) / n,
            "portrait_route_without_human_rate": float(bucket["portrait_no_human_count"]) / max(
                1, int(bucket["portrait_route_count"])
            ),
            "group_route_without_human_rate": float(bucket["group_no_human_count"]) / n,
        },
        "p0_release_gates": {
            "head_top_cut_rate": float(bucket["head_top_cut_count"]) / n,
            "face_cut_rate": float(bucket["face_cut_count"]) / n,
            "joint_cut_rate": float(bucket["joint_cut_count"]) / n,
            "text_document_ocr_unavailable_rate": (
                float(bucket["text_document_ocr_unavailable_count"]) / max(1, text_document_count)
            ),
            "portrait_route_without_human_rate": float(bucket["portrait_no_human_count"]) / max(
                1, int(bucket["portrait_route_count"])
            ),
            "scene_horizon_na_rate": float(bucket["scene_horizon_na_count"]) / max(1, scene_mode_count),
        },
        "copyspace": {
            "subset_count": int(bucket["copyspace_subset_count"]),
            "preserve_rate": float(bucket["copyspace_preserve_count"]) / max(
                1, int(bucket["copyspace_subset_count"])
            ),
            "mode_fired_by_counts": dict(bucket.get("copyspace_mode_fired_by", Counter())),
        },
        "geometry_qa": {
            "needs_leveling_rate": float(bucket.get("needs_leveling_count", 0)) / n,
            "roll_abs_p50": percentile(roll_abs, 0.50),
            "roll_abs_p90": percentile(roll_abs, 0.90),
        },
        "expensive_source_counts": dict(bucket["expensive_source"]),
        "fallback": {
            "activated_rate": float(bucket["fallback_activated_count"]) / n,
            "mode_counts": dict(bucket["fallback_mode"]),
        },
        "norm_size_source_counts": dict(bucket["norm_size_source"]),
        "router_rule_id_counts": dict(bucket["router_rule_id"]),
    }


def flatten_for_csv(ar: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    out = {
        "target_ar": ar,
        "count": summary.get("count", 0),
        "subject_mode_top1": (
            sorted(summary.get("subject_mode_counts", {}).items(), key=lambda kv: kv[1], reverse=True)[0][0]
            if summary.get("subject_mode_counts")
            else "unknown"
        ),
        "keep_full_rate": summary.get("decision_rates", {}).get("keep_full", 0.0),
        "minimal_crop_rate": summary.get("decision_rates", {}).get("minimal_crop", 0.0),
        "crop_rate": summary.get("decision_rates", {}).get("crop", 0.0),
        "candidate_input_mean": summary.get("candidate_volume", {}).get("num_input_mean", 0.0),
        "c2_num_inst_mean": summary.get("candidate_volume", {}).get("c2_num_inst_mean", 0.0),
        "bg_selected_rate": summary.get("subject_mode_kpi", {}).get("bg_selected_rate", 0.0),
        "multi_subject_detect_rate": summary.get("subject_mode_kpi", {}).get("multi_subject_detect_rate", 0.0),
        "union_used_rate": summary.get("subject_mode_kpi", {}).get("union_used_rate", 0.0),
        "subject_mode_conflict_rate": summary.get("subject_mode_conf", {}).get("conflict_rate", 0.0),
        "proposal_injected_rate": summary.get("proposal_injection", {}).get("proposal_injected_rate", 0.0),
        "teacher_seed_top1_rate_all": summary.get("proposal_injection", {}).get("teacher_seed_top1_rate_all", 0.0),
        "teacher_seed_top1_risk_rate_all": summary.get("proposal_injection", {}).get(
            "teacher_seed_top1_risk_rate_all", 0.0
        ),
        "teacher_consensus_rate": summary.get("proposal_injection", {}).get("teacher_consensus_rate", 0.0),
        "teacher_disagreement_rate": summary.get("proposal_injection", {}).get("teacher_disagreement_rate", 0.0),
        "teacher_override_rate_given_consensus": summary.get("proposal_injection", {}).get(
            "teacher_override_rate_given_consensus", 0.0
        ),
        "proposal_rescue_rate_proxy": summary.get("proposal_injection", {}).get("proposal_rescue_rate_proxy", 0.0),
        "final_mean": summary.get("final_score", {}).get("mean", 0.0),
        "final_p50": summary.get("final_score", {}).get("p50", 0.0),
        "final_p90": summary.get("final_score", {}).get("p90", 0.0),
        "final_neg_rate": summary.get("final_score", {}).get("neg_rate", 0.0),
        "checklist_present_rate": summary.get("explainability", {}).get("checklist_present_rate", 0.0),
        "why_text_present_rate": summary.get("explainability", {}).get("why_text_present_rate", 0.0),
        "why_tag_count_mean": summary.get("explainability", {}).get("why_tag_count_mean", 0.0),
        "delta_mean": summary.get("delta_improve", {}).get("mean", 0.0),
        "face_cut_rate": summary.get("risk_rates", {}).get("face_cut_rate", 0.0),
        "head_top_cut_rate": summary.get("risk_rates", {}).get("head_top_cut_rate", 0.0),
        "joint_cut_rate": summary.get("risk_rates", {}).get("joint_cut_rate", 0.0),
        "subject_cov_fail_rate": summary.get("risk_rates", {}).get("subject_coverage_fail_rate", 0.0),
        "subject_cut_risk_rate": summary.get("risk_rates", {}).get("subject_cut_risk_rate", 0.0),
        "portrait_route_rate": summary.get("routing_consistency", {}).get("portrait_route_rate", 0.0),
        "portrait_no_human_rate": summary.get("routing_consistency", {}).get(
            "portrait_route_without_human_rate", 0.0
        ),
        "group_no_human_rate": summary.get("routing_consistency", {}).get(
            "group_route_without_human_rate", 0.0
        ),
        "copyspace_subset_count": summary.get("copyspace", {}).get("subset_count", 0),
        "copyspace_preserve_rate": summary.get("copyspace", {}).get("preserve_rate", 0.0),
        "fallback_activated_rate": summary.get("fallback", {}).get("activated_rate", 0.0),
    }
    return out


def build_mode_qa_summary(by_subject_mode: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for mode, summary in sorted(by_subject_mode.items()):
        out[str(mode)] = {
            "count": int(summary.get("count", 0)),
            "num_person_mean": summary.get("candidate_volume", {}).get("num_person_mean", 0.0),
            "num_c2_instances_mean": summary.get("candidate_volume", {}).get("c2_num_inst_mean", 0.0),
            "num_effective_subjects_mean": summary.get("candidate_volume", {}).get("num_effective_subjects_mean", 0.0),
            "primary_subject_exists_rate": summary.get("subject_mode_kpi", {}).get("primary_subject_exists_rate", 0.0),
            "head_top_cut_rate": summary.get("p0_release_gates", {}).get("head_top_cut_rate", 0.0),
            "face_cut_rate": summary.get("p0_release_gates", {}).get("face_cut_rate", 0.0),
            "joint_cut_rate": summary.get("p0_release_gates", {}).get("joint_cut_rate", 0.0),
            "text_document_ocr_unavailable_rate": (
                summary.get("p0_release_gates", {}).get("text_document_ocr_unavailable_rate", 0.0)
            ),
            "portrait_route_without_human_rate": (
                summary.get("p0_release_gates", {}).get("portrait_route_without_human_rate", 0.0)
            ),
            "scene_horizon_na_rate": summary.get("p0_release_gates", {}).get("scene_horizon_na_rate", 0.0),
        }
    return out


def main() -> None:
    args = parse_args()

    in_path = Path(args.teacher_scores_jsonl)
    out_json = Path(args.output_json)
    out_csv = Path(args.output_by_ar_csv) if args.output_by_ar_csv else None

    if not in_path.exists():
        raise FileNotFoundError(f"teacher_scores_jsonl not found: {in_path}")

    global_bucket = make_bucket()
    per_ar: Dict[str, Dict[str, Any]] = {}
    per_mode: Dict[str, Dict[str, Any]] = {}
    per_mode_ar: Dict[str, Dict[str, Any]] = {}
    per_mode_shot_ar: Dict[str, Dict[str, Any]] = {}
    image_ids = set()
    total_ar_results = 0

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            image_id = str(rec.get("image_id", ""))
            if image_id:
                image_ids.add(image_id)

            route_global = rec.get("route_global", {}) if isinstance(rec.get("route_global"), dict) else {}
            by_ar = (
                rec.get("teacher_scorer", {}).get("results_by_ar", {})
                if isinstance(rec.get("teacher_scorer"), dict)
                else {}
            )
            if not isinstance(by_ar, dict):
                continue

            for ar_text, ar_res in by_ar.items():
                if not isinstance(ar_res, dict):
                    continue
                total_ar_results += 1
                if ar_text not in per_ar:
                    per_ar[ar_text] = make_bucket()
                ar_routing = ar_res.get("routing", {}) if isinstance(ar_res.get("routing"), dict) else {}
                mode_key = str(ar_routing.get("subject_mode", route_global.get("subject_mode", "other_ambiguous")))
                shot_key = str(ar_routing.get("shot_type", route_global.get("shot_type", "unknown")))
                mode_ar_key = f"{mode_key}__{ar_text}"
                mode_shot_ar_key = f"{mode_key}__{shot_key}__{ar_text}"
                if mode_key not in per_mode:
                    per_mode[mode_key] = make_bucket()
                if mode_ar_key not in per_mode_ar:
                    per_mode_ar[mode_ar_key] = make_bucket()
                if mode_shot_ar_key not in per_mode_shot_ar:
                    per_mode_shot_ar[mode_shot_ar_key] = make_bucket()

                update_bucket(
                    global_bucket,
                    rec=rec,
                    ar_res=ar_res,
                    route_global=route_global,
                    negative_score_thr=float(args.negative_score_thr),
                    subject_coverage_fail_thr=float(args.subject_coverage_fail_thr),
                    copyspace_keep_thr=float(args.copyspace_keep_thr),
                    teacher_override_iou_thr=float(args.teacher_override_iou_thr),
                )
                update_bucket(
                    per_ar[ar_text],
                    rec=rec,
                    ar_res=ar_res,
                    route_global=route_global,
                    negative_score_thr=float(args.negative_score_thr),
                    subject_coverage_fail_thr=float(args.subject_coverage_fail_thr),
                    copyspace_keep_thr=float(args.copyspace_keep_thr),
                    teacher_override_iou_thr=float(args.teacher_override_iou_thr),
                )
                update_bucket(
                    per_mode[mode_key],
                    rec=rec,
                    ar_res=ar_res,
                    route_global=route_global,
                    negative_score_thr=float(args.negative_score_thr),
                    subject_coverage_fail_thr=float(args.subject_coverage_fail_thr),
                    copyspace_keep_thr=float(args.copyspace_keep_thr),
                    teacher_override_iou_thr=float(args.teacher_override_iou_thr),
                )
                update_bucket(
                    per_mode_ar[mode_ar_key],
                    rec=rec,
                    ar_res=ar_res,
                    route_global=route_global,
                    negative_score_thr=float(args.negative_score_thr),
                    subject_coverage_fail_thr=float(args.subject_coverage_fail_thr),
                    copyspace_keep_thr=float(args.copyspace_keep_thr),
                    teacher_override_iou_thr=float(args.teacher_override_iou_thr),
                )
                update_bucket(
                    per_mode_shot_ar[mode_shot_ar_key],
                    rec=rec,
                    ar_res=ar_res,
                    route_global=route_global,
                    negative_score_thr=float(args.negative_score_thr),
                    subject_coverage_fail_thr=float(args.subject_coverage_fail_thr),
                    copyspace_keep_thr=float(args.copyspace_keep_thr),
                    teacher_override_iou_thr=float(args.teacher_override_iou_thr),
                )

    global_summary = summarize_bucket(global_bucket)
    by_ar_summary = {k: summarize_bucket(v) for k, v in sorted(per_ar.items())}
    by_subject_mode = {k: summarize_bucket(v) for k, v in sorted(per_mode.items())}
    by_subject_mode_ar = {k: summarize_bucket(v) for k, v in sorted(per_mode_ar.items())}
    by_subject_mode_shot_ar = {k: summarize_bucket(v) for k, v in sorted(per_mode_shot_ar.items())}

    report = {
        "input_jsonl": str(in_path),
        "num_images": len(image_ids),
        "num_ar_results": int(total_ar_results),
        "config": {
            "negative_score_thr": float(args.negative_score_thr),
            "subject_coverage_fail_thr": float(args.subject_coverage_fail_thr),
            "copyspace_keep_thr": float(args.copyspace_keep_thr),
            "teacher_override_iou_thr": float(args.teacher_override_iou_thr),
        },
        "global": global_summary,
        "by_ar": by_ar_summary,
        "mode_qa_summary": build_mode_qa_summary(by_subject_mode),
        "by_subject_mode": by_subject_mode,
        "by_subject_mode_ar": by_subject_mode_ar,
        "by_subject_mode_shot_ar": by_subject_mode_shot_ar,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        rows = [flatten_for_csv(ar, sm) for ar, sm in sorted(by_ar_summary.items())]
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"[done] report_json={out_json}")
    if out_csv is not None:
        print(f"[done] report_by_ar_csv={out_csv}")


if __name__ == "__main__":
    main()
