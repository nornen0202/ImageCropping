from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


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


def summarize_numeric(values: Sequence[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
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


def dedupe_candidates(ar_res: Dict[str, Any]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for key in (
        "cheap_top_m",
        "selected_topk",
        "hard_negatives",
        "also_considered_rejected",
    ):
        for cand in ar_res.get(key, []) or []:
            cid = str(cand.get("candidate_id", ""))
            if not cid or cid in seen:
                continue
            merged.append(copy.deepcopy(cand))
            seen.add(cid)
    for key in ("best_candidate", "baseline_candidate"):
        cand = ar_res.get(key)
        if not isinstance(cand, dict):
            continue
        cid = str(cand.get("candidate_id", ""))
        if not cid or cid in seen:
            continue
        merged.append(copy.deepcopy(cand))
        seen.add(cid)
    return merged


def has_soft_issue(candidate: Dict[str, Any]) -> bool:
    labels = candidate.get("checklist_labels", {}) if isinstance(candidate.get("checklist_labels"), dict) else {}
    for label in labels.values():
        text = str(label).strip().lower()
        if any(token in text for token in SOFT_ISSUE_TOKENS):
            return True
    return False


def is_unsafe_candidate(candidate: Dict[str, Any]) -> bool:
    reject_tags = {str(tag) for tag in (candidate.get("reject_tags") or [])}
    if bool(candidate.get("hard_reject", False)):
        return True
    return len(reject_tags.intersection(SEVERE_REJECT_TAGS)) > 0


def candidate_training_view(candidate: Dict[str, Any]) -> Dict[str, Any]:
    scores = candidate.get("scores", {}) if isinstance(candidate.get("scores"), dict) else {}
    return {
        "candidate_id": str(candidate.get("candidate_id", "")),
        "bbox": [round(safe_float(v), 6) for v in candidate.get("bbox_norm_xyxy", [0, 0, 1, 1])],
        "source": str(candidate.get("source", "")),
        "area_ratio": round(safe_float(candidate.get("area_ratio", 0.0)), 6),
        "score_raw_final": round(safe_float(scores.get("rank", scores.get("final", 0.0))), 6),
        "score_raw_rank": round(safe_float(scores.get("rank", scores.get("final", 0.0))), 6),
        "score_raw_policy": round(safe_float(scores.get("policy", scores.get("final", 0.0))), 6),
        "score_raw_final_legacy": round(safe_float(scores.get("final_legacy", 0.0)), 6),
        "score_rank_pct": round(safe_float(candidate.get("score_rank_pct", 0.0)), 6),
        "score_z_local": round(safe_float(candidate.get("score_z_local", 0.0)), 6),
        "score_softmax_local": round(safe_float(candidate.get("score_softmax_local", 0.0)), 6),
        "pseudo_mos_1to5": round(safe_float(candidate.get("pseudo_mos_1to5", 1.0)), 6),
        "label_type": str(candidate.get("label_type", "")),
        "is_positive_candidate": bool(candidate.get("is_positive_candidate", False)),
        "is_soft_positive": bool(candidate.get("is_soft_positive", False)),
        "is_hard_negative": bool(candidate.get("is_hard_negative", False)),
        "is_unsafe_negative": bool(candidate.get("is_unsafe_negative", False)),
        "macro_scores": candidate.get("macro_scores", {}),
        "checklist_labels": candidate.get("checklist_labels", {}),
        "why_tags": candidate.get("why_tags", []),
        "reject_tags": candidate.get("reject_tags", []),
    }


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
    selected_ids = [str(c.get("candidate_id", "")) for c in (ar_res.get("selected_topk") or [])]
    selected_id_set = {cid for cid in selected_ids if cid}
    top1_id = selected_ids[0] if selected_ids else str((ar_res.get("best_candidate") or {}).get("candidate_id", ""))
    best_id = str((ar_res.get("best_candidate") or {}).get("candidate_id", ""))
    baseline_id = str((ar_res.get("baseline_candidate") or {}).get("candidate_id", ""))
    chosen_id = str((ar_res.get("decision") or {}).get("chosen_candidate_id", ""))
    decision_type = str((ar_res.get("decision") or {}).get("decision_type", ""))

    raw_scores = [safe_float(c.get("scores", {}).get("rank", c.get("scores", {}).get("final", 0.0))) for c in candidates]
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
        out["mode"] = str((ar_res.get("routing") or {}).get("subject_mode", ""))
        out["policy_id"] = str((ar_res.get("routing") or {}).get("policy_id", ""))
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
            out["is_unsafe_negative"]
            and out["score_rank_pct"] <= float(hard_negative_rank_pct_max)
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
            safe_float(cand.get("scores", {}).get("rank", cand.get("scores", {}).get("final", 0.0))),
            safe_float(cand.get("score_softmax_local", 0.0)),
        ),
        reverse=True,
    )
    return annotated


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
    decision = ar_res.get("decision", {}) if isinstance(ar_res.get("decision"), dict) else {}
    mode = str((ar_res.get("routing") or {}).get("subject_mode", ""))
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
        score_a = safe_float(a.get("scores", {}).get("rank", a.get("scores", {}).get("final", 0.0)))
        score_b = safe_float(b.get("scores", {}).get("rank", b.get("scores", {}).get("final", 0.0)))
        margin = score_a - score_b
        if margin < float(pair_margin_min):
            return
        out.append(
            {
                "image_id": image_id,
                "target_ar": target_ar,
                "mode": mode,
                "decision_type": str(decision.get("decision_type", "")),
                "bbox_a": [round(safe_float(v), 6) for v in a.get("bbox_norm_xyxy", [0, 0, 1, 1])],
                "bbox_b": [round(safe_float(v), 6) for v in b.get("bbox_norm_xyxy", [0, 0, 1, 1])],
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
    mode = str((ar_res.get("routing") or {}).get("subject_mode", ""))
    decision = ar_res.get("decision", {}) if isinstance(ar_res.get("decision"), dict) else {}
    positives = [c for c in candidates if bool(c.get("is_positive_candidate", False))]
    hard_negs = [c for c in candidates if str(c.get("label_type", "")) in {"hard_negative", "unsafe_negative"}]
    other_negs = [
        c
        for c in candidates
        if str(c.get("label_type", "")) in {"near_negative", "negative"}
    ]
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
    raw_scores = [safe_float(c.get("scores", {}).get("rank", c.get("scores", {}).get("final", 0.0))) for c in chosen]
    probs = softmax_local(raw_scores, tau=softmax_tau)
    candidate_rows = []
    for cand, prob in zip(chosen, probs):
        row = candidate_training_view(cand)
        row["score_softmax_local"] = round(float(prob), 6)
        candidate_rows.append(row)
    return {
        "image_id": image_id,
        "target_ar": target_ar,
        "decision_type": str(decision.get("decision_type", "")),
        "mode": mode,
        "policy_id": str((ar_res.get("routing") or {}).get("policy_id", "")),
        "candidates": candidate_rows,
    }


def build_decision_record(
    image_id: str,
    target_ar: str,
    ar_res: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    decision = ar_res.get("decision", {}) if isinstance(ar_res.get("decision"), dict) else {}
    baseline = ar_res.get("baseline_candidate", {}) if isinstance(ar_res.get("baseline_candidate"), dict) else {}
    best = ar_res.get("best_candidate", {}) if isinstance(ar_res.get("best_candidate"), dict) else {}
    chosen_id = str(decision.get("chosen_candidate_id", ""))
    chosen = next((c for c in candidates if str(c.get("candidate_id", "")) == chosen_id), None)
    if chosen is None and chosen_id == str(baseline.get("candidate_id", "")):
        chosen = baseline
    if chosen is None and chosen_id == str(best.get("candidate_id", "")):
        chosen = best
    return {
        "image_id": image_id,
        "target_ar": target_ar,
        "decision_type": str(decision.get("decision_type", "")),
        "delta_vs_base": round(safe_float(decision.get("delta_improve", 0.0)), 6),
        "tau_improve": round(safe_float(decision.get("tau_improve", 0.0)), 6),
        "mode": str((ar_res.get("routing") or {}).get("subject_mode", "")),
        "policy_id": str((ar_res.get("routing") or {}).get("policy_id", "")),
        "base_bbox": [round(safe_float(v), 6) for v in baseline.get("bbox_norm_xyxy", [0, 0, 1, 1])],
        "winner_pre_gate_bbox": [round(safe_float(v), 6) for v in best.get("bbox_norm_xyxy", [0, 0, 1, 1])],
        "winner_post_gate_bbox": [round(safe_float(v), 6) for v in (chosen or {}).get("bbox_norm_xyxy", [0, 0, 1, 1])],
        "base_candidate_id": str(baseline.get("candidate_id", "")),
        "winner_pre_gate_candidate_id": str(best.get("candidate_id", "")),
        "winner_post_gate_candidate_id": chosen_id,
        "best_score_rank": round(safe_float((best.get("scores") or {}).get("rank", 0.0)), 6),
        "best_score_policy": round(safe_float((best.get("scores") or {}).get("policy", 0.0)), 6),
        "base_score_rank": round(safe_float((baseline.get("scores") or {}).get("rank", 0.0)), 6),
        "base_score_policy": round(safe_float((baseline.get("scores") or {}).get("policy", 0.0)), 6),
        "chosen_score_rank": round(safe_float(decision.get("chosen_score_rank", 0.0)), 6),
        "chosen_score_policy": round(safe_float(decision.get("chosen_score_policy", 0.0)), 6),
    }


def build_checklist_records(
    image_id: str,
    target_ar: str,
    ar_res: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    mode = str((ar_res.get("routing") or {}).get("subject_mode", ""))
    decision_type = str((ar_res.get("decision") or {}).get("decision_type", ""))
    out = []
    for cand in candidates:
        row = candidate_training_view(cand)
        row.update(
            {
                "image_id": image_id,
                "target_ar": target_ar,
                "mode": mode,
                "decision_type": decision_type,
                "policy_id": str((ar_res.get("routing") or {}).get("policy_id", "")),
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
    mode = str((ar_res.get("routing") or {}).get("subject_mode", ""))
    decision_type = str((ar_res.get("decision") or {}).get("decision_type", ""))
    rows = []
    for cand in candidates:
        rows.append(
            {
                "image_id": image_id,
                "target_ar": target_ar,
                "mode": mode,
                "decision_type": decision_type,
                **candidate_training_view(cand),
            }
        )
    return rows


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
) -> Dict[str, List[Dict[str, Any]]]:
    pairwise_rows: List[Dict[str, Any]] = []
    listwise_rows: List[Dict[str, Any]] = []
    decision_rows: List[Dict[str, Any]] = []
    checklist_rows: List[Dict[str, Any]] = []
    regression_rows: List[Dict[str, Any]] = []

    for rec in teacher_records:
        image_id = str(rec.get("image_id", ""))
        results_by_ar = ((rec.get("teacher_scorer") or {}).get("results_by_ar") or {})
        for target_ar, ar_res in results_by_ar.items():
            candidates = annotate_group_candidates(
                image_id=image_id,
                target_ar=str(target_ar),
                ar_res=ar_res,
                softmax_tau=softmax_tau,
                hard_negative_rank_pct_max=hard_negative_rank_pct_max,
                near_margin_max=near_margin_max,
            )
            if not candidates:
                continue
            pairwise_rows.extend(
                build_pairwise_records(
                    image_id=image_id,
                    target_ar=str(target_ar),
                    ar_res=ar_res,
                    candidates=candidates,
                    pair_margin_min=pair_margin_min,
                    max_hard_pairs=max_hard_pairs,
                    max_near_pairs=max_near_pairs,
                )
            )
            listwise = build_listwise_record(
                image_id=image_id,
                target_ar=str(target_ar),
                ar_res=ar_res,
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
                    ar_res=ar_res,
                    candidates=candidates,
                )
            )
            checklist_rows.extend(build_checklist_records(image_id, str(target_ar), ar_res, candidates))
            regression_rows.extend(build_regression_records(image_id, str(target_ar), ar_res, candidates))

    return {
        "pairwise": pairwise_rows,
        "listwise": listwise_rows,
        "decision": decision_rows,
        "checklist": checklist_rows,
        "regression": regression_rows,
    }


def build_qa_summary(datasets: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    pairwise_rows = datasets["pairwise"]
    listwise_rows = datasets["listwise"]
    decision_rows = datasets["decision"]
    checklist_rows = datasets["checklist"]
    regression_rows = datasets["regression"]

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

    return {
        "counts": {
            "pairwise": len(pairwise_rows),
            "listwise": len(listwise_rows),
            "decision": len(decision_rows),
            "checklist": len(checklist_rows),
            "regression": len(regression_rows),
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
    }


def build_markdown_report(
    teacher_scores_jsonl: Path,
    out_dir: Path,
    datasets: Dict[str, List[Dict[str, Any]]],
    qa_summary: Dict[str, Any],
) -> str:
    counts = qa_summary["counts"]
    pairwise = qa_summary["pairwise"]
    listwise = qa_summary["listwise"]
    decision = qa_summary["decision"]
    regression = qa_summary["regression"]
    checklist = qa_summary["checklist"]
    lines = [
        "# FinalScore Training Data Generator Report",
        "",
        f"- input teacher scores: `{teacher_scores_jsonl}`",
        f"- output dir: `{out_dir}`",
        "",
        "## 1. Generated Outputs",
        "",
        f"- pairwise: `{counts['pairwise']}`",
        f"- listwise: `{counts['listwise']}`",
        f"- decision: `{counts['decision']}`",
        f"- checklist: `{counts['checklist']}`",
        f"- regression: `{counts['regression']}`",
        "",
        "## 2. Pairwise QA",
        "",
        f"- margin distribution: `{pairwise['margin_distribution']}`",
        f"- unsafe_negative_ratio: `{pairwise['unsafe_negative_ratio']}`",
        f"- pair_type_counts: `{pairwise['pair_type_counts']}`",
        "",
        "## 3. Listwise QA",
        "",
        f"- candidate_count_distribution: `{listwise['candidate_count_distribution']}`",
        f"- softmax_entropy_distribution: `{listwise['softmax_entropy_distribution']}`",
        f"- top1_prob_distribution: `{listwise['top1_prob_distribution']}`",
        f"- label_type_counts: `{listwise['label_type_counts']}`",
        "",
        "## 4. Decision QA",
        "",
        f"- decision_counts: `{decision['decision_counts']}`",
        f"- baseline_dominance_rate: `{decision['baseline_dominance_rate']}`",
        "",
        "## 5. Regression QA",
        "",
        f"- rank_pct_distribution: `{regression['rank_pct_distribution']}`",
        f"- z_local_distribution: `{regression['z_local_distribution']}`",
        f"- softmax_local_distribution: `{regression['softmax_local_distribution']}`",
        f"- pseudo_mos_1to5_distribution: `{regression['pseudo_mos_1to5_distribution']}`",
        "",
        "## 6. Checklist QA",
        "",
        f"- label_type_counts: `{checklist['label_type_counts']}`",
        f"- mode_counts: `{checklist['mode_counts']}`",
        "",
        "## 7. Analysis",
        "",
        "- The generator uses local `(image_id, target_ar)` normalization rather than global regression labels.",
        "- Pairwise/listwise are built from current `score_rank`, while decision labels keep `score_policy` semantics.",
        "- Hard and unsafe negatives are explicitly separated from safe near-negatives.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build training labels from teacher score jsonl")
    parser.add_argument("--teacher_scores_jsonl", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--softmax_tau", type=float, default=0.2)
    parser.add_argument("--listwise_top_pos", type=int, default=5)
    parser.add_argument("--listwise_neg", type=int, default=15)
    parser.add_argument("--pair_margin_min", type=float, default=0.01)
    parser.add_argument("--hard_negative_rank_pct_max", type=float, default=0.2)
    parser.add_argument("--near_margin_max", type=float, default=0.05)
    parser.add_argument("--max_hard_pairs", type=int, default=4)
    parser.add_argument("--max_near_pairs", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    teacher_scores_jsonl = Path(args.teacher_scores_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    teacher_records = load_teacher_records(teacher_scores_jsonl)
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
    )
    counts = {
        "train_pairwise.jsonl": write_jsonl(out_dir / "train_pairwise.jsonl", datasets["pairwise"]),
        "train_listwise.jsonl": write_jsonl(out_dir / "train_listwise.jsonl", datasets["listwise"]),
        "train_decision.jsonl": write_jsonl(out_dir / "train_decision.jsonl", datasets["decision"]),
        "train_checklist.jsonl": write_jsonl(out_dir / "train_checklist.jsonl", datasets["checklist"]),
        "train_regression.jsonl": write_jsonl(out_dir / "train_regression.jsonl", datasets["regression"]),
    }
    qa_summary = build_qa_summary(datasets)
    (out_dir / "qa_summary.json").write_text(json.dumps(qa_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "generation_counts.json").write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")
    report_text = build_markdown_report(
        teacher_scores_jsonl=teacher_scores_jsonl,
        out_dir=out_dir,
        datasets=datasets,
        qa_summary=qa_summary,
    )
    (out_dir / "TRAINING_DATA_REPORT_KO.md").write_text(report_text, encoding="utf-8")

    print(json.dumps({"out_dir": str(out_dir), "counts": counts, "qa_summary_path": str(out_dir / "qa_summary.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
