from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import torch

from mobilecropnet_v4.data import (
    CHECKLIST_CLASS_KEYS,
    CHECKLIST_CLASS_SIZES,
    CHECKLIST_CLASS_VOCABS,
    CHECKLIST_SCORE_KEYS,
    DECISION_VOCAB,
    DEFAULT_IMAGE_MEAN,
    DEFAULT_IMAGE_STD,
    PERSON_ONLY_WHY_TAGS,
    SUBJECT_MODE_VOCAB,
    LetterboxTransform,
    WHY_TAG_VOCAB,
    box_iou_xyxy,
    checklist_applicability_by_mode,
    letterbox_to_original_box,
    safe_float,
    target_ar_value,
    why_tag_applicability_by_mode,
)
from mobilecropnet_v4.model import MobileCropNetV4

CHECKLIST_LABELS = ("aesthetic", "subject", "composition", "technical")
AR_COMPATIBILITY_LOG_ERROR_THRESHOLD = 0.08
RUNTIME_BASELINE_LABELS = ("full", "minimal")
RUNTIME_BASELINE_DECISION_CONFIDENCE_THRESHOLD = 0.60
RUNTIME_BASELINE_SCORE_MARGIN = 0.03
SUBJECT_VALID_ROUTE_LABELS = frozenset({"object_multi", "object_single", "portrait_group", "portrait_single"})
SUBJECT_VALID_ROUTE_LABELS_WITH_SCENE = frozenset({*SUBJECT_VALID_ROUTE_LABELS, "scene_general"})
SUBJECT_VALID_ROUTE_LABELS_WITH_SCENE_BACKGROUND = frozenset(
    {*SUBJECT_VALID_ROUTE_LABELS_WITH_SCENE, "background_texture_copyspace"}
)
SUBJECT_VALID_POLICIES = (
    "confidence",
    "route",
    "route_or_conf",
    "route_and_conf",
    "route_scene",
    "route_scene_or_conf",
    "route_scene_and_conf",
    "route_scene_background",
    "route_scene_background_or_conf",
    "route_scene_background_and_conf",
)
SCORE_SURFACES = (
    "deployment",
    "source_mixture",
    "decision_conditioned",
    "raw_return",
    "utility",
)


def _clamp_norm_box(box: Sequence[float]) -> list[float]:
    if not isinstance(box, Sequence) or len(box) < 4:
        return [0.0, 0.0, 1.0, 1.0]
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    x1 = min(max(0.0, x1), 1.0)
    y1 = min(max(0.0, y1), 1.0)
    x2 = min(max(0.0, x2), 1.0)
    y2 = min(max(0.0, y2), 1.0)
    if x2 <= x1:
        x1, x2 = 0.0, 1.0
    if y2 <= y1:
        y1, y2 = 0.0, 1.0
    return [x1, y1, x2, y2]


def _row_score(row: dict[str, Any] | None) -> float:
    if not isinstance(row, dict):
        return 0.0
    return safe_float(row.get("utility_score", row.get("score", 0.0)), 0.0)


def _decision_probability_payload(decision_logits: Sequence[float] | torch.Tensor | None) -> tuple[dict[str, float], float | None]:
    if decision_logits is None:
        return {}, None
    if torch.is_tensor(decision_logits):
        logits = decision_logits.detach().float().view(-1).cpu()
    else:
        logits = torch.tensor([float(v) for v in decision_logits], dtype=torch.float32)
    if logits.numel() == 0:
        return {}, None
    probs = torch.softmax(logits, dim=0).tolist()
    payload = {class_label(DECISION_VOCAB, idx): float(prob) for idx, prob in enumerate(probs)}
    return payload, float(max(probs))


def box_target_ar_stats(box: Sequence[float], *, width: int, height: int, target_ar: str) -> dict[str, Any]:
    crop_ar, ar_error = _target_ar_log_error(box, width=width, height=height, target_ar=target_ar)
    return {
        "crop_ar": crop_ar,
        "target_ar_log_error": ar_error,
        "target_ar_compatible": bool(ar_error is None or ar_error <= AR_COMPATIBILITY_LOG_ERROR_THRESHOLD),
    }


def build_runtime_baselines(*, width: int, height: int, target_ar: str) -> dict[str, dict[str, Any]]:
    width = max(1, int(width))
    height = max(1, int(height))
    image_ar = float(width) / float(height)
    full_box = [0.0, 0.0, 1.0, 1.0]
    target = target_ar_value(target_ar)
    if target is None:
        minimal_box = list(full_box)
    elif image_ar >= target:
        crop_w = min(1.0, float(target) / max(1e-6, image_ar))
        x0 = 0.5 * (1.0 - crop_w)
        minimal_box = [x0, 0.0, x0 + crop_w, 1.0]
    else:
        crop_h = min(1.0, max(1e-6, image_ar) / float(target))
        y0 = 0.5 * (1.0 - crop_h)
        minimal_box = [0.0, y0, 1.0, y0 + crop_h]

    def make_row(label: str, box: Sequence[float]) -> dict[str, Any]:
        return {
            "baseline_label": label,
            "bbox_norm_xyxy": [float(v) for v in _clamp_norm_box(box)],
            "target_ar": str(target_ar),
            "target_ar_value": target_ar_value(target_ar),
            **box_target_ar_stats(box, width=width, height=height, target_ar=target_ar),
        }

    return {
        "full": make_row("full", full_box),
        "minimal": make_row("minimal", minimal_box),
    }


def repair_box_to_target_ar(
    box: Sequence[float],
    *,
    width: int,
    height: int,
    target_ar: str,
) -> dict[str, Any]:
    clamped = _clamp_norm_box(box)
    target = target_ar_value(target_ar)
    if target is None:
        crop_ar, ar_error = _target_ar_log_error(clamped, width=width, height=height, target_ar=target_ar)
        return {
            "bbox_norm_xyxy": clamped,
            "crop_ar": crop_ar,
            "target_ar_log_error": ar_error,
            "target_ar_compatible": True,
            "exact_ar_postprocessed": False,
        }
    width = max(1, int(width))
    height = max(1, int(height))
    image_ar = float(width) / float(height)
    x1, y1, x2, y2 = clamped
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    area = max(1e-6, (x2 - x1) * (y2 - y1))
    norm_ratio = float(target) / max(1e-6, image_ar)
    crop_h = math.sqrt(area / max(1e-6, norm_ratio))
    crop_w = norm_ratio * crop_h
    scale = min(1.0, 1.0 / max(1e-6, crop_w), 1.0 / max(1e-6, crop_h))
    crop_w *= scale
    crop_h *= scale
    half_w = 0.5 * crop_w
    half_h = 0.5 * crop_h
    cx = min(max(cx, half_w), 1.0 - half_w)
    cy = min(max(cy, half_h), 1.0 - half_h)
    repaired = [cx - half_w, cy - half_h, cx + half_w, cy + half_h]
    crop_ar, ar_error = _target_ar_log_error(repaired, width=width, height=height, target_ar=target_ar)
    return {
        "bbox_norm_xyxy": [float(v) for v in _clamp_norm_box(repaired)],
        "crop_ar": crop_ar,
        "target_ar_log_error": ar_error,
        "target_ar_compatible": bool(ar_error is None or ar_error <= AR_COMPATIBILITY_LOG_ERROR_THRESHOLD),
        "exact_ar_postprocessed": bool(any(abs(float(a) - float(b)) > 1e-6 for a, b in zip(repaired, clamped))),
    }


def execute_runtime_decision(
    *,
    decision_id: int,
    target_ar: str,
    width: int,
    height: int,
    proposals: Sequence[dict[str, Any]],
    selected_proposal_index: int = 0,
    exact_target_ar_postprocess: bool = True,
    baseline_rows: dict[str, dict[str, Any]] | None = None,
    decision_logits: Sequence[float] | torch.Tensor | None = None,
    decision_source_logit: float | torch.Tensor | None = None,
    decision_source_raw_logit: float | torch.Tensor | None = None,
    decision_source_logit_threshold: float | torch.Tensor | None = None,
    use_decision_source_action_gate: bool = False,
    enforce_baseline_decision_gate: bool = False,
    baseline_decision_confidence_threshold: float = RUNTIME_BASELINE_DECISION_CONFIDENCE_THRESHOLD,
    baseline_score_margin: float = RUNTIME_BASELINE_SCORE_MARGIN,
) -> dict[str, Any]:
    baselines = baseline_rows or build_runtime_baselines(width=width, height=height, target_ar=target_ar)
    decision_label = class_label(DECISION_VOCAB, int(decision_id))
    decision_probs, decision_top_confidence = _decision_probability_payload(decision_logits)
    decision_confidence = decision_probs.get(decision_label)
    chosen_crop = None
    crop_action_source = "baseline_minimal_fallback"
    if proposals:
        chosen_crop = dict(proposals[max(0, min(int(selected_proposal_index), len(proposals) - 1))])
        crop_action_source = str(chosen_crop.get("proposal_id", selected_proposal_index))
    requested_baseline = None
    requested_action_source = None
    if decision_label == "keep_full":
        requested_baseline = dict(baselines["full"])
        requested_action_source = "baseline_full"
    elif decision_label == "minimal_crop":
        requested_baseline = dict(baselines["minimal"])
        requested_action_source = "baseline_minimal"

    source_value = None
    source_raw_value = None
    source_threshold_value = None
    source_gate_action = None
    runtime_decision_label = decision_label
    if use_decision_source_action_gate and decision_source_logit is not None:
        if torch.is_tensor(decision_source_logit):
            source_value = float(decision_source_logit.detach().cpu().view(-1)[0].item())
        else:
            source_value = float(decision_source_logit)
        if decision_source_raw_logit is not None:
            if torch.is_tensor(decision_source_raw_logit):
                source_raw_value = float(decision_source_raw_logit.detach().cpu().view(-1)[0].item())
            else:
                source_raw_value = float(decision_source_raw_logit)
        if decision_source_logit_threshold is not None:
            if torch.is_tensor(decision_source_logit_threshold):
                source_threshold_value = float(decision_source_logit_threshold.detach().cpu().view(-1)[0].item())
            else:
                source_threshold_value = float(decision_source_logit_threshold)
        source_gate_action = "crop" if source_value >= 0.0 else "baseline"
        if source_gate_action == "crop":
            requested_baseline = None
            requested_action_source = crop_action_source
            runtime_decision_label = "crop"
        elif requested_baseline is None:
            full_score = _row_score(baselines["full"])
            minimal_score = _row_score(baselines["minimal"])
            if full_score >= minimal_score:
                requested_baseline = dict(baselines["full"])
                requested_action_source = "baseline_full"
                runtime_decision_label = "keep_full"
            else:
                requested_baseline = dict(baselines["minimal"])
                requested_action_source = "baseline_minimal"
                runtime_decision_label = "minimal_crop"
        elif requested_action_source == "baseline_full":
            runtime_decision_label = "keep_full"
        else:
            runtime_decision_label = "minimal_crop"

    gate_reasons: list[str] = []
    baseline_score = None
    crop_score = None
    requested_action_source = requested_action_source or crop_action_source
    if requested_baseline is not None and enforce_baseline_decision_gate:
        baseline_score = _row_score(requested_baseline)
        crop_score = _row_score(chosen_crop)
        if decision_confidence is not None and decision_confidence < float(baseline_decision_confidence_threshold):
            gate_reasons.append("low_decision_confidence")
        if chosen_crop is not None and baseline_score < (crop_score + float(baseline_score_margin)):
            gate_reasons.append("baseline_not_better_than_crop")

    if requested_baseline is not None and not gate_reasons:
        chosen = requested_baseline
        action_source = requested_action_source
    elif chosen_crop is not None:
        chosen = chosen_crop
        action_source = crop_action_source
    elif requested_baseline is not None:
        chosen = requested_baseline
        action_source = requested_action_source
    else:
        chosen = dict(baselines["minimal"])
        action_source = "baseline_minimal_fallback"
    post = repair_box_to_target_ar(chosen.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0]), width=width, height=height, target_ar=target_ar)
    if exact_target_ar_postprocess:
        chosen["bbox_norm_xyxy"] = post["bbox_norm_xyxy"]
        chosen["crop_ar"] = post["crop_ar"]
        chosen["target_ar_log_error"] = post["target_ar_log_error"]
        chosen["target_ar_compatible"] = post["target_ar_compatible"]
    chosen["exact_ar_postprocessed"] = bool(exact_target_ar_postprocess and post["exact_ar_postprocessed"])
    chosen["raw_decision_label"] = decision_label
    chosen["decision_label"] = runtime_decision_label
    chosen["runtime_decision_label"] = runtime_decision_label
    chosen["runtime_decision_id"] = int(DECISION_VOCAB.index(runtime_decision_label))
    chosen["action_source"] = action_source
    chosen["selected_proposal_index"] = int(selected_proposal_index)
    chosen["decision_probs"] = decision_probs
    chosen["decision_confidence"] = decision_confidence
    chosen["decision_top_confidence"] = decision_top_confidence
    chosen["decision_source_logit"] = source_value
    chosen["decision_source_raw_logit"] = source_raw_value
    chosen["decision_source_logit_threshold"] = source_threshold_value
    chosen["decision_source_gate_action"] = source_gate_action
    chosen["use_decision_source_action_gate"] = bool(use_decision_source_action_gate)
    chosen["requested_action_source"] = requested_action_source
    chosen["requested_baseline_score"] = baseline_score
    chosen["selected_crop_score"] = crop_score
    chosen["enforce_baseline_decision_gate"] = bool(enforce_baseline_decision_gate)
    chosen["baseline_decision_confidence_threshold"] = float(baseline_decision_confidence_threshold)
    chosen["baseline_score_margin"] = float(baseline_score_margin)
    chosen["executor_override_to_crop"] = bool(requested_baseline is not None and bool(gate_reasons) and not action_source.startswith("baseline"))
    chosen["executor_gate_reasons"] = gate_reasons
    chosen["executor_decision_consistent"] = float(
        (decision_label == "keep_full" and action_source.startswith("baseline_full"))
        or (decision_label == "minimal_crop" and action_source.startswith("baseline_minimal"))
        or (decision_label == "crop" and not action_source.startswith("baseline"))
    )
    return chosen


def quality_label(score: float) -> str:
    value = float(score)
    if value >= 0.72:
        return "good"
    if value >= 0.45:
        return "ok"
    return "bad"


def class_label(vocab: Sequence[str], idx: int) -> str:
    return vocab[int(idx)] if 0 <= int(idx) < len(vocab) else f"class_{int(idx)}"


def subject_valid_from_policy(
    *,
    confidence: float,
    threshold: float,
    route_label: str | None = None,
    policy: str = "confidence",
) -> bool:
    conf_valid = float(confidence) >= float(threshold)
    route_known = route_label is not None and str(route_label) != ""
    if policy.startswith("route_scene_background"):
        route_labels = SUBJECT_VALID_ROUTE_LABELS_WITH_SCENE_BACKGROUND
    elif policy.startswith("route_scene"):
        route_labels = SUBJECT_VALID_ROUTE_LABELS_WITH_SCENE
    else:
        route_labels = SUBJECT_VALID_ROUTE_LABELS
    route_valid = bool(str(route_label) in route_labels) if route_known else conf_valid
    if policy in {"route", "route_scene", "route_scene_background"}:
        return bool(route_valid)
    if policy in {"route_or_conf", "route_scene_or_conf", "route_scene_background_or_conf"}:
        return bool(route_valid or conf_valid)
    if policy in {"route_and_conf", "route_scene_and_conf", "route_scene_background_and_conf"}:
        return bool(route_valid and conf_valid)
    return bool(conf_valid)


def _target_ar_log_error(box: Sequence[float], *, width: int, height: int, target_ar: str) -> tuple[float | None, float | None]:
    expected = target_ar_value(target_ar)
    if expected is None:
        return None, 0.0
    if not isinstance(box, Sequence) or len(box) < 4:
        return None, None
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    crop_w = max(1e-6, (x2 - x1) * float(max(1, int(width))))
    crop_h = max(1e-6, (y2 - y1) * float(max(1, int(height))))
    crop_ar = crop_w / crop_h
    return float(crop_ar), float(abs(math.log(max(1e-6, crop_ar) / max(1e-6, expected))))


def _letterbox_ar(box: Sequence[float]) -> float | None:
    if not isinstance(box, Sequence) or len(box) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    return float(max(1e-6, x2 - x1) / max(1e-6, y2 - y1))


def _select_target_ar_proposal(proposals: Sequence[dict[str, Any]], target_ar: str) -> dict[str, Any] | None:
    if not proposals:
        return None
    if target_ar_value(target_ar) is None:
        return dict(proposals[0])
    compatible = [p for p in proposals if bool(p.get("target_ar_compatible", False))]
    if compatible:
        return dict(compatible[0])
    return dict(min(proposals, key=lambda p: safe_float(p.get("target_ar_log_error"), 1e9)))


def select_generated_proposal_index(
    *,
    utility_scores: Sequence[float],
    proposal_scores: Sequence[float],
    proposals: Sequence[dict[str, Any]] | None = None,
    selection_policy: str = "utility_top1",
    proposal_top_m: int = 8,
) -> int:
    if not utility_scores:
        return 0
    count = min(len(utility_scores), len(proposal_scores)) if proposal_scores else len(utility_scores)
    indices = list(range(count))
    if not indices:
        return 0
    if selection_policy == "proposal_top1":
        return max(indices, key=lambda idx: float(proposal_scores[idx]))
    if selection_policy == "proposal_topk_rerank":
        top_m = max(1, min(int(proposal_top_m), len(indices)))
        shortlist = sorted(indices, key=lambda idx: float(proposal_scores[idx]), reverse=True)[:top_m]
        if proposals:
            compatible = [idx for idx in shortlist if idx < len(proposals) and bool(proposals[idx].get("target_ar_compatible", True))]
            if compatible:
                shortlist = compatible
        return max(shortlist, key=lambda idx: float(utility_scores[idx]))
    return max(indices, key=lambda idx: float(utility_scores[idx]))


def select_score_logits(outputs: dict[str, torch.Tensor], *, score_surface: str = "deployment") -> torch.Tensor:
    surface = str(score_surface or "deployment")
    if surface == "deployment":
        return outputs.get(
            "source_mixture_return_logits",
            outputs.get("decision_conditioned_return_logits", outputs.get("return_logits", outputs["utility_logits"])),
        )
    if surface == "source_mixture":
        return outputs.get("source_mixture_return_logits", outputs.get("return_logits", outputs["utility_logits"]))
    if surface == "decision_conditioned":
        return outputs.get("decision_conditioned_return_logits", outputs.get("return_logits", outputs["utility_logits"]))
    if surface == "raw_return":
        return outputs.get("return_logits", outputs["utility_logits"])
    if surface == "utility":
        return outputs["utility_logits"]
    raise ValueError(f"unsupported score_surface={score_surface!r}; expected one of {SCORE_SURFACES}")


def detailed_explanation_from_logits(
    *,
    checklist_logits: torch.Tensor | None,
    checklist_applicability_logits: torch.Tensor | None = None,
    detail_score_logits: torch.Tensor | None = None,
    why_tag_logits: torch.Tensor | None = None,
    subject_mode_label: str | None = None,
    applicability_threshold: float = 0.5,
    top_why_tags: int = 8,
) -> dict[str, Any]:
    detailed: dict[str, Any] = {}
    mode = str(subject_mode_label or "other_ambiguous")
    class_applicable = checklist_applicability_by_mode(mode)
    app_prob_values = (
        torch.sigmoid(checklist_applicability_logits.float()).detach().cpu().tolist()
        if checklist_applicability_logits is not None
        else [1.0] * len(CHECKLIST_CLASS_KEYS)
    )
    if checklist_logits is not None:
        offset = 0
        for key_idx, (key, size) in enumerate(zip(CHECKLIST_CLASS_KEYS, CHECKLIST_CLASS_SIZES)):
            logits = checklist_logits[offset : offset + int(size)]
            offset += int(size)
            prob = torch.softmax(logits.float(), dim=0)
            idx = int(prob.argmax().item())
            label = class_label(CHECKLIST_CLASS_VOCABS[key], idx)
            raw_available = label not in {"na", "n/a", "none", "null", ""}
            mode_applicable = bool(class_applicable[key_idx] > 0.0)
            applicability_score = float(app_prob_values[key_idx]) if key_idx < len(app_prob_values) else 1.0
            model_applicable = applicability_score >= float(applicability_threshold)
            available = bool(mode_applicable and model_applicable and raw_available)
            if available:
                display_label = label
            elif not mode_applicable:
                display_label = "not_applicable"
            elif not model_applicable:
                display_label = "unavailable"
            else:
                display_label = "unlabeled"
            detailed[key] = {
                "label": label,
                "display_label": display_label,
                "available": bool(available),
                "mode_applicable": bool(mode_applicable),
                "model_applicable": bool(model_applicable),
                "applicability_score": applicability_score,
                "raw_label": label,
                "raw_score": float(prob[idx].item()),
                "score": float(prob[idx].item()) if available else None,
            }
    detail_scores: dict[str, float] = {}
    if detail_score_logits is not None:
        values = torch.sigmoid(detail_score_logits.float()).detach().cpu().tolist()
        detail_scores = {key: float(values[idx]) for idx, key in enumerate(CHECKLIST_SCORE_KEYS)}
    why_tags: list[dict[str, Any]] = []
    if why_tag_logits is not None:
        probs = torch.sigmoid(why_tag_logits.float()).detach().cpu().tolist()
        tag_applicable = why_tag_applicability_by_mode(mode)
        ranked = sorted(enumerate(probs), key=lambda item: float(item[1]), reverse=True)
        for idx, score in ranked[: max(0, int(top_why_tags))]:
            if idx < len(tag_applicable) and tag_applicable[idx] <= 0.0:
                continue
            if float(score) >= 0.2:
                why_tags.append({"tag": WHY_TAG_VOCAB[idx], "score": float(score)})
    return {
        "detailed_checklist": detailed,
        "detail_scores": detail_scores,
        "why_tags": why_tags,
    }


def mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(vals) / len(vals)) if vals else 0.0


def pearson_corr(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    ax = mean(a)
    bx = mean(b)
    num = sum((float(x) - ax) * (float(y) - bx) for x, y in zip(a, b))
    da = math.sqrt(sum((float(x) - ax) ** 2 for x in a))
    db = math.sqrt(sum((float(y) - bx) ** 2 for y in b))
    return float(num / (da * db)) if da > 0 and db > 0 else 0.0


def _rank(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: float(values[i]))
    ranks = [0.0] * len(values)
    for r, idx in enumerate(order):
        ranks[idx] = float(r)
    return ranks


def spearman_corr(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return 0.0
    return pearson_corr(_rank(a), _rank(b))


def ndcg_at_k(labels: Sequence[float], scores: Sequence[float], k: int) -> float:
    if not labels or len(labels) != len(scores):
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:k]
    ideal = sorted(range(len(labels)), key=lambda i: float(labels[i]), reverse=True)[:k]

    def dcg(indices: list[int]) -> float:
        total = 0.0
        for rank, idx in enumerate(indices, 1):
            total += (2.0 ** float(labels[idx]) - 1.0) / math.log2(rank + 1.0)
        return total

    ideal_dcg = dcg(ideal)
    return float(dcg(order) / ideal_dcg) if ideal_dcg > 0 else 0.0


def topk_any_positive(labels: Sequence[float], scores: Sequence[float], k: int) -> float:
    if not labels:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:k]
    return float(any(float(labels[i]) > 0 for i in order))


def topk_exact_best(labels: Sequence[float], scores: Sequence[float], k: int = 1) -> float:
    if not labels:
        return 0.0
    best = max(range(len(labels)), key=lambda i: float(labels[i]))
    order = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:k]
    return float(best in order)


def load_mobilecropnet_v4_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[MobileCropNetV4, dict[str, Any]]:
    load_mode = str(os.environ.get("MOBILECROPNET_CHECKPOINT_LOAD_MODE", "cpu")).strip().lower()
    load_on_device = load_mode in {"device", "gpu", "cuda"} and device.type != "cpu"
    ckpt = torch.load(
        checkpoint_path,
        map_location=device if load_on_device else "cpu",
        weights_only=False,
    )
    ckpt.pop("optimizer", None)
    config = dict(ckpt.get("model_config", {}))
    init_config = dict(config)
    # The checkpoint state_dict contains the trained backbone weights. Avoid a
    # redundant external pretrained download during evaluation/inference reload.
    init_config["backbone_pretrained"] = False
    model = MobileCropNetV4(**init_config)
    if load_on_device:
        model.to(device)
    state_dict = ckpt.get("model")
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"checkpoint does not contain a model state_dict: {checkpoint_path}")
    incompatible = model.load_state_dict(state_dict, strict=False)
    allowed_missing_prefixes = (
        "checklist_class_head.",
        "checklist_applicability_head.",
        "detail_score_head.",
        "why_tag_head.",
        "subject_box_head.",
        "subject_valid_head.",
        "subject_proposal_head.",
        "subject_refine_proj.",
        "subject_refine_box_head.",
        "subject_refine_valid_head.",
    )
    bad_missing = [key for key in incompatible.missing_keys if not key.startswith(allowed_missing_prefixes)]
    if bad_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint is not compatible with MobileCropNetV4: "
            f"missing={bad_missing[:20]}, unexpected={incompatible.unexpected_keys[:20]}"
        )
    if any(key.startswith("checklist_applicability_head.") for key in incompatible.missing_keys):
        with torch.no_grad():
            model.checklist_applicability_head.weight.zero_()
            model.checklist_applicability_head.bias.fill_(5.0)
    if not load_on_device:
        model.to(device)
    model.eval()
    return model, ckpt


def infer_input_size(ckpt: dict[str, Any], explicit_input_size: int | None) -> int:
    if explicit_input_size is not None:
        return int(explicit_input_size)
    return int(ckpt.get("train_config", {}).get("input_size", 256))


def infer_subject_valid_threshold(
    ckpt: dict[str, Any],
    explicit_threshold: float | None = None,
    *,
    default: float = 0.5,
) -> float:
    if explicit_threshold is not None:
        return float(explicit_threshold)
    metrics = ckpt.get("metrics", {})
    if isinstance(metrics, dict):
        metric_sections = [metrics]
        val_metrics = metrics.get("val")
        if isinstance(val_metrics, dict):
            metric_sections.insert(0, val_metrics)
        for section in metric_sections:
            for key in ("subject_box_valid_best_threshold", "subject_valid_threshold", "subject_box_valid_threshold"):
                value = section.get(key)
                if value is not None:
                    return float(value)
    for section_name in ("train_config", "eval_config", "model_config"):
        section = ckpt.get(section_name, {})
        if not isinstance(section, dict):
            continue
        for key in ("subject_valid_threshold", "subject_box_valid_threshold", "subject_box_valid_best_threshold"):
            value = section.get(key)
            if value is not None:
                return float(value)
    return float(default)


def parse_norm_triplet(value: Any, *, default: Sequence[float] | None = None) -> list[float] | None:
    if value is None:
        return [float(v) for v in default] if default is not None else None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return [float(v) for v in default] if default is not None else None
        items = [float(part.strip()) for part in raw.split(",")]
    else:
        items = [float(part) for part in value]
    if len(items) != 3:
        raise ValueError(f"expected 3 normalization values, got {len(items)}")
    return [float(items[0]), float(items[1]), float(items[2])]


def _backbone_pretrained_norm(backbone_name: str) -> tuple[list[float] | None, list[float] | None]:
    name = str(backbone_name or "").strip()
    if not name or name in {"custom_depthwise", "depthwise"}:
        return None, None
    try:
        import timm
    except ImportError:
        return None, None
    cfg = timm.models.get_pretrained_cfg(name)
    if cfg is None:
        return None, None
    mean = list(getattr(cfg, "mean", ()) or ())
    std = list(getattr(cfg, "std", ()) or ())
    return (mean if len(mean) == 3 else None, std if len(std) == 3 else None)


def infer_image_norm(
    ckpt: dict[str, Any],
    *,
    explicit_mean: Any = None,
    explicit_std: Any = None,
) -> tuple[list[float], list[float]]:
    train_config = ckpt.get("train_config", {}) if isinstance(ckpt.get("train_config", {}), dict) else {}
    model_config = ckpt.get("model_config", {}) if isinstance(ckpt.get("model_config", {}), dict) else {}
    cfg_mean, cfg_std = _backbone_pretrained_norm(str(model_config.get("backbone_name", "")))
    mean_default = parse_norm_triplet(train_config.get("image_mean"), default=cfg_mean or DEFAULT_IMAGE_MEAN)
    std_default = parse_norm_triplet(train_config.get("image_std"), default=cfg_std or DEFAULT_IMAGE_STD)
    mean = parse_norm_triplet(explicit_mean, default=mean_default or list(DEFAULT_IMAGE_MEAN))
    std = parse_norm_triplet(explicit_std, default=std_default or list(DEFAULT_IMAGE_STD))
    return mean or list(DEFAULT_IMAGE_MEAN), std or list(DEFAULT_IMAGE_STD)


def batch_predictions(
    *,
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    save_topk: int = 8,
    subject_valid_threshold: float = 0.5,
    subject_valid_policy: str = "confidence",
) -> list[dict[str, Any]]:
    score_logits = select_score_logits(outputs, score_surface="deployment")
    utility = torch.sigmoid(score_logits).detach().cpu()
    positive = torch.sigmoid(outputs["positive_logits"]).detach().cpu()
    risk = torch.sigmoid(outputs["risk_logits"]).detach().cpu()
    macro = torch.sigmoid(outputs["macro_logits"]).detach().cpu()
    checklist_logits = outputs.get("checklist_class_logits")
    checklist_applicability_logits = outputs.get("checklist_applicability_logits")
    detail_score_logits = outputs.get("detail_score_logits")
    why_tag_logits = outputs.get("why_tag_logits")
    checklist_logits_cpu = checklist_logits.detach().cpu() if torch.is_tensor(checklist_logits) else None
    checklist_applicability_logits_cpu = checklist_applicability_logits.detach().cpu() if torch.is_tensor(checklist_applicability_logits) else None
    detail_score_logits_cpu = detail_score_logits.detach().cpu() if torch.is_tensor(detail_score_logits) else None
    why_tag_logits_cpu = why_tag_logits.detach().cpu() if torch.is_tensor(why_tag_logits) else None
    proposal_prob = torch.sigmoid(outputs["proposal_logits"]).detach().cpu()
    proposal_boxes = outputs["proposal_boxes"].detach().cpu()
    pred_subject_boxes = outputs.get("pred_subject_box")
    pred_subject_valid = outputs.get("pred_subject_valid")
    pred_subject_boxes_cpu = pred_subject_boxes.detach().cpu() if torch.is_tensor(pred_subject_boxes) else None
    pred_subject_valid_cpu = pred_subject_valid.detach().cpu() if torch.is_tensor(pred_subject_valid) else None
    decision_ids = outputs["decision_logits"].argmax(dim=1).detach().cpu()
    route_ids = outputs["route_logits"].argmax(dim=1).detach().cpu()
    decision_probs_cpu = torch.softmax(outputs["decision_logits"].float(), dim=1).detach().cpu()
    route_probs_cpu = torch.softmax(outputs["route_logits"].float(), dim=1).detach().cpu()
    decision_source_logit = outputs.get("decision_source_logit")
    decision_source_raw_logit = outputs.get("decision_source_raw_logit", decision_source_logit)
    decision_source_threshold = outputs.get("decision_source_logit_threshold")
    decision_source_logit_cpu = decision_source_logit.detach().cpu() if torch.is_tensor(decision_source_logit) else None
    decision_source_raw_logit_cpu = (
        decision_source_raw_logit.detach().cpu() if torch.is_tensor(decision_source_raw_logit) else None
    )
    source_gate_logit = outputs.get("source_gate_logit")
    source_gate_logit_cpu = source_gate_logit.detach().cpu() if torch.is_tensor(source_gate_logit) else None
    action_source_signal = "decision_source_logit" if decision_source_logit_cpu is not None else None
    action_source_logit_cpu = decision_source_logit_cpu
    action_source_raw_logit_cpu = decision_source_raw_logit_cpu
    if action_source_logit_cpu is None and source_gate_logit_cpu is not None:
        action_source_signal = "source_gate_logit"
        action_source_logit_cpu = source_gate_logit_cpu
        action_source_raw_logit_cpu = source_gate_logit_cpu
    decision_source_threshold_value = (
        float(decision_source_threshold.detach().cpu().item())
        if torch.is_tensor(decision_source_threshold) and decision_source_threshold.numel() == 1
        else None
    )
    policy_score = outputs.get("policy_score_pred")
    policy_score_cpu = policy_score.detach().cpu() if torch.is_tensor(policy_score) else None
    rows: list[dict[str, Any]] = []
    for i in range(utility.shape[0]):
        valid_count = int(batch["valid"][i].sum().item())
        scores = [float(v) for v in utility[i, :valid_count].tolist()]
        ranked = sorted(range(valid_count), key=lambda idx: scores[idx], reverse=True)
        route_id = int(route_ids[i].item())
        route_label = class_label(SUBJECT_MODE_VOCAB, route_id)
        candidates = []
        for j in range(valid_count):
            record = dict(batch["candidate_records"][i][j])
            record["model_utility"] = scores[j]
            record["model_positive"] = float(positive[i, j].item())
            record["model_risk"] = float(risk[i, j].item())
            checklist = {}
            for cidx, name in enumerate(CHECKLIST_LABELS):
                value = float(macro[i, j, cidx].item())
                checklist[name] = {"score": value, "label": quality_label(value)}
            record["model_checklist"] = checklist
            record["model_explanation"] = {
                "utility": {"score": scores[j], "label": quality_label(scores[j])},
                "positive": {"score": record["model_positive"], "label": "likely" if record["model_positive"] >= 0.5 else "unlikely"},
                "risk": {"score": record["model_risk"], "label": "unsafe" if record["model_risk"] >= 0.5 else "safe"},
                "checklist": checklist,
            }
            detail = detailed_explanation_from_logits(
                checklist_logits=checklist_logits_cpu[i, j] if checklist_logits_cpu is not None else None,
                checklist_applicability_logits=checklist_applicability_logits_cpu[i, j] if checklist_applicability_logits_cpu is not None else None,
                detail_score_logits=detail_score_logits_cpu[i, j] if detail_score_logits_cpu is not None else None,
                why_tag_logits=why_tag_logits_cpu[i, j] if why_tag_logits_cpu is not None else None,
                subject_mode_label=route_label,
            )
            record["model_detailed_checklist"] = detail["detailed_checklist"]
            record["model_detail_scores"] = detail["detail_scores"]
            record["model_why_tags"] = detail["why_tags"]
            record["model_explanation"].update(detail)
            candidates.append(record)
        transform = batch["transform"][i]
        assert isinstance(transform, LetterboxTransform)
        proposals = []
        width = int(batch["width"][i])
        height = int(batch["height"][i])
        target_ar = str(batch["target_ar"][i])
        for q, prob in sorted(
            enumerate([float(v) for v in proposal_prob[i].tolist()]),
            key=lambda item: item[1],
            reverse=True,
        )[:save_topk]:
            padded_box = [float(v) for v in proposal_boxes[i, q].tolist()]
            orig_box = letterbox_to_original_box(padded_box, transform)
            crop_ar, ar_error = _target_ar_log_error(orig_box, width=width, height=height, target_ar=target_ar)
            lb_ar = _letterbox_ar(padded_box)
            proposals.append(
                {
                    "proposal_id": int(q),
                    "score": float(prob),
                    "bbox_letterbox_xyxy": padded_box,
                    "bbox_norm_xyxy": orig_box,
                    "target_ar": target_ar,
                    "target_ar_value": target_ar_value(target_ar),
                    "crop_ar": crop_ar,
                    "letterbox_ar": lb_ar,
                    "target_ar_log_error": ar_error,
                    "target_ar_compatible": bool(ar_error is None or ar_error <= AR_COMPATIBILITY_LOG_ERROR_THRESHOLD),
                }
            )
        raw_top_proposal = dict(proposals[0]) if proposals else None
        target_ar_top_proposal = _select_target_ar_proposal(proposals, target_ar)
        subject_box: dict[str, Any] = {}
        if pred_subject_boxes_cpu is not None and pred_subject_valid_cpu is not None:
            pred_lb = [float(v) for v in pred_subject_boxes_cpu[i].tolist()]
            pred_conf = float(pred_subject_valid_cpu[i].item())
            subject_box["predicted"] = {
                "bbox_letterbox_xyxy": pred_lb,
                "bbox_norm_xyxy": letterbox_to_original_box(pred_lb, transform),
                "confidence": pred_conf,
                "valid_threshold": float(subject_valid_threshold),
                "valid_policy": str(subject_valid_policy),
                "confidence_valid": bool(pred_conf >= float(subject_valid_threshold)),
                "route_valid": bool(route_label in SUBJECT_VALID_ROUTE_LABELS),
                "valid": subject_valid_from_policy(
                    confidence=pred_conf,
                    threshold=float(subject_valid_threshold),
                    route_label=route_label,
                    policy=str(subject_valid_policy),
                ),
            }
        teacher_valid = batch.get("subject_box_valid", batch.get("subject_prior_valid"))
        teacher_box = batch.get("subject_box_target", batch.get("subject_prior_box"))
        if torch.is_tensor(teacher_valid) and torch.is_tensor(teacher_box):
            target_valid = float(teacher_valid[i].item())
            teacher_lb = [float(v) for v in teacher_box[i].tolist()]
            subject_box["target_valid"] = target_valid
            if target_valid > 0.0:
                subject_box["teacher"] = {
                    "bbox_letterbox_xyxy": teacher_lb,
                    "bbox_norm_xyxy": letterbox_to_original_box(teacher_lb, transform),
                    "confidence": float(batch.get("subject_prior_reliability", torch.zeros_like(teacher_valid))[i].item())
                    if torch.is_tensor(batch.get("subject_prior_reliability"))
                    else None,
                }
        if isinstance(subject_box.get("predicted"), dict) and isinstance(subject_box.get("teacher"), dict):
            subject_box["iou_to_teacher"] = box_iou_xyxy(subject_box["predicted"]["bbox_norm_xyxy"], subject_box["teacher"]["bbox_norm_xyxy"])
        top = candidates[ranked[0]] if ranked else {}
        positives_ref = list(batch["positive_records"][i])
        best_positive = max(positives_ref, key=lambda row: safe_float(row.get("score")), default=None)
        decision_source_debug = None
        if action_source_logit_cpu is not None:
            source_logit = float(action_source_logit_cpu[i].item())
            raw_source_logit = (
                float(action_source_raw_logit_cpu[i].item())
                if action_source_raw_logit_cpu is not None
                else source_logit
            )
            decision_source_debug = {
                "signal": action_source_signal,
                "logit": source_logit,
                "raw_logit": raw_source_logit,
                "threshold": decision_source_threshold_value,
                "crop_probability": float(torch.sigmoid(action_source_logit_cpu[i]).item()),
                "raw_crop_probability": float(torch.sigmoid(action_source_raw_logit_cpu[i]).item())
                if action_source_raw_logit_cpu is not None
                else float(torch.sigmoid(action_source_logit_cpu[i]).item()),
                "predicted_source": "crop" if source_logit >= 0.0 else "baseline",
            }
        rows.append(
            {
                "image_id": batch["image_id"][i],
                "image_path": batch["image_path"][i],
                "target_ar": target_ar,
                "width": width,
                "height": height,
                "decision_id": int(decision_ids[i].item()),
                "decision_label": class_label(DECISION_VOCAB, int(decision_ids[i].item())),
                "decision_probabilities": {
                    class_label(DECISION_VOCAB, idx): float(value)
                    for idx, value in enumerate(decision_probs_cpu[i].tolist())
                },
                "decision_source": decision_source_debug,
                "policy_score": float(policy_score_cpu[i].item()) if policy_score_cpu is not None else None,
                "route_id": route_id,
                "subject_mode_label": route_label,
                "route_probabilities": {
                    class_label(SUBJECT_MODE_VOCAB, idx): float(value)
                    for idx, value in enumerate(route_probs_cpu[i].tolist())
                },
                "subject_box": subject_box,
                "ranked_indices": ranked,
                "top_candidate": top,
                "top_candidates": [candidates[idx] for idx in ranked[: min(save_topk, len(ranked))]],
                "candidates": candidates,
                "positive_records": positives_ref,
                "best_positive": best_positive,
                "proposals": proposals,
                "proposal_top1_raw": raw_top_proposal,
                "proposal_top1_target_ar": target_ar_top_proposal,
            }
        )
    return rows


def per_record_metrics(
    pred: dict[str, Any],
    *,
    subject_valid_conf_threshold: float | None = None,
    subject_valid_policy: str | None = None,
) -> dict[str, float]:
    candidates = pred.get("candidates", [])
    if not candidates:
        return {}
    labels = [safe_float(c.get("score_target"), 0.0) for c in candidates]
    positives = [safe_float(c.get("positive_target"), 0.0) for c in candidates]
    scores = [safe_float(c.get("model_utility"), 0.0) for c in candidates]
    ranked = pred.get("ranked_indices") or []
    top_idx = int(ranked[0]) if ranked else 0
    best_positive = pred.get("best_positive")
    top = candidates[top_idx] if 0 <= top_idx < len(candidates) else {}
    top_box = top.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0]) if isinstance(top, dict) else [0.0, 0.0, 1.0, 1.0]
    best_box = best_positive.get("bbox_norm_xyxy") if isinstance(best_positive, dict) else None
    proposal_ious = []
    target_ar_proposal_ious = []
    if isinstance(best_box, list):
        proposal_ious = [box_iou_xyxy(p.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0]), best_box) for p in pred.get("proposals", [])]
        target_ar_props = [p for p in pred.get("proposals", []) if bool(p.get("target_ar_compatible", target_ar_value(pred.get("target_ar")) is None))]
        target_ar_proposal_ious = [box_iou_xyxy(p.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0]), best_box) for p in target_ar_props]
    raw_proposal_top = pred.get("proposal_top1_raw") if isinstance(pred.get("proposal_top1_raw"), dict) else None
    if raw_proposal_top is None:
        raw_proposal_top = (pred.get("proposals") or [{}])[0]
    raw_proposal_ar_error = safe_float(raw_proposal_top.get("target_ar_log_error"), 0.0) if isinstance(raw_proposal_top, dict) else 0.0
    raw_proposal_ar_compatible = float(bool(raw_proposal_top.get("target_ar_compatible", True))) if isinstance(raw_proposal_top, dict) else 0.0
    target_ar_top = pred.get("proposal_top1_target_ar") if isinstance(pred.get("proposal_top1_target_ar"), dict) else None
    target_ar_top_error = safe_float(target_ar_top.get("target_ar_log_error"), 0.0) if isinstance(target_ar_top, dict) else 0.0
    target_ar_top_compatible = float(bool(target_ar_top.get("target_ar_compatible", True))) if isinstance(target_ar_top, dict) else 0.0

    detailed = top.get("model_detailed_checklist") if isinstance(top, dict) else {}
    mode = str(pred.get("subject_mode_label", "other_ambiguous"))
    static_app = dict(zip(CHECKLIST_CLASS_KEYS, checklist_applicability_by_mode(mode)))
    mode_violations = 0
    mode_not_applicable = 0
    raw_not_applicable_fp = 0
    app_tp = app_fp = app_fn = 0
    agreement_ok = agreement_count = 0
    if isinstance(detailed, dict):
        target_app = top.get("checklist_applicability_target") if isinstance(top, dict) else None
        target_valid = top.get("checklist_applicability_valid") if isinstance(top, dict) else None
        class_targets = top.get("checklist_class_target") if isinstance(top, dict) else None
        class_valid = top.get("checklist_class_valid") if isinstance(top, dict) else None
        for key_idx, key in enumerate(CHECKLIST_CLASS_KEYS):
            item = detailed.get(key)
            if not isinstance(item, dict):
                continue
            mode_applicable = bool(static_app.get(key, 0.0) > 0.0)
            available = bool(item.get("available", False))
            model_applicable = bool(item.get("model_applicable", available))
            if not mode_applicable:
                mode_not_applicable += 1
                if available:
                    mode_violations += 1
                if model_applicable:
                    raw_not_applicable_fp += 1
            if isinstance(target_app, list) and isinstance(target_valid, list) and key_idx < len(target_app) and key_idx < len(target_valid) and safe_float(target_valid[key_idx]) > 0:
                target_bool = safe_float(target_app[key_idx]) > 0.0
                if model_applicable and target_bool:
                    app_tp += 1
                elif model_applicable and not target_bool:
                    app_fp += 1
                elif (not model_applicable) and target_bool:
                    app_fn += 1
            if (
                isinstance(class_targets, list)
                and isinstance(class_valid, list)
                and key_idx < len(class_targets)
                and key_idx < len(class_valid)
                and safe_float(class_valid[key_idx]) > 0
                and available
            ):
                vocab = CHECKLIST_CLASS_VOCABS[key]
                target_idx = int(max(0, min(len(vocab) - 1, int(safe_float(class_targets[key_idx], 0)))))
                agreement_count += 1
                agreement_ok += int(str(item.get("label")) == str(vocab[target_idx]))
    why_tags = top.get("model_why_tags") if isinstance(top, dict) else []
    person_tag_on_object = 0.0
    if mode in {"object_single", "object_multi"} and isinstance(why_tags, list):
        person_tag_on_object = float(any(isinstance(tag, dict) and str(tag.get("tag")) in PERSON_ONLY_WHY_TAGS for tag in why_tags))
    subject_box = pred.get("subject_box") if isinstance(pred.get("subject_box"), dict) else {}
    subject_target_valid = safe_float(subject_box.get("target_valid"), 0.0) if isinstance(subject_box, dict) else 0.0
    subject_pred = subject_box.get("predicted") if isinstance(subject_box, dict) else None
    subject_pred_conf = safe_float(subject_pred.get("confidence"), 0.0) if isinstance(subject_pred, dict) else 0.0
    subject_threshold = (
        float(subject_valid_conf_threshold)
        if subject_valid_conf_threshold is not None
        else safe_float(subject_pred.get("valid_threshold"), 0.5)
        if isinstance(subject_pred, dict)
        else 0.5
    )
    subject_policy = str(
        subject_valid_policy
        or (subject_pred.get("valid_policy") if isinstance(subject_pred, dict) else None)
        or "confidence"
    )
    subject_iou = safe_float(subject_box.get("iou_to_teacher"), 0.0) if isinstance(subject_box, dict) else 0.0
    subject_pred_positive = subject_valid_from_policy(
        confidence=subject_pred_conf,
        threshold=subject_threshold,
        route_label=str(pred.get("subject_mode_label", "")),
        policy=subject_policy,
    )
    subject_target_positive = subject_target_valid >= 0.5
    subject_extra_metrics: dict[str, float] = {}
    if subject_target_positive:
        subject_extra_metrics["subject_box_valid_positive_acc"] = float(subject_pred_positive)
    else:
        subject_extra_metrics["subject_box_valid_negative_acc"] = float(not subject_pred_positive)
    return {
        "candidate_top1_hit": float(positives[top_idx] > 0.0),
        "candidate_top1_exact_best": topk_exact_best(labels, scores, 1),
        "candidate_recall_at_3": topk_any_positive(positives, scores, 3),
        "candidate_recall_at_5": topk_any_positive(positives, scores, 5),
        "srcc": spearman_corr(scores, labels),
        "pcc": pearson_corr(scores, labels),
        "ndcg_at_5": ndcg_at_k(labels, scores, 5),
        "ndcg_at_10": ndcg_at_k(labels, scores, 10),
        "top1_iou_to_best_positive": box_iou_xyxy(top_box, best_box) if isinstance(best_box, list) else 0.0,
        "proposal_recall_at_1_iou_0_5": float(bool(proposal_ious[:1]) and max(proposal_ious[:1]) >= 0.5),
        "proposal_recall_at_5_iou_0_5": float(bool(proposal_ious[:5]) and max(proposal_ious[:5]) >= 0.5),
        "proposal_best_iou_at_5": max(proposal_ious[:5]) if proposal_ious else 0.0,
        "proposal_raw_top1_target_ar_log_error": raw_proposal_ar_error,
        "proposal_raw_top1_target_ar_compatible": raw_proposal_ar_compatible,
        "proposal_top1_target_ar_log_error": target_ar_top_error,
        "proposal_top1_target_ar_compatible": target_ar_top_compatible,
        "proposal_target_ar_selected_log_error": target_ar_top_error,
        "proposal_target_ar_selected_compatible": target_ar_top_compatible,
        "proposal_target_ar_recall_at_1_iou_0_5": float(bool(target_ar_proposal_ious[:1]) and max(target_ar_proposal_ious[:1]) >= 0.5),
        "proposal_target_ar_recall_at_5_iou_0_5": float(bool(target_ar_proposal_ious[:5]) and max(target_ar_proposal_ious[:5]) >= 0.5),
        "proposal_target_ar_best_iou_at_5": max(target_ar_proposal_ious[:5]) if target_ar_proposal_ious else 0.0,
        "explain_mode_consistency_violation_rate": float(mode_violations / max(1, mode_not_applicable)),
        "explain_not_applicable_false_positive_rate": float(raw_not_applicable_fp / max(1, mode_not_applicable)),
        "explain_applicability_precision": float(app_tp / max(1, app_tp + app_fp)),
        "explain_applicability_recall": float(app_tp / max(1, app_tp + app_fn)),
        "explain_label_agreement_when_applicable": float(agreement_ok / max(1, agreement_count)),
        "explain_person_tag_on_object_rate": person_tag_on_object,
        "subject_box_target_valid": subject_target_valid,
        "subject_box_pred_conf": subject_pred_conf,
        "subject_box_valid_threshold": subject_threshold,
        "subject_box_valid_acc": float(subject_pred_positive == subject_target_positive),
        "subject_box_iou_to_teacher": subject_iou if subject_target_valid > 0.0 else 0.0,
        **subject_extra_metrics,
    }


def summarize_metrics(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: mean([row[key] for row in rows if key in row]) for key in keys}


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
