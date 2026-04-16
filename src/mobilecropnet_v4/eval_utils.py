from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import torch

from mobilecropnet_v4.data import LetterboxTransform, box_iou_xyxy, letterbox_to_original_box, safe_float
from mobilecropnet_v4.model import MobileCropNetV4


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
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = dict(ckpt.get("model_config", {}))
    model = MobileCropNetV4(**config).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, ckpt


def infer_input_size(ckpt: dict[str, Any], explicit_input_size: int | None) -> int:
    if explicit_input_size is not None:
        return int(explicit_input_size)
    return int(ckpt.get("train_config", {}).get("input_size", 256))


def batch_predictions(
    *,
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    save_topk: int = 8,
) -> list[dict[str, Any]]:
    utility = torch.sigmoid(outputs["utility_logits"]).detach().cpu()
    positive = torch.sigmoid(outputs["positive_logits"]).detach().cpu()
    risk = torch.sigmoid(outputs["risk_logits"]).detach().cpu()
    proposal_prob = torch.sigmoid(outputs["proposal_logits"]).detach().cpu()
    proposal_boxes = outputs["proposal_boxes"].detach().cpu()
    decision_ids = outputs["decision_logits"].argmax(dim=1).detach().cpu()
    route_ids = outputs["route_logits"].argmax(dim=1).detach().cpu()
    rows: list[dict[str, Any]] = []
    for i in range(utility.shape[0]):
        valid_count = int(batch["valid"][i].sum().item())
        scores = [float(v) for v in utility[i, :valid_count].tolist()]
        ranked = sorted(range(valid_count), key=lambda idx: scores[idx], reverse=True)
        candidates = []
        for j in range(valid_count):
            record = dict(batch["candidate_records"][i][j])
            record["model_utility"] = scores[j]
            record["model_positive"] = float(positive[i, j].item())
            record["model_risk"] = float(risk[i, j].item())
            candidates.append(record)
        transform = batch["transform"][i]
        assert isinstance(transform, LetterboxTransform)
        proposals = []
        for q, prob in sorted(
            enumerate([float(v) for v in proposal_prob[i].tolist()]),
            key=lambda item: item[1],
            reverse=True,
        )[:save_topk]:
            padded_box = [float(v) for v in proposal_boxes[i, q].tolist()]
            proposals.append(
                {
                    "proposal_id": int(q),
                    "score": float(prob),
                    "bbox_letterbox_xyxy": padded_box,
                    "bbox_norm_xyxy": letterbox_to_original_box(padded_box, transform),
                }
            )
        top = candidates[ranked[0]] if ranked else {}
        positives_ref = list(batch["positive_records"][i])
        best_positive = max(positives_ref, key=lambda row: safe_float(row.get("score")), default=None)
        rows.append(
            {
                "image_id": batch["image_id"][i],
                "image_path": batch["image_path"][i],
                "target_ar": batch["target_ar"][i],
                "width": int(batch["width"][i]),
                "height": int(batch["height"][i]),
                "decision_id": int(decision_ids[i].item()),
                "route_id": int(route_ids[i].item()),
                "ranked_indices": ranked,
                "top_candidate": top,
                "top_candidates": [candidates[idx] for idx in ranked[: min(save_topk, len(ranked))]],
                "candidates": candidates,
                "positive_records": positives_ref,
                "best_positive": best_positive,
                "proposals": proposals,
            }
        )
    return rows


def per_record_metrics(pred: dict[str, Any]) -> dict[str, float]:
    candidates = pred.get("candidates", [])
    if not candidates:
        return {}
    labels = [safe_float(c.get("score_target"), 0.0) for c in candidates]
    positives = [safe_float(c.get("positive_target"), 0.0) for c in candidates]
    scores = [safe_float(c.get("model_utility"), 0.0) for c in candidates]
    ranked = pred.get("ranked_indices") or []
    top_idx = int(ranked[0]) if ranked else 0
    best_positive = pred.get("best_positive")
    top_box = candidates[top_idx].get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0])
    best_box = best_positive.get("bbox_norm_xyxy") if isinstance(best_positive, dict) else None
    proposal_ious = []
    if isinstance(best_box, list):
        proposal_ious = [box_iou_xyxy(p.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0]), best_box) for p in pred.get("proposals", [])]
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
    }


def summarize_metrics(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: mean([row[key] for row in rows if key in row]) for key in keys}


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
