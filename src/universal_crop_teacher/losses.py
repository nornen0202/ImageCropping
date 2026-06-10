from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class UCTRLossWeights:
    listwise: float = 1.0
    pairwise: float = 0.45
    reg: float = 0.25
    gt_iou: float = 0.20
    top: float = 0.15
    uncertainty: float = 0.05
    safety: float = 0.05
    domain_aux: float = 0.20
    gate: float = 0.05


def _masked_softmax(values: torch.Tensor, mask: torch.Tensor, *, temperature: float) -> torch.Tensor:
    masked = values.masked_fill(~mask.bool(), -1e9)
    return torch.softmax(masked / max(1e-6, float(temperature)), dim=-1)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def pairwise_loss_for_batch(
    scores: torch.Tensor,
    pairwise: list[torch.Tensor],
    *,
    device: torch.device,
    row_indices: list[int] | None = None,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    selected_rows = list(range(len(pairwise))) if row_indices is None else [int(v) for v in row_indices]
    for out_idx, bidx in enumerate(selected_rows):
        pairs = pairwise[bidx]
        if pairs.numel() == 0:
            continue
        pairs = pairs.to(device)
        a_idx = pairs[:, 0].long()
        b_idx = pairs[:, 1].long()
        sign = pairs[:, 2].to(scores.dtype)
        weight = pairs[:, 3].to(scores.dtype).clamp_min(1e-6)
        diff = scores[out_idx, a_idx] - scores[out_idx, b_idx]
        losses.append(F.softplus(-sign * diff) * weight)
        weights.append(weight)
    if not losses:
        return scores.sum() * 0.0
    loss_cat = torch.cat(losses)
    weight_cat = torch.cat(weights)
    return loss_cat.sum() / weight_cat.sum().clamp_min(1.0)


def _select_rows(
    tensor: torch.Tensor,
    row_mask: torch.Tensor | None,
) -> torch.Tensor:
    if row_mask is None:
        return tensor
    return tensor[row_mask]


def _core_ranking_terms(
    logits: torch.Tensor,
    batch: dict[str, Any],
    *,
    target_scores: torch.Tensor,
    label_conf: torch.Tensor,
    mask: torch.Tensor,
    row_mask: torch.Tensor | None = None,
    list_temperature: float = 0.12,
) -> dict[str, torch.Tensor]:
    logits_sel = _select_rows(logits, row_mask)
    target_sel = _select_rows(target_scores, row_mask)
    conf_sel = _select_rows(label_conf, row_mask)
    mask_sel = _select_rows(mask, row_mask)
    if logits_sel.numel() == 0 or not bool(mask_sel.any()):
        zero = logits.sum() * 0.0
        return {"listwise": zero, "pairwise": zero, "reg": zero, "top": zero}

    if row_mask is None:
        row_indices = None
    else:
        row_indices = torch.nonzero(row_mask, as_tuple=False).flatten().tolist()
    target_prob = _masked_softmax(target_sel, mask_sel, temperature=float(list_temperature)).detach()
    pred_log_prob = torch.log_softmax(logits_sel.masked_fill(~mask_sel, -1e9), dim=-1)
    listwise = -(target_prob * pred_log_prob * mask_sel.to(pred_log_prob.dtype)).sum(dim=-1).mean()
    pairwise = pairwise_loss_for_batch(logits_sel, batch["pairwise"], device=logits.device, row_indices=row_indices)
    reg = _masked_mean(
        F.smooth_l1_loss(torch.sigmoid(logits_sel), target_sel, reduction="none") * (0.25 + 0.75 * conf_sel),
        mask_sel,
    )
    top_target = target_sel.masked_fill(~mask_sel, -1e9).argmax(dim=-1)
    top = F.cross_entropy(logits_sel, top_target)
    return {
        "listwise": listwise,
        "pairwise": pairwise,
        "reg": reg,
        "top": top,
    }


def compute_uctr_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    weights: UCTRLossWeights,
    list_temperature: float = 0.12,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = batch["valid_mask"].bool()
    target_scores = batch["target_scores"].to(outputs["utility_logits"].device).float()
    label_conf = batch["label_confidence"].to(outputs["utility_logits"].device).float().clamp(0.0, 1.0)
    gt_iou = batch["gt_iou"].to(outputs["utility_logits"].device).float().clamp(0.0, 1.0)
    utility = outputs["utility_logits"]
    dataset_ids = batch["dataset_ids"].to(outputs["utility_logits"].device).long()
    gaic_idx = 4

    core = _core_ranking_terms(
        utility,
        batch,
        target_scores=target_scores,
        label_conf=label_conf,
        mask=mask,
        list_temperature=float(list_temperature),
    )
    listwise = core["listwise"]
    pairwise = core["pairwise"]
    reg = core["reg"]
    top = core["top"]
    gt_iou_loss = _masked_mean(F.smooth_l1_loss(torch.sigmoid(outputs["gt_iou_logits"]), gt_iou, reduction="none"), mask)

    uncertainty_target = (1.0 - label_conf).detach()
    uncertainty = _masked_mean(
        F.binary_cross_entropy_with_logits(outputs["uncertainty_logits"], uncertainty_target, reduction="none"),
        mask,
    )
    safety_target = (1.0 - gt_iou).clamp(0.0, 1.0).detach()
    safety = _masked_mean(F.binary_cross_entropy_with_logits(outputs["safety_logits"], safety_target, reduction="none"), mask)

    public_row_mask = dataset_ids != int(gaic_idx)
    gaic_row_mask = dataset_ids == int(gaic_idx)
    public_aux_terms = _core_ranking_terms(
        outputs["public_utility_logits"],
        batch,
        target_scores=target_scores,
        label_conf=label_conf,
        mask=mask,
        row_mask=public_row_mask,
        list_temperature=float(list_temperature),
    )
    gaic_aux_terms = _core_ranking_terms(
        outputs["gaic_utility_logits"],
        batch,
        target_scores=target_scores,
        label_conf=label_conf,
        mask=mask,
        row_mask=gaic_row_mask,
        list_temperature=float(list_temperature),
    )
    domain_aux = (
        0.60 * public_aux_terms["listwise"]
        + 0.60 * public_aux_terms["pairwise"]
        + 0.35 * public_aux_terms["reg"]
        + 0.20 * public_aux_terms["top"]
        + 0.80 * gaic_aux_terms["listwise"]
        + 0.80 * gaic_aux_terms["pairwise"]
        + 0.20 * gaic_aux_terms["reg"]
        + 0.30 * gaic_aux_terms["top"]
    )
    gate_target = (dataset_ids == int(gaic_idx)).float().unsqueeze(1).expand_as(mask).to(utility.device)
    gate = outputs.get("domain_gate_logits")
    if gate is None:
        gate_loss = utility.sum() * 0.0
    else:
        gate_loss = _masked_mean(F.binary_cross_entropy_with_logits(gate, gate_target, reduction="none"), mask)

    total = (
        float(weights.listwise) * listwise
        + float(weights.pairwise) * pairwise
        + float(weights.reg) * reg
        + float(weights.gt_iou) * gt_iou_loss
        + float(weights.top) * top
        + float(weights.uncertainty) * uncertainty
        + float(weights.safety) * safety
        + float(weights.domain_aux) * domain_aux
        + float(weights.gate) * gate_loss
    )
    metrics = {
        "loss": float(total.detach().cpu()),
        "listwise": float(listwise.detach().cpu()),
        "pairwise": float(pairwise.detach().cpu()),
        "reg": float(reg.detach().cpu()),
        "gt_iou": float(gt_iou_loss.detach().cpu()),
        "top": float(top.detach().cpu()),
        "uncertainty": float(uncertainty.detach().cpu()),
        "safety": float(safety.detach().cpu()),
        "domain_aux": float(domain_aux.detach().cpu()),
        "gate": float(gate_loss.detach().cpu()),
    }
    return total, metrics
