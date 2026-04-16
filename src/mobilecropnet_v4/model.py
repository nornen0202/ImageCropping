from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mobilecropnet_v4.data import DECISION_VOCAB, SUBJECT_MODE_COUNT, TARGET_AR_VOCAB


class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1, groups: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )


class DepthwiseBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, *, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(in_ch, in_ch, stride=stride, groups=in_ch),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        self.use_residual = stride == 1 and in_ch == out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return x + y if self.use_residual else y


def _xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    w = (x2 - x1).clamp_min(1e-6)
    h = (y2 - y1).clamp_min(1e-6)
    return torch.stack([x1 + 0.5 * w, y1 + 0.5 * h, w, h], dim=-1)


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(dim=-1)
    w = w.clamp(1e-4, 1.0)
    h = h.clamp(1e-4, 1.0)
    return torch.stack(
        [
            (cx - 0.5 * w).clamp(0.0, 1.0),
            (cy - 0.5 * h).clamp(0.0, 1.0),
            (cx + 0.5 * w).clamp(0.0, 1.0),
            (cy + 0.5 * h).clamp(0.0, 1.0),
        ],
        dim=-1,
    )


def _box_iou_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax1, ay1, ax2, ay2 = a.unbind(dim=-1)
    bx1, by1, bx2, by2 = b.unbind(dim=-1)
    ix1 = torch.maximum(ax1.unsqueeze(-1), bx1.unsqueeze(-2))
    iy1 = torch.maximum(ay1.unsqueeze(-1), by1.unsqueeze(-2))
    ix2 = torch.minimum(ax2.unsqueeze(-1), bx2.unsqueeze(-2))
    iy2 = torch.minimum(ay2.unsqueeze(-1), by2.unsqueeze(-2))
    inter = (ix2 - ix1).clamp_min(0.0) * (iy2 - iy1).clamp_min(0.0)
    area_a = (ax2 - ax1).clamp_min(0.0) * (ay2 - ay1).clamp_min(0.0)
    area_b = (bx2 - bx1).clamp_min(0.0) * (by2 - by1).clamp_min(0.0)
    union = area_a.unsqueeze(-1) + area_b.unsqueeze(-2) - inter
    return inter / union.clamp_min(1e-6)


def _box_geom(boxes: torch.Tensor, is_base: torch.Tensor | None = None) -> torch.Tensor:
    cxcywh = _xyxy_to_cxcywh(boxes)
    area = (cxcywh[..., 2] * cxcywh[..., 3]).unsqueeze(-1)
    aspect = (cxcywh[..., 2] / cxcywh[..., 3].clamp_min(1e-6)).unsqueeze(-1)
    if is_base is None:
        is_base = torch.zeros_like(area.squeeze(-1))
    return torch.cat([boxes, cxcywh, area, aspect, is_base.unsqueeze(-1)], dim=-1)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def _masked_box_pool(features: torch.Tensor, boxes: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    bsz, channels, height, width = features.shape
    device = features.device
    dtype = features.dtype
    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / float(height)
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / float(width)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    xx = xx.view(1, 1, height, width)
    yy = yy.view(1, 1, height, width)
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    bw = (x2 - x1).clamp_min(1e-4)
    bh = (y2 - y1).clamp_min(1e-4)
    margin_x = (bw * 0.12).clamp_min(1.0 / float(width))
    margin_y = (bh * 0.12).clamp_min(1.0 / float(height))

    inside = (
        (xx >= x1.unsqueeze(-1).unsqueeze(-1))
        & (xx <= x2.unsqueeze(-1).unsqueeze(-1))
        & (yy >= y1.unsqueeze(-1).unsqueeze(-1))
        & (yy <= y2.unsqueeze(-1).unsqueeze(-1))
    )
    outer = (
        (xx >= (x1 - margin_x).clamp_min(0.0).unsqueeze(-1).unsqueeze(-1))
        & (xx <= (x2 + margin_x).clamp_max(1.0).unsqueeze(-1).unsqueeze(-1))
        & (yy >= (y1 - margin_y).clamp_min(0.0).unsqueeze(-1).unsqueeze(-1))
        & (yy <= (y2 + margin_y).clamp_max(1.0).unsqueeze(-1).unsqueeze(-1))
    )
    context = (
        (xx >= (x1 - 2.0 * margin_x).clamp_min(0.0).unsqueeze(-1).unsqueeze(-1))
        & (xx <= (x2 + 2.0 * margin_x).clamp_max(1.0).unsqueeze(-1).unsqueeze(-1))
        & (yy >= (y1 - 2.0 * margin_y).clamp_min(0.0).unsqueeze(-1).unsqueeze(-1))
        & (yy <= (y2 + 2.0 * margin_y).clamp_max(1.0).unsqueeze(-1).unsqueeze(-1))
    )
    masks = [
        inside.to(dtype),
        (outer & ~inside).to(dtype),
        (context & ~inside).to(dtype),
    ]
    feat = features.unsqueeze(1)
    valid_f = valid.view(bsz, -1, 1, 1).to(dtype)
    pooled: list[torch.Tensor] = []
    for mask in masks:
        mask = mask * valid_f
        pooled.append((feat * mask.unsqueeze(2)).sum(dim=(-2, -1)) / mask.sum(dim=(-2, -1)).clamp_min(1.0).unsqueeze(-1))
    return torch.cat(pooled, dim=-1)


@dataclass(frozen=True)
class MobileCropNetV4Config:
    candidate_k: int = 24
    proposal_q: int = 16
    width_mult: float = 0.75
    token_dim: int = 128
    route_classes: int = SUBJECT_MODE_COUNT
    decision_classes: int = len(DECISION_VOCAB)
    ar_classes: int = len(TARGET_AR_VOCAB)


class MobileCropNetV4(nn.Module):
    def __init__(
        self,
        *,
        candidate_k: int = 24,
        proposal_q: int = 16,
        width_mult: float = 0.75,
        token_dim: int = 128,
        route_classes: int = SUBJECT_MODE_COUNT,
        decision_classes: int = len(DECISION_VOCAB),
        ar_classes: int = len(TARGET_AR_VOCAB),
    ) -> None:
        super().__init__()
        c1 = max(8, int(16 * width_mult))
        c2 = max(12, int(24 * width_mult))
        c3 = max(20, int(40 * width_mult))
        c4 = max(32, int(72 * width_mult))
        self.config = MobileCropNetV4Config(
            candidate_k=int(candidate_k),
            proposal_q=int(proposal_q),
            width_mult=float(width_mult),
            token_dim=int(token_dim),
            route_classes=int(route_classes),
            decision_classes=int(decision_classes),
            ar_classes=int(ar_classes),
        )
        self.backbone = nn.Sequential(
            ConvBNAct(3, c1, stride=2),
            DepthwiseBlock(c1, c2, stride=2),
            DepthwiseBlock(c2, c3, stride=2),
            DepthwiseBlock(c3, c4, stride=2),
            DepthwiseBlock(c4, c4, stride=1),
            DepthwiseBlock(c4, c4, stride=1),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.ar_embed = nn.Embedding(ar_classes, 16)
        self.cond_proj = nn.Sequential(nn.Linear(c4 + 16 + 1, token_dim), nn.SiLU(inplace=True), nn.Linear(token_dim, token_dim))

        self.proposal_queries = nn.Parameter(torch.randn(proposal_q, token_dim) * 0.02)
        self.proposal_box_head = nn.Sequential(nn.Linear(token_dim, token_dim), nn.SiLU(inplace=True), nn.Linear(token_dim, 4), nn.Sigmoid())
        self.proposal_logit_head = nn.Sequential(nn.Linear(token_dim, token_dim // 2), nn.SiLU(inplace=True), nn.Linear(token_dim // 2, 1))
        self.proposal_token_head = nn.Sequential(nn.Linear(token_dim, token_dim), nn.SiLU(inplace=True), nn.Linear(token_dim, token_dim))

        cand_in = c4 * 3 + c4 + 16 + 1 + 11
        self.candidate_encoder = nn.Sequential(
            nn.Linear(cand_in, token_dim),
            nn.SiLU(inplace=True),
            nn.Linear(token_dim, token_dim),
            nn.SiLU(inplace=True),
        )
        self.relation_mlp = nn.Sequential(
            nn.Linear(token_dim * 2 + 7, token_dim),
            nn.SiLU(inplace=True),
            nn.Linear(token_dim, token_dim),
        )
        self.relation_norm = nn.LayerNorm(token_dim)
        self.utility_head = nn.Linear(token_dim, 1)
        self.positive_head = nn.Linear(token_dim, 1)
        self.risk_head = nn.Linear(token_dim, 1)
        self.macro_head = nn.Linear(token_dim, 4)
        self.route_head = nn.Sequential(nn.Linear(c4 + 16 + 1, token_dim // 2), nn.SiLU(inplace=True), nn.Linear(token_dim // 2, route_classes))
        self.policy_head = nn.Sequential(
            nn.Linear(c4 + 16 + 1 + token_dim + 2, token_dim),
            nn.SiLU(inplace=True),
            nn.Linear(token_dim, token_dim // 2),
            nn.SiLU(inplace=True),
        )
        self.decision_head = nn.Linear(token_dim // 2, decision_classes)
        self.delta_head = nn.Linear(token_dim // 2, 1)

    def _encode(self, image: torch.Tensor, target_ar_id: torch.Tensor, image_ar_log: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.backbone(image)
        global_feat = self.global_pool(features).flatten(1)
        ar_emb = self.ar_embed(target_ar_id.clamp(0, self.config.ar_classes - 1))
        image_ar = image_ar_log.float().view(-1, 1)
        cond = torch.cat([global_feat, ar_emb, image_ar], dim=-1)
        cond_token = self.cond_proj(cond)
        return features, global_feat, ar_emb, cond_token

    def _proposals(self, cond_token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = cond_token.shape[0]
        query = self.proposal_queries.unsqueeze(0).expand(bsz, -1, -1)
        token = query + cond_token.unsqueeze(1)
        proposal_token = self.proposal_token_head(token)
        box = _cxcywh_to_xyxy(self.proposal_box_head(token))
        logit = self.proposal_logit_head(proposal_token).squeeze(-1)
        return box, logit, proposal_token

    def _relation_refine(self, token: torch.Tensor, boxes: torch.Tensor, valid: torch.Tensor, is_base: torch.Tensor | None) -> torch.Tensor:
        bsz, k, dim = token.shape
        cxcywh = _xyxy_to_cxcywh(boxes)
        ci = cxcywh.unsqueeze(2)
        cj = cxcywh.unsqueeze(1)
        delta = torch.cat(
            [
                (ci[..., 0:1] - cj[..., 0:1]) / cj[..., 2:3].clamp_min(1e-4),
                (ci[..., 1:2] - cj[..., 1:2]) / cj[..., 3:4].clamp_min(1e-4),
                torch.log(ci[..., 2:3].clamp_min(1e-4) / cj[..., 2:3].clamp_min(1e-4)),
                torch.log(ci[..., 3:4].clamp_min(1e-4) / cj[..., 3:4].clamp_min(1e-4)),
            ],
            dim=-1,
        ).clamp(-5.0, 5.0)
        iou = _box_iou_torch(boxes, boxes).unsqueeze(-1)
        area_delta = ((ci[..., 2:3] * ci[..., 3:4]) - (cj[..., 2:3] * cj[..., 3:4])).clamp(-1.0, 1.0)
        if is_base is None:
            is_base_j = torch.zeros((bsz, 1, k, 1), dtype=token.dtype, device=token.device)
        else:
            is_base_j = is_base.view(bsz, 1, k, 1).to(token.dtype)
        pair_geom = torch.cat([delta, iou, area_delta, is_base_j.expand(bsz, k, k, 1)], dim=-1)
        ti = token.unsqueeze(2).expand(bsz, k, k, dim)
        tj = token.unsqueeze(1).expand(bsz, k, k, dim)
        rel = self.relation_mlp(torch.cat([ti, tj, pair_geom], dim=-1))
        pair_valid = (valid.unsqueeze(1) * valid.unsqueeze(2)).to(token.dtype).unsqueeze(-1)
        rel_mean = (rel * pair_valid).sum(dim=2) / pair_valid.sum(dim=2).clamp_min(1.0)
        return self.relation_norm(token + rel_mean)

    def forward(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
        target_ar_id: torch.Tensor | None = None,
        image_ar_log: torch.Tensor | None = None,
        candidate_is_base: torch.Tensor | None = None,
        box_meta: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        bsz = image.shape[0]
        if target_ar_id is None:
            target_ar_id = torch.zeros((bsz,), dtype=torch.long, device=image.device)
        if image_ar_log is None:
            image_ar_log = torch.zeros((bsz,), dtype=image.dtype, device=image.device)
        features, global_feat, ar_emb, cond_token = self._encode(image, target_ar_id, image_ar_log)
        proposal_boxes, proposal_logits, proposal_tokens = self._proposals(cond_token)

        if boxes is None:
            boxes = proposal_boxes
            valid = torch.ones((bsz, proposal_boxes.shape[1]), dtype=image.dtype, device=image.device)
            candidate_is_base = torch.zeros_like(valid)
        if valid is None:
            valid = torch.ones((bsz, boxes.shape[1]), dtype=image.dtype, device=image.device)
        if candidate_is_base is None:
            candidate_is_base = torch.zeros_like(valid)

        pooled = _masked_box_pool(features, boxes, valid)
        geom = box_meta if box_meta is not None else _box_geom(boxes, candidate_is_base)
        global_rep = global_feat.unsqueeze(1).expand(-1, boxes.shape[1], -1)
        ar_rep = ar_emb.unsqueeze(1).expand(-1, boxes.shape[1], -1)
        image_ar_rep = image_ar_log.float().view(bsz, 1, 1).expand(-1, boxes.shape[1], -1)
        cand_token = self.candidate_encoder(torch.cat([pooled, global_rep, ar_rep, image_ar_rep, geom], dim=-1))
        cand_token = self._relation_refine(cand_token, boxes, valid, candidate_is_base)
        utility_logits = self.utility_head(cand_token).squeeze(-1)
        positive_logits = self.positive_head(cand_token).squeeze(-1)
        risk_logits = self.risk_head(cand_token).squeeze(-1)

        masked_utility = utility_logits.masked_fill(valid <= 0, -1e4)
        attn = torch.softmax(masked_utility, dim=1) * valid.to(utility_logits.dtype)
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-6)
        weighted_token = (cand_token * attn.unsqueeze(-1)).sum(dim=1)
        base_mask = (candidate_is_base > 0).to(utility_logits.dtype) * valid.to(utility_logits.dtype)
        base_score = (torch.sigmoid(utility_logits) * base_mask).sum(dim=1, keepdim=True) / base_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        best_score = torch.sigmoid(masked_utility.max(dim=1, keepdim=True).values)
        global_cond = torch.cat([global_feat, ar_emb, image_ar_log.float().view(bsz, 1)], dim=-1)
        policy_token = self.policy_head(torch.cat([global_cond, weighted_token, base_score, best_score], dim=-1))
        macro_logits = self.macro_head(cand_token)

        return {
            "utility_logits": utility_logits,
            "positive_logits": positive_logits,
            "risk_logits": risk_logits,
            "macro_logits": macro_logits,
            "route_logits": self.route_head(global_cond),
            "decision_logits": self.decision_head(policy_token),
            "delta_pred": self.delta_head(policy_token).squeeze(-1),
            "proposal_boxes": proposal_boxes,
            "proposal_logits": proposal_logits,
            "proposal_tokens": proposal_tokens,
            "candidate_tokens": cand_token,
        }


def _listwise_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, temperature: float = 0.15) -> torch.Tensor:
    masked_logits = logits.masked_fill(valid <= 0, -1e4)
    log_prob = torch.log_softmax(masked_logits, dim=1)
    target_score = (target / float(temperature)).masked_fill(valid <= 0, -1e4)
    target_prob = torch.softmax(target_score, dim=1) * valid
    target_prob = target_prob / target_prob.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return -(target_prob * log_prob).sum(dim=1).mean()


def _pairwise_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, margin: float = 0.03) -> torch.Tensor:
    diff_target = target.unsqueeze(2) - target.unsqueeze(1)
    diff_pred = logits.unsqueeze(2) - logits.unsqueeze(1)
    pair_valid = (valid.unsqueeze(2) * valid.unsqueeze(1)) > 0
    pair_valid = pair_valid & (diff_target.abs() >= float(margin))
    if not torch.any(pair_valid):
        return logits.sum() * 0.0
    sign = torch.where(diff_target > 0, torch.ones_like(diff_target), -torch.ones_like(diff_target))
    return F.softplus(-sign[pair_valid] * diff_pred[pair_valid]).mean()


def _proposal_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    proposal_boxes = outputs["proposal_boxes"]
    proposal_logits = outputs["proposal_logits"]
    positive_boxes = batch["positive_boxes"].to(proposal_boxes.device)
    positive_valid = batch["positive_valid"].to(proposal_boxes.device)
    bsz, q, _ = proposal_boxes.shape
    _, p, _ = positive_boxes.shape
    if p == 0 or positive_valid.sum() <= 0:
        zero = proposal_logits.sum() * 0.0
        return zero, {"proposal_obj_loss": zero.detach(), "proposal_box_loss": zero.detach(), "proposal_recall_0_5": zero.detach()}

    iou = _box_iou_torch(proposal_boxes, positive_boxes)
    valid_pos = positive_valid.unsqueeze(1) > 0
    iou_masked = iou.masked_fill(~valid_pos, -1.0)
    max_iou_per_q, _ = iou_masked.max(dim=2)
    obj_target = (max_iou_per_q >= 0.5).to(proposal_logits.dtype)
    best_q_for_pos = iou_masked.argmax(dim=1)
    for b in range(bsz):
        for j in range(p):
            if positive_valid[b, j] > 0:
                obj_target[b, best_q_for_pos[b, j]] = 1.0
    obj_loss = F.binary_cross_entropy_with_logits(proposal_logits, obj_target)

    l1 = torch.cdist(proposal_boxes, positive_boxes, p=1)
    l1 = l1.masked_fill(~valid_pos, 1e4)
    min_l1 = l1.min(dim=1).values
    box_loss = (min_l1 * positive_valid).sum() / positive_valid.sum().clamp_min(1.0)

    max_iou_per_pos = iou_masked.max(dim=1).values.clamp_min(0.0)
    recall_05 = ((max_iou_per_pos >= 0.5).to(proposal_logits.dtype) * positive_valid).sum() / positive_valid.sum().clamp_min(1.0)

    prop_iou = _box_iou_torch(proposal_boxes, proposal_boxes)
    eye = torch.eye(q, dtype=torch.bool, device=proposal_boxes.device).view(1, q, q)
    diversity_loss = (prop_iou.masked_fill(eye, 0.0).clamp_min(0.7) - 0.7).mean()
    total = obj_loss + 2.0 * box_loss + 0.05 * diversity_loss
    return total, {
        "proposal_obj_loss": obj_loss.detach(),
        "proposal_box_loss": box_loss.detach(),
        "proposal_diversity_loss": diversity_loss.detach(),
        "proposal_recall_0_5": recall_05.detach(),
    }


def compute_mobilecropnet_v4_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    score_weight: float = 1.0,
    listwise_weight: float = 0.6,
    pairwise_weight: float = 0.5,
    positive_weight: float = 0.4,
    risk_weight: float = 0.2,
    macro_weight: float = 0.1,
    route_weight: float = 0.1,
    decision_weight: float = 0.25,
    delta_weight: float = 0.1,
    proposal_weight: float = 0.8,
) -> tuple[torch.Tensor, dict[str, float]]:
    valid = batch["valid"].to(outputs["utility_logits"].device).to(outputs["utility_logits"].dtype)
    score_target = batch["score_target"].to(outputs["utility_logits"].device).to(outputs["utility_logits"].dtype)
    positive_target = batch["positive_target"].to(outputs["utility_logits"].device).to(outputs["utility_logits"].dtype)
    risk_target = batch["risk_target"].to(outputs["utility_logits"].device).to(outputs["utility_logits"].dtype)

    score_bce = F.binary_cross_entropy_with_logits(outputs["utility_logits"], score_target, reduction="none")
    score_loss = _masked_mean(score_bce, valid)
    list_loss = _listwise_loss(outputs["utility_logits"], score_target, valid)
    pair_loss = _pairwise_loss(outputs["utility_logits"], score_target, valid)
    pos_loss = _masked_mean(F.binary_cross_entropy_with_logits(outputs["positive_logits"], positive_target, reduction="none"), valid)
    risk_loss = _masked_mean(F.binary_cross_entropy_with_logits(outputs["risk_logits"], risk_target, reduction="none"), valid)

    macro_valid = batch["macro_valid"].to(outputs["utility_logits"].device).to(outputs["utility_logits"].dtype)
    macro_target = batch["macro_target"].to(outputs["utility_logits"].device).to(outputs["utility_logits"].dtype)
    macro_loss = _masked_mean(F.smooth_l1_loss(torch.sigmoid(outputs["macro_logits"]), macro_target, reduction="none").mean(dim=-1), valid * macro_valid)

    route_loss = F.cross_entropy(outputs["route_logits"], batch["route_target"].to(outputs["route_logits"].device))
    decision_loss = F.cross_entropy(outputs["decision_logits"], batch["decision_target"].to(outputs["decision_logits"].device))
    delta_target = batch["delta_target"].to(outputs["delta_pred"].device).to(outputs["delta_pred"].dtype).clamp(-3.0, 3.0) / 3.0
    delta_loss = F.smooth_l1_loss(torch.tanh(outputs["delta_pred"]), delta_target)
    proposal_loss, prop_metrics = _proposal_loss(outputs, batch)

    total = (
        score_weight * score_loss
        + listwise_weight * list_loss
        + pairwise_weight * pair_loss
        + positive_weight * pos_loss
        + risk_weight * risk_loss
        + macro_weight * macro_loss
        + route_weight * route_loss
        + decision_weight * decision_loss
        + delta_weight * delta_loss
        + proposal_weight * proposal_loss
    )

    with torch.no_grad():
        prob = torch.sigmoid(outputs["utility_logits"])
        masked_prob = prob.masked_fill(valid <= 0, -1.0)
        top_idx = masked_prob.argmax(dim=1)
        top_positive = positive_target.gather(1, top_idx[:, None]).squeeze(1)
        best_idx = score_target.masked_fill(valid <= 0, -1.0).argmax(dim=1)
        top1_exact = (top_idx == best_idx).to(prob.dtype)
        score_mae = _masked_mean((prob - score_target).abs(), valid)
        metrics: dict[str, float] = {
            "loss": float(total.detach().cpu()),
            "score_loss": float(score_loss.detach().cpu()),
            "listwise_loss": float(list_loss.detach().cpu()),
            "pairwise_loss": float(pair_loss.detach().cpu()),
            "positive_loss": float(pos_loss.detach().cpu()),
            "risk_loss": float(risk_loss.detach().cpu()),
            "macro_loss": float(macro_loss.detach().cpu()),
            "route_loss": float(route_loss.detach().cpu()),
            "decision_loss": float(decision_loss.detach().cpu()),
            "delta_loss": float(delta_loss.detach().cpu()),
            "proposal_loss": float(proposal_loss.detach().cpu()),
            "score_mae": float(score_mae.detach().cpu()),
            "top1_hit": float(top_positive.mean().detach().cpu()),
            "top1_exact_best_score": float(top1_exact.mean().detach().cpu()),
            "decision_acc": float((outputs["decision_logits"].argmax(dim=1) == batch["decision_target"].to(outputs["decision_logits"].device)).float().mean().detach().cpu()),
            "route_acc": float((outputs["route_logits"].argmax(dim=1) == batch["route_target"].to(outputs["route_logits"].device)).float().mean().detach().cpu()),
        }
        for key, value in prop_metrics.items():
            metrics[key] = float(value.detach().cpu())
    return total, metrics


def model_config_to_dict(model: MobileCropNetV4) -> dict[str, Any]:
    cfg = model.config
    return {
        "candidate_k": cfg.candidate_k,
        "proposal_q": cfg.proposal_q,
        "width_mult": cfg.width_mult,
        "token_dim": cfg.token_dim,
        "route_classes": cfg.route_classes,
        "decision_classes": cfg.decision_classes,
        "ar_classes": cfg.ar_classes,
    }
