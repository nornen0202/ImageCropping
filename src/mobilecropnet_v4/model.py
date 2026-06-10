from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from mobilecropnet_v4.data import (
    CHECKLIST_CLASS_COUNT,
    CHECKLIST_CLASS_TOTAL,
    CHECKLIST_CLASS_SIZES,
    CHECKLIST_SCORE_COUNT,
    DECISION_VOCAB,
    SUBJECT_MODE_VOCAB,
    SUBJECT_MODE_COUNT,
    TARGET_AR_VOCAB,
    WHY_TAG_COUNT,
)


_TARGET_AR_VALUE_BY_ID = torch.tensor([0.0, 1.0, 9.0 / 16.0, 16.0 / 9.0, 3.0 / 4.0, 4.0 / 3.0], dtype=torch.float32)


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


def _cxcywh_to_xyxy_fit(cx: torch.Tensor, cy: torch.Tensor, w: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    w = w.clamp(1e-4, 1.0)
    h = h.clamp(1e-4, 1.0)
    half_w = 0.5 * w
    half_h = 0.5 * h
    cx = torch.maximum(torch.minimum(cx, 1.0 - half_w), half_w)
    cy = torch.maximum(torch.minimum(cy, 1.0 - half_h), half_h)
    return torch.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dim=-1)


def _letterbox_content_rect(image_ar_log: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    image_ar = torch.exp(image_ar_log.to(device=device, dtype=dtype)).clamp(1e-4, 1e4).view(-1, 1)
    one = torch.ones_like(image_ar)
    wide = image_ar >= 1.0
    content_w = torch.where(wide, one, image_ar).clamp(1e-4, 1.0)
    content_h = torch.where(wide, 1.0 / image_ar, one).clamp(1e-4, 1.0)
    x0 = 0.5 * (1.0 - content_w)
    y0 = 0.5 * (1.0 - content_h)
    return x0, y0, content_w, content_h


def _rect_from_content_box(
    letterbox_content_box: torch.Tensor | None,
    *,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if letterbox_content_box is None:
        x0 = torch.zeros((batch_size, 1), dtype=dtype, device=device)
        y0 = torch.zeros((batch_size, 1), dtype=dtype, device=device)
        w = torch.ones((batch_size, 1), dtype=dtype, device=device)
        h = torch.ones((batch_size, 1), dtype=dtype, device=device)
        return x0, y0, w, h
    content = letterbox_content_box.to(device=device, dtype=dtype).view(batch_size, 4)
    x0 = content[:, 0:1].clamp(0.0, 1.0)
    y0 = content[:, 1:2].clamp(0.0, 1.0)
    x1 = content[:, 2:3].clamp(0.0, 1.0)
    y1 = content[:, 3:4].clamp(0.0, 1.0)
    return x0, y0, (x1 - x0).clamp(1e-4, 1.0), (y1 - y0).clamp(1e-4, 1.0)


def _fit_cxcywh_in_rect(
    cx: torch.Tensor,
    cy: torch.Tensor,
    w: torch.Tensor,
    h: torch.Tensor,
    *,
    rect_x0: torch.Tensor,
    rect_y0: torch.Tensor,
    rect_w: torch.Tensor,
    rect_h: torch.Tensor,
) -> torch.Tensor:
    w = torch.minimum(w.clamp_min(1e-4), rect_w.clamp_min(1e-4))
    h = torch.minimum(h.clamp_min(1e-4), rect_h.clamp_min(1e-4))
    half_w = 0.5 * w
    half_h = 0.5 * h
    min_cx = rect_x0 + half_w
    max_cx = rect_x0 + rect_w - half_w
    min_cy = rect_y0 + half_h
    max_cy = rect_y0 + rect_h - half_h
    cx = torch.maximum(torch.minimum(cx, max_cx), min_cx)
    cy = torch.maximum(torch.minimum(cy, max_cy), min_cy)
    return torch.cat([cx - half_w, cy - half_h, cx + half_w, cy + half_h], dim=-1)


def _runtime_baseline_boxes(
    target_ar_id: torch.Tensor,
    image_ar_log: torch.Tensor,
    letterbox_content_box: torch.Tensor | None,
) -> torch.Tensor:
    batch_size = target_ar_id.shape[0]
    dtype = image_ar_log.dtype
    device = image_ar_log.device
    rect_x0, rect_y0, rect_w, rect_h = _rect_from_content_box(
        letterbox_content_box,
        batch_size=batch_size,
        dtype=dtype,
        device=device,
    )
    full = torch.cat([rect_x0, rect_y0, rect_x0 + rect_w, rect_y0 + rect_h], dim=-1)
    target_ar = _TARGET_AR_VALUE_BY_ID.to(device=device, dtype=dtype)[target_ar_id.clamp(0, len(TARGET_AR_VOCAB) - 1)].view(-1, 1)
    free_mask = target_ar <= 0.0
    safe_target_ar = target_ar.clamp_min(1e-4)
    use_full_height = (rect_h * safe_target_ar) <= rect_w
    crop_w = torch.where(use_full_height, rect_h * safe_target_ar, rect_w)
    crop_h = torch.where(use_full_height, rect_h, rect_w / safe_target_ar)
    cx = rect_x0 + 0.5 * rect_w
    cy = rect_y0 + 0.5 * rect_h
    minimal = _fit_cxcywh_in_rect(cx, cy, crop_w, crop_h, rect_x0=rect_x0, rect_y0=rect_y0, rect_w=rect_w, rect_h=rect_h)
    minimal = torch.where(free_mask.expand_as(minimal), full, minimal)
    return torch.stack([full, minimal], dim=1)


def _decode_subject_box(
    raw_box: torch.Tensor,
    *,
    coord_space: str,
    letterbox_content_box: torch.Tensor | None = None,
) -> torch.Tensor:
    batch_size = raw_box.shape[0]
    rect_x0, rect_y0, rect_w, rect_h = _rect_from_content_box(
        letterbox_content_box if str(coord_space) == "content" else None,
        batch_size=batch_size,
        dtype=raw_box.dtype,
        device=raw_box.device,
    )
    cx_raw, cy_raw, w_raw, h_raw = raw_box.unbind(dim=-1)
    w = w_raw.unsqueeze(-1).clamp(1e-4, 1.0) * rect_w
    h = h_raw.unsqueeze(-1).clamp(1e-4, 1.0) * rect_h
    span_x = (rect_w - w).clamp_min(0.0)
    span_y = (rect_h - h).clamp_min(0.0)
    cx = rect_x0 + 0.5 * w + cx_raw.unsqueeze(-1).clamp(0.0, 1.0) * span_x
    cy = rect_y0 + 0.5 * h + cy_raw.unsqueeze(-1).clamp(0.0, 1.0) * span_y
    return _fit_cxcywh_in_rect(cx, cy, w, h, rect_x0=rect_x0, rect_y0=rect_y0, rect_w=rect_w, rect_h=rect_h)


def _apply_subject_box_delta(
    anchor_box: torch.Tensor,
    delta: torch.Tensor,
    *,
    coord_space: str,
    letterbox_content_box: torch.Tensor | None = None,
    translate_scale: float = 0.35,
    size_scale: float = 0.35,
) -> torch.Tensor:
    anchor_cxcywh = _xyxy_to_cxcywh(anchor_box)
    dx, dy, dw, dh = delta.unbind(dim=-1)
    cx = anchor_cxcywh[:, 0:1] + dx.unsqueeze(-1) * anchor_cxcywh[:, 2:3] * float(translate_scale)
    cy = anchor_cxcywh[:, 1:2] + dy.unsqueeze(-1) * anchor_cxcywh[:, 3:4] * float(translate_scale)
    w = anchor_cxcywh[:, 2:3] * torch.exp(dw.unsqueeze(-1).clamp(-4.0, 4.0) * float(size_scale))
    h = anchor_cxcywh[:, 3:4] * torch.exp(dh.unsqueeze(-1).clamp(-4.0, 4.0) * float(size_scale))
    rect_x0, rect_y0, rect_w, rect_h = _rect_from_content_box(
        letterbox_content_box if str(coord_space) == "content" else None,
        batch_size=anchor_box.shape[0],
        dtype=anchor_box.dtype,
        device=anchor_box.device,
    )
    return _fit_cxcywh_in_rect(cx, cy, w, h, rect_x0=rect_x0, rect_y0=rect_y0, rect_w=rect_w, rect_h=rect_h)


def _ar_constrained_proposal_boxes(
    raw: torch.Tensor,
    target_ar_id: torch.Tensor,
    ar_classes: int,
    image_ar_log: torch.Tensor | None = None,
    letterbox_content_box: torch.Tensor | None = None,
) -> torch.Tensor:
    free_boxes = _cxcywh_to_xyxy(raw)
    values = _TARGET_AR_VALUE_BY_ID.to(device=raw.device, dtype=raw.dtype)
    if int(ar_classes) != int(values.numel()):
        padded = torch.zeros((int(ar_classes),), device=raw.device, dtype=raw.dtype)
        n = min(int(ar_classes), int(values.numel()))
        padded[:n] = values[:n]
        values = padded
    ar_id = target_ar_id.long().clamp(0, int(values.numel()) - 1)
    ratio = values[ar_id].view(-1, 1).clamp_min(1e-4)
    fixed = (ar_id > 0).view(-1, 1, 1)
    cx_raw, cy_raw, scale, _unused = raw.unbind(dim=-1)
    if letterbox_content_box is not None:
        content = letterbox_content_box.to(device=raw.device, dtype=raw.dtype).view(raw.shape[0], 4)
        x0 = content[:, 0:1].clamp(0.0, 1.0)
        y0 = content[:, 1:2].clamp(0.0, 1.0)
        x1 = content[:, 2:3].clamp(0.0, 1.0)
        y1 = content[:, 3:4].clamp(0.0, 1.0)
        content_w = (x1 - x0).clamp(1e-4, 1.0)
        content_h = (y1 - y0).clamp(1e-4, 1.0)
    elif image_ar_log is None:
        x0 = y0 = torch.zeros((raw.shape[0], 1), device=raw.device, dtype=raw.dtype)
        content_w = content_h = torch.ones((raw.shape[0], 1), device=raw.device, dtype=raw.dtype)
    else:
        x0, y0, content_w, content_h = _letterbox_content_rect(image_ar_log, dtype=raw.dtype, device=raw.device)
    box_ratio = ratio
    if image_ar_log is not None:
        image_ar = torch.exp(image_ar_log.to(device=raw.device, dtype=raw.dtype)).clamp(1e-4, 1e4).view(-1, 1)
        content_ratio = (content_w / content_h.clamp_min(1e-4)).clamp(1e-4, 1e4)
        box_ratio = (ratio * content_ratio / image_ar).clamp(1e-4, 1e4)
    max_w = torch.minimum(content_w, box_ratio * content_h).expand_as(scale).clamp_min(1e-4)
    max_h = (max_w / box_ratio.expand_as(scale).clamp_min(1e-4)).clamp_min(1e-4)
    scale = scale.clamp(1e-4, 1.0)
    w = (max_w * scale).clamp_min(1e-4)
    h = (max_h * scale).clamp_min(1e-4)
    half_w = 0.5 * w
    half_h = 0.5 * h
    span_x = (content_w.expand_as(scale) - w).clamp_min(0.0)
    span_y = (content_h.expand_as(scale) - h).clamp_min(0.0)
    cx = x0.expand_as(scale) + half_w + cx_raw * span_x
    cy = y0.expand_as(scale) + half_h + cy_raw * span_y
    fixed_boxes = _cxcywh_to_xyxy_fit(cx, cy, w, h)
    return torch.where(fixed.expand_as(free_boxes), fixed_boxes, free_boxes)


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


def _box_ciou_loss(pred_box: torch.Tensor, target_box: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred_cxcywh = _xyxy_to_cxcywh(pred_box)
    target_cxcywh = _xyxy_to_cxcywh(target_box)
    iou = _box_iou_torch(pred_box.unsqueeze(1), target_box.unsqueeze(1)).view(-1).clamp(0.0, 1.0)
    center_dist = ((pred_cxcywh[:, :2] - target_cxcywh[:, :2]) ** 2).sum(dim=-1)
    enc_x1 = torch.minimum(pred_box[:, 0], target_box[:, 0])
    enc_y1 = torch.minimum(pred_box[:, 1], target_box[:, 1])
    enc_x2 = torch.maximum(pred_box[:, 2], target_box[:, 2])
    enc_y2 = torch.maximum(pred_box[:, 3], target_box[:, 3])
    enc_diag = ((enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2).clamp_min(1e-6)
    pred_ratio = torch.atan(pred_cxcywh[:, 2] / pred_cxcywh[:, 3].clamp_min(1e-6))
    target_ratio = torch.atan(target_cxcywh[:, 2] / target_cxcywh[:, 3].clamp_min(1e-6))
    v = (4.0 / float(torch.pi**2)) * (target_ratio - pred_ratio) ** 2
    alpha = v / (1.0 - iou + v).clamp_min(1e-6)
    ciou = iou - center_dist / enc_diag - alpha * v
    return (1.0 - ciou.clamp(-1.0, 1.0)).clamp_min(0.0), iou


def _box_geom(boxes: torch.Tensor, is_base: torch.Tensor | None = None) -> torch.Tensor:
    cxcywh = _xyxy_to_cxcywh(boxes)
    area = (cxcywh[..., 2] * cxcywh[..., 3]).unsqueeze(-1)
    aspect = (cxcywh[..., 2] / cxcywh[..., 3].clamp_min(1e-6)).unsqueeze(-1)
    if is_base is None:
        is_base = torch.zeros_like(area.squeeze(-1))
    return torch.cat([boxes, cxcywh, area, aspect, is_base.unsqueeze(-1)], dim=-1)


def _box_feature_vector(boxes: torch.Tensor) -> torch.Tensor:
    cxcywh = _xyxy_to_cxcywh(boxes)
    area = (cxcywh[..., 2] * cxcywh[..., 3]).unsqueeze(-1)
    aspect = (cxcywh[..., 2] / cxcywh[..., 3].clamp_min(1e-6)).unsqueeze(-1)
    return torch.cat([boxes, cxcywh, area, aspect], dim=-1)


def _subject_prior_feature(boxes: torch.Tensor, valid: torch.Tensor, reliability: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [
            _box_feature_vector(boxes),
            valid.view(valid.shape[0], 1),
            reliability.view(reliability.shape[0], 1),
        ],
        dim=-1,
    )


_ROUTE_OBJECT_SINGLE_IDX = SUBJECT_MODE_VOCAB.index("object_single")
_ROUTE_MARGIN_MODE_WEIGHTS = {
    SUBJECT_MODE_VOCAB.index("portrait_single"): 1.0,
    SUBJECT_MODE_VOCAB.index("portrait_group"): 1.0,
    SUBJECT_MODE_VOCAB.index("object_multi"): 0.6,
    SUBJECT_MODE_VOCAB.index("scene_general"): 0.4,
}
_ROUTE_KIND_GROUPS = (
    ("background", ("background_texture_copyspace",)),
    ("object", ("object_multi", "object_single")),
    ("portrait", ("portrait_group", "portrait_single")),
    ("scene", ("scene_general",)),
    ("other", ("other_ambiguous",)),
)
_ROUTE_CARDINALITY_GROUPS = (
    ("multi", ("object_multi", "portrait_group")),
    ("single", ("object_single", "portrait_single")),
)


def _route_group_target_map(
    groups: tuple[tuple[str, tuple[str, ...]], ...],
    *,
    class_count: int,
    device: torch.device,
) -> torch.Tensor:
    target_map = torch.full((class_count,), -1, dtype=torch.long, device=device)
    for group_idx, (_, names) in enumerate(groups):
        for name in names:
            if name not in SUBJECT_MODE_VOCAB:
                continue
            class_idx = SUBJECT_MODE_VOCAB.index(name)
            if class_idx < class_count:
                target_map[int(class_idx)] = int(group_idx)
    return target_map


def _route_group_aux_loss(
    route_logits: torch.Tensor,
    route_target: torch.Tensor,
    groups: tuple[tuple[str, tuple[str, ...]], ...],
    *,
    metric_prefix: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = route_logits.sum() * 0.0
    class_count = int(route_logits.shape[1])
    target_map = _route_group_target_map(groups, class_count=class_count, device=route_logits.device)
    group_logits: list[torch.Tensor] = []
    for group_idx, (_, names) in enumerate(groups):
        indices = [SUBJECT_MODE_VOCAB.index(name) for name in names if name in SUBJECT_MODE_VOCAB and SUBJECT_MODE_VOCAB.index(name) < class_count]
        if indices:
            idx = torch.tensor(indices, dtype=torch.long, device=route_logits.device)
            group_logits.append(route_logits.index_select(1, idx).logsumexp(dim=1))
        else:
            group_logits.append(route_logits.new_full((route_logits.shape[0],), -1.0e4))
    target = target_map[route_target.clamp(0, class_count - 1)]
    valid = target >= 0
    if not bool(valid.any().item()):
        return zero, {
            f"{metric_prefix}_loss": zero.detach(),
            f"{metric_prefix}_acc": zero.detach(),
            f"{metric_prefix}_valid_frac": zero.detach(),
        }
    logits = torch.stack(group_logits, dim=1)
    loss = F.cross_entropy(logits[valid], target[valid])
    pred = logits.argmax(dim=1)
    acc = (pred[valid] == target[valid]).to(route_logits.dtype).mean()
    return loss, {
        f"{metric_prefix}_loss": loss.detach(),
        f"{metric_prefix}_acc": acc.detach(),
        f"{metric_prefix}_valid_frac": valid.to(route_logits.dtype).mean().detach(),
    }


def _route_aux_head_loss(
    logits: torch.Tensor,
    route_target: torch.Tensor,
    groups: tuple[tuple[str, tuple[str, ...]], ...],
    *,
    metric_prefix: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = logits.sum() * 0.0
    target_map = _route_group_target_map(groups, class_count=SUBJECT_MODE_COUNT, device=logits.device)
    target = target_map[route_target.clamp(0, SUBJECT_MODE_COUNT - 1)]
    valid = target >= 0
    if not bool(valid.any().item()):
        return zero, {
            f"{metric_prefix}_loss": zero.detach(),
            f"{metric_prefix}_acc": zero.detach(),
            f"{metric_prefix}_valid_frac": zero.detach(),
        }
    loss = F.cross_entropy(logits[valid], target[valid])
    pred = logits.argmax(dim=1)
    acc = (pred[valid] == target[valid]).to(logits.dtype).mean()
    return loss, {
        f"{metric_prefix}_loss": loss.detach(),
        f"{metric_prefix}_acc": acc.detach(),
        f"{metric_prefix}_valid_frac": valid.to(logits.dtype).mean().detach(),
    }


def _compose_hierarchical_route_logits(
    fine_logits: torch.Tensor,
    kind_logits: torch.Tensor | None,
    cardinality_logits: torch.Tensor | None,
    *,
    kind_weight: float,
    cardinality_weight: float,
) -> torch.Tensor:
    if kind_logits is None and cardinality_logits is None:
        return fine_logits
    out = fine_logits.float()
    if kind_logits is not None and float(kind_weight) != 0.0:
        kind_log_prob = F.log_softmax(kind_logits.float(), dim=1)
        kind_map = _route_group_target_map(_ROUTE_KIND_GROUPS, class_count=fine_logits.shape[1], device=fine_logits.device)
        mapped = out.new_zeros(out.shape)
        valid = kind_map >= 0
        if bool(valid.any().item()):
            mapped[:, valid] = kind_log_prob.index_select(1, kind_map[valid])
        out = out + float(kind_weight) * mapped
    if cardinality_logits is not None and float(cardinality_weight) != 0.0:
        card_log_prob = F.log_softmax(cardinality_logits.float(), dim=1)
        card_map = _route_group_target_map(_ROUTE_CARDINALITY_GROUPS, class_count=fine_logits.shape[1], device=fine_logits.device)
        mapped = out.new_zeros(out.shape)
        valid = card_map >= 0
        if bool(valid.any().item()):
            mapped[:, valid] = card_log_prob.index_select(1, card_map[valid])
        out = out + float(cardinality_weight) * mapped
    return out.to(dtype=fine_logits.dtype)


def _route_object_single_margin_loss(
    route_logits: torch.Tensor,
    route_target: torch.Tensor,
    *,
    margin: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = route_logits.sum() * 0.0
    sample_weight = torch.zeros_like(route_target, dtype=route_logits.dtype)
    for class_idx, weight in _ROUTE_MARGIN_MODE_WEIGHTS.items():
        sample_weight = sample_weight + (route_target == int(class_idx)).to(route_logits.dtype) * float(weight)
    active = sample_weight > 0
    if not bool(active.any()):
        return zero, {
            "route_object_single_margin_loss": zero.detach(),
            "route_object_single_margin_active_frac": zero.detach(),
            "route_object_single_margin_violation_rate": zero.detach(),
            "route_object_single_margin_violation_mean": zero.detach(),
        }
    true_logits = route_logits.gather(1, route_target.view(-1, 1)).squeeze(1)
    object_single_logits = route_logits[:, _ROUTE_OBJECT_SINGLE_IDX]
    violation = F.relu(float(margin) + object_single_logits - true_logits)
    active_violation = violation[active]
    active_weight = sample_weight[active]
    loss = (active_violation * active_weight).sum() / active_weight.sum().clamp_min(1.0)
    return loss, {
        "route_object_single_margin_loss": loss.detach(),
        "route_object_single_margin_active_frac": active.to(route_logits.dtype).mean().detach(),
        "route_object_single_margin_violation_rate": (active_violation > 0).to(route_logits.dtype).mean().detach(),
        "route_object_single_margin_violation_mean": active_violation.mean().detach(),
    }


def _decision_logits_from_policy_score(score: torch.Tensor, *, low: float, high: float) -> torch.Tensor:
    low_t = score.new_tensor(float(low))
    high_t = score.new_tensor(float(high))
    mid = 0.5 * (low_t + high_t)
    span = max(1e-4, float(high) - float(low))
    keep = low_t - score
    crop = score - high_t
    minimal = 1.0 - (score - mid).abs() / span
    return torch.stack([keep, minimal, crop], dim=-1)


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


def _local_timm_weight_file(backbone_name: str) -> Path | None:
    root = Path(os.environ.get("MCN_TIMM_LOCAL_WEIGHT_ROOT", "weights/hf/timm"))
    name = str(backbone_name or "").strip()
    if not name:
        return None
    candidates = [
        root / name / "model.safetensors",
        root / name / "pytorch_model.bin",
        root / name / "model.bin",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


@dataclass(frozen=True)
class MobileCropNetV4Config:
    candidate_k: int = 24
    proposal_q: int = 16
    input_size: int = 0
    width_mult: float = 0.75
    token_dim: int = 128
    backbone_name: str = "custom_depthwise"
    backbone_pretrained: bool = False
    ranker_type: str = "relation_lite"
    ranker_depth: int = 1
    route_classes: int = SUBJECT_MODE_COUNT
    decision_classes: int = len(DECISION_VOCAB)
    ar_classes: int = len(TARGET_AR_VOCAB)
    use_subject_prior: bool = False
    route_image_only: bool = False
    route_use_subject_prior: bool = False
    route_use_candidate_context: bool = False
    route_use_subject_box_features: bool = False
    route_use_subject_spatial_token: bool = False
    route_head_depth: int = 1
    route_head_hidden_mult: float = 0.5
    route_head_dropout: float = 0.0
    route_aux_heads: bool = False
    route_image_residual: bool = False
    route_image_residual_weight: float = 1.0
    route_decode_mode: str = "fine"
    route_decode_kind_weight: float = 1.0
    route_decode_cardinality_weight: float = 1.0
    policy_use_subject_prior: bool = False
    proposal_use_subject_prior: bool = False
    subject_box_head: bool = False
    subject_box_spatial_head: bool = False
    subject_box_spatial_multiscale: bool = False
    subject_box_spatial_mix_bias: float = -1.0
    subject_box_spatial_output: str = "mix"
    subject_box_spatial_box_mode: str = "regress"
    subject_box_spatial_extent_scale: float = 1.0
    subject_box_coord_space: str = "letterbox"
    subject_box_refine_with_proposals: bool = False
    subject_valid_route_calibrator: bool = False
    subject_valid_route_calibrator_hidden_mult: float = 0.5
    subject_valid_route_calibrator_detach: bool = True
    use_pred_subject_box_as_prior: bool = False
    detach_pred_subject_prior: bool = True
    policy_score_head: bool = False
    policy_score_threshold_low: float = 0.35
    policy_score_threshold_high: float = 0.70
    decision_source_head: bool = False
    decision_source_logit_threshold: float = 0.0
    decision_source_pair_head: bool = False
    decision_source_pair_hidden_mult: float = 0.5
    decision_source_pair_after_return_deltas: bool = False
    decision_source_pair_subject_state: bool = False
    decision_source_pair_subject_detach: bool = True
    route_condition_candidate_scores: bool = False
    route_condition_detach: bool = True
    return_score_head: bool = False
    return_score_head_depth: int = 1
    return_score_head_hidden_mult: float = 0.5
    return_score_action_source_bias: bool = False
    return_score_decision_source_bias: bool = False
    return_score_decision_source_detach: bool = True
    return_score_decision_source_max_scale: float = 2.0
    return_score_decision_source_from_source_head: bool = False
    return_score_decision_conditioned: bool = False
    return_score_decision_conditioned_mask_value: float = -10000.0
    return_score_decision_conditioned_from_source_head: bool = False
    return_score_source_mixture: bool = False
    return_score_source_mixture_strength: float = 1.0
    return_score_source_mixture_detach: bool = True
    return_score_source_mixture_from_source_head: bool = False
    return_score_source_mixture_gap_threshold: float | None = None
    return_score_source_mixture_gap_mask_value: float = -10000.0
    return_score_source_gate: bool = False
    return_score_source_gate_hidden_mult: float = 0.5
    return_score_source_gate_strength: float = 1.0
    return_score_source_gate_detach: bool = True
    return_score_source_gate_subject_state: bool = False
    return_score_source_gate_subject_detach: bool = True
    return_score_source_gate_logit_threshold: float = 0.0
    return_score_source_specific_head: bool = False
    return_score_source_specific_hidden_mult: float = 0.5
    return_score_context_head: bool = False
    return_score_context_hidden_mult: float = 0.5
    return_score_context_detach: bool = True
    return_score_policy_match_head: bool = False
    return_score_policy_match_hidden_mult: float = 0.5
    return_score_policy_match_detach: bool = True
    return_score_policy_source_head: bool = False
    return_score_policy_source_hidden_mult: float = 0.5
    return_score_policy_source_detach: bool = True
    return_score_competition_head: bool = False
    return_score_competition_hidden_mult: float = 0.5
    return_score_competition_detach: bool = True
    return_score_set_refiner_head: bool = False
    return_score_set_refiner_hidden_mult: float = 1.0
    return_score_set_refiner_layers: int = 1
    return_score_set_refiner_detach: bool = True
    return_score_action_decoder_head: bool = False
    return_score_action_decoder_hidden_mult: float = 1.0
    return_score_action_decoder_layers: int = 1
    return_score_action_decoder_detach: bool = True
    return_score_action_decoder_subject_state: bool = False
    return_score_action_decoder_subject_detach: bool = True
    return_score_action_decoder_replace: bool = False


class MobileCropNetV4(nn.Module):
    def __init__(
        self,
        *,
        candidate_k: int = 24,
        proposal_q: int = 16,
        input_size: int = 0,
        width_mult: float = 0.75,
        token_dim: int = 128,
        backbone_name: str = "custom_depthwise",
        backbone_pretrained: bool = False,
        ranker_type: str = "relation_lite",
        ranker_depth: int = 1,
        route_classes: int = SUBJECT_MODE_COUNT,
        decision_classes: int = len(DECISION_VOCAB),
        ar_classes: int = len(TARGET_AR_VOCAB),
        use_subject_prior: bool = False,
        route_image_only: bool = False,
        route_use_subject_prior: bool = False,
        route_use_candidate_context: bool = False,
        route_use_subject_box_features: bool = False,
        route_use_subject_spatial_token: bool = False,
        route_head_depth: int = 1,
        route_head_hidden_mult: float = 0.5,
        route_head_dropout: float = 0.0,
        route_aux_heads: bool = False,
        route_image_residual: bool = False,
        route_image_residual_weight: float = 1.0,
        route_decode_mode: str = "fine",
        route_decode_kind_weight: float = 1.0,
        route_decode_cardinality_weight: float = 1.0,
        policy_use_subject_prior: bool = False,
        proposal_use_subject_prior: bool = False,
        subject_box_head: bool = False,
        subject_box_spatial_head: bool = False,
        subject_box_spatial_multiscale: bool = False,
        subject_box_spatial_mix_bias: float = -1.0,
        subject_box_spatial_output: str = "mix",
        subject_box_spatial_box_mode: str = "regress",
        subject_box_spatial_extent_scale: float = 1.0,
        subject_box_coord_space: str = "letterbox",
        subject_box_refine_with_proposals: bool = False,
        subject_valid_route_calibrator: bool = False,
        subject_valid_route_calibrator_hidden_mult: float = 0.5,
        subject_valid_route_calibrator_detach: bool = True,
        use_pred_subject_box_as_prior: bool = False,
        detach_pred_subject_prior: bool = True,
        policy_score_head: bool = False,
        policy_score_threshold_low: float = 0.35,
        policy_score_threshold_high: float = 0.70,
        decision_source_head: bool = False,
        decision_source_logit_threshold: float = 0.0,
        decision_source_pair_head: bool = False,
        decision_source_pair_hidden_mult: float = 0.5,
        decision_source_pair_after_return_deltas: bool = False,
        decision_source_pair_subject_state: bool = False,
        decision_source_pair_subject_detach: bool = True,
        route_condition_candidate_scores: bool = False,
        route_condition_detach: bool = True,
        return_score_head: bool = False,
        return_score_head_depth: int = 1,
        return_score_head_hidden_mult: float = 0.5,
        return_score_action_source_bias: bool = False,
        return_score_decision_source_bias: bool = False,
        return_score_decision_source_detach: bool = True,
        return_score_decision_source_max_scale: float = 2.0,
        return_score_decision_source_from_source_head: bool = False,
        return_score_decision_conditioned: bool = False,
        return_score_decision_conditioned_mask_value: float = -10000.0,
        return_score_decision_conditioned_from_source_head: bool = False,
        return_score_source_mixture: bool = False,
        return_score_source_mixture_strength: float = 1.0,
        return_score_source_mixture_detach: bool = True,
        return_score_source_mixture_from_source_head: bool = False,
        return_score_source_mixture_gap_threshold: float | None = None,
        return_score_source_mixture_gap_mask_value: float = -10000.0,
        return_score_source_gate: bool = False,
        return_score_source_gate_hidden_mult: float = 0.5,
        return_score_source_gate_strength: float = 1.0,
        return_score_source_gate_detach: bool = True,
        return_score_source_gate_subject_state: bool = False,
        return_score_source_gate_subject_detach: bool = True,
        return_score_source_gate_logit_threshold: float = 0.0,
        return_score_source_specific_head: bool = False,
        return_score_source_specific_hidden_mult: float = 0.5,
        return_score_context_head: bool = False,
        return_score_context_hidden_mult: float = 0.5,
        return_score_context_detach: bool = True,
        return_score_policy_match_head: bool = False,
        return_score_policy_match_hidden_mult: float = 0.5,
        return_score_policy_match_detach: bool = True,
        return_score_policy_source_head: bool = False,
        return_score_policy_source_hidden_mult: float = 0.5,
        return_score_policy_source_detach: bool = True,
        return_score_competition_head: bool = False,
        return_score_competition_hidden_mult: float = 0.5,
        return_score_competition_detach: bool = True,
        return_score_set_refiner_head: bool = False,
        return_score_set_refiner_hidden_mult: float = 1.0,
        return_score_set_refiner_layers: int = 1,
        return_score_set_refiner_detach: bool = True,
        return_score_action_decoder_head: bool = False,
        return_score_action_decoder_hidden_mult: float = 1.0,
        return_score_action_decoder_layers: int = 1,
        return_score_action_decoder_detach: bool = True,
        return_score_action_decoder_subject_state: bool = False,
        return_score_action_decoder_subject_detach: bool = True,
        return_score_action_decoder_replace: bool = False,
    ) -> None:
        super().__init__()
        c1 = max(8, int(16 * width_mult))
        c2 = max(12, int(24 * width_mult))
        c3 = max(20, int(40 * width_mult))
        c4 = max(32, int(72 * width_mult))
        self.config = MobileCropNetV4Config(
            candidate_k=int(candidate_k),
            proposal_q=int(proposal_q),
            input_size=int(input_size),
            width_mult=float(width_mult),
            token_dim=int(token_dim),
            backbone_name=str(backbone_name),
            backbone_pretrained=bool(backbone_pretrained),
            ranker_type=str(ranker_type),
            ranker_depth=int(ranker_depth),
            route_classes=int(route_classes),
            decision_classes=int(decision_classes),
            ar_classes=int(ar_classes),
            use_subject_prior=bool(use_subject_prior),
            route_image_only=bool(route_image_only),
            route_use_subject_prior=bool(route_use_subject_prior),
            route_use_candidate_context=bool(route_use_candidate_context),
            route_use_subject_box_features=bool(route_use_subject_box_features),
            route_use_subject_spatial_token=bool(route_use_subject_spatial_token),
            route_head_depth=max(1, int(route_head_depth)),
            route_head_hidden_mult=float(route_head_hidden_mult),
            route_head_dropout=max(0.0, float(route_head_dropout)),
            route_aux_heads=bool(route_aux_heads),
            route_image_residual=bool(route_image_residual),
            route_image_residual_weight=float(route_image_residual_weight),
            route_decode_mode=str(route_decode_mode or "fine").lower(),
            route_decode_kind_weight=float(route_decode_kind_weight),
            route_decode_cardinality_weight=float(route_decode_cardinality_weight),
            policy_use_subject_prior=bool(policy_use_subject_prior),
            proposal_use_subject_prior=bool(proposal_use_subject_prior),
            subject_box_head=bool(subject_box_head),
            subject_box_spatial_head=bool(subject_box_spatial_head),
            subject_box_spatial_multiscale=bool(subject_box_spatial_multiscale),
            subject_box_spatial_mix_bias=float(subject_box_spatial_mix_bias),
            subject_box_spatial_output=str(subject_box_spatial_output or "mix").lower(),
            subject_box_spatial_box_mode=str(subject_box_spatial_box_mode or "regress").lower(),
            subject_box_spatial_extent_scale=float(subject_box_spatial_extent_scale),
            subject_box_coord_space=str(subject_box_coord_space),
            subject_box_refine_with_proposals=bool(subject_box_refine_with_proposals),
            subject_valid_route_calibrator=bool(subject_valid_route_calibrator),
            subject_valid_route_calibrator_hidden_mult=float(subject_valid_route_calibrator_hidden_mult),
            subject_valid_route_calibrator_detach=bool(subject_valid_route_calibrator_detach),
            use_pred_subject_box_as_prior=bool(use_pred_subject_box_as_prior),
            detach_pred_subject_prior=bool(detach_pred_subject_prior),
            policy_score_head=bool(policy_score_head),
            policy_score_threshold_low=float(policy_score_threshold_low),
            policy_score_threshold_high=float(policy_score_threshold_high),
            decision_source_head=bool(decision_source_head),
            decision_source_logit_threshold=float(decision_source_logit_threshold),
            decision_source_pair_head=bool(decision_source_pair_head),
            decision_source_pair_hidden_mult=float(decision_source_pair_hidden_mult),
            decision_source_pair_after_return_deltas=bool(decision_source_pair_after_return_deltas),
            decision_source_pair_subject_state=bool(decision_source_pair_subject_state),
            decision_source_pair_subject_detach=bool(decision_source_pair_subject_detach),
            route_condition_candidate_scores=bool(route_condition_candidate_scores),
            route_condition_detach=bool(route_condition_detach),
            return_score_head=bool(return_score_head),
            return_score_head_depth=max(1, int(return_score_head_depth)),
            return_score_head_hidden_mult=float(return_score_head_hidden_mult),
            return_score_action_source_bias=bool(return_score_action_source_bias),
            return_score_decision_source_bias=bool(return_score_decision_source_bias),
            return_score_decision_source_detach=bool(return_score_decision_source_detach),
            return_score_decision_source_max_scale=max(0.0, float(return_score_decision_source_max_scale)),
            return_score_decision_source_from_source_head=bool(return_score_decision_source_from_source_head),
            return_score_decision_conditioned=bool(return_score_decision_conditioned),
            return_score_decision_conditioned_mask_value=float(return_score_decision_conditioned_mask_value),
            return_score_decision_conditioned_from_source_head=bool(return_score_decision_conditioned_from_source_head),
            return_score_source_mixture=bool(return_score_source_mixture),
            return_score_source_mixture_strength=max(0.0, float(return_score_source_mixture_strength)),
            return_score_source_mixture_detach=bool(return_score_source_mixture_detach),
            return_score_source_mixture_from_source_head=bool(return_score_source_mixture_from_source_head),
            return_score_source_mixture_gap_threshold=(
                None
                if return_score_source_mixture_gap_threshold is None
                else float(return_score_source_mixture_gap_threshold)
            ),
            return_score_source_mixture_gap_mask_value=float(return_score_source_mixture_gap_mask_value),
            return_score_source_gate=bool(return_score_source_gate),
            return_score_source_gate_hidden_mult=float(return_score_source_gate_hidden_mult),
            return_score_source_gate_strength=max(0.0, float(return_score_source_gate_strength)),
            return_score_source_gate_detach=bool(return_score_source_gate_detach),
            return_score_source_gate_subject_state=bool(return_score_source_gate_subject_state),
            return_score_source_gate_subject_detach=bool(return_score_source_gate_subject_detach),
            return_score_source_gate_logit_threshold=float(return_score_source_gate_logit_threshold),
            return_score_source_specific_head=bool(return_score_source_specific_head),
            return_score_source_specific_hidden_mult=float(return_score_source_specific_hidden_mult),
            return_score_context_head=bool(return_score_context_head),
            return_score_context_hidden_mult=float(return_score_context_hidden_mult),
            return_score_context_detach=bool(return_score_context_detach),
            return_score_policy_match_head=bool(return_score_policy_match_head),
            return_score_policy_match_hidden_mult=float(return_score_policy_match_hidden_mult),
            return_score_policy_match_detach=bool(return_score_policy_match_detach),
            return_score_policy_source_head=bool(return_score_policy_source_head),
            return_score_policy_source_hidden_mult=float(return_score_policy_source_hidden_mult),
            return_score_policy_source_detach=bool(return_score_policy_source_detach),
            return_score_competition_head=bool(return_score_competition_head),
            return_score_competition_hidden_mult=float(return_score_competition_hidden_mult),
            return_score_competition_detach=bool(return_score_competition_detach),
            return_score_set_refiner_head=bool(return_score_set_refiner_head),
            return_score_set_refiner_hidden_mult=float(return_score_set_refiner_hidden_mult),
            return_score_set_refiner_layers=max(1, int(return_score_set_refiner_layers)),
            return_score_set_refiner_detach=bool(return_score_set_refiner_detach),
            return_score_action_decoder_head=bool(return_score_action_decoder_head),
            return_score_action_decoder_hidden_mult=float(return_score_action_decoder_hidden_mult),
            return_score_action_decoder_layers=max(1, int(return_score_action_decoder_layers)),
            return_score_action_decoder_detach=bool(return_score_action_decoder_detach),
            return_score_action_decoder_subject_state=bool(return_score_action_decoder_subject_state),
            return_score_action_decoder_subject_detach=bool(return_score_action_decoder_subject_detach),
            return_score_action_decoder_replace=bool(return_score_action_decoder_replace),
        )
        self.backbone_name = str(backbone_name)
        self.uses_timm_backbone = self.backbone_name not in {"custom_depthwise", "depthwise", ""}
        if self.uses_timm_backbone:
            try:
                import timm
            except ImportError as exc:
                raise ImportError(f"timm is required for backbone_name={self.backbone_name!r}") from exc
            local_weight = _local_timm_weight_file(self.backbone_name) if bool(backbone_pretrained) else None
            out_indices = (-2, -1) if bool(subject_box_spatial_head and subject_box_spatial_multiscale) else (-1,)
            create_kwargs: dict[str, Any] = {
                "features_only": True,
                "pretrained": bool(backbone_pretrained) and local_weight is None,
                "out_indices": out_indices,
            }
            requested_input_size = int(input_size or 0)
            timm_name_lower = self.backbone_name.lower()
            if requested_input_size > 0 and any(token in timm_name_lower for token in ("swin", "vit", "eva")):
                create_kwargs["img_size"] = requested_input_size
                if "swin" in timm_name_lower:
                    create_kwargs["strict_img_size"] = False
            try:
                self.backbone = timm.create_model(
                    self.backbone_name,
                    **create_kwargs,
                )
            except TypeError:
                if "img_size" not in create_kwargs:
                    raise
                create_kwargs.pop("img_size", None)
                create_kwargs.pop("strict_img_size", None)
                self.backbone = timm.create_model(
                    self.backbone_name,
                    **create_kwargs,
                )
            if local_weight is not None:
                from timm.models import load_checkpoint

                load_checkpoint(self.backbone, str(local_weight), strict=False)
            feature_channels = [int(v) for v in self.backbone.feature_info.channels()]
            c4 = int(feature_channels[-1])
            c_spatial = int(feature_channels[-2]) if len(feature_channels) >= 2 and bool(subject_box_spatial_multiscale) else c4
            self._backbone_feature_channels = feature_channels
        else:
            self.backbone = nn.Sequential(
                ConvBNAct(3, c1, stride=2),
                DepthwiseBlock(c1, c2, stride=2),
                DepthwiseBlock(c2, c3, stride=2),
                DepthwiseBlock(c3, c4, stride=2),
                DepthwiseBlock(c4, c4, stride=1),
                DepthwiseBlock(c4, c4, stride=1),
            )
            self._backbone_feature_channels = [c4]
            c_spatial = c4
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.ar_embed = nn.Embedding(ar_classes, 16)
        self.cond_proj = nn.Sequential(nn.Linear(c4 + 16 + 1, token_dim), nn.SiLU(inplace=True), nn.Linear(token_dim, token_dim))
        self.uses_subject_prior = bool(
            self.config.use_subject_prior
            or self.config.route_use_subject_prior
            or self.config.route_use_subject_box_features
            or self.config.policy_use_subject_prior
            or self.config.proposal_use_subject_prior
        )
        self.subject_prior_encoder: nn.Module
        if self.uses_subject_prior:
            self.subject_prior_encoder = nn.Sequential(
                nn.Linear(12, token_dim),
                nn.SiLU(inplace=True),
                nn.Linear(token_dim, token_dim),
            )
        else:
            self.subject_prior_encoder = nn.Identity()

        self.proposal_queries = nn.Parameter(torch.randn(proposal_q, token_dim) * 0.02)
        self.proposal_box_head = nn.Sequential(nn.Linear(token_dim, token_dim), nn.SiLU(inplace=True), nn.Linear(token_dim, 4), nn.Sigmoid())
        self.proposal_logit_head = nn.Sequential(nn.Linear(token_dim, token_dim // 2), nn.SiLU(inplace=True), nn.Linear(token_dim // 2, 1))
        self.proposal_token_head = nn.Sequential(nn.Linear(token_dim, token_dim), nn.SiLU(inplace=True), nn.Linear(token_dim, token_dim))
        if self.config.subject_box_head:
            self.subject_box_head = nn.Sequential(nn.Linear(token_dim, token_dim), nn.SiLU(inplace=True), nn.Linear(token_dim, 4), nn.Sigmoid())
            self.subject_valid_head = nn.Sequential(nn.Linear(token_dim, token_dim // 2), nn.SiLU(inplace=True), nn.Linear(token_dim // 2, 1))
            if self.config.subject_box_spatial_head:
                self.subject_spatial_proj = nn.Sequential(
                    nn.Conv2d(c_spatial, token_dim, kernel_size=1, bias=False),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(token_dim, token_dim, kernel_size=3, padding=1, bias=False),
                    nn.SiLU(inplace=True),
                )
                self.subject_spatial_heatmap_head = nn.Conv2d(token_dim, 1, kernel_size=1)
                self.subject_spatial_box_head = nn.Sequential(
                    nn.Linear(token_dim * 2, token_dim),
                    nn.SiLU(inplace=True),
                    nn.Linear(token_dim, 4),
                )
                self.subject_spatial_valid_head = nn.Sequential(
                    nn.Linear(token_dim * 2, token_dim // 2),
                    nn.SiLU(inplace=True),
                    nn.Linear(token_dim // 2, 1),
                )
                self.subject_spatial_mix_head = nn.Sequential(
                    nn.Linear(token_dim * 2, token_dim // 2),
                    nn.SiLU(inplace=True),
                    nn.Linear(token_dim // 2, 1),
                )
                nn.init.constant_(self.subject_spatial_mix_head[-1].bias, float(self.config.subject_box_spatial_mix_bias))
            else:
                self.subject_spatial_proj = None
                self.subject_spatial_heatmap_head = None
                self.subject_spatial_box_head = None
                self.subject_spatial_valid_head = None
                self.subject_spatial_mix_head = None
            if self.config.subject_box_refine_with_proposals:
                box_feat_dim = _box_feature_vector(torch.zeros((1, 4), dtype=torch.float32)).shape[-1]
                self.subject_proposal_head = nn.Sequential(
                    nn.Linear(token_dim, token_dim // 2),
                    nn.SiLU(inplace=True),
                    nn.Linear(token_dim // 2, 1),
                )
                self.subject_refine_proj = nn.Sequential(
                    nn.Linear(token_dim * 2 + box_feat_dim * 3, token_dim),
                    nn.SiLU(inplace=True),
                    nn.Linear(token_dim, token_dim),
                    nn.SiLU(inplace=True),
                )
                self.subject_refine_box_head = nn.Sequential(
                    nn.Linear(token_dim, token_dim),
                    nn.SiLU(inplace=True),
                    nn.Linear(token_dim, 4),
                    nn.Tanh(),
                )
                self.subject_refine_valid_head = nn.Sequential(
                    nn.Linear(token_dim, token_dim // 2),
                    nn.SiLU(inplace=True),
                    nn.Linear(token_dim // 2, 1),
                )
            else:
                self.subject_proposal_head = None
                self.subject_refine_proj = None
                self.subject_refine_box_head = None
                self.subject_refine_valid_head = None
            if self.config.subject_valid_route_calibrator:
                route_valid_in_dim = token_dim + route_classes + 2
                route_valid_hidden = max(
                    16,
                    int(round(token_dim * max(0.25, float(self.config.subject_valid_route_calibrator_hidden_mult)))),
                )
                self.subject_valid_route_head = nn.Sequential(
                    nn.Linear(route_valid_in_dim, route_valid_hidden),
                    nn.SiLU(inplace=True),
                    nn.Linear(route_valid_hidden, 1),
                )
                last_valid = self.subject_valid_route_head[-1]
                if isinstance(last_valid, nn.Linear):
                    nn.init.zeros_(last_valid.weight)
                    nn.init.zeros_(last_valid.bias)
            else:
                self.subject_valid_route_head = None
        else:
            self.subject_box_head = None
            self.subject_valid_head = None
            self.subject_spatial_proj = None
            self.subject_spatial_heatmap_head = None
            self.subject_spatial_box_head = None
            self.subject_spatial_valid_head = None
            self.subject_spatial_mix_head = None
            self.subject_proposal_head = None
            self.subject_refine_proj = None
            self.subject_refine_box_head = None
            self.subject_refine_valid_head = None
            self.subject_valid_route_head = None

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
        self.ranker_type = str(ranker_type)
        self.ranker_depth = max(0, int(ranker_depth))
        if self.ranker_type == "set_transformer" and self.ranker_depth > 0:
            nhead = 8 if token_dim % 8 == 0 else 4
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=nhead,
                dim_feedforward=token_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.set_ranker = nn.TransformerEncoder(encoder_layer, num_layers=self.ranker_depth)
        else:
            self.set_ranker = None
        self.utility_head = nn.Linear(token_dim, 1)
        self.positive_head = nn.Linear(token_dim, 1)
        self.risk_head = nn.Linear(token_dim, 1)
        self.return_score_delta_head: nn.Module | None = None
        if self.config.return_score_head:
            if self.config.return_score_head_depth <= 1:
                self.return_score_delta_head = nn.Linear(token_dim, 1)
            else:
                hidden = max(8, int(round(token_dim * float(self.config.return_score_head_hidden_mult))))
                layers: list[nn.Module] = []
                in_dim = token_dim
                for _ in range(max(1, int(self.config.return_score_head_depth) - 1)):
                    layers.extend([nn.Linear(in_dim, hidden), nn.SiLU(inplace=True)])
                    in_dim = hidden
                layers.append(nn.Linear(in_dim, 1))
                self.return_score_delta_head = nn.Sequential(*layers)
        if self.return_score_delta_head is not None:
            last_linear = self.return_score_delta_head
            if isinstance(last_linear, nn.Sequential):
                last_linear = next(module for module in reversed(last_linear) if isinstance(module, nn.Linear))
            if isinstance(last_linear, nn.Linear):
                nn.init.zeros_(last_linear.weight)
                nn.init.zeros_(last_linear.bias)
        self.return_score_action_source_bias: nn.Parameter | None = (
            nn.Parameter(torch.zeros(2)) if self.config.return_score_action_source_bias else None
        )
        self.return_score_decision_source_scale: nn.Parameter | None = (
            nn.Parameter(torch.zeros(())) if self.config.return_score_decision_source_bias else None
        )
        self.return_score_source_gate_head: nn.Module | None = None
        if self.config.return_score_source_gate:
            source_gate_subject_dim = 30 if self.config.return_score_source_gate_subject_state else 0
            source_gate_in_dim = (token_dim // 2) + (4 * token_dim) + route_classes + decision_classes + 10 + source_gate_subject_dim
            source_gate_hidden = max(16, int(round(token_dim * max(0.25, float(self.config.return_score_source_gate_hidden_mult)))))
            self.return_score_source_gate_head = nn.Sequential(
                nn.Linear(source_gate_in_dim, source_gate_hidden),
                nn.SiLU(inplace=True),
                nn.Linear(source_gate_hidden, source_gate_hidden),
                nn.SiLU(inplace=True),
                nn.Linear(source_gate_hidden, 1),
            )
            last_gate = self.return_score_source_gate_head[-1]
            if isinstance(last_gate, nn.Linear):
                nn.init.zeros_(last_gate.weight)
                nn.init.zeros_(last_gate.bias)
        self.return_score_source_specific_head: nn.Module | None = None
        if self.config.return_score_source_specific_head:
            source_hidden = max(16, int(round(token_dim * max(0.25, float(self.config.return_score_source_specific_hidden_mult)))))
            self.return_score_source_specific_head = nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, source_hidden),
                nn.SiLU(inplace=True),
                nn.Linear(source_hidden, 2),
            )
            last_source = self.return_score_source_specific_head[-1]
            if isinstance(last_source, nn.Linear):
                nn.init.zeros_(last_source.weight)
                nn.init.zeros_(last_source.bias)
        self.return_score_context_head: nn.Module | None = None
        if self.config.return_score_context_head:
            context_in_dim = token_dim + (token_dim // 2) + route_classes + decision_classes + 3 + 11
            context_hidden = max(16, int(round(token_dim * max(0.25, float(self.config.return_score_context_hidden_mult)))))
            self.return_score_context_head = nn.Sequential(
                nn.Linear(context_in_dim, context_hidden),
                nn.SiLU(inplace=True),
                nn.Linear(context_hidden, context_hidden),
                nn.SiLU(inplace=True),
                nn.Linear(context_hidden, 1),
            )
            last_context = self.return_score_context_head[-1]
            if isinstance(last_context, nn.Linear):
                nn.init.zeros_(last_context.weight)
                nn.init.zeros_(last_context.bias)
        self.return_score_policy_match_proj: nn.Module | None = None
        self.return_score_policy_match_head: nn.Module | None = None
        if self.config.return_score_policy_match_head:
            self.return_score_policy_match_proj = nn.Linear(token_dim // 2, token_dim)
            match_in_dim = token_dim * 4 + route_classes + decision_classes + 3 + 11
            match_hidden = max(16, int(round(token_dim * max(0.25, float(self.config.return_score_policy_match_hidden_mult)))))
            self.return_score_policy_match_head = nn.Sequential(
                nn.Linear(match_in_dim, match_hidden),
                nn.SiLU(inplace=True),
                nn.Linear(match_hidden, match_hidden),
                nn.SiLU(inplace=True),
                nn.Linear(match_hidden, 1),
            )
            last_match = self.return_score_policy_match_head[-1]
            if isinstance(last_match, nn.Linear):
                nn.init.zeros_(last_match.weight)
                nn.init.zeros_(last_match.bias)
        self.return_score_policy_source_proj: nn.Module | None = None
        self.return_score_policy_source_head: nn.Module | None = None
        if self.config.return_score_policy_source_head:
            self.return_score_policy_source_proj = nn.Linear(token_dim // 2, token_dim)
            source_match_in_dim = token_dim * 4 + route_classes + decision_classes + 3 + 11
            source_match_hidden = max(16, int(round(token_dim * max(0.25, float(self.config.return_score_policy_source_hidden_mult)))))
            self.return_score_policy_source_head = nn.Sequential(
                nn.Linear(source_match_in_dim, source_match_hidden),
                nn.SiLU(inplace=True),
                nn.Linear(source_match_hidden, source_match_hidden),
                nn.SiLU(inplace=True),
                nn.Linear(source_match_hidden, 2),
            )
            last_policy_source = self.return_score_policy_source_head[-1]
            if isinstance(last_policy_source, nn.Linear):
                nn.init.zeros_(last_policy_source.weight)
                nn.init.zeros_(last_policy_source.bias)
        self.return_score_competition_head: nn.Module | None = None
        if self.config.return_score_competition_head:
            competition_in_dim = token_dim * 4 + route_classes + decision_classes + 3 + 11
            competition_hidden = max(16, int(round(token_dim * max(0.25, float(self.config.return_score_competition_hidden_mult)))))
            self.return_score_competition_head = nn.Sequential(
                nn.Linear(competition_in_dim, competition_hidden),
                nn.SiLU(inplace=True),
                nn.Linear(competition_hidden, competition_hidden),
                nn.SiLU(inplace=True),
                nn.Linear(competition_hidden, 1),
            )
            last_competition = self.return_score_competition_head[-1]
            if isinstance(last_competition, nn.Linear):
                nn.init.zeros_(last_competition.weight)
                nn.init.zeros_(last_competition.bias)
        self.return_score_set_refiner_input: nn.Module | None = None
        self.return_score_set_refiner: nn.Module | None = None
        self.return_score_set_refiner_head: nn.Module | None = None
        if self.config.return_score_set_refiner_head:
            set_refiner_in_dim = token_dim + (token_dim // 2) + route_classes + decision_classes + 3 + 11
            set_hidden = max(token_dim, int(round(token_dim * max(0.5, float(self.config.return_score_set_refiner_hidden_mult)))))
            self.return_score_set_refiner_input = nn.Linear(set_refiner_in_dim, token_dim)
            nhead = 8 if token_dim % 8 == 0 else 4
            set_layer = nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=nhead,
                dim_feedforward=max(token_dim * 2, set_hidden * 2),
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.return_score_set_refiner = nn.TransformerEncoder(
                set_layer,
                num_layers=max(1, int(self.config.return_score_set_refiner_layers)),
            )
            self.return_score_set_refiner_head = nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, 1),
            )
            last_set_refiner = self.return_score_set_refiner_head[-1]
            if isinstance(last_set_refiner, nn.Linear):
                nn.init.zeros_(last_set_refiner.weight)
                nn.init.zeros_(last_set_refiner.bias)
        self.return_score_action_query: nn.Module | None = None
        self.return_score_action_memory: nn.Module | None = None
        self.return_score_action_decoder: nn.Module | None = None
        self.return_score_action_decoder_head: nn.Module | None = None
        if self.config.return_score_action_decoder_head:
            subject_state_dim = 12 if self.config.return_score_action_decoder_subject_state else 0
            subject_pair_dim = 6 if self.config.return_score_action_decoder_subject_state else 0
            action_query_in_dim = (token_dim // 2) + route_classes + decision_classes + subject_state_dim
            action_memory_in_dim = token_dim + 3 + 11 + subject_pair_dim
            action_hidden = max(token_dim, int(round(token_dim * max(0.5, float(self.config.return_score_action_decoder_hidden_mult)))))
            self.return_score_action_query = nn.Sequential(
                nn.Linear(action_query_in_dim, token_dim),
                nn.SiLU(inplace=True),
                nn.Linear(token_dim, token_dim),
            )
            self.return_score_action_memory = nn.Linear(action_memory_in_dim, token_dim)
            nhead = 8 if token_dim % 8 == 0 else 4
            action_layer = nn.TransformerDecoderLayer(
                d_model=token_dim,
                nhead=nhead,
                dim_feedforward=max(token_dim * 2, action_hidden * 2),
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.return_score_action_decoder = nn.TransformerDecoder(
                action_layer,
                num_layers=max(1, int(self.config.return_score_action_decoder_layers)),
            )
            action_head_in_dim = token_dim * 4 + route_classes + decision_classes + 3 + 11 + subject_state_dim + subject_pair_dim
            self.return_score_action_decoder_head = nn.Sequential(
                nn.Linear(action_head_in_dim, action_hidden),
                nn.SiLU(inplace=True),
                nn.Linear(action_hidden, 1),
            )
            last_action_decoder = self.return_score_action_decoder_head[-1]
            if isinstance(last_action_decoder, nn.Linear):
                nn.init.zeros_(last_action_decoder.weight)
                nn.init.zeros_(last_action_decoder.bias)
        self.macro_head = nn.Linear(token_dim, 4)
        self.checklist_class_head = nn.Linear(token_dim, CHECKLIST_CLASS_TOTAL)
        self.checklist_applicability_head = nn.Linear(token_dim, CHECKLIST_CLASS_COUNT)
        self.detail_score_head = nn.Linear(token_dim, CHECKLIST_SCORE_COUNT)
        self.why_tag_head = nn.Linear(token_dim, WHY_TAG_COUNT)
        self.route_score_proj: nn.Module | None
        if self.config.route_condition_candidate_scores:
            self.route_score_proj = nn.Sequential(
                nn.Linear(route_classes, token_dim),
                nn.SiLU(inplace=True),
                nn.Linear(token_dim, token_dim),
            )
            nn.init.zeros_(self.route_score_proj[-1].weight)
            nn.init.zeros_(self.route_score_proj[-1].bias)
        else:
            self.route_score_proj = None
        route_in_dim = c4 if self.config.route_image_only else c4 + 16 + 1
        if not self.config.route_image_only:
            if self.config.route_use_subject_prior:
                route_in_dim += token_dim
            if self.config.route_use_candidate_context:
                route_in_dim += token_dim
            if self.config.route_use_subject_box_features:
                route_in_dim += 12
            if self.config.route_use_subject_spatial_token:
                route_in_dim += token_dim
        route_hidden = max(16, int(round(token_dim * max(0.25, float(self.config.route_head_hidden_mult)))))
        route_dropout = max(0.0, float(self.config.route_head_dropout))

        def make_route_head(input_dim: int, out_dim: int) -> nn.Sequential:
            route_layers: list[nn.Module] = [nn.Linear(input_dim, route_hidden), nn.SiLU(inplace=True)]
            for _ in range(max(1, int(self.config.route_head_depth)) - 1):
                if route_dropout > 0.0:
                    route_layers.append(nn.Dropout(p=route_dropout))
                route_layers.extend([nn.Linear(route_hidden, route_hidden), nn.SiLU(inplace=True)])
            if route_dropout > 0.0:
                route_layers.append(nn.Dropout(p=route_dropout))
            route_layers.append(nn.Linear(route_hidden, int(out_dim)))
            return nn.Sequential(*route_layers)

        self.route_head = make_route_head(route_in_dim, route_classes)
        if bool(self.config.route_aux_heads):
            self.route_kind_head = make_route_head(route_in_dim, len(_ROUTE_KIND_GROUPS))
            self.route_cardinality_head = make_route_head(route_in_dim, len(_ROUTE_CARDINALITY_GROUPS))
        else:
            self.route_kind_head = None
            self.route_cardinality_head = None
        if bool(self.config.route_image_residual):
            self.route_image_residual_head = make_route_head(c4, route_classes)
            if bool(self.config.route_aux_heads):
                self.route_image_residual_kind_head = make_route_head(c4, len(_ROUTE_KIND_GROUPS))
                self.route_image_residual_cardinality_head = make_route_head(c4, len(_ROUTE_CARDINALITY_GROUPS))
            else:
                self.route_image_residual_kind_head = None
                self.route_image_residual_cardinality_head = None
        else:
            self.route_image_residual_head = None
            self.route_image_residual_kind_head = None
            self.route_image_residual_cardinality_head = None
        self.policy_head = nn.Sequential(
            nn.Linear(c4 + 16 + 1 + token_dim + 2 + (token_dim if self.config.policy_use_subject_prior else 0), token_dim),
            nn.SiLU(inplace=True),
            nn.Linear(token_dim, token_dim // 2),
            nn.SiLU(inplace=True),
        )
        self.decision_head = nn.Linear(token_dim // 2, decision_classes)
        self.delta_head = nn.Linear(token_dim // 2, 1)
        self.policy_score_predictor = nn.Linear(token_dim // 2, 1) if self.config.policy_score_head else None
        self.decision_source_head = nn.Linear(token_dim // 2, 1) if self.config.decision_source_head else None
        self.decision_source_pair_head: nn.Module | None = None
        if self.config.decision_source_pair_head:
            pair_subject_dim = 30 if self.config.decision_source_pair_subject_state else 0
            pair_in = (token_dim // 2) + (4 * token_dim) + route_classes + decision_classes + 6 + pair_subject_dim
            pair_hidden = max(8, int(round(token_dim * float(self.config.decision_source_pair_hidden_mult))))
            self.decision_source_pair_head = nn.Sequential(
                nn.Linear(pair_in, pair_hidden),
                nn.SiLU(inplace=True),
                nn.Linear(pair_hidden, pair_hidden),
                nn.SiLU(inplace=True),
                nn.Linear(pair_hidden, 1),
            )
            last_linear = next(module for module in reversed(self.decision_source_pair_head) if isinstance(module, nn.Linear))
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

    def _encode(self, image: torch.Tensor, target_ar_id: torch.Tensor, image_ar_log: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_features = self.backbone(image)
        if isinstance(raw_features, (list, tuple)):
            features = raw_features[-1]
            spatial_features = raw_features[-2] if bool(self.config.subject_box_spatial_multiscale) and len(raw_features) >= 2 else features
            features = self._ensure_nchw_feature(features, int(self._backbone_feature_channels[-1]))
            spatial_channels = int(self._backbone_feature_channels[-2]) if bool(self.config.subject_box_spatial_multiscale) and len(self._backbone_feature_channels) >= 2 else int(self._backbone_feature_channels[-1])
            spatial_features = self._ensure_nchw_feature(spatial_features, spatial_channels)
        else:
            features = raw_features
            spatial_features = raw_features
        global_feat = self.global_pool(features).flatten(1)
        ar_emb = self.ar_embed(target_ar_id.clamp(0, self.config.ar_classes - 1))
        image_ar = image_ar_log.float().view(-1, 1)
        cond = torch.cat([global_feat, ar_emb, image_ar], dim=-1)
        cond_token = self.cond_proj(cond)
        return features, spatial_features, global_feat, ar_emb, cond_token

    @staticmethod
    def _ensure_nchw_feature(features: torch.Tensor, channels: int) -> torch.Tensor:
        if features.ndim == 4 and features.shape[1] != channels and features.shape[-1] == channels:
            return features.permute(0, 3, 1, 2).contiguous()
        return features

    def _spatial_subject_box(
        self,
        features: torch.Tensor,
        cond_token: torch.Tensor,
        letterbox_content_box: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            self.subject_spatial_proj is None
            or self.subject_spatial_heatmap_head is None
            or self.subject_spatial_box_head is None
            or self.subject_spatial_valid_head is None
        ):
            raise RuntimeError("subject spatial head is not enabled")
        fmap = self.subject_spatial_proj(features)
        bsz, _channels, height, width = fmap.shape
        dtype = fmap.dtype
        device = fmap.device
        heatmap_logits_2d = self.subject_spatial_heatmap_head(fmap).squeeze(1)
        y = (torch.arange(height, device=device, dtype=dtype) + 0.5) / float(height)
        x = (torch.arange(width, device=device, dtype=dtype) + 0.5) / float(width)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        x_flat = xx.reshape(1, -1).expand(bsz, -1)
        y_flat = yy.reshape(1, -1).expand(bsz, -1)
        mask_box = letterbox_content_box if letterbox_content_box is not None else None
        mx0, my0, mw, mh = _rect_from_content_box(mask_box, batch_size=bsz, dtype=dtype, device=device)
        mx1 = mx0 + mw
        my1 = my0 + mh
        inside_content = (x_flat >= mx0) & (x_flat <= mx1) & (y_flat >= my0) & (y_flat <= my1)
        heatmap_logits = heatmap_logits_2d.flatten(1).masked_fill(~inside_content, -1e4)
        heatmap_prob = torch.softmax(heatmap_logits, dim=1)
        cx = (heatmap_prob * x_flat).sum(dim=1, keepdim=True)
        cy = (heatmap_prob * y_flat).sum(dim=1, keepdim=True)
        spatial_token = (fmap.flatten(2) * heatmap_prob.unsqueeze(1)).sum(dim=2)
        subject_token = torch.cat([cond_token, spatial_token], dim=-1)
        raw = self.subject_spatial_box_head(subject_token)
        rect_x0, rect_y0, rect_w, rect_h = _rect_from_content_box(
            letterbox_content_box if str(self.config.subject_box_coord_space) == "content" else None,
            batch_size=bsz,
            dtype=dtype,
            device=device,
        )
        spatial_box_mode = str(self.config.subject_box_spatial_box_mode or "regress").lower()
        if spatial_box_mode in {"mask_moment", "mask_moment_regress", "heatmap_moment", "heatmap_moment_regress"}:
            if spatial_box_mode.startswith("mask_"):
                mass = torch.sigmoid(heatmap_logits_2d.flatten(1)).masked_fill(~inside_content, 0.0)
            else:
                mass = heatmap_prob
            mass_sum = mass.sum(dim=1, keepdim=True).clamp_min(1e-6)
            moment_cx = (mass * x_flat).sum(dim=1, keepdim=True) / mass_sum
            moment_cy = (mass * y_flat).sum(dim=1, keepdim=True) / mass_sum
            var_x = (mass * (x_flat - moment_cx) ** 2).sum(dim=1, keepdim=True) / mass_sum
            var_y = (mass * (y_flat - moment_cy) ** 2).sum(dim=1, keepdim=True) / mass_sum
            extent_scale = float(self.config.subject_box_spatial_extent_scale)
            w = torch.sqrt((12.0 * var_x).clamp_min(1e-8)) * extent_scale
            h = torch.sqrt((12.0 * var_y).clamp_min(1e-8)) * extent_scale
            cx = moment_cx
            cy = moment_cy
            if spatial_box_mode.endswith("_regress"):
                cx = cx + torch.tanh(raw[:, 0:1]) * 0.12 * rect_w
                cy = cy + torch.tanh(raw[:, 1:2]) * 0.12 * rect_h
                w = w * torch.exp(torch.tanh(raw[:, 2:3]) * 0.35)
                h = h * torch.exp(torch.tanh(raw[:, 3:4]) * 0.35)
            w = torch.minimum(w.clamp_min(1.0 / float(max(1, width))), rect_w)
            h = torch.minimum(h.clamp_min(1.0 / float(max(1, height))), rect_h)
        else:
            dx = torch.tanh(raw[:, 0:1]) * 0.25 * rect_w
            dy = torch.tanh(raw[:, 1:2]) * 0.25 * rect_h
            w = torch.sigmoid(raw[:, 2:3]).clamp(1e-4, 1.0) * rect_w
            h = torch.sigmoid(raw[:, 3:4]).clamp(1e-4, 1.0) * rect_h
            cx = cx + dx
            cy = cy + dy
        box = _fit_cxcywh_in_rect(
            cx,
            cy,
            w,
            h,
            rect_x0=rect_x0,
            rect_y0=rect_y0,
            rect_w=rect_w,
            rect_h=rect_h,
        )
        valid_logit = self.subject_spatial_valid_head(subject_token).squeeze(-1)
        return box, valid_logit, spatial_token, heatmap_prob, heatmap_logits

    def _proposals(
        self,
        cond_token: torch.Tensor,
        target_ar_id: torch.Tensor,
        image_ar_log: torch.Tensor,
        letterbox_content_box: torch.Tensor | None,
        subject_prior_token: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = cond_token.shape[0]
        query = self.proposal_queries.unsqueeze(0).expand(bsz, -1, -1)
        token = query + cond_token.unsqueeze(1)
        if self.config.proposal_use_subject_prior and subject_prior_token is not None:
            token = token + subject_prior_token.unsqueeze(1)
        proposal_token = self.proposal_token_head(token)
        box = _ar_constrained_proposal_boxes(
            self.proposal_box_head(token),
            target_ar_id,
            self.config.ar_classes,
            image_ar_log,
            letterbox_content_box,
        )
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

    def _score_box_set(
        self,
        *,
        features: torch.Tensor,
        global_feat: torch.Tensor,
        ar_emb: torch.Tensor,
        image_ar_log: torch.Tensor,
        boxes: torch.Tensor,
        valid: torch.Tensor,
        candidate_is_base: torch.Tensor | None = None,
        box_meta: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if candidate_is_base is None:
            candidate_is_base = torch.zeros((boxes.shape[0], boxes.shape[1]), dtype=valid.dtype, device=boxes.device)
        pooled = _masked_box_pool(features, boxes, valid)
        geom = box_meta if box_meta is not None else _box_geom(boxes, candidate_is_base)
        global_rep = global_feat.unsqueeze(1).expand(-1, boxes.shape[1], -1)
        ar_rep = ar_emb.unsqueeze(1).expand(-1, boxes.shape[1], -1)
        image_ar_rep = image_ar_log.float().view(boxes.shape[0], 1, 1).expand(-1, boxes.shape[1], -1)
        cand_token = self.candidate_encoder(torch.cat([pooled, global_rep, ar_rep, image_ar_rep, geom], dim=-1))
        cand_token = self._relation_refine(cand_token, boxes, valid, candidate_is_base)
        if self.set_ranker is not None:
            cand_token = self.set_ranker(cand_token, src_key_padding_mask=(valid <= 0))
        return {
            "candidate_tokens": cand_token,
            "utility_logits": self.utility_head(cand_token).squeeze(-1),
            "positive_logits": self.positive_head(cand_token).squeeze(-1),
            "risk_logits": self.risk_head(cand_token).squeeze(-1),
            "macro_logits": self.macro_head(cand_token),
            "checklist_class_logits": self.checklist_class_head(cand_token),
            "checklist_applicability_logits": self.checklist_applicability_head(cand_token),
            "detail_score_logits": self.detail_score_head(cand_token),
            "why_tag_logits": self.why_tag_head(cand_token),
        }

    def forward(
        self,
        image: torch.Tensor,
        boxes: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
        target_ar_id: torch.Tensor | None = None,
        image_ar_log: torch.Tensor | None = None,
        candidate_is_base: torch.Tensor | None = None,
        box_meta: torch.Tensor | None = None,
        letterbox_content_box: torch.Tensor | None = None,
        subject_prior_box: torch.Tensor | None = None,
        subject_prior_valid: torch.Tensor | None = None,
        subject_prior_reliability: torch.Tensor | None = None,
        pred_subject_prior_alpha: float | torch.Tensor | None = None,
        return_generated_scores: bool = False,
    ) -> dict[str, torch.Tensor]:
        bsz = image.shape[0]
        if target_ar_id is None:
            target_ar_id = torch.zeros((bsz,), dtype=torch.long, device=image.device)
        if image_ar_log is None:
            image_ar_log = torch.zeros((bsz,), dtype=image.dtype, device=image.device)
        features, spatial_features, global_feat, ar_emb, cond_token = self._encode(image, target_ar_id, image_ar_log)
        pred_subject_box: torch.Tensor | None = None
        pred_subject_box_coarse: torch.Tensor | None = None
        pred_subject_box_spatial: torch.Tensor | None = None
        pred_subject_valid_logit: torch.Tensor | None = None
        pred_subject_valid_logit_coarse: torch.Tensor | None = None
        pred_subject_valid_logit_spatial: torch.Tensor | None = None
        pred_subject_valid_prob: torch.Tensor | None = None
        pred_subject_spatial_token: torch.Tensor | None = None
        pred_subject_spatial_mix: torch.Tensor | None = None
        pred_subject_spatial_heatmap: torch.Tensor | None = None
        pred_subject_spatial_heatmap_logits: torch.Tensor | None = None
        subject_proposal_logits: torch.Tensor | None = None
        if self.subject_box_head is not None and self.subject_valid_head is not None:
            pred_subject_box_coarse = _decode_subject_box(
                self.subject_box_head(cond_token),
                coord_space=self.config.subject_box_coord_space,
                letterbox_content_box=letterbox_content_box,
            )
            pred_subject_valid_logit_coarse = self.subject_valid_head(cond_token).squeeze(-1)
            pred_subject_box = pred_subject_box_coarse
            pred_subject_valid_logit = pred_subject_valid_logit_coarse
            pred_subject_valid_prob = torch.sigmoid(pred_subject_valid_logit)
            if self.subject_spatial_proj is not None and self.subject_spatial_mix_head is not None:
                (
                    pred_subject_box_spatial,
                    pred_subject_valid_logit_spatial,
                    pred_subject_spatial_token,
                    pred_subject_spatial_heatmap,
                    pred_subject_spatial_heatmap_logits,
                ) = self._spatial_subject_box(
                    spatial_features,
                    cond_token,
                    letterbox_content_box,
                )
                spatial_output = str(self.config.subject_box_spatial_output or "mix").lower()
                if spatial_output == "spatial":
                    pred_subject_spatial_mix = torch.ones((bsz,), dtype=image.dtype, device=image.device)
                    pred_subject_box = pred_subject_box_spatial
                    pred_subject_valid_logit = pred_subject_valid_logit_spatial
                elif spatial_output == "coarse":
                    pred_subject_spatial_mix = torch.zeros((bsz,), dtype=image.dtype, device=image.device)
                    pred_subject_box = pred_subject_box_coarse
                    pred_subject_valid_logit = pred_subject_valid_logit_coarse
                else:
                    spatial_mix_input = torch.cat([cond_token, pred_subject_spatial_token], dim=-1)
                    pred_subject_spatial_mix = torch.sigmoid(self.subject_spatial_mix_head(spatial_mix_input).squeeze(-1))
                    mix = pred_subject_spatial_mix.view(bsz, 1)
                    pred_subject_box = pred_subject_box_coarse * (1.0 - mix) + pred_subject_box_spatial * mix
                    pred_subject_valid_logit = (
                        pred_subject_valid_logit_coarse * (1.0 - pred_subject_spatial_mix)
                        + pred_subject_valid_logit_spatial * pred_subject_spatial_mix
                    )
                pred_subject_valid_prob = torch.sigmoid(pred_subject_valid_logit)
        prior_token: torch.Tensor | None = None
        prior_valid = None
        if self.uses_subject_prior:
            if subject_prior_box is None:
                subject_prior_box = torch.zeros((bsz, 4), dtype=image.dtype, device=image.device)
            else:
                subject_prior_box = subject_prior_box.to(image.device).to(image.dtype)
            if subject_prior_valid is None:
                subject_prior_valid = torch.zeros((bsz,), dtype=image.dtype, device=image.device)
            else:
                subject_prior_valid = subject_prior_valid.to(image.device).to(image.dtype).view(bsz)
            if subject_prior_reliability is None:
                subject_prior_reliability = torch.zeros((bsz,), dtype=image.dtype, device=image.device)
            else:
                subject_prior_reliability = subject_prior_reliability.to(image.device).to(image.dtype).view(bsz)
            if self.config.use_pred_subject_box_as_prior and pred_subject_box is not None and pred_subject_valid_prob is not None:
                pred_box_for_prior = pred_subject_box.detach() if self.config.detach_pred_subject_prior else pred_subject_box
                pred_valid_for_prior = pred_subject_valid_prob.detach() if self.config.detach_pred_subject_prior else pred_subject_valid_prob
                pred_rel_for_prior = pred_valid_for_prior
                if pred_subject_prior_alpha is None:
                    alpha = torch.ones((bsz,), dtype=image.dtype, device=image.device)
                elif torch.is_tensor(pred_subject_prior_alpha):
                    alpha = pred_subject_prior_alpha.to(device=image.device, dtype=image.dtype).view(-1)
                else:
                    alpha = torch.full((bsz,), float(pred_subject_prior_alpha), dtype=image.dtype, device=image.device)
                alpha = alpha.clamp(0.0, 1.0)
                box_alpha = alpha.view(bsz, 1)
                subject_prior_box = subject_prior_box * (1.0 - box_alpha) + pred_box_for_prior * box_alpha
                subject_prior_valid = subject_prior_valid * (1.0 - alpha) + pred_valid_for_prior * alpha
                subject_prior_reliability = subject_prior_reliability * (1.0 - alpha) + pred_rel_for_prior * alpha
            prior_token = self.subject_prior_encoder(
                _subject_prior_feature(subject_prior_box, subject_prior_valid, subject_prior_reliability)
            ) * subject_prior_valid.view(bsz, 1)
            prior_valid = subject_prior_valid
        proposal_boxes, proposal_logits, proposal_tokens = self._proposals(
            cond_token,
            target_ar_id,
            image_ar_log,
            letterbox_content_box,
            prior_token,
        )
        subject_box_for_refine = pred_subject_box if pred_subject_box is not None else pred_subject_box_coarse
        subject_valid_logit_for_refine = pred_subject_valid_logit if pred_subject_valid_logit is not None else pred_subject_valid_logit_coarse
        if (
            subject_box_for_refine is not None
            and subject_valid_logit_for_refine is not None
            and self.subject_proposal_head is not None
            and self.subject_refine_proj is not None
            and self.subject_refine_box_head is not None
            and self.subject_refine_valid_head is not None
        ):
            subject_proposal_logits = self.subject_proposal_head(proposal_tokens).squeeze(-1)
            subject_weights = torch.softmax(subject_proposal_logits, dim=1)
            proposal_summary_token = (proposal_tokens * subject_weights.unsqueeze(-1)).sum(dim=1)
            proposal_summary_box = (proposal_boxes * subject_weights.unsqueeze(-1)).sum(dim=1)
            anchor_box = 0.5 * (proposal_summary_box + subject_box_for_refine)
            refine_input = torch.cat(
                [
                    cond_token,
                    proposal_summary_token,
                    _box_feature_vector(anchor_box),
                    _box_feature_vector(subject_box_for_refine),
                    _box_feature_vector(proposal_summary_box),
                ],
                dim=-1,
            )
            refine_token = self.subject_refine_proj(refine_input)
            pred_subject_box = _apply_subject_box_delta(
                anchor_box,
                self.subject_refine_box_head(refine_token),
                coord_space=self.config.subject_box_coord_space,
                letterbox_content_box=letterbox_content_box,
            )
            pred_subject_valid_logit = subject_valid_logit_for_refine + self.subject_refine_valid_head(refine_token).squeeze(-1)
            pred_subject_valid_prob = torch.sigmoid(pred_subject_valid_logit)
        runtime_baseline_boxes: torch.Tensor | None = None
        runtime_baseline_scored: dict[str, torch.Tensor] | None = None
        using_generated_as_candidates = boxes is None
        if boxes is None:
            boxes = proposal_boxes
            valid = torch.ones((bsz, proposal_boxes.shape[1]), dtype=image.dtype, device=image.device)
            candidate_is_base = torch.zeros_like(valid)
            runtime_baseline_boxes = _runtime_baseline_boxes(target_ar_id, image_ar_log, letterbox_content_box)
            runtime_baseline_valid = torch.ones((bsz, runtime_baseline_boxes.shape[1]), dtype=image.dtype, device=image.device)
            runtime_baseline_is_base = torch.ones_like(runtime_baseline_valid)
            runtime_baseline_scored = self._score_box_set(
                features=features,
                global_feat=global_feat,
                ar_emb=ar_emb,
                image_ar_log=image_ar_log,
                boxes=runtime_baseline_boxes,
                valid=runtime_baseline_valid,
                candidate_is_base=runtime_baseline_is_base,
                box_meta=None,
            )
            runtime_baseline_scored = {
                **runtime_baseline_scored,
                "return_logits": self._return_logits(
                    runtime_baseline_scored["candidate_tokens"],
                    runtime_baseline_scored["utility_logits"],
                    candidate_is_base=runtime_baseline_is_base,
                ),
            }
        if valid is None:
            valid = torch.ones((bsz, boxes.shape[1]), dtype=image.dtype, device=image.device)
        if candidate_is_base is None:
            candidate_is_base = torch.zeros_like(valid)

        scored = self._score_box_set(
            features=features,
            global_feat=global_feat,
            ar_emb=ar_emb,
            image_ar_log=image_ar_log,
            boxes=boxes,
            valid=valid,
            candidate_is_base=candidate_is_base,
            box_meta=box_meta,
        )
        cand_token = scored["candidate_tokens"]
        utility_logits = scored["utility_logits"]
        positive_logits = scored["positive_logits"]
        risk_logits = scored["risk_logits"]
        return_logits = self._return_logits(cand_token, utility_logits, candidate_is_base=candidate_is_base)

        masked_utility = utility_logits.masked_fill(valid <= 0, -1e4)
        attn = torch.softmax(masked_utility, dim=1) * valid.to(utility_logits.dtype)
        attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-6)
        weighted_token = (cand_token * attn.unsqueeze(-1)).sum(dim=1)
        base_mask = (candidate_is_base > 0).to(utility_logits.dtype) * valid.to(utility_logits.dtype)
        has_base = base_mask.sum(dim=1, keepdim=True) > 0
        base_score = torch.where(
            has_base,
            torch.sigmoid(utility_logits.masked_fill(base_mask <= 0, -1e4).max(dim=1, keepdim=True).values),
            torch.zeros((bsz, 1), dtype=utility_logits.dtype, device=utility_logits.device),
        )
        if runtime_baseline_scored is not None:
            base_score = torch.sigmoid(runtime_baseline_scored["utility_logits"].max(dim=1, keepdim=True).values)
        best_score = torch.sigmoid(masked_utility.max(dim=1, keepdim=True).values)
        global_cond = torch.cat([global_feat, ar_emb, image_ar_log.float().view(bsz, 1)], dim=-1)
        route_input = global_feat if self.config.route_image_only else global_cond
        if not self.config.route_image_only and self.config.route_use_subject_prior and prior_token is not None:
            route_input = torch.cat([route_input, prior_token], dim=-1)
        if not self.config.route_image_only and self.config.route_use_candidate_context:
            route_input = torch.cat([route_input, weighted_token], dim=-1)
        if not self.config.route_image_only and self.config.route_use_subject_box_features:
            if subject_prior_box is None:
                route_subject_box = torch.zeros((bsz, 4), dtype=image.dtype, device=image.device)
            else:
                route_subject_box = subject_prior_box.to(image.device).to(image.dtype)
            if subject_prior_valid is None:
                route_subject_valid = torch.zeros((bsz,), dtype=image.dtype, device=image.device)
            else:
                route_subject_valid = subject_prior_valid.to(image.device).to(image.dtype).view(bsz)
            if subject_prior_reliability is None:
                route_subject_reliability = torch.zeros((bsz,), dtype=image.dtype, device=image.device)
            else:
                route_subject_reliability = subject_prior_reliability.to(image.device).to(image.dtype).view(bsz)
            route_input = torch.cat(
                [route_input, _subject_prior_feature(route_subject_box, route_subject_valid, route_subject_reliability)],
                dim=-1,
            )
        if not self.config.route_image_only and self.config.route_use_subject_spatial_token:
            if pred_subject_spatial_token is None:
                pred_subject_spatial_token = torch.zeros_like(cond_token)
            route_input = torch.cat([route_input, pred_subject_spatial_token], dim=-1)

        route_fine_logits = self.route_head(route_input)
        route_kind_logits = self.route_kind_head(route_input) if self.route_kind_head is not None else None
        route_cardinality_logits = self.route_cardinality_head(route_input) if self.route_cardinality_head is not None else None
        if self.route_image_residual_head is not None:
            residual_weight = float(self.config.route_image_residual_weight)
            route_fine_logits = route_fine_logits + residual_weight * self.route_image_residual_head(global_feat)
            if route_kind_logits is not None and self.route_image_residual_kind_head is not None:
                route_kind_logits = route_kind_logits + residual_weight * self.route_image_residual_kind_head(global_feat)
            if route_cardinality_logits is not None and self.route_image_residual_cardinality_head is not None:
                route_cardinality_logits = route_cardinality_logits + residual_weight * self.route_image_residual_cardinality_head(global_feat)
        route_logits = route_fine_logits
        if str(self.config.route_decode_mode) == "hierarchical":
            route_logits = _compose_hierarchical_route_logits(
                route_fine_logits,
                route_kind_logits,
                route_cardinality_logits,
                kind_weight=float(self.config.route_decode_kind_weight),
                cardinality_weight=float(self.config.route_decode_cardinality_weight),
            )

        if self.subject_valid_route_head is not None and pred_subject_valid_logit is not None:
            route_valid_logits = route_logits.detach() if self.config.subject_valid_route_calibrator_detach else route_logits
            route_valid_probs = torch.softmax(route_valid_logits.float(), dim=1).to(dtype=cond_token.dtype)
            valid_logit_in = pred_subject_valid_logit.to(dtype=cond_token.dtype).view(bsz, 1).clamp(-10.0, 10.0)
            valid_prob_in = torch.sigmoid(valid_logit_in)
            route_valid_input = torch.cat([cond_token, route_valid_probs, valid_logit_in, valid_prob_in], dim=-1)
            pred_subject_valid_logit = pred_subject_valid_logit + self.subject_valid_route_head(route_valid_input).squeeze(-1)
            pred_subject_valid_prob = torch.sigmoid(pred_subject_valid_logit)

        if self.route_score_proj is not None:
            route_context_logits = route_logits.detach() if self.config.route_condition_detach else route_logits
            route_context = self.route_score_proj(torch.softmax(route_context_logits.float(), dim=1).to(dtype=cand_token.dtype))
            cand_token = cand_token + route_context.unsqueeze(1)
            utility_logits = self.utility_head(cand_token).squeeze(-1)
            positive_logits = self.positive_head(cand_token).squeeze(-1)
            risk_logits = self.risk_head(cand_token).squeeze(-1)
            return_logits = self._return_logits(cand_token, utility_logits, candidate_is_base=candidate_is_base)
            scored = {
                **scored,
                "candidate_tokens": cand_token,
                "utility_logits": utility_logits,
                "positive_logits": positive_logits,
                "risk_logits": risk_logits,
                "macro_logits": self.macro_head(cand_token),
                "checklist_class_logits": self.checklist_class_head(cand_token),
                "checklist_applicability_logits": self.checklist_applicability_head(cand_token),
                "detail_score_logits": self.detail_score_head(cand_token),
                "why_tag_logits": self.why_tag_head(cand_token),
            }
            if runtime_baseline_scored is not None:
                baseline_token = runtime_baseline_scored["candidate_tokens"] + route_context.unsqueeze(1)
                baseline_utility_logits = self.utility_head(baseline_token).squeeze(-1)
                runtime_baseline_scored = {
                    **runtime_baseline_scored,
                    "candidate_tokens": baseline_token,
                    "utility_logits": baseline_utility_logits,
                    "return_logits": self._return_logits(
                        baseline_token,
                        baseline_utility_logits,
                        candidate_is_base=runtime_baseline_is_base,
                    ),
                    "positive_logits": self.positive_head(baseline_token).squeeze(-1),
                    "risk_logits": self.risk_head(baseline_token).squeeze(-1),
                    "macro_logits": self.macro_head(baseline_token),
                    "checklist_class_logits": self.checklist_class_head(baseline_token),
                    "checklist_applicability_logits": self.checklist_applicability_head(baseline_token),
                    "detail_score_logits": self.detail_score_head(baseline_token),
                    "why_tag_logits": self.why_tag_head(baseline_token),
                }
            masked_utility = utility_logits.masked_fill(valid <= 0, -1e4)
            attn = torch.softmax(masked_utility, dim=1) * valid.to(utility_logits.dtype)
            attn = attn / attn.sum(dim=1, keepdim=True).clamp_min(1e-6)
            weighted_token = (cand_token * attn.unsqueeze(-1)).sum(dim=1)
            base_mask = (candidate_is_base > 0).to(utility_logits.dtype) * valid.to(utility_logits.dtype)
            has_base = base_mask.sum(dim=1, keepdim=True) > 0
            base_score = torch.where(
                has_base,
                torch.sigmoid(utility_logits.masked_fill(base_mask <= 0, -1e4).max(dim=1, keepdim=True).values),
                torch.zeros((bsz, 1), dtype=utility_logits.dtype, device=utility_logits.device),
            )
            if runtime_baseline_scored is not None:
                base_score = torch.sigmoid(runtime_baseline_scored["utility_logits"].max(dim=1, keepdim=True).values)
            best_score = torch.sigmoid(masked_utility.max(dim=1, keepdim=True).values)

        policy_input = torch.cat([global_cond, weighted_token, base_score, best_score], dim=-1)
        if self.config.policy_use_subject_prior and prior_token is not None:
            policy_input = torch.cat([policy_input, prior_token], dim=-1)
        policy_token = self.policy_head(policy_input)
        decision_logits = self.decision_head(policy_token)
        decision_source_logit = self.decision_source_head(policy_token).squeeze(-1) if self.decision_source_head is not None else None
        decision_source_pair_logit_value = None
        use_late_pair_source = bool(self.config.decision_source_pair_after_return_deltas)
        if not use_late_pair_source:
            decision_source_pair_logit = self._decision_source_pair_logit(
                policy_token=policy_token,
                candidate_tokens=cand_token,
                boxes=boxes,
                return_logits=return_logits,
                utility_logits=utility_logits,
                valid=valid,
                candidate_is_base=candidate_is_base,
                route_logits=route_logits,
                decision_logits=decision_logits,
                pred_subject_box=pred_subject_box,
                pred_subject_valid_logit=pred_subject_valid_logit,
            )
            if decision_source_pair_logit is not None:
                decision_source_pair_logit_value = decision_source_pair_logit
                if decision_source_logit is None:
                    decision_source_logit = decision_source_pair_logit
                else:
                    decision_source_logit = decision_source_logit + decision_source_pair_logit
        decision_source_raw_logit = decision_source_logit
        if (
            not use_late_pair_source
            and decision_source_logit is not None
            and float(self.config.decision_source_logit_threshold) != 0.0
        ):
            decision_source_logit = decision_source_logit - float(self.config.decision_source_logit_threshold)
        policy_score_pred = None
        if self.policy_score_predictor is not None:
            policy_score_pred = torch.sigmoid(self.policy_score_predictor(policy_token).squeeze(-1))
            decision_logits = decision_logits + _decision_logits_from_policy_score(
                policy_score_pred,
                low=self.config.policy_score_threshold_low,
                high=self.config.policy_score_threshold_high,
            )
        if not use_late_pair_source:
            return_logits = self._apply_decision_source_bias(
                return_logits,
                decision_logits,
                candidate_is_base=candidate_is_base,
                decision_source_logit=decision_source_logit,
            )
        return_logits = self._apply_return_context_delta(
            return_logits,
            candidate_tokens=cand_token,
            boxes=boxes,
            candidate_is_base=candidate_is_base,
            policy_token=policy_token,
            route_logits=route_logits,
            decision_logits=decision_logits,
            utility_logits=utility_logits,
            positive_logits=positive_logits,
            risk_logits=risk_logits,
        )
        return_logits = self._apply_return_policy_match_delta(
            return_logits,
            candidate_tokens=cand_token,
            boxes=boxes,
            valid=valid,
            candidate_is_base=candidate_is_base,
            policy_token=policy_token,
            route_logits=route_logits,
            decision_logits=decision_logits,
            utility_logits=utility_logits,
            positive_logits=positive_logits,
            risk_logits=risk_logits,
        )
        return_logits = self._apply_return_policy_source_delta(
            return_logits,
            candidate_tokens=cand_token,
            boxes=boxes,
            valid=valid,
            candidate_is_base=candidate_is_base,
            policy_token=policy_token,
            route_logits=route_logits,
            decision_logits=decision_logits,
            utility_logits=utility_logits,
            positive_logits=positive_logits,
            risk_logits=risk_logits,
        )
        return_logits = self._apply_return_competition_delta(
            return_logits,
            candidate_tokens=cand_token,
            boxes=boxes,
            valid=valid,
            candidate_is_base=candidate_is_base,
            route_logits=route_logits,
            decision_logits=decision_logits,
            utility_logits=utility_logits,
            positive_logits=positive_logits,
            risk_logits=risk_logits,
        )
        return_logits = self._apply_return_set_refiner_delta(
            return_logits,
            candidate_tokens=cand_token,
            boxes=boxes,
            valid=valid,
            candidate_is_base=candidate_is_base,
            policy_token=policy_token,
            route_logits=route_logits,
            decision_logits=decision_logits,
            utility_logits=utility_logits,
            positive_logits=positive_logits,
            risk_logits=risk_logits,
        )
        return_logits = self._apply_return_action_decoder_delta(
            return_logits,
            candidate_tokens=cand_token,
            boxes=boxes,
            valid=valid,
            candidate_is_base=candidate_is_base,
            policy_token=policy_token,
            route_logits=route_logits,
            decision_logits=decision_logits,
            utility_logits=utility_logits,
            positive_logits=positive_logits,
            risk_logits=risk_logits,
            pred_subject_box=pred_subject_box,
            pred_subject_valid_logit=pred_subject_valid_logit,
        )
        if use_late_pair_source:
            decision_source_pair_logit = self._decision_source_pair_logit(
                policy_token=policy_token,
                candidate_tokens=cand_token,
                boxes=boxes,
                return_logits=return_logits,
                utility_logits=utility_logits,
                valid=valid,
                candidate_is_base=candidate_is_base,
                route_logits=route_logits,
                decision_logits=decision_logits,
                pred_subject_box=pred_subject_box,
                pred_subject_valid_logit=pred_subject_valid_logit,
            )
            if decision_source_pair_logit is not None:
                decision_source_pair_logit_value = decision_source_pair_logit
                if decision_source_logit is None:
                    decision_source_logit = decision_source_pair_logit
                else:
                    decision_source_logit = decision_source_logit + decision_source_pair_logit
            decision_source_raw_logit = decision_source_logit
            if decision_source_logit is not None and float(self.config.decision_source_logit_threshold) != 0.0:
                decision_source_logit = decision_source_logit - float(self.config.decision_source_logit_threshold)
            return_logits = self._apply_decision_source_bias(
                return_logits,
                decision_logits,
                candidate_is_base=candidate_is_base,
                decision_source_logit=decision_source_logit,
            )
        source_gate_logit = self._return_score_source_gate_logit(
            policy_token=policy_token,
            candidate_tokens=cand_token,
            boxes=boxes,
            return_logits=return_logits,
            utility_logits=utility_logits,
            positive_logits=positive_logits,
            risk_logits=risk_logits,
            valid=valid,
            candidate_is_base=candidate_is_base,
            route_logits=route_logits,
            decision_logits=decision_logits,
            pred_subject_box=pred_subject_box,
            pred_subject_valid_logit=pred_subject_valid_logit,
        )
        source_gate_return_logits = self._apply_return_score_source_gate(
            return_logits,
            source_gate_logit,
            candidate_is_base=candidate_is_base,
            valid=valid,
        )
        decision_conditioned_return_logits = self._apply_decision_conditioned_return_mask(
            return_logits,
            decision_logits,
            candidate_is_base=candidate_is_base,
            valid=valid,
            decision_source_logit=decision_source_logit,
        )
        source_mixture_return_logits = self._apply_decision_source_mixture_prior(
            return_logits,
            decision_logits,
            candidate_is_base=candidate_is_base,
            valid=valid,
            decision_source_logit=decision_source_logit,
        )
        if runtime_baseline_scored is not None and "return_logits" in runtime_baseline_scored:
            baseline_return_logits = self._apply_decision_source_bias(
                runtime_baseline_scored["return_logits"],
                decision_logits,
                candidate_is_base=runtime_baseline_is_base,
                decision_source_logit=decision_source_logit,
            )
            baseline_return_logits = self._apply_return_context_delta(
                baseline_return_logits,
                candidate_tokens=runtime_baseline_scored["candidate_tokens"],
                boxes=runtime_baseline_boxes,
                candidate_is_base=runtime_baseline_is_base,
                policy_token=policy_token,
                route_logits=route_logits,
                decision_logits=decision_logits,
                utility_logits=runtime_baseline_scored["utility_logits"],
                positive_logits=runtime_baseline_scored["positive_logits"],
                risk_logits=runtime_baseline_scored["risk_logits"],
            )
            runtime_baseline_scored = {
                **runtime_baseline_scored,
                "return_logits": self._apply_return_policy_match_delta(
                    baseline_return_logits,
                    candidate_tokens=runtime_baseline_scored["candidate_tokens"],
                    boxes=runtime_baseline_boxes,
                    valid=runtime_baseline_valid,
                    candidate_is_base=runtime_baseline_is_base,
                    policy_token=policy_token,
                    route_logits=route_logits,
                    decision_logits=decision_logits,
                    utility_logits=runtime_baseline_scored["utility_logits"],
                    positive_logits=runtime_baseline_scored["positive_logits"],
                    risk_logits=runtime_baseline_scored["risk_logits"],
                ),
            }
            runtime_baseline_scored = {
                **runtime_baseline_scored,
                "return_logits": self._apply_return_policy_source_delta(
                    runtime_baseline_scored["return_logits"],
                    candidate_tokens=runtime_baseline_scored["candidate_tokens"],
                    boxes=runtime_baseline_boxes,
                    valid=runtime_baseline_valid,
                    candidate_is_base=runtime_baseline_is_base,
                    policy_token=policy_token,
                    route_logits=route_logits,
                    decision_logits=decision_logits,
                    utility_logits=runtime_baseline_scored["utility_logits"],
                    positive_logits=runtime_baseline_scored["positive_logits"],
                    risk_logits=runtime_baseline_scored["risk_logits"],
                ),
            }
            runtime_baseline_scored = {
                **runtime_baseline_scored,
                "return_logits": self._apply_return_competition_delta(
                    runtime_baseline_scored["return_logits"],
                    candidate_tokens=runtime_baseline_scored["candidate_tokens"],
                    boxes=runtime_baseline_boxes,
                    valid=runtime_baseline_valid,
                    candidate_is_base=runtime_baseline_is_base,
                    route_logits=route_logits,
                    decision_logits=decision_logits,
                    utility_logits=runtime_baseline_scored["utility_logits"],
                    positive_logits=runtime_baseline_scored["positive_logits"],
                    risk_logits=runtime_baseline_scored["risk_logits"],
                ),
            }
            runtime_baseline_scored = {
                **runtime_baseline_scored,
                "return_logits": self._apply_return_set_refiner_delta(
                    runtime_baseline_scored["return_logits"],
                    candidate_tokens=runtime_baseline_scored["candidate_tokens"],
                    boxes=runtime_baseline_boxes,
                    valid=runtime_baseline_valid,
                    candidate_is_base=runtime_baseline_is_base,
                    policy_token=policy_token,
                    route_logits=route_logits,
                    decision_logits=decision_logits,
                    utility_logits=runtime_baseline_scored["utility_logits"],
                    positive_logits=runtime_baseline_scored["positive_logits"],
                    risk_logits=runtime_baseline_scored["risk_logits"],
                ),
            }
            runtime_baseline_scored = {
                **runtime_baseline_scored,
                "return_logits": self._apply_return_action_decoder_delta(
                    runtime_baseline_scored["return_logits"],
                    candidate_tokens=runtime_baseline_scored["candidate_tokens"],
                    boxes=runtime_baseline_boxes,
                    valid=runtime_baseline_valid,
                    candidate_is_base=runtime_baseline_is_base,
                    policy_token=policy_token,
                    route_logits=route_logits,
                    decision_logits=decision_logits,
                    utility_logits=runtime_baseline_scored["utility_logits"],
                    positive_logits=runtime_baseline_scored["positive_logits"],
                    risk_logits=runtime_baseline_scored["risk_logits"],
                    pred_subject_box=pred_subject_box,
                    pred_subject_valid_logit=pred_subject_valid_logit,
                ),
            }

        out = {
            "utility_logits": utility_logits,
            "positive_logits": positive_logits,
            "risk_logits": risk_logits,
            "return_logits": return_logits,
            "macro_logits": scored["macro_logits"],
            "checklist_class_logits": scored["checklist_class_logits"],
            "checklist_applicability_logits": scored["checklist_applicability_logits"],
            "detail_score_logits": scored["detail_score_logits"],
            "why_tag_logits": scored["why_tag_logits"],
            "route_logits": route_logits,
            "route_fine_logits": route_fine_logits,
            "decision_logits": decision_logits,
            "delta_pred": self.delta_head(policy_token).squeeze(-1),
            "proposal_boxes": proposal_boxes,
            "proposal_logits": proposal_logits,
            "proposal_tokens": proposal_tokens,
            "candidate_tokens": cand_token,
        }
        if self.config.return_score_decision_conditioned:
            out["decision_conditioned_return_logits"] = decision_conditioned_return_logits
        if self.config.return_score_source_mixture:
            out["source_mixture_return_logits"] = source_mixture_return_logits
        if source_gate_logit is not None:
            out["source_gate_logit"] = source_gate_logit
        if self.config.return_score_source_gate:
            out["source_gate_return_logits"] = source_gate_return_logits
        if route_kind_logits is not None:
            out["route_kind_logits"] = route_kind_logits
        if route_cardinality_logits is not None:
            out["route_cardinality_logits"] = route_cardinality_logits
        if decision_source_logit is not None:
            out["decision_source_logit"] = decision_source_logit
            out["decision_source_logit_threshold"] = decision_source_logit.new_tensor(float(self.config.decision_source_logit_threshold))
            if decision_source_raw_logit is not decision_source_logit:
                out["decision_source_raw_logit"] = decision_source_raw_logit
        if decision_source_pair_logit_value is not None:
            out["decision_source_pair_logit"] = decision_source_pair_logit_value
        if pred_subject_box is not None and pred_subject_valid_logit is not None:
            out["pred_subject_box"] = pred_subject_box
            out["pred_subject_valid_logit"] = pred_subject_valid_logit
            out["pred_subject_valid"] = torch.sigmoid(pred_subject_valid_logit)
        if pred_subject_box_coarse is not None and pred_subject_valid_logit_coarse is not None:
            out["pred_subject_box_coarse"] = pred_subject_box_coarse
            out["pred_subject_valid_logit_coarse"] = pred_subject_valid_logit_coarse
            out["pred_subject_valid_coarse"] = torch.sigmoid(pred_subject_valid_logit_coarse)
        if pred_subject_box_spatial is not None and pred_subject_valid_logit_spatial is not None:
            out["pred_subject_box_spatial"] = pred_subject_box_spatial
            out["pred_subject_valid_logit_spatial"] = pred_subject_valid_logit_spatial
            out["pred_subject_valid_spatial"] = torch.sigmoid(pred_subject_valid_logit_spatial)
        if pred_subject_spatial_mix is not None:
            out["pred_subject_spatial_mix"] = pred_subject_spatial_mix
        if pred_subject_spatial_heatmap is not None:
            out["pred_subject_spatial_heatmap"] = pred_subject_spatial_heatmap
        if pred_subject_spatial_heatmap_logits is not None:
            out["pred_subject_spatial_heatmap_logits"] = pred_subject_spatial_heatmap_logits
        if subject_proposal_logits is not None:
            out["subject_proposal_logits"] = subject_proposal_logits
        if policy_score_pred is not None:
            out["policy_score_pred"] = policy_score_pred
        if runtime_baseline_scored is not None and runtime_baseline_boxes is not None:
            out["runtime_baseline_boxes"] = runtime_baseline_boxes
            out["runtime_baseline_utility_logits"] = runtime_baseline_scored["utility_logits"]
            if "return_logits" in runtime_baseline_scored:
                out["runtime_baseline_return_logits"] = runtime_baseline_scored["return_logits"]
                if self.config.return_score_source_gate:
                    out["runtime_baseline_source_gate_return_logits"] = self._apply_return_score_source_gate(
                        runtime_baseline_scored["return_logits"],
                        source_gate_logit,
                        candidate_is_base=runtime_baseline_is_base,
                        valid=runtime_baseline_valid,
                    )
                if self.config.return_score_decision_conditioned:
                    out["runtime_baseline_decision_conditioned_return_logits"] = self._apply_decision_conditioned_return_mask(
                        runtime_baseline_scored["return_logits"],
                        decision_logits,
                        candidate_is_base=runtime_baseline_is_base,
                        valid=runtime_baseline_valid,
                        decision_source_logit=decision_source_logit,
                    )
                if self.config.return_score_source_mixture:
                    out["runtime_baseline_source_mixture_return_logits"] = self._apply_decision_source_mixture_prior(
                        runtime_baseline_scored["return_logits"],
                        decision_logits,
                        candidate_is_base=runtime_baseline_is_base,
                        valid=runtime_baseline_valid,
                        decision_source_logit=decision_source_logit,
                    )
            out["runtime_baseline_positive_logits"] = runtime_baseline_scored["positive_logits"]
            out["runtime_baseline_risk_logits"] = runtime_baseline_scored["risk_logits"]
            out["runtime_baseline_macro_logits"] = runtime_baseline_scored["macro_logits"]
            out["runtime_baseline_checklist_class_logits"] = runtime_baseline_scored["checklist_class_logits"]
            out["runtime_baseline_checklist_applicability_logits"] = runtime_baseline_scored["checklist_applicability_logits"]
            out["runtime_baseline_detail_score_logits"] = runtime_baseline_scored["detail_score_logits"]
            out["runtime_baseline_why_tag_logits"] = runtime_baseline_scored["why_tag_logits"]
        if prior_valid is not None and subject_prior_box is not None:
            out["subject_prior_box"] = subject_prior_box
            out["subject_prior_valid"] = prior_valid
        if return_generated_scores and not using_generated_as_candidates:
            generated_valid = torch.ones((bsz, proposal_boxes.shape[1]), dtype=valid.dtype, device=proposal_boxes.device)
            generated_is_base = torch.zeros_like(generated_valid)
            generated_scored = self._score_box_set(
                features=features,
                global_feat=global_feat,
                ar_emb=ar_emb,
                image_ar_log=image_ar_log,
                boxes=proposal_boxes,
                valid=generated_valid,
                candidate_is_base=generated_is_base,
                box_meta=None,
            )
            if self.route_score_proj is not None:
                route_context_logits = route_logits.detach() if self.config.route_condition_detach else route_logits
                route_context = self.route_score_proj(torch.softmax(route_context_logits.float(), dim=1).to(dtype=proposal_boxes.dtype))
                generated_token = generated_scored["candidate_tokens"] + route_context.unsqueeze(1)
                generated_utility_logits = self.utility_head(generated_token).squeeze(-1)
                generated_scored = {
                    **generated_scored,
                    "candidate_tokens": generated_token,
                    "utility_logits": generated_utility_logits,
                    "positive_logits": self.positive_head(generated_token).squeeze(-1),
                    "risk_logits": self.risk_head(generated_token).squeeze(-1),
                }
            generated_return_logits = self._return_logits(
                generated_scored["candidate_tokens"],
                generated_scored["utility_logits"],
                candidate_is_base=generated_is_base,
            )
            generated_return_logits = self._apply_decision_source_bias(
                generated_return_logits,
                decision_logits,
                candidate_is_base=generated_is_base,
                decision_source_logit=decision_source_logit,
            )
            generated_return_logits = self._apply_return_context_delta(
                generated_return_logits,
                candidate_tokens=generated_scored["candidate_tokens"],
                boxes=proposal_boxes,
                candidate_is_base=generated_is_base,
                policy_token=policy_token,
                route_logits=route_logits,
                decision_logits=decision_logits,
                utility_logits=generated_scored["utility_logits"],
                positive_logits=generated_scored["positive_logits"],
                risk_logits=generated_scored["risk_logits"],
            )
            generated_return_logits = self._apply_return_policy_match_delta(
                generated_return_logits,
                candidate_tokens=generated_scored["candidate_tokens"],
                boxes=proposal_boxes,
                valid=generated_valid,
                candidate_is_base=generated_is_base,
                policy_token=policy_token,
                route_logits=route_logits,
                decision_logits=decision_logits,
                utility_logits=generated_scored["utility_logits"],
                positive_logits=generated_scored["positive_logits"],
                risk_logits=generated_scored["risk_logits"],
            )
            generated_return_logits = self._apply_return_policy_source_delta(
                generated_return_logits,
                candidate_tokens=generated_scored["candidate_tokens"],
                boxes=proposal_boxes,
                valid=generated_valid,
                candidate_is_base=generated_is_base,
                policy_token=policy_token,
                route_logits=route_logits,
                decision_logits=decision_logits,
                utility_logits=generated_scored["utility_logits"],
                positive_logits=generated_scored["positive_logits"],
                risk_logits=generated_scored["risk_logits"],
            )
            generated_return_logits = self._apply_return_competition_delta(
                generated_return_logits,
                candidate_tokens=generated_scored["candidate_tokens"],
                boxes=proposal_boxes,
                valid=generated_valid,
                candidate_is_base=generated_is_base,
                route_logits=route_logits,
                decision_logits=decision_logits,
                utility_logits=generated_scored["utility_logits"],
                positive_logits=generated_scored["positive_logits"],
                risk_logits=generated_scored["risk_logits"],
            )
            generated_return_logits = self._apply_return_set_refiner_delta(
                generated_return_logits,
                candidate_tokens=generated_scored["candidate_tokens"],
                boxes=proposal_boxes,
                valid=generated_valid,
                candidate_is_base=generated_is_base,
                policy_token=policy_token,
                route_logits=route_logits,
                decision_logits=decision_logits,
                utility_logits=generated_scored["utility_logits"],
                positive_logits=generated_scored["positive_logits"],
                risk_logits=generated_scored["risk_logits"],
            )
            generated_return_logits = self._apply_return_action_decoder_delta(
                generated_return_logits,
                candidate_tokens=generated_scored["candidate_tokens"],
                boxes=proposal_boxes,
                valid=generated_valid,
                candidate_is_base=generated_is_base,
                policy_token=policy_token,
                route_logits=route_logits,
                decision_logits=decision_logits,
                utility_logits=generated_scored["utility_logits"],
                positive_logits=generated_scored["positive_logits"],
                risk_logits=generated_scored["risk_logits"],
                pred_subject_box=pred_subject_box,
                pred_subject_valid_logit=pred_subject_valid_logit,
            )
            generated_decision_conditioned_return_logits = self._apply_decision_conditioned_return_mask(
                generated_return_logits,
                decision_logits,
                candidate_is_base=generated_is_base,
                valid=generated_valid,
                decision_source_logit=decision_source_logit,
            )
            generated_source_mixture_return_logits = self._apply_decision_source_mixture_prior(
                generated_return_logits,
                decision_logits,
                candidate_is_base=generated_is_base,
                valid=generated_valid,
                decision_source_logit=decision_source_logit,
            )
            generated_source_gate_return_logits = self._apply_return_score_source_gate(
                generated_return_logits,
                source_gate_logit,
                candidate_is_base=generated_is_base,
                valid=generated_valid,
            )
            out.update(
                {
                    "generated_boxes": proposal_boxes,
                    "generated_valid": generated_valid,
                    "generated_candidate_tokens": generated_scored["candidate_tokens"],
                    "generated_utility_logits": generated_scored["utility_logits"],
                    "generated_return_logits": generated_return_logits,
                    "generated_positive_logits": generated_scored["positive_logits"],
                    "generated_risk_logits": generated_scored["risk_logits"],
                }
            )
            if self.config.return_score_decision_conditioned:
                out["generated_decision_conditioned_return_logits"] = generated_decision_conditioned_return_logits
            if self.config.return_score_source_mixture:
                out["generated_source_mixture_return_logits"] = generated_source_mixture_return_logits
            if self.config.return_score_source_gate:
                out["generated_source_gate_return_logits"] = generated_source_gate_return_logits
        return out

    def _return_logits(
        self,
        candidate_tokens: torch.Tensor,
        utility_logits: torch.Tensor,
        *,
        candidate_is_base: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.return_score_delta_head is None:
            return utility_logits
        out = utility_logits + self.return_score_delta_head(candidate_tokens).squeeze(-1)
        if self.return_score_action_source_bias is not None and candidate_is_base is not None:
            source_idx = (candidate_is_base > 0).to(dtype=torch.long, device=utility_logits.device).clamp(0, 1)
            out = out + self.return_score_action_source_bias.to(device=utility_logits.device, dtype=utility_logits.dtype)[source_idx]
        if self.return_score_source_specific_head is not None and candidate_is_base is not None:
            source_idx = (candidate_is_base > 0).to(dtype=torch.long, device=utility_logits.device).clamp(0, 1)
            source_delta = self.return_score_source_specific_head(candidate_tokens).to(dtype=utility_logits.dtype)
            out = out + source_delta.gather(-1, source_idx.unsqueeze(-1)).squeeze(-1)
        return out

    def _return_score_source_gate_logit(
        self,
        *,
        policy_token: torch.Tensor,
        candidate_tokens: torch.Tensor,
        boxes: torch.Tensor,
        return_logits: torch.Tensor,
        utility_logits: torch.Tensor,
        positive_logits: torch.Tensor,
        risk_logits: torch.Tensor,
        valid: torch.Tensor,
        candidate_is_base: torch.Tensor,
        route_logits: torch.Tensor,
        decision_logits: torch.Tensor,
        pred_subject_box: torch.Tensor | None = None,
        pred_subject_valid_logit: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.return_score_source_gate_head is None:
            return None
        valid_bool = valid > 0
        base_mask = (candidate_is_base > 0) & valid_bool
        crop_mask = (~(candidate_is_base > 0)) & valid_bool
        has_base = base_mask.any(dim=1)
        has_crop = crop_mask.any(dim=1)
        best_base = return_logits.masked_fill(~base_mask, -1e4).max(dim=1)
        best_crop = return_logits.masked_fill(~crop_mask, -1e4).max(dim=1)
        base_idx = best_base.indices.clamp(0, candidate_tokens.shape[1] - 1)
        crop_idx = best_crop.indices.clamp(0, candidate_tokens.shape[1] - 1)
        gather_shape = (-1, 1, candidate_tokens.shape[-1])
        base_token = candidate_tokens.gather(1, base_idx.view(-1, 1, 1).expand(gather_shape)).squeeze(1)
        crop_token = candidate_tokens.gather(1, crop_idx.view(-1, 1, 1).expand(gather_shape)).squeeze(1)
        dtype = policy_token.dtype
        device = policy_token.device

        def best_value(values: torch.Tensor, mask: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
            return torch.where(
                mask.any(dim=1),
                values.masked_fill(~mask, -1e4).max(dim=1).values,
                fallback,
            ).to(device=device, dtype=dtype)

        zero = torch.zeros_like(best_base.values)
        base_ret = torch.where(has_base, best_base.values, zero).to(device=device, dtype=dtype)
        crop_ret = torch.where(has_crop, best_crop.values, zero).to(device=device, dtype=dtype)
        base_util = best_value(utility_logits, base_mask, zero)
        crop_util = best_value(utility_logits, crop_mask, zero)
        base_pos = best_value(positive_logits, base_mask, zero)
        crop_pos = best_value(positive_logits, crop_mask, zero)
        base_risk = best_value(risk_logits, base_mask, zero)
        crop_risk = best_value(risk_logits, crop_mask, zero)
        pair_scalars = torch.stack(
            [
                crop_ret - base_ret,
                crop_util - base_util,
                crop_pos - base_pos,
                crop_risk - base_risk,
                base_ret,
                crop_ret,
                base_util,
                crop_util,
                has_base.to(device=device, dtype=dtype),
                has_crop.to(device=device, dtype=dtype),
            ],
            dim=1,
        )
        route_prob = torch.softmax(route_logits.float(), dim=1).to(device=device, dtype=dtype)
        feature_parts = [
            policy_token.to(dtype=dtype),
            base_token.to(device=device, dtype=dtype),
            crop_token.to(device=device, dtype=dtype),
            (crop_token - base_token).to(device=device, dtype=dtype),
            torch.abs(crop_token - base_token).to(device=device, dtype=dtype),
            route_prob,
            decision_logits.to(device=device, dtype=dtype),
            pair_scalars,
        ]
        if self.config.return_score_source_gate_subject_state:
            subject_state, subject_pair = self._return_score_source_gate_subject_features(
                boxes=boxes,
                pred_subject_box=pred_subject_box,
                pred_subject_valid_logit=pred_subject_valid_logit,
                detach=bool(self.config.return_score_source_gate_subject_detach),
            )
            subject_dim = int(subject_pair.shape[-1])
            base_subject = subject_pair.gather(1, base_idx.view(-1, 1, 1).expand(-1, 1, subject_dim)).squeeze(1)
            crop_subject = subject_pair.gather(1, crop_idx.view(-1, 1, 1).expand(-1, 1, subject_dim)).squeeze(1)
            feature_parts.extend([subject_state, base_subject, crop_subject, crop_subject - base_subject])
        features = torch.cat(feature_parts, dim=1)
        if self.config.return_score_source_gate_detach:
            features = features.detach()
        source_gate_logit = self.return_score_source_gate_head(features).squeeze(-1)
        threshold = float(self.config.return_score_source_gate_logit_threshold)
        if threshold != 0.0:
            source_gate_logit = source_gate_logit - threshold
        return source_gate_logit

    def _return_score_source_gate_subject_features(
        self,
        *,
        boxes: torch.Tensor,
        pred_subject_box: torch.Tensor | None,
        pred_subject_valid_logit: torch.Tensor | None,
        detach: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = int(boxes.shape[0])
        count = int(boxes.shape[1])
        device = boxes.device
        dtype = boxes.dtype
        if pred_subject_box is None or pred_subject_valid_logit is None:
            return boxes.new_zeros((bsz, 12)), boxes.new_zeros((bsz, count, 6))
        subject_box = pred_subject_box.to(device=device, dtype=dtype)
        subject_valid_logit = pred_subject_valid_logit.to(device=device, dtype=dtype).view(bsz, 1).clamp(-10.0, 10.0)
        subject_valid_prob = torch.sigmoid(subject_valid_logit)
        subject_state = torch.cat([_box_feature_vector(subject_box), subject_valid_logit, subject_valid_prob], dim=-1)
        candidate_cxcywh = _xyxy_to_cxcywh(boxes.to(device=device, dtype=dtype))
        subject_cxcywh = _xyxy_to_cxcywh(subject_box).unsqueeze(1)
        subject_iou = _box_iou_torch(boxes.to(device=device, dtype=dtype), subject_box.unsqueeze(1)).squeeze(-1).unsqueeze(-1)
        center_delta = candidate_cxcywh[..., :2] - subject_cxcywh[..., :2]
        size_log_ratio = torch.log(candidate_cxcywh[..., 2:4].clamp_min(1e-6) / subject_cxcywh[..., 2:4].clamp_min(1e-6)).clamp(-4.0, 4.0)
        subject_pair = torch.cat(
            [
                subject_iou.clamp(0.0, 1.0),
                center_delta,
                size_log_ratio,
                subject_valid_prob.unsqueeze(1).expand(-1, count, -1),
            ],
            dim=-1,
        )
        if detach:
            subject_state = subject_state.detach()
            subject_pair = subject_pair.detach()
        return subject_state, subject_pair

    def _apply_return_score_source_gate(
        self,
        logits: torch.Tensor,
        source_gate_logit: torch.Tensor | None,
        *,
        candidate_is_base: torch.Tensor | None,
        valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.config.return_score_source_gate or source_gate_logit is None or candidate_is_base is None:
            return logits
        valid_bool = valid > 0 if valid is not None else torch.ones_like(candidate_is_base, dtype=torch.bool, device=logits.device)
        valid_bool = valid_bool.to(device=logits.device)
        is_base = (candidate_is_base > 0).to(device=logits.device)
        base_mask = is_base & valid_bool
        crop_mask = (~is_base) & valid_bool
        has_base = base_mask.any(dim=1, keepdim=True)
        has_crop = crop_mask.any(dim=1, keepdim=True)
        base_max = logits.masked_fill(~base_mask, -1e4).max(dim=1, keepdim=True).values
        crop_max = logits.masked_fill(~crop_mask, -1e4).max(dim=1, keepdim=True).values
        base_max = torch.where(has_base, base_max, torch.zeros_like(base_max))
        crop_max = torch.where(has_crop, crop_max, torch.zeros_like(crop_max))
        normalized = torch.where(is_base, logits - base_max, logits - crop_max)
        gate = source_gate_logit.view(-1, 1).to(device=logits.device, dtype=logits.dtype)
        base_log_prob = F.logsigmoid(-gate)
        crop_log_prob = F.logsigmoid(gate)
        source_bonus = torch.where(is_base, base_log_prob, crop_log_prob)
        gated = normalized + float(self.config.return_score_source_gate_strength) * source_bonus
        if valid is not None:
            gated = gated.masked_fill(~valid_bool, float(-1e4))
        return gated

    def _decision_source_pair_logit(
        self,
        *,
        policy_token: torch.Tensor,
        candidate_tokens: torch.Tensor,
        boxes: torch.Tensor,
        return_logits: torch.Tensor,
        utility_logits: torch.Tensor,
        valid: torch.Tensor,
        candidate_is_base: torch.Tensor,
        route_logits: torch.Tensor,
        decision_logits: torch.Tensor,
        pred_subject_box: torch.Tensor | None = None,
        pred_subject_valid_logit: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.decision_source_pair_head is None:
            return None
        valid_bool = valid > 0
        base_mask = (candidate_is_base > 0) & valid_bool
        crop_mask = (~(candidate_is_base > 0)) & valid_bool
        has_base = base_mask.any(dim=1)
        has_crop = crop_mask.any(dim=1)
        best_base = return_logits.masked_fill(~base_mask, -1e4).max(dim=1)
        best_crop = return_logits.masked_fill(~crop_mask, -1e4).max(dim=1)
        base_idx = best_base.indices.clamp(0, candidate_tokens.shape[1] - 1)
        crop_idx = best_crop.indices.clamp(0, candidate_tokens.shape[1] - 1)
        gather_shape = (-1, 1, candidate_tokens.shape[-1])
        base_token = candidate_tokens.gather(1, base_idx.view(-1, 1, 1).expand(gather_shape)).squeeze(1)
        crop_token = candidate_tokens.gather(1, crop_idx.view(-1, 1, 1).expand(gather_shape)).squeeze(1)
        dtype = policy_token.dtype
        device = policy_token.device
        base_ret = torch.where(has_base, best_base.values, torch.zeros_like(best_base.values)).to(device=device, dtype=dtype)
        crop_ret = torch.where(has_crop, best_crop.values, torch.zeros_like(best_crop.values)).to(device=device, dtype=dtype)
        base_util = torch.where(
            has_base,
            utility_logits.masked_fill(~base_mask, -1e4).max(dim=1).values,
            torch.zeros_like(base_ret),
        ).to(device=device, dtype=dtype)
        crop_util = torch.where(
            has_crop,
            utility_logits.masked_fill(~crop_mask, -1e4).max(dim=1).values,
            torch.zeros_like(crop_ret),
        ).to(device=device, dtype=dtype)
        pair_scalars = torch.stack(
            [
                crop_ret - base_ret,
                crop_util - base_util,
                base_ret,
                crop_ret,
                has_base.to(device=device, dtype=dtype),
                has_crop.to(device=device, dtype=dtype),
            ],
            dim=1,
        )
        route_prob = torch.softmax(route_logits.float(), dim=1).to(device=device, dtype=dtype)
        decision_context = decision_logits.to(device=device, dtype=dtype)
        feature_parts = [
            policy_token.to(dtype=dtype),
            base_token.to(device=device, dtype=dtype),
            crop_token.to(device=device, dtype=dtype),
            (crop_token - base_token).to(device=device, dtype=dtype),
            torch.abs(crop_token - base_token).to(device=device, dtype=dtype),
            route_prob,
            decision_context,
            pair_scalars,
        ]
        if self.config.decision_source_pair_subject_state:
            subject_state, subject_pair = self._decision_source_pair_subject_features(
                boxes=boxes,
                return_logits=return_logits,
                pred_subject_box=pred_subject_box,
                pred_subject_valid_logit=pred_subject_valid_logit,
            )
            subject_dim = int(subject_pair.shape[-1])
            base_subject = subject_pair.gather(1, base_idx.view(-1, 1, 1).expand(-1, 1, subject_dim)).squeeze(1)
            crop_subject = subject_pair.gather(1, crop_idx.view(-1, 1, 1).expand(-1, 1, subject_dim)).squeeze(1)
            feature_parts.extend([subject_state, base_subject, crop_subject, crop_subject - base_subject])
        features = torch.cat(feature_parts, dim=1)
        return self.decision_source_pair_head(features).squeeze(-1)

    def _decision_source_pair_subject_features(
        self,
        *,
        boxes: torch.Tensor,
        return_logits: torch.Tensor,
        pred_subject_box: torch.Tensor | None,
        pred_subject_valid_logit: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = int(return_logits.shape[0])
        count = int(return_logits.shape[1])
        device = return_logits.device
        dtype = return_logits.dtype
        if pred_subject_box is None or pred_subject_valid_logit is None:
            return return_logits.new_zeros((bsz, 12)), return_logits.new_zeros((bsz, count, 6))
        subject_box = pred_subject_box.to(device=device, dtype=dtype)
        subject_valid_logit = pred_subject_valid_logit.to(device=device, dtype=dtype).view(bsz, 1).clamp(-10.0, 10.0)
        subject_valid_prob = torch.sigmoid(subject_valid_logit)
        subject_state = torch.cat([_box_feature_vector(subject_box), subject_valid_logit, subject_valid_prob], dim=-1)
        candidate_cxcywh = _xyxy_to_cxcywh(boxes.to(device=device, dtype=dtype))
        subject_cxcywh = _xyxy_to_cxcywh(subject_box).unsqueeze(1)
        subject_iou = _box_iou_torch(boxes.to(device=device, dtype=dtype), subject_box.unsqueeze(1)).squeeze(-1).unsqueeze(-1)
        center_delta = candidate_cxcywh[..., :2] - subject_cxcywh[..., :2]
        size_log_ratio = torch.log(candidate_cxcywh[..., 2:4].clamp_min(1e-6) / subject_cxcywh[..., 2:4].clamp_min(1e-6)).clamp(-4.0, 4.0)
        subject_pair = torch.cat(
            [
                subject_iou.clamp(0.0, 1.0),
                center_delta,
                size_log_ratio,
                subject_valid_prob.unsqueeze(1).expand(-1, count, -1),
            ],
            dim=-1,
        )
        if self.config.decision_source_pair_subject_detach:
            subject_state = subject_state.detach()
            subject_pair = subject_pair.detach()
        return subject_state, subject_pair

    def _apply_decision_source_bias(
        self,
        logits: torch.Tensor,
        decision_logits: torch.Tensor | None,
        *,
        candidate_is_base: torch.Tensor | None = None,
        decision_source_logit: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.return_score_decision_source_scale is None or candidate_is_base is None:
            return logits
        crop_id = DECISION_VOCAB.index("crop")
        source_logits = None
        if (
            bool(self.config.return_score_decision_source_from_source_head)
            and decision_source_logit is not None
        ):
            source_logits = decision_source_logit.view(-1)
        else:
            if decision_logits is None:
                return logits
            if int(decision_logits.shape[-1]) <= crop_id:
                return logits
            base_ids = [idx for idx in range(int(decision_logits.shape[-1])) if idx != crop_id]
            if not base_ids:
                return logits
            source_logits = decision_logits[:, crop_id] - torch.logsumexp(decision_logits[:, base_ids], dim=1)
        if self.config.return_score_decision_source_detach:
            source_logits = source_logits.detach()
        crop_margin = source_logits
        source_sign = torch.where(
            candidate_is_base > 0,
            torch.full_like(candidate_is_base, -1.0),
            torch.ones_like(candidate_is_base),
        ).to(dtype=logits.dtype, device=logits.device)
        scale = torch.tanh(self.return_score_decision_source_scale.to(device=logits.device, dtype=logits.dtype))
        scale = scale * float(self.config.return_score_decision_source_max_scale)
        return logits + scale * crop_margin.to(device=logits.device, dtype=logits.dtype).unsqueeze(1) * source_sign

    def _apply_decision_conditioned_return_mask(
        self,
        logits: torch.Tensor,
        decision_logits: torch.Tensor | None,
        *,
        candidate_is_base: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
        decision_source_logit: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.config.return_score_decision_conditioned:
            return logits
        return _mask_logits_by_decision_source(
            logits,
            decision_logits,
            candidate_is_base,
            valid=valid,
            mask_value=float(self.config.return_score_decision_conditioned_mask_value),
            decision_source_logit=decision_source_logit
            if bool(self.config.return_score_decision_conditioned_from_source_head)
            else None,
        )

    def _apply_decision_source_mixture_prior(
        self,
        logits: torch.Tensor,
        decision_logits: torch.Tensor | None,
        *,
        candidate_is_base: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
        decision_source_logit: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.config.return_score_source_mixture:
            return logits
        if candidate_is_base is None:
            return logits
        source_log_probs = _decision_source_log_probs(
            decision_logits,
            decision_source_logit
            if bool(self.config.return_score_source_mixture_from_source_head)
            else None,
            detach=bool(self.config.return_score_source_mixture_detach),
            dtype=logits.dtype,
            device=logits.device,
        )
        if source_log_probs is None:
            return logits
        base_log_prob = source_log_probs[:, 0].unsqueeze(1)
        crop_log_prob = source_log_probs[:, 1].unsqueeze(1)
        is_base = (candidate_is_base > 0).to(device=logits.device)
        source_bonus = torch.where(is_base, base_log_prob, crop_log_prob)
        if valid is not None:
            source_bonus = source_bonus.masked_fill((valid <= 0).to(device=logits.device), 0.0)
        mixed_logits = logits + float(self.config.return_score_source_mixture_strength) * source_bonus
        if self.config.return_score_source_mixture_gap_threshold is None:
            return mixed_logits
        valid_bool = valid > 0 if valid is not None else torch.ones_like(mixed_logits, dtype=torch.bool)
        base_mask = (candidate_is_base > 0).to(device=mixed_logits.device) & valid_bool
        crop_mask = (~(candidate_is_base > 0).to(device=mixed_logits.device)) & valid_bool
        has_base = base_mask.any(dim=1, keepdim=True)
        has_crop = crop_mask.any(dim=1, keepdim=True)
        if not torch.any(has_base & has_crop):
            return mixed_logits
        best_base = mixed_logits.masked_fill(~base_mask, -1e4).max(dim=1, keepdim=True).values
        best_crop = mixed_logits.masked_fill(~crop_mask, -1e4).max(dim=1, keepdim=True).values
        choose_crop = (best_crop - best_base) > float(self.config.return_score_source_mixture_gap_threshold)
        keep_mask = torch.where(choose_crop, crop_mask, base_mask)
        keep_mask = torch.where(has_base & has_crop, keep_mask, valid_bool)
        return mixed_logits.masked_fill(~keep_mask, float(self.config.return_score_source_mixture_gap_mask_value))

    def _apply_return_context_delta(
        self,
        logits: torch.Tensor,
        *,
        candidate_tokens: torch.Tensor,
        boxes: torch.Tensor,
        candidate_is_base: torch.Tensor,
        policy_token: torch.Tensor,
        route_logits: torch.Tensor,
        decision_logits: torch.Tensor,
        utility_logits: torch.Tensor,
        positive_logits: torch.Tensor,
        risk_logits: torch.Tensor,
    ) -> torch.Tensor:
        if self.return_score_context_head is None:
            return logits
        token = candidate_tokens
        policy = policy_token
        route = torch.softmax(route_logits.float(), dim=1).to(device=logits.device, dtype=logits.dtype)
        decision = decision_logits.to(device=logits.device, dtype=logits.dtype)
        geom = _box_geom(boxes, candidate_is_base).to(device=logits.device, dtype=logits.dtype)
        score_features = torch.stack([utility_logits, positive_logits, risk_logits], dim=-1).to(device=logits.device, dtype=logits.dtype)
        if self.config.return_score_context_detach:
            token = token.detach()
            policy = policy.detach()
            route = route.detach()
            decision = decision.detach()
            geom = geom.detach()
            score_features = score_features.detach()
        count = int(candidate_tokens.shape[1])
        context = torch.cat(
            [
                token.to(device=logits.device, dtype=logits.dtype),
                policy.to(device=logits.device, dtype=logits.dtype).unsqueeze(1).expand(-1, count, -1),
                route.unsqueeze(1).expand(-1, count, -1),
                decision.unsqueeze(1).expand(-1, count, -1),
                score_features,
                geom,
            ],
            dim=-1,
        )
        return logits + self.return_score_context_head(context).squeeze(-1)

    def _apply_return_policy_match_delta(
        self,
        logits: torch.Tensor,
        *,
        candidate_tokens: torch.Tensor,
        boxes: torch.Tensor,
        valid: torch.Tensor,
        candidate_is_base: torch.Tensor,
        policy_token: torch.Tensor,
        route_logits: torch.Tensor,
        decision_logits: torch.Tensor,
        utility_logits: torch.Tensor,
        positive_logits: torch.Tensor,
        risk_logits: torch.Tensor,
    ) -> torch.Tensor:
        if self.return_score_policy_match_head is None or self.return_score_policy_match_proj is None:
            return logits
        token = candidate_tokens
        policy = policy_token
        route = torch.softmax(route_logits.float(), dim=1).to(device=logits.device, dtype=logits.dtype)
        decision = decision_logits.to(device=logits.device, dtype=logits.dtype)
        geom = _box_geom(boxes, candidate_is_base).to(device=logits.device, dtype=logits.dtype)
        score_features = torch.stack([utility_logits, positive_logits, risk_logits], dim=-1).to(device=logits.device, dtype=logits.dtype)
        if self.config.return_score_policy_match_detach:
            token = token.detach()
            policy = policy.detach()
            route = route.detach()
            decision = decision.detach()
            geom = geom.detach()
            score_features = score_features.detach()
        token = token.to(device=logits.device, dtype=logits.dtype)
        policy_match = self.return_score_policy_match_proj(policy.to(device=logits.device, dtype=logits.dtype))
        count = int(candidate_tokens.shape[1])
        policy_expanded = policy_match.unsqueeze(1).expand(-1, count, -1)
        match = torch.cat(
            [
                token,
                policy_expanded,
                token * policy_expanded,
                torch.abs(token - policy_expanded),
                route.unsqueeze(1).expand(-1, count, -1),
                decision.unsqueeze(1).expand(-1, count, -1),
                score_features,
                geom,
            ],
            dim=-1,
        )
        return logits + self.return_score_policy_match_head(match).squeeze(-1)

    def _apply_return_policy_source_delta(
        self,
        logits: torch.Tensor,
        *,
        candidate_tokens: torch.Tensor,
        boxes: torch.Tensor,
        valid: torch.Tensor,
        candidate_is_base: torch.Tensor,
        policy_token: torch.Tensor,
        route_logits: torch.Tensor,
        decision_logits: torch.Tensor,
        utility_logits: torch.Tensor,
        positive_logits: torch.Tensor,
        risk_logits: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.return_score_policy_source_head is None
            or self.return_score_policy_source_proj is None
            or candidate_is_base is None
        ):
            return logits
        token = candidate_tokens
        policy = policy_token
        route = torch.softmax(route_logits.float(), dim=1).to(device=logits.device, dtype=logits.dtype)
        decision = decision_logits.to(device=logits.device, dtype=logits.dtype)
        geom = _box_geom(boxes, candidate_is_base).to(device=logits.device, dtype=logits.dtype)
        score_features = torch.stack([utility_logits, positive_logits, risk_logits], dim=-1).to(device=logits.device, dtype=logits.dtype)
        if self.config.return_score_policy_source_detach:
            token = token.detach()
            policy = policy.detach()
            route = route.detach()
            decision = decision.detach()
            geom = geom.detach()
            score_features = score_features.detach()
        token = token.to(device=logits.device, dtype=logits.dtype)
        policy_source = self.return_score_policy_source_proj(policy.to(device=logits.device, dtype=logits.dtype))
        count = int(candidate_tokens.shape[1])
        policy_expanded = policy_source.unsqueeze(1).expand(-1, count, -1)
        match = torch.cat(
            [
                token,
                policy_expanded,
                token * policy_expanded,
                torch.abs(token - policy_expanded),
                route.unsqueeze(1).expand(-1, count, -1),
                decision.unsqueeze(1).expand(-1, count, -1),
                score_features,
                geom,
            ],
            dim=-1,
        )
        source_delta = self.return_score_policy_source_head(match).to(device=logits.device, dtype=logits.dtype)
        source_idx = (candidate_is_base.to(device=logits.device) > 0).long().clamp(0, 1)
        selected_delta = source_delta.gather(-1, source_idx.unsqueeze(-1)).squeeze(-1)
        return logits + selected_delta

    def _apply_return_competition_delta(
        self,
        logits: torch.Tensor,
        *,
        candidate_tokens: torch.Tensor,
        boxes: torch.Tensor,
        valid: torch.Tensor,
        candidate_is_base: torch.Tensor,
        route_logits: torch.Tensor,
        decision_logits: torch.Tensor,
        utility_logits: torch.Tensor,
        positive_logits: torch.Tensor,
        risk_logits: torch.Tensor,
    ) -> torch.Tensor:
        if self.return_score_competition_head is None:
            return logits
        token = candidate_tokens
        route = torch.softmax(route_logits.float(), dim=1).to(device=logits.device, dtype=logits.dtype)
        decision = decision_logits.to(device=logits.device, dtype=logits.dtype)
        geom = _box_geom(boxes, candidate_is_base).to(device=logits.device, dtype=logits.dtype)
        score_features = torch.stack([utility_logits, positive_logits, risk_logits], dim=-1).to(device=logits.device, dtype=logits.dtype)
        valid_mask = (valid > 0).to(device=logits.device)
        masked_logits = logits.masked_fill(~valid_mask, -1e4)
        weights = torch.softmax(masked_logits.float(), dim=1).to(device=logits.device, dtype=logits.dtype)
        weights = weights * valid_mask.to(dtype=logits.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        context_token = (candidate_tokens.to(device=logits.device, dtype=logits.dtype) * weights.unsqueeze(-1)).sum(dim=1)
        if self.config.return_score_competition_detach:
            token = token.detach()
            context_token = context_token.detach()
            route = route.detach()
            decision = decision.detach()
            geom = geom.detach()
            score_features = score_features.detach()
        token = token.to(device=logits.device, dtype=logits.dtype)
        count = int(candidate_tokens.shape[1])
        context_expanded = context_token.to(device=logits.device, dtype=logits.dtype).unsqueeze(1).expand(-1, count, -1)
        match = torch.cat(
            [
                token,
                context_expanded,
                token * context_expanded,
                torch.abs(token - context_expanded),
                route.unsqueeze(1).expand(-1, count, -1),
                decision.unsqueeze(1).expand(-1, count, -1),
                score_features,
                geom,
            ],
            dim=-1,
        )
        return logits + self.return_score_competition_head(match).squeeze(-1)

    def _apply_return_set_refiner_delta(
        self,
        logits: torch.Tensor,
        *,
        candidate_tokens: torch.Tensor,
        boxes: torch.Tensor,
        valid: torch.Tensor,
        candidate_is_base: torch.Tensor,
        policy_token: torch.Tensor,
        route_logits: torch.Tensor,
        decision_logits: torch.Tensor,
        utility_logits: torch.Tensor,
        positive_logits: torch.Tensor,
        risk_logits: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.return_score_set_refiner_input is None
            or self.return_score_set_refiner is None
            or self.return_score_set_refiner_head is None
        ):
            return logits
        token = candidate_tokens.to(device=logits.device, dtype=logits.dtype)
        policy = policy_token.to(device=logits.device, dtype=logits.dtype)
        route = torch.softmax(route_logits.float(), dim=1).to(device=logits.device, dtype=logits.dtype)
        decision = decision_logits.to(device=logits.device, dtype=logits.dtype)
        geom = _box_geom(boxes, candidate_is_base).to(device=logits.device, dtype=logits.dtype)
        score_features = torch.stack([utility_logits, positive_logits, risk_logits], dim=-1).to(device=logits.device, dtype=logits.dtype)
        if self.config.return_score_set_refiner_detach:
            token = token.detach()
            policy = policy.detach()
            route = route.detach()
            decision = decision.detach()
            geom = geom.detach()
            score_features = score_features.detach()
        count = int(candidate_tokens.shape[1])
        refiner_input = torch.cat(
            [
                token,
                policy.unsqueeze(1).expand(-1, count, -1),
                route.unsqueeze(1).expand(-1, count, -1),
                decision.unsqueeze(1).expand(-1, count, -1),
                score_features,
                geom,
            ],
            dim=-1,
        )
        refiner_token = self.return_score_set_refiner_input(refiner_input)
        key_padding_mask = valid.to(device=logits.device) <= 0
        refiner_token = self.return_score_set_refiner(refiner_token, src_key_padding_mask=key_padding_mask)
        return logits + self.return_score_set_refiner_head(refiner_token).squeeze(-1)

    def _apply_return_action_decoder_delta(
        self,
        logits: torch.Tensor,
        *,
        candidate_tokens: torch.Tensor,
        boxes: torch.Tensor,
        valid: torch.Tensor,
        candidate_is_base: torch.Tensor,
        policy_token: torch.Tensor,
        route_logits: torch.Tensor,
        decision_logits: torch.Tensor,
        utility_logits: torch.Tensor,
        positive_logits: torch.Tensor,
        risk_logits: torch.Tensor,
        pred_subject_box: torch.Tensor | None = None,
        pred_subject_valid_logit: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            self.return_score_action_query is None
            or self.return_score_action_memory is None
            or self.return_score_action_decoder is None
            or self.return_score_action_decoder_head is None
        ):
            return logits
        token = candidate_tokens.to(device=logits.device, dtype=logits.dtype)
        policy = policy_token.to(device=logits.device, dtype=logits.dtype)
        route = torch.softmax(route_logits.float(), dim=1).to(device=logits.device, dtype=logits.dtype)
        decision = decision_logits.to(device=logits.device, dtype=logits.dtype)
        geom = _box_geom(boxes, candidate_is_base).to(device=logits.device, dtype=logits.dtype)
        score_features = torch.stack([utility_logits, positive_logits, risk_logits], dim=-1).to(device=logits.device, dtype=logits.dtype)
        subject_state, subject_pair = self._return_action_subject_features(
            boxes=boxes,
            logits=logits,
            pred_subject_box=pred_subject_box,
            pred_subject_valid_logit=pred_subject_valid_logit,
        )
        if self.config.return_score_action_decoder_detach:
            token = token.detach()
            policy = policy.detach()
            route = route.detach()
            decision = decision.detach()
            geom = geom.detach()
            score_features = score_features.detach()
        query_parts = [policy, route, decision]
        if subject_state is not None:
            query_parts.append(subject_state)
        query_input = torch.cat(query_parts, dim=-1)
        action_query = self.return_score_action_query(query_input).unsqueeze(1)
        memory_parts = [token, score_features, geom]
        if subject_pair is not None:
            memory_parts.append(subject_pair)
        memory_input = torch.cat(memory_parts, dim=-1)
        memory = self.return_score_action_memory(memory_input)
        key_padding_mask = valid.to(device=logits.device) <= 0
        action_context = self.return_score_action_decoder(
            action_query,
            memory,
            memory_key_padding_mask=key_padding_mask,
        ).squeeze(1)
        count = int(candidate_tokens.shape[1])
        context = action_context.unsqueeze(1).expand(-1, count, -1)
        action_feature_parts = [
            token,
            context,
            token * context,
            torch.abs(token - context),
            route.unsqueeze(1).expand(-1, count, -1),
            decision.unsqueeze(1).expand(-1, count, -1),
            score_features,
            geom,
        ]
        if subject_state is not None:
            action_feature_parts.append(subject_state.unsqueeze(1).expand(-1, count, -1))
        if subject_pair is not None:
            action_feature_parts.append(subject_pair)
        action_features = torch.cat(action_feature_parts, dim=-1)
        action_delta = self.return_score_action_decoder_head(action_features).squeeze(-1)
        if self.config.return_score_action_decoder_replace:
            return action_delta.masked_fill(valid.to(device=logits.device) <= 0, float(-1e4))
        return logits + action_delta

    def _return_action_subject_features(
        self,
        *,
        boxes: torch.Tensor,
        logits: torch.Tensor,
        pred_subject_box: torch.Tensor | None,
        pred_subject_valid_logit: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.config.return_score_action_decoder_subject_state:
            return None, None
        bsz = int(logits.shape[0])
        count = int(logits.shape[1])
        device = logits.device
        dtype = logits.dtype
        if pred_subject_box is None or pred_subject_valid_logit is None:
            subject_state = logits.new_zeros((bsz, 12))
            subject_pair = logits.new_zeros((bsz, count, 6))
            return subject_state, subject_pair
        subject_box = pred_subject_box.to(device=device, dtype=dtype)
        subject_valid_logit = pred_subject_valid_logit.to(device=device, dtype=dtype).view(bsz, 1)
        subject_valid_logit = subject_valid_logit.clamp(-10.0, 10.0)
        subject_valid_prob = torch.sigmoid(subject_valid_logit)
        subject_state = torch.cat([_box_feature_vector(subject_box), subject_valid_logit, subject_valid_prob], dim=-1)
        candidate_cxcywh = _xyxy_to_cxcywh(boxes.to(device=device, dtype=dtype))
        subject_cxcywh = _xyxy_to_cxcywh(subject_box).unsqueeze(1)
        subject_iou = _box_iou_torch(boxes.to(device=device, dtype=dtype), subject_box.unsqueeze(1)).squeeze(-1).unsqueeze(-1)
        center_delta = candidate_cxcywh[..., :2] - subject_cxcywh[..., :2]
        size_log_ratio = torch.log(candidate_cxcywh[..., 2:4].clamp_min(1e-6) / subject_cxcywh[..., 2:4].clamp_min(1e-6)).clamp(-4.0, 4.0)
        subject_pair = torch.cat(
            [
                subject_iou.clamp(0.0, 1.0),
                center_delta,
                size_log_ratio,
                subject_valid_prob.unsqueeze(1).expand(-1, count, -1),
            ],
            dim=-1,
        )
        if self.config.return_score_action_decoder_subject_detach:
            subject_state = subject_state.detach()
            subject_pair = subject_pair.detach()
        return subject_state, subject_pair


def _listwise_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, temperature: float = 0.15) -> torch.Tensor:
    masked_logits = logits.masked_fill(valid <= 0, -1e4)
    log_prob = torch.log_softmax(masked_logits, dim=1)
    target_score = (target / float(temperature)).masked_fill(valid <= 0, -1e4)
    target_prob = torch.softmax(target_score, dim=1) * valid
    target_prob = target_prob / target_prob.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return -(target_prob * log_prob).sum(dim=1).mean()


def _teacher_distribution_loss(
    logits: torch.Tensor,
    teacher_prob: torch.Tensor,
    valid: torch.Tensor,
    *,
    temperature: float = 0.12,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid_bool = valid > 0
    teacher = teacher_prob * valid
    row_valid = (valid_bool.sum(dim=1) > 1) & (teacher.sum(dim=1) > 0)
    if not torch.any(row_valid):
        zero = logits.sum() * 0.0
        return zero, {"teacher_distill_loss": zero.detach(), "teacher_distill_entropy": zero.detach()}
    teacher = teacher / teacher.sum(dim=1, keepdim=True).clamp_min(1e-6)
    scaled_logits = logits / max(1e-4, float(temperature))
    log_prob = torch.log_softmax(scaled_logits.masked_fill(~valid_bool, -1e4), dim=1)
    loss = -(teacher * log_prob).sum(dim=1)
    out = (loss * row_valid.to(loss.dtype)).sum() / row_valid.to(loss.dtype).sum().clamp_min(1.0)
    with torch.no_grad():
        entropy = -(teacher.clamp_min(1e-8) * teacher.clamp_min(1e-8).log()).sum(dim=1)
        entropy = (entropy * row_valid.to(entropy.dtype)).sum() / row_valid.to(entropy.dtype).sum().clamp_min(1.0)
    return out, {"teacher_distill_loss": out.detach(), "teacher_distill_entropy": entropy.detach()}


def _pairwise_loss(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, margin: float = 0.03, candidate_weight: torch.Tensor | None = None) -> torch.Tensor:
    diff_target = target.unsqueeze(2) - target.unsqueeze(1)
    diff_pred = logits.unsqueeze(2) - logits.unsqueeze(1)
    pair_valid = (valid.unsqueeze(2) * valid.unsqueeze(1)) > 0
    pair_valid = pair_valid & (diff_target.abs() >= float(margin))
    if not torch.any(pair_valid):
        return logits.sum() * 0.0
    sign = torch.where(diff_target > 0, torch.ones_like(diff_target), -torch.ones_like(diff_target))
    loss = F.softplus(-sign * diff_pred)
    if candidate_weight is not None:
        pair_weight = torch.sqrt(candidate_weight.unsqueeze(2).clamp_min(0.05) * candidate_weight.unsqueeze(1).clamp_min(0.05))
        loss = loss * pair_weight
        denom = (pair_valid.to(loss.dtype) * pair_weight).sum().clamp_min(1.0)
        return (loss * pair_valid.to(loss.dtype)).sum() / denom
    return loss[pair_valid].mean()


def _explicit_pairwise_loss(logits: torch.Tensor, batch: dict[str, torch.Tensor], *, margin: float = 0.0) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if "pair_valid" not in batch:
        zero = logits.sum() * 0.0
        return zero, {"explicit_pairwise_acc": zero.detach(), "explicit_pairwise_count": zero.detach()}
    pair_valid = batch["pair_valid"].to(logits.device).to(logits.dtype)
    if pair_valid.sum() <= 0:
        zero = logits.sum() * 0.0
        return zero, {"explicit_pairwise_acc": zero.detach(), "explicit_pairwise_count": zero.detach()}
    pair_i = batch["pair_i"].to(logits.device).long().clamp(0, logits.shape[1] - 1)
    pair_j = batch["pair_j"].to(logits.device).long().clamp(0, logits.shape[1] - 1)
    pair_label = batch["pair_label"].to(logits.device).to(logits.dtype)
    pair_weight = batch["pair_weight"].to(logits.device).to(logits.dtype).clamp_min(0.05)
    pred_i = logits.gather(1, pair_i)
    pred_j = logits.gather(1, pair_j)
    signed_margin = pair_label * (pred_i - pred_j)
    loss = F.softplus(-(signed_margin - float(margin)))
    weighted_valid = pair_valid * pair_weight
    out = (loss * weighted_valid).sum() / weighted_valid.sum().clamp_min(1.0)
    with torch.no_grad():
        acc = (((signed_margin > 0).to(logits.dtype) * pair_valid).sum() / pair_valid.sum().clamp_min(1.0)).detach()
    return out, {"explicit_pairwise_acc": acc, "explicit_pairwise_count": pair_valid.sum().detach()}


def _top1_risk_loss(
    logits: torch.Tensor,
    positive_target: torch.Tensor,
    risk_target: torch.Tensor,
    valid: torch.Tensor,
    *,
    margin: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pos_mask = (positive_target > 0).to(logits.dtype) * valid
    risk_mask = (risk_target > 0).to(logits.dtype) * valid
    if pos_mask.sum() <= 0 or risk_mask.sum() <= 0:
        zero = logits.sum() * 0.0
        return zero, {"risk_suppression_acc": zero.detach(), "top1_risk_rate": zero.detach()}
    pos_logits = logits.masked_fill(pos_mask <= 0, -1e4)
    risk_logits = logits.masked_fill(risk_mask <= 0, -1e4)
    best_pos = pos_logits.max(dim=1).values
    best_risk = risk_logits.max(dim=1).values
    row_valid = ((pos_mask.sum(dim=1) > 0) & (risk_mask.sum(dim=1) > 0)).to(logits.dtype)
    loss = F.softplus(-(best_pos - best_risk - float(margin)))
    out = (loss * row_valid).sum() / row_valid.sum().clamp_min(1.0)
    with torch.no_grad():
        acc = (((best_pos > best_risk).to(logits.dtype) * row_valid).sum() / row_valid.sum().clamp_min(1.0)).detach()
        prob = torch.sigmoid(logits).masked_fill(valid <= 0, -1.0)
        top_idx = prob.argmax(dim=1)
        top_risk = risk_target.gather(1, top_idx[:, None]).squeeze(1)
        top1_risk_rate = (top_risk * (valid.sum(dim=1) > 0).to(logits.dtype)).sum() / (valid.sum(dim=1) > 0).to(logits.dtype).sum().clamp_min(1.0)
    return out, {"risk_suppression_acc": acc, "top1_risk_rate": top1_risk_rate.detach()}


def _positive_margin_loss(
    logits: torch.Tensor,
    positive_target: torch.Tensor,
    valid: torch.Tensor,
    *,
    margin: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pos_mask = (positive_target > 0).to(torch.bool) & (valid > 0)
    neg_mask = (~pos_mask) & (valid > 0)
    row_valid = pos_mask.any(dim=1) & neg_mask.any(dim=1)
    if not torch.any(row_valid):
        zero = logits.sum() * 0.0
        return zero, {
            "positive_margin_loss": zero.detach(),
            "positive_margin_acc": zero.detach(),
            "positive_margin_gap": zero.detach(),
            "positive_margin_active_frac": zero.detach(),
        }
    best_pos = logits.masked_fill(~pos_mask, -1e4).max(dim=1).values
    best_neg = logits.masked_fill(~neg_mask, -1e4).max(dim=1).values
    gap = best_pos - best_neg
    loss = F.softplus(-(gap - float(margin)))
    row_weight = row_valid.to(logits.dtype)
    out = (loss * row_weight).sum() / row_weight.sum().clamp_min(1.0)
    with torch.no_grad():
        metrics = {
            "positive_margin_loss": out.detach(),
            "positive_margin_acc": (((gap > 0.0).to(logits.dtype) * row_weight).sum() / row_weight.sum().clamp_min(1.0)).detach(),
            "positive_margin_gap": ((gap * row_weight).sum() / row_weight.sum().clamp_min(1.0)).detach(),
            "positive_margin_active_frac": row_weight.mean().detach(),
        }
    return out, metrics


def _top_return_loss(
    logits: torch.Tensor,
    score_target: torch.Tensor,
    valid: torch.Tensor,
    candidate_weight: torch.Tensor,
    *,
    score_margin: float = 0.03,
    logit_margin: float = 0.25,
    temperature: float = 0.12,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid_bool = valid > 0
    row_valid = valid_bool.sum(dim=1) > 1
    if not torch.any(row_valid):
        zero = logits.sum() * 0.0
        return zero, {
            "top_return_loss": zero.detach(),
            "top_return_bag_loss": zero.detach(),
            "top_return_margin_loss": zero.detach(),
            "top_return_hit": zero.detach(),
            "top_return_score": zero.detach(),
            "top_return_regret": zero.detach(),
            "top_return_bag_size": zero.detach(),
        }

    masked_target = score_target.masked_fill(~valid_bool, -1.0)
    best_score = masked_target.max(dim=1).values
    top_bag = valid_bool & ((best_score.unsqueeze(1) - score_target) <= float(score_margin))
    top_bag = top_bag & row_valid.unsqueeze(1)

    scaled_logits = logits / max(1e-4, float(temperature))
    scaled_valid = scaled_logits.masked_fill(~valid_bool, -1e4)
    log_denom = torch.logsumexp(scaled_valid, dim=1)
    log_top = torch.logsumexp(scaled_logits.masked_fill(~top_bag, -1e4), dim=1)
    bag_loss = -(log_top - log_denom)

    top_logits = logits.masked_fill(~top_bag, -1e4).max(dim=1).values
    non_top = valid_bool & ~top_bag
    has_non_top = non_top.any(dim=1) & row_valid
    non_top_logits = logits.masked_fill(~non_top, -1e4).max(dim=1).values
    margin_loss = F.softplus(-(top_logits - non_top_logits - float(logit_margin)))

    top_weight = (candidate_weight * top_bag.to(candidate_weight.dtype)).sum(dim=1) / top_bag.to(candidate_weight.dtype).sum(dim=1).clamp_min(1.0)
    row_weight = top_weight.clamp(0.05, 1.0) * row_valid.to(logits.dtype)
    bag_out = (bag_loss * row_weight).sum() / row_weight.sum().clamp_min(1.0)
    margin_weight = row_weight * has_non_top.to(logits.dtype)
    margin_out = (margin_loss * margin_weight).sum() / margin_weight.sum().clamp_min(1.0)
    total = bag_out + margin_out

    with torch.no_grad():
        prob = torch.sigmoid(logits).masked_fill(~valid_bool, -1.0)
        top_idx = prob.argmax(dim=1)
        top_score = score_target.gather(1, top_idx[:, None]).squeeze(1)
        top_hit = top_bag.gather(1, top_idx[:, None]).squeeze(1).to(logits.dtype)
        regret = (best_score - top_score).clamp_min(0.0)
        denom = row_valid.to(logits.dtype).sum().clamp_min(1.0)
        metrics = {
            "top_return_loss": total.detach(),
            "top_return_bag_loss": bag_out.detach(),
            "top_return_margin_loss": margin_out.detach(),
            "top_return_hit": (top_hit * row_valid.to(logits.dtype)).sum() / denom,
            "top_return_score": (top_score * row_valid.to(logits.dtype)).sum() / denom,
            "top_return_regret": (regret * row_valid.to(logits.dtype)).sum() / denom,
            "top_return_bag_size": (top_bag.to(logits.dtype).sum(dim=1) * row_valid.to(logits.dtype)).sum() / denom,
        }
    return total, metrics


def _return_exact_loss(
    logits: torch.Tensor,
    score_target: torch.Tensor,
    valid: torch.Tensor,
    candidate_weight: torch.Tensor,
    *,
    min_gap: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid_bool = valid > 0
    row_valid = valid_bool.sum(dim=1) > 1
    if not torch.any(row_valid):
        zero = logits.sum() * 0.0
        return zero, {
            "return_exact_loss": zero.detach(),
            "return_exact_hit": zero.detach(),
            "return_exact_score": zero.detach(),
            "return_exact_gap": zero.detach(),
            "return_exact_active_frac": zero.detach(),
        }

    masked_target = score_target.masked_fill(~valid_bool, -1.0)
    top2 = masked_target.topk(k=min(2, int(masked_target.shape[1])), dim=1).values
    best_score = top2[:, 0]
    if top2.shape[1] > 1:
        second_score = top2[:, 1]
    else:
        second_score = torch.zeros_like(best_score)
    score_gap = (best_score - second_score).clamp_min(0.0)
    if float(min_gap) > 0.0:
        row_valid = row_valid & (score_gap >= float(min_gap))
    if not torch.any(row_valid):
        zero = logits.sum() * 0.0
        return zero, {
            "return_exact_loss": zero.detach(),
            "return_exact_hit": zero.detach(),
            "return_exact_score": zero.detach(),
            "return_exact_gap": zero.detach(),
            "return_exact_active_frac": zero.detach(),
        }

    target_idx = masked_target.argmax(dim=1)
    masked_logits = logits.masked_fill(~valid_bool, -1e4)
    loss = F.cross_entropy(masked_logits, target_idx, reduction="none")
    best_weight = candidate_weight.gather(1, target_idx[:, None]).squeeze(1).clamp(0.05, 1.0)
    row_weight = best_weight * row_valid.to(logits.dtype)
    out = (loss * row_weight).sum() / row_weight.sum().clamp_min(1.0)
    with torch.no_grad():
        pred_idx = masked_logits.argmax(dim=1)
        pred_score = score_target.gather(1, pred_idx[:, None]).squeeze(1)
        hit = (pred_idx == target_idx).to(logits.dtype)
        denom = row_valid.to(logits.dtype).sum().clamp_min(1.0)
        metrics = {
            "return_exact_loss": out.detach(),
            "return_exact_hit": (hit * row_valid.to(logits.dtype)).sum() / denom,
            "return_exact_score": (pred_score * row_valid.to(logits.dtype)).sum() / denom,
            "return_exact_gap": (score_gap * row_valid.to(logits.dtype)).sum() / denom,
            "return_exact_active_frac": row_valid.to(logits.dtype).mean(),
        }
    return out, metrics


def _decision_action_loss(
    logits: torch.Tensor,
    decision_target: torch.Tensor,
    valid: torch.Tensor,
    candidate_is_base: torch.Tensor,
    *,
    margin: float = 0.15,
    source_balanced: bool = False,
    source_balance_max_weight: float = 4.0,
    source_crop_weight_mult: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid_bool = valid > 0
    base_mask = (candidate_is_base > 0) & valid_bool
    crop_mask = (~(candidate_is_base > 0)) & valid_bool
    row_valid = base_mask.any(dim=1) & crop_mask.any(dim=1)
    if not torch.any(row_valid):
        zero = logits.sum() * 0.0
        return zero, {
            "action_consistency_loss": zero.detach(),
            "action_consistency_acc": zero.detach(),
            "top1_action_source_acc": zero.detach(),
            "baseline_preserve_target_rate": zero.detach(),
            "baseline_preserve_top1_hit": zero.detach(),
            "crop_action_top1_hit": zero.detach(),
        }

    best_base = logits.masked_fill(~base_mask, -1e4).max(dim=1).values
    best_crop = logits.masked_fill(~crop_mask, -1e4).max(dim=1).values
    crop_decision_id = DECISION_VOCAB.index("crop")
    target_is_base = (decision_target != crop_decision_id).to(logits.dtype)
    loss_base = F.softplus(-(best_base - best_crop - float(margin)))
    loss_crop = F.softplus(-(best_crop - best_base - float(margin)))
    metric_row_weight = row_valid.to(logits.dtype)
    row_weight = metric_row_weight
    if bool(source_balanced):
        base_rows_for_weight = metric_row_weight * target_is_base
        crop_rows_for_weight = metric_row_weight * (1.0 - target_is_base)
        base_count = base_rows_for_weight.sum()
        crop_count = crop_rows_for_weight.sum()
        if bool((base_count > 0).item()) and bool((crop_count > 0).item()):
            total_count = base_count + crop_count
            max_weight = max(1.0, float(source_balance_max_weight))
            base_scale = (total_count / (2.0 * base_count)).clamp(max=max_weight)
            crop_scale = (total_count / (2.0 * crop_count)).clamp(max=max_weight)
            row_weight = torch.where(target_is_base > 0.5, base_scale, crop_scale) * metric_row_weight
    crop_mult = max(0.0, float(source_crop_weight_mult))
    if crop_mult != 1.0:
        row_weight = row_weight * torch.where(
            target_is_base > 0.5,
            torch.ones_like(target_is_base),
            torch.full_like(target_is_base, crop_mult),
        )
    loss = torch.where(target_is_base > 0.5, loss_base, loss_crop)
    out = (loss * row_weight).sum() / row_weight.sum().clamp_min(1.0)

    with torch.no_grad():
        pred_is_base = (best_base >= best_crop).to(logits.dtype)
        top_idx = torch.sigmoid(logits).masked_fill(~valid_bool, -1.0).argmax(dim=1)
        top_is_base = candidate_is_base.gather(1, top_idx[:, None]).squeeze(1).to(logits.dtype)
        target_crop = 1.0 - target_is_base
        base_rows = metric_row_weight * target_is_base
        crop_rows = metric_row_weight * target_crop
        metrics = {
            "action_consistency_loss": out.detach(),
            "action_consistency_acc": (((pred_is_base == target_is_base).to(logits.dtype) * metric_row_weight).sum() / metric_row_weight.sum().clamp_min(1.0)).detach(),
            "top1_action_source_acc": (((top_is_base == target_is_base).to(logits.dtype) * metric_row_weight).sum() / metric_row_weight.sum().clamp_min(1.0)).detach(),
            "baseline_preserve_target_rate": (base_rows.sum() / metric_row_weight.sum().clamp_min(1.0)).detach(),
            "baseline_preserve_top1_hit": ((top_is_base * base_rows).sum() / base_rows.sum().clamp_min(1.0)).detach(),
            "crop_action_top1_hit": (((1.0 - top_is_base) * crop_rows).sum() / crop_rows.sum().clamp_min(1.0)).detach(),
            "action_source_balanced": torch.as_tensor(float(bool(source_balanced)), device=logits.device, dtype=logits.dtype),
            "action_source_crop_weight_mult": torch.as_tensor(crop_mult, device=logits.device, dtype=logits.dtype),
            }
    return out, metrics


def _decision_source_loss(
    source_logit: torch.Tensor | None,
    decision_target: torch.Tensor,
    reference: torch.Tensor,
    *,
    balanced_bce: bool = False,
    crop_rebalance_max_weight: float = 4.0,
    focal_gamma: float = 0.0,
    focal_alpha: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if source_logit is None:
        zero = reference.sum() * 0.0
        return zero, {
            "policy_source_loss": zero.detach(),
            "policy_source_acc": zero.detach(),
            "policy_source_base_recall": zero.detach(),
            "policy_source_crop_recall": zero.detach(),
            "policy_source_crop_target_rate": zero.detach(),
        }
    crop_id = DECISION_VOCAB.index("crop")
    logits = source_logit.view(-1)
    target = (decision_target == crop_id).to(logits.device).to(logits.dtype)
    loss_raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    gamma = max(0.0, float(focal_gamma))
    if gamma > 0.0:
        prob = torch.sigmoid(logits)
        p_t = torch.where(target > 0.5, prob, 1.0 - prob)
        loss_raw = loss_raw * (1.0 - p_t).clamp(min=0.0, max=1.0).pow(gamma)
    alpha_value = None if focal_alpha is None else float(focal_alpha)
    if alpha_value is not None:
        alpha_value = min(1.0, max(0.0, alpha_value))
        alpha_t = torch.where(
            target > 0.5,
            torch.full_like(target, alpha_value),
            torch.full_like(target, 1.0 - alpha_value),
        )
        loss_raw = loss_raw * alpha_t
    weight = torch.ones_like(loss_raw)
    if bool(balanced_bce):
        crop_count = target.sum()
        base_count = (1.0 - target).sum()
        if bool((crop_count > 0).item()) and bool((base_count > 0).item()):
            total_count = crop_count + base_count
            max_weight = max(1.0, float(crop_rebalance_max_weight))
            crop_scale = (total_count / (2.0 * crop_count)).clamp(max=max_weight)
            base_scale = (total_count / (2.0 * base_count)).clamp(max=max_weight)
            weight = torch.where(target > 0.5, crop_scale, base_scale)
    out = (loss_raw * weight).sum() / weight.sum().clamp_min(1.0)
    with torch.no_grad():
        pred_crop = (logits >= 0.0).to(logits.dtype)
        crop_rows = target > 0.5
        base_rows = ~crop_rows
        crop_denom = crop_rows.to(logits.dtype).sum().clamp_min(1.0)
        base_denom = base_rows.to(logits.dtype).sum().clamp_min(1.0)
        metrics = {
            "policy_source_loss": out.detach(),
            "policy_source_acc": (pred_crop == target).to(logits.dtype).mean().detach(),
            "policy_source_base_recall": (((1.0 - pred_crop) * base_rows.to(logits.dtype)).sum() / base_denom).detach(),
            "policy_source_crop_recall": ((pred_crop * crop_rows.to(logits.dtype)).sum() / crop_denom).detach(),
            "policy_source_crop_target_rate": target.mean().detach(),
            "policy_source_focal_gamma": torch.as_tensor(gamma, device=logits.device, dtype=logits.dtype),
            "policy_source_focal_alpha": torch.as_tensor(
                -1.0 if alpha_value is None else alpha_value,
                device=logits.device,
                dtype=logits.dtype,
            ),
        }
    return out, metrics


def _decision_source_margin_loss(
    source_logit: torch.Tensor | None,
    decision_target: torch.Tensor,
    reference: torch.Tensor,
    *,
    logit_margin: float = 0.35,
    source_balanced: bool = False,
    source_balance_max_weight: float = 4.0,
    source_crop_weight_mult: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if source_logit is None:
        zero = reference.sum() * 0.0
        return zero, {
            "decision_source_margin_loss": zero.detach(),
            "decision_source_margin_acc": zero.detach(),
            "decision_source_margin_crop_hit": zero.detach(),
            "decision_source_margin_base_hit": zero.detach(),
            "decision_source_margin_target_gap": zero.detach(),
            "decision_source_margin_crop_target_rate": zero.detach(),
        }
    crop_id = DECISION_VOCAB.index("crop")
    logits = source_logit.view(-1)
    target_is_crop = decision_target.to(device=logits.device) == crop_id
    target_gap = torch.where(target_is_crop, logits, -logits)
    loss_raw = F.softplus(-(target_gap - float(logit_margin)))
    row_weight = torch.ones_like(loss_raw)
    if bool(source_balanced):
        crop_count = target_is_crop.to(logits.dtype).sum()
        base_count = (~target_is_crop).to(logits.dtype).sum()
        if bool((crop_count > 0).item()) and bool((base_count > 0).item()):
            total_count = crop_count + base_count
            max_weight = max(1.0, float(source_balance_max_weight))
            crop_scale = (total_count / (2.0 * crop_count)).clamp(max=max_weight)
            base_scale = (total_count / (2.0 * base_count)).clamp(max=max_weight)
            row_weight = torch.where(target_is_crop, crop_scale, base_scale).to(dtype=logits.dtype, device=logits.device)
    crop_mult = max(0.0, float(source_crop_weight_mult))
    if crop_mult != 1.0:
        row_weight = row_weight * torch.where(
            target_is_crop,
            torch.full_like(row_weight, crop_mult),
            torch.ones_like(row_weight),
        )
    out = (loss_raw * row_weight).sum() / row_weight.sum().clamp_min(1.0)
    with torch.no_grad():
        pred_is_crop = logits >= 0.0
        crop_rows = target_is_crop
        base_rows = ~target_is_crop
        source_hit = (pred_is_crop == target_is_crop).to(logits.dtype)
        metrics = {
            "decision_source_margin_loss": out.detach(),
            "decision_source_margin_acc": source_hit.mean().detach(),
            "decision_source_margin_crop_hit": ((source_hit * crop_rows.to(logits.dtype)).sum() / crop_rows.to(logits.dtype).sum().clamp_min(1.0)).detach(),
            "decision_source_margin_base_hit": ((source_hit * base_rows.to(logits.dtype)).sum() / base_rows.to(logits.dtype).sum().clamp_min(1.0)).detach(),
            "decision_source_margin_target_gap": target_gap.mean().detach(),
            "decision_source_margin_crop_target_rate": crop_rows.to(logits.dtype).mean().detach(),
            "decision_source_margin_source_balanced": torch.as_tensor(float(bool(source_balanced)), device=logits.device, dtype=logits.dtype),
            "decision_source_margin_crop_weight_mult": torch.as_tensor(crop_mult, device=logits.device, dtype=logits.dtype),
        }
    return out, metrics


def _decision_source_log_probs(
    decision_logits: torch.Tensor | None,
    decision_source_logit: torch.Tensor | None,
    *,
    detach: bool = True,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor | None:
    crop_id = DECISION_VOCAB.index("crop")
    if decision_source_logit is not None:
        source_logits = decision_source_logit.view(-1)
        if detach:
            source_logits = source_logits.detach()
        crop_log_prob = F.logsigmoid(source_logits)
        base_log_prob = F.logsigmoid(-source_logits)
        out = torch.stack([base_log_prob, crop_log_prob], dim=1)
    else:
        if decision_logits is None or int(decision_logits.shape[-1]) <= crop_id:
            return None
        base_ids = [idx for idx in range(int(decision_logits.shape[-1])) if idx != crop_id]
        if not base_ids:
            return None
        source_logits = torch.stack(
            [
                torch.logsumexp(decision_logits[:, base_ids], dim=1),
                decision_logits[:, crop_id],
            ],
            dim=1,
        )
        if detach:
            source_logits = source_logits.detach()
        out = F.log_softmax(source_logits, dim=1)
    if dtype is not None or device is not None:
        out = out.to(dtype=dtype or out.dtype, device=device or out.device)
    return out


def _mask_logits_by_decision_source(
    logits: torch.Tensor,
    decision_logits: torch.Tensor | None,
    candidate_is_base: torch.Tensor | None,
    *,
    valid: torch.Tensor | None = None,
    mask_value: float = -10000.0,
    decision_source_logit: torch.Tensor | None = None,
) -> torch.Tensor:
    if candidate_is_base is None:
        return logits
    crop_id = DECISION_VOCAB.index("crop")
    if decision_source_logit is None and decision_logits is None:
        return logits
    if decision_source_logit is None and int(decision_logits.shape[-1]) <= crop_id:
        return logits
    valid_bool = valid > 0 if valid is not None else torch.ones_like(candidate_is_base, dtype=torch.bool, device=logits.device)
    valid_bool = valid_bool.to(device=logits.device)
    is_base = (candidate_is_base > 0).to(device=logits.device)
    if decision_source_logit is not None:
        pred_is_crop = decision_source_logit.view(-1).to(device=logits.device) >= 0.0
    else:
        pred_is_crop = decision_logits.argmax(dim=1).to(device=logits.device) == crop_id
    keep = torch.where(pred_is_crop.view(-1, 1), ~is_base, is_base) & valid_bool
    fallback = valid_bool if valid is not None else torch.ones_like(keep, dtype=torch.bool, device=logits.device)
    keep = torch.where(keep.any(dim=1, keepdim=True), keep, fallback)
    return logits.masked_fill(~keep, float(mask_value))


def _decision_conditioned_action_metrics(
    logits: torch.Tensor,
    decision_logits: torch.Tensor,
    decision_target: torch.Tensor,
    valid: torch.Tensor,
    candidate_is_base: torch.Tensor,
    return_target: torch.Tensor,
    score_target: torch.Tensor,
    *,
    score_margin: float = 0.03,
    decision_source_logit: torch.Tensor | None = None,
    sweep_logits: torch.Tensor | None = None,
    logits_already_conditioned: bool = False,
) -> dict[str, torch.Tensor]:
    valid_bool = valid > 0
    base_mask = (candidate_is_base > 0) & valid_bool
    crop_mask = (~(candidate_is_base > 0)) & valid_bool
    row_valid_bool = base_mask.any(dim=1) & crop_mask.any(dim=1)
    if not torch.any(row_valid_bool):
        zero = logits.sum() * 0.0
        return {
            "decision_source_acc": zero.detach(),
            "decision_conditioned_action_source_acc": zero.detach(),
            "decision_conditioned_baseline_preserve_top1_hit": zero.detach(),
            "decision_conditioned_crop_action_top1_hit": zero.detach(),
            "decision_conditioned_top_return_hit": zero.detach(),
            "decision_conditioned_top_return_score": zero.detach(),
            "decision_conditioned_top_return_regret": zero.detach(),
            "decision_conditioned_raw_top_return_hit": zero.detach(),
            "decision_conditioned_raw_top_return_score": zero.detach(),
            "decision_conditioned_raw_top_return_regret": zero.detach(),
            "decision_conditioned_active_frac": zero.detach(),
        }

    if bool(logits_already_conditioned):
        masked_logits = logits
    else:
        masked_logits = _mask_logits_by_decision_source(
            logits,
            decision_logits,
            candidate_is_base,
            valid=valid,
            decision_source_logit=decision_source_logit,
        )
    crop_id = DECISION_VOCAB.index("crop")
    target_is_base = (decision_target != crop_id).to(logits.dtype)
    if decision_source_logit is not None:
        pred_is_base = (decision_source_logit.view(-1).to(device=logits.device) < 0.0).to(logits.dtype)
    else:
        pred_is_base = (decision_logits.argmax(dim=1) != crop_id).to(logits.dtype)
    row_weight = row_valid_bool.to(logits.dtype)
    top_idx = torch.sigmoid(masked_logits).masked_fill(~valid_bool, -1.0).argmax(dim=1)
    top_is_base = candidate_is_base.gather(1, top_idx[:, None]).squeeze(1).to(logits.dtype)
    base_rows = row_weight * target_is_base
    crop_rows = row_weight * (1.0 - target_is_base)

    def top_metrics(target: torch.Tensor, prefix: str) -> dict[str, torch.Tensor]:
        masked_target = target.masked_fill(~valid_bool, -1.0)
        best_score = masked_target.max(dim=1).values
        top_bag = valid_bool & ((best_score.unsqueeze(1) - target) <= float(score_margin))
        top_bag = top_bag & row_valid_bool.unsqueeze(1)
        top_score = target.gather(1, top_idx[:, None]).squeeze(1)
        top_hit = top_bag.gather(1, top_idx[:, None]).squeeze(1).to(logits.dtype)
        regret = (best_score - top_score).clamp_min(0.0)
        denom = row_weight.sum().clamp_min(1.0)
        return {
            f"{prefix}_hit": ((top_hit * row_weight).sum() / denom).detach(),
            f"{prefix}_score": ((top_score * row_weight).sum() / denom).detach(),
            f"{prefix}_regret": ((regret * row_weight).sum() / denom).detach(),
        }

    metrics = {
        "decision_source_acc": (((pred_is_base == target_is_base).to(logits.dtype) * row_weight).sum() / row_weight.sum().clamp_min(1.0)).detach(),
        "decision_conditioned_action_source_acc": (((top_is_base == target_is_base).to(logits.dtype) * row_weight).sum() / row_weight.sum().clamp_min(1.0)).detach(),
        "decision_conditioned_baseline_preserve_top1_hit": ((top_is_base * base_rows).sum() / base_rows.sum().clamp_min(1.0)).detach(),
        "decision_conditioned_crop_action_top1_hit": (((1.0 - top_is_base) * crop_rows).sum() / crop_rows.sum().clamp_min(1.0)).detach(),
        "decision_conditioned_active_frac": row_valid_bool.to(logits.dtype).mean().detach(),
    }
    metrics.update(top_metrics(return_target, "decision_conditioned_top_return"))
    metrics.update(top_metrics(score_target, "decision_conditioned_raw_top_return"))
    if decision_source_logit is not None:
        metrics.update(
            _decision_source_threshold_sweep_metrics(
                sweep_logits if sweep_logits is not None else logits,
                decision_source_logit,
                valid,
                candidate_is_base,
                decision_target,
                return_target,
                score_target,
                score_margin=score_margin,
                prefix="decision_source_threshold_sweep",
            )
        )
    return metrics


def _decision_source_threshold_sweep_metrics(
    logits: torch.Tensor,
    decision_source_logit: torch.Tensor,
    valid: torch.Tensor,
    candidate_is_base: torch.Tensor,
    decision_target: torch.Tensor,
    return_target: torch.Tensor,
    score_target: torch.Tensor,
    *,
    score_margin: float = 0.03,
    prefix: str = "decision_source_threshold_sweep",
) -> dict[str, torch.Tensor]:
    valid_bool = valid > 0
    base_mask = (candidate_is_base > 0) & valid_bool
    crop_mask = (~(candidate_is_base > 0)) & valid_bool
    row_valid_bool = base_mask.any(dim=1) & crop_mask.any(dim=1)
    if not torch.any(row_valid_bool):
        zero = logits.sum() * 0.0
        return {
            f"{prefix}_best_balanced_acc": zero.detach(),
            f"{prefix}_best_threshold": zero.detach(),
            f"{prefix}_best_base_recall": zero.detach(),
            f"{prefix}_best_crop_recall": zero.detach(),
            f"{prefix}_best_source_acc": zero.detach(),
            f"{prefix}_best_min_recall": zero.detach(),
            f"{prefix}_best_min_recall_threshold": zero.detach(),
            f"{prefix}_best_min_recall_base_recall": zero.detach(),
            f"{prefix}_best_min_recall_crop_recall": zero.detach(),
            f"{prefix}_best_min_recall_source_acc": zero.detach(),
            f"{prefix}_best_min_recall_top_return_hit": zero.detach(),
            f"{prefix}_best_min_recall_top_return_score": zero.detach(),
            f"{prefix}_best_min_recall_top_return_regret": zero.detach(),
            f"{prefix}_best_min_recall_raw_top_return_hit": zero.detach(),
            f"{prefix}_best_min_recall_raw_top_return_score": zero.detach(),
            f"{prefix}_best_min_recall_raw_top_return_regret": zero.detach(),
            f"{prefix}_top_return_hit": zero.detach(),
            f"{prefix}_top_return_score": zero.detach(),
            f"{prefix}_top_return_regret": zero.detach(),
            f"{prefix}_raw_top_return_hit": zero.detach(),
            f"{prefix}_raw_top_return_score": zero.detach(),
            f"{prefix}_raw_top_return_regret": zero.detach(),
            f"{prefix}_active_frac": zero.detach(),
        }

    crop_id = DECISION_VOCAB.index("crop")
    target_is_crop_bool = decision_target == crop_id
    target_is_base_bool = ~target_is_crop_bool
    metric_weight = row_valid_bool.to(logits.dtype)
    base_rows = row_valid_bool & target_is_base_bool
    crop_rows = row_valid_bool & target_is_crop_bool
    base_count = base_rows.to(logits.dtype).sum().clamp_min(1.0)
    crop_count = crop_rows.to(logits.dtype).sum().clamp_min(1.0)

    source_signal = decision_source_logit.view(-1).to(device=logits.device, dtype=logits.dtype)
    valid_signal = source_signal[row_valid_bool].to(torch.float32)
    if int(valid_signal.numel()) > 1:
        quantiles = torch.linspace(0.0, 1.0, steps=101, device=logits.device)
        thresholds = torch.quantile(valid_signal, quantiles).to(logits.dtype)
        thresholds = torch.cat(
            [
                thresholds,
                valid_signal.min().to(logits.dtype).view(1) - logits.new_tensor(1e-4),
                valid_signal.max().to(logits.dtype).view(1) + logits.new_tensor(1e-4),
                logits.new_tensor([0.0]),
            ],
            dim=0,
        )
    else:
        thresholds = logits.new_tensor([0.0])

    pred_crop = source_signal.unsqueeze(0) > thresholds.view(-1, 1)
    row_valid = row_valid_bool.unsqueeze(0)
    base_ok = ((~pred_crop) & base_rows.unsqueeze(0) & row_valid).to(logits.dtype).sum(dim=1) / base_count
    crop_ok = (pred_crop & crop_rows.unsqueeze(0) & row_valid).to(logits.dtype).sum(dim=1) / crop_count
    balanced = 0.5 * (base_ok + crop_ok)
    source_acc = (
        ((pred_crop == target_is_crop_bool.unsqueeze(0)) & row_valid).to(logits.dtype).sum(dim=1)
        / metric_weight.sum().clamp_min(1.0)
    )
    best_sweep_idx = balanced.argmax()
    best_threshold = thresholds[best_sweep_idx]
    min_recall = torch.minimum(base_ok, crop_ok)
    best_min_idx = min_recall.argmax()
    best_min_threshold = thresholds[best_min_idx]

    base_logits = logits.masked_fill(~base_mask, -1e4)
    crop_logits = logits.masked_fill(~crop_mask, -1e4)
    best_base = base_logits.max(dim=1)
    best_crop = crop_logits.max(dim=1)

    def selected_top_metrics(selected_idx: torch.Tensor, target: torch.Tensor, name: str) -> dict[str, torch.Tensor]:
        masked_target = target.masked_fill(~valid_bool, -1.0)
        best_score = masked_target.max(dim=1).values
        top_bag = valid_bool & ((best_score.unsqueeze(1) - target) <= float(score_margin))
        top_bag = top_bag & row_valid_bool.unsqueeze(1)
        top_score = target.gather(1, selected_idx[:, None]).squeeze(1)
        top_hit = top_bag.gather(1, selected_idx[:, None]).squeeze(1).to(logits.dtype)
        regret = (best_score - top_score).clamp_min(0.0)
        denom = metric_weight.sum().clamp_min(1.0)
        return {
            f"{prefix}_{name}_hit": ((top_hit * metric_weight).sum() / denom).detach(),
            f"{prefix}_{name}_score": ((top_score * metric_weight).sum() / denom).detach(),
            f"{prefix}_{name}_regret": ((regret * metric_weight).sum() / denom).detach(),
        }

    best_pred_crop = source_signal > best_threshold
    selected_idx = torch.where(best_pred_crop, best_crop.indices, best_base.indices)
    best_min_pred_crop = source_signal > best_min_threshold
    best_min_selected_idx = torch.where(best_min_pred_crop, best_crop.indices, best_base.indices)
    best_min_return = selected_top_metrics(best_min_selected_idx, return_target, "best_min_recall_top_return")
    best_min_raw = selected_top_metrics(best_min_selected_idx, score_target, "best_min_recall_raw_top_return")
    metrics = {
        f"{prefix}_best_balanced_acc": balanced[best_sweep_idx].detach(),
        f"{prefix}_best_threshold": best_threshold.detach(),
        f"{prefix}_best_base_recall": base_ok[best_sweep_idx].detach(),
        f"{prefix}_best_crop_recall": crop_ok[best_sweep_idx].detach(),
        f"{prefix}_best_source_acc": source_acc[best_sweep_idx].detach(),
        f"{prefix}_best_min_recall": min_recall[best_min_idx].detach(),
        f"{prefix}_best_min_recall_threshold": best_min_threshold.detach(),
        f"{prefix}_best_min_recall_base_recall": base_ok[best_min_idx].detach(),
        f"{prefix}_best_min_recall_crop_recall": crop_ok[best_min_idx].detach(),
        f"{prefix}_best_min_recall_source_acc": source_acc[best_min_idx].detach(),
        f"{prefix}_active_frac": row_valid_bool.to(logits.dtype).mean().detach(),
    }
    metrics.update(selected_top_metrics(selected_idx, return_target, "top_return"))
    metrics.update(selected_top_metrics(selected_idx, score_target, "raw_top_return"))
    metrics.update(best_min_return)
    metrics.update(best_min_raw)
    return metrics


def _source_mixture_action_metrics(
    logits: torch.Tensor,
    decision_logits: torch.Tensor,
    decision_target: torch.Tensor,
    valid: torch.Tensor,
    candidate_is_base: torch.Tensor,
    return_target: torch.Tensor,
    score_target: torch.Tensor,
    *,
    score_margin: float = 0.03,
    decision_source_logit: torch.Tensor | None = None,
    prefix: str = "source_mixture",
) -> dict[str, torch.Tensor]:
    valid_bool = valid > 0
    base_mask = (candidate_is_base > 0) & valid_bool
    crop_mask = (~(candidate_is_base > 0)) & valid_bool
    row_valid_bool = base_mask.any(dim=1) & crop_mask.any(dim=1)
    if not torch.any(row_valid_bool):
        zero = logits.sum() * 0.0
        return {
            f"{prefix}_action_source_acc": zero.detach(),
            f"{prefix}_baseline_preserve_top1_hit": zero.detach(),
            f"{prefix}_crop_action_top1_hit": zero.detach(),
            f"{prefix}_top_return_hit": zero.detach(),
            f"{prefix}_top_return_score": zero.detach(),
            f"{prefix}_top_return_regret": zero.detach(),
            f"{prefix}_raw_top_return_hit": zero.detach(),
            f"{prefix}_raw_top_return_score": zero.detach(),
            f"{prefix}_raw_top_return_regret": zero.detach(),
            f"{prefix}_active_frac": zero.detach(),
        }

    crop_id = DECISION_VOCAB.index("crop")
    target_is_base = (decision_target != crop_id).to(logits.dtype)
    row_weight = row_valid_bool.to(logits.dtype)
    top_idx = torch.sigmoid(logits).masked_fill(~valid_bool, -1.0).argmax(dim=1)
    top_is_base = candidate_is_base.gather(1, top_idx[:, None]).squeeze(1).to(logits.dtype)
    base_rows = row_weight * target_is_base
    crop_rows = row_weight * (1.0 - target_is_base)

    def top_metrics(target: torch.Tensor, prefix: str) -> dict[str, torch.Tensor]:
        masked_target = target.masked_fill(~valid_bool, -1.0)
        best_score = masked_target.max(dim=1).values
        top_bag = valid_bool & ((best_score.unsqueeze(1) - target) <= float(score_margin))
        top_bag = top_bag & row_valid_bool.unsqueeze(1)
        top_score = target.gather(1, top_idx[:, None]).squeeze(1)
        top_hit = top_bag.gather(1, top_idx[:, None]).squeeze(1).to(logits.dtype)
        regret = (best_score - top_score).clamp_min(0.0)
        denom = row_weight.sum().clamp_min(1.0)
        return {
            f"{prefix}_hit": ((top_hit * row_weight).sum() / denom).detach(),
            f"{prefix}_score": ((top_score * row_weight).sum() / denom).detach(),
            f"{prefix}_regret": ((regret * row_weight).sum() / denom).detach(),
        }

    metrics = {
        f"{prefix}_action_source_acc": (((top_is_base == target_is_base).to(logits.dtype) * row_weight).sum() / row_weight.sum().clamp_min(1.0)).detach(),
        f"{prefix}_baseline_preserve_top1_hit": ((top_is_base * base_rows).sum() / base_rows.sum().clamp_min(1.0)).detach(),
        f"{prefix}_crop_action_top1_hit": (((1.0 - top_is_base) * crop_rows).sum() / crop_rows.sum().clamp_min(1.0)).detach(),
        f"{prefix}_active_frac": row_valid_bool.to(logits.dtype).mean().detach(),
    }
    metrics.update(top_metrics(return_target, f"{prefix}_top_return"))
    metrics.update(top_metrics(score_target, f"{prefix}_raw_top_return"))
    metrics.update(
        _source_gap_threshold_sweep_metrics(
            logits,
            valid,
            candidate_is_base,
            decision_target,
            return_target,
            score_target,
            score_margin=score_margin,
            prefix=f"{prefix}_gap_sweep",
        )
    )
    return metrics


def _source_gap_threshold_sweep_metrics(
    logits: torch.Tensor,
    valid: torch.Tensor,
    candidate_is_base: torch.Tensor,
    decision_target: torch.Tensor,
    return_target: torch.Tensor,
    score_target: torch.Tensor,
    *,
    score_margin: float = 0.03,
    prefix: str = "source_gap_sweep",
) -> dict[str, torch.Tensor]:
    valid_bool = valid > 0
    base_mask = (candidate_is_base > 0) & valid_bool
    crop_mask = (~(candidate_is_base > 0)) & valid_bool
    row_valid_bool = base_mask.any(dim=1) & crop_mask.any(dim=1)
    if not torch.any(row_valid_bool):
        zero = logits.sum() * 0.0
        return {
            f"{prefix}_best_balanced_acc": zero.detach(),
            f"{prefix}_best_threshold": zero.detach(),
            f"{prefix}_best_base_recall": zero.detach(),
            f"{prefix}_best_crop_recall": zero.detach(),
            f"{prefix}_best_source_acc": zero.detach(),
            f"{prefix}_top_return_hit": zero.detach(),
            f"{prefix}_top_return_score": zero.detach(),
            f"{prefix}_top_return_regret": zero.detach(),
            f"{prefix}_raw_top_return_hit": zero.detach(),
            f"{prefix}_raw_top_return_score": zero.detach(),
            f"{prefix}_raw_top_return_regret": zero.detach(),
            f"{prefix}_active_frac": zero.detach(),
        }

    crop_id = DECISION_VOCAB.index("crop")
    target_is_crop_bool = decision_target == crop_id
    target_is_base_bool = ~target_is_crop_bool
    metric_weight = row_valid_bool.to(logits.dtype)
    base_rows = row_valid_bool & target_is_base_bool
    crop_rows = row_valid_bool & target_is_crop_bool
    base_count = base_rows.to(logits.dtype).sum().clamp_min(1.0)
    crop_count = crop_rows.to(logits.dtype).sum().clamp_min(1.0)

    base_logits = logits.masked_fill(~base_mask, -1e4)
    crop_logits = logits.masked_fill(~crop_mask, -1e4)
    best_base = base_logits.max(dim=1)
    best_crop = crop_logits.max(dim=1)
    source_gap = best_crop.values - best_base.values
    valid_gap = source_gap[row_valid_bool].to(torch.float32)
    if int(valid_gap.numel()) > 1:
        quantiles = torch.linspace(0.0, 1.0, steps=101, device=logits.device)
        thresholds = torch.quantile(valid_gap, quantiles).to(logits.dtype)
        thresholds = torch.cat(
            [
                thresholds,
                valid_gap.min().to(logits.dtype).view(1) - logits.new_tensor(1e-4),
                valid_gap.max().to(logits.dtype).view(1) + logits.new_tensor(1e-4),
                logits.new_tensor([0.0]),
            ],
            dim=0,
        )
    else:
        thresholds = logits.new_tensor([0.0])

    pred_crop = source_gap.unsqueeze(0) > thresholds.view(-1, 1)
    row_valid = row_valid_bool.unsqueeze(0)
    base_ok = ((~pred_crop) & base_rows.unsqueeze(0) & row_valid).to(logits.dtype).sum(dim=1) / base_count
    crop_ok = (pred_crop & crop_rows.unsqueeze(0) & row_valid).to(logits.dtype).sum(dim=1) / crop_count
    balanced = 0.5 * (base_ok + crop_ok)
    source_acc = (
        ((pred_crop == target_is_crop_bool.unsqueeze(0)) & row_valid).to(logits.dtype).sum(dim=1)
        / metric_weight.sum().clamp_min(1.0)
    )
    best_sweep_idx = balanced.argmax()
    best_threshold = thresholds[best_sweep_idx]
    best_pred_crop = source_gap > best_threshold
    selected_idx = torch.where(best_pred_crop, best_crop.indices, best_base.indices)

    def selected_top_metrics(target: torch.Tensor, name: str) -> dict[str, torch.Tensor]:
        masked_target = target.masked_fill(~valid_bool, -1.0)
        best_score = masked_target.max(dim=1).values
        top_bag = valid_bool & ((best_score.unsqueeze(1) - target) <= float(score_margin))
        top_bag = top_bag & row_valid_bool.unsqueeze(1)
        top_score = target.gather(1, selected_idx[:, None]).squeeze(1)
        top_hit = top_bag.gather(1, selected_idx[:, None]).squeeze(1).to(logits.dtype)
        regret = (best_score - top_score).clamp_min(0.0)
        denom = metric_weight.sum().clamp_min(1.0)
        return {
            f"{prefix}_{name}_hit": ((top_hit * metric_weight).sum() / denom).detach(),
            f"{prefix}_{name}_score": ((top_score * metric_weight).sum() / denom).detach(),
            f"{prefix}_{name}_regret": ((regret * metric_weight).sum() / denom).detach(),
        }

    metrics = {
        f"{prefix}_best_balanced_acc": balanced[best_sweep_idx].detach(),
        f"{prefix}_best_threshold": best_threshold.detach(),
        f"{prefix}_best_base_recall": base_ok[best_sweep_idx].detach(),
        f"{prefix}_best_crop_recall": crop_ok[best_sweep_idx].detach(),
        f"{prefix}_best_source_acc": source_acc[best_sweep_idx].detach(),
        f"{prefix}_active_frac": row_valid_bool.to(logits.dtype).mean().detach(),
    }
    metrics.update(selected_top_metrics(return_target, "top_return"))
    metrics.update(selected_top_metrics(score_target, "raw_top_return"))
    return metrics


def _decision_source_return_target(
    score_target: torch.Tensor,
    valid: torch.Tensor,
    candidate_is_base: torch.Tensor,
    decision_target: torch.Tensor,
) -> torch.Tensor:
    valid_bool = valid > 0
    crop_decision_id = DECISION_VOCAB.index("crop")
    target_is_crop = decision_target == crop_decision_id
    base_mask = (candidate_is_base > 0) & valid_bool
    crop_mask = (~(candidate_is_base > 0)) & valid_bool
    source_mask = torch.where(target_is_crop.view(-1, 1), crop_mask, base_mask)
    source_mask = torch.where(source_mask.any(dim=1, keepdim=True), source_mask, valid_bool)
    return score_target.masked_fill(~source_mask, 0.0)


def _action_return_joint_loss(
    logits: torch.Tensor,
    score_target: torch.Tensor,
    valid: torch.Tensor,
    candidate_is_base: torch.Tensor,
    decision_target: torch.Tensor,
    candidate_weight: torch.Tensor,
    *,
    score_margin: float = 0.03,
    logit_margin: float = 0.20,
    temperature: float = 0.12,
    source_balanced: bool = False,
    source_balance_max_weight: float = 4.0,
    source_crop_weight_mult: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid_bool = valid > 0
    base_mask = (candidate_is_base > 0) & valid_bool
    crop_mask = (~(candidate_is_base > 0)) & valid_bool
    crop_decision_id = DECISION_VOCAB.index("crop")
    target_is_crop = decision_target == crop_decision_id
    row_valid = base_mask.any(dim=1) & crop_mask.any(dim=1)
    if not torch.any(row_valid):
        zero = logits.sum() * 0.0
        return zero, {
            "action_return_joint_loss": zero.detach(),
            "action_return_joint_bag_loss": zero.detach(),
            "action_return_joint_margin_loss": zero.detach(),
            "action_return_joint_hit": zero.detach(),
            "action_return_joint_source_acc": zero.detach(),
            "action_return_joint_crop_hit": zero.detach(),
            "action_return_joint_base_hit": zero.detach(),
            "action_return_joint_crop_target_rate": zero.detach(),
        }

    crop_score = score_target.masked_fill(~crop_mask, -1.0)
    base_score = score_target.masked_fill(~base_mask, -1.0)
    best_crop = crop_score.max(dim=1).values
    best_base = base_score.max(dim=1).values
    crop_target_bag = crop_mask & ((best_crop.unsqueeze(1) - score_target) <= float(score_margin))
    base_target_bag = base_mask & ((best_base.unsqueeze(1) - score_target) <= float(score_margin))
    target_bag = torch.where(target_is_crop.view(-1, 1), crop_target_bag, base_target_bag)
    row_valid = row_valid & target_bag.any(dim=1)
    if not torch.any(row_valid):
        zero = logits.sum() * 0.0
        return zero, {
            "action_return_joint_loss": zero.detach(),
            "action_return_joint_bag_loss": zero.detach(),
            "action_return_joint_margin_loss": zero.detach(),
            "action_return_joint_hit": zero.detach(),
            "action_return_joint_source_acc": zero.detach(),
            "action_return_joint_crop_hit": zero.detach(),
            "action_return_joint_base_hit": zero.detach(),
            "action_return_joint_crop_target_rate": zero.detach(),
        }

    scaled_logits = logits / max(1e-4, float(temperature))
    log_denom = torch.logsumexp(scaled_logits.masked_fill(~valid_bool, -1e4), dim=1)
    log_target = torch.logsumexp(scaled_logits.masked_fill(~target_bag, -1e4), dim=1)
    bag_loss = -(log_target - log_denom)
    target_logits = logits.masked_fill(~target_bag, -1e4).max(dim=1).values
    non_target_mask = valid_bool & ~target_bag
    non_target_logits = logits.masked_fill(~non_target_mask, -1e4).max(dim=1).values
    margin_loss = F.softplus(-(target_logits - non_target_logits - float(logit_margin)))
    bag_weight = (candidate_weight * target_bag.to(candidate_weight.dtype)).sum(dim=1) / target_bag.to(candidate_weight.dtype).sum(dim=1).clamp_min(1.0)
    row_weight = bag_weight.clamp(0.05, 1.0) * row_valid.to(logits.dtype)
    if bool(source_balanced):
        target_is_base_float = (~target_is_crop).to(logits.dtype)
        metric_row_weight = row_valid.to(logits.dtype)
        base_count = (metric_row_weight * target_is_base_float).sum()
        crop_count = (metric_row_weight * (1.0 - target_is_base_float)).sum()
        if bool((base_count > 0).item()) and bool((crop_count > 0).item()):
            total_count = base_count + crop_count
            max_weight = max(1.0, float(source_balance_max_weight))
            base_scale = (total_count / (2.0 * base_count)).clamp(max=max_weight)
            crop_scale = (total_count / (2.0 * crop_count)).clamp(max=max_weight)
            source_weight = torch.where(target_is_base_float > 0.5, base_scale, crop_scale)
            row_weight = row_weight * source_weight
    crop_mult = max(0.0, float(source_crop_weight_mult))
    if crop_mult != 1.0:
        row_weight = row_weight * torch.where(
            target_is_crop,
            torch.full_like(row_weight, crop_mult),
            torch.ones_like(row_weight),
        )
    bag_out = (bag_loss * row_weight).sum() / row_weight.sum().clamp_min(1.0)
    margin_out = (margin_loss * row_weight).sum() / row_weight.sum().clamp_min(1.0)
    total = bag_out + margin_out

    with torch.no_grad():
        top_idx = torch.sigmoid(logits).masked_fill(~valid_bool, -1.0).argmax(dim=1)
        top_in_target = target_bag.gather(1, top_idx[:, None]).squeeze(1).to(logits.dtype)
        top_is_base = candidate_is_base.gather(1, top_idx[:, None]).squeeze(1).to(logits.dtype)
        target_is_base = (~target_is_crop).to(logits.dtype)
        source_hit = (top_is_base == target_is_base).to(logits.dtype)
        crop_rows = row_valid & target_is_crop
        base_rows = row_valid & ~target_is_crop
        row_weight_bool = row_valid.to(logits.dtype)
        metrics = {
            "action_return_joint_loss": total.detach(),
            "action_return_joint_bag_loss": bag_out.detach(),
            "action_return_joint_margin_loss": margin_out.detach(),
            "action_return_joint_hit": ((top_in_target * row_weight_bool).sum() / row_weight_bool.sum().clamp_min(1.0)).detach(),
            "action_return_joint_source_acc": ((source_hit * row_weight_bool).sum() / row_weight_bool.sum().clamp_min(1.0)).detach(),
            "action_return_joint_crop_hit": ((top_in_target * crop_rows.to(logits.dtype)).sum() / crop_rows.to(logits.dtype).sum().clamp_min(1.0)).detach(),
            "action_return_joint_base_hit": ((top_in_target * base_rows.to(logits.dtype)).sum() / base_rows.to(logits.dtype).sum().clamp_min(1.0)).detach(),
            "action_return_joint_crop_target_rate": (crop_rows.to(logits.dtype).sum() / row_weight_bool.sum().clamp_min(1.0)).detach(),
            "action_return_joint_source_balanced": torch.as_tensor(float(bool(source_balanced)), device=logits.device, dtype=logits.dtype),
            "action_return_joint_source_crop_weight_mult": torch.as_tensor(crop_mult, device=logits.device, dtype=logits.dtype),
        }
    return total, metrics


def _return_source_margin_loss(
    logits: torch.Tensor,
    valid: torch.Tensor,
    candidate_is_base: torch.Tensor,
    decision_target: torch.Tensor,
    *,
    logit_margin: float = 0.25,
    source_balanced: bool = False,
    source_balance_max_weight: float = 4.0,
    source_crop_weight_mult: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid_bool = valid > 0
    base_mask = (candidate_is_base > 0) & valid_bool
    crop_mask = (~(candidate_is_base > 0)) & valid_bool
    row_valid = base_mask.any(dim=1) & crop_mask.any(dim=1)
    if not torch.any(row_valid):
        zero = logits.sum() * 0.0
        return zero, {
            "return_source_margin_loss": zero.detach(),
            "return_source_margin_acc": zero.detach(),
            "return_source_margin_crop_hit": zero.detach(),
            "return_source_margin_base_hit": zero.detach(),
            "return_source_margin_target_gap": zero.detach(),
            "return_source_margin_crop_target_rate": zero.detach(),
        }

    crop_decision_id = DECISION_VOCAB.index("crop")
    target_is_crop = decision_target == crop_decision_id
    best_base = logits.masked_fill(~base_mask, -1e4).max(dim=1).values
    best_crop = logits.masked_fill(~crop_mask, -1e4).max(dim=1).values
    target_gap = torch.where(target_is_crop, best_crop - best_base, best_base - best_crop)
    margin_loss = F.softplus(-(target_gap - float(logit_margin)))

    row_weight = row_valid.to(logits.dtype)
    if bool(source_balanced):
        target_is_base_float = (~target_is_crop).to(logits.dtype)
        base_count = (row_weight * target_is_base_float).sum()
        crop_count = (row_weight * (1.0 - target_is_base_float)).sum()
        if bool((base_count > 0).item()) and bool((crop_count > 0).item()):
            total_count = base_count + crop_count
            max_weight = max(1.0, float(source_balance_max_weight))
            base_scale = (total_count / (2.0 * base_count)).clamp(max=max_weight)
            crop_scale = (total_count / (2.0 * crop_count)).clamp(max=max_weight)
            row_weight = row_weight * torch.where(target_is_base_float > 0.5, base_scale, crop_scale)
    crop_mult = max(0.0, float(source_crop_weight_mult))
    if crop_mult != 1.0:
        row_weight = row_weight * torch.where(
            target_is_crop,
            torch.full_like(row_weight, crop_mult),
            torch.ones_like(row_weight),
        )

    total = (margin_loss * row_weight).sum() / row_weight.sum().clamp_min(1.0)
    with torch.no_grad():
        pred_is_crop = best_crop > best_base
        source_hit = (pred_is_crop == target_is_crop).to(logits.dtype)
        crop_rows = row_valid & target_is_crop
        base_rows = row_valid & ~target_is_crop
        metric_weight = row_valid.to(logits.dtype)
        metrics = {
            "return_source_margin_loss": total.detach(),
            "return_source_margin_acc": ((source_hit * metric_weight).sum() / metric_weight.sum().clamp_min(1.0)).detach(),
            "return_source_margin_crop_hit": ((source_hit * crop_rows.to(logits.dtype)).sum() / crop_rows.to(logits.dtype).sum().clamp_min(1.0)).detach(),
            "return_source_margin_base_hit": ((source_hit * base_rows.to(logits.dtype)).sum() / base_rows.to(logits.dtype).sum().clamp_min(1.0)).detach(),
            "return_source_margin_target_gap": ((target_gap * metric_weight).sum() / metric_weight.sum().clamp_min(1.0)).detach(),
            "return_source_margin_crop_target_rate": (crop_rows.to(logits.dtype).sum() / metric_weight.sum().clamp_min(1.0)).detach(),
            "return_source_margin_source_balanced": torch.as_tensor(float(bool(source_balanced)), device=logits.device, dtype=logits.dtype),
            "return_source_margin_crop_weight_mult": torch.as_tensor(crop_mult, device=logits.device, dtype=logits.dtype),
        }
    return total, metrics


def _decision_utility_alignment_loss(
    decision_logits: torch.Tensor,
    utility_logits: torch.Tensor,
    valid: torch.Tensor,
    candidate_is_base: torch.Tensor,
    *,
    temperature: float = 0.35,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid_bool = valid > 0
    base_mask = (candidate_is_base > 0) & valid_bool
    crop_mask = (~(candidate_is_base > 0)) & valid_bool
    row_valid = base_mask.any(dim=1) & crop_mask.any(dim=1)
    if not torch.any(row_valid):
        zero = utility_logits.sum() * 0.0
        return zero, {
            "decision_utility_align_loss": zero.detach(),
            "decision_utility_align_gap": zero.detach(),
            "decision_utility_align_acc": zero.detach(),
            "decision_crop_prob_mean": zero.detach(),
            "utility_crop_prob_mean": zero.detach(),
        }

    best_base = utility_logits.masked_fill(~base_mask, -1e4).max(dim=1).values
    best_crop = utility_logits.masked_fill(~crop_mask, -1e4).max(dim=1).values
    utility_crop_prob = torch.sigmoid((best_crop - best_base) / max(1e-4, float(temperature)))
    crop_decision_id = DECISION_VOCAB.index("crop")
    base_decision_ids = [idx for idx in range(decision_logits.shape[1]) if idx != crop_decision_id]
    decision_crop_logit = decision_logits[:, crop_decision_id] - torch.logsumexp(decision_logits[:, base_decision_ids], dim=1)
    decision_crop_prob = torch.sigmoid(decision_crop_logit)
    row_weight = row_valid.to(decision_logits.dtype)
    loss = F.binary_cross_entropy_with_logits(decision_crop_logit, utility_crop_prob.detach(), reduction="none")
    out = (loss * row_weight).sum() / row_weight.sum().clamp_min(1.0)

    with torch.no_grad():
        decision_is_crop = decision_crop_prob >= 0.5
        utility_is_crop = utility_crop_prob >= 0.5
        metrics = {
            "decision_utility_align_loss": out.detach(),
            "decision_utility_align_gap": ((decision_crop_prob - utility_crop_prob).abs() * row_weight).sum() / row_weight.sum().clamp_min(1.0),
            "decision_utility_align_acc": (((decision_is_crop == utility_is_crop).to(decision_logits.dtype) * row_weight).sum() / row_weight.sum().clamp_min(1.0)),
            "decision_crop_prob_mean": (decision_crop_prob * row_weight).sum() / row_weight.sum().clamp_min(1.0),
            "utility_crop_prob_mean": (utility_crop_prob * row_weight).sum() / row_weight.sum().clamp_min(1.0),
        }
    return out, metrics


def _topk_coverage_loss(
    logits: torch.Tensor,
    score_target: torch.Tensor,
    valid: torch.Tensor,
    *,
    k: int = 4,
    temperature: float = 0.12,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid_bool = valid > 0
    row_valid = valid_bool.sum(dim=1) > 1
    if not torch.any(row_valid):
        zero = logits.sum() * 0.0
        return zero, {"topk_coverage_loss": zero.detach(), "topk_teacher_hit": zero.detach(), "topk_teacher_score": zero.detach()}
    masked_target = score_target.masked_fill(~valid_bool, -1.0)
    topk = max(1, min(int(k), score_target.shape[1]))
    top_idx = masked_target.topk(topk, dim=1).indices
    top_mask = torch.zeros_like(valid_bool)
    top_mask.scatter_(1, top_idx, True)
    top_mask = top_mask & valid_bool & row_valid.unsqueeze(1)
    scaled_logits = logits / max(1e-4, float(temperature))
    log_denom = torch.logsumexp(scaled_logits.masked_fill(~valid_bool, -1e4), dim=1)
    log_top = torch.logsumexp(scaled_logits.masked_fill(~top_mask, -1e4), dim=1)
    loss = -(log_top - log_denom)
    out = (loss * row_valid.to(loss.dtype)).sum() / row_valid.to(loss.dtype).sum().clamp_min(1.0)
    with torch.no_grad():
        prob = torch.sigmoid(logits).masked_fill(~valid_bool, -1.0)
        pred_top = prob.argmax(dim=1)
        hit = top_mask.gather(1, pred_top[:, None]).squeeze(1).to(logits.dtype)
        score = score_target.gather(1, pred_top[:, None]).squeeze(1)
        denom = row_valid.to(logits.dtype).sum().clamp_min(1.0)
    return out, {
        "topk_coverage_loss": out.detach(),
        "topk_teacher_hit": (hit * row_valid.to(logits.dtype)).sum() / denom,
        "topk_teacher_score": (score * row_valid.to(logits.dtype)).sum() / denom,
    }


def _proposal_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], *, use_positive_scores: bool = True) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
    positive_weight = positive_valid
    if use_positive_scores and "positive_scores" in batch:
        positive_scores = batch["positive_scores"].to(proposal_boxes.device).to(proposal_boxes.dtype).clamp(0.05, 1.0)
        positive_weight = positive_valid * positive_scores
    box_loss = (min_l1 * positive_weight).sum() / positive_weight.sum().clamp_min(1.0)

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


def _proposal_subject_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if "subject_prior_box" not in batch or "subject_prior_valid" not in batch:
        zero = outputs["proposal_logits"].sum() * 0.0
        return zero, {"proposal_subject_loss": zero.detach(), "proposal_subject_max_iou": zero.detach(), "proposal_top1_subject_iou": zero.detach()}
    proposal_boxes = outputs["proposal_boxes"]
    proposal_logits = outputs["proposal_logits"]
    subject_box = batch.get("subject_box_target", batch["subject_prior_box"]).to(proposal_boxes.device).to(proposal_boxes.dtype)
    subject_valid = batch.get("subject_box_valid", batch["subject_prior_valid"]).to(proposal_boxes.device).to(proposal_boxes.dtype).view(-1)
    subject_reliability = batch.get("subject_box_weight")
    if subject_reliability is None:
        subject_reliability = batch.get("subject_prior_reliability")
    if subject_reliability is None:
        subject_reliability = torch.ones_like(subject_valid)
    else:
        subject_reliability = subject_reliability.to(proposal_boxes.device).to(proposal_boxes.dtype).view(-1).clamp(0.0, 1.0)
    if subject_valid.sum() <= 0:
        zero = proposal_logits.sum() * 0.0
        return zero, {"proposal_subject_loss": zero.detach(), "proposal_subject_max_iou": zero.detach(), "proposal_top1_subject_iou": zero.detach()}
    iou = _box_iou_torch(proposal_boxes, subject_box.unsqueeze(1)).squeeze(-1)
    max_iou = iou.max(dim=1).values
    top_idx = proposal_logits.argmax(dim=1)
    top_iou = iou.gather(1, top_idx[:, None]).squeeze(1)
    weight = subject_valid * (0.25 + 0.75 * subject_reliability)
    loss = ((1.0 - max_iou.clamp(0.0, 1.0)) * weight).sum() / weight.sum().clamp_min(1.0)
    return loss, {
        "proposal_subject_loss": loss.detach(),
        "proposal_subject_max_iou": ((max_iou * subject_valid).sum() / subject_valid.sum().clamp_min(1.0)).detach(),
        "proposal_top1_subject_iou": ((top_iou * subject_valid).sum() / subject_valid.sum().clamp_min(1.0)).detach(),
    }


def _subject_proposal_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits = outputs.get("subject_proposal_logits")
    if logits is None:
        zero = outputs["proposal_logits"].sum() * 0.0
        return zero, {
            "subject_proposal_align_loss": zero.detach(),
            "subject_proposal_top1_iou": zero.detach(),
            "subject_proposal_best_iou": zero.detach(),
        }
    target_box = batch.get("subject_box_target", batch.get("subject_prior_box"))
    target_valid = batch.get("subject_box_valid", batch.get("subject_prior_valid"))
    if target_box is None or target_valid is None:
        zero = logits.sum() * 0.0
        return zero, {
            "subject_proposal_align_loss": zero.detach(),
            "subject_proposal_top1_iou": zero.detach(),
            "subject_proposal_best_iou": zero.detach(),
        }
    proposal_boxes = outputs["proposal_boxes"]
    target_box = target_box.to(proposal_boxes.device).to(proposal_boxes.dtype)
    target_valid = target_valid.to(proposal_boxes.device).to(proposal_boxes.dtype).view(-1)
    sample_weight = batch.get("subject_box_weight")
    if sample_weight is None:
        sample_weight = torch.ones_like(target_valid)
    else:
        sample_weight = sample_weight.to(proposal_boxes.device).to(proposal_boxes.dtype).view(-1).clamp_min(0.05)
    iou = _box_iou_torch(proposal_boxes, target_box.unsqueeze(1)).squeeze(-1).clamp(0.0, 1.0)
    target = iou * target_valid.view(-1, 1)
    row_weight = torch.where(target_valid.view(-1, 1) > 0.5, sample_weight.view(-1, 1), 0.25 * sample_weight.view(-1, 1))
    loss_raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    loss = (loss_raw * row_weight).sum() / row_weight.sum().clamp_min(1.0)
    top_idx = logits.argmax(dim=1)
    top_iou = iou.gather(1, top_idx[:, None]).squeeze(1)
    best_iou = iou.max(dim=1).values
    positive_weight = target_valid * sample_weight
    denom = positive_weight.sum().clamp_min(1.0)
    return loss, {
        "subject_proposal_align_loss": loss.detach(),
        "subject_proposal_top1_iou": ((top_iou * positive_weight).sum() / denom).detach(),
        "subject_proposal_best_iou": ((best_iou * positive_weight).sum() / denom).detach(),
    }


def _subject_box_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    valid_weight: float = 1.0,
    l1_weight: float = 2.0,
    iou_weight: float = 1.0,
    center_weight: float = 0.0,
    size_weight: float = 0.0,
    aspect_weight: float = 0.0,
    ciou_weight: float = 0.0,
    spatial_aux_weight: float = 0.0,
    spatial_heatmap_weight: float = 0.0,
    spatial_mask_weight: float = 0.0,
    valid_balanced_bce: bool = False,
    valid_negative_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if "pred_subject_box" not in outputs or "pred_subject_valid_logit" not in outputs:
        zero = outputs["proposal_logits"].sum() * 0.0
        return zero, {
            "subject_box_loss": zero.detach(),
            "subject_box_valid_loss": zero.detach(),
            "subject_box_l1_loss": zero.detach(),
            "subject_box_iou_loss": zero.detach(),
            "subject_box_center_loss": zero.detach(),
            "subject_box_size_loss": zero.detach(),
            "subject_box_aspect_loss": zero.detach(),
            "subject_box_ciou_loss": zero.detach(),
            "subject_box_spatial_aux_loss": zero.detach(),
            "subject_box_spatial_heatmap_loss": zero.detach(),
            "subject_box_spatial_heatmap_acc": zero.detach(),
            "subject_box_spatial_mask_loss": zero.detach(),
            "subject_box_spatial_mask_iou": zero.detach(),
            "subject_box_spatial_iou": zero.detach(),
            "subject_box_spatial_mix": zero.detach(),
            "subject_box_iou": zero.detach(),
            "subject_box_valid_acc": zero.detach(),
            "subject_box_valid_recall": zero.detach(),
            "subject_box_valid_precision": zero.detach(),
            "subject_box_valid_specificity": zero.detach(),
            "subject_box_valid_balanced_acc": zero.detach(),
            "subject_box_valid_best_balanced_acc": zero.detach(),
            "subject_box_valid_best_threshold": zero.detach(),
            "subject_box_valid_f1": zero.detach(),
        }
    pred_box = outputs["pred_subject_box"]
    valid_logit = outputs["pred_subject_valid_logit"]
    target_box = batch.get("subject_box_target", batch.get("subject_prior_box"))
    target_valid = batch.get("subject_box_valid", batch.get("subject_prior_valid"))
    if target_box is None or target_valid is None:
        zero = pred_box.sum() * 0.0
        return zero, {
            "subject_box_loss": zero.detach(),
            "subject_box_valid_loss": zero.detach(),
            "subject_box_l1_loss": zero.detach(),
            "subject_box_iou_loss": zero.detach(),
            "subject_box_center_loss": zero.detach(),
            "subject_box_size_loss": zero.detach(),
            "subject_box_aspect_loss": zero.detach(),
            "subject_box_ciou_loss": zero.detach(),
            "subject_box_spatial_aux_loss": zero.detach(),
            "subject_box_spatial_heatmap_loss": zero.detach(),
            "subject_box_spatial_heatmap_acc": zero.detach(),
            "subject_box_spatial_mask_loss": zero.detach(),
            "subject_box_spatial_mask_iou": zero.detach(),
            "subject_box_spatial_iou": zero.detach(),
            "subject_box_spatial_mix": zero.detach(),
            "subject_box_iou": zero.detach(),
            "subject_box_valid_acc": zero.detach(),
            "subject_box_valid_recall": zero.detach(),
            "subject_box_valid_precision": zero.detach(),
            "subject_box_valid_specificity": zero.detach(),
            "subject_box_valid_balanced_acc": zero.detach(),
            "subject_box_valid_best_balanced_acc": zero.detach(),
            "subject_box_valid_best_threshold": zero.detach(),
            "subject_box_valid_f1": zero.detach(),
        }
    target_box = target_box.to(pred_box.device).to(pred_box.dtype)
    target_valid = target_valid.to(pred_box.device).to(pred_box.dtype).view(-1).clamp(0.0, 1.0)
    sample_weight = batch.get("subject_box_weight")
    if sample_weight is None:
        sample_weight = torch.ones_like(target_valid)
    else:
        sample_weight = sample_weight.to(pred_box.device).to(pred_box.dtype).view(-1).clamp_min(0.05)
    valid_loss_raw = F.binary_cross_entropy_with_logits(valid_logit, target_valid, reduction="none")
    valid_loss_weight = sample_weight
    if valid_balanced_bce:
        pos_mask = (target_valid >= 0.5).to(pred_box.dtype)
        neg_mask = 1.0 - pos_mask
        pos_count = (pos_mask * sample_weight).sum()
        neg_count = (neg_mask * sample_weight).sum()
        total_count = (pos_count + neg_count).clamp_min(1.0)
        pos_scale = total_count / (2.0 * pos_count.clamp_min(1.0))
        neg_scale = total_count / (2.0 * neg_count.clamp_min(1.0))
        valid_loss_weight = sample_weight * torch.where(pos_mask > 0.0, pos_scale, neg_scale)
    if float(valid_negative_scale) != 1.0:
        neg_scale = torch.ones_like(valid_loss_weight)
        neg_scale = torch.where(target_valid < 0.5, neg_scale * float(valid_negative_scale), neg_scale)
        valid_loss_weight = valid_loss_weight * neg_scale
    valid_loss = (valid_loss_raw * valid_loss_weight).sum() / valid_loss_weight.sum().clamp_min(1.0)
    positive_weight = target_valid * sample_weight
    if positive_weight.sum() > 0:
        pred_cxcywh = _xyxy_to_cxcywh(pred_box)
        target_cxcywh = _xyxy_to_cxcywh(target_box)
        l1 = F.smooth_l1_loss(pred_box, target_box, reduction="none").mean(dim=-1)
        iou = _box_iou_torch(pred_box.unsqueeze(1), target_box.unsqueeze(1)).view(-1).clamp(0.0, 1.0)
        center = F.smooth_l1_loss(pred_cxcywh[:, :2], target_cxcywh[:, :2], reduction="none").mean(dim=-1)
        size = F.smooth_l1_loss(
            torch.log(pred_cxcywh[:, 2:].clamp_min(1e-6)),
            torch.log(target_cxcywh[:, 2:].clamp_min(1e-6)),
            reduction="none",
        ).mean(dim=-1)
        pred_aspect = torch.log((pred_cxcywh[:, 2] / pred_cxcywh[:, 3].clamp_min(1e-6)).clamp_min(1e-6))
        target_aspect = torch.log((target_cxcywh[:, 2] / target_cxcywh[:, 3].clamp_min(1e-6)).clamp_min(1e-6))
        aspect = F.smooth_l1_loss(pred_aspect, target_aspect, reduction="none")
        ciou_raw, iou = _box_ciou_loss(pred_box, target_box)
        l1_loss = (l1 * positive_weight).sum() / positive_weight.sum().clamp_min(1.0)
        iou_loss = ((1.0 - iou) * positive_weight).sum() / positive_weight.sum().clamp_min(1.0)
        center_loss = (center * positive_weight).sum() / positive_weight.sum().clamp_min(1.0)
        size_loss = (size * positive_weight).sum() / positive_weight.sum().clamp_min(1.0)
        aspect_loss = (aspect * positive_weight).sum() / positive_weight.sum().clamp_min(1.0)
        ciou_loss = (ciou_raw * positive_weight).sum() / positive_weight.sum().clamp_min(1.0)
        mean_iou = (iou * target_valid).sum() / target_valid.sum().clamp_min(1.0)
    else:
        l1_loss = pred_box.sum() * 0.0
        iou_loss = pred_box.sum() * 0.0
        center_loss = pred_box.sum() * 0.0
        size_loss = pred_box.sum() * 0.0
        aspect_loss = pred_box.sum() * 0.0
        ciou_loss = pred_box.sum() * 0.0
        mean_iou = pred_box.sum() * 0.0
    total = (
        float(valid_weight) * valid_loss
        + float(l1_weight) * l1_loss
        + float(iou_weight) * iou_loss
        + float(center_weight) * center_loss
        + float(size_weight) * size_loss
        + float(aspect_weight) * aspect_loss
        + float(ciou_weight) * ciou_loss
    )
    spatial_aux_loss = pred_box.sum() * 0.0
    spatial_heatmap_loss = pred_box.sum() * 0.0
    spatial_heatmap_acc = pred_box.sum() * 0.0
    spatial_mask_loss = pred_box.sum() * 0.0
    spatial_mask_iou = pred_box.sum() * 0.0
    spatial_mean_iou = pred_box.sum() * 0.0
    if (
        float(spatial_aux_weight) > 0.0
        and "pred_subject_box_spatial" in outputs
        and "pred_subject_valid_logit_spatial" in outputs
    ):
        spatial_box = outputs["pred_subject_box_spatial"].to(pred_box.device).to(pred_box.dtype)
        spatial_valid_logit = outputs["pred_subject_valid_logit_spatial"].to(pred_box.device).to(pred_box.dtype)
        spatial_valid_raw = F.binary_cross_entropy_with_logits(spatial_valid_logit, target_valid, reduction="none")
        spatial_valid_loss = (spatial_valid_raw * valid_loss_weight).sum() / valid_loss_weight.sum().clamp_min(1.0)
        if positive_weight.sum() > 0:
            spatial_cxcywh = _xyxy_to_cxcywh(spatial_box)
            target_cxcywh = _xyxy_to_cxcywh(target_box)
            spatial_l1 = F.smooth_l1_loss(spatial_box, target_box, reduction="none").mean(dim=-1)
            spatial_iou = _box_iou_torch(spatial_box.unsqueeze(1), target_box.unsqueeze(1)).view(-1).clamp(0.0, 1.0)
            spatial_center = F.smooth_l1_loss(spatial_cxcywh[:, :2], target_cxcywh[:, :2], reduction="none").mean(dim=-1)
            spatial_size = F.smooth_l1_loss(
                torch.log(spatial_cxcywh[:, 2:].clamp_min(1e-6)),
                torch.log(target_cxcywh[:, 2:].clamp_min(1e-6)),
                reduction="none",
            ).mean(dim=-1)
            spatial_l1_loss = (spatial_l1 * positive_weight).sum() / positive_weight.sum().clamp_min(1.0)
            spatial_iou_loss = ((1.0 - spatial_iou) * positive_weight).sum() / positive_weight.sum().clamp_min(1.0)
            spatial_center_loss = (spatial_center * positive_weight).sum() / positive_weight.sum().clamp_min(1.0)
            spatial_size_loss = (spatial_size * positive_weight).sum() / positive_weight.sum().clamp_min(1.0)
            spatial_mean_iou = (spatial_iou * target_valid).sum() / target_valid.sum().clamp_min(1.0)
        else:
            spatial_l1_loss = pred_box.sum() * 0.0
            spatial_iou_loss = pred_box.sum() * 0.0
            spatial_center_loss = pred_box.sum() * 0.0
            spatial_size_loss = pred_box.sum() * 0.0
            spatial_mean_iou = pred_box.sum() * 0.0
        spatial_aux_loss = (
            0.5 * float(valid_weight) * spatial_valid_loss
            + float(l1_weight) * spatial_l1_loss
            + float(iou_weight) * spatial_iou_loss
            + float(center_weight) * spatial_center_loss
            + float(size_weight) * spatial_size_loss
        )
        total = total + float(spatial_aux_weight) * spatial_aux_loss
    if (float(spatial_heatmap_weight) > 0.0 or float(spatial_mask_weight) > 0.0) and "pred_subject_spatial_heatmap_logits" in outputs:
        heatmap_logits = outputs["pred_subject_spatial_heatmap_logits"].to(pred_box.device).to(pred_box.dtype)
        if positive_weight.sum() > 0 and heatmap_logits.ndim == 2 and heatmap_logits.shape[1] > 0:
            grid_count = int(heatmap_logits.shape[1])
            grid_h = int(round(grid_count**0.5))
            grid_w = grid_count // max(1, grid_h)
            if grid_h * grid_w == grid_count:
                target_cxcywh = _xyxy_to_cxcywh(target_box)
                active = target_valid >= 0.5
                support_target = None
                support_active = None
                support_mask = batch.get("subject_support_mask")
                support_valid = batch.get("subject_support_mask_valid")
                if torch.is_tensor(support_mask) and torch.is_tensor(support_valid) and support_mask.ndim == 3:
                    support_target = F.interpolate(
                        support_mask.to(pred_box.device).to(pred_box.dtype).unsqueeze(1),
                        size=(grid_h, grid_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1).clamp(0.0, 1.0)
                    support_row_max = support_target.flatten(1).amax(dim=1).view(-1, 1, 1)
                    support_target = support_target / support_row_max.clamp_min(1e-6)
                    support_active = (
                        active
                        & (support_valid.to(pred_box.device).to(pred_box.dtype).view(-1) >= 0.5)
                        & (support_row_max.view(-1) > 0.0)
                    )
                if bool(active.any().item()) and float(spatial_heatmap_weight) > 0.0:
                    tx = (target_cxcywh[:, 0] * float(grid_w)).floor().long().clamp(0, grid_w - 1)
                    ty = (target_cxcywh[:, 1] * float(grid_h)).floor().long().clamp(0, grid_h - 1)
                    target_index = ty * grid_w + tx
                    if support_target is not None and support_active is not None and bool(support_active.any().item()):
                        support_index = support_target.flatten(1).argmax(dim=1)
                        target_index = torch.where(support_active, support_index, target_index)
                    ce = F.cross_entropy(heatmap_logits[active], target_index[active], reduction="none")
                    heat_weight = sample_weight[active].clamp_min(0.05)
                    spatial_heatmap_loss = (ce * heat_weight).sum() / heat_weight.sum().clamp_min(1.0)
                    pred_index = heatmap_logits[active].argmax(dim=1)
                    spatial_heatmap_acc = (pred_index == target_index[active]).to(pred_box.dtype).mean()
                    total = total + float(spatial_heatmap_weight) * spatial_heatmap_loss
                if bool(active.any().item()) and float(spatial_mask_weight) > 0.0:
                    xs = (torch.arange(grid_w, device=pred_box.device, dtype=pred_box.dtype) + 0.5) / float(grid_w)
                    ys = (torch.arange(grid_h, device=pred_box.device, dtype=pred_box.dtype) + 0.5) / float(grid_h)
                    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
                    xx = xx.view(1, grid_h, grid_w)
                    yy = yy.view(1, grid_h, grid_w)
                    box = target_box.clamp(0.0, 1.0)
                    x1 = box[:, 0].view(-1, 1, 1)
                    y1 = box[:, 1].view(-1, 1, 1)
                    x2 = box[:, 2].view(-1, 1, 1)
                    y2 = box[:, 3].view(-1, 1, 1)
                    hard_mask = ((xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)).to(pred_box.dtype)
                    cx = target_cxcywh[:, 0].view(-1, 1, 1)
                    cy = target_cxcywh[:, 1].view(-1, 1, 1)
                    sx = (0.5 * target_cxcywh[:, 2]).clamp_min(1.0 / float(grid_w)).view(-1, 1, 1)
                    sy = (0.5 * target_cxcywh[:, 3]).clamp_min(1.0 / float(grid_h)).view(-1, 1, 1)
                    gaussian = torch.exp(-0.5 * (((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2))
                    mask_target = torch.maximum(hard_mask, 0.35 * gaussian).clamp(0.0, 1.0)
                    if support_target is not None and support_active is not None and bool(support_active.any().item()):
                        support_soft = torch.maximum(support_target, 0.20 * gaussian).clamp(0.0, 1.0)
                        mask_target = torch.where(support_active.view(-1, 1, 1), support_soft, mask_target)
                    mask_logits = heatmap_logits.view(-1, grid_h, grid_w)
                    mask_raw = F.binary_cross_entropy_with_logits(mask_logits, mask_target, reduction="none")
                    mask_pixel_weight = (0.25 + 0.75 * mask_target) * sample_weight.view(-1, 1, 1).clamp_min(0.05)
                    mask_row_weight = target_valid.view(-1, 1, 1)
                    mask_weight = mask_pixel_weight * mask_row_weight
                    spatial_mask_loss = (mask_raw * mask_weight).sum() / mask_weight.sum().clamp_min(1.0)
                    pred_mask = (torch.sigmoid(mask_logits) >= 0.5) & active.view(-1, 1, 1)
                    tgt_mask = (mask_target >= 0.35) & active.view(-1, 1, 1)
                    inter = (pred_mask & tgt_mask).to(pred_box.dtype).sum(dim=(1, 2))
                    union = (pred_mask | tgt_mask).to(pred_box.dtype).sum(dim=(1, 2)).clamp_min(1.0)
                    spatial_mask_iou = (inter[active] / union[active]).mean()
                    total = total + float(spatial_mask_weight) * spatial_mask_loss
    with torch.no_grad():
        pred_valid = (torch.sigmoid(valid_logit) >= 0.5).to(target_valid.dtype)
        tp = (pred_valid * target_valid).sum()
        fp = (pred_valid * (1.0 - target_valid)).sum()
        fn = ((1.0 - pred_valid) * target_valid).sum()
        tn = ((1.0 - pred_valid) * (1.0 - target_valid)).sum()
        acc = (pred_valid == target_valid).to(target_valid.dtype).mean()
        recall = tp / (tp + fn).clamp_min(1.0)
        precision = tp / (tp + fp).clamp_min(1.0)
        specificity = tn / (tn + fp).clamp_min(1.0)
        balanced_acc = 0.5 * (recall + specificity)
        f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-6)
        prob = torch.sigmoid(valid_logit)
        best_balanced_acc = balanced_acc
        best_threshold = torch.full((), 0.5, dtype=prob.dtype, device=prob.device)
        for threshold_value in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            pred_at = (prob >= float(threshold_value)).to(target_valid.dtype)
            tp_at = (pred_at * target_valid).sum()
            fp_at = (pred_at * (1.0 - target_valid)).sum()
            fn_at = ((1.0 - pred_at) * target_valid).sum()
            tn_at = ((1.0 - pred_at) * (1.0 - target_valid)).sum()
            recall_at = tp_at / (tp_at + fn_at).clamp_min(1.0)
            specificity_at = tn_at / (tn_at + fp_at).clamp_min(1.0)
            balanced_at = 0.5 * (recall_at + specificity_at)
            if bool((balanced_at > best_balanced_acc).item()):
                best_balanced_acc = balanced_at
                best_threshold = torch.full((), float(threshold_value), dtype=prob.dtype, device=prob.device)
        spatial_mix = outputs.get("pred_subject_spatial_mix")
        if spatial_mix is None:
            spatial_mix_mean = pred_box.sum() * 0.0
        else:
            spatial_mix_mean = spatial_mix.to(pred_box.device).to(pred_box.dtype).mean()
    return total, {
        "subject_box_loss": total.detach(),
        "subject_box_valid_loss": valid_loss.detach(),
        "subject_box_l1_loss": l1_loss.detach(),
        "subject_box_iou_loss": iou_loss.detach(),
        "subject_box_center_loss": center_loss.detach(),
        "subject_box_size_loss": size_loss.detach(),
        "subject_box_aspect_loss": aspect_loss.detach(),
        "subject_box_ciou_loss": ciou_loss.detach(),
        "subject_box_spatial_aux_loss": spatial_aux_loss.detach(),
        "subject_box_spatial_heatmap_loss": spatial_heatmap_loss.detach(),
        "subject_box_spatial_heatmap_acc": spatial_heatmap_acc.detach(),
        "subject_box_spatial_mask_loss": spatial_mask_loss.detach(),
        "subject_box_spatial_mask_iou": spatial_mask_iou.detach(),
        "subject_box_spatial_iou": spatial_mean_iou.detach(),
        "subject_box_spatial_mix": spatial_mix_mean.detach(),
        "subject_box_iou": mean_iou.detach(),
        "subject_box_valid_acc": acc.detach(),
        "subject_box_valid_recall": recall.detach(),
        "subject_box_valid_precision": precision.detach(),
        "subject_box_valid_specificity": specificity.detach(),
        "subject_box_valid_balanced_acc": balanced_acc.detach(),
        "subject_box_valid_best_balanced_acc": best_balanced_acc.detach(),
        "subject_box_valid_best_threshold": best_threshold.detach(),
        "subject_box_valid_f1": f1.detach(),
    }


def _checklist_applicability_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], valid: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if "checklist_applicability_logits" not in outputs or "checklist_applicability_target" not in batch:
        zero = valid.sum() * 0.0
        return zero, {
            "checklist_applicability_precision": zero.detach(),
            "checklist_applicability_recall": zero.detach(),
            "checklist_applicability_count": zero.detach(),
        }
    logits = outputs["checklist_applicability_logits"]
    target = batch["checklist_applicability_target"].to(logits.device).to(logits.dtype)
    target_valid = batch["checklist_applicability_valid"].to(logits.device).to(logits.dtype) * valid.unsqueeze(-1)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    loss = (bce * target_valid).sum() / target_valid.sum().clamp_min(1.0)
    with torch.no_grad():
        pred = (torch.sigmoid(logits) >= 0.5).to(logits.dtype)
        tp = (pred * target * target_valid).sum()
        fp = (pred * (1.0 - target) * target_valid).sum()
        fn = ((1.0 - pred) * target * target_valid).sum()
        precision = tp / (tp + fp).clamp_min(1.0)
        recall = tp / (tp + fn).clamp_min(1.0)
    return loss, {
        "checklist_applicability_precision": precision.detach(),
        "checklist_applicability_recall": recall.detach(),
        "checklist_applicability_count": target_valid.sum().detach(),
    }


def _checklist_class_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], valid: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if "checklist_class_logits" not in outputs or "checklist_class_target" not in batch:
        zero = valid.sum() * 0.0
        return zero, {"checklist_class_acc": zero.detach(), "checklist_class_count": zero.detach()}
    logits_all = outputs["checklist_class_logits"]
    targets = batch["checklist_class_target"].to(logits_all.device).long()
    target_valid = batch["checklist_class_valid"].to(logits_all.device).to(logits_all.dtype) * valid.unsqueeze(-1)
    if "checklist_applicability_target" in batch:
        target_valid = target_valid * batch["checklist_applicability_target"].to(logits_all.device).to(logits_all.dtype)
    offset = 0
    losses: list[torch.Tensor] = []
    correct_sum = logits_all.sum() * 0.0
    count_sum = logits_all.sum() * 0.0
    for group_idx, size in enumerate(CHECKLIST_CLASS_SIZES):
        logits = logits_all[..., offset : offset + int(size)]
        offset += int(size)
        group_target = targets[..., group_idx].clamp(0, int(size) - 1)
        group_valid = target_valid[..., group_idx]
        ce = F.cross_entropy(logits.reshape(-1, int(size)), group_target.reshape(-1), reduction="none").view_as(group_valid)
        losses.append((ce * group_valid).sum() / group_valid.sum().clamp_min(1.0))
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            correct_sum = correct_sum + ((pred == group_target).to(logits_all.dtype) * group_valid).sum()
            count_sum = count_sum + group_valid.sum()
    total = torch.stack(losses).mean() if losses else logits_all.sum() * 0.0
    return total, {
        "checklist_class_acc": (correct_sum / count_sum.clamp_min(1.0)).detach(),
        "checklist_class_count": count_sum.detach(),
    }


def _detail_score_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], valid: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if "detail_score_logits" not in outputs or "detail_score_target" not in batch:
        zero = valid.sum() * 0.0
        return zero, {"detail_score_mae": zero.detach(), "detail_score_count": zero.detach()}
    pred = torch.sigmoid(outputs["detail_score_logits"])
    target = batch["detail_score_target"].to(pred.device).to(pred.dtype)
    score_valid = batch["detail_score_valid"].to(pred.device).to(pred.dtype) * valid.unsqueeze(-1)
    loss = (F.smooth_l1_loss(pred, target, reduction="none") * score_valid).sum() / score_valid.sum().clamp_min(1.0)
    with torch.no_grad():
        mae = ((pred - target).abs() * score_valid).sum() / score_valid.sum().clamp_min(1.0)
    return loss, {"detail_score_mae": mae.detach(), "detail_score_count": score_valid.sum().detach()}


def _why_tag_loss(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], valid: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if "why_tag_logits" not in outputs or "why_tag_target" not in batch:
        zero = valid.sum() * 0.0
        return zero, {"why_tag_precision": zero.detach(), "why_tag_recall": zero.detach(), "why_tag_count": zero.detach()}
    logits = outputs["why_tag_logits"]
    target = batch["why_tag_target"].to(logits.device).to(logits.dtype)
    tag_valid = batch["why_tag_valid"].to(logits.device).to(logits.dtype).unsqueeze(-1) * valid.unsqueeze(-1)
    if "why_tag_applicable" in batch:
        tag_valid = tag_valid * batch["why_tag_applicable"].to(logits.device).to(logits.dtype)
    tag_mask = tag_valid.expand_as(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    loss = (bce * tag_mask).sum() / tag_mask.sum().clamp_min(1.0)
    with torch.no_grad():
        pred = (torch.sigmoid(logits) >= 0.5).to(logits.dtype)
        tp = (pred * target * tag_mask).sum()
        fp = (pred * (1.0 - target) * tag_mask).sum()
        fn = ((1.0 - pred) * target * tag_mask).sum()
        precision = tp / (tp + fp).clamp_min(1.0)
        recall = tp / (tp + fn).clamp_min(1.0)
    return loss, {"why_tag_precision": precision.detach(), "why_tag_recall": recall.detach(), "why_tag_count": tag_mask.sum().detach()}


def _generated_proposal_alignment_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    score_weight: float = 1.0,
    listwise_weight: float = 0.5,
    positive_weight: float = 0.3,
    risk_weight: float = 0.2,
    positive_utility_weight: float = 0.0,
    positive_margin_weight: float = 0.0,
    positive_margin: float = 0.25,
    match_iou: float = 0.35,
    positive_iou: float = 0.5,
    risk_iou: float = 0.5,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if "generated_utility_logits" not in outputs or "generated_boxes" not in outputs:
        zero = outputs["utility_logits"].sum() * 0.0
        return zero, {
            "generated_align_loss": zero.detach(),
            "generated_align_match_rate": zero.detach(),
            "generated_align_top1_hit": zero.detach(),
            "generated_align_top1_iou": zero.detach(),
            "generated_align_positive_utility_loss": zero.detach(),
            "generated_align_positive_margin_loss": zero.detach(),
            "generated_align_any_positive_0_5": zero.detach(),
        }

    proposal_boxes = outputs["generated_boxes"]
    logits = outputs.get("generated_return_logits", outputs["generated_utility_logits"])
    positive_logits = outputs["generated_positive_logits"]
    risk_logits = outputs["generated_risk_logits"]
    proposal_valid = outputs.get("generated_valid")
    if proposal_valid is None:
        proposal_valid = torch.ones_like(logits)
    else:
        proposal_valid = proposal_valid.to(logits.device).to(logits.dtype)

    cand_boxes = batch["boxes"].to(proposal_boxes.device)
    cand_valid = batch["valid"].to(proposal_boxes.device).to(proposal_boxes.dtype)
    cand_score = batch["score_target"].to(proposal_boxes.device).to(proposal_boxes.dtype)
    cand_risk = batch["risk_target"].to(proposal_boxes.device).to(proposal_boxes.dtype)
    pos_boxes = batch["positive_boxes"].to(proposal_boxes.device)
    pos_valid = batch["positive_valid"].to(proposal_boxes.device).to(proposal_boxes.dtype)

    iou_to_cand = _box_iou_torch(proposal_boxes, cand_boxes)
    iou_to_cand = iou_to_cand.masked_fill(cand_valid.unsqueeze(1) <= 0, -1.0)
    best_iou_to_cand, best_cand_idx = iou_to_cand.max(dim=2)
    matched = (best_iou_to_cand >= float(match_iou)).to(logits.dtype) * proposal_valid
    gathered_score = cand_score.gather(1, best_cand_idx)
    matched_weight = best_iou_to_cand.clamp(0.05, 1.0) * matched

    score_loss = logits.sum() * 0.0
    list_loss = logits.sum() * 0.0
    if matched_weight.sum() > 0:
        score_bce = F.binary_cross_entropy_with_logits(logits, gathered_score, reduction="none")
        score_loss = (score_bce * matched_weight).sum() / matched_weight.sum().clamp_min(1.0)
        list_loss = _listwise_loss(logits, gathered_score, matched)

    iou_to_pos = _box_iou_torch(proposal_boxes, pos_boxes)
    iou_to_pos = iou_to_pos.masked_fill(pos_valid.unsqueeze(1) <= 0, -1.0)
    best_iou_to_pos, _ = iou_to_pos.max(dim=2)
    pos_target = ((best_iou_to_pos >= float(positive_iou)).to(logits.dtype) * proposal_valid).detach()
    pos_loss = _masked_mean(
        F.binary_cross_entropy_with_logits(positive_logits, pos_target, reduction="none"),
        proposal_valid,
    )
    positive_utility_loss = logits.sum() * 0.0
    if float(positive_utility_weight) > 0.0:
        pos_mask = (pos_target > 0.5).to(logits.dtype) * proposal_valid
        neg_mask = (proposal_valid - pos_mask).clamp_min(0.0)
        pos_count = pos_mask.sum()
        neg_count = neg_mask.sum()
        total_count = (pos_count + neg_count).clamp_min(1.0)
        pos_scale = total_count / (2.0 * pos_count.clamp_min(1.0))
        neg_scale = total_count / (2.0 * neg_count.clamp_min(1.0))
        utility_weight = torch.where(pos_mask > 0.0, pos_scale, neg_scale) * proposal_valid
        utility_bce = F.binary_cross_entropy_with_logits(logits, pos_target, reduction="none")
        positive_utility_loss = (utility_bce * utility_weight).sum() / utility_weight.sum().clamp_min(1.0)
    positive_margin_loss = logits.sum() * 0.0
    if float(positive_margin_weight) > 0.0:
        pos_mask_bool = (pos_target > 0.5) & (proposal_valid > 0.0)
        neg_mask_bool = (pos_target <= 0.5) & (proposal_valid > 0.0)
        row_valid_margin = (pos_mask_bool.sum(dim=1) > 0) & (neg_mask_bool.sum(dim=1) > 0)
        if torch.any(row_valid_margin):
            pos_max = logits.masked_fill(~pos_mask_bool, -1e4).max(dim=1).values
            neg_max = logits.masked_fill(~neg_mask_bool, -1e4).max(dim=1).values
            margin_raw = F.softplus(-(pos_max - neg_max - float(positive_margin)))
            positive_margin_loss = margin_raw[row_valid_margin].mean()

    risk_mask = (cand_risk > 0).to(proposal_boxes.dtype) * cand_valid
    risk_iou_all = _box_iou_torch(proposal_boxes, cand_boxes)
    risk_iou_all = risk_iou_all.masked_fill(risk_mask.unsqueeze(1) <= 0, -1.0)
    best_iou_to_risk, _ = risk_iou_all.max(dim=2)
    risk_target = ((best_iou_to_risk >= float(risk_iou)).to(logits.dtype) * proposal_valid).detach()
    risk_loss = _masked_mean(
        F.binary_cross_entropy_with_logits(risk_logits, risk_target, reduction="none"),
        proposal_valid,
    )

    total = (
        float(score_weight) * score_loss
        + float(listwise_weight) * list_loss
        + float(positive_weight) * pos_loss
        + float(risk_weight) * risk_loss
        + float(positive_utility_weight) * positive_utility_loss
        + float(positive_margin_weight) * positive_margin_loss
    )

    with torch.no_grad():
        masked_prob = torch.sigmoid(logits).masked_fill(proposal_valid <= 0, -1.0)
        top_idx = masked_prob.argmax(dim=1)
        top_hit = pos_target.gather(1, top_idx[:, None]).squeeze(1)
        top_iou = best_iou_to_pos.gather(1, top_idx[:, None]).squeeze(1)
        row_valid = (proposal_valid.sum(dim=1) > 0).to(logits.dtype)
        any_positive = ((best_iou_to_pos >= 0.5).to(logits.dtype) * proposal_valid).sum(dim=1) > 0
        denom = row_valid.sum().clamp_min(1.0)
        metrics = {
            "generated_align_loss": total.detach(),
            "generated_align_score_loss": score_loss.detach(),
            "generated_align_listwise_loss": list_loss.detach(),
            "generated_align_positive_loss": pos_loss.detach(),
            "generated_align_risk_loss": risk_loss.detach(),
            "generated_align_positive_utility_loss": positive_utility_loss.detach(),
            "generated_align_positive_margin_loss": positive_margin_loss.detach(),
            "generated_align_match_rate": (matched.sum(dim=1) / proposal_valid.sum(dim=1).clamp_min(1.0)).mean().detach(),
            "generated_align_top1_hit": ((top_hit * row_valid).sum() / denom).detach(),
            "generated_align_top1_iou": ((top_iou * row_valid).sum() / denom).detach(),
            "generated_align_positive_recall_0_5": (((best_iou_to_pos >= 0.5).to(logits.dtype) * proposal_valid).sum() / proposal_valid.sum().clamp_min(1.0)).detach(),
            "generated_align_any_positive_0_5": (any_positive.to(logits.dtype) * row_valid).sum().detach() / denom,
        }
    return total, metrics


def compute_mobilecropnet_v4_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    score_weight: float = 1.0,
    listwise_weight: float = 0.6,
    pairwise_weight: float = 0.5,
    explicit_pairwise_weight: float = 0.0,
    positive_weight: float = 0.4,
    risk_weight: float = 0.2,
    top1_risk_weight: float = 0.0,
    top1_risk_margin: float = 0.2,
    top_return_weight: float = 0.0,
    raw_top_return_weight: float = 0.0,
    top_return_score_margin: float = 0.03,
    top_return_logit_margin: float = 0.25,
    top_return_temperature: float = 0.12,
    return_score_weight: float = 0.0,
    return_listwise_weight: float = 0.0,
    return_positive_weight: float = 0.0,
    return_positive_margin_weight: float = 0.0,
    return_positive_margin: float = 0.2,
    return_risk_suppression_weight: float = 0.0,
    return_risk_suppression_margin: float = 0.2,
    return_exact_weight: float = 0.0,
    return_exact_min_gap: float = 0.0,
    return_target_mode: str = "score",
    return_explicit_pairwise_weight: float = 0.0,
    return_explicit_pairwise_margin: float = 0.0,
    teacher_distill_weight: float = 0.0,
    teacher_distill_temperature: float = 0.12,
    topk_coverage_weight: float = 0.0,
    topk_coverage_k: int = 4,
    topk_coverage_temperature: float = 0.12,
    macro_weight: float = 0.1,
    checklist_class_weight: float = 0.08,
    checklist_applicability_weight: float = 0.04,
    detail_score_weight: float = 0.05,
    why_tag_weight: float = 0.04,
    route_weight: float = 0.1,
    route_balanced_ce: bool = False,
    route_focal_gamma: float = 0.0,
    route_object_single_margin_weight: float = 0.0,
    route_object_single_margin: float = 0.18,
    route_hierarchy_weight: float = 0.0,
    route_cardinality_weight: float = 0.0,
    route_expert_distill_weight: float = 0.0,
    route_expert_distill_temperature: float = 1.0,
    route_expert_distill_hard_weight: float = 0.0,
    decision_weight: float = 0.25,
    decision_crop_rebalance: bool = False,
    decision_crop_rebalance_max_weight: float = 4.0,
    decision_source_weight: float = 0.0,
    decision_source_balanced_bce: bool = False,
    decision_source_crop_rebalance_max_weight: float = 4.0,
    decision_source_focal_gamma: float = 0.0,
    decision_source_focal_alpha: float | None = None,
    decision_source_margin_weight: float = 0.0,
    decision_source_logit_margin: float = 0.35,
    decision_source_margin_balanced: bool = False,
    decision_source_margin_balance_max_weight: float = 4.0,
    decision_source_margin_crop_weight_mult: float = 1.0,
    decision_source_pair_supervision_weight: float = 0.0,
    decision_source_pair_margin_weight: float = 0.0,
    decision_source_pair_balanced_bce: bool = False,
    decision_source_pair_focal_gamma: float = 0.0,
    decision_source_pair_focal_alpha: float | None = None,
    source_gate_supervision_weight: float = 0.0,
    source_gate_balanced_bce: bool = False,
    source_gate_crop_rebalance_max_weight: float = 4.0,
    source_gate_focal_gamma: float = 0.0,
    source_gate_focal_alpha: float | None = None,
    source_gate_margin_weight: float = 0.0,
    source_gate_logit_margin: float = 0.35,
    source_gate_margin_balanced: bool = False,
    source_gate_margin_balance_max_weight: float = 4.0,
    source_gate_margin_crop_weight_mult: float = 1.0,
    action_consistency_weight: float = 0.0,
    action_consistency_margin: float = 0.15,
    action_source_balanced: bool = False,
    action_source_balance_max_weight: float = 4.0,
    action_source_crop_weight_mult: float = 1.0,
    action_return_joint_weight: float = 0.0,
    action_return_joint_score_margin: float = 0.03,
    action_return_joint_logit_margin: float = 0.20,
    action_return_joint_temperature: float = 0.12,
    return_source_margin_weight: float = 0.0,
    return_source_margin_logit_margin: float = 0.25,
    return_source_margin_balanced: bool = False,
    return_source_margin_balance_max_weight: float = 4.0,
    return_source_margin_crop_weight_mult: float = 1.0,
    decision_utility_align_weight: float = 0.0,
    decision_utility_align_temperature: float = 0.35,
    policy_score_weight: float = 0.0,
    delta_weight: float = 0.1,
    proposal_weight: float = 0.8,
    proposal_subject_weight: float = 0.0,
    proposal_use_positive_scores: bool = True,
    subject_proposal_align_weight: float = 0.0,
    subject_box_weight: float = 0.0,
    subject_box_valid_weight: float = 1.0,
    subject_box_l1_weight: float = 2.0,
    subject_box_iou_weight: float = 1.0,
    subject_box_center_weight: float = 0.0,
    subject_box_size_weight: float = 0.0,
    subject_box_aspect_weight: float = 0.0,
    subject_box_ciou_weight: float = 0.0,
    subject_box_spatial_aux_weight: float = 0.0,
    subject_box_spatial_heatmap_weight: float = 0.0,
    subject_box_spatial_mask_weight: float = 0.0,
    subject_box_valid_balanced_bce: bool = False,
    subject_box_valid_negative_scale: float = 1.0,
    generated_proposal_align_weight: float = 0.0,
    generated_proposal_score_weight: float = 1.0,
    generated_proposal_listwise_weight: float = 0.5,
    generated_proposal_positive_weight: float = 0.3,
    generated_proposal_risk_weight: float = 0.2,
    generated_proposal_positive_utility_weight: float = 0.0,
    generated_proposal_positive_margin_weight: float = 0.0,
    generated_proposal_positive_margin: float = 0.25,
    generated_proposal_match_iou: float = 0.35,
) -> tuple[torch.Tensor, dict[str, float]]:
    valid = batch["valid"].to(outputs["utility_logits"].device).to(outputs["utility_logits"].dtype)
    score_target = batch["score_target"].to(outputs["utility_logits"].device).to(outputs["utility_logits"].dtype)
    teacher_soft_target = batch.get("teacher_soft_target")
    if teacher_soft_target is None:
        teacher_soft_target = torch.zeros_like(score_target)
    else:
        teacher_soft_target = teacher_soft_target.to(outputs["utility_logits"].device).to(outputs["utility_logits"].dtype)
    positive_target = batch["positive_target"].to(outputs["utility_logits"].device).to(outputs["utility_logits"].dtype)
    risk_target = batch["risk_target"].to(outputs["utility_logits"].device).to(outputs["utility_logits"].dtype)
    candidate_weight = batch.get("candidate_weight")
    if candidate_weight is None:
        candidate_weight = torch.ones_like(valid)
    else:
        candidate_weight = candidate_weight.to(outputs["utility_logits"].device).to(outputs["utility_logits"].dtype)
    candidate_is_base = batch["candidate_is_base"].to(outputs["utility_logits"].device).to(outputs["utility_logits"].dtype)
    decision_target = batch["decision_target"].to(outputs["decision_logits"].device)
    return_target = score_target
    return_target_mode_norm = str(return_target_mode or "score").lower()
    if return_target_mode_norm in {"decision_source", "decision_consistent", "action_source"}:
        return_target = _decision_source_return_target(score_target, valid, candidate_is_base, decision_target)

    score_bce = F.binary_cross_entropy_with_logits(outputs["utility_logits"], score_target, reduction="none")
    weighted_valid = valid * candidate_weight.clamp_min(0.05)
    score_loss = _masked_mean(score_bce, weighted_valid)
    list_loss = _listwise_loss(outputs["utility_logits"], score_target, valid)
    teacher_distill_loss, teacher_distill_metrics = _teacher_distribution_loss(
        outputs["utility_logits"],
        teacher_soft_target,
        valid,
        temperature=teacher_distill_temperature,
    )
    pair_loss = _pairwise_loss(outputs["utility_logits"], score_target, valid, candidate_weight=candidate_weight)
    explicit_pair_loss, explicit_pair_metrics = _explicit_pairwise_loss(outputs["utility_logits"], batch)
    pos_loss = _masked_mean(F.binary_cross_entropy_with_logits(outputs["positive_logits"], positive_target, reduction="none"), valid)
    risk_loss = _masked_mean(F.binary_cross_entropy_with_logits(outputs["risk_logits"], risk_target, reduction="none"), valid)
    top1_risk_loss, top1_risk_metrics = _top1_risk_loss(outputs["utility_logits"], positive_target, risk_target, valid, margin=top1_risk_margin)
    action_logits = outputs.get(
        "source_gate_return_logits",
        outputs.get("source_mixture_return_logits", outputs.get("return_logits", outputs["utility_logits"])),
    )
    return_score_loss = _masked_mean(
        F.binary_cross_entropy_with_logits(action_logits, return_target, reduction="none"),
        weighted_valid,
    )
    return_listwise_loss = _listwise_loss(action_logits, return_target, valid)
    top_return_loss, top_return_metrics = _top_return_loss(
        action_logits,
        return_target,
        valid,
        candidate_weight,
        score_margin=top_return_score_margin,
        logit_margin=top_return_logit_margin,
        temperature=top_return_temperature,
    )
    raw_top_return_loss, raw_top_return_metrics = _top_return_loss(
        action_logits,
        score_target,
        valid,
        candidate_weight,
        score_margin=top_return_score_margin,
        logit_margin=top_return_logit_margin,
        temperature=top_return_temperature,
    )
    return_positive_loss = _masked_mean(
        F.binary_cross_entropy_with_logits(action_logits, positive_target, reduction="none"),
        valid,
    )
    return_positive_margin_loss, return_positive_margin_metrics = _positive_margin_loss(
        action_logits,
        positive_target,
        valid,
        margin=return_positive_margin,
    )
    return_risk_suppression_loss, return_risk_suppression_metrics = _top1_risk_loss(
        action_logits,
        positive_target,
        risk_target,
        valid,
        margin=return_risk_suppression_margin,
    )
    return_exact_loss, return_exact_metrics = _return_exact_loss(
        action_logits,
        return_target,
        valid,
        candidate_weight,
        min_gap=return_exact_min_gap,
    )
    return_explicit_pair_loss, return_explicit_pair_metrics = _explicit_pairwise_loss(
        action_logits,
        batch,
        margin=return_explicit_pairwise_margin,
    )
    topk_coverage_loss, topk_coverage_metrics = _topk_coverage_loss(
        outputs["utility_logits"],
        score_target,
        valid,
        k=topk_coverage_k,
        temperature=topk_coverage_temperature,
    )

    macro_valid = batch["macro_valid"].to(outputs["utility_logits"].device).to(outputs["utility_logits"].dtype)
    macro_target = batch["macro_target"].to(outputs["utility_logits"].device).to(outputs["utility_logits"].dtype)
    macro_loss = _masked_mean(F.smooth_l1_loss(torch.sigmoid(outputs["macro_logits"]), macro_target, reduction="none").mean(dim=-1), valid * macro_valid)
    checklist_applicability_loss, checklist_applicability_metrics = _checklist_applicability_loss(outputs, batch, valid)
    checklist_class_loss, checklist_class_metrics = _checklist_class_loss(outputs, batch, valid)
    detail_score_loss, detail_score_metrics = _detail_score_loss(outputs, batch, valid)
    why_tag_loss, why_tag_metrics = _why_tag_loss(outputs, batch, valid)

    route_logits_for_loss = outputs.get("route_fine_logits", outputs["route_logits"])
    route_target = batch["route_target"].to(route_logits_for_loss.device)
    if route_balanced_ce:
        class_count = route_logits_for_loss.shape[1]
        counts = torch.bincount(route_target.view(-1), minlength=class_count).to(route_logits_for_loss.dtype)
        weights = counts.sum().clamp_min(1.0) / counts.clamp_min(1.0)
        weights = weights / weights.mean().clamp_min(1e-6)
        route_ce = F.cross_entropy(route_logits_for_loss, route_target, weight=weights, reduction="none")
    else:
        route_ce = F.cross_entropy(route_logits_for_loss, route_target, reduction="none")
    route_focal_factor = torch.ones_like(route_ce)
    if float(route_focal_gamma) > 0.0:
        route_prob = route_logits_for_loss.softmax(dim=1).gather(1, route_target.view(-1, 1)).squeeze(1)
        route_focal_factor = (1.0 - route_prob).clamp(0.0, 1.0).pow(float(route_focal_gamma))
    route_loss = (route_focal_factor * route_ce).mean()
    route_expert_distill_loss = route_logits_for_loss.sum() * 0.0
    route_expert_hard_loss = route_logits_for_loss.sum() * 0.0
    route_expert_metrics: dict[str, torch.Tensor] = {}
    route_expert_logits = batch.get("route_expert_logits")
    if torch.is_tensor(route_expert_logits):
        route_expert_logits = route_expert_logits.to(route_logits_for_loss.device).to(route_logits_for_loss.dtype)
        temperature = max(1.0e-6, float(route_expert_distill_temperature))
        route_expert_prob = F.softmax(route_expert_logits / temperature, dim=1)
        route_student_log_prob = F.log_softmax(route_logits_for_loss / temperature, dim=1)
        route_expert_distill_loss = F.kl_div(route_student_log_prob, route_expert_prob, reduction="batchmean") * (temperature * temperature)
        route_expert_pred = route_expert_logits.argmax(dim=1)
        route_expert_hard_loss = F.cross_entropy(route_logits_for_loss, route_expert_pred)
        route_student_pred = route_logits_for_loss.argmax(dim=1)
        route_expert_correct = (route_expert_pred == route_target).to(route_logits_for_loss.dtype)
        route_student_expert_agree = (route_student_pred == route_expert_pred).to(route_logits_for_loss.dtype)
        expert_recalls = []
        agreement_recalls = []
        for class_idx in range(route_logits_for_loss.shape[1]):
            mask = route_target == int(class_idx)
            if bool(mask.any().item()):
                expert_recalls.append(route_expert_correct[mask].mean())
                agreement_recalls.append(route_student_expert_agree[mask].mean())
        route_expert_metrics = {
            "route_expert_acc": route_expert_correct.mean().detach(),
            "route_expert_agreement": route_student_expert_agree.mean().detach(),
            "route_expert_balanced_acc": (
                torch.stack(expert_recalls).mean().detach() if expert_recalls else route_expert_correct.mean().detach()
            ),
            "route_expert_agreement_balanced": (
                torch.stack(agreement_recalls).mean().detach() if agreement_recalls else route_student_expert_agree.mean().detach()
            ),
        }
    route_object_single_margin_loss = route_logits_for_loss.sum() * 0.0
    route_object_single_margin_metrics: dict[str, torch.Tensor] = {}
    if float(route_object_single_margin_weight) > 0.0:
        route_object_single_margin_loss, route_object_single_margin_metrics = _route_object_single_margin_loss(
            route_logits_for_loss,
            route_target,
            margin=float(route_object_single_margin),
        )
    if torch.is_tensor(outputs.get("route_kind_logits")):
        route_hierarchy_loss, route_hierarchy_metrics = _route_aux_head_loss(
            outputs["route_kind_logits"],
            route_target,
            _ROUTE_KIND_GROUPS,
            metric_prefix="route_hierarchy",
        )
    else:
        route_hierarchy_loss, route_hierarchy_metrics = _route_group_aux_loss(
            route_logits_for_loss,
            route_target,
            _ROUTE_KIND_GROUPS,
            metric_prefix="route_hierarchy",
        )
    if torch.is_tensor(outputs.get("route_cardinality_logits")):
        route_cardinality_loss, route_cardinality_metrics = _route_aux_head_loss(
            outputs["route_cardinality_logits"],
            route_target,
            _ROUTE_CARDINALITY_GROUPS,
            metric_prefix="route_cardinality",
        )
    else:
        route_cardinality_loss, route_cardinality_metrics = _route_group_aux_loss(
            route_logits_for_loss,
            route_target,
            _ROUTE_CARDINALITY_GROUPS,
            metric_prefix="route_cardinality",
        )
    decision_class_weight = None
    if bool(decision_crop_rebalance):
        crop_decision_id = DECISION_VOCAB.index("crop")
        crop_count = int((decision_target == crop_decision_id).sum().item())
        non_crop_count = int(decision_target.numel() - crop_count)
        if crop_count > 0 and non_crop_count > 0:
            decision_class_weight = torch.ones(outputs["decision_logits"].shape[1], dtype=outputs["decision_logits"].dtype, device=outputs["decision_logits"].device)
            crop_weight = min(float(decision_crop_rebalance_max_weight), float(non_crop_count) / float(max(1, crop_count)))
            decision_class_weight[crop_decision_id] = float(max(1.0, crop_weight))
    decision_loss = F.cross_entropy(outputs["decision_logits"], decision_target, weight=decision_class_weight)
    decision_source_loss, decision_source_metrics = _decision_source_loss(
        outputs.get("decision_source_logit"),
        decision_target,
        outputs["decision_logits"],
        balanced_bce=decision_source_balanced_bce,
        crop_rebalance_max_weight=decision_source_crop_rebalance_max_weight,
        focal_gamma=decision_source_focal_gamma,
        focal_alpha=decision_source_focal_alpha,
    )
    decision_source_margin_loss, decision_source_margin_metrics = _decision_source_margin_loss(
        outputs.get("decision_source_logit"),
        decision_target,
        outputs["decision_logits"],
        logit_margin=decision_source_logit_margin,
        source_balanced=decision_source_margin_balanced,
        source_balance_max_weight=decision_source_margin_balance_max_weight,
        source_crop_weight_mult=decision_source_margin_crop_weight_mult,
    )
    decision_source_pair_loss, decision_source_pair_metrics = _decision_source_loss(
        outputs.get("decision_source_pair_logit"),
        decision_target,
        outputs["decision_logits"],
        balanced_bce=decision_source_pair_balanced_bce,
        crop_rebalance_max_weight=decision_source_crop_rebalance_max_weight,
        focal_gamma=decision_source_pair_focal_gamma,
        focal_alpha=decision_source_pair_focal_alpha,
    )
    decision_source_pair_margin_loss, decision_source_pair_margin_metrics = _decision_source_margin_loss(
        outputs.get("decision_source_pair_logit"),
        decision_target,
        outputs["decision_logits"],
        logit_margin=decision_source_logit_margin,
        source_balanced=decision_source_margin_balanced,
        source_balance_max_weight=decision_source_margin_balance_max_weight,
        source_crop_weight_mult=decision_source_margin_crop_weight_mult,
    )
    source_gate_supervision_loss, source_gate_supervision_raw_metrics = _decision_source_loss(
        outputs.get("source_gate_logit"),
        decision_target,
        outputs["decision_logits"],
        balanced_bce=source_gate_balanced_bce,
        crop_rebalance_max_weight=source_gate_crop_rebalance_max_weight,
        focal_gamma=source_gate_focal_gamma,
        focal_alpha=source_gate_focal_alpha,
    )
    source_gate_supervision_metrics = {
        key.replace("policy_source", "source_gate_source", 1): value for key, value in source_gate_supervision_raw_metrics.items()
    }
    source_gate_margin_loss, source_gate_margin_raw_metrics = _decision_source_margin_loss(
        outputs.get("source_gate_logit"),
        decision_target,
        outputs["decision_logits"],
        logit_margin=source_gate_logit_margin,
        source_balanced=source_gate_margin_balanced,
        source_balance_max_weight=source_gate_margin_balance_max_weight,
        source_crop_weight_mult=source_gate_margin_crop_weight_mult,
    )
    source_gate_margin_metrics = {
        key.replace("decision_source_margin", "source_gate_margin", 1): value for key, value in source_gate_margin_raw_metrics.items()
    }
    action_consistency_loss, action_consistency_metrics = _decision_action_loss(
        action_logits,
        decision_target,
        valid,
        candidate_is_base,
        margin=action_consistency_margin,
        source_balanced=action_source_balanced,
        source_balance_max_weight=action_source_balance_max_weight,
        source_crop_weight_mult=action_source_crop_weight_mult,
    )
    decision_conditioned_action_metrics = _decision_conditioned_action_metrics(
        outputs.get("decision_conditioned_return_logits", action_logits),
        outputs["decision_logits"],
        decision_target,
        valid,
        candidate_is_base,
        return_target,
        score_target,
        score_margin=top_return_score_margin,
        decision_source_logit=outputs.get("decision_source_logit"),
        sweep_logits=action_logits,
        logits_already_conditioned="decision_conditioned_return_logits" in outputs,
    )
    source_mixture_action_metrics: dict[str, torch.Tensor] = {}
    if "source_mixture_return_logits" in outputs:
        source_mixture_action_metrics = _source_mixture_action_metrics(
            outputs["source_mixture_return_logits"],
            outputs["decision_logits"],
            decision_target,
            valid,
            candidate_is_base,
            return_target,
            score_target,
            score_margin=top_return_score_margin,
            decision_source_logit=outputs.get("decision_source_logit"),
        )
    source_gate_action_metrics: dict[str, torch.Tensor] = {}
    if "source_gate_return_logits" in outputs:
        source_gate_action_metrics = _source_mixture_action_metrics(
            outputs["source_gate_return_logits"],
            outputs["decision_logits"],
            decision_target,
            valid,
            candidate_is_base,
            return_target,
            score_target,
            score_margin=top_return_score_margin,
            decision_source_logit=outputs.get("source_gate_logit"),
            prefix="source_gate",
        )
    action_return_joint_loss, action_return_joint_metrics = _action_return_joint_loss(
        action_logits,
        score_target,
        valid,
        candidate_is_base,
        decision_target,
        candidate_weight,
        score_margin=action_return_joint_score_margin,
        logit_margin=action_return_joint_logit_margin,
        temperature=action_return_joint_temperature,
        source_balanced=action_source_balanced,
        source_balance_max_weight=action_source_balance_max_weight,
        source_crop_weight_mult=action_source_crop_weight_mult,
    )
    return_source_margin_loss, return_source_margin_metrics = _return_source_margin_loss(
        action_logits,
        valid,
        candidate_is_base,
        decision_target,
        logit_margin=return_source_margin_logit_margin,
        source_balanced=return_source_margin_balanced,
        source_balance_max_weight=return_source_margin_balance_max_weight,
        source_crop_weight_mult=return_source_margin_crop_weight_mult,
    )
    decision_utility_align_loss, decision_utility_align_metrics = _decision_utility_alignment_loss(
        outputs["decision_logits"],
        action_logits,
        valid,
        candidate_is_base,
        temperature=decision_utility_align_temperature,
    )
    policy_score_loss = outputs["utility_logits"].sum() * 0.0
    if "policy_score_pred" in outputs and "decision_score_target" in batch:
        policy_score_target = batch["decision_score_target"].to(outputs["utility_logits"].device).to(outputs["utility_logits"].dtype)
        policy_score_loss = F.smooth_l1_loss(outputs["policy_score_pred"], policy_score_target)
    delta_target = batch["delta_target"].to(outputs["delta_pred"].device).to(outputs["delta_pred"].dtype).clamp(-3.0, 3.0) / 3.0
    delta_loss = F.smooth_l1_loss(torch.tanh(outputs["delta_pred"]), delta_target)
    proposal_loss, prop_metrics = _proposal_loss(outputs, batch, use_positive_scores=proposal_use_positive_scores)
    proposal_subject_loss, proposal_subject_metrics = _proposal_subject_loss(outputs, batch)
    subject_proposal_loss, subject_proposal_metrics = _subject_proposal_loss(outputs, batch)
    subject_box_loss, subject_box_metrics = _subject_box_loss(
        outputs,
        batch,
        valid_weight=subject_box_valid_weight,
        l1_weight=subject_box_l1_weight,
        iou_weight=subject_box_iou_weight,
        center_weight=subject_box_center_weight,
        size_weight=subject_box_size_weight,
        aspect_weight=subject_box_aspect_weight,
        ciou_weight=subject_box_ciou_weight,
        spatial_aux_weight=subject_box_spatial_aux_weight,
        spatial_heatmap_weight=subject_box_spatial_heatmap_weight,
        spatial_mask_weight=subject_box_spatial_mask_weight,
        valid_balanced_bce=subject_box_valid_balanced_bce,
        valid_negative_scale=subject_box_valid_negative_scale,
    )
    generated_align_loss, generated_align_metrics = _generated_proposal_alignment_loss(
        outputs,
        batch,
        score_weight=generated_proposal_score_weight,
        listwise_weight=generated_proposal_listwise_weight,
        positive_weight=generated_proposal_positive_weight,
        risk_weight=generated_proposal_risk_weight,
        positive_utility_weight=generated_proposal_positive_utility_weight,
        positive_margin_weight=generated_proposal_positive_margin_weight,
        positive_margin=generated_proposal_positive_margin,
        match_iou=generated_proposal_match_iou,
    )

    total = (
        score_weight * score_loss
        + listwise_weight * list_loss
        + pairwise_weight * pair_loss
        + explicit_pairwise_weight * explicit_pair_loss
        + teacher_distill_weight * teacher_distill_loss
        + positive_weight * pos_loss
        + risk_weight * risk_loss
        + top1_risk_weight * top1_risk_loss
        + top_return_weight * top_return_loss
        + raw_top_return_weight * raw_top_return_loss
        + return_score_weight * return_score_loss
        + return_listwise_weight * return_listwise_loss
        + return_positive_weight * return_positive_loss
        + return_positive_margin_weight * return_positive_margin_loss
        + return_risk_suppression_weight * return_risk_suppression_loss
        + return_exact_weight * return_exact_loss
        + return_explicit_pairwise_weight * return_explicit_pair_loss
        + topk_coverage_weight * topk_coverage_loss
        + macro_weight * macro_loss
        + checklist_applicability_weight * checklist_applicability_loss
        + checklist_class_weight * checklist_class_loss
        + detail_score_weight * detail_score_loss
        + why_tag_weight * why_tag_loss
        + route_weight * route_loss
        + route_object_single_margin_weight * route_object_single_margin_loss
        + route_hierarchy_weight * route_hierarchy_loss
        + route_cardinality_weight * route_cardinality_loss
        + route_expert_distill_weight * (route_expert_distill_loss + route_expert_distill_hard_weight * route_expert_hard_loss)
        + decision_weight * decision_loss
        + decision_source_weight * decision_source_loss
        + decision_source_margin_weight * decision_source_margin_loss
        + decision_source_pair_supervision_weight * decision_source_pair_loss
        + decision_source_pair_margin_weight * decision_source_pair_margin_loss
        + source_gate_supervision_weight * source_gate_supervision_loss
        + source_gate_margin_weight * source_gate_margin_loss
        + action_consistency_weight * action_consistency_loss
        + action_return_joint_weight * action_return_joint_loss
        + return_source_margin_weight * return_source_margin_loss
        + decision_utility_align_weight * decision_utility_align_loss
        + policy_score_weight * policy_score_loss
        + delta_weight * delta_loss
        + proposal_weight * proposal_loss
        + proposal_subject_weight * proposal_subject_loss
        + subject_proposal_align_weight * subject_proposal_loss
        + subject_box_weight * subject_box_loss
        + generated_proposal_align_weight * generated_align_loss
    )

    with torch.no_grad():
        prob = torch.sigmoid(outputs["utility_logits"])
        action_prob = torch.sigmoid(action_logits)
        masked_prob = prob.masked_fill(valid <= 0, -1.0)
        top_idx = masked_prob.argmax(dim=1)
        top_positive = positive_target.gather(1, top_idx[:, None]).squeeze(1)
        action_top_idx = action_prob.masked_fill(valid <= 0, -1.0).argmax(dim=1)
        action_top_positive = positive_target.gather(1, action_top_idx[:, None]).squeeze(1)
        best_idx = score_target.masked_fill(valid <= 0, -1.0).argmax(dim=1)
        top1_exact = (top_idx == best_idx).to(prob.dtype)
        action_top1_exact = (action_top_idx == best_idx).to(prob.dtype)
        score_mae = _masked_mean((prob - score_target).abs(), valid)
        route_pred = outputs["route_logits"].argmax(dim=1)
        route_correct = (route_pred == route_target).to(prob.dtype)
        route_fine_logits_for_metric = outputs.get("route_fine_logits", outputs["route_logits"])
        route_fine_pred = route_fine_logits_for_metric.argmax(dim=1)
        route_fine_correct = (route_fine_pred == route_target).to(prob.dtype)
        route_present = []
        route_recalls = []
        route_fine_recalls = []
        for class_idx in range(outputs["route_logits"].shape[1]):
            mask = route_target == int(class_idx)
            if bool(mask.any().item()):
                route_present.append(outputs["route_logits"].new_tensor(1.0))
                route_recalls.append(route_correct[mask].mean())
                route_fine_recalls.append(route_fine_correct[mask].mean())
        if route_recalls:
            route_balanced_acc = torch.stack(route_recalls).mean()
            route_fine_balanced_acc = torch.stack(route_fine_recalls).mean()
            route_present_class_frac = torch.stack(route_present).sum() / float(outputs["route_logits"].shape[1])
        else:
            route_balanced_acc = route_correct.mean()
            route_fine_balanced_acc = route_fine_correct.mean()
            route_present_class_frac = route_correct.sum() * 0.0
        metrics: dict[str, float] = {
            "loss": float(total.detach().cpu()),
            "score_loss": float(score_loss.detach().cpu()),
            "listwise_loss": float(list_loss.detach().cpu()),
            "pairwise_loss": float(pair_loss.detach().cpu()),
            "explicit_pairwise_loss": float(explicit_pair_loss.detach().cpu()),
            "teacher_distill_loss": float(teacher_distill_loss.detach().cpu()),
            "positive_loss": float(pos_loss.detach().cpu()),
            "risk_loss": float(risk_loss.detach().cpu()),
            "top1_risk_loss": float(top1_risk_loss.detach().cpu()),
            "top_return_loss": float(top_return_loss.detach().cpu()),
            "raw_top_return_loss": float(raw_top_return_loss.detach().cpu()),
            "return_score_loss": float(return_score_loss.detach().cpu()),
            "return_listwise_loss": float(return_listwise_loss.detach().cpu()),
            "return_positive_loss": float(return_positive_loss.detach().cpu()),
            "return_positive_margin_loss": float(return_positive_margin_loss.detach().cpu()),
            "return_risk_suppression_loss": float(return_risk_suppression_loss.detach().cpu()),
            "return_exact_loss": float(return_exact_loss.detach().cpu()),
            "return_explicit_pairwise_loss": float(return_explicit_pair_loss.detach().cpu()),
            "topk_coverage_loss": float(topk_coverage_loss.detach().cpu()),
            "macro_loss": float(macro_loss.detach().cpu()),
            "checklist_applicability_loss": float(checklist_applicability_loss.detach().cpu()),
            "checklist_class_loss": float(checklist_class_loss.detach().cpu()),
            "detail_score_loss": float(detail_score_loss.detach().cpu()),
            "why_tag_loss": float(why_tag_loss.detach().cpu()),
            "route_loss": float(route_loss.detach().cpu()),
            "route_expert_distill_loss": float(route_expert_distill_loss.detach().cpu()),
            "route_expert_hard_loss": float(route_expert_hard_loss.detach().cpu()),
            "route_focal_factor_mean": float(route_focal_factor.mean().detach().cpu()),
            "route_object_single_margin_loss": float(route_object_single_margin_loss.detach().cpu()),
            "route_hierarchy_loss": float(route_hierarchy_loss.detach().cpu()),
            "route_cardinality_loss": float(route_cardinality_loss.detach().cpu()),
            "decision_loss": float(decision_loss.detach().cpu()),
            "policy_source_loss": float(decision_source_loss.detach().cpu()),
            "decision_source_margin_loss": float(decision_source_margin_loss.detach().cpu()),
            "decision_source_pair_loss": float(decision_source_pair_loss.detach().cpu()),
            "decision_source_pair_margin_loss": float(decision_source_pair_margin_loss.detach().cpu()),
            "source_gate_source_loss": float(source_gate_supervision_loss.detach().cpu()),
            "source_gate_margin_loss": float(source_gate_margin_loss.detach().cpu()),
            "action_consistency_loss": float(action_consistency_loss.detach().cpu()),
            "action_return_joint_loss": float(action_return_joint_loss.detach().cpu()),
            "return_source_margin_loss": float(return_source_margin_loss.detach().cpu()),
            "decision_utility_align_loss": float(decision_utility_align_loss.detach().cpu()),
            "policy_score_loss": float(policy_score_loss.detach().cpu()),
            "delta_loss": float(delta_loss.detach().cpu()),
            "proposal_loss": float(proposal_loss.detach().cpu()),
            "proposal_subject_loss": float(proposal_subject_loss.detach().cpu()),
            "subject_proposal_align_loss": float(subject_proposal_loss.detach().cpu()),
            "subject_box_loss": float(subject_box_loss.detach().cpu()),
            "generated_align_loss": float(generated_align_loss.detach().cpu()),
            "score_mae": float(score_mae.detach().cpu()),
            "top1_hit": float(top_positive.mean().detach().cpu()),
            "top1_exact_best_score": float(top1_exact.mean().detach().cpu()),
            "return_top1_hit": float(action_top_positive.mean().detach().cpu()),
            "return_top1_exact_best_score": float(action_top1_exact.mean().detach().cpu()),
            "decision_acc": float((outputs["decision_logits"].argmax(dim=1) == decision_target).float().mean().detach().cpu()),
            "route_acc": float(route_correct.mean().detach().cpu()),
            "route_balanced_acc": float(route_balanced_acc.detach().cpu()),
            "route_fine_acc": float(route_fine_correct.mean().detach().cpu()),
            "route_fine_balanced_acc": float(route_fine_balanced_acc.detach().cpu()),
            "route_present_class_frac": float(route_present_class_frac.detach().cpu()),
        }
        if decision_class_weight is not None:
            metrics["decision_crop_class_weight"] = float(decision_class_weight[DECISION_VOCAB.index("crop")].detach().cpu())
        if "policy_score_pred" in outputs and "decision_score_target" in batch:
            metrics["policy_score_mae"] = float((outputs["policy_score_pred"] - batch["decision_score_target"].to(outputs["policy_score_pred"].device)).abs().mean().detach().cpu())
        for key, value in explicit_pair_metrics.items():
            metrics[key] = float(value.detach().cpu())
        for key, value in teacher_distill_metrics.items():
            metrics[key] = float(value.detach().cpu())
        for key, value in top1_risk_metrics.items():
            metrics[key] = float(value.detach().cpu())
        for key, value in top_return_metrics.items():
            metrics[key] = float(value.detach().cpu())
        if return_target_mode_norm in {"decision_source", "decision_consistent", "action_source"}:
            for key, value in raw_top_return_metrics.items():
                metrics[f"raw_{key}"] = float(value.detach().cpu())
            metrics["return_target_mode_decision_source"] = 1.0
        for key, value in return_risk_suppression_metrics.items():
            metrics[f"return_{key}"] = float(value.detach().cpu())
        for key, value in return_positive_margin_metrics.items():
            metrics[f"return_{key}"] = float(value.detach().cpu())
        for key, value in return_exact_metrics.items():
            metrics[key] = float(value.detach().cpu())
        for key, value in return_explicit_pair_metrics.items():
            metrics[f"return_{key}"] = float(value.detach().cpu())
        for key, value in topk_coverage_metrics.items():
            metrics[key] = float(value.detach().cpu())
        for key, value in action_consistency_metrics.items():
            metrics[key] = float(value.detach().cpu())
        for key, value in decision_source_metrics.items():
            metrics[key] = float(value.detach().cpu())
        for key, value in decision_source_margin_metrics.items():
            metrics[key] = float(value.detach().cpu())
        for key, value in decision_source_pair_metrics.items():
            metrics[f"decision_source_pair_{key}"] = float(value.detach().cpu())
        for key, value in decision_source_pair_margin_metrics.items():
            metrics[f"decision_source_pair_{key}"] = float(value.detach().cpu())
        for key, value in source_gate_supervision_metrics.items():
            metrics[key] = float(value.detach().cpu())
        for key, value in source_gate_margin_metrics.items():
            metrics[key] = float(value.detach().cpu())
        for key, value in decision_conditioned_action_metrics.items():
            metrics[key] = float(value.detach().cpu())
        for key, value in source_mixture_action_metrics.items():
            metrics[key] = float(value.detach().cpu())
        for key, value in source_gate_action_metrics.items():
            metrics[key] = float(value.detach().cpu())
        for key, value in action_return_joint_metrics.items():
            metrics[key] = float(value.detach().cpu())
        for key, value in return_source_margin_metrics.items():
            metrics[key] = float(value.detach().cpu())
        for key, value in decision_utility_align_metrics.items():
            metrics[key] = float(value.detach().cpu())
        for key, value in route_object_single_margin_metrics.items():
            metrics[key] = float(value.detach().cpu())
        for key, value in route_hierarchy_metrics.items():
            metrics[key] = float(value.detach().cpu())
        for key, value in route_cardinality_metrics.items():
            metrics[key] = float(value.detach().cpu())
        for key, value in route_expert_metrics.items():
            metrics[key] = float(value.detach().cpu())
        for key, value in prop_metrics.items():
            metrics[key] = float(value.detach().cpu())
        for key, value in proposal_subject_metrics.items():
            metrics[key] = float(value.detach().cpu())
        for key, value in subject_proposal_metrics.items():
            metrics[key] = float(value.detach().cpu())
        for key, value in subject_box_metrics.items():
            metrics[key] = float(value.detach().cpu())
        for key, value in generated_align_metrics.items():
            metrics[key] = float(value.detach().cpu())
        for key, value in checklist_applicability_metrics.items():
            metrics[key] = float(value.detach().cpu())
        for key, value in checklist_class_metrics.items():
            metrics[key] = float(value.detach().cpu())
        for key, value in detail_score_metrics.items():
            metrics[key] = float(value.detach().cpu())
        for key, value in why_tag_metrics.items():
            metrics[key] = float(value.detach().cpu())
    return total, metrics


def model_config_to_dict(model: MobileCropNetV4) -> dict[str, Any]:
    cfg = model.config
    return {
        "candidate_k": cfg.candidate_k,
        "proposal_q": cfg.proposal_q,
        "input_size": cfg.input_size,
        "width_mult": cfg.width_mult,
        "token_dim": cfg.token_dim,
        "backbone_name": cfg.backbone_name,
        "backbone_pretrained": cfg.backbone_pretrained,
        "ranker_type": cfg.ranker_type,
        "ranker_depth": cfg.ranker_depth,
        "route_classes": cfg.route_classes,
        "decision_classes": cfg.decision_classes,
        "ar_classes": cfg.ar_classes,
        "use_subject_prior": cfg.use_subject_prior,
        "route_image_only": cfg.route_image_only,
        "route_use_subject_prior": cfg.route_use_subject_prior,
        "route_use_candidate_context": cfg.route_use_candidate_context,
        "route_use_subject_box_features": cfg.route_use_subject_box_features,
        "route_use_subject_spatial_token": cfg.route_use_subject_spatial_token,
        "route_head_depth": cfg.route_head_depth,
        "route_head_hidden_mult": cfg.route_head_hidden_mult,
        "route_head_dropout": cfg.route_head_dropout,
        "route_aux_heads": cfg.route_aux_heads,
        "route_image_residual": cfg.route_image_residual,
        "route_image_residual_weight": cfg.route_image_residual_weight,
        "route_decode_mode": cfg.route_decode_mode,
        "route_decode_kind_weight": cfg.route_decode_kind_weight,
        "route_decode_cardinality_weight": cfg.route_decode_cardinality_weight,
        "policy_use_subject_prior": cfg.policy_use_subject_prior,
        "proposal_use_subject_prior": cfg.proposal_use_subject_prior,
        "subject_box_head": cfg.subject_box_head,
        "subject_box_spatial_head": cfg.subject_box_spatial_head,
        "subject_box_spatial_multiscale": cfg.subject_box_spatial_multiscale,
        "subject_box_spatial_mix_bias": cfg.subject_box_spatial_mix_bias,
        "subject_box_spatial_output": cfg.subject_box_spatial_output,
        "subject_box_spatial_box_mode": cfg.subject_box_spatial_box_mode,
        "subject_box_spatial_extent_scale": cfg.subject_box_spatial_extent_scale,
        "subject_box_coord_space": cfg.subject_box_coord_space,
        "subject_box_refine_with_proposals": cfg.subject_box_refine_with_proposals,
        "use_pred_subject_box_as_prior": cfg.use_pred_subject_box_as_prior,
        "detach_pred_subject_prior": cfg.detach_pred_subject_prior,
        "policy_score_head": cfg.policy_score_head,
        "policy_score_threshold_low": cfg.policy_score_threshold_low,
        "policy_score_threshold_high": cfg.policy_score_threshold_high,
        "decision_source_head": cfg.decision_source_head,
        "decision_source_logit_threshold": cfg.decision_source_logit_threshold,
        "decision_source_pair_head": cfg.decision_source_pair_head,
        "decision_source_pair_hidden_mult": cfg.decision_source_pair_hidden_mult,
        "decision_source_pair_after_return_deltas": cfg.decision_source_pair_after_return_deltas,
        "decision_source_pair_subject_state": cfg.decision_source_pair_subject_state,
        "decision_source_pair_subject_detach": cfg.decision_source_pair_subject_detach,
        "route_condition_candidate_scores": cfg.route_condition_candidate_scores,
        "route_condition_detach": cfg.route_condition_detach,
        "return_score_head": cfg.return_score_head,
        "return_score_head_depth": cfg.return_score_head_depth,
        "return_score_head_hidden_mult": cfg.return_score_head_hidden_mult,
        "return_score_action_source_bias": cfg.return_score_action_source_bias,
        "return_score_decision_source_bias": cfg.return_score_decision_source_bias,
        "return_score_decision_source_detach": cfg.return_score_decision_source_detach,
        "return_score_decision_source_max_scale": cfg.return_score_decision_source_max_scale,
        "return_score_decision_source_from_source_head": cfg.return_score_decision_source_from_source_head,
        "return_score_decision_conditioned": cfg.return_score_decision_conditioned,
        "return_score_decision_conditioned_mask_value": cfg.return_score_decision_conditioned_mask_value,
        "return_score_decision_conditioned_from_source_head": cfg.return_score_decision_conditioned_from_source_head,
        "return_score_source_mixture": cfg.return_score_source_mixture,
        "return_score_source_mixture_strength": cfg.return_score_source_mixture_strength,
        "return_score_source_mixture_detach": cfg.return_score_source_mixture_detach,
        "return_score_source_mixture_from_source_head": cfg.return_score_source_mixture_from_source_head,
        "return_score_source_mixture_gap_threshold": cfg.return_score_source_mixture_gap_threshold,
        "return_score_source_mixture_gap_mask_value": cfg.return_score_source_mixture_gap_mask_value,
        "return_score_source_gate": cfg.return_score_source_gate,
        "return_score_source_gate_hidden_mult": cfg.return_score_source_gate_hidden_mult,
        "return_score_source_gate_strength": cfg.return_score_source_gate_strength,
        "return_score_source_gate_detach": cfg.return_score_source_gate_detach,
        "return_score_source_gate_subject_state": cfg.return_score_source_gate_subject_state,
        "return_score_source_gate_subject_detach": cfg.return_score_source_gate_subject_detach,
        "return_score_source_gate_logit_threshold": cfg.return_score_source_gate_logit_threshold,
        "return_score_source_specific_head": cfg.return_score_source_specific_head,
        "return_score_source_specific_hidden_mult": cfg.return_score_source_specific_hidden_mult,
        "return_score_context_head": cfg.return_score_context_head,
        "return_score_context_hidden_mult": cfg.return_score_context_hidden_mult,
        "return_score_context_detach": cfg.return_score_context_detach,
        "return_score_policy_match_head": cfg.return_score_policy_match_head,
        "return_score_policy_match_hidden_mult": cfg.return_score_policy_match_hidden_mult,
        "return_score_policy_match_detach": cfg.return_score_policy_match_detach,
        "return_score_policy_source_head": cfg.return_score_policy_source_head,
        "return_score_policy_source_hidden_mult": cfg.return_score_policy_source_hidden_mult,
        "return_score_policy_source_detach": cfg.return_score_policy_source_detach,
        "return_score_competition_head": cfg.return_score_competition_head,
        "return_score_competition_hidden_mult": cfg.return_score_competition_hidden_mult,
        "return_score_competition_detach": cfg.return_score_competition_detach,
        "return_score_set_refiner_head": cfg.return_score_set_refiner_head,
        "return_score_set_refiner_hidden_mult": cfg.return_score_set_refiner_hidden_mult,
        "return_score_set_refiner_layers": cfg.return_score_set_refiner_layers,
        "return_score_set_refiner_detach": cfg.return_score_set_refiner_detach,
        "return_score_action_decoder_head": cfg.return_score_action_decoder_head,
        "return_score_action_decoder_hidden_mult": cfg.return_score_action_decoder_hidden_mult,
        "return_score_action_decoder_layers": cfg.return_score_action_decoder_layers,
        "return_score_action_decoder_detach": cfg.return_score_action_decoder_detach,
        "return_score_action_decoder_subject_state": cfg.return_score_action_decoder_subject_state,
        "return_score_action_decoder_subject_detach": cfg.return_score_action_decoder_subject_detach,
        "return_score_action_decoder_replace": cfg.return_score_action_decoder_replace,
    }
