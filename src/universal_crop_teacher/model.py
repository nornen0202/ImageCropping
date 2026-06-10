from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from universal_crop_teacher.data import CANDIDATE_FEATURE_DIM, DATASET_VOCAB, TARGET_AR_VOCAB


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.skip: nn.Module
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class ImageEncoder(nn.Module):
    def __init__(self, *, base_channels: int, token_dim: int) -> None:
        super().__init__()
        c = int(base_channels)
        self.net = nn.Sequential(
            nn.Conv2d(3, c, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            ConvBlock(c, c * 2, stride=2),
            ConvBlock(c * 2, c * 4, stride=2),
            ConvBlock(c * 4, c * 4, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c * 4, int(token_dim)),
            nn.LayerNorm(int(token_dim)),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class UniversalCropTeacherHConfig:
    candidate_feature_dim: int = CANDIDATE_FEATURE_DIM
    token_dim: int = 192
    base_channels: int = 48
    transformer_layers: int = 3
    transformer_heads: int = 6
    transformer_ff_mult: int = 4
    dropout: float = 0.10
    dataset_count: int = len(DATASET_VOCAB)
    target_ar_count: int = len(TARGET_AR_VOCAB)
    gate_bias_public: float = -2.0
    gate_bias_gaic: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UniversalCropTeacherHConfig":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{key: value for key, value in dict(payload).items() if key in allowed})


class UniversalCropTeacherH(nn.Module):
    """Heavy candidate-set ranker for crop-label curation.

    The model deliberately keeps global image context, per-crop visual context,
    geometry/source metadata, dataset conditioning, and target-AR conditioning
    separate until the cross-candidate transformer.
    """

    def __init__(self, config: UniversalCropTeacherHConfig | None = None) -> None:
        super().__init__()
        self.config = config or UniversalCropTeacherHConfig()
        d = int(self.config.token_dim)
        self.gaic_dataset_idx = DATASET_VOCAB.index("gaic")
        self.global_encoder = ImageEncoder(base_channels=int(self.config.base_channels), token_dim=d)
        self.crop_encoder = ImageEncoder(base_channels=int(self.config.base_channels), token_dim=d)
        self.feature_encoder = nn.Sequential(
            nn.Linear(int(self.config.candidate_feature_dim), d),
            nn.LayerNorm(d),
            nn.SiLU(),
            nn.Dropout(float(self.config.dropout)),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )
        self.dataset_embedding = nn.Embedding(int(self.config.dataset_count), d)
        self.target_ar_embedding = nn.Embedding(int(self.config.target_ar_count), d)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=int(self.config.transformer_heads),
            dim_feedforward=d * int(self.config.transformer_ff_mult),
            dropout=float(self.config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.relation = nn.TransformerEncoder(encoder_layer, num_layers=int(self.config.transformer_layers))
        self.token_norm = nn.LayerNorm(d)
        self.shared_utility_head = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(float(self.config.dropout)),
            nn.Linear(d, 1),
        )
        self.public_utility_head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        self.gaic_utility_head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        self.domain_gate_head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        self.domain_gate_bias = nn.Embedding(int(self.config.dataset_count), 1)
        self.dataset_scale = nn.Embedding(int(self.config.dataset_count), 1)
        self.dataset_bias = nn.Embedding(int(self.config.dataset_count), 1)
        self.aesthetic_head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        self.gt_iou_head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        self.safety_head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        self.uncertainty_head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        self.calibration_head = nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Linear(d // 2, 1))
        with torch.no_grad():
            self.domain_gate_bias.weight.fill_(float(self.config.gate_bias_public))
            self.domain_gate_bias.weight[self.gaic_dataset_idx].fill_(float(self.config.gate_bias_gaic))
            self.dataset_scale.weight.fill_(0.541324854612918)
            self.dataset_bias.weight.zero_()

    def forward(
        self,
        global_images: torch.Tensor,
        crop_images: torch.Tensor,
        candidate_features: torch.Tensor,
        valid_mask: torch.Tensor,
        dataset_ids: torch.Tensor,
        target_ar_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        bsz, max_k = candidate_features.shape[:2]
        global_token = self.global_encoder(global_images).unsqueeze(1)
        crop_flat = crop_images.view(bsz * max_k, *crop_images.shape[2:])
        crop_token = self.crop_encoder(crop_flat).view(bsz, max_k, -1)
        feature_token = self.feature_encoder(candidate_features)
        dataset_token = self.dataset_embedding(dataset_ids.clamp(0, self.config.dataset_count - 1)).unsqueeze(1)
        ar_token = self.target_ar_embedding(target_ar_ids.clamp(0, self.config.target_ar_count - 1)).unsqueeze(1)
        token = crop_token + feature_token + global_token + dataset_token + ar_token
        key_padding_mask = ~valid_mask.bool()
        token = self.relation(token, src_key_padding_mask=key_padding_mask)
        token = self.token_norm(token)
        shared_utility_logits = self.shared_utility_head(token).squeeze(-1)
        public_utility_logits = self.public_utility_head(token).squeeze(-1)
        gaic_utility_logits = self.gaic_utility_head(token).squeeze(-1)
        calibration_logits = self.calibration_head(token).squeeze(-1)
        gate_bias = self.domain_gate_bias(dataset_ids.clamp(0, self.config.dataset_count - 1)).unsqueeze(1)
        domain_gate_logits = self.domain_gate_head(token) + gate_bias
        domain_gate = torch.sigmoid(domain_gate_logits).squeeze(-1)
        dataset_scale = F.softplus(self.dataset_scale(dataset_ids.clamp(0, self.config.dataset_count - 1))).unsqueeze(1)
        dataset_bias = self.dataset_bias(dataset_ids.clamp(0, self.config.dataset_count - 1)).unsqueeze(1)
        utility_logits = (
            dataset_scale.squeeze(-1)
            * (
                shared_utility_logits
                + calibration_logits
                + (1.0 - domain_gate) * public_utility_logits
                + domain_gate * gaic_utility_logits
            )
            + dataset_bias.squeeze(-1)
        )
        aesthetic_logits = self.aesthetic_head(token).squeeze(-1)
        gt_iou_logits = self.gt_iou_head(token).squeeze(-1)
        safety_logits = self.safety_head(token).squeeze(-1)
        uncertainty_logits = self.uncertainty_head(token).squeeze(-1)
        neg_inf = torch.full_like(utility_logits, -1e9)
        return {
            "utility_logits": torch.where(valid_mask.bool(), utility_logits, neg_inf),
            "shared_utility_logits": torch.where(valid_mask.bool(), shared_utility_logits, neg_inf),
            "public_utility_logits": torch.where(valid_mask.bool(), public_utility_logits, neg_inf),
            "gaic_utility_logits": torch.where(valid_mask.bool(), gaic_utility_logits, neg_inf),
            "aesthetic_logits": torch.where(valid_mask.bool(), aesthetic_logits, neg_inf),
            "gt_iou_logits": torch.where(valid_mask.bool(), gt_iou_logits, neg_inf),
            "safety_logits": torch.where(valid_mask.bool(), safety_logits, neg_inf),
            "uncertainty_logits": torch.where(valid_mask.bool(), uncertainty_logits, neg_inf),
            "calibration_logits": torch.where(valid_mask.bool(), calibration_logits, neg_inf),
            "domain_gate_logits": torch.where(valid_mask.bool(), domain_gate_logits.squeeze(-1), neg_inf),
            "domain_gate": torch.where(valid_mask.bool(), domain_gate, torch.zeros_like(domain_gate)),
            "dataset_scale": dataset_scale.squeeze(-1),
            "dataset_bias": dataset_bias.squeeze(-1),
            "tokens": token,
        }
