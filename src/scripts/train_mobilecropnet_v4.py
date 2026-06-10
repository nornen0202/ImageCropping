#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from time import time
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet_v4.data import (
    DECISION_VOCAB,
    MobileCropNetV4BatchDataset,
    SUBJECT_BOX_TARGET_SOURCES,
    SUBJECT_BOX_VALID_TARGET_MODES,
    SUBJECT_MODE_COUNT,
    decision_id,
    mobilecropnet_v4_collate,
)
from mobilecropnet_v4.model import MobileCropNetV4, compute_mobilecropnet_v4_loss, model_config_to_dict
from mobilecropnet_v4.route_expert import (
    load_route_expert_checkpoint,
    route_expert_ensemble_logits_from_paths,
    route_expert_logits_from_paths,
)


def _split_cli_list(value: str | None) -> tuple[str, ...]:
    """Split comma and whitespace separated CLI list values."""
    return tuple(part for part in re.split(r"[\s,]+", str(value or "").strip()) if part)


IMAGENET_DEFAULT_MEAN = [0.485, 0.456, 0.406]
IMAGENET_DEFAULT_STD = [0.229, 0.224, 0.225]
EVA02_DEFAULT_MEAN = [0.48145466, 0.4578275, 0.40821073]
EVA02_DEFAULT_STD = [0.26862954, 0.26130258, 0.27577711]


MODEL_PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "q24_288_w075": {
        "input_size": 288,
        "candidate_k": 24,
        "proposal_q": 24,
        "width_mult": 0.75,
        "token_dim": 128,
        "backbone_name": "custom_depthwise",
        "backbone_pretrained": False,
        "ranker_type": "relation_lite",
        "ranker_depth": 1,
        "image_mean": IMAGENET_DEFAULT_MEAN,
        "image_std": IMAGENET_DEFAULT_STD,
    },
    "hq_320": {
        "input_size": 320,
        "candidate_k": 32,
        "proposal_q": 32,
        "width_mult": 1.0,
        "token_dim": 192,
        "backbone_name": "mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k",
        "backbone_pretrained": True,
        "ranker_type": "set_transformer",
        "ranker_depth": 2,
        "image_mean": IMAGENET_DEFAULT_MEAN,
        "image_std": IMAGENET_DEFAULT_STD,
    },
    "balanced_288": {
        "input_size": 288,
        "candidate_k": 24,
        "proposal_q": 24,
        "width_mult": 0.75,
        "token_dim": 128,
        "backbone_name": "mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k",
        "backbone_pretrained": True,
        "ranker_type": "relation_lite",
        "ranker_depth": 1,
        "image_mean": IMAGENET_DEFAULT_MEAN,
        "image_std": IMAGENET_DEFAULT_STD,
    },
    "turbo_256": {
        "input_size": 256,
        "candidate_k": 16,
        "proposal_q": 16,
        "width_mult": 0.75,
        "token_dim": 128,
        "backbone_name": "mobilenetv4_conv_small.e3600_r256_in1k",
        "backbone_pretrained": True,
        "ranker_type": "relation_lite",
        "ranker_depth": 1,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    },
    "rank_320": {
        "input_size": 320,
        "candidate_k": 32,
        "proposal_q": 32,
        "width_mult": 0.75,
        "token_dim": 192,
        "backbone_name": "mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k",
        "backbone_pretrained": True,
        "ranker_type": "set_transformer",
        "ranker_depth": 2,
        "image_mean": IMAGENET_DEFAULT_MEAN,
        "image_std": IMAGENET_DEFAULT_STD,
    },
    "prod_rank_288": {
        "input_size": 288,
        "candidate_k": 32,
        "proposal_q": 32,
        "width_mult": 0.85,
        "token_dim": 160,
        "backbone_name": "mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k",
        "backbone_pretrained": True,
        "ranker_type": "set_transformer",
        "ranker_depth": 2,
        "image_mean": IMAGENET_DEFAULT_MEAN,
        "image_std": IMAGENET_DEFAULT_STD,
    },
    "plus_384": {
        "input_size": 384,
        "candidate_k": 48,
        "proposal_q": 48,
        "width_mult": 1.0,
        "token_dim": 256,
        "backbone_name": "convnextv2_tiny.fcmae_ft_in22k_in1k_384",
        "backbone_pretrained": True,
        "ranker_type": "set_transformer",
        "ranker_depth": 3,
        "image_mean": IMAGENET_DEFAULT_MEAN,
        "image_std": IMAGENET_DEFAULT_STD,
    },
    "hybrid_384": {
        "input_size": 384,
        "candidate_k": 48,
        "proposal_q": 48,
        "width_mult": 1.0,
        "token_dim": 256,
        "backbone_name": "mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k",
        "backbone_pretrained": True,
        "ranker_type": "set_transformer",
        "ranker_depth": 3,
        "image_mean": IMAGENET_DEFAULT_MEAN,
        "image_std": IMAGENET_DEFAULT_STD,
    },
    "quality_cnv2b_448": {
        "input_size": 448,
        "candidate_k": 64,
        "proposal_q": 64,
        "width_mult": 1.0,
        "token_dim": 384,
        "backbone_name": "convnextv2_base.fcmae_ft_in22k_in1k_384",
        "backbone_pretrained": True,
        "ranker_type": "set_transformer",
        "ranker_depth": 4,
        "image_mean": IMAGENET_DEFAULT_MEAN,
        "image_std": IMAGENET_DEFAULT_STD,
    },
    "quality_cnv2l_512": {
        "input_size": 512,
        "candidate_k": 80,
        "proposal_q": 80,
        "width_mult": 1.0,
        "token_dim": 512,
        "backbone_name": "convnextv2_large.fcmae_ft_in22k_in1k_384",
        "backbone_pretrained": True,
        "ranker_type": "set_transformer",
        "ranker_depth": 4,
        "image_mean": IMAGENET_DEFAULT_MEAN,
        "image_std": IMAGENET_DEFAULT_STD,
    },
    "quality_cnv2h_512": {
        "input_size": 512,
        "candidate_k": 96,
        "proposal_q": 96,
        "width_mult": 1.0,
        "token_dim": 768,
        "backbone_name": "convnextv2_huge.fcmae_ft_in22k_in1k_512",
        "backbone_pretrained": True,
        "ranker_type": "set_transformer",
        "ranker_depth": 4,
        "image_mean": IMAGENET_DEFAULT_MEAN,
        "image_std": IMAGENET_DEFAULT_STD,
    },
    "quality_clip_cnvb_384": {
        "input_size": 384,
        "candidate_k": 64,
        "proposal_q": 64,
        "width_mult": 1.0,
        "token_dim": 512,
        "backbone_name": "convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384",
        "backbone_pretrained": True,
        "ranker_type": "set_transformer",
        "ranker_depth": 4,
    },
    "quality_siglipb_384": {
        "input_size": 384,
        "candidate_k": 64,
        "proposal_q": 64,
        "width_mult": 1.0,
        "token_dim": 512,
        "backbone_name": "vit_base_patch16_siglip_384.webli",
        "backbone_pretrained": True,
        "ranker_type": "set_transformer",
        "ranker_depth": 4,
    },
    "quality_eva02b_448": {
        "input_size": 448,
        "candidate_k": 64,
        "proposal_q": 64,
        "width_mult": 1.0,
        "token_dim": 512,
        "backbone_name": "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
        "backbone_pretrained": True,
        "ranker_type": "set_transformer",
        "ranker_depth": 4,
        "image_mean": EVA02_DEFAULT_MEAN,
        "image_std": EVA02_DEFAULT_STD,
    },
    "quality_eva02l_448": {
        "input_size": 448,
        "candidate_k": 80,
        "proposal_q": 80,
        "width_mult": 1.0,
        "token_dim": 768,
        "backbone_name": "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        "backbone_pretrained": True,
        "ranker_type": "set_transformer",
        "ranker_depth": 4,
        "image_mean": EVA02_DEFAULT_MEAN,
        "image_std": EVA02_DEFAULT_STD,
    },
    "quality_eva02b_subject_448": {
        "input_size": 448,
        "candidate_k": 16,
        "proposal_q": 64,
        "width_mult": 1.0,
        "token_dim": 512,
        "backbone_name": "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
        "backbone_pretrained": True,
        "ranker_type": "set_transformer",
        "ranker_depth": 1,
        "image_mean": EVA02_DEFAULT_MEAN,
        "image_std": EVA02_DEFAULT_STD,
    },
    "quality_eva02l_subject_448": {
        "input_size": 448,
        "candidate_k": 16,
        "proposal_q": 80,
        "width_mult": 1.0,
        "token_dim": 768,
        "backbone_name": "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        "backbone_pretrained": True,
        "ranker_type": "set_transformer",
        "ranker_depth": 1,
        "image_mean": EVA02_DEFAULT_MEAN,
        "image_std": EVA02_DEFAULT_STD,
    },
    "quality_swinv2b_384": {
        "input_size": 384,
        "candidate_k": 64,
        "proposal_q": 64,
        "width_mult": 1.0,
        "token_dim": 512,
        "backbone_name": "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
        "backbone_pretrained": True,
        "ranker_type": "set_transformer",
        "ranker_depth": 4,
        "image_mean": IMAGENET_DEFAULT_MEAN,
        "image_std": IMAGENET_DEFAULT_STD,
    },
    "quality_swinv2l_384": {
        "input_size": 384,
        "candidate_k": 80,
        "proposal_q": 80,
        "width_mult": 1.0,
        "token_dim": 768,
        "backbone_name": "swinv2_large_window12to24_192to384.ms_in22k_ft_in1k",
        "backbone_pretrained": True,
        "ranker_type": "set_transformer",
        "ranker_depth": 4,
        "image_mean": IMAGENET_DEFAULT_MEAN,
        "image_std": IMAGENET_DEFAULT_STD,
    },
    "quality_swinv2l_512": {
        "input_size": 512,
        "candidate_k": 80,
        "proposal_q": 80,
        "width_mult": 1.0,
        "token_dim": 768,
        "backbone_name": "swinv2_large_window12to24_192to384.ms_in22k_ft_in1k",
        "backbone_pretrained": True,
        "ranker_type": "set_transformer",
        "ranker_depth": 4,
        "image_mean": IMAGENET_DEFAULT_MEAN,
        "image_std": IMAGENET_DEFAULT_STD,
    },
    "quality_swinv2b_subject_384": {
        "input_size": 384,
        "candidate_k": 16,
        "proposal_q": 64,
        "width_mult": 1.0,
        "token_dim": 512,
        "backbone_name": "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
        "backbone_pretrained": True,
        "ranker_type": "set_transformer",
        "ranker_depth": 1,
        "image_mean": IMAGENET_DEFAULT_MEAN,
        "image_std": IMAGENET_DEFAULT_STD,
    },
    "quality_dinov2b_subject_518": {
        "input_size": 518,
        "candidate_k": 16,
        "proposal_q": 64,
        "width_mult": 1.0,
        "token_dim": 512,
        "backbone_name": "vit_base_patch14_dinov2.lvd142m",
        "backbone_pretrained": True,
        "ranker_type": "set_transformer",
        "ranker_depth": 1,
        "image_mean": IMAGENET_DEFAULT_MEAN,
        "image_std": IMAGENET_DEFAULT_STD,
    },
    "quality_dinov2b_518": {
        "input_size": 518,
        "candidate_k": 64,
        "proposal_q": 64,
        "width_mult": 1.0,
        "token_dim": 512,
        "backbone_name": "vit_base_patch14_dinov2.lvd142m",
        "backbone_pretrained": True,
        "ranker_type": "set_transformer",
        "ranker_depth": 4,
        "image_mean": IMAGENET_DEFAULT_MEAN,
        "image_std": IMAGENET_DEFAULT_STD,
    },
    "quality_dinov2l_subject_518": {
        "input_size": 518,
        "candidate_k": 16,
        "proposal_q": 64,
        "width_mult": 1.0,
        "token_dim": 768,
        "backbone_name": "vit_large_patch14_dinov2.lvd142m",
        "backbone_pretrained": True,
        "ranker_type": "set_transformer",
        "ranker_depth": 1,
        "image_mean": IMAGENET_DEFAULT_MEAN,
        "image_std": IMAGENET_DEFAULT_STD,
    },
    "quality_dinov2l_518": {
        "input_size": 518,
        "candidate_k": 80,
        "proposal_q": 80,
        "width_mult": 1.0,
        "token_dim": 768,
        "backbone_name": "vit_large_patch14_dinov2.lvd142m",
        "backbone_pretrained": True,
        "ranker_type": "set_transformer",
        "ranker_depth": 4,
        "image_mean": IMAGENET_DEFAULT_MEAN,
        "image_std": IMAGENET_DEFAULT_STD,
    },
}
PROFILE_CONTROLLED_ARGS = set(next(iter(MODEL_PROFILE_PRESETS.values())).keys())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MobileCropNet v4 on current SSTK/GAIC batch JSONL labels.")
    parser.add_argument("--train_jsonl", required=True, type=Path)
    parser.add_argument("--val_jsonl", required=True, type=Path)
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--image_root", type=Path, default=None)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--model_profile", choices=[*MODEL_PROFILE_PRESETS.keys(), "none"], default="hq_320")
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--candidate_k", type=int, default=24)
    parser.add_argument("--proposal_q", type=int, default=16)
    parser.add_argument("--width_mult", type=float, default=0.75)
    parser.add_argument("--token_dim", type=int, default=128)
    parser.add_argument("--backbone_name", default="custom_depthwise")
    parser.add_argument("--backbone_pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image_mean", default=None, help="Comma-separated RGB mean. Defaults to the selected profile or backbone pretrained_cfg.")
    parser.add_argument("--image_std", default=None, help="Comma-separated RGB std. Defaults to the selected profile or backbone pretrained_cfg.")
    parser.add_argument("--ranker_type", choices=["relation_lite", "set_transformer"], default="relation_lite")
    parser.add_argument("--ranker_depth", type=int, default=1)
    parser.add_argument("--use_subject_prior", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--route_image_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--route_use_subject_prior", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--route_use_candidate_context", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--route_use_subject_box_features", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--route_use_subject_spatial_token", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--route_head_depth", type=int, default=1)
    parser.add_argument("--route_head_hidden_mult", type=float, default=0.5)
    parser.add_argument("--route_head_dropout", type=float, default=0.0)
    parser.add_argument("--route_aux_heads", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--route_image_residual", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--route_image_residual_weight", type=float, default=1.0)
    parser.add_argument("--route_decode_mode", choices=["fine", "hierarchical"], default="fine")
    parser.add_argument("--route_decode_kind_weight", type=float, default=1.0)
    parser.add_argument("--route_decode_cardinality_weight", type=float, default=1.0)
    parser.add_argument("--policy_use_subject_prior", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--proposal_use_subject_prior", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--subject_box_head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--subject_box_spatial_head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--subject_box_spatial_multiscale", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--subject_box_spatial_mix_bias", type=float, default=-1.0)
    parser.add_argument("--subject_box_spatial_output", choices=["mix", "spatial", "coarse"], default="mix")
    parser.add_argument(
        "--subject_box_spatial_box_mode",
        choices=["regress", "mask_moment", "mask_moment_regress", "heatmap_moment", "heatmap_moment_regress"],
        default="regress",
    )
    parser.add_argument("--subject_box_spatial_extent_scale", type=float, default=1.0)
    parser.add_argument("--subject_box_coord_space", choices=["letterbox", "content"], default="letterbox")
    parser.add_argument("--subject_box_refine_with_proposals", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--subject_valid_route_calibrator", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--subject_valid_route_calibrator_hidden_mult", type=float, default=0.5)
    parser.add_argument("--subject_valid_route_calibrator_detach", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use_pred_subject_box_as_prior", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--detach_pred_subject_prior", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pred_subject_prior_start_epoch", type=int, default=1)
    parser.add_argument("--pred_subject_prior_warmup_epochs", type=int, default=0)
    parser.add_argument("--policy_score_head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--policy_score_threshold_low", type=float, default=0.35)
    parser.add_argument("--policy_score_threshold_high", type=float, default=0.70)
    parser.add_argument("--decision_source_head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--decision_source_logit_threshold", type=float, default=0.0)
    parser.add_argument("--decision_source_pair_head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--decision_source_pair_hidden_mult", type=float, default=0.5)
    parser.add_argument("--decision_source_pair_after_return_deltas", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--decision_source_pair_subject_state", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--decision_source_pair_subject_detach", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--route_condition_candidate_scores", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--route_condition_detach", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--return_score_head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--return_score_head_depth", type=int, default=1)
    parser.add_argument("--return_score_head_hidden_mult", type=float, default=0.5)
    parser.add_argument("--return_score_action_source_bias", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--return_score_decision_source_bias", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--return_score_decision_source_detach", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--return_score_decision_source_max_scale", type=float, default=2.0)
    parser.add_argument("--return_score_decision_source_from_source_head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--return_score_decision_conditioned", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--return_score_decision_conditioned_mask_value", type=float, default=-10000.0)
    parser.add_argument("--return_score_decision_conditioned_from_source_head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--return_score_source_mixture", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--return_score_source_mixture_strength", type=float, default=1.0)
    parser.add_argument("--return_score_source_mixture_detach", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--return_score_source_mixture_from_source_head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--return_score_source_mixture_gap_threshold", type=float, default=None)
    parser.add_argument("--return_score_source_mixture_gap_mask_value", type=float, default=-10000.0)
    parser.add_argument("--return_score_source_gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--return_score_source_gate_hidden_mult", type=float, default=0.5)
    parser.add_argument("--return_score_source_gate_strength", type=float, default=1.0)
    parser.add_argument("--return_score_source_gate_detach", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--return_score_source_gate_subject_state", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--return_score_source_gate_subject_detach", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--return_score_source_gate_logit_threshold", type=float, default=0.0)
    parser.add_argument("--return_score_source_specific_head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--return_score_source_specific_hidden_mult", type=float, default=0.5)
    parser.add_argument("--return_score_context_head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--return_score_context_hidden_mult", type=float, default=0.5)
    parser.add_argument("--return_score_context_detach", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--return_score_policy_match_head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--return_score_policy_match_hidden_mult", type=float, default=0.5)
    parser.add_argument("--return_score_policy_match_detach", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--return_score_policy_source_head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--return_score_policy_source_hidden_mult", type=float, default=0.5)
    parser.add_argument("--return_score_policy_source_detach", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--return_score_competition_head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--return_score_competition_hidden_mult", type=float, default=0.5)
    parser.add_argument("--return_score_competition_detach", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--return_score_set_refiner_head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--return_score_set_refiner_hidden_mult", type=float, default=1.0)
    parser.add_argument("--return_score_set_refiner_layers", type=int, default=1)
    parser.add_argument("--return_score_set_refiner_detach", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--return_score_action_decoder_head", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--return_score_action_decoder_hidden_mult", type=float, default=1.0)
    parser.add_argument("--return_score_action_decoder_layers", type=int, default=1)
    parser.add_argument("--return_score_action_decoder_detach", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--return_score_action_decoder_subject_state", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--return_score_action_decoder_subject_detach", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--return_score_action_decoder_replace", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=None)
    parser.add_argument("--pin_memory_device", default="")
    parser.add_argument("--train_route_balanced_sampler", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train_route_balanced_sampler_power", type=float, default=1.0)
    parser.add_argument("--train_route_balanced_sampler_min_weight", type=float, default=0.25)
    parser.add_argument("--train_route_balanced_sampler_max_weight", type=float, default=4.0)
    parser.add_argument("--train_decision_source_balanced_sampler", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train_decision_source_balanced_sampler_power", type=float, default=1.0)
    parser.add_argument("--train_decision_source_balanced_sampler_min_weight", type=float, default=0.25)
    parser.add_argument("--train_decision_source_balanced_sampler_max_weight", type=float, default=4.0)
    parser.add_argument("--precompute_sample_tensors", action="store_true")
    parser.add_argument("--image_tensor_cache_size", type=int, default=0)
    parser.add_argument("--pairwise_jsonl", type=Path, default=None)
    parser.add_argument("--listwise_jsonl", type=Path, default=None)
    parser.add_argument("--val_pairwise_jsonl", type=Path, default=None)
    parser.add_argument("--val_listwise_jsonl", type=Path, default=None)
    parser.add_argument("--max_pairwise_pairs", type=int, default=32)
    parser.add_argument("--score_target_mode", default="default")
    parser.add_argument("--subject_box_target_source", choices=SUBJECT_BOX_TARGET_SOURCES, default="legacy")
    parser.add_argument("--subject_box_valid_target_mode", choices=SUBJECT_BOX_VALID_TARGET_MODES, default="route_gated")
    parser.add_argument("--subject_box_valid_reliability_min", type=float, default=0.25)
    parser.add_argument("--include_ignored_candidates", action="store_true")
    parser.add_argument("--include_overflow_candidates", action="store_true")
    parser.add_argument("--high_score_safe_positive_threshold", type=float, default=0.72)
    parser.add_argument("--score_weight", type=float, default=1.0)
    parser.add_argument("--listwise_weight", type=float, default=0.6)
    parser.add_argument("--pairwise_weight", type=float, default=0.5)
    parser.add_argument("--explicit_pairwise_weight", type=float, default=0.3)
    parser.add_argument("--positive_weight", type=float, default=0.4)
    parser.add_argument("--risk_weight", type=float, default=0.2)
    parser.add_argument("--top1_risk_weight", type=float, default=0.2)
    parser.add_argument("--top1_risk_margin", type=float, default=0.2)
    parser.add_argument("--top_return_weight", type=float, default=0.0)
    parser.add_argument("--raw_top_return_weight", type=float, default=0.0)
    parser.add_argument("--top_return_score_margin", type=float, default=0.03)
    parser.add_argument("--top_return_logit_margin", type=float, default=0.25)
    parser.add_argument("--top_return_temperature", type=float, default=0.12)
    parser.add_argument("--return_score_weight", type=float, default=0.0)
    parser.add_argument("--return_listwise_weight", type=float, default=0.0)
    parser.add_argument("--return_positive_weight", type=float, default=0.0)
    parser.add_argument("--return_positive_margin_weight", type=float, default=0.0)
    parser.add_argument("--return_positive_margin", type=float, default=0.2)
    parser.add_argument("--return_risk_suppression_weight", type=float, default=0.0)
    parser.add_argument("--return_risk_suppression_margin", type=float, default=0.2)
    parser.add_argument("--return_exact_weight", type=float, default=0.0)
    parser.add_argument("--return_exact_min_gap", type=float, default=0.0)
    parser.add_argument("--return_target_mode", default="score", choices=["score", "decision_source", "decision_consistent", "action_source"])
    parser.add_argument("--return_explicit_pairwise_weight", type=float, default=0.0)
    parser.add_argument("--return_explicit_pairwise_margin", type=float, default=0.0)
    parser.add_argument("--macro_weight", type=float, default=0.1)
    parser.add_argument("--checklist_class_weight", type=float, default=0.08)
    parser.add_argument("--checklist_applicability_weight", type=float, default=0.04)
    parser.add_argument("--detail_score_weight", type=float, default=0.05)
    parser.add_argument("--why_tag_weight", type=float, default=0.04)
    parser.add_argument("--route_weight", type=float, default=0.1)
    parser.add_argument("--route_balanced_ce", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--route_focal_gamma", type=float, default=0.0)
    parser.add_argument("--route_object_single_margin_weight", type=float, default=0.0)
    parser.add_argument("--route_object_single_margin", type=float, default=0.18)
    parser.add_argument("--route_hierarchy_weight", type=float, default=0.0)
    parser.add_argument("--route_cardinality_weight", type=float, default=0.0)
    parser.add_argument("--route_expert_distill_checkpoint", type=Path, default=None)
    parser.add_argument("--route_expert_distill_checkpoints", type=Path, nargs="*", default=None)
    parser.add_argument("--route_expert_distill_weights", default=None, help="Comma-separated weights for route expert ensemble distillation.")
    parser.add_argument("--route_expert_distill_weight", type=float, default=0.0)
    parser.add_argument("--route_expert_distill_temperature", type=float, default=1.0)
    parser.add_argument("--route_expert_distill_hard_weight", type=float, default=0.0)
    parser.add_argument("--decision_weight", type=float, default=0.25)
    parser.add_argument("--decision_crop_rebalance", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--decision_crop_rebalance_max_weight", type=float, default=4.0)
    parser.add_argument("--decision_source_weight", type=float, default=0.0)
    parser.add_argument("--decision_source_balanced_bce", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--decision_source_crop_rebalance_max_weight", type=float, default=4.0)
    parser.add_argument("--decision_source_focal_gamma", type=float, default=0.0)
    parser.add_argument("--decision_source_focal_alpha", type=float, default=None)
    parser.add_argument("--decision_source_margin_weight", type=float, default=0.0)
    parser.add_argument("--decision_source_logit_margin", type=float, default=0.35)
    parser.add_argument("--decision_source_margin_balanced", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--decision_source_margin_balance_max_weight", type=float, default=4.0)
    parser.add_argument("--decision_source_margin_crop_weight_mult", type=float, default=1.0)
    parser.add_argument("--decision_source_pair_supervision_weight", type=float, default=0.0)
    parser.add_argument("--decision_source_pair_margin_weight", type=float, default=0.0)
    parser.add_argument("--decision_source_pair_balanced_bce", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--decision_source_pair_focal_gamma", type=float, default=0.0)
    parser.add_argument("--decision_source_pair_focal_alpha", type=float, default=None)
    parser.add_argument("--source_gate_supervision_weight", type=float, default=0.0)
    parser.add_argument("--source_gate_balanced_bce", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--source_gate_crop_rebalance_max_weight", type=float, default=4.0)
    parser.add_argument("--source_gate_focal_gamma", type=float, default=0.0)
    parser.add_argument("--source_gate_focal_alpha", type=float, default=None)
    parser.add_argument("--source_gate_margin_weight", type=float, default=0.0)
    parser.add_argument("--source_gate_logit_margin", type=float, default=0.35)
    parser.add_argument("--source_gate_margin_balanced", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--source_gate_margin_balance_max_weight", type=float, default=4.0)
    parser.add_argument("--source_gate_margin_crop_weight_mult", type=float, default=1.0)
    parser.add_argument("--action_consistency_weight", type=float, default=0.0)
    parser.add_argument("--action_consistency_margin", type=float, default=0.15)
    parser.add_argument("--action_source_balanced", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--action_source_balance_max_weight", type=float, default=4.0)
    parser.add_argument("--action_source_crop_weight_mult", type=float, default=1.0)
    parser.add_argument("--action_return_joint_weight", type=float, default=0.0)
    parser.add_argument("--action_return_joint_score_margin", type=float, default=0.03)
    parser.add_argument("--action_return_joint_logit_margin", type=float, default=0.20)
    parser.add_argument("--action_return_joint_temperature", type=float, default=0.12)
    parser.add_argument("--return_source_margin_weight", type=float, default=0.0)
    parser.add_argument("--return_source_margin_logit_margin", type=float, default=0.25)
    parser.add_argument("--return_source_margin_balanced", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--return_source_margin_balance_max_weight", type=float, default=4.0)
    parser.add_argument("--return_source_margin_crop_weight_mult", type=float, default=1.0)
    parser.add_argument("--decision_utility_align_weight", type=float, default=0.0)
    parser.add_argument("--decision_utility_align_temperature", type=float, default=0.35)
    parser.add_argument("--policy_score_weight", type=float, default=0.0)
    parser.add_argument("--delta_weight", type=float, default=0.1)
    parser.add_argument("--proposal_weight", type=float, default=0.8)
    parser.add_argument("--proposal_subject_weight", type=float, default=0.0)
    parser.add_argument("--subject_proposal_align_weight", type=float, default=0.0)
    parser.add_argument("--subject_box_weight", type=float, default=0.0)
    parser.add_argument("--subject_box_valid_weight", type=float, default=1.0)
    parser.add_argument("--subject_box_l1_weight", type=float, default=2.0)
    parser.add_argument("--subject_box_iou_weight", type=float, default=1.0)
    parser.add_argument("--subject_box_center_weight", type=float, default=0.0)
    parser.add_argument("--subject_box_size_weight", type=float, default=0.0)
    parser.add_argument("--subject_box_aspect_weight", type=float, default=0.0)
    parser.add_argument("--subject_box_ciou_weight", type=float, default=0.0)
    parser.add_argument("--subject_box_spatial_aux_weight", type=float, default=0.0)
    parser.add_argument("--subject_box_spatial_heatmap_weight", type=float, default=0.0)
    parser.add_argument("--subject_box_spatial_mask_weight", type=float, default=0.0)
    parser.add_argument("--subject_box_valid_balanced_bce", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--subject_box_valid_negative_scale", type=float, default=1.0)
    parser.add_argument("--generated_proposal_align_weight", type=float, default=0.0)
    parser.add_argument("--generated_proposal_score_weight", type=float, default=1.0)
    parser.add_argument("--generated_proposal_listwise_weight", type=float, default=0.5)
    parser.add_argument("--generated_proposal_positive_weight", type=float, default=0.3)
    parser.add_argument("--generated_proposal_risk_weight", type=float, default=0.2)
    parser.add_argument("--generated_proposal_positive_utility_weight", type=float, default=0.0)
    parser.add_argument("--generated_proposal_positive_margin_weight", type=float, default=0.0)
    parser.add_argument("--generated_proposal_positive_margin", type=float, default=0.25)
    parser.add_argument("--generated_proposal_match_iou", type=float, default=0.35)
    parser.add_argument(
        "--teacher_distill_weight",
        type=float,
        default=0.0,
        help="Weight for explicit teacher soft distribution distillation from listwise labels.",
    )
    parser.add_argument("--teacher_distill_temperature", type=float, default=0.12)
    parser.add_argument("--topk_coverage_weight", type=float, default=0.0)
    parser.add_argument("--topk_coverage_k", type=int, default=4)
    parser.add_argument("--topk_coverage_temperature", type=float, default=0.12)
    parser.add_argument(
        "--selection_metric",
        choices=[
            "legacy",
            "sstk_composite",
            "sstk_topreturn",
            "sstk_balanced_topreturn",
            "public_score_distill",
            "deploy_align_topreturn",
            "release_gate",
            "release_gate_joint",
            "subject_box",
            "route_only",
            "route_balanced",
        ],
        default="sstk_composite",
    )
    parser.add_argument("--max_train_rows", type=int, default=None)
    parser.add_argument("--max_val_rows", type=int, default=None)
    parser.add_argument("--limit_train_steps", type=int, default=None)
    parser.add_argument("--limit_val_steps", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--init_checkpoint", type=Path, default=None)
    parser.add_argument(
        "--init_checkpoint_skip_prefixes",
        default="",
        help="Comma- or whitespace-separated state_dict prefixes to skip when warm-starting from --init_checkpoint.",
    )
    parser.add_argument("--freeze_backbone", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--freeze_condition_encoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--freeze_subject_box_heads", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--freeze_subject_box_stack",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freeze backbone, condition encoder, and subject-box heads to preserve an init checkpoint subject prior.",
    )
    parser.add_argument(
        "--trainable_module_prefixes",
        default="",
        help=(
            "Optional comma-separated parameter/module prefixes to keep trainable. "
            "When set, all other parameters are frozen after the explicit freeze options."
        ),
    )
    parser.add_argument("--gpu_usage_sample_interval", type=int, default=10)
    parser.add_argument("--progress_log_interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260415)
    return parser


def _move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return out


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    out: dict[str, float] = {}
    for key in keys:
        vals = []
        for row in rows:
            value = float(row.get(key, 0.0))
            if math.isfinite(value):
                vals.append(value)
        out[key] = float(sum(vals) / len(vals)) if vals else float("nan")
    return out


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _query_nvidia_smi(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}
    gpu_id = 0 if device.index is None else int(device.index)
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_id}",
                "--query-gpu=utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return {}
    if not raw:
        return {}
    parts = [part.strip() for part in raw.splitlines()[0].split(",")]
    if len(parts) < 3:
        return {}
    try:
        return {
            "nvidia_gpu_util_pct": float(parts[0]),
            "nvidia_mem_used_mb": float(parts[1]),
            "nvidia_power_w": float(parts[2]),
        }
    except ValueError:
        return {}


def _cuda_peak_metrics(device: torch.device) -> dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}
    dev = device if device.index is not None else torch.device("cuda:0")
    return {
        "cuda_peak_allocated_mb": float(torch.cuda.max_memory_allocated(dev)) / 1024.0 / 1024.0,
        "cuda_peak_reserved_mb": float(torch.cuda.max_memory_reserved(dev)) / 1024.0 / 1024.0,
    }


def _summarize_gpu_samples(samples: list[dict[str, float]], device: torch.device) -> dict[str, float]:
    metrics = _cuda_peak_metrics(device)
    if not samples:
        return metrics
    for key in ("nvidia_gpu_util_pct", "nvidia_mem_used_mb", "nvidia_power_w"):
        vals = [float(sample[key]) for sample in samples if key in sample]
        if vals:
            metrics[f"{key}_mean"] = float(sum(vals) / len(vals))
            metrics[f"{key}_max"] = float(max(vals))
    metrics["gpu_usage_sample_count"] = float(len(samples))
    return metrics


def _explicit_arg_names(argv: list[str] | None) -> set[str]:
    raw = list(sys.argv[1:] if argv is None else argv)
    names: set[str] = set()
    for token in raw:
        if not token.startswith("--"):
            continue
        name = token[2:].split("=", 1)[0].replace("-", "_")
        if name:
            names.add(name)
            if name.startswith("no_"):
                names.add(name[3:])
    return names


def _apply_model_profile(args: argparse.Namespace, explicit_args: set[str]) -> None:
    profile = str(args.model_profile)
    args.model_profile_effective = profile
    if profile == "none":
        return
    preset = MODEL_PROFILE_PRESETS[profile]
    profile_explicit = "model_profile" in explicit_args
    controlled_explicit = PROFILE_CONTROLLED_ARGS.intersection(explicit_args)
    if not profile_explicit and controlled_explicit:
        args.model_profile_effective = f"custom_explicit_from_{profile}"
        return
    for key, value in preset.items():
        if profile_explicit and key in explicit_args:
            continue
        setattr(args, key, value)


def _parse_norm_triplet(value: Any, *, default: list[float] | None = None) -> list[float] | None:
    if value is None:
        return list(default) if default is not None else None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return list(default) if default is not None else None
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


def _resolve_image_norm_args(args: argparse.Namespace) -> None:
    cfg_mean, cfg_std = _backbone_pretrained_norm(str(args.backbone_name))
    args.image_mean = _parse_norm_triplet(args.image_mean, default=cfg_mean or IMAGENET_DEFAULT_MEAN)
    args.image_std = _parse_norm_triplet(args.image_std, default=cfg_std or IMAGENET_DEFAULT_STD)


def resolve_epoch_lr(*, base_lr: float, min_lr: float, scheduler: str, warmup_epochs: int, epoch: int, epochs: int) -> float:
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return float(base_lr) * float(epoch) / float(max(1, warmup_epochs))
    if scheduler != "cosine":
        return float(base_lr)
    decay_epochs = max(1, int(epochs) - max(0, int(warmup_epochs)))
    decay_index = min(decay_epochs, max(0, int(epoch) - max(0, int(warmup_epochs)) - 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * float(decay_index) / float(decay_epochs)))
    return float(min_lr) + (float(base_lr) - float(min_lr)) * cosine


def resolve_pred_subject_prior_alpha(*, epoch: int, training: bool, start_epoch: int, warmup_epochs: int) -> float:
    if not training:
        return 1.0
    if epoch < int(start_epoch):
        return 0.0
    if int(warmup_epochs) <= 0:
        return 1.0
    progress = float(epoch - int(start_epoch) + 1) / float(max(1, int(warmup_epochs)))
    return max(0.0, min(1.0, progress))


def run_epoch(
    *,
    model: MobileCropNetV4,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    amp: bool,
    limit_steps: int | None,
    grad_clip_norm: float,
    loss_kwargs: dict[str, Any],
    phase: str = "train",
    epoch: int = 0,
    progress_log_interval: int = 0,
    pred_subject_prior_start_epoch: int = 1,
    pred_subject_prior_warmup_epochs: int = 0,
    route_expert_distill_models: list[tuple[torch.nn.Module, dict[str, Any]]] | None = None,
    route_expert_distill_weights: list[float] | None = None,
    route_expert_distill_cache: dict[str, torch.Tensor] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if route_expert_distill_models:
        for route_expert_distill_model, _route_expert_distill_config in route_expert_distill_models:
            route_expert_distill_model.eval()
    rows: list[dict[str, float]] = []
    gpu_samples: list[dict[str, float]] = []
    data_wait_sec = 0.0
    compute_sec = 0.0
    sample_count = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device if device.index is not None else None)
    iterator = iter(loader)
    step = 0
    loop_start = time()
    return_generated_scores = float(loss_kwargs.get("generated_proposal_align_weight", 0.0)) > 0.0
    pred_subject_prior_alpha = resolve_pred_subject_prior_alpha(
        epoch=epoch,
        training=training,
        start_epoch=pred_subject_prior_start_epoch,
        warmup_epochs=pred_subject_prior_warmup_epochs,
    )
    while True:
        if limit_steps is not None and step >= int(limit_steps):
            break
        fetch_start = time()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        data_wait_sec += time() - fetch_start
        step += 1
        compute_start = time()
        batch = _move_to_device(batch, device)
        if route_expert_distill_models:
            with torch.no_grad():
                image_paths = [str(path) for path in batch["image_path"]]
                if route_expert_distill_cache is None:
                    if len(route_expert_distill_models) == 1:
                        route_expert_distill_model, route_expert_distill_config = route_expert_distill_models[0]
                        batch["route_expert_logits"] = route_expert_logits_from_paths(
                            route_expert_distill_model,
                            route_expert_distill_config,
                            image_paths,
                            device=device,
                        )
                    else:
                        batch["route_expert_logits"] = route_expert_ensemble_logits_from_paths(
                            route_expert_distill_models,
                            image_paths,
                            device=device,
                            weights=route_expert_distill_weights,
                        )
                else:
                    missing_paths: list[str] = []
                    missing_keys: list[str] = []
                    for image_path in image_paths:
                        key = str(Path(image_path))
                        if key not in route_expert_distill_cache:
                            missing_paths.append(image_path)
                            missing_keys.append(key)
                    if missing_paths:
                        if len(route_expert_distill_models) == 1:
                            route_expert_distill_model, route_expert_distill_config = route_expert_distill_models[0]
                            missing_logits = route_expert_logits_from_paths(
                                route_expert_distill_model,
                                route_expert_distill_config,
                                missing_paths,
                                device=device,
                            ).detach().cpu()
                        else:
                            missing_logits = route_expert_ensemble_logits_from_paths(
                                route_expert_distill_models,
                                missing_paths,
                                device=device,
                                weights=route_expert_distill_weights,
                            ).detach().cpu()
                        for key, logit in zip(missing_keys, missing_logits, strict=True):
                            route_expert_distill_cache[key] = logit
                    batch["route_expert_logits"] = torch.stack(
                        [route_expert_distill_cache[str(Path(image_path))] for image_path in image_paths],
                        dim=0,
                    ).to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
                outputs = model(
                    batch["image"],
                    batch["boxes"],
                    batch["valid"],
                    batch["target_ar_id"],
                    batch["image_ar_log"],
                    batch["candidate_is_base"],
                    batch["box_meta"],
                    batch.get("letterbox_content_box"),
                    batch.get("subject_prior_box"),
                    batch.get("subject_prior_valid"),
                    batch.get("subject_prior_reliability"),
                    pred_subject_prior_alpha=pred_subject_prior_alpha,
                    return_generated_scores=return_generated_scores,
                )
                loss, metrics = compute_mobilecropnet_v4_loss(outputs, batch, **loss_kwargs)
        metrics["pred_subject_prior_alpha"] = float(pred_subject_prior_alpha)
        metrics["nonfinite_loss_step"] = 0.0
        metrics["nonfinite_grad_step"] = 0.0
        metrics["optimizer_step_skipped"] = 0.0
        loss_is_finite = bool(torch.isfinite(loss.detach()).all().item())
        if training and not loss_is_finite:
            optimizer.zero_grad(set_to_none=True)
            metrics["nonfinite_loss_step"] = 1.0
            metrics["optimizer_step_skipped"] = 1.0
        elif training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = None
            if grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm), error_if_nonfinite=False)
            grad_is_finite = grad_norm is None or bool(torch.isfinite(grad_norm.detach()).all().item())
            if grad_is_finite:
                scaler.step(optimizer)
            else:
                optimizer.zero_grad(set_to_none=True)
                metrics["nonfinite_grad_step"] = 1.0
                metrics["optimizer_step_skipped"] = 1.0
            scaler.update()
        if device.type == "cuda":
            interval = max(1, int(getattr(loader, "gpu_usage_sample_interval", 10)))
            if step == 1 or step % interval == 0:
                torch.cuda.synchronize(device)
                sample = _query_nvidia_smi(device)
                if sample:
                    gpu_samples.append(sample)
        rows.append(metrics)
        if torch.is_tensor(batch.get("image")):
            sample_count += int(batch["image"].shape[0])
        compute_sec += time() - compute_start
        log_interval = max(0, int(progress_log_interval or 0))
        if log_interval > 0 and (step == 1 or step % log_interval == 0):
            progress_payload: dict[str, Any] = {
                "event": "progress",
                "phase": str(phase),
                "epoch": int(epoch),
                "step": int(step),
                "samples": int(sample_count),
                "elapsed_sec": round(time() - loop_start, 3),
                "data_wait_sec": round(data_wait_sec, 3),
                "compute_sec": round(compute_sec, 3),
            }
            print(
                json.dumps(progress_payload, ensure_ascii=False),
                file=sys.stderr,
                flush=True,
            )
            if progress_callback is not None:
                try:
                    progress_callback(progress_payload)
                except Exception as exc:  # pragma: no cover - best-effort durable progress.
                    print(
                        json.dumps(
                            {
                                "event": "progress_status_write_failed",
                                "phase": str(phase),
                                "epoch": int(epoch),
                                "step": int(step),
                                "error": str(exc),
                            },
                            ensure_ascii=False,
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
    out = _mean_metrics(rows)
    out.update(_summarize_gpu_samples(gpu_samples, device))
    steps = max(1, len(rows))
    out.update(
        {
            "loader_data_wait_sec_total": float(data_wait_sec),
            "loader_data_wait_sec_per_step": float(data_wait_sec / steps),
            "loader_compute_sec_total": float(compute_sec),
            "loader_compute_sec_per_step": float(compute_sec / steps),
            "loader_samples_per_sec": float(sample_count / max(1e-9, data_wait_sec + compute_sec)),
            "loader_step_count": float(len(rows)),
        }
    )
    return out


def _selection_score(metrics: dict[str, float], mode: str) -> float:
    def metric(key: str, default: float = 0.0) -> float:
        value = float(metrics.get(key, default))
        return value if math.isfinite(value) else float(default)

    def bounded_inverse_loss(key: str, default: float = 1.0) -> float:
        value = max(0.0, metric(key, default))
        return 1.0 / (1.0 + value)

    def surface_metric(name: str, default: float = 0.0) -> float:
        return metric(f"source_mixture_{name}", metric(f"decision_conditioned_{name}", metric(name, default)))

    if mode == "legacy":
        return metric("top1_hit") + 0.25 * metric("proposal_recall_0_5") + 0.1 * metric("decision_acc")
    if mode == "route_only":
        return metric("route_acc")
    if mode == "route_balanced":
        return metric("route_balanced_acc", metric("route_acc"))
    if mode == "subject_box":
        subject_iou = metric("subject_box_iou")
        spatial_iou = metric("subject_box_spatial_iou", subject_iou)
        valid_balance = metric("subject_box_valid_best_balanced_acc", metric("subject_box_valid_balanced_acc", metric("subject_box_valid_acc")))
        heatmap_acc = metric("subject_box_spatial_heatmap_acc")
        mask_iou = metric("subject_box_spatial_mask_iou")
        return 0.44 * subject_iou + 0.20 * spatial_iou + 0.18 * valid_balance + 0.08 * heatmap_acc + 0.10 * mask_iou
    if mode == "sstk_topreturn":
        top_return_regret = metric("top_return_regret", 1.0)
        top_return_score = metric("top_return_score")
        return (
            0.34 * metric("top_return_hit", metric("top1_hit"))
            + 0.18 * metric("top1_exact_best_score")
            + 0.18 * top_return_score
            + 0.12 * max(0.0, 1.0 - top_return_regret)
            + 0.10 * metric("explicit_pairwise_acc")
            + 0.08 * metric("risk_suppression_acc")
            + 0.04 * metric("proposal_recall_0_5")
            - 0.24 * metric("top1_risk_rate")
        )
    if mode == "sstk_balanced_topreturn":
        top_return_regret = metric("top_return_regret", 1.0)
        score_calibration = max(0.0, 1.0 - metric("score_mae", 1.0))
        listwise_quality = bounded_inverse_loss("listwise_loss")
        pairwise_count = metric("explicit_pairwise_count")
        pairwise_quality = metric("explicit_pairwise_acc") if pairwise_count > 0.5 else listwise_quality
        return (
            0.20 * metric("top_return_hit", metric("top1_hit"))
            + 0.12 * metric("top1_exact_best_score")
            + 0.12 * metric("top_return_score")
            + 0.10 * max(0.0, 1.0 - top_return_regret)
            + 0.12 * pairwise_quality
            + 0.10 * listwise_quality
            + 0.08 * metric("risk_suppression_acc")
            + 0.08 * metric("proposal_recall_0_5")
            + 0.08 * score_calibration
            - 0.20 * metric("top1_risk_rate")
        )
    if mode == "public_score_distill":
        score_calibration = max(0.0, 1.0 - metric("score_mae", 1.0))
        listwise_quality = bounded_inverse_loss("listwise_loss")
        teacher_quality = bounded_inverse_loss("teacher_distill_loss")
        top_return_regret = metric("top_return_regret", 1.0)
        pairwise_count = metric("explicit_pairwise_count")
        pairwise_quality = metric("explicit_pairwise_acc") if pairwise_count > 0.5 else listwise_quality
        return (
            0.20 * pairwise_quality
            + 0.18 * listwise_quality
            + 0.16 * teacher_quality
            + 0.12 * metric("topk_teacher_hit", metric("top_return_hit"))
            + 0.10 * metric("top_return_hit", metric("top1_hit"))
            + 0.10 * metric("top_return_score")
            + 0.08 * score_calibration
            + 0.06 * max(0.0, 1.0 - top_return_regret)
            + 0.04 * metric("risk_suppression_acc")
            - 0.12 * metric("top1_risk_rate")
        )
    if mode == "deploy_align_topreturn":
        generated_top1_hit = metric("generated_align_top1_hit", metric("top_return_hit", metric("top1_hit")))
        generated_top1_iou = metric("generated_align_top1_iou", metric("top1_iou_to_best_positive"))
        generated_positive_recall = metric("generated_align_any_positive_0_5", metric("generated_align_positive_recall_0_5", metric("proposal_recall_0_5")))
        generated_match_rate = metric("generated_align_match_rate")
        generated_score_quality = bounded_inverse_loss("generated_align_score_loss")
        generated_listwise_quality = bounded_inverse_loss("generated_align_listwise_loss")
        subject_proposal_iou = metric("proposal_subject_max_iou")
        policy_score_quality = bounded_inverse_loss("policy_score_loss")
        top_return_regret = surface_metric("top_return_regret", metric("top_return_regret", 1.0))
        action_consistency = surface_metric("action_source_acc", metric("action_consistency_acc", metric("top1_action_source_acc")))
        baseline_preserve = surface_metric("baseline_preserve_top1_hit", metric("baseline_preserve_top1_hit"))
        crop_action = surface_metric("crop_action_top1_hit", metric("crop_action_top1_hit"))
        return (
            0.18 * generated_top1_hit
            + 0.14 * generated_top1_iou
            + 0.10 * generated_positive_recall
            + 0.07 * generated_match_rate
            + 0.08 * generated_score_quality
            + 0.07 * generated_listwise_quality
            + 0.06 * subject_proposal_iou
            + 0.10 * action_consistency
            + 0.05 * baseline_preserve
            + 0.04 * crop_action
            + 0.05 * policy_score_quality
            + 0.10 * surface_metric("top_return_hit", metric("top_return_hit", metric("top1_hit")))
            + 0.06 * surface_metric("top_return_score", metric("top_return_score"))
            + 0.04 * max(0.0, 1.0 - top_return_regret)
            + 0.01 * metric("decision_acc")
            + 0.02 * metric("route_acc")
            - 0.10 * metric("top1_risk_rate")
        )
    if mode == "release_gate":
        generated_top1_hit = metric("generated_align_top1_hit", metric("top_return_hit", metric("top1_hit")))
        generated_top1_iou = metric("generated_align_top1_iou", metric("top1_iou_to_best_positive"))
        subject_iou = metric("subject_box_iou")
        valid_recall = metric("subject_box_valid_recall")
        valid_specificity = metric("subject_box_valid_specificity")
        valid_calibration = min(valid_recall, valid_specificity)
        valid_balance = metric("subject_box_valid_balanced_acc", metric("subject_box_valid_acc"))
        top_return_regret = surface_metric("top_return_regret", metric("top_return_regret", 1.0))
        return (
            0.13 * surface_metric("top_return_hit", metric("top_return_hit", metric("top1_hit")))
            + 0.09 * surface_metric("top_return_score", metric("top_return_score"))
            + 0.08 * max(0.0, 1.0 - top_return_regret)
            + 0.12 * generated_top1_hit
            + 0.10 * generated_top1_iou
            + 0.08 * metric("proposal_recall_0_5")
            + 0.13 * metric("route_acc")
            + 0.12 * subject_iou
            + 0.07 * valid_balance
            + 0.06 * valid_calibration
            + 0.05 * surface_metric("action_source_acc", metric("action_consistency_acc", metric("top1_action_source_acc")))
            + 0.03 * metric("decision_acc")
            - 0.12 * metric("top1_risk_rate")
        )
    if mode == "release_gate_joint":
        route_balanced = metric("route_balanced_acc", metric("route_acc"))
        generated_top1_hit = metric("generated_align_top1_hit", 0.0)
        generated_top1_iou = metric("generated_align_top1_iou", metric("top1_iou_to_best_positive"))
        generated_positive_recall = metric("generated_align_any_positive_0_5", metric("generated_align_positive_recall_0_5", metric("proposal_recall_0_5")))
        subject_iou = metric("subject_box_iou")
        valid_balance = metric("subject_box_valid_balanced_acc", metric("subject_box_valid_acc"))
        action_consistency = surface_metric("action_source_acc", metric("action_consistency_acc", metric("top1_action_source_acc")))
        joint_floor = min(
            route_balanced / 0.52,
            subject_iou / 0.50,
            valid_balance / 0.52,
            max(generated_top1_hit, generated_positive_recall * 0.5) / 0.20,
        )
        joint_floor = max(0.0, min(1.0, joint_floor))
        return (
            0.18 * joint_floor
            + 0.18 * route_balanced
            + 0.14 * subject_iou
            + 0.10 * valid_balance
            + 0.14 * generated_top1_hit
            + 0.10 * generated_top1_iou
            + 0.07 * generated_positive_recall
            + 0.04 * action_consistency
            + 0.03 * surface_metric("top_return_hit", metric("top_return_hit", metric("top1_hit")))
            + 0.02 * metric("decision_acc")
            - 0.12 * metric("top1_risk_rate")
        )
    return (
        0.35 * metric("top1_hit")
        + 0.20 * metric("top1_exact_best_score")
        + 0.20 * metric("proposal_recall_0_5")
        + 0.15 * metric("explicit_pairwise_acc")
        + 0.15 * metric("risk_suppression_acc")
        + 0.05 * metric("decision_acc")
        - 0.20 * metric("top1_risk_rate")
    )


def _dataloader_kwargs(
    *,
    dataset: MobileCropNetV4BatchDataset,
    batch_size: int,
    shuffle: bool,
    sampler: WeightedRandomSampler | None,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int | None,
    pin_memory_device: str,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle) if sampler is None else False,
        "sampler": sampler,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "collate_fn": mobilecropnet_v4_collate,
        "drop_last": False,
    }
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = bool(persistent_workers)
        if prefetch_factor is not None:
            kwargs["prefetch_factor"] = int(prefetch_factor)
    if pin_memory and str(pin_memory_device).strip():
        kwargs["pin_memory_device"] = str(pin_memory_device)
    return kwargs


def _build_route_balanced_sampler(
    dataset: MobileCropNetV4BatchDataset,
    *,
    power: float,
    min_weight: float,
    max_weight: float,
) -> tuple[WeightedRandomSampler, dict[str, Any]]:
    route_counts: dict[int, int] = {}
    route_labels: list[int] = []
    for row in dataset.records:
        routing = row.get("routing") if isinstance(row.get("routing"), dict) else {}
        try:
            route_id = int(routing.get("subject_mode_id", 0))
        except (TypeError, ValueError):
            route_id = 0
        route_id = max(0, min(SUBJECT_MODE_COUNT - 1, route_id))
        route_labels.append(route_id)
        route_counts[route_id] = route_counts.get(route_id, 0) + 1
    weights = []
    for route_id in route_labels:
        count = max(1, route_counts.get(route_id, 1))
        weight = float(len(route_labels)) / float(count)
        if float(power) != 1.0:
            weight = weight ** float(power)
        weight = max(float(min_weight), min(float(max_weight), weight))
        weights.append(weight)
    weight_tensor = torch.tensor(weights, dtype=torch.double)
    weight_mean = float(weight_tensor.mean().item()) if len(weight_tensor) > 0 else 1.0
    if weight_mean > 0:
        weight_tensor = weight_tensor / weight_mean
    debug_counts = {str(route_id): int(count) for route_id, count in sorted(route_counts.items())}
    summary = {
        "enabled": True,
        "num_samples": int(len(route_labels)),
        "replacement": True,
        "power": float(power),
        "min_weight": float(min_weight),
        "max_weight": float(max_weight),
        "route_target_counts": debug_counts,
        "sample_weight_mean": float(weight_tensor.mean().item()) if len(weight_tensor) > 0 else None,
        "sample_weight_min": float(weight_tensor.min().item()) if len(weight_tensor) > 0 else None,
        "sample_weight_max": float(weight_tensor.max().item()) if len(weight_tensor) > 0 else None,
    }
    sampler = WeightedRandomSampler(weight_tensor, num_samples=len(route_labels), replacement=True)
    return sampler, summary


def _row_route_id(row: dict[str, Any]) -> int:
    routing = row.get("routing") if isinstance(row.get("routing"), dict) else {}
    try:
        route_id = int(routing.get("subject_mode_id", 0))
    except (TypeError, ValueError):
        route_id = 0
    return max(0, min(SUBJECT_MODE_COUNT - 1, route_id))


def _row_decision_source_id(row: dict[str, Any]) -> int:
    decision = row.get("decision_target") if isinstance(row.get("decision_target"), dict) else {}
    crop_id = DECISION_VOCAB.index("crop")
    target = decision_id(decision.get("decision_id", decision.get("decision_type", crop_id)))
    return 1 if int(target) == crop_id else 0


def _balanced_component_weights(
    labels: list[int],
    *,
    label_name: str,
    power: float,
    min_weight: float,
    max_weight: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    counts: dict[int, int] = {}
    for label in labels:
        counts[int(label)] = counts.get(int(label), 0) + 1
    weights = []
    for label in labels:
        count = max(1, counts.get(int(label), 1))
        weight = float(len(labels)) / float(count)
        if float(power) != 1.0:
            weight = weight ** float(power)
        weight = max(float(min_weight), min(float(max_weight), weight))
        weights.append(weight)
    tensor = torch.tensor(weights, dtype=torch.double)
    summary = {
        "label_name": str(label_name),
        "power": float(power),
        "min_weight": float(min_weight),
        "max_weight": float(max_weight),
        "target_counts": {str(label): int(count) for label, count in sorted(counts.items())},
        "raw_weight_mean": float(tensor.mean().item()) if len(tensor) > 0 else None,
        "raw_weight_min": float(tensor.min().item()) if len(tensor) > 0 else None,
        "raw_weight_max": float(tensor.max().item()) if len(tensor) > 0 else None,
    }
    return tensor, summary


def _build_train_balanced_sampler(
    dataset: MobileCropNetV4BatchDataset,
    *,
    route_enabled: bool,
    route_power: float,
    route_min_weight: float,
    route_max_weight: float,
    decision_source_enabled: bool,
    decision_source_power: float,
    decision_source_min_weight: float,
    decision_source_max_weight: float,
) -> tuple[WeightedRandomSampler, dict[str, Any]]:
    sample_count = int(len(dataset.records))
    weight_tensor = torch.ones((sample_count,), dtype=torch.double)
    components: dict[str, Any] = {}
    if bool(route_enabled):
        route_weights, route_summary = _balanced_component_weights(
            [_row_route_id(row) for row in dataset.records],
            label_name="route",
            power=float(route_power),
            min_weight=float(route_min_weight),
            max_weight=float(route_max_weight),
        )
        weight_tensor = weight_tensor * route_weights
        components["route"] = route_summary
    if bool(decision_source_enabled):
        source_weights, source_summary = _balanced_component_weights(
            [_row_decision_source_id(row) for row in dataset.records],
            label_name="decision_source",
            power=float(decision_source_power),
            min_weight=float(decision_source_min_weight),
            max_weight=float(decision_source_max_weight),
        )
        weight_tensor = weight_tensor * source_weights
        components["decision_source"] = source_summary
    weight_mean = float(weight_tensor.mean().item()) if sample_count > 0 else 1.0
    if weight_mean > 0:
        weight_tensor = weight_tensor / weight_mean
    summary = {
        "enabled": True,
        "num_samples": sample_count,
        "replacement": True,
        "components": components,
        "sample_weight_mean": float(weight_tensor.mean().item()) if sample_count > 0 else None,
        "sample_weight_min": float(weight_tensor.min().item()) if sample_count > 0 else None,
        "sample_weight_max": float(weight_tensor.max().item()) if sample_count > 0 else None,
    }
    sampler = WeightedRandomSampler(weight_tensor, num_samples=sample_count, replacement=True)
    return sampler, summary


def _freeze_module(module: torch.nn.Module | None) -> int:
    if module is None:
        return 0
    frozen = 0
    for param in module.parameters():
        if param.requires_grad:
            param.requires_grad_(False)
            frozen += int(param.numel())
    return frozen


def _apply_freeze_args(model: MobileCropNetV4, args: argparse.Namespace) -> dict[str, Any]:
    freeze_subject_stack = bool(args.freeze_subject_box_stack)
    trainable_prefixes = tuple(prefix.strip() for prefix in str(args.trainable_module_prefixes or "").split(",") if prefix.strip())
    summary: dict[str, Any] = {
        "freeze_backbone": bool(args.freeze_backbone) or freeze_subject_stack,
        "freeze_condition_encoder": bool(args.freeze_condition_encoder) or freeze_subject_stack,
        "freeze_subject_box_heads": bool(args.freeze_subject_box_heads) or freeze_subject_stack,
        "trainable_module_prefixes": list(trainable_prefixes),
        "frozen_parameter_count": 0,
        "frozen_modules": {},
        "prefix_frozen_parameter_count": 0,
    }
    if summary["freeze_backbone"]:
        summary["frozen_modules"]["backbone"] = _freeze_module(model.backbone)
    if summary["freeze_condition_encoder"]:
        summary["frozen_modules"]["ar_embed"] = _freeze_module(model.ar_embed)
        summary["frozen_modules"]["cond_proj"] = _freeze_module(model.cond_proj)
    if summary["freeze_subject_box_heads"]:
        for name in [
            "subject_box_head",
            "subject_valid_head",
            "subject_spatial_proj",
            "subject_spatial_heatmap_head",
            "subject_spatial_box_head",
            "subject_spatial_valid_head",
            "subject_spatial_mix_head",
            "subject_proposal_head",
            "subject_refine_proj",
            "subject_refine_box_head",
            "subject_refine_valid_head",
        ]:
            summary["frozen_modules"][name] = _freeze_module(getattr(model, name, None))
    if trainable_prefixes:
        prefix_frozen = 0
        trainable_matched = 0
        for name, param in model.named_parameters():
            keep_trainable = any(name == prefix or name.startswith(prefix + ".") for prefix in trainable_prefixes)
            if keep_trainable:
                if param.requires_grad:
                    trainable_matched += int(param.numel())
            elif param.requires_grad:
                param.requires_grad_(False)
                prefix_frozen += int(param.numel())
        summary["prefix_frozen_parameter_count"] = int(prefix_frozen)
        summary["trainable_prefix_parameter_count"] = int(trainable_matched)
    summary["frozen_parameter_count"] = int(sum(int(v) for v in summary["frozen_modules"].values()))
    summary["frozen_parameter_count"] += int(summary.get("prefix_frozen_parameter_count", 0))
    total = int(sum(param.numel() for param in model.parameters()))
    trainable = int(sum(param.numel() for param in model.parameters() if param.requires_grad))
    summary["total_parameter_count"] = total
    summary["trainable_parameter_count"] = trainable
    return summary


def save_checkpoint(
    path: Path,
    *,
    model: MobileCropNetV4,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "model_config": model_config_to_dict(model),
            "train_config": config,
        },
        path,
    )


def main(argv: list[str] | None = None) -> int:
    explicit_args = _explicit_arg_names(argv)
    args = build_parser().parse_args(argv)
    _apply_model_profile(args, explicit_args)
    _resolve_image_norm_args(args)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    train_status_path = args.output_dir / "train_status.json"
    startup_time = time()

    def _write_startup_status(phase: str) -> None:
        payload = {
            "state": "starting",
            "phase": phase,
            "start_time_unix": startup_time,
            "last_update_time_unix": time(),
            "current_epoch": 0,
            "total_epochs": int(args.epochs),
            "output_dir": str(args.output_dir),
            "config_path": str(args.output_dir / "config.json"),
            "metrics_path": str(args.output_dir / "metrics.json"),
            "summary_path": str(args.output_dir / "summary.json"),
            "best_checkpoint": str(args.output_dir / "best.pt"),
            "latest_checkpoint": str(args.output_dir / "last.pt"),
            "history_length": 0,
            "best_selection_score": -1.0,
            "best_subject_box_score": -1.0,
            "selection_metric": str(args.selection_metric),
            "model_profile": str(args.model_profile),
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint is not None else None,
        }
        train_status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_startup_status("dataset_build")
    train_ds = MobileCropNetV4BatchDataset(
        jsonl_path=args.train_jsonl,
        project_root=args.project_root,
        image_root=args.image_root,
        input_size=args.input_size,
        candidate_k=args.candidate_k,
        max_rows=args.max_train_rows,
        include_ignored_candidates=args.include_ignored_candidates,
        include_overflow_candidates=args.include_overflow_candidates,
        precompute_sample_tensors=args.precompute_sample_tensors,
        image_tensor_cache_size=args.image_tensor_cache_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        pairwise_jsonl=args.pairwise_jsonl,
        listwise_jsonl=args.listwise_jsonl,
        max_pairwise_pairs=args.max_pairwise_pairs,
        score_target_mode=args.score_target_mode,
        subject_box_target_source=args.subject_box_target_source,
        subject_box_valid_target_mode=args.subject_box_valid_target_mode,
        subject_box_valid_reliability_min=args.subject_box_valid_reliability_min,
        high_score_safe_positive_threshold=args.high_score_safe_positive_threshold,
    )
    val_ds = MobileCropNetV4BatchDataset(
        jsonl_path=args.val_jsonl,
        project_root=args.project_root,
        image_root=args.image_root,
        input_size=args.input_size,
        candidate_k=args.candidate_k,
        max_rows=args.max_val_rows,
        include_ignored_candidates=args.include_ignored_candidates,
        include_overflow_candidates=args.include_overflow_candidates,
        precompute_sample_tensors=args.precompute_sample_tensors,
        image_tensor_cache_size=args.image_tensor_cache_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        pairwise_jsonl=args.val_pairwise_jsonl if args.val_pairwise_jsonl is not None else args.pairwise_jsonl,
        listwise_jsonl=args.val_listwise_jsonl if args.val_listwise_jsonl is not None else args.listwise_jsonl,
        max_pairwise_pairs=args.max_pairwise_pairs,
        score_target_mode=args.score_target_mode,
        subject_box_target_source=args.subject_box_target_source,
        subject_box_valid_target_mode=args.subject_box_valid_target_mode,
        subject_box_valid_reliability_min=args.subject_box_valid_reliability_min,
        high_score_safe_positive_threshold=args.high_score_safe_positive_threshold,
    )
    train_sampler = None
    train_sampler_summary: dict[str, Any] = {"enabled": False}
    if bool(args.train_decision_source_balanced_sampler):
        train_sampler, train_sampler_summary = _build_train_balanced_sampler(
            train_ds,
            route_enabled=bool(args.train_route_balanced_sampler),
            route_power=float(args.train_route_balanced_sampler_power),
            route_min_weight=float(args.train_route_balanced_sampler_min_weight),
            route_max_weight=float(args.train_route_balanced_sampler_max_weight),
            decision_source_enabled=True,
            decision_source_power=float(args.train_decision_source_balanced_sampler_power),
            decision_source_min_weight=float(args.train_decision_source_balanced_sampler_min_weight),
            decision_source_max_weight=float(args.train_decision_source_balanced_sampler_max_weight),
        )
    elif bool(args.train_route_balanced_sampler):
        train_sampler, train_sampler_summary = _build_route_balanced_sampler(
            train_ds,
            power=float(args.train_route_balanced_sampler_power),
            min_weight=float(args.train_route_balanced_sampler_min_weight),
            max_weight=float(args.train_route_balanced_sampler_max_weight),
        )
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(
            {
                "train": train_ds.summary(),
                "val": val_ds.summary(),
                "train_sampler": train_sampler_summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_startup_status("dataloader_build")
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        **_dataloader_kwargs(
            dataset=train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory_device=args.pin_memory_device,
        )
    )
    setattr(train_loader, "gpu_usage_sample_interval", args.gpu_usage_sample_interval)
    val_loader = DataLoader(
        **_dataloader_kwargs(
            dataset=val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=None,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=args.persistent_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory_device=args.pin_memory_device,
        )
    )
    setattr(val_loader, "gpu_usage_sample_interval", args.gpu_usage_sample_interval)

    _write_startup_status("model_build")
    model = MobileCropNetV4(
        candidate_k=args.candidate_k,
        proposal_q=args.proposal_q,
        input_size=args.input_size,
        width_mult=args.width_mult,
        token_dim=args.token_dim,
        backbone_name=args.backbone_name,
        backbone_pretrained=args.backbone_pretrained,
        ranker_type=args.ranker_type,
        ranker_depth=args.ranker_depth,
        use_subject_prior=args.use_subject_prior,
        route_image_only=args.route_image_only,
        route_use_subject_prior=args.route_use_subject_prior,
        route_use_candidate_context=args.route_use_candidate_context,
        route_use_subject_box_features=args.route_use_subject_box_features,
        route_use_subject_spatial_token=args.route_use_subject_spatial_token,
        route_head_depth=args.route_head_depth,
        route_head_hidden_mult=args.route_head_hidden_mult,
        route_head_dropout=args.route_head_dropout,
        route_aux_heads=args.route_aux_heads,
        route_image_residual=args.route_image_residual,
        route_image_residual_weight=args.route_image_residual_weight,
        route_decode_mode=args.route_decode_mode,
        route_decode_kind_weight=args.route_decode_kind_weight,
        route_decode_cardinality_weight=args.route_decode_cardinality_weight,
        policy_use_subject_prior=args.policy_use_subject_prior,
        proposal_use_subject_prior=args.proposal_use_subject_prior,
        subject_box_head=args.subject_box_head,
        subject_box_spatial_head=args.subject_box_spatial_head,
        subject_box_spatial_multiscale=args.subject_box_spatial_multiscale,
        subject_box_spatial_mix_bias=args.subject_box_spatial_mix_bias,
        subject_box_spatial_output=args.subject_box_spatial_output,
        subject_box_spatial_box_mode=args.subject_box_spatial_box_mode,
        subject_box_spatial_extent_scale=args.subject_box_spatial_extent_scale,
        subject_box_coord_space=args.subject_box_coord_space,
        subject_box_refine_with_proposals=args.subject_box_refine_with_proposals,
        subject_valid_route_calibrator=args.subject_valid_route_calibrator,
        subject_valid_route_calibrator_hidden_mult=args.subject_valid_route_calibrator_hidden_mult,
        subject_valid_route_calibrator_detach=args.subject_valid_route_calibrator_detach,
        use_pred_subject_box_as_prior=args.use_pred_subject_box_as_prior,
        detach_pred_subject_prior=args.detach_pred_subject_prior,
        policy_score_head=args.policy_score_head,
        policy_score_threshold_low=args.policy_score_threshold_low,
        policy_score_threshold_high=args.policy_score_threshold_high,
        decision_source_head=args.decision_source_head,
        decision_source_logit_threshold=args.decision_source_logit_threshold,
        decision_source_pair_head=args.decision_source_pair_head,
        decision_source_pair_hidden_mult=args.decision_source_pair_hidden_mult,
        decision_source_pair_after_return_deltas=args.decision_source_pair_after_return_deltas,
        decision_source_pair_subject_state=args.decision_source_pair_subject_state,
        decision_source_pair_subject_detach=args.decision_source_pair_subject_detach,
        route_condition_candidate_scores=args.route_condition_candidate_scores,
        route_condition_detach=args.route_condition_detach,
        return_score_head=args.return_score_head,
        return_score_head_depth=args.return_score_head_depth,
        return_score_head_hidden_mult=args.return_score_head_hidden_mult,
        return_score_action_source_bias=args.return_score_action_source_bias,
        return_score_decision_source_bias=args.return_score_decision_source_bias,
        return_score_decision_source_detach=args.return_score_decision_source_detach,
        return_score_decision_source_max_scale=args.return_score_decision_source_max_scale,
        return_score_decision_source_from_source_head=args.return_score_decision_source_from_source_head,
        return_score_decision_conditioned=args.return_score_decision_conditioned,
        return_score_decision_conditioned_mask_value=args.return_score_decision_conditioned_mask_value,
        return_score_decision_conditioned_from_source_head=args.return_score_decision_conditioned_from_source_head,
        return_score_source_mixture=args.return_score_source_mixture,
        return_score_source_mixture_strength=args.return_score_source_mixture_strength,
        return_score_source_mixture_detach=args.return_score_source_mixture_detach,
        return_score_source_mixture_from_source_head=args.return_score_source_mixture_from_source_head,
        return_score_source_mixture_gap_threshold=args.return_score_source_mixture_gap_threshold,
        return_score_source_mixture_gap_mask_value=args.return_score_source_mixture_gap_mask_value,
        return_score_source_gate=args.return_score_source_gate,
        return_score_source_gate_hidden_mult=args.return_score_source_gate_hidden_mult,
        return_score_source_gate_strength=args.return_score_source_gate_strength,
        return_score_source_gate_detach=args.return_score_source_gate_detach,
        return_score_source_gate_subject_state=args.return_score_source_gate_subject_state,
        return_score_source_gate_subject_detach=args.return_score_source_gate_subject_detach,
        return_score_source_gate_logit_threshold=args.return_score_source_gate_logit_threshold,
        return_score_source_specific_head=args.return_score_source_specific_head,
        return_score_source_specific_hidden_mult=args.return_score_source_specific_hidden_mult,
        return_score_context_head=args.return_score_context_head,
        return_score_context_hidden_mult=args.return_score_context_hidden_mult,
        return_score_context_detach=args.return_score_context_detach,
        return_score_policy_match_head=args.return_score_policy_match_head,
        return_score_policy_match_hidden_mult=args.return_score_policy_match_hidden_mult,
        return_score_policy_match_detach=args.return_score_policy_match_detach,
        return_score_policy_source_head=args.return_score_policy_source_head,
        return_score_policy_source_hidden_mult=args.return_score_policy_source_hidden_mult,
        return_score_policy_source_detach=args.return_score_policy_source_detach,
        return_score_competition_head=args.return_score_competition_head,
        return_score_competition_hidden_mult=args.return_score_competition_hidden_mult,
        return_score_competition_detach=args.return_score_competition_detach,
        return_score_set_refiner_head=args.return_score_set_refiner_head,
        return_score_set_refiner_hidden_mult=args.return_score_set_refiner_hidden_mult,
        return_score_set_refiner_layers=args.return_score_set_refiner_layers,
        return_score_set_refiner_detach=args.return_score_set_refiner_detach,
        return_score_action_decoder_head=args.return_score_action_decoder_head,
        return_score_action_decoder_hidden_mult=args.return_score_action_decoder_hidden_mult,
        return_score_action_decoder_layers=args.return_score_action_decoder_layers,
        return_score_action_decoder_detach=args.return_score_action_decoder_detach,
        return_score_action_decoder_subject_state=args.return_score_action_decoder_subject_state,
        return_score_action_decoder_subject_detach=args.return_score_action_decoder_subject_detach,
        return_score_action_decoder_replace=args.return_score_action_decoder_replace,
    ).to(device)
    init_checkpoint_summary: dict[str, Any] | None = None
    if args.init_checkpoint is not None:
        _write_startup_status("init_checkpoint_load")
        checkpoint = torch.load(args.init_checkpoint, map_location="cpu")
        state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state, dict):
            raise SystemExit(f"invalid init checkpoint state: {args.init_checkpoint}")
        model_state = model.state_dict()
        loadable_state = {}
        skipped_shape_keys = []
        skipped_type_keys = []
        skip_prefixes = _split_cli_list(args.init_checkpoint_skip_prefixes)
        skipped_prefix_keys = []
        for key, value in state.items():
            if skip_prefixes and any(key == prefix or key.startswith(prefix + ".") for prefix in skip_prefixes):
                skipped_prefix_keys.append(key)
                continue
            if key not in model_state:
                continue
            target_value = model_state[key]
            if not torch.is_tensor(value) or not torch.is_tensor(target_value):
                skipped_type_keys.append(key)
                continue
            if tuple(value.shape) != tuple(target_value.shape):
                skipped_shape_keys.append(
                    {
                        "key": key,
                        "checkpoint_shape": list(value.shape),
                        "model_shape": list(target_value.shape),
                    }
                )
                continue
            loadable_state[key] = value
        if not loadable_state:
            raise SystemExit(f"init checkpoint has no shape-compatible tensors: {args.init_checkpoint}")
        incompatible = model.load_state_dict(loadable_state, strict=False)
        init_checkpoint_summary = {
            "path": str(args.init_checkpoint),
            "checkpoint_keys": int(len(state)),
            "loaded_keys": int(len(loadable_state)),
            "skipped_shape_keys": skipped_shape_keys[:100],
            "skipped_shape_key_count": int(len(skipped_shape_keys)),
            "skipped_type_keys": skipped_type_keys[:100],
            "skipped_type_key_count": int(len(skipped_type_keys)),
            "skipped_prefixes": list(skip_prefixes),
            "skipped_prefix_key_count": int(len(skipped_prefix_keys)),
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }
        print(json.dumps({"event": "init_checkpoint_loaded", **init_checkpoint_summary}, ensure_ascii=False), flush=True)
    _write_startup_status("freeze_setup")
    freeze_summary = _apply_freeze_args(model, args)
    if int(freeze_summary.get("frozen_parameter_count", 0)) > 0:
        print(json.dumps({"event": "modules_frozen", **freeze_summary}, ensure_ascii=False), flush=True)
    if args.compile and hasattr(torch, "compile"):
        _write_startup_status("compile")
        model = torch.compile(model)  # type: ignore[assignment]
    route_expert_distill_models: list[tuple[torch.nn.Module, dict[str, Any]]] | None = None
    route_expert_distill_weights: list[float] | None = None
    route_expert_distill_summary: dict[str, Any] | None = None
    route_expert_distill_cache: dict[str, torch.Tensor] | None = None
    route_expert_distill_paths: list[Path] = []
    if args.route_expert_distill_checkpoint is not None:
        route_expert_distill_paths.append(args.route_expert_distill_checkpoint)
    route_expert_distill_paths.extend(args.route_expert_distill_checkpoints or [])
    if route_expert_distill_paths and float(args.route_expert_distill_weight) > 0.0:
        _write_startup_status("route_expert_distill_load")
        if args.route_expert_distill_weights:
            route_expert_distill_weights = [float(item) for item in str(args.route_expert_distill_weights).split(",") if item.strip()]
            if len(route_expert_distill_weights) != len(route_expert_distill_paths):
                raise SystemExit(
                    "route expert distill weight count mismatch: "
                    f"{len(route_expert_distill_weights)} != {len(route_expert_distill_paths)}"
                )
        route_expert_distill_models = []
        route_expert_entries: list[dict[str, Any]] = []
        for checkpoint in route_expert_distill_paths:
            route_expert_distill_model, route_expert_distill_config, route_expert_ckpt = load_route_expert_checkpoint(
                checkpoint,
                device=device,
            )
            for param in route_expert_distill_model.parameters():
                param.requires_grad_(False)
            route_expert_distill_models.append((route_expert_distill_model, route_expert_distill_config))
            route_expert_entries.append(
                {
                    "checkpoint": str(checkpoint),
                    "config": route_expert_distill_config,
                    "checkpoint_epoch": route_expert_ckpt.get("epoch"),
                    "checkpoint_metrics": route_expert_ckpt.get("metrics"),
                }
            )
        route_expert_distill_summary = {
            "checkpoints": [str(path) for path in route_expert_distill_paths],
            "weights": route_expert_distill_weights,
            "experts": route_expert_entries,
            "weight": float(args.route_expert_distill_weight),
            "temperature": float(args.route_expert_distill_temperature),
            "hard_weight": float(args.route_expert_distill_hard_weight),
        }
        print(json.dumps({"event": "route_expert_distill_loaded", **route_expert_distill_summary}, ensure_ascii=False), flush=True)
        route_expert_distill_cache = {}
    _write_startup_status("optimizer_build")
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if not trainable_parameters:
        raise SystemExit("no trainable parameters remain after freeze options")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    def _json_safe(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        return value

    config = _json_safe(vars(args).copy())
    if init_checkpoint_summary is not None:
        config["init_checkpoint_summary"] = init_checkpoint_summary
    if route_expert_distill_summary is not None:
        config["route_expert_distill_summary"] = route_expert_distill_summary
    config["freeze_summary"] = freeze_summary
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    best_score = -1.0
    best_subject_box_score = -1.0
    epochs_since_improve = 0
    history: list[dict[str, Any]] = []
    loss_kwargs = {
        "score_weight": float(args.score_weight),
        "listwise_weight": float(args.listwise_weight),
        "pairwise_weight": float(args.pairwise_weight),
        "explicit_pairwise_weight": float(args.explicit_pairwise_weight),
        "positive_weight": float(args.positive_weight),
        "risk_weight": float(args.risk_weight),
        "top1_risk_weight": float(args.top1_risk_weight),
        "top1_risk_margin": float(args.top1_risk_margin),
        "top_return_weight": float(args.top_return_weight),
        "raw_top_return_weight": float(args.raw_top_return_weight),
        "top_return_score_margin": float(args.top_return_score_margin),
        "top_return_logit_margin": float(args.top_return_logit_margin),
        "top_return_temperature": float(args.top_return_temperature),
        "return_score_weight": float(args.return_score_weight),
        "return_listwise_weight": float(args.return_listwise_weight),
        "return_positive_weight": float(args.return_positive_weight),
        "return_positive_margin_weight": float(args.return_positive_margin_weight),
        "return_positive_margin": float(args.return_positive_margin),
        "return_risk_suppression_weight": float(args.return_risk_suppression_weight),
        "return_risk_suppression_margin": float(args.return_risk_suppression_margin),
        "return_exact_weight": float(args.return_exact_weight),
        "return_exact_min_gap": float(args.return_exact_min_gap),
        "return_target_mode": str(args.return_target_mode),
        "return_explicit_pairwise_weight": float(args.return_explicit_pairwise_weight),
        "return_explicit_pairwise_margin": float(args.return_explicit_pairwise_margin),
        "teacher_distill_weight": float(args.teacher_distill_weight),
        "teacher_distill_temperature": float(args.teacher_distill_temperature),
        "topk_coverage_weight": float(args.topk_coverage_weight),
        "topk_coverage_k": int(args.topk_coverage_k),
        "topk_coverage_temperature": float(args.topk_coverage_temperature),
        "macro_weight": float(args.macro_weight),
        "checklist_class_weight": float(args.checklist_class_weight),
        "checklist_applicability_weight": float(args.checklist_applicability_weight),
        "detail_score_weight": float(args.detail_score_weight),
        "why_tag_weight": float(args.why_tag_weight),
        "route_weight": float(args.route_weight),
        "route_balanced_ce": bool(args.route_balanced_ce),
        "route_focal_gamma": float(args.route_focal_gamma),
        "route_object_single_margin_weight": float(args.route_object_single_margin_weight),
        "route_object_single_margin": float(args.route_object_single_margin),
        "route_hierarchy_weight": float(args.route_hierarchy_weight),
        "route_cardinality_weight": float(args.route_cardinality_weight),
        "route_expert_distill_weight": float(args.route_expert_distill_weight),
        "route_expert_distill_temperature": float(args.route_expert_distill_temperature),
        "route_expert_distill_hard_weight": float(args.route_expert_distill_hard_weight),
        "decision_weight": float(args.decision_weight),
        "decision_crop_rebalance": bool(args.decision_crop_rebalance),
        "decision_crop_rebalance_max_weight": float(args.decision_crop_rebalance_max_weight),
        "decision_source_weight": float(args.decision_source_weight),
        "decision_source_balanced_bce": bool(args.decision_source_balanced_bce),
        "decision_source_crop_rebalance_max_weight": float(args.decision_source_crop_rebalance_max_weight),
        "decision_source_focal_gamma": float(args.decision_source_focal_gamma),
        "decision_source_focal_alpha": args.decision_source_focal_alpha,
        "decision_source_margin_weight": float(args.decision_source_margin_weight),
        "decision_source_logit_margin": float(args.decision_source_logit_margin),
        "decision_source_margin_balanced": bool(args.decision_source_margin_balanced),
        "decision_source_margin_balance_max_weight": float(args.decision_source_margin_balance_max_weight),
        "decision_source_margin_crop_weight_mult": float(args.decision_source_margin_crop_weight_mult),
        "decision_source_pair_supervision_weight": float(args.decision_source_pair_supervision_weight),
        "decision_source_pair_margin_weight": float(args.decision_source_pair_margin_weight),
        "decision_source_pair_balanced_bce": bool(args.decision_source_pair_balanced_bce),
        "decision_source_pair_focal_gamma": float(args.decision_source_pair_focal_gamma),
        "decision_source_pair_focal_alpha": args.decision_source_pair_focal_alpha,
        "source_gate_supervision_weight": float(args.source_gate_supervision_weight),
        "source_gate_balanced_bce": bool(args.source_gate_balanced_bce),
        "source_gate_crop_rebalance_max_weight": float(args.source_gate_crop_rebalance_max_weight),
        "source_gate_focal_gamma": float(args.source_gate_focal_gamma),
        "source_gate_focal_alpha": args.source_gate_focal_alpha,
        "source_gate_margin_weight": float(args.source_gate_margin_weight),
        "source_gate_logit_margin": float(args.source_gate_logit_margin),
        "source_gate_margin_balanced": bool(args.source_gate_margin_balanced),
        "source_gate_margin_balance_max_weight": float(args.source_gate_margin_balance_max_weight),
        "source_gate_margin_crop_weight_mult": float(args.source_gate_margin_crop_weight_mult),
        "action_consistency_weight": float(args.action_consistency_weight),
        "action_consistency_margin": float(args.action_consistency_margin),
        "action_source_balanced": bool(args.action_source_balanced),
        "action_source_balance_max_weight": float(args.action_source_balance_max_weight),
        "action_source_crop_weight_mult": float(args.action_source_crop_weight_mult),
        "action_return_joint_weight": float(args.action_return_joint_weight),
        "action_return_joint_score_margin": float(args.action_return_joint_score_margin),
        "action_return_joint_logit_margin": float(args.action_return_joint_logit_margin),
        "action_return_joint_temperature": float(args.action_return_joint_temperature),
        "return_source_margin_weight": float(args.return_source_margin_weight),
        "return_source_margin_logit_margin": float(args.return_source_margin_logit_margin),
        "return_source_margin_balanced": bool(args.return_source_margin_balanced),
        "return_source_margin_balance_max_weight": float(args.return_source_margin_balance_max_weight),
        "return_source_margin_crop_weight_mult": float(args.return_source_margin_crop_weight_mult),
        "decision_utility_align_weight": float(args.decision_utility_align_weight),
        "decision_utility_align_temperature": float(args.decision_utility_align_temperature),
        "policy_score_weight": float(args.policy_score_weight),
        "delta_weight": float(args.delta_weight),
        "proposal_weight": float(args.proposal_weight),
        "proposal_subject_weight": float(args.proposal_subject_weight),
        "proposal_use_positive_scores": True,
        "subject_proposal_align_weight": float(args.subject_proposal_align_weight),
        "subject_box_weight": float(args.subject_box_weight),
        "subject_box_valid_weight": float(args.subject_box_valid_weight),
        "subject_box_l1_weight": float(args.subject_box_l1_weight),
        "subject_box_iou_weight": float(args.subject_box_iou_weight),
        "subject_box_center_weight": float(args.subject_box_center_weight),
        "subject_box_size_weight": float(args.subject_box_size_weight),
        "subject_box_aspect_weight": float(args.subject_box_aspect_weight),
        "subject_box_ciou_weight": float(args.subject_box_ciou_weight),
        "subject_box_spatial_aux_weight": float(args.subject_box_spatial_aux_weight),
        "subject_box_spatial_heatmap_weight": float(args.subject_box_spatial_heatmap_weight),
        "subject_box_spatial_mask_weight": float(args.subject_box_spatial_mask_weight),
        "subject_box_valid_balanced_bce": bool(args.subject_box_valid_balanced_bce),
        "subject_box_valid_negative_scale": float(args.subject_box_valid_negative_scale),
        "generated_proposal_align_weight": float(args.generated_proposal_align_weight),
        "generated_proposal_score_weight": float(args.generated_proposal_score_weight),
        "generated_proposal_listwise_weight": float(args.generated_proposal_listwise_weight),
        "generated_proposal_positive_weight": float(args.generated_proposal_positive_weight),
        "generated_proposal_risk_weight": float(args.generated_proposal_risk_weight),
        "generated_proposal_positive_utility_weight": float(args.generated_proposal_positive_utility_weight),
        "generated_proposal_positive_margin_weight": float(args.generated_proposal_positive_margin_weight),
        "generated_proposal_positive_margin": float(args.generated_proposal_positive_margin),
        "generated_proposal_match_iou": float(args.generated_proposal_match_iou),
    }
    start = time()
    summary_path = args.output_dir / "summary.json"

    def _best_history_row() -> dict[str, Any] | None:
        if not history:
            return None
        return max(history, key=lambda item: float(item.get("selection_score", float("-inf"))))

    def _write_terminal_summary(*, state: str, last_row: dict[str, Any] | None) -> None:
        best_row = _best_history_row()
        payload: dict[str, Any] = {
            "state": str(state),
            "start_time_unix": start,
            "end_time_unix": time(),
            "elapsed_sec": time() - start,
            "output_dir": str(args.output_dir),
            "config_path": str(args.output_dir / "config.json"),
            "metrics_path": str(args.output_dir / "metrics.json"),
            "status_path": str(train_status_path),
            "best_checkpoint": str(args.output_dir / "best.pt"),
            "latest_checkpoint": str(args.output_dir / "last.pt"),
            "best_checkpoint_exists": bool((args.output_dir / "best.pt").exists()),
            "latest_checkpoint_exists": bool((args.output_dir / "last.pt").exists()),
            "history_length": len(history),
            "selection_metric": str(args.selection_metric),
            "best_selection_score": float(best_score),
            "best_subject_box_score": float(best_subject_box_score),
        }
        if best_row is not None:
            payload["best_epoch"] = int(best_row.get("epoch", 0))
            payload["best_row"] = best_row
        if last_row is not None:
            payload["last_row"] = last_row
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_train_status(*, state: str, epoch: int, last_row: dict[str, Any] | None) -> None:
        payload: dict[str, Any] = {
            "state": str(state),
            "start_time_unix": start,
            "last_update_time_unix": time(),
            "current_epoch": int(epoch),
            "total_epochs": int(args.epochs),
            "output_dir": str(args.output_dir),
            "config_path": str(args.output_dir / "config.json"),
            "metrics_path": str(args.output_dir / "metrics.json"),
            "summary_path": str(summary_path),
            "best_checkpoint": str(args.output_dir / "best.pt"),
            "latest_checkpoint": str(args.output_dir / "last.pt"),
            "history_length": len(history),
            "best_selection_score": float(best_score),
            "best_subject_box_score": float(best_subject_box_score),
            "selection_metric": str(args.selection_metric),
        }
        if last_row is not None:
            payload["last_row"] = last_row
        train_status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_progress_status(progress_payload: dict[str, Any]) -> None:
        payload: dict[str, Any] = {
            "state": "running",
            "phase": str(progress_payload.get("phase", "")),
            "start_time_unix": start,
            "last_update_time_unix": time(),
            "current_epoch": int(progress_payload.get("epoch", 0) or 0),
            "current_step": int(progress_payload.get("step", 0) or 0),
            "current_samples": int(progress_payload.get("samples", 0) or 0),
            "phase_elapsed_sec": float(progress_payload.get("elapsed_sec", 0.0) or 0.0),
            "phase_data_wait_sec": float(progress_payload.get("data_wait_sec", 0.0) or 0.0),
            "phase_compute_sec": float(progress_payload.get("compute_sec", 0.0) or 0.0),
            "total_epochs": int(args.epochs),
            "output_dir": str(args.output_dir),
            "config_path": str(args.output_dir / "config.json"),
            "metrics_path": str(args.output_dir / "metrics.json"),
            "summary_path": str(summary_path),
            "best_checkpoint": str(args.output_dir / "best.pt"),
            "latest_checkpoint": str(args.output_dir / "last.pt"),
            "history_length": len(history),
            "best_selection_score": float(best_score),
            "best_subject_box_score": float(best_subject_box_score),
            "selection_metric": str(args.selection_metric),
        }
        train_status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_train_status(state="running", epoch=0, last_row=None)
    terminal_state = "completed"
    for epoch in range(1, args.epochs + 1):
        lr = resolve_epoch_lr(
            base_lr=args.lr,
            min_lr=args.min_lr,
            scheduler=args.scheduler,
            warmup_epochs=args.warmup_epochs,
            epoch=epoch,
            epochs=args.epochs,
        )
        _set_optimizer_lr(optimizer, lr)
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp=args.amp,
            limit_steps=args.limit_train_steps,
            grad_clip_norm=args.grad_clip_norm,
            loss_kwargs=loss_kwargs,
            phase="train",
            epoch=epoch,
            progress_log_interval=args.progress_log_interval,
            pred_subject_prior_start_epoch=args.pred_subject_prior_start_epoch,
            pred_subject_prior_warmup_epochs=args.pred_subject_prior_warmup_epochs,
            route_expert_distill_models=route_expert_distill_models,
            route_expert_distill_weights=route_expert_distill_weights,
            route_expert_distill_cache=route_expert_distill_cache,
            progress_callback=_write_progress_status,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                device=device,
                optimizer=None,
                scaler=scaler,
                amp=args.amp,
                limit_steps=args.limit_val_steps,
                grad_clip_norm=args.grad_clip_norm,
                loss_kwargs=loss_kwargs,
                phase="val",
                epoch=epoch,
                progress_log_interval=args.progress_log_interval,
                pred_subject_prior_start_epoch=args.pred_subject_prior_start_epoch,
                pred_subject_prior_warmup_epochs=args.pred_subject_prior_warmup_epochs,
                route_expert_distill_models=route_expert_distill_models,
                route_expert_distill_weights=route_expert_distill_weights,
                route_expert_distill_cache=route_expert_distill_cache,
                progress_callback=_write_progress_status,
            )
        selection_score = _selection_score(val_metrics, args.selection_metric)
        row = {
            "epoch": epoch,
            "lr": lr,
            "train": train_metrics,
            "val": val_metrics,
            "selection_metric": args.selection_metric,
            "selection_score": selection_score,
            "elapsed_sec": time() - start,
        }
        history.append(row)
        (args.output_dir / "metrics.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        save_checkpoint(args.output_dir / "last.pt", model=model, optimizer=optimizer, epoch=epoch, metrics=row, config=config)

        if selection_score > best_score + float(args.early_stop_min_delta):
            best_score = selection_score
            epochs_since_improve = 0
            save_checkpoint(args.output_dir / "best.pt", model=model, optimizer=optimizer, epoch=epoch, metrics=row, config=config)
        else:
            epochs_since_improve += 1
        subject_box_score = (
            0.50 * float(val_metrics.get("subject_box_iou", 0.0))
            + 0.30 * float(val_metrics.get("subject_box_valid_balanced_acc", val_metrics.get("subject_box_valid_acc", 0.0)))
            + 0.20 * float(val_metrics.get("subject_box_valid_f1", 0.0))
        )
        if math.isfinite(subject_box_score) and subject_box_score > best_subject_box_score + float(args.early_stop_min_delta):
            best_subject_box_score = subject_box_score
            save_checkpoint(args.output_dir / "subject_box_best.pt", model=model, optimizer=optimizer, epoch=epoch, metrics=row, config=config)
        _write_train_status(state="running", epoch=epoch, last_row=row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if args.early_stop_patience > 0 and epochs_since_improve >= args.early_stop_patience:
            terminal_state = "early_stopped"
            break
    last_row = history[-1] if history else None
    _write_terminal_summary(state=terminal_state, last_row=last_row)
    _write_train_status(state=terminal_state, epoch=len(history), last_row=last_row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
