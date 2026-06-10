from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from mobilecropnet_v4.data import (
    CHECKLIST_CLASS_KEYS,
    CHECKLIST_CLASS_TOTAL,
    DECISION_VOCAB,
    TARGET_AR_VOCAB,
    WHY_TAG_VOCAB,
    MobileCropNetV4BatchDataset,
    compute_letterbox_transform,
    letterbox_to_original_box,
    mobilecropnet_v4_collate,
)
from mobilecropnet_v4.eval_utils import (
    build_runtime_baselines,
    detailed_explanation_from_logits,
    execute_runtime_decision,
    select_generated_proposal_index,
    subject_valid_from_policy,
)
from mobilecropnet_v4.model import MobileCropNetV4, compute_mobilecropnet_v4_loss
from scripts.evaluate_mobilecropnet_v4 import main as evaluate_v4_main
from scripts.evaluate_mobilecropnet_v4_product_ar_direct import main as evaluate_v4_direct_main
from scripts.compare_mobilecropnet_v4_methods import main as compare_v4_main
from scripts.infer_mobilecropnet_v4 import main as infer_v4_main
from scripts.prepare_mobilecropnet_v4_splits import main as prepare_v4_splits_main
from scripts.train_mobilecropnet_v4 import _apply_model_profile, _resolve_image_norm_args, _selection_score, build_parser, main as train_v4_main
from scripts.build_mobilecropnet_v4_qualitative_review_pack import main as build_review_pack_main
from scripts.visualize_mobilecropnet_v4_predictions import _label_with_score, _teacher_detail_payload
from scripts.visualize_mobilecropnet_v4_predictions import main as visualize_v4_main


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (80, 60), color=color).save(path)


def _row(image_id: str, image_path: str, target_ar: str = "FREE", subject_mode_id: int = 4) -> dict:
    return {
        "schema_version": "sstk_conditional_detr_batch_v3",
        "image_id": image_id,
        "image_path": image_path,
        "target_ar": target_ar,
        "routing": {
            "subject_mode_id": subject_mode_id,
            "subject_prior_bbox_norm_xyxy": [0.12, 0.08, 0.88, 0.9],
            "flags": {"subject_reliability": 0.75},
        },
        "baseline": {
            "candidate_id": f"{target_ar}_full",
            "bbox_norm_xyxy": [0.0, 0.0, 1.0, 1.0],
            "score_policy": 0.45,
            "crop_utility_raw": 0.45,
        },
        "decision_target": {"decision_type": "crop", "decision_id": 2, "delta_vs_base": 0.35},
        "matching_targets": [
            {
                "target_id": "pos0",
                "candidate_id": "pos0",
                "bbox_norm_xyxy": [0.18, 0.12, 0.82, 0.78],
                "score_targets": {"crop_utility_prob": 0.92, "score_prob": 0.9},
                "macro_targets": {"aesthetic_prob": 0.8, "subject_prob": 0.9, "composition_prob": 0.7, "technical_prob": 0.8},
                "checklist_labels": {
                    "subject_coverage": "good",
                    "subject_scale": "ideal_scale",
                    "headroom": "headroom_ok",
                    "lookroom": "lookroom_adequate",
                    "face_cut": "no_face_cut",
                    "joint_cut": "no_joint_cut",
                    "copyspace": "copyspace_preserved",
                    "context": "context_preserved",
                    "third_dist": "rule_of_thirds_strong",
                    "phi_dist": "phi_grid_weak",
                    "center_dist": "center_comp_weak",
                    "ar": "ar_fits_well",
                },
                "checklist_scores": {
                    "subject_coverage_ratio": 0.8,
                    "subject_scale_ratio": 0.6,
                    "headroom_ratio": 0.08,
                    "lookroom_ratio": 1.4,
                    "third_dist": 0.04,
                    "phi_dist": 0.16,
                    "center_dist": 0.22,
                    "horizon_y": 0.52,
                    "horizon_visible_ratio": 0.4,
                    "context_value": 0.7,
                    "symmetry_score": 0.75,
                    "copyspace_blank_ratio_keep": 0.2,
                },
                "composition_focus": {
                    "placement_score": 0.7,
                    "placement_family_margin": 0.3,
                    "placement_reward_third": 0.9,
                    "placement_reward_phi": 0.4,
                    "placement_reward_center": 0.2,
                },
                "macro_components": {"C_headroom": 0.8, "C_lookroom": 0.75, "C_context": 0.7},
                "safety_penalty": {"soft_total": 0.0, "hard_total": 0.0},
                "why_tags": ["rule_of_thirds", "avoid_face_cut", "context_preserved", "headroom_ok", "lookroom_ok"],
            }
        ],
        "candidate_pool": [
            {
                "candidate_id": "soft0",
                "bbox_norm_xyxy": [0.2, 0.15, 0.84, 0.8],
                "score_targets": {"crop_utility_prob": 0.76},
                "is_soft_positive": True,
                "is_positive_candidate": False,
                "is_hard_negative": False,
                "is_unsafe_negative": False,
                "is_ignore_candidate": False,
                "is_overflow_candidate": False,
            },
            {
                "candidate_id": "neg0",
                "bbox_norm_xyxy": [0.0, 0.0, 0.38, 0.38],
                "score_targets": {"crop_utility_prob": 0.08},
                "is_positive_candidate": False,
                "is_hard_negative": True,
                "is_unsafe_negative": False,
                "is_ignore_candidate": False,
                "is_overflow_candidate": False,
            },
        ],
        "ignored_candidates": [],
        "overflow_candidates": [],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_subject_valid_route_soft_policies() -> None:
    assert subject_valid_from_policy(confidence=0.9, threshold=0.5, route_label="scene_general", policy="route_and_conf") is False
    assert (
        subject_valid_from_policy(confidence=0.9, threshold=0.5, route_label="scene_general", policy="route_scene_and_conf")
        is True
    )
    assert (
        subject_valid_from_policy(
            confidence=0.2, threshold=0.5, route_label="scene_general", policy="route_scene_or_conf"
        )
        is True
    )
    assert (
        subject_valid_from_policy(
            confidence=0.9, threshold=0.5, route_label="background_texture_copyspace", policy="route_scene_and_conf"
        )
        is False
    )
    assert (
        subject_valid_from_policy(
            confidence=0.9,
            threshold=0.5,
            route_label="background_texture_copyspace",
            policy="route_scene_background_and_conf",
        )
        is True
    )


def test_mobilecropnet_v4_dataset_forward_loss(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_image(image_dir / "a.jpg", (140, 80, 60))
    _write_image(image_dir / "b.jpg", (30, 120, 180))
    jsonl = tmp_path / "batch.jsonl"
    _write_jsonl(jsonl, [_row("a", "images/a.jpg"), _row("b", "images/b.jpg", "1:1")])
    pairwise_jsonl = tmp_path / "pairwise.jsonl"
    _write_jsonl(
        pairwise_jsonl,
        [
            {
                "image_id": "a",
                "target_ar": "FREE",
                "candidate_id_a": "pos0",
                "candidate_id_b": "neg0",
                "label": 1,
                "score_margin": 0.5,
                "crop_utility_margin": 0.5,
                "pair_type": "top1_vs_negative",
            }
        ],
    )
    listwise_jsonl = tmp_path / "listwise.jsonl"
    _write_jsonl(
        listwise_jsonl,
        [
            {
                "image_id": "a",
                "target_ar": "FREE",
                "candidates": [
                    {"candidate_id": "pos0", "crop_utility_rank_pct": 1.0, "crop_utility_softmax_local": 0.8},
                    {"candidate_id": "neg0", "crop_utility_rank_pct": 0.0, "crop_utility_softmax_local": 0.1},
                ],
            }
        ],
    )

    ds = MobileCropNetV4BatchDataset(jsonl_path=jsonl, project_root=tmp_path, input_size=64, candidate_k=6)
    ds_pre = MobileCropNetV4BatchDataset(
        jsonl_path=jsonl,
        project_root=tmp_path,
        input_size=64,
        candidate_k=6,
        precompute_sample_tensors=True,
        image_tensor_cache_size=2,
    )
    assert ds.summary()["image_count"] == 2
    assert ds.summary()["subject_mode_counts"]["portrait_single"] == 2
    assert ds.summary()["decision_type_counts"]["crop"] == 2
    assert ds.summary()["subject_prior_box_rate"] == 1.0
    assert ds_pre.summary()["precompute_sample_tensors"] is True
    assert ds_pre.summary()["image_tensor_cache_size"] == 2
    ds_rank = MobileCropNetV4BatchDataset(
        jsonl_path=jsonl,
        project_root=tmp_path,
        input_size=64,
        candidate_k=6,
        pairwise_jsonl=pairwise_jsonl,
        listwise_jsonl=listwise_jsonl,
        score_target_mode="hybrid_rank",
        max_pairwise_pairs=4,
    )
    assert ds_rank.summary()["pairwise_label_group_count"] == 1
    assert ds_rank[0]["pair_valid"].sum().item() == 1
    assert ds_rank[0]["teacher_soft_target"].sum().item() > 0
    assert ds_rank[0]["candidate_weight"].shape == (6,)
    assert ds_rank[0]["checklist_class_valid"].sum().item() > 0
    pos_slot = int(ds_rank[0]["valid"].nonzero()[1].item()) if ds_rank[0]["valid"].ndim == 2 else 1
    assert ds_rank[0]["checklist_class_valid"][pos_slot].sum().item() == 12
    assert ds_rank[0]["checklist_applicability_target"].shape == (6, len(CHECKLIST_CLASS_KEYS))
    assert ds_rank[0]["checklist_applicability_valid"][pos_slot].sum().item() == len(CHECKLIST_CLASS_KEYS)
    assert ds_rank[0]["detail_score_valid"].sum().item() > 0
    assert ds_rank[0]["why_tag_valid"].sum().item() > 0
    assert ds_rank[0]["subject_prior_valid"].item() == 1.0
    assert ds_rank[0]["subject_prior_reliability"].item() > 0.0
    bg_jsonl = tmp_path / "batch_background.jsonl"
    _write_jsonl(bg_jsonl, [_row("bg", "images/a.jpg", subject_mode_id=0)])
    ds_bg_default = MobileCropNetV4BatchDataset(jsonl_path=bg_jsonl, project_root=tmp_path, input_size=64, candidate_k=6)
    ds_bg_reliable = MobileCropNetV4BatchDataset(
        jsonl_path=bg_jsonl,
        project_root=tmp_path,
        input_size=64,
        candidate_k=6,
        subject_box_valid_target_mode="box_reliability",
    )
    assert ds_bg_default[0]["subject_box_valid"].item() == 0.0
    assert ds_bg_reliable[0]["subject_box_valid"].item() == 1.0
    assert ds_bg_reliable.summary()["subject_box_valid_target_mode"] == "box_reliability"
    low_rel_jsonl = tmp_path / "batch_low_reliability.jsonl"
    low_rel_row = _row("low_rel", "images/a.jpg", subject_mode_id=4)
    low_rel_row["routing"]["flags"]["subject_reliability"] = 0.10
    _write_jsonl(low_rel_jsonl, [low_rel_row])
    ds_fg_default = MobileCropNetV4BatchDataset(jsonl_path=low_rel_jsonl, project_root=tmp_path, input_size=64, candidate_k=6)
    ds_fg_route_reliable = MobileCropNetV4BatchDataset(
        jsonl_path=low_rel_jsonl,
        project_root=tmp_path,
        input_size=64,
        candidate_k=6,
        subject_box_valid_target_mode="route_reliable",
    )
    assert ds_fg_default[0]["subject_box_valid"].item() == 1.0
    assert ds_fg_route_reliable[0]["subject_box_valid"].item() == 0.0
    assert ds_fg_route_reliable.summary()["subject_box_valid_target_mode"] == "route_reliable"
    uncached = ds[0]
    cached = ds_pre[0]
    for key in ("boxes", "box_meta", "score_target", "positive_boxes", "target_ar_id", "image_ar_log"):
        assert torch.allclose(uncached[key], cached[key])
    batch = mobilecropnet_v4_collate([ds[0], ds[1]])
    assert batch["image"].shape == (2, 3, 64, 64)
    assert batch["boxes"].shape == (2, 6, 4)
    assert batch["letterbox_content_box"].shape == (2, 4)
    assert batch["positive_valid"].sum().item() >= 2

    model = MobileCropNetV4(
        candidate_k=6,
        proposal_q=4,
        width_mult=0.5,
        token_dim=64,
        use_subject_prior=True,
        route_use_subject_prior=True,
        route_use_candidate_context=True,
        route_use_subject_box_features=True,
        route_use_subject_spatial_token=True,
        policy_use_subject_prior=True,
        proposal_use_subject_prior=True,
        subject_box_head=True,
        subject_box_spatial_head=True,
        subject_box_spatial_output="spatial",
        subject_box_spatial_box_mode="mask_moment_regress",
        subject_box_coord_space="content",
        subject_box_refine_with_proposals=True,
        use_pred_subject_box_as_prior=True,
        policy_score_head=True,
    )
    outputs = model(
        batch["image"],
        batch["boxes"],
        batch["valid"],
        batch["target_ar_id"],
        batch["image_ar_log"],
        batch["candidate_is_base"],
        batch["box_meta"],
        batch["letterbox_content_box"],
        batch["subject_prior_box"],
        batch["subject_prior_valid"],
        batch["subject_prior_reliability"],
        return_generated_scores=True,
    )
    assert outputs["utility_logits"].shape == (2, 6)
    assert torch.allclose(outputs["return_logits"], outputs["utility_logits"])
    assert outputs["proposal_boxes"].shape == (2, 4, 4)
    assert outputs["checklist_class_logits"].shape[:2] == (2, 6)
    assert outputs["checklist_applicability_logits"].shape[:2] == (2, 6)
    assert outputs["detail_score_logits"].shape[:2] == (2, 6)
    assert outputs["why_tag_logits"].shape[:2] == (2, 6)
    assert outputs["policy_score_pred"].shape == (2,)
    assert outputs["pred_subject_box"].shape == (2, 4)
    assert torch.all(outputs["pred_subject_box"] >= 0.0)
    assert torch.all(outputs["pred_subject_box"] <= 1.0)
    assert outputs["pred_subject_spatial_heatmap_logits"].ndim == 2
    assert outputs["pred_subject_box_coarse"].shape == (2, 4)
    assert outputs["pred_subject_box_spatial"].shape == (2, 4)
    assert outputs["pred_subject_spatial_mix"].shape == (2,)
    assert outputs["subject_proposal_logits"].shape == (2, 4)
    assert outputs["generated_utility_logits"].shape == (2, 4)
    fixed_ar_boxes = outputs["proposal_boxes"][1].detach()
    fixed_ar_ratio = (fixed_ar_boxes[:, 2] - fixed_ar_boxes[:, 0]) / (fixed_ar_boxes[:, 3] - fixed_ar_boxes[:, 1]).clamp_min(1e-6)
    assert torch.allclose(fixed_ar_ratio, torch.ones_like(fixed_ar_ratio), atol=1e-4)
    loss, metrics = compute_mobilecropnet_v4_loss(
        outputs,
        batch,
        top_return_weight=0.5,
        return_positive_weight=0.2,
        return_risk_suppression_weight=0.2,
        teacher_distill_weight=0.2,
        topk_coverage_weight=0.1,
        action_consistency_weight=0.2,
        decision_utility_align_weight=0.1,
        policy_score_weight=0.2,
        route_balanced_ce=True,
        route_object_single_margin_weight=0.1,
        route_object_single_margin=0.18,
        proposal_subject_weight=0.2,
        subject_proposal_align_weight=0.1,
        subject_box_weight=0.2,
        subject_box_center_weight=0.1,
        subject_box_ciou_weight=0.2,
        subject_box_spatial_aux_weight=0.2,
        generated_proposal_align_weight=0.2,
        generated_proposal_positive_utility_weight=0.3,
        generated_proposal_positive_margin_weight=0.2,
    )
    assert torch.isfinite(loss)
    assert 0.0 <= metrics["top1_hit"] <= 1.0
    assert "return_top1_hit" in metrics
    assert "top_return_hit" in metrics
    assert "return_positive_loss" in metrics
    assert "return_risk_suppression_loss" in metrics
    assert "return_risk_suppression_acc" in metrics
    assert "teacher_distill_loss" in metrics
    assert "topk_coverage_loss" in metrics
    assert "route_object_single_margin_loss" in metrics
    assert "action_consistency_loss" in metrics
    assert "baseline_preserve_top1_hit" in metrics
    assert "decision_source_acc" in metrics
    assert "decision_conditioned_crop_action_top1_hit" in metrics
    assert "decision_conditioned_top_return_hit" in metrics
    assert "decision_utility_align_loss" in metrics
    assert "decision_utility_align_gap" in metrics
    assert "policy_score_loss" in metrics
    assert "proposal_subject_loss" in metrics
    assert "subject_proposal_align_loss" in metrics
    assert "subject_box_center_loss" in metrics
    assert "subject_box_ciou_loss" in metrics
    assert "subject_box_spatial_aux_loss" in metrics
    assert "subject_box_spatial_iou" in metrics
    assert "subject_box_spatial_mix" in metrics
    assert "subject_box_valid_best_balanced_acc" in metrics
    assert "subject_box_valid_best_threshold" in metrics
    assert "generated_align_loss" in metrics
    assert "generated_align_positive_utility_loss" in metrics
    assert "generated_align_positive_margin_loss" in metrics
    assert "generated_align_any_positive_0_5" in metrics
    assert 0.0 <= metrics["top_return_hit"] <= 1.0

    return_model = MobileCropNetV4(candidate_k=6, proposal_q=4, width_mult=0.5, token_dim=64, return_score_head=True)
    return_outputs = return_model(
        batch["image"],
        batch["boxes"],
        batch["valid"],
        batch["target_ar_id"],
        batch["image_ar_log"],
        batch["candidate_is_base"],
        batch["box_meta"],
        batch["letterbox_content_box"],
        return_generated_scores=True,
    )
    assert return_outputs["return_logits"].shape == return_outputs["utility_logits"].shape
    assert torch.allclose(return_outputs["return_logits"], return_outputs["utility_logits"])
    assert return_outputs["generated_return_logits"].shape == return_outputs["generated_utility_logits"].shape

    context_return_model = MobileCropNetV4(
        candidate_k=6,
        proposal_q=4,
        width_mult=0.5,
        token_dim=64,
        return_score_head=True,
        return_score_action_source_bias=True,
        return_score_context_head=True,
    )
    context_return_outputs = context_return_model(
        batch["image"],
        batch["boxes"],
        batch["valid"],
        batch["target_ar_id"],
        batch["image_ar_log"],
        batch["candidate_is_base"],
        batch["box_meta"],
        batch["letterbox_content_box"],
        return_generated_scores=True,
    )
    assert context_return_outputs["return_logits"].shape == context_return_outputs["utility_logits"].shape
    assert torch.allclose(context_return_outputs["return_logits"], context_return_outputs["utility_logits"])
    assert context_return_outputs["generated_return_logits"].shape == context_return_outputs["generated_utility_logits"].shape

    conditioned_model = MobileCropNetV4(
        candidate_k=6,
        proposal_q=4,
        width_mult=0.5,
        token_dim=64,
        return_score_head=True,
        return_score_decision_conditioned=True,
    )
    conditioned_outputs = conditioned_model(
        batch["image"],
        batch["boxes"],
        batch["valid"],
        batch["target_ar_id"],
        batch["image_ar_log"],
        batch["candidate_is_base"],
        batch["box_meta"],
        batch["letterbox_content_box"],
        return_generated_scores=True,
    )
    assert conditioned_outputs["decision_conditioned_return_logits"].shape == conditioned_outputs["return_logits"].shape
    crop_id = DECISION_VOCAB.index("crop")
    for row_idx, decision_id in enumerate(conditioned_outputs["decision_logits"].argmax(dim=1).tolist()):
        keep_base = int(decision_id) != crop_id
        masked_source = batch["candidate_is_base"][row_idx] <= 0 if keep_base else batch["candidate_is_base"][row_idx] > 0
        masked_source = masked_source & (batch["valid"][row_idx] > 0)
        assert torch.all(conditioned_outputs["decision_conditioned_return_logits"][row_idx][masked_source] < -999.0)

    mixture_model = MobileCropNetV4(
        candidate_k=6,
        proposal_q=4,
        width_mult=0.5,
        token_dim=64,
        return_score_head=True,
        return_score_source_mixture=True,
        return_score_source_mixture_strength=0.5,
    )
    mixture_outputs = mixture_model(
        batch["image"],
        batch["boxes"],
        batch["valid"],
        batch["target_ar_id"],
        batch["image_ar_log"],
        batch["candidate_is_base"],
        batch["box_meta"],
        batch["letterbox_content_box"],
        return_generated_scores=True,
    )
    assert mixture_outputs["source_mixture_return_logits"].shape == mixture_outputs["return_logits"].shape
    assert mixture_outputs["generated_source_mixture_return_logits"].shape == mixture_outputs["generated_return_logits"].shape
    mixture_loss, mixture_metrics = compute_mobilecropnet_v4_loss(mixture_outputs, batch)
    assert torch.isfinite(mixture_loss)
    assert "source_mixture_crop_action_top1_hit" in mixture_metrics

    source_gate_model = MobileCropNetV4(
        candidate_k=6,
        proposal_q=4,
        width_mult=0.5,
        token_dim=64,
        return_score_head=True,
        return_score_source_gate=True,
        return_score_source_gate_subject_state=True,
        return_score_source_gate_logit_threshold=0.0,
        subject_box_head=True,
    )
    source_gate_model.eval()
    source_gate_outputs = source_gate_model(
        batch["image"],
        batch["boxes"],
        batch["valid"],
        batch["target_ar_id"],
        batch["image_ar_log"],
        batch["candidate_is_base"],
        batch["box_meta"],
        batch["letterbox_content_box"],
        return_generated_scores=True,
    )
    assert source_gate_outputs["source_gate_logit"].shape == (2,)
    assert source_gate_outputs["source_gate_return_logits"].shape == source_gate_outputs["return_logits"].shape
    threshold_source_gate_model = MobileCropNetV4(
        candidate_k=6,
        proposal_q=4,
        width_mult=0.5,
        token_dim=64,
        return_score_head=True,
        return_score_source_gate=True,
        return_score_source_gate_subject_state=True,
        return_score_source_gate_logit_threshold=0.75,
        subject_box_head=True,
    )
    threshold_source_gate_model.load_state_dict(source_gate_model.state_dict())
    threshold_source_gate_model.eval()
    threshold_outputs = threshold_source_gate_model(
        batch["image"],
        batch["boxes"],
        batch["valid"],
        batch["target_ar_id"],
        batch["image_ar_log"],
        batch["candidate_is_base"],
        batch["box_meta"],
        batch["letterbox_content_box"],
        return_generated_scores=True,
    )
    assert torch.allclose(threshold_outputs["source_gate_logit"], source_gate_outputs["source_gate_logit"] - 0.75, atol=1e-5)
    source_gate_loss, source_gate_metrics = compute_mobilecropnet_v4_loss(
        source_gate_outputs,
        batch,
        source_gate_supervision_weight=0.2,
        source_gate_balanced_bce=True,
        source_gate_margin_weight=0.1,
        source_gate_margin_balanced=True,
    )
    assert torch.isfinite(source_gate_loss)
    assert "source_gate_source_acc" in source_gate_metrics
    assert "source_gate_margin_acc" in source_gate_metrics
    assert "source_gate_crop_action_top1_hit" in source_gate_metrics

    action_replace_model = MobileCropNetV4(
        candidate_k=6,
        proposal_q=4,
        width_mult=0.5,
        token_dim=64,
        return_score_head=True,
        return_score_action_decoder_head=True,
        return_score_action_decoder_replace=True,
        return_score_action_decoder_subject_state=True,
        subject_box_head=True,
    )
    action_replace_outputs = action_replace_model(
        batch["image"],
        batch["boxes"],
        batch["valid"],
        batch["target_ar_id"],
        batch["image_ar_log"],
        batch["candidate_is_base"],
        batch["box_meta"],
        batch["letterbox_content_box"],
        return_generated_scores=True,
    )
    assert action_replace_outputs["return_logits"].shape == action_replace_outputs["utility_logits"].shape
    assert not torch.allclose(action_replace_outputs["return_logits"], action_replace_outputs["utility_logits"])
    assert action_replace_outputs["generated_return_logits"].shape == action_replace_outputs["generated_utility_logits"].shape
    action_replace_loss, action_replace_metrics = compute_mobilecropnet_v4_loss(
        action_replace_outputs,
        batch,
        return_target_mode="decision_source",
        action_return_joint_weight=0.3,
        return_source_margin_weight=0.2,
    )
    assert torch.isfinite(action_replace_loss)
    assert "action_return_joint_crop_hit" in action_replace_metrics
    assert "return_source_margin_crop_hit" in action_replace_metrics
    assert _selection_score(
        {
            "source_mixture_top_return_hit": 0.9,
            "decision_conditioned_top_return_hit": 0.1,
            "top_return_hit": 0.1,
            "source_mixture_top_return_score": 0.9,
            "source_mixture_top_return_regret": 0.1,
            "generated_align_top1_hit": 0.8,
            "generated_align_top1_iou": 0.8,
            "proposal_recall_0_5": 0.9,
            "route_acc": 0.8,
            "subject_box_iou": 0.7,
            "subject_box_valid_balanced_acc": 0.7,
            "source_mixture_action_source_acc": 0.8,
            "decision_acc": 0.8,
        },
        "release_gate",
    ) > 0.0

    mixed_batch = {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}
    mixed_batch["decision_target"] = torch.tensor([0, 2], dtype=torch.long)
    mixed_loss, mixed_metrics = compute_mobilecropnet_v4_loss(
        outputs,
        mixed_batch,
        decision_crop_rebalance=True,
        decision_utility_align_weight=0.1,
    )
    assert torch.isfinite(mixed_loss)
    assert mixed_metrics["decision_crop_class_weight"] >= 1.0
    assert "decision_utility_align_acc" in mixed_metrics


def test_select_generated_proposal_index_prefers_proposal_shortlist() -> None:
    utility = [0.9, 0.8, 0.2]
    proposal = [0.1, 0.95, 0.8]
    proposals = [
        {"target_ar_compatible": True},
        {"target_ar_compatible": True},
        {"target_ar_compatible": False},
    ]
    assert select_generated_proposal_index(utility_scores=utility, proposal_scores=proposal, selection_policy="utility_top1") == 0
    assert select_generated_proposal_index(utility_scores=utility, proposal_scores=proposal, selection_policy="proposal_top1") == 1
    assert (
        select_generated_proposal_index(
            utility_scores=utility,
            proposal_scores=proposal,
            proposals=proposals,
            selection_policy="proposal_topk_rerank",
            proposal_top_m=2,
        )
        == 1
    )


def test_runtime_baselines_and_executor_align_to_decision() -> None:
    baselines = build_runtime_baselines(width=1200, height=800, target_ar="1:1")
    strong_baselines = build_runtime_baselines(width=1200, height=800, target_ar="1:1")
    full = baselines["full"]["bbox_norm_xyxy"]
    minimal = baselines["minimal"]["bbox_norm_xyxy"]
    baselines["full"]["utility_score"] = 0.25
    baselines["minimal"]["utility_score"] = 0.30
    strong_baselines["full"]["utility_score"] = 0.95
    strong_baselines["minimal"]["utility_score"] = 0.30
    assert full == [0.0, 0.0, 1.0, 1.0]
    assert minimal[1] == 0.0 and minimal[3] == 1.0
    assert abs((minimal[2] - minimal[0]) - (800.0 / 1200.0)) < 1e-6

    proposals = [{"proposal_id": 0, "bbox_norm_xyxy": [0.1, 0.1, 0.8, 0.9], "target_ar_compatible": False, "utility_score": 0.82}]
    keep = execute_runtime_decision(
        decision_id=0,
        target_ar="1:1",
        width=1200,
        height=800,
        proposals=proposals,
        selected_proposal_index=0,
        baseline_rows=strong_baselines,
        exact_target_ar_postprocess=True,
        decision_logits=[4.0, 0.2, -1.0],
    )
    crop = execute_runtime_decision(
        decision_id=2,
        target_ar="1:1",
        width=1200,
        height=800,
        proposals=proposals,
        selected_proposal_index=0,
        baseline_rows=baselines,
        exact_target_ar_postprocess=True,
        decision_logits=[-1.0, 0.1, 4.0],
    )
    low_conf_keep = execute_runtime_decision(
        decision_id=0,
        target_ar="1:1",
        width=1200,
        height=800,
        proposals=proposals,
        selected_proposal_index=0,
        baseline_rows=baselines,
        exact_target_ar_postprocess=True,
        decision_logits=[0.2, 0.1, 0.0],
        enforce_baseline_decision_gate=True,
    )
    assert keep["action_source"] == "baseline_full"
    assert crop["action_source"] == "0"
    assert keep["target_ar_compatible"] is True
    assert crop["target_ar_compatible"] is True
    assert keep["executor_decision_consistent"] == 1.0
    assert crop["executor_decision_consistent"] == 1.0
    assert keep["executor_override_to_crop"] is False
    assert keep["executor_gate_reasons"] == []
    assert low_conf_keep["action_source"] == "0"
    assert low_conf_keep["executor_override_to_crop"] is True
    assert "low_decision_confidence" in low_conf_keep["executor_gate_reasons"]


def test_mobilecropnet_v4_object_mode_explanation_masks_person_terms() -> None:
    checklist_logits = torch.zeros((CHECKLIST_CLASS_TOTAL,), dtype=torch.float32)
    checklist_app = torch.full((len(CHECKLIST_CLASS_KEYS),), 8.0, dtype=torch.float32)
    why_logits = torch.full((len(WHY_TAG_VOCAB),), -8.0, dtype=torch.float32)
    why_logits[WHY_TAG_VOCAB.index("avoid_person_cut")] = 9.0
    why_logits[WHY_TAG_VOCAB.index("rule_of_thirds")] = 8.5
    detail = detailed_explanation_from_logits(
        checklist_logits=checklist_logits,
        checklist_applicability_logits=checklist_app,
        why_tag_logits=why_logits,
        subject_mode_label="object_single",
    )
    assert detail["detailed_checklist"]["headroom"]["display_label"] == "not_applicable"
    assert detail["detailed_checklist"]["lookroom"]["display_label"] == "not_applicable"
    assert "avoid_person_cut" not in {item["tag"] for item in detail["why_tags"]}
    assert "rule_of_thirds" in {item["tag"] for item in detail["why_tags"]}


def test_mobilecropnet_v4_fixed_ar_proposals_stay_inside_letterbox_content() -> None:
    model = MobileCropNetV4(candidate_k=4, proposal_q=8, width_mult=0.5, token_dim=64)
    model.eval()
    image = torch.zeros((1, 3, 64, 64), dtype=torch.float32)
    target_ar_id = torch.tensor([TARGET_AR_VOCAB.index("16:9")], dtype=torch.long)
    transform = compute_letterbox_transform(width=480, height=160, input_size=64)
    content_box = torch.tensor([[float(transform.pad_x) / transform.input_size, float(transform.pad_y) / transform.input_size, float(transform.pad_x + transform.resized_width) / transform.input_size, float(transform.pad_y + transform.resized_height) / transform.input_size]])
    with torch.no_grad():
        outputs = model(image, target_ar_id=target_ar_id, image_ar_log=torch.tensor([transform.image_ar_log]), letterbox_content_box=content_box)
    boxes = outputs["proposal_boxes"][0]
    assert outputs["runtime_baseline_boxes"].shape == (1, 2, 4)
    y0 = float(transform.pad_y) / float(transform.input_size)
    y1 = float(transform.pad_y + transform.resized_height) / float(transform.input_size)
    assert torch.all(boxes[:, 1] >= y0 - 1e-5)
    assert torch.all(boxes[:, 3] <= y1 + 1e-5)
    for box in boxes.tolist():
        orig = letterbox_to_original_box(box, transform)
        crop_w = (orig[2] - orig[0]) * float(transform.original_width)
        crop_h = (orig[3] - orig[1]) * float(transform.original_height)
        assert abs((crop_w / max(1e-6, crop_h)) - (16.0 / 9.0)) < 1e-3


def test_mobilecropnet_v4_profile_presets_use_pretrained_backbones() -> None:
    expected = {
        "hq_320": ("mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k", [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "balanced_288": ("mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k", [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "turbo_256": ("mobilenetv4_conv_small.e3600_r256_in1k", [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        "plus_384": ("convnextv2_tiny.fcmae_ft_in22k_in1k_384", [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "quality_cnv2b_448": ("convnextv2_base.fcmae_ft_in22k_in1k_384", [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "quality_cnv2l_512": ("convnextv2_large.fcmae_ft_in22k_in1k_384", [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        "quality_eva02b_subject_448": ("eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
        "quality_swinv2b_subject_384": ("swinv2_base_window12to24_192to384.ms_in22k_ft_in1k", [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    }
    parser = build_parser()
    for profile, (backbone, mean, std) in expected.items():
        args = parser.parse_args(
            [
                "--train_jsonl",
                "train.jsonl",
                "--val_jsonl",
                "val.jsonl",
                "--output_dir",
                "out",
                "--model_profile",
                profile,
            ]
        )
        _apply_model_profile(args, {"model_profile"})
        _resolve_image_norm_args(args)
        assert args.backbone_name == backbone
        assert args.backbone_pretrained is True
        assert args.image_mean == mean
        assert args.image_std == std


def test_mobilecropnet_v4_balanced_topreturn_selection_metric() -> None:
    metrics = {
        "top_return_hit": 0.4,
        "top1_exact_best_score": 0.3,
        "top_return_score": 0.7,
        "top_return_regret": 0.2,
        "explicit_pairwise_acc": 0.8,
        "risk_suppression_acc": 0.6,
        "proposal_recall_0_5": 0.9,
        "score_mae": 0.15,
        "top1_risk_rate": 0.1,
        "listwise_loss": 0.25,
    }
    balanced = _selection_score(metrics, "sstk_balanced_topreturn")
    topreturn = _selection_score(metrics, "sstk_topreturn")
    assert balanced > 0.0
    assert balanced != topreturn
    explicit = dict(metrics)
    explicit["explicit_pairwise_count"] = 4.0
    explicit["explicit_pairwise_acc"] = 0.95
    assert _selection_score(explicit, "sstk_balanced_topreturn") > balanced
    worse = dict(metrics)
    worse["top1_risk_rate"] = 0.8
    worse["score_mae"] = 0.8
    assert _selection_score(worse, "sstk_balanced_topreturn") < balanced


def test_mobilecropnet_v4_scripts_smoke(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_image(image_dir / "a.jpg", (120, 80, 40))
    _write_image(image_dir / "b.jpg", (40, 120, 180))
    label_dir = tmp_path / "labels"
    label_dir.mkdir()
    batch_jsonl = label_dir / "train_conditional_detr_batch.jsonl"
    _write_jsonl(batch_jsonl, [_row("a", "images/a.jpg"), _row("b", "images/b.jpg", "1:1")])

    split_dir = tmp_path / "splits"
    assert (
        prepare_v4_splits_main(
            [
                "--label_dir",
                str(label_dir),
                "--output_dir",
                str(split_dir),
                "--val_fraction",
                "0.5",
            ]
        )
        == 0
    )
    train_jsonl = split_dir / "gaic_personv6_v4_train_dev.jsonl"
    val_jsonl = split_dir / "gaic_personv6_v4_val_dev.jsonl"
    assert train_jsonl.exists()
    assert val_jsonl.exists()

    run_dir = tmp_path / "run"
    assert (
        train_v4_main(
            [
                "--train_jsonl",
                str(train_jsonl),
                "--val_jsonl",
                str(val_jsonl),
                "--project_root",
                str(tmp_path),
                "--output_dir",
                str(run_dir),
                "--input_size",
                "64",
                "--candidate_k",
                "6",
                "--proposal_q",
                "4",
                "--width_mult",
                "0.5",
                "--token_dim",
                "64",
                "--use_subject_prior",
                "--route_use_subject_prior",
                "--policy_use_subject_prior",
                "--proposal_use_subject_prior",
                "--subject_box_head",
                "--subject_box_coord_space",
                "content",
                "--subject_box_refine_with_proposals",
                "--use_pred_subject_box_as_prior",
                "--pred_subject_prior_start_epoch",
                "1",
                "--subject_proposal_align_weight",
                "0.1",
                "--subject_box_weight",
                "0.2",
                "--subject_box_center_weight",
                "0.1",
                "--subject_box_ciou_weight",
                "0.2",
                "--epochs",
                "1",
                "--batch_size",
                "1",
                "--num_workers",
                "0",
                "--train_route_balanced_sampler",
                "--precompute_sample_tensors",
                "--image_tensor_cache_size",
                "2",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "subject_box_best.pt").exists()
    train_dataset_summary = json.loads((run_dir / "dataset_summary.json").read_text(encoding="utf-8"))
    assert train_dataset_summary["train_sampler"]["enabled"] is True

    eval_dir = tmp_path / "eval"
    assert (
        evaluate_v4_main(
            [
                "--checkpoint",
                str(run_dir / "best.pt"),
                "--eval_jsonl",
                str(val_jsonl),
                "--project_root",
                str(tmp_path),
                "--output_dir",
                str(eval_dir),
                "--batch_size",
                "1",
                "--num_workers",
                "0",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert (eval_dir / "metrics.json").exists()
    assert (eval_dir / "predictions.jsonl").exists()

    direct_eval_dir = tmp_path / "direct_eval"
    assert (
        evaluate_v4_direct_main(
            [
                "--checkpoint",
                str(run_dir / "best.pt"),
                "--eval_jsonl",
                str(val_jsonl),
                "--project_root",
                str(tmp_path),
                "--output_dir",
                str(direct_eval_dir),
                "--batch_size",
                "1",
                "--num_workers",
                "0",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    direct_metrics = json.loads((direct_eval_dir / "metrics.json").read_text(encoding="utf-8"))
    assert direct_metrics["runtime_subject_prior_mode"] == "none"
    assert direct_metrics["exact_target_ar_postprocess"] is True

    compare_dir = tmp_path / "compare"
    assert (
        compare_v4_main(
            [
                "--predictions_jsonl",
                str(eval_dir / "predictions.jsonl"),
                "--output_dir",
                str(compare_dir),
            ]
        )
        == 0
    )
    assert (compare_dir / "comparison_metrics.json").exists()

    viz_dir = tmp_path / "viz"
    assert (
        visualize_v4_main(
            [
                "--predictions_jsonl",
                str(eval_dir / "predictions.jsonl"),
                "--output_dir",
                str(viz_dir),
                "--sample_size",
                "1",
            ]
        )
        == 0
    )
    assert (viz_dir / "contact_sheet.png").exists()

    infer_json = tmp_path / "infer.json"
    infer_png = tmp_path / "infer.png"
    assert (
        infer_v4_main(
            [
                "--checkpoint",
                str(run_dir / "best.pt"),
                "--image",
                str(image_dir / "a.jpg"),
                "--output_json",
                str(infer_json),
                "--output_png",
                str(infer_png),
                "--input_size",
                "64",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert infer_json.exists()
    assert infer_png.exists()
    payload = json.loads(infer_json.read_text(encoding="utf-8"))
    assert payload["decision"]["label"] in {"keep_full", "minimal_crop", "crop"}
    assert "subject_mode" in payload
    assert "runtime_output" in payload
    assert "checklist" in payload["predictions"][0]
    assert "detailed_checklist" in payload["predictions"][0]
    assert "why_tags" in payload["predictions"][0]


def test_mobilecropnet_v4_detail_visualization_fallbacks() -> None:
    top = {
        "candidate_id": "baseline",
        "bbox_norm_xyxy": [0.0, 0.0, 1.0, 1.0],
        "teacher_checklist_labels": {},
        "teacher_checklist_scores": {},
    }
    row = {
        "top_candidate": top,
        "candidates": [
            top,
            {
                "candidate_id": "near_labeled",
                "bbox_norm_xyxy": [0.05, 0.05, 0.95, 0.95],
                "teacher_checklist_labels": {
                    "third_dist": "rule_of_thirds_strong",
                    "headroom": "headroom_loose",
                    "lookroom": "lookroom_excessive",
                },
                "teacher_checklist_scores": {"third_dist": 0.04, "headroom_ratio": 0.2},
                "teacher_why_tags": ["rule_of_thirds"],
            },
        ],
    }
    labels, scores, tags, source = _teacher_detail_payload(row, top)
    assert labels["third_dist"] == "rule_of_thirds_strong"
    assert scores["third_dist"] == 0.04
    assert tags == ["rule_of_thirds"]
    assert "nearest labeled candidate near_labeled" in source
    assert _label_with_score({"label": "na", "score": 0.98}) == "unlabeled p=0.98"


def test_mobilecropnet_v4_qualitative_review_pack(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    overlay_dir = run_dir / "viz_test" / "overlays"
    overlay_dir.mkdir(parents=True)
    _write_image(overlay_dir / "0001_sstk_image_1_9_16.png", (80, 40, 40))
    eval_dir = tmp_path / "eval"
    failures_dir = eval_dir / "head_analysis"
    failures_dir.mkdir(parents=True)
    _write_jsonl(
        failures_dir / "head_analysis_failures.jsonl",
        [
            {
                "severity": 6.5,
                "image_id": "sstk_image_1",
                "image_path": "data/SSTK/Full_10000/images/sstk_image_1.jpg",
                "target_ar": "9:16",
                "teacher_subject_mode": "portrait_single",
                "pred_subject_mode": "object_multi",
                "teacher_decision": "crop",
                "pred_decision": "minimal_crop",
                "proposal_recall_at_5_iou_0_5": 0.0,
                "selected_to_subject_iou": 0.2,
                "top1_iou_to_best_positive": 0.3,
                "explain_label_agreement_when_applicable": 0.0,
                "model_why_tags": [{"tag": "subject_poor", "score": 0.9}],
            }
        ],
    )
    leaderboard_json = tmp_path / "leaderboard.json"
    leaderboard_json.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "track": "SSTK public",
                        "profile": "balanced_288",
                        "run_name": "mcn-test",
                        "run_dir": str(run_dir),
                        "eval_output_dir": str(eval_dir),
                        "equal4_zscore": 1.0,
                        "head_sanity_score": 0.6,
                        "direct_alignment_score": 0.7,
                        "route_balanced_accuracy": 0.3,
                        "route_collapse_flag": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "review"
    assert (
        build_review_pack_main(
            [
                "--leaderboard_json",
                str(leaderboard_json),
                "--output_dir",
                str(out_dir),
                "--candidate_limit",
                "1",
                "--max_total_rows",
                "10",
            ]
        )
        == 0
    )
    payload = json.loads((out_dir / "qualitative_review_pack.json").read_text(encoding="utf-8"))
    assert payload["sample_count"] == 1
    assert "route_mismatch" in payload["samples"][0]["failure_buckets"]
    assert payload["samples"][0]["overlay_path"].endswith("0001_sstk_image_1_9_16.png")
    assert (out_dir / "QUALITATIVE_REVIEW_PACK.md").exists()
