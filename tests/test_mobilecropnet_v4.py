from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from mobilecropnet_v4.data import MobileCropNetV4BatchDataset, mobilecropnet_v4_collate
from mobilecropnet_v4.model import MobileCropNetV4, compute_mobilecropnet_v4_loss
from scripts.evaluate_mobilecropnet_v4 import main as evaluate_v4_main
from scripts.compare_mobilecropnet_v4_methods import main as compare_v4_main
from scripts.infer_mobilecropnet_v4 import main as infer_v4_main
from scripts.prepare_mobilecropnet_v4_splits import main as prepare_v4_splits_main
from scripts.train_mobilecropnet_v4 import main as train_v4_main
from scripts.visualize_mobilecropnet_v4_predictions import main as visualize_v4_main


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (80, 60), color=color).save(path)


def _row(image_id: str, image_path: str, target_ar: str = "FREE") -> dict:
    return {
        "schema_version": "sstk_conditional_detr_batch_v3",
        "image_id": image_id,
        "image_path": image_path,
        "target_ar": target_ar,
        "routing": {"subject_mode_id": 2},
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


def test_mobilecropnet_v4_dataset_forward_loss(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_image(image_dir / "a.jpg", (140, 80, 60))
    _write_image(image_dir / "b.jpg", (30, 120, 180))
    jsonl = tmp_path / "batch.jsonl"
    _write_jsonl(jsonl, [_row("a", "images/a.jpg"), _row("b", "images/b.jpg", "1:1")])

    ds = MobileCropNetV4BatchDataset(jsonl_path=jsonl, project_root=tmp_path, input_size=64, candidate_k=6)
    assert ds.summary()["image_count"] == 2
    batch = mobilecropnet_v4_collate([ds[0], ds[1]])
    assert batch["image"].shape == (2, 3, 64, 64)
    assert batch["boxes"].shape == (2, 6, 4)
    assert batch["positive_valid"].sum().item() >= 2

    model = MobileCropNetV4(candidate_k=6, proposal_q=4, width_mult=0.5, token_dim=64)
    outputs = model(
        batch["image"],
        batch["boxes"],
        batch["valid"],
        batch["target_ar_id"],
        batch["image_ar_log"],
        batch["candidate_is_base"],
        batch["box_meta"],
    )
    assert outputs["utility_logits"].shape == (2, 6)
    assert outputs["proposal_boxes"].shape == (2, 4, 4)
    loss, metrics = compute_mobilecropnet_v4_loss(outputs, batch)
    assert torch.isfinite(loss)
    assert 0.0 <= metrics["top1_hit"] <= 1.0


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
                "--epochs",
                "1",
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
    assert (run_dir / "best.pt").exists()

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
