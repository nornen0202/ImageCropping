from __future__ import annotations

import math
import json

from mobilecropnet_v4.gaic_benchmark import (
    acc_k_of_top_n,
    coco_bbox_to_norm_xyxy,
    load_teacher_candidate_eval_scores,
    per_image_gaic_metrics,
    rank_weighted_acc_k_of_top_n,
)


def test_gaic_return_metrics_match_paper_example() -> None:
    mos = [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    # Return ranks {2, 5, 3, 10}; scores order is exactly these crop indices.
    scores = [0.0, 4.0, 2.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    assert acc_k_of_top_n(mos, scores, k=4, n=5) == 0.75
    expected = (math.exp(-(2 - 1) / 5) + math.exp(-(3 - 2) / 5) + math.exp(-(5 - 3) / 5)) / 4
    assert abs(rank_weighted_acc_k_of_top_n(mos, scores, k=4, n=5) - expected) < 1e-9


def test_per_image_metrics_are_perfect_for_oracle_scores() -> None:
    mos = [1.0, 2.0, 5.0, 4.0, 3.0]
    metrics = per_image_gaic_metrics(mos, mos)
    assert abs(metrics["pcc"] - 1.0) < 1e-9
    assert abs(metrics["srcc"] - 1.0) < 1e-9
    assert metrics["acc1_of_top5"] == 1.0
    assert metrics["acc4_of_top5"] == 1.0
    assert metrics["accw4_of_top5"] == 1.0
    assert metrics["top1_mos_regret"] == 0.0


def test_coco_bbox_to_norm_xyxy() -> None:
    assert coco_bbox_to_norm_xyxy([10, 20, 30, 40], width=100, height=200) == [0.1, 0.1, 0.4, 0.3]


def test_load_teacher_candidate_eval_scores_maps_exact_gaic_annotations(tmp_path) -> None:
    records = [
        {
            "image_id": "1",
            "candidates": [
                {"annotation_id": 10, "mos": 4.0},
                {"annotation_id": 11, "mos": 2.0},
            ],
        }
    ]
    rows = [
        {"image_id": "1", "protocol": "Gc", "candidate_id": "gaic_gt_11", "crop_utility_raw": 0.2},
        {"image_id": "1", "protocol": "Gc", "candidate_id": "gaic_gt_10", "crop_utility_raw": 0.9},
        {"image_id": "1", "protocol": "Ge", "candidate_id": "gaic_gt_10", "crop_utility_raw": 0.1},
    ]
    path = tmp_path / "candidate_eval_rows.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    scores, coverage = load_teacher_candidate_eval_scores(path, records, protocol="Gc", score_field="crop_utility_raw")

    assert scores == {"1": [0.9, 0.2]}
    assert coverage["1"]["annotation_score_rate"] == 1.0


def test_load_teacher_candidate_eval_scores_falls_back_to_bbox_when_annotation_ids_change(tmp_path) -> None:
    records = [
        {
            "image_id": "1",
            "candidates": [
                {"annotation_id": 1000, "bbox_norm_xyxy": [0.0, 0.0, 0.5, 0.5], "mos": 4.0},
                {"annotation_id": 1001, "bbox_norm_xyxy": [0.5, 0.5, 1.0, 1.0], "mos": 2.0},
            ],
        }
    ]
    rows = [
        {"image_id": "1", "protocol": "Gc", "candidate_id": "gaic_gt_10", "bbox_norm_xyxy": [0.5, 0.5, 1.0, 1.0], "crop_utility_raw": 0.2},
        {"image_id": "1", "protocol": "Gc", "candidate_id": "gaic_gt_11", "bbox_norm_xyxy": [0.0, 0.0, 0.5, 0.5], "crop_utility_raw": 0.9},
    ]
    path = tmp_path / "candidate_eval_rows.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    scores, coverage = load_teacher_candidate_eval_scores(path, records, protocol="Gc", score_field="crop_utility_raw")

    assert scores == {"1": [0.9, 0.2]}
    assert coverage["1"]["annotation_score_rate"] == 1.0
