#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from universal_crop_teacher.data import (  # noqa: E402
    UniversalCropTeacherDataset,
    collate_uctr_batch,
    load_warehouse_rows,
)
from universal_crop_teacher.losses import UCTRLossWeights, compute_uctr_loss  # noqa: E402
from universal_crop_teacher.model import UniversalCropTeacherH, UniversalCropTeacherHConfig  # noqa: E402
from universal_crop_teacher.warehouse import gaic_record_to_warehouse_row  # noqa: E402
from mobilecropnet_v4.gaic_benchmark import evaluate_scored_records, load_gaic_annotation_records  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UniversalCropTeacher-H from a unified candidate warehouse.")
    parser.add_argument("--warehouse_jsonl", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_candidates", type=int, default=128)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--crop_size", type=int, default=160)
    parser.add_argument("--token_dim", type=int, default=192)
    parser.add_argument("--base_channels", type=int, default=48)
    parser.add_argument("--transformer_layers", type=int, default=3)
    parser.add_argument("--transformer_heads", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--list_temperature", type=float, default=0.12)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260420)
    parser.add_argument("--max_train_tasks", type=int, default=0)
    parser.add_argument("--max_val_tasks", type=int, default=0)
    parser.add_argument("--max_test_tasks", type=int, default=0)
    parser.add_argument("--limit_train_steps", type=int, default=0)
    parser.add_argument("--include_images", type=int, default=1)
    parser.add_argument("--balanced_train_dataset_sampling", type=int, default=1)
    parser.add_argument("--dataset_balance_power", type=float, default=1.0)
    parser.add_argument("--train_epoch_samples", type=int, default=0)
    parser.add_argument("--listwise_weight", type=float, default=1.0)
    parser.add_argument("--pairwise_weight", type=float, default=0.45)
    parser.add_argument("--reg_weight", type=float, default=0.25)
    parser.add_argument("--gt_iou_weight", type=float, default=0.20)
    parser.add_argument("--top_weight", type=float, default=0.15)
    parser.add_argument("--uncertainty_weight", type=float, default=0.05)
    parser.add_argument("--safety_weight", type=float, default=0.05)
    parser.add_argument("--domain_aux_weight", type=float, default=0.20)
    parser.add_argument("--gate_weight", type=float, default=0.05)
    parser.add_argument("--final_eval_splits", nargs="*", default=["train", "val", "test"])
    parser.add_argument("--gaic_selection_annotations_json", type=Path, default=None)
    parser.add_argument("--gaic_selection_image_roots", nargs="*", type=Path, default=[])
    parser.add_argument("--gaic_selection_max_images", type=int, default=0)
    parser.add_argument("--gaic_selection_batch_size", type=int, default=0)
    parser.add_argument("--gaic_selection_num_workers", type=int, default=0)
    parser.add_argument("--gaic_selection_max_candidates", type=int, default=0)
    parser.add_argument("--gaic_selection_weight", type=float, default=1.0)
    parser.add_argument("--gaic_selection_top1_mos_weight", type=float, default=0.50)
    parser.add_argument("--gaic_selection_srcc_weight", type=float, default=0.35)
    parser.add_argument("--gaic_selection_accw4_weight", type=float, default=0.15)
    parser.add_argument("--gaic_selection_min_top1_mos", type=float, default=0.0)
    parser.add_argument("--gaic_selection_min_srcc", type=float, default=0.0)
    parser.add_argument("--gaic_selection_min_accw4_top10", type=float, default=0.0)
    parser.add_argument("--public_selection_weight", type=float, default=1.0)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = dict(batch)
    for key in (
        "global_images",
        "crop_images",
        "candidate_features",
        "valid_mask",
        "dataset_ids",
        "target_ar_ids",
        "target_scores",
        "label_confidence",
        "gt_iou",
    ):
        out[key] = batch[key].to(device)
    return out


def forward_model(model: UniversalCropTeacherH, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    return model(
        batch["global_images"],
        batch["crop_images"],
        batch["candidate_features"],
        batch["valid_mask"],
        batch["dataset_ids"],
        batch["target_ar_ids"],
    )


def prediction_rows_for_batch(outputs: dict[str, torch.Tensor], batch: dict[str, Any], *, method: str) -> list[dict[str, Any]]:
    scores = torch.sigmoid(outputs["utility_logits"]).detach().float().cpu()
    uncertainty = torch.sigmoid(outputs["uncertainty_logits"]).detach().float().cpu()
    rows: list[dict[str, Any]] = []
    for bidx, source in enumerate(batch["source_rows"]):
        valid = batch["valid_mask"][bidx].detach().cpu().bool()
        candidates = []
        source_candidates = list(source.get("candidates", []) or [])
        # Dataset may have selected a subset. Candidate ids preserve the selected order.
        for cidx, cid in enumerate(batch["candidate_ids"][bidx]):
            if cidx >= int(valid.numel()) or not bool(valid[cidx]):
                continue
            source_match = next((cand for cand in source_candidates if str(cand.get("candidate_id", "")) == str(cid)), {})
            box = source_match.get("bbox_xyxy_norm", source_match.get("bbox_norm_xyxy"))
            if not isinstance(box, list):
                box = batch["candidate_boxes"][bidx, cidx].detach().cpu().tolist()
            score = float(scores[bidx, cidx])
            candidates.append(
                {
                    "candidate_id": str(cid),
                    "bbox_xyxy_norm": [float(v) for v in box[:4]],
                    "score": score,
                    "source": str(source_match.get("source", "")),
                    "label": str(source_match.get("label", "")),
                    "scores": {
                        "uctr_utility": score,
                        "uctr_uncertainty": float(uncertainty[bidx, cidx]),
                    },
                }
            )
        order = sorted(range(len(candidates)), key=lambda idx: float(candidates[idx]["score"]), reverse=True)
        for rank, idx in enumerate(order, 1):
            candidates[idx]["model_rank"] = int(rank)
        rows.append(
            {
                "dataset": str(source.get("dataset", "")).lower(),
                "image_id": str(source.get("image_id", "")),
                "target_ar": source.get("target_ar"),
                "method": method,
                "source": "universal_crop_teacher_h",
                "candidates": candidates,
            }
        )
    return rows


def build_train_sampler(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    balance_power: float,
    num_samples: int,
) -> tuple[WeightedRandomSampler, dict[str, Any]]:
    dataset_counts = Counter(str(row.get("dataset", "unknown")).lower() for row in rows)
    weights = []
    for row in rows:
        dataset = str(row.get("dataset", "unknown")).lower()
        count = max(1, int(dataset_counts.get(dataset, 1)))
        weights.append(1.0 / (float(count) ** max(0.0, float(balance_power))))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=max(1, int(num_samples)),
        replacement=True,
        generator=generator,
    )
    return sampler, {
        "enabled": True,
        "num_samples": int(num_samples),
        "balance_power": float(balance_power),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "min_weight": float(min(weights)) if weights else 0.0,
        "max_weight": float(max(weights)) if weights else 0.0,
    }


def evaluate_predictions(rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]]) -> dict[str, Any]:
    pred_by_task = {
        (str(row.get("dataset", "")), str(row.get("image_id", "")), str(row.get("target_ar", ""))): row
        for row in pred_rows
    }
    metrics_by_dataset: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("dataset", "")), str(row.get("image_id", "")), str(row.get("target_ar", "")))
        pred = pred_by_task.get(key, {})
        candidates = pred.get("candidates", []) if isinstance(pred.get("candidates"), list) else []
        score_by_id = {str(c.get("candidate_id", "")): float(c.get("score", 0.0)) for c in candidates}
        targets = {
            str(c.get("candidate_id", "")): float(c.get("target_score", 0.0))
            for c in row.get("candidates", [])
            if isinstance(c, dict)
        }
        gt_ious = {
            str(c.get("candidate_id", "")): float(c.get("gt_iou", c.get("target_score", 0.0)))
            for c in row.get("candidates", [])
            if isinstance(c, dict)
        }
        if not score_by_id or not targets:
            continue
        sorted_ids = sorted(score_by_id, key=lambda cid: score_by_id[cid], reverse=True)
        target_sorted = sorted(targets, key=lambda cid: targets[cid], reverse=True)
        top_id = sorted_ids[0]
        rank_by_id = {cid: rank for rank, cid in enumerate(sorted_ids, 1)}
        best_id = target_sorted[0]
        pair_hits = 0.0
        pair_weighted = 0.0
        pair_total = 0
        pair_weight_total = 0.0
        for pair in row.get("pairwise", []) or []:
            if not isinstance(pair, dict):
                continue
            a = str(pair.get("candidate_id_a", ""))
            b = str(pair.get("candidate_id_b", ""))
            if a not in score_by_id or b not in score_by_id:
                continue
            preferred = str(pair.get("preferred", "a"))
            hit = 0.5
            if score_by_id[a] > score_by_id[b]:
                hit = 1.0 if preferred == "a" else 0.0
            elif score_by_id[b] > score_by_id[a]:
                hit = 1.0 if preferred == "b" else 0.0
            weight = max(1e-6, float(pair.get("weight", 1.0)))
            pair_hits += hit
            pair_weighted += hit * weight
            pair_total += 1
            pair_weight_total += weight
        metrics_by_dataset[str(row.get("dataset", "unknown"))].append(
            {
                "top1_target": float(targets.get(top_id, 0.0)),
                "top1_gt_iou": float(gt_ious.get(top_id, 0.0)),
                "best_rank_at_1": 1.0 if rank_by_id.get(best_id, 999999) <= 1 else 0.0,
                "best_rank_at_5": 1.0 if rank_by_id.get(best_id, 999999) <= 5 else 0.0,
                "pairwise": pair_hits / pair_total if pair_total else float("nan"),
                "weighted_pairwise": pair_weighted / pair_weight_total if pair_weight_total else float("nan"),
                "pair_count": float(pair_total),
            }
        )

    def mean_metric(items: list[dict[str, float]], key: str) -> float | None:
        vals = [float(item[key]) for item in items if key in item and np.isfinite(float(item[key]))]
        return float(sum(vals) / len(vals)) if vals else None

    datasets: dict[str, Any] = {}
    primary_values: list[float] = []
    for dataset, items in sorted(metrics_by_dataset.items()):
        row = {
            "task_count": len(items),
            "top1_target": mean_metric(items, "top1_target"),
            "top1_gt_iou": mean_metric(items, "top1_gt_iou"),
            "best_rank_at_1": mean_metric(items, "best_rank_at_1"),
            "best_rank_at_5": mean_metric(items, "best_rank_at_5"),
            "pairwise": mean_metric(items, "pairwise"),
            "weighted_pairwise": mean_metric(items, "weighted_pairwise"),
            "pair_count": int(sum(float(item.get("pair_count", 0.0)) for item in items)),
        }
        if dataset == "cpc" and row["weighted_pairwise"] is not None:
            primary = float(row["weighted_pairwise"])
        elif row["top1_gt_iou"] is not None and dataset in {"fcdb", "gnmc"}:
            primary = float(row["top1_gt_iou"])
        else:
            primary = float(row["top1_target"] or 0.0)
        row["primary"] = primary
        primary_values.append(primary)
        datasets[dataset] = row
    return {
        "task_count": int(sum(row["task_count"] for row in datasets.values())),
        "datasets": datasets,
        "macro_primary": float(sum(primary_values) / max(1, len(primary_values))),
        "dataset_count": len(datasets),
    }


def public_macro_primary(metrics: dict[str, Any]) -> float:
    datasets = metrics.get("datasets", {}) if isinstance(metrics.get("datasets"), dict) else {}
    vals = [
        float(row.get("primary", 0.0))
        for dataset, row in datasets.items()
        if str(dataset).lower() != "gaic" and isinstance(row, dict) and row.get("primary") is not None
    ]
    return float(sum(vals) / max(1, len(vals)))


def evaluate_gaic_selection_metrics(
    *,
    model: UniversalCropTeacherH,
    records: list[dict[str, Any]],
    device: torch.device,
    batch_size: int,
    max_candidates: int,
    image_size: int,
    crop_size: int,
    num_workers: int,
) -> dict[str, Any]:
    rows = [
        gaic_record_to_warehouse_row(
            record,
            preserve_official_split=True,
        )
        for record in records
    ]
    dataset = UniversalCropTeacherDataset(
        rows,
        max_candidates=int(max_candidates),
        image_size=int(image_size),
        crop_size=int(crop_size),
        include_images=True,
        training=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=collate_uctr_batch,
    )
    scores_by_image_id: dict[str, list[float]] = {}
    with torch.no_grad():
        for batch in loader:
            batch_dev = move_batch_to_device(batch, device)
            outputs = forward_model(model, batch_dev)
            pred_rows = prediction_rows_for_batch(outputs, batch, method="universal_crop_teacher_h_gaic_selection")
            for source_row, pred_row in zip(batch["source_rows"], pred_rows):
                score_map = {
                    str(cand.get("candidate_id", "")): float(cand.get("score", 0.0))
                    for cand in pred_row.get("candidates", [])
                }
                scores_by_image_id[str(source_row.get("image_id", ""))] = [
                    float(score_map.get(str(cand.get("candidate_id", "")), -1e9))
                    for cand in source_row.get("candidates", [])
                ]
    result = evaluate_scored_records(records, scores_by_image_id, method_name="universal_crop_teacher_h_gaic_selection")
    metrics = result.get("metrics", {}) if isinstance(result.get("metrics"), dict) else {}
    return {
        "image_count": int(result.get("image_count", 0)),
        "missing_image_count": int(result.get("missing_image_count", 0)),
        "metrics": {
            "top1_mos": float(metrics.get("top1_mos", 0.0) or 0.0),
            "top1_mos_regret": float(metrics.get("top1_mos_regret", 0.0) or 0.0),
            "srcc": float(metrics.get("srcc", 0.0) or 0.0),
            "acc1_of_top10": float(metrics.get("acc1_of_top10", 0.0) or 0.0),
            "accw4_of_top10": float(metrics.get("accw4_of_top10", 0.0) or 0.0),
        },
    }


def build_selection_result(
    *,
    val_metrics: dict[str, Any],
    gaic_selection: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    public_score = float(public_macro_primary(val_metrics))
    gaic_metrics = gaic_selection.get("metrics", {}) if isinstance(gaic_selection, dict) else {}
    top1_mos = float(gaic_metrics.get("top1_mos", 0.0) or 0.0)
    srcc = float(gaic_metrics.get("srcc", 0.0) or 0.0)
    accw4 = float(gaic_metrics.get("accw4_of_top10", 0.0) or 0.0)
    gaic_component = (
        float(args.gaic_selection_top1_mos_weight) * (top1_mos / 5.0)
        + float(args.gaic_selection_srcc_weight) * srcc
        + float(args.gaic_selection_accw4_weight) * accw4
    )
    combined = float(args.public_selection_weight) * public_score + float(args.gaic_selection_weight) * gaic_component
    passed = True
    if gaic_selection is not None:
        passed = (
            top1_mos >= float(args.gaic_selection_min_top1_mos)
            and srcc >= float(args.gaic_selection_min_srcc)
            and accw4 >= float(args.gaic_selection_min_accw4_top10)
        )
    effective = combined if passed else combined - 1e6
    return {
        "public_macro_primary": public_score,
        "gaic_component": gaic_component,
        "combined_score": combined,
        "effective_score": effective,
        "passed_gate": bool(passed),
        "gaic_metrics": gaic_metrics,
    }


def predict_loader(
    *,
    model: UniversalCropTeacherH,
    loader: DataLoader,
    device: torch.device,
    method: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            batch_dev = move_batch_to_device(batch, device)
            outputs = forward_model(model, batch_dev)
            rows.extend(prediction_rows_for_batch(outputs, batch, method=method))
            source_rows.extend(batch["source_rows"])
    return source_rows, rows


def main() -> None:
    args = parse_args()
    start = time.time()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_split = {
        "train": load_warehouse_rows(args.warehouse_jsonl, split="train", max_rows=int(args.max_train_tasks)),
        "val": load_warehouse_rows(args.warehouse_jsonl, split="val", max_rows=int(args.max_val_tasks)),
        "test": load_warehouse_rows(args.warehouse_jsonl, split="test", max_rows=int(args.max_test_tasks)),
    }
    if not rows_by_split["train"]:
        raise SystemExit("warehouse has no train rows")
    if not rows_by_split["val"]:
        rows_by_split["val"] = rows_by_split["train"][: min(8, len(rows_by_split["train"]))]
    if not rows_by_split["test"]:
        rows_by_split["test"] = rows_by_split["val"]

    include_images = bool(int(args.include_images) > 0)
    gaic_selection_records: list[dict[str, Any]] | None = None
    if args.gaic_selection_annotations_json is not None:
        max_images = int(args.gaic_selection_max_images) if int(args.gaic_selection_max_images) > 0 else None
        gaic_selection_records = load_gaic_annotation_records(
            args.gaic_selection_annotations_json.resolve(),
            image_roots=args.gaic_selection_image_roots,
            max_images=max_images,
        )
    datasets = {
        split: UniversalCropTeacherDataset(
            rows,
            max_candidates=int(args.max_candidates),
            image_size=int(args.image_size),
            crop_size=int(args.crop_size),
            include_images=include_images,
            training=(split == "train"),
        )
        for split, rows in rows_by_split.items()
    }
    train_sampler = None
    train_sampler_summary = {"enabled": False, "mode": "shuffle"}
    if bool(int(args.balanced_train_dataset_sampling) > 0):
        epoch_samples = int(args.train_epoch_samples) if int(args.train_epoch_samples) > 0 else len(rows_by_split["train"])
        train_sampler, train_sampler_summary = build_train_sampler(
            rows_by_split["train"],
            seed=int(args.seed),
            balance_power=float(args.dataset_balance_power),
            num_samples=epoch_samples,
        )
    loaders = {}
    for split, dataset in datasets.items():
        sampler = train_sampler if split == "train" else None
        loaders[split] = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=(split == "train" and sampler is None),
            sampler=sampler,
            num_workers=int(args.num_workers),
            collate_fn=collate_uctr_batch,
        )

    config = UniversalCropTeacherHConfig(
        token_dim=int(args.token_dim),
        base_channels=int(args.base_channels),
        transformer_layers=int(args.transformer_layers),
        transformer_heads=int(args.transformer_heads),
        dropout=float(args.dropout),
    )
    device = torch.device(str(args.device))
    model = UniversalCropTeacherH(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_weights = UCTRLossWeights(
        listwise=float(args.listwise_weight),
        pairwise=float(args.pairwise_weight),
        reg=float(args.reg_weight),
        gt_iou=float(args.gt_iou_weight),
        top=float(args.top_weight),
        uncertainty=float(args.uncertainty_weight),
        safety=float(args.safety_weight),
        domain_aux=float(args.domain_aux_weight),
        gate=float(args.gate_weight),
    )
    best_score = -1e18
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_selection: dict[str, Any] | None = None
    best_val_metrics: dict[str, Any] | None = None
    best_gaic_selection: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        loss_rows: list[dict[str, float]] = []
        for step, batch in enumerate(loaders["train"], 1):
            optimizer.zero_grad(set_to_none=True)
            batch_dev = move_batch_to_device(batch, device)
            outputs = forward_model(model, batch_dev)
            loss, loss_metrics = compute_uctr_loss(outputs, batch_dev, weights=loss_weights, list_temperature=float(args.list_temperature))
            loss.backward()
            if float(args.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
            optimizer.step()
            loss_rows.append(loss_metrics)
            if int(args.limit_train_steps) > 0 and step >= int(args.limit_train_steps):
                break
        val_source_rows, val_preds = predict_loader(model=model, loader=loaders["val"], device=device, method="universal_crop_teacher_h")
        val_metrics = evaluate_predictions(val_source_rows, val_preds)
        gaic_selection = None
        if gaic_selection_records:
            gaic_selection = evaluate_gaic_selection_metrics(
                model=model,
                records=gaic_selection_records,
                device=device,
                batch_size=int(args.gaic_selection_batch_size) if int(args.gaic_selection_batch_size) > 0 else int(args.batch_size),
                max_candidates=int(args.gaic_selection_max_candidates) if int(args.gaic_selection_max_candidates) > 0 else int(args.max_candidates),
                image_size=int(args.image_size),
                crop_size=int(args.crop_size),
                num_workers=int(args.gaic_selection_num_workers) if int(args.gaic_selection_num_workers) > 0 else int(args.num_workers),
            )
        avg_loss = {
            key: float(sum(row.get(key, 0.0) for row in loss_rows) / max(1, len(loss_rows)))
            for key in sorted({key for row in loss_rows for key in row})
        }
        selection = build_selection_result(val_metrics=val_metrics, gaic_selection=gaic_selection, args=args)
        val_score = float(selection["effective_score"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": avg_loss,
                "val": val_metrics,
                "gaic_selection": gaic_selection,
                "selection": selection,
            }
        )
        print(
            json.dumps(
                {
                    "status": "epoch_done",
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "val_macro_primary": round(float(val_metrics.get("macro_primary", 0.0)), 6),
                    "public_macro_primary": round(float(selection["public_macro_primary"]), 6),
                    "selection_score": round(float(selection["combined_score"]), 6),
                    "selection_passed_gate": bool(selection["passed_gate"]),
                    "gaic_selection": gaic_selection,
                    "val_datasets": val_metrics.get("datasets", {}),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            best_selection = copy.deepcopy(selection)
            best_val_metrics = copy.deepcopy(val_metrics)
            best_gaic_selection = copy.deepcopy(gaic_selection)
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})

    if best_state is not None:
        model.load_state_dict(best_state)
    final_predictions: dict[str, Any] = {}
    final_metrics: dict[str, Any] = {}
    final_eval_splits = [str(split) for split in args.final_eval_splits if str(split) in {"train", "val", "test"}]
    if not final_eval_splits:
        final_eval_splits = ["train", "val", "test"]
    for split in final_eval_splits:
        source_rows, preds = predict_loader(model=model, loader=loaders[split], device=device, method="universal_crop_teacher_h")
        final_predictions[split] = preds
        final_metrics[split] = evaluate_predictions(source_rows, preds)
        write_jsonl(output_dir / f"predictions_{split}.jsonl", preds)

    checkpoint_path = output_dir / "universal_crop_teacher_h_best.pt"
    torch.save(
        {
            "format": "universal_crop_teacher_h_checkpoint_v1",
            "config": config.to_dict(),
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "train_args": vars(args),
            "best_epoch": int(best_epoch),
            "best_selection_score": float(best_score),
            "best_val_macro_primary": float((best_val_metrics or {}).get("macro_primary", 0.0)),
            "best_public_macro_primary": float((best_selection or {}).get("public_macro_primary", 0.0)),
            "best_gaic_selection": best_gaic_selection,
            "best_selection": best_selection,
            "history": history,
        },
        checkpoint_path,
    )
    row_counts = {split: len(rows) for split, rows in rows_by_split.items()}
    dataset_counts = {split: dict(Counter(str(row.get("dataset", "unknown")) for row in rows)) for split, rows in rows_by_split.items()}
    summary = {
        "format": "universal_crop_teacher_h_train_summary_v1",
        "warehouse_jsonl": str(args.warehouse_jsonl.resolve()),
        "output_dir": str(output_dir),
        "checkpoint_path": str(checkpoint_path),
        "row_counts": row_counts,
        "dataset_counts": dataset_counts,
        "config": config.to_dict(),
        "loss_weights": {
            "listwise": float(loss_weights.listwise),
            "pairwise": float(loss_weights.pairwise),
            "reg": float(loss_weights.reg),
            "gt_iou": float(loss_weights.gt_iou),
            "top": float(loss_weights.top),
            "uncertainty": float(loss_weights.uncertainty),
            "safety": float(loss_weights.safety),
            "domain_aux": float(loss_weights.domain_aux),
            "gate": float(loss_weights.gate),
        },
        "best_epoch": int(best_epoch),
        "best_selection_score": float(best_score),
        "best_val_macro_primary": float((best_val_metrics or {}).get("macro_primary", 0.0)),
        "best_public_macro_primary": float((best_selection or {}).get("public_macro_primary", 0.0)),
        "best_selection": best_selection,
        "best_gaic_selection": best_gaic_selection,
        "history": history,
        "final_metrics": final_metrics,
        "train_sampler": train_sampler_summary,
        "duration_sec": round(time.time() - start, 3),
        "device": str(device),
        "include_images": include_images,
        "final_eval_splits": final_eval_splits,
        "gaic_selection_config": {
            "annotations_json": str(args.gaic_selection_annotations_json.resolve()) if args.gaic_selection_annotations_json is not None else "",
            "image_roots": [str(path) for path in args.gaic_selection_image_roots],
            "weight": float(args.gaic_selection_weight),
            "top1_mos_weight": float(args.gaic_selection_top1_mos_weight),
            "srcc_weight": float(args.gaic_selection_srcc_weight),
            "accw4_weight": float(args.gaic_selection_accw4_weight),
            "min_top1_mos": float(args.gaic_selection_min_top1_mos),
            "min_srcc": float(args.gaic_selection_min_srcc),
            "min_accw4_top10": float(args.gaic_selection_min_accw4_top10),
            "record_count": len(gaic_selection_records or []),
        },
    }
    write_json(output_dir / "train_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
