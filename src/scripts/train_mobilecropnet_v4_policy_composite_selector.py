#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "src" / "scripts"
for path in (SRC_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_mobilecropnet_v4_policy_composite import (  # noqa: E402
    _accumulate_counts,
    _best_source_gap,
    _compact_metrics,
    _global_action_metrics,
    _masked_by_source,
    _metrics_to_float,
    _run_loss_metrics,
    _select_logits,
    _source_action_count_metrics,
    _update_outputs_with_composite,
    _write_json,
    _write_progress,
)
from mobilecropnet_v4.data import (  # noqa: E402
    DECISION_VOCAB,
    MobileCropNetV4BatchDataset,
    SUBJECT_BOX_TARGET_SOURCES,
    TARGET_AR_VOCAB,
    mobilecropnet_v4_collate,
)
from mobilecropnet_v4.eval_utils import infer_image_norm, infer_input_size, load_mobilecropnet_v4_checkpoint  # noqa: E402


FEATURE_NAMES = [
    "base_return_best_base",
    "base_return_best_crop",
    "base_return_gap",
    "crop_return_best_base",
    "crop_return_best_crop",
    "crop_return_gap",
    "cross_crop_minus_base",
    "base_utility_best_base",
    "base_utility_best_crop",
    "base_utility_gap",
    "base_decision_keep",
    "base_decision_minimal",
    "base_decision_crop",
    "base_decision_crop_prob",
    "base_decision_source_logit",
    "base_decision_source_prob",
    "subject_valid_prob",
    "subject_box_area",
    "subject_box_aspect_log",
    "image_ar_log",
    "valid_base_count",
    "valid_crop_count",
    *[f"route_prob_{idx}" for idx in range(7)],
    *[f"target_ar_{idx}" for idx in range(len(TARGET_AR_VOCAB))],
]


class SelectorMLP(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 64, dropout: float = 0.05) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(16, hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(max(16, hidden_dim // 2), 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an internal MobileCropNet two-expert source selector.")
    parser.add_argument("--base_checkpoint", required=True, type=Path)
    parser.add_argument("--crop_checkpoint", required=True, type=Path)
    parser.add_argument("--train_jsonl", required=True, type=Path)
    parser.add_argument("--val_jsonl", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--image_root", type=Path, default=None)
    parser.add_argument("--input_size", type=int, default=None)
    parser.add_argument("--candidate_k", type=int, default=None)
    parser.add_argument("--image_mean", default=None)
    parser.add_argument("--image_std", default=None)
    parser.add_argument("--runtime_subject_prior_mode", choices=["none", "teacher"], default="none")
    parser.add_argument("--subject_box_target_source", choices=SUBJECT_BOX_TARGET_SOURCES, default=None)
    parser.add_argument("--base_surface", default="deployment")
    parser.add_argument("--crop_surface", default="deployment")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_train_rows", type=int, default=None)
    parser.add_argument("--max_val_rows", type=int, default=None)
    parser.add_argument("--selector_epochs", type=int, default=120)
    parser.add_argument("--selector_batch_size", type=int, default=512)
    parser.add_argument("--selector_hidden_dim", type=int, default=96)
    parser.add_argument("--selector_lr", type=float, default=1e-3)
    parser.add_argument("--selector_weight_decay", type=float, default=1e-4)
    parser.add_argument("--selector_label_mode", choices=["decision_target", "oracle_score"], default="decision_target")
    parser.add_argument("--oracle_score_margin", type=float, default=None)
    parser.add_argument("--oracle_crop_score_advantage", type=float, default=0.02)
    parser.add_argument("--thresholds", default="0.20,0.30,0.40,0.50,0.60,0.70,0.80")
    parser.add_argument("--score_margin", type=float, default=0.03)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--progress_log_interval", type=int, default=50)
    parser.add_argument("--progress_json", type=Path, default=None)
    return parser


def _parse_thresholds(raw: str) -> list[float]:
    values: list[float] = []
    for part in str(raw or "").replace(",", " ").split():
        try:
            values.append(float(part))
        except ValueError:
            continue
    return sorted({max(0.0, min(1.0, value)) for value in values}) or [0.5]


def _best(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, -10000.0).max(dim=1).values


def _source_features(
    *,
    base_outputs: dict[str, torch.Tensor],
    crop_outputs: dict[str, torch.Tensor],
    base_logits: torch.Tensor,
    crop_logits: torch.Tensor,
    batch: dict[str, Any],
) -> torch.Tensor:
    valid = batch["valid"] > 0
    candidate_is_base = batch["candidate_is_base"] > 0
    base_mask = valid & candidate_is_base
    crop_mask = valid & (~candidate_is_base)

    base_return_best_base = _best(base_logits, base_mask)
    base_return_best_crop = _best(base_logits, crop_mask)
    crop_return_best_base = _best(crop_logits, base_mask)
    crop_return_best_crop = _best(crop_logits, crop_mask)
    base_utility = base_outputs.get("utility_logits", base_logits).float()
    base_utility_best_base = _best(base_utility, base_mask)
    base_utility_best_crop = _best(base_utility, crop_mask)

    decision_logits = base_outputs.get("decision_logits")
    if not torch.is_tensor(decision_logits):
        decision_logits = torch.zeros((base_logits.shape[0], len(DECISION_VOCAB)), device=base_logits.device)
    decision_probs = torch.softmax(decision_logits.float(), dim=1)
    if decision_probs.shape[1] < 3:
        decision_probs = torch.nn.functional.pad(decision_probs, (0, 3 - decision_probs.shape[1]))
    decision_source_logit = base_outputs.get("decision_source_logit")
    if not torch.is_tensor(decision_source_logit):
        decision_source_logit = torch.zeros((base_logits.shape[0],), device=base_logits.device, dtype=base_logits.dtype)
    decision_source_logit = decision_source_logit.float().view(-1)

    route_logits = base_outputs.get("route_logits")
    if torch.is_tensor(route_logits):
        route_probs = torch.softmax(route_logits.float(), dim=1)
    else:
        route_probs = torch.zeros((base_logits.shape[0], 7), device=base_logits.device, dtype=base_logits.dtype)
    if route_probs.shape[1] < 7:
        route_probs = torch.nn.functional.pad(route_probs, (0, 7 - route_probs.shape[1]))
    route_probs = route_probs[:, :7]

    subject_valid_logit = base_outputs.get("pred_subject_valid_logit", base_outputs.get("subject_valid_logit"))
    if torch.is_tensor(subject_valid_logit):
        subject_valid_prob = torch.sigmoid(subject_valid_logit.float().view(-1))
    else:
        subject_valid_prob = torch.zeros((base_logits.shape[0],), device=base_logits.device, dtype=base_logits.dtype)
    subject_box = base_outputs.get("pred_subject_box", base_outputs.get("subject_box"))
    if torch.is_tensor(subject_box):
        box = subject_box.float().clamp(0.0, 1.0)
        wh = (box[:, 2:4] - box[:, 0:2]).clamp_min(1e-4)
        subject_area = wh[:, 0] * wh[:, 1]
        subject_aspect = torch.log((wh[:, 0] / wh[:, 1]).clamp(1e-4, 1e4))
    else:
        subject_area = torch.zeros((base_logits.shape[0],), device=base_logits.device, dtype=base_logits.dtype)
        subject_aspect = torch.zeros_like(subject_area)

    target_ar = torch.nn.functional.one_hot(
        batch["target_ar_id"].long().clamp(0, len(TARGET_AR_VOCAB) - 1),
        num_classes=len(TARGET_AR_VOCAB),
    ).to(base_logits.dtype)
    counts = torch.stack([base_mask.sum(dim=1), crop_mask.sum(dim=1)], dim=1).to(base_logits.dtype) / max(1.0, float(base_logits.shape[1]))
    scalar_features = torch.stack(
        [
            base_return_best_base,
            base_return_best_crop,
            base_return_best_crop - base_return_best_base,
            crop_return_best_base,
            crop_return_best_crop,
            crop_return_best_crop - crop_return_best_base,
            crop_return_best_crop - base_return_best_base,
            base_utility_best_base,
            base_utility_best_crop,
            base_utility_best_crop - base_utility_best_base,
            decision_logits.float()[:, 0],
            decision_logits.float()[:, 1] if decision_logits.shape[1] > 1 else torch.zeros_like(base_return_best_base),
            decision_logits.float()[:, 2] if decision_logits.shape[1] > 2 else torch.zeros_like(base_return_best_base),
            decision_probs[:, 2],
            decision_source_logit,
            torch.sigmoid(decision_source_logit),
            subject_valid_prob,
            subject_area,
            subject_aspect,
            batch["image_ar_log"].float().view(-1),
            counts[:, 0],
            counts[:, 1],
        ],
        dim=1,
    )
    return torch.cat([scalar_features, route_probs, target_ar], dim=1)


def _source_oracle_labels(
    *,
    base_logits: torch.Tensor,
    crop_logits: torch.Tensor,
    batch: dict[str, Any],
    score_margin: float,
    crop_score_advantage: float,
) -> torch.Tensor:
    valid = batch["valid"] > 0
    candidate_is_base = batch["candidate_is_base"] > 0
    score_target = batch["score_target"].float()
    base_mask = valid & candidate_is_base
    crop_mask = valid & (~candidate_is_base)
    row_valid = base_mask.any(dim=1) & crop_mask.any(dim=1)

    def top_target_score(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        top_idx = logits.float().masked_fill(~mask, -10000.0).argmax(dim=1)
        return score_target.gather(1, top_idx[:, None]).squeeze(1)

    base_top_score = top_target_score(base_logits, base_mask)
    crop_top_score = top_target_score(crop_logits, crop_mask)
    best_score = score_target.masked_fill(~valid, -1.0).max(dim=1).values
    base_hit = row_valid & ((best_score - base_top_score) <= float(score_margin))
    crop_hit = row_valid & ((best_score - crop_top_score) <= float(score_margin))
    crop_advantage = crop_top_score > (base_top_score + float(crop_score_advantage))
    choose_crop = (crop_hit & ~base_hit) | ((crop_hit == base_hit) & crop_advantage)
    return (choose_crop & row_valid).to(torch.float32)


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def _make_loader(
    *,
    jsonl_path: Path,
    args: argparse.Namespace,
    input_size: int,
    candidate_k: int,
    image_mean: list[float],
    image_std: list[float],
    subject_box_target_source: str,
    max_rows: int | None,
    device: torch.device,
) -> DataLoader:
    dataset = MobileCropNetV4BatchDataset(
        jsonl_path=jsonl_path,
        project_root=args.project_root,
        image_root=args.image_root,
        input_size=input_size,
        candidate_k=candidate_k,
        image_mean=image_mean,
        image_std=image_std,
        max_rows=max_rows,
        subject_box_target_source=subject_box_target_source,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=mobilecropnet_v4_collate,
    )


def _forward_experts(
    *,
    base_model: nn.Module,
    crop_model: nn.Module,
    batch: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    subject_prior_box = batch.get("subject_prior_box") if args.runtime_subject_prior_mode == "teacher" else None
    subject_prior_valid = batch.get("subject_prior_valid") if args.runtime_subject_prior_mode == "teacher" else None
    subject_prior_reliability = batch.get("subject_prior_reliability") if args.runtime_subject_prior_mode == "teacher" else None
    common_args = (
        batch["image"],
        batch["boxes"],
        batch["valid"],
        batch["target_ar_id"],
        batch["image_ar_log"],
        batch["candidate_is_base"],
        batch["box_meta"],
        batch.get("letterbox_content_box"),
        subject_prior_box,
        subject_prior_valid,
        subject_prior_reliability,
    )
    base_outputs = base_model(*common_args, return_generated_scores=True)
    crop_outputs = crop_model(*common_args, return_generated_scores=False)
    base_logits = _select_logits(base_outputs, args.base_surface).float()
    crop_logits = _select_logits(crop_outputs, args.crop_surface).float()
    return base_outputs, crop_outputs, base_logits, crop_logits


def _collect_features(
    *,
    loader: DataLoader,
    base_model: nn.Module,
    crop_model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    progress_path: Path,
    phase: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    crop_id = DECISION_VOCAB.index("crop")
    oracle_margin = float(args.score_margin if args.oracle_score_margin is None else args.oracle_score_margin)
    started = time.monotonic()
    processed = 0
    with torch.no_grad():
        for batch_idx, raw_batch in enumerate(loader, start=1):
            batch = _to_device(raw_batch, device)
            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                base_outputs, crop_outputs, base_logits, crop_logits = _forward_experts(
                    base_model=base_model,
                    crop_model=crop_model,
                    batch=batch,
                    args=args,
                )
            features.append(
                _source_features(
                    base_outputs=base_outputs,
                    crop_outputs=crop_outputs,
                    base_logits=base_logits.float(),
                    crop_logits=crop_logits.float(),
                    batch=batch,
                )
                .detach()
                .cpu()
            )
            if str(args.selector_label_mode) == "oracle_score":
                label = _source_oracle_labels(
                    base_logits=base_logits.float(),
                    crop_logits=crop_logits.float(),
                    batch=batch,
                    score_margin=oracle_margin,
                    crop_score_advantage=float(args.oracle_crop_score_advantage),
                )
            else:
                label = (batch["decision_target"] == crop_id).to(torch.float32)
            labels.append(label.detach().cpu())
            processed += int(batch["image"].shape[0])
            if args.progress_log_interval > 0 and (batch_idx == 1 or batch_idx % int(args.progress_log_interval) == 0):
                elapsed = max(1e-6, time.monotonic() - started)
                _write_progress(
                    progress_path,
                    {
                        "state": "running",
                        "phase": phase,
                        "processed_batches": int(batch_idx),
                        "processed_images": int(processed),
                        "samples_per_sec": round(processed / elapsed, 4),
                    },
                )
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def _mean_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                continue
            if torch.isfinite(torch.tensor(scalar)):
                sums[key] = sums.get(key, 0.0) + scalar
                counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / max(1, counts[key]) for key in sorted(sums)}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.progress_json or (args.output_dir / "progress.json")
    summary_path = args.output_dir / "summary.json"
    selector_path = args.output_dir / "selector.pt"
    device = torch.device(args.device)

    _write_progress(progress_path, {"state": "running", "phase": "load_checkpoints"})
    base_model, base_ckpt = load_mobilecropnet_v4_checkpoint(args.base_checkpoint, device=device)
    crop_model, crop_ckpt = load_mobilecropnet_v4_checkpoint(args.crop_checkpoint, device=device)
    base_model.eval()
    crop_model.eval()
    input_size = infer_input_size(base_ckpt, args.input_size)
    image_mean, image_std = infer_image_norm(base_ckpt, explicit_mean=args.image_mean, explicit_std=args.image_std)
    candidate_k = int(args.candidate_k or base_ckpt.get("model_config", {}).get("candidate_k", 24))
    crop_candidate_k = int(crop_ckpt.get("model_config", {}).get("candidate_k", candidate_k))
    if crop_candidate_k != candidate_k:
        raise SystemExit(f"candidate_k mismatch: base={candidate_k}, crop={crop_candidate_k}")
    subject_box_target_source = str(
        args.subject_box_target_source
        or base_ckpt.get("train_config", {}).get("subject_box_target_source")
        or base_ckpt.get("dataset_summary", {}).get("subject_box_target_source")
        or "legacy"
    )
    train_loader = _make_loader(
        jsonl_path=args.train_jsonl,
        args=args,
        input_size=input_size,
        candidate_k=candidate_k,
        image_mean=image_mean,
        image_std=image_std,
        subject_box_target_source=subject_box_target_source,
        max_rows=args.max_train_rows,
        device=device,
    )
    val_loader = _make_loader(
        jsonl_path=args.val_jsonl,
        args=args,
        input_size=input_size,
        candidate_k=candidate_k,
        image_mean=image_mean,
        image_std=image_std,
        subject_box_target_source=subject_box_target_source,
        max_rows=args.max_val_rows,
        device=device,
    )
    train_features, train_labels = _collect_features(
        loader=train_loader,
        base_model=base_model,
        crop_model=crop_model,
        args=args,
        device=device,
        progress_path=progress_path,
        phase="extract_train_features",
    )
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0).clamp_min(1e-4)
    train_features_norm = (train_features - mean) / std

    selector = SelectorMLP(train_features.shape[1], hidden_dim=args.selector_hidden_dim).to(device)
    train_ds = TensorDataset(train_features_norm.to(torch.float32), train_labels.to(torch.float32))
    train_select_loader = DataLoader(train_ds, batch_size=args.selector_batch_size, shuffle=True)
    pos = float(train_labels.sum().item())
    neg = float(train_labels.numel() - train_labels.sum().item())
    pos_weight = torch.tensor([neg / max(1.0, pos)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(selector.parameters(), lr=args.selector_lr, weight_decay=args.selector_weight_decay)
    selector.train()
    for epoch in range(1, int(args.selector_epochs) + 1):
        losses: list[float] = []
        for xb, yb in train_select_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(selector(xb), yb)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        if epoch == 1 or epoch == int(args.selector_epochs) or epoch % 20 == 0:
            _write_progress(
                progress_path,
                {
                    "state": "running",
                    "phase": "train_selector",
                    "selector_epoch": int(epoch),
                    "selector_epochs": int(args.selector_epochs),
                    "selector_loss": sum(losses) / max(1, len(losses)),
                },
            )
    selector.eval()
    torch.save(
        {
            "state_dict": selector.state_dict(),
            "feature_mean": mean,
            "feature_std": std,
            "feature_names": FEATURE_NAMES,
            "base_checkpoint": str(args.base_checkpoint),
            "crop_checkpoint": str(args.crop_checkpoint),
            "runtime_subject_prior_mode": str(args.runtime_subject_prior_mode),
            "selector_label_mode": str(args.selector_label_mode),
            "oracle_score_margin": float(args.score_margin if args.oracle_score_margin is None else args.oracle_score_margin),
            "oracle_crop_score_advantage": float(args.oracle_crop_score_advantage),
            "selector_pos_weight": float(pos_weight.detach().cpu().item()),
            "candidate_k": int(candidate_k),
            "input_size": int(input_size),
        },
        selector_path,
    )

    thresholds = _parse_thresholds(args.thresholds)
    metric_rows: dict[str, list[dict[str, float]]] = {f"selector_t{int(round(t * 100)):03d}": [] for t in thresholds}
    count_sums: dict[str, dict[str, float]] = {name: {} for name in metric_rows}
    with torch.no_grad():
        for batch_idx, raw_batch in enumerate(val_loader, start=1):
            batch = _to_device(raw_batch, device)
            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                base_outputs, crop_outputs, base_logits, crop_logits = _forward_experts(
                    base_model=base_model,
                    crop_model=crop_model,
                    batch=batch,
                    args=args,
                )
            feats = _source_features(
                base_outputs=base_outputs,
                crop_outputs=crop_outputs,
                base_logits=base_logits.float(),
                crop_logits=crop_logits.float(),
                batch=batch,
            )
            selector_prob = torch.sigmoid(selector(((feats.cpu() - mean) / std).to(device=device, dtype=torch.float32)))
            gap = _best_source_gap(
                base_logits=base_logits.float(),
                crop_logits=crop_logits.float(),
                valid=batch["valid"],
                candidate_is_base=batch["candidate_is_base"],
            )
            for threshold in thresholds:
                gate_name = f"selector_t{int(round(threshold * 100)):03d}"
                choose_crop = selector_prob >= float(threshold)
                composite_logits = _masked_by_source(
                    base_logits=base_logits.float(),
                    crop_logits=crop_logits.float(),
                    valid=batch["valid"],
                    candidate_is_base=batch["candidate_is_base"],
                    choose_crop=choose_crop,
                )
                outputs = _update_outputs_with_composite(
                    base_outputs=base_outputs,
                    composite_logits=composite_logits,
                    gate_signal=torch.logit(selector_prob.clamp(1e-5, 1.0 - 1e-5)),
                )
                metric_rows[gate_name].append(_run_loss_metrics(outputs, batch, args.score_margin))
                _accumulate_counts(
                    count_sums[gate_name],
                    _source_action_count_metrics(
                        gate_name=gate_name,
                        logits=composite_logits,
                        choose_crop=choose_crop,
                        decision_target=batch["decision_target"],
                        valid=batch["valid"],
                        candidate_is_base=batch["candidate_is_base"],
                        return_target=batch["score_target"],
                        score_target=batch["score_target"],
                        score_margin=float(args.score_margin),
                    ),
                )
            if args.progress_log_interval > 0 and (batch_idx == 1 or batch_idx % int(args.progress_log_interval) == 0):
                _write_progress(
                    progress_path,
                    {
                        "state": "running",
                        "phase": "eval_selector",
                        "processed_batches": int(batch_idx),
                    },
                )

    gates: dict[str, dict[str, Any]] = {}
    for gate_name, rows in metric_rows.items():
        metrics = _mean_metric_rows(rows)
        surface, gate = _global_action_metrics(gate_name, count_sums[gate_name])
        metrics.update(surface)
        gates[gate_name] = {
            "metrics": metrics,
            "gate": gate,
            "compact": _compact_metrics(metrics),
            "global_counts": count_sums[gate_name],
        }
    summary = {
        "state": "completed",
        "base_checkpoint": str(args.base_checkpoint),
        "crop_checkpoint": str(args.crop_checkpoint),
        "train_jsonl": str(args.train_jsonl),
        "val_jsonl": str(args.val_jsonl),
        "selector_checkpoint": str(selector_path),
        "feature_names": FEATURE_NAMES,
        "runtime_subject_prior_mode": str(args.runtime_subject_prior_mode),
        "selector_label_mode": str(args.selector_label_mode),
        "oracle_score_margin": float(args.score_margin if args.oracle_score_margin is None else args.oracle_score_margin),
        "oracle_crop_score_advantage": float(args.oracle_crop_score_advantage),
        "selector_pos_weight": float(pos_weight.detach().cpu().item()),
        "train_rows": int(train_labels.numel()),
        "train_crop_rate": float(train_labels.mean().item()) if train_labels.numel() else 0.0,
        "thresholds": thresholds,
        "input_size": int(input_size),
        "candidate_k": int(candidate_k),
        "subject_box_target_source": str(subject_box_target_source),
        "gates": gates,
    }
    _write_json(summary_path, summary)
    _write_progress(
        progress_path,
        {
            "state": "completed",
            "phase": "eval_selector",
            "summary_json": str(summary_path),
            "selector_checkpoint": str(selector_path),
        },
    )
    print(json.dumps({k: v["compact"] | v["gate"] for k, v in gates.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
