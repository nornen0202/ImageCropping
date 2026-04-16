#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from time import time
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet_v4.data import MobileCropNetV4BatchDataset, mobilecropnet_v4_collate
from mobilecropnet_v4.model import MobileCropNetV4, compute_mobilecropnet_v4_loss, model_config_to_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MobileCropNet v4 on current SSTK/GAIC batch JSONL labels.")
    parser.add_argument("--train_jsonl", required=True, type=Path)
    parser.add_argument("--val_jsonl", required=True, type=Path)
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--image_root", type=Path, default=None)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--candidate_k", type=int, default=24)
    parser.add_argument("--proposal_q", type=int, default=16)
    parser.add_argument("--width_mult", type=float, default=0.75)
    parser.add_argument("--token_dim", type=int, default=128)
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
    parser.add_argument("--max_train_rows", type=int, default=None)
    parser.add_argument("--max_val_rows", type=int, default=None)
    parser.add_argument("--limit_train_steps", type=int, default=None)
    parser.add_argument("--limit_val_steps", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--gpu_usage_sample_interval", type=int, default=10)
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
    return {key: float(sum(row.get(key, 0.0) for row in rows) / len(rows)) for key in keys}


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


def resolve_epoch_lr(*, base_lr: float, min_lr: float, scheduler: str, warmup_epochs: int, epoch: int, epochs: int) -> float:
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return float(base_lr) * float(epoch) / float(max(1, warmup_epochs))
    if scheduler != "cosine":
        return float(base_lr)
    decay_epochs = max(1, int(epochs) - max(0, int(warmup_epochs)))
    decay_index = min(decay_epochs, max(0, int(epoch) - max(0, int(warmup_epochs)) - 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * float(decay_index) / float(decay_epochs)))
    return float(min_lr) + (float(base_lr) - float(min_lr)) * cosine


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
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    rows: list[dict[str, float]] = []
    gpu_samples: list[dict[str, float]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device if device.index is not None else None)
    for step, batch in enumerate(loader, 1):
        if limit_steps is not None and step > int(limit_steps):
            break
        batch = _move_to_device(batch, device)
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
                )
                loss, metrics = compute_mobilecropnet_v4_loss(outputs, batch)
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
        if device.type == "cuda":
            interval = max(1, int(getattr(loader, "gpu_usage_sample_interval", 10)))
            if step == 1 or step % interval == 0:
                torch.cuda.synchronize(device)
                sample = _query_nvidia_smi(device)
                if sample:
                    gpu_samples.append(sample)
        rows.append(metrics)
    out = _mean_metrics(rows)
    out.update(_summarize_gpu_samples(gpu_samples, device))
    return out


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
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_ds = MobileCropNetV4BatchDataset(
        jsonl_path=args.train_jsonl,
        project_root=args.project_root,
        image_root=args.image_root,
        input_size=args.input_size,
        candidate_k=args.candidate_k,
        max_rows=args.max_train_rows,
    )
    val_ds = MobileCropNetV4BatchDataset(
        jsonl_path=args.val_jsonl,
        project_root=args.project_root,
        image_root=args.image_root,
        input_size=args.input_size,
        candidate_k=args.candidate_k,
        max_rows=args.max_val_rows,
    )
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps({"train": train_ds.summary(), "val": val_ds.summary()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=mobilecropnet_v4_collate,
        drop_last=False,
    )
    setattr(train_loader, "gpu_usage_sample_interval", args.gpu_usage_sample_interval)
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=mobilecropnet_v4_collate,
        drop_last=False,
    )
    setattr(val_loader, "gpu_usage_sample_interval", args.gpu_usage_sample_interval)

    model = MobileCropNetV4(
        candidate_k=args.candidate_k,
        proposal_q=args.proposal_q,
        width_mult=args.width_mult,
        token_dim=args.token_dim,
    ).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    config = vars(args).copy()
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    best_score = -1.0
    epochs_since_improve = 0
    history: list[dict[str, Any]] = []
    start = time()
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
            )
        row = {"epoch": epoch, "lr": lr, "train": train_metrics, "val": val_metrics, "elapsed_sec": time() - start}
        history.append(row)
        (args.output_dir / "metrics.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        save_checkpoint(args.output_dir / "last.pt", model=model, optimizer=optimizer, epoch=epoch, metrics=row, config=config)

        score = float(val_metrics.get("top1_hit", 0.0)) + 0.25 * float(val_metrics.get("proposal_recall_0_5", 0.0)) + 0.1 * float(val_metrics.get("decision_acc", 0.0))
        if score > best_score + float(args.early_stop_min_delta):
            best_score = score
            epochs_since_improve = 0
            save_checkpoint(args.output_dir / "best.pt", model=model, optimizer=optimizer, epoch=epoch, metrics=row, config=config)
        else:
            epochs_since_improve += 1
        print(json.dumps(row, ensure_ascii=False))
        if args.early_stop_patience > 0 and epochs_since_improve >= args.early_stop_patience:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
