#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
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

from mobilecropnet.data import MobileCropNetGAICDataset, mobilecropnet_collate
from mobilecropnet.model import MobileCropNet, compute_mobilecropnet_loss, model_config_to_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MobileCropNet on current SSTK/GAIC crop labels.")
    parser.add_argument("--train_json", required=True, type=Path)
    parser.add_argument("--val_json", required=True, type=Path)
    parser.add_argument("--image_root", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--width_mult", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_train_images", type=int, default=None)
    parser.add_argument("--max_val_images", type=int, default=None)
    parser.add_argument("--limit_train_steps", type=int, default=None)
    parser.add_argument("--limit_val_steps", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
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
    keys = sorted(rows[0].keys())
    return {key: float(sum(row.get(key, 0.0) for row in rows) / len(rows)) for key in keys}


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def _resolve_epoch_lr(*, base_lr: float, min_lr: float, scheduler: str, warmup_epochs: int, epoch: int, epochs: int) -> float:
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
    model: MobileCropNet,
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
    for step, batch in enumerate(loader, 1):
        if limit_steps is not None and step > limit_steps:
            break
        batch = _move_to_device(batch, device)
        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
                outputs = model(
                    batch["image"],
                    batch["boxes"],
                    batch["valid"],
                    batch["target_ar_id"],
                    batch["box_meta"],
                )
                loss, metrics = compute_mobilecropnet_loss(outputs, batch)
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        rows.append(metrics)
    return _mean_metrics(rows)


def save_checkpoint(
    path: Path,
    *,
    model: MobileCropNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "model_config": model_config_to_dict(model),
        "train_config": config,
    }
    torch.save(payload, path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    train_ds = MobileCropNetGAICDataset(
        json_path=args.train_json,
        image_root=args.image_root,
        input_size=args.input_size,
        k=args.k,
        max_images=args.max_train_images,
    )
    val_ds = MobileCropNetGAICDataset(
        json_path=args.val_json,
        image_root=args.image_root,
        input_size=args.input_size,
        k=args.k,
        max_images=args.max_val_images,
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
        collate_fn=mobilecropnet_collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=mobilecropnet_collate,
        drop_last=False,
    )

    model = MobileCropNet(k=args.k, width_mult=args.width_mult).to(device)
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
        epoch_lr = _resolve_epoch_lr(
            base_lr=args.lr,
            min_lr=args.min_lr,
            scheduler=args.scheduler,
            warmup_epochs=args.warmup_epochs,
            epoch=epoch,
            epochs=args.epochs,
        )
        _set_optimizer_lr(optimizer, epoch_lr)
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
        epoch_row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "lr": epoch_lr,
            "elapsed_sec": round(time() - start, 3),
        }
        history.append(epoch_row)
        (args.output_dir / "metrics.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        save_checkpoint(
            args.output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=epoch_row,
            config=config,
        )
        val_score = float(val_metrics.get("top1_hit", 0.0)) - 0.01 * float(val_metrics.get("loss", 0.0))
        if val_score > best_score + float(args.early_stop_min_delta):
            best_score = val_score
            epochs_since_improve = 0
            save_checkpoint(
                args.output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=epoch_row,
                config=config,
            )
        else:
            epochs_since_improve += 1
        print(json.dumps(epoch_row, ensure_ascii=False), flush=True)
        if args.early_stop_patience > 0 and epochs_since_improve >= args.early_stop_patience:
            print(
                json.dumps(
                    {
                        "event": "early_stop",
                        "epoch": epoch,
                        "best_score": best_score,
                        "patience": args.early_stop_patience,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
