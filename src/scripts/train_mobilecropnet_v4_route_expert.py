#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet_v4.data import (  # noqa: E402
    SUBJECT_MODE_COUNT,
    SUBJECT_MODE_VOCAB,
    _resolve_image_path,
    load_image_tensor_letterbox,
)
from routing.routing_v2_v16_modes import IMAGE_ROUTE_NO_PLACEMENT_NAMES  # noqa: E402


PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "convnextv2_base_448": {
        "backbone": "convnextv2_base.fcmae_ft_in22k_in1k_384",
        "input_size": 448,
        "batch_size": 20,
        "lr": 2.5e-5,
    },
    "convnextv2_large_512": {
        "backbone": "convnextv2_large.fcmae_ft_in22k_in1k_384",
        "input_size": 512,
        "batch_size": 8,
        "lr": 1.2e-5,
    },
    "convnextv2_huge_512": {
        "backbone": "convnextv2_huge.fcmae_ft_in22k_in1k_512",
        "input_size": 512,
        "batch_size": 2,
        "lr": 4.0e-6,
    },
    "swinv2_base_384": {
        "backbone": "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
        "input_size": 384,
        "batch_size": 18,
        "lr": 1.8e-5,
    },
    "eva02_base_448": {
        "backbone": "eva02_base_patch14_448.mim_in22k_ft_in22k_in1k",
        "input_size": 448,
        "batch_size": 8,
        "lr": 8.0e-6,
    },
    "eva02_large_448": {
        "backbone": "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        "input_size": 448,
        "batch_size": 4,
        "lr": 5.0e-6,
    },
    "dinov2_large_518": {
        "backbone": "vit_large_patch14_dinov2.lvd142m",
        "input_size": 518,
        "batch_size": 4,
        "lr": 4.0e-6,
    },
}

ROUTING_V2_ROUTE_FAMILY_LABELS = ["scene", "person_single", "person_group", "pet_dogcat"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an image-only MobileCropNet v4 route expert.")
    parser.add_argument("--train_jsonl", required=True, type=Path)
    parser.add_argument("--val_jsonl", required=True, type=Path)
    parser.add_argument(
        "--label_source",
        choices=[
            "legacy_subject_mode",
            "routing_v2_route_family",
            "routing_v2_flat",
            "routing_v2_image_route_no_placement",
        ],
        default="legacy_subject_mode",
    )
    parser.add_argument("--routing_v2_vocab", type=Path, default=None)
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--image_root", type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--profile", choices=[*PROFILE_PRESETS.keys(), "custom"], default="convnextv2_base_448")
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--backbone_pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--input_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max_train_images", type=int, default=0)
    parser.add_argument("--max_val_images", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.02)
    parser.add_argument("--warmup_epochs", type=int, default=1)
    parser.add_argument("--min_lr", type=float, default=1.0e-6)
    parser.add_argument("--route_balanced_sampler", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sampler_max_weight", type=float, default=12.0)
    parser.add_argument("--class_weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--focal_gamma", type=float, default=0.0)
    parser.add_argument("--hflip_prob", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress_log_interval", type=int, default=50)
    return parser


def _profile_value(args: argparse.Namespace, key: str, default: Any = None) -> Any:
    preset = PROFILE_PRESETS.get(str(args.profile), {})
    value = getattr(args, key)
    return preset.get(key, default) if value is None else value


def _load_label_vocab(args: argparse.Namespace) -> list[str]:
    if str(args.label_source) == "legacy_subject_mode":
        return list(SUBJECT_MODE_VOCAB)
    if str(args.label_source) == "routing_v2_route_family":
        if args.routing_v2_vocab and Path(args.routing_v2_vocab).exists():
            payload = json.loads(Path(args.routing_v2_vocab).read_text(encoding="utf-8"))
            labels = (((payload.get("hierarchical") or {}).get("route_family_v2")) or [])
            if labels:
                return [str(label) for label in labels]
        return list(ROUTING_V2_ROUTE_FAMILY_LABELS)
    if str(args.label_source) == "routing_v2_image_route_no_placement":
        if args.routing_v2_vocab and Path(args.routing_v2_vocab).exists():
            payload = json.loads(Path(args.routing_v2_vocab).read_text(encoding="utf-8"))
            labels = payload.get("route_names")
            if isinstance(labels, list) and labels:
                return [str(label) for label in labels]
            mapping = payload.get("route_to_id")
            if isinstance(mapping, dict) and mapping:
                ordered = ["" for _ in range(max(int(v) for v in mapping.values()) + 1)]
                for label, idx in mapping.items():
                    ordered[int(idx)] = str(label)
                if all(ordered):
                    return ordered
        return list(IMAGE_ROUTE_NO_PLACEMENT_NAMES)
    if not args.routing_v2_vocab:
        raise ValueError("--routing_v2_vocab is required for --label_source routing_v2_flat")
    payload = json.loads(Path(args.routing_v2_vocab).read_text(encoding="utf-8"))
    mapping = payload.get("flat_route_class")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"routing_v2 vocab has no flat_route_class mapping: {args.routing_v2_vocab}")
    labels = ["" for _ in range(max(int(v) for v in mapping.values()) + 1)]
    for label, idx in mapping.items():
        labels[int(idx)] = str(label)
    if any(not label for label in labels):
        raise ValueError(f"routing_v2 flat_route_class ids are not contiguous: {args.routing_v2_vocab}")
    return labels


def _route_label(row: dict[str, Any], *, label_source: str, label_to_id: dict[str, int], num_classes: int) -> int | None:
    if label_source == "routing_v2_route_family":
        routing_v2 = row.get("routing_v2_simple") if isinstance(row.get("routing_v2_simple"), dict) else {}
        label = str(routing_v2.get("route_family_v2") or "")
        if label in label_to_id:
            return int(label_to_id[label])
        hier = row.get("hierarchical_targets") if isinstance(row.get("hierarchical_targets"), dict) else {}
        try:
            idx = int(hier.get("route_family_v2_id"))
        except (TypeError, ValueError):
            return None
        return idx if 0 <= idx < num_classes else None
    if label_source == "routing_v2_flat":
        label = row.get("flat_route_class")
        if isinstance(label, str) and label in label_to_id:
            return int(label_to_id[label])
        try:
            idx = int(row.get("flat_route_class_id"))
        except (TypeError, ValueError):
            return None
        return idx if 0 <= idx < num_classes else None
    if label_source == "routing_v2_image_route_no_placement":
        label = row.get("image_route_name_no_placement")
        if isinstance(label, str) and label in label_to_id:
            return int(label_to_id[label])
        try:
            idx = int(row.get("image_route_id_no_placement"))
        except (TypeError, ValueError):
            return None
        return idx if 0 <= idx < num_classes else None
    routing = row.get("routing") if isinstance(row.get("routing"), dict) else {}
    label = str(routing.get("subject_mode") or "")
    if label in SUBJECT_MODE_VOCAB:
        return int(SUBJECT_MODE_VOCAB.index(label))
    try:
        idx = int(routing.get("subject_mode_id", 0))
    except (TypeError, ValueError):
        idx = 0
    return max(0, min(SUBJECT_MODE_COUNT - 1, idx))


def _image_key(row: dict[str, Any], image_path: Path) -> str:
    for key in ("image_id", "id", "image_name"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return str(image_path)


def _load_route_records(
    jsonl_path: Path,
    *,
    project_root: Path,
    image_root: Path | None,
    max_images: int,
    label_source: str,
    labels: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_image: dict[str, dict[str, Any]] = {}
    raw_rows = 0
    missing_images = 0
    missing_labels = 0
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            raw_rows += 1
            row = json.loads(line)
            image_path = _resolve_image_path(row, project_root=project_root, image_root=image_root)
            if not image_path.exists():
                missing_images += 1
                continue
            key = _image_key(row, image_path)
            if max_images > 0 and key not in by_image and len(by_image) >= int(max_images):
                continue
            label = _route_label(row, label_source=label_source, label_to_id=label_to_id, num_classes=len(labels))
            if label is None:
                missing_labels += 1
                continue
            bucket = by_image.setdefault(
                key,
                {
                    "image_key": key,
                    "image_path": str(image_path),
                    "labels": Counter(),
                    "row_count": 0,
                },
            )
            bucket["labels"][label] += 1
            bucket["row_count"] += 1
    records: list[dict[str, Any]] = []
    ambiguous = 0
    for bucket in by_image.values():
        label_votes: Counter[int] = bucket["labels"]
        top = label_votes.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            ambiguous += 1
        label = int(top[0][0])
        records.append(
            {
                "image_key": bucket["image_key"],
                "image_path": bucket["image_path"],
                "route_target": label,
                "row_count": int(bucket["row_count"]),
                "label_votes": {str(k): int(v) for k, v in sorted(label_votes.items())},
            }
        )
    records.sort(key=lambda item: item["image_key"])
    if max_images > 0:
        records = records[: int(max_images)]
    counts = Counter(int(row["route_target"]) for row in records)
    summary = {
        "jsonl_path": str(jsonl_path),
        "label_source": str(label_source),
        "raw_rows": raw_rows,
        "unique_images": len(records),
        "missing_images": missing_images,
        "missing_labels": missing_labels,
        "ambiguous_majority_images": ambiguous,
        "route_counts": {labels[k]: int(v) for k, v in sorted(counts.items())},
    }
    return records, summary


class RouteImageDataset(Dataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        *,
        input_size: int,
        image_mean: tuple[float, float, float],
        image_std: tuple[float, float, float],
        hflip_prob: float,
    ) -> None:
        self.records = records
        self.input_size = int(input_size)
        self.image_mean = image_mean
        self.image_std = image_std
        self.hflip_prob = max(0.0, min(1.0, float(hflip_prob)))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        image, _ = load_image_tensor_letterbox(
            Path(record["image_path"]),
            self.input_size,
            image_mean=self.image_mean,
            image_std=self.image_std,
        )
        if self.hflip_prob > 0.0 and random.random() < self.hflip_prob:
            image = torch.flip(image, dims=[2])
        return {
            "image": image,
            "route_target": torch.tensor(int(record["route_target"]), dtype=torch.long),
            "image_key": str(record["image_key"]),
        }


def _collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "route_target": torch.stack([item["route_target"] for item in batch], dim=0),
        "image_key": [item["image_key"] for item in batch],
    }


def _build_sampler(records: list[dict[str, Any]], max_weight: float, labels: list[str]) -> tuple[WeightedRandomSampler, dict[str, Any]]:
    counts = Counter(int(row["route_target"]) for row in records)
    weights = []
    for row in records:
        count = max(1, counts[int(row["route_target"])])
        weight = float(len(records)) / float(count)
        weight = min(float(max_weight), max(0.25, weight))
        weights.append(weight)
    tensor = torch.tensor(weights, dtype=torch.double)
    tensor = tensor / tensor.mean().clamp_min(1.0e-6)
    summary = {
        "enabled": True,
        "replacement": True,
        "max_weight": float(max_weight),
        "sample_weight_min": float(tensor.min().item()) if len(tensor) else None,
        "sample_weight_max": float(tensor.max().item()) if len(tensor) else None,
        "route_counts": {labels[k]: int(v) for k, v in sorted(counts.items())},
    }
    return WeightedRandomSampler(tensor, num_samples=len(records), replacement=True), summary


def _class_weights(records: list[dict[str, Any]], device: torch.device, num_classes: int) -> torch.Tensor:
    counts = Counter(int(row["route_target"]) for row in records)
    weights = torch.zeros((num_classes,), dtype=torch.float32, device=device)
    present = []
    total = float(sum(counts.values()))
    for idx in range(num_classes):
        count = int(counts.get(idx, 0))
        if count > 0:
            weights[idx] = total / float(count)
            present.append(idx)
    if present:
        weights[present] = weights[present] / weights[present].mean().clamp_min(1.0e-6)
    return weights


def _loss_fn(logits: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None, focal_gamma: float) -> torch.Tensor:
    ce = nn.functional.cross_entropy(logits.float(), target, weight=weights, reduction="none")
    if float(focal_gamma) > 0.0:
        prob = logits.float().softmax(dim=1).gather(1, target.view(-1, 1)).squeeze(1)
        ce = ce * (1.0 - prob).clamp(0.0, 1.0).pow(float(focal_gamma))
    return ce.mean()


def _confusion_summary(confusion: list[list[int]], labels: list[str]) -> dict[str, Any]:
    total = 0
    correct = 0
    recalls = []
    precisions = []
    per_class = []
    for idx, label in enumerate(labels):
        row_total = int(sum(confusion[idx]))
        col_total = int(sum(confusion[r][idx] for r in range(len(confusion))))
        hit = int(confusion[idx][idx])
        total += row_total
        correct += hit
        recall = float(hit / row_total) if row_total else None
        precision = float(hit / col_total) if col_total else None
        if recall is not None:
            recalls.append(recall)
        if precision is not None:
            precisions.append(precision)
        per_class.append(
            {
                "class_id": idx,
                "label": label,
                "target_count": row_total,
                "pred_count": col_total,
                "correct": hit,
                "recall": recall,
                "precision": precision,
            }
        )
    return {
        "matrix_target_by_pred": confusion,
        "accuracy": float(correct / total) if total else 0.0,
        "balanced_accuracy": float(sum(recalls) / len(recalls)) if recalls else 0.0,
        "macro_precision": float(sum(precisions) / len(precisions)) if precisions else 0.0,
        "per_class": per_class,
    }


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device, labels: list[str]) -> dict[str, Any]:
    confusion = [[0 for _ in range(len(labels))] for _ in range(len(labels))]
    losses = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device, non_blocking=True)
            target = batch["route_target"].to(device, non_blocking=True)
            logits = model(image)
            losses.append(float(nn.functional.cross_entropy(logits.float(), target).detach().cpu()))
            pred = logits.argmax(dim=1).detach().cpu()
            for t, p in zip(target.detach().cpu().tolist(), pred.tolist()):
                confusion[int(t)][int(p)] += 1
    summary = _confusion_summary(confusion, labels)
    summary["loss"] = float(sum(losses) / len(losses)) if losses else 0.0
    return summary


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["last_update_time_unix"] = time.time()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    _write_status(status_path, {"state": "running", "phase": "load_data"})

    backbone = str(_profile_value(args, "backbone"))
    input_size = int(_profile_value(args, "input_size"))
    batch_size = int(_profile_value(args, "batch_size"))
    lr = float(_profile_value(args, "lr"))
    labels = _load_label_vocab(args)
    if not labels:
        raise RuntimeError("empty route expert label vocabulary")

    train_records, train_summary = _load_route_records(
        args.train_jsonl,
        project_root=args.project_root,
        image_root=args.image_root,
        max_images=int(args.max_train_images),
        label_source=str(args.label_source),
        labels=labels,
    )
    val_records, val_summary = _load_route_records(
        args.val_jsonl,
        project_root=args.project_root,
        image_root=args.image_root,
        max_images=int(args.max_val_images),
        label_source=str(args.label_source),
        labels=labels,
    )
    if not train_records or not val_records:
        raise RuntimeError("empty train/val route expert dataset")

    import timm

    model = timm.create_model(backbone, pretrained=bool(args.backbone_pretrained), num_classes=len(labels))
    default_cfg = getattr(model, "default_cfg", {}) or {}
    image_mean = tuple(float(v) for v in default_cfg.get("mean", (0.485, 0.456, 0.406)))
    image_std = tuple(float(v) for v in default_cfg.get("std", (0.229, 0.224, 0.225)))

    train_dataset = RouteImageDataset(
        train_records,
        input_size=input_size,
        image_mean=image_mean,
        image_std=image_std,
        hflip_prob=float(args.hflip_prob),
    )
    val_dataset = RouteImageDataset(
        val_records,
        input_size=input_size,
        image_mean=image_mean,
        image_std=image_std,
        hflip_prob=0.0,
    )
    sampler = None
    sampler_summary = {"enabled": False}
    if bool(args.route_balanced_sampler):
        sampler, sampler_summary = _build_sampler(train_records, max_weight=float(args.sampler_max_weight), labels=labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(args.num_workers),
        pin_memory=str(args.device).startswith("cuda"),
        collate_fn=_collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=str(args.device).startswith("cuda"),
        collate_fn=_collate,
        drop_last=False,
    )

    device = torch.device(args.device)
    model.to(device)
    weights = _class_weights(train_records, device, len(labels)) if bool(args.class_weight) else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(args.weight_decay))
    total_steps = max(1, len(train_loader) * int(args.epochs))
    warmup_steps = max(0, len(train_loader) * int(args.warmup_epochs))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1.0e-3, float(step + 1) / float(warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        return max(float(args.min_lr) / max(lr, 1.0e-12), cosine)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and device.type == "cuda")
    config = {
        "profile": args.profile,
        "backbone": backbone,
        "backbone_pretrained": bool(args.backbone_pretrained),
        "input_size": input_size,
        "label_source": str(args.label_source),
        "routing_v2_vocab": str(args.routing_v2_vocab) if args.routing_v2_vocab else None,
        "batch_size": batch_size,
        "lr": lr,
        "epochs": int(args.epochs),
        "seed": int(args.seed),
        "image_mean": image_mean,
        "image_std": image_std,
        "labels": list(labels),
        "train_summary": train_summary,
        "val_summary": val_summary,
        "sampler": sampler_summary,
    }
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_status(status_path, {"state": "running", "phase": "train", "config": str(output_dir / "config.json")})

    metrics_history: list[dict[str, Any]] = []
    best_score = -1.0
    best_epoch = 0
    global_step = 0
    train_log = output_dir / "train.log"
    with train_log.open("a", encoding="utf-8") as log_fh:
        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            loss_sum = 0.0
            seen = 0
            epoch_start = time.time()
            for step, batch in enumerate(train_loader, start=1):
                image = batch["image"].to(device, non_blocking=True)
                target = batch["route_target"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=bool(args.amp) and device.type == "cuda"):
                    logits = model(image)
                    loss = _loss_fn(logits, target, weights, float(args.focal_gamma))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_step += 1
                batch_size_seen = int(target.numel())
                loss_sum += float(loss.detach().cpu()) * batch_size_seen
                seen += batch_size_seen
                if step == 1 or step % int(args.progress_log_interval) == 0:
                    payload = {
                        "event": "progress",
                        "phase": "train",
                        "epoch": epoch,
                        "step": step,
                        "samples": seen,
                        "loss": float(loss_sum / max(1, seen)),
                        "lr": float(optimizer.param_groups[0]["lr"]),
                        "elapsed_sec": round(time.time() - epoch_start, 3),
                    }
                    log_fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    log_fh.flush()
                    _write_status(status_path, {"state": "running", "phase": "train", **payload})
            val = _evaluate(model, val_loader, device, labels)
            train_loss = float(loss_sum / max(1, seen))
            score = float(val["balanced_accuracy"])
            row = {
                "epoch": epoch,
                "selection_score": score,
                "train": {"loss": train_loss},
                "val": val,
                "global_step": global_step,
            }
            metrics_history.append(row)
            (output_dir / "metrics.json").write_text(json.dumps(metrics_history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            ckpt = {
                "model_state": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "selection_score": score,
                "val": val,
            }
            torch.save(ckpt, output_dir / "last.pt")
            if score > best_score:
                best_score = score
                best_epoch = epoch
                torch.save(ckpt, output_dir / "best.pt")
            log_fh.write(json.dumps({"event": "epoch", **row}, ensure_ascii=False) + "\n")
            log_fh.flush()
            _write_status(status_path, {"state": "running", "phase": "val", "epoch": epoch, "selection_score": score, "best_score": best_score})

    summary = {
        "state": "completed",
        "phase": "completed",
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "best_checkpoint": str(output_dir / "best.pt"),
        "latest_checkpoint": str(output_dir / "last.pt"),
        "best_val": max(metrics_history, key=lambda row: row["selection_score"])["val"] if metrics_history else {},
        "history_length": len(metrics_history),
        "config": config,
        "last_update_time_unix": time.time(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_status(status_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
