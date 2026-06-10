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

from mobilecropnet_v4.data import SUBJECT_BOX_TARGET_SOURCES, SUBJECT_MODE_VOCAB, MobileCropNetV4BatchDataset, mobilecropnet_v4_collate
from mobilecropnet_v4.eval_utils import infer_image_norm, infer_input_size, load_mobilecropnet_v4_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate MobileCropNet v4 route logits with a folded checkpoint bias.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--calibration_jsonl", required=True, type=Path)
    parser.add_argument("--eval_jsonl", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--project_root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--image_root", type=Path)
    parser.add_argument("--input_size", type=int)
    parser.add_argument("--candidate_k", type=int)
    parser.add_argument("--image_mean")
    parser.add_argument("--image_std")
    parser.add_argument("--subject_box_target_source", choices=SUBJECT_BOX_TARGET_SOURCES, default=None)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_calibration_rows", type=int, default=12000)
    parser.add_argument("--max_eval_rows", type=int, default=4200)
    parser.add_argument("--bias_span", type=float, default=3.0)
    parser.add_argument("--bias_step", type=float, default=0.25)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--save_checkpoint_name", default="route_bias_calibrated.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _dataset(
    *,
    jsonl_path: Path,
    project_root: Path,
    image_root: Path | None,
    input_size: int,
    candidate_k: int,
    image_mean: list[float],
    image_std: list[float],
    max_rows: int,
    subject_box_target_source: str,
) -> MobileCropNetV4BatchDataset:
    return MobileCropNetV4BatchDataset(
        jsonl_path=jsonl_path,
        project_root=project_root,
        image_root=image_root,
        input_size=input_size,
        candidate_k=candidate_k,
        image_mean=image_mean,
        image_std=image_std,
        max_rows=max_rows,
        subject_box_target_source=subject_box_target_source,
    )


def collect_route_logits(
    *,
    model: torch.nn.Module,
    dataset: MobileCropNetV4BatchDataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        collate_fn=mobilecropnet_v4_collate,
    )
    logits_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            tensor_batch = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}
            outputs = model(
                tensor_batch["image"],
                tensor_batch["boxes"],
                tensor_batch["valid"],
                tensor_batch["target_ar_id"],
                tensor_batch["image_ar_log"],
                tensor_batch["candidate_is_base"],
                tensor_batch["box_meta"],
                tensor_batch.get("letterbox_content_box"),
                tensor_batch.get("subject_prior_box"),
                tensor_batch.get("subject_prior_valid"),
                tensor_batch.get("subject_prior_reliability"),
            )
            logits_rows.append(outputs["route_logits"].detach().float().cpu())
            target_rows.append(batch["route_target"].detach().long().cpu())
    if not logits_rows:
        raise RuntimeError("no rows were available for route calibration")
    return torch.cat(logits_rows, dim=0), torch.cat(target_rows, dim=0)


def route_confusion(logits: torch.Tensor, targets: torch.Tensor, bias: torch.Tensor | None = None) -> dict[str, Any]:
    logits = logits.float().cpu()
    targets = targets.long().cpu()
    if bias is not None:
        logits = logits + bias.float().cpu().view(1, -1)
    class_count = int(logits.shape[1])
    pred = logits.argmax(dim=1)
    matrix = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for target, out in zip(targets.tolist(), pred.tolist()):
        if 0 <= int(target) < class_count and 0 <= int(out) < class_count:
            matrix[int(target)][int(out)] += 1
    per_class: list[dict[str, Any]] = []
    recalls: list[float] = []
    precisions: list[float] = []
    total_correct = 0
    total = 0
    for idx in range(class_count):
        row_total = int(sum(matrix[idx]))
        col_total = int(sum(matrix[row][idx] for row in range(class_count)))
        correct = int(matrix[idx][idx])
        total += row_total
        total_correct += correct
        recall = float(correct / row_total) if row_total > 0 else None
        precision = float(correct / col_total) if col_total > 0 else None
        if recall is not None:
            recalls.append(recall)
        if precision is not None:
            precisions.append(precision)
        label = SUBJECT_MODE_VOCAB[idx] if idx < len(SUBJECT_MODE_VOCAB) else f"class_{idx}"
        per_class.append(
            {
                "class_id": idx,
                "label": label,
                "target_count": row_total,
                "pred_count": col_total,
                "correct": correct,
                "recall": recall,
                "precision": precision,
            }
        )
    return {
        "accuracy": float(total_correct / total) if total else 0.0,
        "balanced_accuracy": float(sum(recalls) / len(recalls)) if recalls else 0.0,
        "macro_precision": float(sum(precisions) / len(precisions)) if precisions else 0.0,
        "matrix_target_by_pred": matrix,
        "per_class": per_class,
    }


def optimize_route_bias(logits: torch.Tensor, targets: torch.Tensor, *, span: float, step: float, rounds: int) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    class_count = int(logits.shape[1])
    bias = torch.zeros((class_count,), dtype=torch.float32)
    deltas = torch.arange(-float(span), float(span) + float(step) * 0.5, float(step), dtype=torch.float32)
    history: list[dict[str, Any]] = []
    best_score = float(route_confusion(logits, targets, bias)["balanced_accuracy"])
    for round_idx in range(max(1, int(rounds))):
        changed = False
        for class_idx in range(class_count):
            class_best_score = best_score
            class_best_delta = 0.0
            for delta in deltas.tolist():
                candidate = bias.clone()
                candidate[class_idx] += float(delta)
                candidate -= candidate.mean()
                score = float(route_confusion(logits, targets, candidate)["balanced_accuracy"])
                if score > class_best_score + 1e-12:
                    class_best_score = score
                    class_best_delta = float(delta)
            if abs(class_best_delta) > 1e-12:
                bias[class_idx] += class_best_delta
                bias -= bias.mean()
                best_score = class_best_score
                changed = True
            history.append(
                {
                    "round": int(round_idx + 1),
                    "class_id": int(class_idx),
                    "class_label": SUBJECT_MODE_VOCAB[class_idx] if class_idx < len(SUBJECT_MODE_VOCAB) else f"class_{class_idx}",
                    "applied_delta": float(class_best_delta),
                    "calibration_balanced_accuracy": float(best_score),
                    "bias": [float(v) for v in bias.tolist()],
                }
            )
        if not changed:
            break
    return bias, history


def _add_route_bias_to_checkpoint(ckpt: dict[str, Any], bias: torch.Tensor, *, source_checkpoint: Path) -> dict[str, Any]:
    out = dict(ckpt)
    model_state = dict(out["model"])
    candidates: list[tuple[int, str]] = []
    for key, value in model_state.items():
        if not key.startswith("route_head.") or not key.endswith(".bias"):
            continue
        if tuple(value.shape) != tuple(bias.shape):
            continue
        parts = key.split(".")
        layer_idx = int(parts[1]) if len(parts) > 2 and parts[1].isdigit() else -1
        candidates.append((layer_idx, key))
    if not candidates:
        available = {
            key: tuple(value.shape)
            for key, value in model_state.items()
            if key.startswith("route_head.") and key.endswith(".bias")
        }
        raise ValueError(f"no route logits bias key matches calibration shape {tuple(bias.shape)}; available={available}")
    _, key = max(candidates)
    model_state[key] = model_state[key].detach().cpu() + bias.detach().cpu().to(model_state[key].dtype)
    out["model"] = model_state
    train_config = dict(out.get("train_config", {}))
    train_config["route_bias_calibration"] = {
        "source_checkpoint": str(source_checkpoint),
        "folded_state_key": key,
        "bias": [float(v) for v in bias.tolist()],
    }
    out["train_config"] = train_config
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "status.json"
    started = time()
    _write_json(status_path, {"state": "running", "phase": "load", "checkpoint": str(args.checkpoint)})

    device = torch.device(args.device)
    model, ckpt = load_mobilecropnet_v4_checkpoint(args.checkpoint, device=device)
    input_size = infer_input_size(ckpt, args.input_size)
    image_mean, image_std = infer_image_norm(ckpt, explicit_mean=args.image_mean, explicit_std=args.image_std)
    subject_box_target_source = str(
        args.subject_box_target_source
        or ckpt.get("train_config", {}).get("subject_box_target_source")
        or ckpt.get("dataset_summary", {}).get("subject_box_target_source")
        or "legacy"
    )
    candidate_k = int(args.candidate_k or ckpt.get("model_config", {}).get("candidate_k", 24))

    _write_json(status_path, {"state": "running", "phase": "collect_calibration", "checkpoint": str(args.checkpoint)})
    calibration_ds = _dataset(
        jsonl_path=args.calibration_jsonl,
        project_root=args.project_root,
        image_root=args.image_root,
        input_size=input_size,
        candidate_k=candidate_k,
        image_mean=image_mean,
        image_std=image_std,
        max_rows=args.max_calibration_rows,
        subject_box_target_source=subject_box_target_source,
    )
    cal_logits, cal_targets = collect_route_logits(
        model=model,
        dataset=calibration_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    _write_json(status_path, {"state": "running", "phase": "collect_eval", "checkpoint": str(args.checkpoint)})
    eval_ds = _dataset(
        jsonl_path=args.eval_jsonl,
        project_root=args.project_root,
        image_root=args.image_root,
        input_size=input_size,
        candidate_k=candidate_k,
        image_mean=image_mean,
        image_std=image_std,
        max_rows=args.max_eval_rows,
        subject_box_target_source=subject_box_target_source,
    )
    eval_logits, eval_targets = collect_route_logits(
        model=model,
        dataset=eval_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    _write_json(status_path, {"state": "running", "phase": "optimize_bias", "checkpoint": str(args.checkpoint)})
    bias, history = optimize_route_bias(cal_logits, cal_targets, span=args.bias_span, step=args.bias_step, rounds=args.rounds)
    checkpoint_payload = _add_route_bias_to_checkpoint(ckpt, bias, source_checkpoint=args.checkpoint)
    calibrated_checkpoint = args.output_dir / args.save_checkpoint_name
    torch.save(checkpoint_payload, calibrated_checkpoint)

    summary = {
        "state": "completed",
        "checkpoint": str(args.checkpoint),
        "calibrated_checkpoint": str(calibrated_checkpoint),
        "calibration_jsonl": str(args.calibration_jsonl),
        "eval_jsonl": str(args.eval_jsonl),
        "subject_box_target_source": subject_box_target_source,
        "input_size": int(input_size),
        "candidate_k": int(candidate_k),
        "max_calibration_rows": int(args.max_calibration_rows),
        "max_eval_rows": int(args.max_eval_rows),
        "bias": [float(v) for v in bias.tolist()],
        "history": history,
        "calibration_before": route_confusion(cal_logits, cal_targets),
        "calibration_after": route_confusion(cal_logits, cal_targets, bias),
        "eval_before": route_confusion(eval_logits, eval_targets),
        "eval_after": route_confusion(eval_logits, eval_targets, bias),
        "duration_sec": float(time() - started),
    }
    _write_json(args.output_dir / "route_bias_calibration_summary.json", summary)
    _write_json(status_path, {"state": "completed", "phase": "completed", "summary": str(args.output_dir / "route_bias_calibration_summary.json")})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
