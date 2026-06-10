#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet_v4.data import (  # noqa: E402
    DECISION_VOCAB,
    MobileCropNetV4BatchDataset,
    SUBJECT_BOX_TARGET_SOURCES,
    mobilecropnet_v4_collate,
)
from mobilecropnet_v4.eval_utils import (  # noqa: E402
    infer_image_norm,
    infer_input_size,
    load_mobilecropnet_v4_checkpoint,
)
from mobilecropnet_v4.model import compute_mobilecropnet_v4_loss  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an internal two-expert MobileCropNet v4 policy composite. "
            "The base expert supplies route/subject/proposal/policy heads and base-candidate scores; "
            "the crop expert supplies crop-candidate scores. No external teacher prior is used."
        )
    )
    parser.add_argument("--base_checkpoint", required=True, type=Path)
    parser.add_argument("--crop_checkpoint", required=True, type=Path)
    parser.add_argument("--eval_jsonl", required=True, type=Path)
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
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--score_margin", type=float, default=0.03)
    parser.add_argument("--gap_threshold", type=float, default=0.0)
    parser.add_argument(
        "--gap_thresholds",
        default="-0.50,-0.25,-0.10,0.00,0.10,0.25,0.50",
        help=(
            "Comma/whitespace-separated crop-vs-base score-gap thresholds to evaluate as deployment "
            "source gates. The primary --gap_threshold is always included."
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--progress_log_interval", type=int, default=25)
    parser.add_argument("--progress_json", type=Path, default=None)
    return parser


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    enriched = dict(payload)
    enriched["updated_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write_json(path, enriched)


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def _select_logits(outputs: dict[str, torch.Tensor], surface: str) -> torch.Tensor:
    surface_norm = str(surface or "deployment").strip().lower()
    if surface_norm in {"deployment", "auto"}:
        for key in ("source_mixture_return_logits", "decision_conditioned_return_logits", "return_logits", "utility_logits"):
            value = outputs.get(key)
            if torch.is_tensor(value):
                return value
    key_map = {
        "source_mixture": "source_mixture_return_logits",
        "decision_conditioned": "decision_conditioned_return_logits",
        "return": "return_logits",
        "raw_return": "return_logits",
        "utility": "utility_logits",
    }
    key = key_map.get(surface_norm, surface_norm)
    value = outputs.get(key)
    if not torch.is_tensor(value):
        available = ", ".join(sorted(k for k, v in outputs.items() if torch.is_tensor(v) and v.ndim == 2))
        raise KeyError(f"score surface {surface!r} is unavailable; available tensor surfaces: {available}")
    return value


def _masked_by_source(
    *,
    base_logits: torch.Tensor,
    crop_logits: torch.Tensor,
    valid: torch.Tensor,
    candidate_is_base: torch.Tensor,
    choose_crop: torch.Tensor,
    mask_value: float = -10000.0,
) -> torch.Tensor:
    valid_bool = valid > 0
    base_mask = (candidate_is_base > 0) & valid_bool
    crop_mask = (~(candidate_is_base > 0)) & valid_bool
    base_selected = base_logits.masked_fill(~base_mask, float(mask_value))
    crop_selected = crop_logits.masked_fill(~crop_mask, float(mask_value))
    selected = torch.where(choose_crop.view(-1, 1), crop_selected, base_selected)
    fallback = torch.maximum(base_logits, crop_logits).masked_fill(~valid_bool, float(mask_value))
    has_selected = torch.isfinite(selected).any(dim=1, keepdim=True) & (selected > (float(mask_value) / 2.0)).any(dim=1, keepdim=True)
    return torch.where(has_selected, selected, fallback)


def _best_source_gap(
    *,
    base_logits: torch.Tensor,
    crop_logits: torch.Tensor,
    valid: torch.Tensor,
    candidate_is_base: torch.Tensor,
) -> torch.Tensor:
    valid_bool = valid > 0
    base_mask = (candidate_is_base > 0) & valid_bool
    crop_mask = (~(candidate_is_base > 0)) & valid_bool
    best_base = base_logits.masked_fill(~base_mask, -10000.0).max(dim=1).values
    best_crop = crop_logits.masked_fill(~crop_mask, -10000.0).max(dim=1).values
    return best_crop - best_base


def _mean_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                continue
            if not torch.isfinite(torch.tensor(scalar)):
                continue
            sums[key] = sums.get(key, 0.0) + scalar
            counts[key] = counts.get(key, 0) + 1
    return {key: sums[key] / max(1, counts[key]) for key in sorted(sums)}


def _metrics_to_float(metrics: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            if value.numel() == 1:
                out[key] = float(value.detach().cpu().item())
        elif isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def _compact_metrics(metrics: dict[str, float]) -> dict[str, float]:
    keys = (
        "route_balanced_acc",
        "subject_box_iou",
        "subject_box_valid_best_balanced_acc",
        "proposal_recall_0_5",
        "generated_align_top1_hit",
        "source_mixture_action_source_acc",
        "source_mixture_crop_action_top1_hit",
        "source_mixture_top_return_hit",
        "source_mixture_raw_top_return_hit",
        "source_mixture_baseline_preserve_top1_hit",
        "baseline_preserve_target_rate",
        "decision_source_acc",
        "policy_source_base_recall",
        "policy_source_crop_recall",
    )
    return {key: float(metrics[key]) for key in keys if key in metrics}


def _gate_summary(
    *,
    gate_name: str,
    choose_crop: torch.Tensor,
    decision_target: torch.Tensor,
    valid: torch.Tensor,
    candidate_is_base: torch.Tensor,
) -> dict[str, float]:
    crop_id = DECISION_VOCAB.index("crop")
    valid_bool = valid > 0
    row_valid = ((candidate_is_base > 0) & valid_bool).any(dim=1) & ((candidate_is_base <= 0) & valid_bool).any(dim=1)
    target_crop = decision_target == crop_id
    base_rows = row_valid & (~target_crop)
    crop_rows = row_valid & target_crop
    source_acc = ((choose_crop == target_crop) & row_valid).to(torch.float32).sum() / row_valid.to(torch.float32).sum().clamp_min(1.0)
    base_recall = ((~choose_crop) & base_rows).to(torch.float32).sum() / base_rows.to(torch.float32).sum().clamp_min(1.0)
    crop_recall = (choose_crop & crop_rows).to(torch.float32).sum() / crop_rows.to(torch.float32).sum().clamp_min(1.0)
    return {
        f"{gate_name}_source_acc": float(source_acc.detach().cpu().item()),
        f"{gate_name}_base_recall": float(base_recall.detach().cpu().item()),
        f"{gate_name}_crop_recall": float(crop_recall.detach().cpu().item()),
        f"{gate_name}_balanced_acc": float((0.5 * (base_recall + crop_recall)).detach().cpu().item()),
        f"{gate_name}_crop_rate": float((choose_crop & row_valid).to(torch.float32).sum().detach().cpu().item() / max(1.0, float(row_valid.to(torch.float32).sum().detach().cpu().item()))),
    }


def _safe_div(numer: float, denom: float) -> float:
    return float(numer) / max(1.0, float(denom))


def _parse_float_list(value: str | None) -> list[float]:
    if value is None:
        return []
    out: list[float] = []
    for part in str(value).replace(",", " ").split():
        try:
            out.append(float(part))
        except ValueError:
            continue
    return out


def _threshold_gate_name(prefix: str, threshold: float) -> str:
    sign = "m" if float(threshold) < 0 else "p"
    value = abs(float(threshold))
    return f"{prefix}_{sign}{int(round(value * 1000)):04d}"


def _source_action_count_metrics(
    *,
    gate_name: str,
    logits: torch.Tensor,
    choose_crop: torch.Tensor,
    decision_target: torch.Tensor,
    valid: torch.Tensor,
    candidate_is_base: torch.Tensor,
    return_target: torch.Tensor,
    score_target: torch.Tensor,
    score_margin: float,
) -> dict[str, float]:
    crop_id = DECISION_VOCAB.index("crop")
    valid_bool = valid > 0
    base_mask = (candidate_is_base > 0) & valid_bool
    crop_mask = (~(candidate_is_base > 0)) & valid_bool
    row_valid = base_mask.any(dim=1) & crop_mask.any(dim=1)
    target_is_crop = decision_target == crop_id
    target_is_base = ~target_is_crop
    base_rows = row_valid & target_is_base
    crop_rows = row_valid & target_is_crop
    total_count = row_valid.to(torch.float32).sum()
    base_count = base_rows.to(torch.float32).sum()
    crop_count = crop_rows.to(torch.float32).sum()

    top_idx = logits.masked_fill(~valid_bool, -10000.0).argmax(dim=1)
    top_is_base = candidate_is_base.gather(1, top_idx[:, None]).squeeze(1) > 0
    source_correct = ((top_is_base == target_is_base) & row_valid).to(torch.float32).sum()
    baseline_hit = (top_is_base & base_rows).to(torch.float32).sum()
    crop_hit = ((~top_is_base) & crop_rows).to(torch.float32).sum()

    def top_hit_count(target: torch.Tensor) -> torch.Tensor:
        masked_target = target.masked_fill(~valid_bool, -1.0)
        best_score = masked_target.max(dim=1).values
        top_bag = valid_bool & ((best_score.unsqueeze(1) - target) <= float(score_margin))
        top_bag = top_bag & row_valid.unsqueeze(1)
        return top_bag.gather(1, top_idx[:, None]).squeeze(1).to(torch.float32).sum()

    gate_source_correct = ((choose_crop == target_is_crop) & row_valid).to(torch.float32).sum()
    gate_base_hit = ((~choose_crop) & base_rows).to(torch.float32).sum()
    gate_crop_hit = (choose_crop & crop_rows).to(torch.float32).sum()
    gate_crop_pred = (choose_crop & row_valid).to(torch.float32).sum()
    values = {
        "total_count": total_count,
        "base_count": base_count,
        "crop_count": crop_count,
        "source_correct_count": source_correct,
        "baseline_hit_count": baseline_hit,
        "crop_hit_count": crop_hit,
        "return_top_hit_count": top_hit_count(return_target),
        "raw_top_hit_count": top_hit_count(score_target),
        "gate_source_correct_count": gate_source_correct,
        "gate_base_hit_count": gate_base_hit,
        "gate_crop_hit_count": gate_crop_hit,
        "gate_crop_pred_count": gate_crop_pred,
    }
    return {f"{gate_name}_{key}": float(value.detach().cpu().item()) for key, value in values.items()}


def _accumulate_counts(target: dict[str, float], source: dict[str, float]) -> None:
    for key, value in source.items():
        target[key] = float(target.get(key, 0.0)) + float(value)


def _global_action_metrics(gate_name: str, counts: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    prefix = f"{gate_name}_"
    total = counts.get(prefix + "total_count", 0.0)
    base = counts.get(prefix + "base_count", 0.0)
    crop = counts.get(prefix + "crop_count", 0.0)
    surface = {
        "decision_source_acc": _safe_div(counts.get(prefix + "gate_source_correct_count", 0.0), total),
        "source_mixture_action_source_acc": _safe_div(counts.get(prefix + "source_correct_count", 0.0), total),
        "source_mixture_baseline_preserve_top1_hit": _safe_div(counts.get(prefix + "baseline_hit_count", 0.0), base),
        "source_mixture_crop_action_top1_hit": _safe_div(counts.get(prefix + "crop_hit_count", 0.0), crop),
        "source_mixture_top_return_hit": _safe_div(counts.get(prefix + "return_top_hit_count", 0.0), total),
        "source_mixture_raw_top_return_hit": _safe_div(counts.get(prefix + "raw_top_hit_count", 0.0), total),
        "baseline_preserve_target_rate": _safe_div(base, total),
    }
    base_recall = _safe_div(counts.get(prefix + "gate_base_hit_count", 0.0), base)
    crop_recall = _safe_div(counts.get(prefix + "gate_crop_hit_count", 0.0), crop)
    gate = {
        f"{gate_name}_source_acc": _safe_div(counts.get(prefix + "gate_source_correct_count", 0.0), total),
        f"{gate_name}_base_recall": base_recall,
        f"{gate_name}_crop_recall": crop_recall,
        f"{gate_name}_balanced_acc": 0.5 * (base_recall + crop_recall),
        f"{gate_name}_crop_rate": _safe_div(counts.get(prefix + "gate_crop_pred_count", 0.0), total),
        f"{gate_name}_total_count": float(total),
        f"{gate_name}_base_count": float(base),
        f"{gate_name}_crop_count": float(crop),
    }
    return surface, gate


def _run_loss_metrics(outputs: dict[str, torch.Tensor], batch: dict[str, Any], score_margin: float) -> dict[str, float]:
    with torch.amp.autocast(device_type="cuda", enabled=False):
        _, metrics = compute_mobilecropnet_v4_loss(
            outputs,
            batch,
            score_weight=0.0,
            listwise_weight=0.0,
            pairwise_weight=0.0,
            explicit_pairwise_weight=0.0,
            positive_weight=0.0,
            risk_weight=0.0,
            top1_risk_weight=0.0,
            top_return_weight=0.0,
            raw_top_return_weight=0.0,
            return_score_weight=0.0,
            return_listwise_weight=0.0,
            return_positive_weight=0.0,
            macro_weight=0.0,
            checklist_class_weight=0.0,
            checklist_applicability_weight=0.0,
            detail_score_weight=0.0,
            why_tag_weight=0.0,
            route_weight=0.0,
            decision_weight=0.0,
            action_consistency_weight=0.0,
            decision_utility_align_weight=0.0,
            policy_score_weight=0.0,
            delta_weight=0.0,
            proposal_weight=0.0,
            proposal_subject_weight=0.0,
            subject_proposal_align_weight=0.0,
            subject_box_weight=0.0,
            generated_proposal_align_weight=0.0,
            top_return_score_margin=float(score_margin),
        )
    return _metrics_to_float(metrics)


def _update_outputs_with_composite(
    *,
    base_outputs: dict[str, torch.Tensor],
    composite_logits: torch.Tensor,
    gate_signal: torch.Tensor,
) -> dict[str, torch.Tensor]:
    outputs = dict(base_outputs)
    outputs["return_logits"] = composite_logits
    outputs["source_mixture_return_logits"] = composite_logits
    outputs["decision_conditioned_return_logits"] = composite_logits
    outputs["decision_source_logit"] = gate_signal
    outputs["decision_source_raw_logit"] = gate_signal
    outputs["decision_source_logit_threshold"] = torch.zeros_like(gate_signal)
    return outputs


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.progress_json or (args.output_dir / "progress.json")
    summary_path = args.output_dir / "summary.json"
    metrics_path = args.output_dir / "metrics.json"
    device = torch.device(args.device)

    _write_progress(
        progress_path,
        {
            "state": "running",
            "phase": "load_checkpoints",
            "base_checkpoint": str(args.base_checkpoint),
            "crop_checkpoint": str(args.crop_checkpoint),
            "eval_jsonl": str(args.eval_jsonl),
            "output_dir": str(args.output_dir),
        },
    )
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
    dataset = MobileCropNetV4BatchDataset(
        jsonl_path=args.eval_jsonl,
        project_root=args.project_root,
        image_root=args.image_root,
        input_size=input_size,
        candidate_k=candidate_k,
        image_mean=image_mean,
        image_std=image_std,
        max_rows=args.max_rows,
        subject_box_target_source=subject_box_target_source,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=mobilecropnet_v4_collate,
    )

    gap_thresholds = [float(args.gap_threshold)]
    for threshold in _parse_float_list(args.gap_thresholds):
        if all(abs(float(threshold) - existing) > 1e-9 for existing in gap_thresholds):
            gap_thresholds.append(float(threshold))
    metric_rows: dict[str, list[dict[str, float]]] = {
        "oracle": [],
        "base_decision": [],
        "base_source": [],
        "gap_threshold": [],
    }
    for threshold in gap_thresholds:
        metric_rows.setdefault(_threshold_gate_name("gap", threshold), [])
    gate_rows: dict[str, list[dict[str, float]]] = {name: [] for name in metric_rows}
    count_sums: dict[str, dict[str, float]] = {name: {} for name in metric_rows}
    started_at = time.monotonic()
    processed = 0
    crop_id = DECISION_VOCAB.index("crop")
    _write_progress(
        progress_path,
        {
            "state": "running",
            "phase": "composite_eval",
            "image_count": len(dataset),
            "batch_size": int(args.batch_size),
            "processed_batches": 0,
            "processed_images": 0,
            "summary_json": str(summary_path),
            "metrics_json": str(metrics_path),
        },
    )

    with torch.no_grad():
        for batch_idx, raw_batch in enumerate(loader, start=1):
            batch = _to_device(raw_batch, device)
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
            with torch.amp.autocast(device_type="cuda", enabled=device.type == "cuda"):
                base_outputs = base_model(*common_args, return_generated_scores=True)
                crop_outputs = crop_model(*common_args, return_generated_scores=False)
            base_logits = _select_logits(base_outputs, args.base_surface).float()
            crop_logits = _select_logits(crop_outputs, args.crop_surface).float()
            valid = batch["valid"]
            candidate_is_base = batch["candidate_is_base"]
            decision_target = batch["decision_target"]
            gap = _best_source_gap(
                base_logits=base_logits,
                crop_logits=crop_logits,
                valid=valid,
                candidate_is_base=candidate_is_base,
            )
            gates = {
                "oracle": decision_target == crop_id,
                "base_decision": base_outputs["decision_logits"].argmax(dim=1) == crop_id,
                "base_source": base_outputs.get("decision_source_logit", torch.full_like(gap, -1.0)).view(-1) >= 0.0,
                "gap_threshold": gap >= float(args.gap_threshold),
            }
            for threshold in gap_thresholds:
                gates[_threshold_gate_name("gap", threshold)] = gap >= float(threshold)
            for gate_name, choose_crop in gates.items():
                composite_logits = _masked_by_source(
                    base_logits=base_logits,
                    crop_logits=crop_logits,
                    valid=valid,
                    candidate_is_base=candidate_is_base,
                    choose_crop=choose_crop,
                )
                outputs = _update_outputs_with_composite(
                    base_outputs=base_outputs,
                    composite_logits=composite_logits,
                    gate_signal=gap if gate_name == "gap_threshold" else choose_crop.to(base_logits.dtype) * 2.0 - 1.0,
                )
                metric_rows[gate_name].append(_run_loss_metrics(outputs, batch, args.score_margin))
                gate_rows[gate_name].append(
                    _gate_summary(
                        gate_name=gate_name,
                        choose_crop=choose_crop,
                        decision_target=decision_target,
                        valid=valid,
                        candidate_is_base=candidate_is_base,
                    )
                )
                _accumulate_counts(
                    count_sums[gate_name],
                    _source_action_count_metrics(
                        gate_name=gate_name,
                        logits=composite_logits,
                        choose_crop=choose_crop,
                        decision_target=decision_target,
                        valid=valid,
                        candidate_is_base=candidate_is_base,
                        return_target=batch["score_target"],
                        score_target=batch["score_target"],
                        score_margin=float(args.score_margin),
                    ),
                )
            processed += int(batch["image"].shape[0])
            if args.progress_log_interval > 0 and (batch_idx == 1 or batch_idx % int(args.progress_log_interval) == 0):
                elapsed = max(1e-6, time.monotonic() - started_at)
                payload = {
                    "state": "running",
                    "phase": "composite_eval",
                    "image_count": len(dataset),
                    "processed_batches": batch_idx,
                    "processed_images": processed,
                    "samples_per_sec": round(processed / elapsed, 4),
                    "elapsed_sec": round(elapsed, 3),
                    "summary_json": str(summary_path),
                    "metrics_json": str(metrics_path),
                }
                _write_progress(progress_path, payload)
                print(json.dumps({"event": "policy_composite_progress", **payload}, ensure_ascii=False), file=sys.stderr, flush=True)

    metrics_by_gate: dict[str, dict[str, Any]] = {}
    for gate_name, rows in metric_rows.items():
        metrics = _mean_metric_rows(rows)
        gate_metrics = _mean_metric_rows(gate_rows[gate_name])
        global_surface_metrics, global_gate_metrics = _global_action_metrics(gate_name, count_sums[gate_name])
        metrics.update(global_surface_metrics)
        gate_metrics.update(global_gate_metrics)
        metrics_by_gate[gate_name] = {
            "metrics": metrics,
            "gate": gate_metrics,
            "compact": _compact_metrics(metrics),
            "global_counts": count_sums[gate_name],
        }
    summary = {
        "state": "completed",
        "base_checkpoint": str(args.base_checkpoint),
        "crop_checkpoint": str(args.crop_checkpoint),
        "eval_jsonl": str(args.eval_jsonl),
        "output_dir": str(args.output_dir),
        "image_count": int(len(dataset)),
        "input_size": int(input_size),
        "candidate_k": int(candidate_k),
        "base_surface": str(args.base_surface),
        "crop_surface": str(args.crop_surface),
        "runtime_subject_prior_mode": str(args.runtime_subject_prior_mode),
        "gap_threshold": float(args.gap_threshold),
        "gap_thresholds": [float(value) for value in gap_thresholds],
        "subject_box_target_source": str(subject_box_target_source),
        "dataset_summary": dataset.summary(),
        "gates": metrics_by_gate,
    }
    _write_json(summary_path, summary)
    _write_json(metrics_path, metrics_by_gate)
    _write_progress(
        progress_path,
        {
            "state": "completed",
            "phase": "composite_eval",
            "image_count": len(dataset),
            "processed_batches": len(loader),
            "processed_images": len(dataset),
            "summary_json": str(summary_path),
            "metrics_json": str(metrics_path),
        },
    )
    print(json.dumps({k: v["compact"] | v["gate"] for k, v in metrics_by_gate.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
