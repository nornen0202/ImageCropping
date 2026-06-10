#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from scripts.evaluate_gaic_score_explanation_consistency import build_explanation_payload  # noqa: E402
from scripts.fit_gaic_teacher_mos_calibration import (  # noqa: E402
    DEFAULT_EXTRA_FEATURE_FIELDS,
    DEFAULT_SCORE_FIELDS,
    build_feature_spec,
    evaluate_rows,
    objective,
    row_to_features,
    safe_float,
    target_values,
)
from scripts.train_gaic_visual_crop_ranker import load_feature_cache, prediction_rows  # noqa: E402


DEFAULT_PROFILE_SPECS = "score_only,exp_weighted,exp_margin"


def parse_csv(text: str) -> list[str]:
    return [part.strip() for part in str(text or "").split(",") if part.strip()]


def read_rows(path: Path, *, protocol: str, max_rows: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("protocol", "")) != str(protocol):
                continue
            rows.append(row)
            if int(max_rows) > 0 and len(rows) >= int(max_rows):
                break
    return rows


def group_indices(rows: Sequence[dict[str, Any]]) -> list[np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[str(row.get("image_id", ""))].append(idx)
    return [np.asarray(indices, dtype=np.int64) for _, indices in sorted(groups.items())]


def build_compact_matrix(rows: Sequence[dict[str, Any]], spec: dict[str, Any]) -> np.ndarray:
    return np.asarray([row_to_features(row, spec) for row in rows], dtype=np.float32)


def build_explanation_arrays(rows: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
    scores: list[float] = []
    confidence: list[float] = []
    fatal: list[float] = []
    warning_count: list[float] = []
    for row in rows:
        payload = build_explanation_payload(row)
        scores.append(safe_float(payload.get("explanation_score"), 0.0))
        confidence.append(safe_float(payload.get("explanation_confidence"), 0.0))
        fatal.append(1.0 if payload.get("fatal_flag") else 0.0)
        warnings = payload.get("warnings", [])
        warning_count.append(float(len(warnings)) if isinstance(warnings, list) else 0.0)
    return {
        "explanation_score": np.asarray(scores, dtype=np.float32),
        "explanation_confidence": np.asarray(confidence, dtype=np.float32),
        "fatal_flag": np.asarray(fatal, dtype=np.float32),
        "warning_count": np.asarray(warning_count, dtype=np.float32),
    }


def method_name(profile: str) -> str:
    return f"deep_crop_ranker_{profile}"


class DeepCropSetRanker(nn.Module):
    def __init__(self, input_dim: int, *, hidden_dim: int = 256, dropout: float = 0.05) -> None:
        super().__init__()
        self.candidate_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.relation = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.score_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token = self.candidate_encoder(x)
        context = token.mean(dim=0, keepdim=True).expand_as(token)
        refined = self.relation(torch.cat([token, context, token - context], dim=-1))
        return self.score_head(token + refined).squeeze(-1)


def profile_config(profile: str) -> dict[str, float]:
    configs = {
        "score_only": {
            "listwise": 1.0,
            "pairwise": 0.35,
            "reg": 0.12,
            "align": 0.0,
            "fatal_margin": 0.0,
            "fatal_downweight": 1.0,
            "target_shift": 0.0,
            "margin": 0.08,
        },
        "exp_weighted": {
            "listwise": 1.0,
            "pairwise": 0.35,
            "reg": 0.12,
            "align": 0.04,
            "fatal_margin": 0.10,
            "fatal_downweight": 0.55,
            "target_shift": 0.0,
            "margin": 0.08,
        },
        "exp_margin": {
            "listwise": 1.0,
            "pairwise": 0.40,
            "reg": 0.10,
            "align": 0.08,
            "fatal_margin": 0.18,
            "fatal_downweight": 0.45,
            "target_shift": 0.08,
            "margin": 0.08,
        },
    }
    if profile not in configs:
        raise ValueError(f"unsupported profile: {profile}")
    return configs[profile]


def compute_group_loss(
    scores: torch.Tensor,
    rankpct: torch.Tensor,
    explanation_score: torch.Tensor,
    explanation_confidence: torch.Tensor,
    fatal_flag: torch.Tensor,
    *,
    config: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    target_rank = rankpct
    if config["target_shift"] > 0:
        target_rank = (target_rank - config["target_shift"] * fatal_flag + 0.05 * (explanation_score - 0.5)).clamp(0.0, 1.0)

    confidence_weight = (0.35 + 0.65 * explanation_confidence).clamp(0.15, 1.0)
    fatal_weight = torch.where(fatal_flag > 0.5, torch.full_like(confidence_weight, float(config["fatal_downweight"])), torch.ones_like(confidence_weight))
    candidate_weight = (confidence_weight * fatal_weight).detach()

    target_prob = torch.softmax(target_rank / 0.12, dim=0).detach()
    pred_log_prob = torch.log_softmax(scores, dim=0)
    listwise_loss = -(target_prob * pred_log_prob).sum()

    diff_target = target_rank.unsqueeze(1) - target_rank.unsqueeze(0)
    diff_pred = scores.unsqueeze(1) - scores.unsqueeze(0)
    pair_valid = diff_target.abs() >= float(config["margin"])
    if torch.any(pair_valid):
        sign = torch.where(diff_target > 0, torch.ones_like(diff_target), -torch.ones_like(diff_target))
        pair_loss = F.softplus(-sign * diff_pred)
        pair_weight = torch.sqrt(candidate_weight.unsqueeze(1).clamp_min(0.05) * candidate_weight.unsqueeze(0).clamp_min(0.05))
        pairwise_loss = (pair_loss * pair_weight * pair_valid.to(pair_loss.dtype)).sum() / (pair_weight * pair_valid.to(pair_loss.dtype)).sum().clamp_min(1.0)
    else:
        pairwise_loss = scores.sum() * 0.0

    score_prob = torch.sigmoid(scores)
    reg_loss = (((score_prob - rankpct) ** 2) * candidate_weight).sum() / candidate_weight.sum().clamp_min(1.0)

    nonfatal = fatal_flag <= 0.5
    if float(config["align"]) > 0.0 and torch.any(nonfatal):
        align_weight = explanation_confidence[nonfatal].clamp_min(0.10)
        align_loss = (((score_prob[nonfatal] - explanation_score[nonfatal]) ** 2) * align_weight).sum() / align_weight.sum().clamp_min(1.0)
    else:
        align_loss = scores.sum() * 0.0

    if float(config["fatal_margin"]) > 0.0 and torch.any(fatal_flag > 0.5) and torch.any(nonfatal):
        best_nonfatal = scores[nonfatal].max().detach()
        fatal_loss = F.relu(scores[fatal_flag > 0.5] - best_nonfatal + 0.25).mean()
    else:
        fatal_loss = scores.sum() * 0.0

    total = (
        float(config["listwise"]) * listwise_loss
        + float(config["pairwise"]) * pairwise_loss
        + float(config["reg"]) * reg_loss
        + float(config["align"]) * align_loss
        + float(config["fatal_margin"]) * fatal_loss
    )
    metrics = {
        "loss": float(total.detach().cpu()),
        "listwise": float(listwise_loss.detach().cpu()),
        "pairwise": float(pairwise_loss.detach().cpu()),
        "reg": float(reg_loss.detach().cpu()),
        "align": float(align_loss.detach().cpu()),
        "fatal_margin": float(fatal_loss.detach().cpu()),
    }
    return total, metrics


def normalize_features(x_train: np.ndarray, x_by_split: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    out = {split: ((x.astype(np.float32) - mean) / std).astype(np.float32) for split, x in x_by_split.items()}
    return out, {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "mean_shape": list(mean.shape),
        "std_min": float(std.min()),
        "std_max": float(std.max()),
    }


def predict_split(model: nn.Module, x: np.ndarray, groups: Sequence[np.ndarray], *, device: torch.device) -> list[float]:
    model.eval()
    out = np.zeros((x.shape[0],), dtype=np.float32)
    with torch.no_grad():
        for indices in groups:
            tensor = torch.from_numpy(x[indices]).to(device)
            scores = model(tensor).detach().float().cpu().numpy()
            out[indices] = scores.astype(np.float32)
    return [float(v) for v in out]


def evaluate_scores(rows: Sequence[dict[str, Any]], scores: Sequence[float], *, method: str, family: str, profile: str, epoch: int) -> dict[str, Any]:
    result = evaluate_rows(rows, scores, method=method)
    result["family"] = family
    result["profile"] = profile
    result["epoch"] = int(epoch)
    result["objective"] = round(objective(result.get("metrics", {})), 9)
    return result


def train_profile(
    *,
    profile: str,
    x_by_split: dict[str, np.ndarray],
    rows_by_split: dict[str, list[dict[str, Any]]],
    rankpct_by_split: dict[str, np.ndarray],
    explanation_by_split: dict[str, dict[str, np.ndarray]],
    groups_by_split: dict[str, list[np.ndarray]],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    config = profile_config(profile)
    model = DeepCropSetRanker(x_by_split["train"].shape[1], hidden_dim=int(args.hidden_dim), dropout=float(args.dropout)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_value = -1e18
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    history: list[dict[str, Any]] = []
    rng = random.Random(int(args.random_seed))

    train_groups = list(groups_by_split["train"])
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        rng.shuffle(train_groups)
        loss_rows: list[dict[str, float]] = []
        for step, indices in enumerate(train_groups, start=1):
            optimizer.zero_grad(set_to_none=True)
            x = torch.from_numpy(x_by_split["train"][indices]).to(device)
            rankpct = torch.from_numpy(rankpct_by_split["train"][indices]).to(device)
            expl_score = torch.from_numpy(explanation_by_split["train"]["explanation_score"][indices]).to(device)
            expl_conf = torch.from_numpy(explanation_by_split["train"]["explanation_confidence"][indices]).to(device)
            fatal = torch.from_numpy(explanation_by_split["train"]["fatal_flag"][indices]).to(device)
            scores = model(x)
            loss, loss_metrics = compute_group_loss(
                scores,
                rankpct,
                expl_score,
                expl_conf,
                fatal,
                config=config,
            )
            loss.backward()
            if float(args.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip_norm))
            optimizer.step()
            loss_rows.append(loss_metrics)
            if int(args.limit_train_steps) > 0 and step >= int(args.limit_train_steps):
                break

        val_scores = predict_split(model, x_by_split["val"], groups_by_split["val"], device=device)
        val_result = evaluate_scores(rows_by_split["val"], val_scores, method=method_name(profile), family="deep_crop_ranker", profile=profile, epoch=epoch)
        val_value = objective(val_result.get("metrics", {}))
        loss_summary = {
            key: float(sum(row.get(key, 0.0) for row in loss_rows) / max(1, len(loss_rows)))
            for key in sorted({key for row in loss_rows for key in row})
        }
        history.append({"epoch": epoch, "val": val_result, "train_loss": loss_summary})
        print(
            json.dumps(
                {
                    "status": "epoch_done",
                    "profile": profile,
                    "epoch": epoch,
                    "val_objective": round(val_value, 9),
                    "val_pcc": round(safe_float(val_result["metrics"].get("pcc"), 0.0), 6),
                    "val_srcc": round(safe_float(val_result["metrics"].get("srcc"), 0.0), 6),
                    "val_acc1_10": round(safe_float(val_result["metrics"].get("acc1_of_top10"), 0.0), 6),
                    "loss": loss_summary,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if val_value > best_value:
            best_value = float(val_value)
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            best_epoch = epoch

    if best_state is not None:
        model.load_state_dict(best_state)
    final_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
    scores_by_split = {split: predict_split(model, x_by_split[split], groups_by_split[split], device=device) for split in ("train", "val", "test")}
    evaluations = {
        split: evaluate_scores(rows_by_split[split], scores_by_split[split], method=method_name(profile), family="deep_crop_ranker", profile=profile, epoch=best_epoch)
        for split in ("train", "val", "test")
    }
    return {
        "profile": profile,
        "method": method_name(profile),
        "family": "deep_crop_ranker",
        "best_epoch": int(best_epoch),
        "best_val_objective": float(best_value),
        "config": config,
        "history": history,
        "evaluations": evaluations,
        "_state_dict": final_state,
        "_scores_by_split": scores_by_split,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def save_inference_bundle(
    path: Path,
    *,
    candidate: dict[str, Any],
    compact_spec: dict[str, Any],
    norm_meta: dict[str, Any],
    feature_fields: Sequence[str],
    visual_dim: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = candidate.get("_state_dict")
    if state is None:
        raise RuntimeError(f"candidate has no _state_dict: {candidate.get('method')}")
    torch.save(
        {
            "format": "gaic_deep_crop_ranker_bundle_v1",
            "method": str(candidate.get("method", "")),
            "profile": str(candidate.get("profile", "")),
            "config": candidate.get("config", {}),
            "model_state": state,
            "input_dim": int(len(norm_meta.get("mean", []))),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "protocol": str(args.protocol),
            "compact_spec": compact_spec,
            "feature_fields": list(feature_fields),
            "normalization": norm_meta,
            "visual_dim": int(visual_dim),
            "feature_backend": str(args.feature_backend),
            "feature_cache_key": str(args.feature_cache_key),
            "train_args": {
                "epochs": int(args.epochs),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "random_seed": int(args.random_seed),
            },
            "best_epoch": int(candidate.get("best_epoch", 0)),
            "best_val_objective": float(candidate.get("best_val_objective", 0.0)),
        },
        path,
    )


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def metric_row(candidate: dict[str, Any], split: str) -> list[str]:
    result = candidate.get("evaluations", {}).get(split, {})
    metrics = result.get("metrics", {})
    return [
        str(candidate.get("method", "")),
        str(candidate.get("profile", "")),
        str(candidate.get("best_epoch", "")),
        f"{safe_float(result.get('objective'), 0.0):.6f}",
        f"{safe_float(metrics.get('pcc'), 0.0):.6f}",
        f"{safe_float(metrics.get('srcc'), 0.0):.6f}",
        f"{safe_float(metrics.get('acc1_of_top10'), 0.0):.6f}",
        f"{safe_float(metrics.get('top1_mos'), 0.0):.6f}",
        f"{safe_float(metrics.get('top1_mos_regret'), 0.0):.6f}",
    ]


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# GAIC Deep Crop-Aware Ranker Report",
        "",
        f"- protocol: `{summary.get('protocol')}`",
        f"- selected_method_by_val: `{summary.get('selected_method_by_val')}`",
        f"- row_counts: `{summary.get('row_counts')}`",
        f"- feature_source: `{summary.get('feature_source')}`",
        "",
    ]
    for split in ("val", "test"):
        rows = [metric_row(candidate, split) for candidate in summary.get("model_candidates", [])]
        lines.extend(
            [
                f"## {split.title()} Comparison",
                "",
                markdown_table(["method", "profile", "best epoch", "objective", "PCC", "SRCC", "Acc1/10", "top1 MOS", "regret"], rows),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train deep crop-aware GAIC rankers with optional explanation-aware losses.")
    parser.add_argument("--train_candidate_eval_jsonl", type=Path, required=True)
    parser.add_argument("--val_candidate_eval_jsonl", type=Path, required=True)
    parser.add_argument("--test_candidate_eval_jsonl", type=Path, required=True)
    parser.add_argument("--protocol", choices=["Gc", "Ge"], default="Gc")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--feature_cache_dir", type=Path, required=True)
    parser.add_argument("--feature_cache_key", default="official_full")
    parser.add_argument("--feature_backend", default="timm_roi_mobilenetv4_conv_small_e3600_r256_in1k_raw")
    parser.add_argument("--score_fields", default=",".join(DEFAULT_SCORE_FIELDS))
    parser.add_argument("--extra_feature_fields", default=",".join(DEFAULT_EXTRA_FEATURE_FIELDS))
    parser.add_argument("--profiles", default=DEFAULT_PROFILE_SPECS)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--random_seed", type=int, default=20260417)
    parser.add_argument("--max_train_rows", type=int, default=0)
    parser.add_argument("--max_val_rows", type=int, default=0)
    parser.add_argument("--max_test_rows", type=int, default=0)
    parser.add_argument("--limit_train_steps", type=int, default=0)
    parser.add_argument("--write_predictions", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(int(args.random_seed))
    np.random.seed(int(args.random_seed))
    torch.manual_seed(int(args.random_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.random_seed))
    device = torch.device(str(args.device))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_split = {
        "train": read_rows(args.train_candidate_eval_jsonl, protocol=args.protocol, max_rows=int(args.max_train_rows)),
        "val": read_rows(args.val_candidate_eval_jsonl, protocol=args.protocol, max_rows=int(args.max_val_rows)),
        "test": read_rows(args.test_candidate_eval_jsonl, protocol=args.protocol, max_rows=int(args.max_test_rows)),
    }
    print(json.dumps({"status": "loaded_rows", "protocol": args.protocol, "row_counts": {k: len(v) for k, v in rows_by_split.items()}}), flush=True)

    feature_fields = list(dict.fromkeys(parse_csv(args.score_fields) + parse_csv(args.extra_feature_fields)))
    compact_spec = build_feature_spec(rows_by_split["train"], feature_fields)
    visual_by_split: dict[str, np.ndarray] = {}
    for split_name, rows in rows_by_split.items():
        visual = load_feature_cache(
            args.feature_cache_dir,
            backend=str(args.feature_backend),
            cache_key=str(args.feature_cache_key),
            split_name=split_name,
            rows=rows,
        )
        if visual is None:
            raise RuntimeError(f"missing visual feature cache for split={split_name}, backend={args.feature_backend}")
        visual_by_split[split_name] = visual.astype(np.float32, copy=False)

    x_raw = {}
    for split_name, rows in rows_by_split.items():
        compact = build_compact_matrix(rows, compact_spec)
        x_raw[split_name] = np.concatenate([visual_by_split[split_name], compact], axis=1).astype(np.float32)
    x_by_split, norm_meta = normalize_features(x_raw["train"], x_raw)
    norm_log = {key: value for key, value in norm_meta.items() if key not in {"mean", "std"}}
    rankpct_by_split = {split: target_values(rows, target="rankpct").astype(np.float32) for split, rows in rows_by_split.items()}
    explanation_by_split = {split: build_explanation_arrays(rows) for split, rows in rows_by_split.items()}
    groups_by_split = {split: group_indices(rows) for split, rows in rows_by_split.items()}

    print(
        json.dumps(
            {
                "status": "features_ready",
                "input_dim": int(x_by_split["train"].shape[1]),
                "visual_dim": int(visual_by_split["train"].shape[1]),
                "compact_dim": int(x_raw["train"].shape[1] - visual_by_split["train"].shape[1]),
                "normalization": norm_log,
                "device": str(device),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    candidates: list[dict[str, Any]] = []
    for profile in parse_csv(args.profiles):
        candidate = train_profile(
            profile=profile,
            x_by_split=x_by_split,
            rows_by_split=rows_by_split,
            rankpct_by_split=rankpct_by_split,
            explanation_by_split=explanation_by_split,
            groups_by_split=groups_by_split,
            args=args,
            device=device,
        )
        candidates.append(candidate)

    candidates.sort(key=lambda item: objective(item["evaluations"]["val"].get("metrics", {})), reverse=True)
    selected = candidates[0]
    bundle_paths: dict[str, str] = {}
    for candidate in candidates:
        method = str(candidate["method"])
        bundle_path = output_dir / f"{method}_bundle.pt"
        save_inference_bundle(
            bundle_path,
            candidate=candidate,
            compact_spec=compact_spec,
            norm_meta=norm_meta,
            feature_fields=feature_fields,
            visual_dim=int(visual_by_split["train"].shape[1]),
            args=args,
        )
        bundle_paths[method] = str(bundle_path)
    selected_bundle_path = output_dir / "selected_deep_crop_ranker_bundle.pt"
    save_inference_bundle(
        selected_bundle_path,
        candidate=selected,
        compact_spec=compact_spec,
        norm_meta=norm_meta,
        feature_fields=feature_fields,
        visual_dim=int(visual_by_split["train"].shape[1]),
        args=args,
    )
    if int(args.write_predictions):
        for candidate in candidates:
            method = str(candidate["method"])
            for split_name, rows in rows_by_split.items():
                write_jsonl(output_dir / f"predictions_{split_name}_{method}.jsonl", prediction_rows(rows, candidate["_scores_by_split"][split_name], method=method))
        for split_name, rows in rows_by_split.items():
            write_jsonl(output_dir / f"predictions_{split_name}.jsonl", prediction_rows(rows, selected["_scores_by_split"][split_name], method=str(selected["method"])))

    clean_candidates = []
    for candidate in candidates:
        clean_candidates.append({key: value for key, value in candidate.items() if not key.startswith("_")})
    summary = {
        "protocol": str(args.protocol),
        "inputs": {
            "train_candidate_eval_jsonl": str(args.train_candidate_eval_jsonl),
            "val_candidate_eval_jsonl": str(args.val_candidate_eval_jsonl),
            "test_candidate_eval_jsonl": str(args.test_candidate_eval_jsonl),
            "feature_cache_dir": str(args.feature_cache_dir),
            "feature_cache_key": str(args.feature_cache_key),
            "feature_backend": str(args.feature_backend),
        },
        "row_counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "feature_source": str(args.feature_backend),
        "input_dim": int(x_by_split["train"].shape[1]),
        "visual_dim": int(visual_by_split["train"].shape[1]),
        "compact_dim": int(x_by_split["train"].shape[1] - visual_by_split["train"].shape[1]),
        "profiles": parse_csv(args.profiles),
        "selected_method_by_val": str(selected["method"]),
        "bundle_paths": bundle_paths,
        "selected_bundle_path": str(selected_bundle_path),
        "model_candidates": clean_candidates,
        "prediction_counts": {split: len(rows) for split, rows in rows_by_split.items()} if int(args.write_predictions) else {},
    }
    write_json(output_dir / "deep_ranker_summary.json", summary)
    (output_dir / "DEEP_RANKER_REPORT.md").write_text(build_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir),
                "protocol": str(args.protocol),
                "selected_method_by_val": str(selected["method"]),
                "row_counts": summary["row_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
