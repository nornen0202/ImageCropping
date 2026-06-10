#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from scripts.fit_gaic_teacher_mos_calibration import row_to_features  # noqa: E402
from scripts.train_gaic_deep_crop_ranker import DeepCropSetRanker  # noqa: E402
from scripts.train_gaic_visual_crop_ranker import load_feature_cache  # noqa: E402


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except (TypeError, ValueError):
        return float(default)


def read_rows(path: Path, *, protocol: str | None = None, max_rows: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if protocol is not None and str(row.get("protocol", "")) != str(protocol):
                continue
            rows.append(row)
            if int(max_rows) > 0 and len(rows) >= int(max_rows):
                break
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def group_indices(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> list[np.ndarray]:
    groups: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[tuple(str(row.get(key, "")) for key in keys)].append(idx)
    return [np.asarray(indices, dtype=np.int64) for _, indices in sorted(groups.items())]


def rank_pct_by_groups(scores: Sequence[float], groups: Sequence[np.ndarray]) -> list[float]:
    out = [0.0] * len(scores)
    for indices in groups:
        if len(indices) <= 1:
            for idx in indices:
                out[int(idx)] = 1.0
            continue
        sorted_indices = sorted([int(idx) for idx in indices], key=lambda idx: (float(scores[idx]), -idx))
        denom = float(max(1, len(sorted_indices) - 1))
        for rank, idx in enumerate(sorted_indices):
            out[idx] = float(rank) / denom
    return out


def build_feature_matrix(rows: Sequence[dict[str, Any]], *, compact_spec: dict[str, Any], visual: np.ndarray, norm: dict[str, Any]) -> np.ndarray:
    compact = np.asarray([row_to_features(row, compact_spec) for row in rows], dtype=np.float32)
    raw = np.concatenate([visual.astype(np.float32, copy=False), compact], axis=1).astype(np.float32)
    mean = np.asarray(norm.get("mean"), dtype=np.float32)
    std = np.asarray(norm.get("std"), dtype=np.float32)
    if mean.shape[0] != raw.shape[1] or std.shape[0] != raw.shape[1]:
        raise RuntimeError(f"normalization dim mismatch: raw={raw.shape[1]} mean={mean.shape} std={std.shape}")
    std = np.where(std < 1e-6, 1.0, std)
    return ((raw - mean) / std).astype(np.float32)


def predict_scores(model: DeepCropSetRanker, x: np.ndarray, groups: Sequence[np.ndarray], *, device: torch.device) -> list[float]:
    model.eval()
    out = np.zeros((x.shape[0],), dtype=np.float32)
    with torch.no_grad():
        for indices in groups:
            tensor = torch.from_numpy(x[indices]).to(device)
            out[indices] = model(tensor).detach().float().cpu().numpy().astype(np.float32)
    return [float(v) for v in out]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score arbitrary candidate rows with a saved T6 deep crop ranker bundle.")
    parser.add_argument("--candidate_eval_jsonl", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--feature_cache_dir", type=Path, required=True)
    parser.add_argument("--feature_cache_key", required=True)
    parser.add_argument("--feature_backend", default=None)
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, required=True)
    parser.add_argument("--group_keys", default="image_id,target_ar")
    parser.add_argument("--score_source", default="t6_deep_crop_ranker_bundle")
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = torch.load(args.bundle, map_location="cpu")
    if str(bundle.get("format", "")) != "gaic_deep_crop_ranker_bundle_v1":
        raise RuntimeError(f"unsupported bundle format: {bundle.get('format')}")
    rows = read_rows(args.candidate_eval_jsonl, protocol=args.protocol, max_rows=int(args.max_rows))
    backend = str(args.feature_backend or bundle.get("feature_backend"))
    visual = load_feature_cache(
        args.feature_cache_dir,
        backend=backend,
        cache_key=str(args.feature_cache_key),
        split_name=str(args.split_name),
        rows=rows,
    )
    if visual is None:
        raise RuntimeError(f"missing feature cache split={args.split_name} backend={backend} key={args.feature_cache_key}")
    x = build_feature_matrix(rows, compact_spec=bundle["compact_spec"], visual=visual, norm=bundle["normalization"])
    device = torch.device(str(args.device))
    model = DeepCropSetRanker(int(bundle["input_dim"]), hidden_dim=int(bundle["hidden_dim"]), dropout=float(bundle["dropout"])).to(device)
    model.load_state_dict(bundle["model_state"])
    keys = [part.strip() for part in str(args.group_keys).split(",") if part.strip()]
    groups = group_indices(rows, keys)
    scores = predict_scores(model, x, groups, device=device)
    rank_pct = rank_pct_by_groups(scores, groups)
    method = str(bundle.get("method", "deep_crop_ranker"))

    def iter_rows() -> Iterable[dict[str, Any]]:
        selected = set()
        for indices in groups:
            if len(indices):
                selected.add(int(max([int(idx) for idx in indices], key=lambda idx: (scores[idx], -idx))))
        for idx, row in enumerate(rows):
            out = dict(row)
            out.update(
                {
                    "score_source": str(args.score_source),
                    "ranker_method": method,
                    "ranker_score": float(scores[idx]),
                    "score_rank_pct_by_group": float(rank_pct[idx]),
                    "score_rank_pct_by_image": float(rank_pct[idx]),
                    "selected_by_ranker": bool(idx in selected),
                    "t6_bundle": str(args.bundle),
                    "t6_group_keys": keys,
                }
            )
            yield out

    count = write_jsonl(args.output_jsonl, iter_rows())
    summary = {
        "status": "ok",
        "candidate_eval_jsonl": str(args.candidate_eval_jsonl),
        "bundle": str(args.bundle),
        "output_jsonl": str(args.output_jsonl),
        "row_count": int(count),
        "group_count": int(len(groups)),
        "group_keys": keys,
        "feature_backend": backend,
        "feature_cache_key": str(args.feature_cache_key),
        "split_name": str(args.split_name),
        "method": method,
    }
    write_json(args.summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
