#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet.bank import build_static_micro_bank
from mobilecropnet.data import DECISION_VOCAB, load_image_tensor, target_ar_id
from mobilecropnet.geometry import box_area_norm, xyxy_norm_to_xywh_pixels
from mobilecropnet.model import MobileCropNet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MobileCropNet inference with host static micro-bank candidates.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--image_list", type=Path)
    parser.add_argument("--output_jsonl", required=True, type=Path)
    parser.add_argument("--target_ar", default="FREE")
    parser.add_argument("--input_size", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--static_area_prior", type=float, default=None)
    parser.add_argument("--static_area_penalty", type=float, default=0.0)
    parser.add_argument("--static_full_penalty", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--include_candidates", action="store_true")
    return parser


def _is_full_crop(box: list[float]) -> bool:
    if len(box) < 4:
        return False
    return (
        abs(float(box[0])) < 1e-6
        and abs(float(box[1])) < 1e-6
        and abs(float(box[2]) - 1.0) < 1e-6
        and abs(float(box[3]) - 1.0) < 1e-6
    )


def _adjust_static_score(
    *,
    raw_score: float,
    box: list[float],
    area_prior: float | None,
    area_penalty: float,
    full_penalty: float,
) -> tuple[float, float]:
    penalty = 0.0
    if area_prior is not None and area_penalty > 0.0:
        penalty += float(area_penalty) * abs(box_area_norm(box) - max(0.0, min(1.0, float(area_prior))))
    if full_penalty > 0.0 and _is_full_crop(box):
        penalty += float(full_penalty)
    return float(raw_score) - penalty, penalty


def _load_model(checkpoint_path: Path, device: torch.device, k_override: int | None) -> tuple[MobileCropNet, dict[str, Any]]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = dict(ckpt.get("model_config", {}))
    if k_override is not None:
        model_config["k"] = int(k_override)
    model = MobileCropNet(**model_config).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, ckpt


def _iter_images(args: argparse.Namespace) -> list[Path]:
    paths = [Path(p) for p in args.image]
    if args.image_list:
        for line in args.image_list.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                paths.append(Path(line))
    if not paths:
        raise ValueError("provide --image or --image_list")
    return paths


def _predict_one(
    *,
    model: MobileCropNet,
    image_path: Path,
    target_ar: str,
    input_size: int,
    k: int,
    device: torch.device,
    include_candidates: bool,
    static_area_prior: float | None,
    static_area_penalty: float,
    static_full_penalty: float,
) -> dict[str, Any]:
    with Image.open(image_path) as img:
        width, height = img.size
    candidates = build_static_micro_bank(image_width=width, image_height=height, target_ar=target_ar, k=k)
    boxes = torch.zeros((1, k, 4), dtype=torch.float32, device=device)
    valid = torch.zeros((1, k), dtype=torch.float32, device=device)
    for idx, cand in enumerate(candidates[:k]):
        boxes[0, idx] = torch.tensor(cand.bbox_norm_xyxy, dtype=torch.float32, device=device)
        valid[0, idx] = 1.0
    image = load_image_tensor(image_path, input_size).unsqueeze(0).to(device)
    ar_id = torch.tensor([target_ar_id(target_ar)], dtype=torch.long, device=device)
    with torch.inference_mode():
        outputs = model(image, boxes, valid, ar_id)
        raw_scores = torch.sigmoid(outputs["score_logits"]).masked_fill(valid <= 0, -1.0)
        adjusted_scores = raw_scores.clone()
        score_penalties: list[float] = []
        for idx, cand in enumerate(candidates[:k]):
            adjusted, penalty = _adjust_static_score(
                raw_score=float(raw_scores[0, idx].detach().cpu()),
                box=cand.bbox_norm_xyxy,
                area_prior=static_area_prior,
                area_penalty=static_area_penalty,
                full_penalty=static_full_penalty,
            )
            adjusted_scores[0, idx] = adjusted
            score_penalties.append(float(penalty))
        best_idx = int(adjusted_scores.argmax(dim=1).item())
        decision_id = int(outputs["decision_logits"].argmax(dim=1).item())
    best_box = boxes[0, best_idx].detach().cpu().tolist()
    record: dict[str, Any] = {
        "image_path": str(image_path),
        "target_ar": target_ar,
        "width": width,
        "height": height,
        "chosen_candidate_id": candidates[best_idx].candidate_id,
        "chosen_source": candidates[best_idx].source,
        "bbox_norm_xyxy": [round(float(v), 9) for v in best_box],
        "bbox_xywh": [round(float(v), 3) for v in xyxy_norm_to_xywh_pixels(best_box, width=width, height=height)],
        "score": round(float(adjusted_scores[0, best_idx].detach().cpu()), 9),
        "raw_score": round(float(raw_scores[0, best_idx].detach().cpu()), 9),
        "score_penalty": round(float(score_penalties[best_idx]), 9),
        "decision_id": decision_id,
        "decision_type": DECISION_VOCAB[decision_id] if 0 <= decision_id < len(DECISION_VOCAB) else str(decision_id),
    }
    if include_candidates:
        score_values = adjusted_scores[0].detach().cpu().tolist()
        raw_score_values = raw_scores[0].detach().cpu().tolist()
        record["candidates"] = [
            {
                "candidate_id": cand.candidate_id,
                "source": cand.source,
                "bbox_norm_xyxy": [round(float(v), 9) for v in cand.bbox_norm_xyxy],
                "score": round(float(score_values[idx]), 9),
                "raw_score": round(float(raw_score_values[idx]), 9),
                "score_penalty": round(float(score_penalties[idx]), 9),
            }
            for idx, cand in enumerate(candidates[:k])
        ]
    return record


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    model, ckpt = _load_model(args.checkpoint, device, args.k)
    train_config = ckpt.get("train_config", {})
    input_size = int(args.input_size or train_config.get("input_size", 256))
    k = int(args.k or model.config.k)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for image_path in _iter_images(args):
            record = _predict_one(
                model=model,
                image_path=image_path,
                target_ar=args.target_ar,
                input_size=input_size,
                k=k,
                device=device,
                include_candidates=args.include_candidates,
                static_area_prior=args.static_area_prior,
                static_area_penalty=args.static_area_penalty,
                static_full_penalty=args.static_full_penalty,
            )
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
