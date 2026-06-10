#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet_v4.model import MobileCropNetV4
from scripts.train_mobilecropnet_v4 import MODEL_PROFILE_PRESETS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Instantiate MobileCropNet v4 profiles and run a tiny forward pass.")
    parser.add_argument("--profiles", nargs="+", default=["q24_288_w075", "rank_320", "plus_384"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} device={device}")
    if device.type == "cuda":
        print(f"device_name={torch.cuda.get_device_name(0)}")
    for profile in args.profiles:
        cfg = dict(MODEL_PROFILE_PRESETS[profile])
        input_size = int(cfg.pop("input_size", 96))
        cfg.pop("image_mean", None)
        cfg.pop("image_std", None)
        cfg["input_size"] = input_size
        print(f"profile={profile} input_size={input_size} cfg={cfg}")
        model = MobileCropNetV4(**cfg).to(device).eval()
        image = torch.zeros((1, 3, input_size, input_size), dtype=torch.float32, device=device)
        with torch.no_grad():
            out = model(image)
        print(
            f"profile={profile} utility_shape={tuple(out['utility_logits'].shape)} "
            f"proposal_shape={tuple(out['proposal_boxes'].shape)}"
        )
    print("smoke_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
