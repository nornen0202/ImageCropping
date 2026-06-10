#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    return float(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="QA gate for public teacher proposal injection rate."
    )
    p.add_argument("--candidate_overview_json", required=True)
    p.add_argument("--expected_enabled", type=int, default=1)
    p.add_argument("--min_injected_rate", type=float, default=0.95)
    p.add_argument("--strict", type=int, default=1, help="1=fail(exit 1) on gate miss")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    overview_path = Path(args.candidate_overview_json)
    expected_enabled = int(args.expected_enabled) == 1
    min_rate = max(0.0, min(1.0, float(args.min_injected_rate)))
    strict = int(args.strict) == 1

    if not expected_enabled:
        print("[gate] expected_enabled=0 -> skip")
        return

    if not overview_path.exists():
        msg = f"[gate] candidate overview not found: {overview_path}"
        if strict:
            raise FileNotFoundError(msg)
        print(f"[warn] {msg}")
        return

    with overview_path.open("r", encoding="utf-8") as f:
        payload: Dict[str, Any] = json.load(f)

    num_images = int(payload.get("num_images", 0))
    rate = _safe_float(payload.get("proposal_injected_rate", 0.0), 0.0)

    print(
        f"[gate] proposal_injected_rate={rate:.6f} (min={min_rate:.6f}, num_images={num_images})"
    )

    if num_images <= 0:
        msg = "[gate] num_images is 0 in overview; cannot verify injection quality."
        if strict:
            raise RuntimeError(msg)
        print(f"[warn] {msg}")
        return

    if rate + 1e-12 < min_rate:
        msg = (
            "[gate] proposal injection QA FAILED: "
            f"proposal_injected_rate={rate:.6f} < min_injected_rate={min_rate:.6f}"
        )
        if strict:
            raise RuntimeError(msg)
        print(f"[warn] {msg}")
        return

    print("[gate] PASS")


if __name__ == "__main__":
    main()

