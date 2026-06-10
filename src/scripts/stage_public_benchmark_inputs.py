#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from public_benchmark.staging import stage_public_benchmark_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage FCDB/CPC/GNMC benchmark images/tasks for SSTK candidate/teacher runs.")
    parser.add_argument("--data_root", default="data/Publics")
    parser.add_argument("--datasets", nargs="+", default=["fcdb", "cpc", "gnmc"])
    parser.add_argument("--split", default="all")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_tasks_per_dataset", type=int, default=0)
    parser.add_argument("--max_pairwise_per_task", type=int, default=200)
    parser.add_argument("--min_pairwise_gap", type=float, default=0.0)
    parser.add_argument("--link_mode", choices=["symlink", "copy", "hardlink", "none"], default="symlink")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = stage_public_benchmark_inputs(
        data_root=Path(args.data_root).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        datasets=[str(item).lower() for item in args.datasets],
        split=str(args.split),
        max_tasks_per_dataset=int(args.max_tasks_per_dataset),
        max_pairwise_per_task=int(args.max_pairwise_per_task),
        min_pairwise_gap=float(args.min_pairwise_gap),
        link_mode=str(args.link_mode),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
