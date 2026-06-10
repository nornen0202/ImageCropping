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

from public_benchmark.regression import build_regression_report, compare_summaries, load_gate_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare a public benchmark summary against a locked baseline.")
    parser.add_argument("--current_summary_json", required=True)
    parser.add_argument("--baseline_summary_json", required=True)
    parser.add_argument("--gate_config_json", default="")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_report_md", default="")
    parser.add_argument("--fail_on_regression", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    current = json.loads(Path(args.current_summary_json).read_text(encoding="utf-8"))
    baseline = json.loads(Path(args.baseline_summary_json).read_text(encoding="utf-8"))
    gate_config = load_gate_config(Path(args.gate_config_json).resolve() if str(args.gate_config_json).strip() else None)
    result = compare_summaries(current=current, baseline=baseline, gate_config=gate_config)
    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    if str(args.output_report_md).strip():
        report_path = Path(args.output_report_md).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(build_regression_report(result), encoding="utf-8")
    print(json.dumps({"status": result["status"], "failure_count": result["failure_count"], "output_json": str(output_json)}, ensure_ascii=False, indent=2))
    if result["status"] == "fail" and int(args.fail_on_regression) > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
