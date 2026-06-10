#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet_v4.data import target_ar_id
from mobilecropnet_v4.eval_utils import infer_input_size, load_mobilecropnet_v4_checkpoint
from mobilecropnet_v4.route_expert import load_route_expert_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark MobileCropNet v4 inference latency.")
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Repeated NAME=PATH entry. PATH is usually a best.pt checkpoint.",
    )
    parser.add_argument("--output_json", type=Path, required=True)
    parser.add_argument("--output_md", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target_ar", default="FREE")
    parser.add_argument(
        "--candidate_counts",
        default="auto,86",
        help="Comma-separated candidate counts. `auto` uses checkpoint train_config/model_config candidate_k.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--amp", type=int, default=1, help="Use CUDA fp16 autocast when device is cuda.")
    parser.add_argument("--seed", type=int, default=260420)
    parser.add_argument("--route_expert_checkpoint", action="append", default=None, help="Optional route expert checkpoint included in composite latency.")
    parser.add_argument("--runtime_no_prior", action="store_true", help="Benchmark image+target_ar runtime path without external candidate boxes.")
    return parser


def parse_checkpoint_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.parent.name or path.stem, path
    name, raw_path = value.split("=", 1)
    return name.strip(), Path(raw_path.strip())


def box_meta_from_boxes(boxes: torch.Tensor, candidate_is_base: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    w = (x2 - x1).clamp_min(1e-6)
    h = (y2 - y1).clamp_min(1e-6)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    area = (w * h).clamp(0.0, 1.0)
    aspect = (w / h).clamp(0.0, 100.0)
    return torch.stack([x1, y1, x2, y2, cx, cy, w, h, area, aspect, candidate_is_base], dim=-1)


def make_inputs(
    *,
    batch_size: int,
    input_size: int,
    candidate_count: int,
    target_ar: str,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    image = torch.rand((batch_size, 3, input_size, input_size), dtype=dtype, device=device)
    xy1 = torch.rand((batch_size, candidate_count, 2), dtype=dtype, device=device) * 0.82
    max_wh = (1.0 - xy1).clamp_min(0.02)
    wh = (0.08 + 0.55 * torch.rand((batch_size, candidate_count, 2), dtype=dtype, device=device)) * max_wh
    xy2 = (xy1 + wh).clamp(0.0, 1.0)
    boxes = torch.cat([xy1, xy2], dim=-1)
    valid = torch.ones((batch_size, candidate_count), dtype=dtype, device=device)
    candidate_is_base = torch.zeros((batch_size, candidate_count), dtype=dtype, device=device)
    candidate_is_base[:, 0] = 1.0
    return {
        "image": image,
        "boxes": boxes,
        "valid": valid,
        "target_ar_id": torch.full((batch_size,), target_ar_id(target_ar), dtype=torch.long, device=device),
        "image_ar_log": torch.zeros((batch_size,), dtype=dtype, device=device),
        "candidate_is_base": candidate_is_base,
        "box_meta": box_meta_from_boxes(boxes, candidate_is_base),
    }


def summarize_ms(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    sorted_values = sorted(float(v) for v in values)

    def pct(q: float) -> float:
        if len(sorted_values) == 1:
            return sorted_values[0]
        idx = (len(sorted_values) - 1) * q
        lo = int(idx)
        hi = min(len(sorted_values) - 1, lo + 1)
        frac = idx - lo
        return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac

    mean_ms = float(statistics.fmean(sorted_values))
    return {
        "mean_ms": mean_ms,
        "median_ms": float(statistics.median(sorted_values)),
        "p90_ms": pct(0.90),
        "p95_ms": pct(0.95),
        "min_ms": sorted_values[0],
        "max_ms": sorted_values[-1],
        "images_per_sec": float(1000.0 / mean_ms) if mean_ms > 0.0 else 0.0,
    }


def benchmark_forward(
    model: torch.nn.Module,
    *,
    route_experts: list[tuple[torch.nn.Module, dict[str, Any]]] | None = None,
    runtime_no_prior: bool = False,
    input_size: int,
    candidate_count: int,
    batch_size: int,
    target_ar: str,
    warmup: int,
    iterations: int,
    amp: bool,
    device: torch.device,
) -> dict[str, Any]:
    inputs = make_inputs(
        batch_size=batch_size,
        input_size=input_size,
        candidate_count=candidate_count,
        target_ar=target_ar,
        device=device,
        dtype=torch.float32,
    )
    expert_inputs = []
    for _, config in route_experts or []:
        expert_input_size = int(config.get("input_size") or 384)
        expert_inputs.append(torch.rand((batch_size, 3, expert_input_size, expert_input_size), dtype=torch.float32, device=device))
    autocast_enabled = bool(amp and device.type == "cuda")

    def run_model_once() -> None:
        if runtime_no_prior:
            _ = model(
                inputs["image"],
                None,
                None,
                inputs["target_ar_id"],
                inputs["image_ar_log"],
                None,
                None,
            )
        else:
            _ = model(**inputs)
        for (route_expert, _), expert_image in zip(route_experts or [], expert_inputs):
            _ = route_expert(expert_image)

    with torch.no_grad():
        for _ in range(max(0, int(warmup))):
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
                run_model_once()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        timings: list[float] = []
        for _ in range(max(1, int(iterations))):
            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
                    run_model_once()
                end_event.record()
                torch.cuda.synchronize(device)
                timings.append(float(start_event.elapsed_time(end_event)))
            else:
                start = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
                    run_model_once()
                timings.append(float((time.perf_counter() - start) * 1000.0))
    peak_mem_mb = 0.0
    if device.type == "cuda":
        peak_mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
    return {
        "candidate_count": int(candidate_count),
        "batch_size": int(batch_size),
        "input_size": int(input_size),
        "amp": bool(autocast_enabled),
        "warmup": int(warmup),
        "iterations": int(iterations),
        "runtime_no_prior": bool(runtime_no_prior),
        "peak_memory_mb": peak_mem_mb,
        **summarize_ms(timings),
    }


def resolve_candidate_counts(raw: str, *, auto_count: int) -> list[int]:
    out: list[int] = []
    for part in str(raw).split(","):
        item = part.strip()
        if not item:
            continue
        count = int(auto_count) if item.lower() == "auto" else int(item)
        if count > 0 and count not in out:
            out.append(count)
    return out or [int(auto_count)]


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# MobileCropNet v4 추론 지연시간 벤치마크",
        "",
        f"- device: `{payload.get('device')}`",
        f"- cuda_device_name: `{payload.get('cuda_device_name')}`",
        f"- torch: `{payload.get('torch_version')}`",
        "- 벤치마크 범위: synthetic model forward only, image decoding/disk I/O 제외",
        "- route expert checkpoint가 지정되면 body forward와 route expert forward를 같은 iteration에 포함",
        "- candidate count `auto`: checkpoint candidate_k; candidate count `86`: GAIC v2 test mean annotation count proxy",
        "",
        "| 모델 | 입력 | 후보 수 | 파라미터 M | AMP | mean ms | p50 ms | p95 ms | img/s | peak mem MB |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload.get("results", []):
        if row.get("status") != "ok":
            lines.append(f"| `{row.get('name')}` | - | - | - | - | - | - | - | - | - |")
            continue
        params_m = float(row.get("parameter_count", 0)) / 1_000_000.0
        for bench in row.get("benchmarks", []):
            lines.append(
                "| {name} | {input_size} | {candidates} | {params_m:.3f} | {amp} | {mean:.3f} | {median:.3f} | {p95:.3f} | {ips:.2f} | {mem:.1f} |".format(
                    name=f"`{row['name']}`",
                    input_size=int(bench.get("input_size", 0)),
                    candidates=int(bench.get("candidate_count", 0)),
                    params_m=params_m,
                    amp="fp16" if bench.get("amp") else "fp32",
                    mean=float(bench.get("mean_ms", 0.0)),
                    median=float(bench.get("median_ms", 0.0)),
                    p95=float(bench.get("p95_ms", 0.0)),
                    ips=float(bench.get("images_per_sec", 0.0)),
                    mem=float(bench.get("peak_memory_mb", 0.0)),
                )
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    output: dict[str, Any] = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "",
        "target_ar": str(args.target_ar),
        "route_expert_checkpoints": [str(item) for item in (args.route_expert_checkpoint or [])],
        "runtime_no_prior": bool(args.runtime_no_prior),
        "results": [],
    }
    route_experts: list[tuple[torch.nn.Module, dict[str, Any]]] = []
    for checkpoint in args.route_expert_checkpoint or []:
        expert_model, expert_config, _ = load_route_expert_checkpoint(Path(checkpoint), device=device)
        route_experts.append((expert_model, expert_config))
    for spec in args.checkpoint:
        name, path = parse_checkpoint_spec(spec)
        row: dict[str, Any] = {"name": name, "checkpoint": str(path)}
        if not path.exists():
            row.update({"status": "missing", "error": "checkpoint file does not exist"})
            output["results"].append(row)
            continue
        try:
            model, ckpt = load_mobilecropnet_v4_checkpoint(path, device=device)
            model.eval()
            train_config = ckpt.get("train_config", {}) if isinstance(ckpt.get("train_config"), dict) else {}
            model_config = ckpt.get("model_config", {}) if isinstance(ckpt.get("model_config"), dict) else {}
            input_size = infer_input_size(ckpt, None)
            auto_candidate_count = int(train_config.get("candidate_k") or model_config.get("candidate_k") or 32)
            candidate_counts = resolve_candidate_counts(args.candidate_counts, auto_count=auto_candidate_count)
            row.update(
                {
                    "status": "ok",
                    "input_size": int(input_size),
                    "candidate_k": int(auto_candidate_count),
                    "proposal_q": int(model_config.get("proposal_q") or train_config.get("proposal_q") or 0),
                    "backbone": str(model_config.get("backbone_name") or train_config.get("backbone_name") or ""),
                    "parameter_count": count_parameters(model) + sum(count_parameters(route_expert) for route_expert, _ in route_experts),
                    "body_parameter_count": count_parameters(model),
                    "route_expert_parameter_count": sum(count_parameters(route_expert) for route_expert, _ in route_experts),
                    "benchmarks": [],
                }
            )
            for candidate_count in candidate_counts:
                row["benchmarks"].append(
                    benchmark_forward(
                        model,
                        route_experts=route_experts,
                        runtime_no_prior=bool(args.runtime_no_prior),
                        input_size=int(input_size),
                        candidate_count=int(candidate_count),
                        batch_size=int(args.batch_size),
                        target_ar=str(args.target_ar),
                        warmup=int(args.warmup),
                        iterations=int(args.iterations),
                        amp=bool(int(args.amp)),
                        device=device,
                    )
                )
        except Exception as exc:
            row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
        output["results"].append(row)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.output_md, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
