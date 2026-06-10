#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build MobileCropNet training diagnostics from a run directory.")
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        out = float(value)
        return out if math.isfinite(out) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _nested_metric(row: dict[str, Any], split: str, key: str) -> float:
    split_row = row.get(split, {})
    return _safe_float(split_row.get(key), 0.0) if isinstance(split_row, dict) else 0.0


def _best_epoch(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {}
    return max(
        history,
        key=lambda row: _nested_metric(row, "val", "top1_hit") - 0.01 * _nested_metric(row, "val", "loss"),
    )


def _health_summary(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {"status": "failed", "reasons": ["metrics history is empty"]}
    first = history[0]
    last = history[-1]
    best = _best_epoch(history)
    val_loss_delta = _nested_metric(first, "val", "loss") - _nested_metric(last, "val", "loss")
    train_loss_delta = _nested_metric(first, "train", "loss") - _nested_metric(last, "train", "loss")
    val_top1_delta = _nested_metric(last, "val", "top1_hit") - _nested_metric(first, "val", "top1_hit")
    val_score_mae_delta = _nested_metric(first, "val", "score_mae") - _nested_metric(last, "val", "score_mae")
    final_gap = _nested_metric(last, "val", "loss") - _nested_metric(last, "train", "loss")
    finite = all(
        math.isfinite(_nested_metric(row, split, key))
        for row in history
        for split in ("train", "val")
        for key in ("loss", "top1_hit", "score_mae")
    )
    checks = {
        "finite_metrics": bool(finite),
        "train_loss_improved": bool(train_loss_delta > 0.0),
        "val_loss_improved": bool(val_loss_delta > 0.0),
        "val_top1_improved": bool(val_top1_delta > 0.0),
        "val_score_mae_improved": bool(val_score_mae_delta > 0.0),
        "overfit_gap_warning": bool(final_gap > 0.25),
        "short_run_warning": bool(len(history) < 5),
    }
    pass_count = sum(1 for key in ("finite_metrics", "train_loss_improved", "val_loss_improved", "val_top1_improved") if checks[key])
    status = "pass" if pass_count >= 3 and finite else "needs_attention"
    return {
        "status": status,
        "epoch_count": len(history),
        "best_epoch": int(best.get("epoch", 0)),
        "best_val_loss": _nested_metric(best, "val", "loss"),
        "best_val_top1_hit": _nested_metric(best, "val", "top1_hit"),
        "train_loss_delta": train_loss_delta,
        "val_loss_delta": val_loss_delta,
        "val_top1_delta": val_top1_delta,
        "val_score_mae_delta": val_score_mae_delta,
        "final_train_val_loss_gap": final_gap,
        "checks": checks,
    }


def _plot_curves(history: list[dict[str, Any]], output_path: Path) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional local install
        return f"plot_skipped: {type(exc).__name__}: {exc}"

    epochs = [int(row.get("epoch", idx + 1)) for idx, row in enumerate(history)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=140)
    panels = [
        ("loss", "Loss"),
        ("top1_hit", "Top-1 Hit"),
        ("score_mae", "Score MAE"),
        ("anchor_loss", "Anchor Loss"),
    ]
    for ax, (metric, title) in zip(axes.ravel(), panels):
        ax.plot(epochs, [_nested_metric(row, "train", metric) for row in history], marker="o", label="train")
        ax.plot(epochs, [_nested_metric(row, "val", metric) for row in history], marker="o", label="val")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return str(output_path)


def _write_csv(history: list[dict[str, Any]], output_path: Path) -> None:
    keys = ["loss", "top1_hit", "score_mae", "score_loss", "positive_loss", "risk_loss", "decision_loss", "route_loss", "anchor_loss"]
    lines = ["epoch,split," + ",".join(keys)]
    for row in history:
        epoch = int(row.get("epoch", 0))
        for split in ("train", "val"):
            values = [_nested_metric(row, split, key) for key in keys]
            lines.append(f"{epoch},{split}," + ",".join(f"{value:.9f}" for value in values))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(path: Path, *, summary: dict[str, Any], config: dict[str, Any], dataset_summary: dict[str, Any], plot_status: str) -> None:
    checks = summary["checks"]
    lines = [
        "# MobileCropNet Training Report",
        "",
        f"- status: `{summary['status']}`",
        f"- epochs: {summary['epoch_count']}",
        f"- best_epoch: {summary['best_epoch']}",
        f"- best_val_top1_hit: {summary['best_val_top1_hit']:.6f}",
        f"- best_val_loss: {summary['best_val_loss']:.6f}",
        f"- plot: `{plot_status}`",
        "",
        "## Health Checks",
        "",
        "| check | value |",
        "| --- | ---: |",
    ]
    for key, value in sorted(checks.items()):
        lines.append(f"| {key} | {str(bool(value)).lower()} |")
    lines.extend(
        [
            "",
            "## Deltas",
            "",
            "| metric | value |",
            "| --- | ---: |",
            f"| train_loss_delta | {summary['train_loss_delta']:.6f} |",
            f"| val_loss_delta | {summary['val_loss_delta']:.6f} |",
            f"| val_top1_delta | {summary['val_top1_delta']:.6f} |",
            f"| val_score_mae_delta | {summary['val_score_mae_delta']:.6f} |",
            f"| final_train_val_loss_gap | {summary['final_train_val_loss_gap']:.6f} |",
            "",
            "## Run Config",
            "",
            f"- input_size: `{config.get('input_size', '')}`",
            f"- k: `{config.get('k', '')}`",
            f"- width_mult: `{config.get('width_mult', '')}`",
            f"- batch_size: `{config.get('batch_size', '')}`",
            f"- lr: `{config.get('lr', '')}`",
            f"- weight_decay: `{config.get('weight_decay', '')}`",
            "",
            "## Dataset",
            "",
            f"- train_images: `{dataset_summary.get('train', {}).get('image_count', '')}`",
            f"- val_images: `{dataset_summary.get('val', {}).get('image_count', '')}`",
            f"- train_positive_annotations: `{dataset_summary.get('train', {}).get('positive_annotation_count', '')}`",
            f"- val_positive_annotations: `{dataset_summary.get('val', {}).get('positive_annotation_count', '')}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or args.run_dir / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    history = _read_json(args.run_dir / "metrics.json", [])
    config = _read_json(args.run_dir / "config.json", {})
    dataset_summary = _read_json(args.run_dir / "dataset_summary.json", {})
    summary = _health_summary(history if isinstance(history, list) else [])
    plot_status = _plot_curves(history if isinstance(history, list) else [], output_dir / "training_curves.png")
    _write_csv(history if isinstance(history, list) else [], output_dir / "training_curves.csv")

    payload = {
        "run_dir": str(args.run_dir),
        "summary": summary,
        "config": config,
        "dataset_summary": dataset_summary,
        "plot_status": plot_status,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_report(
        output_dir / "TRAINING_REPORT.md",
        summary=summary,
        config=config if isinstance(config, dict) else {},
        dataset_summary=dataset_summary if isinstance(dataset_summary, dict) else {},
        plot_status=plot_status,
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

