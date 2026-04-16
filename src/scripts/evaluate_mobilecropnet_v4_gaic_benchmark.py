#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet_v4.data import (
    box_area,
    compute_letterbox_transform,
    load_image_tensor_letterbox,
    original_to_letterbox_box,
    target_ar_id,
    xyxy_to_cxcywh,
)
from mobilecropnet_v4.eval_utils import infer_input_size, load_mobilecropnet_v4_checkpoint, write_jsonl
from mobilecropnet_v4.gaic_benchmark import (
    DEFAULT_RETURN_K,
    DEFAULT_TOP_N,
    evaluate_scored_records,
    load_gaic_annotation_records,
    load_teacher_candidate_eval_scores,
    load_teacher_rows,
    project_teacher_scores_to_annotations,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate MobileCropNet v4 against official GAIC MOS annotations.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--annotations_json", type=Path, default=PROJECT_ROOT / "data/Publics/GAIC/annotations_json/instances_test.json")
    parser.add_argument(
        "--image_roots",
        nargs="*",
        type=Path,
        default=[
            PROJECT_ROOT / "data/Publics/GAIC/images/test",
            PROJECT_ROOT / "data/Publics/GAIC_v2/images/test",
        ],
    )
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--model_name", default="mobilecropnet_v4")
    parser.add_argument("--target_ar", default="FREE", help="Conditioning AR used by the target-aware model for official GAIC scoring.")
    parser.add_argument("--input_size", type=int, default=None)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--save_per_image", action="store_true")
    parser.add_argument("--teacher_jsonl", type=Path, default=None)
    parser.add_argument(
        "--teacher_candidate_eval_jsonl",
        type=Path,
        default=None,
        help="Optional candidate_eval_rows.jsonl from run_gaic_benchmark_eval.py. This is the preferred exact teacher-vs-official-GAIC evaluation input.",
    )
    parser.add_argument("--teacher_name", default="sstk_teacher")
    parser.add_argument("--teacher_protocol", default="Gc", help="Protocol to read from --teacher_candidate_eval_jsonl, usually Gc.")
    parser.add_argument("--teacher_target_ar", default="FREE")
    parser.add_argument("--teacher_score_field", default="crop_utility_raw")
    parser.add_argument("--teacher_match_iou_threshold", type=float, default=0.5)
    parser.add_argument(
        "--add_teacher_overlap_model",
        type=int,
        default=1,
        help="When exact teacher rows cover only a subset, add a model row on the same image subset for fair comparison.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def _box_meta(box: Sequence[float], *, is_base: float = 0.0) -> list[float]:
    cxcywh = xyxy_to_cxcywh(box)
    area = box_area(box)
    aspect = cxcywh[2] / max(1e-6, cxcywh[3])
    return [*box, *cxcywh, area, aspect, float(is_base)]


def _score_model_records(
    *,
    checkpoint: Path,
    records: list[dict[str, Any]],
    target_ar: str,
    input_size: int | None,
    device: torch.device,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    model, ckpt = load_mobilecropnet_v4_checkpoint(checkpoint, device=device)
    size = infer_input_size(ckpt, input_size)
    model.eval()
    scores_by_image: dict[str, list[float]] = {}
    skipped: list[dict[str, str]] = []
    with torch.no_grad():
        for record in records:
            image_path = Path(str(record.get("image_path", "")))
            if not image_path.exists():
                skipped.append({"image_id": str(record.get("image_id", "")), "reason": "missing_image"})
                continue
            candidates = list(record.get("candidates") or [])
            if not candidates:
                skipped.append({"image_id": str(record.get("image_id", "")), "reason": "no_candidates"})
                continue
            image, transform = load_image_tensor_letterbox(image_path, size)
            boxes_orig = [list(c["bbox_norm_xyxy"]) for c in candidates]
            boxes = [original_to_letterbox_box(box, transform) for box in boxes_orig]
            box_meta = [_box_meta(box) for box in boxes]
            valid = [1.0] * len(boxes)
            candidate_is_base = [0.0] * len(boxes)
            outputs = model(
                image.unsqueeze(0).to(device),
                torch.tensor([boxes], dtype=torch.float32, device=device),
                torch.tensor([valid], dtype=torch.float32, device=device),
                torch.tensor([target_ar_id(str(target_ar))], dtype=torch.long, device=device),
                torch.tensor([transform.image_ar_log], dtype=torch.float32, device=device),
                torch.tensor([candidate_is_base], dtype=torch.float32, device=device),
                torch.tensor([box_meta], dtype=torch.float32, device=device),
            )
            scores = torch.sigmoid(outputs["utility_logits"]).squeeze(0).detach().cpu().tolist()
            scores_by_image[str(record["image_id"])] = [float(v) for v in scores]
    metadata = {
        "checkpoint": str(checkpoint),
        "input_size": int(size),
        "target_ar": str(target_ar),
        "device": str(device),
        "skipped_count": len(skipped),
        "skipped": skipped[:50],
    }
    return scores_by_image, metadata


def _oracle_scores(records: list[dict[str, Any]]) -> dict[str, list[float]]:
    return {str(record["image_id"]): [float(c.get("mos", 0.0)) for c in record.get("candidates", [])] for record in records}


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    def _score_coverage(row: dict[str, Any], metrics: dict[str, Any]) -> float:
        if "coverage_annotation_match_rate" in metrics:
            return float(metrics["coverage_annotation_match_rate"])
        if "coverage_annotation_score_rate" in metrics:
            return float(metrics["coverage_annotation_score_rate"])
        if int(row.get("missing_image_count", 0)) == 0 and int(row.get("image_count", 0)) == int(summary["image_count"]):
            return 1.0
        return float(row.get("image_count", 0)) / float(max(1, int(summary["image_count"])))

    lines = [
        "# MobileCropNet v4 Official GAIC Benchmark Evaluation",
        "",
        f"- annotations_json: `{summary['annotations_json']}`",
        f"- image_count: {summary['image_count']}",
        f"- candidate_count: {summary['candidate_count']}",
        f"- model_name: `{summary['model_name']}`",
        f"- target_ar_conditioning: `{summary['model_metadata'].get('target_ar')}`",
        f"- teacher_candidate_eval_jsonl: `{summary.get('teacher_candidate_eval_jsonl') or ''}`",
        f"- teacher_protocol: `{summary.get('teacher_protocol') or ''}`",
        "",
        "## Metric Definitions",
        "",
        "- `pcc` and `srcc`: per-image correlation between official GAIC MOS and predicted crop scores, averaged over images.",
        "- `accK_of_topN`: return-K-of-top-N accuracy from the GAIC paper. Higher means more returned crops fall inside the MOS top-N set.",
        "- `accwK_of_topN`: rank-weighted return-K-of-top-N accuracy. It rewards returning better-ranked top-N crops earlier.",
        "- `top1_mos`: official MOS of the first returned crop. Higher is better.",
        "- `top1_mos_regret`: best official MOS minus returned top-1 MOS. Lower is better.",
        "- `top1_rank_percentile`: official MOS rank percentile of top-1 crop, where 1.0 is best.",
        "",
        "## Method Comparison",
        "",
        "| method | images | PCC | SRCC | Acc1/5 | Acc1/10 | Acc4/5 | Accw4/5 | top1 MOS | top1 rank pct | MOS regret | eval/score coverage |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in summary["method_order"]:
        row = summary["methods"][method]
        metrics = row.get("metrics", {})
        lines.append(
            "| {method} | {images} | {pcc:.6f} | {srcc:.6f} | {acc15:.6f} | {acc110:.6f} | {acc45:.6f} | {accw45:.6f} | {mos:.6f} | {pct:.6f} | {regret:.6f} | {coverage:.6f} |".format(
                method=method,
                images=int(row.get("image_count", 0)),
                pcc=float(metrics.get("pcc", 0.0)),
                srcc=float(metrics.get("srcc", 0.0)),
                acc15=float(metrics.get("acc1_of_top5", 0.0)),
                acc110=float(metrics.get("acc1_of_top10", 0.0)),
                acc45=float(metrics.get("acc4_of_top5", 0.0)),
                accw45=float(metrics.get("accw4_of_top5", 0.0)),
                mos=float(metrics.get("top1_mos", 0.0)),
                pct=float(metrics.get("top1_rank_percentile", 0.0)),
                regret=float(metrics.get("top1_mos_regret", 0.0)),
                coverage=_score_coverage(row, metrics),
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `gaic_mos_oracle` is not deployable; it is the upper bound that ranks crops by the official MOS itself.",
            "- When `teacher_candidate_eval_jsonl` is used, teacher scores are read from exact official GAIC crop candidates. Otherwise teacher scores are projected to official GAIC annotation crops by nearest-IoU matching.",
            "- The coverage column should be considered when interpreting subset or projected teacher metrics.",
            "- This report intentionally does not use SSTK teacher labels as ground truth.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_gaic_annotation_records(args.annotations_json, image_roots=args.image_roots, max_images=args.max_images)
    records_with_images = [record for record in records if str(record.get("image_path", "")).strip()]
    candidate_count = sum(len(record.get("candidates") or []) for record in records_with_images)

    device = torch.device(args.device)
    model_scores, model_metadata = _score_model_records(
        checkpoint=args.checkpoint,
        records=records_with_images,
        target_ar=args.target_ar,
        input_size=args.input_size,
        device=device,
    )
    methods: dict[str, Any] = {}
    method_order = [args.model_name]
    methods[args.model_name] = evaluate_scored_records(
        records_with_images,
        model_scores,
        method_name=args.model_name,
        return_k_values=DEFAULT_RETURN_K,
        top_n_values=DEFAULT_TOP_N,
    )

    teacher_overlap_records: list[dict[str, Any]] = []
    if args.teacher_candidate_eval_jsonl is not None:
        teacher_scores, teacher_coverage = load_teacher_candidate_eval_scores(
            args.teacher_candidate_eval_jsonl,
            records_with_images,
            protocol=args.teacher_protocol,
            score_field=args.teacher_score_field,
        )
        teacher_overlap_ids = set(teacher_scores.keys())
        teacher_overlap_records = [record for record in records_with_images if str(record["image_id"]) in teacher_overlap_ids]
        if teacher_overlap_records and int(args.add_teacher_overlap_model) > 0:
            overlap_name = f"{args.model_name}__teacher_overlap"
            method_order.append(overlap_name)
            methods[overlap_name] = evaluate_scored_records(
                teacher_overlap_records,
                model_scores,
                method_name=overlap_name,
                return_k_values=DEFAULT_RETURN_K,
                top_n_values=DEFAULT_TOP_N,
            )
        method_order.append(args.teacher_name)
        methods[args.teacher_name] = evaluate_scored_records(
            records_with_images,
            teacher_scores,
            method_name=args.teacher_name,
            coverage_by_image_id=teacher_coverage,
            return_k_values=DEFAULT_RETURN_K,
            top_n_values=DEFAULT_TOP_N,
        )
    elif args.teacher_jsonl is not None:
        teacher_candidates = load_teacher_rows(
            args.teacher_jsonl,
            image_ids=[str(record["image_id"]) for record in records_with_images],
            target_ar=args.teacher_target_ar,
            score_field=args.teacher_score_field,
        )
        teacher_scores, teacher_coverage = project_teacher_scores_to_annotations(
            records_with_images,
            teacher_candidates,
            match_iou_threshold=args.teacher_match_iou_threshold,
        )
        method_order.append(args.teacher_name)
        methods[args.teacher_name] = evaluate_scored_records(
            records_with_images,
            teacher_scores,
            method_name=args.teacher_name,
            coverage_by_image_id=teacher_coverage,
            return_k_values=DEFAULT_RETURN_K,
            top_n_values=DEFAULT_TOP_N,
        )

    method_order.append("gaic_mos_oracle")
    methods["gaic_mos_oracle"] = evaluate_scored_records(
        records_with_images,
        _oracle_scores(records_with_images),
        method_name="gaic_mos_oracle",
        return_k_values=DEFAULT_RETURN_K,
        top_n_values=DEFAULT_TOP_N,
    )

    summary = {
        "annotations_json": str(args.annotations_json),
        "image_roots": [str(p) for p in args.image_roots],
        "image_count": len(records_with_images),
        "annotation_image_count": len(records),
        "missing_image_count": len(records) - len(records_with_images),
        "candidate_count": candidate_count,
        "candidate_count_mean": float(candidate_count / max(1, len(records_with_images))),
        "model_name": args.model_name,
        "model_metadata": model_metadata,
        "teacher_jsonl": str(args.teacher_jsonl) if args.teacher_jsonl is not None else None,
        "teacher_candidate_eval_jsonl": str(args.teacher_candidate_eval_jsonl) if args.teacher_candidate_eval_jsonl is not None else None,
        "teacher_protocol": args.teacher_protocol if args.teacher_candidate_eval_jsonl is not None else None,
        "teacher_target_ar": args.teacher_target_ar if args.teacher_jsonl is not None else None,
        "teacher_score_field": args.teacher_score_field if args.teacher_jsonl is not None or args.teacher_candidate_eval_jsonl is not None else None,
        "teacher_match_iou_threshold": args.teacher_match_iou_threshold if args.teacher_jsonl is not None else None,
        "method_order": method_order,
        "methods": {name: {k: v for k, v in row.items() if k != "per_image"} for name, row in methods.items()},
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.save_per_image:
        for name, row in methods.items():
            safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
            write_jsonl(args.output_dir / f"{safe_name}_per_image.jsonl", row.get("per_image", []))
    _write_report(args.output_dir / "GAIC_BENCHMARK_EVAL_REPORT.md", summary)
    print(json.dumps(summary["methods"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
