#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
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
from mobilecropnet_v4.eval_utils import (
    SCORE_SURFACES,
    infer_image_norm,
    infer_input_size,
    load_mobilecropnet_v4_checkpoint,
    select_score_logits,
    write_jsonl,
)
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
    parser.add_argument("--image_mean", default=None, help="Comma-separated RGB mean. Defaults to checkpoint train_config/backbone pretrained_cfg.")
    parser.add_argument("--image_std", default=None, help="Comma-separated RGB std. Defaults to checkpoint train_config/backbone pretrained_cfg.")
    parser.add_argument(
        "--score_surface",
        choices=SCORE_SURFACES,
        default="deployment",
        help="Score surface for GAIC ranking. deployment matches runtime inference/direct evaluation.",
    )
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
    parser.add_argument("--progress_json", type=Path, default=None)
    parser.add_argument("--progress_every", type=int, default=25)
    return parser


def _write_progress(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
    image_mean: Sequence[float] | None,
    image_std: Sequence[float] | None,
    score_surface: str,
    device: torch.device,
    progress_path: Path | None = None,
    progress_every: int = 25,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    model, ckpt = load_mobilecropnet_v4_checkpoint(checkpoint, device=device)
    size = infer_input_size(ckpt, input_size)
    norm_mean, norm_std = infer_image_norm(ckpt, explicit_mean=image_mean, explicit_std=image_std)
    model.eval()
    scores_by_image: dict[str, list[float]] = {}
    skipped: list[dict[str, str]] = []
    started_at = time.time()
    total = len(records)
    _write_progress(
        progress_path,
        {
            "state": "running",
            "phase": "model_scoring",
            "checkpoint": str(checkpoint),
            "processed_images": 0,
            "total_images": total,
            "scored_images": 0,
            "skipped_count": 0,
            "started_at_unix": started_at,
            "updated_at_unix": started_at,
        },
    )
    with torch.no_grad():
        for index, record in enumerate(records, start=1):
            image_path = Path(str(record.get("image_path", "")))
            if not image_path.exists():
                skipped.append({"image_id": str(record.get("image_id", "")), "reason": "missing_image"})
                if index % max(1, progress_every) == 0 or index == total:
                    _write_progress(
                        progress_path,
                        {
                            "state": "running",
                            "phase": "model_scoring",
                            "checkpoint": str(checkpoint),
                            "processed_images": index,
                            "total_images": total,
                            "scored_images": len(scores_by_image),
                            "skipped_count": len(skipped),
                            "elapsed_sec": time.time() - started_at,
                            "updated_at_unix": time.time(),
                        },
                    )
                continue
            candidates = list(record.get("candidates") or [])
            if not candidates:
                skipped.append({"image_id": str(record.get("image_id", "")), "reason": "no_candidates"})
                if index % max(1, progress_every) == 0 or index == total:
                    _write_progress(
                        progress_path,
                        {
                            "state": "running",
                            "phase": "model_scoring",
                            "checkpoint": str(checkpoint),
                            "processed_images": index,
                            "total_images": total,
                            "scored_images": len(scores_by_image),
                            "skipped_count": len(skipped),
                            "elapsed_sec": time.time() - started_at,
                            "updated_at_unix": time.time(),
                        },
                    )
                continue
            image, transform = load_image_tensor_letterbox(image_path, size, image_mean=norm_mean, image_std=norm_std)
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
            score_logits = select_score_logits(outputs, score_surface=score_surface)
            scores = torch.sigmoid(score_logits).squeeze(0).detach().cpu().tolist()
            scores_by_image[str(record["image_id"])] = [float(v) for v in scores]
            if index % max(1, progress_every) == 0 or index == total:
                _write_progress(
                    progress_path,
                    {
                        "state": "running",
                        "phase": "model_scoring",
                        "checkpoint": str(checkpoint),
                        "processed_images": index,
                        "total_images": total,
                        "scored_images": len(scores_by_image),
                        "skipped_count": len(skipped),
                        "elapsed_sec": time.time() - started_at,
                        "updated_at_unix": time.time(),
                    },
                )
    metadata = {
        "checkpoint": str(checkpoint),
        "input_size": int(size),
        "target_ar": str(target_ar),
        "image_mean": norm_mean,
        "image_std": norm_std,
        "score_surface": str(score_surface),
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
        "# MobileCropNet v4 GAIC 공식 벤치마크 평가 보고서",
        "",
        f"- annotations_json: `{summary['annotations_json']}`",
        f"- 이미지 수: {summary['image_count']}",
        f"- 후보 수: {summary['candidate_count']}",
        f"- 모델명: `{summary['model_name']}`",
        f"- target AR 조건: `{summary['model_metadata'].get('target_ar')}`",
        f"- score surface: `{summary['model_metadata'].get('score_surface', 'deployment')}`",
        f"- teacher_candidate_eval_jsonl: `{summary.get('teacher_candidate_eval_jsonl') or ''}`",
        f"- teacher protocol: `{summary.get('teacher_protocol') or ''}`",
        "",
        "## 지표 정의",
        "",
        "- `pcc`와 `srcc`: 이미지별 공식 GAIC MOS와 예측 crop score의 상관을 평균한 값이다.",
        "- `accK_of_topN`: GAIC 논문의 return-K-of-top-N 정확도이며, 반환 crop이 MOS top-N 안에 많이 들어갈수록 높다.",
        "- `accwK_of_topN`: rank-weighted return-K-of-top-N 정확도이며, 더 좋은 순위의 top-N crop을 앞에 반환할수록 높다.",
        "- `top1_mos`: 첫 번째 반환 crop의 공식 MOS이며 높을수록 좋다.",
        "- `top1_mos_regret`: 이미지 내 최고 공식 MOS와 top-1 반환 crop MOS의 차이며 낮을수록 좋다.",
        "- `top1_rank_percentile`: top-1 crop의 공식 MOS 순위 백분위이며 1.0이 최상위다.",
        "",
        "## 방법별 비교",
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
            "## 해석 메모",
            "",
            "- `gaic_mos_oracle`은 배포 가능 모델이 아니며, 공식 MOS 자체로 crop을 정렬한 상한이다.",
            "- `teacher_candidate_eval_jsonl`을 쓰면 teacher score는 공식 GAIC crop 후보에서 직접 읽는다. 그 외에는 nearest-IoU matching으로 teacher score를 공식 GAIC annotation crop에 투영한다.",
            "- subset 또는 투영 teacher 지표를 해석할 때는 coverage 열을 함께 봐야 한다.",
            "- 이 보고서는 SSTK teacher label을 ground truth로 사용하지 않는다.",
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
        image_mean=args.image_mean,
        image_std=args.image_std,
        score_surface=str(args.score_surface),
        device=device,
        progress_path=args.progress_json or args.output_dir / "progress.json",
        progress_every=args.progress_every,
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
    _write_progress(
        args.progress_json or args.output_dir / "progress.json",
        {
            "state": "completed",
            "phase": "completed",
            "checkpoint": str(args.checkpoint),
            "processed_images": len(records_with_images),
            "total_images": len(records_with_images),
            "metrics_json": str(args.output_dir / "metrics.json"),
            "updated_at_unix": time.time(),
        },
    )
    print(json.dumps(summary["methods"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
