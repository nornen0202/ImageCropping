#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mobilecropnet_v4.data import (
    DECISION_VOCAB,
    SUBJECT_MODE_VOCAB,
    TARGET_AR_VOCAB,
    letterbox_content_box,
    letterbox_to_original_box,
    load_image_tensor_letterbox,
    original_to_letterbox_box,
    target_ar_id,
)
from mobilecropnet_v4.eval_utils import (
    CHECKLIST_LABELS,
    RUNTIME_BASELINE_DECISION_CONFIDENCE_THRESHOLD,
    RUNTIME_BASELINE_LABELS,
    RUNTIME_BASELINE_SCORE_MARGIN,
    SUBJECT_VALID_POLICIES,
    box_target_ar_stats,
    build_runtime_baselines,
    class_label,
    detailed_explanation_from_logits,
    execute_runtime_decision,
    quality_label,
)
from mobilecropnet_v4.eval_utils import infer_image_norm, load_mobilecropnet_v4_checkpoint
from mobilecropnet_v4.eval_utils import infer_subject_valid_threshold
from mobilecropnet_v4.eval_utils import select_generated_proposal_index
from mobilecropnet_v4.eval_utils import subject_valid_from_policy
from mobilecropnet_v4.route_expert import (
    load_embedded_route_experts,
    load_route_expert_checkpoint,
    override_outputs_route_logits,
    route_expert_ensemble_logits_from_paths,
    route_expert_logits_from_paths,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MobileCropNet v4 inference on one image.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--target_ar", default="FREE", choices=TARGET_AR_VOCAB)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--output_png", type=Path, default=None)
    parser.add_argument("--candidate_boxes_json", type=Path, default=None, help="Optional JSON list of normalized xyxy boxes.")
    parser.add_argument("--input_size", type=int, default=None)
    parser.add_argument("--image_mean", default=None, help="Comma-separated RGB mean. Defaults to checkpoint train_config/backbone pretrained_cfg.")
    parser.add_argument("--image_std", default=None, help="Comma-separated RGB std. Defaults to checkpoint train_config/backbone pretrained_cfg.")
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--selection_policy", choices=["utility_top1", "proposal_top1", "proposal_topk_rerank"], default="proposal_topk_rerank")
    parser.add_argument("--proposal_top_m", type=int, default=8)
    parser.add_argument("--exact_target_ar_postprocess", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--use_decision_source_action_gate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use calibrated decision_source_logit to choose baseline-vs-crop final runtime action. Defaults to checkpoint train_config.",
    )
    parser.add_argument("--enforce_baseline_decision_gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--baseline_decision_confidence_threshold", type=float, default=RUNTIME_BASELINE_DECISION_CONFIDENCE_THRESHOLD)
    parser.add_argument("--baseline_score_margin", type=float, default=RUNTIME_BASELINE_SCORE_MARGIN)
    parser.add_argument("--subject_valid_threshold", type=float, default=None)
    parser.add_argument("--subject_valid_policy", choices=SUBJECT_VALID_POLICIES, default="confidence")
    parser.add_argument("--route_expert_checkpoint", type=Path, default=None)
    parser.add_argument("--route_expert_checkpoints", type=Path, nargs="*", default=None)
    parser.add_argument("--route_expert_weights", default=None, help="Comma-separated weights for route expert ensemble.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def _load_font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _draw_png(image_path: Path, rows: list[dict[str, Any]], output_path: Path) -> None:
    with Image.open(image_path) as img:
        canvas = img.convert("RGB")
    scale = min(1.0, 1400.0 / float(max(canvas.size)))
    if scale < 1.0:
        canvas = canvas.resize((int(canvas.width * scale), int(canvas.height * scale)), Image.BILINEAR)
    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(18, canvas.width // 55))
    width = max(3, canvas.width // 350)
    colors = [(196, 45, 45), (224, 138, 34), (45, 102, 210), (36, 150, 80)]
    for idx, row in enumerate(rows[:4]):
        box = row.get("bbox_norm_xyxy", [0.0, 0.0, 1.0, 1.0])
        x1, y1, x2, y2 = [float(v) for v in box]
        xy = [int(x1 * canvas.width), int(y1 * canvas.height), int(x2 * canvas.width), int(y2 * canvas.height)]
        color = colors[idx % len(colors)]
        draw.rectangle(xy, outline=color, width=width)
        label = f"#{idx + 1} {float(row.get('score', 0.0)):.2f}"
        tw = int(draw.textlength(label, font=font))
        th = int(getattr(font, "size", 18) * 1.35)
        draw.rectangle([xy[0], max(0, xy[1] - th - 4), xy[0] + tw + 10, xy[1]], fill=color)
        draw.text((xy[0] + 5, max(0, xy[1] - th - 2)), label, fill=(255, 255, 255), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", compress_level=3)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    model, ckpt = load_mobilecropnet_v4_checkpoint(args.checkpoint, device=device)
    route_expert_paths: list[Path] = []
    if args.route_expert_checkpoint is not None:
        route_expert_paths.append(args.route_expert_checkpoint)
    route_expert_paths.extend(args.route_expert_checkpoints or [])
    route_experts: list[tuple[torch.nn.Module, dict[str, Any]]] = []
    route_expert_source = "none"
    embedded_route_expert_count = 0
    for checkpoint in route_expert_paths:
        expert_model, expert_config, _ = load_route_expert_checkpoint(checkpoint, device=device)
        route_experts.append((expert_model, expert_config))
    if route_experts:
        route_expert_source = "external_checkpoint"
    route_expert_weights = None
    if args.route_expert_weights:
        route_expert_weights = [float(item) for item in str(args.route_expert_weights).split(",") if item.strip()]
    if not route_experts:
        route_experts, route_expert_weights = load_embedded_route_experts(ckpt, device=device)
        embedded_route_expert_count = len(route_experts)
        if route_experts:
            route_expert_source = "embedded_checkpoint"
    input_size = int(args.input_size or ckpt.get("train_config", {}).get("input_size", 256))
    image_mean, image_std = infer_image_norm(ckpt, explicit_mean=args.image_mean, explicit_std=args.image_std)
    subject_valid_threshold = infer_subject_valid_threshold(ckpt, args.subject_valid_threshold)
    use_decision_source_action_gate = (
        bool(args.use_decision_source_action_gate)
        if args.use_decision_source_action_gate is not None
        else bool(ckpt.get("train_config", {}).get("runtime_use_decision_source_action_gate", False))
    )
    image_tensor, transform = load_image_tensor_letterbox(args.image, input_size, image_mean=image_mean, image_std=image_std)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    ar = torch.tensor([target_ar_id(args.target_ar)], dtype=torch.long, device=device)
    image_ar = torch.tensor([transform.image_ar_log], dtype=torch.float32, device=device)
    content_box = torch.tensor([letterbox_content_box(transform)], dtype=torch.float32, device=device)

    boxes = None
    valid = None
    is_base = None
    if args.candidate_boxes_json is not None:
        raw_boxes = json.loads(args.candidate_boxes_json.read_text(encoding="utf-8"))
        padded = [original_to_letterbox_box(box, transform) for box in raw_boxes]
        boxes = torch.tensor(padded, dtype=torch.float32, device=device).unsqueeze(0)
        valid = torch.ones((1, len(padded)), dtype=torch.float32, device=device)
        is_base = torch.zeros_like(valid)

    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor, boxes, valid, ar, image_ar, is_base, None, content_box)
        if route_experts:
            if len(route_experts) == 1:
                expert_model, expert_config = route_experts[0]
                expert_logits = route_expert_logits_from_paths(expert_model, expert_config, [args.image], device=device)
            else:
                expert_logits = route_expert_ensemble_logits_from_paths(
                    route_experts,
                    [args.image],
                    device=device,
                    weights=route_expert_weights,
                )
            outputs = override_outputs_route_logits(outputs, expert_logits)
    scores = torch.sigmoid(outputs["utility_logits"][0]).detach().cpu().tolist()
    selection_logits = outputs.get(
        "source_gate_return_logits",
        outputs.get(
            "source_mixture_return_logits",
            outputs.get("decision_conditioned_return_logits", outputs.get("return_logits", outputs["utility_logits"])),
        ),
    )
    selection_scores = torch.sigmoid(selection_logits[0]).detach().cpu().tolist()
    positives = torch.sigmoid(outputs["positive_logits"][0]).detach().cpu().tolist()
    risks = torch.sigmoid(outputs["risk_logits"][0]).detach().cpu().tolist()
    macro = torch.sigmoid(outputs["macro_logits"][0]).detach().cpu().tolist()
    checklist_logits = outputs.get("checklist_class_logits")
    checklist_applicability_logits = outputs.get("checklist_applicability_logits")
    detail_score_logits = outputs.get("detail_score_logits")
    why_tag_logits = outputs.get("why_tag_logits")
    checklist_cpu = checklist_logits.detach().cpu()[0] if torch.is_tensor(checklist_logits) else None
    checklist_app_cpu = checklist_applicability_logits.detach().cpu()[0] if torch.is_tensor(checklist_applicability_logits) else None
    detail_cpu = detail_score_logits.detach().cpu()[0] if torch.is_tensor(detail_score_logits) else None
    why_cpu = why_tag_logits.detach().cpu()[0] if torch.is_tensor(why_tag_logits) else None
    decision_id = int(outputs["decision_logits"].argmax(dim=1).detach().cpu()[0].item())
    decision_logits = outputs["decision_logits"][0].detach().cpu().tolist()
    decision_probs = torch.softmax(outputs["decision_logits"][0], dim=0).detach().cpu().tolist()
    decision_source_logit = outputs.get("decision_source_logit")
    decision_source_raw_logit = outputs.get("decision_source_raw_logit", decision_source_logit)
    decision_source_threshold = outputs.get("decision_source_logit_threshold")
    decision_source_value = (
        float(decision_source_logit.detach().cpu().view(-1)[0].item())
        if torch.is_tensor(decision_source_logit)
        else None
    )
    decision_source_raw_value = (
        float(decision_source_raw_logit.detach().cpu().view(-1)[0].item())
        if torch.is_tensor(decision_source_raw_logit)
        else None
    )
    decision_source_threshold_value = (
        float(decision_source_threshold.detach().cpu().view(-1)[0].item())
        if torch.is_tensor(decision_source_threshold) and decision_source_threshold.numel() >= 1
        else None
    )
    route_id = int(outputs["route_logits"].argmax(dim=1).detach().cpu()[0].item())
    route_probs = torch.softmax(outputs["route_logits"][0], dim=0).detach().cpu().tolist()
    route_label = class_label(SUBJECT_MODE_VOCAB, route_id)
    out_boxes = boxes.detach().cpu()[0].tolist() if boxes is not None else outputs["proposal_boxes"].detach().cpu()[0].tolist()
    ranked = sorted(range(len(selection_scores)), key=lambda i: float(selection_scores[i]), reverse=True)
    selected_idx = 0
    proposal_scores = [float(v) for v in torch.sigmoid(outputs["proposal_logits"][0]).detach().cpu().tolist()] if "proposal_logits" in outputs else []
    width = int(transform.original_width)
    height = int(transform.original_height)
    baseline_rows = build_runtime_baselines(width=width, height=height, target_ar=args.target_ar)
    runtime_output: dict[str, Any] | None = None
    generated_proposals: list[dict[str, Any]] | None = None
    subject_box_payload: dict[str, Any] = {"available": False}
    pred_subject_box = outputs.get("pred_subject_box")
    pred_subject_valid = outputs.get("pred_subject_valid")
    if torch.is_tensor(pred_subject_box) and torch.is_tensor(pred_subject_valid):
        pred_lb = [float(v) for v in pred_subject_box.detach().cpu()[0].tolist()]
        pred_conf = float(pred_subject_valid.detach().cpu()[0].item())
        subject_box_payload = {
            "available": True,
            "bbox_letterbox_xyxy": pred_lb,
            "bbox_norm_xyxy": letterbox_to_original_box(pred_lb, transform),
            "confidence": pred_conf,
            "valid_threshold": float(subject_valid_threshold),
            "valid_policy": str(args.subject_valid_policy),
            "route_label": route_label,
            "confidence_valid": bool(pred_conf >= float(subject_valid_threshold)),
            "valid": subject_valid_from_policy(
                confidence=pred_conf,
                threshold=float(subject_valid_threshold),
                route_label=route_label,
                policy=str(args.subject_valid_policy),
            ),
        }
        coarse_box = outputs.get("pred_subject_box_coarse")
        coarse_valid = outputs.get("pred_subject_valid_coarse")
        if torch.is_tensor(coarse_box) and torch.is_tensor(coarse_valid):
            coarse_lb = [float(v) for v in coarse_box.detach().cpu()[0].tolist()]
            subject_box_payload["coarse"] = {
                "bbox_letterbox_xyxy": coarse_lb,
                "bbox_norm_xyxy": letterbox_to_original_box(coarse_lb, transform),
                "confidence": float(coarse_valid.detach().cpu()[0].item()),
            }
    if boxes is None:
        proposal_rows = []
        for idx, box_lb in enumerate(out_boxes):
            proposal_rows.append(
                {
                    "proposal_id": int(idx),
                    "bbox_letterbox_xyxy": [float(v) for v in box_lb],
                    "bbox_norm_xyxy": letterbox_to_original_box(box_lb, transform),
                    "score": float(proposal_scores[idx]) if idx < len(proposal_scores) else 0.0,
                    "utility_score": float(scores[idx]),
                    "return_score": float(selection_scores[idx]) if idx < len(selection_scores) else float(scores[idx]),
                    "proposal_score": float(proposal_scores[idx]) if idx < len(proposal_scores) else 0.0,
                    **box_target_ar_stats(letterbox_to_original_box(box_lb, transform), width=width, height=height, target_ar=args.target_ar),
                }
            )
        generated_proposals = proposal_rows
        selected_idx = select_generated_proposal_index(
            utility_scores=selection_scores,
            proposal_scores=proposal_scores,
            proposals=proposal_rows,
            selection_policy=args.selection_policy,
            proposal_top_m=args.proposal_top_m,
        )
        ranked = [selected_idx] + [idx for idx in ranked if idx != selected_idx]
        baseline_utility = outputs.get(
            "runtime_baseline_source_gate_return_logits",
            outputs.get(
                "runtime_baseline_source_mixture_return_logits",
                outputs.get(
                    "runtime_baseline_decision_conditioned_return_logits",
                    outputs.get("runtime_baseline_return_logits", outputs.get("runtime_baseline_utility_logits")),
                ),
            ),
        )
        baseline_positive = outputs.get("runtime_baseline_positive_logits")
        baseline_risk = outputs.get("runtime_baseline_risk_logits")
        baseline_macro = outputs.get("runtime_baseline_macro_logits")
        baseline_checklist = outputs.get("runtime_baseline_checklist_class_logits")
        baseline_checklist_app = outputs.get("runtime_baseline_checklist_applicability_logits")
        baseline_detail = outputs.get("runtime_baseline_detail_score_logits")
        baseline_why = outputs.get("runtime_baseline_why_tag_logits")
        for base_idx, label in enumerate(RUNTIME_BASELINE_LABELS):
            row = baseline_rows[label]
            if torch.is_tensor(baseline_utility):
                row["score"] = float(torch.sigmoid(baseline_utility[0, base_idx]).detach().cpu().item())
                row["utility_score"] = row["score"]
            if torch.is_tensor(baseline_positive):
                row["positive_score"] = float(torch.sigmoid(baseline_positive[0, base_idx]).detach().cpu().item())
            if torch.is_tensor(baseline_risk):
                row["risk_score"] = float(torch.sigmoid(baseline_risk[0, base_idx]).detach().cpu().item())
            if torch.is_tensor(baseline_macro):
                macro_row = torch.sigmoid(baseline_macro[0, base_idx]).detach().cpu().tolist()
                row["checklist"] = {
                    name: {"score": float(macro_row[cidx]), "label": quality_label(float(macro_row[cidx]))}
                    for cidx, name in enumerate(CHECKLIST_LABELS)
                }
            detail = detailed_explanation_from_logits(
                checklist_logits=baseline_checklist.detach().cpu()[0, base_idx] if torch.is_tensor(baseline_checklist) else None,
                checklist_applicability_logits=baseline_checklist_app.detach().cpu()[0, base_idx] if torch.is_tensor(baseline_checklist_app) else None,
                detail_score_logits=baseline_detail.detach().cpu()[0, base_idx] if torch.is_tensor(baseline_detail) else None,
                why_tag_logits=baseline_why.detach().cpu()[0, base_idx] if torch.is_tensor(baseline_why) else None,
                subject_mode_label=route_label,
            )
            row["detailed_checklist"] = detail["detailed_checklist"]
            row["detail_scores"] = detail["detail_scores"]
            row["why_tags"] = detail["why_tags"]
        runtime_output = execute_runtime_decision(
            decision_id=decision_id,
            target_ar=args.target_ar,
            width=width,
            height=height,
            proposals=proposal_rows,
            selected_proposal_index=selected_idx,
            exact_target_ar_postprocess=args.exact_target_ar_postprocess,
            baseline_rows=baseline_rows,
            decision_logits=decision_logits,
            decision_source_logit=decision_source_value,
            decision_source_raw_logit=decision_source_raw_value,
            decision_source_logit_threshold=decision_source_threshold_value,
            use_decision_source_action_gate=bool(use_decision_source_action_gate),
            enforce_baseline_decision_gate=args.enforce_baseline_decision_gate,
            baseline_decision_confidence_threshold=args.baseline_decision_confidence_threshold,
            baseline_score_margin=args.baseline_score_margin,
        )
    predictions = []
    for rank, idx in enumerate(ranked[: args.topk]):
        detail = detailed_explanation_from_logits(
            checklist_logits=checklist_cpu[idx] if checklist_cpu is not None else None,
            checklist_applicability_logits=checklist_app_cpu[idx] if checklist_app_cpu is not None else None,
            detail_score_logits=detail_cpu[idx] if detail_cpu is not None else None,
            why_tag_logits=why_cpu[idx] if why_cpu is not None else None,
            subject_mode_label=route_label,
        )
        explanation = {
            "utility": {"score": float(scores[idx]), "label": quality_label(float(scores[idx]))},
            "positive": {"score": float(positives[idx]), "label": "likely" if float(positives[idx]) >= 0.5 else "unlikely"},
            "risk": {"score": float(risks[idx]), "label": "unsafe" if float(risks[idx]) >= 0.5 else "safe"},
        }
        explanation.update(detail)
        predictions.append(
            {
                "rank": rank + 1,
                "proposal_id": int(idx),
                "score": float(scores[idx]),
                "return_score": float(selection_scores[idx]) if idx < len(selection_scores) else float(scores[idx]),
                "proposal_score": float(proposal_scores[idx]) if idx < len(proposal_scores) else None,
                "positive_score": float(positives[idx]),
                "risk_score": float(risks[idx]),
                "checklist": {
                    name: {"score": float(macro[idx][cidx]), "label": quality_label(float(macro[idx][cidx]))}
                    for cidx, name in enumerate(CHECKLIST_LABELS)
                },
                "detailed_checklist": detail["detailed_checklist"],
                "detail_scores": detail["detail_scores"],
                "why_tags": detail["why_tags"],
                "explanation": explanation,
                "bbox_letterbox_xyxy": [float(v) for v in out_boxes[idx]],
                "bbox_norm_xyxy": letterbox_to_original_box(out_boxes[idx], transform),
            }
        )
    if runtime_output is not None:
        baseline_payload = None
        if str(runtime_output.get("action_source", "")).startswith("baseline_full"):
            baseline_payload = baseline_rows["full"]
        elif str(runtime_output.get("action_source", "")).startswith("baseline_minimal"):
            baseline_payload = baseline_rows["minimal"]
        if baseline_payload is not None:
            runtime_output["score"] = baseline_payload.get("score")
            runtime_output["positive_score"] = baseline_payload.get("positive_score")
            runtime_output["risk_score"] = baseline_payload.get("risk_score")
            runtime_output["checklist"] = baseline_payload.get("checklist", {})
            runtime_output["detailed_checklist"] = baseline_payload.get("detailed_checklist", {})
            runtime_output["detail_scores"] = baseline_payload.get("detail_scores", {})
            runtime_output["why_tags"] = baseline_payload.get("why_tags", [])
        elif 0 <= selected_idx < len(predictions):
            runtime_output["score"] = predictions[0].get("score")
            runtime_output["positive_score"] = predictions[0].get("positive_score")
            runtime_output["risk_score"] = predictions[0].get("risk_score")
            runtime_output["checklist"] = predictions[0].get("checklist", {})
            runtime_output["detailed_checklist"] = predictions[0].get("detailed_checklist", {})
            runtime_output["detail_scores"] = predictions[0].get("detail_scores", {})
            runtime_output["why_tags"] = predictions[0].get("why_tags", [])
    payload = {
        "checkpoint": str(args.checkpoint),
        "route_expert_checkpoint": str(args.route_expert_checkpoint) if args.route_expert_checkpoint is not None else None,
        "route_expert_checkpoints": [str(path) for path in route_expert_paths],
        "route_expert_weights": route_expert_weights,
        "route_expert_source": route_expert_source,
        "embedded_route_expert_count": embedded_route_expert_count,
        "image": str(args.image),
        "target_ar": args.target_ar,
        "input_size": input_size,
        "image_mean": image_mean,
        "image_std": image_std,
        "subject_valid_threshold": float(subject_valid_threshold),
        "subject_valid_policy": str(args.subject_valid_policy),
        "subject_box": subject_box_payload,
        "decision": {
            "id": decision_id,
            "label": class_label(DECISION_VOCAB, decision_id),
            "probabilities": {class_label(DECISION_VOCAB, idx): float(prob) for idx, prob in enumerate(decision_probs)},
        },
        "subject_mode": {
            "id": route_id,
            "label": route_label,
            "source": "route_expert" if route_expert_paths else "mobilecropnet",
            "probabilities": {class_label(SUBJECT_MODE_VOCAB, idx): float(prob) for idx, prob in enumerate(route_probs)},
        },
        "selection_policy": args.selection_policy,
        "exact_target_ar_postprocess": bool(args.exact_target_ar_postprocess),
        "use_decision_source_action_gate": bool(use_decision_source_action_gate),
        "enforce_baseline_decision_gate": bool(args.enforce_baseline_decision_gate),
        "baseline_decision_confidence_threshold": float(args.baseline_decision_confidence_threshold),
        "baseline_score_margin": float(args.baseline_score_margin),
        "selected_top1_index": int(selected_idx),
        "policy": {
            "selection_policy": args.selection_policy,
            "proposal_top_m": int(args.proposal_top_m),
            "selected_proposal_index": int(selected_idx),
            "decision": class_label(DECISION_VOCAB, decision_id),
            "route": route_label,
            "action_source": runtime_output.get("action_source") if isinstance(runtime_output, dict) else None,
            "target_ar_repaired": bool((runtime_output or {}).get("target_ar_compatible", False)) if isinstance(runtime_output, dict) else None,
        },
        "generated_proposals": generated_proposals,
        "runtime_baselines": baseline_rows if boxes is None else None,
        "runtime_output": runtime_output,
        "predictions": predictions,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.output_png is not None:
        draw_rows = [runtime_output] + predictions if runtime_output is not None else predictions
        _draw_png(args.image, [row for row in draw_rows if isinstance(row, dict)], args.output_png)
    preview = {"runtime_output": payload.get("runtime_output"), "predictions": payload["predictions"][: min(3, len(predictions))]}
    print(json.dumps(preview, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
