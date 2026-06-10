#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from routing.routing_v2_v16_modes import IMAGE_ROUTE_NO_PLACEMENT_NAMES, V16_MODE_SPECS  # noqa: E402


TARGET_AR_NAMES = ["FREE", "1:1", "9:16", "16:9", "3:4", "4:3"]
DECISION_NAMES = ["keep_full", "minimal_crop", "crop"]
V16_MODE_NAMES = [spec.name for spec in V16_MODE_SPECS]


@dataclass
class Candidate:
    candidate_id: str
    bbox: list[float]
    score: float
    is_best: bool
    route_name: str
    mode_name: str
    target_ar: str
    why_tags: list[str]
    checklist: dict[str, Any]


@dataclass
class Task:
    task_id: str
    image_id: str
    target_ar: str
    route_name: str
    mode_name: str
    candidates: list[Candidate]
    decision_id: int


class CandidateFirstSelector(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, max(16, hidden_dim // 2)),
            nn.GELU(),
            nn.Linear(max(16, hidden_dim // 2), 1),
        )
        self.decision_head = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(DECISION_NAMES)),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.scorer(features).squeeze(-1)
        pooled = torch.cat([features.mean(dim=0), features[scores.argmax(dim=0)]], dim=0)
        decision_logits = self.decision_head(pooled.unsqueeze(0)).squeeze(0)
        return scores, decision_logits


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-train a candidate-first routing_v2/v16 crop selector adapter.")
    parser.add_argument("--label-json", type=Path, required=True)
    parser.add_argument("--query-status-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=32)
    parser.add_argument("--max-candidates-per-task", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--seed", type=int, default=20260609)
    return parser.parse_args(argv)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _bbox_norm(ann: dict[str, Any], image: dict[str, Any]) -> list[float]:
    x, y, w, h = [float(v) for v in ann.get("bbox", [0, 0, 1, 1])[:4]]
    iw = max(1.0, float(image.get("width") or 1.0))
    ih = max(1.0, float(image.get("height") or 1.0))
    return [
        max(0.0, min(1.0, x / iw)),
        max(0.0, min(1.0, y / ih)),
        max(0.0, min(1.0, (x + w) / iw)),
        max(0.0, min(1.0, (y + h) / ih)),
    ]


def _candidate_id(ann: dict[str, Any]) -> str:
    attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
    return str(attrs.get("candidate_id") or ann.get("candidate_id") or ann.get("id"))


def _target_ar(ann: dict[str, Any]) -> str:
    attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
    return str(attrs.get("target_ar") or ann.get("target_ar") or "FREE")


def _score(ann: dict[str, Any]) -> float:
    value = ann.get("score_mode", ann.get("score", 0.0))
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = 0.0
    return out if math.isfinite(out) else 0.0


def _load_query_meta(path: Path, source_image_ids: set[str]) -> dict[str, dict[str, Any]]:
    meta: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        source_id = str(row.get("source_image_id") or "")
        if source_id in source_image_ids:
            meta[str(row.get("query_id") or "")] = row
    return meta


def _decision_from_candidates(candidates: list[Candidate]) -> int:
    best = max(candidates, key=lambda item: item.score)
    x1, y1, x2, y2 = best.bbox
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if area >= 0.96:
        return 0
    if area >= 0.82:
        return 1
    return 2


def _load_tasks(label_json: Path, query_status_jsonl: Path, max_images: int, max_candidates_per_task: int) -> tuple[list[Task], dict[str, Any]]:
    payload = json.loads(label_json.read_text(encoding="utf-8"))
    images = payload.get("images") if isinstance(payload.get("images"), list) else []
    annotations = payload.get("annotations") if isinstance(payload.get("annotations"), list) else []
    selected_images = images[: int(max_images)] if int(max_images) > 0 else images
    id_to_image = {int(row.get("id")): row for row in selected_images if row.get("id") is not None}
    source_ids = {str(row.get("source_image_id") or Path(str(row.get("file_name") or "")).stem) for row in selected_images}
    query_meta = _load_query_meta(query_status_jsonl, source_ids)

    grouped: dict[tuple[str, str, str], list[Candidate]] = defaultdict(list)
    for ann in annotations:
        image = id_to_image.get(int(ann.get("image_id") or -1))
        if image is None:
            continue
        source_id = str(image.get("source_image_id") or Path(str(image.get("file_name") or "")).stem)
        target_ar = _target_ar(ann)
        query_id = str(ann.get("query_id") or "")
        meta = query_meta.get(query_id, {})
        attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
        route_name = str(meta.get("query_route_name_no_placement") or image.get("image_route_name_no_placement") or "scene")
        mode_name = str(meta.get("v16_mode_name") or ann.get("mode_name") or "scene_center")
        score_components = attrs.get("score_components") if isinstance(attrs.get("score_components"), dict) else {}
        why_tags = [str(x) for x in attrs.get("why_tags", [])] if isinstance(attrs.get("why_tags"), list) else []
        checklist = {
            "score_components_present": bool(score_components),
            "score_component_keys": sorted(str(k) for k in score_components.keys())[:20],
            "is_best": int(ann.get("is_best") or 0),
        }
        candidate = Candidate(
            candidate_id=_candidate_id(ann),
            bbox=_bbox_norm(ann, image),
            score=_score(ann),
            is_best=bool(int(ann.get("is_best") or 0)),
            route_name=route_name if route_name in IMAGE_ROUTE_NO_PLACEMENT_NAMES else "scene",
            mode_name=mode_name if mode_name in V16_MODE_NAMES else "scene_center",
            target_ar=target_ar if target_ar in TARGET_AR_NAMES else "FREE",
            why_tags=why_tags,
            checklist=checklist,
        )
        grouped[(source_id, target_ar, candidate.mode_name)].append(candidate)

    tasks: list[Task] = []
    for (source_id, target_ar, mode_name), candidates in sorted(grouped.items()):
        if len(candidates) < 2:
            continue
        candidates = sorted(candidates, key=lambda item: item.score, reverse=True)[: int(max_candidates_per_task)]
        route_name = candidates[0].route_name
        tasks.append(
            Task(
                task_id=f"{source_id}::{target_ar}::{mode_name}",
                image_id=source_id,
                target_ar=target_ar,
                route_name=route_name,
                mode_name=mode_name,
                candidates=candidates,
                decision_id=_decision_from_candidates(candidates),
            )
        )
    split = max(1, int(round(len(tasks) * 0.8))) if len(tasks) > 1 else len(tasks)
    summary = {
        "source_images_loaded": len(selected_images),
        "annotations_loaded_for_selected_images": sum(len(v) for v in grouped.values()),
        "tasks": len(tasks),
        "train_tasks": split,
        "val_tasks": max(0, len(tasks) - split),
        "route_counts": dict(Counter(task.route_name for task in tasks)),
        "mode_counts": dict(Counter(task.mode_name for task in tasks)),
        "decision_counts": dict(Counter(DECISION_NAMES[task.decision_id] for task in tasks)),
    }
    return tasks, summary


def _one_hot(name: str, names: list[str]) -> list[float]:
    return [1.0 if name == item else 0.0 for item in names]


def _features(task: Task) -> torch.Tensor:
    rows: list[list[float]] = []
    for cand in task.candidates:
        x1, y1, x2, y2 = cand.bbox
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        area = w * h
        cx = x1 + 0.5 * w
        cy = y1 + 0.5 * h
        aspect = math.log(max(1e-4, w) / max(1e-4, h))
        rows.append(
            [
                x1,
                y1,
                x2,
                y2,
                w,
                h,
                area,
                cx,
                cy,
                aspect,
                float(cand.is_best),
            ]
            + _one_hot(cand.target_ar, TARGET_AR_NAMES)
            + _one_hot(cand.route_name, list(IMAGE_ROUTE_NO_PLACEMENT_NAMES))
            + _one_hot(cand.mode_name, V16_MODE_NAMES)
        )
    return torch.tensor(rows, dtype=torch.float32)


def _score_targets(task: Task) -> torch.Tensor:
    raw = torch.tensor([cand.score for cand in task.candidates], dtype=torch.float32)
    if float(raw.max() - raw.min()) > 1e-6:
        raw = (raw - raw.min()) / (raw.max() - raw.min())
    return raw


def _loss_for_task(model: CandidateFirstSelector, task: Task) -> tuple[torch.Tensor, dict[str, float]]:
    feats = _features(task)
    scores, decision_logits = model(feats)
    target_scores = _score_targets(task)
    list_target = torch.softmax(target_scores * 4.0, dim=0)
    listwise_loss = -(list_target * torch.log_softmax(scores, dim=0)).sum()
    top_idx = int(torch.argmax(target_scores).item())
    bottom_idx = int(torch.argmin(target_scores).item())
    pairwise_loss = torch.nn.functional.softplus(-(scores[top_idx] - scores[bottom_idx]))
    decision_target = torch.tensor([task.decision_id], dtype=torch.long)
    decision_loss = torch.nn.functional.cross_entropy(decision_logits.unsqueeze(0), decision_target)
    total = listwise_loss + pairwise_loss + 0.5 * decision_loss
    with torch.no_grad():
        pred_top = int(torch.argmax(scores).item())
        pred_decision = int(torch.argmax(decision_logits).item())
    return total, {
        "listwise_loss": float(listwise_loss.detach()),
        "pairwise_loss": float(pairwise_loss.detach()),
        "decision_loss": float(decision_loss.detach()),
        "top1_correct": float(pred_top == top_idx),
        "decision_correct": float(pred_decision == task.decision_id),
    }


def _eval(model: CandidateFirstSelector, tasks: list[Task]) -> dict[str, float]:
    if not tasks:
        return {"tasks": 0}
    totals: Counter[str] = Counter()
    with torch.no_grad():
        for task in tasks:
            _, metrics = _loss_for_task(model, task)
            for key, value in metrics.items():
                totals[key] += float(value)
    return {key: float(value / len(tasks)) for key, value in totals.items()} | {"tasks": len(tasks)}


def _preview(model: CandidateFirstSelector, tasks: list[Task], limit: int = 8) -> list[dict[str, Any]]:
    rows = []
    with torch.no_grad():
        for task in tasks[:limit]:
            scores, decision_logits = model(_features(task))
            idx = int(torch.argmax(scores).item())
            cand = task.candidates[idx]
            rows.append(
                {
                    "task_id": task.task_id,
                    "image_id": task.image_id,
                    "target_ar": task.target_ar,
                    "bbox_norm_xyxy": cand.bbox,
                    "utility": float(scores[idx].item()),
                    "decision": DECISION_NAMES[int(torch.argmax(decision_logits).item())],
                    "route": task.route_name,
                    "mode": task.mode_name,
                    "risk": float(max(0.0, 1.0 - cand.score)),
                    "checklist": cand.checklist,
                    "why_tags": cand.why_tags,
                    "candidate_id": cand.candidate_id,
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch.manual_seed(int(args.seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "status.json"
    _write_json(status_path, {"state": "running", "phase": "load_data", "started_at_unix": time.time()})
    tasks, data_summary = _load_tasks(args.label_json, args.query_status_jsonl, int(args.max_images), int(args.max_candidates_per_task))
    if len(tasks) < 2:
        raise RuntimeError(f"not enough candidate tasks for smoke training: tasks={len(tasks)}")
    split = int(data_summary["train_tasks"])
    train_tasks = tasks[:split]
    val_tasks = tasks[split:] or tasks[-1:]
    feature_dim = int(_features(train_tasks[0]).shape[1])
    model = CandidateFirstSelector(feature_dim=feature_dim, hidden_dim=int(args.hidden_dim))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1.0e-4)
    train_log = args.output_dir / "train.log"
    history = []
    best_score = -1.0
    best_epoch = 0
    with train_log.open("a", encoding="utf-8") as log:
        for epoch in range(1, int(args.epochs) + 1):
            model.train()
            totals: Counter[str] = Counter()
            for task in train_tasks:
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = _loss_for_task(model, task)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                totals["loss"] += float(loss.detach())
                for key, value in metrics.items():
                    totals[key] += float(value)
            train_metrics = {key: float(value / len(train_tasks)) for key, value in totals.items()}
            model.eval()
            val_metrics = _eval(model, val_tasks)
            score = float(val_metrics.get("top1_correct", 0.0))
            row = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "selection_score": score}
            history.append(row)
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
            _write_json(status_path, {"state": "running", "phase": "train", "epoch": epoch, "selection_score": score})
            checkpoint = {"model_state": model.state_dict(), "feature_dim": feature_dim, "data_summary": data_summary, "history": history}
            torch.save(checkpoint, args.output_dir / "last.pt")
            if score >= best_score:
                best_score = score
                best_epoch = epoch
                torch.save(checkpoint, args.output_dir / "best.pt")
    preview = _preview(model, val_tasks, limit=8)
    _write_json(args.output_dir / "inference_preview.json", preview)
    summary = {
        "state": "completed",
        "phase": "completed",
        "label_json": str(args.label_json),
        "query_status_jsonl": str(args.query_status_jsonl),
        "data_summary": data_summary,
        "feature_dim": feature_dim,
        "losses_supported": ["pairwise_ranking", "listwise_ranking", "ordinal_decision_policy"],
        "optional_aux_supported_in_payload": ["score_components", "checklist", "why_tags"],
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "best_checkpoint": str(args.output_dir / "best.pt"),
        "latest_checkpoint": str(args.output_dir / "last.pt"),
        "history": history,
        "inference_preview": str(args.output_dir / "inference_preview.json"),
        "last_update_time_unix": time.time(),
    }
    _write_json(args.output_dir / "summary.json", summary)
    _write_json(status_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
