#!/usr/bin/env python3
"""
Visualization/analytics utility for Stage-10 VLM teacher labels.

Inputs:
  - crop_label_v1*.jsonl (required)
  - teacher_scores*.jsonl (optional; for source/final enrichment)
  - filtered parquet + tar_dir (+ optional image_dir) for image decoding

Outputs:
  - per-task overlays: out_dir/by_ar/<ar>/<image_id>.jpg
  - analytics png/csv/json under out_dir/analytics
  - viz_overview.json
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image


TOPK_COLORS_BGR: List[Tuple[int, int, int]] = [
    (0, 0, 255),
    (0, 200, 0),
    (0, 165, 255),
    (255, 0, 255),
    (255, 255, 0),
]
ALSO_CONSIDERED_COLOR = (170, 170, 170)
BASELINE_COLOR = (255, 90, 20)
IMAGE_EXTS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize Stage-10 VLM teacher labels")
    p.add_argument("--vlm_labels_jsonl", required=True)
    p.add_argument("--teacher_scores_jsonl", default="", help="optional; enrich source/final from teacher candidates")
    p.add_argument("--parquet", required=True, help="filtered parquet with tar_name/bucket")
    p.add_argument("--tar_dir", required=True)
    p.add_argument("--image_dir", default="", help="optional local curated image dir (<image_id>.<ext>)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--target_ar", default="all", help="all or one of FREE,1:1,9:16,16:9,3:4,4:3")
    p.add_argument("--num_samples", type=int, default=120, help="0=all selected tasks")
    p.add_argument(
        "--image_ids",
        nargs="*",
        default=[],
        help="explicit image ids (space/comma mixed). if set, rendering is limited to these ids",
    )
    p.add_argument("--image_ids_file", default="", help="text file with image_id entries (1 per line)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--draw_also_considered", type=int, default=1)
    p.add_argument("--analytics_top_n_tags", type=int, default=20)
    p.add_argument(
        "--analytics_use_full_labels",
        type=int,
        default=1,
        help="1: analytics over all rows filtered only by target_ar, 0: analytics over rendered subset",
    )
    return p.parse_args()


def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not np.isfinite(x):
        return float(default)
    return float(x)


def safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def normalize_box(box: Any) -> Optional[List[float]]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(x) for x in box]
    except Exception:
        return None
    x1 = clamp(x1, 0.0, 1.0)
    y1 = clamp(y1, 0.0, 1.0)
    x2 = clamp(x2, 0.0, 1.0)
    y2 = clamp(y2, 0.0, 1.0)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if (x2 - x1) <= 1e-8 or (y2 - y1) <= 1e-8:
        return None
    return [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)]


def denorm_box(box_norm: Sequence[float], w: int, h: int) -> Tuple[int, int, int, int]:
    x1n, y1n, x2n, y2n = [float(v) for v in box_norm]
    x1 = int(round(clamp(x1n, 0.0, 1.0) * w))
    y1 = int(round(clamp(y1n, 0.0, 1.0) * h))
    x2 = int(round(clamp(x2n, 0.0, 1.0) * w))
    y2 = int(round(clamp(y2n, 0.0, 1.0) * h))
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def parse_image_ids_arg(values: Sequence[str]) -> List[str]:
    out: List[str] = []
    for raw in values:
        toks = [tok.strip() for tok in str(raw).split(",")]
        for tok in toks:
            if tok:
                out.append(tok)
    return out


def parse_image_ids_file(path: str) -> List[str]:
    if not path:
        return []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"image_ids_file not found: {path}")
    out: List[str] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.extend([tok.strip() for tok in s.split(",") if tok.strip()])
    return out


def unique_keep_order(ids: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in ids:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def find_image_in_dir(image_dir: str, image_id: str) -> Optional[Path]:
    if not image_dir:
        return None
    root = Path(image_dir)
    for ext in IMAGE_EXTS:
        p = root / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


def resolve_tar_path(tar_dir: str, bucket: str, tar_name: str) -> Optional[str]:
    if not tar_name:
        return None
    cand = [
        os.path.join(tar_dir, tar_name),
        os.path.join(tar_dir, str(bucket), tar_name) if bucket else None,
    ]
    for p in cand:
        if p and os.path.exists(p):
            return p
    return None


def draw_text_block(img: np.ndarray, lines: Sequence[str], x: int = 8, y: int = 20) -> None:
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    th = 1
    line_h = 17
    max_w = 0
    for s in lines:
        (tw, _), _ = cv2.getTextSize(s, font, scale, th)
        max_w = max(max_w, tw)
    h_block = line_h * len(lines) + 8
    cv2.rectangle(img, (x - 6, y - 15), (x + max_w + 8, y - 15 + h_block), (0, 0, 0), -1)
    for i, s in enumerate(lines):
        cv2.putText(img, s, (x, y + i * line_h), font, scale, (232, 232, 232), th, cv2.LINE_AA)


def candidate_to_compact(c: Dict[str, Any]) -> Dict[str, Any]:
    box = normalize_box(c.get("bbox_norm_xyxy"))
    return {
        "candidate_id": str(c.get("candidate_id", "")),
        "bbox_norm_xyxy": box,
        "source": str(c.get("source", "")),
        "score_final": safe_float(c.get("scores", {}).get("final", np.nan), np.nan),
    }


def load_teacher_candidate_index(path: str) -> Dict[Tuple[str, str], Dict[str, Dict[str, Any]]]:
    if not path or not Path(path).exists():
        return {}
    index: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            image_id = str(rec.get("image_id", ""))
            by_ar = rec.get("teacher_scorer", {}).get("results_by_ar", {})
            if not image_id or not isinstance(by_ar, dict):
                continue
            for ar, ar_res in by_ar.items():
                if not isinstance(ar_res, dict):
                    continue
                pool: List[Dict[str, Any]] = []
                for k in ("cheap_top_m", "selected_topk", "hard_negatives"):
                    v = ar_res.get(k, [])
                    if isinstance(v, list):
                        pool.extend(v)
                for k in ("baseline_candidate", "best_candidate"):
                    v = ar_res.get(k)
                    if isinstance(v, dict):
                        pool.append(v)
                key = (image_id, str(ar))
                for c in pool:
                    if not isinstance(c, dict):
                        continue
                    cid = str(c.get("candidate_id", ""))
                    if not cid:
                        continue
                    if cid in index[key]:
                        continue
                    index[key][cid] = candidate_to_compact(c)
    return index


def load_vlm_tasks(path: Path, target_ar: str) -> List[Dict[str, Any]]:
    tasks: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            image_id = str(rec.get("image_id", ""))
            ar = str(rec.get("target_ar", ""))
            if not image_id or not ar:
                continue
            if target_ar != "all" and ar != target_ar:
                continue
            selected = rec.get("selected_topk", [])
            also = rec.get("also_considered", [])
            if not isinstance(selected, list):
                selected = []
            if not isinstance(also, list):
                also = []
            tasks.append(
                {
                    "image_id": image_id,
                    "target_ar": ar,
                    "decision_type": str(rec.get("decision_type", "")),
                    "delta_improve_vs_baseline": safe_float(rec.get("delta_improve_vs_baseline", 0.0), 0.0),
                    "baseline_used_candidate_id": str(rec.get("baseline_used_candidate_id", "")),
                    "selected_topk": selected,
                    "also_considered": also,
                    "teacher_backend": str(rec.get("teacher", {}).get("backend", "")),
                    "teacher_id": str(rec.get("teacher", {}).get("teacher_id", "")),
                    "teacher_confidence": safe_float(rec.get("teacher", {}).get("teacher_confidence", 0.0), 0.0),
                    "validator_note": str(rec.get("validator", {}).get("notes", "")),
                    "short_expl": str(rec.get("explanations", {}).get("short", "")),
                }
            )
    return tasks


def enrich_tasks_with_teacher(
    tasks: List[Dict[str, Any]],
    teacher_index: Dict[Tuple[str, str], Dict[str, Dict[str, Any]]],
) -> None:
    for t in tasks:
        key = (str(t["image_id"]), str(t["target_ar"]))
        cmap = teacher_index.get(key, {})
        baseline_cid = str(t.get("baseline_used_candidate_id", ""))
        if baseline_cid and baseline_cid in cmap:
            t["baseline_candidate"] = cmap[baseline_cid]
        else:
            t["baseline_candidate"] = None

        for row in t.get("selected_topk", []):
            cid = str(row.get("candidate_id", ""))
            cinfo = cmap.get(cid)
            if not cinfo:
                continue
            row["source"] = str(cinfo.get("source", ""))
            row["score_final"] = safe_float(cinfo.get("score_final", np.nan), np.nan)
            if normalize_box(row.get("bbox_norm_xyxy")) is None and normalize_box(cinfo.get("bbox_norm_xyxy")) is not None:
                row["bbox_norm_xyxy"] = cinfo.get("bbox_norm_xyxy")

        for row in t.get("also_considered", []):
            cid = str(row.get("candidate_id", ""))
            cinfo = cmap.get(cid)
            if not cinfo:
                continue
            row["source"] = str(cinfo.get("source", ""))
            row["score_final"] = safe_float(cinfo.get("score_final", np.nan), np.nan)
            if normalize_box(row.get("bbox_norm_xyxy")) is None and normalize_box(cinfo.get("bbox_norm_xyxy")) is not None:
                row["bbox_norm_xyxy"] = cinfo.get("bbox_norm_xyxy")


def load_images_from_tars(
    ids: Sequence[str],
    mapping: Dict[str, Dict[str, Any]],
    tar_dir: str,
    image_dir: str = "",
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}

    pending: List[str] = []
    for image_id in ids:
        p = find_image_in_dir(image_dir, image_id)
        if p is not None:
            raw = cv2.imread(str(p))
            if raw is not None:
                out[image_id] = raw
                continue
        pending.append(image_id)

    tar_to_ids: Dict[str, List[str]] = defaultdict(list)
    for image_id in pending:
        info = mapping.get(image_id)
        if not info:
            continue
        tar_name = str(info.get("tar_name", ""))
        bucket = str(info.get("bucket", "")) if "bucket" in info else ""
        tar_path = resolve_tar_path(tar_dir, bucket, tar_name)
        if tar_path:
            tar_to_ids[tar_path].append(image_id)

    for tar_path, wanted_ids in tar_to_ids.items():
        wanted = set(wanted_ids)
        try:
            with tarfile.open(tar_path, "r|") as tf:
                for member in tf:
                    name = os.path.basename(member.name)
                    base, ext = os.path.splitext(name)
                    if ext.lower() not in IMAGE_EXTS:
                        continue
                    if base not in wanted:
                        continue
                    fobj = tf.extractfile(member)
                    if fobj is None:
                        wanted.remove(base)
                        if not wanted:
                            break
                        continue

                    arr = np.asarray(Image.open(io.BytesIO(fobj.read())).convert("RGB"))
                    out[base] = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                    wanted.remove(base)
                    if not wanted:
                        break
        except Exception as e:
            print(f"[warn] failed reading tar={tar_path}: {e}")

    return out


def draw_one(
    raw_bgr: np.ndarray,
    task: Dict[str, Any],
    draw_also_considered: bool,
) -> np.ndarray:
    out = raw_bgr.copy()
    h, w = out.shape[:2]

    baseline = task.get("baseline_candidate")
    if isinstance(baseline, dict):
        b = normalize_box(baseline.get("bbox_norm_xyxy"))
        if b is not None:
            x1, y1, x2, y2 = denorm_box(b, w, h)
            cv2.rectangle(out, (x1, y1), (x2, y2), BASELINE_COLOR, 2)
            btxt = f"baseline {baseline.get('candidate_id','')}"
            if baseline.get("source"):
                btxt += f" src={baseline.get('source')}"
            cv2.putText(out, btxt, (x1, min(h - 6, y2 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, BASELINE_COLOR, 1, cv2.LINE_AA)

    selected = task.get("selected_topk", [])
    for i, c in enumerate(selected):
        b = normalize_box(c.get("bbox_norm_xyxy"))
        if b is None:
            continue
        x1, y1, x2, y2 = denorm_box(b, w, h)
        color = TOPK_COLORS_BGR[i % len(TOPK_COLORS_BGR)]
        thickness = 3 if i == 0 else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        rank = safe_int(c.get("rank", i + 1), i + 1)
        cid = str(c.get("candidate_id", ""))
        s_final = c.get("score_final", None)
        src = str(c.get("source", ""))
        label = f"#{rank} {cid}"
        if s_final is not None and np.isfinite(safe_float(s_final, np.nan)):
            label += f" s={safe_float(s_final, 0.0):.3f}"
        if src:
            label += f" {src}"
        cv2.putText(out, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    if draw_also_considered:
        for c in task.get("also_considered", []):
            b = normalize_box(c.get("bbox_norm_xyxy"))
            if b is None:
                continue
            x1, y1, x2, y2 = denorm_box(b, w, h)
            cv2.rectangle(out, (x1, y1), (x2, y2), ALSO_CONSIDERED_COLOR, 1)

    top1 = selected[0] if selected else {}
    top1_tags = []
    if isinstance(top1, dict):
        top1_tags = [str(x) for x in top1.get("why_tags", [])[:3]]

    note = str(task.get("validator_note", "")).strip()
    if not note:
        note = "none"

    lines = [
        f"{task.get('image_id','')} | AR={task.get('target_ar','')} | decision={task.get('decision_type','')}",
        f"delta={safe_float(task.get('delta_improve_vs_baseline',0.0),0.0):.4f} k={len(selected)} conf={safe_float(task.get('teacher_confidence',0.0),0.0):.3f}",
        f"backend={task.get('teacher_backend','')} fallback_note={note}",
        f"top1_why={','.join(top1_tags) if top1_tags else 'n/a'}",
    ]
    draw_text_block(out, lines)
    return out


def source_group(source: str) -> str:
    s = str(source or "").lower()
    if s.startswith("baseline"):
        return "baseline"
    if "teacher" in s:
        return "teacher_seed"
    if "phi" in s:
        return "phi"
    if "jitter" in s:
        return "jitter"
    if "grid" in s:
        return "grid"
    return "other"


def build_analytics(
    analytics_tasks: List[Dict[str, Any]],
    out_dir: Path,
    top_n_tags: int,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    task_rows: List[Dict[str, Any]] = []
    sel_rows: List[Dict[str, Any]] = []
    why_counter: Counter[str] = Counter()

    for t in analytics_tasks:
        selected = t.get("selected_topk", [])
        task_rows.append(
            {
                "image_id": t.get("image_id", ""),
                "target_ar": t.get("target_ar", ""),
                "decision_type": t.get("decision_type", ""),
                "selected_k": len(selected),
                "delta_improve_vs_baseline": safe_float(t.get("delta_improve_vs_baseline", 0.0), 0.0),
                "teacher_backend": t.get("teacher_backend", ""),
                "teacher_confidence": safe_float(t.get("teacher_confidence", 0.0), 0.0),
                "validator_note": t.get("validator_note", ""),
                "baseline_used_candidate_id": t.get("baseline_used_candidate_id", ""),
            }
        )
        for c in selected:
            rank = safe_int(c.get("rank", len(sel_rows) + 1), len(sel_rows) + 1)
            src = str(c.get("source", ""))
            sel_rows.append(
                {
                    "image_id": t.get("image_id", ""),
                    "target_ar": t.get("target_ar", ""),
                    "rank": rank,
                    "candidate_id": str(c.get("candidate_id", "")),
                    "source": src,
                    "source_group": source_group(src),
                    "score_final": safe_float(c.get("score_final", np.nan), np.nan),
                    "why_tags": "|".join([str(x) for x in c.get("why_tags", [])]),
                }
            )
            for tag in c.get("why_tags", []):
                s = str(tag).strip()
                if s:
                    why_counter[s] += 1

    df_tasks = pd.DataFrame(task_rows)
    df_sel = pd.DataFrame(sel_rows)
    df_tasks.to_csv(out_dir / "vlm_task_summary.csv", index=False)
    df_sel.to_csv(out_dir / "vlm_selected_rows.csv", index=False)

    why_top = pd.DataFrame(
        [{"why_tag": k, "count": int(v)} for k, v in why_counter.most_common(max(1, int(top_n_tags)))]
    )
    why_top.to_csv(out_dir / "vlm_why_tags_top.csv", index=False)

    summary = {
        "num_tasks": int(len(df_tasks)),
        "num_selected_rows": int(len(df_sel)),
        "decision_counts": Counter(df_tasks["decision_type"].tolist()) if not df_tasks.empty else {},
        "selected_k_distribution": Counter(df_tasks["selected_k"].tolist()) if not df_tasks.empty else {},
        "per_ar_tasks": Counter(df_tasks["target_ar"].tolist()) if not df_tasks.empty else {},
        "backend_counts": Counter(df_tasks["teacher_backend"].tolist()) if not df_tasks.empty else {},
        "validator_note_counts": Counter(
            [x if str(x).strip() else "<empty>" for x in (df_tasks["validator_note"].tolist() if not df_tasks.empty else [])]
        ),
        "teacher_confidence": {
            "mean": float(df_tasks["teacher_confidence"].mean()) if not df_tasks.empty else 0.0,
            "p50": float(np.percentile(df_tasks["teacher_confidence"], 50)) if not df_tasks.empty else 0.0,
            "p90": float(np.percentile(df_tasks["teacher_confidence"], 90)) if not df_tasks.empty else 0.0,
            "min": float(df_tasks["teacher_confidence"].min()) if not df_tasks.empty else 0.0,
            "max": float(df_tasks["teacher_confidence"].max()) if not df_tasks.empty else 0.0,
        },
        "why_tags_top": {k: int(v) for k, v in why_counter.most_common(max(1, int(top_n_tags)))},
        "score_final": {
            "mean": float(df_sel["score_final"].mean()) if (not df_sel.empty and "score_final" in df_sel.columns) else 0.0,
            "p50": float(np.percentile(df_sel["score_final"].dropna(), 50))
            if (not df_sel.empty and df_sel["score_final"].notna().any())
            else 0.0,
            "p90": float(np.percentile(df_sel["score_final"].dropna(), 90))
            if (not df_sel.empty and df_sel["score_final"].notna().any())
            else 0.0,
            "min": float(df_sel["score_final"].min()) if (not df_sel.empty and df_sel["score_final"].notna().any()) else 0.0,
            "max": float(df_sel["score_final"].max()) if (not df_sel.empty and df_sel["score_final"].notna().any()) else 0.0,
        },
    }

    # Counter -> serializable dict
    for k in ("decision_counts", "selected_k_distribution", "per_ar_tasks", "backend_counts", "validator_note_counts"):
        v = summary[k]
        if isinstance(v, Counter):
            summary[k] = {str(x): int(y) for x, y in sorted(v.items(), key=lambda it: str(it[0]))}

    (out_dir / "vlm_analytics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib unavailable. csv/json only: {e}")
        return summary

    # decision distribution
    if not df_tasks.empty:
        cnt = Counter(df_tasks["decision_type"].tolist())
        keys = sorted(cnt.keys())
        vals = [cnt[k] for k in keys]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(keys, vals, color="#2563eb", edgecolor="black")
        ax.set_title("VLM Decision Type Distribution")
        ax.set_xlabel("decision_type")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(out_dir / "vlm_decision_type_distribution.png", dpi=150)
        plt.close(fig)

    # selected-k distribution
    if not df_tasks.empty:
        cnt = Counter(df_tasks["selected_k"].tolist())
        keys = sorted(cnt.keys())
        vals = [cnt[k] for k in keys]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar([str(k) for k in keys], vals, color="#16a34a", edgecolor="black")
        ax.set_title("VLM Selected-K Distribution")
        ax.set_xlabel("selected_k")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(out_dir / "vlm_selected_k_distribution.png", dpi=150)
        plt.close(fig)

    # confidence histogram
    if not df_tasks.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(df_tasks["teacher_confidence"].tolist(), bins=40, color="#9333ea", edgecolor="black", alpha=0.85)
        ax.set_title("VLM Teacher Confidence Histogram")
        ax.set_xlabel("teacher_confidence")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(out_dir / "vlm_teacher_confidence_hist.png", dpi=150)
        plt.close(fig)

    # per ar
    if not df_tasks.empty:
        cnt = Counter(df_tasks["target_ar"].tolist())
        keys = sorted(cnt.keys())
        vals = [cnt[k] for k in keys]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(keys, vals, color="#0ea5e9", edgecolor="black")
        ax.set_title("VLM Tasks per Aspect Ratio")
        ax.set_xlabel("target_ar")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(out_dir / "vlm_tasks_per_ar.png", dpi=150)
        plt.close(fig)

    # why tags top
    if not why_top.empty:
        wdf = why_top.sort_values("count", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.barh(wdf["why_tag"], wdf["count"], color="#f59e0b", edgecolor="black")
        ax.set_title(f"VLM Why-Tags Top {len(wdf)}")
        ax.set_xlabel("count")
        fig.tight_layout()
        fig.savefig(out_dir / "vlm_why_tags_top.png", dpi=150)
        plt.close(fig)

    # source mix by rank
    if not df_sel.empty and "rank" in df_sel.columns:
        by_rank_group: Dict[int, Counter[str]] = defaultdict(Counter)
        for _, row in df_sel.iterrows():
            by_rank_group[int(row["rank"])][str(row.get("source_group", "other"))] += 1
        ranks = sorted(by_rank_group.keys())
        groups = ["baseline", "grid", "phi", "jitter", "teacher_seed", "other"]
        stack_vals = {g: [] for g in groups}
        for r in ranks:
            total = float(sum(by_rank_group[r].values()))
            for g in groups:
                v = float(by_rank_group[r].get(g, 0))
                stack_vals[g].append((v / total) if total > 0 else 0.0)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        bottom = np.zeros(len(ranks), dtype=np.float64)
        color_map = {
            "baseline": "#1f77b4",
            "grid": "#2ca02c",
            "phi": "#17becf",
            "jitter": "#ff7f0e",
            "teacher_seed": "#d62728",
            "other": "#7f7f7f",
        }
        for g in groups:
            vals = np.asarray(stack_vals[g], dtype=np.float64)
            ax.bar([str(r) for r in ranks], vals, bottom=bottom, label=g, color=color_map.get(g, None))
            bottom += vals
        ax.set_title("VLM Selected Source Mix by Rank")
        ax.set_xlabel("rank")
        ax.set_ylabel("ratio")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "vlm_source_mix_by_rank.png", dpi=150)
        plt.close(fig)

    return summary


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    labels_path = Path(args.vlm_labels_jsonl)
    parquet_path = Path(args.parquet)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "by_ar").mkdir(parents=True, exist_ok=True)

    if not labels_path.exists():
        raise FileNotFoundError(f"vlm_labels_jsonl not found: {labels_path}")
    if not parquet_path.exists():
        raise FileNotFoundError(f"parquet not found: {parquet_path}")

    print(f"[info] loading labels: {labels_path}")
    tasks_all = load_vlm_tasks(labels_path, target_ar=args.target_ar)
    if not tasks_all:
        raise RuntimeError("No VLM tasks found. Check --target_ar / input jsonl.")

    print(f"[info] loading teacher index (optional): {args.teacher_scores_jsonl or '<none>'}")
    teacher_index = load_teacher_candidate_index(args.teacher_scores_jsonl)
    if teacher_index:
        enrich_tasks_with_teacher(tasks_all, teacher_index)

    # analytics pool: full target_ar filtered tasks by default
    analytics_tasks = list(tasks_all)

    explicit_ids = unique_keep_order(parse_image_ids_arg(args.image_ids) + parse_image_ids_file(args.image_ids_file))

    render_tasks = list(tasks_all)
    if explicit_ids:
        allow = set(explicit_ids)
        render_tasks = [t for t in render_tasks if str(t.get("image_id", "")) in allow]
        if not render_tasks:
            raise RuntimeError("No tasks selected after --image_ids filter.")

    if args.num_samples > 0 and len(render_tasks) > args.num_samples:
        render_tasks = random.sample(render_tasks, args.num_samples)

    if int(args.analytics_use_full_labels) == 0:
        analytics_tasks = list(render_tasks)

    # Build analytics first.
    analytics_summary = build_analytics(
        analytics_tasks=analytics_tasks,
        out_dir=out_dir / "analytics",
        top_n_tags=max(1, int(args.analytics_top_n_tags)),
    )

    # Render overlays
    df = pd.read_parquet(parquet_path)
    if "bucket" in df.columns:
        mapping = df.set_index("image_id")[["tar_name", "bucket"]].to_dict("index")
    else:
        mapping = df.set_index("image_id")[["tar_name"]].to_dict("index")

    image_ids = sorted(set(str(t.get("image_id", "")) for t in render_tasks))
    img_map = load_images_from_tars(ids=image_ids, mapping=mapping, tar_dir=args.tar_dir, image_dir=args.image_dir)

    rendered = 0
    missing_images = 0
    by_ar_count: Dict[str, int] = defaultdict(int)
    by_decision_count: Dict[str, int] = defaultdict(int)

    for t in render_tasks:
        image_id = str(t.get("image_id", ""))
        ar = str(t.get("target_ar", ""))
        raw = img_map.get(image_id)
        if raw is None:
            missing_images += 1
            continue
        ar_tok = ar.replace(":", "x")
        subdir = out_dir / "by_ar" / ar_tok
        subdir.mkdir(parents=True, exist_ok=True)

        vis = draw_one(
            raw_bgr=raw,
            task=t,
            draw_also_considered=int(args.draw_also_considered) != 0,
        )
        out_path = subdir / f"{image_id}.jpg"
        cv2.imwrite(str(out_path), vis)

        rendered += 1
        by_ar_count[ar] += 1
        by_decision_count[str(t.get("decision_type", ""))] += 1

    overview = {
        "inputs": {
            "vlm_labels_jsonl": str(labels_path),
            "teacher_scores_jsonl": str(args.teacher_scores_jsonl),
            "parquet": str(parquet_path),
            "tar_dir": str(args.tar_dir),
            "image_dir": str(args.image_dir),
        },
        "filters": {
            "target_ar": args.target_ar,
            "num_samples": int(args.num_samples),
            "explicit_image_ids": explicit_ids,
            "analytics_use_full_labels": int(args.analytics_use_full_labels),
        },
        "summary": {
            "analytics_num_tasks": int(analytics_summary.get("num_tasks", 0)),
            "analytics_num_selected_rows": int(analytics_summary.get("num_selected_rows", 0)),
            "render_tasks": int(len(render_tasks)),
            "rendered": int(rendered),
            "missing_images": int(missing_images),
            "by_ar_rendered": dict(sorted(by_ar_count.items())),
            "by_decision_rendered": dict(sorted(by_decision_count.items())),
        },
        "analytics_summary_json": str((out_dir / "analytics" / "vlm_analytics_summary.json")),
        "output_dir": str(out_dir),
    }

    with (out_dir / "viz_overview.json").open("w", encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)

    print(f"[done] analytics tasks={analytics_summary.get('num_tasks', 0)} selected_rows={analytics_summary.get('num_selected_rows', 0)}")
    print(f"[done] rendered={rendered} missing={missing_images} out_dir={out_dir}")
    print(f"[done] overview={out_dir / 'viz_overview.json'}")


if __name__ == "__main__":
    main()
