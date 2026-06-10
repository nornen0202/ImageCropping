#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build teacher Top-K analytics assets for report.")
    p.add_argument("--teacher_scores_jsonl", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def source_group(source: str, source_types: List[str]) -> str:
    s = str(source or "").lower()
    st = [str(x).lower() for x in (source_types or [])]
    joined = " ".join([s] + st)
    if s.startswith("baseline"):
        return "baseline"
    if "teacher:" in joined or "teacher:jitter" in joined:
        return "teacher_seed"
    if "phi" in joined:
        return "phi"
    if "jitter" in joined:
        return "jitter"
    if "grid" in joined:
        return "grid"
    return "other"


def load_rows(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            image_id = str(rec.get("image_id", ""))
            by_ar = rec.get("teacher_scorer", {}).get("results_by_ar", {})
            if not isinstance(by_ar, dict):
                continue
            for ar_text, ar_res in by_ar.items():
                if not isinstance(ar_res, dict):
                    continue
                selected = ar_res.get("selected_topk", [])
                if not isinstance(selected, list):
                    selected = []
                k = len(selected)
                for rank, cand in enumerate(selected, start=1):
                    if not isinstance(cand, dict):
                        continue
                    source = str(cand.get("source", "unknown"))
                    stypes = cand.get("source_types", [])
                    if not isinstance(stypes, list):
                        stypes = []
                    score_final = float(cand.get("scores", {}).get("final", 0.0))
                    rows.append(
                        {
                            "image_id": image_id,
                            "target_ar": str(ar_text),
                            "selected_k": int(k),
                            "rank": int(rank),
                            "candidate_id": str(cand.get("candidate_id", "")),
                            "source": source,
                            "source_group": source_group(source, stypes),
                            "score_final": score_final,
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    in_path = Path(args.teacher_scores_jsonl)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_rows(in_path)
    if df.empty:
        summary = {
            "input_jsonl": str(in_path),
            "num_selected_rows": 0,
            "num_tasks": 0,
            "selected_k_distribution": {},
            "source_group_distribution": {},
            "rank_count": {},
        }
        (out_dir / "teacher_topk_distribution_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        pd.DataFrame([]).to_csv(out_dir / "teacher_selected_topk_rows.csv", index=False)
        pd.DataFrame([]).to_csv(out_dir / "teacher_task_level_summary.csv", index=False)
        print(f"[done] empty input. outputs initialized under: {out_dir}")
        return

    task_df = (
        df.groupby(["image_id", "target_ar"], as_index=False)
        .agg(selected_k=("selected_k", "max"), top1_final=("score_final", "max"), mean_final=("score_final", "mean"))
    )
    df.to_csv(out_dir / "teacher_selected_topk_rows.csv", index=False)
    task_df.to_csv(out_dir / "teacher_task_level_summary.csv", index=False)

    k_dist = Counter(task_df["selected_k"].tolist())
    src_dist = Counter(df["source_group"].tolist())
    rank_dist = Counter(df["rank"].tolist())

    summary = {
        "input_jsonl": str(in_path),
        "num_selected_rows": int(len(df)),
        "num_tasks": int(len(task_df)),
        "selected_k_distribution": {str(k): int(v) for k, v in sorted(k_dist.items())},
        "source_group_distribution": {str(k): int(v) for k, v in sorted(src_dist.items())},
        "rank_count": {str(k): int(v) for k, v in sorted(rank_dist.items())},
        "score_final": {
            "mean": float(df["score_final"].mean()),
            "p50": float(np.percentile(df["score_final"], 50)),
            "p90": float(np.percentile(df["score_final"], 90)),
            "min": float(df["score_final"].min()),
            "max": float(df["score_final"].max()),
        },
    }
    (out_dir / "teacher_topk_distribution_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[warn] matplotlib unavailable. csv/json only: {e}")
        print(f"[done] analytics tables written to: {out_dir}")
        return

    # 1) selected_k distribution
    k_keys = sorted(k_dist.keys())
    k_vals = [k_dist[k] for k in k_keys]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(k) for k in k_keys], k_vals, color="#2563eb", edgecolor="black")
    ax.set_title("Teacher Selected-K Distribution")
    ax.set_xlabel("selected_k")
    ax.set_ylabel("num(image, ar) tasks")
    fig.tight_layout()
    fig.savefig(out_dir / "teacher_selected_k_distribution.png", dpi=int(args.dpi))
    plt.close(fig)

    # 2) final score histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["score_final"].tolist(), bins=40, color="#10b981", edgecolor="black", alpha=0.85)
    ax.set_title("Teacher Top-K Final Score Histogram")
    ax.set_xlabel("score_final")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_dir / "teacher_topk_final_score_hist.png", dpi=int(args.dpi))
    plt.close(fig)

    # 3) source mix by rank (stacked ratio)
    by_rank_group: Dict[int, Counter[str]] = defaultdict(Counter)
    for _, row in df.iterrows():
        by_rank_group[int(row["rank"])][str(row["source_group"])] += 1
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
    ax.set_title("Teacher Top-K Source Mix by Rank")
    ax.set_xlabel("rank")
    ax.set_ylabel("ratio")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "teacher_topk_source_mix_by_rank.png", dpi=int(args.dpi))
    plt.close(fig)

    print(f"[done] analytics built under: {out_dir}")


if __name__ == "__main__":
    main()

