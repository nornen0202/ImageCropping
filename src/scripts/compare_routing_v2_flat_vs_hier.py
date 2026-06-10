#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from routing.routing_v2_simple import HIERARCHICAL_VOCABS, flat_route_class_key  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare flat-class vs hierarchical routing_v2_simple route heads.")
    parser.add_argument("--train_jsonl", required=True, type=Path)
    parser.add_argument("--val_jsonl", required=True, type=Path)
    parser.add_argument("--vocabs", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--min_train_conf", type=float, default=0.65)
    parser.add_argument("--min_eval_conf", type=float, default=0.0)
    parser.add_argument("--max_train_rows", type=int, default=0)
    parser.add_argument("--max_val_rows", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260608)
    return parser


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_rows(path: Path, *, max_rows: int = 0, min_conf: float = 0.0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rv2 = row.get("routing_v2_simple") if isinstance(row.get("routing_v2_simple"), dict) else {}
            conf = float(rv2.get("routing_confidence") or row.get("label_weight") or 0.0)
            if conf < float(min_conf):
                continue
            rows.append(row)
            if max_rows > 0 and len(rows) >= max_rows:
                break
    return rows


def _feature_names(rows: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        feats = row.get("probe_features") if isinstance(row.get("probe_features"), dict) else {}
        names.update(str(k) for k in feats)
    return sorted(names)


def _matrix(rows: list[dict[str, Any]], names: list[str]) -> np.ndarray:
    data = []
    for row in rows:
        feats = row.get("probe_features") if isinstance(row.get("probe_features"), dict) else {}
        data.append([float(feats.get(name, 0.0) or 0.0) for name in names])
    return np.asarray(data, dtype=np.float32)


def _target(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    if key == "flat":
        return np.asarray([int(row.get("flat_route_class_id", -1)) for row in rows], dtype=np.int64)
    targets = []
    for row in rows:
        hier = row.get("hierarchical_targets") if isinstance(row.get("hierarchical_targets"), dict) else {}
        targets.append(int(hier.get(key, -1)))
    return np.asarray(targets, dtype=np.int64)


def _fit_classifier(x: np.ndarray, y: np.ndarray, *, seed: int) -> Any:
    unique = sorted(set(int(v) for v in y.tolist()))
    if len(unique) <= 1:
        clf = DummyClassifier(strategy="most_frequent")
        clf.fit(x, y)
        return clf
    max_iter = 1000
    clf = LogisticRegression(
        max_iter=max_iter,
        class_weight="balanced",
        random_state=int(seed),
        n_jobs=1,
        solver="lbfgs",
        multi_class="auto",
    )
    return make_pipeline(StandardScaler(), clf).fit(x, y)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    present_ids = sorted(set(int(v) for v in y_true.tolist()) | set(int(v) for v in y_pred.tolist()))
    names = [labels[idx] if 0 <= idx < len(labels) else str(idx) for idx in present_ids]
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(set(y_true.tolist())) > 1 else float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) else 0.0,
        "labels": names,
        "label_ids": present_ids,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=present_ids).tolist() if len(y_true) else [],
        "classification_report": classification_report(y_true, y_pred, labels=present_ids, target_names=names, zero_division=0, output_dict=True) if len(y_true) else {},
    }


def _flat_labels(vocabs: dict[str, Any]) -> list[str]:
    flat = vocabs.get("flat_route_class") if isinstance(vocabs.get("flat_route_class"), dict) else {}
    labels = ["" for _ in range(len(flat))]
    for key, idx in flat.items():
        if 0 <= int(idx) < len(labels):
            labels[int(idx)] = str(key)
    return labels


def _compose_hier_predictions(
    *,
    rows: list[dict[str, Any]],
    family_pred: np.ndarray,
    shot_pred: np.ndarray | None,
    placement_pred: np.ndarray,
    context_pred: np.ndarray,
    feasible_pred: np.ndarray,
    flat_vocab: dict[str, int],
) -> np.ndarray:
    out = []
    shot_cursor = 0
    for idx, row in enumerate(rows):
        family = HIERARCHICAL_VOCABS["route_family_v2"][int(family_pred[idx])]
        shot = None
        if family == "person_single" and shot_pred is not None and shot_cursor < len(shot_pred):
            shot = HIERARCHICAL_VOCABS["person_shot_type"][int(shot_pred[shot_cursor])]
            shot_cursor += 1
        placement = HIERARCHICAL_VOCABS["placement_intent"][int(placement_pred[idx])]
        context = HIERARCHICAL_VOCABS["context_intent"][int(context_pred[idx])]
        feasible = bool(int(feasible_pred[idx]))
        key = flat_route_class_key(
            {
                "route_family_v2": family,
                "person_shot_type": shot,
                "placement_intent": placement,
                "context_intent": context,
                "mode_feasible": feasible,
            }
        )
        out.append(int(flat_vocab.get(key, -1)))
    return np.asarray(out, dtype=np.int64)


def _counter(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    c = Counter()
    for row in rows:
        if key == "flat":
            c[str(row.get("flat_route_class"))] += 1
        else:
            rv2 = row.get("routing_v2_simple") if isinstance(row.get("routing_v2_simple"), dict) else {}
            c[str(rv2.get(key))] += 1
    return dict(c)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output_dir / "status.json"
    _write_json(status_path, {"state": "running", "phase": "load_data"})
    start = time.time()

    vocabs = _read_json(args.vocabs)
    flat_vocab = vocabs.get("flat_route_class") if isinstance(vocabs.get("flat_route_class"), dict) else {}
    flat_labels = _flat_labels(vocabs)
    train_rows = _load_rows(args.train_jsonl, max_rows=int(args.max_train_rows), min_conf=float(args.min_train_conf))
    val_rows = _load_rows(args.val_jsonl, max_rows=int(args.max_val_rows), min_conf=float(args.min_eval_conf))
    if not train_rows or not val_rows:
        raise RuntimeError("empty routing_v2 train/val rows after confidence filtering")
    names = _feature_names(train_rows + val_rows)
    x_train = _matrix(train_rows, names)
    x_val = _matrix(val_rows, names)
    _write_json(status_path, {"state": "running", "phase": "fit_flat", "train_rows": len(train_rows), "val_rows": len(val_rows)})

    y_flat_train = _target(train_rows, "flat")
    y_flat_val = _target(val_rows, "flat")
    flat_model = _fit_classifier(x_train, y_flat_train, seed=int(args.seed))
    flat_pred = flat_model.predict(x_val)
    flat_metrics = _metrics(y_flat_val, flat_pred, flat_labels)

    _write_json(status_path, {"state": "running", "phase": "fit_hierarchical"})
    y_family_train = _target(train_rows, "route_family_v2_id")
    y_family_val = _target(val_rows, "route_family_v2_id")
    family_model = _fit_classifier(x_train, y_family_train, seed=int(args.seed))
    family_pred = family_model.predict(x_val)

    y_place_train = _target(train_rows, "placement_intent_id")
    y_place_val = _target(val_rows, "placement_intent_id")
    place_model = _fit_classifier(x_train, y_place_train, seed=int(args.seed) + 1)
    place_pred = place_model.predict(x_val)

    y_context_train = _target(train_rows, "context_intent_id")
    y_context_val = _target(val_rows, "context_intent_id")
    context_model = _fit_classifier(x_train, y_context_train, seed=int(args.seed) + 2)
    context_pred = context_model.predict(x_val)

    y_feas_train = _target(train_rows, "mode_feasible_id")
    y_feas_val = _target(val_rows, "mode_feasible_id")
    feas_model = _fit_classifier(x_train, y_feas_train, seed=int(args.seed) + 3)
    feas_pred = feas_model.predict(x_val)

    train_shot_mask = y_family_train == HIERARCHICAL_VOCABS["route_family_v2"].index("person_single")
    val_shot_mask = y_family_val == HIERARCHICAL_VOCABS["route_family_v2"].index("person_single")
    shot_metrics: dict[str, Any]
    shot_pred_full = None
    if int(train_shot_mask.sum()) > 0 and int(val_shot_mask.sum()) > 0:
        y_shot_train = _target(train_rows, "person_shot_type_id")[train_shot_mask]
        y_shot_val = _target(val_rows, "person_shot_type_id")[val_shot_mask]
        shot_model = _fit_classifier(x_train[train_shot_mask], y_shot_train, seed=int(args.seed) + 4)
        shot_pred_full = shot_model.predict(x_val[val_shot_mask])
        shot_metrics = _metrics(y_shot_val, shot_pred_full, HIERARCHICAL_VOCABS["person_shot_type"])
    else:
        shot_metrics = {"skipped": True, "reason": "no person_single rows after filtering"}

    hier_flat_pred = _compose_hier_predictions(
        rows=val_rows,
        family_pred=family_pred,
        shot_pred=shot_pred_full,
        placement_pred=place_pred,
        context_pred=context_pred,
        feasible_pred=feas_pred,
        flat_vocab={str(k): int(v) for k, v in flat_vocab.items()},
    )
    valid_hier_mask = hier_flat_pred >= 0
    hier_flat_metrics = _metrics(y_flat_val[valid_hier_mask], hier_flat_pred[valid_hier_mask], flat_labels)
    hier_metrics = {
        "route_family_v2": _metrics(y_family_val, family_pred, HIERARCHICAL_VOCABS["route_family_v2"]),
        "person_shot_type_masked": shot_metrics,
        "placement_intent": _metrics(y_place_val, place_pred, HIERARCHICAL_VOCABS["placement_intent"]),
        "context_intent": _metrics(y_context_val, context_pred, HIERARCHICAL_VOCABS["context_intent"]),
        "mode_feasible": _metrics(y_feas_val, feas_pred, ["False", "True"]),
        "composed_flat_exact": hier_flat_metrics,
        "composed_flat_valid_rate": float(valid_hier_mask.mean()) if len(valid_hier_mask) else 0.0,
    }
    summary = {
        "state": "completed",
        "phase": "completed",
        "train_jsonl": str(args.train_jsonl),
        "val_jsonl": str(args.val_jsonl),
        "vocabs": str(args.vocabs),
        "min_train_conf": float(args.min_train_conf),
        "min_eval_conf": float(args.min_eval_conf),
        "feature_count": len(names),
        "feature_names": names,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_counts": {
            "route_family_v2": _counter(train_rows, "route_family_v2"),
            "person_shot_type": _counter(train_rows, "person_shot_type"),
            "flat": _counter(train_rows, "flat"),
        },
        "val_counts": {
            "route_family_v2": _counter(val_rows, "route_family_v2"),
            "person_shot_type": _counter(val_rows, "person_shot_type"),
            "flat": _counter(val_rows, "flat"),
        },
        "flat_head": flat_metrics,
        "hierarchical_head": hier_metrics,
        "winner_by_balanced_flat_exact": "hierarchical" if hier_flat_metrics["balanced_accuracy"] > flat_metrics["balanced_accuracy"] else "flat",
        "elapsed_sec": round(time.time() - start, 3),
    }
    _write_json(args.output_dir / "summary.json", summary)
    _write_json(args.output_dir / "status.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
