#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
try:
    from progress_utils import ProgressTracker, progress_log
except ModuleNotFoundError:
    from scripts.progress_utils import ProgressTracker, progress_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate GAIC e2e outputs produced by the GAIC wrapper.")
    parser.add_argument("--manifest_parquet", required=True)
    parser.add_argument("--prepare_summary_json", default="")
    parser.add_argument("--precompute_jsonl", default="")
    parser.add_argument("--routed_jsonl", default="")
    parser.add_argument("--candidates_jsonl", default="")
    parser.add_argument("--teacher_jsonl", default="")
    parser.add_argument("--vlm_labels_jsonl", default="")
    parser.add_argument("--vlm_meta_jsonl", default="")
    parser.add_argument("--vlm_summary_json", default="")
    parser.add_argument("--training_validation_json", default="")
    parser.add_argument("--training_gaic_like_summary_json", default="")
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--progress", type=int, default=1)
    parser.add_argument("--progress_every", type=int, default=1000)
    parser.add_argument("--progress_min_seconds", type=float, default=10.0)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl_rows(
    path: Path,
    *,
    progress: bool = False,
    progress_every: int = 1000,
    progress_min_seconds: float = 10.0,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    tracker = ProgressTracker(
        f"validate_gaic_e2e_outputs:read_jsonl:{path.name}",
        unit="rows",
        every=progress_every,
        min_seconds=progress_min_seconds,
        enabled=progress,
    )
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            tracker.update(idx)
    tracker.finish(len(rows), extra=f"path={path.name}")
    return rows


def safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def validate_subset_ids(
    *,
    name: str,
    rows: Iterable[Dict[str, Any]],
    manifest_ids: Set[str],
    errors: List[str],
) -> Tuple[int, Set[str]]:
    row_ids: List[str] = []
    for row in rows:
        image_id = str(row.get("image_id", ""))
        if not image_id:
            errors.append(f"{name}: found row with empty image_id")
            continue
        row_ids.append(image_id)
    row_id_set = set(row_ids)
    if len(row_ids) != len(row_id_set):
        errors.append(f"{name}: duplicate image_id rows detected ({len(row_ids) - len(row_id_set)} duplicates)")
    extra = sorted(row_id_set - manifest_ids)
    if extra:
        errors.append(f"{name}: found ids not present in manifest (sample={extra[:5]})")
    return len(row_ids), row_id_set


def main() -> None:
    args = parse_args()
    progress_enabled = bool(int(args.progress))
    progress_every = max(1, int(args.progress_every))
    progress_min_seconds = max(0.0, float(args.progress_min_seconds))

    manifest_parquet = Path(args.manifest_parquet).resolve()
    summary_json = Path(args.summary_json).resolve()
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    if not manifest_parquet.exists():
        raise FileNotFoundError(f"manifest parquet not found: {manifest_parquet}")

    progress_log(
        f"validate_gaic_e2e_outputs: start | manifest_parquet={manifest_parquet} | summary_json={summary_json}",
        enabled=progress_enabled,
    )
    df = pd.read_parquet(manifest_parquet)
    required_cols = {"image_id", "width", "height", "source_image_path", "flat_image_path"}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"manifest missing required columns: {missing_cols}")

    manifest_rows = int(len(df))
    manifest_ids = {str(x) for x in df["image_id"].astype(str).tolist()}
    errors: List[str] = []
    warnings: List[str] = []
    sections: Dict[str, Any] = {}

    if len(manifest_ids) != manifest_rows:
        errors.append("manifest: duplicate image_id rows detected")
    if (df["width"].astype(int) <= 0).any() or (df["height"].astype(int) <= 0).any():
        errors.append("manifest: non-positive width/height detected")

    missing_source = [str(x) for x in df["source_image_path"].tolist() if not Path(str(x)).exists()]
    missing_flat = [str(x) for x in df["flat_image_path"].tolist() if not Path(str(x)).exists()]
    if missing_source:
        errors.append(f"manifest: missing source_image_path count={len(missing_source)} sample={missing_source[:3]}")
    if missing_flat:
        errors.append(f"manifest: missing flat_image_path count={len(missing_flat)} sample={missing_flat[:3]}")

    sections["manifest"] = {
        "rows": manifest_rows,
        "unique_image_ids": len(manifest_ids),
        "split_counts": {
            str(key): int(value)
            for key, value in df["split"].fillna("unknown").astype(str).value_counts().sort_index().items()
        }
        if "split" in df.columns
        else {},
    }

    if str(args.prepare_summary_json).strip():
        prepare_path = Path(args.prepare_summary_json).resolve()
        if not prepare_path.exists():
            errors.append(f"prepare_summary_json missing: {prepare_path}")
        else:
            progress_log(f"validate_gaic_e2e_outputs: load prepare summary {prepare_path.name}", enabled=progress_enabled)
            payload = load_json(prepare_path)
            prepared = int(payload.get("num_images_prepared", -1))
            if prepared != manifest_rows:
                errors.append(
                    f"prepare_summary_json mismatch: num_images_prepared={prepared}, manifest_rows={manifest_rows}"
                )
            sections["prepare_summary"] = payload

    precompute_count = None
    if str(args.precompute_jsonl).strip():
        path = Path(args.precompute_jsonl).resolve()
        if not path.exists():
            errors.append(f"precompute_jsonl missing: {path}")
        else:
            rows = read_jsonl_rows(
                path,
                progress=progress_enabled,
                progress_every=progress_every,
                progress_min_seconds=progress_min_seconds,
            )
            count, id_set = validate_subset_ids(
                name="precompute_jsonl",
                rows=rows,
                manifest_ids=manifest_ids,
                errors=errors,
            )
            precompute_count = count
            if count != manifest_rows:
                errors.append(f"precompute_jsonl row count mismatch: expected={manifest_rows}, got={count}")
            sections["precompute"] = {
                "path": str(path),
                "rows": count,
                "has_c2_seg_rows": sum(1 for row in rows if isinstance(row.get("c2_seg"), list)),
                "has_c3_pose_rows": sum(1 for row in rows if isinstance(row.get("c3_pose"), list)),
                "has_c4_ocr_rows": sum(1 for row in rows if "c4_ocr" in row),
                "has_c5_geom_rows": sum(1 for row in rows if isinstance(row.get("c5_geom"), dict)),
                "unique_image_ids": len(id_set),
            }

    if str(args.routed_jsonl).strip():
        path = Path(args.routed_jsonl).resolve()
        if not path.exists():
            errors.append(f"routed_jsonl missing: {path}")
        else:
            rows = read_jsonl_rows(
                path,
                progress=progress_enabled,
                progress_every=progress_every,
                progress_min_seconds=progress_min_seconds,
            )
            count, id_set = validate_subset_ids(
                name="routed_jsonl",
                rows=rows,
                manifest_ids=manifest_ids,
                errors=errors,
            )
            if count != manifest_rows:
                errors.append(f"routed_jsonl row count mismatch: expected={manifest_rows}, got={count}")
            routing_rows = sum(1 for row in rows if isinstance(row.get("routing"), dict))
            if routing_rows != count:
                errors.append(f"routed_jsonl missing routing payloads: expected={count}, got={routing_rows}")
            sections["routed"] = {
                "path": str(path),
                "rows": count,
                "routing_rows": routing_rows,
                "unique_image_ids": len(id_set),
            }

    candidate_rows: List[Dict[str, Any]] = []
    candidate_ids: Set[str] = set()
    if str(args.candidates_jsonl).strip():
        path = Path(args.candidates_jsonl).resolve()
        if not path.exists():
            errors.append(f"candidates_jsonl missing: {path}")
        else:
            candidate_rows = read_jsonl_rows(
                path,
                progress=progress_enabled,
                progress_every=progress_every,
                progress_min_seconds=progress_min_seconds,
            )
            count, candidate_ids = validate_subset_ids(
                name="candidates_jsonl",
                rows=candidate_rows,
                manifest_ids=manifest_ids,
                errors=errors,
            )
            missing_candidates_by_ar = sum(
                1 for row in candidate_rows if not isinstance(row.get("candidates_by_ar"), dict) or not row.get("candidates_by_ar")
            )
            if missing_candidates_by_ar:
                errors.append(f"candidates_jsonl rows missing candidates_by_ar: {missing_candidates_by_ar}")
            sections["candidates"] = {
                "path": str(path),
                "rows": count,
                "unique_image_ids": len(candidate_ids),
                "subset_of_manifest": count <= manifest_rows,
            }
            if count < manifest_rows:
                warnings.append(
                    f"candidates_jsonl covers only {count}/{manifest_rows} images. This is expected for subset smoke runs."
                )

    teacher_rows: List[Dict[str, Any]] = []
    teacher_ids: Set[str] = set()
    expected_vlm_tasks = 0
    if str(args.teacher_jsonl).strip():
        path = Path(args.teacher_jsonl).resolve()
        if not path.exists():
            errors.append(f"teacher_jsonl missing: {path}")
        else:
            teacher_rows = read_jsonl_rows(
                path,
                progress=progress_enabled,
                progress_every=progress_every,
                progress_min_seconds=progress_min_seconds,
            )
            count, teacher_ids = validate_subset_ids(
                name="teacher_jsonl",
                rows=teacher_rows,
                manifest_ids=manifest_ids,
                errors=errors,
            )
            if candidate_ids and not teacher_ids.issubset(candidate_ids):
                errors.append("teacher_jsonl contains image_ids not present in candidates_jsonl")
            result_rows = 0
            for row in teacher_rows:
                scorer = safe_dict(row.get("teacher_scorer"))
                results_by_ar = safe_dict(scorer.get("results_by_ar"))
                if results_by_ar:
                    result_rows += 1
                    expected_vlm_tasks += len(results_by_ar)
            if result_rows != count:
                errors.append(f"teacher_jsonl rows missing teacher_scorer.results_by_ar: {count - result_rows}")
            sections["teacher"] = {
                "path": str(path),
                "rows": count,
                "unique_image_ids": len(teacher_ids),
                "expected_vlm_tasks_all_ar": expected_vlm_tasks,
            }
            if count < manifest_rows:
                warnings.append(
                    f"teacher_jsonl covers only {count}/{manifest_rows} images. This is expected when subset scoring was requested."
                )

    if str(args.vlm_meta_jsonl).strip():
        path = Path(args.vlm_meta_jsonl).resolve()
        if not path.exists():
            errors.append(f"vlm_meta_jsonl missing: {path}")
        else:
            rows = read_jsonl_rows(
                path,
                progress=progress_enabled,
                progress_every=progress_every,
                progress_min_seconds=progress_min_seconds,
            )
            count, meta_ids = validate_subset_ids(
                name="vlm_meta_jsonl",
                rows=rows,
                manifest_ids=manifest_ids,
                errors=errors,
            )
            if teacher_ids and not meta_ids.issubset(teacher_ids):
                errors.append("vlm_meta_jsonl contains image_ids not present in teacher_jsonl")
            sections["vlm_meta"] = {
                "path": str(path),
                "rows": count,
                "unique_image_ids": len(meta_ids),
            }

    if str(args.vlm_labels_jsonl).strip():
        path = Path(args.vlm_labels_jsonl).resolve()
        if not path.exists():
            errors.append(f"vlm_labels_jsonl missing: {path}")
        else:
            rows = read_jsonl_rows(
                path,
                progress=progress_enabled,
                progress_every=progress_every,
                progress_min_seconds=progress_min_seconds,
            )
            count = len(rows)
            bad_rows = 0
            label_ids: Set[str] = set()
            for row in rows:
                image_id = str(row.get("image_id", ""))
                target_ar = str(row.get("target_ar", ""))
                label_ids.add(image_id)
                if image_id not in manifest_ids:
                    bad_rows += 1
                if not target_ar:
                    bad_rows += 1
                if not isinstance(row.get("selected_topk"), list) or not row.get("selected_topk"):
                    bad_rows += 1
            if bad_rows:
                errors.append(f"vlm_labels_jsonl contains invalid rows: {bad_rows}")
            sections["vlm_labels"] = {
                "path": str(path),
                "rows": count,
                "unique_image_ids": len(label_ids),
            }
            if expected_vlm_tasks and count != expected_vlm_tasks:
                warnings.append(
                    f"vlm_labels_jsonl rows ({count}) != expected all-AR tasks from teacher ({expected_vlm_tasks}). "
                    "This can happen when target_ar/max_images filters were used."
                )

    if str(args.vlm_summary_json).strip():
        path = Path(args.vlm_summary_json).resolve()
        if not path.exists():
            errors.append(f"vlm_summary_json missing: {path}")
        else:
            payload = load_json(path)
            task_written = int(safe_dict(payload.get("counts")).get("task_written", 0))
            if "vlm_labels" in sections:
                vlm_rows = int(sections["vlm_labels"]["rows"])
                if task_written != vlm_rows:
                    errors.append(f"vlm_summary_json task_written mismatch: expected={vlm_rows}, got={task_written}")
            sections["vlm_summary"] = payload

    if str(args.training_validation_json).strip():
        path = Path(args.training_validation_json).resolve()
        if not path.exists():
            errors.append(f"training_validation_json missing: {path}")
        else:
            payload = load_json(path)
            if str(payload.get("status", "")).lower() not in {"ok", "passed", "success"}:
                errors.append(f"training_validation_json status not ok: {payload.get('status')}")
            sections["training_validation"] = payload

    if str(args.training_gaic_like_summary_json).strip():
        path = Path(args.training_gaic_like_summary_json).resolve()
        if not path.exists():
            errors.append(f"training_gaic_like_summary_json missing: {path}")
        else:
            payload = load_json(path)
            validation = safe_dict(payload.get("validation"))
            if validation and str(validation.get("status", "")).lower() != "ok":
                errors.append(
                    "training_gaic_like_summary_json validation status not ok: "
                    f"{validation.get('status')}"
                )
            sections["training_gaic_like"] = payload

    summary = {
        "status": "ok" if not errors else "failed",
        "manifest_rows": manifest_rows,
        "precompute_rows": precompute_count,
        "errors": errors,
        "warnings": warnings,
        "sections": sections,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    progress_log(
        f"validate_gaic_e2e_outputs: finished | status={summary['status']} | errors={len(errors)} | warnings={len(warnings)}",
        enabled=progress_enabled,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
