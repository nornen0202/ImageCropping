#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from label_artifacts.paths import LEGACY_COCO_DIRNAME, iter_legacy_mapping_paths, legacy_to_label_json_path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def discover_legacy_dirs(root: Path) -> list[Path]:
    skip_names = {
        ".git",
        ".idea",
        "__pycache__",
        "images",
        "image",
        "imgs",
        "jpg",
        "jpeg",
        "png",
        "tar",
        "tars",
        "weights",
        "venv",
        ".venv",
        "node_modules",
        "overlays",
        "crops",
        "per_image",
    }
    out: list[Path] = []
    for dirpath, dirnames, _filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in skip_names and not name.startswith(".cache")]
        current = Path(dirpath)
        if current.name == LEGACY_COCO_DIRNAME:
            out.append(current)
            dirnames[:] = []
    return sorted(out)


def link_or_copy(src: Path, dst: Path, *, link_mode: str, dry_run: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return "dry_run"
    if dst.exists():
        try:
            if dst.stat().st_size != src.stat().st_size:
                dst.unlink()
            else:
                return "exists"
        except OSError:
            dst.unlink(missing_ok=True)
    if dst.exists():
        return "exists"
    if link_mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            raise
    if link_mode == "symlink":
        try:
            os.symlink(os.path.relpath(src, start=dst.parent), dst)
            return "symlink"
        except OSError:
            raise
    if link_mode == "move":
        shutil.move(str(src), str(dst))
        return "move"
    shutil.copy2(src, dst)
    return "copy"


def replace_paths_and_keys(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            new_key = {
                "coco_json": "label_json",
                "target_ar_only_coco_json": "target_ar_only_label_json",
                "output_coco_json": "output_label_json",
                "input_coco_json": "input_label_json",
                "multimode_coco_json": "multimode_label_json",
                "coco_image_count": "source_image_entry_count",
                "coco_images_missing_from_image_root": "label_images_missing_from_image_root",
                "coco_images_missing_from_features": "label_images_missing_from_features",
                "coco_images_missing_from_candidates": "label_images_missing_from_candidates",
            }.get(str(key), str(key))
            out[new_key] = replace_paths_and_keys(item)
            if new_key != key:
                out[str(key)] = replace_paths_and_keys(item)
        return out
    if isinstance(value, list):
        return [replace_paths_and_keys(item) for item in value]
    if isinstance(value, str) and "/coco/" in value:
        return str(legacy_to_label_json_path(value))
    return value


def update_metadata_json(path: Path, *, dry_run: bool) -> bool:
    if not path.is_file() or path.suffix.lower() != ".json":
        return False
    try:
        payload = read_json(path)
    except Exception:
        return False
    updated = replace_paths_and_keys(payload)
    if updated == payload:
        return False
    if not dry_run:
        write_json(path, updated)
    return True


def replace_text_paths(text: str) -> str:
    path_replacements = {
        "/coco/instances_multimode_training_labels_train_target_ar_only.json": "/label_json/multimode_labels_target_ar_only_train.json",
        "/coco/instances_multimode_training_labels_val_target_ar_only.json": "/label_json/multimode_labels_target_ar_only_val.json",
        "/coco/instances_multimode_training_labels_target_ar_only.json": "/label_json/multimode_labels_target_ar_only_full.json",
        "/coco/instances_multimode_training_labels_train.json": "/label_json/multimode_labels_train.json",
        "/coco/instances_multimode_training_labels_val.json": "/label_json/multimode_labels_val.json",
        "/coco/instances_multimode_training_labels.json": "/label_json/multimode_labels_full.json",
        "/coco/instances_conditional_detr_batch_gaic_like_unassigned.json": "/label_json/gaic_like_labels_unassigned.json",
        "/coco/instances_conditional_detr_batch_gaic_like_train.json": "/label_json/gaic_like_labels_train.json",
        "/coco/instances_conditional_detr_batch_gaic_like_val.json": "/label_json/gaic_like_labels_val.json",
        "/coco/instances_conditional_detr_batch_gaic_like_test.json": "/label_json/gaic_like_labels_test.json",
        "/coco/instances_conditional_detr_batch_gaic_like.json": "/label_json/gaic_like_labels_full.json",
        "/coco/instances_conditional_detr_canonical.json": "/label_json/conditional_detr_labels_canonical.json",
        "/coco/instances_conditional_detr_batch.json": "/label_json/conditional_detr_labels_batch.json",
        "/coco/coco_conversion_summary.json": "/label_json/annotation_format_conversion_summary.json",
        "/coco/gaic_like_conversion_summary.json": "/label_json/gaic_like_conversion_summary.json",
        "/coco/GAIC_INSTANCES_TRAIN_FORMAT_KO.md": "/label_json/GAIC_LIKE_LABEL_FORMAT_KO.md",
        "coco/instances_multimode_training_labels_train_target_ar_only.json": "label_json/multimode_labels_target_ar_only_train.json",
        "coco/instances_multimode_training_labels_val_target_ar_only.json": "label_json/multimode_labels_target_ar_only_val.json",
        "coco/instances_multimode_training_labels_target_ar_only.json": "label_json/multimode_labels_target_ar_only_full.json",
        "coco/instances_multimode_training_labels_train.json": "label_json/multimode_labels_train.json",
        "coco/instances_multimode_training_labels_val.json": "label_json/multimode_labels_val.json",
        "coco/instances_multimode_training_labels.json": "label_json/multimode_labels_full.json",
        "coco/instances_conditional_detr_batch_gaic_like_unassigned.json": "label_json/gaic_like_labels_unassigned.json",
        "coco/instances_conditional_detr_batch_gaic_like_train.json": "label_json/gaic_like_labels_train.json",
        "coco/instances_conditional_detr_batch_gaic_like_val.json": "label_json/gaic_like_labels_val.json",
        "coco/instances_conditional_detr_batch_gaic_like_test.json": "label_json/gaic_like_labels_test.json",
        "coco/instances_conditional_detr_batch_gaic_like.json": "label_json/gaic_like_labels_full.json",
        "coco/instances_conditional_detr_canonical.json": "label_json/conditional_detr_labels_canonical.json",
        "coco/instances_conditional_detr_batch.json": "label_json/conditional_detr_labels_batch.json",
        "coco/coco_conversion_summary.json": "label_json/annotation_format_conversion_summary.json",
        "coco/gaic_like_conversion_summary.json": "label_json/gaic_like_conversion_summary.json",
        "coco/GAIC_INSTANCES_TRAIN_FORMAT_KO.md": "label_json/GAIC_LIKE_LABEL_FORMAT_KO.md",
        "`coco/instances_conditional_detr_batch_gaic_like_train.json`": "`label_json/gaic_like_labels_train.json`",
        "`coco/instances_multimode_training_labels.json`": "`label_json/multimode_labels_full.json`",
        "`coco/": "`label_json/",
    }
    out = text
    for old, new in path_replacements.items():
        out = out.replace(old, new)
    token_replacements = {
        "--coco_json": "--label_json",
        "--multimode_coco_json": "--multimode_label_json",
        "--target_ar_only_coco_json": "--target_ar_only_label_json",
        "--size_reference_coco_json": "--size_reference_label_json",
        "coco_json": "label_json",
        "multimode_coco_json": "multimode_label_json",
        "target_ar_only_coco_json": "target_ar_only_label_json",
        "size_reference_coco_json": "size_reference_label_json",
        "output_coco_json": "output_label_json",
        "input_coco_json": "input_label_json",
        "coco_image_count": "source_image_entry_count",
        "coco images": "source image entries",
        "COCO images": "source image entries",
        "COCO image": "source image entry",
        "multimode COCO": "multimode label JSON",
        "COCO annotation 포맷": "annotation JSON 포맷",
        "COCO annotation schema": "annotation JSON schema",
        "coco conversion summary": "annotation-format conversion summary",
    }
    for old, new in token_replacements.items():
        out = out.replace(old, new)
    return out


def update_text_file(path: Path, *, dry_run: bool) -> bool:
    if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return False
    updated = replace_text_paths(text)
    if updated == text:
        return False
    if not dry_run:
        path.write_text(updated, encoding="utf-8")
    return True


def iter_run_doc_files(run_dir: Path) -> list[Path]:
    skip_names = {
        "per_image",
        "overlays",
        "crops",
        "debug_viz",
        "debug_visualizations_balanced50_bottomneg",
        "debug_visualizations",
        "label_json",
        "coco",
    }
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(run_dir):
        rel_depth = len(Path(dirpath).relative_to(run_dir).parts)
        if rel_depth > 3:
            dirnames[:] = []
            continue
        dirnames[:] = [name for name in dirnames if name not in skip_names]
        for name in filenames:
            path = Path(dirpath) / name
            if path.suffix.lower() in {".md", ".txt"}:
                out.append(path)
    return sorted(out)


def migrate_one_run(run_dir: Path, *, link_mode: str, dry_run: bool, remove_legacy_coco_dir: bool) -> dict[str, Any]:
    legacy_dir = run_dir / LEGACY_COCO_DIRNAME
    mappings = []
    if legacy_dir.is_dir():
        for old, new in iter_legacy_mapping_paths(legacy_dir):
            size = old.stat().st_size
            action = link_or_copy(old, new, link_mode=link_mode, dry_run=dry_run)
            mappings.append({"old_path": str(old), "new_path": str(new), "action": action, "bytes": size})

    metadata_candidates = [
        run_dir / "summary.json",
        run_dir / "status.json",
        run_dir / "validation_summary.json",
        run_dir / "validation_summary_strict.json",
        run_dir / "splits" / "split_manifest.json",
    ]
    metadata_updated = [str(path) for path in metadata_candidates if update_metadata_json(path, dry_run=dry_run)]
    docs_updated = [str(path) for path in iter_run_doc_files(run_dir) if update_text_file(path, dry_run=dry_run)]
    removed = False
    if remove_legacy_coco_dir and legacy_dir.is_dir() and not dry_run:
        shutil.rmtree(legacy_dir)
        removed = True

    manifest = {
        "schema_version": "label_artifact_layout_migration_v1",
        "run_dir": str(run_dir),
        "source_layout": "legacy_coco_dir" if legacy_dir.is_dir() or removed else "no_legacy_coco_dir",
        "target_layout": "label_json_v1",
        "link_mode": link_mode,
        "dry_run": bool(dry_run),
        "legacy_removed": removed,
        "mapping_count": len(mappings),
        "mappings": mappings,
        "metadata_updated": metadata_updated,
        "docs_updated": docs_updated,
    }
    if not dry_run and (mappings or metadata_updated or removed):
        write_json(run_dir / "artifact_layout_migration.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate SSTK/GAIC/TestImages label artifacts from legacy coco/ to label_json/.")
    parser.add_argument("--run_dir", default="")
    parser.add_argument("--root", default="")
    parser.add_argument("--discover", type=int, default=0)
    parser.add_argument("--link_mode", choices=["hardlink", "symlink", "copy", "move"], default="hardlink")
    parser.add_argument("--dry_run", type=int, default=0)
    parser.add_argument("--remove_legacy_coco_dir", type=int, default=0)
    parser.add_argument("--out_report", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dry_run = bool(int(args.dry_run))
    run_dirs: list[Path] = []
    if args.run_dir:
        run_dirs.append(Path(args.run_dir))
    if args.root:
        root = Path(args.root)
        if int(args.discover) == 1:
            run_dirs.extend(sorted({path.parent for path in discover_legacy_dirs(root)}))
        else:
            run_dirs.append(root)
    if not run_dirs:
        raise ValueError("provide --run_dir or --root")
    reports = [
        migrate_one_run(
            run_dir,
            link_mode=str(args.link_mode),
            dry_run=dry_run,
            remove_legacy_coco_dir=bool(int(args.remove_legacy_coco_dir)),
        )
        for run_dir in sorted(set(run_dirs))
    ]
    payload = {
        "schema_version": "label_artifact_layout_migration_report_v1",
        "run_count": len(reports),
        "mapping_count": sum(int(row.get("mapping_count", 0)) for row in reports),
        "dry_run": dry_run,
        "reports": reports,
    }
    if args.out_report:
        write_json(Path(args.out_report), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
