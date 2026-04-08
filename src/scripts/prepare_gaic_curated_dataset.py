#!/usr/bin/env python3
from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from PIL import Image

from progress_utils import ProgressTracker, progress_log


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an image-only dataset for the existing SSTK e2e pipeline "
            "by materializing a flat curated image dir and a pseudo-filtered parquet."
        )
    )
    parser.add_argument("--image_root", required=True, help="image root (can contain nested split dirs)")
    parser.add_argument("--output_parquet", required=True, help="output pseudo-filtered parquet path")
    parser.add_argument("--flat_image_dir", required=True, help="flat image dir used as curated_pool for downstream steps")
    parser.add_argument("--summary_json", required=True, help="preparation summary json")
    parser.add_argument(
        "--reference_json_out",
        default="",
        help="optional GAIC reference json generated from available train/test annotations",
    )
    parser.add_argument(
        "--annotation_jsons",
        nargs="*",
        default=[],
        help="optional GAIC annotation json files to merge/filter for available local images",
    )
    parser.add_argument("--bucket", default="gaic_all", help="pseudo bucket name stored in parquet")
    parser.add_argument(
        "--pseudo_tar_chunk_size",
        type=int,
        default=256,
        help="number of images per pseudo tar_name group to keep precompute memory bounded",
    )
    parser.add_argument(
        "--link_mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="how to materialize flat curated images",
    )
    parser.add_argument("--max_images", type=int, default=0, help="0=all, >0 keeps the first N sorted images")
    parser.add_argument(
        "--caption_jsonl",
        default="",
        help="optional caption jsonl keyed by image_id; merged into parquet caption field",
    )
    parser.add_argument(
        "--skip_existing_links",
        type=int,
        default=1,
        help="1=keep already matching flat image files, 0=revalidate and overwrite only when absent",
    )
    parser.add_argument(
        "--dataset_name",
        default="GAIC",
        help="dataset name recorded in the manifest parquet",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="CPU workers for flat-link/materialization + image-size probing (0=all cores, 1=single).",
    )
    parser.add_argument("--progress", type=int, default=1)
    parser.add_argument("--progress_every", type=int, default=500)
    parser.add_argument("--progress_min_seconds", type=float, default=10.0)
    return parser.parse_args()


def int_if_possible(value: str) -> Any:
    try:
        return int(value)
    except Exception:
        return value


def scan_images(
    image_root: Path,
    *,
    progress: bool = False,
    progress_every: int = 1000,
    progress_min_seconds: float = 10.0,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen_ids: Dict[str, Path] = {}
    candidates = sorted(image_root.rglob("*"))
    tracker = ProgressTracker(
        "prepare_gaic_curated_dataset:scan_images",
        total=len(candidates),
        unit="paths",
        every=progress_every,
        min_seconds=progress_min_seconds,
        enabled=progress,
    )
    for idx, path in enumerate(candidates, start=1):
        if not path.is_file():
            tracker.update(idx)
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            tracker.update(idx)
            continue
        image_id = path.stem
        prev = seen_ids.get(image_id)
        if prev is not None:
            raise ValueError(f"duplicate image_id detected: {image_id} -> {prev} | {path}")
        seen_ids[image_id] = path
        rel = path.relative_to(image_root)
        split = rel.parts[0] if len(rel.parts) > 1 else "root"
        rows.append(
            {
                "image_id": str(image_id),
                "source_path": path.resolve(),
                "relative_path": rel.as_posix(),
                "split": split,
                "ext": path.suffix.lower(),
            }
        )
        tracker.update(idx, extra=f"images={len(rows)}")
    tracker.finish(len(candidates), extra=f"images={len(rows)}")
    return rows


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_flat_image(
    *,
    source_path: Path,
    flat_path: Path,
    link_mode: str,
    skip_existing_links: bool,
) -> Tuple[str, bool]:
    ensure_parent(flat_path)
    same_id_matches = sorted(flat_path.parent.glob(f"{flat_path.stem}.*"))
    for existing in same_id_matches:
        if existing == flat_path:
            continue
        if existing.exists():
            try:
                if os.path.samefile(existing, source_path):
                    continue
            except Exception:
                pass
            raise ValueError(
                f"flat image dir already contains another file for image_id={flat_path.stem}: {existing}"
            )

    if flat_path.exists() or flat_path.is_symlink():
        try:
            if os.path.samefile(flat_path, source_path):
                return "existing_match", False
        except Exception:
            pass
        try:
            if flat_path.is_file() and source_path.is_file() and filecmp.cmp(flat_path, source_path, shallow=False):
                return "existing_match", False
        except Exception:
            pass
        if int(skip_existing_links) == 1:
            raise ValueError(f"existing flat image does not match source: {flat_path} vs {source_path}")
        flat_path.unlink()

    requested_mode = str(link_mode)
    if requested_mode == "symlink":
        try:
            flat_path.symlink_to(source_path)
            return "symlink", True
        except OSError:
            shutil.copy2(source_path, flat_path)
            return "copy_fallback_from_symlink", True
    if requested_mode == "hardlink":
        try:
            os.link(source_path, flat_path)
            return "hardlink", True
        except OSError:
            shutil.copy2(source_path, flat_path)
            return "copy_fallback_from_hardlink", True
    shutil.copy2(source_path, flat_path)
    return "copy", True


def load_image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as image:
        return int(image.width), int(image.height)


def resolve_num_workers(requested_workers: int, num_items: int) -> int:
    if num_items <= 1:
        return 1
    req = int(requested_workers)
    if req == 1:
        return 1
    if req <= 0:
        req = max(1, int(os.cpu_count() or 1))
    return max(1, min(req, num_items))


def load_annotation_payloads(
    paths: Iterable[Path],
    *,
    progress: bool = False,
    progress_every: int = 1,
    progress_min_seconds: float = 1.0,
) -> List[Tuple[Path, Dict[str, Any]]]:
    payloads: List[Tuple[Path, Dict[str, Any]]] = []
    path_list = list(paths)
    tracker = ProgressTracker(
        "prepare_gaic_curated_dataset:load_annotation_payloads",
        total=len(path_list),
        unit="files",
        every=progress_every,
        min_seconds=progress_min_seconds,
        enabled=progress,
    )
    for idx, path in enumerate(path_list, start=1):
        if not path.exists():
            tracker.update(idx, extra=f"skip_missing={path.name}")
            continue
        with path.open("r", encoding="utf-8") as handle:
            payloads.append((path, json.load(handle)))
        tracker.update(idx, extra=f"path={path.name}")
    tracker.finish(len(path_list), extra=f"loaded={len(payloads)}")
    return payloads


def load_caption_map(
    path: Optional[Path],
    *,
    progress: bool = False,
    progress_every: int = 1000,
    progress_min_seconds: float = 10.0,
) -> Dict[str, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    tracker = ProgressTracker(
        "prepare_gaic_curated_dataset:load_caption_map",
        unit="rows",
        every=progress_every,
        min_seconds=progress_min_seconds,
        enabled=progress,
    )
    row_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            row_count += 1
            image_id = str(rec.get("image_id", "")).strip()
            if not image_id:
                tracker.update(row_count)
                continue
            out[image_id] = rec
            tracker.update(row_count, extra=f"mapped={len(out)}")
    tracker.finish(row_count, extra=f"mapped={len(out)}")
    return out


def build_available_reference_json(
    *,
    manifest_rows: List[Dict[str, Any]],
    payloads: List[Tuple[Path, Dict[str, Any]]],
    progress: bool = False,
    progress_every: int = 1,
    progress_min_seconds: float = 1.0,
) -> Dict[str, Any]:
    manifest_meta = {
        str(row["image_id"]): {
            "file_name": Path(str(row["source_image_path"])).name,
            "width": int(row["width"]),
            "height": int(row["height"]),
        }
        for row in manifest_rows
    }
    available_ids = set(manifest_meta.keys())

    images_by_id: Dict[str, Dict[str, Any]] = {}
    annotations: List[Dict[str, Any]] = []
    categories: List[Dict[str, Any]] = [{"supercategory": "none", "id": 0, "name": "crop"}]
    type_value = "instances"
    source_stats: Dict[str, Dict[str, int]] = {}

    payload_tracker = ProgressTracker(
        "prepare_gaic_curated_dataset:build_reference:payloads",
        total=len(payloads),
        unit="jsons",
        every=progress_every,
        min_seconds=progress_min_seconds,
        enabled=progress,
    )
    for payload_idx, (path, payload) in enumerate(payloads, start=1):
        source_name = path.name
        images = payload.get("images", []) if isinstance(payload.get("images"), list) else []
        anns = payload.get("annotations", []) if isinstance(payload.get("annotations"), list) else []
        cats = payload.get("categories", []) if isinstance(payload.get("categories"), list) else []
        if cats:
            categories = cats
        if isinstance(payload.get("type"), str) and payload.get("type"):
            type_value = str(payload.get("type"))
        local_image_ids = {str(image.get("id")) for image in images if str(image.get("id")) in available_ids}
        source_stats[source_name] = {
            "images_in_json": len(images),
            "annotations_in_json": len(anns),
            "matched_available_images": len(local_image_ids),
        }
        for image in images:
            image_id = str(image.get("id"))
            if image_id not in available_ids:
                continue
            images_by_id[image_id] = dict(image)
        for ann in anns:
            image_id = str(ann.get("image_id"))
            if image_id not in available_ids:
                continue
            annotations.append(dict(ann))
        payload_tracker.update(payload_idx, extra=f"path={source_name} matched_images={len(local_image_ids)}")
    payload_tracker.finish(len(payloads), extra=f"images={len(images_by_id)} anns={len(annotations)}")

    images: List[Dict[str, Any]] = []
    image_ids_sorted = sorted(available_ids)
    image_tracker = ProgressTracker(
        "prepare_gaic_curated_dataset:build_reference:images",
        total=len(image_ids_sorted),
        unit="images",
        every=max(100, progress_every * 100),
        min_seconds=progress_min_seconds,
        enabled=progress,
    )
    for idx, image_id in enumerate(image_ids_sorted, start=1):
        base = images_by_id.get(image_id, {})
        meta = manifest_meta[image_id]
        images.append(
            {
                "file_name": Path(str(base.get("file_name") or meta["file_name"])).name,
                "height": int(base.get("height") or meta["height"]),
                "width": int(base.get("width") or meta["width"]),
                "id": int_if_possible(image_id),
            }
        )
        image_tracker.update(idx)
    image_tracker.finish(len(image_ids_sorted))

    filtered_annotations: List[Dict[str, Any]] = []
    ann_tracker = ProgressTracker(
        "prepare_gaic_curated_dataset:build_reference:annotations",
        total=len(annotations),
        unit="annotations",
        every=max(1000, progress_every * 1000),
        min_seconds=progress_min_seconds,
        enabled=progress,
    )
    for idx, ann in enumerate(annotations, start=1):
        out = dict(ann)
        out["image_id"] = int_if_possible(str(ann.get("image_id")))
        filtered_annotations.append(out)
        ann_tracker.update(idx)
    ann_tracker.finish(len(annotations))

    return {
        "images": images,
        "type": type_value,
        "annotations": filtered_annotations,
        "categories": categories,
        "_meta": {
            "source_annotation_jsons": [str(path) for path, _ in payloads],
            "source_stats": source_stats,
            "available_image_count": len(images),
            "available_annotation_count": len(filtered_annotations),
        },
    }


def prepare_manifest_row(
    *,
    idx: int,
    item: Dict[str, Any],
    flat_image_dir: Path,
    link_mode: str,
    skip_existing_links: bool,
    bucket: str,
    pseudo_tar_chunk_size: int,
    caption_map: Dict[str, Dict[str, Any]],
    dataset_name: str,
) -> Dict[str, Any]:
    source_path = Path(str(item["source_path"]))
    flat_path = flat_image_dir / f"{item['image_id']}{item['ext']}"
    link_status, created = ensure_flat_image(
        source_path=source_path,
        flat_path=flat_path,
        link_mode=link_mode,
        skip_existing_links=skip_existing_links,
    )
    width, height = load_image_size(source_path)
    tar_name = f"gaic_chunk_{idx // max(1, int(pseudo_tar_chunk_size)):05d}.local"
    caption_rec = caption_map.get(str(item["image_id"]), {})
    caption_text = str(caption_rec.get("caption", "") or "").strip()
    return {
        "split": str(item["split"]),
        "ext": str(item["ext"]),
        "link_status": link_status if created else "existing_match",
        "tar_name": tar_name,
        "caption_applied": bool(caption_text),
        "manifest_row": {
            "image_id": str(item["image_id"]),
            "width": int(width),
            "height": int(height),
            "bucket": str(bucket),
            "tar_name": tar_name,
            "tags": "",
            "caption": caption_text,
            "caption_backend": str(caption_rec.get("caption_backend", "") or ""),
            "caption_model_id": str(caption_rec.get("caption_model_id", "") or ""),
            "caption_prompt": str(caption_rec.get("caption_prompt", "") or ""),
            "super_cat": "",
            "split": str(item["split"]),
            "relative_path": str(item["relative_path"]),
            "source_image_path": str(source_path),
            "flat_image_path": str(flat_path.resolve()),
            "dataset_name": str(dataset_name),
        },
    }


def main() -> None:
    args = parse_args()
    progress_enabled = bool(int(args.progress))
    progress_every = max(1, int(args.progress_every))
    progress_min_seconds = max(0.0, float(args.progress_min_seconds))

    image_root = Path(args.image_root).resolve()
    output_parquet = Path(args.output_parquet).resolve()
    flat_image_dir = Path(args.flat_image_dir).resolve()
    summary_json = Path(args.summary_json).resolve()
    reference_json_out = Path(args.reference_json_out).resolve() if str(args.reference_json_out).strip() else None
    caption_jsonl = Path(args.caption_jsonl).resolve() if str(args.caption_jsonl).strip() else None

    if not image_root.exists():
        raise FileNotFoundError(f"image_root not found: {image_root}")

    progress_log(
        f"prepare_gaic_curated_dataset: start | image_root={image_root} | output_parquet={output_parquet}",
        enabled=progress_enabled,
    )
    scanned = scan_images(
        image_root,
        progress=progress_enabled,
        progress_every=progress_every,
        progress_min_seconds=progress_min_seconds,
    )
    total_scanned = len(scanned)
    if int(args.max_images) > 0:
        scanned = scanned[: int(args.max_images)]
    if not scanned:
        raise RuntimeError(f"no images found under: {image_root}")

    flat_image_dir.mkdir(parents=True, exist_ok=True)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    split_counter: Counter[str] = Counter()
    ext_counter: Counter[str] = Counter()
    link_counter: Counter[str] = Counter()
    pseudo_tar_counter: Counter[str] = Counter()
    caption_map = load_caption_map(
        caption_jsonl,
        progress=progress_enabled,
        progress_every=max(500, progress_every),
        progress_min_seconds=progress_min_seconds,
    )
    caption_coverage = 0
    num_workers = resolve_num_workers(int(args.num_workers), len(scanned))
    progress_log(
        f"prepare_gaic_curated_dataset: scanned={len(scanned)} | num_workers={num_workers} | captions={len(caption_map)}",
        enabled=progress_enabled,
    )

    manifest_rows: List[Dict[str, Any]] = []
    indexed_scanned = list(enumerate(scanned))
    prepare_tracker = ProgressTracker(
        "prepare_gaic_curated_dataset:prepare_manifest_rows",
        total=len(indexed_scanned),
        unit="images",
        every=max(100, progress_every),
        min_seconds=progress_min_seconds,
        enabled=progress_enabled,
    )
    if num_workers <= 1:
        prepared_rows = []
        for idx, item in indexed_scanned:
            prepared_rows.append(
                prepare_manifest_row(
                idx=idx,
                item=item,
                flat_image_dir=flat_image_dir,
                link_mode=str(args.link_mode),
                skip_existing_links=bool(int(args.skip_existing_links)),
                bucket=str(args.bucket),
                pseudo_tar_chunk_size=int(args.pseudo_tar_chunk_size),
                caption_map=caption_map,
                dataset_name=str(args.dataset_name),
            )
            )
            prepare_tracker.update(len(prepared_rows), extra=f"image_id={item['image_id']}")
    else:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            prepared_rows = []
            for prepared in executor.map(
                    lambda pair: prepare_manifest_row(
                        idx=pair[0],
                        item=pair[1],
                        flat_image_dir=flat_image_dir,
                        link_mode=str(args.link_mode),
                        skip_existing_links=bool(int(args.skip_existing_links)),
                        bucket=str(args.bucket),
                        pseudo_tar_chunk_size=int(args.pseudo_tar_chunk_size),
                        caption_map=caption_map,
                        dataset_name=str(args.dataset_name),
                    ),
                    indexed_scanned,
                ):
                prepared_rows.append(prepared)
                prepare_tracker.update(len(prepared_rows))
    prepare_tracker.finish(len(prepared_rows))

    for prepared in prepared_rows:
        split_counter[str(prepared["split"])] += 1
        ext_counter[str(prepared["ext"])] += 1
        link_counter[str(prepared["link_status"])] += 1
        pseudo_tar_counter[str(prepared["tar_name"])] += 1
        if bool(prepared["caption_applied"]):
            caption_coverage += 1
        manifest_rows.append(dict(prepared["manifest_row"]))

    df = pd.DataFrame(manifest_rows)
    if df["image_id"].duplicated().any():
        dupes = df.loc[df["image_id"].duplicated(), "image_id"].tolist()
        raise RuntimeError(f"duplicate image_id rows detected before parquet write: {dupes[:5]}")
    df.to_parquet(output_parquet, index=False)

    reference_summary: Dict[str, Any] = {"generated": False}
    if reference_json_out is not None:
        annotation_paths = [Path(path).resolve() for path in args.annotation_jsons if str(path).strip()]
        payloads = load_annotation_payloads(
            annotation_paths,
            progress=progress_enabled,
            progress_every=1,
            progress_min_seconds=progress_min_seconds,
        )
        reference_payload = build_available_reference_json(
            manifest_rows=manifest_rows,
            payloads=payloads,
            progress=progress_enabled,
            progress_every=1,
            progress_min_seconds=progress_min_seconds,
        )
        ensure_parent(reference_json_out)
        reference_json_out.write_text(
            json.dumps(reference_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        ref_meta = reference_payload.get("_meta", {}) if isinstance(reference_payload.get("_meta"), dict) else {}
        reference_summary = {
            "generated": True,
            "path": str(reference_json_out),
            "image_count": len(reference_payload.get("images", [])),
            "annotation_count": len(reference_payload.get("annotations", [])),
            "source_annotation_jsons": ref_meta.get("source_annotation_jsons", []),
            "source_stats": ref_meta.get("source_stats", {}),
        }

    summary = {
        "status": "ok",
        "image_root": str(image_root),
        "output_parquet": str(output_parquet),
        "flat_image_dir": str(flat_image_dir),
        "bucket": str(args.bucket),
        "link_mode": str(args.link_mode),
        "num_workers": int(num_workers),
        "pseudo_tar_chunk_size": int(args.pseudo_tar_chunk_size),
        "max_images": int(args.max_images),
        "num_images_scanned_total": total_scanned,
        "num_images_prepared": len(manifest_rows),
        "split_counts": dict(sorted(split_counter.items())),
        "extension_counts": dict(sorted(ext_counter.items())),
        "link_counts": dict(sorted(link_counter.items())),
        "pseudo_tar_groups": len(pseudo_tar_counter),
        "largest_pseudo_tar_group": max(pseudo_tar_counter.values()) if pseudo_tar_counter else 0,
        "caption_jsonl": (str(caption_jsonl) if caption_jsonl is not None else ""),
        "caption_rows_available": len(caption_map),
        "caption_rows_applied": caption_coverage,
        "reference": reference_summary,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    progress_log(
        f"prepare_gaic_curated_dataset: finished | prepared={len(manifest_rows)} | parquet={output_parquet}",
        enabled=progress_enabled,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
