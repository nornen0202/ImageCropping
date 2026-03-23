#!/usr/bin/env python3
from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a GAIC image-only dataset for the existing SSTK e2e pipeline "
            "by materializing a flat curated image dir and a pseudo-filtered parquet."
        )
    )
    parser.add_argument("--image_root", required=True, help="GAIC image root (can contain nested split dirs)")
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
    return parser.parse_args()


def int_if_possible(value: str) -> Any:
    try:
        return int(value)
    except Exception:
        return value


def scan_images(image_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen_ids: Dict[str, Path] = {}
    for path in sorted(image_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
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


def load_annotation_payloads(paths: Iterable[Path]) -> List[Tuple[Path, Dict[str, Any]]]:
    payloads: List[Tuple[Path, Dict[str, Any]]] = []
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            payloads.append((path, json.load(handle)))
    return payloads


def load_caption_map(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            image_id = str(rec.get("image_id", "")).strip()
            if not image_id:
                continue
            out[image_id] = rec
    return out


def build_available_reference_json(
    *,
    manifest_rows: List[Dict[str, Any]],
    payloads: List[Tuple[Path, Dict[str, Any]]],
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

    for path, payload in payloads:
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

    images: List[Dict[str, Any]] = []
    for image_id in sorted(available_ids):
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

    filtered_annotations: List[Dict[str, Any]] = []
    for ann in annotations:
        out = dict(ann)
        out["image_id"] = int_if_possible(str(ann.get("image_id")))
        filtered_annotations.append(out)

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


def main() -> None:
    args = parse_args()

    image_root = Path(args.image_root).resolve()
    output_parquet = Path(args.output_parquet).resolve()
    flat_image_dir = Path(args.flat_image_dir).resolve()
    summary_json = Path(args.summary_json).resolve()
    reference_json_out = Path(args.reference_json_out).resolve() if str(args.reference_json_out).strip() else None
    caption_jsonl = Path(args.caption_jsonl).resolve() if str(args.caption_jsonl).strip() else None

    if not image_root.exists():
        raise FileNotFoundError(f"image_root not found: {image_root}")

    scanned = scan_images(image_root)
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
    caption_map = load_caption_map(caption_jsonl)
    caption_coverage = 0

    manifest_rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(scanned):
        source_path = Path(str(item["source_path"]))
        flat_path = flat_image_dir / f"{item['image_id']}{item['ext']}"
        link_status, created = ensure_flat_image(
            source_path=source_path,
            flat_path=flat_path,
            link_mode=str(args.link_mode),
            skip_existing_links=bool(int(args.skip_existing_links)),
        )
        width, height = load_image_size(source_path)
        tar_name = f"gaic_chunk_{idx // max(1, int(args.pseudo_tar_chunk_size)):05d}.local"
        split_counter[str(item["split"])] += 1
        ext_counter[str(item["ext"])] += 1
        link_counter[link_status if created else "existing_match"] += 1
        pseudo_tar_counter[tar_name] += 1
        caption_rec = caption_map.get(str(item["image_id"]), {})
        caption_text = str(caption_rec.get("caption", "") or "").strip()
        if caption_text:
            caption_coverage += 1
        manifest_rows.append(
            {
                "image_id": str(item["image_id"]),
                "width": int(width),
                "height": int(height),
                "bucket": str(args.bucket),
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
                "dataset_name": "GAIC",
            }
        )

    df = pd.DataFrame(manifest_rows)
    if df["image_id"].duplicated().any():
        dupes = df.loc[df["image_id"].duplicated(), "image_id"].tolist()
        raise RuntimeError(f"duplicate image_id rows detected before parquet write: {dupes[:5]}")
    df.to_parquet(output_parquet, index=False)

    reference_summary: Dict[str, Any] = {"generated": False}
    if reference_json_out is not None:
        annotation_paths = [Path(path).resolve() for path in args.annotation_jsons if str(path).strip()]
        payloads = load_annotation_payloads(annotation_paths)
        reference_payload = build_available_reference_json(manifest_rows=manifest_rows, payloads=payloads)
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
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
