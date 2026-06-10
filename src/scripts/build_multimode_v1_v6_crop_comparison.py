from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


MODE_COLORS = {
    "landscape": (35, 82, 220),
    "single_person_center": (30, 170, 70),
    "single_person_rot": (230, 60, 60),
    "group_center": (190, 70, 230),
    "group_rot": (220, 70, 160),
    "face": (230, 215, 30),
    "object_single_center": (40, 160, 220),
    "object_single_rot": (20, 120, 200),
    "object_multi_center": (70, 190, 130),
    "object_multi_rot": (110, 170, 70),
}

MODE_PRIORITY = {
    "face": 0,
    "single_person_center": 1,
    "single_person_rot": 2,
    "group_center": 3,
    "group_rot": 4,
    "object_single_center": 5,
    "object_single_rot": 6,
    "object_multi_center": 7,
    "object_multi_rot": 8,
    "landscape": 9,
}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _source_id(row: Dict[str, Any]) -> str:
    return str(row.get("source_image_id") or Path(str(row.get("file_name") or "")).stem)


def _image_path(image_root: Path, image_row: Dict[str, Any], source_id: str) -> Path:
    file_name = str(image_row.get("file_name") or "")
    if file_name:
        direct = image_root / file_name
        if direct.is_file():
            return direct
    for suffix in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        candidate = image_root / f"{source_id}{suffix}"
        if candidate.is_file():
            return candidate
    return image_root / file_name


def _load_label_index(path: Path) -> Dict[str, Any]:
    payload = _load_json(path)
    category_names = {
        int(row.get("id", -1)): str(row.get("name") or row.get("mode_name") or row.get("id"))
        for row in payload.get("categories", [])
        if isinstance(row, dict)
    }
    image_by_id: Dict[int, Dict[str, Any]] = {}
    source_to_image: Dict[str, Dict[str, Any]] = {}
    for row in payload.get("images", []):
        if not isinstance(row, dict):
            continue
        image_id = int(row["id"])
        source_id = _source_id(row)
        image_by_id[image_id] = row
        source_to_image[source_id] = row

    anns_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ann in payload.get("annotations", []):
        if not isinstance(ann, dict):
            continue
        if int(ann.get("gt_flag", 0)) != 1:
            continue
        image_row = image_by_id.get(int(ann.get("image_id", -1)))
        if image_row is None:
            continue
        source_id = _source_id(image_row)
        ann = dict(ann)
        mode_name = str(ann.get("mode_name") or category_names.get(int(ann.get("category_id", -1)), "unknown"))
        ann["mode_name"] = mode_name
        anns_by_source[source_id].append(ann)

    return {
        "source_to_image": source_to_image,
        "anns_by_source": anns_by_source,
        "category_names": category_names,
    }


def _read_ids(path: Path) -> List[str]:
    seen = set()
    ids: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        source_id = line.strip()
        if not source_id or source_id.startswith("#") or source_id in seen:
            continue
        seen.add(source_id)
        ids.append(source_id)
    return ids


def _find_overlay(source_id: str, primary: Path, fallbacks: Sequence[Path]) -> Optional[Path]:
    for directory in (primary, *fallbacks):
        path = directory / f"vis_{source_id}.jpg"
        if path.is_file():
            return path
    return None


def _fit_width(image: Image.Image, width: int) -> Image.Image:
    image = image.convert("RGB")
    if image.width == width:
        return image
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _fit_box(image: Image.Image, width: int, height: int) -> Image.Image:
    image = image.convert("RGB")
    scale = min(width / image.width, height / image.height)
    resized = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), (248, 248, 248))
    canvas.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return canvas


def _make_overlay_image(
    *,
    source_id: str,
    image_root: Path,
    image_row: Optional[Dict[str, Any]],
    anns: List[Dict[str, Any]],
    title: str,
    max_boxes_per_mode: int = 12,
) -> Image.Image:
    title_font = _font(24, bold=True)
    small_font = _font(15)
    if image_row is None:
        image = Image.new("RGB", (960, 540), (235, 235, 235))
        draw = ImageDraw.Draw(image)
        draw.text((24, 240), f"{title}: image metadata missing", fill=(120, 120, 120), font=title_font)
        return image
    image_path = _image_path(image_root, image_row, source_id)
    if not image_path.is_file():
        image = Image.new("RGB", (960, 540), (235, 235, 235))
        draw = ImageDraw.Draw(image)
        draw.text((24, 240), f"{title}: source image missing", fill=(120, 120, 120), font=title_font)
        draw.text((24, 276), str(image_path), fill=(120, 120, 120), font=small_font)
        return image
    base = Image.open(image_path).convert("RGB")
    header_h = 34
    canvas = Image.new("RGB", (base.width, base.height + header_h), (0, 0, 0))
    canvas.paste(base, (0, header_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), f"{title}: {source_id}", fill=(245, 245, 245), font=small_font)
    by_mode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ann in anns:
        by_mode[str(ann.get("mode_name") or "unknown")].append(ann)
    for mode_name in sorted(by_mode, key=lambda value: MODE_PRIORITY.get(value, 99)):
        color = MODE_COLORS.get(mode_name, (120, 120, 120))
        mode_anns = sorted(by_mode[mode_name], key=_ann_sort_key)[:max(1, int(max_boxes_per_mode))]
        for ann in mode_anns:
            x, y, w, h = [_safe_float(value) for value in (ann.get("bbox") or [0, 0, 0, 0])[:4]]
            x1 = int(round(x))
            y1 = int(round(y + header_h))
            x2 = int(round(x + w))
            y2 = int(round(y + h + header_h))
            for offset in range(2):
                draw.rectangle((x1 - offset, y1 - offset, x2 + offset, y2 + offset), outline=color)
            attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
            label = f"{mode_name} {attrs.get('target_ar', '')} {_safe_float(ann.get('score_mode'), 0.0):.2f}".strip()
            if len(label) > 34:
                label = label[:33] + "."
            text_y = max(header_h, y1 - 16)
            draw.rectangle((x1, text_y, x1 + min(260, 7 * len(label) + 8), text_y + 15), fill=(0, 0, 0))
            draw.text((x1 + 3, text_y + 1), label, fill=color, font=small_font)
    return canvas


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _positive_counts(anns: Iterable[Dict[str, Any]]) -> Counter:
    counter: Counter[str] = Counter()
    for ann in anns:
        counter[str(ann.get("mode_name") or "unknown")] += 1
    return counter


def _ann_sort_key(ann: Dict[str, Any]) -> Tuple[int, float]:
    mode_name = str(ann.get("mode_name") or "unknown")
    return (MODE_PRIORITY.get(mode_name, 99), -_safe_float(ann.get("score_mode", ann.get("score", 0.0))))


def _make_crop_strip(
    *,
    source_id: str,
    image_root: Path,
    image_row: Optional[Dict[str, Any]],
    anns: List[Dict[str, Any]],
    title: str,
    max_crops: int,
    strip_width: int,
    crop_thumb_width: int,
    crop_thumb_height: int,
    crop_cols: int,
) -> Image.Image:
    title_font = _font(22, bold=True)
    small_font = _font(15)
    thumb_w = max(96, int(crop_thumb_width))
    thumb_h = max(80, int(crop_thumb_height))
    gap = 8
    label_h = 34
    header_h = 40
    auto_cols = max(1, (strip_width - gap) // (thumb_w + gap))
    cols = min(auto_cols, max(1, int(crop_cols))) if int(crop_cols) > 0 else auto_cols
    sorted_anns = sorted(anns, key=_ann_sort_key)
    selected = sorted_anns if int(max_crops) <= 0 else sorted_anns[:max_crops]
    rows = max(1, (len(selected) + cols - 1) // cols)
    strip_h = header_h + rows * (thumb_h + label_h + gap) + gap
    canvas = Image.new("RGB", (strip_width, strip_h), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), f"{title} positive crops ({len(anns)})", fill=(20, 20, 20), font=title_font)
    if not selected:
        draw.text((10, header_h + 18), "No positive crop annotations in this label JSON.", fill=(105, 105, 105), font=small_font)
        return canvas
    if image_row is None:
        draw.text((10, header_h + 18), "Image metadata missing in this label JSON.", fill=(150, 40, 40), font=small_font)
        return canvas
    image_path = _image_path(image_root, image_row, source_id)
    if not image_path.is_file():
        draw.text((10, header_h + 18), f"Source image missing: {image_path}", fill=(150, 40, 40), font=small_font)
        return canvas
    source = Image.open(image_path).convert("RGB")
    for idx, ann in enumerate(selected):
        bbox = ann.get("bbox") or [0, 0, 0, 0]
        x, y, w, h = [_safe_float(value) for value in bbox[:4]]
        left = max(0, min(source.width, int(round(x))))
        top = max(0, min(source.height, int(round(y))))
        right = max(left + 1, min(source.width, int(round(x + w))))
        bottom = max(top + 1, min(source.height, int(round(y + h))))
        crop = source.crop((left, top, right, bottom))
        thumb = _fit_box(crop, thumb_w, thumb_h)
        col = idx % cols
        row = idx // cols
        px = gap + col * (thumb_w + gap)
        py = header_h + gap + row * (thumb_h + label_h + gap)
        mode_name = str(ann.get("mode_name") or "unknown")
        color = MODE_COLORS.get(mode_name, (120, 120, 120))
        canvas.paste(thumb, (px, py))
        draw.rectangle((px, py, px + thumb_w - 1, py + thumb_h - 1), outline=color, width=4)
        attrs = ann.get("attributes") if isinstance(ann.get("attributes"), dict) else {}
        target_ar = str(attrs.get("target_ar") or "")
        score = _safe_float(ann.get("score_mode", ann.get("score", 0.0)))
        label = f"{mode_name} {target_ar} {score:.2f}".strip()
        if len(label) > 26:
            label = label[:25] + "."
        draw.text((px, py + thumb_h + 3), label, fill=(20, 20, 20), font=small_font)
    return canvas


def _counts_text(counts: Counter, max_items: int = 6) -> str:
    if not counts:
        return "none"
    pairs = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:max_items]
    return ", ".join(f"{key}:{value}" for key, value in pairs)


def _build_comparison(
    *,
    source_id: str,
    v1_overlay: Optional[Path],
    v6_overlay: Optional[Path],
    v1_index: Dict[str, Any],
    v6_index: Dict[str, Any],
    image_root: Path,
    out_path: Path,
    max_crops: int,
    left_title: str,
    right_title: str,
    left_key: str,
    right_key: str,
    crop_thumb_width: int,
    crop_thumb_height: int,
    crop_cols: int,
) -> Dict[str, Any]:
    panel_w = 960
    header_h = 96
    bg = (244, 244, 244)
    title_font = _font(30, bold=True)
    small_font = _font(18)

    v1_anns = list(v1_index["anns_by_source"].get(source_id, []))
    v6_anns = list(v6_index["anns_by_source"].get(source_id, []))
    overlays: List[Image.Image] = []
    overlay_specs = [
        (v1_overlay, v1_index["source_to_image"].get(source_id) or v6_index["source_to_image"].get(source_id), v1_anns, left_title),
        (v6_overlay, v6_index["source_to_image"].get(source_id) or v1_index["source_to_image"].get(source_id), v6_anns, right_title),
    ]
    for path, image_row, anns, title in overlay_specs:
        if path and path.is_file():
            overlays.append(_fit_width(Image.open(path), panel_w))
        else:
            overlays.append(_fit_width(_make_overlay_image(source_id=source_id, image_root=image_root, image_row=image_row, anns=anns, title=title), panel_w))
    top_h = max(image.height for image in overlays)
    v1_strip = _make_crop_strip(
        source_id=source_id,
        image_root=image_root,
        image_row=v1_index["source_to_image"].get(source_id) or v6_index["source_to_image"].get(source_id),
        anns=v1_anns,
        title=left_title,
        max_crops=max_crops,
        strip_width=panel_w,
        crop_thumb_width=crop_thumb_width,
        crop_thumb_height=crop_thumb_height,
        crop_cols=crop_cols,
    )
    v6_strip = _make_crop_strip(
        source_id=source_id,
        image_root=image_root,
        image_row=v6_index["source_to_image"].get(source_id) or v1_index["source_to_image"].get(source_id),
        anns=v6_anns,
        title=right_title,
        max_crops=max_crops,
        strip_width=panel_w,
        crop_thumb_width=crop_thumb_width,
        crop_thumb_height=crop_thumb_height,
        crop_cols=crop_cols,
    )
    strip_h = max(v1_strip.height, v6_strip.height)
    canvas = Image.new("RGB", (panel_w * 2, header_h + top_h + strip_h), bg)
    draw = ImageDraw.Draw(canvas)
    v1_counts = _positive_counts(v1_anns)
    v6_counts = _positive_counts(v6_anns)
    draw.text((18, 12), f"{left_title}: {source_id}", fill=(20, 20, 20), font=title_font)
    draw.text((18, 52), _counts_text(v1_counts), fill=(70, 70, 70), font=small_font)
    draw.text((panel_w + 18, 12), f"{right_title}: {source_id}", fill=(20, 20, 20), font=title_font)
    draw.text((panel_w + 18, 52), _counts_text(v6_counts), fill=(70, 70, 70), font=small_font)
    canvas.paste(overlays[0], (0, header_h))
    canvas.paste(overlays[1], (panel_w, header_h))
    canvas.paste(v1_strip, (0, header_h + top_h))
    canvas.paste(v6_strip, (panel_w, header_h + top_h))
    draw.line((panel_w - 1, 0, panel_w - 1, canvas.height), fill=(180, 180, 180), width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)
    row = {
        "source_image_id": source_id,
        "comparison_image": str(out_path),
        "v1_overlay": str(v1_overlay) if v1_overlay else "",
        "v6_overlay": str(v6_overlay) if v6_overlay else "",
        "v1_positive_total": len(v1_anns),
        "v6_positive_total": len(v6_anns),
        "v1_mode_counts": dict(v1_counts),
        "v6_mode_counts": dict(v6_counts),
    }
    row[f"{left_key}_overlay"] = str(v1_overlay) if v1_overlay else ""
    row[f"{right_key}_overlay"] = str(v6_overlay) if v6_overlay else ""
    row[f"{left_key}_positive_total"] = len(v1_anns)
    row[f"{right_key}_positive_total"] = len(v6_anns)
    row[f"{left_key}_mode_counts"] = dict(v1_counts)
    row[f"{right_key}_mode_counts"] = dict(v6_counts)
    return row


def _save_contact_sheet_chunk(
    *,
    rows: List[Dict[str, Any]],
    thumbs: List[Image.Image],
    out_path: Path,
    cols: int,
    thumb_w: int,
    cell_h: int,
) -> None:
    sheet = Image.new("RGB", (cols * thumb_w, ((len(thumbs) + cols - 1) // cols) * cell_h), (246, 246, 246))
    draw = ImageDraw.Draw(sheet)
    title_font = _font(22, bold=True)
    for idx, (row, thumb) in enumerate(zip(rows, thumbs)):
        x = (idx % cols) * thumb_w
        y = (idx // cols) * cell_h
        draw.text((x + 8, y + 8), str(row["source_image_id"]), fill=(20, 20, 20), font=title_font)
        sheet.paste(thumb, (x, y + 42))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def _contact_sheet(rows: List[Dict[str, Any]], out_path: Path, cols: int = 2) -> List[str]:
    if not rows:
        return []
    thumb_w = 760
    thumbs = []
    for row in rows:
        image = Image.open(row["comparison_image"]).convert("RGB")
        scale = thumb_w / image.width
        thumbs.append(image.resize((thumb_w, max(1, round(image.height * scale))), Image.Resampling.LANCZOS))
    cell_h = max(thumb.height for thumb in thumbs) + 52
    max_jpeg_dimension = 60000
    max_rows_per_sheet = max(1, max_jpeg_dimension // max(1, cell_h))
    max_items_per_sheet = max(cols, max_rows_per_sheet * cols)
    if len(rows) <= max_items_per_sheet:
        _save_contact_sheet_chunk(rows=rows, thumbs=thumbs, out_path=out_path, cols=cols, thumb_w=thumb_w, cell_h=cell_h)
        return [str(out_path)]

    out_paths: List[str] = []
    stem = out_path.stem
    suffix = out_path.suffix or ".jpg"
    for start in range(0, len(rows), max_items_per_sheet):
        end = min(len(rows), start + max_items_per_sheet)
        chunk_path = out_path.with_name(f"{stem}_part{len(out_paths) + 1:03d}{suffix}")
        _save_contact_sheet_chunk(
            rows=rows[start:end],
            thumbs=thumbs[start:end],
            out_path=chunk_path,
            cols=cols,
            thumb_w=thumb_w,
            cell_h=cell_h,
        )
        out_paths.append(str(chunk_path))
    return out_paths


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v1/v6 overlay plus crop-thumbnail comparison sheets.")
    parser.add_argument("--v1_label_json", required=True)
    parser.add_argument("--v6_label_json", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--selected_ids", required=True)
    parser.add_argument("--v1_overlay_dir", required=True)
    parser.add_argument("--v6_overlay_dir", required=True)
    parser.add_argument("--v1_overlay_fallback_dir", action="append", default=[])
    parser.add_argument("--v6_overlay_fallback_dir", action="append", default=[])
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--max_crops", type=int, default=12, help="Maximum positive crops per side. Use 0 or negative to include all positive crops.")
    parser.add_argument("--crop_thumb_width", type=int, default=150)
    parser.add_argument("--crop_thumb_height", type=int, default=128)
    parser.add_argument("--crop_cols", type=int, default=0, help="Crop thumbnail columns per side. Use 0 for automatic width-based layout.")
    parser.add_argument("--left_title", default="v1 260413")
    parser.add_argument("--right_title", default="v6 260528")
    parser.add_argument("--left_key", default="v1")
    parser.add_argument("--right_key", default="v6")
    parser.add_argument("--manifest_prefix", default="v1_v6")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    pairs_dir = out_dir / "pairs_with_crops"
    ids = _read_ids(Path(args.selected_ids))
    v1_index = _load_label_index(Path(args.v1_label_json))
    v6_index = _load_label_index(Path(args.v6_label_json))
    image_root = Path(args.image_root)
    rows: List[Dict[str, Any]] = []
    for source_id in ids:
        v1_overlay = _find_overlay(source_id, Path(args.v1_overlay_dir), [Path(p) for p in args.v1_overlay_fallback_dir])
        v6_overlay = _find_overlay(source_id, Path(args.v6_overlay_dir), [Path(p) for p in args.v6_overlay_fallback_dir])
        row = _build_comparison(
            source_id=source_id,
            v1_overlay=v1_overlay,
            v6_overlay=v6_overlay,
            v1_index=v1_index,
            v6_index=v6_index,
            image_root=image_root,
            out_path=pairs_dir / f"compare_{source_id}_with_crops.jpg",
            max_crops=int(args.max_crops),
            left_title=str(args.left_title),
            right_title=str(args.right_title),
            left_key=str(args.left_key),
            right_key=str(args.right_key),
            crop_thumb_width=int(args.crop_thumb_width),
            crop_thumb_height=int(args.crop_thumb_height),
            crop_cols=int(args.crop_cols),
        )
        rows.append(row)
    manifest_prefix = str(args.manifest_prefix)
    contact_path = out_dir / f"{manifest_prefix}_overlay_crop_comparison_contact_sheet.jpg"
    contact_sheets = _contact_sheet(rows, contact_path, cols=2)
    manifest_json = out_dir / f"{manifest_prefix}_overlay_crop_comparison_manifest.json"
    manifest_csv = out_dir / f"{manifest_prefix}_overlay_crop_comparison_manifest.csv"
    manifest_json.write_text(
        json.dumps(
            {
                "selected_count": len(rows),
                "contact_sheet": contact_sheets[0] if contact_sheets else "",
                "contact_sheets": contact_sheets,
                "max_crops_per_version": int(args.max_crops),
                "crop_thumb_width": int(args.crop_thumb_width),
                "crop_thumb_height": int(args.crop_thumb_height),
                "crop_cols": int(args.crop_cols),
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "source_image_id",
            f"{args.left_key}_positive_total",
            f"{args.right_key}_positive_total",
            "v1_positive_total",
            "v6_positive_total",
            "comparison_image",
            f"{args.left_key}_overlay",
            f"{args.right_key}_overlay",
            "v1_overlay",
            "v6_overlay",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "selected_count": len(rows),
                "contact_sheet": contact_sheets[0] if contact_sheets else "",
                "contact_sheets": contact_sheets,
                "manifest_json": str(manifest_json),
                "manifest_csv": str(manifest_csv),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
