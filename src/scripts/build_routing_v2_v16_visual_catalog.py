#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from routing.routing_v2_v16_modes import V16_MODE_SPECS, image_route_no_placement_from_routing  # noqa: E402


DEFAULT_PHASE_ROOT = Path("data/SSTK/PhaseA_PhotoPrimary_10K_SigLIP2_v5_20260529")
DEFAULT_RUN_TAG = "260609_v16_routing_v2_mode_catalog_nofood_petstrict_shotstrict_nms_v2"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a unified visual catalog for routing_v2 image-route labels.")
    parser.add_argument("--phase_root", type=Path, default=DEFAULT_PHASE_ROOT)
    parser.add_argument("--run_tag", default=DEFAULT_RUN_TAG)
    parser.add_argument("--image_root", type=Path, default=DEFAULT_PHASE_ROOT / "images")
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in paths:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONT_TITLE = _font(22, bold=True)
FONT = _font(14)
FONT_SMALL = _font(12)
FONT_MONO = _font(11)

ROUTE_COLORS = {
    "scene": (70, 130, 180),
    "person_single": (32, 160, 100),
    "person_group": (155, 92, 210),
    "pet_dogcat": (230, 126, 34),
}


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _resolve_image(row: dict[str, Any], image_root: Path) -> Path:
    image_path = str(row.get("image_path") or "")
    if image_path:
        path = Path(image_path)
        if path.exists():
            return path
        candidate = image_root / path.name
        if candidate.exists():
            return candidate
    image_id = str(row.get("image_id") or row.get("source_image_id") or "")
    for suffix in (".jpg", ".jpeg", ".png", ".webp"):
        candidate = image_root / f"{image_id}{suffix}"
        if candidate.exists():
            return candidate
    return image_root / f"{image_id}.jpg"


def _fit(im: Image.Image, max_w: int, max_h: int) -> Image.Image:
    scale = min(max_w / max(1, im.width), max_h / max(1, im.height))
    size = (max(1, int(round(im.width * scale))), max(1, int(round(im.height * scale))))
    return im.resize(size, Image.Resampling.LANCZOS)


def _draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, *, width: int, font: ImageFont.ImageFont, fill: tuple[int, int, int]) -> int:
    x, y = xy
    line = ""
    for token in str(text).split():
        trial = token if not line else f"{line} {token}"
        if draw.textlength(trial, font=font) <= width:
            line = trial
        else:
            if line:
                draw.text((x, y), line, font=font, fill=fill)
                y += 17
            line = token
    if line:
        draw.text((x, y), line, font=font, fill=fill)
        y += 17
    return y


def _card(row: dict[str, Any], *, title: str, subtitle: str, image_root: Path, output_path: Path) -> dict[str, Any]:
    routing = row.get("routing_v2_simple") if isinstance(row.get("routing_v2_simple"), dict) else {}
    family = str(routing.get("route_family_v2") or "scene")
    image_route_name = str(row.get("image_route_name_no_placement") or image_route_no_placement_from_routing(routing))
    color = ROUTE_COLORS.get(family, (90, 90, 90))
    image_path = _resolve_image(row, image_root)
    im = Image.open(image_path).convert("RGB")
    fitted = _fit(im, 520, 360)
    canvas = Image.new("RGB", (720, 560), (246, 247, 249))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 720, 44), fill=color)
    title_font = FONT_TITLE
    if draw.textlength(title, font=title_font) > 680:
        title_font = _font(18, bold=True)
    if draw.textlength(title, font=title_font) > 680:
        title_font = _font(15, bold=True)
    draw.text((18, 11), title, font=title_font, fill=(255, 255, 255))
    canvas.paste(fitted, (18, 64))
    x = 558
    y = 64
    fields = [
        ("image", row.get("image_id") or row.get("source_image_id")),
        ("image_route_target", image_route_name),
        ("image_family", family),
        ("image_shot", routing.get("person_shot_type") or "na"),
        ("image_context", routing.get("context_intent")),
        ("image_feasible", routing.get("mode_feasible")),
        ("image_conf", f"{float(routing.get('routing_confidence') or 0.0):.3f}"),
        ("image_placement_aux", routing.get("placement_intent")),
        ("v16_summary_aux", row.get("v16_mode_name") or row.get("mode_name")),
        ("flat_aux", row.get("flat_route_class")),
    ]
    for key, value in fields:
        draw.text((x, y), str(key), font=FONT_SMALL, fill=(95, 99, 106))
        y += 14
        y = _draw_wrapped(
            draw,
            (x, y),
            str(value),
            width=140,
            font=FONT_MONO if key in {"flat_aux", "v16_summary_aux", "image_route_target"} else FONT,
            fill=(22, 26, 30),
        )
        y += 4
    y = max(438, fitted.height + 74)
    draw.text((18, y), subtitle[:120], font=FONT, fill=(40, 44, 48))
    y += 22
    reasons = routing.get("reasons") if isinstance(routing.get("reasons"), list) else []
    _draw_wrapped(draw, (18, y), "reasons: " + ", ".join(str(v) for v in reasons[:4]), width=670, font=FONT_SMALL, fill=(80, 84, 90))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)
    return {
        "image_id": row.get("image_id") or row.get("source_image_id"),
        "image_route_name_no_placement": image_route_name,
        "v16_summary_mode_aux": row.get("v16_mode_name") or row.get("mode_name"),
        "flat_route_class_aux": row.get("flat_route_class"),
        "route_family_v2": family,
        "person_shot_type": routing.get("person_shot_type") or "na",
        "image_placement_intent_aux": routing.get("placement_intent"),
        "context_intent": routing.get("context_intent"),
        "mode_feasible": routing.get("mode_feasible"),
        "routing_confidence": routing.get("routing_confidence"),
        "card": str(output_path),
    }


def _contact_sheet(paths: list[Path], output_path: Path, *, cols: int = 3, thumb_w: int = 320) -> None:
    if not paths:
        return
    thumbs = []
    for path in paths:
        im = Image.open(path).convert("RGB")
        scale = thumb_w / max(1, im.width)
        thumbs.append(im.resize((thumb_w, max(1, int(round(im.height * scale)))), Image.Resampling.LANCZOS))
    gap = 10
    rows = math.ceil(len(thumbs) / cols)
    h = max(t.height for t in thumbs)
    sheet = Image.new("RGB", (cols * thumb_w + (cols + 1) * gap, rows * h + (rows + 1) * gap), (235, 237, 240))
    for idx, thumb in enumerate(thumbs):
        x = gap + (idx % cols) * (thumb_w + gap)
        y = gap + (idx // cols) * (h + gap)
        sheet.paste(thumb, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=90)


def _row_rank(row: dict[str, Any]) -> tuple[float, float, str]:
    routing = row.get("routing_v2_simple") if isinstance(row.get("routing_v2_simple"), dict) else {}
    feasible = 1.0 if bool(routing.get("mode_feasible", True)) else 0.0
    return (feasible, float(routing.get("routing_confidence") or 0.0), str(row.get("image_id") or row.get("source_image_id") or ""))


def _write_count_csv(path: Path, *, key_name: str, counts: Counter[str]) -> None:
    rows = [{key_name: key, "count": int(value)} for key, value in sorted(counts.items())]
    _write_csv(path, rows)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    phase_root: Path = args.phase_root
    out_dir = args.output_dir or phase_root / "artifacts/training_labels_multimode" / args.run_tag / "routing_v2_unified_visual_catalog"
    tl_path = phase_root / "artifacts/training_labels" / args.run_tag / "routing_v2_v16_full.jsonl"
    qs_path = phase_root / "artifacts/training_labels_multimode" / args.run_tag / "mode_query_status_v16.jsonl"
    rows = _iter_jsonl(tl_path)
    query_rows = _iter_jsonl(qs_path) if qs_path.exists() else []

    image_route_best: dict[str, dict[str, Any]] = {}
    image_route_counts: Counter[str] = Counter()
    image_v16_summary_counts: Counter[str] = Counter()
    flat_counts: Counter[str] = Counter()
    for row in rows:
        routing = row.get("routing_v2_simple") if isinstance(row.get("routing_v2_simple"), dict) else {}
        image_route = str(row.get("image_route_name_no_placement") or image_route_no_placement_from_routing(routing))
        image_route_counts[image_route] += 1
        if row.get("v16_mode_name"):
            image_v16_summary_counts[str(row.get("v16_mode_name"))] += 1
        if row.get("flat_route_class"):
            flat_counts[str(row.get("flat_route_class"))] += 1
        if image_route not in image_route_best or _row_rank(row) > _row_rank(image_route_best[image_route]):
            image_route_best[image_route] = row

    query_crop_mode_counts: Counter[str] = Counter()
    query_crop_route_counts: Counter[str] = Counter()
    for row in query_rows:
        mode = str(row.get("mode_name") or row.get("v16_mode_name") or "")
        if mode:
            query_crop_mode_counts[mode] += 1
        routing = row.get("routing_v2_simple") if isinstance(row.get("routing_v2_simple"), dict) else {}
        family = str(routing.get("route_family_v2") or "")
        if family:
            query_crop_route_counts[family] += 1

    manifest = []
    card_paths = []
    for idx, key in enumerate(sorted(image_route_best), start=1):
        row = image_route_best[key]
        path = out_dir / "unified_image_route_cards" / f"{idx:03d}_{key}.jpg"
        manifest.append(
            _card(
                row,
                title=f"IMAGE ROUTE: {key}",
                subtitle="Image-level target has no placement. Crop/query modes are reviewed in crop-review panels.",
                image_root=args.image_root,
                output_path=path,
            )
        )
        card_paths.append(path)

    hierarchy_rows = []
    axes = {
        "route_family_v2": ["scene", "person_single", "person_group", "pet_dogcat"],
        "person_shot_type": ["face_headshot", "upper_half_body", "full_body"],
        "placement_intent": ["center", "rot"],
        "context_intent": ["tight_subject", "balanced", "environmental"],
        "mode_feasible": [True, False],
    }
    for axis, values in axes.items():
        for value in values:
            match = None
            for row in sorted(rows, key=_row_rank, reverse=True):
                routing = row.get("routing_v2_simple") if isinstance(row.get("routing_v2_simple"), dict) else {}
                row_value = routing.get(axis)
                if axis == "person_shot_type" and row_value is None:
                    row_value = "na"
                if row_value == value:
                    match = row
                    break
            hierarchy_rows.append(
                {
                    "axis": axis,
                    "value": value,
                    "image_id": match.get("image_id") if match else "",
                    "image_route_name_no_placement": match.get("image_route_name_no_placement") if match else "",
                    "v16_summary_mode_aux": match.get("v16_mode_name") if match else "",
                    "flat_route_class_aux": match.get("flat_route_class") if match else "",
                }
            )

    _contact_sheet(card_paths, out_dir / "unified_image_route_contact_sheet.jpg")
    _write_csv(out_dir / "unified_image_route_manifest.csv", manifest)
    _write_csv(out_dir / "hierarchical_axis_manifest.csv", hierarchy_rows)
    _write_count_csv(out_dir / "image_route_counts.csv", key_name="image_route_name_no_placement", counts=image_route_counts)
    _write_count_csv(out_dir / "image_v16_summary_mode_counts_aux.csv", key_name="v16_summary_mode_aux", counts=image_v16_summary_counts)
    _write_count_csv(out_dir / "flat_route_class_counts_aux.csv", key_name="flat_route_class_aux", counts=flat_counts)
    _write_count_csv(out_dir / "query_crop_v16_mode_counts.csv", key_name="query_crop_v16_mode", counts=query_crop_mode_counts)
    _write_count_csv(out_dir / "query_crop_route_family_counts.csv", key_name="query_crop_route_family", counts=query_crop_route_counts)
    summary = {
        "state": "completed",
        "run_tag": args.run_tag,
        "unified_image_route_cards": len(manifest),
        "hierarchical_axis_rows": len(hierarchy_rows),
        "output_dir": str(out_dir),
        "unified_image_route_contact_sheet": str(out_dir / "unified_image_route_contact_sheet.jpg"),
        "deprecated_split_visuals": [
            "flat_class_cards",
            "v16_mode_cards",
            "flat_class_contact_sheet.jpg",
            "v16_mode_contact_sheet.jpg",
        ],
    }
    _write_json(out_dir / "summary.json", summary)
    lines = [
        "# routing_v2 v16 label visual catalog",
        "",
        f"- run_tag: `{args.run_tag}`",
        f"- unified image route cards: {len(manifest)}",
        "- title policy: IMAGE ROUTE TARGET=<route_without_placement>",
        "- query/crop v16 modes and flat classes are coverage CSVs, not separate image-level card sets.",
        "- crop/query mode with placement must be inspected on crop-review panels.",
        "",
        "## Contact Sheets",
        "",
        "![](unified_image_route_contact_sheet.jpg)",
        "",
        "## Coverage Tables",
        "",
        "- `query_crop_v16_mode_counts.csv`: query/crop v16 mode distribution with placement.",
        "- `flat_route_class_counts_aux.csv`: flat class distribution for auxiliary/ablation only.",
        "- `image_v16_summary_mode_counts_aux.csv`: image-row v16 summary distribution; not an image-level primary target.",
        "",
        "## Hierarchical Axis Representatives",
        "",
        "| axis | value | image_id | image_route | v16_summary_aux | flat_aux |",
        "|---|---|---|---|---|---|",
    ]
    for row in hierarchy_rows:
        lines.append(
            f"| {row['axis']} | {row['value']} | {row['image_id']} | "
            f"{row['image_route_name_no_placement']} | {row['v16_summary_mode_aux']} | `{row['flat_route_class_aux']}` |"
        )
    (out_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
