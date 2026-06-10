#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


DEFAULT_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
CATEGORY_COLUMN_CANDIDATES = ("super_cat", "super_cat_x", "super_cat_y", "category", "category_x", "category_y")


def _safe_value(value: Any) -> Any:
    try:
        import numpy as np

        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
    except Exception:
        pass
    return value


def resolve_image_path(image_dir: Path, image_id: str) -> Path | None:
    stem = Path(str(image_id)).stem
    for ext in DEFAULT_IMAGE_EXTS:
        path = image_dir / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def normalize_category_column(df: pd.DataFrame) -> pd.DataFrame:
    if "super_cat" in df.columns:
        return df
    for col in CATEGORY_COLUMN_CANDIDATES:
        if col in df.columns:
            out = df.copy()
            out["super_cat"] = out[col]
            return out
    return df


def select_photo_primary_pool(
    df: pd.DataFrame,
    *,
    target_size: int,
    random_state: int,
) -> pd.DataFrame:
    if "media_decision" not in df.columns:
        raise ValueError("input parquet must contain media_decision")

    df = normalize_category_column(df)
    keep = df[df["media_decision"].astype(str) == "keep_photo_primary"].copy()
    if len(keep) <= target_size:
        return keep.reset_index(drop=True)

    if "super_cat" not in keep.columns:
        return keep.sample(n=target_size, random_state=random_state).reset_index(drop=True)

    cats = keep["super_cat"].value_counts()
    target = int(target_size)
    cap = 0
    n_cats = len(cats)
    cats_sorted = cats.sort_values(ascending=True)
    for i, count in enumerate(cats_sorted):
        alloc = target // max(1, n_cats - i)
        if count <= alloc:
            target -= int(count)
        else:
            cap = int(alloc)
            break
    if cap == 0 and target > 0:
        cap = max(1, target // max(1, n_cats))

    def sample_group(group: pd.DataFrame) -> pd.DataFrame:
        n_samples = min(len(group), cap)
        weights = group["sampling_weight"] if "sampling_weight" in group.columns else None
        return group.sample(n=n_samples, random_state=random_state, weights=weights)

    selected_parts = [sample_group(group) for _, group in keep.groupby("super_cat", sort=False)]
    selected = pd.concat(selected_parts, ignore_index=False) if selected_parts else keep.head(0)
    remaining = int(target_size) - len(selected)
    if remaining > 0:
        leftovers = keep.loc[~keep.index.isin(selected.index)]
        if len(leftovers) > 0:
            weights = leftovers["sampling_weight"] if "sampling_weight" in leftovers.columns else None
            selected = pd.concat(
                [
                    selected,
                    leftovers.sample(
                        n=min(remaining, len(leftovers)),
                        random_state=random_state,
                        weights=weights,
                    ),
                ],
                ignore_index=False,
            )
    return selected.reset_index(drop=True)


def export_selected_images(
    df: pd.DataFrame,
    *,
    input_image_dir: Path,
    output_image_dir: Path,
    mode: str,
) -> dict[str, Any]:
    output_image_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    missing: list[str] = []
    errors: list[dict[str, str]] = []
    for image_id in df["image_id"].astype(str):
        src = resolve_image_path(input_image_dir, image_id)
        if src is None:
            missing.append(image_id)
            continue
        dst = output_image_dir / src.name
        try:
            if dst.exists():
                copied += 1
                continue
            if mode == "hardlink":
                os.link(src, dst)
            elif mode == "symlink":
                os.symlink(src, dst)
            else:
                shutil.copy2(src, dst)
            copied += 1
        except Exception as exc:
            errors.append({"image_id": image_id, "src": str(src), "error": str(exc)})
    return {
        "copied_or_existing": int(copied),
        "missing": int(len(missing)),
        "missing_ids": missing[:100],
        "errors": errors[:100],
        "error_count": int(len(errors)),
    }


def write_contact_sheet(df: pd.DataFrame, image_dir: Path, path: Path, *, max_images: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subset = df.copy()
    if "nonphoto_score" in subset.columns:
        subset = subset.sort_values("nonphoto_score", ascending=False)
    subset = subset.head(max_images)

    thumb_w, thumb_h = 160, 120
    label_h = 34
    cols = 5
    rows = max(1, int((len(subset) + cols - 1) // cols))
    canvas = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for idx, (_, row) in enumerate(subset.iterrows()):
        x = (idx % cols) * thumb_w
        y = (idx // cols) * (thumb_h + label_h)
        src = resolve_image_path(image_dir, str(row["image_id"]))
        if src is not None:
            try:
                with Image.open(src) as im:
                    im = im.convert("RGB")
                    im.thumbnail((thumb_w, thumb_h), Image.Resampling.BILINEAR)
                    canvas.paste(im, (x + (thumb_w - im.width) // 2, y + (thumb_h - im.height) // 2))
            except Exception:
                pass
        score = float(row.get("nonphoto_score", 0.0) or 0.0)
        draw.text((x + 3, y + thumb_h + 2), f"{row['image_id']}\n{score:.2f}"[:54], fill="black", font=font)
    canvas.save(path, quality=90)


def write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize exact 10K SSTK photo-primary Phase A pool.")
    parser.add_argument("--audited_parquet", required=True)
    parser.add_argument("--input_image_dir", required=True)
    parser.add_argument("--output_parquet", required=True)
    parser.add_argument("--output_image_dir", required=True)
    parser.add_argument("--summary_json", required=True)
    parser.add_argument("--target_size", type=int, default=10000)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--copy_mode", choices=["copy", "hardlink", "symlink"], default="copy")
    parser.add_argument("--contact_sheet", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audited_path = Path(args.audited_parquet)
    input_image_dir = Path(args.input_image_dir)
    output_parquet = Path(args.output_parquet)
    output_image_dir = Path(args.output_image_dir)
    summary_json = Path(args.summary_json)

    df = pd.read_parquet(audited_path)
    selected = select_photo_primary_pool(
        df,
        target_size=int(args.target_size),
        random_state=int(args.random_state),
    )

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(output_parquet, index=False)
    export_summary = export_selected_images(
        selected,
        input_image_dir=input_image_dir,
        output_image_dir=output_image_dir,
        mode=str(args.copy_mode),
    )
    if str(args.contact_sheet).strip():
        write_contact_sheet(selected, output_image_dir, Path(args.contact_sheet), max_images=80)

    summary = {
        "audited_parquet": str(audited_path),
        "input_rows": int(len(df)),
        "target_size": int(args.target_size),
        "photo_primary_available": int((df["media_decision"].astype(str) == "keep_photo_primary").sum()),
        "selected_rows": int(len(selected)),
        "output_parquet": str(output_parquet),
        "output_image_dir": str(output_image_dir),
        "selection_complete": bool(len(selected) == int(args.target_size)),
        "media_decision_counts_selected": {
            str(k): int(v) for k, v in selected["media_decision"].value_counts(dropna=False).to_dict().items()
        },
        "category_counts_selected": {
            str(k): int(v) for k, v in selected.get("super_cat", pd.Series(dtype=str)).value_counts(dropna=False).to_dict().items()
        },
        "export": export_summary,
    }
    for col in ["curation_policy_version", "prompt_classifier_model"]:
        if col in selected.columns:
            summary[f"{col}_values"] = {
                str(k): int(v) for k, v in selected[col].value_counts(dropna=False).head(20).to_dict().items()
            }
    write_summary(summary_json, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if not summary["selection_complete"] or export_summary["missing"] or export_summary["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
