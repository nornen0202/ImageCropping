#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scripts.migrate_label_artifact_layout import replace_text_paths


def iter_patchable_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    skip_names = {
        ".git",
        ".idea",
        "__pycache__",
        "images",
        "image",
        "imgs",
        "label_json",
        "coco",
        "crops",
        "overlays",
        "per_image",
        "debug_visualizations",
        "debug_visualizations_balanced50_bottomneg",
        "weights",
        ".venv",
        "venv",
        "node_modules",
    }
    out: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        if any(part in skip_names for part in path.parts):
            continue
        out.append(path)
    return sorted(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch Markdown/text documentation references from legacy coco/ label paths to label_json/.")
    parser.add_argument("--root", action="append", required=True)
    parser.add_argument("--dry_run", type=int, default=0)
    parser.add_argument("--out_report", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dry_run = bool(int(args.dry_run))
    updated: list[str] = []
    scanned = 0
    for root_text in args.root:
        root = Path(root_text)
        for path in iter_patchable_files(root):
            scanned += 1
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            patched = replace_text_paths(text)
            if patched == text:
                continue
            updated.append(str(path))
            if not dry_run:
                path.write_text(patched, encoding="utf-8")
    payload = {
        "schema_version": "label_artifact_doc_path_patch_v1",
        "dry_run": dry_run,
        "scanned_count": scanned,
        "updated_count": len(updated),
        "updated_paths": updated,
    }
    if args.out_report:
        out = Path(args.out_report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
