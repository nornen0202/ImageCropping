from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path


def extract_archive(archive: Path, dest_dir: Path) -> Path:
    marker = dest_dir / f".{archive.name}.extracted"
    if marker.exists():
        return dest_dir

    dest_dir.mkdir(parents=True, exist_ok=True)
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as tf:
            tf.extractall(dest_dir)
    elif zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(dest_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive}")

    marker.write_text("ok", encoding="utf-8")
    return dest_dir
