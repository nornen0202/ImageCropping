from __future__ import annotations

from pathlib import Path


def gdrive_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?id={file_id}"


def download_gdrive(file_id: str, dest: Path) -> Path:
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive download requires gdown. Install via `pip install crop_datasets[gdrive]`."
        ) from exc

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = gdrive_url(file_id)
    out = gdown.download(url, str(dest), quiet=False, resume=True, fuzzy=True)
    if not out:
        raise RuntimeError(
            "Google Drive download failed. Check permission/link validity. "
            "If this is access-controlled, download manually and run register."
        )
    return dest
