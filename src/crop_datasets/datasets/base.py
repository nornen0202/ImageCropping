from __future__ import annotations

import hashlib
from pathlib import Path

from crop_datasets.manifest import DatasetInfo


class DatasetManager:
    def __init__(self, info: DatasetInfo):
        self.info = info

    def verify_structure(self, path: Path) -> tuple[bool, list[str]]:
        missing: list[str] = []
        expected = self.info.expected_structure or {}
        for rel in expected.get("required_paths", []):
            if not (path / rel).exists():
                missing.append(rel)

        any_of_groups = expected.get("any_of", [])
        for group in any_of_groups:
            if not any((path / entry).exists() for entry in group):
                missing.append(f"one of: {group}")

        return (len(missing) == 0, missing)

    @staticmethod
    def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
