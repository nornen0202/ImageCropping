from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DatasetInfo:
    name: str
    description: str
    download_type: str
    source_url: str
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    expected_structure: dict[str, Any] = field(default_factory=dict)
    license_url: str = ""
    citation: str = ""
    size_estimate: str = "unknown"
    notes: str = ""
    file_id: str | None = None


class Manifest:
    def __init__(self, path: Path):
        self.path = path
        self._datasets = self._load(path)

    @staticmethod
    def _load(path: Path) -> dict[str, DatasetInfo]:
        text = path.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise RuntimeError("Manifest is not JSON and PyYAML is unavailable.") from exc
            data = yaml.safe_load(text)

        entries = data.get("datasets", [])
        return {entry["name"]: DatasetInfo(**entry) for entry in entries}

    def names(self) -> list[str]:
        return sorted(self._datasets.keys())

    def get(self, name: str) -> DatasetInfo:
        try:
            return self._datasets[name]
        except KeyError as exc:
            raise KeyError(f"Unknown dataset '{name}'. Use 'list' to see supported datasets.") from exc

    def all(self) -> list[DatasetInfo]:
        return [self._datasets[name] for name in self.names()]
