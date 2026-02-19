from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class Registry:
    def __init__(self, root: Path):
        self.root = root
        self.path = root / "registry.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save(self, data: dict[str, Any]) -> None:
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    def get(self, name: str) -> dict[str, Any] | None:
        return self._load().get(name)

    def set(self, name: str, payload: dict[str, Any]) -> None:
        data = self._load()
        payload = dict(payload)
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        data[name] = payload
        self._save(data)

    def all(self) -> dict[str, Any]:
        return self._load()
