from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def _read_ids(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def _row_id(row: Dict[str, Any], key: str) -> str:
    return str(row.get(key) or row.get("source_image_id") or row.get("id") or "")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter JSONL rows by image id list.")
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--image_ids", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--key", default="image_id")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    ids = _read_ids(Path(args.image_ids))
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    read_count = 0
    written_count = 0
    with Path(args.input_jsonl).open(encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            read_count += 1
            row = json.loads(line)
            if _row_id(row, str(args.key)) in ids:
                dst.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                written_count += 1
    summary = {
        "input_jsonl": str(args.input_jsonl),
        "image_ids": str(args.image_ids),
        "output_jsonl": str(output_path),
        "requested_id_count": len(ids),
        "read_count": read_count,
        "written_count": written_count,
        "missing_count": max(0, len(ids) - written_count),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
