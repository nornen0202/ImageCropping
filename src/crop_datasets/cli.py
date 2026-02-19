from __future__ import annotations

import argparse
import json
from pathlib import Path

from crop_datasets.datasets import manager_for
from crop_datasets.download.extract import extract_archive
from crop_datasets.download.gdrive import download_gdrive
from crop_datasets.download.http import download_http
from crop_datasets.manifest import Manifest
from crop_datasets.registry import Registry


def default_manifest_path() -> Path:
    return Path(__file__).resolve().parents[2] / "datasets.yaml"


def get_paths(args: argparse.Namespace) -> tuple[Path, Registry, Manifest]:
    root = Path(args.root).expanduser().resolve()
    registry = Registry(root)
    manifest = Manifest(Path(args.manifest).resolve())
    return root, registry, manifest


def cmd_list(args: argparse.Namespace) -> None:
    root, registry, manifest = get_paths(args)
    records = registry.all()
    print(f"Data root: {root}")
    for item in manifest.all():
        status = "registered" if item.name in records else "not-installed"
        print(
            f"- {item.name:9} [{item.download_type:6}] {status:13} size={item.size_estimate}\n"
            f"  license: {item.license_url}\n"
            f"  source:  {item.source_url}"
        )


def _download_dataset(name: str, ds, dataset_root: Path) -> dict:
    install_path = dataset_root / name
    download_dir = install_path / "downloads"
    extracted_dir = install_path / "extracted"
    install_path.mkdir(parents=True, exist_ok=True)

    downloaded: list[str] = []
    if ds.download_type == "http":
        for artifact in ds.artifacts:
            dest = download_dir / artifact["filename"]
            download_http(artifact["url"], dest)
            downloaded.append(str(dest))
            if dest.suffix in {".tar", ".zip", ".gz", ".tgz"}:
                extract_archive(dest, extracted_dir)
    elif ds.download_type == "gdrive":
        filename = ds.artifacts[0]["filename"] if ds.artifacts else f"{name}.download"
        dest = download_dir / filename
        download_gdrive(ds.file_id, dest)
        downloaded.append(str(dest))
        if dest.suffix in {".tar", ".zip", ".gz", ".tgz"}:
            extract_archive(dest, extracted_dir)
    else:
        raise RuntimeError(
            f"Dataset '{name}' requires manual/script-based setup ({ds.download_type}). "
            f"Use register with a local path. Source: {ds.source_url}"
        )

    return {
        "name": name,
        "path": str(install_path),
        "downloaded_files": downloaded,
        "download_type": ds.download_type,
        "source_url": ds.source_url,
    }


def cmd_download(args: argparse.Namespace) -> None:
    root, registry, manifest = get_paths(args)
    ds = manifest.get(args.name)
    try:
        payload = _download_dataset(args.name, ds, root)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download '{args.name}': {exc}.\n"
            f"Next steps: verify network/link permissions, or download manually from {ds.source_url} and run register."
        ) from exc

    manager = manager_for(ds)
    ok, missing = manager.verify_structure(Path(payload["path"]) / "extracted")
    payload["verify_ok"] = ok
    payload["verify_missing"] = missing
    registry.set(args.name, payload)
    print(json.dumps(payload, indent=2))


def cmd_register(args: argparse.Namespace) -> None:
    _, registry, manifest = get_paths(args)
    ds = manifest.get(args.name)
    path = Path(args.path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    manager = manager_for(ds)
    verify_target = path if path.is_dir() else path.parent
    ok, missing = manager.verify_structure(verify_target)
    if not ok:
        raise RuntimeError(
            f"Structure verification failed for {args.name}. Missing: {missing}. "
            "Please check dataset layout or README guidance."
        )

    payload = {
        "name": args.name,
        "path": str(path),
        "download_type": "manual",
        "source_url": ds.source_url,
        "verify_ok": ok,
    }
    registry.set(args.name, payload)
    print(f"Registered {args.name} -> {path}")


def cmd_verify(args: argparse.Namespace) -> None:
    _, registry, manifest = get_paths(args)
    ds = manifest.get(args.name)
    rec = registry.get(args.name)
    if not rec:
        raise RuntimeError(f"Dataset '{args.name}' is not registered. Use download/register first.")

    path = Path(rec["path"])
    target = path / "extracted" if (path / "extracted").exists() else path
    manager = manager_for(ds)
    ok, missing = manager.verify_structure(target)
    if ok:
        print(f"{args.name}: OK ({target})")
    else:
        raise RuntimeError(f"{args.name}: FAILED, missing {missing} in {target}")


def cmd_prepare_unsplash(args: argparse.Namespace) -> None:
    root, registry, manifest = get_paths(args)
    ds = manifest.get("unsplash")
    unsplash_root = Path(args.unsplash_root).expanduser().resolve()
    manager = manager_for(ds)
    ok, missing = manager.verify_structure(unsplash_root)
    if not ok:
        raise RuntimeError(
            f"Unsplash root validation failed. Missing {missing}. "
            "Provide an approved Unsplash dataset export root."
        )

    out_dir = root / "unsplash"
    out_dir.mkdir(parents=True, exist_ok=True)
    photos_tsv = unsplash_root / "photos.tsv"
    photos_parquet = unsplash_root / "photos.parquet"
    source_file = photos_tsv if photos_tsv.exists() else photos_parquet
    index_path = out_dir / "index.jsonl"

    count = 0
    with source_file.open("r", encoding="utf-8", errors="ignore") as src, index_path.open("w", encoding="utf-8") as dst:
        for idx, line in enumerate(src):
            if idx == 0 and "\t" in line:
                continue
            dst.write(json.dumps({"row": idx, "raw": line.strip()}) + "\n")
            count += 1
            if count >= args.max_rows:
                break

    payload = {
        "name": "unsplash",
        "path": str(unsplash_root),
        "prepared_index": str(index_path),
        "indexed_rows": count,
        "verify_ok": True,
    }
    registry.set("unsplash", payload)
    print(json.dumps(payload, indent=2))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="crop-datasets")
    p.add_argument("--root", default="./data", help="Dataset storage root (default: ./data)")
    p.add_argument("--manifest", default=str(default_manifest_path()), help="Path to datasets manifest")

    sub = p.add_subparsers(dest="cmd", required=True)

    s_list = sub.add_parser("list", help="List supported datasets and status")
    s_list.set_defaults(func=cmd_list)

    s_dl = sub.add_parser("download", help="Download supported auto-download datasets")
    s_dl.add_argument("name")
    s_dl.set_defaults(func=cmd_download)

    s_reg = sub.add_parser("register", help="Register locally prepared dataset path")
    s_reg.add_argument("name")
    s_reg.add_argument("--path", required=True)
    s_reg.set_defaults(func=cmd_register)

    s_v = sub.add_parser("verify", help="Verify registered dataset structure")
    s_v.add_argument("name")
    s_v.set_defaults(func=cmd_verify)

    s_p = sub.add_parser("prepare", help="Prepare dataset-specific indexes")
    prep_sub = s_p.add_subparsers(dest="prepare_cmd", required=True)
    s_u = prep_sub.add_parser("unsplash", help="Prepare local Unsplash export")
    s_u.add_argument("--unsplash_root", required=True)
    s_u.add_argument("--max-rows", type=int, default=50000)
    s_u.set_defaults(func=cmd_prepare_unsplash)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
