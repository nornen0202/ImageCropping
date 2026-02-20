from __future__ import annotations

import argparse
import json
import sys
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


def _is_archive(p: Path) -> bool:
    """Return True if the path looks like a downloadable archive."""
    compound = "".join(p.suffixes[-2:]).lower()
    if compound in {".tar.gz", ".tar.bz2", ".tar.xz", ".tar.zst"}:
        return True
    return p.suffix.lower() in {".tar", ".zip", ".tgz", ".gz"}


def cmd_list(args: argparse.Namespace) -> None:
    root, registry, manifest = get_paths(args)
    records = registry.all()
    print(f"Data root : {root}")
    print(f"Manifest  : {args.manifest}")
    print()
    header = f"{'NAME':<10} {'TYPE':<8} {'STATUS':<14} {'SIZE':<18} {'LICENSE'}"
    print(header)
    print("-" * len(header))
    for item in manifest.all():
        status = "registered" if item.name in records else "not-installed"
        size = item.size_estimate or "unknown"
        print(f"{item.name:<10} {item.download_type:<8} {status:<14} {size:<18}  {item.license_url}")
        print(f"           source : {item.source_url}")
        if item.notes:
            print(f"           notes : {item.notes}")
    print()


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
            if _is_archive(dest):
                extract_archive(dest, extracted_dir)
    elif ds.download_type == "gdrive":
        filename = ds.artifacts[0]["filename"] if ds.artifacts else f"{name}.download"
        dest = download_dir / filename
        download_gdrive(ds.file_id, dest)
        downloaded.append(str(dest))
        if _is_archive(dest):
            extract_archive(dest, extracted_dir)
    else:
        raise RuntimeError(
            f"Dataset '{name}' requires manual/script-based setup ({ds.download_type}).\n"
            f"  → Use 'crop-datasets register {name} --path <local_path>' after downloading.\n"
            f"  → Source: {ds.source_url}"
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
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download '{args.name}': {exc}\n"
            f"  → Check network/link permissions, or download manually from:\n"
            f"    {ds.source_url}\n"
            f"  → Then register with: crop-datasets register {args.name} --path <local_path>"
        ) from exc

    manager = manager_for(ds)
    extracted = Path(payload["path"]) / "extracted"
    verify_target = extracted if extracted.exists() else Path(payload["path"])
    ok, missing = manager.verify_structure(verify_target)
    payload["verify_ok"] = ok
    payload["verify_missing"] = missing
    registry.set(args.name, payload)
    print(json.dumps(payload, indent=2))
    if not ok:
        print(
            f"\n[WARNING] Structure check found missing items: {missing}\n"
            "  → Extraction may have a different directory layout than expected.\n"
            "  → The dataset is still registered; run 'verify' to recheck.",
            file=sys.stderr,
        )


def cmd_register(args: argparse.Namespace) -> None:
    _, registry, manifest = get_paths(args)
    ds = manifest.get(args.name)
    path = Path(args.path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Path does not exist: {path}\n"
            f"  → Download the dataset first from: {ds.source_url}\n"
            f"  → Then re-run: crop-datasets register {args.name} --path <local_path>"
        )

    manager = manager_for(ds)
    verify_target = path if path.is_dir() else path.parent
    ok, missing = manager.verify_structure(verify_target)
    if not ok:
        raise RuntimeError(
            f"Structure verification failed for '{args.name}'.\n"
            f"  Missing: {missing}\n"
            "  → Check dataset layout or consult README for expected structure."
        )

    payload = {
        "name": args.name,
        "path": str(path),
        "download_type": "manual",
        "source_url": ds.source_url,
        "verify_ok": ok,
    }
    registry.set(args.name, payload)
    print(f"Registered '{args.name}' → {path}")


def cmd_verify(args: argparse.Namespace) -> None:
    _, registry, manifest = get_paths(args)
    ds = manifest.get(args.name)
    rec = registry.get(args.name)
    if not rec:
        raise RuntimeError(
            f"Dataset '{args.name}' is not registered.\n"
            f"  → Run 'crop-datasets download {args.name}' or "
            f"'crop-datasets register {args.name} --path <local_path>' first."
        )

    path = Path(rec["path"])
    extracted = path / "extracted"
    target = extracted if extracted.exists() else path
    manager = manager_for(ds)
    ok, missing = manager.verify_structure(target)
    if ok:
        print(f"{args.name}: OK ({target})")
    else:
        raise RuntimeError(
            f"{args.name}: FAILED\n"
            f"  Missing in {target}: {missing}"
        )


def cmd_prepare_unsplash(args: argparse.Namespace) -> None:
    root, registry, manifest = get_paths(args)
    ds = manifest.get("unsplash")
    unsplash_root = Path(args.unsplash_root).expanduser().resolve()
    manager = manager_for(ds)
    ok, missing = manager.verify_structure(unsplash_root)
    if not ok:
        raise RuntimeError(
            f"Unsplash root validation failed.\n"
            f"  Missing: {missing}\n"
            "  → Provide an approved Unsplash dataset export root (photos.tsv or photos.parquet required)."
        )

    out_dir = root / "unsplash"
    out_dir.mkdir(parents=True, exist_ok=True)
    photos_tsv = unsplash_root / "photos.tsv"
    photos_parquet = unsplash_root / "photos.parquet"
    source_file = photos_tsv if photos_tsv.exists() else photos_parquet
    index_path = out_dir / "index.jsonl"

    count = 0
    with (
        source_file.open("r", encoding="utf-8", errors="ignore") as src,
        index_path.open("w", encoding="utf-8") as dst,
    ):
        for idx, line in enumerate(src):
            if idx == 0 and "\t" in line:
                continue  # skip TSV header
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
    # Common arguments shared by all subcommands
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--root",
        default="./data",
        help="Dataset storage root directory (default: ./data)",
    )
    common.add_argument(
        "--manifest",
        default=str(default_manifest_path()),
        help="Path to datasets manifest YAML/JSON",
    )

    p = argparse.ArgumentParser(
        prog="crop-datasets",
        description="Unified dataset manager for image cropping/composition research.",
    )
    # Also add to the top-level parser so `crop-datasets --root X list` works too
    p.add_argument("--root", default="./data", help="Dataset storage root (default: ./data)")
    p.add_argument("--manifest", default=str(default_manifest_path()), help="Path to datasets manifest")

    sub = p.add_subparsers(dest="cmd", required=True)

    # list
    s_list = sub.add_parser("list", parents=[common], help="List supported datasets and their status")
    s_list.set_defaults(func=cmd_list)

    # download <name>
    s_dl = sub.add_parser("download", parents=[common], help="Download an auto-downloadable dataset")
    s_dl.add_argument("name", help="Dataset name (e.g. cpc, xpview, flms)")
    s_dl.set_defaults(func=cmd_download)

    # register <name> --path <path>
    s_reg = sub.add_parser("register", parents=[common], help="Register a locally prepared dataset")
    s_reg.add_argument("name", help="Dataset name")
    s_reg.add_argument("--path", required=True, help="Local path to the dataset root or archive")
    s_reg.set_defaults(func=cmd_register)

    # verify <name>
    s_v = sub.add_parser("verify", parents=[common], help="Verify registered dataset structure")
    s_v.add_argument("name", help="Dataset name")
    s_v.set_defaults(func=cmd_verify)

    # prepare <subcmd>
    s_p = sub.add_parser("prepare", help="Dataset-specific preparation commands")
    prep_sub = s_p.add_subparsers(dest="prepare_cmd", required=True)
    s_u = prep_sub.add_parser("unsplash", parents=[common], help="Index a local Unsplash export")
    s_u.add_argument("--unsplash_root", required=True, help="Path to the Unsplash dataset export directory")
    s_u.add_argument("--max-rows", type=int, default=50000, dest="max_rows", help="Max rows to index")
    s_u.set_defaults(func=cmd_prepare_unsplash)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except (FileNotFoundError, KeyError, RuntimeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
