#!/usr/bin/env python3
from __future__ import annotations

import argparse
from fnmatch import fnmatch
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Iterator, List, Literal, Sequence


VERSION_RE = re.compile(r"(?<!\d)(\d{6})(?:_[A-Za-z0-9]+)*?_(r|v)(\d+)(?!\d)")


@dataclass(frozen=True, order=True)
class RunVersion:
    date_code: int
    ordinal: int
    raw: str


@dataclass(frozen=True)
class MovePlan:
    source: Path
    destination: Path
    reason: str


ConflictMode = Literal["fail", "skip", "overwrite"]

DEFAULT_REUSABLE_PREFIXES = (
    "artifacts/metadata",
    "artifacts/public_teachers",
)

GAIC_ALL_REUSABLE_GLOBS = (
    "artifacts/candidates/candidates_ar_gaic_260320_r0*",
    "artifacts/reports/gaic_benchmark_eval_gaic_260320_r0*",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Archive old artifact outputs into a sibling *_olds directory while preserving "
            "the dataset-root-relative structure."
        )
    )
    parser.add_argument(
        "targets",
        nargs="+",
        help=(
            "Dataset roots such as data/GAIC/All, or parent directories such as data/GAIC. "
            "Parent directories are expanded to immediate child dataset roots."
        ),
    )
    parser.add_argument(
        "--cutoff",
        required=True,
        help="Archive versioned outputs older than this run token, for example 260330_r0.",
    )
    parser.add_argument(
        "--scope",
        action="append",
        default=None,
        help="Subdirectories inside each dataset root to scan. Defaults to artifacts and logs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned moves without changing the filesystem.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every move instead of just the summary.",
    )
    conflict_group = parser.add_mutually_exclusive_group()
    conflict_group.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a move when the archive destination already exists.",
    )
    conflict_group.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing archive destination before moving the source.",
    )
    return parser.parse_args()


def parse_run_version(text: str) -> RunVersion | None:
    match = VERSION_RE.search(text)
    if not match:
        return None
    return RunVersion(
        date_code=int(match.group(1)),
        ordinal=int(match.group(3)),
        raw=match.group(0),
    )


def require_cutoff_version(text: str) -> RunVersion:
    version = parse_run_version(text)
    if version is None:
        raise SystemExit(f"invalid --cutoff value: {text!r}")
    return version


def default_scopes() -> List[str]:
    return ["artifacts", "logs"]


def looks_like_dataset_root(path: Path) -> bool:
    return any((path / name).exists() for name in default_scopes())


def iter_dataset_roots(target: Path) -> Iterator[Path]:
    if looks_like_dataset_root(target):
        yield target
        return

    if not target.exists():
        raise FileNotFoundError(f"target does not exist: {target}")
    if not target.is_dir():
        raise NotADirectoryError(f"target is not a directory: {target}")

    for child in sorted(target.iterdir()):
        if not child.is_dir():
            continue
        if child.name.endswith("_olds"):
            continue
        if looks_like_dataset_root(child):
            yield child


def archive_root_for(dataset_root: Path) -> Path:
    return dataset_root.parent / f"{dataset_root.name}_olds"


def reusable_prefixes_for(dataset_root: Path) -> List[str]:
    _ = dataset_root
    return list(DEFAULT_REUSABLE_PREFIXES)


def reusable_globs_for(dataset_root: Path) -> List[str]:
    globs: List[str] = []
    if tuple(dataset_root.parts[-2:]) == ("GAIC", "All"):
        globs.extend(GAIC_ALL_REUSABLE_GLOBS)
    return globs


def is_reusable_artifact(relative_path: Path, dataset_root: Path) -> bool:
    relative_posix = relative_path.as_posix()
    for prefix in reusable_prefixes_for(dataset_root):
        if relative_posix == prefix or relative_posix.startswith(f"{prefix}/"):
            return True
    return any(fnmatch(relative_posix, pattern) for pattern in reusable_globs_for(dataset_root))


def is_old_version(name: str, cutoff: RunVersion) -> bool:
    version = parse_run_version(name)
    return version is not None and version < cutoff


def strip_legacy_olds(relative_path: Path) -> Path:
    parts = [part for part in relative_path.parts if part != "olds"]
    return Path(*parts) if parts else Path()


def plan_moves_for_scope(
    dataset_root: Path,
    scope_root: Path,
    archive_root: Path,
    cutoff: RunVersion,
) -> List[MovePlan]:
    plans: List[MovePlan] = []

    def plan_legacy_olds(directory: Path) -> None:
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            relative = child.relative_to(dataset_root)
            normalized_relative = strip_legacy_olds(relative)
            plans.append(
                MovePlan(
                    source=child,
                    destination=archive_root / normalized_relative,
                    reason="legacy olds",
                )
            )

    def visit(directory: Path) -> None:
        for child in sorted(directory.iterdir(), key=lambda item: item.name):
            if child.name in {".DS_Store", "Thumbs.db"}:
                continue

            relative = child.relative_to(dataset_root)
            if is_reusable_artifact(relative, dataset_root):
                continue
            if child.is_dir() and child.name == "olds":
                plan_legacy_olds(child)
                continue

            if is_old_version(child.name, cutoff):
                plans.append(
                    MovePlan(
                        source=child,
                        destination=archive_root / relative,
                        reason=f"older than {cutoff.raw}",
                    )
                )
                continue

            if child.is_dir():
                visit(child)

    visit(scope_root)
    return plans


def dedupe_plans(plans: Iterable[MovePlan]) -> List[MovePlan]:
    seen_sources: set[Path] = set()
    deduped: List[MovePlan] = []
    for plan in sorted(plans, key=lambda item: (len(item.source.parts), str(item.source))):
        if plan.source in seen_sources:
            continue
        if any(parent in seen_sources for parent in plan.source.parents):
            continue
        deduped.append(plan)
        seen_sources.add(plan.source)
    return sorted(deduped, key=lambda item: str(item.source))


def dedupe_paths(paths: Iterable[Path]) -> List[Path]:
    deduped: List[Path] = []
    for path in sorted(set(paths), key=lambda item: (len(item.parts), str(item))):
        if any(parent in deduped for parent in path.parents):
            continue
        deduped.append(path)
    return deduped


def plan_moves(
    dataset_root: Path,
    scopes: Sequence[str],
    cutoff: RunVersion,
) -> List[MovePlan]:
    archive_root = archive_root_for(dataset_root)
    plans: List[MovePlan] = []
    for scope in scopes:
        scope_root = dataset_root / scope
        if not scope_root.exists():
            continue
        plans.extend(plan_moves_for_scope(dataset_root, scope_root, archive_root, cutoff))
    return dedupe_plans(plans)


def conflict_mode_from_args(args: argparse.Namespace) -> ConflictMode:
    if args.skip_existing:
        return "skip"
    if args.overwrite:
        return "overwrite"
    return "fail"


def remove_existing_destination(destination: Path) -> None:
    if destination.is_dir() and not destination.is_symlink():
        shutil.rmtree(destination)
        return
    destination.unlink()


def resolve_destination(destination: Path, conflict_mode: ConflictMode) -> str:
    if destination.exists():
        if conflict_mode == "skip":
            return "skipped_existing"
        if conflict_mode == "overwrite":
            remove_existing_destination(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            return "overwrote_existing"
        raise FileExistsError(f"destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return "move"


def protected_restore_sources(dataset_root: Path) -> List[Path]:
    archive_root = archive_root_for(dataset_root)
    matches: List[Path] = []
    for prefix in reusable_prefixes_for(dataset_root):
        source = archive_root / prefix
        if source.exists():
            matches.append(source)
    for pattern in reusable_globs_for(dataset_root):
        matches.extend(archive_root.glob(pattern))
    return dedupe_paths(matches)


def copy_missing_entry(source: Path, destination: Path, dry_run: bool) -> int:
    if source.is_dir() and not source.is_symlink():
        copied_files = 0
        if not dry_run:
            destination.mkdir(parents=True, exist_ok=True)
        for child in sorted(source.iterdir(), key=lambda item: item.name):
            copied_files += copy_missing_entry(child, destination / child.name, dry_run)
        return copied_files

    if destination.exists():
        return 0
    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return 1


def restore_reusable_artifacts(dataset_root: Path, dry_run: bool) -> int:
    archive_root = archive_root_for(dataset_root)
    restored_files = 0
    for source in protected_restore_sources(dataset_root):
        relative = source.relative_to(archive_root)
        restored_files += copy_missing_entry(source, dataset_root / relative, dry_run)
    return restored_files


def prune_empty_dirs(start: Path, stop: Path) -> None:
    current = start
    while current != stop and current.exists():
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def apply_plan(
    plan: MovePlan,
    dataset_root: Path,
    dry_run: bool,
    conflict_mode: ConflictMode,
) -> str:
    if dry_run:
        return "dry_run"
    result = resolve_destination(plan.destination, conflict_mode)
    if result == "skipped_existing":
        return result
    shutil.move(str(plan.source), str(plan.destination))
    prune_empty_dirs(plan.source.parent, dataset_root.parent)
    return result


def render_plan(plan: MovePlan, dataset_root: Path) -> str:
    source = plan.source.relative_to(dataset_root)
    destination = plan.destination.relative_to(dataset_root.parent)
    return f"{source} -> {destination} [{plan.reason}]"


def main() -> int:
    args = parse_args()
    cutoff = require_cutoff_version(args.cutoff)
    conflict_mode = conflict_mode_from_args(args)
    scopes = args.scope or default_scopes()

    targets = [Path(target).resolve() for target in args.targets]
    dataset_roots: List[Path] = []
    for target in targets:
        dataset_roots.extend(iter_dataset_roots(target))

    if not dataset_roots:
        print("no dataset roots found", file=sys.stderr)
        return 1

    total_moves = 0
    total_restored_reusable_files = 0
    total_skipped_existing = 0
    total_overwritten_existing = 0
    for dataset_root in dataset_roots:
        restored_reusable_files = restore_reusable_artifacts(dataset_root, args.dry_run)
        plans = plan_moves(dataset_root, scopes, cutoff)
        archive_root = archive_root_for(dataset_root)
        print(f"[dataset] {dataset_root}")
        print(f"archive_root={archive_root}")
        if restored_reusable_files:
            print(f"restored_reusable_files={restored_reusable_files}")
        print(f"planned_moves={len(plans)}")
        if args.verbose or args.dry_run:
            for plan in plans:
                print(f"  {render_plan(plan, dataset_root)}")
        dataset_skipped_existing = 0
        dataset_overwritten_existing = 0
        for plan in plans:
            result = apply_plan(plan, dataset_root, args.dry_run, conflict_mode)
            if result == "skipped_existing":
                dataset_skipped_existing += 1
                if args.verbose:
                    print(f"  skip existing: {render_plan(plan, dataset_root)}")
            elif result == "overwrote_existing":
                dataset_overwritten_existing += 1
        total_moves += len(plans)
        total_restored_reusable_files += restored_reusable_files
        total_skipped_existing += dataset_skipped_existing
        total_overwritten_existing += dataset_overwritten_existing
        if dataset_skipped_existing:
            print(f"skipped_existing={dataset_skipped_existing}")
        if dataset_overwritten_existing:
            print(f"overwritten_existing={dataset_overwritten_existing}")

    print(f"total_moves={total_moves}")
    if total_restored_reusable_files:
        print(f"total_restored_reusable_files={total_restored_reusable_files}")
    if total_skipped_existing:
        print(f"total_skipped_existing={total_skipped_existing}")
    if total_overwritten_existing:
        print(f"total_overwritten_existing={total_overwritten_existing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
