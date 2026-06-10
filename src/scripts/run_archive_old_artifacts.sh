#!/bin/bash
# ==============================================================================
# run_archive_old_artifacts.sh
# Archive old dataset artifacts into sibling *_olds directories.
# ------------------------------------------------------------------------------
# Examples:
#   src/scripts/run_archive_old_artifacts.sh --cutoff 260330_r0 data/GAIC/All
#   src/scripts/run_archive_old_artifacts.sh --cutoff 260330_r0 --dry_run data/GAIC/All
#   src/scripts/run_archive_old_artifacts.sh --cutoff 260330_r0 --skip-existing data/GAIC/All
#   src/scripts/run_archive_old_artifacts.sh --cutoff 260330_r0 --overwrite data/GAIC/All
#   src/scripts/run_archive_old_artifacts.sh --cutoff 260319_cond_detr_v1 data/SSTK/Test_100
# ==============================================================================

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DRY_RUN=0
VERBOSE=0
CUTOFF=""
SKIP_EXISTING=0
OVERWRITE=0
SCOPES=()
TARGETS=()

usage() {
  sed -n '1,220p' "$0"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --cutoff)
      CUTOFF="$2"
      shift 2
      ;;
    --scope)
      SCOPES+=("$2")
      shift 2
      ;;
    --dry-run|--dry_run)
      DRY_RUN=1
      shift 1
      ;;
    --verbose)
      VERBOSE=1
      shift 1
      ;;
    --skip-existing)
      SKIP_EXISTING=1
      shift 1
      ;;
    --overwrite)
      OVERWRITE=1
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      TARGETS+=("$1")
      shift 1
      ;;
  esac
done

if [ -z "$CUTOFF" ]; then
  echo "[error] --cutoff is required"
  usage
  exit 1
fi

if [ "$SKIP_EXISTING" -eq 1 ] && [ "$OVERWRITE" -eq 1 ]; then
  echo "[error] --skip-existing and --overwrite are mutually exclusive"
  usage
  exit 1
fi

if [ "${#TARGETS[@]}" -eq 0 ]; then
  echo "[error] at least one target path is required"
  usage
  exit 1
fi

CMD=("$PYTHON_BIN" "src/scripts/archive_old_artifacts.py" "--cutoff" "$CUTOFF")

if [ "$DRY_RUN" -eq 1 ]; then
  CMD+=("--dry-run")
fi
if [ "$VERBOSE" -eq 1 ]; then
  CMD+=("--verbose")
fi
if [ "$SKIP_EXISTING" -eq 1 ]; then
  CMD+=("--skip-existing")
fi
if [ "$OVERWRITE" -eq 1 ]; then
  CMD+=("--overwrite")
fi
for scope in "${SCOPES[@]}"; do
  CMD+=("--scope" "$scope")
done
for target in "${TARGETS[@]}"; do
  CMD+=("$target")
done

echo "[archive] cutoff=$CUTOFF dry_run=$DRY_RUN verbose=$VERBOSE"
echo "[archive] targets=${TARGETS[*]}"
if [ "$SKIP_EXISTING" -eq 1 ]; then
  echo "[archive] conflict_mode=skip-existing"
elif [ "$OVERWRITE" -eq 1 ]; then
  echo "[archive] conflict_mode=overwrite"
fi
if [ "${#SCOPES[@]}" -gt 0 ]; then
  echo "[archive] scopes=${SCOPES[*]}"
fi

"${CMD[@]}"
