#!/bin/bash
set -euo pipefail

VENV_PATH="/media/jyju25/Disk_JY/Projects_26/Venvs/ImageCropping_Py310/bin/activate"
if [ -f "$VENV_PATH" ]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH"
fi

python src/scripts/setup_public_cropping_teachers.py "$@"

