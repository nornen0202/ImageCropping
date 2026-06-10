import os
import sys
from pathlib import Path

# Ensure src/ is on the path so tests work without pip install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Change to project root so relative paths like "datasets.yaml" work in tests
os.chdir(Path(__file__).resolve().parents[1])
