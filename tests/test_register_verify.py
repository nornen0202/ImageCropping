from pathlib import Path

from crop_datasets.datasets.base import DatasetManager
from crop_datasets.manifest import DatasetInfo


def test_register_structure_validation():
    info = DatasetInfo(
        name="dummy",
        description="",
        download_type="manual",
        source_url="",
        expected_structure={"required_paths": ["a", "b"]},
    )
    manager = DatasetManager(info)
    ok, missing = manager.verify_structure(Path("."))
    assert not ok
    assert "a" in missing and "b" in missing
