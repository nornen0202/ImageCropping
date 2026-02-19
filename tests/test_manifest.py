from pathlib import Path

from crop_datasets.manifest import Manifest


def test_manifest_parsing():
    manifest = Manifest(Path("datasets.yaml"))
    names = manifest.names()
    assert "cpc" in names
    assert "flms" in names
    assert manifest.get("cpc").download_type == "gdrive"
