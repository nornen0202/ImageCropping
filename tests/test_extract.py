import tarfile
import zipfile
from pathlib import Path

from crop_datasets.download.extract import extract_archive


def test_extract_tar_idempotent(tmp_path: Path):
    src = tmp_path / "hello.txt"
    src.write_text("hello", encoding="utf-8")
    tar_path = tmp_path / "sample.tar"
    with tarfile.open(tar_path, "w") as tf:
        tf.add(src, arcname="hello.txt")

    out = tmp_path / "out"
    extract_archive(tar_path, out)
    assert (out / "hello.txt").exists()
    extract_archive(tar_path, out)
    assert (out / ".sample.tar.extracted").exists()


def test_extract_zip_idempotent(tmp_path: Path):
    zpath = tmp_path / "sample.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("a.txt", "a")
    out = tmp_path / "out_zip"
    extract_archive(zpath, out)
    extract_archive(zpath, out)
    assert (out / "a.txt").read_text(encoding="utf-8") == "a"
