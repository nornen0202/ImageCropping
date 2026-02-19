from pathlib import Path

from crop_datasets.download.http import download_http


class FakeHeaders(dict):
    pass


class FakeResponse:
    def __init__(self, status, headers, chunks):
        self.status = status
        self.headers = headers
        self._chunks = chunks
        self._idx = 0

    def read(self, _chunk_size):
        if self._idx >= len(self._chunks):
            return b""
        c = self._chunks[self._idx]
        self._idx += 1
        return c

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_http_resume(monkeypatch, tmp_path: Path):
    target = tmp_path / "file.bin"
    target.write_bytes(b"abc")
    captured = {}

    def fake_urlopen(req, timeout=30):
        captured["range"] = req.headers.get("Range")
        return FakeResponse(206, FakeHeaders({"Content-Length": "3"}), [b"def"])

    monkeypatch.setattr("crop_datasets.download.http.urlopen", fake_urlopen)
    download_http("http://example.com/f", target)
    assert captured["range"] == "bytes=3-"
    assert target.read_bytes() == b"abcdef"
