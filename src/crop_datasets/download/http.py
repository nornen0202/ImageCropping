from __future__ import annotations

import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    class tqdm:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def update(self, n):
            return None


class DownloadError(RuntimeError):
    pass


def download_http(url: str, dest: Path, chunk_size: int = 1024 * 1024, retries: int = 3) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    resume_from = dest.stat().st_size if dest.exists() else 0

    for attempt in range(retries + 1):
        try:
            headers = {"Range": f"bytes={resume_from}-"} if resume_from > 0 else {}
            req = Request(url, headers=headers)
            with urlopen(req, timeout=30) as resp:
                code = getattr(resp, "status", 200)
                if code >= 400:
                    raise DownloadError(f"HTTP {code} while downloading {url}")

                content_length = resp.headers.get("Content-Length")
                total = int(content_length) + resume_from if content_length else None
                mode = "ab" if resume_from > 0 and code == 206 else "wb"
                if mode == "wb":
                    resume_from = 0

                with dest.open(mode) as f, tqdm(total=total, initial=resume_from, unit="B", unit_scale=True, desc=dest.name) as bar:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        bar.update(len(chunk))
                return dest
        except HTTPError as exc:
            if exc.code == 416:
                return dest
            if attempt >= retries:
                raise DownloadError(f"HTTP {exc.code} while downloading {url}") from exc
        except URLError as exc:
            if attempt >= retries:
                raise DownloadError(f"Network error while downloading {url}: {exc}") from exc

        time.sleep(2**attempt)

    raise DownloadError(f"Exhausted retries for {url}")
