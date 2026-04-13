from __future__ import annotations

import time
from pathlib import Path
from typing import Optional


def format_duration(seconds: float) -> str:
    total_seconds = max(0.0, float(seconds))
    if total_seconds < 60.0:
        return f"{total_seconds:.1f}s"
    minutes, secs = divmod(int(round(total_seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{secs:02d}s"


def progress_log(message: str, *, enabled: bool = True) -> None:
    if not enabled:
        return
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[progress {timestamp}] {message}", flush=True)


def count_nonempty_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def build_progress_bar(count: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[?]"
    ratio = max(0.0, min(1.0, float(count) / float(total)))
    filled = int(ratio * width)
    if filled >= width:
        return "[" + "=" * width + "]"
    if filled <= 0:
        return "[" + "." * width + "]"
    return "[" + "=" * max(0, filled - 1) + ">" + "." * max(0, width - filled) + "]"


class ProgressTracker:
    def __init__(
        self,
        label: str,
        *,
        total: Optional[int] = None,
        unit: str = "items",
        every: int = 100,
        min_seconds: float = 10.0,
        enabled: bool = True,
    ) -> None:
        self.label = str(label)
        self.total = int(total) if total is not None else None
        self.unit = str(unit)
        self.every = max(1, int(every))
        self.min_seconds = max(0.0, float(min_seconds))
        self.enabled = bool(enabled)
        self.started_at = time.monotonic()
        self.last_emit_at = self.started_at
        self.last_count = 0
        if self.enabled:
            total_part = f" total={self.total}" if self.total is not None else ""
            progress_log(f"{self.label}: start{total_part} unit={self.unit}", enabled=True)

    def _build_message(self, count: int, extra: str = "") -> str:
        elapsed = time.monotonic() - self.started_at
        if self.total is not None and self.total > 0:
            pct = 100.0 * float(count) / float(self.total)
            parts = [
                f"{self.label}: {build_progress_bar(count, self.total)} {count}/{self.total} ({pct:.1f}%)",
            ]
            if count > 0 and elapsed > 0.0 and count < self.total:
                rate = float(count) / float(elapsed)
                remaining = max(0, self.total - count)
                eta_seconds = float(remaining) / max(rate, 1e-9)
                parts.append(f"{format_duration(elapsed)}<{format_duration(eta_seconds)}")
            else:
                parts.append(format_duration(elapsed))
        else:
            parts = [
                f"{self.label}: {count} {self.unit}",
                format_duration(elapsed),
            ]
        if extra:
            parts.append(extra)
        return " | ".join(parts)

    def update(self, count: int, *, extra: str = "") -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        force = bool(self.total is not None and count >= self.total)
        count_due = (count - self.last_count) >= self.every
        time_due = (now - self.last_emit_at) >= self.min_seconds and count > self.last_count
        if not (force or count_due or time_due):
            return
        progress_log(self._build_message(int(count), extra=extra), enabled=True)
        self.last_count = int(count)
        self.last_emit_at = now

    def finish(self, count: int, *, extra: str = "") -> None:
        if not self.enabled:
            return
        progress_log(self._build_message(int(count), extra=extra), enabled=True)
