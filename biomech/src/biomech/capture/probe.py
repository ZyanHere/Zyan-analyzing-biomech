"""Camera diagnostic.

Owns: measuring what the camera actually delivers, as opposed to what it claims.
Does NOT own: the pipeline, or any model.

Reports *unique* frames per second, not read() calls per second. These are not
the same number: MSMF returns the same image up to four times, which inflated an
apparent 30 FPS to a real 7.8 (FINDINGS.md F26). Counting reads measures your
loop; hashing the pixels measures the camera.

Useful to a reviewer as well as to us - it is the fastest way to find out
whether their camera and lighting can support the measurements at all.
"""

from __future__ import annotations

import hashlib
import logging
import time

import numpy as np

from ..config import CaptureConfig
from .camera import CameraSource

log = logging.getLogger(__name__)


def _digest(image: np.ndarray) -> bytes:
    """Cheap content hash, to tell a new frame from a repeat."""
    return hashlib.blake2b(image.tobytes(), digest_size=8).digest()


def probe_camera(cfg: CaptureConfig, seconds: float = 5.0) -> dict[str, float]:
    """Measure real capture behaviour for a few seconds.

    Returns a dict of measurements, also logged in readable form.
    """
    source = CameraSource(cfg)
    log.info("probing %s for %.0fs...", source.description, seconds)

    reads = 0
    unique = 0
    last_digest: bytes | None = None
    gaps: list[float] = []
    brightness: list[float] = []
    prev_ts: float | None = None

    deadline = time.perf_counter() + seconds
    try:
        while time.perf_counter() < deadline:
            frame = source.next_frame()
            if frame is None:
                time.sleep(0.001)
                continue
            reads += 1
            digest = _digest(frame.image)
            if digest != last_digest:
                unique += 1
                last_digest = digest
                if prev_ts is not None:
                    gaps.append((frame.capture_ts - prev_ts) * 1000.0)
                prev_ts = frame.capture_ts
                if unique % 5 == 0:
                    brightness.append(float(frame.image.mean()))
    finally:
        drops = source.drops
        source.close()

    gaps_sorted = sorted(gaps)
    result = {
        "reads_per_s": reads / seconds,
        "unique_fps": unique / seconds,
        "duplicate_pct": 100.0 * (1.0 - unique / reads) if reads else 0.0,
        "gap_p50_ms": _percentile(gaps_sorted, 50),
        "gap_p95_ms": _percentile(gaps_sorted, 95),
        "brightness": float(np.mean(brightness)) if brightness else 0.0,
        "dropped_by_slot": float(drops),
    }
    _report(result)
    return result


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = min(int(len(sorted_values) * pct / 100.0), len(sorted_values) - 1)
    return sorted_values[idx]


def _report(r: dict[str, float]) -> None:
    """Print the measurements with the interpretation that matters."""
    log.info("read() calls per second : %6.1f", r["reads_per_s"])
    log.info("UNIQUE frames per second: %6.1f   <- the real frame rate", r["unique_fps"])
    log.info("duplicates              : %5.0f%%", r["duplicate_pct"])
    log.info("new-frame gap p50 / p95 : %6.1f / %.1f ms", r["gap_p50_ms"], r["gap_p95_ms"])
    log.info("mean brightness         : %6.1f / 255", r["brightness"])
    log.info("dropped by the slot     : %6.0f   (expected: the consumer is slow here)",
             r["dropped_by_slot"])

    if r["unique_fps"] < 29.0:
        log.warning(
            "Only %.1f unique FPS. The usual cause is auto-exposure: it lengthens "
            "exposure time in dim light and the camera drops frames to compensate. "
            "More light, or a shorter exposure in config.py (FINDINGS.md F5, F7).",
            r["unique_fps"],
        )
    if r["duplicate_pct"] > 5.0:
        log.warning(
            "%.0f%% of reads returned a repeated image. The backend is serving stale "
            "frames; DirectShow gave 0%% on the test hardware (FINDINGS.md F26).",
            r["duplicate_pct"],
        )
