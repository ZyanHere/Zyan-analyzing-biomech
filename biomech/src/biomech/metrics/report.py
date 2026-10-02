"""The machine-readable performance record.

Owns: serialising a finished run - its timings, its settings and the machine it
ran on - to JSON.
Does NOT own: measuring any of it (see timing.py).

Separate from `timing.py` because they answer to different readers. `RollingStats`
answers "how is it behaving right now" for the health panel; this answers "what
did this run prove, on what hardware, with which settings" for a document. A
performance claim without its hardware and configuration is not a claim, and
retyping either into prose is how published numbers drift from measured ones.
"""

from __future__ import annotations

import json
import logging
import platform
from datetime import UTC, datetime
from pathlib import Path

from ..config import Config
from .timing import PipelineMetrics

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def hardware() -> dict[str, object]:
    """What the machine is, collected rather than retyped.

    Deliberately stdlib-only: adding a dependency to describe the host would
    make the measurement harder to reproduce than the thing being measured.
    """
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "cpu_count": _cpu_count(),
    }


def _cpu_count() -> int | None:
    """Logical processors, or None if the OS will not say."""
    import os

    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        return os.cpu_count()


def build_report(
    cfg: Config,
    metrics: PipelineMetrics,
    source: str,
    label: str,
    latency_kind: str,
) -> dict[str, object]:
    """Assemble one run's complete record.

    `latency_kind` is carried explicitly because the end-to-end figure means
    different things live and on replay, and a reader of the JSON has no other
    way to tell which they are looking at.
    """
    return {
        "schema": SCHEMA_VERSION,
        "recorded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "label": label,
        "source": source,
        "latency_kind": latency_kind,
        "hardware": hardware(),
        "config": cfg.to_dict(),
        "metrics": metrics.snapshot(),
    }


def write_report(path: Path, report: dict[str, object]) -> None:
    """Write the record, creating its directory if need be."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log.info("metrics -> %s", path)
