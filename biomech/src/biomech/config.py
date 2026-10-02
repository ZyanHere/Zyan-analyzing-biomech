"""Every tuned constant in the system, in one place.

Owns: the values that were settled by experiment, and the reference to the
experiment that settled each one.
Does NOT own: any behaviour. Nothing here does work.

Each value carries its finding reference. A reader must be able to trace any
number in this system to the measurement that produced it - that traceability
is the point of this module, more than the centralisation.

The whole config is serialised into every recording, so a replay is
self-describing and a result can always be reproduced with the settings that
produced it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .errors import ConfigError


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    """Camera device settings. See FINDINGS.md F5, F7, F26."""

    device_index: int = 0

    # 1280x720 caps at 10 FPS on the test camera regardless of exposure (F5).
    width: int = 640
    height: int = 480

    # MSMF returns the same frame up to four times - 74% duplicates before the
    # exposure fix, still 11% after - and roughly double the p95 jitter (F26).
    backend: str = "dshow"

    # Auto-exposure lengthens exposure time indoors, costing 75% of the frame
    # rate AND making a motionless elbow read +/- 23 deg instead of 1.5 (F7).
    # The darker image costs nothing in accuracy; exposure *time* is what
    # matters, not brightness.
    exposure_log2: float = -5.0
    use_manual_exposure: bool = True


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Pose model selection. See FINDINGS.md F16, F24, F36."""

    # `lite` is 8 ms faster but ~3x noisier at the knee; `heavy` is 2.3x slower
    # AND drifts 21.6 deg out of plane (F16, F24).
    variant: str = "full"

    # VIDEO mode carries tracking state between frames: 2.4x faster than IMAGE
    # and 82% less frame-to-frame knee jump. It requires a single contiguous
    # stream, which is why the pipeline has one worker and not a pool (F36).
    running_mode: str = "video"


@dataclass(frozen=True, slots=True)
class ValidityConfig:
    """Thresholds for refusing to report a measurement. See F30, F37."""

    # At 0.50 a 76 deg error slipped through at 0.52 (F37).
    min_visibility: float = 0.55

    # Per-side bone-length variation over time. Catches the seated case, which
    # visibility misses at any threshold (F37).
    max_bone_cv_pct: float = 6.0

    # Bilateral, so it cannot be computed when one side is hidden, and it
    # false-positives on camera elevation - 14.1% asymmetry on a measurement
    # accurate to 0.2 deg. ADVISORY ONLY: it never rejects (F37).
    advisory_bone_asymmetry_pct: float = 8.0

    # Window for the bone-length variance estimate, in frames. One second at
    # 30 FPS. NOT measured - chosen so the signal reports the reconstruction's
    # current stability rather than its history, short enough that a limb
    # entering occlusion is flagged within a second. The thresholds it feeds
    # were measured (F37); this window was not swept.
    bone_history_frames: int = 30


@dataclass(frozen=True, slots=True)
class OrientationConfig:
    """Which plane the subject is presenting. See F27, F34, F35, F36."""

    # 2D sagittal error exceeds 15 deg below ~45 deg of rotation (F27).
    sagittal_min_yaw_deg: float = 45.0

    # Abduction is a frontal-plane measurement and needs the opposite
    # orientation from the other five. One camera cannot serve both (F34).
    frontal_max_yaw_deg: float = 30.0

    # Yaw noise is ~1 deg sd and real postures sit at 44-46 deg. A bare
    # threshold flipped state up to 3 times inside one phase; this gives
    # zero (F36).
    hysteresis_deg: float = 5.0
    hysteresis_frames: int = 5

    # Anterior direction sets the sign of every sagittal measurement. Updated
    # only when all three cues agree, because the nose and ear cues are both
    # head-based and fail together - majority voting scored 0.51 where the best
    # single cue scored 1.00 (F35).
    require_unanimous_anterior: bool = True


@dataclass(frozen=True, slots=True)
class FilterConfig:
    """Temporal smoothing. See FINDINGS.md F15, F27."""

    # One Euro adapts its cutoff to velocity. A fixed EMA at alpha=0.15 bought
    # the lowest elbow jitter at 240 ms of lag, and was worse on both axes at
    # once for the knee (F15).
    beta: float = 0.01
    min_cutoff_hz: float = 1.0
    d_cutoff_hz: float = 1.0

    # Applied to landmark POSITIONS, not to the resulting angle: 40-45% better
    # during real movement, because it keeps the skeleton geometrically
    # consistent (F27).
    filter_landmarks: bool = True

    # Below the lower bound, numerical blow-up. Above the upper bound the
    # previous sample is not evidence about the current one, so the filter
    # resets rather than interpolating across the gap.
    dt_min_s: float = 0.001
    dt_max_s: float = 0.250


@dataclass(frozen=True, slots=True)
class UIConfig:
    """Rendering.

    The canvas is composed at the video's native resolution; these are only the
    initial window size, which the window manager scales for free. Resizing the
    canvas in numpy instead costs the inference thread directly.
    """

    # None of these came from an experiment: they are display preferences, and
    # the window is resizable, so nothing downstream depends on them. Kept here
    # so a reviewer does not go looking for a finding that does not exist.
    window_width: int = 1000
    window_height: int = 576
    show_health_panel: bool = True
    draw_skeleton: bool = True


@dataclass(frozen=True, slots=True)
class MetricsConfig:
    """Timing. p95 matters because dropped-frame stalls live in the tail."""

    # Four seconds at 30 FPS. A choice, not a measurement: long enough that one
    # slow frame does not swing the displayed rate, short enough that the panel
    # still answers "how is it behaving now" rather than "on average since
    # start". The 72 s benchmark run (F38) reads this window at the END of the
    # run, which is what makes "no thermal decay" a claim about the last four
    # seconds rather than an average that would hide a decline.
    window_frames: int = 120

    # p50 and p95 because the assignment asks for typical and worst-case
    # latency; the tail is what a user actually notices.
    percentiles: tuple[float, ...] = (50.0, 95.0)


@dataclass(frozen=True, slots=True)
class Config:
    """The complete configuration for one run."""

    capture: CaptureConfig = field(default_factory=CaptureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    validity: ValidityConfig = field(default_factory=ValidityConfig)
    orientation: OrientationConfig = field(default_factory=OrientationConfig)
    filtering: FilterConfig = field(default_factory=FilterConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    def to_dict(self) -> dict[str, Any]:
        """Serialisable form, written into every recording."""
        return asdict(self)

    def validate(self) -> None:
        """Reject configurations that cannot produce a meaningful result."""
        if self.model.variant not in ("lite", "full", "heavy"):
            raise ConfigError(
                f"Unknown model variant {self.model.variant!r}. "
                "Expected one of: lite, full, heavy."
            )
        if self.model.running_mode not in ("video", "image"):
            raise ConfigError(
                f"Unknown running mode {self.model.running_mode!r}. "
                "Expected 'video' (tracked, faster) or 'image' (stateless)."
            )
        if self.orientation.frontal_max_yaw_deg >= self.orientation.sagittal_min_yaw_deg:
            raise ConfigError(
                "Orientation bands overlap: frontal_max_yaw_deg "
                f"({self.orientation.frontal_max_yaw_deg}) must be below "
                f"sagittal_min_yaw_deg ({self.orientation.sagittal_min_yaw_deg}). "
                "The gap between them is the band where neither plane is valid."
            )
        if not 0.0 < self.validity.min_visibility < 1.0:
            raise ConfigError(
                f"min_visibility must be between 0 and 1, got "
                f"{self.validity.min_visibility}."
            )
        if self.filtering.dt_min_s >= self.filtering.dt_max_s:
            raise ConfigError(
                "dt_min_s must be below dt_max_s; they bound the same quantity."
            )
