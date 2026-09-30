"""Turning a signed number into what the goniometry chart calls it.

Owns: display names, normal ranges, and which plane each measurement needs.
Does NOT own: any computation.

The chart lists the two directions of a joint as **separate named ranges** -
shoulder flexion 0-180 and extension 0-60 - not as one range with a sign. So a
negative value is displayed as its named opposite, never as a minus sign:
`-30` on the hip reads as "30 deg of extension", because that is the quantity
a clinician would recognise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..types import MeasurementName, Plane

# Which way the subject must face. Sagittal and frontal are mutually exclusive,
# so at most ten of the twelve measurements can be valid at once (F34).
REQUIRED_PLANE: dict[MeasurementName, Plane] = {
    MeasurementName.ELBOW_FLEXION: Plane.SAGITTAL,
    MeasurementName.KNEE_FLEXION: Plane.SAGITTAL,
    MeasurementName.SHOULDER_FLEXION: Plane.SAGITTAL,
    MeasurementName.SHOULDER_ABDUCTION: Plane.FRONTAL,
    MeasurementName.HIP_FLEXION: Plane.SAGITTAL,
    MeasurementName.ANKLE_ANGLE: Plane.SAGITTAL,
}


@dataclass(frozen=True, slots=True)
class ChartEntry:
    """One row of the goniometry chart.

    `negative_label` is None for joints the chart gives a single direction:
    an elbow does not meaningfully extend past a straight arm.
    """

    positive_label: str
    negative_label: str | None
    positive_max: float
    negative_max: float


CHART: dict[MeasurementName, ChartEntry] = {
    MeasurementName.ELBOW_FLEXION: ChartEntry("flexion", None, 150.0, 0.0),
    MeasurementName.KNEE_FLEXION: ChartEntry("flexion", None, 135.0, 0.0),
    MeasurementName.SHOULDER_FLEXION: ChartEntry("flexion", "extension", 180.0, 60.0),
    MeasurementName.SHOULDER_ABDUCTION: ChartEntry("abduction", "adduction", 180.0, 45.0),
    MeasurementName.HIP_FLEXION: ChartEntry("flexion", "extension", 120.0, 30.0),
    MeasurementName.ANKLE_ANGLE: ChartEntry("dorsiflexion", "plantarflexion", 20.0, 50.0),
}

# A joint held at exactly its charted maximum computes to 150.0000001. Flagging
# that as abnormal would report a rounding artefact as a clinical finding.
_RANGE_TOLERANCE_DEG = 0.5


@dataclass(frozen=True, slots=True)
class Presentation:
    """A measurement as it should appear on screen."""

    label: str
    magnitude_deg: float
    within_normal_range: bool

    def __str__(self) -> str:
        if math.isnan(self.magnitude_deg):
            return "--"
        return f"{self.magnitude_deg:.0f} deg {self.label}"


def present(name: MeasurementName, value_deg: float) -> Presentation:
    """Render a signed value in the chart's own terms."""
    entry = CHART[name]
    if math.isnan(value_deg):
        return Presentation(label="no data", magnitude_deg=float("nan"),
                            within_normal_range=False)

    if value_deg >= 0.0 or entry.negative_label is None:
        return Presentation(
            label=entry.positive_label,
            magnitude_deg=abs(value_deg),
            within_normal_range=abs(value_deg) <= entry.positive_max + _RANGE_TOLERANCE_DEG,
        )
    return Presentation(
        label=entry.negative_label,
        magnitude_deg=abs(value_deg),
        within_normal_range=abs(value_deg) <= entry.negative_max + _RANGE_TOLERANCE_DEG,
    )
