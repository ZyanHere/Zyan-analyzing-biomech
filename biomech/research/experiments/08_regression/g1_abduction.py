"""Does raising the arm corrupt the shoulder landmark, and what does that break?

Two reviews of a recorded demo disagreed about an interval where the subject
raised both arms to horizontal while facing the camera:

  - abduction displayed "52 deg adduction" when the arms were visibly at ~90
  - both abduction rows were then vetoed "unstable tracking" for five seconds
  - body yaw wobbled between 1 and 26 degrees

The hypothesis under test was that these are ONE defect: raising the arm
elevates and protracts the shoulder, moving the landmark that the abduction
angle, the bone-length veto and the yaw estimator all read. F36 documented that
mechanism when it rejected the ratio-based yaw estimator.

The variance is computed the way the application computes it - a rolling window
of `bone_history_frames`, not a statistic over the whole session - because the
veto fires on the rolling value and nothing else is comparable to it.

Run against a session recorded with --record-landmarks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from biomech.biomechanics.angles import CHAIN, compute_angles  # noqa: E402
from biomech.capture.landmarks import LandmarkSource  # noqa: E402
from biomech.config import Config  # noqa: E402
from biomech.errors import SourceExhaustedError  # noqa: E402
from biomech.types import MeasurementName, Side  # noqa: E402
from biomech.validity.orientation import yaw_degrees  # noqa: E402
from biomech.validity.signals import SEGMENTS  # noqa: E402

L_SH, L_EL, L_HIP, R_HIP = 11, 13, 23, 24


def load(path: Path) -> dict[str, np.ndarray]:
    src = LandmarkSource(path)
    xy, xyz, abd_l, abd_r, yaw = [], [], [], [], []
    while True:
        try:
            _, result = src.next_pose()
        except SourceExhaustedError:
            break
        m = result.landmarks
        if m is None:
            continue
        xy.append(m.image_xy)
        xyz.append(m.world_xyz)
        angles = compute_angles(m, 1.0)
        abd_l.append(angles[(MeasurementName.SHOULDER_ABDUCTION, Side.LEFT)])
        abd_r.append(angles[(MeasurementName.SHOULDER_ABDUCTION, Side.RIGHT)])
        yaw.append(yaw_degrees(m.world_xyz))
    return {"xy": np.array(xy), "xyz": np.array(xyz), "abd_l": np.array(abd_l),
            "abd_r": np.array(abd_r), "yaw": np.array(yaw)}


def arm_elevation(xy: np.ndarray) -> np.ndarray:
    """Elbow height above the shoulder, as a fraction of torso length.

    A reference-free proxy for "is the arm raised", computed from landmarks the
    measurement under test does not use for its sign - so it cannot beg the
    question. A hanging arm is strongly negative; a horizontal arm is near zero.
    """
    torso = np.linalg.norm(xy[:, L_SH] - xy[:, L_HIP], axis=1)
    return (xy[:, L_SH, 1] - xy[:, L_EL, 1]) / torso


def rolling_cv_pct(lengths: np.ndarray, window: int) -> np.ndarray:
    """Coefficient of variation over a trailing window, as the tracker does it."""
    out = np.full(len(lengths), np.nan)
    for i in range(window, len(lengths)):
        w = lengths[i - window:i]
        out[i] = 100.0 * w.std() / w.mean()
    return out


def bands_of(raised: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "hanging": raised < -0.35,
        "part-way": (raised >= -0.35) & (raised < -0.12),
        "horizontal": (raised >= -0.12) & (raised < 0.12),
    }


def report(path: Path) -> None:
    cfg = Config()
    window = cfg.validity.bone_history_frames
    threshold = cfg.validity.max_bone_cv_pct
    d = load(path)
    xy, xyz = d["xy"], d["xyz"]
    raised = arm_elevation(xy)
    bands = bands_of(raised)

    print(f"{len(xy)} frames with a person; veto window {window} frames, "
          f"threshold {threshold}% \n")
    for name, mask in bands.items():
        print(f"  {name:11s} {mask.sum():5d} frames")

    print("\n--- Does arm elevation trip the bone-length veto? ---")
    print(f"{'segment':10s} {'side':5s} " + "".join(f"{b:>22s}" for b in bands))
    print(f"{'':16s}" + "".join(f"{'% frames over veto':>22s}" for _ in bands))
    for segment, (prox, dist) in SEGMENTS.items():
        for side in (Side.LEFT, Side.RIGHT):
            chain = CHAIN[side]
            length = np.linalg.norm(xy[:, chain[prox]] - xy[:, chain[dist]], axis=1)
            cv = rolling_cv_pct(length, window)
            cells = []
            for mask in bands.values():
                sel = cv[mask]
                sel = sel[~np.isnan(sel)]
                cells.append(f"{100 * (sel > threshold).mean():21.1f}%" if len(sel) else f"{'-':>22s}")
            print(f"{segment:10s} {side.value:5s} " + "".join(cells))

    print("\n--- Abduction by arm elevation (horizontal truth is near +90) ---")
    for name, mask in bands.items():
        for label, series in (("L", d["abd_l"]), ("R", d["abd_r"])):
            v = series[mask]
            v = v[~np.isnan(v)]
            if len(v) < 10:
                continue
            print(f"  {name:11s} {label}  median {np.median(v):7.1f}   "
                  f"p5 {np.percentile(v, 5):7.1f}  p95 {np.percentile(v, 95):7.1f}   "
                  f"reads as adduction in {100 * (v < 0).mean():5.1f}% of frames")

    print("\n--- Yaw and shoulder position by arm elevation ---")
    hip_y = 0.5 * (xy[:, L_HIP, 1] + xy[:, R_HIP, 1])
    torso = np.linalg.norm(xy[:, L_SH] - xy[:, L_HIP], axis=1)
    shoulder_h = (hip_y - xy[:, L_SH, 1]) / torso
    for name, mask in bands.items():
        if mask.sum() < 10:
            continue
        print(f"  {name:11s} yaw {d['yaw'][mask].mean():5.1f} "
              f"(sd {d['yaw'][mask].std():4.1f})   "
              f"shoulder height {shoulder_h[mask].mean():.3f} "
              f"(sd {shoulder_h[mask].std():.3f})")


FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "sessions" / "abduction.jsonl"
)

if __name__ == "__main__":
    report(Path(sys.argv[1]) if len(sys.argv) > 1 else FIXTURE)
