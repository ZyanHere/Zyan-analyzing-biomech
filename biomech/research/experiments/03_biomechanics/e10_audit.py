"""E10: full audit of every mandatory measurement.

Answers, per measurement:
  1. does the 3D formula recover exact known angles, in BOTH directions?
  2. does the 2D formula recover them, at the orientation it requires?
  3. which orientation IS required, measured rather than assumed?
  4. is the sign correct - does positive mean what the chart says it means?
  5. how is a negative value displayed?

No camera, no model. This isolates our mathematics from everything else.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import sys
import math
import numpy as np
from lib import synth
from lib import biomech_ref as B

# name -> (synth kwarg, 3D fn, 2D fn, sweep values covering BOTH directions)
CASES = [
    ("elbow_flexion",      "elbow",    B.elbow_flexion,      B.elbow_flexion,
     [0, 15, 30, 60, 90, 120, 150]),
    ("knee_flexion",       "knee",     B.knee_flexion,       B.knee_flexion,
     [0, 15, 30, 60, 90, 120, 135]),
    ("shoulder_flexion",   "sh_flex",  B.shoulder_flexion,   B.shoulder_flexion_2d,
     [-60, -30, -10, 0, 30, 90, 150, 180]),
    ("shoulder_abduction", "sh_abd",   B.shoulder_abduction, B.shoulder_abduction_2d,
     [-45, -20, 0, 30, 90, 150, 180]),
    ("hip_flexion",        "hip_flex", B.hip_flexion,        B.hip_flexion_2d,
     [-30, -15, 0, 20, 60, 90, 120]),
    ("ankle_angle",        "ankle",    B.ankle_angle,        B.ankle_angle_2d,
     [-50, -30, -10, 0, 10, 20]),
]

# yaw that presents each plane to the camera: 0 = facing, 90 = side-on
VIEW_YAW = {"sagittal": 90, "frontal": 0}

print("=" * 86)
print("1. 3D formulas vs exact synthetic truth, BOTH directions, both sides")
print("=" * 86)
print(f"  {'measurement':<20}{'range tested':>22}{'max |err|':>12}{'verdict':>10}")
worst3 = 0.0
for name, arg, f3, _f2, vals in CASES:
    errs = []
    for side in ("L", "R"):
        for v in vals:
            lm = synth.build(**{arg: v}, side=side)
            errs.append(B.wrap180(f3(lm, side) - v))
    e = float(np.max(np.abs(errs)))
    worst3 = max(worst3, e)
    print(f"  {name:<20}{f'{min(vals)} to {max(vals)} deg':>22}{e:>12.6f}"
          f"{'ok' if e < 0.01 else 'FAIL':>10}")
print(f"\n  worst 3D error across every mandatory measurement: {worst3:.6f} deg")

print()
print("=" * 86)
print("2. 2D formulas, each at the orientation its plane requires")
print("=" * 86)
print(f"  {'measurement':<20}{'plane':<11}{'view':<9}{'max |err|':>12}{'verdict':>10}")
worst2 = 0.0
for name, arg, _f3, f2, vals in CASES:
    view = B.REQUIRED_VIEW[name]
    yaw = VIEW_YAW[view]
    errs = []
    for side in ("L", "R"):
        for v in vals:
            P = synth.project(synth.build(**{arg: v}, side=side, body_yaw=yaw))
            errs.append(B.wrap180(f2(P, side) - v))
    e = float(np.max(np.abs(errs)))
    worst2 = max(worst2, e)
    print(f"  {name:<20}{view:<11}{('side-on' if yaw else 'face-on'):<9}{e:>12.4f}"
          f"{'ok' if e < 1.0 else 'FAIL':>10}")
print(f"\n  worst 2D error at the correct orientation: {worst2:.4f} deg")

print()
print("=" * 86)
print("3. Required orientation, MEASURED: max |error| vs body yaw")
print("=" * 86)
print(f"  {'measurement':<20}" + "".join(f"{y:>8}" for y in (0, 30, 45, 60, 90)))
for name, arg, _f3, f2, vals in CASES:
    row = f"  {name:<20}"
    for yaw in (0, 30, 45, 60, 90):
        errs = []
        for v in vals:
            P = synth.project(synth.build(**{arg: v}, side="R", body_yaw=yaw))
            got = f2(P, "R")
            errs.append(abs(B.wrap180(got - v)) if not np.isnan(got) else 999.0)
        row += f"{min(max(errs), 999):>8.1f}"
    print(row)
print("  0 = facing the camera, 90 = fully side-on. Lower is better.")

print()
print("=" * 86)
print("4. Sign convention: does positive mean what the chart says?")
print("=" * 86)
print(f"  {'measurement':<20}{'positive input':>16}{'reads':>9}{'negative input':>17}{'reads':>9}{'ok':>5}")
for name, arg, f3, _f2, vals in CASES:
    pos = max(v for v in vals if v > 0)
    negs = [v for v in vals if v < 0]
    lm_p = synth.build(**{arg: pos}, side="R")
    vp = f3(lm_p, "R")
    if negs:
        neg = min(negs)
        vn = f3(synth.build(**{arg: neg}, side="R"), "R")
        ok = "ok" if vp > 0 and vn < 0 else "SIGN?"
        print(f"  {name:<20}{pos:>16}{vp:>9.1f}{neg:>17}{vn:>9.1f}{ok:>5}")
    else:
        ok = "ok" if vp > 0 else "SIGN?"
        print(f"  {name:<20}{pos:>16}{vp:>9.1f}{'n/a (unsigned)':>17}{'-':>9}{ok:>5}")

print()
print("=" * 86)
print("5. Display: a negative value must become its NAMED opposite, not a minus sign")
print("=" * 86)
print(f"  {'measurement':<20}{'raw':>8}   {'shown as':<34}{'within normal range?':>21}")
for name, arg, f3, _f2, vals in CASES:
    for v in (max(vals), min(vals)):
        lab, mag, inr = B.present(name, float(f3(synth.build(**{arg: v}, side="R"), "R")))
        print(f"  {name:<20}{v:>8}   {f'{mag:.0f} deg of {lab}':<34}{('yes' if inr else 'OUT OF RANGE'):>21}")
