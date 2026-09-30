"""E2 (item 5): exact-ground-truth validation of the conventions. No model involved.

Any error here is OUR mathematics. Model error is measured separately (E5).
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
from lib import synth
from lib import biomech_ref as B

SWEEPS = {
    "elbow_flexion":      ("elbow",    [0,15,30,45,60,90,120,150]),
    "knee_flexion":       ("knee",     [0,15,30,45,60,90,120,135]),
    "shoulder_flexion":   ("sh_flex",  [-60,-30,0,30,60,90,120,150,180]),
    "shoulder_abduction": ("sh_abd",   [-45,-20,0,30,60,90,120,150,180]),
    "hip_flexion":        ("hip_flex", [-30,-15,0,20,45,70,90,120]),
    "ankle_angle":        ("ankle",    [-50,-30,-15,0,10,20]),
}

print("="*76)
print("A. 3D conventions, subject facing camera. Expect EXACT recovery.")
print("="*76)
worst = 0.0
for meas, (arg, vals) in SWEEPS.items():
    for side in ("L","R"):
        errs = []
        for v in vals:
            lm = synth.build(**{arg: v}, side=side)
            errs.append(B.ALL[meas](lm, side) - v)
        e = np.max(np.abs(errs)); worst = max(worst, e)
        print(f"  {meas:<20} {side}  max |err| = {e:8.5f} deg   {'OK' if e<0.01 else 'FAIL'}")
print(f"\n  worst error across all measurements and both sides: {worst:.6f} deg")

print()
print("="*76)
print("B. 3D conventions with the body rotated. Frame is body-relative, so")
print("   rotation should change NOTHING.")
print("="*76)
print(f"  {'measurement':<20}" + "".join(f"{y:>9}" for y in (0,30,60,90,135,180)))
for meas, (arg, vals) in SWEEPS.items():
    test = vals[len(vals)//2] or vals[-1]
    row = f"  {meas:<20}"
    for yaw in (0,30,60,90,135,180):
        lm = synth.build(**{arg: test}, side="L", body_yaw=yaw)
        row += f"{B.ALL[meas](lm,'L') - test:>+9.4f}"
    print(row)
print("   (values are error in degrees vs the exact input angle)")

print()
print("="*76)
print("C. 2D image coordinates, hinge joints, as the body rotates away.")
print("   This is pure projection error - the maths is identical.")
print("="*76)
def interior2d(P, a, b, c):
    ba, bc = P[a][:2]-P[b][:2], P[c][:2]-P[b][:2]
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    return np.degrees(np.arccos(np.clip(np.dot(ba,bc)/(na*nc),-1,1)))

print(f"  {'joint':<8}{'true':>7}" + "".join(f"{y:>8}" for y in (0,15,30,45,60,75,90)))
for label, arg, idx, true_vals in [("elbow","elbow",(11,13,15),[30,60,90,120]),
                                   ("knee","knee",(23,25,27),[30,60,90,120])]:
    for tv in true_vals:
        row = f"  {label:<8}{tv:>7}"
        for yaw in (0,15,30,45,60,75,90):
            lm = synth.build(**{arg: tv}, side="L", body_yaw=yaw)
            P = synth.project(lm)
            row += f"{(180-interior2d(P,*idx)) - tv:>+8.1f}"
        print(row)
print("   (error in degrees; the subject is side-on to the camera at yaw=90)")

print()
print("="*76)
print("D. Can 2D measure SAGITTAL flexion at all while facing the camera?")
print("="*76)
for yaw in (0, 30, 60, 90):
    lm = synth.build(sh_flex=90, side="L", body_yaw=yaw)
    P  = synth.project(lm)
    o, up, left, ant = B.frame(P)
    hum = P[13][:2] - P[11][:2]
    apparent = np.degrees(np.arctan2(hum[0], hum[1]))
    print(f"  yaw {yaw:>3} deg   true shoulder flexion 90.0   "
          f"apparent in-image angle from vertical: {abs(apparent):6.1f} deg")
print("   at yaw=0 the sagittal plane is perpendicular to the image plane,")
print("   so forward arm motion projects onto almost nothing.")
