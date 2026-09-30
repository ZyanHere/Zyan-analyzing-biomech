"""E4: run the verified conventions over the REAL recording.

F19 proved the maths is exact on synthetic input. Everything that deviates here
is model error, capture error, or the subject not holding the pose.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from lib.paths import fixture

import json, numpy as np
from lib import biomech_ref as B
from collections import defaultdict

ph = defaultdict(list)
for line in open(fixture("session.jsonl")):
    r = json.loads(line)
    if r.get("found") and "w" in r: ph[r["phase"]].append(r)

def lms(r):
    """No axis flipping. The raw world frame is already right-handed and
    self-consistent (E1: up=-y, left=+x, anterior=-z, cross(left,up)=-z), and
    biomech_ref derives its frame from the landmarks. Flipping a single axis
    reverses handedness and silently inverts every sagittal sign."""
    return np.array(r["w"], dtype=float)

def vis(r, idx): return min(r["lm"][i][3] for i in idx)

MEAS = [("elbow_flexion","L",(11,13,15)), ("elbow_flexion","R",(12,14,16)),
        ("knee_flexion","L",(23,25,27)),  ("knee_flexion","R",(24,26,28)),
        ("shoulder_flexion","L",(23,11,13)), ("shoulder_flexion","R",(24,12,14)),
        ("shoulder_abduction","L",(23,11,13)), ("shoulder_abduction","R",(24,12,14)),
        ("hip_flexion","L",(11,23,25)), ("hip_flexion","R",(12,24,26)),
        ("ankle_angle","L",(25,27,29,31)), ("ankle_angle","R",(26,28,30,32))]

for phase in ("still","elbow","squat","rotate"):
    rs = ph[phase]
    print("="*78)
    print(f"{phase}   ({len(rs)} frames)")
    print("="*78)
    print(f"  {'measurement':<22}{'median':>9}{'min':>9}{'max':>9}{'sd':>8}{'vis':>7}")
    for name, side, idx in MEAS:
        v = np.array([B.ALL[name](lms(r), side) for r in rs], dtype=float)
        vv = np.mean([vis(r, idx) for r in rs])
        print(f"  {name+' '+side:<22}{np.nanmedian(v):>9.1f}{np.nanmin(v):>9.1f}"
              f"{np.nanmax(v):>9.1f}{np.nanstd(v):>8.1f}{vv:>7.2f}")
    print()

print("="*78)
print("SANITY CHECKS against the goniometry chart")
print("="*78)
st = ph["still"]
print("  standing still -> every measurement should sit near anatomical neutral (0):")
for name, side, idx in MEAS:
    v = np.nanmedian([B.ALL[name](lms(r), side) for r in st])
    flag = "ok" if abs(v) < 15 else "OFF NEUTRAL"
    print(f"    {name+' '+side:<22}{v:>+8.1f}   {flag}")

print()
print("  elbow sweep vs chart range 0-150:")
ev = np.array([B.elbow_flexion(lms(r), "R") for r in ph["elbow"]])
print(f"    right elbow: {np.nanmin(ev):.1f} to {np.nanmax(ev):.1f} deg")
print()
print("  knee sweep during squat vs chart range 0-135:")
for side in ("L","R"):
    kv = np.array([B.knee_flexion(lms(r), side) for r in ph["squat"]])
    print(f"    {side} knee: {np.nanmin(kv):.1f} to {np.nanmax(kv):.1f} deg")
