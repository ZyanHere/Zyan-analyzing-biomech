"""F4b: salvage the anchored session.

Two suspicions to test against the recorded data:
  1. the knee was measured on the FAR leg, hidden behind the near one
  2. the 180 deg shoulder pose was never achieved, so its 'truth' is wrong
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from lib.paths import fixture

import sys
import json, numpy as np
from lib import biomech_ref as B
from collections import defaultdict

W, H = 640, 480
by = defaultdict(list)
for line in open(fixture("anchored.jsonl")):
    r = json.loads(line)
    by[r["key"]].append(r)

CHAIN = {"knee": {"L": (23, 25, 27), "R": (24, 26, 28)},
         "shoulder_flexion": {"L": (23, 11, 13), "R": (24, 12, 14)}}
FN2D = {"knee": B.knee_flexion, "shoulder_flexion": B.shoulder_flexion_2d}
FN3D = {"knee": B.knee_flexion, "shoulder_flexion": B.shoulder_flexion}
TRUTH = {"knee_0": ("knee", 0), "knee_90": ("knee", 90),
         "sh_0": ("shoulder_flexion", 0), "sh_90": ("shoulder_flexion", 90),
         "sh_180": ("shoulder_flexion", 180)}


def img(r):
    return np.array([[p[0] * W, p[1] * H, 0.0] for p in r["lm"]])


def vis(r, idx):
    return min(r["lm"][i][3] for i in idx)


print("=" * 84)
print("A. Which side was actually visible? The script measured RIGHT unconditionally.")
print("=" * 84)
print(f"  {'pose':<10}{'joint':<18}{'vis LEFT':>10}{'vis RIGHT':>11}{'nearer camera':>16}")
better = {}
for key, (joint, truth) in TRUTH.items():
    rs = by.get(key, [])
    if not rs:
        continue
    vL = np.mean([vis(r, CHAIN[joint]["L"]) for r in rs])
    vR = np.mean([vis(r, CHAIN[joint]["R"]) for r in rs])
    side = "L" if vL > vR else "R"
    better[key] = side
    print(f"  {key:<10}{joint:<18}{vL:>10.2f}{vR:>11.2f}{('LEFT' if side == 'L' else 'RIGHT'):>16}")

print()
print("=" * 84)
print("B. Re-measured on the better-visible side")
print("=" * 84)
print(f"  {'pose':<10}{'true':>6}{'side':>6}{'2D':>9}{'err':>8}{'3D':>9}{'err':>8}{'vis':>7}")
res = {}
for key, (joint, truth) in TRUTH.items():
    rs = by.get(key, [])
    if not rs:
        continue
    side = better[key]
    l = np.array([img(r) for r in rs])
    w3 = np.array([r["w"] for r in rs])
    v2 = float(np.nanmedian([abs(FN2D[joint](x, side)) for x in l]))
    v3 = float(np.nanmedian([abs(FN3D[joint](x, side)) for x in w3]))
    vv = float(np.mean([vis(r, CHAIN[joint][side]) for r in rs]))
    res[key] = (joint, truth, v2, v3, vv)
    print(f"  {key:<10}{truth:>6}{side:>6}{v2:>9.1f}{v2 - truth:>+8.1f}"
          f"{v3:>9.1f}{v3 - truth:>+8.1f}{vv:>7.2f}")

print()
print("=" * 84)
print("C. Was the 180 deg pose reached? Compare the arm against the trunk.")
print("=" * 84)
for key in ("sh_90", "sh_180"):
    rs = by.get(key, [])
    if not rs:
        continue
    side = better[key]
    s_i, e_i = (11, 13) if side == "L" else (12, 14)
    arr = np.array([img(r) for r in rs])
    up = []
    for x in arr:
        mid_sh = (x[11][:2] + x[12][:2]) / 2
        mid_hp = (x[23][:2] + x[24][:2]) / 2
        trunk_up = (mid_sh - mid_hp) / np.linalg.norm(mid_sh - mid_hp)
        hum = x[e_i][:2] - x[s_i][:2]
        up.append(np.degrees(np.arccos(np.clip(
            np.dot(hum / np.linalg.norm(hum), trunk_up), -1, 1))))
    print(f"  {key:<8} angle between upper arm and trunk-UP: {np.median(up):5.1f} deg")
print("  0 deg would mean the arm is perfectly in line with the trunk, i.e. true 180.")

print()
print("=" * 84)
print("D. Summary excluding poses that were not achieved")
print("=" * 84)
for joint in ("knee", "shoulder_flexion"):
    e2, e3, used = [], [], []
    for key, (j, truth, v2, v3, vv) in res.items():
        if j != joint or key == "sh_180":
            continue
        e2.append(v2 - truth)
        e3.append(v3 - truth)
        used.append(key)
    if not e2:
        continue
    for nm, e in (("2D", np.array(e2)), ("3D", np.array(e3))):
        print(f"  {joint:<18}{nm}:  MAE {np.abs(e).mean():5.1f}   bias {e.mean():+5.1f}   "
              f"n={len(e)}  ({', '.join(used)})")
print("  sh_180 excluded: the pose was not reached, so its nominal truth is wrong.")
