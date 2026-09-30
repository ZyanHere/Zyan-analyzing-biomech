"""E11: test the two falsifiable claims from the architecture review.

A. Is yaw = arccos(shoulder_width / max_shoulder_width) reliable when the subject
   changes DISTANCE? Pixel width depends on rotation AND distance, so the estimator
   may be conflating them.

B. (review item G) Is the nose-based anterior direction stable? It sets the SIGN of
   every sagittal measurement, so a flip turns flexion into extension.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from lib.paths import fixture

import sys
import json
import math
import numpy as np
from collections import defaultdict

W, H = 640, 480


def load(path, key):
    by = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if r.get("lm"):
            by[r[key]].append(r)
    return by


def px(r, i):
    return np.array([r["lm"][i][0] * W, r["lm"][i][1] * H])


def shoulder_w(r):
    return float(np.linalg.norm(px(r, 11) - px(r, 12)))


def trunk_len(r):
    mid_sh = (px(r, 11) + px(r, 12)) / 2
    mid_hp = (px(r, 23) + px(r, 24)) / 2
    return float(np.linalg.norm(mid_sh - mid_hp))


print("=" * 88)
print("A. Does the yaw estimator survive a change of DISTANCE?")
print("=" * 88)
rob = load(fixture("robustness_session.jsonl"), "phase")
SAME_ORIENT = ["baseline", "near", "far", "cam_low", "cam_high"]
print("  These phases were all recorded facing the camera. A sound estimator should")
print("  report roughly the SAME yaw for all of them despite the distance changes.\n")
print(f"  {'phase':<11}{'shoulder px':>13}{'trunk px':>10}{'sw/trunk':>11}")
rows = {}
for ph in SAME_ORIENT:
    rs = rob.get(ph, [])
    if not rs:
        continue
    sw = float(np.median([shoulder_w(r) for r in rs]))
    tl = float(np.median([trunk_len(r) for r in rs]))
    rows[ph] = (sw, tl, sw / tl)
    print(f"  {ph:<11}{sw:>13.1f}{tl:>10.1f}{sw/tl:>11.3f}")

sw_all = [v[0] for v in rows.values()]
ratio_all = [v[2] for v in rows.values()]
print(f"\n  raw shoulder px  spread: {min(sw_all):.0f} to {max(sw_all):.0f}  "
      f"({100*(max(sw_all)-min(sw_all))/np.mean(sw_all):.0f}% of mean)")
print(f"  shoulder/trunk   spread: {min(ratio_all):.3f} to {max(ratio_all):.3f}  "
      f"({100*(max(ratio_all)-min(ratio_all))/np.mean(ratio_all):.0f}% of mean)")

print("\n  yaw each estimator would report, using the max seen in THIS set as reference:")
sw_max, ratio_max = max(sw_all), max(ratio_all)
print(f"  {'phase':<11}{'yaw (raw px)':>14}{'yaw (normalised)':>19}")
for ph, (sw, tl, ratio) in rows.items():
    y1 = math.degrees(math.acos(min(max(sw / sw_max, 0.0), 1.0)))
    y2 = math.degrees(math.acos(min(max(ratio / ratio_max, 0.0), 1.0)))
    print(f"  {ph:<11}{y1:>14.1f}{y2:>19.1f}")
print("\n  Truth for every row above is the same orientation, so any spread is error.")

print()
print("=" * 88)
print("A2. Does the normalised estimator still respond to REAL rotation?")
print("=" * 88)
print("  A scale-invariant ratio is worthless if it no longer detects turning.\n")
acc = load(fixture("accuracy_session.jsonl"), "target")
print(f"  {'source':<26}{'sw/trunk':>11}{'recorded yaw':>15}")
for ph in ("baseline", "rotated"):
    rs = rob.get(ph, [])
    if rs:
        print(f"  robustness/{ph:<15}{np.median([shoulder_w(r)/trunk_len(r) for r in rs]):>11.3f}"
              f"{'facing' if ph == 'baseline' else 'side-on':>15}")
for t in sorted(acc)[:2]:
    rs = acc[t]
    yaw = np.median([r.get("yaw", float('nan')) for r in rs])
    print(f"  accuracy/target {t:<10}{np.median([shoulder_w(r)/trunk_len(r) for r in rs]):>11.3f}"
          f"{yaw:>14.0f} deg")

print()
print("=" * 88)
print("B. Is the nose-based anterior direction stable? It sets every sagittal SIGN.")
print("=" * 88)


def anterior_from(r, mode):
    mid_sh = (px(r, 11) + px(r, 12)) / 2
    mid_hp = (px(r, 23) + px(r, 24)) / 2
    down = mid_hp - mid_sh
    down = down / (np.linalg.norm(down) + 1e-9)
    perp = np.array([-down[1], down[0]])
    if mode == "nose":
        ref = px(r, 0) - mid_sh
    elif mode == "ear_to_nose":
        ref = px(r, 0) - (px(r, 7) + px(r, 8)) / 2
    elif mode == "foot":
        ref = ((px(r, 31) - px(r, 29)) + (px(r, 32) - px(r, 30))) / 2
    return 1.0 if np.dot(perp, ref) > 0 else -1.0


for path, key, label in [(fixture("robustness_session.jsonl"), "phase", "robustness"),
                         (fixture("anchored.jsonl"), "key", "anchored")]:
    by = load(path, key)
    print(f"\n  [{label}]  fraction of frames agreeing with that phase's MAJORITY sign")
    print(f"  {'phase':<12}{'nose':>8}{'ear->nose':>12}{'heel->toe':>12}{'nose vis':>10}")
    for ph, rs in by.items():
        out = []
        for mode in ("nose", "ear_to_nose", "foot"):
            signs = np.array([anterior_from(r, mode) for r in rs])
            out.append(max((signs > 0).mean(), (signs < 0).mean()))
        nv = float(np.mean([r["lm"][0][3] for r in rs]))
        print(f"  {ph:<12}{out[0]:>8.2f}{out[1]:>12.2f}{out[2]:>12.2f}{nv:>10.2f}")
print("\n  1.00 = perfectly consistent within the phase. Anything below ~0.95 means the")
print("  sign flips mid-phase, which would swap flexion and extension on screen.")
