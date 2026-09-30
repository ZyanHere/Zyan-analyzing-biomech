"""F1b: was the subject side-on? If yes, 2D is geometrically valid (F20) and the
2D/3D disagreement is entirely 3D's fault, circularity notwithstanding.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from lib.paths import fixture

import json, numpy as np
from lib import biomech_ref as B
from collections import defaultdict

W, H = 640, 480
by = defaultdict(list)
for line in open(fixture("accuracy_session.jsonl")):
    r = json.loads(line); by[r["target"]].append(r)

print("="*78)
print("A. Body orientation, from the yaw RECORDED at capture time.")
print("   Yaw is measured against a face-on calibration, not against the session's")
print("   own spread - normalising within the session reports 0 when the subject was")
print("   side-on the whole time, which is exactly backwards.")
print("="*78)
have_yaw = all("yaw" in r for rs in by.values() for r in rs)
print(f"  {'target':>7}{'yaw median':>12}{'min':>8}{'max':>8}{'2D valid?':>12}")
for t in sorted(by):
    rs = by[t]
    if have_yaw:
        y = np.array([r["yaw"] for r in rs])
    else:
        y = np.array([np.nan])
    med = np.nanmedian(y)
    verdict = "YES" if med >= 60 else ("marginal" if med >= 45 else "NO")
    print(f"  {t:>7}{med:>12.1f}{np.nanmin(y):>8.1f}{np.nanmax(y):>8.1f}{verdict:>12}")
if not have_yaw:
    print("  (no recorded yaw in this file - it predates the orientation gate)")

print()
print("="*78)
print("B. Is the 3D skeleton even self-consistent? Bone lengths must not change.")
print("="*78)
print(f"  {'target':>7}{'upper arm cm':>14}{'cv %':>8}{'forearm cm':>13}{'cv %':>8}")
for t in sorted(by):
    w = np.array([r["w"] for r in by[t]])
    ua = np.linalg.norm(w[:,12]-w[:,14], axis=1)*100
    fa = np.linalg.norm(w[:,14]-w[:,16], axis=1)*100
    print(f"  {t:>7}{ua.mean():>14.1f}{100*ua.std()/ua.mean():>8.1f}"
          f"{fa.mean():>13.1f}{100*fa.std()/fa.mean():>8.1f}")
ua_all = np.concatenate([np.linalg.norm(np.array([r["w"] for r in by[t]])[:,12]
                                       -np.array([r["w"] for r in by[t]])[:,14],axis=1) for t in by])*100
print(f"\n  upper arm across ALL poses: {ua_all.mean():.1f} cm, "
      f"range {ua_all.min():.1f} to {ua_all.max():.1f}, cv {100*ua_all.std()/ua_all.mean():.1f}%")
print("  Your humerus did not change length. Any variation is reconstruction error.")

print()
print("="*78)
print("C. The orientation-independent check: a straight arm is straight everywhere.")
print("="*78)
rs = by[0]
w = np.array([r["w"] for r in rs])
l = np.array([[[p[0]*W, p[1]*H, 0.0] for p in r["lm"]] for r in rs])
v3 = np.array([B.elbow_flexion(x, "R") for x in w])
v2 = np.array([B.elbow_flexion(x, "R") for x in l])
print(f"  at a definitionally 0 deg pose (arm fully extended), n={len(rs)}")
print(f"    3D: median {np.nanmedian(v3):5.1f}   range {np.nanmin(v3):5.1f} to {np.nanmax(v3):5.1f}")
print(f"    2D: median {np.nanmedian(v2):5.1f}   range {np.nanmin(v2):5.1f} to {np.nanmax(v2):5.1f}")
print("  This pose has no plane dependence, so no projection argument can excuse 3D.")

print()
print("="*78)
print("D. Linear fit of reported vs true (ideal: slope 1.0, intercept 0)")
print("="*78)
tt = np.array(sorted(by))
for nm, use3d in (("3D world", True), ("2D image", False)):
    got = []
    for t in tt:
        rs = by[t]
        if use3d: arr = np.array([r["w"] for r in rs])
        else:     arr = np.array([[[p[0]*W,p[1]*H,0.0] for p in r["lm"]] for r in rs])
        got.append(np.nanmedian([B.elbow_flexion(x, "R") for x in arr]))
    m, c = np.polyfit(tt, got, 1)
    r2 = np.corrcoef(tt, got)[0,1]**2
    print(f"  {nm}: reported = {m:.3f} x true + {c:+.1f}   (r2 = {r2:.3f})")
print("  A slope well under 1 means real movement is being compressed, not just offset,")
print("  so no single calibration constant can repair it.")
