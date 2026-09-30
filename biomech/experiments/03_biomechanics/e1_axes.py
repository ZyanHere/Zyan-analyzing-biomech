"""E1: What do MediaPipe's world-landmark axes actually mean?

Every anatomical plane derives from this. Assuming a handedness and being wrong
silently flips flexion into extension, so it gets measured, not assumed.

Known facts used as probes, from the recorded session:
  - subject stands upright  -> mid_shoulder is ABOVE mid_hip
  - subject faces the camera in 'still'
  - subject rotates in 'rotate'
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from lib.paths import fixture

import json, numpy as np
from collections import defaultdict

ph = defaultdict(list)
for line in open(fixture("session.jsonl")):
    r = json.loads(line)
    if r.get("found") and "w" in r: ph[r["phase"]].append(r)

def W(r, i): return np.array(r["w"][i])
def N(r, i): return np.array(r["lm"][i][:3])

still = ph["still"]
print("=== axis 1: which world axis is vertical, and which way is up? ===")
d = np.mean([ (W(r,11)+W(r,12))/2 - (W(r,23)+W(r,24))/2 for r in still ], axis=0)
print(f"  mid_shoulder - mid_hip  = [{d[0]:+.3f} {d[1]:+.3f} {d[2]:+.3f}]")
ax = int(np.argmax(np.abs(d)))
print(f"  -> vertical axis is index {ax}, and 'up' is {'+' if d[ax]>0 else '-'}{'xyz'[ax]}")

print("\n=== cross-check with image coords (y grows DOWNWARD in an image) ===")
dn = np.mean([ (N(r,11)+N(r,12))/2 - (N(r,23)+N(r,24))/2 for r in still ], axis=0)
print(f"  same vector in normalised image coords = [{dn[0]:+.3f} {dn[1]:+.3f} {dn[2]:+.3f}]")
print(f"  image y is {dn[1]:+.3f}: negative confirms shoulders are higher up the picture")

print("\n=== axis 2: which axis separates left from right? ===")
h = np.mean([ W(r,23) - W(r,24) for r in still ], axis=0)
print(f"  L_hip - R_hip  = [{h[0]:+.3f} {h[1]:+.3f} {h[2]:+.3f}]")
axh = int(np.argmax(np.abs(h)))
print(f"  -> lateral axis is index {axh}; subject's LEFT is {'+' if h[axh]>0 else '-'}{'xyz'[axh]}")

print("\n=== axis 3: is the remaining axis anterior (forward) or posterior? ===")
rem = [i for i in (0,1,2) if i not in (ax, axh)][0]
nose_off = np.mean([ W(r,0)[rem] - ((W(r,23)+W(r,24))/2)[rem] for r in still ])
print(f"  nose offset from hip centre on axis {'xyz'[rem]} = {nose_off:+.4f}")
ear = np.mean([ (W(r,7)+W(r,8))/2 - W(r,0) for r in still ], axis=0)
print(f"  ear_centre - nose = [{ear[0]:+.3f} {ear[1]:+.3f} {ear[2]:+.3f}]")
print(f"  ears sit BEHIND the nose, so anterior is {'-' if ear[rem]>0 else '+'}{'xyz'[rem]}")

print("\n=== handedness of the resulting frame ===")
up = np.zeros(3); up[ax] = 1.0 if d[ax] > 0 else -1.0
left = np.zeros(3); left[axh] = 1.0 if h[axh] > 0 else -1.0
ant = np.zeros(3); ant[rem] = -1.0 if ear[rem] > 0 else 1.0
print(f"  up={up}  left={left}  anterior={ant}")
print(f"  cross(left, up) = {np.cross(left, up)}   (should equal anterior if right-handed)")
print(f"  dot with anterior = {np.dot(np.cross(left, up), ant):+.1f}")

print("\n=== does the frame rotate with the body, or stay camera-locked? ===")
print("  if world landmarks were body-locked, hip vector would NOT change during 'rotate'")
for tag in ("still", "rotate"):
    v = np.array([ W(r,23) - W(r,24) for r in ph[tag] ])
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    spread = np.degrees(np.arccos(np.clip(v @ v.mean(0)/np.linalg.norm(v.mean(0)), -1, 1)))
    print(f"  {tag:<7} hip-vector angular spread: mean {spread.mean():5.1f} deg, max {spread.max():5.1f} deg")
