"""E5 (item 6): isolate MODEL error. The maths is exact (F19), so anything wrong
in a known pose comes from the pose estimator.

Standing still is a known pose: knees and elbows should be near straight.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from lib.paths import fixture

import json, numpy as np
from lib import biomech_ref as B
from collections import defaultdict

W, H = 640, 480
ph = defaultdict(list)
for line in open(fixture("session.jsonl")):
    r = json.loads(line)
    if r.get("found") and "w" in r: ph[r["phase"]].append(r)

def w3(r):  return np.array(r["w"], dtype=float)
def img(r):
    a = np.array([[p[0]*W, p[1]*H, 0.0] for p in r["lm"]], dtype=float)
    return a

print("="*76)
print("Standing still. Anatomically a relaxed stance is ~0-10 deg at knee and elbow.")
print("="*76)
st = ph["still"]
print(f"  {'joint':<14}{'3D world':>11}{'2D image':>11}{'gap':>8}{'vis':>7}")
for nm, side, idx in [("knee_flexion","L",(23,25,27)), ("knee_flexion","R",(24,26,28)),
                      ("elbow_flexion","L",(11,13,15)), ("elbow_flexion","R",(12,14,16))]:
    v3 = np.nanmedian([B.ALL[nm](w3(r), side) for r in st])
    v2 = np.nanmedian([B.ALL[nm](img(r), side) for r in st])
    vv = np.mean([min(r["lm"][i][3] for i in idx) for r in st])
    print(f"  {nm+' '+side:<14}{v3:>11.1f}{v2:>11.1f}{v3-v2:>+8.1f}{vv:>7.2f}")

print()
print("="*76)
print("Bone-length consistency: a rigid body has constant segment lengths.")
print("Variation is reconstruction error, and it needs no ground truth at all.")
print("="*76)
SEGS = {"upper arm L":(11,13), "forearm L":(13,15), "femur L":(23,25),
        "shank L":(25,27), "foot L":(29,31),
        "upper arm R":(12,14), "forearm R":(14,16), "femur R":(24,26),
        "shank R":(26,28), "foot R":(30,32)}
print(f"  {'segment':<14}{'mean cm':>9}{'sd cm':>8}{'cv %':>7}   {'L/R symmetry':>14}")
means = {}
for nm,(a,b) in SEGS.items():
    d = np.array([np.linalg.norm(w3(r)[a]-w3(r)[b]) for r in st])*100
    means[nm] = d.mean()
    print(f"  {nm:<14}{d.mean():>9.1f}{d.std():>8.2f}{100*d.std()/d.mean():>7.1f}", end="")
    if nm.endswith(" R"):
        l = means.get(nm[:-2]+" L")
        print(f"{100*abs(d.mean()-l)/l:>13.1f}%" if l else "")
    else:
        print()

print()
print("="*76)
print("Does the knee error move with body rotation, or is it a constant offset?")
print("="*76)
rot_rs = ph["rotate"]
sw = np.array([np.linalg.norm(np.array(r["lm"][11][:2])*[W,H] - np.array(r["lm"][12][:2])*[W,H])
               for r in rot_rs])
yaw = np.degrees(np.arccos(np.clip(sw/np.percentile(sw,95),0,1)))
print(f"  {'yaw band':>12}{'n':>5}{'knee L 3D':>11}{'knee R 3D':>11}")
for lo,hi in [(0,15),(15,30),(30,45),(45,60),(60,90)]:
    sel = [r for r,y in zip(rot_rs,yaw) if lo<=y<hi]
    if len(sel) < 5: continue
    kl = np.nanmedian([B.knee_flexion(w3(r),"L") for r in sel])
    kr = np.nanmedian([B.knee_flexion(w3(r),"R") for r in sel])
    print(f"  {str(lo)+'-'+str(hi):>12}{len(sel):>5}{kl:>11.1f}{kr:>11.1f}")
