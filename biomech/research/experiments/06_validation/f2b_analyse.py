"""F2b (items 12, 13): re-analysed in 2D, the representation F29 selected.

Two questions:
  DETECTABILITY - when a measurement goes wrong, does some signal we can compute
                  at runtime actually move? A failure we cannot detect is the
                  dangerous kind.
  SENSITIVITY   - how much do distance and camera geometry shift a reading?
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from lib.paths import fixture

import json, numpy as np
from lib import biomech_ref as B
from collections import defaultdict

W, H = 640, 480
by = defaultdict(list)
for line in open(fixture("robustness_session.jsonl")):
    r = json.loads(line); by[r["phase"]].append(r)

def img(r): return np.array([[p[0]*W, p[1]*H, 0.0] for p in r["lm"]])
def vis(r, idx): return min(r["lm"][i][3] for i in idx)

ARM = (12, 14, 16); LEG = (24, 26, 28)
ORDER = ["baseline","occl_arm","occl_leg","partial","fast","rotated","seated",
         "far","near","cam_low","cam_high"]

def stats(name):
    rs = by.get(name, [])
    found = [r for r in rs if r.get("found") and "lm" in r]
    if not found: return None
    L = np.array([img(r) for r in found])
    d = dict(det=100*len(found)/len(rs))
    d["vis_arm"] = np.mean([vis(r, ARM) for r in found])
    d["vis_leg"] = np.mean([vis(r, LEG) for r in found])
    d["elbow"] = np.nanmedian([B.elbow_flexion(x,"R") for x in L])
    d["knee"]  = np.nanmedian([B.knee_flexion(x,"R")  for x in L])
    d["elbow_sd"] = np.nanstd([B.elbow_flexion(x,"R") for x in L])
    d["knee_sd"]  = np.nanstd([B.knee_flexion(x,"R")  for x in L])
    sw = np.array([np.linalg.norm(x[11][:2]-x[12][:2]) for x in L])
    d["shoulder_px"] = np.median(sw)
    fl = np.array([np.linalg.norm(x[23][:2]-x[25][:2]) for x in L])
    fr = np.array([np.linalg.norm(x[24][:2]-x[26][:2]) for x in L])
    d["bone_asym"] = 100*abs(np.median(fl)-np.median(fr))/np.median(fr)
    d["bone_cv"] = 100*np.std(fr)/np.mean(fr)
    return d

base = stats("baseline")
print("="*100)
print("A. DETECTABILITY - does any runtime signal move when the measurement breaks?")
print("="*100)
print(f"  {'phase':<11}{'det%':>6}{'vis arm':>9}{'vis leg':>9}{'elbow':>8}{'knee':>8}"
      f"{'knee sd':>9}{'bone asym':>11}{'bone cv':>9}")
for nm in ORDER:
    d = stats(nm)
    if not d: continue
    print(f"  {nm:<11}{d['det']:>5.0f}%{d['vis_arm']:>9.2f}{d['vis_leg']:>9.2f}"
          f"{d['elbow']:>8.1f}{d['knee']:>8.1f}{d['knee_sd']:>9.1f}"
          f"{d['bone_asym']:>10.1f}%{d['bone_cv']:>9.1f}")

print()
print("="*100)
print("B. Which signal catches which failure? (deviation from baseline)")
print("="*100)
TH = dict(vis=0.60, asym=8.0, cv=6.0, det=90.0)
print(f"  {'phase':<11}{'knee err':>10}   {'caught by':<46}{'verdict':>10}")
for nm in ORDER:
    if nm == "baseline": continue
    d = stats(nm)
    if not d: continue
    err = d["knee"] - base["knee"]
    fired = []
    if d["det"] < TH["det"]:        fired.append("detection rate")
    if d["vis_leg"] < TH["vis"]:    fired.append("visibility")
    if d["bone_asym"] > TH["asym"]: fired.append("bone asymmetry")
    if d["bone_cv"] > TH["cv"]:     fired.append("bone variance")
    bad = abs(err) > 15
    verdict = ("ok" if not bad else ("CAUGHT" if fired else "*** MISSED ***"))
    print(f"  {nm:<11}{err:>+10.1f}   {', '.join(fired) if fired else '-':<46}{verdict:>10}")

print()
print("="*100)
print("C. SENSITIVITY (item 13) - does the setup itself shift the reading?")
print("="*100)
print(f"  {'phase':<11}{'shoulder px':>13}{'elbow':>9}{'vs base':>9}{'knee':>9}{'vs base':>9}")
for nm in ["baseline","near","far","cam_low","cam_high"]:
    d = stats(nm)
    if not d: continue
    print(f"  {nm:<11}{d['shoulder_px']:>13.0f}{d['elbow']:>9.1f}{d['elbow']-base['elbow']:>+9.1f}"
          f"{d['knee']:>9.1f}{d['knee']-base['knee']:>+9.1f}")
print("\n  shoulder px is a proxy for apparent size, i.e. distance from the camera.")
