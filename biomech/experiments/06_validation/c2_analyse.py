"""C2: Standing still, so ALL variation is error. Which exposure is actually usable?

Reports jitter in degrees, because degrees are what the app displays - pixel noise
only matters insofar as it moves an angle.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from lib.paths import fixture

import json, numpy as np
from collections import defaultdict

W, H = 640, 480
NAMES = {11:"shoulderL",12:"shoulderR",13:"elbowL",14:"elbowR",15:"wristL",16:"wristR",
         23:"hipL",24:"hipR",25:"kneeL",26:"kneeR",27:"ankleL",28:"ankleR",31:"footL",32:"footR"}
CHAINS = {"elbowL":(11,13,15), "elbowR":(12,14,16),
          "kneeL":(23,25,27),  "kneeR":(24,26,28),
          "shoulderL":(23,11,13), "shoulderR":(24,12,14),
          "hipL":(11,23,25),   "hipR":(12,24,26),
          "ankleL":(25,27,31), "ankleR":(26,28,32)}

def angle(a, b, c):
    ba, bc = a-b, c-b
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-9 or nc < 1e-9: return np.nan
    return np.degrees(np.arccos(np.clip(np.dot(ba,bc)/(na*nc), -1, 1)))

by_exp = defaultdict(list)
for line in open(fixture("exposure_session.jsonl")):
    r = json.loads(line)
    if r.get("found"): by_exp[r["exp"]].append(r)

print("VISIBILITY - model's own confidence (higher is better)\n")
print(f"{'joint':<10}" + "".join(f"{e:>9}" for e in by_exp))
for idx, nm in NAMES.items():
    row = f"{nm:<10}"
    for e in by_exp:
        v = np.mean([r["lm"][idx][3] for r in by_exp[e]])
        row += f"{v:>9.2f}"
    print(row)

print("\n\nANGLE JITTER while standing still, 2D image coords (degrees std dev, lower is better)\n")
print(f"{'joint':<10}" + "".join(f"{e:>9}" for e in by_exp))
jit2d = {}
for nm, (a,b,c) in CHAINS.items():
    row = f"{nm:<10}"
    for e in by_exp:
        vals = []
        for r in by_exp[e]:
            L = r["lm"]
            P = lambda i: np.array([L[i][0]*W, L[i][1]*H])
            vals.append(angle(P(a), P(b), P(c)))
        s = np.nanstd(vals); jit2d[(nm,e)] = s
        row += f"{s:>9.2f}"
    print(row)

print("\n\nANGLE JITTER, model's 3D world coords (degrees std dev)\n")
print(f"{'joint':<10}" + "".join(f"{e:>9}" for e in by_exp))
for nm, (a,b,c) in CHAINS.items():
    row = f"{nm:<10}"
    for e in by_exp:
        vals = []
        for r in by_exp[e]:
            if "w" not in r: continue
            Wl = r["w"]
            P = lambda i: np.array(Wl[i][:3])
            vals.append(angle(P(a), P(b), P(c)))
        row += f"{np.nanstd(vals):>9.2f}" if vals else f"{'--':>9}"
    print(row)

print("\n\nSUMMARY per exposure\n")
print(f"{'exposure':<10} {'frames':>7} {'mean vis':>9} {'median 2D jitter':>18} {'worst joint':>22}")
for e in by_exp:
    rs = by_exp[e]
    mv = np.mean([r["lm"][i][3] for r in rs for i in NAMES])
    js = {nm: jit2d[(nm,e)] for nm,_ in CHAINS.items()}
    worst = max(js, key=lambda k: js[k])
    print(f"{e:<10} {len(rs):>7} {mv:>9.2f} {np.median(list(js.values())):>17.2f}deg "
          f"{worst+' '+format(js[worst],'.1f')+'deg':>22}")
