
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from lib.paths import fixture

"""C5: Replay the session and answer the open questions. No camera involved."""
import json, numpy as np
from collections import defaultdict

W, H = 640, 480
CH = {"elbowL":(11,13,15), "elbowR":(12,14,16), "kneeL":(23,25,27), "kneeR":(24,26,28),
      "shoulderL":(23,11,13), "shoulderR":(24,12,14), "hipL":(11,23,25), "hipR":(12,24,26),
      "ankleL":(25,27,31), "ankleR":(26,28,32)}

def ang(a,b,c):
    ba, bc = a-b, c-b
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na<1e-9 or nc<1e-9: return np.nan
    return np.degrees(np.arccos(np.clip(np.dot(ba,bc)/(na*nc),-1,1)))

def p2(L,i): return np.array([L[i][0]*W, L[i][1]*H])
def p3(Wl,i): return np.array(Wl[i][:3])

ph = defaultdict(list)
for line in open(fixture("session.jsonl")):
    r = json.loads(line)
    if r.get("found"): ph[r["phase"]].append(r)

print("="*74)
print("1. JITTER WHILE STILL - all variation is error (deg std dev)")
print("="*74)
print(f"{'joint':<11}{'2D':>8}{'3D':>8}{'min vis':>9}   verdict")
for nm,(a,b,c) in CH.items():
    rs = ph["still"]
    v2 = [ang(p2(r['lm'],a),p2(r['lm'],b),p2(r['lm'],c)) for r in rs]
    v3 = [ang(p3(r['w'],a),p3(r['w'],b),p3(r['w'],c)) for r in rs if 'w' in r]
    vis = np.mean([min(r['lm'][i][3] for i in (a,b,c)) for r in rs])
    s2, s3 = np.nanstd(v2), np.nanstd(v3)
    verdict = "2D" if s2 < s3 else "3D"
    flag = "" if vis > 0.6 else "  (low confidence)"
    print(f"{nm:<11}{s2:>8.2f}{s3:>8.2f}{vis:>9.2f}   {verdict} quieter{flag}")

print()
print("="*74)
print("2. ROTATION SWEEP - elbow was HELD FIXED, so all change is projection error")
print("="*74)
rs = ph["rotate"]
sw = [np.linalg.norm(p2(r['lm'],11)-p2(r['lm'],12)) for r in rs]
swmax = np.percentile(sw,95)
rot = [np.degrees(np.arccos(np.clip(s/swmax,0,1))) for s in sw]
e2 = [ang(p2(r['lm'],12),p2(r['lm'],14),p2(r['lm'],16)) for r in rs]
e3 = [ang(p3(r['w'],12),p3(r['w'],14),p3(r['w'],16)) for r in rs if 'w' in r]
ref2 = np.nanmedian([v for v,q in zip(e2,rot) if q < 12])
ref3 = np.nanmedian([v for v,q in zip(e3,rot) if q < 12])
print(f"reference (facing camera):  2D {ref2:.1f}deg    3D {ref3:.1f}deg\n")
print(f"{'rotation':>10}{'n':>5}{'2D angle':>10}{'2D error':>10}{'3D angle':>10}{'3D error':>10}")
for lo,hi in [(0,12),(12,25),(25,40),(40,55),(55,70),(70,90)]:
    idx = [i for i,q in enumerate(rot) if lo<=q<hi]
    if len(idx) < 3: continue
    a2 = np.nanmedian([e2[i] for i in idx]); a3 = np.nanmedian([e3[i] for i in idx])
    print(f"{lo:>4}-{hi:<5}{len(idx):>5}{a2:>10.1f}{a2-ref2:>+10.1f}{a3:>10.1f}{a3-ref3:>+10.1f}")

print()
print("="*74)
print("3. MOVEMENT PHASES - does the signal track through a real range?")
print("="*74)
for phase, joints in [("elbow",["elbowR","elbowL"]), ("squat",["kneeL","kneeR","hipL","ankleL"])]:
    rs = ph[phase]
    print(f"\n[{phase}]  {len(rs)} frames")
    print(f"{'joint':<11}{'min':>8}{'max':>8}{'range':>8}{'vis':>7}   frame-to-frame jump p95")
    for nm in joints:
        a,b,c = CH[nm]
        v = np.array([ang(p2(r['lm'],a),p2(r['lm'],b),p2(r['lm'],c)) for r in rs])
        vis = np.mean([min(r['lm'][i][3] for i in (a,b,c)) for r in rs])
        d = np.abs(np.diff(v)); d = d[~np.isnan(d)]
        print(f"{nm:<11}{np.nanmin(v):>8.1f}{np.nanmax(v):>8.1f}{np.nanmax(v)-np.nanmin(v):>8.1f}"
              f"{vis:>7.2f}{np.percentile(d,95):>22.1f}deg")
