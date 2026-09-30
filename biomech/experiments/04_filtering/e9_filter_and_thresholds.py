
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from lib.paths import fixture

"""E9: item 8 (filter placement) and item 3 (plane-validity thresholds)."""
import json, numpy as np
from lib import biomech_ref as B, synth
from collections import defaultdict

W,H = 640,480
ph = defaultdict(list)
for line in open(fixture("session.jsonl")):
    r = json.loads(line)
    if r.get("found") and "w" in r: ph[r["phase"]].append(r)

def one_euro(x, t, min_cutoff=1.0, beta=0.01, d_cutoff=1.0):
    def a(c, te): tau = 1/(2*np.pi*c); return 1/(1+tau/te)
    y = np.array(x, dtype=float).copy(); dxp = np.zeros(np.shape(x)[1:]) if np.ndim(x)>1 else 0.0
    for i in range(1, len(x)):
        te = max(t[i]-t[i-1], 1e-4)
        dx = (x[i]-y[i-1])/te
        dxh = a(d_cutoff,te)*dx + (1-a(d_cutoff,te))*dxp
        cut = min_cutoff + beta*np.abs(dxh)
        al = 1/(1+(1/(2*np.pi*cut))/te)
        y[i] = al*x[i] + (1-al)*y[i-1]; dxp = dxh
    return y

print("="*78)
print("Item 8: filter the LANDMARKS, or filter the resulting ANGLE?")
print("="*78)
print(f"  {'measurement':<20}{'phase':<8}{'raw sd':>8}{'filt angle':>12}{'filt lm':>10}{'winner':>9}")
for phase in ("still","squat"):
    rs = ph[phase]
    t = np.array([r["t"] for r in rs])
    lm = np.array([r["w"] for r in rs], dtype=float)
    lm_f = one_euro(lm, t)
    for name, side in [("elbow_flexion","R"), ("knee_flexion","L"),
                       ("hip_flexion","L"), ("ankle_angle","L")]:
        raw  = np.array([B.ALL[name](x, side) for x in lm])
        fa   = one_euro(raw, t)
        fl   = np.array([B.ALL[name](x, side) for x in lm_f])
        if phase == "still":
            a, b, c = np.nanstd(raw), np.nanstd(fa), np.nanstd(fl)
        else:
            d = lambda v: np.nanpercentile(np.abs(np.diff(v)), 95)
            a, b, c = d(raw), d(fa), d(fl)
        win = "landmarks" if c < b else "angle"
        print(f"  {name+' '+side:<20}{phase:<8}{a:>8.2f}{b:>12.2f}{c:>10.2f}{win:>9}")
print("  still = jitter sd; squat = p95 frame-to-frame jump. Lower is better.")

print()
print("="*78)
print("Item 3: at what body rotation does a 2D measurement become unusable?")
print("="*78)
print("  Synthetic, exact truth. Max |error| in degrees across the chart range.")
print(f"  {'measurement':<20}" + "".join(f"{y:>8}" for y in (0,15,30,45,60,75,90)))
def interior2d(P,a,b,c):
    ba,bc = P[a][:2]-P[b][:2], P[c][:2]-P[b][:2]
    return np.degrees(np.arccos(np.clip(np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)),-1,1)))
SW = {"elbow_flexion":("elbow",(11,13,15),[15,30,60,90,120,150]),
      "knee_flexion":("knee",(23,25,27),[15,30,60,90,120])}
limits = {}
for nm,(arg,idx,vals) in SW.items():
    row = f"  {nm:<20}"; per_yaw = {}
    for yaw in (0,15,30,45,60,75,90):
        e = []
        for tv in vals:
            P = synth.project(synth.build(**{arg:tv}, side="L", body_yaw=yaw))
            e.append(abs((180-interior2d(P,*idx)) - tv))
        per_yaw[yaw] = max(e); row += f"{max(e):>8.1f}"
    limits[nm] = per_yaw
    print(row)
print()
print("  Minimum body rotation required to keep 2D error under a budget:")
print(f"  {'measurement':<20}{'<5 deg':>9}{'<10 deg':>10}{'<15 deg':>10}")
for nm, per in limits.items():
    row = f"  {nm:<20}"
    for budget in (5,10,15):
        ok = [y for y,e in per.items() if e <= budget]
        row += f"{(str(min(ok))+' deg') if ok else 'never':>9}" if budget==5 else \
               f"{(str(min(ok))+' deg') if ok else 'never':>10}"
    print(row)
print()
print("  0 deg = facing the camera, 90 deg = fully side-on.")
print("  Sagittal measurements (shoulder/hip flex, ankle) are geometrically")
print("  undefined in 2D near 0 deg - see F20 test D - so they have no valid band there.")
