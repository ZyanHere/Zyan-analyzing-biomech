"""D1: Which filter? Jitter and lag are a trade-off; measure both on the SAME data.

Replays the recorded session. No camera. Deterministic - same answer every run,
which is the entire point of having recorded it.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from lib.paths import fixture

import json, numpy as np
from collections import defaultdict

W,H = 640,480
def ang(a,b,c):
    ba,bc = a-b, c-b
    na,nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na<1e-9 or nc<1e-9: return np.nan
    return np.degrees(np.arccos(np.clip(np.dot(ba,bc)/(na*nc),-1,1)))

ph = defaultdict(list)
for line in open(fixture("session.jsonl")):
    r = json.loads(line)
    if r.get("found"): ph[r["phase"]].append(r)

def series(phase, chain, use3d=True):
    out, ts = [], []
    for r in ph[phase]:
        src = r.get("w") if use3d else r["lm"]
        if src is None: continue
        P = (lambda i: np.array(src[i][:3])) if use3d else (lambda i: np.array([src[i][0]*W, src[i][1]*H]))
        out.append(ang(P(chain[0]), P(chain[1]), P(chain[2]))); ts.append(r["t"])
    return np.array(out), np.array(ts)

def ema(x, a):
    y = np.empty_like(x); y[0] = x[0]
    for i in range(1,len(x)): y[i] = a*x[i] + (1-a)*y[i-1]
    return y

def median_f(x, k):
    y = x.copy()
    for i in range(len(x)):
        y[i] = np.median(x[max(0,i-k+1):i+1])
    return y

def one_euro(x, t, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
    def alpha(cut, te):
        tau = 1.0/(2*np.pi*cut)
        return 1.0/(1.0 + tau/te)
    y = np.empty_like(x); y[0] = x[0]; dx_prev = 0.0
    for i in range(1,len(x)):
        te = max(t[i]-t[i-1], 1e-4)
        dx = (x[i]-y[i-1])/te
        dx_hat = alpha(d_cutoff,te)*dx + (1-alpha(d_cutoff,te))*dx_prev
        cutoff = min_cutoff + beta*abs(dx_hat)
        a = alpha(cutoff, te)
        y[i] = a*x[i] + (1-a)*y[i-1]
        dx_prev = dx_hat
    return y

def lag_ms(raw, filt, ts):
    """Filtered output trails the input: filt[t] ~= raw[t-k]. Find k by normalised
    correlation (Pearson), otherwise shrinking overlap biases every answer to zero."""
    raw = np.asarray(raw, float); filt = np.asarray(filt, float)
    ok = ~(np.isnan(raw) | np.isnan(filt))
    raw, filt = raw[ok], filt[ok]
    if len(raw) < 10: return float("nan")
    best_k, best_c = 0, -2.0
    for k in range(0, min(30, len(raw)//3)):
        a = filt[k:]; b = raw[:len(raw)-k] if k else raw
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        sa, sb = a.std(), b.std()
        if sa < 1e-9 or sb < 1e-9: continue
        c = float(np.mean((a-a.mean())*(b-b.mean()))/(sa*sb))
        if c > best_c: best_c, best_k = c, k
    return best_k * float(np.median(np.diff(ts))) * 1000

FILTERS = {
    "raw":            lambda x,t: x,
    "EMA a=0.5":      lambda x,t: ema(x,0.5),
    "EMA a=0.3":      lambda x,t: ema(x,0.3),
    "EMA a=0.15":     lambda x,t: ema(x,0.15),
    "median k=5":     lambda x,t: median_f(x,5),
    "1euro b=0.001":  lambda x,t: one_euro(x,t,1.0,0.001),
    "1euro b=0.01":   lambda x,t: one_euro(x,t,1.0,0.01),
    "1euro b=0.05":   lambda x,t: one_euro(x,t,1.0,0.05),
}
CH = {"elbowR":(12,14,16), "kneeL":(23,25,27)}

for jname, chain in CH.items():
    still,_   = series("still", chain)
    mv, mvt   = series("elbow" if "elbow" in jname else "squat", chain)
    print("="*78)
    print(f"{jname}   jitter from 'still' (deg sd)   lag + tracking from movement phase")
    print("="*78)
    print(f"{'filter':<16}{'jitter':>9}{'vs raw':>9}{'lag ms':>9}{'p95 jump':>10}{'range kept':>12}")
    raw_j = np.nanstd(still); raw_range = np.nanmax(mv)-np.nanmin(mv)
    for name, fn in FILTERS.items():
        s = fn(still, np.arange(len(still))*0.05)
        m = fn(mv, mvt)
        j = np.nanstd(s)
        d = np.abs(np.diff(m)); d = d[~np.isnan(d)]
        rng = np.nanmax(m)-np.nanmin(m)
        print(f"{name:<16}{j:>9.2f}{100*j/raw_j-100:>8.0f}%{lag_ms(mv,m,mvt):>9.0f}"
              f"{np.percentile(d,95):>9.1f}d{100*rng/raw_range:>11.0f}%")
    print()
