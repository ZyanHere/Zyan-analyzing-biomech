
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from lib.paths import fixture, model_path

"""A1: How fast is each model variant, and does it find a person? A2: do threads help?"""
import time, threading, statistics as st, cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

FIX = fixture("frame_raw.png")
bgr = cv2.imread(FIX)
print(f"fixture: {bgr.shape[1]}x{bgr.shape[0]}  mean brightness {bgr.mean():.1f}/255\n")
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

def make(variant):
    return vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path(variant)),
        running_mode=vision.RunningMode.IMAGE, num_poses=1))

def bench(variant, iters=80):
    t0 = time.perf_counter(); det = make(variant); load_ms = (time.perf_counter()-t0)*1000
    for _ in range(10): det.detect(mp_img)
    lat = []
    for _ in range(iters):
        t = time.perf_counter(); r = det.detect(mp_img); lat.append((time.perf_counter()-t)*1000)
        last = r
    lat.sort()
    found = len(last.pose_landmarks) > 0
    vis = "n/a"
    if found:
        lms = last.pose_landmarks[0]
        key = {"elbowL":13,"wristL":15,"kneeL":25,"ankleL":27,"footL":31}
        vis = " ".join(f"{k}={lms[i].visibility:.2f}" for k,i in key.items())
    det.close()
    return dict(v=variant, load=load_ms, p50=lat[len(lat)//2], p95=lat[int(len(lat)*.95)],
                mean=st.mean(lat), found=found, vis=vis)

print(f"{'model':<7} {'load ms':>8} {'p50 ms':>8} {'p95 ms':>8} {'max fps':>8}  person?")
res = {}
for v in ["lite","full","heavy"]:
    r = bench(v); res[v]=r
    print(f"{r['v']:<7} {r['load']:>8.0f} {r['p50']:>8.2f} {r['p95']:>8.2f} {1000/r['p50']:>8.0f}  {r['found']}")
    if r['found']: print(f"        visibility: {r['vis']}")

print("\n--- A2: does MediaPipe release Python's lock? ---")
def worker(variant, n, out, idx):
    det = make(variant)
    for _ in range(5): det.detect(mp_img)
    t0 = time.perf_counter()
    for _ in range(n): det.detect(mp_img)
    out[idx] = time.perf_counter()-t0
    det.close()

for variant in ["lite","full"]:
    N = 40
    for nt in [1, 4]:
        out = [0.0]*nt
        ts = [threading.Thread(target=worker, args=(variant, N, out, i)) for i in range(nt)]
        t0=time.perf_counter(); [t.start() for t in ts]; [t.join() for t in ts]
        wall = time.perf_counter()-t0
        tput = (N*nt)/wall
        if nt==1: base=tput
        print(f"{variant:<6} {nt} thread(s): {wall:6.2f}s wall, {tput:6.1f} infer/s", end="")
        print(f"   speedup {tput/base:.2f}x" if nt>1 else "   (baseline)")
