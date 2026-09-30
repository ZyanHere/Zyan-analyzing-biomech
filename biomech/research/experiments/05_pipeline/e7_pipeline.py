"""E7 (items 10, 11): does a real threaded pipeline hold 30 FPS, and where is
the bottleneck? Measurement harness, not the application.

Stages: source -> [N inference workers] -> biomech -> sink
Bounded queues, drop-oldest. Per-stage and end-to-end latency recorded.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from lib.paths import fixture, model_path

import cv2, time, threading, queue, numpy as np
from lib import biomech_ref as B
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL = model_path("full")

def load(phase):
    cap = cv2.VideoCapture(fixture(f"sess_{phase}.mp4")); fr=[]
    while True:
        ok,f = cap.read()
        if not ok: break
        fr.append(f)
    cap.release(); return fr

FRAMES = load("squat") + load("elbow")

def run(n_workers, secs=8.0):
    # qout must be unbounded and actively drained: a bounded output queue with no
    # consumer blocks the workers and silently caps throughput at its own size.
    qin = queue.Queue(maxsize=2)
    results, rlock = [], threading.Lock()
    stop = threading.Event()
    drops = [0]

    def source():
        i = 0
        while not stop.is_set():
            f = FRAMES[i % len(FRAMES)]; i += 1
            item = (time.perf_counter(), f)
            try:
                qin.put_nowait(item)
            except queue.Full:
                try: qin.get_nowait(); drops[0]+= 1          # drop OLDEST
                except queue.Empty: pass
                try: qin.put_nowait(item)
                except queue.Full: pass
            time.sleep(1/120)

    def worker():
        det = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL),
            running_mode=vision.RunningMode.IMAGE, num_poses=1))
        while not stop.is_set():
            try: t0, f = qin.get(timeout=0.2)
            except queue.Empty: continue
            ti = time.perf_counter()
            r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                    data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
            t_inf = time.perf_counter() - ti
            tb = time.perf_counter()
            if r.pose_world_landmarks:
                w = np.array([[l.x,l.y,l.z] for l in r.pose_world_landmarks[0]])
                for nm in B.ALL:
                    for side in ("L","R"): B.ALL[nm](w, side)
            t_bio = time.perf_counter() - tb
            with rlock:
                results.append((t0, t_inf, t_bio, time.perf_counter()))
        det.close()

    ts = [threading.Thread(target=source, daemon=True)] + \
         [threading.Thread(target=worker, daemon=True) for _ in range(n_workers)]
    [t.start() for t in ts]
    time.sleep(secs)
    stop.set(); [t.join(timeout=2) for t in ts]

    with rlock:
        rows = list(results)
    if not rows: return None
    e2e = np.array([(r[3]-r[0])*1000 for r in rows])
    inf = np.array([r[1]*1000 for r in rows])
    bio = np.array([r[2]*1000 for r in rows])
    return dict(n=n_workers, fps=len(rows)/secs, drops=drops[0],
                inf50=np.percentile(inf,50), inf95=np.percentile(inf,95),
                bio50=np.percentile(bio,50), bio95=np.percentile(bio,95),
                e2e50=np.percentile(e2e,50), e2e95=np.percentile(e2e,95))

print("="*84)
print("Item 11: full pipeline throughput and latency vs inference worker count")
print("="*84)
print(f"  {'workers':>8}{'FPS':>8}{'drops':>7}{'infer p50':>11}{'p95':>8}"
      f"{'biomech p50':>13}{'e2e p50':>10}{'e2e p95':>10}")
for n in (1,2,3,4,6,8):
    r = run(n)
    if not r: continue
    print(f"  {r['n']:>8}{r['fps']:>8.1f}{r['drops']:>7}{r['inf50']:>11.1f}{r['inf95']:>8.1f}"
          f"{r['bio50']:>13.3f}{r['e2e50']:>10.1f}{r['e2e95']:>10.1f}")
print()
print("  FPS = end-to-end results produced per second. drops = frames discarded")
print("  by the drop-oldest policy. e2e = source timestamp to result, in ms.")
