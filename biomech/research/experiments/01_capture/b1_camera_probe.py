
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from lib.paths import fixture

"""B1: What does this webcam actually support? Vendor specs are unreliable."""
import cv2, time, sys, os

BACKENDS = [("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW)]
MODES = [(320,240),(640,480),(800,600),(1280,720),(1920,1080)]
FOURCCS = ["MJPG", "YUY2"]

def fourcc_str(v):
    v = int(v)
    return "".join(chr((v >> (8*i)) & 0xFF) for i in range(4))

def probe(backend_name, backend, w, h, cc):
    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        acc = fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
        claimed = cap.get(cv2.CAP_PROP_FPS)
        for _ in range(10):      # warm up / let exposure settle
            cap.read()
        stamps = []
        t_end = time.perf_counter() + 2.0
        while time.perf_counter() < t_end:
            ok, f = cap.read()
            if ok:
                stamps.append(time.perf_counter())
        if len(stamps) < 5:
            return None
        dur = stamps[-1] - stamps[0]
        fps = (len(stamps) - 1) / dur if dur > 0 else 0
        gaps = [(stamps[i+1]-stamps[i])*1000 for i in range(len(stamps)-1)]
        gaps_sorted = sorted(gaps)
        return dict(backend=backend_name, req=f"{w}x{h}", req_cc=cc,
                    actual=f"{aw}x{ah}", actual_cc=acc, claimed_fps=claimed,
                    measured_fps=fps,
                    gap_med=gaps_sorted[len(gaps)//2],
                    gap_p95=gaps_sorted[int(len(gaps)*0.95)],
                    frame=frame)
    finally:
        cap.release()

rows = []
for bname, b in BACKENDS:
    for cc in FOURCCS:
        for (w,h) in MODES:
            r = probe(bname, b, w, h, cc)
            if r:
                rows.append(r)
                print(f"{r['backend']:<6} req {r['req']:>9} {cc}  ->  got {r['actual']:>9} {r['actual_cc']}  "
                      f"claimed {r['claimed_fps']:5.1f}  measured {r['measured_fps']:5.1f} fps  "
                      f"gap med {r['gap_med']:5.1f} p95 {r['gap_p95']:5.1f} ms")
            else:
                print(f"{bname:<6} req {w}x{h:<5} {cc}  ->  unsupported")

if rows:
    best = max(rows, key=lambda r: (r['measured_fps'], r['actual']))
    print(f"\nFastest config: {best['backend']} {best['actual']} {best['actual_cc']} @ {best['measured_fps']:.1f} fps")
    cv2.imwrite(fixture("frame_raw.png"), best['frame'])
    print("Saved a fixture frame -> fixtures/raw/frame_raw.png")
