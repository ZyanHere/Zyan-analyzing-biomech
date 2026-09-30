"""C3: Interactive framing finder. What configuration can this camera actually deliver?

Move around, change resolution, and watch per-region confidence live.
Records the best configuration seen for each body region.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from lib.paths import fixture, model_path

import cv2, time, sys, json
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

RESOS = [(320,240), (640,480), (1280,720)]
UPPER = {"shoulderL":11,"shoulderR":12,"elbowL":13,"elbowR":14,"wristL":15,"wristR":16}
LOWER = {"kneeL":25,"kneeR":26,"ankleL":27,"ankleR":28,"footL":31,"footR":32}

det = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path("lite")),
    running_mode=vision.RunningMode.IMAGE, num_poses=1))

ri = 1
def open_cam(i):
    c = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    c.set(cv2.CAP_PROP_FRAME_WIDTH, RESOS[i][0]); c.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOS[i][1])
    c.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25); c.set(cv2.CAP_PROP_EXPOSURE, -5)
    return c

cap = open_cam(ri)
if not cap.isOpened(): sys.exit("camera would not open")
WIN = "framing finder"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL); cv2.resizeWindow(WIN, 1100, 800)
best = {}
print("keys:  1/2/3 resolution   s = save this config   q = quit")

while True:
    ok, f = cap.read()
    if not ok: continue
    h, w = f.shape[:2]
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    up = lo = 0.0; px_h = 0
    if r.pose_landmarks:
        lms = r.pose_landmarks[0]
        up = min(lms[i].visibility for i in UPPER.values())
        lo = min(lms[i].visibility for i in LOWER.values())
        ys = [lms[i].y for i in list(UPPER.values())+list(LOWER.values())]
        px_h = int((max(ys)-min(ys)) * h)
        for i in list(UPPER.values())+list(LOWER.values()):
            lm = lms[i]
            c = (0,235,0) if lm.visibility > 0.5 else (0,120,255)
            cv2.circle(f, (int(lm.x*w), int(lm.y*h)), 4, c, -1)
    d = cv2.resize(f, (1100, 800))
    cv2.rectangle(d, (0,0), (1100,175), (0,0,0), -1)
    cv2.putText(d, f"{w}x{h}   body height {px_h}px", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 3)
    for j,(lbl,v) in enumerate([("UPPER", up), ("LOWER", lo)]):
        col = (0,235,0) if v > 0.5 else (0,120,255)
        cv2.putText(d, f"{lbl} {v:.2f}", (20+j*330, 115), cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 3)
    cv2.putText(d, "1/2/3 res    s save    q quit", (20,160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,200), 2)
    cv2.imshow(WIN, d)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'): break
    if k in (ord('1'),ord('2'),ord('3')):
        ri = k - ord('1'); cap.release(); cap = open_cam(ri)
    if k == ord('s'):
        key = f"{w}x{h}"
        rec = {"res": key, "px_height": px_h, "upper": round(up,3), "lower": round(lo,3)}
        best[f"{key}@{px_h}px"] = rec
        print(f"saved: {key}  body {px_h}px  upper {up:.2f}  lower {lo:.2f}  "
              f"{'BOTH OK' if up>0.5 and lo>0.5 else 'upper only' if up>0.5 else 'neither'}")

cap.release(); cv2.destroyAllWindows(); det.close()
if best:
    json.dump(best, open(fixture("framing.json"),"w"), indent=2)
    print(f"\nwrote {len(best)} configs -> fixtures/robustness/framing.json")
