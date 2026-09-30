"""C4: The master fixture recording. Everything downstream replays this.

Four phases:
  still    - all variation is error (jitter baseline, full body)
  elbow    - does a tracked angle move smoothly through its range?
  squat    - knee / hip / ankle under real motion
  rotate   - elbow held FIXED while the body turns. True angle is constant,
             so all measured change is monocular projection error. This is
             ground truth obtained from the subject's own body, no equipment.

Saves 2D landmarks, 3D world landmarks and raw video per phase.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from lib.paths import fixture, model_path

import cv2, time, json, os, sys
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


PHASES = [
    ("still",  6.0, "STAND STILL", "face camera, arms slightly away from body"),
    ("elbow", 12.0, "BEND + STRAIGHTEN RIGHT ELBOW", "slowly, full range, keep facing camera"),
    ("squat", 12.0, "SLOW SQUAT, DOWN AND UP", "stay side-on to camera if you can"),
    ("rotate",12.0, "HOLD ELBOW AT 90, TURN SLOWLY", "keep the arm locked - turn your whole body"),
]
NAMES = {11:"shoulderL",12:"shoulderR",13:"elbowL",14:"elbowR",15:"wristL",16:"wristR",
         23:"hipL",24:"hipR",25:"kneeL",26:"kneeR",27:"ankleL",28:"ankleR",31:"footL",32:"footR"}

det = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path("lite")),
    running_mode=vision.RunningMode.IMAGE, num_poses=1))

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25); cap.set(cv2.CAP_PROP_EXPOSURE, -5)
if not cap.isOpened(): sys.exit("camera would not open")

WIN = "session recorder - q aborts"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL); cv2.resizeWindow(WIN, 1100, 820)

def draw(frame, lms, big, small, col=(0,235,0)):
    f = frame.copy(); h, w = f.shape[:2]
    if lms is not None:
        for i in NAMES:
            lm = lms[i]
            c = (0,235,0) if lm.visibility > 0.5 else (0,120,255)
            cv2.circle(f, (int(lm.x*w), int(lm.y*h)), 4, c, -1)
    d = cv2.resize(f, (1100, 820))
    cv2.rectangle(d, (0,0), (1100,165), (0,0,0), -1)
    cv2.putText(d, big, (20,65), cv2.FONT_HERSHEY_SIMPLEX, 1.25, col, 3)
    cv2.putText(d, small, (20,120), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (210,210,210), 2)
    cv2.imshow(WIN, d)
    return cv2.waitKey(1) & 0xFF

def detect(f):
    r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    return (r.pose_landmarks[0] if r.pose_landmarks else None,
            r.pose_world_landmarks[0] if r.pose_world_landmarks else None)

ok_since, t0 = None, time.time()
while True:
    ok, f = cap.read()
    if not ok: continue
    lms, _ = detect(f)
    full, detail = False, "no person detected"
    if lms is not None:
        worst = min(lms[i].visibility for i in (27,28,31,32))
        full = worst > 0.5
        detail = f"feet confidence {worst:.2f} (need > 0.50)"
    if full:
        ok_since = ok_since or time.time()
        held = time.time() - ok_since
        if held >= 2.5: break
        k = draw(f, lms, f"holding... {2.5-held:.1f}s", detail)
    else:
        ok_since = None
        k = draw(f, lms, "STEP BACK", detail, (0,120,255))
    if k == ord('q'): sys.exit("aborted")
    if time.time()-t0 > 180: sys.exit("timed out waiting for full body")

records, summary = [], []
for name, secs, big, small in PHASES:
    for n in (3,2,1):
        t=time.time()
        while time.time()-t < 1.0:
            ok,f = cap.read()
            if ok: draw(f, None, f"{big}", f"starting in {n}...", (0,200,255))
    vw = cv2.VideoWriter(fixture(f"sess_{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640,480))
    n_f = found = 0; t_start = time.time()
    while time.time()-t_start < secs:
        ok, f = cap.read()
        if not ok: continue
        n_f += 1; vw.write(f)
        lms, wlm = detect(f)
        rec = {"phase": name, "t": round(time.time()-t_start,4), "found": lms is not None}
        if lms is not None:
            found += 1
            rec["lm"] = [[round(l.x,5),round(l.y,5),round(l.z,5),round(l.visibility,4)] for l in lms]
            if wlm: rec["w"] = [[round(l.x,5),round(l.y,5),round(l.z,5)] for l in wlm]
        records.append(rec)
        if draw(f, lms, f"{big}   {secs-(time.time()-t_start):3.1f}s", small) == ord('q'):
            sys.exit("aborted")
    vw.release()
    summary.append((name, n_f, found, n_f/secs))

cap.release(); cv2.destroyAllWindows(); det.close()
with open(fixture("session.jsonl"),"w") as fh:
    for r in records: fh.write(json.dumps(r)+"\n")

print(f"\n{'phase':<8} {'frames':>7} {'fps':>6} {'detected':>9}")
for nm, n_f, fnd, fps in summary:
    print(f"{nm:<8} {n_f:>7} {fps:>6.1f} {100*fnd/max(n_f,1):>8.0f}%")
print(f"\nwrote {len(records)} records -> {fixture("session.jsonl")}")
print(f"wrote 4 videos -> fixtures/sessions/sess_*.mp4")
