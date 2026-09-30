"""C1: Record the same still pose at four exposure settings.

Answers: does the dark image needed for 30 FPS wreck landmark quality?
Also produces the first replayable fixture: raw video + landmarks for every frame.
Stateless IMAGE mode on purpose - VIDEO mode's tracking would smooth over the
very degradation we are trying to measure.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from lib.paths import fixture, model_path

import cv2, time, json, os, sys
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
os.makedirs(OUT, exist_ok=True)
EXPOSURES = [("auto", None), ("e4", -4), ("e5", -5), ("e6", -6)]
RECORD_SECS, SETUP_SECS = 6.0, 15

NAMES = {11:"shoulderL",12:"shoulderR",13:"elbowL",14:"elbowR",15:"wristL",16:"wristR",
         23:"hipL",24:"hipR",25:"kneeL",26:"kneeR",27:"ankleL",28:"ankleR",
         31:"footL",32:"footR"}

det = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path("lite")),
    running_mode=vision.RunningMode.IMAGE, num_poses=1))

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    sys.exit("camera would not open")

WIN = "recording - press q to abort"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL); cv2.resizeWindow(WIN, 1000, 750)

def show(frame, big, small="", colour=(0,235,0)):
    d = cv2.resize(frame, (1000, 750))
    cv2.rectangle(d, (0,0), (1000,150), (0,0,0), -1)
    cv2.putText(d, big, (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.9, colour, 4)
    if small:
        cv2.putText(d, small, (25, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (215,215,215), 2)
    cv2.imshow(WIN, d)
    return cv2.waitKey(1) & 0xFF

def overlay_pose(frame, lms):
    h, w = frame.shape[:2]
    for i, nm in NAMES.items():
        lm = lms[i]
        c = (0,235,0) if lm.visibility > 0.5 else (0,120,255)
        cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 5, c, -1)
    return frame

# Gate: do not start until the lower body is genuinely visible for a sustained
# period. The previous version only *displayed* a warning and started anyway,
# which produced a run where every knee/ankle number was the model guessing.
GATE_HOLD = 2.5
ok_since = None
t0 = time.time()
while True:
    ok, f = cap.read()
    if not ok: continue
    rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    full = False
    detail = "no person detected"
    if r.pose_landmarks:
        lms = r.pose_landmarks[0]
        f = overlay_pose(f, lms)
        vis = {n: lms[i].visibility for i, n in NAMES.items()}
        worst = min(vis["ankleL"], vis["ankleR"], vis["footL"], vis["footR"])
        full = worst > 0.5
        detail = f"feet confidence {worst:.2f} (need > 0.50)"
    if full:
        ok_since = ok_since or time.time()
        held = time.time() - ok_since
        if held >= GATE_HOLD:
            break
        big, small, col = f"holding... {GATE_HOLD-held:.1f}s", detail, (0,235,0)
    else:
        ok_since = None
        big, small, col = "STEP BACK", detail, (0,120,255)
    if time.time() - t0 > 120:
        sys.exit("gave up waiting for full body after 2 minutes")
    if show(f, big, small, col) == ord('q'):
        sys.exit("aborted")

for n in (3, 2, 1):
    t = time.time()
    while time.time() - t < 1.0:
        ok, f = cap.read()
        if ok: show(f, f"starting in {n}", "STAND STILL, arms slightly out")

records, summary = [], []
for label, ev in EXPOSURES:
    if ev is None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25); cap.set(cv2.CAP_PROP_EXPOSURE, ev)
    t = time.time()
    while time.time() - t < 1.5:
        cap.read(); show(np.zeros((480,640,3), np.uint8), f"switching: {label}", "hold still")

    vw = cv2.VideoWriter(fixture(f"exp_{label}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640,480))
    n = found = 0; bright = []; t_start = time.time()
    while time.time() - t_start < RECORD_SECS:
        ok, f = cap.read()
        if not ok: continue
        n += 1; bright.append(float(f.mean())); vw.write(f)
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        rec = {"exp": label, "t": time.time()-t_start, "found": bool(r.pose_landmarks)}
        disp = f.copy()
        if r.pose_landmarks:
            found += 1
            lms = r.pose_landmarks[0]
            rec["lm"] = [[round(l.x,5),round(l.y,5),round(l.z,5),round(l.visibility,4)] for l in lms]
            if r.pose_world_landmarks:
                rec["w"] = [[round(l.x,5),round(l.y,5),round(l.z,5)] for l in r.pose_world_landmarks[0]]
            disp = overlay_pose(disp, lms)
        records.append(rec)
        left = RECORD_SECS - (time.time()-t_start)
        show(disp, f"{label}  {left:3.1f}s", "STAND STILL, arms slightly out")
    vw.release()
    summary.append((label, n, found, float(np.mean(bright)), n/RECORD_SECS))

cap.release(); cv2.destroyAllWindows(); det.close()

with open(fixture("exposure_session.jsonl"), "w") as fh:
    for r in records: fh.write(json.dumps(r) + "\n")

print(f"{'exposure':<9} {'frames':>7} {'fps':>6} {'detected':>9} {'brightness':>11}")
for label, n, found, br, fps in summary:
    print(f"{label:<9} {n:>7} {fps:>6.1f} {100*found/max(n,1):>8.0f}% {br:>10.1f}")
print(f"\nwrote {len(records)} landmark records -> {OUT}/exposure_session.jsonl")
print(f"wrote 4 raw videos -> {OUT}/exp_*.mp4")
