"""F2 (items 12, 13): what breaks the measurements, and how sensitive is setup?

Self-paced: press SPACE to start each phase, it records 5s, then moves on.
Records landmarks plus the candidate validity signals (visibility, bone-length
variance, estimated body yaw) so we can test whether failures are DETECTABLE,
not merely observable after the fact.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from lib.paths import fixture, model_path

import cv2, time, json, numpy as np
from lib import biomech_ref as B
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

W, H, HOLD = 640, 480, 5.0
PHASES = [
    ("baseline",    "STAND NORMALLY, full body, facing camera"),
    ("occl_arm",    "PUT YOUR RIGHT ARM BEHIND YOUR BACK"),
    ("occl_leg",    "STAND WITH ONE LEG BEHIND THE OTHER, side-on"),
    ("edge",        "STEP TO THE LEFT EDGE, half out of frame"),
    ("partial",     "STEP CLOSE so your legs are out of frame"),
    ("fast",        "WAVE BOTH ARMS AS FAST AS YOU CAN"),
    ("rotated",     "TURN FULLY SIDE-ON to the camera"),
    ("seated",      "SIT DOWN, side-on to the camera"),
    ("far",         "STAND AS FAR BACK as the room allows"),
    ("near",        "STAND CLOSE, about half your normal distance"),
    ("cam_low",     "PUT THE LAPTOP ON THE FLOOR, then stand normally"),
    ("cam_high",    "RAISE THE LAPTOP as high as you can, stand normally"),
]

det = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path("full")),
    running_mode=vision.RunningMode.IMAGE, num_poses=1))
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25); cap.set(cv2.CAP_PROP_EXPOSURE, -5)
if not cap.isOpened(): sys.exit("camera would not open")
cv2.namedWindow("robustness", cv2.WINDOW_NORMAL); cv2.resizeWindow("robustness", 1050, 800)

def detect(f):
    r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    if not r.pose_landmarks: return None, None
    return r.pose_landmarks[0], (r.pose_world_landmarks[0] if r.pose_world_landmarks else None)

def show(f, lm, big, small, col=(0,235,0)):
    d = f.copy()
    if lm is not None:
        for i in range(33):
            if i in (11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32):
                cv2.circle(d,(int(lm[i].x*W),int(lm[i].y*H)),4,
                           (0,235,0) if lm[i].visibility>.5 else (0,120,255),-1)
    d = cv2.resize(d,(1050,800)); cv2.rectangle(d,(0,0),(1050,150),(0,0,0),-1)
    cv2.putText(d,big,(20,58),cv2.FONT_HERSHEY_SIMPLEX,1.0,col,2)
    cv2.putText(d,small,(20,112),cv2.FONT_HERSHEY_SIMPLEX,0.75,(210,210,210),2)
    cv2.imshow("robustness",d); return cv2.waitKey(1)&0xFF

recs = []
for name, instruction in PHASES:
    while True:
        ok,f = cap.read()
        if not ok: continue
        lm,_ = detect(f)
        k = show(f, lm, instruction, "press SPACE when ready   |   s = skip   q = quit", (0,200,255))
        if k == ord(' '): break
        if k == ord('s'): name = None; break
        if k == ord('q'): cap.release(); cv2.destroyAllWindows(); sys.exit("aborted")
    if name is None: continue
    t0 = time.time(); n = 0
    while time.time()-t0 < HOLD:
        ok,f = cap.read()
        if not ok: continue
        lm,wl = detect(f)
        rec = {"phase":name, "t":time.time()-t0, "found": lm is not None}
        if lm is not None:
            n += 1
            rec["lm"] = [[l.x,l.y,l.z,l.visibility] for l in lm]
            if wl: rec["w"] = [[l.x,l.y,l.z] for l in wl]
        recs.append(rec)
        show(f, lm, f"{name}   {HOLD-(time.time()-t0):3.1f}s", f"{n} frames")

cap.release(); cv2.destroyAllWindows(); det.close()
with open(fixture("robustness_session.jsonl"),"w") as fh:
    for r in recs: fh.write(json.dumps(r)+"\n")

print()
print("="*94)
print("Detectability: do our validity signals FIRE when the measurement is bad?")
print("="*94)
print(f"  {'phase':<11}{'det%':>6}{'minvis':>8}{'knee L':>9}{'knee R':>9}"
      f"{'|L-R| bone':>12}{'femur cv%':>11}{'yaw est':>9}")
byp = {}
for r in recs: byp.setdefault(r["phase"], []).append(r)
for name,_ in PHASES:
    rs = byp.get(name, [])
    if not rs: continue
    ok = [r for r in rs if r.get("found") and "w" in r]
    if not ok:
        print(f"  {name:<11}{0:>5.0f}%{'--':>8}{'--':>9}{'--':>9}{'--':>12}{'--':>11}{'--':>9}")
        continue
    w = np.array([r["w"] for r in ok])
    mv = np.mean([min(r["lm"][i][3] for i in (25,26,27,28)) for r in ok])
    kl = np.nanmedian([B.knee_flexion(x,"L") for x in w])
    kr = np.nanmedian([B.knee_flexion(x,"R") for x in w])
    fl = np.linalg.norm(w[:,23]-w[:,25],axis=1); fr = np.linalg.norm(w[:,24]-w[:,26],axis=1)
    asym = 100*abs(fl.mean()-fr.mean())/fr.mean(); cv = 100*fl.std()/fl.mean()
    sw = np.array([np.linalg.norm(np.array(r["lm"][11][:2])-np.array(r["lm"][12][:2])) for r in ok])
    yaw = np.degrees(np.arccos(np.clip(sw.mean()/np.nanpercentile(sw,95),0,1)))
    print(f"  {name:<11}{100*len(ok)/len(rs):>5.0f}%{mv:>8.2f}{kl:>9.1f}{kr:>9.1f}"
          f"{asym:>11.1f}%{cv:>11.1f}{yaw:>9.1f}")
print()
print("  A good validity signal moves when the measurement goes wrong.")
print("  Compare every row against 'baseline'.")
print(f"\n  wrote {len(recs)} frames -> {OUT}/robustness_session.jsonl")
