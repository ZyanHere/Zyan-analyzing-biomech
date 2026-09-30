"""E6 (item 9): is heavy's -23.6 deg rotation drift real, or an artefact?

F16 estimated body yaw from each model's OWN shoulder width, so each model was
binned on its own ruler. Here every model is binned by a single shared estimate,
and bone-length consistency gives an independent quality check.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from lib.paths import fixture, model_path

import cv2, numpy as np, time
from lib import biomech_ref as B
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

cap = cv2.VideoCapture(fixture("sess_rotate.mp4"))
FR = []
while True:
    ok, f = cap.read()
    if not ok: break
    FR.append(f)
cap.release()
print(f"{len(FR)} frames replayed through each model\n")

out = {}
for variant in ["lite","full","heavy"]:
    det = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path(variant)),
        running_mode=vision.RunningMode.IMAGE, num_poses=1))
    w, sw = [], []
    for f in FR:
        r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
        if not r.pose_landmarks or not r.pose_world_landmarks:
            w.append(None); sw.append(np.nan); continue
        lm = r.pose_landmarks[0]
        w.append(np.array([[l.x,l.y,l.z] for l in r.pose_world_landmarks[0]]))
        sw.append(np.linalg.norm(np.array([lm[11].x*640, lm[11].y*480])
                                -np.array([lm[12].x*640, lm[12].y*480])))
    det.close()
    out[variant] = (w, np.array(sw))

ref_sw = out["full"][1]
yaw = np.degrees(np.arccos(np.clip(ref_sw/np.nanpercentile(ref_sw,95),0,1)))
print("="*72)
print("A. Elbow drift, ALL models binned by the SAME (full-derived) yaw")
print("="*72)
print(f"  {'yaw':>9}{'n':>5}" + "".join(f"{v:>10}" for v in out))
bands = [(0,15),(15,30),(30,45),(45,60),(60,90)]
refs = {}
for v,(w,_) in out.items():
    sel = [w[i] for i,y in enumerate(yaw) if y<15 and w[i] is not None]
    refs[v] = np.nanmedian([B.elbow_flexion(x,"R") for x in sel])
for lo,hi in bands:
    idx = [i for i,y in enumerate(yaw) if lo<=y<hi]
    if len(idx)<5: continue
    row = f"  {str(lo)+'-'+str(hi):>9}{len(idx):>5}"
    for v,(w,_) in out.items():
        vals = [B.elbow_flexion(w[i],"R") for i in idx if w[i] is not None]
        row += f"{np.nanmedian(vals)-refs[v]:>+10.1f}"
    print(row)
print("  (drift in degrees from each model's own facing-camera reference)")

print()
print("="*72)
print("B. Independent quality check: bone-length consistency per model")
print("="*72)
SEG = {"upper arm R":(12,14), "forearm R":(14,16), "femur R":(24,26), "shank R":(26,28)}
print(f"  {'segment':<14}" + "".join(f"{v:>22}" for v in out))
print(f"  {'':<14}" + "".join(f"{'mean cm':>11}{'cv %':>11}" for v in out))
for nm,(a,b) in SEG.items():
    row = f"  {nm:<14}"
    for v,(w,_) in out.items():
        d = np.array([np.linalg.norm(x[a]-x[b]) for x in w if x is not None])*100
        row += f"{d.mean():>11.1f}{100*d.std()/d.mean():>11.1f}"
    print(row)

print()
print("="*72)
print("C. Do the models agree with each other at all? (median elbow, deg)")
print("="*72)
for lo,hi in bands:
    idx = [i for i,y in enumerate(yaw) if lo<=y<hi]
    if len(idx)<5: continue
    vals = {}
    for v,(w,_) in out.items():
        vals[v] = np.nanmedian([B.elbow_flexion(w[i],"R") for i in idx if w[i] is not None])
    spread = max(vals.values())-min(vals.values())
    print(f"  yaw {str(lo)+'-'+str(hi):<7}" + "".join(f"{v}={vals[v]:6.1f}  " for v in vals)
          + f"  spread {spread:5.1f}")
