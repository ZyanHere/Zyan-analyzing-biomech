"""D2: lite vs full vs heavy, replayed through the SAME recorded frames.

Identical input for every model - impossible with a live subject, which is the
whole reason the session was recorded.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from lib.paths import fixture, model_path

import cv2, time, json, numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

def ang(a,b,c):
    ba,bc = a-b,c-b
    na,nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na<1e-9 or nc<1e-9: return np.nan
    return np.degrees(np.arccos(np.clip(np.dot(ba,bc)/(na*nc),-1,1)))

def frames(phase):
    cap = cv2.VideoCapture(fixture(f"sess_{phase}.mp4"))
    out = []
    while True:
        ok, f = cap.read()
        if not ok: break
        out.append(f)
    cap.release(); return out

STILL, ROT = frames("still"), frames("rotate")
print(f"replaying {len(STILL)} still + {len(ROT)} rotate frames through each model\n")

CH = {"elbowR":(12,14,16), "kneeL":(23,25,27), "ankleR":(26,28,32), "shoulderR":(24,12,14)}
res = {}
for variant in ["lite","full","heavy"]:
    det = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path(variant)),
        running_mode=vision.RunningMode.IMAGE, num_poses=1))
    lat, still_w, still_v, rot_w, rot_sw = [], [], [], [], []
    for tag, seq in (("still", STILL), ("rotate", ROT)):
        for f in seq:
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            t0 = time.perf_counter(); r = det.detect(img); lat.append((time.perf_counter()-t0)*1000)
            if not r.pose_landmarks: continue
            lm = r.pose_landmarks[0]
            w = [[l.x,l.y,l.z] for l in r.pose_world_landmarks[0]] if r.pose_world_landmarks else None
            if tag == "still":
                still_w.append(w); still_v.append([l.visibility for l in lm])
            else:
                rot_w.append(w)
                rot_sw.append(np.linalg.norm(np.array([lm[11].x*640, lm[11].y*480])
                                            -np.array([lm[12].x*640, lm[12].y*480])))
    det.close()
    lat.sort()
    jit = {}
    for nm,(a,b,c) in CH.items():
        v = [ang(np.array(w[a]), np.array(w[b]), np.array(w[c])) for w in still_w if w]
        jit[nm] = np.nanstd(v)
    sw = np.array(rot_sw); swmax = np.percentile(sw,95)
    rot = np.degrees(np.arccos(np.clip(sw/swmax,0,1)))
    e = np.array([ang(np.array(w[12]),np.array(w[14]),np.array(w[16])) for w in rot_w if w])
    ref = np.nanmedian(e[rot<12]) if (rot<12).any() else np.nan
    far = np.nanmedian(e[rot>55]) if (rot>55).any() else np.nan
    res[variant] = dict(p50=lat[len(lat)//2], p95=lat[int(len(lat)*.95)],
                        vis=np.mean([np.mean(v) for v in still_v]),
                        ankle_vis=np.mean([ (v[28]+v[32])/2 for v in still_v]),
                        jit=jit, drift=far-ref)

print(f"{'model':<7}{'p50 ms':>8}{'p95 ms':>8}{'mean vis':>10}{'ankle vis':>11}")
for v,r in res.items():
    print(f"{v:<7}{r['p50']:>8.1f}{r['p95']:>8.1f}{r['vis']:>10.2f}{r['ankle_vis']:>11.2f}")

print(f"\n3D jitter while still (deg sd)")
print(f"{'model':<7}" + "".join(f"{k:>11}" for k in CH))
for v,r in res.items():
    print(f"{v:<7}" + "".join(f"{r['jit'][k]:>11.2f}" for k in CH))

print(f"\nrotation drift, elbow held fixed (deg, closer to 0 is better)")
for v,r in res.items():
    print(f"  {v:<7}{r['drift']:>+8.1f}")
