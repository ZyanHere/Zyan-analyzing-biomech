"""E12: the three checks required before freezing the architecture.

1. Does the orientation estimate chatter at the 30/45 deg thresholds?
2. IMAGE vs VIDEO running mode - does temporal tracking beat two workers?
3. Does the scale-normalised yaw survive posture change at one orientation?
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from lib.paths import fixture, model_path

import sys
import json
import math
import time
import numpy as np
import cv2
from collections import defaultdict
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from lib import biomech_ref as B

W, H = 640, 480
MODEL = model_path("full")


def load(path, key):
    by = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if r.get("lm"):
            by[r[key]].append(r)
    return by


def px(r, i):
    return np.array([r["lm"][i][0] * W, r["lm"][i][1] * H])


def ratio_of(r):
    sw = np.linalg.norm(px(r, 11) - px(r, 12))
    mid_sh = (px(r, 11) + px(r, 12)) / 2
    mid_hp = (px(r, 23) + px(r, 24)) / 2
    return float(sw / (np.linalg.norm(mid_sh - mid_hp) + 1e-9))


print("=" * 86)
print("1. Does the orientation estimate CHATTER at a threshold?")
print("=" * 86)
rob = load(fixture("robustness_session.jsonl"), "phase")
anch = load(fixture("anchored.jsonl"), "key")
all_r = {**{f"rob/{k}": v for k, v in rob.items()},
         **{f"anch/{k}": v for k, v in anch.items()}}
rmax = max(ratio_of(r) for rs in all_r.values() for r in rs)


def yaw_series(rs):
    return np.array([math.degrees(math.acos(min(max(ratio_of(r) / rmax, 0.0), 1.0)))
                     for r in rs])


print(f"  {'phase':<16}{'yaw mean':>10}{'frame sd':>10}{'p95 jump':>10}"
      f"{'raw flips':>11}{'+hysteresis':>13}")


def count_flips(y, lo, hi, hyst=0.0, hold=1):
    """State flips crossing a threshold band, optionally with hysteresis+hold."""
    state, flips, streak, pending = None, 0, 0, None
    for v in y:
        if v >= hi + hyst:
            s = "sagittal"
        elif v <= lo - hyst:
            s = "frontal"
        elif state is not None and (lo <= v <= hi or True):
            s = state          # inside the band or the hysteresis margin: keep state
        else:
            s = "neither"
        if s != state:
            if s == pending:
                streak += 1
            else:
                pending, streak = s, 1
            if streak >= hold:
                state, flips, streak, pending = s, flips + 1, 0, None
        else:
            pending, streak = None, 0
    return max(flips - 1, 0)


worst = 0
for name, rs in all_r.items():
    if len(rs) < 20:
        continue
    y = yaw_series(rs)
    d = np.abs(np.diff(y))
    raw = count_flips(y, 30, 45)
    hys = count_flips(y, 30, 45, hyst=5.0, hold=5)
    worst = max(worst, raw)
    print(f"  {name:<16}{y.mean():>10.1f}{y.std():>10.2f}{np.percentile(d, 95):>10.2f}"
          f"{raw:>11}{hys:>13}")
print(f"\n  'raw flips' = state changes with a bare 30/45 threshold.")
print(f"  '+hysteresis' = 5 deg margin, state must persist 5 frames (~0.25 s).")

print()
print("=" * 86)
print("2. IMAGE vs VIDEO running mode: does temporal tracking beat two workers?")
print("=" * 86)


def frames(phase):
    cap = cv2.VideoCapture(fixture(f"sess_{phase}.mp4"))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    cap.release()
    return out


CH = {"elbowR": (12, 14, 16), "kneeL": (23, 25, 27), "ankleR": (26, 28, 32)}
for phase in ("still", "squat"):
    FR = frames(phase)
    print(f"\n  [{phase}]  {len(FR)} frames")
    print(f"  {'mode':<8}{'p50 ms':>9}{'p95 ms':>9}" +
          "".join(f"{k:>10}" for k in CH))
    for mode in ("IMAGE", "VIDEO"):
        opts = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL),
            running_mode=getattr(vision.RunningMode, mode), num_poses=1)
        det = vision.PoseLandmarker.create_from_options(opts)
        lat, sets = [], []
        for i, f in enumerate(FR):
            img = mp.Image(image_format=mp.ImageFormat.SRGB,
                           data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            t0 = time.perf_counter()
            if mode == "IMAGE":
                r = det.detect(img)
            else:
                r = det.detect_for_video(img, int(i * 1000 / 30))
            lat.append((time.perf_counter() - t0) * 1000)
            if r.pose_landmarks:
                sets.append(np.array([[l.x * W, l.y * H, 0.0] for l in r.pose_landmarks[0]]))
        det.close()
        lat.sort()
        row = f"  {mode:<8}{lat[len(lat)//2]:>9.1f}{lat[int(len(lat)*.95)]:>9.1f}"
        for name, (a, b, c) in CH.items():
            fn = B.elbow_flexion if "elbow" in name else (
                B.knee_flexion if "knee" in name else B.ankle_angle_2d)
            side = "R" if name.endswith("R") else "L"
            v = [fn(x, side) for x in sets]
            stat = np.nanstd(v) if phase == "still" else np.nanpercentile(
                np.abs(np.diff(v)), 95)
            row += f"{stat:>10.2f}"
        print(row)
    print("  still -> jitter sd (deg); squat -> p95 frame-to-frame jump (deg)")

print()
print("=" * 86)
print("2b. Does VIDEO mode survive a SUBSAMPLED stream, as two workers would see?")
print("=" * 86)
FR = frames("squat")
print(f"  {'stream':<28}{'p95 jump kneeL':>18}")
for label, step in (("every frame (1 worker)", 1), ("every 2nd frame (2 workers)", 2)):
    det = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL),
        running_mode=vision.RunningMode.VIDEO, num_poses=1))
    sets = []
    for i in range(0, len(FR), step):
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=cv2.cvtColor(FR[i], cv2.COLOR_BGR2RGB))
        r = det.detect_for_video(img, int(i * 1000 / 30))
        if r.pose_landmarks:
            sets.append(np.array([[l.x * W, l.y * H, 0.0] for l in r.pose_landmarks[0]]))
    det.close()
    v = [B.knee_flexion(x, "L") for x in sets]
    print(f"  {label:<28}{np.nanpercentile(np.abs(np.diff(v)), 95):>18.2f}")
print("  Each worker in a 2-worker pool sees roughly every 2nd frame.")

print()
print("=" * 86)
print("3. Does the normalised yaw survive POSTURE change at one orientation?")
print("=" * 86)
print("  Phases below were all recorded side-on. Ratio should not move with posture.\n")
print(f"  {'phase':<16}{'posture':<28}{'ratio':>8}{'implied yaw':>13}")
POSTURE = [("anch/knee_0", "standing, legs locked"),
           ("anch/knee_90", "SEATED on a chair"),
           ("anch/sh_0", "arm hanging down"),
           ("anch/sh_90", "arm raised to horizontal"),
           ("anch/sh_180", "arm raised high"),
           ("rob/rotated", "standing side-on"),
           ("rob/seated", "SEATED side-on")]
vals = []
for k, desc in POSTURE:
    rs = all_r.get(k, [])
    if not rs:
        continue
    rt = float(np.median([ratio_of(r) for r in rs]))
    y = math.degrees(math.acos(min(max(rt / rmax, 0.0), 1.0)))
    vals.append((desc, rt, y))
    print(f"  {k:<16}{desc:<28}{rt:>8.3f}{y:>12.1f} deg")
if vals:
    ys = [v[2] for v in vals]
    print(f"\n  spread across postures at one orientation: {min(ys):.1f} to {max(ys):.1f} deg"
          f"  (range {max(ys)-min(ys):.1f} deg)")
