"""F1 (item 4): absolute accuracy at known elbow angles.

Two reference modes, toggled live with 'h':

  GUIDE ON  - an on-screen protractor fan is drawn from your detected elbow.
              Easy, needs no printer. BUT the target line is derived from the
              model's own landmarks, so aligning to it makes the 2D reading
              partly self-referential. Still informative for 3D, which is
              computed from different numbers.

  GUIDE OFF - align to the PRINTED sheet instead. Fully independent reference.
              This is the mode whose numbers can be quoted without caveat.

The live angle readout is hidden while you align and only revealed after the
hold is recorded, so you cannot tune yourself to the system's own answer.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from lib.paths import fixture, model_path

import cv2, time, json, math, numpy as np
from lib import biomech_ref as B
from collections import deque
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

W, H, HOLD, SIDE = 640, 480, 5.0, "R"
TOL = 5.0          # degrees of alignment counted as on-target
STEADY_PX = 3.5    # wrist movement below this counts as holding still
AUTO_AFTER = 1.2   # seconds of sustained green before auto-recording
MIN_YAW = 55.0     # degrees of body rotation required before recording is allowed
TARGETS = [0, 30, 60, 90, 120, 150]
FAN = [0, 30, 60, 90, 120, 150]
SH, EL, WR = (12, 14, 16) if SIDE == "R" else (11, 13, 15)
DISP = (1100, 825)

det = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path("full")),
    running_mode=vision.RunningMode.IMAGE, num_poses=1))
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25); cap.set(cv2.CAP_PROP_EXPOSURE, -5)
if not cap.isOpened(): sys.exit("camera would not open")
cv2.namedWindow("accuracy", cv2.WINDOW_NORMAL); cv2.resizeWindow("accuracy", *DISP)

def detect(f):
    r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    if not r.pose_landmarks: return None, None
    return r.pose_landmarks[0], (r.pose_world_landmarks[0] if r.pose_world_landmarks else None)

def px(lm, i, w, h): return np.array([lm[i].x * w, lm[i].y * h])

def rotate(v, deg):
    t = math.radians(deg); c, s = math.cos(t), math.sin(t)
    return np.array([v[0]*c - v[1]*s, v[0]*s + v[1]*c])

def draw(frame, lm, target, guide, big, small, colour, live=None, ok=False,
         aligned=None, steady=False):
    d = cv2.resize(frame, DISP); w, h = DISP
    if lm is not None:
        sh, el, wr = px(lm, SH, w, h), px(lm, EL, w, h), px(lm, WR, w, h)
        ua = el - sh
        L = max(np.linalg.norm(wr - el), 90.0)
        if np.linalg.norm(ua) > 1e-3:
            straight = ua / np.linalg.norm(ua)
            fv = wr - el
            # NumPy 2.x dropped the 2-D np.cross; the scalar z-component is all
            # we need, and it tells us which way the forearm currently bends.
            side = 1.0 if (straight[0]*fv[1] - straight[1]*fv[0]) > 0 else -1.0
            if guide:
                for a in FAN:
                    p = el + rotate(straight, side * a) * L
                    hot = (a == target)
                    cv2.line(d, tuple(el.astype(int)), tuple(p.astype(int)),
                             (0, 0, 255) if hot else (90, 90, 190), 4 if hot else 1)
                    lp = el + rotate(straight, side * a) * (L + 34)
                    cv2.putText(d, str(a), tuple((lp - [10, -6]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85 if hot else 0.55,
                                (0, 0, 255) if hot else (120, 120, 200), 3 if hot else 1)
            cv2.line(d, tuple(sh.astype(int)), tuple(el.astype(int)), (0, 0, 0), 5)
        fore_col = (0, 235, 0) if (ok or aligned) else (0, 190, 255)
        cv2.line(d, tuple(el.astype(int)), tuple(wr.astype(int)), fore_col, 5)
        for p in (sh, el, wr):
            cv2.circle(d, tuple(p.astype(int)), 7, (255, 255, 255), -1)
            cv2.circle(d, tuple(p.astype(int)), 7, (0, 0, 0), 2)
    cv2.rectangle(d, (0, 0), (w, 150), (0, 0, 0), -1)
    cv2.putText(d, big, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.25, colour, 3)
    cv2.putText(d, small, (20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (205, 205, 205), 2)
    mode = "GUIDE ON (on-screen)" if guide else "GUIDE OFF (printed sheet)"
    cv2.putText(d, mode, (w - 430, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (120, 190, 255) if guide else (120, 255, 190), 2)
    if live is not None:
        cv2.putText(d, f"{live:5.1f}", (w - 260, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.9,
                    (0, 235, 0), 4)
    # Binary alignment light. Deliberately has NO number attached: a readout would
    # let you tune your arm until the system agreed with itself.
    if aligned is not None:
        if aligned and steady:   bar, txt = (0, 190, 0),   "ON TARGET - HOLD STILL"
        elif aligned:            bar, txt = (0, 180, 230), "ON TARGET - stop moving"
        else:                    bar, txt = (40, 40, 190), "ADJUST - turn side-on / find the red line"
        cv2.rectangle(d, (0, h - 92), (w, h), bar, -1)
        cv2.putText(d, txt, (26, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (255, 255, 255), 3)
    cv2.imshow("accuracy", d)
    return cv2.waitKey(1) & 0xFF

# --- calibration: measure shoulder width face-on, so body yaw can be estimated ---
print("calibrating: face the camera for a moment...")
widths = []
t0 = time.time()
while time.time() - t0 < 3.0:
    ok, f = cap.read()
    if not ok: continue
    lm, _ = detect(f)
    if lm is not None:
        widths.append(np.linalg.norm(px(lm, 11, W, H) - px(lm, 12, W, H)))
    draw(f, lm, 0, False, "FACE THE CAMERA", "measuring your shoulder width...", (0, 200, 255))
SW_MAX = np.percentile(widths, 90) if widths else 100.0
print(f"face-on shoulder width = {SW_MAX:.1f} px")

def body_yaw(lm):
    """Apparent shoulder width shrinks as the subject turns. 0 = facing, 90 = side-on."""
    sw = np.linalg.norm(px(lm, 11, W, H) - px(lm, 12, W, H))
    return math.degrees(math.acos(min(max(sw / SW_MAX, 0.0), 1.0)))

recs, guide, modes = [], True, {}
for target in TARGETS:
    hint = ("straighten your arm COMPLETELY" if target == 0
            else f"put your forearm on the line marked {target}")
    wrist_hist, green_since = deque(maxlen=6), None
    while True:
        ok, f = cap.read()
        if not ok: continue
        lm, _ = detect(f)
        aligned, steady, yaw = None, False, 0.0
        if lm is not None:
            yaw = body_yaw(lm)
            wrist_hist.append(px(lm, WR, W, H))
            if len(wrist_hist) >= 4:
                steady = max(np.linalg.norm(wrist_hist[i+1] - wrist_hist[i])
                             for i in range(len(wrist_hist) - 1)) < STEADY_PX
            if guide:
                cur = B.elbow_flexion(
                    np.array([[l.x * W, l.y * H, 0.0] for l in lm]), SIDE)
                if not np.isnan(cur):
                    aligned = abs(cur - target) <= TOL
        # F20: an image-plane measurement is only valid when the plane of motion
        # faces the camera. Facing the camera forces the forearm across the body,
        # which is both anatomically unusual and geometrically wrong.
        if yaw < MIN_YAW:
            aligned = False
        # Auto-record once the light has been green for a moment, so you do not
        # have to walk back to the keyboard while holding an awkward pose.
        if aligned and steady:
            green_since = green_since or time.time()
            if time.time() - green_since >= AUTO_AFTER: break
        else:
            green_since = None
        msg = (hint if yaw >= MIN_YAW
               else f"TURN SIDE-ON FIRST  (rotation {yaw:.0f} deg, need {MIN_YAW:.0f})")
        k = draw(f, lm, target, guide, f"SET ELBOW TO {target} deg",
                 msg + "   |   SPACE record   h toggle guide   q quit", (0, 200, 255),
                 aligned=aligned, steady=steady)
        if k == ord(' '): break
        if k == ord('h'): guide = not guide; green_since = None
        if k == ord('q'): cap.release(); cv2.destroyAllWindows(); sys.exit("aborted")
    modes[target] = "guide" if guide else "printed"
    t0, n = time.time(), 0
    while time.time() - t0 < HOLD:
        ok, f = cap.read()
        if not ok: continue
        lm, wl = detect(f)
        if lm is None or wl is None: continue
        n += 1
        recs.append({"target": target, "mode": modes[target], "yaw": round(body_yaw(lm), 1),
                     "t": time.time() - t0,
                     "lm": [[l.x, l.y, l.z, l.visibility] for l in lm],
                     "w":  [[l.x, l.y, l.z] for l in wl]})
        draw(f, lm, target, guide, f"RECORDING {target} deg   {HOLD-(time.time()-t0):3.1f}s",
             f"{n} frames  -  keep still", (0, 235, 0), ok=True)
    rs = [r for r in recs if r["target"] == target]
    got = np.nanmedian([B.elbow_flexion(np.array(r["w"]), SIDE) for r in rs])
    t0 = time.time()
    while time.time() - t0 < 2.0:
        ok, f = cap.read()
        if ok: draw(f, None, target, False, f"target {target}  ->  measured {got:.1f} deg",
                    "next angle in a moment...", (0, 235, 0), live=got)

cap.release(); cv2.destroyAllWindows(); det.close()
with open(fixture("accuracy_session.jsonl"), "w") as fh:
    for r in recs: fh.write(json.dumps(r) + "\n")

print()
print("=" * 78)
print("Absolute accuracy vs reference   (reference error ~ +/- 5 deg)")
print("=" * 78)
print(f"  {'true':>6}{'mode':>9}{'n':>5}{'3D':>9}{'err':>8}{'2D':>9}{'err':>8}")
e3, e2 = [], []
for t in TARGETS:
    rs = [r for r in recs if r["target"] == t]
    if not rs: continue
    w = np.array([r["w"] for r in rs])
    l = np.array([[[p[0]*W, p[1]*H, 0.0] for p in r["lm"]] for r in rs])
    v3 = np.nanmedian([B.elbow_flexion(x, SIDE) for x in w])
    v2 = np.nanmedian([B.elbow_flexion(x, SIDE) for x in l])
    e3.append(v3 - t); e2.append(v2 - t)
    print(f"  {t:>6}{modes[t]:>9}{len(rs):>5}{v3:>9.1f}{v3-t:>+8.1f}{v2:>9.1f}{v2-t:>+8.1f}")
for nm, e in (("3D world", e3), ("2D image", e2)):
    e = np.array(e)
    print(f"\n  {nm}:  MAE {np.mean(np.abs(e)):5.1f}   bias {np.mean(e):+5.1f}   "
          f"95% LoA {np.mean(e)-1.96*np.std(e):+.1f} to {np.mean(e)+1.96*np.std(e):+.1f}")
if any(m == "guide" for m in modes.values()):
    print("\n  NOTE: angles recorded in 'guide' mode used an on-screen target derived")
    print("  from the model's own landmarks, so their 2D column is partly")
    print("  self-referential. Quote 'printed' rows without caveat.")
print(f"\n  wrote {len(recs)} frames -> {OUT}/accuracy_session.jsonl")
