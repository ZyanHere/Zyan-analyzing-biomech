"""F3: absolute accuracy for KNEE and SHOULDER FLEXION.

Completes the trio the assignment requires: elbow (F28), knee, and one
challenging out-of-plane measurement. Shoulder flexion is the harder case by
design - it is a signed, plane-projected measurement built on the trunk frame
rather than a three-point hinge, so the elbow result cannot be assumed to carry.

Two fan geometries:
  hinge    - angle from the proximal segment's continuation (knee)
  vertical - angle from straight down in the trunk frame (shoulder, hip), which
             is what the goniometry chart means by anatomical neutral
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from lib.paths import fixture, model_path

import sys
import cv2, time, json, math, numpy as np
from lib import biomech_ref as B
from collections import deque
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

W, H, HOLD, SIDE = 640, 480, 4.0, "R"
# Loosened from 6.0/4.0/55.0: holding a single-leg pose cannot meet a 4 px
# steadiness gate, and the reference itself is only good to about +/- 5 deg,
# so a tighter alignment tolerance was false precision.
TOL, STEADY_PX, AUTO_AFTER, MIN_YAW = 9.0, 9.0, 0.8, 45.0
DISP = (1100, 825)

JOINTS = {
    # fn2d and fn3d are DIFFERENT functions for the sagittal measurements.
    # The 3D version projects onto `anterior`, which degenerates to zero on 2D
    # input; the 2D version treats the image plane as the sagittal plane, valid
    # only while the subject is side-on. Hinge joints share one implementation.
    # 120 dropped: heel-to-buttock on one leg cannot be held still enough to
    # measure, and an unheld pose is not a reference.
    "knee": dict(
        kind="hinge", prox=24, joint=26, dist=28, flex="posterior",
        targets=[0, 30, 60, 90],
        fn2d=B.knee_flexion, fn3d=B.knee_flexion,
        how="stand side-on, bend your knee BACK, heel toward your buttock"),
    "shoulder_flexion": dict(
        kind="vertical", prox=24, joint=12, dist=14, flex="anterior",
        targets=[0, 45, 90, 135, 180],
        fn2d=B.shoulder_flexion_2d, fn3d=B.shoulder_flexion,
        how="stand side-on, raise your arm FORWARD, keep the elbow straight"),
}

det = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path("full")),
    running_mode=vision.RunningMode.IMAGE, num_poses=1))
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv2.CAP_PROP_EXPOSURE, -5)
if not cap.isOpened():
    sys.exit("camera would not open")
cv2.namedWindow("accuracy", cv2.WINDOW_NORMAL)
cv2.resizeWindow("accuracy", *DISP)


def detect(f):
    r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    if not r.pose_landmarks:
        return None, None
    return r.pose_landmarks[0], (r.pose_world_landmarks[0] if r.pose_world_landmarks else None)


def P(lm, i, w, h):
    return np.array([lm[i].x * w, lm[i].y * h])


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


def trunk_frame(lm, w, h):
    """(down, anterior) in image pixels. Anterior comes from where the head sits
    relative to the hips, so it follows whichever way the subject is facing."""
    mid_sh = (P(lm, 11, w, h) + P(lm, 12, w, h)) / 2
    mid_hp = (P(lm, 23, w, h) + P(lm, 24, w, h)) / 2
    down = unit(mid_hp - mid_sh)
    perp = np.array([-down[1], down[0]])
    if np.dot(perp, P(lm, 0, w, h) - mid_sh) < 0:
        perp = -perp
    return down, perp


def reference(lm, spec, w, h):
    """(origin, zero-direction, perpendicular pointing the way the joint flexes).

    The flex direction is ANATOMICAL, not observed. Deriving it from the current
    limb position fails at exactly the pose that matters most: standing straight,
    the limb is parallel to the zero line, so the sign is noise and the fan flips
    to a random side. The knee flexes posteriorly; elbow, shoulder and hip flex
    anteriorly. That never changes, so it is not something to infer per frame.
    """
    j = P(lm, spec["joint"], w, h)
    down, ant = trunk_frame(lm, w, h)
    if spec["kind"] == "hinge":
        zero = unit(j - P(lm, spec["prox"], w, h))
        perp = np.array([-zero[1], zero[0]])
        if np.dot(perp, ant) < 0:
            perp = -perp
    else:
        zero, perp = down, ant
    if spec["flex"] == "posterior":
        perp = -perp
    return j, zero, perp


def fan_dir(zero, perp, deg):
    t = math.radians(deg)
    return zero * math.cos(t) + perp * math.sin(t)


def draw(frame, lm, spec, target, big, small, colour, aligned=None, steady=False):
    d = cv2.resize(frame, DISP)
    w, h = DISP
    if lm is not None:
        j, zero, perp = reference(lm, spec, w, h)
        dist = P(lm, spec["dist"], w, h)
        L = max(np.linalg.norm(dist - j), 110.0)
        for a in spec["targets"]:
            p = j + fan_dir(zero, perp, a) * L
            hot = (a == target)
            cv2.line(d, tuple(j.astype(int)), tuple(p.astype(int)),
                     (0, 0, 255) if hot else (95, 95, 190), 4 if hot else 1)
            lp = j + fan_dir(zero, perp, a) * (L + 36)
            cv2.putText(d, str(a), tuple((lp - [12, -6]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85 if hot else 0.55,
                        (0, 0, 255) if hot else (125, 125, 200), 3 if hot else 1)
        cv2.line(d, tuple(P(lm, spec["prox"], w, h).astype(int)),
                 tuple(j.astype(int)), (0, 0, 0), 5)
        cv2.line(d, tuple(j.astype(int)), tuple(dist.astype(int)),
                 (0, 235, 0) if aligned else (0, 190, 255), 5)
        for q in (P(lm, spec["prox"], w, h), j, dist):
            cv2.circle(d, tuple(q.astype(int)), 7, (255, 255, 255), -1)
            cv2.circle(d, tuple(q.astype(int)), 7, (0, 0, 0), 2)
    cv2.rectangle(d, (0, 0), (w, 150), (0, 0, 0), -1)
    cv2.putText(d, big, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.15, colour, 3)
    cv2.putText(d, small, (20, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (205, 205, 205), 2)
    if aligned is not None:
        if aligned and steady:
            bar, txt = (0, 190, 0), "ON TARGET - HOLD STILL"
        elif aligned:
            bar, txt = (0, 180, 230), "ON TARGET - stop moving"
        else:
            bar, txt = (40, 40, 190), "ADJUST"
        cv2.rectangle(d, (0, h - 92), (w, h), bar, -1)
        cv2.putText(d, txt, (26, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    cv2.imshow("accuracy", d)
    return cv2.waitKey(1) & 0xFF


# Calibration measures shoulder width SQUARE-ON. Every later rotation estimate is
# a ratio against it, so a calibration taken while already turned records a narrow
# width as if it were the full one - after which side-on reads as 0 deg rotation
# and the gate can never open. It is confirmed by keypress rather than by a timer
# for exactly that reason.
print("calibration: face the camera square-on, then press SPACE")
widths = []
while True:
    ok, f = cap.read()
    if not ok:
        continue
    lm, _ = detect(f)
    now = np.linalg.norm(P(lm, 11, W, H) - P(lm, 12, W, H)) if lm is not None else 0.0
    k = draw(f, None, JOINTS["knee"], 0, "FACE THE CAMERA SQUARE-ON",
             f"shoulder width {now:.0f} px   |   SPACE when you are facing it   q quit",
             (0, 200, 255))
    if k == ord("q"):
        cap.release(); cv2.destroyAllWindows(); sys.exit("aborted")
    if k == ord(" ") and lm is not None:
        t0 = time.time()
        while time.time() - t0 < 1.5:
            ok, f = cap.read()
            if not ok:
                continue
            lm, _ = detect(f)
            if lm is not None:
                widths.append(np.linalg.norm(P(lm, 11, W, H) - P(lm, 12, W, H)))
            draw(f, None, JOINTS["knee"], 0, "HOLD STILL", "measuring...", (0, 235, 0))
        if widths:
            break

SW_MAX = float(np.percentile(widths, 90))
print(f"face-on shoulder width = {SW_MAX:.1f} px")


def yaw_of(lm):
    """Self-correcting: the true face-on width is the widest the shoulders can
    ever appear, so anything wider than the calibration means the calibration was
    taken while partly turned. Widen it rather than reporting impossible angles."""
    global SW_MAX
    sw = np.linalg.norm(P(lm, 11, W, H) - P(lm, 12, W, H))
    if sw > SW_MAX:
        SW_MAX = sw
    return math.degrees(math.acos(min(max(sw / SW_MAX, 0.0), 1.0)))


recs = []
for jname, spec in JOINTS.items():
    for target in spec["targets"]:
        hist, green = deque(maxlen=6), None
        while True:
            ok, f = cap.read()
            if not ok:
                continue
            lm, _ = detect(f)
            aligned, steady, yaw = None, False, 0.0
            if lm is not None:
                yaw = yaw_of(lm)
                hist.append(P(lm, spec["dist"], W, H))
                if len(hist) >= 4:
                    steady = max(np.linalg.norm(hist[i + 1] - hist[i])
                                 for i in range(len(hist) - 1)) < STEADY_PX
                cur = spec["fn2d"](np.array([[l.x * W, l.y * H, 0.0] for l in lm]), SIDE)
                aligned = (not np.isnan(cur)) and abs(abs(cur) - target) <= TOL
                if yaw < MIN_YAW:
                    aligned = False
            if aligned and steady:
                green = green or time.time()
                if time.time() - green >= AUTO_AFTER:
                    break
            else:
                green = None
            # Say WHICH gate is blocking. "ADJUST" with three invisible
            # conditions behind it is unusable feedback.
            why = []
            if lm is None:
                why.append("no person detected")
            else:
                if yaw < MIN_YAW:
                    why.append(f"turn more side-on ({yaw:.0f} of {MIN_YAW:.0f} deg)")
                if not aligned and yaw >= MIN_YAW:
                    off = abs(cur) - target if not np.isnan(cur) else float("nan")
                    why.append(f"off the line by {off:+.0f} deg")
                if aligned and not steady:
                    why.append("hold still")
            msg = "   |   ".join(why) if why else spec["how"]
            k = draw(f, lm, spec, target, f"{jname.upper()}  ->  {target} deg",
                     msg + "   |   SPACE force   s skip   q quit", (0, 200, 255),
                     aligned, steady)
            if k == ord(" "):
                break
            if k == ord("s"):
                target = None
                break
            if k == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                sys.exit("aborted")
        if target is None:
            continue
        t0, n = time.time(), 0
        while time.time() - t0 < HOLD:
            ok, f = cap.read()
            if not ok:
                continue
            lm, wl = detect(f)
            if lm is None or wl is None:
                continue
            n += 1
            recs.append({"joint": jname, "target": target, "yaw": round(yaw_of(lm), 1),
                         "lm": [[l.x, l.y, l.z, l.visibility] for l in lm],
                         "w": [[l.x, l.y, l.z] for l in wl]})
            draw(f, lm, spec, target,
                 f"RECORDING {target} deg   {HOLD - (time.time() - t0):3.1f}s",
                 f"{n} frames - keep still", (0, 235, 0), True, True)

cap.release()
cv2.destroyAllWindows()
det.close()
with open(fixture("accuracy_multi.jsonl"), "w") as fh:
    for r in recs:
        fh.write(json.dumps(r) + "\n")

for jname, spec in JOINTS.items():
    print()
    print("=" * 74)
    print(f"{jname}  -  absolute accuracy (reference error ~ +/- 5 deg)")
    print("=" * 74)
    print(f"  {'true':>6}{'n':>5}{'yaw':>7}{'2D':>9}{'err':>8}{'3D':>9}{'err':>8}")
    e2, e3 = [], []
    for t in spec["targets"]:
        rs = [r for r in recs if r["joint"] == jname and r["target"] == t]
        if not rs:
            continue
        l = np.array([[[p[0] * W, p[1] * H, 0.0] for p in r["lm"]] for r in rs])
        w3 = np.array([r["w"] for r in rs])
        v2 = np.nanmedian([abs(spec["fn2d"](x, SIDE)) for x in l])
        v3 = np.nanmedian([abs(spec["fn3d"](x, SIDE)) for x in w3])
        e2.append(v2 - t)
        e3.append(v3 - t)
        print(f"  {t:>6}{len(rs):>5}{np.median([r['yaw'] for r in rs]):>7.0f}"
              f"{v2:>9.1f}{v2 - t:>+8.1f}{v3:>9.1f}{v3 - t:>+8.1f}")
    for nm, e in (("2D", e2), ("3D", e3)):
        if not e:
            continue
        e = np.array(e)
        print(f"  {nm}: MAE {np.mean(np.abs(e)):5.1f}  bias {np.mean(e):+5.1f}  "
              f"95% LoA {np.mean(e) - 1.96 * np.std(e):+.1f} to {np.mean(e) + 1.96 * np.std(e):+.1f}")

print(f"\nwrote {len(recs)} frames -> {OUT}/accuracy_multi.jsonl")
