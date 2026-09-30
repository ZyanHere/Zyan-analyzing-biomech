"""F4: absolute accuracy from ANATOMICALLY ANCHORED poses.

Replaces f3, which gated on rotation + alignment + steadiness simultaneously and
could lock the subject out with no indication of which gate was failing.

Design change: nothing is gated. You press SPACE, it records. Body rotation is
measured and REPORTED so validity can be judged from the data afterwards, rather
than enforced by a gate that can refuse every pose.

No protractor is needed because each pose defines its own angle:

  knee 0      standing, legs locked straight      - extension is a hard stop
  knee 90     seated on an ordinary chair         - a chair IS a 90 degree jig
  shoulder 0  arm hanging at your side            - gravity defines it
  shoulder 90 arm horizontal, forward             - check against any shelf or
                                                    table edge; horizontal is
                                                    something people judge well
  shoulder 180 arm straight up beside your ear    - vertical, same argument

Reference error is larger than a printed scale (call it +/- 8 deg rather than
+/- 5) but the poses are reachable, repeatable, and need no equipment.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from lib.paths import fixture, model_path

import sys
import cv2, time, json, math, numpy as np
from lib import biomech_ref as B
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

W, H, HOLD, SIDE = 640, 480, 3.0, "R"
DISP = (1100, 825)

POSES = [
    dict(key="knee_0", joint="knee", truth=0,
         title="KNEE 0 deg", how="stand up straight, side-on, both legs locked",
         fn2d=B.knee_flexion, fn3d=B.knee_flexion, chain=(24, 26, 28)),
    dict(key="knee_90", joint="knee", truth=90,
         title="KNEE 90 deg", how="sit on a normal chair, side-on, feet flat, shins vertical",
         fn2d=B.knee_flexion, fn3d=B.knee_flexion, chain=(24, 26, 28)),
    dict(key="sh_0", joint="shoulder_flexion", truth=0,
         title="SHOULDER 0 deg", how="stand side-on, right arm hanging straight down",
         fn2d=B.shoulder_flexion_2d, fn3d=B.shoulder_flexion, chain=(24, 12, 14)),
    dict(key="sh_90", joint="shoulder_flexion", truth=90,
         title="SHOULDER 90 deg", how="raise right arm FORWARD to horizontal - check against a shelf",
         fn2d=B.shoulder_flexion_2d, fn3d=B.shoulder_flexion, chain=(24, 12, 14)),
    dict(key="sh_180", joint="shoulder_flexion", truth=180,
         title="SHOULDER 180 deg", how="right arm straight UP beside your ear",
         fn2d=B.shoulder_flexion_2d, fn3d=B.shoulder_flexion, chain=(24, 12, 14)),
]

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
cv2.namedWindow("anchored", cv2.WINDOW_NORMAL)
cv2.resizeWindow("anchored", *DISP)

SW_SEEN = [1.0]   # widest shoulders ever observed = the face-on reference


def detect(f):
    r = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    if not r.pose_landmarks:
        return None, None
    return r.pose_landmarks[0], (r.pose_world_landmarks[0] if r.pose_world_landmarks else None)


def px(lm, i):
    return np.array([lm[i].x * W, lm[i].y * H])


def yaw_of(lm):
    """Measured, never enforced. Self-referencing against the widest shoulders
    seen so far, so no separate calibration step exists to get wrong."""
    sw = np.linalg.norm(px(lm, 11) - px(lm, 12))
    SW_SEEN[0] = max(SW_SEEN[0], sw)
    return math.degrees(math.acos(min(max(sw / SW_SEEN[0], 0.0), 1.0)))


def draw(frame, lm, pose, big, small, colour, recording=False):
    d = cv2.resize(frame, DISP)
    w, h = DISP
    sx, sy = w / W, h / H
    if lm is not None:
        a, b, c = pose["chain"]
        pts = [(px(lm, i)[0] * sx, px(lm, i)[1] * sy) for i in (a, b, c)]
        for p, q in ((pts[0], pts[1]), (pts[1], pts[2])):
            cv2.line(d, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])),
                     (0, 235, 0) if recording else (0, 190, 255), 6)
        for p in pts:
            cv2.circle(d, (int(p[0]), int(p[1])), 9, (255, 255, 255), -1)
            cv2.circle(d, (int(p[0]), int(p[1])), 9, (0, 0, 0), 2)
    cv2.rectangle(d, (0, 0), (w, 160), (0, 0, 0), -1)
    cv2.putText(d, big, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.3, colour, 3)
    cv2.putText(d, small, (20, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (215, 215, 215), 2)
    cv2.imshow("anchored", d)
    return cv2.waitKey(1) & 0xFF


recs = []
for i, pose in enumerate(POSES):
    while True:
        ok, f = cap.read()
        if not ok:
            continue
        lm, _ = detect(f)
        yaw = yaw_of(lm) if lm is not None else 0.0
        seen = "person detected" if lm is not None else "NO PERSON DETECTED"
        k = draw(f, lm, pose, f"[{i+1}/{len(POSES)}]  {pose['title']}",
                 f"{pose['how']}   |   {seen}, rotation {yaw:.0f} deg"
                 "   |   SPACE record   n skip   q quit", (0, 200, 255))
        if k == ord(" ") and lm is not None:
            break
        if k == ord("n"):
            pose = None
            break
        if k == ord("q"):
            cap.release(); cv2.destroyAllWindows(); sys.exit("aborted")
    if pose is None:
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
        recs.append({"key": pose["key"], "joint": pose["joint"], "truth": pose["truth"],
                     "yaw": round(yaw_of(lm), 1),
                     "lm": [[l.x, l.y, l.z, l.visibility] for l in lm],
                     "w": [[l.x, l.y, l.z] for l in wl]})
        draw(f, lm, pose, f"RECORDING  {HOLD - (time.time() - t0):3.1f}s",
             f"{n} frames - hold the pose", (0, 235, 0), recording=True)

cap.release()
cv2.destroyAllWindows()
det.close()

if not recs:
    sys.exit("nothing recorded")
with open(fixture("anchored.jsonl"), "w") as fh:
    for r in recs:
        fh.write(json.dumps(r) + "\n")

print()
print("=" * 82)
print("Absolute accuracy at anatomically anchored poses (reference ~ +/- 8 deg)")
print("=" * 82)
print(f"  {'pose':<12}{'true':>6}{'n':>5}{'yaw':>6}{'2D':>9}{'err':>8}{'3D':>9}{'err':>8}{'valid?':>9}")
err2, err3 = {}, {}
for pose in POSES:
    rs = [r for r in recs if r["key"] == pose["key"]]
    if not rs:
        continue
    l = np.array([[[p[0] * W, p[1] * H, 0.0] for p in r["lm"]] for r in rs])
    w3 = np.array([r["w"] for r in rs])
    v2 = float(np.nanmedian([abs(pose["fn2d"](x, SIDE)) for x in l]))
    v3 = float(np.nanmedian([abs(pose["fn3d"](x, SIDE)) for x in w3]))
    yaw = float(np.median([r["yaw"] for r in rs]))
    ok2d = "yes" if yaw >= 45 else "2D SUSPECT"
    err2.setdefault(pose["joint"], []).append(v2 - pose["truth"])
    err3.setdefault(pose["joint"], []).append(v3 - pose["truth"])
    print(f"  {pose['key']:<12}{pose['truth']:>6}{len(rs):>5}{yaw:>6.0f}"
          f"{v2:>9.1f}{v2 - pose['truth']:>+8.1f}{v3:>9.1f}{v3 - pose['truth']:>+8.1f}{ok2d:>9}")

print()
for joint in err2:
    for nm, e in (("2D", err2[joint]), ("3D", err3[joint])):
        e = np.array(e)
        lo = e.mean() - 1.96 * e.std()
        hi = e.mean() + 1.96 * e.std()
        print(f"  {joint:<18}{nm}:  MAE {np.abs(e).mean():5.1f}   bias {e.mean():+5.1f}"
              f"   95% LoA {lo:+.1f} to {hi:+.1f}   (n={len(e)})")
print()
print("  'valid?' flags poses recorded below 45 deg of body rotation, where a 2D")
print("  sagittal measurement is geometrically unreliable (F27). Reported, not blocked -")
print("  a gate that refuses every pose yields no data at all.")
print(f"\n  wrote {len(recs)} frames -> {OUT}/anchored.jsonl")
