"""Reference implementation of the goniometry conventions. Investigation build.

Anatomical frame is rebuilt per frame from the landmarks, because E1 showed the
model's world frame is camera-aligned, not body-locked.

Frame (right-handed, verified in E1):
    up        = mid_shoulder - mid_hip          (superior)
    left      = L_hip - R_hip, orthogonalised   (subject's own left)
    anterior  = cross(left, up)                 (forward)

Planes:
    sagittal = span(up, anterior)   normal = left       -> flexion / extension
    frontal  = span(up, left)       normal = anterior   -> abduction / adduction
"""
import numpy as np

L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 23, 24, 25, 26, 27, 28
L_HEEL, R_HEEL, L_FOOT, R_FOOT = 29, 30, 31, 32

S = {"L": dict(sh=L_SH, el=L_EL, wr=L_WR, hip=L_HIP, kn=L_KN, an=L_AN, heel=L_HEEL, foot=L_FOOT),
     "R": dict(sh=R_SH, el=R_EL, wr=R_WR, hip=R_HIP, kn=R_KN, an=R_AN, heel=R_HEEL, foot=R_FOOT)}

def _n(v):
    m = np.linalg.norm(v)
    return v / m if m > 1e-9 else v

def frame(lm):
    """Anatomical unit frame from landmarks. lm: (33,3)."""
    mid_hip = (lm[L_HIP] + lm[R_HIP]) / 2
    mid_sh  = (lm[L_SH]  + lm[R_SH])  / 2
    up   = _n(mid_sh - mid_hip)
    raw  = lm[L_HIP] - lm[R_HIP]
    left = _n(raw - np.dot(raw, up) * up)          # orthogonalise against up
    ant  = _n(np.cross(left, up))
    return mid_hip, up, left, ant

def interior(a, b, c):
    """Unsigned interior angle at b. 180 = straight."""
    ba, bc = a - b, c - b
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-9 or nc < 1e-9: return np.nan
    return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (na * nc), -1, 1)))

def _signed_from_down(vec, up, in_plane_axis):
    """Angle of `vec` away from straight-down, measured inside the plane
    spanned by (up, in_plane_axis). Positive means toward +in_plane_axis.
    Anatomical neutral (limb hanging down) is exactly 0."""
    return np.degrees(np.arctan2(np.dot(vec, in_plane_axis), -np.dot(vec, up)))

# --- hinge joints: interior angle, neutral is a straight limb = 0 -------------

def elbow_flexion(lm, side):
    """Chart: flexion 0-150. Straight arm = 0. Unsigned (elbow does not extend past 0)."""
    s = S[side]
    return 180.0 - interior(lm[s["sh"]], lm[s["el"]], lm[s["wr"]])

def knee_flexion(lm, side):
    """Chart: flexion 0-135. Straight leg = 0. Unsigned."""
    s = S[side]
    return 180.0 - interior(lm[s["hip"]], lm[s["kn"]], lm[s["an"]])

# --- ball joints: signed, plane-projected ------------------------------------

def shoulder_flexion(lm, side):
    """Chart: flexion 0-180 (forward), extension 0-60 (backward). Arm down = 0.
    Humerus projected into the SAGITTAL plane. Positive = flexion."""
    s = S[side]
    _, up, left, ant = frame(lm)
    return _signed_from_down(lm[s["el"]] - lm[s["sh"]], up, ant)

def shoulder_abduction(lm, side):
    """Chart: abduction 0-180 (out to the side), adduction 0-45 (across body).
    Humerus projected into the FRONTAL plane. Positive = abduction, i.e. away
    from the midline, so the lateral axis is flipped for the right side."""
    s = S[side]
    _, up, left, ant = frame(lm)
    lateral = left if side == "L" else -left
    return _signed_from_down(lm[s["el"]] - lm[s["sh"]], up, lateral)

def hip_flexion(lm, side):
    """Chart: flexion 0-120 (forward), extension 0-30 (backward). Standing = 0.
    Femur projected into the SAGITTAL plane. Positive = flexion."""
    s = S[side]
    _, up, left, ant = frame(lm)
    return _signed_from_down(lm[s["kn"]] - lm[s["hip"]], up, ant)

# --- ankle: neutral is ~90 between shank and foot, not 180 -------------------

def ankle_angle(lm, side, use_heel=True, project=True, neutral=90.0):
    """Chart: dorsiflexion 0-20, plantarflexion 0-50. Neutral standing = 0.
    Positive = dorsiflexion (toes toward shin), negative = plantarflexion.

    use_heel: define the foot by heel->foot_index (the foot's long axis) rather
              than ankle->foot_index. Anatomically the former is correct.
    project:  restrict to the sagittal plane, excluding inversion/eversion.
    """
    s = S[side]
    _, up, left, ant = frame(lm)
    shank = lm[s["kn"]] - lm[s["an"]]
    foot  = lm[s["foot"]] - (lm[s["heel"]] if use_heel else lm[s["an"]])
    if project:
        for v in (shank, foot):
            v -= np.dot(v, left) * left
    na, nc = np.linalg.norm(shank), np.linalg.norm(foot)
    if na < 1e-9 or nc < 1e-9: return np.nan
    theta = np.degrees(np.arccos(np.clip(np.dot(shank, foot) / (na * nc), -1, 1)))
    return neutral - theta

# --- 2D variants of the plane-projected measurements ------------------------
#
# The 3D functions above project onto `anterior = cross(left, up)`. Given 2D
# image coordinates every z is 0, so `anterior` degenerates to pure Z and the
# dot product is always zero: the function can only ever return 0 or 180.
# It is structurally incapable of running on 2D input.
#
# So the sagittal measurements need a SEPARATE 2D implementation. It is valid
# only while the subject is side-on, because that is the condition under which
# the image plane IS the sagittal plane (F20 test D, F27 thresholds).
#
# Hinge joints (elbow, knee) need no 2D variant: an interior angle between three
# points is the same computation in any number of dimensions.

def _image_frame(lm):
    """(down, anterior) unit vectors in the image plane. Anterior is taken from
    which way the head sits relative to the hips, so it follows the subject."""
    mid_sh = (lm[L_SH][:2] + lm[R_SH][:2]) / 2
    mid_hp = (lm[L_HIP][:2] + lm[R_HIP][:2]) / 2
    down = _n(mid_hp - mid_sh)
    perp = np.array([-down[1], down[0]])
    nose = lm[0][:2] if len(lm) > 0 else mid_sh
    if np.dot(perp, nose - mid_sh) < 0:
        perp = -perp
    return down, perp

def _signed_in_image(vec2, down, ant):
    return np.degrees(np.arctan2(np.dot(vec2, ant), np.dot(vec2, down)))

def shoulder_flexion_2d(lm, side):
    """Sagittal shoulder flexion from image coordinates. Subject must be side-on."""
    s = S[side]
    down, ant = _image_frame(lm)
    return _signed_in_image(lm[s["el"]][:2] - lm[s["sh"]][:2], down, ant)

def hip_flexion_2d(lm, side):
    """Sagittal hip flexion from image coordinates. Subject must be side-on."""
    s = S[side]
    down, ant = _image_frame(lm)
    return _signed_in_image(lm[s["kn"]][:2] - lm[s["hip"]][:2], down, ant)

def shoulder_abduction_2d(lm, side):
    """Frontal-plane abduction from image coordinates.

    Note the orientation requirement is the OPPOSITE of the sagittal measurements:
    abduction lives in the frontal plane, so the subject must FACE the camera,
    whereas flexion/extension need them side-on. One application cannot satisfy
    both at once, which is a constraint on the UI, not on the maths.
    """
    s = S[side]
    mid_sh = (lm[L_SH][:2] + lm[R_SH][:2]) / 2
    mid_hp = (lm[L_HIP][:2] + lm[R_HIP][:2]) / 2
    down = _n(mid_hp - mid_sh)
    # lateral = away from the midline for this side, taken from the hips in-image
    hips = lm[L_HIP][:2] - lm[R_HIP][:2]
    lateral = _n(hips - np.dot(hips, down) * down)
    if side == "R":
        lateral = -lateral
    h = lm[s["el"]][:2] - lm[s["sh"]][:2]
    return np.degrees(np.arctan2(np.dot(h, lateral), np.dot(h, down)))

def ankle_angle_2d(lm, side, use_heel=True, neutral=90.0):
    """Ankle from image coordinates. No sagittal projection: subtracting the
    lateral component would discard real in-image information, because in 2D
    the `left` axis lies inside the image plane rather than along depth."""
    s = S[side]
    shank = lm[s["kn"]][:2] - lm[s["an"]][:2]
    foot = lm[s["foot"]][:2] - (lm[s["heel"]][:2] if use_heel else lm[s["an"]][:2])
    na, nc = np.linalg.norm(shank), np.linalg.norm(foot)
    if na < 1e-9 or nc < 1e-9: return np.nan
    theta = np.degrees(np.arccos(np.clip(np.dot(shank, foot) / (na * nc), -1, 1)))
    return neutral - theta

ALL = {
    "elbow_flexion":      elbow_flexion,
    "knee_flexion":       knee_flexion,
    "shoulder_flexion":   shoulder_flexion,
    "shoulder_abduction": shoulder_abduction,
    "hip_flexion":        hip_flexion,
    "ankle_angle":        ankle_angle,
}

# Which implementation to use for which representation. Hinge joints share one.
ALL_2D = {
    "elbow_flexion":      elbow_flexion,
    "knee_flexion":       knee_flexion,
    "shoulder_flexion":   shoulder_flexion_2d,
    "shoulder_abduction": shoulder_abduction_2d,
    "hip_flexion":        hip_flexion_2d,
    "ankle_angle":        ankle_angle_2d,
}

# Which way the subject must face for each 2D measurement to be geometrically
# valid. SAGITTAL measurements need side-on; FRONTAL needs face-on. Hinge joints
# are measured in the limb's own plane, so they follow whichever plane the limb
# moves in - in practice sagittal for both elbow and knee.
REQUIRED_VIEW = {
    "elbow_flexion":      "sagittal",
    "knee_flexion":       "sagittal",
    "shoulder_flexion":   "sagittal",
    "shoulder_abduction": "frontal",
    "hip_flexion":        "sagittal",
    "ankle_angle":        "sagittal",
}

# How a signed value is presented. The chart lists the two directions as separate
# named ranges, so a negative number must be shown as its named opposite rather
# than as a minus sign.
# (positive label, negative label, positive max, negative max). The two maxima are
# the chart's two named ranges, NOT the endpoints of one range: writing the elbow's
# "0 to 150" as (0, 150) sets the positive maximum to zero and flags every normal
# bend as abnormal.
DISPLAY = {
    "elbow_flexion":      ("flexion", None, 150, 0),
    "knee_flexion":       ("flexion", None, 135, 0),
    "shoulder_flexion":   ("flexion", "extension", 180, 60),
    "shoulder_abduction": ("abduction", "adduction", 180, 45),
    "hip_flexion":        ("flexion", "extension", 120, 30),
    "ankle_angle":        ("dorsiflexion", "plantarflexion", 20, 50),
}

def present(name, value, tol=0.5):
    """(label, magnitude, in_normal_range) for display.

    `tol` exists because a joint held at exactly its charted maximum computes to
    150.0000001, and flagging that as out of range would be a rounding artefact
    reported as a clinical finding.
    """
    pos, neg, pos_max, neg_max = DISPLAY[name]
    if np.isnan(value):
        return ("no data", float("nan"), False)
    if value >= 0 or neg is None:
        return (pos, abs(value), abs(value) <= pos_max + tol)
    return (neg, abs(value), abs(value) <= neg_max + tol)

def wrap180(deg):
    """Signed difference folded into (-180, 180]. Comparing angles by plain
    subtraction reports -179.8 vs +180 as a 360 degree error."""
    return (deg + 180.0) % 360.0 - 180.0
