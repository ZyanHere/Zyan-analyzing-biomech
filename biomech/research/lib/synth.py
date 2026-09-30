"""Synthetic skeleton with EXACT known joint angles.

Forward kinematics in a clean right-handed frame: up=+Y, subject's left=+X,
anterior=+Z, so cross(left, up) = anterior, matching what E1 measured.

Sagittal rotations use axis = -left, except the knee, which flexes backwards
and therefore uses +left. That asymmetry is anatomy, not a sign bug.
"""
import numpy as np

UP    = np.array([0.0, 1.0, 0.0])
LEFT  = np.array([1.0, 0.0, 0.0])
ANT   = np.array([0.0, 0.0, 1.0])
DOWN  = -UP

SEG = dict(trunk=0.50, sh_half=0.18, hip_half=0.15,
           upper_arm=0.30, forearm=0.26, femur=0.42, shank=0.42,
           heel_back=0.05, foot_fwd=0.20, foot_drop=0.06)

def rot(v, axis, deg):
    """Rodrigues rotation of v about a unit axis."""
    a = axis / np.linalg.norm(axis); th = np.radians(deg)
    return (v * np.cos(th) + np.cross(a, v) * np.sin(th)
            + a * np.dot(a, v) * (1 - np.cos(th)))

def build(sh_flex=0.0, sh_abd=0.0, elbow=0.0, hip_flex=0.0, knee=0.0,
          ankle=0.0, side="L", body_yaw=0.0):
    """Return (33,3) landmarks. Angles in degrees, in CHART convention:
       sh_flex  +forward / -backward     elbow    +flexion
       sh_abd   +away from midline       hip_flex +forward / -backward
       knee     +flexion (heel back)     ankle    +dorsiflexion / -plantarflexion
       body_yaw rotates the whole body about vertical (0 = facing +Z / camera)
    """
    lm = np.zeros((33, 3))
    sgn = 1.0 if side == "L" else -1.0
    sag = -LEFT                      # sagittal rotation axis, + = anterior
    abd_axis = ANT * sgn             # frontal axis, + = away from this side's midline

    mid_hip = np.zeros(3)
    mid_sh  = mid_hip + UP * SEG["trunk"]
    lm[23] = mid_hip + LEFT * SEG["hip_half"];  lm[24] = mid_hip - LEFT * SEG["hip_half"]
    lm[11] = mid_sh  + LEFT * SEG["sh_half"];   lm[12] = mid_sh  - LEFT * SEG["sh_half"]
    lm[0]  = mid_sh + UP * 0.25 + ANT * 0.05
    lm[7]  = lm[0] - ANT * 0.10 + LEFT * 0.07
    lm[8]  = lm[0] - ANT * 0.10 - LEFT * 0.07

    sh_i, el_i, wr_i = (11,13,15) if side == "L" else (12,14,16)
    hp_i, kn_i, an_i, hl_i, ft_i = (23,25,27,29,31) if side=="L" else (24,26,28,30,32)

    hum = rot(rot(DOWN, sag, sh_flex), abd_axis, sh_abd)
    lm[el_i] = lm[sh_i] + hum * SEG["upper_arm"]
    lm[wr_i] = lm[el_i] + rot(hum, sag, elbow) * SEG["forearm"]

    fem = rot(DOWN, sag, hip_flex)
    lm[kn_i] = lm[hp_i] + fem * SEG["femur"]
    shank = rot(fem, LEFT, knee)
    lm[an_i] = lm[kn_i] + shank * SEG["shank"]

    # The foot is attached to the SHANK, so it inherits the knee and hip rotations.
    # Applying only the ankle rotation leaves the foot in its global orientation and
    # fabricates a bias whenever the leg is bent. Compose local-to-global:
    #   world = R_hip( R_knee( R_ankle( offset ) ) )
    heel_off = -ANT * SEG["heel_back"] - UP * SEG["foot_drop"]
    toe_off  =  ANT * SEG["foot_fwd"]  - UP * SEG["foot_drop"]
    def to_world(off):
        return rot(rot(rot(off, sag, ankle), LEFT, knee), sag, hip_flex)
    lm[hl_i] = lm[an_i] + to_world(heel_off)
    lm[ft_i] = lm[an_i] + to_world(toe_off)

    # mirror the idle limbs so the anatomical frame is well conditioned
    o_sh, o_el, o_wr = (12,14,16) if side=="L" else (11,13,15)
    o_hp, o_kn, o_an, o_hl, o_ft = (24,26,28,30,32) if side=="L" else (23,25,27,29,31)
    lm[o_el] = lm[o_sh] + DOWN * SEG["upper_arm"]
    lm[o_wr] = lm[o_el] + DOWN * SEG["forearm"]
    lm[o_kn] = lm[o_hp] + DOWN * SEG["femur"]
    lm[o_an] = lm[o_kn] + DOWN * SEG["shank"]
    lm[o_hl] = lm[o_an] + heel_off
    lm[o_ft] = lm[o_an] + toe_off

    if body_yaw:
        lm = np.array([rot(p, UP, body_yaw) for p in lm])
    return lm

def project(lm, f=600.0, cam_dist=3.0, w=640, h=480):
    """Pinhole projection to pixels. Camera on +Z looking back toward origin."""
    out = np.zeros((33, 3))
    for i, p in enumerate(lm):
        z = cam_dist - p[2]
        if z < 0.1: z = 0.1
        out[i] = [w/2 + f * p[0] / z, h/2 - f * p[1] / z, 0.0]
    return out
