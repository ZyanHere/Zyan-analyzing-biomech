"""E3 (item 7 + noise sensitivity): why the ankle is worst, and what helps.

Injects known Gaussian noise into synthetic landmarks and measures how much
angular error each measurement produces per mm of landmark error. Short
segments amplify; this quantifies by how much.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
from lib import synth
from lib import biomech_ref as B

rng = np.random.default_rng(7)
POSE = dict(elbow=60, knee=45, sh_flex=45, sh_abd=0, hip_flex=25, ankle=-10)

def noisy(lm, sigma_mm):
    return lm + rng.normal(0, sigma_mm/1000.0, lm.shape)

print("="*74)
print("A. Noise amplification: degrees of angle error per mm of landmark error")
print("="*74)
base = synth.build(**POSE, side="L")
print(f"  {'measurement':<20}{'1mm':>8}{'3mm':>8}{'5mm':>8}{'10mm':>8}   deg per mm")
amp = {}
for nm, fn in B.ALL.items():
    truth = fn(base, "L")
    row, slope = f"  {nm:<20}", None
    for s in (1,3,5,10):
        errs = [fn(noisy(base, s), "L") - truth for _ in range(600)]
        sd = np.nanstd(errs); row += f"{sd:>8.2f}"
        if s == 5: slope = sd/5
    amp[nm] = slope
    print(row + f"{slope:>13.2f}")
print("\n  (std dev of the reported angle, in degrees, for that landmark noise)")

print()
print("="*74)
print("B. Segment lengths - the cause")
print("="*74)
for nm, a, b in [("forearm", 13, 15), ("upper arm", 11, 13), ("femur", 23, 25),
                 ("shank", 25, 27), ("foot ankle->toe", 27, 31), ("foot heel->toe", 29, 31)]:
    print(f"  {nm:<18}{np.linalg.norm(base[a]-base[b])*100:>7.1f} cm")

print()
print("="*74)
print("C. Ankle: which foot definition, and does sagittal projection help?")
print("="*74)
print(f"  {'variant':<34}{'err@0':>9}{'err@-30':>9}{'noise 5mm':>11}")
for use_heel in (False, True):
    for project in (False, True):
        lab = f"{'heel->toe' if use_heel else 'ankle->toe':<12} {'sagittal' if project else 'raw 3D':<9}"
        errs0, errs30 = [], []
        for true_v in (0, -30):
            lm = synth.build(**{**POSE, "ankle": true_v}, side="L")
            got = B.ankle_angle(lm, "L", use_heel=use_heel, project=project)
            (errs0 if true_v == 0 else errs30).append(got - true_v)
        lm = synth.build(**{**POSE, "ankle": -10}, side="L")
        t = B.ankle_angle(lm, "L", use_heel=use_heel, project=project)
        sd = np.nanstd([B.ankle_angle(noisy(lm,5), "L", use_heel=use_heel, project=project) - t
                        for _ in range(600)])
        print(f"  {lab:<34}{errs0[0]:>+9.2f}{errs30[0]:>+9.2f}{sd:>11.2f}")
print("\n  err@0 is the bias at anatomical neutral: how far from 0 the reading sits")
print("  when the true ankle angle is exactly 0.")

print()
print("="*74)
print("D. Does a neutral-calibration offset fix the ankle->toe definition?")
print("="*74)
lm0 = synth.build(**{**POSE, "ankle": 0}, side="L")
off = B.ankle_angle(lm0, "L", use_heel=False, project=True)
print(f"  measured offset at true neutral (ankle->toe): {off:+.2f} deg")
print(f"  {'true':>7}{'uncalibrated':>15}{'calibrated':>13}{'heel->toe':>12}")
for tv in (-50,-30,-15,0,10,20):
    lm = synth.build(**{**POSE, "ankle": tv}, side="L")
    raw = B.ankle_angle(lm, "L", use_heel=False, project=True)
    heel = B.ankle_angle(lm, "L", use_heel=True, project=True)
    print(f"  {tv:>7}{raw:>15.2f}{raw-off:>13.2f}{heel:>12.2f}")

print()
print("="*74)
print("E. Is the ankle->toe bias constant, or does it move with leg posture?")
print("   A constant bias is fixable by calibration. A varying one is not.")
print("="*74)
print(f"  {'hip':>5}{'knee':>6}{'ankle':>7}{'ankle->toe bias':>18}{'heel->toe bias':>17}")
biases = []
for hip in (0, 25, 60):
    for knee in (0, 45, 90):
        for tv in (-30, 0, 20):
            lm = synth.build(hip_flex=hip, knee=knee, ankle=tv, side="L")
            b1 = B.ankle_angle(lm,"L",use_heel=False) - tv
            b2 = B.ankle_angle(lm,"L",use_heel=True)  - tv
            biases.append(b1)
            if tv == 0:
                print(f"  {hip:>5}{knee:>6}{tv:>7}{b1:>+18.2f}{b2:>+17.2f}")
print()
print(f"  ankle->toe bias across all postures: mean {np.mean(biases):+.2f}, "
      f"sd {np.std(biases):.2f}, range {np.min(biases):+.1f} to {np.max(biases):+.1f}")
print("  if sd is large, a single calibration constant cannot fix this definition.")
