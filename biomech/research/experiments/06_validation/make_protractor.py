"""Printable wall protractor for absolute-angle validation (item 4).

Elbow flexion is measured from a STRAIGHT arm:
    0 deg   = forearm continues straight on from the upper arm
    90 deg  = forearm perpendicular
    150 deg = forearm folded back toward the shoulder

So the fan is measured from the DOWNWARD direction (away from the shoulder),
not from the upper-arm line itself.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from lib.paths import fixture

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANGLES = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150]
CX, CY, R = 0.30, 0.58, 0.46

fig, ax = plt.subplots(figsize=(8.27, 11.69))
ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.42); ax.axis("off")
ax.set_aspect("equal")

ax.plot([CX, CX], [CY, CY + 0.38], lw=4, color="black", solid_capstyle="butt")
ax.text(CX - 0.02, CY + 0.40, "UPPER ARM\n(toward shoulder)", fontsize=11,
        fontweight="bold", ha="center", va="bottom")

for a in ANGLES:
    th = np.radians(a)
    dx, dy = np.sin(th), -np.cos(th)          # 0 deg = straight down
    ax.plot([CX, CX + R*dx], [CY, CY + R*dy], lw=1.6, color="#c62828", alpha=.9)
    lx, ly = CX + (R + 0.055)*dx, CY + (R + 0.055)*dy
    ax.text(lx, ly, f"{a}", fontsize=16, fontweight="bold", color="#c62828",
            ha="center", va="center")

ax.plot(CX, CY, "o", ms=15, color="black", zorder=5)
ax.text(CX - 0.035, CY, "ELBOW\nHERE", fontsize=10, fontweight="bold",
        ha="right", va="center")
ax.text(0.5, 1.38, "Elbow flexion reference", fontsize=17, fontweight="bold", ha="center")
ax.text(0.5, 1.335,
        "Print at 100% scale - turn OFF 'fit to page'.\n"
        "Tape to a wall at elbow height. Stand SIDE-ON to the camera, arm next to the wall.\n"
        "Elbow on the dot, upper arm along the BLACK line, forearm along a RED line.\n"
        "0 = arm completely straight.   150 = hand near your shoulder.",
        fontsize=10, ha="center", va="top", color="#333")

plt.savefig(fixture("protractor_A4.pdf"), dpi=300, bbox_inches="tight")
plt.savefig(fixture("protractor_A4.png"), dpi=170, bbox_inches="tight")
print("regenerated protractor")
