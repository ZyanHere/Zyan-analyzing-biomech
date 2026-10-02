# Five things here that the brief did not ask for

The assignment asked for twelve joint angles at 30 FPS, with documented performance,
accuracy validation, an indication when a measurement cannot be made, and code
documentation. All of that is in the [README](../../README.md).

This document is about what is in the repository *beyond* those requirements, because it is
spread across a 1500-line findings log and a reviewer should not have to go looking for it.

**None of these is a feature.** They are engineering choices. That distinction is worth
being honest about: nothing here adds a capability the brief did not ask for. What they add
is a reason to believe the numbers.

---

## 1. Separating "is my maths wrong" from "is the model wrong"

The brief asked for accuracy validation against a physical reference. It did not ask for a
test with *exact* ground truth, and a physical reference cannot provide one — a printed
protractor carries +/- 5 degrees of its own error, so it can never tell you whether a 6
degree discrepancy came from your geometry or from the pose model.

So a skeleton is built at known joint angles by forward kinematics, projected to 2D, and
measured back through the production code. Worst error across all twelve measurements, both
sides, every body orientation:

    0.000000 degrees

That eliminates the geometry as an error source entirely. Every remaining error in the
system is attributable to the model, which is what makes statements like "BlazePose adds
15-27 degrees of phantom flexion at a known-straight pose" sayable at all.

It also caught two real bugs before they shipped: a sign flip in the image-frame lateral
axis (2D abduction read +45 where truth was -45, because a subject facing the camera has
their left arm on the image right), and a convention error where four of six measurements
turned out not to be simple three-point interior angles.

> **In one line:** a physical reference can only tell you the total error. Synthetic
> ground truth tells you whose fault it is.

*Evidence: F17, F19, F20. Script: `research/experiments/03_biomechanics/`,
`tests/synthetic.py`.*

---

## 2. A validity signal that needs no ground truth, no calibration and no cooperation

Your femur does not change length. So if the reconstructed one varies 14% from frame to
frame, the reconstruction is wrong — and you can know that without a reference, without
calibrating the camera, and without the subject doing anything in particular.

This matters because the obvious signal, landmark confidence, has a blind spot:

| Failure | Knee error | Landmark visibility | Bone-length variance |
|---|---|---|---|
| seated | **147 degrees** | 0.88 — looks fine | **14.2%** — caught |
| occluded leg | 76 degrees | 0.52 — caught, but only at a 0.55 threshold | 5.2% |

**The seated pose is caught by nothing else.** The model is confident and wrong, which is
the worst failure mode a measurement system can have, and no visibility threshold at any
value detects it.

The same investigation also *removed* a signal. Bone-length **asymmetry** was adopted as a
veto, then demoted to advisory after measurement showed it added no unique coverage and
false-positived on camera height — 14.1% asymmetry on a reading that was accurate to 0.2
degrees. Arguing your own safety check down on evidence is harder than adding one.

> **In one line:** rigid bodies have constant segment lengths, so variation is measurable
> error — no reference required.

*Evidence: F22, F30, F37.*

---

## 3. The refusal is the product, and it comes with a proof

The brief asks the system to indicate when a measurement cannot be made. The interesting
part is a case where it refuses and **cannot be argued out of it**.

Knee flexion could not be validated. Not because of a bug — because of geometry:

    2D sagittal validity  requires  side-on
    knee landmark quality requires  the legs not overlapping
    side-on               causes    the legs overlapping

The orientation that makes a 2D knee measurement valid is the same orientation that makes
one leg occlude the other. Measured: visibility 0.37-0.49 against a 0.55 threshold. **The
validity layer refused the measurement during the validation attempt itself** — the
refusal was not noticed afterwards by a human reading a table.

A single camera cannot satisfy both constraints. That is a property of monocular
measurement, not of this implementation, and the honest response is to withhold the number
and say why. Six of twelve measurements therefore ship with synthetic validation only, and
the README says so in those words.

The same logic produces the system's most visible behaviour: at most **10 of 12**
measurements can ever be valid at once, because flexion lives in the sagittal plane and
abduction in the frontal plane, and one lens cannot present both.

> **In one line:** a confident wrong number is worse than no number, and here is the
> geometric proof that no number is the correct answer.

*Evidence: F29, F32, F33, F34.*

---

## 4. Three rates, because reporting one of them would have been a lie

"30 FPS" is three different numbers, and conflating them is the easiest way to publish a
misleading benchmark:

    source FPS         30    what the camera can deliver
    pipeline throughput 39   what the code can process  (measured on file input)
    displayed FPS      30    results actually drawn

Reporting `1 / displayed_FPS` as latency would have given 33 ms against a true 24.7 ms —
wrong in both directions. Latency is measured from a timestamp stamped at capture and
carried on the frame itself, which is the only reason the number can be honest.

Two consequences that only fall out of keeping them separate:

- **The 30 FPS live figure is the camera's ceiling, not the pipeline's.** The headroom
  claim is defensible because it was measured where the camera is not in the way.
- **A replay has no capture instant**, so its latency is a different quantity. The
  application knows this, labels it `processing` rather than `end-to-end`, and the
  metrics JSON carries the distinction. An early version reported 127,089,632 ms before
  that was separated.

The performance section of the README is **generated** from recorded runs rather than
typed, and `tools/performance_table.py --check` fails if the document and the data disagree.

> **In one line:** a rate is not a latency, and a benchmark that cannot tell them apart
> cannot be trusted about either.

*Evidence: F25, F35, F38.*

---

## 5. Findings that reverse themselves — including three killed fixes

The investigation log contains **seven reversals**, kept rather than tidied away. Three of
them killed changes that looked obviously correct:

| Looked right | Measurement said | Outcome |
|---|---|---|
| Two inference workers for throughput | VIDEO mode is 2.4x faster *and* 82% less jittery, and needs a single worker | Worker pool deleted (F25 -> F36) |
| Yaw from shoulder width / trunk length | 50 degrees of spread at one orientation when posture changes | Replaced with the world-3D shoulder axis (F35 -> F36) |
| Move bone-length variance to 3D to fix a projection artefact | 57.7% vs 57.2% — buys nothing | Not changed (F39) |

And one reversal is simply an error corrected in place: an earlier entry claimed *"no
projection argument can excuse a 33.8 degree reading."* It was wrong; body orientation
excused roughly 80% of it, and the log says so where the original claim was made.

The most recent example is the one worth reading. Two separate reviews of a demo recording
reported that shoulder abduction was broken and that the orientation estimator failed when
the arms were raised. Replaying the recorded landmarks showed that abduction held at a
steady **80-83 degrees** against a true 90, the orientation estimator never
misclassified a single frame, and the real finding was something neither review proposed:
the bone-length validity window is **motion-sensitive**, firing on 57-87% of frames during
fast movement against 0-25% when still.

That limitation was then **deliberately not fixed**, because the same signal is the only
one that catches the seated 147 degree error. It is written up as a limitation instead:

> a biomechanics tool whose trust signal is least willing to answer while the subject is
> moving, which is when the measurement is most wanted.

> **In one line:** the log records what was believed, what was measured, and which of the
> two won — including the times measurement beat me.

*Evidence: F36, F37, F38, F39. The full list is under "Settled by measurement" in
[FINDINGS.md](FINDINGS.md).*

---

## What this is not

It is worth saying plainly: **no additional capability was built.** There is no rep
counter, no session report, no second camera, no phone support. The brief's scope was
delivered and not expanded.

What was spent instead was effort on knowing whether the numbers are true, and on saying
so precisely where they are not. Twelve measurements ship; six have synthetic validation
only; one is unvalidated and withheld in the exact pose that would validate it. All three
of those statements are in the README rather than in a footnote.
