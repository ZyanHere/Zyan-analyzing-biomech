# Findings log

Measured facts only. Every claim here has a script in `experiments/` that produced it.
Written as the investigation runs, so later entries may correct earlier ones.

## Test hardware

| | |
|---|---|
| CPU | AMD Ryzen 7 4800H, 8 cores / 16 threads |
| GPU | NVIDIA RTX 3050 Laptop 4GB + AMD Radeon integrated |
| RAM | 15.4 GB |
| OS | Windows 11 Home Single Language 26200 |
| Camera | USB2.0 HD UVC WebCam (integrated) |
| Python | 3.14.7 |
| MediaPipe | 1.0.1 |
| OpenCV | 5.0.0 |

---

## F1 - The GPU cannot be used. Not a configuration problem.

From the installed MediaPipe source, `tasks/python/core/base_options.py`:

> `delegate: Acceleration to use. Supported values are GPU and CPU.`
> `GPU support is currently limited to Ubuntu platforms.`

The RTX 3050 is unusable by MediaPipe on Windows at any setting. All inference is CPU.
Not a flag we failed to find - the binding does not implement it for this platform.

**Decision:** target the CPU deliberately rather than pretend otherwise. Reaching
30 FPS becomes a scheduling problem, not an accelerator problem.

## F2 - MediaPipe 1.0 removed the legacy API

`mediapipe` now exports only `Image`, `ImageFormat`, `tasks`. The old `mp.solutions.pose`
API most tutorials use is gone; the Tasks API is the only option. Tasks is now a C binding
(`base_options_c_lib`, `to_ctypes`), so pre-1.0 behaviour cannot be assumed - hence F3.

## F3 - MediaPipe releases the GIL. Threads give real parallelism.

`experiments/a1_model_bench.py`

| Workload | 1 thread | 4 threads | Speedup |
|---|---|---|---|
| Pure Python (control) | 380 ms | 1522 ms | **1.00x** |
| NumPy (control) | 621 ms | 899 ms | 2.76x |
| MediaPipe lite | 18.5 inf/s | 55.9 inf/s | **3.03x** |
| MediaPipe full | 15.3 inf/s | 45.6 inf/s | **2.98x** |

Controls included because a 3x speedup is only meaningful next to a workload that
gets 1.0x on the same machine.

**Decision:** threads, not processes. No IPC, no frame serialisation across
process boundaries, shared-memory frames are free.

## F4 - No model variant reaches 30 FPS single-threaded

`experiments/a1_model_bench.py`, 80 iterations, 1280x720 input, warm.

| Model | Load | p50 | p95 | Single-thread ceiling |
|---|---|---|---|---|
| lite | 12.0 s (first ever load) | 41.4 ms | 49.6 ms | 24 FPS |
| full | 219 ms | 49.6 ms | 58.1 ms | 20 FPS |
| heavy | 274 ms | 111.0 ms | 116.7 ms | 9 FPS |

The 12 s on lite is one-time XNNPACK delegate init, not per-model cost.

**This is the central fact of the project.** A sequential
`read -> detect -> draw` loop cannot meet the mandatory 30 FPS on this hardware.
Concurrency is what makes the requirement reachable, so the pipeline architecture
is load-bearing rather than decorative.

Note the latency/throughput split: 4 threads give 55.9 inferences/sec of *throughput*,
but any single frame still takes ~41 ms to process. The assignment asks for both
numbers separately, and this is why they differ.

## F5 - The camera delivered 7.8 real FPS. Auto-exposure was the cause.

`experiments/b3_distinct.py`, `experiments/b4_exposure.py`

First measurement suggested MSMF gave 30 FPS and DirectShow 7.5 FPS. **That was wrong.**
Hashing frame contents showed MSMF returns each frame up to 4 times:

| Backend | reads/sec | **unique frames/sec** | duplicates |
|---|---|---|---|
| MSMF | 30.0 | **7.8** | 74% |
| DSHOW | 7.8 | **7.8** | 0% |

Both backends delivered the same true rate. MSMF inflated it with repeats; DirectShow
was honest. Counting `read()` calls measures the loop, not the camera.

Root cause is auto-exposure lengthening exposure time in low light:

| Exposure | Unique FPS | Mean brightness |
|---|---|---|
| auto (default) | 7.7 | 123 / 255 |
| manual 2^-4 s | 16.0 | 77 / 255 |
| manual 2^-5 s | **30.0** | 47 / 255 |
| manual 2^-6 s | 30.0 | 25 / 255 |

Resolution also matters: 320x240 and 640x480 both reach 30 FPS at manual exposure;
**1280x720 caps at 10 FPS** regardless.

**Decision:** 640x480, manual exposure, and the app must set exposure explicitly -
leaving it automatic costs 75% of the frame rate. A dedicated capture thread keeping
only the newest frame is also required, since MSMF will otherwise serve stale repeats.

**Open trade-off:** 30 FPS costs ~2.6x image brightness. Whether that darkness degrades
landmark quality is unresolved and is the next experiment. If it does, the honest answer
may be a lower frame rate with a better image, or more light in the room.

## F6 - Visibility scores differ sharply between variants (preliminary)

Same seated frame, legs occluded by a desk:

| Model | left elbow | left knee | left ankle |
|---|---|---|---|
| lite | 0.64 | 0.25 | 0.07 |
| full | 0.08 | 0.01 | 0.01 |
| heavy | 0.46 | 0.00 | 0.00 |

For genuinely invisible joints, `full` and `heavy` report near-zero while `lite` reports
0.25. If that holds on full-body frames, lite is overconfident about joints it cannot see -
which matters directly for the "indicate unreliable state" requirement, where an
overconfident score is worse than a slow model.

Preliminary: one frame, occluded lower body. Needs a full-body test before it counts.

## F7 - Auto-exposure degrades accuracy 10x, not just frame rate

`experiments/c1_record.py`, `experiments/c2_analyse.py`. Subject standing still, so all
variation is measurement error. Only joints with visibility > 0.9 are included.

Angle jitter, standard deviation in degrees, 2D image coordinates:

| Joint | auto | e4 | **e5** | e6 |
|---|---|---|---|---|
| elbow L | 23.35 | 5.04 | **2.10** | 3.61 |
| elbow R | 21.98 | 2.46 | **1.53** | 1.31 |
| shoulder L | 10.77 | 2.22 | **1.12** | 1.85 |
| shoulder R | 11.34 | 2.11 | **0.70** | 1.41 |
| hip L | 2.25 | 2.35 | **1.81** | 1.52 |
| hip R | 1.31 | 2.77 | **1.86** | 1.86 |

At auto exposure a motionless elbow reads +/- 23 degrees. Cause is motion blur: the long
exposure that auto-exposure selects in indoor light smears each frame, so landmarks
wander. Setting exposure manually improves jitter ~10x *and* triples frame rate.

F5 framed exposure as a frame-rate/brightness trade-off. That framing was incomplete -
there is no accuracy cost to the darker image. Brightness is not the quantity that
matters; exposure *time* is.

**Decision:** exposure 2^-5 s, set explicitly at startup. Never leave it automatic.

## F8 - 2D image coordinates are less noisy than the model's 3D output

Same session, joints with visibility > 0.9, degrees std dev:

| Joint | 2D | 3D world | 
|---|---|---|
| shoulder L | **1.12** | 2.91 |
| shoulder R | **0.70** | 2.68 |
| hip L | **1.81** | 4.20 |
| hip R | **1.86** | 3.50 |
| elbow L | **2.10** | 2.03 |
| elbow R | **1.53** | 1.94 |

The regressed depth coordinate adds 2-3x noise for joints facing the camera. Expected:
`z` is inferred from a flat image, not measured.

**Preliminary decision:** prefer 2D for in-plane joints; 3D remains a candidate only
where 2D is geometrically degenerate. Not yet tested on out-of-plane poses, which is
where 3D might earn its noise.

## F9 - The visibility score is a usable trust signal

Same session. Upper body fully in frame, lower legs out of frame:

| Joint group | reported visibility |
|---|---|
| shoulders, hips | 0.98 - 1.00 |
| elbows, wrists | 0.92 - 1.00 |
| knees | 0.65 - 0.89 |
| ankles | 0.06 - 0.19 |
| feet | 0.03 - 0.10 |

The model correctly reported near-zero confidence for joints outside the frame. This is
a sound basis for the mandatory "indicate that state rather than displaying a misleading
value" requirement.

## F10 - Invalid run, kept deliberately: stable is not the same as correct

The knee and ankle numbers from the F7 session are void - the subject's lower legs were
outside the frame, so the model was extrapolating. Recorded here because of what the bad
data looked like:

| Joint | auto | e4 | e5 | e6 |
|---|---|---|---|---|
| knee L | 51.88 | **2.48** | 50.38 | 42.33 |
| ankle R | 53.12 | **7.03** | 44.75 | 42.98 |

At e4 the knee jitter looks superb - better than any legitimate reading in F7. It is
worthless: the model had locked onto a confidently wrong constant guess for a joint it
could not see. Visibility for those joints was 0.04 - 0.17 at the time.

**Consequence for validation:** jitter alone cannot establish accuracy, because a
constant wrong answer scores perfectly on it. Any validation must include an
independent reference, not just stability. The recording script now blocks until the
feet are genuinely visible.

---

## F11 - CORRECTS F8. 2D vs 3D is a per-joint decision, not a global one

`experiments/c4_session.py`, `experiments/c5_analyse.py`. Full body in frame,
subject still, exposure 2^-5. Degrees std dev.

| Joint | 2D | 3D | Quieter |
|---|---|---|---|
| shoulder L | **1.10** | 1.87 | 2D |
| shoulder R | **0.98** | 1.48 | 2D |
| hip L | **2.57** | 3.09 | 2D |
| hip R | **1.49** | 1.69 | 2D |
| elbow L | 2.10 | **1.79** | 3D |
| elbow R | 1.57 | **1.40** | 3D |
| knee L | 9.32 | **6.83** | 3D |
| knee R | 1.82 | **1.26** | 3D |
| ankle L | 8.93 | **2.88** | 3D (3x) |
| ankle R | 6.05 | **1.11** | 3D (5x) |

F8 concluded "2D is less noisy than 3D". That was measured on a recording with the
lower body out of frame, i.e. on shoulders and hips only, and generalised wrongly.

The real pattern: **2D wins for torso-referenced joints** (shoulder, hip) - the torso is
large and near-parallel to the image plane, so its projection is stable and `z` only adds
noise. **3D wins for limb joints** (elbow, knee, ankle), decisively at the ankle, where
the foot segment is short and frequently angled out of plane.

**Decision:** representation is chosen per joint, justified by this table.

## F12 - Out-of-plane error measured: 2D carries an 18 degree bias past 40 degrees

`rotate` phase. Elbow held fixed while the body turned, so the true angle is constant by
construction and every deviation is projection error. Body rotation estimated from
apparent shoulder width against its own 95th percentile.

| Body rotation | 2D error | 3D error |
|---|---|---|
| 0-12 deg | 0.0 | 0.0 |
| 12-25 deg | +1.1 | -0.4 |
| 25-40 deg | -3.5 | -1.8 |
| **40-55 deg** | **-18.1** | **-7.0** |
| **55-70 deg** | **-17.4** | **-5.0** |
| **70-90 deg** | **-18.4** | **-2.8** |

2D survives to roughly 40 degrees of rotation, then settles into a **stable -18 degree
bias**. Stable, not noisy - it is a systematic lie, which is worse, because averaging and
smoothing will not remove it and it looks perfectly calm on screen. 3D stays within
7 degrees across the whole sweep.

Ground truth here costs nothing: the subject's own held joint is the reference. No
goniometer, no mocap, fully reproducible from the saved fixture.

**Decision:** 3D primary for limb joints. Rotation is *detectable* at runtime from
apparent shoulder width, so the app can also flag "subject too rotated for this
measurement" rather than silently reporting an 18 degree error.

**Caveat:** at 0 rotation 2D read 172.3 deg and 3D read 146.3 deg for the same elbow -
26 degrees apart. The arm was angled toward the camera even at nominal zero, so the
2D reading was already badly foreshortened at what we assumed was the best case.
Absolute accuracy at zero rotation is therefore still unvalidated; this experiment
measures *change*, not absolute correctness. An external reference is still required.

**Protocol note:** the subject was asked to hold 90 degrees and actually held ~172 deg
(2D). Method is unaffected - it needs the angle constant, not any particular value.

## F13 - Raw output is unusable live. Filtering is mandatory, quantified.

Frame-to-frame change, 95th percentile, during the `squat` phase at ~20 FPS:

| Phase | Joint | p95 jump between consecutive frames |
|---|---|---|
| elbow | elbow R | 10.3 deg |
| squat | knee L | **46.6 deg** |
| squat | knee R | **39.0 deg** |
| squat | ankle L | **35.8 deg** |

A knee does not move 46 degrees in 50 ms during a slow squat. That is display noise.
Any unfiltered readout is unreadable during real movement.

Also: knee R confidence fell to **0.50** during the side-on squat - the near leg occludes
the far leg. A genuine single-camera limitation, not a defect.

## F14 - First external accuracy signal: elbow range matches the goniometry chart

`elbow` phase, right elbow swept through its full range:

- measured interior angle: 30.8 deg to 177.7 deg
- converted to chart convention (flexion = 180 - interior): **2.3 deg to 149.2 deg**
- goniometry chart, elbow flexion: **0 to 150 deg**

An independent reference agreeing to within ~2 degrees at both ends of the range.
Weak evidence - it confirms the range endpoints and the convention conversion, not
accuracy at intermediate angles - but it is the first datapoint not generated by the
system grading itself.

---

## F15 - One Euro filter chosen. EMA is strictly dominated on the knee.

`experiments/d1_filters.py`. Replayed from the saved session, so deterministic and
rerunnable. Jitter measured on `still`, lag and range on the movement phases.

| Filter | Elbow jitter | Elbow lag | Knee jitter | Knee lag | Range kept |
|---|---|---|---|---|---|
| raw | 1.40 | 0 ms | 6.83 | 0 ms | 100% |
| EMA a=0.5 | 1.05 (-25%) | 0 ms | 4.85 (-29%) | 0 ms | 91-94% |
| EMA a=0.3 | 0.94 (-32%) | 96 ms | 4.65 (-32%) | 0 ms | 85-90% |
| EMA a=0.15 | 0.86 (-38%) | **240 ms** | 5.20 (-24%) | **192 ms** | 77-87% |
| median k=5 | 1.03 (-26%) | 96 ms | 4.84 (-29%) | 48 ms | 84-93% |
| **1euro b=0.01** | **0.92 (-34%)** | **48 ms** | **4.39 (-36%)** | **0 ms** | 88-89% |
| 1euro b=0.05 | 0.96 (-31%) | 0 ms | 4.85 (-29%) | 0 ms | 92-93% |

EMA a=0.15 is the clearest illustration of why a fixed smoothing constant is the wrong
tool: on the elbow it buys the best jitter at 240 ms of lag (7 frames at 30 FPS, plainly
visible), and on the knee it is **worse on both axes at once** - less smoothing *and*
192 ms of lag. A single constant cannot suit both a slow joint and a fast one.

One Euro varies its cutoff with observed velocity, so it smooths hard when the joint is
still and gets out of the way when it moves. Equal or better jitter reduction at 0-48 ms.

**Decision:** One Euro, beta ~= 0.01, per angle.

**Caveat carried forward:** every filter shaves the peaks. One Euro retains 88-89% of
the movement range, so a filtered signal under-reports *maximum* range of motion by
roughly 10%. Live display should be filtered; any peak or ROM statistic must be taken
from the unfiltered signal. Two consumers, two signals.

**Not yet tested:** filtering the landmark positions *before* computing angles, rather
than filtering the resulting angle. That may do better, since it keeps the skeleton
geometrically consistent. Worth an experiment if time allows.

---

## F16 - Model choice: `full`. Corrects the F6 suspicion about `lite`.

`experiments/d2_models.py`. All three variants replayed through the *same* recorded
frames - impossible with a live subject, and the reason the session was recorded.

| Model | p50 | p95 | mean vis | ankle vis | knee jitter | rotation drift |
|---|---|---|---|---|---|---|
| lite | **36.8 ms** | 44.8 | 0.97 | 0.97 | 5.85 | -3.9 |
| **full** | 44.5 ms | 56.6 | 0.96 | 0.94 | **2.01** | **-3.2** |
| heavy | 101.1 ms | 114.0 | 0.99 | 0.99 | 2.76 | **-23.6** |

3D jitter while still, deg sd:

| Model | elbow R | knee L | ankle R | shoulder R |
|---|---|---|---|---|
| lite | 1.44 | 5.85 | **1.19** | 1.37 |
| full | 1.03 | **2.01** | 1.75 | **0.99** |
| heavy | **0.81** | 2.76 | 1.54 | 1.03 |

**Decision: `full`.** 8 ms slower than `lite` buys ~3x lower knee jitter on a mandatory
joint. At 44.5 ms it still yields ~67 inferences/sec across 4 threads (F3), comfortably
above the 30 FPS requirement, so the extra cost is affordable.

`heavy` is rejected on two counts: 2.3x the latency, and -23.6 deg of drift during the
rotation sweep against -3.2 for `full`. The largest model is by far the worst on the
out-of-plane case that matters most. Surprising enough to warrant re-checking before it
goes in the final report, but the test was identical across all three variants.

**This corrects F6.** F6 suspected `lite` of overconfidence about joints it could not
see. On valid full-body data all three report 0.94-0.99 with ankles properly visible.
F6 was measured on a seated frame with the legs under a desk; the model was correctly
reporting low confidence, and that correct behaviour was misread as a flaw.

**Caveat:** replay is from mp4v-compressed video, so all three saw slightly degraded
frames versus live capture. It affects the variants equally, so the comparison holds,
but absolute latency and jitter may differ marginally from live.

---

## F17 - Item 1: the conventions, derived and executable

`experiments/biomech_ref.py`. Anatomical frame rebuilt per frame (see F18):

    up       = normalise(mid_shoulder - mid_hip)
    left     = normalise(L_hip - R_hip, orthogonalised against up)
    anterior = cross(left, up)
    sagittal = span(up, anterior)    frontal = span(up, left)

| Measurement | Landmarks | Plane | Neutral | Signed | Conversion |
|---|---|---|---|---|---|
| Elbow flexion | shoulder, elbow, wrist | limb plane (hinge) | straight arm | no | `180 - interior` |
| Knee flexion | hip, knee, ankle | limb plane (hinge) | straight leg | no | `180 - interior` |
| Shoulder flex/ext | hip, shoulder, elbow | **sagittal** | arm hanging | **yes** | `atan2(h.ant, -h.up)` |
| Shoulder abd/add | hip, shoulder, elbow | **frontal** | arm hanging | **yes** | `atan2(h.lat, -h.up)` |
| Hip flex/ext | shoulder, hip, knee | **sagittal** | standing | **yes** | `atan2(f.ant, -f.up)` |
| Ankle dorsi/plantar | knee, ankle, foot_index | **sagittal** | ~90 shank-to-foot | **yes** | `90 - interior` |

Key points that a naive three-point implementation gets wrong:

- Shoulder flex/ext and abd/add use the **same three landmarks** but project into
  **different planes**. One interior angle cannot produce both.
- Flexion/extension and abduction/adduction are **signed**; the chart's two ranges
  (e.g. flexion 0-180, extension 0-60) are the two signs of one measurement.
- The lateral axis is **flipped for the right side** so that positive abduction means
  "away from the midline" on both sides.
- Ankle neutral is ~90 degrees, not 180. Elbow and knee neutral are 180 interior -> 0.
- The knee flexes **backwards** while elbow, hip and shoulder flex **forwards**. In the
  synthetic model this is an opposite rotation axis. Anatomy, not a sign bug.

## F18 - The model's 3D frame is camera-aligned, not body-locked

`experiments/e1_axes.py`. Measured axes: up = **-y**, subject's left = **+x**,
anterior = **-z**, right-handed (`dot(cross(left,up), anterior) = +1.0`).

Hip-vector angular spread across a phase:

| Phase | mean | max |
|---|---|---|
| still | 3.3 deg | 9.0 deg |
| rotate | **34.2 deg** | **138.2 deg** |

World landmarks rotate with the body relative to a camera-fixed frame. They are metric
3D in the *camera's* orientation, not an anatomical frame.

**Consequence:** projecting onto a fixed world plane and calling it sagittal is wrong.
The anatomical frame must be reconstructed from landmarks every frame. This is what
makes an angle mean the same thing when the subject turns.

## F19 - Item 5: the conventions maths is exact. Error is 0.000000 degrees.

`experiments/synth.py`, `experiments/e2_synth_validate.py`. A synthetic skeleton is
posed at exactly known angles by forward kinematics, then measured by F17's code.

- **A.** Every measurement, both sides, full chart range: worst error **0.000000 deg**.
- **B.** Body yawed 0/30/60/90/135/180 deg: worst error **0.0000 deg**.

Any error the finished system exhibits therefore originates in the *pose model* or the
*capture*, never in the angle mathematics or the convention conversion. That separation
is the point of the experiment, and it is what lets model error be attributed honestly
later (E5).

## F20 - CORRECTS F11. 2D projection error is catastrophic, not moderate.

Same synthetic rig, landmarks projected through a pinhole camera. Error in degrees for
2D hinge angles as the body rotates (side-on = 90):

| True elbow flexion | 0 (facing) | 15 | 30 | 45 | 60 | 75 | 90 |
|---|---|---|---|---|---|---|---|
| 30 | **-27.9** | -19.3 | -12.0 | -6.4 | -2.6 | -0.5 | 0.0 |
| 60 | **-53.3** | -28.6 | -13.8 | -5.9 | -1.8 | -0.1 | 0.0 |
| 90 | **+48.0** | +11.4 | +5.9 | +3.5 | +2.1 | +1.0 | 0.0 |
| 120 | **+54.7** | +33.9 | +19.4 | +10.2 | +4.7 | +1.5 | 0.0 |

Facing the camera, a true 60 degree elbow bend reads **6.7 degrees**. F12 measured
-18 deg live; the true worst case is -53 deg, because the subject's arm was never fully
perpendicular to the image plane during that recording.

Test D: a true 90 deg shoulder flexion appears as a 160 deg in-image angle at yaw 0,
and exactly 90.0 at yaw 90. Facing the camera, the sagittal plane is perpendicular to
the image plane, so flexion projects onto almost nothing.

**This invalidates F11.** F11 compared *interior angles between three landmarks*, not
the anatomical measurements F17 defines. Facing the camera with arms down, the
hip-shoulder-elbow interior angle sits near zero and is beautifully stable - and it is
**not shoulder flexion**. F11 measured low noise on a quantity that was not the thing
being asked for.

**Corrected decision:** 3D is primary for all six measurements. 2D is a cross-check,
valid only when the relevant plane is near-parallel to the image plane. This supersedes
the per-joint split in settled-decision 6.

---

## F21 - Item 6: the model's 3D output carries 15-27 deg of phantom flexion

`experiments/e5_model_error.py`. F19 proved the maths exact, so error at a known pose
is attributable to the pose estimator. Standing relaxed, a knee sits at 0-10 deg:

| Joint | 3D world | 2D image | gap | visibility |
|---|---|---|---|---|
| knee L | 30.3 | 15.2 | +15.1 | 0.95 |
| knee R | 24.0 | **2.0** | +22.0 | 0.98 |
| elbow L | 28.6 | 11.7 | +17.0 | 0.98 |
| elbow R | 31.7 | **5.0** | +26.7 | 0.96 |

A straight standing leg is vertical and lies in the image plane, so 2D suffers no
foreshortening in this pose - and it reads 2.0 deg, which is correct. The 3D
reconstruction adds 15-27 deg of flexion that is not there.

Error grows with rotation, so it is not a fixed offset:

| Body yaw | knee L 3D | knee R 3D |
|---|---|---|
| 0-15 | 21.8 | 23.8 |
| 30-45 | 32.7 | 22.2 |
| 60-90 | **35.7** | 28.8 |

**A single calibration constant cannot remove this.**

## F22 - Bone-length consistency: 3D error measured with no ground truth

Segment lengths are constant in a rigid body, so variation is reconstruction error.
Requires no reference, no equipment, and no subject cooperation.

| Segment | Left | Right | L/R asymmetry | frame-to-frame cv |
|---|---|---|---|---|
| upper arm | 24.7 cm | 25.5 cm | 3.4% | 1.3-1.4% |
| forearm | 24.3 cm | 23.4 cm | 3.8% | 1.3-1.8% |
| femur | 36.1 cm | 40.1 cm | **11.2%** | **8.0%** / 2.5% |
| shank | 35.8 cm | 40.2 cm | **12.3%** | 3.6% / 1.5% |
| foot | 12.3 cm | 13.9 cm | **12.5%** | 8.1% / 3.1% |

A human is symmetric to ~1-2%. The model's legs differ by 11-12%, and the left femur's
reported length varies 8% frame to frame **while the subject stands still**.

Arms reconstruct far better than legs (1.3-1.8% cv vs 2.5-8.0%). Plausibly because the
legs are further from the camera, lower in frame, and partly self-occluding.

**Also: the foot measures 12-14 cm heel-to-toe against a real ~25 cm.** The model places
`heel` and `foot_index` at roughly half their true separation. That halves the segment
and doubles noise amplification (E3), and is the actual mechanism behind the ankle's
poor behaviour - distinct from both the "short segment" and "bad landmark" hypotheses,
though closest to the latter.

**Use as a runtime signal:** bone-length variance is computable live and is a candidate
validity check, independent of the model's own `visibility` score.

## F23 - CONTESTS F20. 2D vs 3D is now genuinely open, pending item 4.

F20 concluded "3D primary for all measurements" from synthetic data - where landmarks
are perfect by construction. That experiment measured the *geometry* and assumed the
*inputs*. F21 and F22 measure the inputs.

| | 2D image coordinates | 3D world landmarks |
|---|---|---|
| Landmark accuracy | good (2.0 deg at a known 0 deg pose) | poor (24 deg at the same pose) |
| Geometry | wrong when the plane is not facing the camera (up to 53 deg, F20) | exact at any orientation (F19) |
| Fails when | subject faces the camera for a sagittal measurement | always, by a roughly constant-to-growing margin |

Neither dominates. This reframes the brief's remark that the subject "may change their
orientation... to expose the sagittal plane" - that may not be a convenience but the
only accurate path, since a correctly oriented 2D measurement avoids both failure modes.

**Cannot be settled from the existing recording.** Both candidates need comparing
against a *known true angle*, which is item 4. Until then, settled-decision 6 is
provisional and the choice is open.

---

## F24 - Item 9: the heavy anomaly is real, and models disagree by 16 deg

`experiments/e6_heavy_recheck.py`. F16 binned each model by its *own* shoulder-width
yaw estimate, so each was scored on its own ruler. Re-run with a single shared estimate:

| Body yaw | lite | full | heavy |
|---|---|---|---|
| 15-30 | +0.9 | +1.5 | +2.7 |
| 30-45 | +3.0 | +6.4 | +12.3 |
| 60-90 | **+3.9** | **+4.2** | **+21.6** |

Confirmed: `heavy` genuinely degrades out of plane. F16's rejection stands.

**The larger finding is panel C.** On identical frames the three models disagree with
each other by 10-18 degrees:

| Yaw | lite | full | heavy | spread |
|---|---|---|---|---|
| 0-15 | 34.2 | 37.3 | 21.3 | **16.0** |
| 45-60 | 38.2 | 34.9 | 20.1 | **18.1** |
| 60-90 | 38.1 | 41.5 | 42.9 | 4.8 |

Model choice alone injects more error than most effects measured in this investigation,
and without an external reference there is no way to know which is closest to truth.
This makes item 4 load-bearing rather than merely outstanding.

## F25 - Item 11: inference is 96-99% of the budget; more threads make it worse

`experiments/e7_pipeline.py`. Threaded pipeline, bounded queues, drop-oldest.

| Workers | FPS | Inference p50 | Biomech p50 | e2e p50 | e2e p95 | drops |
|---|---|---|---|---|---|---|
| **1** | 34.4 | 27.2 | 0.69 | **41.7** | 46.8 | 643 |
| **2** | 49.9 | 38.9 | 0.81 | 52.9 | 63.7 | 516 |
| 3 | 69.9 | 41.7 | 0.88 | 54.5 | 65.1 | 354 |
| 6 | 95.4 | 60.7 | 1.01 | 69.7 | 84.2 | 147 |
| 8 | 100.0 | 76.5 | 1.15 | 84.8 | 101.3 | 103 |

- **Bottleneck is inference**, at 96-99% of the budget. Biomechanics costs under 1.2 ms.
  Optimising the angle maths would be wasted effort.
- **Throughput and latency move in opposite directions.** 8 workers triple throughput
  and double end-to-end latency. Since the target is 30 FPS, throughput above ~50 is
  worthless while latency is always felt.
- A **single worker already meets 30 FPS** (34.4) at the lowest achievable latency.

**Decision: 2 inference workers.** 49.9 FPS, 52.9 ms e2e, margin for slow frames.

**Note:** this harness initially reported a flat 8.0 FPS at every worker count - its
output queue was bounded at 64 with no consumer, so it measured itself. Fixed.

## F26 - Item 10: DirectShow, not MSMF. Corrects F5 a second time.

`experiments/e8_capture_final.py`, measured after the exposure fix and in brighter light.

| Backend | Exposure | reads/s | **unique/s** | dupes | gap p50 | p95 |
|---|---|---|---|---|---|---|
| MSMF | auto | 30.2 | 15.2 | 50% | 62.6 | 78.4 |
| MSMF | fixed | 30.4 | 27.2 | 11% | 31.4 | **92.2** |
| **DSHOW** | auto | 30.0 | **30.0** | **0%** | 32.1 | **47.9** |
| **DSHOW** | fixed | 30.0 | **30.0** | **0%** | 32.1 | 48.2 |

DirectShow delivers zero duplicates in every condition and roughly half the p95 jitter.
MSMF still serves stale repeats 11% of the time even with exposure fixed.

DSHOW-auto now reaches 30 FPS where it managed 7.7 in F5's session: the room is
brighter. The auto-exposure failure is therefore **lighting-dependent and fragile**.
Exposure is still set explicitly - for reproducibility now, rather than for speed.

**Decision: DirectShow backend, exposure set explicitly, 640x480.**

## F27 - Item 8 and item 3: filter landmarks, and quantified plane-validity

**Item 8 - filter placement.** `experiments/e9_filter_and_thresholds.py`

| Measurement (squat, p95 jump) | raw | filter angle | **filter landmarks** |
|---|---|---|---|
| elbow R | 11.33 | 3.32 | **2.40** |
| knee L | 31.29 | 9.15 | **5.56** |
| hip L | 27.03 | 9.22 | **5.64** |
| ankle L | 32.71 | 10.53 | **5.84** |

Filtering landmarks is 40-45% better during motion. It keeps the skeleton geometrically
consistent; smoothing the angle afterwards cannot repair inconsistent geometry. While
standing still the two are equivalent, which is why F15 - which only ever filtered
angles - could not detect the difference.

**Decision: One Euro applied to landmark positions, before angle computation.**

**Item 3 - plane-validity thresholds.** Synthetic, exact truth, max error across range:

| measurement | 0 | 15 | 30 | 45 | 60 | 75 | 90 |
|---|---|---|---|---|---|---|---|
| elbow (2D) | 54.7 | 33.9 | 19.4 | 10.2 | 4.7 | 1.5 | 0.0 |
| knee (2D) | 70.3 | 36.8 | 22.4 | 12.6 | 6.3 | 2.3 | 0.0 |

Minimum body rotation to hold 2D error within budget:

| measurement | <5 deg | <10 deg | <15 deg |
|---|---|---|---|
| elbow flexion | 60 deg | 60 deg | 45 deg |
| knee flexion | 75 deg | 60 deg | 45 deg |

**Rule:** below ~45 deg of body rotation a 2D sagittal measurement is unusable, and
near 0 deg it is geometrically undefined (F20 test D). Body yaw is estimable at runtime
from apparent shoulder width, so this is an implementable "cannot measure" condition.

---

## F28 - Item 4: absolute accuracy. 3D compresses movement by 23% at best.

`experiments/f1_accuracy.py`, `experiments/f1b_analyse.py`. Elbow held at six known
angles against an on-screen protractor. Reference error ~ +/- 5 deg.

**Run 1 was invalid and is kept for what it shows.** Body yaw was 6-13 deg - the subject
faced the camera, which forces the forearm across the body in the frontal plane: an
anatomically unusual pose, and geometrically the worst case for an image-plane guide.
The orientation gate in the script now prevents this.

| | Run 1 (facing camera) | Run 2 (side-on, 73-83 deg) |
|---|---|---|
| 3D MAE | 27.9 deg | **12.8 deg** |
| 3D bias | -8.1 | -7.4 |
| 3D 95% LoA | -70.6 to +54.4 | **-31.5 to +16.7** |
| 3D gain (slope) | 0.381 | **0.773** |
| 3D at true 0 deg | 33.8 | **5.9** |

Run 2, side-on, per target:

| True | 3D | error | 2D | error |
|---|---|---|---|---|
| 0 | 5.9 | +5.9 | 3.1 | +3.1 |
| 30 | 40.2 | +10.2 | 31.9 | +1.9 |
| 60 | 50.9 | -9.1 | 60.6 | +0.6 |
| 90 | 81.1 | -8.9 | 90.4 | +0.4 |
| 120 | 102.9 | -17.1 | 121.5 | +1.5 |
| 150 | 124.7 | -25.3 | 149.5 | -0.5 |

Linear fit, run 2:

    3D: reported = 0.773 x true + 9.6   (r2 = 0.990)
    2D: reported = 0.982 x true + 2.5   (r2 = 1.000)

**Findings:**

1. **Orientation dominates 3D accuracy.** MAE more than halves between facing the
   camera and side-on. F21's "phantom flexion" was mostly pose, not an inherent defect.
   An earlier claim in this log that no projection argument could excuse the 33.8 deg
   reading was wrong: orientation excused roughly 80% of it.
2. **Even at its best, 3D compresses movement 23%.** Bend 150 deg, it reports 125.
   Publishable error bar for 3D would be **-31.5 to +16.7 deg**.
3. **The defect is a gain error, not an offset.** r2 = 0.990 means it is highly linear
   and therefore calibratable in principle - but the gain itself moved 0.381 -> 0.773
   with orientation, so a fixed per-subject calibration breaks as soon as the subject
   turns.
4. **Mechanism:** upper arm reconstructs at 25.4 cm against a real humerus of ~30 cm
   (cv 5.4%). A short proximal segment with a correct distal one compresses this angle.

**Method limitation, stated plainly:** the 2D column is partly self-referential. The
on-screen guide is drawn from the model's own landmarks, so aligning to it makes 2D
agree with itself; its MAE of 1.3 deg cannot be quoted as accuracy. The exceptions are
the 0 deg rows, where "arm fully extended" is anatomically anchored and needs no
reference at all - there 2D reads **3.1 deg** and 3D reads **5.9 deg**.

**Outstanding:** one run against the printed sheet with the guide off (press `h`) would
give independent 2D numbers. Until then 2D's accuracy rests on: exact geometry when
in-plane (F19, F20), landmark quality (F21), and the 0 deg anchor above.

## F29 - Item 2 RESOLVED: 2D with a side-on requirement

Combining F19, F20, F21, F22 and F28:

| | 2D image | 3D world |
|---|---|---|
| Geometry | exact when the plane faces the camera (F19) | exact at any orientation (F19) |
| Landmarks | accurate (2.0 deg at a known 0 deg pose, F21) | 23% gain compression (F28) |
| Measured MAE | 1.3 (circular) / 3.1 at the anchored 0 deg | **12.8** side-on, 27.9 facing |
| 95% LoA | -1.1 to +3.4 (circular) | **-31.5 to +16.7** |
| Failure mode | wrong below ~45 deg rotation (F27) | always compressed; worsens with rotation |
| Fixable? | yes - require and *detect* the correct orientation | no - gain varies with orientation |

**Decision: 2D image coordinates, with a runtime orientation requirement.**

The deciding argument is that 2D's failure mode is *detectable and avoidable* while 3D's
is neither. Body yaw is measurable at runtime from apparent shoulder width against a
face-on calibration (F28 does exactly this), so the application can require the correct
plane, verify it frame by frame, and refuse to display an angle when the subject is not
presenting it. 3D's 23% compression cannot be detected from the output, cannot be
calibrated away, and is silently wrong.

This supersedes settled-decision 6, F11, F20 and F23.

**Consequence for the application:** it must guide the subject into the correct plane
per measurement, which is what the brief anticipates in saying the subject "may change
their orientation... to expose the sagittal plane". That is now a hard requirement
backed by measurement, not a convenience.

---

## F30 - Item 12: every large failure is detectable, by four complementary signals

`experiments/f2_robustness.py`, `experiments/f2b_analyse.py`. Twelve phases, analysed
in 2D (the representation selected in F29).

| Phase | det% | vis arm | vis leg | elbow | knee | bone asym | bone cv |
|---|---|---|---|---|---|---|---|
| baseline | 100% | 0.93 | 0.95 | **3.2** | **6.3** | 3.1% | 1.8 |
| occl_arm | 100% | **0.50** | 0.95 | 94.5 | 9.3 | 1.9% | 5.1 |
| occl_leg | 100% | 0.88 | **0.52** | 4.3 | **82.4** | **9.7%** | 5.2 |
| partial | 100% | 0.95 | **0.38** | 14.7 | 14.5 | 6.2% | 2.1 |
| fast | 100% | 0.93 | **0.23** | 136.4 | 13.1 | 0.1% | 5.4 |
| rotated | 100% | 0.97 | 0.99 | 21.2 | 13.2 | **12.8%** | 4.4 |
| seated | 99% | 0.96 | 0.88 | 47.9 | **153.2** | **11.8%** | **14.2** |
| far | **43%** | 0.93 | 0.67 | 6.0 | 14.9 | 5.2% | 4.5 |
| near | 100% | 0.94 | 0.94 | 1.7 | 8.0 | 3.0% | 2.8 |
| cam_low | 99% | 0.94 | 0.89 | 1.7 | 6.1 | 9.5% | 2.2 |
| cam_high | 100% | 0.96 | 0.96 | 3.6 | 8.1 | 4.9% | 2.1 |

**Unplanned but important: the baseline row independently confirms F29.** Standing
relaxed is anatomically 0-10 deg. 2D reads elbow **3.2** and knee **6.3**; the same pose
measured in 3D (F28 run 1, F21) gives 24-32. Nothing here was aligned to a guide, so
this is not circular.

**Detection results - no failure went unnoticed:**

| Phase | knee error vs baseline | caught by |
|---|---|---|
| occl_leg | **+76.2** | visibility (0.52), bone asymmetry (9.7%) |
| seated | **+147.0** | bone asymmetry (11.8%), bone variance (14.2%) |
| occl_arm | elbow -> 94.5 | arm visibility (0.50) |
| far | +8.6 | detection rate (43%) |
| rotated | +6.9 | bone asymmetry (12.8%) |

Four complementary signals, each catching failures the others miss:

1. **Detection rate** - subject too far or not found at all
2. **Per-joint visibility** - occlusion and partial framing. Must be computed per
   measurement: leg visibility stayed at 0.95 while the arm was hidden behind the back.
3. **Bone-length asymmetry** - catches rotation and seated posture, which visibility
   does **not** flag (vis 0.88-0.99 in both). This is the signal that cannot be obtained
   from the model's own confidence, and it validates F22's proposal.
4. **Bone-length variance** - seated (cv 14.2 vs 1.8 baseline)

**Decision:** all four run at runtime. A measurement is displayed only when its own
joint chain passes visibility, the skeleton passes bone-length checks, and body yaw is
within the band that measurement requires (F27, F29).

## F31 - Item 13: camera height is irrelevant; only distance matters

| Setup | shoulder px | elbow vs baseline | knee vs baseline |
|---|---|---|---|
| baseline | 79 | 0.0 | 0.0 |
| near | 74 | -1.5 | +1.7 |
| **camera on the floor** | 69 | **-1.5** | **-0.2** |
| **camera raised high** | 82 | **+0.4** | **+1.8** |
| far | 63 | +2.9 | +8.6 |

**Camera height changes readings by under 2 degrees** across a floor-to-overhead range.
No height calibration is needed, and the setup instructions can be permissive about it.

**Distance is the only setup variable that matters**, and mostly through outright
detection failure rather than drift: at 63 px shoulder width the model found the subject
in only 43% of frames.

**Decision:** no camera calibration step. Instead a runtime minimum-size check -
apparent shoulder width below ~70 px prompts "move closer". Camera height and angle are
left to the user.

---

## F32 - Item 4 continued: shoulder flexion validated; knee blocked by visibility

`experiments/f4_anchored.py`, `experiments/f4b_analyse.py`. Anatomically anchored
poses, no protractor: each pose defines its own angle (locked knee = extension stop,
ordinary chair = 90 deg, hanging arm = gravity, horizontal arm checked against a shelf).
Reference error ~ +/- 8 deg, looser than a printed scale but reachable without equipment.

**Shoulder flexion - the assignment's "challenging out-of-plane measurement":**

| Pose | True | 2D | error | 3D | error | visibility |
|---|---|---|---|---|---|---|
| arm at side | 0 | 3.2 | **+3.2** | 7.3 | +7.3 | 0.77 |
| arm horizontal forward | 90 | 87.8 | **-2.2** | 87.2 | -2.8 | 0.98 |

**2D: MAE 2.7, bias +0.5. 3D: MAE 5.1, bias +2.3.** Both usable; 2D better, consistent
with F29. Body yaw was 87-88 deg, so the sagittal plane genuinely faced the camera.

**Knee - not validated, and the reason matters:**

| Pose | True | 2D | error | visibility |
|---|---|---|---|---|
| standing, legs locked | 0 | 45.3 | +45.3 | **0.49** |
| seated on a chair | 90 | 67.2 | -22.8 | **0.37** |

Visibility 0.37-0.49 sits **below the 0.50 trust threshold** established in F30. Standing
or sitting side-on, the near leg occludes the far one and the model reports low
confidence for both. These are therefore values the application would **refuse to
display**, not values it would show incorrectly. The validity layer behaved correctly;
this script reports rather than gates, so the numbers were printed anyway.

**Consequence:** knee flexion cannot be validated in the orientation that 2D requires,
on this camera, because the orientation that makes 2D valid is the same orientation that
occludes the leg. This is a genuine monocular conflict, not a tuning problem:

    2D sagittal validity  requires  side-on  (F27: >= 45 deg rotation)
    knee landmark quality requires  legs not overlapping
    side-on               causes    legs overlapping

**Poses excluded:**
- `sh_180` - the subject could not reach 180 deg. Panel C measured the upper arm at
  164 deg from the trunk axis where a true 180 deg would read 0. Scoring it against its
  nominal target would have measured shoulder mobility, not the system.
- Auto-selecting "the more visible side" is unsound for the arm: the *stationary* arm
  scores higher visibility than the raised one, so at `sh_180` it silently switched to
  the arm hanging at the subject's side. Side must be specified, not inferred.

## F33 - Validation status against the assignment's required trio

The brief requires absolute validation of elbow, knee, and one challenging out-of-plane
measurement.

| Measurement | Synthetic | Absolute | Result |
|---|---|---|---|
| elbow | exact | **yes** (F28) | 3D MAE 12.8 side-on, gain 0.773; 2D anchored at 0 deg reads 3.1 |
| **shoulder flex** | exact | **yes** (F32) | **2D MAE 2.7**, 2 points |
| knee | exact | **attempted, refused** (F32) | landmark visibility below trust threshold when side-on |
| hip flex | exact | no | - |
| shoulder abd | exact | no | - |
| ankle | exact | no | - |

Elbow and the challenging out-of-plane case are validated. Knee is attempted, fails for
a documented and *detected* reason, and the failure mode is itself a finding about
single-camera measurement. Hip, abduction and ankle rest on synthetic exactness only.

---

## F34 - Full audit of all six mandatory measurements

`experiments/e10_audit.py`. Prompted by an external review asking whether the angle
mathematics had been proven for *every* mandatory measurement, not only the elbow.
Five panels: 3D exactness in both directions, 2D exactness at the required orientation,
orientation requirement measured rather than assumed, sign convention, and display.

**1. 3D formulas, both directions, both sides: worst error 0.000000 deg**

| Measurement | range tested | max error |
|---|---|---|
| elbow flexion | 0 to 150 | 0.000000 |
| knee flexion | 0 to 135 | 0.000000 |
| shoulder flexion | **-60 to 180** | 0.000000 |
| shoulder abduction | **-45 to 180** | 0.000000 |
| hip flexion | **-30 to 120** | 0.000000 |
| ankle | **-50 to 20** | 0.000000 |

F19 swept these too, but this run covers the **negative** direction explicitly -
extension, adduction, plantarflexion - which the chart lists as separate named ranges.

**2. 2D formulas at their required orientation: worst error 0.0000 deg**

Includes `shoulder_abduction_2d`, which did not exist before this audit. F29 chose 2D
for all measurements while two of them - abduction and, earlier, flexion - had no 2D
implementation at all.

**3. THE ARCHITECTURAL FINDING: abduction requires the opposite orientation**

Max error in degrees versus body yaw (0 = facing camera, 90 = side-on):

| Measurement | 0 | 30 | 45 | 60 | 90 |
|---|---|---|---|---|---|
| elbow flexion | 54.7 | 24.5 | 12.8 | 5.8 | **0.0** |
| knee flexion | 70.3 | 26.7 | 14.9 | 7.3 | **0.0** |
| shoulder flexion | 70.2 | 27.8 | 15.8 | 7.9 | **0.0** |
| hip flexion | 55.1 | 21.5 | 10.5 | 4.3 | **0.0** |
| ankle | 80.5 | 33.8 | 20.2 | 11.3 | **0.0** |
| **shoulder abduction** | **0.0** | 5.7 | 11.9 | 20.6 | **90.0** |

Abduction is a **frontal**-plane measurement, so it needs the subject facing the camera -
precisely the orientation in which all five sagittal measurements are catastrophically
wrong, and the reverse holds at side-on.

**One fixed camera cannot present all six mandatory measurements at once.** This is
geometry, not a tuning parameter. The application must be built around two distinct
subject positions and must know which measurements each one supports. This was invisible
until abduction was audited alongside the others, and it changes the UI design.

**4. Sign conventions correct** for every measurement: positive reads as flexion,
abduction and dorsiflexion; negative as extension, adduction and plantarflexion.

**5. Two bugs found and fixed**

- `DISPLAY` had the elbow as `("flexion", None, 0, 150)` - the chart's range "0 to 150"
  typed into the (positive max, negative max) fields, setting the positive maximum to
  **zero**. Every normal elbow bend would have been flagged abnormal. Same for the knee.
- The audit's own error metric compared angles by subtraction, reporting -179.8 against
  +180 as a 360 degree error. Fixed with `wrap180`.

**Display convention added:** a signed value is shown as its NAMED opposite, never as a
minus sign, because the chart lists the two directions as separate ranges. `-30` on the
hip displays as "30 deg of extension", not "-30 deg of flexion".

---

## F35 - Architecture review: two design bugs found and fixed before implementation

`experiments/e11_review.py`. An external review raised seven concerns about
ARCHITECTURE.md. Two were testable claims; both were confirmed by experiment.

### A. The yaw estimator conflated rotation with distance

Draft design: `yaw = arccos(shoulder_px / max_shoulder_px)`. Pixel width depends on
rotation **and** distance, so it cannot separate them. Measured across five phases
recorded at the **same** orientation but different distance and camera height:

| Phase | shoulder px | trunk px | ratio | yaw (raw px) | yaw (normalised) |
|---|---|---|---|---|---|
| baseline | 78.7 | 112.1 | 0.702 | 16.0 | 12.3 |
| near | 74.4 | 103.6 | 0.719 | 24.6 | 0.0 |
| far | 63.0 | 89.6 | 0.704 | **39.7** | 11.7 |
| cam_low | 68.5 | 96.9 | 0.707 | 33.2 | 10.5 |
| cam_high | 81.9 | 117.9 | 0.694 | 0.0 | 15.0 |

Truth is identical for every row, so all spread is error. Raw pixel width spreads **26%**
of its mean and implies **0 to 39.7 deg** of phantom rotation - enough to cross the 45 deg
gate and silently change which measurements the application trusts, purely because the
subject walked backwards.

Normalising by trunk length (unaffected by rotation about the vertical axis) reduces the
spread to **3%**. Sensitivity to real rotation survives: ratio 0.702 facing, 0.246
side-on.

**Change: `yaw = arccos(ratio / ratio_max_seen)` where `ratio = shoulder_width /
trunk_length`.**

### G. Nose-based anterior detection is unreliable, and voting makes it worse

The anterior direction sets the sign of every signed sagittal measurement, so a flip
turns flexion into extension. Consistency within each recorded phase:

| Phase | nose | ear->nose | heel->toe | majority vote | **unanimous-hold** |
|---|---|---|---|---|---|
| cam_high | **0.67** | **0.54** | 1.00 | **0.51** | **1.00** |
| cam_low | **0.81** | **0.73** | 1.00 | 0.90 | **1.00** |
| seated | 1.00 | 1.00 | **0.73** | 1.00 | **1.00** |
| fast | **0.75** | 0.96 | **0.68** | 0.78 | 0.87 |
| far | **0.85** | 0.89 | 1.00 | 0.89 | **1.00** |
| sh_180 | **0.55** | 1.00 | **0.79** | 0.80 | **1.00** |

Nose visibility was 1.00 throughout, so this is **geometric, not occlusion**: camera
elevation changes how the head projects relative to the shoulder line.

**Majority voting is worse than the best single cue** (cam_high 0.51 vs 1.00). The nose
and ear cues are both head-based and fail together with camera elevation, outvoting the
foot cue that remains correct. Correlated failures are not repaired by averaging.

**Change: update the stored direction only when all three cues agree; otherwise hold the
previous value; if never established, withhold signed sagittal measurements.** Ambiguous
frames are discarded, not averaged. Perfect consistency everywhere except fast motion
(0.87), where visibility has already collapsed to 0.23 (F30) and the measurement is
rejected on other grounds. The policy also reports when direction was never established -
100% of frames with the legs out of view, 46% at long range.

### Five further corrections, from analysis rather than experiment

- **Measurement count was wrong.** The design said "nine"; it is **twelve** - six types
  across two sides, because the brief assigns the shoulder two measurements in different
  planes. At most 10 can be valid at once (F34).
- **Source interface was incoherent.** A recorded-landmark source has no image and cannot
  satisfy `Frame(image, ...)`. Split into `VideoSource` (yields frames, exercises the
  model, keeps F16/F24 model comparison reproducible) and `LandmarkSource` (yields
  landmark sets, enters at the filter stage, cannot exercise the model).
- **49.9 FPS was throughput, not frame rate.** Separated into source FPS (30, camera
  ceiling), pipeline throughput (~50, measured by replay) and displayed FPS (live = 30).
  This is also what answers the 60 FPS question honestly: the pipeline has headroom, the
  capture device does not.
- **Model ownership was unstated.** Each worker constructs and owns its own
  `PoseLandmarker`; none is shared. The 3.03x measurement in F3 was taken under this rule.
  The result queue becomes a **single-slot latest-wins holder**: only the newest result is
  ever displayed, so a queue would grow without bound if the consumer stalled.
- **Validity must run on RAW landmarks.** Filtering removes precisely the variation the
  geometric checks look for - One Euro cuts frame-to-frame jump 40-45% (F27), which would
  suppress the bone-length variance that flags an unstable reconstruction. Raw feeds
  `validity`, filtered feeds `angles`.
- **One Euro timestamps** come from `capture_ts`, never wall-clock; monotonicity follows
  from discarding out-of-order results by `seq`; `dt` is clamped to [1 ms, 250 ms] and the
  filter resets beyond that rather than interpolating across a gap.

**Assessment:** two of the seven were outright errors that would have shipped - a yaw
estimator that breaks when the user moves, and a wrong measurement count. Both were
cheaper to find in a document than in a running application.

---

## F36 - Pre-freeze review: VIDEO mode replaces the two-worker design

`experiments/e12_freeze.py`. Three checks demanded before freezing the architecture. All
three found real problems; one overturned a core decision.

### 1. The orientation gate chatters at its thresholds

Frame-to-frame yaw noise is ~1 deg sd with p95 jumps of 2.3-3.9 deg, and several recorded
phases sit at 44-46 deg - directly on the 45 deg boundary. A bare threshold produced up
to **3 state flips inside a single phase**, which on screen means a measurement blinking
between a value and "turn side-on".

**A 5 deg margin plus a 5-frame hold (~0.25 s) gives zero flips in every recorded phase.**

### 2. VIDEO mode beats two workers in IMAGE mode, on every axis

All experiments to this point used `RunningMode.IMAGE`, which re-runs the person detector
on every frame. `RunningMode.VIDEO` carries tracking state between frames. Replayed over
identical recorded frames:

| | IMAGE p50 | VIDEO p50 | IMAGE knee | VIDEO knee |
|---|---|---|---|---|
| still | 40.9 ms | **23.2 ms** | 2.00 deg sd | **1.54 deg sd** |
| squat | 39.3 ms | **16.6 ms** | 29.60 deg p95 jump | **5.31 deg** |

**2.4x faster and 82% less frame-to-frame knee jump.** At 16.6-23.2 ms that is 43-60 FPS
single-threaded - past the requirement with no worker pool.

It does not achieve this by discarding movement:

| Phase | Joint | range retained | lag |
|---|---|---|---|
| squat | knee (moving) | 93% | 0 ms |
| elbow | elbow (moving) | **100%** | 0 ms |
| elbow | knee (**stationary**) | **15%** | 0 ms |

Full range on joints that move; 85% suppressed on a joint that did not. The 84 deg of
apparent range IMAGE mode reported on a stationary knee was noise. Zero added lag.

**And it requires a single worker.** VIDEO mode assumes a contiguous stream. Giving each
of two detectors every second frame raises knee p95 jump from **5.31 to 9.32 deg**.

**Decision: one sequential worker in VIDEO mode.** This supersedes F25's two-worker
choice. F25 was not wrong - two workers really are better than one *in IMAGE mode* - but
it optimised around a cost that VIDEO mode removes. The revised design deletes the worker
pool, the result queue, out-of-order rejection and per-worker model ownership, and is
faster and more accurate than what it replaces.

### 3. The scale-normalised yaw estimator fails under posture change

F35 fixed the distance problem. It does not survive posture. Phases recorded at one
orientation (side-on):

| Posture | shoulder/trunk ratio | implied yaw |
|---|---|---|
| standing, legs locked | 0.291 | 73.0 |
| seated on a chair | 0.308 | 72.0 |
| arm raised to horizontal | 0.123 | 82.9 |
| **arm raised high** | **0.806** | **36.1** |
| seated side-on | 0.068 | 86.1 |

**50 deg of spread at one orientation.** Raising an arm elevates and protracts the
shoulder, moving both terms of the ratio; that single outlier also poisoned the running
maximum the estimator depends on, which is why phases recorded facing the camera were
reading 45 deg.

**Replacement: the world-space shoulder axis.**

    v   = world[LEFT_SHOULDER] - world[RIGHT_SHOULDER]
    yaw = degrees(atan2(|v.z|, |v.x|))

| Estimator | fails on |
|---|---|
| raw shoulder pixels | distance: 0 to 39.7 deg error at one orientation (F35) |
| shoulder / trunk ratio | posture: 50 deg spread at one orientation |
| **world-3D shoulder axis** | facing group **1.1-6.2 deg**, side-on group **62.3-87.7 deg** |

It measures a **direction**, not a length, so neither distance nor posture disturbs it.
It needs no reference, so it cannot be mis-calibrated - removing the failure class that
locked the system out during F32 when a startup calibration window caught the subject
turned.

**This does not contradict choosing 2D for angles (F29).** The 23% gain compression that
disqualified 3D distorts limb angle *magnitudes*; this uses only the direction of the
trunk's shoulder axis, which is measured accurately across six known face-on phases.

`sh_180` reads 11.9 deg on all three estimators, consistent with the subject having
turned toward the camera to raise the arm - the pose they reported being unable to reach.
That is the estimator being correct, not failing.

---

## F37 - Bone-length asymmetry is advisory, not a veto. Visibility threshold 0.55.

Prompted by a build-plan review asking what happens when a *bilateral* validity signal
cannot be computed - if the right leg is hidden, should a perfectly visible left knee be
rejected? The brief says to compute each side whenever **that side's** landmarks are
reliable, so a bilateral veto would be stricter than required.

Rather than write a rule for the uncomputable case, the question was whether asymmetry
still earns a veto at all. It was adopted in F30, before the world-3D orientation
estimator (F36) existed to catch rotation directly.

Signal coverage across eleven recorded failure phases, scored against actual knee error:

| Phase | knee error | visibility | variance (per side) | asymmetry (bilateral) |
|---|---|---|---|---|
| occl_leg | **76.2 deg** | 0.52 | 5.2% | 9.3% |
| seated | **147.0 deg** | 0.88 | **14.2%** | 12.2% |
| cam_low | **0.2 deg** (correct) | 0.89 | 2.2% | **14.1%** |
| all others | < 9 deg | > 0.55 or n/a | < 6% | < 8% |

- **`seated`** is caught only by bone-length **variance** (cv 14.2% against a 6% threshold).
  Visibility is 0.88 and no threshold catches it.
- **`occl_leg`** is caught by visibility, but only at **0.55**; at the F30 threshold of
  0.50 a 76 degree error slips through at 0.52.
- **`cam_low`** is a **false positive** for asymmetry: 14.1% asymmetry on a measurement
  that was accurate to 0.2 deg. A low camera foreshortens the near leg differently from
  the far one, manufacturing apparent asymmetry with no error.

Threshold sweep for visibility: 0.50 catches 0 of 2 genuine failures; 0.55, 0.60 and 0.65
each catch 1 (`occl_leg`), with `seated` never caught by visibility at any threshold.

**Decisions:**

1. **Visibility threshold 0.50 -> 0.55.**
2. **Bone-length variance** (per side, one segment over time) becomes the second
   veto-capable geometric signal. It is per-side, so the bilateral problem does not arise.
3. **Bone-length asymmetry becomes advisory only** - displayed as a warning, never
   vetoing. It adds no unique coverage and has a false-positive mode tied to camera
   elevation.
4. **General rule:** a veto-capable signal that cannot be computed **blocks** the
   measurement; an advisory signal that cannot be computed is **omitted**.

The review's concern was real, but the fix is not a special case for uncomputable
bilateral signals - it is that the bilateral signal should not have had a veto.

---

---

## Settled by measurement

1. CPU only - GPU unavailable on this platform (F1)
2. ONE sequential worker in VIDEO mode; threads only for capture (F36, supersedes F25)
3. Concurrency is mandatory to reach 30 FPS, not an optimisation (F4)
4. DirectShow backend, 640x480, exposure fixed at 2^-5 s (F5, F7, F26)
5. Dedicated capture thread holding only the newest frame (F5)
6. **2D image coordinates**, with a runtime side-on orientation requirement and yaw gate (F28, F29)
7. Visibility drives the "cannot measure" state (F9)
8. Validation needs an external reference, not jitter alone (F10)
9. One Euro on LANDMARKS (not angles), beta ~= 0.01; peaks read unfiltered (F15, F27)
10. Runtime rotation detection to flag unreliable 2D measurements (F12)
11. Model: `pose_landmarker_full`; heavy rejected, re-verified (F16, F24)
16. Two inference workers: 49.9 FPS at 52.9 ms e2e (F25)
17. Plane-validity: 2D unusable below ~45 deg body rotation (F27)
18. Veto signals: detection rate, per-joint visibility (< 0.55), per-side bone-length
    variance (cv > 6%), plane presented. Bone-length asymmetry is ADVISORY only (F30, F37)
19. No camera calibration; runtime minimum-size check instead (F31)
20. TWO subject positions required: side-on for the five sagittal measurements,
    face-on for shoulder abduction. No single camera view serves both (F34)
21. Signed values display as their named opposite, not a minus sign (F34)
22. Yaw from the WORLD-3D shoulder axis; no calibration (F36, supersedes F35)
23. Anterior direction by unanimous three-cue agreement with hold (F35)
24. Validity on raw landmarks, angles on filtered (F35)
25. Twelve measurements, at most 10 simultaneously valid (F34, F35)
26. Orientation gate needs 5 deg hysteresis and a 5-frame hold (F36)
12. Anatomical frame rebuilt per frame from landmarks, never a fixed world plane (F18)
13. Conventions per F17; verified exact to 0.000000 deg against synthetic truth (F19)
14. Ankle uses heel->foot_index, sagittal-projected: zero bias and 18% less noise (E3)
15. Bone-length variance as a reference-free validity signal (F22)

## Still open

- **Absolute** accuracy at a known angle. F12 measured *change*, F14 only the range
  endpoints. Needs a physical reference - printed protractor or goniometer.
- Does MSMF still duplicate frames now that exposure is fixed? (F5 measured it before
  the exposure fix, so the number may no longer hold)
- Why `heavy` drifts 23 deg out of plane when smaller models do not (F16) - surprising,
  re-check before publishing
- Rotation threshold for flagging unreliable 2D: F12 suggests ~40 deg, needs
  confirming per joint rather than assumed uniform
- Can self-occlusion in side-on squats (F13, knee R confidence 0.50) be detected and
  flagged reliably, or only observed after the fact?
- Filtering landmarks before computing angles, vs filtering the angle itself (F15)
- Anatomical conventions: nothing here yet converts an interior angle into the
  goniometry chart's signed, neutral-referenced values. F14 only did the elbow, by hand.

---

# Gap analysis

| # | Question | Hypothesis going in | Experiment | Result | Confidence | Decision |
|---|---|---|---|---|---|---|
| 1 | What are the exact conventions for all 9 measurements? | Three-point interior angles would mostly do | E2 derivation + synthetic check (F17, F19) | Interior angles are wrong for 4 of 6. Shoulder needs two different plane projections from the same landmarks; ankle neutral is 90 not 180; knee flexes backwards | **High** - verified exact | Conventions per F17, implemented in `biomech_ref.py` |
| 2 | 2D or 3D, per joint? | 3D everywhere (F20) | Synthetic (F20), real (F21, F22), absolute (F28) | **2D.** Its failure mode is detectable and avoidable; 3D's 23% gain compression is neither | **High** | 2D + runtime orientation gate (F29) |
| 3 | When does rotation invalidate a measurement? | Somewhere near 40 deg | Synthetic sweep (F27) | 2D needs >=45 deg rotation for <15 deg error, >=60 for <5 deg; undefined near 0 | **High** for 2D hinges | Flag "cannot measure" below 45 deg body yaw |
| 4 | What does the system report at a *known* angle? | Unknown | `f1_accuracy.py`, six held angles (F28) | 3D MAE **12.8** side-on, gain 0.773, LoA -31.5 to +16.7. 2D column circular; 0 deg anchor gives 2D 3.1 / 3D 5.9 | **High** for 3D; 2D partly self-referential | 2D selected (F29) |
| 5 | Is the angle maths itself correct? | Probably | Synthetic forward kinematics (F19) | Worst error **0.000000 deg**, all measurements, both sides, all orientations | **Very high** | Maths eliminated as an error source |
| 6 | How much error does BlazePose add? | Some | Known-pose + bone-length (F21, F22) | 15-27 deg phantom flexion at a known-straight pose; 11-12% L/R bone asymmetry; models disagree by 16 deg | **High** that it is large; **low** on magnitude without item 4 | Model error dominates; needs absolute reference to quantify |
| 7 | How bad is the ankle, and what helps? | Short foot segment amplifies noise | Noise injection + foot definitions (E3, F22) | Elbow is more noise-sensitive (0.45 vs 0.37 deg/mm). Real ankle problem is the model placing heel/toe **12-14 cm apart vs ~25 cm true**. `heel->toe` beats `ankle->toe`: zero bias vs -16.7, and 18% less noise | **High** | `heel->toe`, sagittal-projected |
| 8 | Filter landmarks or the angle? | Probably equivalent | Replay both (F27) | Landmarks better by **40-45%** during motion; equivalent at rest | **High** | One Euro on landmarks, before angle computation |
| 9 | Is heavy's 23 deg drift real? | Possibly an artefact | Shared-yaw re-run (F24) | Real: +21.6 vs +4.2 for full. Also models disagree with each other by 10-18 deg | **High** | heavy rejected; `full` retained |
| 10 | MSMF duplicates and the real FPS ceiling? | Fixed by the exposure change | Re-measure both backends (F26) | MSMF still 11% duplicates and 2x the p95 jitter. **DirectShow: 0% duplicates, 30 FPS, p95 48 ms** | **High** | DirectShow, exposure explicit, 640x480 |
| 11 | Can the full pipeline hold 30 FPS, and where is the bottleneck? | Inference-bound, more threads better | Threaded harness (F25) | Inference is 96-99% of budget. 1 worker already gives 34.4 FPS. **More workers trade latency for throughput we do not need** | **High** | 2 workers: 49.9 FPS at 52.9 ms |
| 12 | What breaks it? | Occlusion and frame edges | 12-phase session (F30) | Every large failure detected. Bone asymmetry catches rotation and seated, which visibility misses | **High** | Four runtime validity signals |
| 13 | How sensitive to camera setup? | Distance matters most | near/far/low/high (F31) | Camera height under 2 deg effect. Distance matters via detection failure (43% at 63 px) | **High** | No calibration; runtime size check |

## These are the remaining unknowns that could still change the final architecture

The four listed when this section was first written are now all answered: 2D vs 3D
(F29), whether model error is correctable (F28 - no, the gain moves with orientation),
failure detectability (F30 - yes, four signals), and setup sensitivity (F31 - height
irrelevant, distance matters). What follows is what actually remains.

### 1. Only the ELBOW has been validated against known angles

F28 measured absolute accuracy for elbow flexion. **Knee, shoulder flexion/extension,
shoulder abduction/adduction, hip flexion/extension and ankle dorsi/plantarflexion have
no absolute validation at all.** What exists for them:

| Measurement | Synthetic exactness | Jitter | Absolute accuracy |
|---|---|---|---|
| elbow | yes (F19) | yes (F11) | **yes (F28)** |
| knee | yes (F19) | yes (F11) | **no** |
| shoulder flex/ext | yes (F19) | yes (F11) | **no** |
| shoulder abd/add | yes (F19) | no | **no** |
| hip flex/ext | yes (F19) | yes (F11) | **no** |
| ankle | yes (F19) | yes (F11) | **no** |

The elbow result may not generalise: the ankle has a landmark placement defect (F22,
heel-to-toe at half its true length) that the elbow does not, and the shoulder
measurements depend on the trunk frame rather than a simple three-point chain.

**This is the largest remaining gap.** It does not change the architecture, but it
does change what the validation section can honestly claim. The printed protractor
works for the knee; shoulder and hip need a different reference; the ankle may not be
validatable with the equipment available, and saying so is a legitimate result.

### 2. 2D absolute accuracy is still partly self-referential

F28's 2D column came from aligning to a model-drawn guide. The supporting evidence is
strong - exact in-plane geometry (F19, F20), correct readings at anatomically anchored
poses (F30 baseline: elbow 3.2, knee 6.3 where truth is 0-10), and landmark quality
(F21) - but one run against the printed sheet with the guide off would replace
inference with measurement. Cheap, and it upgrades the headline claim.

### 3. Open, but not architecture-changing

- Inter-model disagreement of 10-18 deg (F24) has never been resolved against a true
  reference. `full` was chosen on jitter and rotation stability, not on accuracy.
- Why the legs reconstruct so much worse than the arms (F22: 11-12% vs 3.4% asymmetry).
- Whether the ankle's `heel->foot_index` advantage (E3, synthetic) survives the model's
  actual landmark placement, given the segment comes out at half its true length (F22).
- Rotation thresholds (F27) were derived for 2D hinge joints only. Shoulder and hip
  are sagittal-plane projections and may need different bands.
