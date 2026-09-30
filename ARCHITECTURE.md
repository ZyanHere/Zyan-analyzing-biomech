# Architecture

Design for the real-time biomechanical analysis application.

Every decision here was settled by experiment during the investigation phase. Findings
are cited as `F<n>` and live in [FINDINGS.md](FINDINGS.md); each has a rerunnable script
in `experiments/`. Where a decision was reversed by later evidence, the reversal is
noted rather than hidden.

---

## 1. Components

| Component | Owns | Does not own |
|---|---|---|
| `capture` | camera device, exposure, delivering the newest frame | timing analysis, any model |
| `source` | frame provenance: camera, video file, or recorded landmarks | frame content |
| `inference` | model lifetime, VIDEO-mode tracking state, landmark extraction | angle meaning |
| `filters` | temporal smoothing of landmark positions | which landmarks matter |
| `angles` | goniometry conventions, 2D/3D geometry | whether a value is trustworthy |
| `validity` | trust signals, orientation gate, the "cannot measure" verdict | angle computation |
| `metrics` | per-stage latency, FPS, queue depth, drop counts | display |
| `ui` | overlay, value panel, orientation guidance | any computation |
| `record` | writing and replaying sessions | live capture |
| `app` | wiring, lifecycle, shutdown | domain logic |

The boundary that matters most is **`angles` versus `validity`**. `angles` always returns
a number; it never decides whether that number is believable. `validity` never computes
geometry. Keeping these apart is what makes the "indicate that state" requirement a
composable rule rather than conditionals sprinkled through the maths.

---

## 2. Data flow

```
                    +------------------+
   camera / file /  |      source      |
   recorded JSONL   +--------+---------+
                             | Frame(image, capture_ts, seq)
                             v
                    +------------------+
                    |    inference     |   1 worker, VIDEO mode (F36)
                    +--------+---------+
                             | LandmarkSet(image_xy, world_xyz, visibility, ts)
                             v
                    +------------------+
                    |     filters      |   One Euro on POSITIONS (F27)
                    +--------+---------+
                             | LandmarkSet (smoothed)
                             v
              +--------------+--------------+
              |                             |
              v                             v
      +---------------+            +----------------+
      |    angles     |            |    validity    |
      +-------+-------+            +--------+-------+
              | Measurement[]               | Verdict[]
              +--------------+--------------+
                             v
                    +------------------+
                    |        ui        |
                    +------------------+
```

Four data types, all immutable:

- `Frame` - image, capture timestamp, sequence number
- `LandmarkSet` - 33 image-space points, 33 world-space points, per-point visibility
- `Measurement` - name, side, signed value, representation used
- `Verdict` - valid or not, and if not, which signal failed

`capture_ts` travels with the frame all the way to display. That single field is what
makes honest end-to-end latency measurable (section 9).

### Two source levels, not one

A recorded-landmark source has no image, so it cannot satisfy a `Frame(image, ...)`
interface. There are **two distinct replay levels**, and conflating them would make the
model comparison irreproducible:

| Source | Yields | Enters the pipeline at | Exercises |
|---|---|---|---|
| `CameraSource` | `Frame` | inference | everything |
| `VideoSource` (mp4) | `Frame` | inference | everything, including **the model** |
| `LandmarkSource` (JSONL) | `LandmarkSet` | filters | filters, angles, validity, UI - **not the model** |

`VideoSource` is what keeps model comparison reproducible: F16 and F24 compared lite /
full / heavy by replaying identical *frames*, which a landmark stream cannot do because
the landmarks are already the output of one specific model.

`LandmarkSource` is faster, far smaller, and sufficient for everything downstream of
inference - the filter comparison (F15, F27) and every angle analysis ran on it. It is
also what ships in the repository, since `fixtures/*.mp4` contain video of the subject
while `session.jsonl` holds only coordinates.

**Frame buffer ownership.** `cv2.VideoCapture.read()` allocates a fresh array per call in
the Python bindings, but the pipeline does not depend on that: the capture thread copies
the image before queueing it. A queued frame is owned by the queue and is never written
again by the producer, so "immutable" is enforced rather than assumed.

---

## 3. Concurrency

```
capture thread ---> [slot, newest frame only] ---> main thread:
                                                     detect_for_video  (VIDEO mode)
                                                     -> filter -> angles
                                                     -> validity -> draw
```

**One inference worker, running MediaPipe in VIDEO mode.** This replaces an earlier
two-worker design that used IMAGE mode, and it is faster, more accurate and simpler at
once (F36).

MediaPipe offers three running modes. `IMAGE` treats every frame independently, re-running
the person detector each time. `VIDEO` carries tracking state between frames: it locates
the person once and then follows the landmarks, requiring monotonically increasing
timestamps on a single detector instance. Measured on identical recorded frames:

| | IMAGE p50 | VIDEO p50 | IMAGE knee jitter | VIDEO knee jitter |
|---|---|---|---|---|
| still | 40.9 ms | **23.2 ms** | 2.00 deg | **1.54 deg** |
| squat | 39.3 ms | **16.6 ms** | 29.60 deg p95 jump | **5.31 deg** |

VIDEO mode is **2.4x faster** and cuts frame-to-frame knee jump by **82%**, because it
skips the detector stage. At 16.6-23.2 ms that is **43-60 FPS single-threaded**, well past
the 30 FPS requirement with no worker pool at all.

**It does not buy smoothness by discarding movement** (F36):

| Phase | Joint | range retained | lag added |
|---|---|---|---|
| squat | knee (moving) | 93% | 0 ms |
| elbow | elbow (moving) | **100%** | 0 ms |
| elbow | knee (**stationary**) | **15%** | 0 ms |

Full range on joints that move, 85% suppressed on a joint that did not - the 84 deg of
"range" IMAGE mode reported on a stationary knee was noise. This is noise rejection, not
signal loss.

**Why this forces a single worker.** VIDEO mode's tracking assumes it sees a contiguous
stream. Splitting frames across two workers gives each detector every second frame, which
breaks that assumption: knee p95 jump rises from **5.31 to 9.32 deg** on the same footage
(F36). Two workers in IMAGE mode were an answer to IMAGE mode being slow; VIDEO mode
removes the problem rather than parallelising around it.

**What this deletes:** the worker pool, the result queue, out-of-order rejection by
sequence number, and per-worker model ownership. A sequential pipeline has none of those
concerns, and monotonic timestamps - which VIDEO mode requires - are automatic.

**Threads are still used for capture.** The capture thread keeps only the newest frame,
so a slow consumer receives fresh data rather than a backlog. MediaPipe does release the
GIL (3.03x on 4 threads, against a pure-Python control at exactly 1.00x, F3), so the
parallel design was viable - it simply is not needed.

**Everything after inference runs on the main thread**, because it costs **0.7-1.1 ms**
against inference's 27-77 ms (F25). Inference is 96-99% of the budget. Parallelising the
biomechanics would be optimising 1% of the system.

**One `PoseLandmarker`, owned by the main thread, never shared.** With a sequential
pipeline there is no concurrent `detect_for_video()` call to worry about, and VIDEO mode's
tracking state belongs to exactly one stream. The Tasks API makes no thread-safety
guarantee for concurrent calls on one instance; this design never makes one.

---

## 4. What happens when inference is slow

**Keep only the newest frame. Never block the capture thread.**

The capture thread writes into a single slot. If the consumer has not taken the previous
frame, it is overwritten. A queue would keep FPS looking healthy while displayed latency
grew without bound - the user would see smooth video of where their arm was a second ago.
Discarding is a correctness property here, not an optimisation.

Dropped frames are counted and shown, since a high drop rate means the pipeline is not
keeping up even though the displayed output still looks smooth.

**Ordering is not a concern.** Processing is sequential, so results arrive in capture
order by construction. The earlier two-worker design needed sequence numbers to reject
out-of-order results; a single worker removes the failure mode rather than handling it.
This also satisfies VIDEO mode's requirement for monotonically increasing timestamps
(section 3) without any extra mechanism.

---

## 5. What happens when landmarks are unreliable

Four signals, each catching failures the others miss (F30). All are computable at
runtime and none requires a reference:

| Signal | Threshold | Catches |
|---|---|---|
| detection rate | pose found at all | subject too far - 43% detection at 63 px shoulder width |
| per-joint visibility | < 0.50 | occlusion, partial framing |
| bone-length asymmetry | > 8% L/R | rotation, seated posture |
| bone-length variance | cv > 6% | unstable reconstruction |

**Visibility must be evaluated per measurement**, over that measurement's own landmark
chain. In testing, leg visibility stayed at 0.95 while an arm was hidden behind the back
(F30). A single global confidence number would have missed it.

**Bone-length asymmetry earns its place** by catching what visibility does not. Seated
and rotated postures both reported visibility 0.88-0.99 while producing knee errors of
+147 deg and +6.9 deg respectively; asymmetry flagged both at 11.8% and 12.8%. A human is
symmetric to 1-2%, so this needs no calibration and no ground truth.

When a measurement fails any signal, the UI shows **why** - "left leg occluded", "turn
side-on" - not a blank and not a number.

**Validity is computed on RAW landmarks, before filtering.** Filtering exists to remove
exactly the variation these signals look for: One Euro cuts frame-to-frame jump by 40-45%
(F27), which would suppress the bone-length variance that flags an unstable
reconstruction. Judging trustworthiness from a smoothed signal means asking whether the
smoother did its job, not whether the data was sound. Visibility is per-frame from the
model and unaffected either way, but the geometric checks are not.

So each frame carries two landmark sets: **raw** feeds `validity`, **filtered** feeds
`angles`. They are the same measurement judged and displayed by different criteria.

### How many measurements

**Twelve**, not nine. Six measurement types, each computed for both sides:

| Type | Sides | Signed | Required view |
|---|---|---|---|
| elbow flexion | L, R | no | sagittal |
| knee flexion | L, R | no | sagittal |
| shoulder flexion/extension | L, R | yes | sagittal |
| shoulder abduction/adduction | L, R | yes | **frontal** |
| hip flexion/extension | L, R | yes | sagittal |
| ankle dorsi/plantarflexion | L, R | yes | sagittal |

The brief lists five joints but assigns the shoulder **two** measurements in different
planes, so the count is 6 x 2 = 12. At any instant at most **10** can be valid, because
abduction's plane is mutually exclusive with the other five (section 6).

---

## 6. Side-view versus front-view

**This is the constraint that shapes the UI, and it is not negotiable.**

Measured error versus body rotation (F34), 0 = facing the camera, 90 = side-on:

| Measurement | plane | 0 | 45 | 90 |
|---|---|---|---|---|
| elbow flexion | sagittal | 54.7 | 12.8 | **0.0** |
| knee flexion | sagittal | 70.3 | 14.9 | **0.0** |
| shoulder flexion/extension | sagittal | 70.2 | 15.8 | **0.0** |
| hip flexion/extension | sagittal | 55.1 | 10.5 | **0.0** |
| ankle dorsi/plantarflexion | sagittal | 80.5 | 20.2 | **0.0** |
| **shoulder abduction/adduction** | **frontal** | **0.0** | 11.9 | **90.0** |

Abduction lives in the frontal plane and needs the subject facing the camera. The other
five are sagittal and need side-on. **One fixed camera cannot serve both at once.** This
is geometry, not tuning, and no filtering or model change affects it.

### Decision: continuous orientation estimate, all measurements always listed

Three options were considered:

1. **Explicit mode switch** - user picks "sagittal" or "frontal". Simple, but makes the
   user responsible for a constraint they have no reason to understand, and a wrong mode
   produces confident nonsense.
2. **Auto-detect and show only what is valid** - measurements appear and vanish as the
   user turns. No wrong numbers, but disappearing rows read as bugs.
3. **Auto-detect, always list all twelve, mark the invalid ones with the reason.**

**Option 3.** Every measurement is always on screen. Those whose plane is presented show
a value; those whose plane is not show `--` with "turn side-on" or "face the camera".
At most 10 of the 12 can be valid at once, since abduction's plane excludes the other
five - so the list never goes fully green, and that is the honest picture.
The constraint becomes visible and teachable instead of hidden, and it is a direct
implementation of the brief's requirement to indicate that state rather than display a
misleading value.

### Estimating orientation

**From the world-space shoulder axis. No calibration, no reference, no ratio.**

```
v   = world_landmark[LEFT_SHOULDER] - world_landmark[RIGHT_SHOULDER]
yaw = degrees( atan2( |v.z| , |v.x| ) )        # 0 = facing camera, 90 = side-on
```

Two earlier estimators were tried and discarded, both measured against recorded sessions
whose true orientation is known (F35, F36):

| Estimator | facing group | side-on group | fails on |
|---|---|---|---|
| raw shoulder pixels | - | - | **distance**: 0 to 39.7 deg error at one orientation |
| shoulder / trunk ratio | - | - | **posture**: 50 deg spread at one orientation |
| **world-3D shoulder axis** | **1.1 to 6.2 deg** | **62.3 to 87.7 deg** | - |

The ratio estimator solved distance but not posture: raising an arm elevates and
protracts the shoulder, moving both terms at once, and a single such frame also poisoned
the running maximum it depended on - which is why "facing the camera" was reading 45 deg.

The world-space form has no such failure because it measures a **direction**, not a
length. It needs no reference to be established, so it cannot be mis-calibrated, and it
removes an entire class of bug: a fixed startup calibration window was observed to lock
the system out permanently when the subject happened to be turned during it (F32).

**This does not contradict choosing 2D for angles.** The 23% gain compression that
disqualified 3D (F28) distorts limb angle *magnitudes*. Orientation uses only the
*direction* of the trunk's shoulder axis, a different quantity, and it is measured
accurately: 1.1-6.2 deg across six phases known to be face-on.

### Threshold hysteresis

Valid bands, from F27 and F34: sagittal needs yaw >= 45 deg, frontal needs yaw <= 30 deg,
and between them both are unavailable.

A bare threshold chatters. Measured frame-to-frame yaw noise is ~1 deg sd with p95 jumps
of 2.3-3.9 deg, and several recorded phases sit at 44-46 deg - directly on the boundary.
Counted across the recorded sessions, a bare threshold produced up to **3 state flips**
within a single phase, which would make measurements blink between a value and "turn
side-on" (F36).

**A 5 deg margin plus a 5-frame hold (~0.25 s) reduces this to zero flips in every
recorded phase.** A state change must therefore clear the threshold by 5 deg and persist
for 5 frames before the UI acts on it.

### Determining which way the subject faces

The anterior direction sets the **sign** of every signed sagittal measurement, so a flip
turns flexion into extension - a 2x error that looks entirely plausible on screen.

Three candidate cues, measured for consistency within each recorded phase (F35):

| Phase | nose | ear->nose | heel->toe | unanimous-hold |
|---|---|---|---|---|
| cam_high | **0.67** | **0.54** | 1.00 | **1.00** |
| cam_low | **0.81** | **0.73** | 1.00 | **1.00** |
| seated | 1.00 | 1.00 | **0.73** | **1.00** |
| fast | **0.75** | 0.96 | **0.68** | 0.87 |
| sh_180 | **0.55** | 1.00 | **0.79** | **1.00** |

No single cue is reliable, and **majority voting makes it worse** (cam_high falls to
0.51): the nose and ear cues are both head-based, so they fail together with camera
elevation and outvote the one cue that is right. Correlated failures are not fixed by
averaging.

**Policy: update only on unanimous agreement, otherwise hold.**

```
if nose_cue == ear_cue == foot_cue:   anterior = that direction   # confident
else:                                  keep the previous value     # ambiguous, ignore
if never established:                  withhold all signed sagittal measurements
```

Ambiguous frames are discarded rather than averaged. This gives perfect consistency
everywhere except fast motion (0.87), where landmark visibility has already collapsed to
0.23 (F30) and the measurements are rejected on other grounds anyway.

The policy also reports when direction was **never** established - 100% of frames when
the legs are out of view, 46% at long range - and in that state signed sagittal
measurements show "orientation unknown" rather than a guess. Unsigned hinge joints
(elbow, knee) are unaffected, since they need no anterior direction at all.

---

## 7. How measurements are calculated

Full conventions in F17; verified exact to **0.000000 deg** against synthetic forward
kinematics across every measurement, both sides, both directions (F19, F34).

| Measurement | Landmarks | Plane | Neutral | Signed | Conversion |
|---|---|---|---|---|---|
| Elbow flexion | shoulder, elbow, wrist | limb (hinge) | straight arm | no | `180 - interior` |
| Knee flexion | hip, knee, ankle | limb (hinge) | straight leg | no | `180 - interior` |
| Shoulder flex/ext | hip, shoulder, elbow | sagittal | arm hanging | yes | `atan2(h.ant, -h.up)` |
| Shoulder abd/add | hip, shoulder, elbow | frontal | arm hanging | yes | `atan2(h.lat, -h.up)` |
| Hip flex/ext | shoulder, hip, knee | sagittal | standing | yes | `atan2(f.ant, -f.up)` |
| Ankle dorsi/plantar | knee, ankle, **heel**, foot_index | sagittal | ~90 shank-to-foot | yes | `90 - interior` |

Points a three-point implementation gets wrong:

- **Shoulder flexion and abduction use the same three landmarks but different planes.**
  One interior angle cannot produce both.
- **The knee flexes backwards** while elbow, hip and shoulder flex forwards. Opposite
  rotation sense; anatomy, not a sign error.
- **Ankle neutral is ~90 deg, not 180.** It also uses `heel -> foot_index` as the foot
  axis, not `ankle -> foot_index`: the latter carries a **-16.7 deg constant bias** and
  18% more noise (E3).
- **The anatomical frame is rebuilt every frame** from the landmarks. The model's world
  frame is camera-aligned, not body-locked - the hip vector swings 34 deg mean, 138 deg
  max as the subject turns (F18) - so projecting onto a fixed world plane and calling it
  sagittal is wrong.

### 2D or 3D

**2D image coordinates**, with the orientation requirement of section 6.

| | 2D | 3D world |
|---|---|---|
| Geometry | exact when the plane faces the camera | exact at any orientation |
| Landmarks | accurate: reads 3.2 deg elbow, 6.3 deg knee on a relaxed stance (F30) | 23% gain compression (F28) |
| Measured accuracy | shoulder flexion **MAE 2.7 deg** (F32) | elbow **MAE 12.8 deg** side-on (F28) |
| Failure mode | wrong below ~45 deg rotation | always compressed; worse with rotation |
| Detectable? | **yes** - rotation is measurable at runtime | **no** |

The deciding argument is not the error figures but the failure modes. 2D fails in a way
the application can detect and refuse; 3D's compression is invisible from its output,
cannot be calibrated away because the gain itself shifts with orientation (0.381 facing
the camera, 0.773 side-on), and is silently wrong. A measurement that lies quietly is
worse than one that declines to answer.

This reverses an earlier decision (F11, F20) that favoured 3D on geometric grounds. That
conclusion came from synthetic data, where landmarks are perfect by construction; it
measured the geometry and assumed the inputs. F21, F22 and F28 measured the inputs.

Hinge joints share one implementation between representations. **Plane-projected
measurements need two separate implementations**: the 3D form projects onto
`anterior = cross(left, up)`, which degenerates to pure Z when every z is 0, so it can
only ever return 0 or 180 on image coordinates (F34).

### Display

Signed values are shown as their **named opposite**, never as a minus sign, because the
chart lists the two directions as separate ranges. Hip at `-30` displays as "30 deg of
extension". Values outside the chart's normal range are flagged.

### Filtering

**One Euro on landmark POSITIONS, before angles are computed.** 40-45% better than
filtering the resulting angle during real movement, and equivalent at rest (F27) -
smoothing positions keeps the skeleton geometrically consistent, which smoothing the
output angle cannot repair afterwards.

Beta ~ 0.01. Against a fixed EMA this matters: EMA alpha=0.15 bought the lowest elbow
jitter at **240 ms of lag** (7 frames), and on the knee was worse on both axes at once
(F15). One Euro adapts its cutoff to velocity, giving equal or better smoothing at
0-48 ms.

**Peaks are read from the unfiltered signal.** Every filter shaves the extremes; One Euro
retains 88-89% of movement range, so a filtered signal under-reports maximum range of
motion by ~10%. Display filtered, measure peaks raw.

**Timestamps under drops and reordering.** One Euro's cutoff depends on `dt` between
consecutive samples, so the behaviour must be defined when frames go missing:

- `dt` comes from **`capture_ts`**, never from wall-clock time at the filter. Wall-clock
  would fold queueing delay into the velocity estimate and make smoothing depend on
  system load.
- Out-of-order results are already discarded by `seq` (section 4), so the filter only
  ever sees **monotonically increasing** timestamps. This is a consequence of that rule,
  not a second mechanism.
- Dropped frames simply produce a larger `dt`, which One Euro handles by construction -
  this is an argument *for* it over a fixed-alpha EMA, whose smoothing silently changes
  meaning when the frame rate varies.
- `dt` is clamped to [1 ms, 250 ms]. Below that, numerical blow-up; above it, the gap is
  long enough that the previous sample is not evidence about the current one, and the
  filter is reset rather than asked to interpolate across it.
- On reset - first frame, or after a gap exceeding the clamp - the filter outputs the
  raw value and begins accumulating again.

---

## 8. Capture

**DirectShow, 640x480, exposure fixed at 2^-5 s.**

- **DirectShow, not MSMF** (F26). MSMF returns the same frame up to four times - 74%
  duplicates in the original session, still 11% after the exposure fix - and roughly
  double the p95 jitter. DirectShow: 0% duplicates, 30 FPS, p95 48 ms. Counting `read()`
  calls measures the loop, not the camera.
- **Exposure set explicitly.** Auto-exposure lengthens exposure time indoors, costing 75%
  of frame rate *and* 10x accuracy: a motionless elbow read +/- 23 deg at auto against
  1.5-2 deg fixed (F7). The effect is lighting-dependent and therefore fragile, so the
  value is set for reproducibility rather than speed.
- **640x480.** 1280x720 caps at 10 FPS on this camera regardless of exposure (F5).
- **Dedicated capture thread holding only the newest frame**, so a slow consumer gets
  fresh data rather than a backlog.

**GPU is unavailable.** MediaPipe's own source states GPU support is limited to Ubuntu
(F1). All inference is CPU; reaching 30 FPS is a scheduling problem, not an accelerator
one. Model: `pose_landmarker_full` - 8 ms slower than `lite` but ~3x lower knee jitter,
while `heavy` is 2.3x slower and drifts 21.6 deg out of plane (F16, F24).

---

## 9. How latency and FPS are measured

Five distinct numbers. Three of them are rates, and conflating them is the single easiest
way to publish a misleading benchmark:

| Metric | Definition | Live (camera) | Replay (file) |
|---|---|---|---|
| **Source FPS** | frames the source can deliver | **30** (camera ceiling, F26) | unbounded |
| **Pipeline throughput** | frames the pipeline *can* process | **~43-60** (F36, VIDEO mode) | same |
| **Displayed FPS** | results actually drawn per second | **min(30, ~50) = 30** | ~43-60 |
| **Model inference latency** | around `detect_for_video()` alone | ~17-23 ms p50 | same |
| **End-to-end latency** | `capture_ts` to the moment its result is drawn | ~20-26 ms p50 | same |

**Throughput is measured by replaying a file, and is not a live frame rate.** The camera
delivers 30 unique frames per second and cannot be made to deliver more (F5, F26), so the
live application displays 30 FPS and no more. Quoting the replay figure as the
application's frame rate would be wrong.

Latency figures are far better than the earlier two-worker IMAGE-mode design (~39 ms
inference, ~53 ms end-to-end) because VIDEO mode skips the detector stage and the
sequential pipeline has no queueing delay between stages (F36).

The distinction is what answers the assignment's 60 FPS question honestly: **the pipeline
has headroom past 30, the capture device does not.** Throughput is measured by replay
precisely so the claim does not depend on a camera that caps below the target.

Reporting `1/displayed_FPS` as latency would give 33 ms against a true 53 ms. The brief
asks for inference latency and end-to-end latency separately for exactly this reason.

**What the README will report:**

- test hardware, OS, camera model, resolution, exposure setting, model variant
- source FPS (live) = 30, stated as a device limit with the measurement behind it
- displayed FPS (live), mean and p95
- pipeline throughput (replay), showing headroom above the requirement
- model inference latency, p50 and p95
- end-to-end latency, p50 and p95
- dropped-frame count and queue depth over the run
- the measurement method for each, and the script that produces them

Every stage records p50 and p95 over a rolling window. p95 matters because dropped-frame
stalls live in the tail. The whole set is written as JSON at exit, so the performance
section is generated rather than retyped.


---

## 10. Recording, replay and tests

**The frame source is an interface.** Three implementations: live camera, video file,
recorded landmark stream (JSONL). Everything downstream is identical in all three.

This is the instrument the investigation ran on, not a feature added afterwards. Two
smoothing filters cannot be compared on a live human, because the human moves differently
each time; they have to be replayed against one recording. The same applies to model
variants (F16, F24) and to every accuracy analysis.

What it buys:

- **Deterministic tests.** Golden-file tests over recorded sessions catch regressions in
  the conventions. 30 experiment scripts already exist and become the suite.
- **Reproducible by a reviewer with no webcam.** `fixtures/session.jsonl` holds 865
  frames across four movement phases.
- **Honest performance numbers.** Replaying a fixed stream benchmarks the pipeline
  independently of a camera that caps at 30 FPS, which is how the 60 FPS question gets
  answered rather than dodged.

Test layers:

1. **Synthetic exactness** - forward kinematics gives known angles; conventions must
   return them to 0.000000 deg (F19, F34). Isolates our maths from model error entirely.
2. **Golden replay** - recorded sessions produce stable expected outputs.
3. **Validity unit tests** - each signal fires on its recorded failure case (F30).
4. **Pipeline integration** - drop-oldest under load, no unbounded latency growth,
   out-of-order results rejected.

---

## Known limitations

- **Absolute accuracy is validated for elbow (F28) and shoulder flexion (F32, MAE 2.7
  deg).** Knee, hip, abduction and ankle rest on synthetic exactness plus jitter
  measurement; they have no physical reference.
- **Knee flexion could not be validated at all** in the orientation 2D requires. Side-on
  is what makes a sagittal measurement valid, and side-on is also what makes one leg
  occlude the other: visibility fell to 0.37-0.49, below the 0.50 trust threshold (F32).
  The system correctly refuses these values rather than reporting them. This is a genuine
  single-camera conflict, not a tuning problem.
- **The ankle is the least trustworthy measurement.** The model places `heel` and
  `foot_index` **12-14 cm apart against a true ~25 cm** (F22), halving the segment and
  doubling noise amplification.
- **Legs reconstruct far worse than arms** - 11-12% L/R bone asymmetry versus 3.4% (F22).
  Cause not established.
- **The three model variants disagree with each other by 10-18 deg** on identical frames
  (F24). `full` was chosen on jitter and rotation stability, not on absolute accuracy,
  which remains unmeasured for the other two.
- **No multi-person support.** `num_poses=1`.
