# Build plan (frozen)

Thirteen phases. Each is a **single idea** you should be able to explain on its own, has
**explicit pass criteria**, and cites the findings that justify it.

This is an interview assignment, not a production product. Operational resilience is
deliberately scoped to *demo-killers* only - the brief grades inference design, accuracy,
real-time performance, technical reasoning, code quality, and UI, in that order. There is
no criterion for unattended operation and we are not building for one.

Findings cited as `F<n>` are in [FINDINGS.md](FINDINGS.md); design in
[ARCHITECTURE.md](../../ARCHITECTURE.md).

**Every phase reports back in the same five fields:**

    What changed | Why | How tested | PASS criteria | Weak point

The last one is the important one. Anyone can explain why a design works.

| Phase | Idea | Build |
|---|---|---|
| 0 | Environment - config, dependencies, CLI, logging | 30 min |
| 1 | Capture - get frames out of the camera, correctly | 45 min |
| 2 | Inference - turn one frame into 33 points | 30 min |
| 3 | Pipeline - keep the picture fresh under load | 45 min |
| 4 | Geometry - turn points into anatomical angles | 60 min |
| 5 | Orientation - know which plane the subject presents | 30 min |
| 6 | Trust - decide when a number is not worth showing | 40 min |
| 7 | Smoothing - remove jitter without adding lag | 25 min |
| 8 | Display - show twelve measurements honestly | 50 min |
| 9 | Replay and tests - make it reproducible | 60 min |
| 11 | Full-system benchmark - re-prove 30 FPS with everything on | 30 min |
| 12 | README, code docs, clean-clone test | 60 min |

**~8.5 h of build**, plus your review cycle. Budget 12-14 h elapsed.

**Phase 10 (physical validation) sits outside this chain.** It depends on daylight, a
printed reference, and poses that have already failed twice. It runs when conditions
allow and **must not block implementation**.

One git commit per phase, named `phase-N-<idea>`. Not a ceremony - it is what makes
"show me the diff" answerable.

---

# PHASE 0 - ENVIRONMENT

**Idea:** every threshold in this system came from an experiment. Scattering them as
literals across seven modules throws that away.

## 0.1 Dependencies
### 0.1.1 `requirements.txt`, pinned: `mediapipe==1.0.1`, `opencv-python==5.0.0.93`,
        `numpy==2.5.3`, `matplotlib`
### 0.1.2 Note Python 3.14 explicitly - unusual enough that a reviewer will hit it
### 0.1.3 Model download command, since `models/*.task` is gitignored

## 0.2 Configuration
### 0.2.1 One `config.py`, every tuned value in it
### 0.2.2 Each value carries its finding reference as a comment
### 0.2.3 Values to centralise:
- capture: device index, 640x480, DirectShow, exposure `2^-5` (F5, F7, F26)
- validity: visibility `0.55`, bone cv `6%`, asymmetry advisory `8%` (F30, F37)
- orientation: sagittal `>=45`, frontal `<=30`, hysteresis `5 deg` / `5 frames` (F27, F34, F36)
- filter: One Euro `beta=0.01`, dt clamp `[1ms, 250ms]` (F15, F27)
- model: `full` (F16, F24)

### 0.2.4 Config is serialised into every recording, so a replay is self-describing

## 0.3 CLI surface
### 0.3.1 `--source camera|video|landmarks` (default camera)
### 0.3.2 `--path` for the two replay sources
### 0.3.3 `--record-landmarks` / `--record-video` (video off by default, it captures the subject)
### 0.3.4 `--device N`, `--model lite|full|heavy`
### 0.3.5 `--metrics-out` for the JSON dump

## 0.4 Logging
### 0.4.1 `logging` with INFO default, `--verbose` for DEBUG
### 0.4.2 Startup logs the resolved config, so a session is reconstructible from its log

**PASS:** `pip install -r requirements.txt` works in a fresh venv; `--help` lists every
flag; startup prints the resolved config; no tuned value appears outside `config.py`.

**Weak point:** pinned versions on Python 3.14 are a narrow path. If a reviewer is on
3.12, mediapipe 1.0.1 should still resolve, but this is untested.

---

# PHASE 1 - CAPTURE

**Idea:** a camera is an unreliable narrator. Getting frames is not the same as getting
*fresh, distinct, correctly-exposed* frames.

## 1.1 Device layer
### 1.1.1 Open with the **DirectShow** backend explicitly
### 1.1.2 Request 640x480; read back actual width/height/FOURCC
### 1.1.3 Disable auto-exposure, set `2^-5 s`; warn loudly if it did not apply
### 1.1.4 Refuse to start on a resolution we did not ask for

## 1.2 Freshness
### 1.2.1 Capture thread looping `read()`, never blocking the consumer
### 1.2.2 Copy the frame before publishing; stamp `capture_ts` and `seq`
### 1.2.3 Single newest-frame slot, not a queue; overwrite counts as a drop

## 1.3 Source abstraction
### 1.3.1 `FrameSource` - `next_frame() -> Frame | None`
### 1.3.2 `CameraSource`, `VideoSource`

## 1.4 Failure handling (demo-killers only)
### 1.4.1 Camera unavailable -> clear message naming the device index, exit 1
### 1.4.2 Camera disconnects mid-run -> on-screen banner, keep the last frame, retry
### 1.4.3 Invalid/empty frame -> skip and count, never crash

**PASS:** probe reports >= 29 unique FPS; exposure confirmed applied; unplugging the
camera mid-run shows a banner rather than a traceback.

**Must explain:**
- Why DirectShow. *MSMF returns the same frame up to four times - 74% duplicates - so
  counting `read()` calls measures your loop, not the camera (F26).*
- Why manual exposure. *Auto-exposure costs 75% of frame rate and makes a motionless
  elbow read +/- 23 deg instead of 1.5 (F5, F7).*
- Why a slot, not a queue. *A queue keeps FPS healthy-looking while latency grows
  unbounded.*

**Weak point:** exposure control is driver-dependent. A camera that ignores
`CAP_PROP_EXPOSURE` silently reverts to the auto-exposure behaviour that costs 10x
accuracy. The warning is the only defence.

**Implements:** F5, F7, F26

---

# PHASE 2 - INFERENCE

**Idea:** the model is a function from image to 33 labelled points. Everything anatomical
happens outside it.

**Scope boundary:** proves `frame -> model -> landmarks` **only**, on a saved video file.
No camera, no loop, no window. Never debug inference and threading at once.

## 2.1 Detector lifetime
### 2.1.1 `PoseLandmarker` in **VIDEO** running mode
### 2.1.2 One instance, one thread, never shared; closed on shutdown

## 2.2 Per-frame call
### 2.2.1 BGR to RGB; wrap as `mp.Image`
### 2.2.2 `detect_for_video(image, timestamp_ms)`, **monotonic** timestamps
### 2.2.3 Time the call in isolation for the inference metric

## 2.3 Output normalisation
### 2.3.1 `LandmarkSet`: image xy, world xyz, per-point visibility
### 2.3.2 Carry `capture_ts` and `seq` through unchanged
### 2.3.3 "No person detected" is an explicit state, never empty landmarks

## 2.4 Failure handling
### 2.4.1 Model file missing -> message with the download command, exit 1
### 2.4.2 First load takes ~12 s (F4) -> log it so a frozen window is explained

**PASS:** model loads; landmark count == 33; no-person case returns the explicit state;
inference p50 recorded and within 15-30 ms on `sess_squat.mp4`.

**Must explain:**
- IMAGE vs VIDEO. *IMAGE re-runs the detector every frame; VIDEO locates once then
  tracks. 2.4x faster, 82% less knee jump (F36).*
- Why one worker. *VIDEO assumes a contiguous stream. Two workers each see every second
  frame: knee jump 5.31 -> 9.32 deg. The biggest performance win came from deleting
  concurrency, not adding it (F36, supersedes F25).*
- Why the GPU is idle. *MediaPipe's own source: GPU support is Ubuntu-only (F1).*

**Weak point:** VIDEO mode holds tracking state. If tracking locks onto the wrong person
or a reflection, there is no reset path in this design short of restarting the detector.

**Implements:** F1, F16, F36

---

# PHASE 3 - PIPELINE

**Idea:** throughput and latency are different numbers, and only one of them is felt.

**Scope boundary:** wires 1 and 2 and proves the result is *fresh and fast*. Still no
biomechanics - proving 30 FPS before anything is layered on top is why this is third.

## 3.1 The loop
### 3.1.1 Newest frame -> infer -> hand on; skip cleanly when empty
### 3.1.2 Shutdown joins the capture thread and closes the device

## 3.2 Timing
### 3.2.1 Inference latency around `detect_for_video` only
### 3.2.2 End-to-end latency, `capture_ts` to draw
### 3.2.3 **Render time measured separately** - never measured before, and a 1100x825
        resize plus ~30 `putText` calls is not free
### 3.2.4 Rolling window, p50 and p95

## 3.3 Backpressure
### 3.3.1 Count slot drops; expose drop rate as a health signal

**PASS:** >= 30 displayed FPS with a bare skeleton; inference p50 recorded; e2e p50
recorded; render p50 recorded; drop rate reported; clean Ctrl-C shutdown.

**Must explain:**
- Why `1/FPS` is not latency. *Displayed FPS is camera-capped at 30; e2e latency is
  ~20-26 ms. `1/30 = 33 ms` is wrong in both directions.*
- The three rates. *Source 30 (camera ceiling), throughput 43-60 (replay-measured),
  displayed 30. The pipeline has headroom; the capture device does not. That is the
  honest answer to the 60 FPS question.*
- Why p95. *Dropped-frame stalls live in the tail.*

**Weak point:** everything after capture is on one thread, including rendering. If render
time grows with the panel in phase 8, it eats the inference budget directly. Phase 11
exists because of this.

**Implements:** F25, F26, F36

---

# PHASE 4 - GEOMETRY (the numpy work)

**Idea:** the model gives coordinates. Anatomy is arithmetic you do yourself, and the
conventions are not obvious.

## 4.1 Anatomical frame
### 4.1.1 `up` = mid_shoulder - mid_hip
### 4.1.2 `left` = L_hip - R_hip, orthogonalised against `up`
### 4.1.3 `anterior` = cross(left, up)
### 4.1.4 Rebuilt **every frame**; never cached, never a fixed world axis

## 4.2 Hinge joints (one implementation, both representations)
### 4.2.1 Elbow `180 - interior(shoulder, elbow, wrist)`
### 4.2.2 Knee `180 - interior(hip, knee, ankle)`, unsigned, clamped at 0

## 4.3 Plane-projected joints (separate 2D and 3D implementations)
### 4.3.1 Shoulder flexion/extension - **sagittal**, signed
### 4.3.2 Shoulder abduction/adduction - **frontal**, signed, lateral flipped on the right
### 4.3.3 Hip flexion/extension - sagittal, signed
### 4.3.4 Ankle `90 - interior(knee, ankle, heel->foot_index)`, sagittal-projected

## 4.4 Presentation
### 4.4.1 Signed -> named direction ("30 deg of extension", never "-30")
### 4.4.2 Flag values outside the chart's normal range
### 4.4.3 Twelve measurements: six types x two sides

## 4.5 Proof
### 4.5.1 Port `lib/biomech_ref.py` unchanged where possible
### 4.5.2 Re-run the synthetic audit against the ported module
### 4.5.3 Require **0.000000 deg** before proceeding

**PASS:** synthetic audit returns 0.000000 deg for all twelve, both directions; neutral
pose reads 0; twelve values appear live.

**Must explain:**
- Why a straight elbow reads 0. *Chart neutral is 0; raw interior is 180.*
- Why shoulder flexion and abduction share landmarks. *Same humerus vector, two planes.
  One interior angle cannot produce both (F17).*
- Why the knee flexes the other way. *Anatomy, not a sign bug.*
- Why the frame is rebuilt per frame. *The model's world frame is camera-aligned; the hip
  vector swings 34 deg mean, 138 deg max as the subject turns (F18).*
- Why the ankle uses the heel. *`ankle->toe` carries -16.7 deg constant bias and 18% more
  noise (E3).*
- Why 2D over 3D. *3D geometry is exact but its landmarks compress movement 23% - bend
  150 deg, it reports 125. 2D is wrong only when the plane is not presented, and that is
  **detectable**. A measurement that lies quietly is worse than one that declines (F28, F29).*

**Weak point:** the anatomical frame degenerates if hips or shoulders are not both
visible. Nothing in this phase handles that - phase 6 must.

**Implements:** F17, F18, F19, F28, F29, F34, E3

---

# PHASE 5 - ORIENTATION

**Idea:** one fixed camera physically cannot show all twelve measurements at once.

## 5.1 Yaw estimate
### 5.1.1 `v = world[L_shoulder] - world[R_shoulder]`
### 5.1.2 `yaw = atan2(|v.z|, |v.x|)`; 0 facing, 90 side-on
### 5.1.3 No calibration, no reference, no running maximum

## 5.2 Plane bands
### 5.2.1 Sagittal `>= 45`, frontal `<= 30`, between them neither

## 5.3 Hysteresis
### 5.3.1 5 deg margin; state must persist 5 frames (~0.25 s)
### 5.3.2 Report current state and live yaw

**PASS:** facing the camera reads < 10 deg, side-on reads > 60 deg; turning slowly through
the boundary produces zero state flicker.

**Must explain:**
- Why abduction is special. *Frontal plane needs face-on - the exact orientation in which
  the five sagittal measurements are catastrophically wrong (F34).*
- Why 3D here after rejecting it for angles. *Gain compression distorts **magnitudes**;
  this uses only the **direction** of the shoulder axis, accurate to 1.1-6.2 deg (F36).*
- Why two earlier estimators were discarded. *Raw pixels conflate rotation with distance
  (0-39.7 deg error); shoulder/trunk ratio fixes distance but not posture (50 deg spread)
  (F35, F36).*
- Why hysteresis. *Yaw noise ~1 deg sd; real postures sit at 44-46 deg. Bare threshold
  flipped 3 times in one phase; 5 deg + 5 frames gives zero (F36).*

**Weak point:** depends on world landmarks, which F28 showed are unreliable for
magnitudes. Direction has been validated across eleven phases - but only for this subject,
this camera.

**Implements:** F27, F34, F35, F36

---

# PHASE 6 - TRUST

**Idea:** the hardest requirement in the brief is refusing to answer.

## 6.1 Signals, and which may veto

A signal that cannot be computed must not block a measurement. The brief says compute each
side whenever **that side's** landmarks are reliable, so a bilateral veto would be stricter
than required.

| # | Signal | Scope | Threshold | May veto? |
|---|---|---|---|---|
| 6.1.1 | Detection rate | global | person found | **yes** |
| 6.1.2 | Per-joint visibility | that measurement's own landmarks | < **0.55** | **yes** |
| 6.1.3 | Bone-length variance | **per side**, one segment over time | cv > 6% | **yes** |
| 6.1.4 | Plane presented | that measurement's plane | phase 5 bands | **yes** |
| 6.1.5 | Frame landmarks present | hips + shoulders, for the anatomical frame | visible | **yes** |
| 6.1.6 | Bone-length asymmetry | **bilateral** | > 8% | **no - advisory** |

## 6.2 Why asymmetry is advisory only (F37)

| Phase | knee error | visibility | variance | asymmetry |
|---|---|---|---|---|
| occl_leg | **76.2 deg** | 0.52 - catches at 0.55 | 5.2% | 9.3% |
| seated | **147.0 deg** | 0.88 - misses at any threshold | **14.2% - catches** | 12.2% |
| cam_low | **0.2 deg** (fine) | 0.89 | 2.2% | **14.1% - FALSE POSITIVE** |

Visibility and variance catch both genuine failures. Asymmetry adds no unique coverage and
false-positives on camera elevation. Threshold change: **visibility 0.50 -> 0.55**.

## 6.3 Composition
### 6.3.1 Valid only if every **veto-capable** signal in scope passes
### 6.3.2 A veto signal that cannot be computed **blocks**; an advisory one is **omitted**
### 6.3.3 Failure carries a **reason**, not a boolean
### 6.3.4 All signals on **raw** landmarks, before smoothing

## 6.4 Anterior direction
### 6.4.1 Three cues: nose, ear->nose, heel->toe
### 6.4.2 Update only on unanimous agreement; otherwise hold
### 6.4.3 If never established, withhold signed sagittal measurements

**PASS:** hiding one arm invalidates that arm only, leaving the other reporting; sitting
invalidates the legs; turning invalidates the wrong-plane measurements; every blank shows
a reason.

**Must explain:**
- Why visibility is per-measurement. *Leg visibility stayed at 0.95 while an arm was
  hidden behind the back (F30).*
- Why asymmetry does not veto. *Bilateral, so uncomputable when one side is hidden - and
  false-positives on camera elevation. Variance gives the same coverage per-side (F37).*
- Why validity runs on raw landmarks. *Smoothing removes exactly the variation these
  checks look for.*
- Why unanimous, not majority. *Nose and ear cues are both head-based and fail together
  with camera elevation, outvoting the foot cue that is right - voting scored 0.51 where
  the best single cue scored 1.00. Voting assumes independent failures (F35).*

**Weak point:** thresholds were tuned on one subject in one room. Someone in dark
trousers, or a different body shape, could sit on the wrong side of 0.55 permanently.

**Implements:** F22, F30, F35, F37

---

# PHASE 7 - SMOOTHING

**Idea:** every filter trades jitter against lag. One Euro mostly dissolves the trade.

## 7.1 Filter
### 7.1.1 One Euro, `beta=0.01`, on **landmark positions**
### 7.1.2 One instance per coordinate, reset together

## 7.2 Time handling
### 7.2.1 `dt` from `capture_ts`, never wall-clock
### 7.2.2 Clamp to `[1 ms, 250 ms]`; reset beyond rather than interpolate across a gap

## 7.3 Two signals
### 7.3.1 Filtered -> angles -> display
### 7.3.2 Raw -> validity, and -> peak/ROM statistics

**PASS:** standing still, displayed angles vary < 2 deg; fast movement shows no visible
lag; the same clip replayed with and without smoothing differs as F27 predicts.

**Must explain:**
- Why positions, not the angle. *Smoothing positions keeps the skeleton geometrically
  consistent; smoothing the output cannot repair inconsistent geometry. 40-45% better
  during movement (F27).*
- Why One Euro, not a moving average. *EMA alpha=0.15 bought the lowest elbow jitter at
  240 ms lag, and on the knee was worse on both axes at once (F15).*
- Why peaks come from raw. *One Euro retains 88-89% of range, so filtered under-reports
  maximum ROM by ~10%.*

**Weak point:** `beta=0.01` was tuned on elbow and knee only. Ankle and shoulder may want
different values; one constant for all twelve is an untested simplification.

**Implements:** F13, F15, F27

---

# PHASE 8 - DISPLAY

**Idea:** show all twelve always, and make the reason for every blank visible.

**Semantic boundary:** the skeleton shows **landmark** confidence; the panel shows
**measurement** validity. An elbow landmark can be perfectly visible while shoulder
flexion is invalid because the subject faces the wrong way. Colouring the skeleton by
measurement validity would conflate the two.

## 8.1 Video pane
### 8.1.1 Camera image; skeleton coloured by **per-landmark visibility**
### 8.1.2 Orientation indicator: live yaw and current plane

## 8.2 Measurement panel
### 8.2.1 Twelve rows, always present, never reordered
### 8.2.2 Valid: named direction and magnitude ("flexion 87 deg")
### 8.2.3 Invalid: `--` plus the **measurement-level** reason
### 8.2.4 Advisory warnings shown separately, never as a blank
### 8.2.5 Out-of-range flagged against the chart

## 8.3 Health panel
### 8.3.1 Displayed FPS, inference p50/p95, e2e p50/p95, render p50
### 8.3.2 Drop rate, orientation state

**PASS:** all twelve rows visible at all times; no row ever blank without a reason; health
panel live; **render p50 recorded for phase 11**.

**Must explain:**
- Why all twelve always. *Rows appearing and vanishing read as bugs. A permanent list with
  reasons teaches the constraint - a direct implementation of "indicate that state".*
- Why at most ten valid at once. *Abduction's plane excludes the other five (F34).*
- Why skeleton colour and panel state differ. *Landmark confidence and measurement
  validity are not the same question.*

**Weak point:** this is where frame time grows. Every element added here is subtracted
from the inference budget on the same thread.

**Implements:** F30, F34

---

# PHASE 9 - REPLAY AND TESTS

**Idea:** the instrument the investigation ran on, kept as a product capability.

## 9.1 Recording
### 9.1.1 Landmarks + timestamps as JSONL
### 9.1.2 Raw video alongside, a **first-class capability** behind an explicit flag - video
        replay is the only thing that can compare pose models, because a landmark stream
        is already one model's output
### 9.1.3 Serialise the config, so a replay is self-describing
### 9.1.4 Video off by default and clearly labelled - it captures the subject

## 9.2 Replay sources
### 9.2.1 `VideoSource` - frames, exercises the model
### 9.2.2 `LandmarkSource` - landmark sets, enters at the filter stage
### 9.2.3 Identical downstream code in all three modes

## 9.3 Tests
### 9.3.1 Synthetic exactness - conventions return 0.000000 deg
### 9.3.2 Golden replay - recorded sessions produce stable expected output
### 9.3.3 Validity units - each signal fires on its recorded failure case
### 9.3.4 Pipeline - drop-oldest under load; latency does not grow unbounded
### 9.3.5 **Coverage measured and reported** - criterion 5 names it explicitly

**PASS:** `--source landmarks --path fixtures/sessions/session.jsonl` runs with no webcam;
all tests green; coverage number recorded.

**Must explain:**
- Why two replay levels. *A landmark stream is already one model's output, so it cannot
  compare models. Video replay can - that is how lite/full/heavy were compared (F16, F24).*
- Why this preceded being a feature. *Two filters cannot be compared on a live human; the
  human moves differently each time.*
- What it buys the reviewer. *They can run it with no webcam.*

**Weak point:** golden tests freeze current behaviour, including any bug present when they
were written.

**Implements:** F15, F16, F24, F27

---

# PHASE 11 - FULL-SYSTEM BENCHMARK

**Idea:** the submission is the whole application, not phase 3.

## 11.1 Measure everything on
### 11.1.1 Camera + inference + filtering + angles + validity + rendering
### 11.1.2 Displayed FPS, inference p50/p95, e2e p50/p95, render p50/p95, drop rate
### 11.1.3 Over a representative run (>= 60 s), both standing and moving

## 11.2 Compare against phase 3
### 11.2.1 How much did phases 4-8 cost?
### 11.2.2 Attribute the delta: geometry, validity, rendering

## 11.3 Fallback ladder, if below 30 FPS
Apply in order, stopping as soon as 30 is met, and **record which step was needed**:
### 11.3.1 Reduce display resize from 1100x825 to native 640x480
### 11.3.2 Draw the panel every 2nd frame; angles still computed every frame
### 11.3.3 Simplify the skeleton overlay to key landmarks only
### 11.3.4 Drop to the `lite` model - **last resort**, it costs 3x knee jitter (F16)

**PASS:** >= 30 displayed FPS with everything enabled, or a documented fallback step with
the measured result.

**Weak point:** measured on one machine. The claim is about this hardware and says so.

---

# PHASE 12 - DOCUMENTATION

## 12.1 Metrics export
### 12.1.1 JSON at exit: all rates and latencies, p50 and p95
### 12.1.2 Test-hardware block filled automatically
### 12.1.3 Performance section generated, not retyped

## 12.2 Code documentation (an explicit deliverable in the brief)
### 12.2.1 Module docstrings stating what each owns and does **not** own
### 12.2.2 Every tuned constant annotated with its finding reference
### 12.2.3 Non-obvious geometry commented with the convention it implements

## 12.3 README
### 12.3.1 Install, model download, run - every CLI flag
### 12.3.2 Setup requirements: side-on for sagittal, face-on for abduction, distance
### 12.3.3 Performance results from phase 11
### 12.3.4 Accuracy results from phase 10, **including what is not validated**
### 12.3.5 Known limitations

## 12.4 Final checks
### 12.4.1 Fresh-clone install test in a new venv
### 12.4.2 Run with no camera attached (replay path)
### 12.4.3 No absolute paths remain

## 12.5 Personal data removal

Run **after** the application is finished and its results are recorded, because some of
this data is what makes the results reproducible. Removing it earlier would mean
re-recording to finish the work.

### 12.5.1 Inventory first
```
find biomech/research/fixtures -type f \( -name "*.mp4" -o -name "*.png" -o -name "*.jpg" \)
git log --all --pretty=format: --name-only --diff-filter=A | sort -u | grep -iE "\.(mp4|png|jpg|jsonl)$"
```
Disk and history are different questions. A file deleted from disk that was ever
committed is still in history and still published.

### 12.5.2 Local media - delete unconditionally

Four session videos, four exposure-test videos, and one raw still. None was ever
committed (the `.gitignore` caught them), so deleting from disk removes them completely.

**Cost:** the model-comparison findings (F16, F24) replay *identical frames* through
lite/full/heavy and cannot be reproduced without video. A landmark stream will not do -
it is already one model's output. Those findings become documented-but-unreproducible.

### 12.5.3 Landmark JSONL - a decision, not a default

Five files are in git history. They contain 3D body coordinates, not images: body
proportions and movement patterns across every recorded session.

| Option | Keeps | Costs |
|---|---|---|
| **Keep all** | Every finding reproducible; reviewer runs replay with no webcam | Body-coordinate data stays public |
| **Keep one short session** | Replay demo works; the "no webcam needed" claim holds | Filter and accuracy findings stop being reproducible |
| **Remove entirely** | No personal data at all | Replay is undemonstrable; **requires a history rewrite** |

Removal is not `git rm`. It needs `git filter-repo` or BFG, a force-push, and it breaks
every existing clone:
```
git filter-repo --path-glob 'biomech/**/fixtures/**/*.jsonl' --invert-paths
```
Note the paths in history are the **pre-restructure** ones (`biomech/fixtures/...`), not
the current ones.

**Default recommendation: keep one short session, remove the rest.** The reviewer keeps
a working `--source landmarks` demo, and the volume of retained personal data drops from
five sessions to one. If the submission is due imminently, keeping all is defensible -
the data is coordinates rather than imagery, and the repository can be made private or
deleted after review.

### 12.5.4 Decide the repository's visibility
It is currently **public**. If it stays public after submission, everything in it is
permanently crawlable. Making it private, or deleting it once the interview concludes,
is a cheaper control than a history rewrite.

### 12.5.5 Verify
Re-run 12.5.1 and confirm the output matches the decision. Then confirm the application
still starts and the test suite still passes: removing fixtures will skip tests that
depend on them, and a skip must not be mistaken for a pass.

**PASS:** a stranger can clone, install, and run both live and replay from the README
alone; and the personal data remaining in the repository is what 12.5.3 deliberately
chose to keep, verified in history rather than only on disk.

---

# PHASE 10 - PHYSICAL VALIDATION (off the critical path)

**Idea:** the brief requires physical validation of elbow, knee, and one challenging
out-of-plane measurement. It depends on daylight, a printed reference, and poses that have
already failed twice, so it **runs when conditions allow and never blocks implementation**.

**Status:** elbow validated (F28). Shoulder flexion validated, MAE 2.7 deg (F32). **Knee
unvalidated** - the attempt was refused by the system's own validity layer.

## 10.1 Knee, in daylight
### 10.1.1 Re-run the anchored protocol with better light
### 10.1.2 Target: leg visibility above 0.55
### 10.1.3 Two anchored poses: standing locked (0 deg), seated on a chair (90 deg)
### 10.1.4 **If visibility still fails, that is a result.** Record it and stop

## 10.2 Shoulder against an independent reference
### 10.2.1 Printed protractor, on-screen guide **off**
### 10.2.2 Replaces a partly self-referential column with a measured one

## 10.3 Analysis
### 10.3.1 MAE, bias, 95% limits of agreement per measurement
### 10.3.2 State the reference's own error (+/- 5 deg printed, +/- 8 deg anchored)
### 10.3.3 Record which measurements remain unvalidated, and why

**Do not fabricate a result.** Six of the twelve measurements will ship with synthetic
validation only. Saying so precisely is stronger than implying otherwise.

**Must explain:**
- Why the knee is hard. *Side-on makes a 2D sagittal measurement valid, and side-on is
  also what makes one leg occlude the other. A genuine single-camera conflict (F32).*
- Why refusing is the correct outcome. *A confident wrong number would be worse. The brief
  asks for the limitations of the validation method.*

**Implements:** F28, F32, F33
