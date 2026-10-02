# Code standards

Operational resilience was deliberately scoped down (see BUILD_PLAN). **Code quality was
not.** A developer who has never seen this repository should be able to open it, find the
part they care about, and understand it without reading the rest.

This document is the contract. If the code disagrees with it, the code is wrong.

---

## 1. Repository map

```
zeuron/
  README.md               install, run, results          <- start here
  ARCHITECTURE.md         the design and its evidence
  requirements.txt        pinned dependencies
  biomech/
    pyproject.toml        package metadata, lint config
    src/biomech/          THE APPLICATION - production code
    tests/                mirrors src/ one-to-one
    tools/                documentation generators - held to this document
    research/             the investigation record - NOT production code
      experiments/        37 scripts that produced the findings
      lib/                shared helpers for those scripts
      fixtures/           recorded sessions, replay data
    docs/                 FINDINGS, BUILD_PLAN, DATA, this file
      benchmarks/         recorded performance runs, with hardware and config
    models/              downloaded .task files (gitignored)
```

`tools/` is production-quality but not part of the application: it generates
documentation from recorded measurements, so that a published number is never a retyped
one. It is linted and typed like `src/`, and `src/` never imports from it.

### The `research/` boundary is deliberate and stated

`research/experiments/` contains exploratory scripts. They have hardcoded assumptions,
inconsistent structure, and print-based output. **They are not held to this document, and
they are not pretending to be.**

They are kept because they are the evidence behind every decision in `src/`, and because
every finding must remain reproducible from the data that is still here. Six of them read
video that was deleted as personal data, and fail confusingly without it - see
`research/experiments/README.md`, which lists them and says how to supply your own
recording. Dressing one-off experiments up as production code
would cost real effort and make the repository *less* honest, not more.

The rule: **`src/` never imports from `research/`.** Anything the application needs is in
`src/`, even where that means the logic exists in both places. `research/lib/biomech_ref.py`
is the investigation's copy; `src/biomech/biomechanics/` is the application's, and the
synthetic audit runs against **both** to prove they agree.

---

## 2. Package layout

```
src/biomech/
  __init__.py           version, public exports
  __main__.py           entry point: python -m biomech
  cli.py                argument parsing only
  config.py             every tuned constant, each with its finding reference
  errors.py             exception hierarchy
  types.py              Frame, LandmarkSet, Measurement, Verdict

  capture/              frames in
    source.py           FrameSource protocol
    camera.py           live device
    video.py            mp4 replay
    landmarks.py        JSONL replay (enters downstream of inference)

  inference/
    pose.py             PoseEstimator - MediaPipe, VIDEO mode

  biomechanics/         coordinates -> angles. Knows nothing about trust.
    frame.py            anatomical frame construction
    angles.py           the twelve measurements
    conventions.py      display names, normal ranges, required view

  validity/             is this number believable? Computes no geometry.
    signals.py          the individual checks
    orientation.py      yaw, hysteresis, anterior direction
    rules.py            composition: which signals may veto what

  filtering/
    one_euro.py

  pipeline/
    runner.py           the composition root - the only module that knows all others

  ui/
    overlay.py          skeleton on the video pane
    panel.py            measurement and health panels

  metrics/
    timing.py           rolling p50/p95, drop counting
    report.py           JSON export

  recording/
    writer.py           JSONL + optional video
```

One idea per module. If a module needs "and" to describe it, it is two modules.

---

## 3. Dependency direction

Enforced by review, and by a test that fails on violation.

```
        types, errors, config          (leaves - import nothing internal)
              ^        ^
    capture, inference, filtering, metrics
              ^
        biomechanics                   (may use types, config)
              ^
          validity                     (may use biomechanics for the frame)
              ^
             ui                        (reads everything, mutates nothing)
              ^
           pipeline                    (composition root - knows all)
```

Two rules that matter more than the rest:

- **`biomechanics` must never import `validity`.** Angles are computed unconditionally;
  whether to believe them is a separate question asked by a separate module. This is the
  boundary that makes "indicate that state" a composable rule instead of conditionals
  smeared through the maths.
- **`ui` mutates nothing.** It receives values and draws them. A display layer that can
  change state is a display layer that will.

---

## 4. Types

### 4.1 Frozen dataclasses for data that crosses a module boundary

```python
@dataclass(frozen=True, slots=True)
class Frame:
    image: np.ndarray
    capture_ts: float
    seq: int
```

Immutable by default. A queued frame that a producer can still write to is a race
condition waiting for load.

### 4.2 Full type hints on every public function

Internal helpers may omit them where the type is obvious from two lines of context.
Anything crossing a module boundary is annotated without exception.

### 4.3 `Protocol` for interfaces, not inheritance

`FrameSource` is a `Protocol`. `CameraSource` does not inherit from anything - it simply
satisfies the shape. Three implementations, no base class, no `NotImplementedError` stubs.

### 4.4 No bare tuples across boundaries

`(87.3, True, "ok")` is unreadable at the call site. Use a named type.

---

## 5. Errors

```python
class BiomechError(Exception):          """Base for everything this package raises."""
class ConfigError(BiomechError):        """Invalid configuration."""
class CaptureError(BiomechError):       """Camera or source failure."""
class ModelError(BiomechError):         """Model file missing or unloadable."""
```

- Every raise is one of these, never a bare `Exception`.
- Messages state **what failed, what was expected, and what to do**:
  `"Camera 0 unavailable. Is another application using it? Try --device 1."`
- The entry point catches `BiomechError` and prints the message without a traceback.
  Everything else keeps its traceback, because an unexpected error should look unexpected.
- **Never a bare `except:`.** Never `except Exception: pass`.

---

## 6. Documentation

### 6.1 Module docstrings state what the module owns AND what it does not

```python
"""Anatomical angle computation.

Owns: the anatomical frame, the twelve measurements, chart conventions.
Does NOT own: whether a measurement is trustworthy (see validity/), smoothing
(see filtering/), or how anything is displayed (see ui/).
"""
```

The "does not own" half is what stops a module accreting.

### 6.2 Every tuned constant carries its finding reference

```python
# Visibility below this rejects the measurement.
# 0.50 let a 76 deg error through at 0.52; see FINDINGS.md F37.
MIN_VISIBILITY: float = 0.55
```

A reader must be able to trace any number to the experiment that produced it. This is
the single highest-value convention in the repository.

### 6.3 Comment the non-obvious, never the obvious

```python
# The knee flexes posteriorly while elbow, hip and shoulder flex anteriorly,
# so the rotation sense is inverted here. Anatomy, not a sign error.
```

Not `# increment the counter`.

### 6.4 Docstrings explain why and what, not how

The body says how. If the how needs explaining, the function is too long.

---

## 7. Naming

- Modules, functions, variables: `snake_case`. Classes: `PascalCase`. Constants: `UPPER_SNAKE`.
- **Anatomical names in full.** `shoulder_flexion`, not `sh_flex`. The research scripts use
  abbreviations; the application does not.
- Units in the name where ambiguous: `capture_ts` (seconds), `latency_ms`, `yaw_deg`.
- Booleans read as assertions: `is_valid`, `has_person`, not `flag` or `status`.
- No single letters outside tight numeric loops. `v` for a vector in a three-line geometry
  function is fine; `v` as a parameter name is not.

---

## 8. Functions

- **Under 40 lines.** Longer means it is doing two things.
- **One level of abstraction per function.** Do not mix "compute the anatomical frame" with
  "clamp the arccos argument" in the same body.
- **No side effects in anything named like a query.** `compute_angles()` returns; it does
  not draw, log at INFO, or mutate its input.
- **Arguments over globals.** `config` is passed in, never reached for.

---

## 9. Tests

```
tests/
  test_biomechanics/
    test_angles.py          synthetic exactness - must return 0.000000
    test_conventions.py     display names, ranges, sign handling
  test_validity/
    test_signals.py         each signal fires on its recorded failure case
    test_orientation.py     hysteresis produces no flicker
  test_filtering/
    test_one_euro.py        dt clamping, reset behaviour
  test_capture/
    test_sources.py         all three sources satisfy the protocol
  test_pipeline/
    test_runner.py          drop-oldest under load; latency stays bounded
  test_architecture.py      dependency direction is not violated
```

- **Mirrors `src/` one-to-one.** Finding the test for a module requires no searching.
- **Every test names what it asserts**: `test_straight_elbow_reads_zero_not_180`.
- **No live camera in tests.** Everything replays from `research/fixtures/`.
- **Coverage measured and reported** - criterion 5 names it. Target 80% on `src/biomech/`,
  excluding `ui/`, which is verified by eye.

---

## 10. Tooling

`pyproject.toml` carries:

- `ruff` for lint and format - one tool, no black/isort/flake8 stack
- line length 100
- `pytest` config with `--cov=src/biomech`
- package metadata and the `biomech` console entry point

Committed configuration, so `ruff check` and `pytest` behave identically for a reviewer.

---

## 11. What a new developer reads, in order

1. `README.md` - what it is, how to run it
2. `ARCHITECTURE.md` - the design and the evidence behind it
3. `src/biomech/types.py` - the four data types everything passes around
4. `src/biomech/pipeline/runner.py` - the composition root, ~100 lines, shows the shape
5. Whichever module they came for

If step 4 does not make the whole system legible, the runner is doing too much.

---

## 12. Definition of done, per module

- [ ] Module docstring states what it owns and what it does not
- [ ] Public functions fully type-hinted
- [ ] Tuned constants live in `config.py` with finding references
- [ ] No import that violates section 3
- [ ] Tests mirror the module and name their assertions
- [ ] `ruff check` clean
- [ ] Under 40 lines per function
