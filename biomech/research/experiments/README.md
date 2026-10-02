# The investigation scripts

These produced the findings in [`../../docs/FINDINGS.md`](../../docs/FINDINGS.md). They are
exploratory: hardcoded assumptions, inconsistent structure, print-based output. They are
**deliberately not held to `docs/CODE_STANDARDS.md`** and are kept because they are the
evidence, not because they are production code.

`src/` never imports from here.

---

## Six of these need a video recording that is not in the repository

The video fixtures were images of the subject and this repository is public, so they were
deleted rather than published ([`docs/DATA.md`](../../docs/DATA.md)). Landmark recordings
(`.jsonl`) **are** here, so anything that replays landmarks still runs.

These read `fixtures/sessions/*.mp4` and will **not** run as-is:

| Script | What it established |
|---|---|
| `02_model/d2_models.py` | lite / full / heavy comparison (F16) |
| `02_model/e6_heavy_recheck.py` | heavy's 21.6 deg out-of-plane drift (F24) |
| `05_pipeline/e7_pipeline.py` | threaded pipeline throughput (F25) |
| `06_validation/c4_session.py` | the twelve-phase failure session (F30) |
| `07_architecture/e12_freeze.py` | VIDEO vs IMAGE mode (F36) |
| `01_capture/b1_camera_probe.py` | exposure and backend probing (F5, F26) |

**They fail badly, not clearly.** With no video to read they collect zero frames and then
die inside numpy — `IndexError: index -1 is out of bounds for axis 0 with size 0` — rather
than saying the file is missing. That is a real rough edge, left as-is because these
scripts are a historical record and editing them now would mean re-verifying findings they
already produced.

### To run them

Record your own clip first, then point the script at it:

```bash
biomech --source camera --duration 30 --record-video biomech/research/fixtures/sessions/sess_squat.mp4
```

The numbers will be your subject on your hardware, so they will not reproduce the published
values exactly. What they can reproduce is the *shape* of each result — that VIDEO mode
beats IMAGE mode, that `heavy` drifts where `full` does not.

Only `a1_model_bench.py` of the model scripts needs no video; it benchmarks the model on
synthetic input.

---

## Everything else runs from what is here

The landmark fixtures are committed, so the filter comparison, the convention checks, the
accuracy analyses, the validity-signal coverage and the orientation estimators all replay
unchanged:

```bash
python biomech/research/experiments/08_regression/g1_abduction.py
```

That one is the most recent and the easiest to follow: it tests three competing
explanations for a reported bug and refutes two of them (F39).
