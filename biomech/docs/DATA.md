# What personal data is in this repository, and why

This repository is **public** and contains recorded data of one human subject — the author.
That was a deliberate decision with a stated cost, taken at the end of the build (build plan
phase 12.5). This document records what is here, what is not, and why.

Decision date: **2026-10-01**. Reaffirmed **2026-10-02** when `abduction.jsonl` was
added: F39 cannot be reproduced without it, which is the same reason the other five are
here.

---

## What is here

Six landmark recordings, 23.0 MB, committed and kept:

| File | Size | Duration | What depends on it |
|---|---|---|---|
| `fixtures/robustness/robustness_session.jsonl` | 7.40 MB | 54 s | F30 failure detection, F35, F36 |
| `fixtures/accuracy/accuracy_session.jsonl` | 4.31 MB | 30 s | F28 absolute accuracy |
| `fixtures/sessions/session.jsonl` | 1.90 MB | 29 s | **15 experiments**; the replay demo |
| `fixtures/accuracy/anchored.jsonl` | 1.39 MB | 10 s | F32 shoulder validated, knee refused |
| `fixtures/sessions/exposure_session.jsonl` | 0.82 MB | 12 s | F5 exposure and frame rate |
| `fixtures/sessions/abduction.jsonl` | 7.15 MB | 127 s | F39 motion sensitivity of the bone-length veto |

These are **coordinates, not imagery**: 33 body landmarks per frame in image and world
space, plus per-point confidence. They disclose body proportions and movement patterns.
They do not contain a face, a photograph, or anything that renders as a picture of a person.

One non-personal file is also committed on purpose: `fixtures/reference/protractor_A4.png`,
the printed validation reference.

## What is not here

Nine video and still files, 9.2 MB — four session recordings, four exposure-test
recordings, one raw frame. `.gitignore` caught them from the first commit, so they were
**never published**, and on **2026-10-02** they were **deleted from disk** as well. Because
they were never committed, deleting them removes them completely; no history rewrite was
needed or performed.

**What that cost, stated rather than glossed:** F16 and F24 compared pose models by
replaying *identical frames* through `lite`, `full` and `heavy`. A landmark recording
cannot stand in — its landmarks are already one model's output. Those two findings are now
**documented but no longer reproducible**, and anyone wanting to re-derive them must record
their own video. That was accepted knowingly: the models were chosen and the finding
written up, so the data had done its work, and it was the only imagery of the subject
anywhere in the project.

The nine video-dependent tests now **skip** rather than fail, each naming its reason
(`recorded video is gitignored; run a session first`). The suite reports 166 passed and 9
skipped, against 175 passed before — the arithmetic matches, so no failure is hiding behind
a skip.

Verified in history, not only on disk:

```bash
git log --all --pretty=format: --name-only --diff-filter=A | sort -u | grep -iE "\.(mp4|png|jpg|jsonl)$"
```

That returns exactly the `.jsonl` files and the protractor sheet, and no `.mp4` or `.jpg`
has ever been added on any branch. The protractor is the printed validation reference, not
imagery of anyone, and is deliberately kept. The paths it prints are the **pre-restructure**
ones (`biomech/fixtures/...`) rather than today's `biomech/research/fixtures/...`, which is
worth knowing before writing any filter against them.

---

## Why keep the landmark files

Removing them was the alternative, and it costs more than it buys.

**They are the evidence.** `session.jsonl` alone is read by fifteen experiment scripts.
Deleting it does not make the findings wrong, it makes them unverifiable — a reader would
have to take F27's filter comparison and F28's accuracy numbers on trust. The whole point of
keeping `research/` in the repository is that a reviewer can re-run the thing that produced
a claim.

**They are what makes the application runnable without a webcam.** `--source landmarks`
replays a recorded session deterministically, which is how anyone without this exact camera
can see the system work:

```bash
biomech --source landmarks --path biomech/research/fixtures/sessions/session.jsonl
```

**Removal would not have been `git rm`.** These files are in history. Actually removing them
needs `git filter-repo` or BFG, a force-push that rewrites public history, and it breaks
every existing clone — including a reviewer's, mid-review.

**And repository visibility is the cheaper control.** Making this repository private, or
deleting it, removes the exposure in one step without rewriting anything, and it covers the
whole repository rather than a file list someone has to keep correct.

## The trade-off, stated plainly

Keeping these files means body-coordinate data for one consenting person — the author — is
publicly crawlable for as long as this repository is public. That is the cost. It was
accepted because the subject is the author, the data is coordinates rather than imagery, and
reproducibility is the point of the repository.

**This is not a recommendation for anyone else's data.** A recording of a third party does
not belong in a public repository, and the `--record-video` flag is off by default and logs
a warning precisely because video of a subject is a different category of thing from
coordinates.

## Planned action

**Revisit repository visibility after the interview review.** Making it private at that
point removes the exposure at no cost to the work, since reproducibility only needs to hold
while someone is reviewing it.

If the files ever do need to come out of history, the inventory command above is the first
step, and the paths to target are the pre-restructure ones.
