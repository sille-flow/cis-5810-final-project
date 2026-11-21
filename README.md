# Real-Time Gym Form Correction

CIS 5810 final project — Group 53. **Status: November 20, 2026.**

- **Ellis Charles:** chaelli@engineering.upenn.edu
- **Kevin Xue:** kevinxue@engineering.upenn.edu

## Context and scope

Build a webcam-based exercise assistant for people exercising without a trainer.
The evaluated core is **squat repetition counting and shallow-depth detection**.
The research question is whether reference-relative pose analysis and temporal
tracking improve on fixed 2D knee-angle thresholds across viewpoints and speeds.

The proposal introduced MediaPipe pose estimation, a rule-based squat counter,
and possible reference-video comparison. TA feedback identified the main gap:
baseline and iteration descriptions lacked quantitative results and comparisons.
The current priority is therefore a reproducible baseline-plus-three-iterations
experiment, supported by labeled videos and a working demo.

The midterm PDF is due **November 20** and requires group information, title and
summary, goals/audience, pipeline, baseline results, at least three development
iterations with results, and a timeline with responsibilities. Submit through
**Gradescope**. Demo Day is **December 6, approximately 12–5 PM**. Confirm the
approved two-person arrangement because the supplied rubric specifies three.

## Pipeline and three development stages

```text
Videos → pretrained MediaPipe → cached landmarks/features/timestamps
       → baseline and three configurations → repetition events + depth labels
       → comparison against human annotations → tables, plots, failure review
```

**Baseline:** the original 2D knee-angle state machine, using 150° to start,
95° for depth acceptance, and 160° to return to standing.

| Stage | Implemented method | Question to evaluate |
| --- | --- | --- |
| 1 — Reliable tracking | `confidence_2d`: time-based smoothing and cancellation after tracking gaps | Does it reduce noisy or interrupted detections? |
| 2 — Reference-relative depth | `reference_3d`: hip height and bilateral knee flexion normalized to labeled standing/bottom reference anchors | Does it improve depth decisions across people and views? |
| 3 — Temporal attempt feedback | `temporal_3d`: duration, ascent, depth dwell, and coverage checks; separate attempts from accepted-depth reps | Does it reject spurious motion while identifying shallow attempts? |

These are implemented experiment configurations, **not demonstrated improvements
yet**. MediaPipe remains pretrained; fitting reference anchors and tuning
development-set thresholds do not require training a new neural network.

An older PCA phase-template matcher and automatic exercise selector also exist.
The new controlled experiment uses standing/bottom anchors, not DTW or the older
phase matcher. Multi-exercise recognition remains outside the evaluated core.

## Data available

| Source | Inspected contents | Use and limitations |
| --- | --- | --- |
| Local `mm-fit/` | 21 sessions, 616 labeled sets, including 64 squat sets; poses and wearable signals, no RGB videos | Useful for future counting experiments. Requires a verified skeleton adapter and participant mapping. No form labels; three sets have incomplete pose coverage. |
| Downloaded exercise archive | 652 videos across 22 folders; 29 squat clips totaling about five minutes | Real RGB input for MediaPipe. Counts/depth need annotation. Related source clips must stay in one split; split-screen/cropped footage needs curation. |
| Team recordings | Original notes mention four reference clips, absent from this checkout | Recover these and record controlled normal/shallow attempts plus standing/walking negatives. |

The archive was inspected at `/Users/kevin/Downloads/archive (1)/`; its raw videos
are external to this repository. Candidate notes are in
[archive_squat_review.csv](data/annotations/archive_squat_review.csv).
MM-Fit sources: [dataset page](https://mmfit.github.io/) and
[author code](https://github.com/KDMStromback/mm-fit). Additional downloads are
not required to begin the focused squat evaluation.

## Progress made

- Restored the original baseline as a reusable module.
- Implemented all three experiment stages and reference-only anchor fitting.
- Added pose caching with video/model/code hashes, timestamps, visibility,
  missing-feature preservation, and extraction timing.
- Added annotation validation, duplicate-video checks, source-group split checks,
  and explicit handling of participant overlap.
- Added event matching, count MAE/exact-count rate, depth metrics, repetition and
  shallow-detection precision/recall/F1, uncertainty coverage, and negative counts.
- Added per-clip predictions, comparison tables, plots, failure-review CSVs,
  and a demo using the same temporal counter.
- Fixed model lookup and missing geometry being interpreted as zero angles.
- **19 tests passed**, including synthetic behavior across frame rates, missing
  poses, label/split checks, event matching, and report generation.
- Verified real-video extraction and cache reuse. Usable bilateral depth features
  covered 100% of frames in `squat_11`, `squat_18`, and `squat_19`, 40% in
  `squat_17`, and 0% in `squat_10` under current visibility rules.

**The coverage figures are not accuracy results.** Synthetic tests and report
smoke checks are complete; labeled real-video comparisons remain pending.

## Quick start

Install dependencies in a Python environment using `requirements.txt`.
`requirements-tested.txt` records versions used for the local Python 3.13 macOS
smoke checks. An existing local `.venv` is available on the development machine.

```sh
# Regression tests
.venv/bin/python -m unittest discover -s tests -v

# Extract/cache one video and inspect pose quality
.venv/bin/python scripts/cache_poses.py '/absolute/path/squat.mp4'

# Once clip and repetition annotations are populated
.venv/bin/python scripts/run_experiment.py \
  --manifest data/annotations/clips.csv \
  --repetitions data/annotations/repetitions.csv \
  --split dev --out output/dev --allow-participant-overlap

# Demo after reference.json has been fitted
.venv/bin/python scripts/demo_squat.py --reference output/dev/reference.json
```

`--allow-participant-overlap` explicitly labels a within-person pilot; omit it
for participant-disjoint evaluation. It does not bypass source-group checks.
Tune on development data only and keep the test set untouched until settings
are fixed. Raw recordings, caches, and generated outputs are ignored by Git.

## Remaining work and ownership

**Kevin and Ellis:** recover/record footage, define the observable depth criterion,
label each completed attempt, review uncertain cases, and assign reference/dev/test
splits. Each person can start with two reference clips, four normal/shallow
evaluation clips across side/oblique views, and two standing/walking negatives.
Keep the whole body visible and start/end with 2–3 seconds standing still.

**Implementation/evaluation work:** run the four methods when labels are ready,
inspect errors, tune on development data, freeze settings, evaluate held-out
clips, and prepare report figures and demo evidence. Low bilateral visibility
is a known limitation to measure rather than conceal with filled-in angles.

**Joint deliverable:** a baseline-versus-three-stages comparison table, supported
by real labels, failure examples, and a repeatable demo. No claim of general
form correctness or validated coaching follows from depth classification alone.

## Key files

- [Experiment workflow](docs/experiments.md): commands, metrics, splits, outputs.
- [Recording and labeling guide](docs/evaluation.md): footage and annotation rules.
- [Clip manifest](data/annotations/clips.csv) and [repetition labels](data/annotations/repetitions.csv).
- [Experiment parameters](configs/experiment.json).
- `src/squat_experiment.py`: reference fitting and four methods.
- `src/pose_cache.py`: extraction/cache; `src/experiment_metrics.py`: evaluation.
- `scripts/run_experiment.py`: full experiment; `scripts/demo_squat.py`: demo.
- `scripts/audit_mmfit.py`: local dataset inventory and coverage audit.
