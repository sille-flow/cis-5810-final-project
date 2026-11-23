# Squat experiment workflow

## Implemented methods

All four methods produce completed movement events with start/bottom/end times
and an acceptable/shallow label (the temporal method can abstain as unclear).

1. `baseline`: original fixed 2D knee-angle state machine (150/95/160 degrees).
2. `confidence_2d`: same thresholds, causal time-based smoothing, cancellation
   of pending events after tracking gaps.
3. `reference_3d`: reference-relative hip height and bilateral knee flexion,
   smoothed over time; separate thresholds for movement and acceptable depth.
4. `temporal_3d`: adds minimum duration, ascent detection, observed depth dwell,
   and uncertainty based on missing-pose coverage.

The experimental 3D methods use fitted standing/bottom reference anchors,
not the older PCA phase matcher. This deliberately isolates depth and temporal
validation in the first experiment. The original phase matcher remains available
in run_live.py. Do not describe the new experiment as a DTW or phase-alignment
experiment. Thresholds in configs/experiment.json are starting hypotheses, not
validated biomechanical standards. `direction_epsilon` is depth change per second.

## Setup

Use Python with numpy, mediapipe, OpenCV and matplotlib installed (requirements.txt).
On this machine a .venv is used. Commands below assume its Python:

```sh
.venv/bin/python -m unittest discover -s tests -v
```

## Labels and splits

Edit the unified annotation CSV and export `data/annotations/clips.csv` and
`repetitions.csv` as described in evaluation.md. Do not maintain labels twice.
Every completed attempt must have a repetition row, including reference attempts.
For a clip with any unclear depth labels, leave acceptable_depth_reps blank.
Negatives have attempts=0, acceptable_depth_reps=0, and no repetition rows.
Reference clips may include shallow attempts; only acceptable ones fit anchors.

Add a source_group for related edits, simultaneous cameras and common source
recordings. A source group cannot cross splits. Exact duplicate video content is
rejected even under different filenames. All source files must exist.

Default evaluation requires participant-disjoint clips. With just two people
and each contributing reference/test footage, use --allow-participant-overlap
for an explicitly labeled within-person pilot. This does NOT waive source-group
or duplicate checks. For unseen-person evaluation, reserve a person entirely
from reference and development. Do not tune parameters against test results.

## Cache and quality inspection

```sh
.venv/bin/python scripts/data/cache_poses.py '/absolute/path/squat.mp4'
```

data/review/pose_quality.json records usable 2D/depth fractions and per-feature
coverage. Cache files preserve every decoded frame, timestamps, missing values,
2D and 3D landmarks/visibility, feature vectors, camera-view estimates and
extraction times. Keys hash video/model/code and dependency versions. Caches
are automatically reused unless inputs change. Videos are never modified.
Cropping is not implemented: prepare a single-view file first and keep its
source group consistent. Do not feed split screens into the study unchanged.

## Run the experiment

```sh
.venv/bin/python scripts/experimental/run_event_experiment.py \
  --manifest data/annotations/clips.csv \
  --repetitions data/annotations/repetitions.csv \
  --split dev --out data/results/dev --allow-participant-overlap
```

Outputs:
- reference.json: fitted anchors and reference-video hashes.
- results.json: config, code/label hashes, participant overlap, all metrics.
- clips.csv: per-clip and per-method results, runtime and coverage.
- report.md and counting_comparison.png: comparison table and chart.
- clip_*_events.json / clip_*_frames.csv: auditable predictions and signals.
- trace_*.png: trajectories with human repetition annotations shaded.

Reference anchors are fitted ONLY on reference clips, using the annotated
standing/bottom neighborhoods. Insufficient motion or poor reference geometry
raises an error instead of silently constructing an unusable template.

Metrics include count MAE/exact-count fraction, accepted-depth count MAE,
one-to-one temporal event precision/recall/F1, shallow-detection precision/recall/F1,
form decision coverage, and false counts on zero-attempt clips. Matching uses
an order-preserving maximum-match assignment with configurable interval IoU.
Missed shallow attempts count as false negatives. Unknown ground-truth depth
is excluded from form classification, but its attempt still counts.

Algorithm and extraction runtimes are reported separately; they exclude
video decoding/rendering and are not end-to-end webcam latency. Synthetic
regression checks are not real-world accuracy measurements.

## Demo the same temporal method

After the experiment produces a reference.json:

```sh
.venv/bin/python scripts/demo/demo_squat.py --reference data/results/dev/reference.json
.venv/bin/python scripts/demo/demo_squat.py --reference data/results/dev/reference.json --source '/absolute/path/squat.mp4'
```

The demo displays completed attempts, depth-accepted repetitions, tracking state
and shallow/uncertain feedback. Use the same --config as the experiment if you
change parameters. Press q to close. No camera is opened by the experiment runner.

## What still depends on human input

The missing human annotations prevent producing an accuracy report
until real clips are labeled. This workflow uses manual annotations and reference calibration; it does not
train a neural network. MM-Fit skeleton conversion and general
exercise classification remain outside this focused squat experiment.
