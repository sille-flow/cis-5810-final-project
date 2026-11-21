# Evaluation and recording workflow

## Local MM-Fit audit

Run `python scripts/data/audit_mmfit.py --root mm-fit --out output/mmfit`.
The inspected download has 21 sessions, 616 labeled sets, 64 squat sets,
65 push-up sets, and 65 sit-up sets. It has no RGB video files. Three sets
have incomplete 3D pose frame coverage (using inclusive label endpoints).
See output/mmfit/sets.csv for individual coverage flags.

Arrays are not MediaPipe landmarks: observed 3D shape is (3, frames, 18),
with frame ID in column 0 and 17 joint columns. 2D shape is (2, frames, 19).
Frame IDs can start well above zero. Slice labels by frame IDs, never by raw
array offsets. Label rows contain start frame, end frame, repetition count,
and exercise. They do not contain per-repetition boundaries or form ratings.

Useful now: dataset inventory and set-level counting/recognition experiments.
Required before running our matcher on skeletons: verify the authors' joint
ordering, coordinate axes, frame rate and preprocessing; implement and visually
validate a separate adapter; build reference templates from that same backend.
Do not reinterpret these arrays as MediaPipe's 33 landmarks or mix their
features with existing MediaPipe templates without validation.

Use subject-disjoint splits after obtaining the session-to-participant mapping.
Until then, a session-disjoint split is only exploratory and does not establish
unseen-person performance. Exclude incomplete sets from full-set counting metrics.
Unlabeled gaps may contain transitions; do not automatically label them walking.

RGB downloads are linked separately at https://mmfit.github.io/ (Zenodo).
Author starter code: https://github.com/KDMStromback/mm-fit .
Obtain matching RGB sessions for end-to-end MediaPipe testing. Our own labeled
footage remains necessary for shallow-depth feedback evaluation.

## Record movements, not still poses

Start with bodyweight squats. For EACH participant record:

- Reference: one separate 5-repetition normal-depth clip at each of side,
  front, and 45-degree oblique views (3 clips).
- Evaluation: at each view, record 5 normal repetitions, 5 shallow attempts,
  and 5 mixed-tempo repetitions as separate clips (9 clips).
- Negatives: one 15-second standing clip and one 15-second walking/turning clip.
- Optional robustness: one clip with a brief occlusion or exit/re-entry.

Use a fixed phone/webcam, landscape 1080p/30fps if available, no zoom or cuts,
level at approximately hip height, with the whole body and floor visible.
Keep lighting steady and leave space around head and feet. Start and finish
with 2-3 seconds standing still. Keep source files and native timestamps.
Use comfortable unweighted movement. Shallow means a deliberate shorter range;
there is no need to simulate risky knee collapse or other injury-related faults.

Name files p01_side_normal_01.mp4, p01_front_shallow_01.mp4, etc.
Store in data/recordings/ (ignored by Git). Record participant/session IDs.
For frontal depth labels, simultaneously record a side-view phone if possible;
otherwise mark ambiguous depth as unclear. Keep both views of the SAME action
in the same split. Reference videos must not be evaluation videos.

## Labeling

Fill data/annotations/clips.csv, one row per video. Paths are relative to that
CSV, for example ../recordings/p01_side_normal_01.mp4. split is reference,
dev, or test. attempts counts all completed down/up cycles, including shallow
ones. acceptable_depth_reps counts only cycles meeting the agreed depth rule.
Leave acceptable_depth_reps empty if depth cannot be judged reliably. Negatives
have zero attempts and zero depth reps. Use unique clip IDs.

Fill repetitions.csv with one row per completed attempt, times in seconds:
start = departure from standing; bottom = deepest point; end = return to
standing. depth_label is acceptable, shallow, or unclear. visibility is clear,
partial, or occluded. Record incomplete attempts in notes and exclude them
from completed-attempt ground truth.

For a reproducible project label, agree on a visible side-view depth landmark
before annotation (for example hip crease reaching knee-top height), annotate
according to that rule, and mark borderline cases unclear. This is a project
depth criterion, not a general assessment of safe/correct form. Both teammates
should independently check a subset and resolve disagreements without looking
at predictions. Never label every MM-Fit repetition as good form by assumption.

## Evaluation commands

Use the four-method workflow in [experiments.md](experiments.md) for the project
comparison. The recording checklist above is the same 14-clip-per-person pilot
summarized in the README. Reference clips calibrate anchors; dev clips support
tuning; test clips supply held-out results after settings are frozen.

## Earlier phase-template evaluator

`scripts/experimental/evaluate_videos.py` is retained for the older PCA phase
matcher. It compares the baseline with a `.npz` template, not with the three
current experiment iterations. Build a template using reference clips only via
`scripts/experimental/build_template.py` before using it.
The older live demo (`scripts/experimental/run_live.py`) can bootstrap a synthetic
template; do not present that output as calibrated study evidence.
Use `scripts/demo/demo_squat.py` with the experiment's `reference.json` for the
project demo. MM-Fit conversion and multi-exercise recognition remain outside
the evaluated squat core.
