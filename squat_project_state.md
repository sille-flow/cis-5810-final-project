# Squat Form Correction Project --- Current State

**Last updated:** November 7, 2026
**Course context:** CIS 5810 computer vision project
**Current artifacts:** `src/`, `scripts/`, `pipeline.ipynb`, `data/`

## 1. Project goal

Build a real-time computer-vision coach that compares a live exerciser to
reference workout videos, counts repetitions, and flags joints that leave the
reference pose. The first exercise is the squat, but the matcher is not a
squat-specific angle state machine.

## 2. Approach

The proposal asked for reference-based analysis instead of only rule-based
squat thresholds. Classic **batch DTW is the wrong tool for the live loop**:
it wants two complete sequences, so feedback would arrive after the rep (or
after a long window), not while the user is still moving.

This project uses **online phase following** plus a **template-relative rep
counter**:

```text
Reference videos (offline)
    -> 3D joint features per frame (yaw invariant)
    -> PCA motion axis + cyclic phase template (mean/std per phase)
Live frame (online)
    -> same features
    -> search only nearby template phases (pose cost + direction cue)
    -> residual vs reference pose at that phase -> per-joint feedback
    -> separate depth score -> hysteresis rep counter
```

The live path never stores the rest of the user's repetition. Cost per frame
is `O(search_radius x n_features)`, independent of video length.

Batch DTW (or full-sequence evaluation) can still be used later as an
**offline metric** on saved clips. It is not required for coaching.

## 3. Implementation

### Features (`src/features.py`)

Each usable pose becomes seven features, all computed from MediaPipe **world
landmarks** (metric 3D, hip-centered): left/right knee angle, left/right hip
angle, torso incline (0 deg = upright, measured against world vertical), hip
height (vertical hip-to-ankle distance divided by torso length), and knee
width (knee separation divided by hip width, a knee-cave signal).

These are chosen to be **invariant to rotation about the vertical axis**, so a
template recorded from the side can follow a squat done facing the camera.
Image-space (2D) angles are no longer used: they collapse head-on, and
normalized pixel coordinates depend on the video's aspect ratio. There are no
95 / 150 / 160 degree cutoffs.

Camera view (`front` / `side` / `oblique`) is still estimated, from the yaw of
the 3D shoulder and hip line (front <= 35 deg, side >= 60 deg). It is used only
to re-weight channels (`VIEW_WEIGHTS`): head-on, depth-based angles are noisier,
so hip height and knee width are trusted more; side-on, knee width is nearly
ignored because lateral spread lies along the noisy depth axis.

### Phase template (`src/template.py`)

`build_phase_template` concatenates reference sequences, takes the first PCA
axis as a data-driven "how far through the movement" coordinate, splits
cycles at standing peaks of that axis, time-normalizes each cycle to `n_phases`
bins, and stores per-bin mean and std (with per-feature std floors).
`MultiViewTemplates` can hold separate front / side / oblique templates.
`build_template.py` **pools all clips into one template by default** (since the
features are yaw invariant); `--per-view` keeps separate templates.

### Live matcher (`src/realtime_matcher.py`)

`RealtimeReferenceMatcher` keeps a phase index. On each frame it:

1. Exponentially smooths the live feature vector.
2. Scores a small window of phases around the current index with a weighted,
   std-normalized squared residual.
3. Adds a **direction cue**: a squat looks nearly the same going down and
   coming up, so the live progress velocity along the PCA axis is compared
   with the template's slope at each candidate phase. A mild forward bias
   also keeps the phase from jittering backward on ambiguous plateaus.
4. Steps to the best neighbor and reports per-joint residuals / z-scores.
5. Turns the largest weighted z-score into a feedback string
   (`left knee: higher than reference`).

**Rep counting is separate from the phase tracker.** A `depth` score
(0 = standing, 1 = as deep as the reference bottom) is computed from hip drop
and the *less-bent* of the two knees, scaled by the template's own standing and
bottom poses. A rep counts only after: standing (depth <= 0.25), then at least
`bottom_hold_frames` (3) consecutive frames at depth >= 0.55, then a return to
standing. This hysteresis prevents double counts, and requiring both knees to
bend keeps walking from counting. Tunable constructor arguments:
`bottom_depth`, `stand_depth`, `bottom_hold_frames`.

### Pose I/O (`src/pose_stream.py`)

MediaPipe Pose Landmarker in `VIDEO` mode, with timestamps that always
increase (fixes the webcam `CAP_PROP_POS_MSEC == 0` issue). The lite model is
downloaded to `models/pose_landmarker_lite.task` on first use. The harmless
native TFLite/glog warnings are muted by silencing stderr during model creation
and the first few detections.

### Overlay (`src/overlay.py`)

Reps, view, depth %, phase, feedback, and the three largest z-scores are drawn
on a dark semi-transparent panel with outlined text so they stay readable on
any background.

### How to use reference videos

1. Put good-form clips in `data/references/squat/good/` (side, front, or
   anything in between).
2. `python scripts/build_template.py --videos data/references/squat/good --out data/templates/squat_good.npz`
   (prints the detected view per clip). **Rebuild whenever `features.py`
   changes**; templates built with the older 2D feature definitions are not
   compatible.
3. `python scripts/run_live.py --template data/templates/squat_good.npz`

Public datasets (InfiniteForm/InfiniteRep, MM-Fit, UESTC RGB-D, Fit3D) can
feed the same extractor. Details are in `data/README.md`. Until real clips
exist, `build_template.py --synthetic` writes a bootstrap kinematic template
(`src/synthetic.py` generates rows directly in feature space).

The old `SquatTracker` angle machine is no longer the live path. It remains
useful only as a documented baseline from the proposal.

## 4. Current data and validation status

- **Reference data:** 4 clips so far (3 side, 1 front), all recorded by the team.
- **Validation so far:** mostly on a simulated 3D skeleton (not real MediaPipe
  output) plus live webcam testing. In simulation, with a side-only template,
  rep counts were exact at yaws of 0, 45 and 90 deg, at fast and normal
  tempo, with moderate depth noise; simulated walking produced no reps; partial
  squats did not count. These numbers are sanity checks, not an accuracy result.
  A proper evaluation on held-out real clips is still to do (see next steps).

## 5. Known limitations

- MediaPipe's depth estimate is noisier head-on, so 3D knee/hip angles are less
  reliable in front view; the view weights compensate only partly.
- Camera tilt biases the hip-height feature somewhat, because world "down" is
  not corrected for camera pitch.
- Very fast reps combined with heavy landmark noise can still drop reps in
  simulation.
- Depth is relative to the reference bottom. If reference clips are much deeper
  than a user's normal squat, depth percentages run low and `bottom_depth` may
  need lowering (and vice versa).
- Shallow squats are not flagged through large knee residuals: the phase tracker
  simply sits at an earlier phase and no rep is counted. Explicit depth feedback
  would need a separate check at the bottom of the rep.
- With only 4 clips, per-phase std is mostly its floor value, so z-score
  feedback is coarse. Synthetic templates are not coaching ground truth;
  replace them with labeled good-form video.
- The phase template assumes a cyclic exercise whose template starts near
  standing.
- Feedback strings name the worst feature, not a validated biomechanical fault.

## 6. Next steps

1. Record 6-10 good-form clips (side, front, and in between, varied depth and
   tempo) and rebuild the template.
2. Hold out a few user clips (good, shallow, and non-squat motion such as
   walking) and report rep accuracy and false-positive counts; tune
   `bottom_depth` / `bottom_hold_frames` on a development split only.
3. Add explicit depth feedback (bottom-of-rep knee angle / hip height versus
   reference) so shallow reps are flagged instead of silently uncounted.
4. Add a second exercise folder using the same matcher.
5. Optionally compute batch DTW only when writing the evaluation table, not
   in the webcam loop.
6. If public datasets are used, document license and whether quality labels
   exist before treating clips as "correct form."

## 7. Scope statement

The system is a real-time reference follower with approximate joint-level
feedback. It is not a validated trainer, and synthetic templates should not
be presented as biomechanical standards.