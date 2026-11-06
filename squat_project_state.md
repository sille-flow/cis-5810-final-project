# Squat Form Correction Project --- Current State

**Last updated:** October 11, 2026\
**Course context:** CIS 5810 computer vision project\
**Current artifact:** `pipeline.ipynb`

## 1. Project goal

Build a minimal computer-vision prototype that tracks a person's squat,
counts completed repetitions, and gives basic feedback about whether the
squat reached sufficient depth.

The current implementation combines MediaPipe Pose Landmarker with a
rule-based state tracker. It is a baseline, not a complete exercise-form
assessment system.

## 2. Current implementation

### Environment and libraries

The notebook imports:

-   OpenCV (`cv2`) for webcam capture, drawing, and the preview window.
-   MediaPipe Tasks API (`mediapipe`, `mediapipe.tasks.python`,
    `mediapipe.tasks.python.vision`) for pose estimation.
-   NumPy for vector calculations.
-   `math` for converting angles to degrees.

The first notebook cell currently runs `%pip install opencv`; the
commonly used PyPI package for these imports is `opencv-python`.
MediaPipe is installed with `%pip install mediapipe`.

### Pose model and camera

The code loads the local model asset `pose_landmarker_lite.task` and
creates a `PoseLandmarker` in `VIDEO` running mode. Detection,
pose-presence, and tracking confidence thresholds are each set to `0.5`.
OpenCV opens webcam index `0`.

Each frame is converted from BGR to RGB, wrapped as a MediaPipe
`mp.Image`, and passed to `landmarker.detect_for_video(...)`. The
notebook displays the camera feed in a window resized to 960 × 540;
pressing `q` exits the loop.

### Skeleton visualization

The notebook draws the 33 MediaPipe pose landmarks as green circles and
connects selected landmark pairs using the `CONNECTIONS` list. These
connections include the face, shoulders, arms, torso, hips, and legs.
The visualization is a drawing layer over the model output; it does not
change pose-estimation accuracy.

### Knee-angle calculation

`calculate_angle(a, b, c)` computes the 2D angle at point `b` from three
`(x, y)` coordinates. It guards against a near-zero vector norm and
clips the cosine to `[-1, 1]` before applying `acos`.

`get_knee_angle(landmarks, ids)` receives the hip, knee, and ankle
landmark indices. It returns no angle when the minimum visibility of
those three landmarks is below `0.5`; otherwise, it calculates the angle
using MediaPipe's normalized image coordinates.

Landmark indices used:

-   Left hip / knee / ankle: `23, 25, 27`
-   Right hip / knee / ankle: `24, 26, 28`

For each frame, the code computes both knee angles and selects the side
with the higher minimum landmark visibility. If both sides are
sufficiently visible, it still chooses the side with the higher
visibility score.

### Rule-based squat tracker

The `SquatTracker` class currently uses these defaults:

  ------------------------------------------------------------------------
  Parameter                                  Default Purpose
  --------------------- ---------------------------- ---------------------
  `deep_angle`                                   95° Minimum knee angle
                                                     required to mark a
                                                     rep as deep enough

  `start_angle`                                 150° Angle below which a
                                                     squat begins

  `stand_angle`                                 160° Angle above which the
                                                     rep is considered
                                                     complete
  ------------------------------------------------------------------------

The tracker stores:

-   `reps`: number of completed reps
-   `good_reps`: completed reps whose minimum observed knee angle
    reached the depth threshold
-   `state`: `"standing"` or `"squatting"`
-   `min_angle`: smallest knee angle observed during the current rep

State logic:

1.  While standing, an angle below `start_angle` begins a squat and
    initializes `min_angle`.
2.  While squatting, the tracker records the smallest angle observed.
3.  When the angle rises above `stand_angle`, the tracker increments
    `reps`. If `min_angle <= deep_angle`, it also increments `good_reps`
    and reports `"Good depth!"`; otherwise, it reports `"Go deeper!"`.
4.  Between rep completions, feedback is `"Stand tall"`,
    `"Depth reached"`, or `"Keep going lower"`, depending on the state
    and minimum angle.
5.  If no usable knee angle is available, it returns `"Pose unclear"`
    without updating the state.

The overlay shows rep count, good-rep count, state, knee angle (or
unavailable), and feedback.

## 3. Current status

The notebook has the components for a working baseline:

-   Webcam input and real-time pose-landmark detection.
-   Pose skeleton drawing.
-   Knee-angle computation with a visibility threshold.
-   A small state machine for squat phases.
-   Rep counting and a depth-based good-rep counter.
-   On-screen feedback.

The supplied notebook code has not established quantitative accuracy or
demonstrated robust performance across camera angles, body types,
clothing, lighting, or occlusion. The 95° threshold is a tunable
heuristic rather than a universal clinical or coaching standard. Knee
angle is only a proxy for squat depth.

## 4. Known issues and limitations

### Timestamp handling

The current loop uses `cap.get(cv2.CAP_PROP_POS_MSEC)` and, when it
returns zero, substitutes OpenCV's elapsed tick count. For a live
webcam, this fallback is not a reliable per-frame timestamp because it
can represent elapsed time since the process began, while the video
timestamp can be zero or otherwise unsuitable. `VIDEO` mode expects
timestamps to increase monotonically. Use a monotonic elapsed-time
timestamp for webcam frames and ensure each timestamp is greater than
the previous one, or use MediaPipe's `LIVE_STREAM` mode with a callback.

### Missing-pose handling

The tracker is only updated inside
`if detection_result.pose_landmarks:`. If no pose is detected in a
frame, the overlay is not updated and the tracker does not explicitly
handle the missing frame. Consider showing a persistent `"Pose unclear"`
message and avoiding state transitions based on unreliable landmarks.

### Angle and depth limitations

-   The angle is computed from 2D normalized image coordinates.
    Perspective and camera placement affect it.
-   A side-view camera is a good initial setup because the hip, knee,
    and ankle are easier to interpret in one plane.
-   The code chooses the more visible side, but visibility alone does
    not guarantee a geometrically reliable angle.
-   The current `95°` cutoff should be tested on sample videos and
    adjusted based on the project's intended definition of sufficient
    depth.
-   The tracker does not currently evaluate torso lean, heel lift, knee
    alignment, balance, or other form criteria.

### Rep-state limitations

-   There is no temporal smoothing, hysteresis, or minimum-duration
    requirement, so noisy landmarks may cause unstable feedback or
    missed/false transitions.
-   A completed rep is counted when the knee angle returns above
    `stand_angle` after the tracker entered the squat state.
-   The tracker has no explicit reset method, per-rep log, or
    rep-quality history.
-   `min_angle` is reset when a new squat starts, but the displayed
    angle and feedback are only refreshed on frames where a pose is
    detected.

### Notebook cleanup

-   The code uses `%pip install opencv`; change this to
    `%pip install opencv-python` if the package installation fails.
-   The skeleton-drawing loop iterates over
    `detection_result.pose_landmarks`, but uses the `landmarks` variable
    set to the first detected pose. This is acceptable for the current
    single-person use case, but could be simplified or made consistent.
-   The `pixel_landmarks` list is populated with a coordinate for every
    landmark, even if a landmark is outside the frame or low-confidence.
    Connections are currently drawn without confidence filtering.
-   A commented-out older skeleton drawing implementation remains in the
    notebook and can be removed for clarity.
-   The camera object is opened before entering the landmarker context
    manager. For reliability, use `try/finally` so camera and OpenCV
    resources are released if an exception occurs.

## 5. Recommended next steps

Prioritize these steps before adding a trained classifier:

1.  **Run and test the baseline:** verify the model file path, camera
    capture, skeleton overlay, knee angle, and rep counter.
2.  **Fix webcam timestamps:** use monotonically increasing timestamps
    appropriate for `VIDEO` mode, or switch to `LIVE_STREAM`.
3.  **Test representative clips:** include standing, partial squats,
    deep squats, pauses at the bottom, and repetitions at different
    speeds.
4.  **Tune thresholds:** test `start_angle`, `deep_angle`, and
    `stand_angle` rather than treating defaults as final.
5.  **Add smoothing and transition stability:** smooth the angle across
    frames and require a stable transition before changing state.
6.  **Handle lost landmarks:** avoid counting a rep while key joints are
    unreliable and make uncertainty visible in the UI.
7.  **Record evaluation data:** log timestamps, selected side, knee
    angle, state, feedback, and rep events. Manually label a small set
    of clips to measure missed reps, false reps, and
    depth-classification errors.
8.  **Expand form feedback only after validating the baseline:**
    possible next features include torso inclination and additional
    joint-angle checks.
9.  **Consider reference-video matching later:** compare normalized
    joint-angle sequences with a reference using Dynamic Time Warping
    (DTW). A learned temporal classifier would require labelled
    sequences and is not necessary for this initial state-tracker
    baseline.

## 6. Current architecture

``` text
Webcam frame
    ↓
OpenCV BGR → RGB conversion
    ↓
MediaPipe Pose Landmarker (33 pose landmarks)
    ↓
Choose the more visible side
    ↓
Hip–knee–ankle angle
    ↓
SquatTracker state machine
    ├── state: standing / squatting
    ├── reps: completed repetitions
    └── good_reps: reps reaching the depth threshold
    ↓
Overlay feedback and skeleton
    ↓
OpenCV preview window
```

## 7. Scope statement

At the current stage, this project is a real-time squat repetition
counter with a heuristic knee-angle depth check. It is not yet a
validated general-purpose fitness coach, and its feedback should be
presented as approximate. The immediate goal is to make the state
transitions reliable and evaluate the baseline before adding more
complex pose matching or machine learning.
