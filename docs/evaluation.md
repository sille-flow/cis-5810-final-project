# Minimal midpoint data entry

The pilot uses seven videos: one calibration video and six evaluation videos.
These CSVs are already populated and versioned; the following rules apply when
reviewing or extending the labels. The recorded criterion is hip crease reaching
knee-top height.
For a counting-only run, enter six rep totals and three reference timestamps.
The current dataset also includes acceptable-depth totals for all six clips.
You do not need timestamps for each evaluation rep.

Open [the focused video player](../data/review/pilot.html) and the two CSVs below.
MOV playback depends on browser codecs; use a local player if it does not play.

## 1. Enter six counts

In [clip_counts.csv](../data/annotations/clip_counts.csv), fill `human_attempts`
for the six dev rows: clips 2, 4, 5, 17, 18 and 26. The reference row for clip 11
can keep both count fields blank. Metadata, paths and provisional source groups
are already filled. Participant identities are unknown, so these groups may include the same people. Review the source grouping and correct it if necessary.

Count every complete standing-down-standing cycle, including shallow ones.
Exclude a partial cycle already underway when the clip starts or unfinished when
it ends. Clip 17 has a partial opening movement to check carefully. Use 0 only
when you actually observed zero complete cycles. A blank is unfinished input.

`human_acceptable_depth_reps` is optional. Leave it blank during the first pass.
If you use it, first agree on a visible depth criterion and count only completed
cycles meeting that criterion. Leave the total blank when any cycle's depth is
ambiguous; a front view is not sufficient to assume acceptable depth. This
supports depth-count error, not per-repetition classification accuracy.

Use notes only for issues worth explaining: partial boundary cycles, occlusion,
uncertain depth or source-group corrections. To omit a video, set split=exclude
and explain why. Keep challenging usable footage rather than selecting by model
performance. Do not put a source group in more than one split.

## 2. Calibrate one acceptable reference squat

In [reference_reps.csv](../data/annotations/reference_reps.csv), enter only:

| Field | Meaning |
| --- | --- |
| start_s | Standing just before descent begins |
| bottom_s | First deepest frame |
| end_s | Return to standing |

Times are decimal seconds from the exact file. Review clip 11 and choose one
complete squat that visibly meets your agreed depth criterion. Each row in this
file explicitly declares an acceptable calibration squat. Do not use it if depth
cannot be judged; choose another suitable source group for the reference instead.
You may add a second reference squat, but it is not required. This is the only
video requiring repetition timestamps. Reference total counts are unnecessary.

## 3. Validate and run

Save both CSVs, then from the repository root:

```sh
.venv/bin/python scripts/evaluation/run_clip_counts.py --include-hybrid --check
.venv/bin/python scripts/evaluation/run_clip_counts.py --include-hybrid
```

The check lists missing inputs and does not extract video or write results.
Running verifies reference geometry and reuses existing pose caches. Results go
to data/results/post_midpoint. If calibration fails, check the reference timestamps and missing pose features.

This is a preliminary development-set study, not a final held-out benchmark.
Keep remaining source groups available for later evaluation. The old all-video,
per-repetition workflow is preserved under data/annotations/legacy and in
[legacy/event_labeling.md](legacy/event_labeling.md); it is not required for this pilot.
