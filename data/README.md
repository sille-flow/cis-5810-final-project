# Project data

The active human inputs are complete and versioned:

- [clip_counts.csv](annotations/clip_counts.csv): reference clip 11 and six dev clips.
- [reference_reps.csv](annotations/reference_reps.csv): one acceptable reference cycle.
- [annotation_provenance.json](annotations/annotation_provenance.json): Numbers
  source names, dataset attribution and the agreed depth rule.

The depth criterion is hip crease reaching knee-top height.
See [labeling rules](../docs/evaluation.md) before adding annotations.

```text
data/
  annotations/                Annotation CSVs, versioned
    legacy/                   Optional detailed event-label inventory
  videos/squat/               29 original RGB videos
  datasets/mm-fit/             Downloaded skeleton/wearable data, not scored
  models/                     MediaPipe model
  templates/                  Experimental phase templates
  cache/poses/                Current pose caches
  review/                     Frame/video player, inventory and coverage
  results/midpoint/            Historical original four-method run
  results/post_midpoint/       Current five-method run and diagnostics
  reports/midpoint/            Current PDF, LaTeX source and results package
  reports/archive/             Previous report
  backups/annotations/        Previous input/export versions
```

The annotation CSVs and this guide are versioned. Videos, models, caches,
results, reports, and backups are ignored by Git. Run `make report` from the
repository root to regenerate the current results and report.

The [focused review player](review/pilot.html) shows the seven pilot videos;
[index.html](review/index.html) covers all 29. Its sampled-frame notes and
candidate timing windows are preliminary observations, not ground truth.
The archive contains 297.1 seconds of footage and no byte-identical files;
related edits still need source grouping. Split-screen clips 21--25 need
single-panel derivatives, 10/12 need shot-continuity review, and 27/28 omit
required lower-body geometry.

Optional inventory commands:

```sh
.venv/bin/python scripts/data/analyze_unlabeled.py
.venv/bin/python scripts/data/cache_poses.py /absolute/path/to/video.mp4
```

The all-video inventory uses annotations/legacy/squat_annotations.csv. Earlier
MM-Fit, event-label and phase-template workflows are documented in
[docs/legacy](../docs/legacy/README.md).
