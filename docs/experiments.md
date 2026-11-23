# Evaluation and report workflow

To rerun the current experiment and build the report:

```sh
make check
make report
```

make report runs the five-method comparison, prepares figures, generates LaTeX
tables, compiles the PDF, and creates the source and results ZIP. It requires
the local videos, pose model, and `latexmk`.

## Individual commands

```sh
.venv/bin/python scripts/evaluation/run_clip_counts.py --include-hybrid --check
.venv/bin/python scripts/evaluation/run_clip_counts.py --include-hybrid
.venv/bin/python scripts/report/prepare_figures.py
.venv/bin/python scripts/report/build_midpoint_report.py
```

The runner reads data/annotations/clip_counts.csv, reference_reps.csv and
configs/experiment.json. It reuses pose caches. `--include-hybrid` adds the hybrid
method to the original four methods. It writes to
data/results/post_midpoint and refuses hybrid output inside the historical
midpoint directory. Without the flag, only the original four configurations run.

## Outputs

- data/results/post_midpoint/comparison.csv: human and all five predicted totals.
- per_clip.csv and failure_review.csv: errors, coverage, uncertainty and timing.
- results.json and reference.json: parameters, hashes, metrics and calibration.
- clip_*_frames.csv and clip_*_events.json: predictions for diagnosis, not labels.
- diagnostics.json and experiment_log.csv: report analysis and iteration record.
- data/reports/midpoint/: current PDF, portable LaTeX, bibliography, figures,
  generated tables and source/evidence ZIP.

The report builder checks input hashes, clip counts, and metrics against the
values discussed in the report. If clips or results change, update
`docs/report/midpoint.tex` before building again. Tables and plots come from the saved measurements; do not edit
them by hand. The first report is preserved under data/reports/archive.

## Reading the results

The [main README](../README.md) summarizes the metrics, and the report discusses
individual clips and errors. The CSV labels contain per-video totals, so they
support count MAE but not per-rep F1. Processing times exclude video decoding
and display.

## Demo

```sh
make demo
.venv/bin/python scripts/demo/demo_squat.py --source data/videos/squat/squat_26.mp4 --reference data/results/post_midpoint/reference.json --headless
```

The hybrid is the default. It requires standing before a rep, a 0.6--15s cycle,
and sufficient 2D tracking coverage. Reference calibration is optional for
counting; without it, depth is marked unclear.
--headless requires --source and opens no camera/window. Use --method temporal_3d
with reference.json for the original midpoint demo. The demo and evaluation use the same tracker.
To add reference cycles, edit `reference_reps.csv` and keep calibration sources
separate from evaluation clips.
