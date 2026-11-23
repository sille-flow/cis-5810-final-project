# Real-Time Gym Form Correction

CIS 5810 final project, Group 53.
Ellis Charles (chaelli@engineering.upenn.edu) and Kevin Xue
(kevinxue@engineering.upenn.edu).

We use MediaPipe pose tracking to count squats and give depth feedback. The
current version counts reps from 2D knee angles and judges depth using 3D
features calibrated against a reference squat. If depth tracking is incomplete,
it can still count the rep and mark its depth as unclear.

## Current report and results

- [Compiled LaTeX report](data/reports/midpoint/Group_53_Midpoint_Report.pdf)
- [Report source](docs/report/midpoint.tex) and [bibliography](docs/report/references.bib)
- [Source and results package](data/reports/midpoint/Group_53_Midpoint_Source_and_Evidence.zip)
- [Latest per-video comparison](data/results/post_midpoint/comparison.csv)

| Configuration | Count MAE | Exact clips | Depth-count MAE | Depth-unclear events |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 2.17 | 1/6 | 1.50 | 0 |
| Smoothing | 1.00 | 1/6 | 1.83 | 0 |
| Reference-relative | 1.00 | 2/6 | 2.33 | 0 |
| Temporal validation | 1.00 | 2/6 | 2.33 | 0 |
| Hybrid 2D + depth | **0.67** | 2/6 | 2.00 | 2 |

These results use six development clips with 17 completed reps, including 15
at acceptable depth. A separate squat provides calibration. The hybrid was
developed using these clips; we have not evaluated on a separate test set.
See the report for per-clip results and limitations.

## Run

Install Python dependencies from `requirements.txt` into `.venv`.
`requirements-tested.txt` lists the versions used for the reported run.
Building the PDF also requires `latexmk` and a TeX installation.

```sh
make check       # regression tests
make evaluate    # original four methods plus hybrid; reuses pose caches
make report      # evaluate, figures, LaTeX PDF, and source/evidence ZIP
make demo        # camera demo with optional calibrated depth
```

After the first `make report`, you can also compile `docs/report/midpoint.tex`
directly in your editor. It resolves generated tables and figures under
`data/reports/midpoint/`. Run `make report` again when results change or to
refresh the submission PDF and evidence ZIP after editing the narrative.

Prerequisites are the source videos under data/videos/squat and the pose model
under data/models. To obtain the official model if absent:

```sh
.venv/bin/python -c 'from src.pose.stream import ensure_model; ensure_model()'
```

For a prerecorded demo without a camera/window:

```sh
.venv/bin/python scripts/demo/demo_squat.py \
  --source data/videos/squat/squat_26.mp4 \
  --reference data/results/post_midpoint/reference.json --headless
```

The default demo method is hybrid_2d. Use --method temporal_3d to replay the
original temporal method. Reference calibration is optional for hybrid counting;
without it, depth labels remain unclear. See [experiment commands](docs/experiments.md).

## Human inputs

The completed [clip totals](data/annotations/clip_counts.csv) and
[reference timestamps](data/annotations/reference_reps.csv) are the only active
annotation sheets. They were transcribed from the team's Numbers files, with
[provenance](data/annotations/annotation_provenance.json). The depth criterion is
hip crease reaching knee-top height. Full labeling instructions are in
[docs/evaluation.md](docs/evaluation.md).

Dataset: Seyed Vala Khorasani, *Gym Members Exercise Dataset*, Kaggle, 2024,
[DOI 10.34740/KAGGLE/DSV/9559728](https://www.kaggle.com/dsv/9559728).
The local archive contains 29 squat clips. Source credits and the AI-assistance
disclosure are in the report's final subsection.

## Repository layout

| Location | Purpose |
| --- | --- |
| src/pose/ | Pose extraction, geometry and caching |
| src/squat/ | Original methods and the hybrid counter |
| src/evaluation/ | Input validation and metrics |
| src/visualization/ | Video overlays |
| scripts/evaluation/ | Clip-count experiment |
| scripts/report/ | Figures, LaTeX tables and report build |
| scripts/demo/ | Webcam and video demo |
| scripts/data/ | Pose caching and video inventory |
| configs/ | Algorithm parameters |
| docs/report/ | LaTeX report source and bibliography |
| data/ | Inputs, local raw data, caches, results and compiled deliverables |
| tests/ | Regression and integration tests |

Earlier experiments are listed in [docs/legacy](docs/legacy/README.md).
See [data/README.md](data/README.md) for data and output locations.
