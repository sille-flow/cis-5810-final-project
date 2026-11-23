PYTHON ?= .venv/bin/python

.PHONY: check evaluate figures report demo

check:
	$(PYTHON) -m unittest discover -s tests -v

evaluate:
	$(PYTHON) scripts/evaluation/run_clip_counts.py --include-hybrid

figures: evaluate
	$(PYTHON) scripts/report/prepare_figures.py

report: figures
	$(PYTHON) scripts/report/build_midpoint_report.py

demo:
	$(PYTHON) scripts/demo/demo_squat.py --reference data/results/post_midpoint/reference.json
