"""Generate LaTeX tables from measured results and compile the midpoint report.

Run after the five-method comparison and prepare_figures.py. Requires latexmk.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.pose.cache import load_cache


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(headers, rows, columns):
    return "\n".join(
        [
            r"\small",
            r"\begin{tabular}{@{}" + columns + r"@{}}",
            r"\toprule",
            " & ".join(headers) + r"\\",
            r"\midrule",
            *[" & ".join(str(x) for x in row) + r"\\" for row in rows],
            r"\bottomrule",
            r"\end{tabular}",
            "",
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()
    results = ROOT / "data/results/post_midpoint"
    out = ROOT / "data/reports/midpoint"
    report = json.loads((results / "results.json").read_text())
    if (
        digest(ROOT / "data/annotations/clip_counts.csv") != report["counts_sha256"]
        or digest(ROOT / "data/annotations/reference_reps.csv")
        != report["references_sha256"]
    ):
        raise ValueError(
            "Human inputs changed after evaluation. Rerun run_clip_counts.py first."
        )
    if json.loads((ROOT / "configs/experiment.json").read_text()) != report["config"]:
        raise ValueError(
            "Configuration changed after evaluation. Rerun the experiment first."
        )
    # Check that the generated tables still match the written results.
    expected = {
        "archive_squat_2": 4,
        "archive_squat_4": 4,
        "archive_squat_5": 4,
        "archive_squat_17": 1,
        "archive_squat_18": 1,
        "archive_squat_26": 3,
    }
    if {r["clip_id"]: r["human_count"] for r in report["comparison"]} != expected:
        raise ValueError(
            "Pilot cohort or counts changed: update the report narrative before building."
        )
    expected_metrics = {
        "baseline": (13 / 6, 1.5),
        "confidence_2d": (1.0, 11 / 6),
        "reference_3d": (1.0, 14 / 6),
        "temporal_3d": (1.0, 14 / 6),
        "hybrid_2d": (2 / 3, 2.0),
    }
    for variant, (count_mae, depth_mae) in expected_metrics.items():
        if (
            abs(report["summary"][variant]["attempt_mae"] - count_mae) > 1e-8
            or abs(report["summary"][variant]["depth_mae"] - depth_mae) > 1e-8
        ):
            raise ValueError(
                "Measured outcomes changed: update the iteration/failure narrative before building."
            )
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    names = {
        "baseline": "Baseline",
        "confidence_2d": "Smoothing",
        "reference_3d": "Reference-relative",
        "temporal_3d": "Temporal validation",
        "hybrid_2d": "Hybrid 2D + depth",
    }
    rows = []
    for variant, values in report["summary"].items():
        exact = round(values["exact_count_fraction"] * values["clips"])
        rows.append(
            [
                names[variant],
                f'{values["attempt_mae"]:.2f}',
                f"{exact}/6",
                f'{values["depth_mae"]:.2f}',
                values.get("uncertain_depth_predictions", 0),
            ]
        )
    (tables / "results.tex").write_text(
        table(
            ["Configuration", "Count MAE", "Exact clips", "Depth MAE", "Unclear"],
            rows,
            "lrrrr",
        )
    )
    summary = report["summary"]
    macros = {
        "BaseMAE": summary["baseline"]["attempt_mae"],
        "SmoothMAE": summary["confidence_2d"]["attempt_mae"],
        "RefMAE": summary["reference_3d"]["attempt_mae"],
        "BaseDepthMAE": summary["baseline"]["depth_mae"],
        "SmoothDepthMAE": summary["confidence_2d"]["depth_mae"],
        "RefDepthMAE": summary["reference_3d"]["depth_mae"],
        "HybridMAE": summary["hybrid_2d"]["attempt_mae"],
        "HybridDepthMAE": summary["hybrid_2d"]["depth_mae"],
    }
    text = "\n".join(
        "\\newcommand{\\" + name + "}{" + f"{value:.2f}" + "}"
        for name, value in macros.items()
    )
    reduction = (
        100
        * (summary["baseline"]["attempt_mae"] - summary["confidence_2d"]["attempt_mae"])
        / summary["baseline"]["attempt_mae"]
    )
    text += "\n\\newcommand{\\CountReduction}{" + f"{reduction:.1f}" + "}\n"
    timing = {
        "ExtractMin": min(r["extraction_median_ms"] for r in report["clips"]),
        "ExtractMax": max(r["extraction_median_ms"] for r in report["clips"]),
        "CounterMin": min(r["algorithm_median_ms"] for r in report["clips"]),
        "CounterMax": max(r["algorithm_median_ms"] for r in report["clips"]),
    }
    for name, value in timing.items():
        digits = 4 if name.startswith("Counter") else 2
        text += "\\newcommand{\\" + name + "}{" + f"{value:.{digits}f}" + "}\n"
    (tables / "metrics_macros.tex").write_text(text)
    with (ROOT / "data/annotations/clip_counts.csv").open(newline="") as stream:
        inputs = list(csv.DictReader(stream))
    quality = {
        Path(q["video"]).stem: q
        for q in json.loads((ROOT / "data/review/pose_quality.json").read_text())
    }
    rows = []
    for row in inputs:
        number = row["clip_id"].rsplit("_", 1)[1]
        q = quality["squat_" + number]
        role = (
            "Reference / " + row["view"]
            if row["split"] == "reference"
            else "Dev / " + row["view"]
        )
        rows.append(
            [
                number,
                role,
                row["human_attempts"],
                row["human_acceptable_depth_reps"],
                f'{q["duration_s"]:.2f}',
                f'{100*q["usable_depth_fraction"]:.1f}',
            ]
        )
    (tables / "dataset.tex").write_text(
        table(
            [
                "Clip",
                "Role / view",
                "Attempts",
                "Depth reps",
                "Seconds",
                "Coverage (\\%)",
            ],
            rows,
            "llrrrr",
        )
    )
    rows = [
        [
            r["clip_id"].rsplit("_", 1)[1],
            r["human_count"],
            r["baseline"],
            r["confidence_2d"],
            r["reference_3d"],
            r["temporal_3d"],
            r["hybrid_2d"],
        ]
        for r in report["comparison"]
    ]
    (tables / "counts.tex").write_text(
        table(
            ["Clip", "Human", "Baseline", "Smooth", "Reference", "Temporal", "Hybrid"],
            rows,
            "lrrrrrr",
        )
    )
    for name in ["midpoint.tex", "references.bib"]:
        target = "Group_53_Midpoint_Report.tex" if name == "midpoint.tex" else name
        shutil.copy2(ROOT / "docs/report" / name, out / target)
    (out / "README.txt").write_text(
        "Group 53 midpoint report, November 22, 2026\n\n"
        "Compile this portable LaTeX folder with:\n"
        "  latexmk -pdf -interaction=nonstopmode -halt-on-error Group_53_Midpoint_Report.tex\n\n"
        "Repository reproduction:\n"
        "  .venv/bin/python scripts/evaluation/run_clip_counts.py --include-hybrid\n"
        "  .venv/bin/python scripts/report/prepare_figures.py\n"
        "  .venv/bin/python scripts/report/build_midpoint_report.py\n\n"
        "Primary narrative source: docs/report/midpoint.tex. Tables are generated\n"
        "from data/results/post_midpoint/results.json. The hybrid was developed after\n"
        "reviewing these six clips and still needs testing on new videos.\n"
        "If clips or results change, update the report text before rebuilding.\n"
    )
    provenance = json.loads(
        (ROOT / "data/annotations/annotation_provenance.json").read_text()
    )
    cache_signatures = {
        cid: load_cache(path)["metadata"]["signature"]
        for cid, path in report["cache_paths"].items()
    }
    model_hash = digest(ROOT / "data/models/pose_landmarker_lite.task")
    if any(s["model_sha256"] != model_hash for s in cache_signatures.values()):
        raise ValueError("Current model differs from the evaluated cache signatures.")
    (results / "evidence_manifest.json").write_text(
        json.dumps(
            {
                "cache_signatures": cache_signatures,
                "model_sha256": model_hash,
                "code_sha256": report["code_sha256"],
                "human_inputs": {
                    name: digest(ROOT / "data/annotations" / name)
                    for name in ["clip_counts.csv", "reference_reps.csv"]
                },
            },
            indent=2,
        )
        + "\n"
    )
    (out / "build_manifest.json").write_text(
        json.dumps(
            dict(
                inputs_sha256={
                    name: digest(ROOT / name)
                    for name in [
                        "data/annotations/clip_counts.csv",
                        "data/annotations/reference_reps.csv",
                        "configs/experiment.json",
                    ]
                },
                results_sha256=digest(results / "results.json"),
                template_sha256=digest(ROOT / "docs/report/midpoint.tex"),
                annotation_provenance=provenance,
                model_sha256=model_hash,
                cache_signatures=cache_signatures,
            ),
            indent=2,
        )
        + "\n"
    )
    if args.no_compile:
        print("Generated LaTeX source and measured tables under", out)
        return
    compiler = shutil.which("latexmk")
    if not compiler:
        raise RuntimeError(
            "latexmk is required; LaTeX source and tables have been generated."
        )
    build = Path(tempfile.mkdtemp(prefix="cis5810-midpoint-latex-"))
    env = dict(
        os.environ,
        TEXMFVAR=str(build / "texmf-var"),
        TEXMFCONFIG=str(build / "texmf-config"),
    )
    command = [
        compiler,
        "-pdf",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-outdir=" + str(build),
        "Group_53_Midpoint_Report.tex",
    ]
    completed = subprocess.run(
        command, cwd=out, env=env, capture_output=True, text=True
    )
    (out / "compilation.log").write_text(completed.stdout + "\n" + completed.stderr)
    if completed.returncode:
        print((completed.stdout + "\n" + completed.stderr)[-7000:])
        raise RuntimeError("LaTeX compilation failed; inspect compilation.log")
    shutil.copy2(
        build / "Group_53_Midpoint_Report.pdf", out / "Group_53_Midpoint_Report.pdf"
    )
    package_evidence(out, results)
    print("Compiled", out / "Group_53_Midpoint_Report.pdf")


def package_evidence(out, results):
    """Bundle portable LaTeX and the matching code/inputs without raw media."""
    import zipfile

    package = out / "Group_53_Midpoint_Source_and_Evidence.zip"
    with zipfile.ZipFile(package, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "README.txt",
            "Group 53 revised midpoint report\n\nlatex/ compiles with latexmk -pdf "
            "Group_53_Midpoint_Report.tex.\nproject/ contains the evaluated code, human "
            "inputs, configuration and predictions. Raw videos, model binary and "
            "environments are omitted. Restore originals under project/data/videos/squat "
            "and the model under project/data/models before rerunning make report.\n"
            "The hybrid was developed using these six development clips and has not "
            "been evaluated on a separate test set.\n",
        )
        for path in out.rglob("*"):
            if path.is_file() and path.suffix in {
                ".pdf",
                ".tex",
                ".bib",
                ".png",
                ".txt",
                ".json",
            }:
                archive.write(path, "latex/" + str(path.relative_to(out)))
        for folder in [
            ROOT / "src",
            ROOT / "scripts",
            ROOT / "tests",
            ROOT / "configs",
            ROOT / "docs",
            ROOT / "data/annotations",
            results,
        ]:
            for path in folder.rglob("*"):
                if (
                    path.is_file()
                    and "__pycache__" not in path.parts
                    and "legacy" not in path.parts
                    and path.suffix in {".py", ".json", ".csv", ".md", ".tex", ".bib"}
                ):
                    archive.write(path, "project/" + str(path.relative_to(ROOT)))
        for name in [
            "Makefile",
            "README.md",
            "requirements.txt",
            "requirements-tested.txt",
            "data/README.md",
        ]:
            archive.write(ROOT / name, "project/" + name)
    with zipfile.ZipFile(package) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Evidence ZIP failed verification")


if __name__ == "__main__":
    main()
