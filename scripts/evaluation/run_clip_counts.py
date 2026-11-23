"""Compare midpoint squat methods, optionally adding the post-midpoint hybrid.

Fill data/annotations/clip_counts.csv and three timestamps in reference_reps.csv.
No timestamps or per-repetition depth labels are required for evaluation clips.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.clip_counts import (
    load_inputs,
    missing_inputs,
    summarize_counts,
    validate_config,
)
from src.evaluation.data import sha256
from src.pose.cache import extract_cached
from src.squat.tracking import VARIANTS, fit_reference, run_method
from src.squat.hybrid import HYBRID_VARIANT, run_hybrid


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(out, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4))
    variants = list(summary)
    errors = [summary[v]["attempt_mae"] for v in variants]
    bars = axis.bar(variants, errors)
    axis.bar_label(bars, fmt="%.2f", padding=3)
    axis.set_ylim(0, max(1, max(errors) * 1.2))
    axis.set(
        ylabel="Mean absolute count error per video", title="Clip-count comparison"
    )
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()
    figure.savefig(out / "counting_comparison.png", dpi=160)
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--counts", type=Path, default=ROOT / "data/annotations/clip_counts.csv"
    )
    parser.add_argument(
        "--references", type=Path, default=ROOT / "data/annotations/reference_reps.csv"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.json")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/cache/poses")
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--include-hybrid",
        action="store_true",
        help="Compare the post-midpoint 2D counter with all four original methods; defaults to data/results/post_midpoint",
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate inputs without extraction or writing results",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    variants = (*VARIANTS, HYBRID_VARIANT) if args.include_hybrid else VARIANTS
    if args.out is None:
        args.out = ROOT / (
            "data/results/post_midpoint"
            if args.include_hybrid
            else "data/results/midpoint"
        )
    midpoint = (ROOT / "data/results/midpoint").resolve()
    if args.include_hybrid and (
        args.out.resolve() == midpoint or midpoint in args.out.resolve().parents
    ):
        raise ValueError(
            "Save the hybrid experiment separately from the frozen midpoint results."
        )
    config = json.loads(args.config.read_text())
    validate_config(config)
    inputs, gaps = missing_inputs(args.counts, args.references, args.split)
    print(
        f'{sum(r["split"] == args.split for r in inputs)} evaluation count cells; '
        f'{sum(r["split"] == "reference" for r in inputs)} reference video(s).'
    )
    if gaps:
        raise ValueError("Human inputs still needed:\n  " + "\n  ".join(gaps))
    reference, selected, annotations = load_inputs(
        args.counts, args.references, args.split
    )
    if args.check:
        print(
            f"Inputs valid: {len(selected)} labeled evaluation clips, "
            f"{sum(map(len, annotations.values()))} acceptable reference squat(s). "
            "No files written; reference geometry will be checked when running."
        )
        return

    caches, cache_paths = {}, {}
    for clip in reference + selected:
        cid = clip["clip_id"]
        print("Extract/cache:", cid, flush=True)
        cache, path = extract_cached(clip["path"], args.cache, args.model)
        if (
            annotations.get(cid)
            and max(e["end_s"] for e in annotations[cid])
            > cache["times"][-1] + 1 / cache["metadata"]["fps"]
        ):
            raise ValueError(f"{cid}: reference annotation extends beyond the video")
        caches[cid], cache_paths[cid] = cache, str(path)
    anchors = fit_reference(
        {c["clip_id"]: caches[c["clip_id"]] for c in reference}, annotations
    )
    anchors["sources"] = [
        dict(
            clip_id=c["clip_id"],
            video_sha256=c["video_sha256"],
            annotated_cycles=annotations[c["clip_id"]],
        )
        for c in reference
    ]
    # Only produce results after complete labels and usable calibration exist.
    args.out.mkdir(parents=True, exist_ok=True)
    rows, comparison = [], []
    for index, clip in enumerate(selected):
        cid = clip["clip_id"]
        cache = caches[cid]
        counts = dict(
            clip_id=cid,
            human_count=clip["attempts"],
            human_acceptable_depth_count=clip["acceptable_depth_reps"],
        )
        for variant in variants:
            events, history, runtime = (
                run_hybrid(cache, config, anchors)
                if variant == HYBRID_VARIANT
                else run_method(cache, variant, config, anchors)
            )
            accepted = sum(e["label"] == "acceptable" for e in events)
            depth = clip["acceptable_depth_reps"]
            row = dict(
                clip_id=cid,
                variant=variant,
                participant=clip["participant"] or "unknown",
                view=clip["view"],
                source_group=clip["source_group"],
                true_attempts=clip["attempts"],
                predicted_attempts=len(events),
                attempt_error=abs(len(events) - clip["attempts"]),
                true_depth=depth,
                predicted_depth=accepted,
                depth_error=abs(accepted - depth) if depth is not None else None,
                uncertain_predictions=sum(e["label"] == "unclear" for e in events),
                usable_2d_fraction=float(np.isfinite(cache["angles"]).mean()),
                usable_depth_fraction=float(
                    np.isfinite(cache["features"][:, [5, 0, 1]]).all(axis=1).mean()
                ),
                algorithm_median_ms=float(np.median(runtime)),
                algorithm_p95_ms=float(np.percentile(runtime, 95)),
                extraction_median_ms=float(np.median(cache["extraction_ms"])),
                extraction_p95_ms=float(np.percentile(cache["extraction_ms"], 95)),
            )
            rows.append(row)
            counts[variant] = len(events)
            counts[variant + "_accepted_depth"] = accepted
            stem = f"clip_{index:03d}_{variant}"
            write_csv(args.out / (stem + "_frames.csv"), history)
            (args.out / (stem + "_events.json")).write_text(
                json.dumps(events, indent=2) + "\n"
            )
        comparison.append(counts)
    summary = {
        v: summarize_counts([r for r in rows if r["variant"] == v]) for v in variants
    }
    if args.include_hybrid:
        for variant, values in summary.items():
            selected_rows = [r for r in rows if r["variant"] == variant]
            total = sum(r["predicted_attempts"] for r in selected_rows)
            uncertain = sum(r["uncertain_predictions"] for r in selected_rows)
            values["uncertain_depth_predictions"] = uncertain
            values["predicted_depth_decision_fraction"] = (
                (total - uncertain) / total if total else None
            )
    by_view = {
        view: {
            v: summarize_counts(
                [r for r in rows if r["view"] == view and r["variant"] == v]
            )
            for v in variants
        }
        for view in sorted({r["view"] for r in rows})
    }
    ref_people = {c["participant"].strip().lower() for c in reference}
    eval_people = {c["participant"].strip().lower() for c in selected}
    placeholder_ids = {"", "unknown", "video"}
    unknown = bool((ref_people | eval_people) & placeholder_ids)
    overlap = sorted((ref_people & eval_people) - placeholder_ids)
    report = dict(
        evaluation="clip_counts_only",
        split=args.split,
        config=config,
        summary=summary,
        by_view=by_view,
        clips=rows,
        comparison=comparison,
        reference=anchors,
        cache_paths=cache_paths,
        evaluation_clips=len(selected),
        reference_clips=len(reference),
        source_groups={c["clip_id"]: c["source_group"] for c in reference + selected},
        source_group_note="Groups are declared from source review; prefilled candidate groups require human verification.",
        participant_disjoint=not unknown and not overlap,
        participant_overlap=overlap,
        participant_identity_unknown=unknown,
        not_evaluated=[
            "per-repetition detection precision/recall/F1",
            "shallow-detection precision/recall/F1",
            "per-repetition depth classification accuracy",
        ],
        timing="Extraction and algorithms timed separately; excludes decoding/rendering, not end-to-end webcam latency.",
        interpretation=(
            "Preliminary development-set pilot; not a final held-out benchmark."
            if args.split == "dev"
            else "Held-out clip-count evaluation only if configuration and references were frozen before inspecting these labels."
        ),
        counts_sha256=sha256(args.counts),
        references_sha256=sha256(args.references),
        code_sha256={
            str(p.relative_to(ROOT)): sha256(p)
            for p in sorted((ROOT / "src").rglob("*.py"))
        },
    )
    report["code_sha256"][str(Path(__file__).relative_to(ROOT))] = sha256(__file__)
    if args.include_hybrid:
        report["interpretation"] = (
            "Exploratory post-midpoint development experiment on previously inspected clips; "
            "not independent validation of the new counter."
            if args.split == "dev"
            else "Post-midpoint test evaluation. Independent validation requires untouched clips "
            "and configuration frozen before examining their labels."
        )
        report["post_midpoint_changes"] = [
            f'{config["min_duration_s"]:g}s minimum duration on smoothed 2D counts',
            "standing required before arming and after 2D tracking loss",
            "counting independent of bilateral depth availability",
            "depth abstention on insufficient coverage or long depth-feature gaps",
        ]
        report["depth_output_note"] = (
            "Accepted-depth counts include only decided acceptable predictions. "
            "Unclear events are counted as attempts; a zero accepted-depth count "
            "with unclear events does not mean all attempts were shallow."
        )
    (args.out / "reference.json").write_text(json.dumps(anchors, indent=2) + "\n")
    (args.out / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    write_csv(args.out / "comparison.csv", comparison)
    write_csv(args.out / "per_clip.csv", rows)
    write_csv(
        args.out / "failure_review.csv",
        sorted(rows, key=lambda r: r["attempt_error"], reverse=True),
    )
    fmt = lambda x: "N/A" if x is None else f"{x:.3f}"
    table = [
        "# "
        + (
            "Post-midpoint hybrid comparison"
            if args.include_hybrid
            else "Midpoint clip-count results"
        ),
        "",
        report["interpretation"],
        "",
        f"Evaluation: {len(selected)} clips. Calibration: {len(reference)} separate clip(s).",
        f'Participant-disjoint: {report["participant_disjoint"]}. Identity unknown: {unknown}.',
        "",
        "| Method | Count MAE | Exact count fraction | Depth-labeled clips | Depth count MAE |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant, values in summary.items():
        table.append(
            "| "
            + " | ".join(
                [
                    variant,
                    fmt(values["attempt_mae"]),
                    fmt(values["exact_count_fraction"]),
                    str(values["depth_labeled_clips"]),
                    fmt(values["depth_mae"]),
                ]
            )
            + " |"
        )
    table += [
        "",
        "No per-repetition detection or shallow-classification F1 is computed from clip totals.",
        "Blank human depth totals are excluded from depth metrics; they are not zeros.",
        "",
        report["timing"],
        "",
        "Inspect per_clip.csv and failure_review.csv for coverage, uncertainty and individual count errors.",
    ]
    if args.include_hybrid:
        table += [
            "",
            report["depth_output_note"],
            "",
            "| Method | Depth-unclear events | Depth decision fraction |",
            "|---|---:|---:|",
        ]
        for variant, values in summary.items():
            table.append(
                f'| {variant} | {values["uncertain_depth_predictions"]} | {fmt(values["predicted_depth_decision_fraction"])} |'
            )
    (args.out / "report.md").write_text("\n".join(table) + "\n")
    if not args.no_plots:
        plot_comparison(args.out, summary)
    print(
        f"Wrote comparison.csv, report.md, results.json and predictions under {args.out}"
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError) as exc:
        sys.exit(f"Clip-count input error: {exc}")
