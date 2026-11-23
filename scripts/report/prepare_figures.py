"""Create auditable diagnostics and report figures from a completed count pilot."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.pose.cache import load_cache
from src.squat.tracking import depth_from_features

DEFAULT_RESULTS = ROOT / "data/results/post_midpoint"
DEFAULT_FIGURES = ROOT / "data/reports/midpoint/figures"
NAMES = {
    "baseline": "Baseline",
    "confidence_2d": "Smoothing",
    "reference_3d": "Reference",
    "temporal_3d": "Temporal",
    "hybrid_2d": "Hybrid",
}
COLORS = ["#53667D", "#2E877A", "#6375AB", "#BD7B35", "#874763"]


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def hybrid_figure(results, figures, result):
    index = next(
        i
        for i, c in enumerate(result["comparison"])
        if c["clip_id"] == "archive_squat_26"
    )
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9.2, 3.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
        layout="constrained",
    )
    for variant, color, style in [
        ("confidence_2d", COLORS[1], "-"),
        ("temporal_3d", COLORS[3], "-"),
        ("hybrid_2d", COLORS[4], "--"),
    ]:
        history = read_csv(results / f"clip_{index:03d}_{variant}_frames.csv")
        axes[0].step(
            [float(r["time_s"]) for r in history],
            [int(r["attempts"]) for r in history],
            where="post",
            label=NAMES[variant],
            color=color,
            ls=style,
            lw=1.8,
        )
    axes[0].set(
        title="Clip 26: 3 reps counted; depth unclear for 1 rep",
        ylabel="Completed attempts",
        ylim=(-0.1, 3.4),
        yticks=[0, 1, 2, 3],
    )
    axes[0].legend(ncol=3, fontsize=8, loc="upper left")
    events = json.loads(
        (results / f"clip_{index:03d}_hybrid_2d_events.json").read_text()
    )
    for event in events:
        color = COLORS[1] if event["label"] == "acceptable" else "#D3A35B"
        axes[1].axvspan(event["start_s"], event["end_s"], color=color, alpha=0.3)
        axes[1].text(
            (event["start_s"] + event["end_s"]) / 2,
            0.5,
            event["label"],
            ha="center",
            va="center",
            fontsize=8,
        )
    axes[1].set(xlabel="Video time (s)", ylabel="Hybrid depth", ylim=(0, 1), yticks=[])
    fig.savefig(figures / "hybrid_case.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIGURES)
    args = parser.parse_args()
    results, figures = args.results, args.figures
    figures.mkdir(parents=True, exist_ok=True)
    result = json.loads((results / "results.json").read_text())
    variants = list(result["summary"])
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    summaries = result["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 2.7), layout="constrained")
    for ax, metric, title in zip(
        axes,
        ["attempt_mae", "depth_mae"],
        ["Completed-attempt count error", "Accepted-depth count error"],
    ):
        values = [summaries[v][metric] for v in variants]
        bars = ax.bar(
            [NAMES[v] for v in variants],
            values,
            color=COLORS[: len(variants)],
            width=0.62,
        )
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
        ax.set(title=title, ylabel="MAE (reps per video)", ylim=(0, max(values) * 1.23))
        ax.tick_params(axis="x", labelsize=8)
    fig.savefig(figures / "method_comparison.png", dpi=220)
    plt.close(fig)

    diagnostics = []
    for index, clip in enumerate(result["comparison"]):
        cid = clip["clip_id"]
        cache = load_cache(result["cache_paths"][cid])
        valid = np.isfinite(cache["features"][:, [5, 0, 1]]).all(axis=1)
        values = np.array(
            [depth_from_features(f, result["reference"]) for f in cache["features"]]
        )
        finite = values[np.isfinite(values)]
        gaps = np.diff(cache["times"][valid])
        base = json.loads(
            (results / f"clip_{index:03d}_baseline_events.json").read_text()
        )
        ref = json.loads(
            (results / f"clip_{index:03d}_reference_3d_events.json").read_text()
        )
        temporal = json.loads(
            (results / f"clip_{index:03d}_temporal_3d_events.json").read_text()
        )
        states = read_csv(results / f"clip_{index:03d}_reference_3d_frames.csv")
        diagnostics.append(
            dict(
                clip_id=cid,
                frames=len(valid),
                duration_s=float(cache["times"][-1]),
                valid_depth_fraction=float(valid.mean()),
                max_between_valid_s=float(gaps.max()) if len(gaps) else None,
                raw_depth_min=float(finite.min()) if len(finite) else None,
                raw_depth_max=float(finite.max()) if len(finite) else None,
                baseline_events=len(base),
                baseline_events_under_0_6s=sum(
                    e["end_s"] - e["start_s"] < 0.6 for e in base
                ),
                baseline_event_durations_s=[e["end_s"] - e["start_s"] for e in base],
                reference_unarmed_fraction=sum(s["state"] == "unarmed" for s in states)
                / len(states),
                reference_temporal_events_identical=ref == temporal,
            )
        )
    (results / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 4.8), layout="constrained")
    # Clip 5: evidence of threshold chatter, with model event ends marked.
    i = next(
        i
        for i, c in enumerate(result["comparison"])
        if c["clip_id"] == "archive_squat_5"
    )
    for variant, color in [("baseline", COLORS[0]), ("confidence_2d", COLORS[1])]:
        rows = read_csv(results / f"clip_{i:03d}_{variant}_frames.csv")
        axes[0].plot(
            [float(r["time_s"]) for r in rows],
            [float(r["signal"]) if r["signal"] else np.nan for r in rows],
            label=variant,
            lw=1.1,
            color=color,
        )
    for threshold in (150, 160):
        axes[0].axhline(threshold, color="#77818C", ls=":", lw=0.8)
    events = json.loads((results / f"clip_{i:03d}_baseline_events.json").read_text())
    axes[0].scatter(
        [e["end_s"] for e in events],
        [168] * len(events),
        marker="v",
        s=18,
        color="#A24635",
        label="Baseline completion",
    )
    axes[0].set(
        title="Clip 5: human count 4; baseline 10; smoothing 5",
        ylabel="Image knee angle (degrees)",
        xlabel="Video time (s)",
    )
    axes[0].legend(loc="lower right", fontsize=7, ncol=3)
    # Clip 26: missing bilateral geometry and state cancellation.
    cid = "archive_squat_26"
    cache = load_cache(result["cache_paths"][cid])
    t = cache["times"]
    depth = np.array(
        [depth_from_features(f, result["reference"]) for f in cache["features"]]
    )
    axes[1].plot(
        t, depth, color=COLORS[2], lw=1, label="Reference-relative depth (raw)"
    )
    axes[1].fill_between(
        t,
        -0.2,
        1.1,
        where=~np.isfinite(depth),
        color="#EACFC9",
        alpha=0.8,
        label="Required geometry missing",
    )
    axes[1].axhline(
        result["config"]["stand_depth"],
        ls=":",
        color="#53667D",
        lw=0.9,
        label="Standing threshold",
    )
    axes[1].axhline(
        result["config"]["acceptable_depth"],
        ls="--",
        color=COLORS[1],
        lw=0.9,
        label="Depth threshold",
    )
    axes[1].set(
        title="Clip 26: human count 3; baseline 3; reference / temporal 0",
        xlabel="Video time (s)",
        ylabel="Relative depth",
        ylim=(-0.2, 1.1),
    )
    axes[1].legend(loc="upper left", fontsize=7, ncol=2)
    fig.savefig(figures / "failure_traces.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 1.45))
    ax.set(xlim=(0, 10), ylim=(0, 1))
    ax.axis("off")
    boxes = [
        ("RGB video", 0.1, 1.4),
        ("MediaPipe\npretrained pose", 1.95, 1.65),
        ("Landmarks +\ntimestamps/cache", 4.05, 1.8),
        (f"{len(variants)} tracking\nconfigurations", 6.3, 1.55),
        ("Human totals\nvs predictions", 8.3, 1.6),
    ]
    for label, x, width in boxes:
        box = FancyBboxPatch(
            (x, 0.25),
            width,
            0.5,
            boxstyle="round,pad=0.035",
            fc="#EFF3F7",
            ec="#71839A",
            lw=0.8,
        )
        ax.add_patch(box)
        ax.text(x + width / 2, 0.5, label, ha="center", va="center", fontsize=8)
    for x in (1.55, 3.7, 5.95, 7.95):
        ax.annotate(
            "",
            xy=(x + 0.32, 0.5),
            xytext=(x, 0.5),
            arrowprops=dict(arrowstyle="->", color="#53667D"),
        )
    fig.savefig(figures / "pipeline.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    stages = [
        ("baseline", "Fixed 2D thresholds", 2.1666666666666665),
        ("confidence_2d", "Causal smoothing and tracking-gap handling", 1.0),
        ("reference_3d", "Reference-relative bilateral depth", 1.0),
        ("temporal_3d", "Duration, ascent, dwell and coverage checks", 1.0),
        ("hybrid_2d", "Validated 2D counts with independent depth uncertainty", 2 / 3),
    ]
    with (results / "experiment_log.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "date",
                "stage",
                "change",
                "count_mae",
                "depth_count_mae",
                "interpretation",
            ],
        )
        writer.writeheader()
        for variant, change, _ in stages:
            if variant not in summaries:
                continue
            writer.writerow(
                dict(
                    date="2026-11-22",
                    stage=variant,
                    change=change,
                    count_mae=summaries[variant]["attempt_mae"],
                    depth_count_mae=summaries[variant]["depth_mae"],
                    interpretation=(
                        "Developed after reviewing these six clips; needs testing on new videos."
                        if variant == "hybrid_2d"
                        else "Original four methods evaluated on the same six development clips."
                    ),
                )
            )
    if "hybrid_2d" in variants:
        hybrid_figure(results, figures, result)
    print("Wrote diagnostics, experiment log, and report figures.")


if __name__ == "__main__":
    main()
