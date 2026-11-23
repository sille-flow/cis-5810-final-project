"""Minimal human inputs for a clip-count pilot; no event-level ground truth."""

import csv
import math
from pathlib import Path

from src.evaluation.data import sha256


def csv_rows(path, required):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'{path}: missing columns: {", ".join(sorted(missing))}')
        rows = []
        for line, raw in enumerate(reader, 2):
            if None in raw:
                raise ValueError(
                    f"{path}:{line}: extra fields; quote notes containing commas"
                )
            row = {key: (value or "").strip() for key, value in raw.items()}
            if any(row.values()):
                rows.append(row)
        return rows


def count_value(value, context, required=False):
    if value == "":
        if required:
            raise ValueError(
                f"{context}: enter the total completed squats (0 is allowed)"
            )
        return None
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(
            f"{context}: count must be a nonnegative whole number"
        ) from exc
    if not math.isfinite(number) or number < 0 or number != int(number):
        raise ValueError(f"{context}: count must be a nonnegative whole number")
    return int(number)


def missing_inputs(manifest, references, split="dev"):
    """Return actionable human-input gaps without creating an output report."""
    clips = csv_rows(manifest, ["clip_id", "split", "human_attempts"])
    anchors = csv_rows(references, ["clip_id", "start_s", "bottom_s", "end_s"])
    gaps = [
        f'{row["clip_id"]}: human_attempts'
        for row in clips
        if row["split"] == split and not row["human_attempts"]
    ]
    for row in anchors:
        missing = [key for key in ("start_s", "bottom_s", "end_s") if not row[key]]
        if missing:
            gaps.append(f'{row["clip_id"]}: reference ' + ", ".join(missing))
    return clips, gaps


def load_inputs(manifest, references, split="dev"):
    manifest = Path(manifest)
    rows = csv_rows(
        manifest,
        [
            "clip_id",
            "split",
            "human_attempts",
            "human_acceptable_depth_reps",
            "path",
            "source_group",
            "participant",
            "view",
        ],
    )
    ids, hashes, groups = set(), {}, {}
    clips = []
    for row in rows:
        cid = row["clip_id"]
        if not cid or cid in ids:
            raise ValueError(f"Empty/duplicate clip_id: {cid}")
        ids.add(cid)
        if row["split"] not in {"reference", "dev", "test", "exclude"}:
            raise ValueError(f"{cid}: split must be reference, dev, test, or exclude")
        if row["split"] == "exclude":
            continue
        if not row["source_group"]:
            raise ValueError(f"{cid}: source_group is required")
        group = row["source_group"]
        if group in groups and groups[group] != row["split"]:
            raise ValueError(f"{cid}: source_group {group} crosses splits")
        groups[group] = row["split"]
        if row["view"] not in {"side", "front", "oblique"}:
            raise ValueError(f"{cid}: invalid view")
        if not row["path"]:
            raise ValueError(f"{cid}: video path is required")
        row["path"] = (manifest.parent / row["path"]).resolve()
        if not row["path"].is_file():
            raise ValueError(f'{cid}: missing video {row["path"]}')
        digest = sha256(row["path"])
        if digest in hashes:
            raise ValueError(f"{cid}: duplicate video content ({hashes[digest]})")
        hashes[digest] = cid
        row["video_sha256"] = digest
        # Test labels are not inspected during a dev run. Reference total counts
        # are unused: just one or two annotated reference cycles are sufficient.
        row["attempts"] = (
            count_value(row["human_attempts"], f"{cid}: human_attempts", required=True)
            if row["split"] == split
            else None
        )
        row["acceptable_depth_reps"] = (
            count_value(
                row["human_acceptable_depth_reps"],
                f"{cid}: human_acceptable_depth_reps",
            )
            if row["split"] == split
            else None
        )
        if (
            row["acceptable_depth_reps"] is not None
            and row["acceptable_depth_reps"] > row["attempts"]
        ):
            raise ValueError(f"{cid}: accepted-depth count exceeds total attempts")
        clips.append(row)
    reference = [r for r in clips if r["split"] == "reference"]
    selected = [r for r in clips if r["split"] == split]
    if not reference or not selected:
        raise ValueError(
            "Need at least one reference clip and one clip in the selected split"
        )
    annotations = {row["clip_id"]: [] for row in reference}
    for row in csv_rows(references, ["clip_id", "start_s", "bottom_s", "end_s"]):
        cid = row["clip_id"]
        if cid not in annotations:
            raise ValueError(f"{cid}: calibration row must name a reference clip")
        if any(not row[key] for key in ("start_s", "bottom_s", "end_s")):
            raise ValueError(
                f"{cid}: enter start_s, bottom_s, end_s for one acceptable reference squat"
            )
        try:
            start, bottom, end = [
                float(row[key]) for key in ("start_s", "bottom_s", "end_s")
            ]
        except ValueError as exc:
            raise ValueError(
                f"{cid}: reference timestamps must be decimal seconds"
            ) from exc
        if (
            not all(math.isfinite(x) for x in (start, bottom, end))
            or not 0 <= start < bottom < end
        ):
            raise ValueError(
                f"{cid}: reference times must satisfy 0 <= start < bottom < end"
            )
        annotations[cid].append(
            dict(start_s=start, bottom_s=bottom, end_s=end, label="acceptable")
        )
    for cid, events in annotations.items():
        events.sort(key=lambda e: e["start_s"])
        if not events:
            raise ValueError(f"{cid}: needs an annotated acceptable reference squat")
        if any(a["end_s"] > b["start_s"] for a, b in zip(events, events[1:])):
            raise ValueError(f"{cid}: overlapping reference intervals")
    return reference, selected, annotations


def summarize_counts(rows):
    if not rows:
        raise ValueError("Cannot summarize an empty count comparison")
    depth = [r["depth_error"] for r in rows if r["depth_error"] is not None]
    negatives = [r for r in rows if r["true_attempts"] == 0]
    return dict(
        clips=len(rows),
        attempt_mae=sum(r["attempt_error"] for r in rows) / len(rows),
        exact_count_fraction=sum(r["attempt_error"] == 0 for r in rows) / len(rows),
        depth_labeled_clips=len(depth),
        depth_mae=sum(depth) / len(depth) if depth else None,
        negative_clips=len(negatives),
        false_reps_on_negatives=sum(r["predicted_attempts"] for r in negatives),
    )


def validate_config(config):
    for key, value in config.items():
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"Invalid numeric config: {key}")
    if (
        not config["smooth_tau_s"] > 0
        or not 0
        <= config["stand_depth"]
        < config["start_depth"]
        < config["acceptable_depth"]
    ):
        raise ValueError("Invalid smoothing/depth configuration")
    if not 0 <= config["min_coverage"] <= 1 or not config["max_gap_s"] > 0:
        raise ValueError("Invalid coverage/gap configuration")
    if (
        not 0 < config["min_duration_s"] < config["max_duration_s"]
        or config["bottom_hold_s"] < 0
        or config["direction_epsilon"] < 0
    ):
        raise ValueError("Invalid temporal configuration")
