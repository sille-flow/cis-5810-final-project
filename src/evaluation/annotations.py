"""Read the unified annotation CSV and prepare validated experiment CSV rows.

Uses the Python standard library; formula caches are never used for counts.
"""

import math
import csv

CLIP_FIELDS = [
    "clip_id",
    "path",
    "participant",
    "view",
    "split",
    "exercise",
    "attempts",
    "acceptable_depth_reps",
    "notes",
    "source_group",
]
REP_FIELDS = [
    "clip_id",
    "rep_index",
    "start_s",
    "bottom_s",
    "end_s",
    "depth_label",
    "visibility",
    "labeler",
    "notes",
]
REVIEW_FIELDS = [
    "clip_id",
    "path",
    "duration_s",
    "review_status",
    "source_group_candidate",
    "notes",
    "attempts",
    "acceptable_depth_reps",
    "split",
    "suggested_decision",
    "visual_review",
    "reviewer",
    "review_scope",
    "reviewed_at",
]


def read_annotations_csv(path):
    """Read one CSV with clip metadata rows and repetition rows.

    Empty repetition slots are intentionally ignored. A partially entered row
    remains a draft until its parent clip is marked complete.
    """
    tables = {"Clips": [], "Repetitions": []}
    with open(path, newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {
            "record_type",
            "clip_id",
            "decision",
            "annotation_state",
            "start_s",
            "bottom_s",
            "end_s",
            "depth_label",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(
                "Annotation CSV is missing required columns: "
                + ", ".join(sorted(required - set(reader.fieldnames or [])))
            )
        for line, raw in enumerate(reader, 2):
            if None in raw:
                raise ValueError(
                    f"CSV line {line}: extra fields; quote notes containing commas"
                )
            row = {key: (value or "").strip() for key, value in raw.items()}
            if not any(row.values()):
                continue
            kind = row.pop("record_type")
            if kind == "clip":
                tables["Clips"].append(row)
            elif kind == "rep":
                # Preallocated slots contain only clip_id and record_type.
                if any(
                    row.get(key)
                    for key in REP_FIELDS
                    if key not in {"clip_id", "rep_index"}
                ):
                    tables["Repetitions"].append(row)
            else:
                raise ValueError(f"CSV line {line}: record_type must be clip or rep")
    return tables


def prepare_annotations(tables):
    """Validate completed inputs and derive counts, never inventing labels."""
    clips, reps = tables["Clips"], tables["Repetitions"]
    by_id = {}
    for clip in clips:
        cid = clip.get("clip_id", "")
        if not cid or cid in by_id:
            raise ValueError(f"Empty/duplicate clip_id: {cid}")
        by_id[cid] = clip
    events = {cid: [] for cid in by_id}
    for row in reps:
        cid = row.get("clip_id", "")
        if cid not in by_id:
            raise ValueError(f"Repetition has missing/unknown clip_id: {cid}")
        events[cid].append(row)
    manifest, labels, review = [], [], []
    for cid, clip in by_id.items():
        decision = clip.get("decision", "")
        state = clip.get("annotation_state", "")
        if decision not in {"", "keep", "count_only", "crop_or_exclude", "exclude"}:
            raise ValueError(f"{cid}: invalid decision {decision}")
        if state not in {"", "not_started", "in_progress", "complete"}:
            raise ValueError(f"{cid}: invalid annotation_state {state}")
        selected = decision in {"keep", "count_only"} and state == "complete"
        attempts, depth = "", ""
        if selected:
            required = [
                "path",
                "participant",
                "view",
                "source_group",
                "split",
                "reviewer",
            ]
            missing = [key for key in required if not clip.get(key, "")]
            if missing:
                raise ValueError(f"{cid}: missing " + ", ".join(missing))
            if clip["view"] not in {"side", "front", "oblique"}:
                raise ValueError(f"{cid}: invalid view")
            if clip["split"] not in {"reference", "dev", "test"}:
                raise ValueError(f"{cid}: invalid split")
            if clip["split"] == "reference" and decision == "count_only":
                raise ValueError(f"{cid}: count_only cannot calibrate reference depth")
            try:
                duration = float(clip["duration_s"])
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"{cid}: duration_s must be a positive number"
                ) from exc
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(f"{cid}: duration_s must be a positive number")
            prepared = []
            for row in events[cid]:
                missing = [
                    key
                    for key in [
                        "start_s",
                        "bottom_s",
                        "end_s",
                        "depth_label",
                        "visibility",
                        "labeler",
                    ]
                    if not row.get(key, "")
                ]
                if missing:
                    raise ValueError(f"{cid}: repetition missing " + ", ".join(missing))
                try:
                    start, bottom, end = [
                        float(row[key]) for key in ["start_s", "bottom_s", "end_s"]
                    ]
                except ValueError as exc:
                    raise ValueError(
                        f"{cid}: timestamps must be decimal seconds"
                    ) from exc
                if (
                    not all(math.isfinite(x) for x in (start, bottom, end))
                    or not 0 <= start <= bottom <= end <= duration
                    or start >= end
                ):
                    raise ValueError(
                        f"{cid}: invalid timestamps or event beyond video duration"
                    )
                if row["depth_label"] not in {"acceptable", "shallow", "unclear"}:
                    raise ValueError(f"{cid}: invalid depth_label")
                if row["visibility"] not in {"clear", "partial", "occluded"}:
                    raise ValueError(f"{cid}: invalid visibility")
                if decision == "count_only" and row["depth_label"] != "unclear":
                    raise ValueError(
                        f"{cid}: count_only repetitions must have unclear depth"
                    )
                prepared.append(
                    {
                        **{key: row.get(key, "") for key in REP_FIELDS},
                        "start_s": start,
                        "bottom_s": bottom,
                        "end_s": end,
                    }
                )
            prepared.sort(key=lambda r: r["start_s"])
            if any(a["end_s"] > b["start_s"] for a, b in zip(prepared, prepared[1:])):
                raise ValueError(f"{cid}: overlapping repetition intervals")
            for i, row in enumerate(prepared, 1):
                row["rep_index"] = i
            attempts = len(prepared)
            depth = (
                ""
                if any(r["depth_label"] == "unclear" for r in prepared)
                else sum(r["depth_label"] == "acceptable" for r in prepared)
            )
            if clip["split"] == "reference" and not any(
                r["depth_label"] == "acceptable" for r in prepared
            ):
                raise ValueError(
                    f"{cid}: reference requires an acceptable annotated attempt"
                )
            labels.extend(prepared)
            manifest.append(
                {
                    **{key: clip.get(key, "") for key in CLIP_FIELDS},
                    "exercise": clip.get("exercise") or "squat",
                    "attempts": attempts,
                    "acceptable_depth_reps": depth,
                }
            )
        review.append(
            dict(
                clip_id=cid,
                path=clip.get("path", ""),
                duration_s=clip.get("duration_s", ""),
                review_status=decision or "pending",
                source_group_candidate=clip.get("suggested_source_group", ""),
                notes=clip.get("notes", ""),
                attempts=attempts,
                acceptable_depth_reps=depth,
                split=clip.get("split", ""),
                suggested_decision=clip.get("suggested_decision", ""),
                visual_review=clip.get("visual_review", ""),
                reviewer=clip.get("reviewer", ""),
                review_scope=clip.get("review_scope", ""),
                reviewed_at=clip.get("reviewed_at", ""),
            )
        )
    return manifest, labels, review
