import unittest
import csv
from pathlib import Path
import tempfile
from src.evaluation.annotations import prepare_annotations, read_annotations_csv


def clip(**overrides):
    return dict(
        clip_id="a",
        path="video.mp4",
        decision="keep",
        view="side",
        participant="p1",
        source_group="source1",
        split="dev",
        annotation_state="complete",
        reviewer="Kevin",
        duration_s="8",
        **overrides
    )


def rep(**overrides):
    row = dict(
        clip_id="a",
        start_s="1",
        bottom_s="2",
        end_s="3",
        depth_label="acceptable",
        visibility="clear",
        labeler="Kevin",
    )
    row.update(overrides)
    return row


class AnnotationExportTests(unittest.TestCase):
    def prepare(self, video, events):
        return prepare_annotations({"Clips": [video], "Repetitions": events})

    def test_csv_blank_slots_ignored_and_filled_rows_preserved(self):
        fields = [
            "record_type",
            "clip_id",
            "decision",
            "annotation_state",
            "start_s",
            "bottom_s",
            "end_s",
            "depth_label",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(dict(record_type="clip", clip_id="a"))
                writer.writerow(dict(record_type="rep", clip_id="a"))
                writer.writerow(dict(record_type="rep", clip_id="a", start_s="0"))
            tables = read_annotations_csv(path)
            self.assertEqual(len(tables["Clips"]), 1)
            self.assertEqual(len(tables["Repetitions"]), 1)
            self.assertEqual(tables["Repetitions"][0]["start_s"], "0")

    def test_csv_unknown_record_type_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.csv"
            path.write_text(
                "record_type,clip_id,decision,annotation_state,start_s,bottom_s,end_s,depth_label\nunknown,a,,,,,,\n"
            )
            with self.assertRaisesRegex(ValueError, "record_type"):
                read_annotations_csv(path)

    def test_draft_counts_remain_missing(self):
        video = clip()
        video["annotation_state"] = "in_progress"
        manifest, labels, review = self.prepare(video, [rep()])
        self.assertEqual((manifest, labels), ([], []))
        self.assertEqual(review[0]["attempts"], "")

    def test_counts_derived_and_indices_sorted(self):
        manifest, labels, _ = self.prepare(
            clip(),
            [rep(start_s="4", bottom_s="5", end_s="6", depth_label="shallow"), rep()],
        )
        self.assertEqual(
            (manifest[0]["attempts"], manifest[0]["acceptable_depth_reps"]), (2, 1)
        )
        self.assertEqual([r["start_s"] for r in labels], [1, 4])
        self.assertEqual([r["rep_index"] for r in labels], [1, 2])

    def test_unknown_depth_does_not_become_zero(self):
        manifest, _, _ = self.prepare(clip(), [rep(depth_label="unclear")])
        self.assertEqual(manifest[0]["acceptable_depth_reps"], "")

    def test_confirmed_negative_is_zero(self):
        manifest, _, _ = self.prepare(clip(), [])
        self.assertEqual(
            (manifest[0]["attempts"], manifest[0]["acceptable_depth_reps"]), (0, 0)
        )

    def test_count_only_rejects_known_depth(self):
        video = clip()
        video["decision"] = "count_only"
        with self.assertRaisesRegex(ValueError, "unclear depth"):
            self.prepare(video, [rep()])

    def test_partial_repetition_is_not_exported_as_complete(self):
        with self.assertRaisesRegex(ValueError, "missing bottom_s"):
            self.prepare(clip(), [rep(bottom_s="")])

    def test_nonfinite_and_out_of_range_timestamps_rejected(self):
        for value in ["nan", "inf", "9"]:
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "timestamps"
            ):
                self.prepare(clip(), [rep(end_s=value)])

    def test_overlapping_events_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlapping"):
            self.prepare(clip(), [rep(), rep(start_s="2", bottom_s="3", end_s="4")])

    def test_reference_needs_acceptable_attempt(self):
        video = clip()
        video["split"] = "reference"
        with self.assertRaisesRegex(ValueError, "acceptable annotated"):
            self.prepare(video, [rep(depth_label="unclear")])

    def test_confirmed_source_group_required(self):
        video = clip()
        video["source_group"] = ""
        with self.assertRaisesRegex(ValueError, "missing source_group"):
            self.prepare(video, [rep()])

    def test_orphan_repetition_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown clip_id"):
            self.prepare(clip(), [rep(clip_id="typo")])
