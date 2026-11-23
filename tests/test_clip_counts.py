import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.evaluation.clip_counts import load_inputs, summarize_counts, missing_inputs
from test_experiments import fixture

ROOT = Path(__file__).resolve().parents[1]


def write_rows(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ClipCountTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ["ref", "evaluation"]:
            (self.root / (name + ".mp4")).write_bytes(name.encode())
        self.counts = self.root / "counts.csv"
        self.references = self.root / "reference.csv"
        self.rows = [
            dict(
                clip_id="ref",
                split="reference",
                human_attempts="",
                human_acceptable_depth_reps="",
                path="ref.mp4",
                source_group="g1",
                participant="unknown",
                view="side",
            ),
            dict(
                clip_id="evaluation",
                split="dev",
                human_attempts="2",
                human_acceptable_depth_reps="",
                path="evaluation.mp4",
                source_group="g2",
                participant="unknown",
                view="oblique",
            ),
        ]
        self.anchors = [dict(clip_id="ref", start_s="1", bottom_s="2", end_s="3")]
        self.save()

    def save(self):
        write_rows(self.counts, self.rows)
        write_rows(self.references, self.anchors)

    def runner(self):
        spec = importlib.util.spec_from_file_location(
            "clip_runner", ROOT / "scripts/evaluation/run_clip_counts.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_total_counts_do_not_require_evaluation_events(self):
        ref, selected, annotations = load_inputs(self.counts, self.references)
        self.assertEqual(selected[0]["attempts"], 2)
        self.assertIsNone(selected[0]["acceptable_depth_reps"])
        self.assertEqual(set(annotations), {"ref"})
        self.assertIsNone(ref[0]["attempts"])

    def test_zero_is_label_but_blank_is_not(self):
        self.rows[1]["human_attempts"] = "0"
        self.save()
        self.assertEqual(load_inputs(self.counts, self.references)[1][0]["attempts"], 0)
        self.rows[1]["human_attempts"] = ""
        self.save()
        with self.assertRaisesRegex(ValueError, "enter the total"):
            load_inputs(self.counts, self.references)

    def test_invalid_counts_rejected(self):
        for value in ["-1", "1.5", "nan", "inf", "two"]:
            self.rows[1]["human_attempts"] = value
            self.save()
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "whole number"
            ):
                load_inputs(self.counts, self.references)

    def test_optional_depth_count_must_not_exceed_total(self):
        self.rows[1]["human_acceptable_depth_reps"] = "3"
        self.save()
        with self.assertRaisesRegex(ValueError, "exceeds"):
            load_inputs(self.counts, self.references)

    def test_source_group_leakage_rejected(self):
        self.rows[1]["source_group"] = "g1"
        self.save()
        with self.assertRaisesRegex(ValueError, "crosses splits"):
            load_inputs(self.counts, self.references)

    def test_duplicate_video_rejected(self):
        (self.root / "evaluation.mp4").write_bytes(b"ref")
        with self.assertRaisesRegex(ValueError, "duplicate video"):
            load_inputs(self.counts, self.references)

    def test_reference_order_and_clip_membership_checked(self):
        self.anchors[0]["bottom_s"] = "0"
        self.save()
        with self.assertRaisesRegex(ValueError, "start < bottom < end"):
            load_inputs(self.counts, self.references)
        self.anchors[0].update(clip_id="evaluation", bottom_s="2")
        self.save()
        with self.assertRaisesRegex(ValueError, "reference clip"):
            load_inputs(self.counts, self.references)

    def test_metrics_use_only_labeled_depth_and_no_event_f1(self):
        summary = summarize_counts(
            [
                dict(
                    attempt_error=0,
                    depth_error=None,
                    true_attempts=2,
                    predicted_attempts=2,
                ),
                dict(
                    attempt_error=1,
                    depth_error=0,
                    true_attempts=0,
                    predicted_attempts=1,
                ),
            ]
        )
        self.assertEqual(summary["attempt_mae"], 0.5)
        self.assertEqual(summary["exact_count_fraction"], 0.5)
        self.assertEqual(summary["depth_labeled_clips"], 1)
        self.assertEqual(summary["false_reps_on_negatives"], 1)
        self.assertNotIn("event_detection", summary)

    def test_unfinished_inputs_produce_no_report(self):
        self.rows[1]["human_attempts"] = ""
        self.save()
        out = self.root / "out"
        with patch("builtins.print"), self.assertRaisesRegex(
            ValueError, "Human inputs still needed"
        ):
            self.runner().main(
                [
                    "--counts",
                    str(self.counts),
                    "--references",
                    str(self.references),
                    "--out",
                    str(out),
                ]
            )
        self.assertFalse(out.exists())
        self.assertEqual(len(missing_inputs(self.counts, self.references)[1]), 1)

    def test_report_runs_all_methods_with_count_only_labels(self):
        # The Numbers input used "video" as a placeholder in every person cell.
        # It must not be mistaken for a verified shared participant identity.
        for row in self.rows:
            row["participant"] = "video"
        self.save()
        runner = self.runner()
        cache, _ = fixture()
        out = self.root / "out"
        with patch.object(
            runner, "extract_cached", return_value=(cache, self.root / "cache.npz")
        ), patch("builtins.print"):
            runner.main(
                [
                    "--counts",
                    str(self.counts),
                    "--references",
                    str(self.references),
                    "--out",
                    str(out),
                    "--no-plots",
                ]
            )
        result = json.loads((out / "results.json").read_text())
        self.assertEqual(result["evaluation"], "clip_counts_only")
        self.assertEqual(len(result["summary"]), 4)
        self.assertFalse(result["participant_disjoint"])
        self.assertTrue(result["participant_identity_unknown"])
        self.assertEqual(result["summary"]["temporal_3d"]["attempt_mae"], 0)
        self.assertIsNone(result["summary"]["temporal_3d"]["depth_mae"])
        self.assertNotIn("event_detection", result["summary"]["baseline"])
        self.assertTrue((out / "comparison.csv").exists())
        self.assertTrue((out / "reference.json").exists())
        self.assertEqual(len(list(out.glob("*_events.json"))), 4)

    def test_hybrid_is_separate_and_reports_depth_uncertainty(self):
        runner = self.runner()
        cache, _ = fixture()
        cache["features"][:] = float("nan")
        # Reference must remain observable; evaluation alone loses 3D features.
        good, _ = fixture()

        def extract(path, *args):
            return (good if path.stem == "ref" else cache), self.root / "cache.npz"

        out = self.root / "post_midpoint"
        with patch.object(runner, "extract_cached", side_effect=extract), patch(
            "builtins.print"
        ):
            runner.main(
                [
                    "--counts",
                    str(self.counts),
                    "--references",
                    str(self.references),
                    "--out",
                    str(out),
                    "--include-hybrid",
                    "--no-plots",
                ]
            )
        report = json.loads((out / "results.json").read_text())
        self.assertEqual(len(report["summary"]), 5)
        self.assertEqual(report["summary"]["hybrid_2d"]["attempt_mae"], 0)
        self.assertEqual(
            report["summary"]["hybrid_2d"]["uncertain_depth_predictions"], 2
        )
        self.assertEqual(
            report["summary"]["hybrid_2d"]["predicted_depth_decision_fraction"], 0
        )

    def test_hybrid_cannot_overwrite_midpoint(self):
        for path in [
            ROOT / "data/results/midpoint",
            ROOT / "data/results/midpoint/nested",
        ]:
            with self.subTest(path=path), self.assertRaisesRegex(
                ValueError, "frozen midpoint"
            ):
                self.runner().main(["--include-hybrid", "--out", str(path)])
