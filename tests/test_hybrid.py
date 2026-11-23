import json
import unittest
import numpy as np

from src.squat.hybrid import HybridSquatTracker, run_hybrid
from test_experiments import CONFIG, fixture

REFERENCE = dict(standing=[1.8, 175, 175], span=[0.7, 85, 85])


class HybridTrackerTests(unittest.TestCase):
    def test_complete_cycles_across_frame_rates(self):
        for fps in [15, 24, 30, 60]:
            cache, _ = fixture(fps=fps)
            events, history, _ = run_hybrid(cache, CONFIG, REFERENCE)
            self.assertEqual(
                [e["label"] for e in events], ["acceptable", "shallow"], fps
            )
            self.assertEqual(len(history), len(cache["times"]))

    def test_starting_at_bottom_cannot_finish_an_attempt(self):
        tracker = HybridSquatTracker(CONFIG)
        for t in np.arange(0, 1.5, 1 / 30):
            tracker.update(t, 100 if t < 0.7 else 175)
        self.assertEqual(tracker.events, [])
        self.assertEqual(tracker.state, "standing")

    def test_brief_motion_is_rejected(self):
        tracker = HybridSquatTracker(CONFIG)
        for t in np.arange(0, 2, 1 / 60):
            tracker.update(t, 90 if 0.5 <= t < 0.65 else 175)
        self.assertEqual(tracker.events, [])
        self.assertEqual(tracker.state, "standing")

    def test_no_depth_features_does_not_erase_2d_counts(self):
        cache, _ = fixture()
        cache["features"][:] = np.nan
        events, _, _ = run_hybrid(cache, CONFIG, REFERENCE)
        self.assertEqual(len(events), 2)
        self.assertTrue(
            all(
                e["label"] == "unclear"
                and e["max_depth"] is None
                and e["coverage"] == 0
                for e in events
            )
        )
        json.dumps(events, allow_nan=False)

    def test_no_calibration_still_counts_with_unclear_depth(self):
        cache, _ = fixture()
        events, _, _ = run_hybrid(cache, CONFIG)
        self.assertEqual(len(events), 2)
        self.assertTrue(all(e["label"] == "unclear" for e in events))

    def test_long_depth_gap_only_abstains_on_depth(self):
        cache, _ = fixture()
        cache["features"][(cache["times"] > 1.7) & (cache["times"] < 2.4)] = np.nan
        events, _, _ = run_hybrid(cache, CONFIG, REFERENCE)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["label"], "unclear")
        self.assertEqual(events[0]["count_coverage"], 1.0)
        self.assertLess(events[0]["coverage"], CONFIG["min_coverage"])

    def test_short_depth_gap_preserves_observed_dwell_without_filling(self):
        cache, _ = fixture()
        cache["features"][(cache["times"] > 1.9) & (cache["times"] < 2.0)] = np.nan
        events, _, _ = run_hybrid(cache, CONFIG, REFERENCE)
        self.assertEqual(events[0]["label"], "acceptable")
        self.assertLess(events[0]["coverage"], 1.0)
        self.assertGreater(events[0]["depth_hold_s"], CONFIG["bottom_hold_s"])

    def test_2d_dropout_cancels_pending_attempt(self):
        cache, _ = fixture()
        cache["angles"][(cache["times"] > 1.7) & (cache["times"] < 2.4)] = np.nan
        events, _, _ = run_hybrid(cache, CONFIG, REFERENCE)
        self.assertEqual(len(events), 1)
        self.assertGreater(events[0]["start_s"], 3.0)

    def test_time_jump_without_missing_rows_still_cancels(self):
        cache, _ = fixture()
        keep = ~((cache["times"] > 1.7) & (cache["times"] < 2.4))
        for name in ["times", "features", "angles", "extraction_ms"]:
            cache[name] = cache[name][keep]
        events, _, _ = run_hybrid(cache, CONFIG, REFERENCE)
        self.assertEqual(len(events), 1)
        self.assertGreater(events[0]["start_s"], 3.0)

    def test_repeated_short_2d_dropouts_do_not_look_fully_observed(self):
        cache, _ = fixture()
        drop = (
            (cache["times"] > 1.3)
            & (cache["times"] < 2.8)
            & (np.arange(len(cache["times"])) % 3 == 0)
        )
        cache["angles"][drop] = np.nan
        events, _, _ = run_hybrid(cache, CONFIG, REFERENCE)
        self.assertEqual(len(events), 1)

    def test_single_deep_frame_does_not_satisfy_dwell(self):
        cache, _ = fixture()
        # Keep the count trajectory, but expose shallow geometry except one
        # isolated apparent deep observation. It must not become accepted depth.
        first = (cache["times"] >= 1) & (cache["times"] <= 3)
        cache["features"][first, 0:2] = 132.5
        cache["features"][first, 5] = 1.45
        i = np.argmin(abs(cache["times"] - 2))
        cache["features"][i, [0, 1, 5]] = [90, 90, 1.1]
        events, _, _ = run_hybrid(cache, CONFIG, REFERENCE)
        self.assertEqual(events[0]["label"], "unclear")
        self.assertEqual(events[0]["depth_hold_s"], 0.0)

    def test_timed_out_cycle_cannot_complete_later(self):
        tracker = HybridSquatTracker(CONFIG)
        for t in np.arange(0, 19, 1 / 30):
            tracker.update(t, 175 if t < 0.5 or t > 17 else 90)
        self.assertEqual(tracker.events, [])

    def test_invalid_timestamps_rejected(self):
        tracker = HybridSquatTracker(CONFIG)
        tracker.update(0, 175)
        for value in [0, -1, np.nan, np.inf]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                tracker.update(value, 175)
