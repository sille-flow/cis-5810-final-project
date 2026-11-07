import unittest

from src.realtime_matcher import RealtimeReferenceMatcher
from src.synthetic import synthetic_squat_sequence
from src.template import MultiViewTemplates, build_multiview_templates, build_phase_template


class RealtimeMatcherTests(unittest.TestCase):
    def test_counts_reps_on_matching_sequence(self):
        template = build_phase_template(
            [synthetic_squat_sequence(n_frames=64, n_reps=1, depth=90.0)]
        )
        matcher = RealtimeReferenceMatcher(template)
        live = synthetic_squat_sequence(n_frames=128, n_reps=2, depth=90.0, noise=1.0)
        last = None
        for row in live:
            last = matcher.update(row, view="side")
        self.assertIsNotNone(last)
        self.assertGreaterEqual(last.reps, 1)

    def test_flags_shallow_live_vs_deep_reference(self):
        template = build_phase_template(
            [synthetic_squat_sequence(n_frames=64, n_reps=1, depth=85.0, torso_lean=20.0)]
        )
        matcher = RealtimeReferenceMatcher(template, z_threshold=1.2)
        live = synthetic_squat_sequence(n_frames=64, n_reps=1, depth=140.0, torso_lean=20.0)
        flagged = False
        for row in live:
            result = matcher.update(row, view="side")
            if "knee" in result.feedback:
                flagged = True
        self.assertTrue(flagged)

    def test_front_view_counts_reps_using_hip_height(self):
        bank = build_multiview_templates(
            {
                "side": [synthetic_squat_sequence(n_frames=64, n_reps=1, view="side")],
                "front": [synthetic_squat_sequence(n_frames=64, n_reps=1, view="front")],
            }
        )
        matcher = RealtimeReferenceMatcher(bank)
        live = synthetic_squat_sequence(n_frames=128, n_reps=2, view="front", noise=0.5)
        last = None
        for row in live:
            last = matcher.update(row, view="front")
        self.assertIsNotNone(last)
        self.assertEqual(last.view, "front")
        self.assertGreaterEqual(last.reps, 1)

    def test_front_live_still_tracks_against_side_only_template(self):
        bank = MultiViewTemplates(
            views={
                "side": build_phase_template(
                    [synthetic_squat_sequence(n_frames=64, n_reps=1, view="side")],
                    view="side",
                )
            }
        )
        matcher = RealtimeReferenceMatcher(bank)
        live = synthetic_squat_sequence(n_frames=128, n_reps=2, view="front", noise=0.4)
        last = None
        for row in live:
            last = matcher.update(row, view="front")
        self.assertIsNotNone(last)
        self.assertGreaterEqual(last.reps, 1)


if __name__ == "__main__":
    unittest.main()
