import unittest
import numpy as np
from src.squat.baseline import SquatTracker
from src.experimental.realtime_matcher import RealtimeReferenceMatcher
from src.experimental.synthetic import synthetic_squat_sequence
from src.experimental.template import build_phase_template


class EvaluationTests(unittest.TestCase):
    def test_baseline_counts_shallow_attempt_separately(self):
        tracker = SquatTracker()
        for angle in (175,140,120,140,175):
            tracker.update(angle)
        self.assertEqual((tracker.reps,tracker.good_reps),(1,0))
        for angle in (140,85,140,175):
            tracker.update(angle)
        self.assertEqual((tracker.reps,tracker.good_reps),(2,1))

    def test_missing_pose_cannot_finish_a_rep(self):
        template=build_phase_template([synthetic_squat_sequence(n_frames=64,n_reps=1)])
        matcher=RealtimeReferenceMatcher(template)
        matcher.rep_counter._visited_bottom=True
        result=matcher.update(np.array([np.nan]*7))
        self.assertFalse(result.aligned)
        self.assertEqual(result.feedback,'Pose unclear')
        standing=template.mean[np.argmax(template.mean[:,5])]
        for _ in range(5):
            result=matcher.update(standing)
        self.assertEqual(result.reps,0)

    def test_schema_mismatch_is_explicit(self):
        template=build_phase_template([synthetic_squat_sequence()])
        with self.assertRaises(ValueError):
            RealtimeReferenceMatcher(template).update(np.ones(6))
