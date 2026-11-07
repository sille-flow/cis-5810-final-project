"""Webcam demo: live frames follow a reference phase template."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.overlay import draw_match, draw_skeleton
from src.pose_stream import features_from_result, make_landmarker, monotonic_timestamp_ms, to_mp_image
from src.realtime_matcher import RealtimeReferenceMatcher
from src.exercise_selector import AutoExerciseSelector
from src.synthetic import synthetic_squat_sequence
from src.template import MultiViewTemplates, build_multiview_templates

import cv2


def load_or_bootstrap(path: Path) -> MultiViewTemplates:
    if path.exists():
        return MultiViewTemplates.load(path)
    print(f"{path} missing; using a synthetic squat template.")
    return build_multiview_templates(
        {
            "side": [synthetic_squat_sequence(n_frames=72, n_reps=2, depth=90.0, view="side")],
            "front": [synthetic_squat_sequence(n_frames=72, n_reps=2, depth=90.0, view="front")],
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=ROOT / "data" / "templates" / "squat_good.npz")
    parser.add_argument("--templates", type=Path, default=None, help="Directory of exercise templates named <exercise>_good.npz")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--source", type=Path, default=None, help="Optional video file instead of webcam")
    args = parser.parse_args()

    if args.templates:
        banks = {
            path.stem.removesuffix("_good"): MultiViewTemplates.load(path)
            for path in sorted(args.templates.glob("*_good.npz"))
        }
        if not banks:
            raise SystemExit(f"No *_good.npz templates found in {args.templates}")
        matcher = AutoExerciseSelector(banks)
    else:
        templates = load_or_bootstrap(args.template)
        matcher = RealtimeReferenceMatcher(templates)
    cap = cv2.VideoCapture(str(args.source) if args.source else args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open camera/video")

    last_ts = -1
    frame_index = 0
    try:
        with make_landmarker() as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                ts = monotonic_timestamp_ms(cap, last_ts, frame_index)
                last_ts = ts
                frame_index += 1
                result = landmarker.detect_for_video(to_mp_image(frame), ts)
                features, view, landmarks = features_from_result(result)
                if landmarks is not None:
                    draw_skeleton(frame, landmarks)
                match = matcher.update(features, view=view)
                draw_match(frame, match)
                display = cv2.resize(frame, (960, 540))
                cv2.imshow("Reference form matching", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
