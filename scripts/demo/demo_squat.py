"""Demo 2D counting with optional depth feedback, or replay the midpoint method."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.squat.tracking import AttemptTracker, depth_from_features
from src.squat.hybrid import HYBRID_VARIANT, HybridSquatTracker
from src.pose.cache import image_knee_angle
from src.evaluation.clip_counts import validate_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--reference",
        type=Path,
        help="Optional depth calibration for hybrid; required for temporal_3d",
    )
    p.add_argument(
        "--method", choices=[HYBRID_VARIANT, "temporal_3d"], default=HYBRID_VARIANT
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "configs/experiment.json",
    )
    p.add_argument("--source", type=Path)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument(
        "--headless",
        action="store_true",
        help="Replay --source without opening a window or camera",
    )
    p.add_argument("--save-json", type=Path, help="Save completed events and counts")
    args = p.parse_args()
    if args.headless and not args.source:
        p.error("--headless requires --source")
    if args.method == "temporal_3d" and args.reference is None:
        p.error("temporal_3d requires --reference")
    import cv2
    from src.pose.stream import make_landmarker, features_from_result, to_mp_image
    from src.visualization.overlay import draw_skeleton

    config = json.loads(args.config.read_text())
    validate_config(config)
    reference = json.loads(args.reference.read_text()) if args.reference else None
    tracker = (
        HybridSquatTracker(config, reference)
        if args.method == HYBRID_VARIANT
        else AttemptTracker(config, True)
    )
    cap = cv2.VideoCapture(str(args.source) if args.source else args.camera)
    if not cap.isOpened():
        raise ValueError("Cannot open video/camera")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    origin = time.monotonic()
    index = 0
    last_ms = -1
    feedback = "Stand fully upright to begin"
    try:
        with make_landmarker() as detector:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if args.source:
                    raw_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                    ms = (
                        round(raw_ms)
                        if np.isfinite(raw_ms) and raw_ms > last_ms
                        else last_ms + max(1, round(1000 / fps))
                    )
                else:
                    ms = max(last_ms + 1, round((time.monotonic() - origin) * 1000))
                last_ms = ms
                index += 1
                result = detector.detect_for_video(to_mp_image(frame), ms)
                features, view, landmarks = features_from_result(result)
                depth = (
                    depth_from_features(features, reference)
                    if features is not None and reference
                    else np.nan
                )
                angle = image_knee_angle(landmarks)
                event = (
                    tracker.update(ms / 1000, angle, features)
                    if args.method == HYBRID_VARIANT
                    else tracker.update(ms / 1000, depth)
                )
                if event:
                    feedback = {
                        "acceptable": "Depth reached",
                        "shallow": "Shallow attempt",
                        "unclear": "Depth uncertain",
                    }[event.label]
                if args.headless:
                    continue
                if landmarks is not None:
                    draw_skeleton(frame, landmarks)
                if args.method == HYBRID_VARIANT:
                    current_feedback = tracker.feedback
                    if not np.isfinite(angle):
                        current_feedback = "2D tracking unclear - show the moving leg"
                    elif not np.isfinite(depth) and tracker.state != "unarmed":
                        current_feedback = "Counting 2D; depth unavailable"
                else:
                    current_feedback = (
                        feedback
                        if np.isfinite(depth)
                        else "Pose unclear - show hips, knees and feet"
                    )
                lines = [
                    f'Attempts: {len(tracker.events)}  Depth accepted: {sum(e.label=="acceptable" for e in tracker.events)}',
                    f'{args.method} / {tracker.state}  Uncertain depth: {sum(e.label=="unclear" for e in tracker.events)}',
                    current_feedback,
                ]
                for j, line in enumerate(lines):
                    cv2.putText(
                        frame,
                        line,
                        (15, 30 + 30 * j),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 0, 0),
                        4,
                    )
                    cv2.putText(
                        frame,
                        line,
                        (15, 30 + 30 * j),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        1,
                    )
                scale = min(960 / frame.shape[1], 720 / frame.shape[0])
                cv2.imshow(
                    "Squat experiment", cv2.resize(frame, None, fx=scale, fy=scale)
                )
                if cv2.waitKey(1) & 255 == ord("q"):
                    break
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
    result = dict(
        method=args.method,
        attempts=len(tracker.events),
        accepted_depth=sum(e.label == "acceptable" for e in tracker.events),
        uncertain_depth=sum(e.label == "unclear" for e in tracker.events),
        events=[asdict(e) for e in tracker.events],
    )
    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
