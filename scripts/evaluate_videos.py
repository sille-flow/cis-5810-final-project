"""Evaluate labeled squat videos, preserving every frame including missing poses.

Manifest paths are relative to the manifest. Reference must be built separately.
Baseline total attempts and template accepted-depth counts have separate targets.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.baseline import SquatTracker
from src.features import angle_abc
from src.template import MultiViewTemplates
from src.realtime_matcher import RealtimeReferenceMatcher


def baseline_angle(landmarks):
    if landmarks is None:
        return None
    candidates = []
    for ids in ((23,25,27), (24,26,28)):
        points = [landmarks[i] for i in ids]
        visibility = min(p.visibility for p in points)
        angle = angle_abc(*(np.array([p.x, p.y]) for p in points)) if visibility >= .5 else None
        candidates.append((visibility, angle))
    return max(candidates, key=lambda x:x[0])[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--template', type=Path, required=True)
    p.add_argument('--split', choices=['dev','test'], default='dev')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    from src.pose_stream import make_landmarker, features_from_result, to_mp_image
    import cv2
    bank = MultiViewTemplates.load(args.template)
    with args.manifest.open() as f:
        clips = [r for r in csv.DictReader(f) if r['split'] == args.split]
    if not clips:
        raise SystemExit('No clips for requested split')
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for clip in clips:
        source = args.manifest.parent / clip['path']
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise ValueError(f'Cannot open {source}')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f'Invalid FPS: {source}')
        baseline, matcher = SquatTracker(), RealtimeReferenceMatcher(bank)
        history, latency = [], []
        try:
            with make_landmarker() as detector:
                frame = 0
                last_ts = -1
                while True:
                    ok, image = cap.read()
                    if not ok:
                        break
                    start = time.perf_counter()
                    ts = max(last_ts+1, round(frame*1000/fps)); last_ts = ts
                    pose = detector.detect_for_video(to_mp_image(image), ts)
                    features, view, landmarks = features_from_result(pose)
                    baseline.update(baseline_angle(landmarks))
                    match = matcher.update(features, view)
                    latency.append((time.perf_counter()-start)*1000)
                    history.append(dict(frame=frame, time_s=frame/fps, baseline_attempts=baseline.reps,
                                        baseline_depth_reps=baseline.good_reps, reference_depth_reps=match.reps,
                                        aligned=match.aligned, feedback=match.feedback))
                    frame += 1
        finally:
            cap.release()
        if not history:
            raise ValueError(f'No decoded frames: {source}')
        # Use a numeric index, never a user-provided ID, as an output filename.
        with (args.out/f'frames_{len(results):03d}.csv').open('w', newline='') as f:
            w=csv.DictWriter(f, fieldnames=list(history[0])); w.writeheader(); w.writerows(history)
        expected_attempts = int(clip['attempts'])
        accepted = clip['acceptable_depth_reps'].strip()
        expected_depth = int(accepted) if accepted else None
        results.append(dict(clip_id=clip['clip_id'], participant=clip['participant'], view=clip['view'],
            attempts_true=expected_attempts, baseline_attempts=baseline.reps,
            baseline_attempt_error=abs(baseline.reps-expected_attempts),
            depth_true=expected_depth, baseline_depth=baseline.good_reps, reference_depth=match.reps,
            reference_depth_error=abs(match.reps-expected_depth) if expected_depth is not None else None,
            baseline_depth_error=abs(baseline.good_reps-expected_depth) if expected_depth is not None else None,
            frames=len(history), valid_fraction=sum(h['aligned'] for h in history)/len(history),
            processing_median_ms=float(np.median(latency)), processing_p95_ms=float(np.percentile(latency,95))))
    summary = {}
    for key in ['baseline_attempt_error','baseline_depth_error','reference_depth_error']:
        errors=[r[key] for r in results if r[key] is not None]
        summary[key] = dict(n=len(errors), mae=float(np.mean(errors)) if errors else None,
                            exact_fraction=float(np.mean(np.array(errors)==0)) if errors else None)
    report=dict(split=args.split, template=str(args.template),
                template_sha256=hashlib.sha256(args.template.read_bytes()).hexdigest(),
                manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                timing='Pose extraction + both algorithms; excludes decoding/rendering; includes warmup',
                summary=summary, clips=results)
    (args.out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
