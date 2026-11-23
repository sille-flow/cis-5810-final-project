"""Inventory local videos and summarize extraction coverage, not accuracy.

Does not write ground-truth labels or expose model repetition predictions.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.annotations import read_annotations_csv
from src.evaluation.data import sha256


def main():
    import cv2
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, default=ROOT / 'data/annotations/legacy/squat_annotations.csv')
    parser.add_argument('--quality', type=Path, default=ROOT / 'data/review/pose_quality.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'data/review')
    args = parser.parse_args()
    clips = read_annotations_csv(args.annotations)['Clips']
    quality = {Path(r['video']).name: r for r in json.loads(args.quality.read_text())} if args.quality.exists() else {}
    rows, hashes = [], {}
    for clip in clips:
        video = (args.annotations.parent / clip['path']).resolve()
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise ValueError(f'Cannot open {video}')
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        digest = sha256(video)
        duplicate = hashes.get(digest, '')
        hashes[digest] = clip['clip_id']
        q = quality.get(video.name, {})
        rows.append(dict(clip_id=clip['clip_id'], filename=video.name, duration_s=round(frames/fps, 3),
                         fps=fps, frames=frames, width=width, height=height, video_sha256=digest,
                         exact_duplicate_of=duplicate, suggested_decision=clip.get('suggested_decision', ''),
                         usable_2d_fraction=q.get('usable_2d_fraction', ''),
                         usable_depth_fraction=q.get('usable_depth_fraction', ''),
                         analysis_scope='Video inventory and feature coverage only; no accuracy or ground-truth counts'))
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/'inventory.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f'{len(rows)} videos, {sum(r["duration_s"] for r in rows):.1f} seconds, '
          f'{sum(bool(r["exact_duplicate_of"]) for r in rows)} byte-identical duplicates, '
          f'{len(quality)} clips with extraction coverage. Wrote {args.out / "inventory.csv"}.')


if __name__ == '__main__':
    main()
