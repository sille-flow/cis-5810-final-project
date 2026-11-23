"""Inventory local MM-Fit labels and pose coverage without loading wearable arrays."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np


def audit(root):
    records, totals = [], Counter()
    for labels in sorted(root.glob('w*/w*_labels.csv')):
        session = labels.parent.name
        pose = np.load(labels.parent / f'{session}_pose_3d.npy', mmap_mode='r', allow_pickle=False)
        if pose.ndim != 3 or pose.shape[0] != 3 or pose.shape[2] != 18:
            raise ValueError(f'Unexpected 3D layout: {labels}: {pose.shape}')
        ids = pose[0, :, 0]
        if not np.all(np.diff(ids) > 0) or not np.all(pose[:, :, 0] == ids):
            raise ValueError(f'Invalid frame IDs: {session}')
        with labels.open() as f:
            for index, row in enumerate(csv.reader(f)):
                if not row:
                    continue
                start, end, reps = map(int, row[:3])
                if start > end or reps < 0:
                    raise ValueError(f'Invalid label: {row}')
                exercise = row[3].strip()
                mask = (ids >= start) & (ids <= end)
                count = int(mask.sum())
                records.append(dict(session=session, set_index=index, start_frame=start,
                                    end_frame=end, repetitions=reps, exercise=exercise,
                                    pose_frames=count, expected_frames=end-start+1,
                                    complete_pose_coverage=count == end-start+1))
                totals[exercise] += 1
    videos = [str(p.relative_to(root)) for p in root.rglob('*') if p.suffix.lower() in {'.mp4','.avi','.mov','.mkv'}]
    return records, dict(sessions=len({r['session'] for r in records}), sets=len(records),
                         sets_by_exercise=dict(totals), videos=videos,
                         incomplete_sets=sum(not r['complete_pose_coverage'] for r in records),
                         note='Coverage assumes inclusive label endpoints; labels are set-level, not form labels.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('data/datasets/mm-fit'))
    p.add_argument('--out', type=Path, default=Path('data/review/mmfit'))
    args = p.parse_args()
    records, summary = audit(args.root)
    if not records:
        raise SystemExit('No workout labels found')
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out/'sets.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
