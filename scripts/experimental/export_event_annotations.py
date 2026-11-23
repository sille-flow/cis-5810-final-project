"""Validate the unified annotation CSV and export the experiment's CSV inputs."""
import argparse
import csv
import os
from datetime import datetime
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.annotations import read_annotations_csv, prepare_annotations, CLIP_FIELDS, REP_FIELDS, REVIEW_FIELDS
from src.evaluation.data import load_manifest, load_repetitions


def write_csv(path, fields, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, default=ROOT / 'data/annotations/legacy/squat_annotations.csv')
    parser.add_argument('--out', type=Path, default=ROOT / 'data/annotations')
    parser.add_argument('--check', action='store_true', help='Validate and summarize without writing any files')
    args = parser.parse_args()
    tables = read_annotations_csv(args.annotations)
    clips, reps, review = prepare_annotations(tables)
    # The optional detailed input lives in a subfolder; rebase exported paths
    # against the output CSV directory rather than assuming matching parents.
    for row in clips + review:
        if row.get('path'):
            row['path'] = os.path.relpath((args.annotations.parent / row['path']).resolve(), args.out.resolve())
    print(f'{len(review)} review rows; {len(clips)} completed retained clips; {len(reps)} labeled attempts.')
    pending = [r['clip_id'] for r in tables['Clips'] if r.get('decision') not in {'exclude', 'crop_or_exclude'} and not (r.get('decision') in {'keep', 'count_only'} and r.get('annotation_state') == 'complete')]
    if pending:
        print('Not exported (curation/annotation unfinished): ' + ', '.join(pending))
    if not clips:
        print('No experiment-ready clips. Confirm metadata, annotate full clips, and mark the clip row complete in the annotation CSV.')
        if args.check:
            return
        raise ValueError('Nothing exported; existing CSVs were left unchanged.')
    # Validate in a temporary directory. Resolve video paths against the intended
    # CSV destination so the existing validators check hashes and split leakage.
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        validation_clips = [{**row, 'path': str((args.out / row['path']).resolve())} for row in clips]
        write_csv(temp / 'clips.csv', CLIP_FIELDS, validation_clips)
        write_csv(temp / 'repetitions.csv', REP_FIELDS, reps)
        parsed = load_manifest(temp / 'clips.csv')
        load_repetitions(temp / 'repetitions.csv', parsed)
        if args.check:
            print('Completed rows validated; --check wrote no files.')
            return
        args.out.mkdir(parents=True, exist_ok=True)
        backup = ROOT / 'data/backups/annotations' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        backed_up = 0
        for filename in ['clips.csv', 'repetitions.csv', 'archive_squat_review.csv']:
            existing = args.out / filename
            if existing.exists():
                backup.mkdir(parents=True, exist_ok=True)
                shutil.copy2(existing, backup / filename)
                backed_up += 1
        for filename, fields, rows in [('clips.csv', CLIP_FIELDS, clips), ('repetitions.csv', REP_FIELDS, reps), ('archive_squat_review.csv', REVIEW_FIELDS, review)]:
            staged = args.out / (filename + '.tmp')
            write_csv(staged, fields, rows)
            staged.replace(args.out / filename)
        print(f'Exported to {args.out}.')
        if backed_up:
            print(f'Previous CSVs backed up under {backup}.')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError) as exc:
        sys.exit(f'Annotation error: {exc}')
