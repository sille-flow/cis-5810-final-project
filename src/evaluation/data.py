"""Strict annotation loading and split/provenance validation."""
import csv
import hashlib
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path):
    path = Path(path)
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Manifest is empty. Add labeled clips first.')
    ids, files, groups = set(), {}, {}
    for row in rows:
        cid = row['clip_id'].strip()
        if not cid or cid in ids:
            raise ValueError(f'Empty/duplicate clip ID: {cid}')
        ids.add(cid)
        row['clip_id'] = cid
        split = row['split']
        if split not in {'reference', 'dev', 'test'}:
            raise ValueError(f'{cid}: split must be reference, dev or test')
        row['path'] = (path.parent / row['path']).resolve()
        if not row['path'].is_file():
            raise ValueError(f'{cid}: missing video {row["path"]}')
        digest = sha256(row['path'])
        if digest in files:
            raise ValueError(f'{cid}: duplicate video content ({files[digest]})')
        files[digest] = cid
        row['video_sha256'] = digest
        group = row.get('source_group', '').strip()
        if group:
            if group in groups and groups[group] != split:
                raise ValueError(f'{cid}: source_group {group} crosses splits')
            groups[group] = split
        for key in ('attempts', 'acceptable_depth_reps'):
            value = row.get(key, '').strip()
            row[key] = int(value) if value else None
            if row[key] is not None and row[key] < 0:
                raise ValueError(f'{cid}: negative {key}')
        if row['attempts'] is None:
            raise ValueError(f'{cid}: attempts is required (use 0 for negatives)')
        if row['acceptable_depth_reps'] is not None and row['acceptable_depth_reps'] > row['attempts']:
            raise ValueError(f'{cid}: depth count exceeds attempts')
    return rows


def load_repetitions(path, clips):
    by_id = {c['clip_id']: [] for c in clips}
    with Path(path).open(newline='') as f:
        for row in csv.DictReader(f):
            cid = row['clip_id']
            if cid not in by_id:
                raise ValueError(f'Repetition references unknown clip: {cid}')
            event = {k: float(row[k]) for k in ('start_s', 'bottom_s', 'end_s')}
            if not 0 <= event['start_s'] <= event['bottom_s'] <= event['end_s'] or event['end_s'] <= event['start_s']:
                raise ValueError(f'{cid}: invalid repetition timestamps')
            label = row['depth_label']
            if label not in {'acceptable', 'shallow', 'unclear'}:
                raise ValueError(f'{cid}: invalid depth label')
            event['label'] = label
            by_id[cid].append(event)
    for clip in clips:
        events = sorted(by_id[clip['clip_id']], key=lambda r:r['start_s'])
        by_id[clip['clip_id']] = events
        if len(events) != clip['attempts']:
            raise ValueError(f'{clip["clip_id"]}: repetition rows disagree with attempts')
        if any(a['end_s'] > b['start_s'] for a,b in zip(events,events[1:])):
            raise ValueError(f'{clip["clip_id"]}: overlapping repetitions')
        known_depth = clip['acceptable_depth_reps']
        if known_depth is not None:
            if any(e['label']=='unclear' for e in events):
                raise ValueError(f'{clip["clip_id"]}: leave depth count blank when any rep is unclear')
            if sum(e['label']=='acceptable' for e in events) != known_depth:
                raise ValueError(f'{clip["clip_id"]}: depth count disagrees with repetition labels')
    return by_id
