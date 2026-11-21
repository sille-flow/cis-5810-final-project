"""Content-addressed MediaPipe extraction; every decoded frame is retained."""
import json
from pathlib import Path
import time
import numpy as np
from src.evaluation.data import sha256
from src.pose.features import FEATURE_NAMES, angle_abc

SCHEMA = 1


def cache_signature(video, model, versions):
    return dict(schema=SCHEMA, video_sha256=sha256(video), model_sha256=sha256(model),
                feature_sha256=sha256(Path(__file__).with_name('features.py')),
                extractor_sha256=sha256(__file__),
                pose_stream_sha256=sha256(Path(__file__).with_name('stream.py')),
                versions=versions)


def cache_key(metadata):
    import hashlib
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()


def image_knee_angle(landmarks):
    if landmarks is None:
        return np.nan
    candidates = []
    for ids in ((23,25,27), (24,26,28)):
        points = [landmarks[i] for i in ids]
        visibility = min(p.visibility for p in points)
        angle = angle_abc(*(np.array([p.x,p.y]) for p in points)) if visibility >= .5 else None
        candidates.append((visibility, np.nan if angle is None else angle))
    return max(candidates, key=lambda c:c[0])[1]


def extract_cached(video, cache_dir, model=None):
    # Lazy imports allow the algorithm/tests to run without MediaPipe.
    import cv2
    import mediapipe as mp
    from src.pose.stream import DEFAULT_MODEL, make_landmarker, features_from_result, to_mp_image
    model = Path(model) if model else DEFAULT_MODEL
    if not model.is_file():
        raise ValueError(f'Model missing: {model}')
    signature = cache_signature(video, model, dict(numpy=np.__version__, opencv=cv2.__version__, mediapipe=mp.__version__))
    target = Path(cache_dir) / (cache_key(signature)+'.npz')
    if target.exists():
        return load_cache(target), target
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f'Cannot open video: {video}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError(f'Invalid frame rate: {video}')
    times, features, angles, image_rows, world_rows, views, latency = [], [], [], [], [], [], []
    fallback_count = 0
    try:
        with make_landmarker(model) as detector:
            last_ms = -1
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                pts = cap.get(cv2.CAP_PROP_POS_MSEC)
                if not np.isfinite(pts) or (times and pts <= times[-1]*1000):
                    pts = (times[-1]*1000 + 1000/fps) if times else 0
                    fallback_count += 1
                times.append(pts/1000)
                ms = max(last_ms+1, round(pts)); last_ms = ms
                start = time.perf_counter()
                result = detector.detect_for_video(to_mp_image(frame), ms)
                feat, view, landmarks = features_from_result(result)
                features.append(np.full(len(FEATURE_NAMES),np.nan) if feat is None else feat)
                angles.append(image_knee_angle(landmarks)); views.append(view)
                for attr, rows in [('pose_landmarks',image_rows), ('pose_world_landmarks',world_rows)]:
                    values = getattr(result, attr)
                    rows.append(np.array([[p.x,p.y,p.z,p.visibility] for p in values[0]]) if values else np.full((33,4),np.nan))
                latency.append((time.perf_counter()-start)*1000)
    finally:
        cap.release()
    if not times:
        raise ValueError(f'No decoded frames: {video}')
    # Label times are relative to the beginning of each video.
    times = np.asarray(times)-times[0]
    metadata = dict(signature=signature, fps=fps, timestamp_fallback_frames=fallback_count,
                    frames=len(times), timing='Pose + feature extraction, excludes decoding; includes warmup')
    payload = dict(times=times, features=np.asarray(features), angles=np.asarray(angles),
                   image_landmarks=np.asarray(image_rows), world_landmarks=np.asarray(world_rows),
                   views=np.asarray(views), extraction_ms=np.asarray(latency), metadata=np.array(json.dumps(metadata)))
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, **payload); tmp.replace(target)
    return load_cache(target), target


def load_cache(path):
    with np.load(path, allow_pickle=False) as data:
        result = {k:data[k] for k in data.files if k != 'metadata'}
        result['metadata'] = json.loads(str(data['metadata']))
    if len(result['times']) < 2 or not np.all(np.diff(result['times']) > 0):
        raise ValueError(f'Invalid cache timestamps: {path}')
    return result
