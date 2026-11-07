"""MediaPipe pose extraction from webcam or video files."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .features import estimate_camera_view, landmarks_to_features

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_lite.task"


@contextlib.contextmanager
def _quiet_stderr():
    """Silence C++ (glog/TFLite) stderr noise. Works on Windows and POSIX."""
    try:
        saved = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
    except OSError:
        yield
        return
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


class _QuietLandmarker:
    """Wraps PoseLandmarker; hides the harmless native warnings.

    They are printed at creation and on the first few inferences
    ("Feedback manager requires a model with a single signature...",
    "Using NORM_RECT without IMAGE_DIMENSIONS..."), so stderr is muted for
    those calls only.
    """

    QUIET_CALLS = 5

    def __init__(self, inner):
        self._inner = inner
        self._calls = 0

    def detect_for_video(self, image, timestamp_ms):
        if self._calls < self.QUIET_CALLS:
            self._calls += 1
            with _quiet_stderr():
                return self._inner.detect_for_video(image, timestamp_ms)
        return self._inner.detect_for_video(image, timestamp_ms)

    def close(self):
        self._inner.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def ensure_model(path: str | Path = DEFAULT_MODEL) -> Path:
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request

    print(f"Downloading pose model to {path} ...")
    urllib.request.urlretrieve(MODEL_URL, path)
    return path


def make_landmarker(model_path: str | Path = DEFAULT_MODEL):
    model_path = ensure_model(model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    with _quiet_stderr():
        inner = vision.PoseLandmarker.create_from_options(options)
    return _QuietLandmarker(inner)


def to_mp_image(frame_bgr):
    rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def monotonic_timestamp_ms(cap: cv2.VideoCapture, last_ts: int, frame_index: int) -> int:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    file_ts = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    if file_ts > last_ts:
        return file_ts
    return last_ts + max(1, int(1000.0 / fps))


def features_from_result(result):
    if not result.pose_landmarks:
        return None, "oblique", None
    landmarks = result.pose_landmarks[0]
    world = result.pose_world_landmarks[0] if result.pose_world_landmarks else None
    feats = landmarks_to_features(landmarks, world)
    view = estimate_camera_view(landmarks, world)
    return feats, view, landmarks


def extract_feature_sequence(
    video_path: str | Path,
    model_path: str | Path = DEFAULT_MODEL,
    max_frames: int | None = None,
):
    """Return (features, camera_view) from a reference or user video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open {video_path}")

    rows: list = []
    views: list[str] = []
    last_ts = -1
    frame_index = 0
    with make_landmarker(model_path) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = monotonic_timestamp_ms(cap, last_ts, frame_index)
            last_ts = ts
            frame_index += 1
            result = landmarker.detect_for_video(to_mp_image(frame), ts)
            parsed = features_from_result(result)
            if parsed[0] is not None:
                rows.append(parsed[0])
                views.append(parsed[1])
            if max_frames is not None and frame_index >= max_frames:
                break
    cap.release()
    if not rows:
        raise RuntimeError(f"No usable pose features in {video_path}")
    view = max(set(views), key=views.count)
    return np.stack(rows, axis=0), view