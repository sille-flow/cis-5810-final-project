"""Per-frame pose features for reference matching.

Everything here is computed from MediaPipe *world* landmarks (metric 3D, origin
at the hips). The features are chosen so they do not change when the person
rotates about the vertical axis:

* 3D joint angles (knee, hip) are rotation invariant.
* torso incline is measured against the world vertical.
* hip height is the vertical hip-to-ankle distance divided by torso length,
  so it does not depend on camera distance, image aspect ratio or yaw.
* knee width is knee separation divided by hip width (knee-cave signal).

That means a template recorded from the side can still follow a squat done
facing the camera. Camera view is still estimated (from the hip/shoulder line
in 3D, not from normalised image coordinates) but is only used to re-weight
channels, because MediaPipe's depth estimate is noisier head-on.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

FEATURE_NAMES = (
    "left_knee",
    "right_knee",
    "left_hip",
    "right_hip",
    "torso_incline",
    "hip_height",
    "knee_width",
)

HIP_HEIGHT_INDEX = FEATURE_NAMES.index("hip_height")

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28

MIN_VISIBILITY = 0.4

# Yaw of the shoulder/hip line relative to the image plane:
# 0 deg = facing (or back to) the camera, 90 deg = pure side view.
FRONT_MAX_YAW = 35.0
SIDE_MIN_YAW = 60.0

# Matching weights. Head-on, MediaPipe's depth is least reliable, so knee/hip
# angles are trusted a bit less and hip height / knee width a bit more.
VIEW_WEIGHTS = {
    "front": np.array([0.60, 0.60, 0.70, 0.70, 0.50, 1.50, 1.00]),
    "side": np.array([1.20, 1.20, 1.00, 1.00, 1.10, 1.30, 0.05]),
    "oblique": np.array([0.90, 0.90, 0.90, 0.90, 0.80, 1.40, 0.30]),
}


def angle_abc(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    """Return angle ABC in degrees, or None if the points are degenerate."""
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return None
    cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _xyz(landmark) -> np.ndarray:
    return np.array([landmark.x, landmark.y, landmark.z], dtype=np.float64)


def _visible(landmarks, indices: Sequence[int], min_vis: float = MIN_VISIBILITY) -> bool:
    return all(getattr(landmarks[i], "visibility", 1.0) >= min_vis for i in indices)


def estimate_yaw_deg(landmarks, world_landmarks, min_vis: float = MIN_VISIBILITY) -> float | None:
    """Body yaw relative to the camera from the 3D shoulder + hip lines.

    0 = facing the camera, 90 = side on. Uses metric world coordinates, so it
    does not depend on image aspect ratio (portrait phone clip vs landscape
    webcam), which the old normalised-2D width/torso ratio did.
    """
    if world_landmarks is None:
        return None
    needed = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
    if not _visible(landmarks, needed, min_vis):
        return None
    vx = vz = 0.0
    for left, right in ((LEFT_SHOULDER, RIGHT_SHOULDER), (LEFT_HIP, RIGHT_HIP)):
        d = _xyz(world_landmarks[right]) - _xyz(world_landmarks[left])
        vx += d[0]
        vz += d[2]
    if abs(vx) + abs(vz) < 1e-6:
        return None
    return math.degrees(math.atan2(abs(vz), abs(vx)))


def estimate_camera_view(landmarks, world_landmarks=None, min_vis: float = MIN_VISIBILITY) -> str:
    yaw = estimate_yaw_deg(landmarks, world_landmarks, min_vis)
    if yaw is None:
        return "oblique"
    if yaw <= FRONT_MAX_YAW:
        return "front"
    if yaw >= SIDE_MIN_YAW:
        return "side"
    return "oblique"


def _angle(world, i, j, k, landmarks, min_vis: float) -> float | None:
    if not _visible(landmarks, (i, j, k), min_vis):
        return None
    return angle_abc(_xyz(world[i]), _xyz(world[j]), _xyz(world[k]))


def landmarks_to_features(
    landmarks,
    world_landmarks=None,
    min_vis: float = MIN_VISIBILITY,
) -> np.ndarray | None:
    """Convert one MediaPipe pose to a feature vector, or None if unusable.

    World landmarks are required: image-space angles are not view invariant.
    """
    if world_landmarks is None:
        return None
    w = world_landmarks

    left_knee = _angle(w, LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, landmarks, min_vis)
    right_knee = _angle(w, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE, landmarks, min_vis)
    left_hip = _angle(w, LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, landmarks, min_vis)
    right_hip = _angle(w, RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE, landmarks, min_vis)

    torso = None
    hip_height = None
    trunk_ids = (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
    if _visible(landmarks, trunk_ids, min_vis):
        shoulder = 0.5 * (_xyz(w[LEFT_SHOULDER]) + _xyz(w[RIGHT_SHOULDER]))
        hip = 0.5 * (_xyz(w[LEFT_HIP]) + _xyz(w[RIGHT_HIP]))
        # World y points down, so "up" is -y. 0 deg = perfectly upright.
        torso = angle_abc(shoulder, hip, hip + np.array([0.0, -1.0, 0.0]))
        torso_len = float(np.linalg.norm(shoulder - hip))
        if torso_len > 1e-3 and _visible(landmarks, (LEFT_ANKLE, RIGHT_ANKLE), min_vis):
            ankle = 0.5 * (_xyz(w[LEFT_ANKLE]) + _xyz(w[RIGHT_ANKLE]))
            hip_height = float((ankle[1] - hip[1]) / torso_len)

    knee_width = None
    if _visible(landmarks, (LEFT_KNEE, RIGHT_KNEE, LEFT_HIP, RIGHT_HIP), min_vis):
        knees = float(np.linalg.norm(_xyz(w[LEFT_KNEE]) - _xyz(w[RIGHT_KNEE])))
        hips = float(np.linalg.norm(_xyz(w[LEFT_HIP]) - _xyz(w[RIGHT_HIP])))
        if hips > 1e-3:
            knee_width = knees / hips

    # Pushups and floor exercises often hide the ankles or knees from the pose
    # model. Keep partially observed poses and let the sequence interpolator fill
    # missing channels; reject frames only when too little usable geometry exists.
    values = [left_knee, right_knee, left_hip, right_hip, torso, hip_height, knee_width]
    if sum(v is not None for v in values) < 3:
        return None
    return np.array([v if v is not None else np.nan for v in values], dtype=np.float64)


def feature_min_std(names: tuple[str, ...] = FEATURE_NAMES) -> np.ndarray:
    floors = []
    for name in names:
        if name == "hip_height":
            floors.append(0.08)  # ratio, roughly 0.7 (deep) to 1.8 (standing)
        elif name == "knee_width":
            floors.append(0.12)  # ratio of knee separation to hip width
        else:
            floors.append(4.0)  # degrees
    return np.array(floors, dtype=np.float64)


def interpolate_nans(sequence: np.ndarray) -> np.ndarray:
    """Linearly fill NaN feature channels independently."""
    out = sequence.copy()
    n, d = out.shape
    idx = np.arange(n)
    for j in range(d):
        col = out[:, j]
        good = np.isfinite(col)
        if good.all():
            continue
        if not good.any():
            # Neutral defaults prevent an entirely occluded channel from
            # becoming a false zero-angle signal in a reference template.
            out[:, j] = 180.0 if j < 4 else (1.7 if j == HIP_HEIGHT_INDEX else (1.0 if j == 6 else 0.0))
            continue
        out[:, j] = np.interp(idx, idx[good], col[good])
    return out
