"""Synthetic squat feature sequences for testing without a camera.

Generates rows in the same layout as ``src.features.FEATURE_NAMES``. Because
the real features are yaw invariant, ``view`` only changes the noise level
(head-on depth estimates are noisier), not the underlying motion.

These are bootstrap fixtures, not biomechanical ground truth.
"""

from __future__ import annotations

import numpy as np


def synthetic_squat_sequence(
    n_frames: int = 72,
    n_reps: int = 3,
    depth: float = 90.0,
    noise: float = 1.0,
    view: str = "side",
    torso_lean: float = 28.0,
    seed: int = 0,
) -> np.ndarray:
    """Return (n_frames, 7) rows: knees, hips, torso incline, hip height, knee width."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_frames) / n_frames * n_reps
    s = 0.5 * (1.0 - np.cos(2.0 * np.pi * t))  # 0 standing, 1 bottom

    knee = 175.0 - s * (175.0 - depth)
    hip = 165.0 - s * (165.0 - (depth - 6.0))
    torso = 5.0 + s * (torso_lean - 5.0)
    # hip height ~1.76 standing; falls roughly with knee flexion.
    hip_height = 1.76 - 0.67 * (175.0 - knee) / (175.0 - 85.0)
    knee_width = np.full(n_frames, 1.2)

    scale = 2.0 if view == "front" else 1.0
    cols = [knee, knee, hip, hip, torso, hip_height, knee_width]
    out = np.stack(cols, axis=1).astype(np.float64)
    out[:, :5] += rng.normal(0.0, noise * scale, size=(n_frames, 5))
    out[:, 5] += rng.normal(0.0, 0.01, size=n_frames)
    out[:, 6] += rng.normal(0.0, 0.03, size=n_frames)
    return out