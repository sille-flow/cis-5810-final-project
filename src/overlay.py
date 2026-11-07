"""Drawing helpers shared by the notebook and live runner."""

from __future__ import annotations

import cv2
import numpy as np

from .features import FEATURE_NAMES
from .realtime_matcher import MatchResult

CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]


def draw_skeleton(frame, landmarks) -> None:
    h, w = frame.shape[:2]
    pts = []
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        pts.append((cx, cy))
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 180, 255), 2)


def _put_text(frame, text, org, color, scale=0.8, thickness=2):
    """Bright text with a thick black outline so it reads on any background."""
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_match(frame, match: MatchResult) -> None:
    h, w = frame.shape[:2]
    scale = max(0.7, min(w, h) / 700.0)
    line_h = int(38 * scale)

    top_z = np.argsort(-np.abs(match.z_scores))[:3]
    lines = [
        *(([(f"Exercise: {match.exercise}", (0, 255, 0), 1.0)] if match.exercise else [])),
        (f"Reps: {match.reps}", (255, 255, 255), 1.25),
        ((f"View: {match.view}   Depth: {int(round(100 * match.depth))}%") if match.exercise.lower().rstrip("s") == "squat" else f"View: {match.view}", (255, 255, 255), 0.9),
        (f"Phase: {match.phase:.2f}", (255, 255, 255), 0.9),
        (match.feedback, (0, 255, 255), 1.0),
    ]
    for idx in top_z:
        names = match.feature_names or FEATURE_NAMES
        if idx < len(names):
            lines.append((f"{names[idx]}: z={float(match.z_scores[idx]):+.1f}", (220, 220, 220), 0.8))

    # Semi-transparent dark panel behind the text.
    panel_w = int(min(w - 20, 560 * scale))
    panel_h = int(line_h * (len(lines) + 0.6))
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    y = 10 + line_h
    for text, color, rel in lines:
        _put_text(frame, text, (22, y), color, scale=0.75 * rel * scale, thickness=2)
        y += line_h if rel >= 0.9 else int(line_h * 0.85)
