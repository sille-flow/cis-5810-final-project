"""Rep counting policies used by the model-independent phase matcher.

The default policy is tuned for squats. Other cyclic exercises can supply a
policy implementing ``update(template, features) -> (count, depth)``.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from src.experimental.template import PhaseTemplate


class RepCounter(Protocol):
    def update(self, template: PhaseTemplate, features: np.ndarray, phase: float = 0.0) -> tuple[int, float]: ...
    def reset(self) -> None: ...


class SquatDepthRepCounter:
    """Count a rep after hip drop and both knees reach reference depth."""

    def __init__(self, bottom_depth: float = 0.55, stand_depth: float = 0.25, bottom_hold_frames: int = 3):
        self.bottom_depth = bottom_depth
        self.stand_depth = stand_depth
        self.bottom_hold_frames = bottom_hold_frames
        self.reps = 0
        self._visited_bottom = False
        self._armed = False
        self._bottom_run = 0
        self._anchor_cache: dict[int, dict[str, float]] = {}

    def reset(self) -> None:
        self.reps = 0
        self._visited_bottom = self._armed = False
        self._bottom_run = 0

    def _anchors(self, template: PhaseTemplate) -> dict[str, float]:
        key = id(template)
        if key not in self._anchor_cache:
            names = list(template.feature_names)
            required = ("hip_height", "left_knee", "right_knee")
            if not all(name in names for name in required):
                raise ValueError("SquatDepthRepCounter requires hip_height and bilateral knee features")
            hh = template.mean[:, names.index("hip_height")]
            knees = 0.5 * (template.mean[:, names.index("left_knee")] + template.mean[:, names.index("right_knee")])
            self._anchor_cache[key] = {
                "hh_stand": float(hh.max()), "hh_range": max(float(hh.max()-hh.min()), 0.2),
                "k_stand": float(knees.max()), "k_range": max(float(knees.max()-knees.min()), 30.0),
                "hh_i": names.index("hip_height"), "lk_i": names.index("left_knee"), "rk_i": names.index("right_knee"),
            }
        return self._anchor_cache[key]

    def update(self, template: PhaseTemplate, features: np.ndarray, phase: float = 0.0) -> tuple[int, float]:
        a = self._anchors(template)
        d_hh = (a["hh_stand"] - features[int(a["hh_i"])]) / a["hh_range"]
        fl = (a["k_stand"] - features[int(a["lk_i"])]) / a["k_range"]
        fr = (a["k_stand"] - features[int(a["rk_i"])]) / a["k_range"]
        depth = float(0.5 * np.clip(d_hh, -0.5, 1.5) + 0.5 * np.clip(min(fl, fr), -0.5, 1.5))
        if depth <= self.stand_depth:
            if self._visited_bottom:
                self.reps += 1
                self._visited_bottom = False
            self._armed, self._bottom_run = True, 0
        elif self._armed:
            self._bottom_run = self._bottom_run + 1 if depth >= self.bottom_depth else 0
            if self._bottom_run >= self.bottom_hold_frames:
                self._visited_bottom = True
        return self.reps, depth


class PhaseCycleRepCounter:
    """Generic rep counter for cyclic exercises without squat depth semantics."""

    def __init__(self, wrap_from: float = 0.8, wrap_to: float = 0.2):
        self.wrap_from, self.wrap_to = wrap_from, wrap_to
        self.reps = 0
        self._previous: float | None = None
        self._armed = False

    def reset(self) -> None:
        self.reps = 0
        self._previous = None
        self._armed = False

    def update(self, template: PhaseTemplate, features: np.ndarray, phase: float = 0.0) -> tuple[int, float]:
        if phase <= self.wrap_to:
            if self._armed and self._previous is not None and self._previous >= self.wrap_from:
                self.reps += 1
            self._armed = True
        self._previous = phase
        return self.reps, 0.0
