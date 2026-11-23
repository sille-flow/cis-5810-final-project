"""Post-midpoint counting: validated 2D cycles with independent depth evidence.

The original four experiment methods remain unchanged. Missing 3D features can
make a completed attempt's depth unclear, but cannot erase reliable 2D motion.
"""

from dataclasses import asdict, dataclass
import time

import numpy as np

from src.squat.tracking import depth_from_features

HYBRID_VARIANT = "hybrid_2d"


@dataclass
class HybridEvent:
    start_s: float
    bottom_s: float
    end_s: float
    label: str
    max_depth: float | None
    coverage: float
    count_coverage: float
    depth_hold_s: float


class HybridSquatTracker:
    """Count smoothed 2D cycles, then assess depth without filling missing data."""

    def __init__(self, config, reference=None):
        self.c = config
        self.reference = reference
        self.events = []
        self.state = "unarmed"
        self.pending = None
        self.last_t = self.last_valid = self.smoothed = None
        self.signal = self.depth_signal = np.nan
        self.previous_angle_seen = self.previous_depth_seen = False
        self.previous_depth = np.nan
        self.feedback = "Stand fully upright to begin"

    def _cancel(self, message):
        self.state = "unarmed"
        self.pending = None
        self.last_valid = self.smoothed = None
        self.signal = np.nan
        self.previous_angle_seen = self.previous_depth_seen = False
        self.previous_depth = np.nan
        self.feedback = message

    def update(self, timestamp, angle, features=None):
        t = float(timestamp)
        if not np.isfinite(t) or (self.last_t is not None and t <= self.last_t):
            raise ValueError("Timestamps must be finite and strictly increasing")
        dt = t - self.last_t if self.last_t is not None else 0.0
        self.last_t = t
        self.depth_signal = (
            depth_from_features(features, self.reference)
            if features is not None and self.reference is not None
            else np.nan
        )
        angle_seen = angle is not None and np.isfinite(angle)
        depth_seen = bool(np.isfinite(self.depth_signal))
        gap = t - self.last_valid if self.last_valid is not None else 0.0
        if self.last_valid is not None and gap > self.c["max_gap_s"]:
            self._cancel("2D tracking lost; stand upright to restart")
        if angle_seen:
            alpha = (
                1 - np.exp(-gap / self.c["smooth_tau_s"])
                if self.smoothed is not None
                else 1.0
            )
            self.smoothed = (
                float(angle)
                if self.smoothed is None
                else self.smoothed + alpha * (float(angle) - self.smoothed)
            )
            self.last_valid = t
            self.signal = self.smoothed
        else:
            self.signal = np.nan

        event = None
        if self.pending is not None:
            p = self.pending
            duration = t - p["start"]
            if duration > self.c["max_duration_s"]:
                self._cancel("Movement timed out; stand upright to restart")
            else:
                # Count observed intervals only when both endpoint samples exist.
                if (
                    angle_seen
                    and self.previous_angle_seen
                    and dt <= self.c["max_gap_s"]
                ):
                    p["count_observed_s"] += dt
                if depth_seen:
                    if self.previous_depth_seen and dt <= self.c["max_gap_s"]:
                        p["depth_observed_s"] += dt
                        if (
                            min(self.previous_depth, self.depth_signal)
                            >= self.c["acceptable_depth"]
                        ):
                            p["depth_hold_s"] += dt
                    depth_gap = t - (
                        p["last_depth_t"]
                        if p["last_depth_t"] is not None
                        else p["start"]
                    )
                    if depth_gap > self.c["max_gap_s"]:
                        p["depth_gap_exceeded"] = True
                    p["last_depth_t"] = t
                    p["peak"] = (
                        float(self.depth_signal)
                        if p["peak"] is None
                        else max(p["peak"], float(self.depth_signal))
                    )
                elif (
                    t
                    - (
                        p["last_depth_t"]
                        if p["last_depth_t"] is not None
                        else p["start"]
                    )
                    > self.c["max_gap_s"]
                ):
                    p["depth_gap_exceeded"] = True
                if angle_seen and self.signal < p["minimum_angle"]:
                    p["minimum_angle"], p["bottom"] = self.signal, t
                if angle_seen and self.signal > 160:
                    count_coverage = min(
                        1.0, p["count_observed_s"] / max(duration, 1e-9)
                    )
                    depth_coverage = min(
                        1.0, p["depth_observed_s"] / max(duration, 1e-9)
                    )
                    if duration < self.c["min_duration_s"]:
                        self.feedback = "Brief motion ignored"
                    elif count_coverage < self.c["min_coverage"]:
                        self.feedback = "Interrupted 2D motion ignored"
                    else:
                        label = "unclear"
                        if (
                            p["peak"] is not None
                            and depth_coverage >= self.c["min_coverage"]
                            and not p["depth_gap_exceeded"]
                        ):
                            if p["peak"] < self.c["acceptable_depth"]:
                                label = "shallow"
                            elif p["depth_hold_s"] >= self.c["bottom_hold_s"]:
                                label = "acceptable"
                        event = HybridEvent(
                            p["start"],
                            p["bottom"],
                            t,
                            label,
                            p["peak"],
                            depth_coverage,
                            count_coverage,
                            p["depth_hold_s"],
                        )
                        self.events.append(event)
                        self.feedback = {
                            "acceptable": "Depth reached",
                            "shallow": "Shallow attempt",
                            "unclear": "Attempt counted; depth uncertain",
                        }[label]
                    self.pending = None
                    self.state = "standing"

        elif angle_seen:
            if self.state == "unarmed":
                # A video beginning at the bottom must not complete a phantom rep.
                if self.signal > 160:
                    self.state = "standing"
                    self.feedback = "Ready"
            elif self.state == "standing" and self.signal < 150:
                self.state = "squatting"
                self.pending = dict(
                    start=t,
                    bottom=t,
                    minimum_angle=self.signal,
                    count_observed_s=0.0,
                    depth_observed_s=0.0,
                    depth_hold_s=0.0,
                    peak=float(self.depth_signal) if depth_seen else None,
                    last_depth_t=t if depth_seen else None,
                    depth_gap_exceeded=False,
                )
        self.previous_angle_seen = bool(angle_seen)
        self.previous_depth_seen = depth_seen
        self.previous_depth = self.depth_signal
        return event


def run_hybrid(cache, config, reference=None):
    tracker = HybridSquatTracker(config, reference)
    history, runtime = [], []
    for t, features, angle in zip(cache["times"], cache["features"], cache["angles"]):
        tick = time.perf_counter()
        tracker.update(t, angle, features)
        runtime.append((time.perf_counter() - tick) * 1000)
        history.append(
            dict(
                time_s=float(t),
                signal=float(tracker.signal) if np.isfinite(tracker.signal) else None,
                depth_signal=(
                    float(tracker.depth_signal)
                    if np.isfinite(tracker.depth_signal)
                    else None
                ),
                state=tracker.state,
                attempts=len(tracker.events),
                acceptable=sum(e.label == "acceptable" for e in tracker.events),
                uncertain=sum(e.label == "unclear" for e in tracker.events),
            )
        )
    return [asdict(e) for e in tracker.events], history, np.asarray(runtime)
