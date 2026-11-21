"""Online reference matching for live video.

Each new frame is compared only to a neighborhood of the current phase on the
template for the detected camera view (falling back to any available template,
since the features are yaw invariant). Head-on, noisier depth-based channels are
down-weighted and hip height is trusted more.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.pose.features import VIEW_WEIGHTS
from src.experimental.template import MultiViewTemplates, PhaseTemplate
from src.experimental.rep_counting import RepCounter, SquatDepthRepCounter


@dataclass
class MatchResult:
    phase_index: int
    phase: float
    distance: float
    residuals: np.ndarray
    z_scores: np.ndarray
    reps: int
    feedback: str
    aligned: bool
    view: str = "side"
    depth: float = 0.0  # 0 = standing, 1 = as deep as the reference bottom
    exercise: str = ""
    feature_names: tuple[str, ...] = ()


class RealtimeReferenceMatcher:
    def __init__(
        self,
        template: PhaseTemplate | MultiViewTemplates,
        search_radius: int = 4,
        z_threshold: float = 2.0,
        smooth: float = 0.35,
        min_bottom_phase: float = 0.35,
        max_bottom_phase: float = 0.65,
        direction_weight: float = 1.50,
        direction_scale: float = 0.015,
        bottom_depth: float = 0.55,
        stand_depth: float = 0.25,
        bottom_hold_frames: int = 3,
        rep_counter: RepCounter | None = None,
    ):
        # Rep counting uses its own hysteresis state machine on "depth", a
        # 0..1 score of how far down the live pose is relative to the template's
        # own standing and bottom poses (hip drop + BOTH knees flexed).
        #   bottom_depth: depth that counts as having reached the bottom.
        #   stand_depth: depth below which the person is "back up".
        #   bottom_hold_frames: frames in a row at the bottom (rejects spikes).
        # Lower bottom_depth if good reps are missed; raise it (or bottom_hold_frames)
        # if shallow bobs or walking get counted.
        self.rep_counter = rep_counter or SquatDepthRepCounter(bottom_depth, stand_depth, bottom_hold_frames)
        # direction_weight: cost bonus/penalty for moving the way the template
        # moves at a candidate phase. direction_scale: per-frame progress speed
        # (as a fraction of the template's range) that counts as "clearly moving".
        self.direction_weight = direction_weight
        self.direction_scale = direction_scale
        if isinstance(template, PhaseTemplate):
            template = MultiViewTemplates(views={template.view: template})
        self.bank = template
        self.search_radius = search_radius
        self.z_threshold = z_threshold
        self.smooth = smooth
        self.min_bottom_phase = min_bottom_phase
        self.max_bottom_phase = max_bottom_phase

        self.active_view = next(iter(self.bank.views))
        self.template = self.bank.get(self.active_view)
        self.phase_index = 0
        self.smoothed: np.ndarray | None = None
        self.reps = 0
        self._visited_bottom = False
        self._view_votes: list[str] = []
        self._prev_progress: float | None = None
        self._velocity = 0.0
        self._progress_cache: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
        self._visited_bottom = False
        self.depth = 0.0

    def _progress_curve(self, view: str, template: PhaseTemplate):
        """Template progress per phase, its cyclic slope, and its total range."""
        cached = self._progress_cache.get(view)
        if cached is None or len(cached[0]) != template.n_phases:
            prog = np.asarray(template.project(template.mean), dtype=np.float64)
            slope = np.roll(prog, -1) - np.roll(prog, 1)
            span = float(prog.max() - prog.min()) or 1.0
            cached = (prog, slope, span)
            self._progress_cache[view] = cached
        return cached

    def reset(self) -> None:
        self.rep_counter.reset()
        self.depth = 0.0
        self.phase_index = 0
        self.smoothed = None
        self.reps = 0
        self._visited_bottom = False
        self._view_votes = []
        self._prev_progress = None
        self._velocity = 0.0

    def _stabilize_view(self, view: str) -> str:
        self._view_votes.append(view)
        self._view_votes = self._view_votes[-12:]
        chosen = max(set(self._view_votes), key=self._view_votes.count)
        if chosen != self.active_view:
            old_template = self.template
            self.active_view = chosen
            new_template = self.bank.get(chosen)
            # Only the weights change when the view label flips but the bank
            # falls back to the same template; keep tracking state untouched.
            if new_template is not old_template:
                old_n = old_template.n_phases
                n = new_template.n_phases
                self.template = new_template
                # Keep the same fractional phase when templates differ in length.
                self.phase_index = int(round(self.phase_index * n / max(old_n, 1))) % n
                self._prev_progress = None
                self._velocity = 0.0
        return self.active_view

    def update(self, features: np.ndarray | None, view: str = "side") -> MatchResult:
        view = self._stabilize_view(view)
        template = self.template
        n = template.n_phases
        # Conservative policy: incomplete geometry is not a zero-valued pose.
        if features is not None:
            features = np.asarray(features, dtype=np.float64)
            if features.shape != (len(template.feature_names),):
                raise ValueError("Feature vector does not match template schema")
            if not np.isfinite(features).all():
                features = None
        if features is None:
            # Do not complete a repetition across an unobserved interval.
            saved_reps = self.reps
            self.rep_counter.reset()
            self.rep_counter.reps = saved_reps
            self.smoothed = None
            self._prev_progress = None
            self._velocity = 0.0
            return MatchResult(
                phase_index=self.phase_index,
                phase=self.phase_index / n,
                distance=float("inf"),
                residuals=np.zeros(len(template.feature_names)),
                z_scores=np.zeros(len(template.feature_names)),
                reps=self.reps,
                feedback="Pose unclear",
                aligned=False,
                view=view,
                feature_names=template.feature_names,
            )

        features = np.nan_to_num(np.asarray(features, dtype=np.float64), nan=0.0)
        if self.smoothed is None:
            self.smoothed = features
        else:
            self.smoothed = (1.0 - self.smooth) * self.smoothed + self.smooth * features

        # Known feature names retain their empirically tuned view weights;
        # additional exercise/backend features get a neutral weight.
        view_weights = VIEW_WEIGHTS.get(view, VIEW_WEIGHTS["oblique"])
        canonical = ("left_knee", "right_knee", "left_hip", "right_hip", "torso_incline", "hip_height", "knee_width")
        weight_by_name = dict(zip(canonical, view_weights))
        weights = np.array([weight_by_name.get(name, 1.0) for name in template.feature_names])
        if len(weights) != len(self.smoothed):
            weights = np.ones_like(self.smoothed)

        # Direction cue. A squat looks almost the same going down and coming up,
        # so pose alone cannot tell descent from ascent and the phase can bounce
        # back and forth. Compare the live progress velocity with the template's.
        prog, slope, span = self._progress_curve(self.active_view, template)
        p_live = float(template.project(self.smoothed))
        if self._prev_progress is not None:
            dv = p_live - self._prev_progress
            self._velocity = 0.7 * self._velocity + 0.3 * dv
        self._prev_progress = p_live
        speed_scale = max(self.direction_scale * span, 1e-9)

        best_i = self.phase_index
        best_cost = float("inf")
        for delta in range(-self.search_radius, self.search_radius + 1):
            i = (self.phase_index + delta) % n
            mean, std = template.pose_at_index(i)
            residual = (self.smoothed - mean) / std
            cost = float(np.mean(weights * residual**2))
            # Prefer moving forward through the cycle; on ambiguous plateaus
            # (standing, bottom) this keeps the phase from jittering backward.
            cost += -0.03 * max(delta, 0) + 0.06 * max(-delta, 0)
            agree = float(np.tanh(self._velocity * np.sign(slope[i]) / speed_scale))
            cost -= self.direction_weight * agree
            if cost < best_cost:
                best_cost = cost
                best_i = i

        self.phase_index = best_i
        phase = best_i / n
        self.reps, self.depth = self.rep_counter.update(template, self.smoothed, phase=phase)
        self._visited_bottom = getattr(self.rep_counter, "_visited_bottom", False)

        mean, std = template.pose_at_index(best_i)
        residuals = self.smoothed - mean
        z_scores = residuals / std
        weighted_z = z_scores * np.sqrt(weights)
        feedback = self._feedback(weighted_z, view)
        return MatchResult(
            phase_index=best_i,
            phase=phase,
            distance=float(np.sqrt(max(best_cost, 0.0))),
            residuals=residuals,
            z_scores=z_scores,
            reps=self.reps,
            feedback=feedback,
            aligned=True,
            view=view,
            depth=self.depth,
            feature_names=template.feature_names,
        )

    def _feedback(self, z_scores: np.ndarray, view: str) -> str:
        worst = int(np.argmax(np.abs(z_scores)))
        mag = abs(float(z_scores[worst]))
        if mag < self.z_threshold:
            if self._visited_bottom:
                return "On track"
            return f"Follow the {view} reference"
        name = self.template.feature_names[worst]
        direction = "higher" if z_scores[worst] > 0 else "lower"
        readable = name.replace("_", " ")
        return f"{readable}: {direction} than reference"
