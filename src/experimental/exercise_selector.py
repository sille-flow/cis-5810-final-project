"""Automatically select among exercise-specific reference templates."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.experimental.rep_counting import PhaseCycleRepCounter, RepCounter, SquatDepthRepCounter
from src.experimental.realtime_matcher import MatchResult, RealtimeReferenceMatcher
from src.experimental.template import MultiViewTemplates


class AutoExerciseSelector:
    """Run candidate phase followers and select the best matching motion."""

    def __init__(self, templates: dict[str, MultiViewTemplates], switch_margin: float = 0.15):
        if not templates:
            raise ValueError("At least one exercise template is required")
        self.matchers: dict[str, RealtimeReferenceMatcher] = {}
        for name, bank in templates.items():
            counter: RepCounter = SquatDepthRepCounter() if name.lower().rstrip("s") == "squat" else PhaseCycleRepCounter()
            self.matchers[name] = RealtimeReferenceMatcher(bank, rep_counter=counter)
        self.switch_margin = switch_margin
        self.active: str | None = None
        self._scores: dict[str, float] = {name: float("inf") for name in templates}

    def reset(self) -> None:
        for matcher in self.matchers.values():
            matcher.reset()
        self.active = None
        self._scores = {name: float("inf") for name in self.matchers}

    def update(self, features: np.ndarray | None, view: str = "side") -> MatchResult:
        results = {name: matcher.update(features, view) for name, matcher in self.matchers.items()}
        for name, result in results.items():
            old = self._scores[name]
            current = result.distance
            self._scores[name] = current if not np.isfinite(old) else 0.75 * old + 0.25 * current
        candidate = min(self._scores, key=self._scores.get)
        if self.active is None:
            self.active = candidate
        elif candidate != self.active:
            # A margin avoids frequent label changes when two templates score similarly.
            active_score, candidate_score = self._scores[self.active], self._scores[candidate]
            if candidate_score + self.switch_margin < active_score:
                self.active = candidate
        return replace(results[self.active], exercise=self.active)
