"""Original proposal knee-angle state machine (commit 2813950)."""
import math


class SquatTracker:
    def __init__(self, deep_angle=95, start_angle=150, stand_angle=160):
        self.deep_angle, self.start_angle, self.stand_angle = deep_angle, start_angle, stand_angle
        self.reps = self.good_reps = 0
        self.state = 'standing'
        self.min_angle = 180

    def update(self, angle):
        if angle is None or not math.isfinite(angle):
            return self.state, 'Pose unclear'
        if self.state == 'standing':
            if angle < self.start_angle:
                self.state, self.min_angle = 'squatting', angle
        elif self.state == 'squatting':
            self.min_angle = min(self.min_angle, angle)
            if angle > self.stand_angle:
                self.reps += 1
                good = self.min_angle <= self.deep_angle
                self.good_reps += int(good)
                self.state = 'standing'
                return self.state, 'Good depth!' if good else 'Go deeper!'
        return self.state, ('Stand tall' if self.state == 'standing' else
                            'Depth reached' if self.min_angle <= self.deep_angle else 'Keep going lower')
