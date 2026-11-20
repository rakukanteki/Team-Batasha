"""Timed steering, coasting and body-command segments shared by planning and flight."""
from dataclasses import dataclass, replace

import numpy as np

from Systems import Core as C

# Legal SetCommand pairs and their delivered acceleration/fuel in the 60 Hz
# offline selector twin for STU4-Sim@0d753a2. Angular delivery is about body Y.
# An engine run is needed to validate the response under different tick times.
BODY_COMMANDS = np.array([
    # lin x, lin y, lin z, ang y, accel x, accel y, accel z, alpha y, kg/s
    [0.5, 0, 0, 0, 0.1745625, 0, 0, 0, 0.7545171],
    [-0.5, 0, 0, 0, -0.0965596, 0, 0, 0, 0.4173661],
    [0, 0.5, 0, 0, 0, 0.1431810, 0, 0, 0.6188756],
    [0, -0.5, 0, 0, 0, -0.0964447, 0, 0, 0.4168679],
    [0.5, 0, 0, 0.02, 0.0968059, 0, 0, 0.0038722, 0.4192531],
])
ZERO = np.zeros(3)


@dataclass(frozen=True)
class Segment:
    duration: float
    direction: tuple[float, float, float] | None = None
    body_command: int | None = None
    pulse_accel: float = 0.0

    def control(self):
        if self.body_command is not None:
            lin = BODY_COMMANDS[self.body_command, :3]
            return "body_command", lin
        if self.direction is not None:
            if self.pulse_accel:
                return "pulse", self.pulse_accel * np.asarray(self.direction)
            return "main", 4.0 * np.asarray(self.direction)
        return "coast", ZERO


@dataclass(frozen=True)
class Maneuver:
    start: float
    segments: tuple[Segment, ...]
    dt: float
    path: np.ndarray
    velocity: np.ndarray
    fallback: tuple
    frame: bool
    initial_body: np.ndarray | None = None
    review_horizon: float = 0.0

    def remaining(self, t):
        elapsed = max(t - self.start, 0.0)
        remaining = []
        for segment in self.segments:
            if elapsed < segment.duration:
                remaining.append(replace(segment, duration=segment.duration - elapsed))
            elapsed -= min(elapsed, segment.duration)
        return tuple(remaining)

    def action(self, t):
        elapsed = t - self.start
        for segment in self.segments:
            if elapsed < segment.duration:
                return segment, *segment.control()
            elapsed -= segment.duration
        return None

    def rebase(self, t, r, v):
        """Move the command start to adoption after a short planning delay."""
        age = t - self.start
        dr, dv = r - self.path[0], v - self.velocity[0]
        if (age < 0.0 or age > 1.2 or np.linalg.norm(dr - age * self.velocity[0]) > 3.0
                or np.linalg.norm(dv) > 0.5):
            return None
        tau = np.arange(len(self.path))[:, None] * self.dt
        path = self.path + dr + tau * dv
        if np.linalg.norm(path - C.T, axis=1).min() > C.WIN - 1.0:
            return None
        return replace(self, start=t, path=path, velocity=self.velocity + dv)

    def check(self, t, r, v, rocks):
        """Check a short measured-state corridor before executing or retaining a maneuver."""
        age = t - self.start
        times = np.arange(len(self.path)) * self.dt
        if age < 0.0 or age >= times[-1] or (not self.review_horizon and age > 4.5):
            return False
        horizon = self.review_horizon or 12.0
        step = 0.5 if self.review_horizon else 0.25
        sample = age + np.arange(0, min(horizon, times[-1] - age) + 0.01, step)
        path = np.stack([np.interp(sample, times, axis) for axis in self.path.T], axis=1)
        velocity = np.array([np.interp(age, times, axis) for axis in self.velocity.T])
        dr, dv = r - path[0], v - velocity
        if np.linalg.norm(dr) > 3.0 or np.linalg.norm(dv) > 0.5:
            return False
        if rocks is None or not len(rocks[0]):
            return True
        rp, rv, hard, sp, sv = rocks
        tau = (sample - age)[:, None]
        debris = rp + tau[..., None] * rv + 0.5 * tau[..., None] ** 2 * C.accel(rp, rv, frame=self.frame)
        path += dr + tau * dv
        need = hard + 2.0 * np.hypot(sp, tau * sv) + 1.0
        return bool(np.all(np.linalg.norm(path[:, None] - debris, axis=2) > need))


def maneuver_of(plan):
    return plan[3] if plan is not None and len(plan) > 3 else None
