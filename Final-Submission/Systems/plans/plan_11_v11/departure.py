"""Radial lift followed by a body-axis burn and a latched speed cutoff."""
import numpy as np

from Systems import Core as C
from Systems.plans.plan_10_real_calibrated.calibrated import apf_waypoint


class AxisDeparture:
    def __init__(self, cfg):
        self.cfg = cfg
        self.lifted = False
        self.cutoff_time = None
        self.peak_speed = 0.0

    def decide(self, t, r, v, rocks, emergency):
        speed = np.linalg.norm(v)
        self.peak_speed = max(self.peak_speed, float(speed))
        if self.cutoff_time is None and speed >= self.cfg["accel_till_speed"]:
            self.cutoff_time = t
        if self.cutoff_time is not None:
            return "coast", np.zeros(3)
        if emergency is not None:
            return "trim", emergency

        radius = np.linalg.norm(r)
        if radius - np.linalg.norm(C.S) >= self.cfg["lift_height_m"]:
            self.lifted = True
        if not self.lifted:
            return "trim", r / radius

        delta = C.T - r
        distance = max(np.linalg.norm(delta), 1e-6)
        toward = delta / distance
        closing = v @ toward
        # Cancel transverse velocity over the remaining flight. Without this
        # correction, even the radial lift can leave the coast outside the sphere.
        tgo = distance / max(closing, 5.0)
        gravity = C.accel(r, v, frame=True)
        lateral = -3.0 * (v - closing * toward) / max(tgo, 1.0)
        lateral -= gravity - (gravity @ toward) * toward
        aim = toward + lateral / C.A_BURN
        if rocks is not None:
            waypoint = apf_waypoint({"r": r, "v": v}, rocks, self.cfg)
            if waypoint is not None:
                direction = waypoint - r
                aim += (direction - (direction @ toward) * toward) / max(direction @ toward, 1.0)
        return "main", aim / max(np.linalg.norm(aim), 1e-9)
