"""Climb clear of the launch area while slewing body +X toward docking."""
import numpy as np

from Systems import Core as C


class ClearanceLaunch:
    def __init__(self, cfg):
        self.cfg = cfg
        self.done = False
        self.started = None
        self.slew_started = None
        self.timed_out = False

    def command(self, r, v, acquired, emergency, t=0.0):
        if self.started is None:
            self.started = t
        up = C.S / np.linalg.norm(C.S)
        height = (r - C.S) @ up
        climb = v @ up
        clearance = self.cfg["launch_clearance_m"]
        clear = height >= clearance and climb >= 0.0
        if clear and self.slew_started is None:
            self.slew_started = t
        still_clear = height + 5.0 * min(climb, 0.0) >= clearance
        slew_expired = self.slew_started is not None and t - self.slew_started >= self.cfg["launch_slew_limit_s"]
        launch_expired = t - self.started >= self.cfg["launch_deadline_s"]
        if (clear and acquired) or (still_clear and (slew_expired or launch_expired)):
            self.done = True
            self.timed_out = not (clear and acquired)
            return None
        if launch_expired:
            return "launch_failed", np.zeros(3)

        full_slew = clear or (self.slew_started is not None and still_clear)
        if full_slew and emergency is not None:
            # Give the dodge the available thrust and suspend the target slew.
            # Keep the push tangent to the launch surface if the guard asks to descend.
            escape = emergency - min(emergency @ up, 0.0) * up
            length = np.linalg.norm(escape)
            if length > 1e-6:
                return "launch_evade", escape / length

        early_slew = (not full_slew and height >= self.cfg["launch_early_slew_m"]
                      and climb >= self.cfg["launch_early_slew_min_climb_mps"])
        mode = "launch_lift"
        if full_slew:
            mode = "launch_slew"
        elif early_slew:
            mode = "launch_slew_climb"

        # Keep climbing through the turn. A guard dodge can add lateral
        # acceleration, but cannot ask for a descent inside the launch area.
        vertical = 0.55 * (self.cfg["launch_climb_mps"] - climb) - C.accel(r, v, frame=True) @ up
        vertical = np.clip(vertical, 0.0 if not clear else -0.04, 0.18)
        sideways = v - climb * up
        lateral = -0.25 * sideways
        if self.cfg.get("launch_coast_climb") and emergency is None:
            if 0.65 <= climb <= 1.05:
                vertical = 0.0
            if np.linalg.norm(sideways) <= 0.15:
                lateral = np.zeros(3)
        if emergency is not None:
            lateral += 0.12 * (emergency - (emergency @ up) * up)
        magnitude = np.linalg.norm(lateral)
        if magnitude > 0.15:
            lateral *= 0.15 / magnitude
        return mode, vertical * up + lateral
