"""Offline twin of the sim's jet selection (submodules/STU4-Sim/Systems/JetSelectionActuator.py).

Same math and the same solver call, so SetCommand laws can be tested without the sim:
    acc, fuel_rate, jets = tick(lin, ang)      # achieved [linear(3), angular(3)] acceleration, kg/s, jets used

TwinSim puts the same jets behind the Sim interface (Systems/Interfaces.py), so the real flight loop
(Systems.BasicController.fly) can run offline with its commands, attitude and all.

Usage:  python tools/actuator_twin.py          # prints the per-axis thrust table and the boost sector
"""
import csv
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import optimize
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tools.upstream_scene import SIM_ROOT, Scene

SCENE = Scene()
ACTUATOR = SCENE.systems["JetSelectionActuator"]
JETS = SIM_ROOT / ACTUATOR["JetConfigName"].removeprefix("Local/Repos/STU4-Sim/")
MAX_ACCEL = SCENE.shuttle["Thrusters"]["MaxAccel_m_s2"]
MAX_ANG_ACCEL = SCENE.shuttle["Thrusters"]["MaxAngAccel_rad_s2"]
ACTUATOR_HZ = ACTUATOR["LoopFreqHz"]
MASS = SCENE.shuttle["Dynamics"]["Mass"]
INERTIA = np.array(SCENE.shuttle["Dynamics"]["Inertia_BodyFrame"][1:]).reshape(3, 3)
INERTIA_INV = np.linalg.inv(INERTIA)

_rows = [list(map(float, r)) for r in list(csv.reader(open(JETS)))[1:]]
A = []                                         # per jet: [linear accel(3), angular accel(3)] at full thrust
for x, y, z, nx, ny, nz, F in _rows:
    n = np.array([nx, ny, nz]); n /= np.linalg.norm(n); f = -F * n
    A.append(np.r_[f / MASS, INERTIA_INV @ np.cross([x, y, z], f)])
A = np.array(A)
FUEL_PER_SECOND = np.linalg.norm(MASS * A[:, :3], axis=1) / (80 * 9.8)


def _jet_times(command):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return optimize.linprog(np.ones(len(A)), A_eq=A.T, b_eq=command, method="revised simplex").x


@lru_cache(maxsize=32)
def _axis_jet_times(command):
    return _jet_times(command)


def tick(lin, ang, dt=1 / ACTUATOR_HZ):
    lin = np.array(lin, float); ang = np.array(ang, float)
    if np.linalg.norm(lin) > MAX_ACCEL:
        lin = MAX_ACCEL * lin / np.linalg.norm(lin)
    if np.linalg.norm(ang) > MAX_ANG_ACCEL:
        ang = MAX_ANG_ACCEL * ang / np.linalg.norm(ang)
    b = np.r_[lin, ang]
    if np.linalg.norm(b) < 1e-9:
        return np.zeros(6), 0.0, 0
    if np.count_nonzero(lin) == 1 and not np.any(ang):
        x = _axis_jet_times(tuple(b))
    else:
        x = _jet_times(b)
    mx = max(x.max(), 0.0)
    if mx < 1e-12:
        return np.zeros(6), 0.0, 0
    on = np.clip(x, 0, None) * min(dt, mx) / mx            # the busiest jet fires for at most the whole tick
    fuel = float(on @ FUEL_PER_SECOND)
    return A.T @ on / dt, fuel / dt, int((x > 1e-9).sum())


def rotation(axis, angle):
    x, y, z = np.asarray(axis, float) / np.linalg.norm(axis); c, s = np.cos(angle), np.sin(angle); C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def fly_burn(command, tilt_deg, seconds, hz=60.0, tilt_axis=(0.3, 1.0, 0.0)):
    """Run a burn law (d_body, w, state) -> (lin, ang) against the jet solver with rigid-body rotation.
    Returns mean thrust along the wanted direction, kg per m/s, and the final pointing error in degrees."""
    dt = 1 / hz; R = np.eye(3); w = np.zeros(3); state = {"kick": False}
    u = rotation(tilt_axis, np.radians(tilt_deg)) @ np.array([0, 0, 1.0]); dv = np.zeros(3); fuel = 0.0
    for _ in range(int(seconds * hz)):
        lin, ang = command(R.T @ u, w, state)
        acc, rate, _ = tick(lin, ang, dt)
        w = w + (acc[3:] - INERTIA_INV @ np.cross(w, INERTIA @ w)) * dt
        dv += R @ acc[:3] * dt; fuel += rate * dt
        if np.linalg.norm(w) > 0:
            R = R @ rotation(w, np.linalg.norm(w) * dt)
    err = np.degrees(np.arccos(np.clip((R @ [0, 0, 1.0]) @ u, -1, 1)))
    return (dv @ u) / seconds, fuel / (dv @ u), err


class TwinSim:
    """Interfaces.Sim on the jet twin: a rigid shuttle at 60 Hz in Phobos gravity, with no rocks and no camera.
    Starts at the pad with body +z a few degrees off the target line, as in the sim."""

    def __init__(self, seconds, hz=ACTUATOR_HZ, tilt_deg=4.0, scene_attitude=False, frame_accel=False):
        from Systems import Core
        self.grav = Core.grav; self.dt = 1 / hz; self.end = seconds; self.t = 0.0
        self.frame_accel = frame_accel
        self.r = Core.S.copy(); self.v = np.array(SCENE.shuttle["Velocity"][1:], float)
        self.w = np.zeros(3); self.used = 0.0; self.goal = Core.T.copy()
        self.R = np.column_stack([Core.E1, np.cross(Core.SH, Core.E1), Core.SH]) @ rotation((0.3, 1.0, 0.0), np.radians(tilt_deg))
        if scene_attitude:
            self.R = SCENE.body_rotation(SCENE.shuttle).as_matrix()
            self.w = self.R.T @ np.array(SCENE.shuttle["AngVelocity"][1:], float)
        self.cmd = (np.zeros(3), np.zeros(3)); self.lines = []          # R: body axes in the world, as columns

    def wait(self):
        self.integrate()
        return self.t < self.end

    def integrate(self, thrust=1.0, kick=0.0):
        from Systems import Core
        # Hold the scene actuator's average response over this integration step.
        # A coarser offline step must not increase the jet selector's tick length.
        acc, rate, _ = tick(*self.cmd)
        acc *= thrust
        self.w = self.w + (acc[3:] - INERTIA_INV @ np.cross(self.w, INERTIA @ self.w)) * self.dt
        self.v = self.v + (self.R @ acc[:3] + Core.accel(self.r, self.v, self.frame_accel) + kick) * self.dt
        self.r = self.r + self.v * self.dt
        if np.linalg.norm(self.w) > 0:
            self.R = self.R @ rotation(self.w, np.linalg.norm(self.w) * self.dt)
        self.used += rate * self.dt; self.t += self.dt

    def body_quaternion(self):
        return Rotation.from_matrix(self.R).as_quat()

    def time(self): return self.t

    def _score_summary(self, arrival, *, clock_offset=0.0, hits=(), clearance=None):
        elapsed = max(0.0, (self.t if arrival is None else arrival) - clock_offset)
        damage = sum(hits)
        failed = damage > SCENE.scoring["DamageThreshold_J"]
        score = None
        if failed:
            distance = np.linalg.norm(self.r - self.goal)
            initial = np.linalg.norm(SCENE.shuttle["Location"][1:] - self.goal)
            score = SCENE.failure_score(distance, initial, self.used, damage)
        elif arrival is not None:
            score = SCENE.arrival_score(elapsed, self.used, damage)
        return {
            "arrived": arrival is not None and not failed,
            "outcome": "damage_limit" if failed else "arrived" if arrival is not None else "incomplete",
            "scoring_rules": SCENE.scoring,
            "scoring_time_s": elapsed,
            "score": score,
            "hits": len(hits),
            "damage_kJ": damage / 1000.0,
            "min_clear": clearance,
        }

    def state(self): return self.r.copy(), self.v.copy()
    def target(self): return self.goal
    def rates(self): return self.w.copy()
    def to_body(self, vec): return self.R.T @ np.asarray(vec, float)
    def fuel(self): return self.used
    def radar(self): return []
    def command(self, lin, ang): self.cmd = (np.array(lin, float), np.array(ang, float))
    def capture(self, top, yaw_deg, pitch_deg, fov_deg=60.0): return None
    def log(self, msg, error=False): self.lines.append(("ERROR " if error else "") + msg)


if __name__ == "__main__":
    print("zero-torque thrust by body axis (m/s^2, kg per m/s):")
    for name, ax in (("+x", [1, 0, 0]), ("-x", [-1, 0, 0]), ("+y", [0, 1, 0]), ("-y", [0, -1, 0]), ("+z", [0, 0, 1]), ("-z", [0, 0, -1])):
        acc, rate, jets = tick(0.5 * np.array(ax, float), [0, 0, 0]); a = np.linalg.norm(acc[:3])
        print(f"  {name}: {a:.3f}  {rate / a:.2f}  ({jets} jets)")
    print("body +z thrust versus the direction of a 0.002 torque command (the boost sector):")
    for phi in range(0, 360, 30):
        acc, _, _ = tick([0, 0, 0.5], [0.002 * np.cos(np.radians(phi)), 0.002 * np.sin(np.radians(phi)), 0])
        print(f"  {phi:3d} deg: {acc[2]:.3f}")
