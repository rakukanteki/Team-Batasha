"""Replay bench: fly the real flight loop (Systems.BasicController.fly) through rocks recorded in the real sim.

    python tools/replay_bench.py FIELD_DIR [FIELD_DIR ...] --seeds 3 [--planner fewest_points] [--burn boost] [--plot]

FIELD_DIR is a lab telemetry folder (debris.csv: true rock centres at 1 Hz from a real Space Teams PRO run,
meta.json: radar/visual radius per rock and the target, shuttle.csv: start state). Unlike tools/offline_bench.py
the rocks move as they really did (start-up push, rock-on-rock collisions, gravity), and the shuttle is a rigid
body behind the jet-selection twin, so attitude, the real thrust per direction and the burn laws are all exercised.

Per seed: random radar noise, a random shift of the field in time (the sim's debris start time varies run to run),
a random thrust error. A hit applies a bounce and a spin to the shuttle, so recovery after a hit is tested too.
The recorded rocks are truth for this bench only; the flight code sees them only through the emulated radar.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from Systems import Core as C                                   # noqa: E402
from Systems import BasicController as BC                      # noqa: E402
from tools.actuator_twin import SCENE, tick, rotation, INERTIA, INERTIA_INV   # noqa: E402
from tools.offline_bench import Radar                          # noqa: E402

RADAR_HZ = 97.0       # entries re-measured per second in the real sim (lab radar log 2026-11-13; 120 Hz nominal)
TRUE_W = np.array([0.0, 0.0, 2.2803e-4])            # the real sim's frame terms (see Core.FRAME); bench truth, always on
TRUE_B = np.array([0.00215, -0.00100, 0.00148])
OFF_AXIS = np.radians(35.0)  # thrust this far off body +z counts as off-axis fuel
HULL = 4.5            # m the shuttle reaches beyond its centre (measured: contact at 4.6 m past the rock's radius)


def quat_to_R(q):
    """Body axes in the world (columns) from the sim's scalar-last quaternion (Quat_WRT the Phobos frame)."""
    x, y, z, w = q
    M = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return M.T                                                    # the sim's quaternion maps world -> body


class Field:
    def __init__(self, folder):
        folder = Path(folder)
        self.meta = json.loads((folder / "meta.json").read_text())
        D = np.loadtxt(folder / "debris.csv", delimiter=",")
        self.t = D[:, 0]; self.P = D[:, 1:].reshape(len(D), -1, 3)
        self.vis = np.array(self.meta["visual_radius"]); self.dia = np.array(self.meta["radar_diameter"])
        self.T = np.array(self.meta["target_r"])
        s = np.genfromtxt(folder / "shuttle.csv", delimiter=",", names=True, max_rows=5)
        self.r0 = np.array([s["rx"][0], s["ry"][0], s["rz"][0]])
        self.q0 = np.array([s["qx"][0], s["qy"][0], s["qz"][0], s["qw"][0]])
        self.name = folder.name

    def at(self, t):
        """True rock positions and velocities at field time t (linear between samples, ballistic past the end)."""
        t = max(t, self.t[0])
        if t >= self.t[-1]:
            v = (self.P[-1] - self.P[-2]) / (self.t[-1] - self.t[-2]); dt = t - self.t[-1]
            a = C.grav(self.P[-1]) + TRUE_B - 2 * np.cross(TRUE_W, v)
            return self.P[-1] + v * dt + 0.5 * a * dt * dt, v + a * dt
        k = int(np.searchsorted(self.t, t)) - 1; k = min(max(k, 0), len(self.t) - 2)
        a = (t - self.t[k]) / (self.t[k + 1] - self.t[k]); v = (self.P[k + 1] - self.P[k]) / (self.t[k + 1] - self.t[k])
        return self.P[k] + a * (self.P[k + 1] - self.P[k]), v


class ReplaySim:
    """Interfaces.Sim over a recorded field: rigid shuttle + jet twin, emulated radar, collisions with bounce."""

    def __init__(self, field, seed, hz=30.0, tmax=420.0, kick=True):
        rng = np.random.default_rng(seed); self.rng = rng; self.f = field
        self.dt = 1 / hz; self.t = 0.0; self.tmax = tmax; self.kick = kick
        self.shift = rng.uniform(-3.0, 3.0); self.thrust = 1 + rng.normal(0, 0.03)
        self.r = field.r0.copy(); self.v = np.zeros(3); self.w = np.zeros(3); self.R = quat_to_R(field.q0)
        self.used = 0.0; self.off_axis = 0.0; self.cmd = (np.zeros(3), np.zeros(3)); self.lines = []
        self.damage = 0.0; self.hits = []; self.cool = {}; self.clear = np.inf; self.arrived_t = None; self.trace = []
        P, V = self._rocks(); self.radar_em = Radar(rng, P, V, field.dia, self.r, self.v); self.racc = 0.0

    def _rocks(self):
        return self.f.at(self.t + self.shift)

    def wait(self):
        acc, rate, _ = tick(*self.cmd); acc = acc * self.thrust
        lin = self.cmd[0]; nl = np.linalg.norm(lin)
        if nl > 1e-9 and lin[2] < np.cos(OFF_AXIS) * nl:                   # thrust far off body +z (FINDINGS 7.17)
            self.off_axis += rate * self.thrust * self.dt
        self.w = self.w + (acc[3:] - INERTIA_INV @ np.cross(self.w, INERTIA @ self.w)) * self.dt
        self.v = self.v + (self.R @ acc[:3] + C.grav(self.r) + TRUE_B - 2 * np.cross(TRUE_W, self.v)) * self.dt; self.r = self.r + self.v * self.dt
        if np.linalg.norm(self.w) > 0:
            self.R = self.R @ rotation(self.w, np.linalg.norm(self.w) * self.dt)
        self.used += rate * self.thrust * self.dt; self.t += self.dt
        P, V = self._rocks()
        d = np.linalg.norm(P - self.r, axis=1); gap = d - self.f.vis - HULL
        self.clear = min(self.clear, float(gap.min()))
        for i in np.flatnonzero(gap < 0):
            if self.cool.get(int(i), -1) > self.t:
                continue
            n = (self.r - P[i]) / d[i]; vrel = self.v - V[i]; vn = vrel @ n
            e = 0.5 * C.MASS * float(vrel @ vrel)
            self.damage += e; self.hits.append((round(self.t, 1), int(i), round(e)))
            if self.kick and vn < 0:
                self.v = self.v - 1.5 * vn * n                                     # bounce off, restitution 0.5
                self.w = self.w + self.rng.normal(0, 0.2, 3)                       # and a tumble
            self.cool[int(i)] = self.t + 2.0
        self.racc += RADAR_HZ * self.dt                                            # the real radar: ~97 entries/s
        while self.racc >= 1:
            self.radar_em.tick(self.t, P, V, self.f.dia, self.r, self.v); self.racc -= 1
        damage_failed = self.damage > SCENE.scoring["DamageThreshold_J"]
        if (not damage_failed and self.arrived_t is None
                and np.linalg.norm(self.r - self.f.T) <= SCENE.scoring["KeepOutRadius_m"]):
            self.arrived_t = self.t
        if int(self.t / self.dt) % int(1 / self.dt) == 0:
            self.trace.append((self.t, *self.r, *self.v, self.used, self.damage))
        stop = self.t >= self.tmax or damage_failed or (self.arrived_t is not None and self.t > self.arrived_t + 1.0)
        return not stop

    def time(self): return self.t
    def state(self): return self.r.copy(), self.v.copy()
    def target(self): return self.f.T.copy()
    def rates(self): return self.w.copy()                                       # the twin keeps w in body axes
    def to_body(self, vec): return self.R.T @ np.asarray(vec, float)
    def fuel(self): return self.used
    def radar(self): return [m.copy() for m in self.radar_em.published]
    def command(self, lin, ang): self.cmd = (np.array(lin, float), np.array(ang, float))
    def capture(self, top, yaw_deg, pitch_deg, fov_deg=60.0): return None
    def log(self, msg, error=False): self.lines.append((round(self.t, 1), ("ERROR " if error else "") + msg))


class TruthWorld:
    """Oracle tracker for the bench only: the true current rock states (what a perfect sensor would give)."""

    def __init__(self, sim):
        self.sim = sim; self.n = len(sim.f.dia); self.keys = {}

    def read(self, entries, t_now, own_at):
        return []

    def radius(self):
        return self.sim.f.dia / 2

    def predict(self, t):
        P, V = self.sim.f.at(t + self.sim.shift)
        hard = C.SHAPE * self.sim.f.dia / 2 + C.R_SH
        return P, V, hard, np.full(self.n, 0.5), np.full(self.n, 0.01)


def run(field, seed, planner=None, burn=None, hz=30.0, tmax=420.0, verbose=False, oracle=False, config=None):
    sim = ReplaySim(field, seed, hz=hz, tmax=tmax)
    keep = BC.RADAR_EVERY; BC.RADAR_EVERY = max(1, int(round(hz / 5)))
    wall = time.time()
    try:
        BC.fly(sim, planner=planner, burn=burn, threaded=False, tracker=TruthWorld(sim) if oracle else None,
               config=config, log_run=False)
    finally:
        BC.RADAR_EVERY = keep
    T = sim.arrived_t; dmg = min(sim.damage, C.DMAX); fuel = sim.used
    score = 100 * (120 / T - 2 * dmg / C.DMAX - fuel / 75 + 0.5) if T else float("nan")
    if verbose:
        for ln in sim.lines:
            print("   ", ln)
    return dict(field=field.name[7:20], seed=seed, arrived=T is not None, T=round(T, 1) if T else None, score=round(score, 1),
                fuel=round(fuel, 1), off_axis=round(sim.off_axis, 1), damage=round(dmg), hits=sim.hits[:4], min_clear=round(sim.clear, 1),
                end_dist=round(float(np.linalg.norm(sim.r - field.T)), 1), errors=sum("ERROR" in l[1] for l in sim.lines), replans=sum(l[1].startswith("plan t=") for l in sim.lines),
                wall_s=round(time.time() - wall, 1)), sim


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("fields", nargs="+"); ap.add_argument("--seeds", default="2")
    ap.add_argument("--planner"); ap.add_argument("--burn"); ap.add_argument("--hz", type=float, default=30.0)
    ap.add_argument("--tmax", type=float, default=420.0); ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",") if x] if "," in a.seeds else range(int(a.seeds))
    out = []
    for fd in a.fields:
        f = Field(fd)
        for s in seeds:
            res, _ = run(f, s, a.planner, a.burn, a.hz, a.tmax, a.verbose); out.append(res); print(res, flush=True)
    sc = np.array([o["score"] for o in out])
    print(f"SUMMARY planner={a.planner} burn={a.burn}: n={len(out)} arrived {sum(o['arrived'] for o in out)}/{len(out)} "
          f"| with a hit {sum(o['damage'] > 0 for o in out)} | score mean {np.nanmean(sc) if np.isfinite(sc).any() else float('nan'):.1f} "
          f"min {np.nanmin(sc) if np.isfinite(sc).any() else float('nan'):.1f} | fuel {np.mean([o['fuel'] for o in out]):.1f} "
          f"| min clear {min(o['min_clear'] for o in out):.1f} m")


if __name__ == "__main__":
    main()
