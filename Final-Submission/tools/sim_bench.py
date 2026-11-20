"""Realistic flight bench: the real flight loop (Systems.BasicController.fly) against the field the sim really flew.

Usage:  python tools/sim_bench.py --plan conjunction --seeds 4
        python tools/sim_bench.py --plan velocity_obstacle --fields all --no-log   # every recorded field
        python tools/sim_bench.py --plan 5 --seeds 0,3 --burn direct --verbose

What differs from tools/offline_bench.py (measured in the sim, docs/FINDINGS.md section 6.13):
  - the rocks gain several m/s in their first seconds (tools/sim_field.npz, built by tools/sim_field.py).
    Seed 0 is the recorded field; other seeds give the same rocks a reshuffle of the measured velocities;
  - the shuttle is pushed for 1.4 s before the controller's first tick at t = 4 s;
  - the shuttle is a rigid body on the jet twin, so the commands, the attitude and the fuel of off-axis
    thrust are the jet solver's, not an ideal point mass;
  - the radar measures 105 rocks/s on a clock 1.17 s behind ours;
  - rocks knock each other now and then, mostly early, and change velocity by a few m/s.
Offline testing only: flight code in Systems/ never reads the field.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from Systems import BasicController       # noqa: E402
from Systems import Core as C             # noqa: E402
import actuator_twin as twin               # noqa: E402
import offline_bench as OB                 # noqa: E402

FIELD = ROOT / "tools" / "sim_field.npz"
FIELDS = ROOT / "tools" / "sim_fields"             # one per 1x recording: python tools/sim_field.py --all
T_ACC = 6.0          # s the rocks accelerate before they coast (best fit to the recording)
T_PUSH = 1.37        # s the shuttle is pushed (fits its recorded position and velocity at the first tick)
RADAR_HZ, RADAR_LAG = 105.0, 1.17
KNOCK, KNOCK_T = 0.009, 75.0   # per rock per second at t = 0, and its decay time: 17 % of the recorded rocks changed
#                                velocity by over 1 m/s between 30 s and 60 s, 6 % around 110 s, 1 % after 150 s


class FieldSim(twin.TwinSim):
    """Interfaces.Sim: the jet twin flying through the measured rock field."""

    def __init__(self, seed, seconds=420.0, frame_accel=False, field=None, score_clock_offset_s=0.0):
        """field: a recorded field (tools/sim_fields/*.npz) flown as recorded; without it, seed 0 is tools/sim_field.npz
        as recorded and other seeds reshuffle its velocities."""
        f = np.load(FIELD if field is None else field); P0, V0, D, S, T = OB.load_field(serialized=True); C.configure(S, T)
        legacy_order = twin.SCENE.serialized_debris_order()
        super().__init__(seconds, frame_accel=frame_accel); rng = self.rng = np.random.default_rng(seed)
        self.score_clock_offset_s = score_clock_offset_s
        self.R = f["axes_first"].copy(); self.r = S.copy(); self.goal = T.copy()
        if seed == 0 or field is not None:                          # the recorded field itself
            ok = f["rock"] >= 0
            # The bundled calibration stored indices in the scene's serialized entity order.
            j = f["rock"][ok]
            Vk = np.r_[f["V"][ok], f["V"][~ok]]; Pk = np.r_[f["P"][ok], f["P"][~ok]] + Vk * (T_ACC - float(f["t_ref"]))
            self.P0 = np.r_[P0[j], Pk[ok.sum():] - Vk[ok.sum():] * T_ACC]; self.V0 = np.r_[V0[j], Vk[ok.sum():]]
            self.D = np.r_[D[j], f["D"][~ok]]; push = 1.0
            self.scene_indices = legacy_order[j]
        else:                                                       # same rocks, the measured velocities reshuffled
            Vk = f["V"][rng.permutation(len(f["V"]))][:len(D)] + rng.normal(0, 0.3, (len(D), 3))
            self.P0 = P0; self.V0 = V0; self.D = D; Pk = P0 + 0.5 * (V0 + Vk) * T_ACC; push = rng.uniform(0.85, 1.25)
            self.scene_indices = legacy_order
        self.Pk, self.Vk = Pk, Vk; self.P = self.P0.copy(); self.V = self.V0.copy()
        self.push = f["v_first"] / T_PUSH * push; self.t_first = float(f["t_first"])
        self.size = self.D / 2 * rng.uniform(1.0, 1.3, len(self.D)) + 2.5     # collision radius incl. shuttle; unknown to the flight
        self.thrust = 1 + rng.normal(0, 0.03)                                  # thrust error; unknown to the flight
        self.scan = OB.Radar(rng, self.P, self.V, self.D, self.r, self.v); self.due = 0.0
        self.hits = {}; self.clear = np.inf; self.arrived = None
        # Warm the debris and radar to the controller's first tick. For a
        # recorded field, start the controller at the measured state and time.
        step = self.dt
        try:
            while self.t < self.t_first:
                if field is not None:
                    self.dt = min(step, self.t_first - self.t)
                self.wait()
        finally:
            self.dt = step
        if field is not None:
            self.r = f["r_first"].copy()
            self.v = f["v_first"].copy()

    def _rocks(self):
        if self.t < T_ACC:                                                     # cubic from the simconf state to the coast
            s = self.t / T_ACC; a, b, c, d = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
            self.P = a * self.P0 + b * T_ACC * self.V0 + c * self.Pk + d * T_ACC * self.Vk
            self.V = ((6 * s**2 - 6 * s) * self.P0 + (-6 * s**2 + 6 * s) * self.Pk) / T_ACC \
                + (3 * s**2 - 4 * s + 1) * self.V0 + (3 * s**2 - 2 * s) * self.Vk
        else:
            hit = self.rng.random(len(self.D)) < KNOCK * np.exp(-self.t / KNOCK_T) * self.dt     # rocks knock each other
            if hit.any():
                self.V[hit] += self.rng.normal(0, 1.5, (int(hit.sum()), 3))
            self.V = self.V + C.accel(self.P, self.V, self.frame_accel) * self.dt; self.P = self.P + self.V * self.dt

    def wait(self):
        kick = self.push if self.t < T_PUSH else 0.0
        self.integrate(self.thrust, kick)
        self._rocks()
        self.due += RADAR_HZ * self.dt
        while self.due >= 1.0 and self.t > RADAR_LAG:
            self.due -= 1.0; self.scan.tick(self.t - RADAR_LAG, self.P, self.V, self.D, self.r, self.v)
        d = np.linalg.norm(self.P - self.r, axis=1); self.clear = min(self.clear, float((d - self.size).min()))
        for i in np.flatnonzero(d < self.size):
            self.hits.setdefault(int(i), 0.5 * C.MASS * np.linalg.norm(self.v - self.V[i]) ** 2)
        if sum(self.hits.values()) > twin.SCENE.scoring["DamageThreshold_J"]:
            return False
        if np.linalg.norm(self.r - self.goal) <= twin.SCENE.scoring["KeepOutRadius_m"]:
            self.arrived = self.t; return False
        return self.t < self.end

    def radar(self):
        return [m.copy() for m in self.scan.published]

    def log_scene(self, logger):
        if not hasattr(self, "_attitudes"):
            self._attitudes = twin.SCENE.debris_attitudes(self.scene_indices)
        count = len(self.scene_indices)
        logger.log_scene(self.t, self.P[:count], self.D[:count], self._attitudes.at(self.t))

    def score_summary(self):
        # Use the scene clock unless a matched engine run supplies a measured offset.
        return self._score_summary(self.arrived, clock_offset=self.score_clock_offset_s, hits=self.hits.values(),
                                   clearance=float(self.clear) if np.isfinite(self.clear) else None)


def run(plan, seed, burn=None, verbose=False, log_dir=None, flush_immediately=False, log_run=True, frame_accel=False, field=None):
    wall = time.time(); sim = FieldSim(seed, frame_accel=frame_accel, field=field)
    flight = BasicController.fly(sim, planner=plan, burn=burn, threaded=False,
                                 log_dir=log_dir, flush_immediately=flush_immediately,
                                 log_run=log_run, seed=seed, score_summary=sim.score_summary)
    summary = sim.score_summary()
    score = summary["score"] if summary["score"] is not None else float("nan")
    if verbose:
        for line in sim.lines:
            print("   ", line)
    main = sum(1 for l in sim.lines if l.startswith("tm ") and "mode=main" in l) * 2
    out = dict(plan=plan, seed=seed, score=round(float(score), 1), arrived=summary["arrived"],
               T=round(summary["scoring_time_s"], 1), fuel=round(float(sim.used), 1), hits=summary["hits"],
               damage_kJ=round(summary["damage_kJ"], 1), min_clear=round(sim.clear, 1),
               plan_changes=len(flight.events), max_points=max([e[2] for e in flight.events], default=0),
               main_burn_s=main, wall_s=round(time.time() - wall, 1))
    if flight.run_dir is not None:
        out["log_dir"] = str(flight.run_dir)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", type=lambda x: int(x) if x.isdigit() else x, default=BasicController.PLANNER)
    ap.add_argument("--burn", default=None, help="a name in Actuation.BURNS (default: the controller's)")
    ap.add_argument("--seeds", default="3", help="a count (5) or a list (0,7,11)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--frame-accel", action="store_true", help="apply the rotating-frame acceleration model to the shuttle and debris")
    ap.add_argument("--flush-immediately", action="store_true", help="flush run.log immediately instead of buffering in memory")
    ap.add_argument("--log-dir", default=None, help="base directory for plan_name__timestamp run logs (default: flightlogs)")
    ap.add_argument("--no-log", action="store_true", help="disable run log / json output")
    ap.add_argument("--fields", help="'all' for tools/sim_fields, or a single fitted field NPZ")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",") if x] if "," in a.seeds else range(int(a.seeds))
    if a.fields == "all":
        jobs = [(0, f) for f in sorted(FIELDS.glob("*.npz"))]
    else:
        jobs = [(s, Path(a.fields) if a.fields else None) for s in seeds]
    out = []
    for s, field in jobs:
        out.append(run(a.plan, s, a.burn, a.verbose, log_dir=a.log_dir,
                       flush_immediately=a.flush_immediately, log_run=not a.no_log,
                       frame_accel=a.frame_accel, field=field))
        print(out[-1], flush=True)
    sc = np.array([o["score"] for o in out])
    print(f"SUMMARY plan {a.plan}: n={len(out)} mean {np.nanmean(sc):.1f} min {np.nanmin(sc):.1f} max {np.nanmax(sc):.1f} | "
          f"runs with a hit: {sum(o['hits'] > 0 for o in out)} | arrived {sum(o['arrived'] for o in out)}/{len(out)} | "
          f"mean fuel {np.mean([o['fuel'] for o in out]):.1f} | mean T {np.mean([o['T'] for o in out]):.1f} | "
          f"min true clearance {min(o['min_clear'] for o in out):.1f} m")


if __name__ == "__main__":
    main()
