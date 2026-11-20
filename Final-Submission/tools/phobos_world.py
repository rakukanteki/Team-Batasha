"""The rock field as the real sim moves it, generated from the world definition instead of a recording.

The world comes from the organizer scene (submodules/STU4-Sim/Sims/FINAL_SUBMISSION.simconf: rock states, masses,
radar and visual radii, shuttle and target) and the behaviour of the upstream systems that move the rocks
(submodules/STU4-Sim/Systems/SpawnDebris.py), measured against 42 real runs with recorded rock truth
(codernayeem/STU4-lab fields/, docs/FINDINGS.md §7):

* start: the scene's rock states plus SpawnDebris's start impulse, N(0, 0.02 m/s) per axis from its fixed seed 8193,
  so the same in every run;
* start-up push: every rock accelerates at about 0.43 m/s^2 along -x until SpawnDebris's round-robin first sets its
  gravity (1.6-4 s + 9-12.6 s * index/500), about -3 m/s in total (41 of 42 recorded fields);
* then Phobos point-mass gravity plus the rotating-frame terms (Coriolis and the frame bias, replay_bench.TRUE_W/B),
  which predicts the smooth motion to 0.0004 m/s^2;
* wander: every rock's velocity drifts slowly off the smooth model (an Ornstein-Uhlenbeck offset, WANDER_SIGMA per axis,
  correlation time WANDER_TAU), fitted to how far recorded rocks stray in 30 s and 60 s (median 2.3 m and 5.1 m);
* bumps: sudden changes over 1 m/s at 0.079 per rock per minute, sizes drawn from the recorded deciles, random directions.

Rock-on-rock contact is not simulated; the recorded jumps don't line up with rock pairs at contact range.
WorldField has the interface of replay_bench.Field, so the replay loop (rigid shuttle, jet twin, radar, collisions)
flies through it unchanged.
"""
from __future__ import annotations

import numpy as np

from Systems import Core as C
from pathlib import Path

from tools.replay_bench import Field, ReplaySim, TRUE_B, TRUE_W
from tools.upstream_scene import Scene

LAB_RUNS = Path(__file__).resolve().parents[2] / "STU4-lab" / "runs"     # a clone of codernayeem/STU4-lab, restored

SPAWN_SEED = 8193                           # SpawnDebris.py: rng = np.random.default_rng(8193)
SPAWN_DV = 0.02                             # SpawnDebris.py: deltaV_1sigma_m_s
KICK_DIR = np.array([-1.0, 0.0, 0.0])
KICK_ACC = (0.34, 0.43)                     # m/s^2, per run (fits over 7 fields; -3.0 m/s median push over 42)
KICK_ON = (1.6, 4.0)                        # s, when the round-robin starts
KICK_SPAN = (9.0, 12.6)                     # s for the round-robin to reach rock 500
WANDER_SIGMA, WANDER_TAU = 0.08, 30.0       # m/s per axis, s (30 s drift median 2.3 m, as recorded)
BUMP_RATE = 0.079 / 60.0                    # per rock per second, changes over 1 m/s
BUMP_Q = np.array([1.0, 1.13, 1.29, 1.52, 1.84, 2.2, 2.68, 3.25, 3.99, 5.66, 33.07])   # |dv| deciles, m/s


def spawn_impulses(mass):
    """SpawnDebris's start impulses as velocity changes, drawn in its order (linear, then angular, per rock)."""
    rng = np.random.default_rng(SPAWN_SEED); dv = np.zeros((len(mass), 3))
    for i, m in enumerate(mass):
        dv[i] = rng.normal(0.0, m * SPAWN_DV, size=3) / m
        rng.normal(0.0, 0.2, size=3)
    return dv


class WorldField(Field):
    """A rock field sampled at 1 Hz like a recording; seed sets the start-up push timing and the bumps."""

    def __init__(self, seed, scene=None, seconds=430.0, dt=0.1, bumps=True, kick=True,
                 wander=(WANDER_SIGMA, WANDER_TAU), bump_rate=BUMP_RATE):
        sc = scene or Scene(); E = sc.entities; rng = np.random.default_rng(seed)
        P, V, D, S, T = sc.field(); N = len(D)
        rocks = [E[f"Debris{i}"] for i in range(N)]
        mass = np.array([e["Dynamics"]["Mass"] for e in rocks])
        self.vis = np.array([e["Graphics"]["VisualRadius_m"] for e in rocks]); self.dia = D
        self.T = T; self.r0 = S; self.q0 = np.array(sc.shuttle["Rotation"][1:], float)
        self.meta = dict(radar_diameter=D.tolist(), visual_radius=self.vis.tolist(), target_r=T.tolist(), seed=seed)
        self.name = f"world_{seed:06d}_generated"
        V = V + spawn_impulses(mass)
        acc = rng.uniform(*KICK_ACC) if kick else 0.0
        t_grav = rng.uniform(*KICK_ON) + rng.uniform(*KICK_SPAN) * np.arange(N) / N
        n = int(round(seconds / dt)); every = int(round(1.0 / dt))
        ts, Ps = [0.0], [P.copy()]
        sig, tau = wander if bumps else (0.0, 1.0); U = rng.normal(0.0, sig, (N, 3))     # wander offset, stationary start
        keep, push = np.exp(-dt / tau), sig * np.sqrt(1 - np.exp(-2 * dt / tau))
        for k in range(1, n + 1):
            t = k * dt
            a = C.grav(P) + TRUE_B - 2 * np.cross(TRUE_W, V)
            a = np.where((t < t_grav)[:, None], acc * KICK_DIR, a)
            if bumps:
                hit = rng.random(N) < bump_rate * dt
                if hit.any():
                    u = rng.normal(size=(hit.sum(), 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
                    mag = np.interp(rng.random(hit.sum()), np.linspace(0, 1, len(BUMP_Q)), BUMP_Q)
                    V[hit] += mag[:, None] * u
                U = keep * U + push * rng.normal(size=(N, 3))
            V = V + a * dt; P = P + (V + U) * dt
            if k % every == 0:
                ts.append(t); Ps.append(P.copy())
        self.t = np.array(ts); self.P = np.array(Ps)



def recorded_fields():
    """Real-sim rock fields recorded in the lab (folders with debris.csv), oldest first."""
    return sorted(p for p in LAB_RUNS.glob("*") if (p / "debris.csv").is_file()) if LAB_RUNS.is_dir() else []


class WorldSim(ReplaySim):
    """The replay loop (rigid shuttle, jet twin, radar, collisions) over a generated field, or a recorded one,
    with the scene and score hooks that BasicController.fly and the run logger use."""

    def __init__(self, seed, tmax=420.0, hz=30.0, field=None, debris_count=None):
        field = field if field is not None else WorldField(seed, seconds=tmax + 10.0)
        self.scene_indices = np.arange(len(field.dia))
        if debris_count is not None:
            if not 1 <= debris_count <= len(field.dia):
                raise ValueError("debris_count must be between 1 and the full field size")
            # Nested, evenly distributed subset: retained rocks keep their original trajectories.
            import copy
            field = copy.copy(field)
            self.scene_indices = np.linspace(0, len(field.dia) - 1, debris_count, dtype=int)
            field.P = field.P[:, self.scene_indices].copy()
            field.dia = field.dia[self.scene_indices].copy()
            field.vis = field.vis[self.scene_indices].copy()
            field.meta = {**field.meta, "radar_diameter": field.dia.tolist(), "visual_radius": field.vis.tolist(),
                          "original_scene_indices": self.scene_indices.tolist()}
            field.name += f"_{debris_count}rocks"
        super().__init__(field, seed, hz=hz, tmax=tmax)
        self.scene = Scene(); self._att = None
        self._camera_seed = seed
        self._camera = None
        self.camera_frames = 0

    def capture(self, top, yaw_deg, pitch_deg, fov_deg=60.0):
        """Render a synthetic camera observation at the current rigid-body pose."""
        from tools.offline_bench import Camera
        if self._camera is None:
            self._camera = Camera(np.random.default_rng(self._camera_seed + 1000))
        positions, _ = self._rocks()
        self._camera.B = self.R.T
        self._camera.scene = (np.vstack([positions, self.f.T]), np.append(self.f.vis, 40.0), self.r)
        self.camera_frames += 1
        return self._camera.capture(top, yaw_deg, pitch_deg, fov_deg)

    def log_scene(self, logger):
        if self._att is None:
            self._att = self.scene.debris_attitudes()
        P, _ = self._rocks(); logger.log_scene(self.t, P, self.f.dia, self._att.at(self.t)[self.scene_indices])

    def score_summary(self):
        sc = self.scene; dmg = self.damage; failed = dmg > sc.scoring["DamageThreshold_J"]
        arrived = self.arrived_t is not None and not failed
        score = (sc.failure_score(float(np.linalg.norm(self.r - self.f.T)), float(np.linalg.norm(self.f.r0 - self.f.T)),
                                  self.used, dmg) if failed else sc.arrival_score(self.arrived_t, self.used, dmg) if arrived else None)
        return {"arrived": arrived, "outcome": "damage_limit" if failed else "arrived" if arrived else "incomplete",
                "scoring_rules": sc.scoring, "scoring_time_s": self.arrived_t if arrived else self.t, "score": score,
                "hits": len(self.hits), "damage_kJ": dmg / 1000.0,
                "min_clear": float(self.clear) if np.isfinite(self.clear) else None}


if __name__ == "__main__":                  # self-check: the push and the smooth motion match the recorded fields
    C.configure()
    f = WorldField(0, bumps=False)
    v0 = (f.P[1] - f.P[0]); v14 = (f.P[15] - f.P[13]) / 2
    push = np.median(v14 - v0, axis=0)
    assert -4.5 < push[0] < -2.0 and abs(push[1]) < 0.2, push
    print("ok: start-up push", push.round(2))
