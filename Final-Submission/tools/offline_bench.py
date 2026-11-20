"""Offline flight bench: fly any plan through the real Systems code against an emulated sim.

Usage:  python tools/offline_bench.py --plan 1 --seeds 5
        python tools/offline_bench.py --plan 3 --seeds 0,7 --verbose

What is emulated (docs/plans/00_FOUNDATION.md section 10): the 500 rocks of the final simconf with a random
kick per run, the radar's measure-one-rock-per-tick and publish-sorted-list behaviour with its noise, a
point-mass shuttle, a random thrust error. Not emulated: shuttle rotation, true rock shapes, sim timing.

The debris list is read from the simconf here for testing only. Flight code in Systems/ never sees it.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from Systems import Actuation, BasicController  # noqa: E402
from Systems import Core as C            # noqa: E402
from Systems import Vision as V_         # noqa: E402
from Systems.Flight import Flight        # noqa: E402
from Systems.RunLogger import DEFAULT_LOG_DIR, RunLogger  # noqa: E402
from Systems.plans import get_config     # noqa: E402
from tools.upstream_scene import FINAL_SCENE, Scene  # noqa: E402

SIMCONF = FINAL_SCENE


def load_field(simconf=SIMCONF, *, serialized=True):
    """Rock positions, velocities, radar diameters, shuttle start and target from the simconf."""
    scene = Scene(simconf)
    positions, velocities, diameters, start, target = scene.field()
    if serialized:
        order = scene.serialized_debris_order()
        positions, velocities, diameters = positions[order], velocities[order], diameters[order]
    return positions, velocities, diameters, start, target


TRUE_W = np.array([0.0, 0.0, 2.2803e-4])            # measured in the real sim 2026-11-13 (see Core.FRAME)
TRUE_B = np.array([0.00215, -0.00100, 0.00148])

class Radar:
    """Emulates STU4-Sim/Systems/Radar.py: one rock re-measured per tick, sorted list republished every 50."""

    def __init__(self, rng, P, V, D, r, v):
        self.rng = rng; self.n = len(D); self.it = 0
        self.data = [self._measure(i, 0.0, P, V, D, r, v) for i in range(self.n)]
        self._publish()

    def _measure(self, i, t, P, V, D, r, v):
        rel = P[i] - r; rho = np.linalg.norm(rel); g = self.rng
        return np.array([rel + g.normal(0, 2, 3),
                         [rho - 0.5 * D[i] + g.normal(0, 2), (V[i] - v) @ rel / rho + g.normal(0, 0.02), D[i] * (1 + 0.05 * g.normal())],
                         [t, 0.0, 0.0]])

    def _publish(self):
        order = sorted(range(self.n), key=lambda i: np.linalg.norm(self.data[i][0]))
        self.published = [self.data[i].copy() for i in order]; self.truth = order      # truth: rock id per entry (tests only)

    def tick(self, t, P, V, D, r, v):
        idx = self.it % self.n; self.data[idx] = self._measure(idx, t, P, V, D, r, v)
        if idx % max(1, self.n // 10) == 0:
            self._publish()
        self.it += 1


class Camera:
    """Emulated sim camera (the two Interfaces.Sim methods a camera use calls): sunlit balls on a black sky with
    stars and a big ball for the mothership, seen with a pointing error and an image layout that the flight code
    is not told. Rendering shares Vision.pixels and Vision.body_to_camera with the flight code, so a mistake common
    to both is invisible here; hidden signs, a transpose and a pointing error are caught."""

    def __init__(self, rng, layout=(1.0, -1.0, True), kappa=0.2):
        self.B = np.array([C.E1, C.E2, C.SH])                          # world -> body, body +z along the path
        self.err = V_.body_to_camera(*rng.normal(0, 0.4, 2))            # hidden pointing error, degrees
        a = rng.uniform(0, 2 * np.pi); self.sun = np.array([np.cos(a), np.sin(a)])
        self.layout = layout; self.kappa = kappa; self.rng = rng; self.scene = None   # scene: (positions, radii, shuttle)

    def to_body(self, vec):
        return self.B @ np.asarray(vec, float)

    def capture(self, top, yaw_deg, pitch_deg, fov_deg=60.0):
        P, R, r = self.scene; n = V_.SIZE; f = V_.focal(fov_deg, n); cam = V_.body_to_camera(yaw_deg, pitch_deg)
        q = (((P - r) @ self.B.T - (V_.TOP_MOUNT if top else V_.BOTTOM_MOUNT)) @ cam.T) @ self.err.T
        img = np.zeros((n, n), np.uint8)
        img[self.rng.integers(0, n, 300), self.rng.integers(0, n, 300)] = 255          # stars
        front = np.flatnonzero(q[:, 0] > 5.0)
        uv, _ = V_.pixels(q[front], f, n, self.layout); rad = f * R[front] / q[front, 0]
        for k in np.argsort(-q[front, 0]):                             # far to near: a near rock hides a far one
            (u, v), rr = uv[k], rad[k]
            u0, u1 = int(max(u - rr - 1, 0)), int(min(u + rr + 2, n)); v0, v1 = int(max(v - rr - 1, 0)), int(min(v + rr + 2, n))
            if u0 >= u1 or v0 >= v1:
                continue
            vv, uu = np.mgrid[v0:v1, u0:u1]; du, dv = uu - u, vv - v
            disc = du ** 2 + dv ** 2 <= rr ** 2
            s = du * self.sun[0] + dv * self.sun[1]; t2 = du ** 2 + dv ** 2 - s ** 2
            lit = disc & (s >= -self.kappa * np.sqrt(np.maximum(rr ** 2 - t2, 0.0)))   # kappa 1 full, 0 half, <0 crescent
            patch = img[v0:v1, u0:u1]; patch[disc] = 0; patch[lit] = 120 + (int(front[k]) * 37) % 136
        return np.dstack([img] * 3)


def _track_error(flight, radar, P, Vel, r, t):
    """Squared position error across and along the line of sight, and squared velocity error, averaged over the
    rocks 20-500 m ahead and within 80 m of the start-to-target line."""
    ids = {k: i for m, i in zip(radar.published, radar.truth) for k in [flight.world.keys.get(m.tobytes())] if k is not None}
    if not ids:
        return None
    trk = np.array(list(ids)); tru = np.array([ids[k] for k in trk])
    pos, vel = flight.world.predict(t)[:2]
    s = (P[tru] - r) @ C.SH; off = np.linalg.norm((P[tru] - r) - s[:, None] * C.SH, axis=1)
    ok = (s > 20) & (s < 500) & (off < 80)
    if not ok.any():
        return None
    e = pos[trk[ok]] - P[tru[ok]]; los = (P[tru[ok]] - r); los /= np.linalg.norm(los, axis=1, keepdims=True)
    along = (e * los).sum(1)
    return (np.mean(np.sum((e - along[:, None] * los) ** 2, 1)), np.mean(along ** 2), np.mean(np.sum((vel[trk[ok]] - Vel[tru[ok]]) ** 2, 1)))


def run(plan_id, seed, field=None, dt=0.1, verbose=False, camera="off", segmenter="classical",
        log_dir=None, flush_immediately=False, log_run=True):
    rng = np.random.default_rng(seed)
    P0, V0, D, S, T = field if field is not None else load_field()
    cfg = get_config(plan_id)
    burn = cfg.get("burn", BasicController.BURN)
    C.configure(S, T, a_burn=cfg.get("a_burn") or Actuation.BURNS[burn].accel)
    P = P0.copy(); V = V0 + rng.normal(0, 0.02, V0.shape); N = len(D)
    true_size = D / 2 * rng.uniform(1.0, 1.3, N) + 2.5          # collision radius incl. shuttle; unknown to the planner
    thrust = 1 + rng.normal(0, 0.03)                           # thrust error; unknown to the planner
    r = S.copy(); v = np.zeros(3); fuel = 0.0; t = 0.0; step = 0
    parts = dict(planner=plan_id, tracker=cfg.get("tracker", "kalman"), burn=burn, camera=camera, segmenter=segmenter)
    run_logger = RunLogger(
        plan_id,
        base_dir=log_dir or DEFAULT_LOG_DIR,
        flush_immediately=flush_immediately,
        runner="offline_bench",
        parts=parts,
        config=get_config(plan_id),
        seed=seed,
    ) if log_run else None
    radar = Radar(rng, P, V, D, r, v)
    flight = Flight(plan_id, run_logger=run_logger)
    flight.radar(radar.published, 0.0)
    cam = V_.camera_use(camera, segmenter); eye = Camera(np.random.default_rng(seed + 1000)) if cam else None
    errs = []; reports = []
    hits = {}; clear = np.inf; t_plan = 0.0; guard_steps = 0; wall = time.time()
    try:
        while t < 420.0:
            mode, vec = flight.decide(t, r, v, fuel, dt)
            guard_steps += flight.guarding
            lin_cmd = np.zeros(3); ang_cmd = np.zeros(3)
            if cam is not None and cam.due(t, mode == "coast"):
                eye.scene = (np.vstack([P, T]), np.append(true_size - 2.5, 40.0), r)
                rep = cam(eye, flight.world, t, r, v)
                if isinstance(rep, dict):
                    reports.append(rep)
                    if verbose:
                        print("    camera", {k: (np.round(x, 2) if k == "shift" else x) for k, x in rep.items()})
                if run_logger is not None:
                    run_logger.log_camera(t, True, 0.0, 0.0, V_.FOV, report=rep)
            acc = C.grav(r) + TRUE_B - 2 * np.cross(TRUE_W, v)          # the real sim's frame terms (Core.FRAME)
            if mode != "coast":
                a = (C.A_BURN if mode == "main" else C.A_TRIM) * thrust; n = np.linalg.norm(vec)
                dv = min(a * dt, n)
                lin_cmd = 0.5 * vec / n
                v = v + dv * vec / n; fuel += C.KF * dv * (1.0 if mode == "main" else C.TRIM_L1)
            if run_logger is not None:
                run_logger.log_tick(t, r, v, fuel, mode, vec, lin_cmd, ang_cmd,
                                    flight.idx, flight.active_point, guarding=flight.guarding,
                                    a_burn=C.A_BURN, barrier_active=flight.barrier_active)
            v = v + acc * dt; r = r + v * dt
            V = V + (C.grav(P) + TRUE_B - 2 * np.cross(TRUE_W, V)) * dt; P = P + V * dt; t += dt; step += 1
            d = np.linalg.norm(P - r, axis=1); clear = min(clear, (d - true_size).min())
            for i in np.flatnonzero(d < true_size):
                hits.setdefault(int(i), 0.5 * C.MASS * np.linalg.norm(v - V[i]) ** 2)
            for _ in range(12):                                    # radar runs at 120 Hz
                radar.tick(t, P, V, D, r, v)
            if step % 2 == 0:                                      # the controller reads it at 5 Hz
                flight.radar(radar.published, t)
                if step % 50 == 0:
                    errs.append(_track_error(flight, radar, P, V, r, t))
            if np.linalg.norm(r - T) <= C.WIN:
                break
        damage = min(sum(hits.values()), C.DMAX); arrived = np.linalg.norm(r - T) <= C.WIN
        score = 100 * (120 / t - 2 * damage / C.DMAX - fuel / 75 + 0.5) if arrived else float("nan")
        if verbose:
            for e in flight.events:
                print("   ", e)
        errs = [e for e in errs if e is not None]; rms = lambda k: float(np.sqrt(np.mean([e[k] for e in errs]))) if errs else float("nan")
        out = dict(plan=plan_id, seed=seed, cross_m=round(rms(0), 2), along_m=round(rms(1), 2), vel_mps=round(rms(2), 3),
                   images=len(reports), camera_used=sum(r_["used"] for r_ in reports), score=round(float(score), 1), arrived=bool(arrived), T=round(t, 1),
                   fuel=round(float(fuel), 1), hits=len(hits), min_clear=round(float(clear), 1),
                   max_points=max([e[2] for e in flight.events], default=0), plan_changes=len(flight.events),
                   guard_steps=int(guard_steps), wall_s=round(time.time() - wall, 1))
        if run_logger is not None:
            out["log_dir"] = str(run_logger.run_dir)
        return out
    finally:
        if run_logger is not None:
            arrived_now = bool(np.linalg.norm(r - T) <= C.WIN)
            dmg_now = min(sum(hits.values()), C.DMAX)
            sc_now = float(100 * (120 / max(t, 1e-6) - 2 * dmg_now / C.DMAX - fuel / 75 + 0.5)) if arrived_now else None
            run_logger.finalize({
                "arrived": arrived_now,
                "duration_s": round(float(t), 4),
                "final_fuel_kg": round(float(fuel), 4),
                "final_position": r.tolist(),
                "final_velocity": v.tolist(),
                "hits": len(hits),
                "score": sc_now,
            })


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", type=lambda x: int(x) if x.isdigit() else x, default=1, help="plan number or a name in Planners.PLANNERS")
    ap.add_argument("--camera", default="off", help="a name in Vision.CAMERAS")
    ap.add_argument("--segmenter", default="classical", help="a name in Vision.SEGMENTERS")
    ap.add_argument("--seeds", default="3", help="a count (5) or a list (0,7,11)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--flush-immediately", action="store_true", help="flush run.log immediately on each entry instead of buffering in memory")
    ap.add_argument("--log-dir", default=None, help="base directory for plan_name__timestamp run logs (default: flightlogs)")
    ap.add_argument("--no-log", action="store_true", help="disable run log / json output")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",") if x] if "," in a.seeds else range(int(a.seeds))
    field = load_field(); out = []
    for s in seeds:
        out.append(run(a.plan, s, field, verbose=a.verbose, camera=a.camera, segmenter=a.segmenter,
                       log_dir=a.log_dir, flush_immediately=a.flush_immediately, log_run=not a.no_log))
        print(out[-1], flush=True)
    sc = np.array([o["score"] for o in out])
    print(f"SUMMARY plan {a.plan}: n={len(out)} mean {np.nanmean(sc):.1f} min {np.nanmin(sc):.1f} max {np.nanmax(sc):.1f} | "
          f"runs with a hit: {sum(o['hits'] > 0 for o in out)} | arrived {sum(o['arrived'] for o in out)}/{len(out)} | "
          f"mean fuel {np.mean([o['fuel'] for o in out]):.1f} | min true clearance {min(o['min_clear'] for o in out):.1f} m")
    print(f"tracking error rms: cross {np.mean([o['cross_m'] for o in out]):.2f} m, along {np.mean([o['along_m'] for o in out]):.2f} m, "
          f"velocity {np.mean([o['vel_mps'] for o in out]):.3f} m/s | camera images used {sum(o['camera_used'] for o in out)}/{sum(o['images'] for o in out)}")


if __name__ == "__main__":
    main()
