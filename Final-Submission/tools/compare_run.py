"""Compare a Space Teams PRO run with the Python simulator under the same plan, scene and rock field.

    python tools/compare_run.py flightlogs/spaceteams/<run>            # all three checks, saves compare.json in the run
    python tools/compare_run.py <run> --seeds 0,1,2 --no-flight        # rocks and shuttle dynamics only
    python tools/compare_run.py <run> --no-flight --recording flightlogs/<ticks>.npz

1. rocks    where the engine's radar saw the rocks vs where phobos_world puts them at the same time (nearest rock,
            median distance and mean offset; radar order is not rock order, so rocks are matched by position)
2. dynamics the engine's own thrust commands (run.json time_to_thrust) replayed through the jet twin from the engine's
            first logged state, no rocks: how far position and speed drift from what the engine logged
3. flight   the same plan flown closed-loop in phobos_world (written to flightlogs/python, so the web GUI lists it next
            to the engine run); distance to goal, speed and fuel against the engine's at the same times
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from Systems import Core as C                       # noqa: E402
from tools.phobos_world import WorldField, WorldSim  # noqa: E402


def load(run_dir: Path) -> dict:
    return json.loads((run_dir / "run.json").read_text(encoding="utf-8"))


def series(run: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Engine time, position, velocity from time_to_point."""
    items = sorted(((float(t), p) for t, p in run["time_to_point"].items()))
    t = np.array([i[0] for i in items])
    return t, np.array([i[1]["shuttle_pos"] for i in items]), np.array([i[1]["shuttle_vel"] for i in items])


def at(t_ref: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    return np.array([np.interp(t, t_ref, y[:, k]) for k in range(y.shape[1])])


def check_rocks(run: dict, seed: int) -> dict:
    from scipy.spatial import cKDTree
    calls = run["radar_calls"]
    if not calls:
        return {"note": "no radar calls in run.json"}
    field = WorldField(seed, seconds=max(c["t"] for c in calls) + 5.0)
    rows = []
    for c in calls[:: max(1, len(calls) // 8)]:
        P, _ = field.at(c["t"])
        pos = np.array(c["positions_abs"])
        if not len(pos):
            continue
        dist, idx = cKDTree(P).query(pos)
        ok = dist < 40.0
        off = (pos[ok] - P[idx[ok]]).mean(axis=0) if ok.any() else np.zeros(3)
        rows.append({"t": round(c["t"], 1), "median_nearest_m": round(float(np.median(dist)), 2),
                     "p90_nearest_m": round(float(np.percentile(dist, 90)), 2),
                     "mean_offset_m": off.round(2).tolist(), "matched": int(ok.sum()), "seen": int(len(pos))})
    return {"seed": seed, "rows": rows,
            "worst_median_m": max(r["median_nearest_m"] for r in rows) if rows else None}


def check_dynamics(run: dict) -> dict:
    """Replay the engine's logged thrust commands through the jet twin with no rocks present."""
    from tools.actuator_twin import tick, rotation, INERTIA, INERTIA_INV
    t, pos, vel = series(run)
    cmds = {float(k): v for k, v in run["time_to_thrust"].items()}
    tk = sorted(cmds)
    if len(tk) < 2 or not len(t):
        return {"note": "at least two commands and a state sample are required"}
    tk = [seconds for seconds in tk if t[0] <= seconds <= t[-1]]
    if len(tk) < 2:
        return {"note": "commands do not overlap recorded states"}
    first = run["time_to_point"].get(f"{tk[0]:.4f}", {})
    if "body_quaternion_xyzw" not in first:
        return {"note": "no initial attitude at the first command; use the NPZ recording"}
    C.configure(np.array(run["metadata"]["core"]["start"]), np.array(run["metadata"]["core"]["target"]),
                a_burn=run["metadata"]["core"]["a_burn"])
    r, v = at(t, pos, tk[0]), at(t, vel, tk[0]); w = np.zeros(3)
    # RunLogger's active body-to-Phobos rotation differs from the lab CSV convention.
    R = Rotation.from_quat(first["body_quaternion_xyzw"]).as_matrix(); used = 0.0
    err = []
    from tools.replay_bench import TRUE_B, TRUE_W
    for a, b in zip(tk[:-1], tk[1:]):
        cmd = cmds[a]
        acc, rate, _ = tick(cmd["lin_cmd"], cmd["ang_cmd"])
        w = w + (acc[3:] - INERTIA_INV @ np.cross(w, INERTIA @ w)) * (b - a)
        v = v + (R @ acc[:3] + C.grav(r) + TRUE_B - 2 * np.cross(TRUE_W, v)) * (b - a)
        r = r + v * (b - a)
        if np.linalg.norm(w) > 0:
            R = R @ rotation(w, np.linalg.norm(w) * (b - a))
        used += rate * (b - a)
        er = float(np.linalg.norm(r - at(t, pos, b))); ev = float(np.linalg.norm(v - at(t, vel, b)))
        err.append((b, er, ev))
    err = np.array(err)
    marks = [err[min(int(len(err) * f), len(err) - 1)] for f in (0.25, 0.5, 0.75, 1.0)]
    return {"duration_s": round(float(tk[-1] - tk[0]), 1),
            "pos_error_m": [{"t": round(float(m[0]), 1), "pos_m": round(float(m[1]), 2), "vel_mps": round(float(m[2]), 3)} for m in marks],
            "final_pos_error_m": round(float(err[-1, 1]), 2), "final_speed_error_mps": round(float(err[-1, 2]), 3),
            "twin_fuel_kg": round(used, 2)}


def check_recorded_dynamics(path: Path) -> dict:
    """Compare recorded velocity/fuel changes with the final-scene actuator on matched windows.

    Use recorded position, velocity, attitude, commands and sample times throughout
    each window so accumulated attitude drift and scene timing do not hide local errors.
    """
    from tools.actuator_twin import tick
    from tools.upstream_scene import Scene

    with np.load(path) as recording:
        ticks = recording["ticks"]
        target = recording["target"]
    scene = Scene()
    _, _, _, start, goal = scene.field()
    if ticks.ndim != 2 or ticks.shape[1] != 26 or len(ticks) < 2:
        raise ValueError("Expected the 26-column flight recorder tick format")
    if not np.isfinite(ticks).all() or np.any(np.diff(ticks[:, 0]) < 0):
        raise ValueError("Recording has invalid data or timestamps")
    if not np.allclose(goal, target, atol=0.1, rtol=0):
        raise ValueError("Recorded target does not match the official final scene")

    seconds = ticks[:, 0]
    coast, burn = [], []
    for i in np.searchsorted(seconds, np.arange(seconds[0] + 1, seconds[-1] - 1.5, 2)):
        j = np.searchsorted(seconds, seconds[i] + 1.5)
        if j >= len(seconds):
            continue
        rows = ticks[i:j]
        dt = np.diff(seconds[i:j + 1])
        if dt.max() > 0.12:
            continue
        duration = dt.sum()
        command = np.linalg.norm(rows[:, 19:25], axis=1)
        measured_dv = ticks[j, 4:7] - ticks[i, 4:7]
        measured_fuel = ticks[j, 25] - ticks[i, 25]
        position, velocity = rows[:, 1:4], rows[:, 4:7]
        gravity_dv = dt @ C.accel(position, velocity)
        framed_dv = dt @ C.accel(position, velocity, frame=True)

        if not np.any(command > 1e-8) and abs(measured_fuel) < 0.0001:
            raw = (measured_dv - gravity_dv) / duration
            if np.linalg.norm(raw) < 0.04:  # exclude impulses outside the coast model
                coast.append((np.linalg.norm(raw), np.linalg.norm(measured_dv - framed_dv) / duration))
        elif np.mean(command > 0.01) > 0.95 and measured_fuel > 0.02:
            responses = [tick(row[19:22], row[22:25]) for row in rows]
            rotation = rows[:, 10:19].reshape(-1, 3, 3)  # body-to-Phobos, stored in rows
            predicted = np.einsum("n,nij,nj->i", dt, rotation, [r[0][:3] for r in responses])
            predicted_fuel = dt @ np.array([r[1] for r in responses])
            if np.linalg.norm(predicted) > 0.02:
                real = measured_dv - framed_dv
                burn.append((float(real @ predicted / (predicted @ predicted)),
                             float(measured_fuel / predicted_fuel),
                             float(np.linalg.norm(real - predicted) / duration)))

    initial_rotation = Rotation.from_matrix(ticks[0, 10:19].reshape(3, 3))
    return {
        "recording": str(path),
        "initial_offset_m": float(np.linalg.norm(ticks[0, 1:4] - start)),
        "initial_attitude_error_deg": float(np.rad2deg(
            (scene.body_rotation(scene.shuttle).inv() * initial_rotation).magnitude())),
        "coast_windows": len(coast),
        "coast_median_acceleration_error_mps2": dict(zip(("point_mass", "with_frame"),
                                                        np.median(coast, axis=0).tolist())) if coast else None,
        "burn_windows": len(burn),
        "burn_median": dict(zip(("measured_to_predicted_impulse_projection", "measured_to_predicted_fuel",
                                 "acceleration_error_mps2"), np.median(burn, axis=0).tolist())) if burn else None,
    }


def engine_result(run: dict) -> dict:
    """The engine's own score lines, from the app's log folder that started within 2 min before this run."""
    from tools.spaceteams_runner import engine_score_for
    r = engine_score_for(run["metadata"].get("timestamp_utc"), run["metadata"].get("plan_name"))
    if not r:
        return {"note": "no engine log folder found for this run"}
    return r if "score" in r else {**r, "note": "no score line yet (run killed or still going)"}


def check_flight(run: dict, run_dir: Path, seed: int, max_time: float, log_dir: Path) -> dict:
    from Systems import SimSuite
    meta = run["metadata"]
    res = SimSuite.run_single(meta["plan_name"], scenario="phobos_world", seed=seed, max_time=max_time,
                              log_dir=log_dir, log_run=True, config_overrides=meta.get("config"))
    sim_run = load(Path(res["run_dir"]))
    te, pe, ve = series(run); ts, ps, vs = series(sim_run)
    T = np.array(C.T)
    rows = []
    for tt in np.arange(max(te[0], ts[0]), min(te[-1], ts[-1]), 10.0):
        pe_t, ps_t = at(te, pe, tt), at(ts, ps, tt)
        rows.append({"t": round(float(tt), 1),
                     "engine_dist_goal": round(float(np.linalg.norm(pe_t - T)), 1), "python_dist_goal": round(float(np.linalg.norm(ps_t - T)), 1),
                     "engine_speed": round(float(np.linalg.norm(at(te, ve, tt))), 2), "python_speed": round(float(np.linalg.norm(at(ts, vs, tt))), 2),
                     "separation_m": round(float(np.linalg.norm(pe_t - ps_t)), 1)})
    return {"seed": seed, "python_run_dir": res["run_dir"], "python_score": res.get("score"), "python_arrived": res.get("arrived"),
            "python_fuel_kg": res.get("fuel"), "hits": res.get("hits"), "rows": rows,
            "engine_summary": meta.get("summary"), "python_summary": sim_run["metadata"].get("summary")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--seeds", default="0", help="comma list; rocks use the first, the flight runs every seed")
    ap.add_argument("--no-flight", action="store_true")
    ap.add_argument("--recording", type=Path, help="matching Recorder NPZ for short-window actuator and coast comparisons")
    ap.add_argument("--log-dir", type=Path, default=ROOT / "flightlogs")
    a = ap.parse_args()
    run = load(a.run_dir)
    seeds = [int(s) for s in a.seeds.split(",")]
    te = series(run)[0]
    out = {"engine_run": str(a.run_dir), "plan": run["metadata"]["plan_name"], "partial": bool(run["metadata"]["summary"].get("partial")),
           "engine_span_s": [round(float(te[0]), 1), round(float(te[-1]), 1)]}
    out["engine_result"] = engine_result(run)
    out["rocks"] = check_rocks(run, seeds[0])
    out["dynamics"] = check_dynamics(run)
    if a.recording:
        out["recorded_dynamics"] = check_recorded_dynamics(a.recording)
    if not a.no_flight:
        out["flights"] = [check_flight(run, a.run_dir, s, min(420.0, float(te[-1]) + 30.0), a.log_dir) for s in seeds]
    (a.run_dir / "compare.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{out['plan']}  engine span {out['engine_span_s']} s{'  (partial run)' if out['partial'] else ''}")
    print("\nROCKS  engine radar vs phobos_world field")
    for r in out["rocks"].get("rows", []):
        print(f"  t={r['t']:6.1f}  median {r['median_nearest_m']:6.2f} m  p90 {r['p90_nearest_m']:6.2f} m  offset {r['mean_offset_m']}  matched {r['matched']}/{r['seen']}")
    d = out["dynamics"]
    print(f"\nDYNAMICS  {d.get('note', 'engine commands replayed through the jet twin')}")
    for m in d.get("pos_error_m", []):
        print(f"  t={m['t']:6.1f}  position off by {m['pos_m']:7.2f} m  speed off by {m['vel_mps']:.3f} m/s")
    if a.recording:
        print(json.dumps(out["recorded_dynamics"], indent=2))
    for f in out.get("flights", []):
        print(f"\nFLIGHT  same plan in phobos_world seed {f['seed']}: score {f['python_score']} arrived {f['python_arrived']} fuel {f['python_fuel_kg']}")
        print("   t     goal-dist engine/python   speed engine/python   separation")
        for r in f["rows"]:
            print(f"  {r['t']:6.1f}  {r['engine_dist_goal']:8.1f} / {r['python_dist_goal']:8.1f}   {r['engine_speed']:5.2f} / {r['python_speed']:5.2f}   {r['separation_m']:8.1f} m")
        print(f"  python run: {f['python_run_dir']}")
    print(f"\nsaved {a.run_dir / 'compare.json'}")


if __name__ == "__main__":
    main()
