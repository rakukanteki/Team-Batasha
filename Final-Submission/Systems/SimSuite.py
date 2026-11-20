"""
Systems/SimSuite.py - unified Python simulator suite across multiple physics, rock-field, and sensor models.

Scenarios:
    point_mass           Point-mass shuttle + FINAL_SUBMISSION.simconf 500 rocks + 120 Hz radar + optional camera
    jet_twin_field       6-DOF rigid-body jet solver (TwinSim) + measured sim_field.npz rocks + 105 Hz lagged radar
    jet_twin_clean       6-DOF rigid-body jet solver (TwinSim) in clear space (burn-law & attitude baseline)
    stress_storm         Point-mass shuttle + high-velocity rock perturbations + boosted radar noise
    competition_simconf  Point-mass shuttle + COMPETITION_SIM.simconf debris field
"""
from __future__ import annotations

import concurrent.futures
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from Systems import Actuation, BasicController, Core as C, Vision as V_
from Systems.Flight import Flight
from Systems.RunLogger import DEFAULT_LOG_DIR, RunLogger
from Systems.plans import get_config, plan_name as canonical_plan_name
from Systems.plans.common import extract_plan_overrides
from tools.upstream_scene import FINAL_SCENE, PRACTICE_SCENE, Scene

ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

FINAL_SIMCONF = FINAL_SCENE
COMPETITION_SIMCONF = PRACTICE_SCENE
RAM_LOG_DIR = Path("/dev/shm/batasha_flightlogs")


def resolve_log_dir(log_dir: str | Path | None = None, ram: bool = False) -> Path:
    """Resolve log output directory, preferring /dev/shm RAM tmpfs when ram=True."""
    if log_dir is not None:
        return Path(log_dir)
    if ram and Path("/dev/shm").is_dir():
        return RAM_LOG_DIR
    return DEFAULT_LOG_DIR

SCENARIOS = {
    "phobos_world": {
        "label": "Phobos world (calibrated to real runs)",
        "description": "Rock field generated from FINAL_SUBMISSION.simconf and STU4-Sim's SpawnDebris (seeded start impulses, "
                       "the start-up push, Phobos gravity and frame terms, rock bumps), measured against 42 recorded real runs; "
                       "rigid-body shuttle behind the jet twin, 97 Hz radar, collisions with bounce (tools/phobos_world.py).",
        "default_dt": 1.0 / 30.0,
        "supports_camera": True,
    },
    "real_fields": {
        "label": "Recorded real-sim fields (STU4-lab)",
        "description": "Rocks exactly as they moved in a real Space Teams PRO run (seed picks the recording), same shuttle and "
                       "radar as Phobos world. Needs ../STU4-lab/runs (restore_fields.py).",
        "default_dt": 1.0 / 30.0,
        "supports_camera": True,
    },
    "point_mass": {
        "label": "Point-Mass + FINAL_SUBMISSION Rocks",
        "description": "Fast point-mass dynamics with 500 rocks from FINAL_SUBMISSION.simconf, 120 Hz radar, and optional emulated camera.",
        "default_dt": 0.1,
        "supports_camera": True,
    },
    "point_mass_clean": {
        "label": "Point-Mass (Asteroid-Free)",
        "description": "Fast asteroid-free point-mass dynamics for sweeping accel_dur_s, accel_till_speed, and burn/steering efficiency without obstacle detours.",
        "default_dt": 0.1,
        "supports_camera": False,
    },
    "jet_twin_field": {
        "label": "6-DOF Jet Twin + Measured Sim Field",
        "description": "Rigid-body 6-DOF jet solver with measured accelerating/knocking rocks (sim_field.npz) and 105 Hz lagged radar.",
        "default_dt": 1.0 / 60.0,
        "supports_camera": False,
    },
    "jet_twin_clean": {
        "label": "6-DOF Jet Twin (Obstacle-Free)",
        "description": "Rigid-body 6-DOF jet solver without debris to verify burn laws, attitude loop, and fuel efficiency.",
        "default_dt": 1.0 / 60.0,
        "supports_camera": False,
    },
    "stress_storm": {
        "label": "Stress Storm (High Rock Perturbation & Noise)",
        "description": "Point-mass dynamics with 0.4 m/s rock velocity dispersion, 3.0 m radar noise, and 5% thrust error.",
        "default_dt": 0.1,
        "supports_camera": True,
    },
    "competition_simconf": {
        "label": "Point-Mass + COMPETITION_SIM Rocks",
        "description": "Point-mass dynamics using submodules/STU4-Sim/Sims/COMPETITION_SIM.simconf debris field.",
        "default_dt": 0.1,
        "supports_camera": True,
    },
}


def effective_plan_config(plan: Any, overrides=None, param_json=None) -> dict:
    """Snapshot a plan with parameterized JSON taking priority over the run's edits."""
    defaults = get_config(plan)
    name = canonical_plan_name(plan)
    config = copy.deepcopy(defaults)
    if overrides:
        config.update(extract_plan_overrides({name: overrides}, name, defaults))
    if isinstance(param_json, (str, Path)):
        param_json = json.loads(Path(param_json).read_text(encoding="utf-8"))
    config.update(extract_plan_overrides(param_json, name, defaults))
    return config


def _run_point_mass(
    plan_id: Any,
    seed: int,
    simconf_path: Path,
    dt: float = 0.1,
    max_time: float = 420.0,
    burn: str | None = None,
    camera: str = "off",
    segmenter: str = "classical",
    vel_noise: float = 0.02,
    pos_noise: float | None = None,
    thrust_sigma: float = 0.03,
    log_dir: str | Path | None = None,
    flush_immediately: bool = False,
    log_run: bool = True,
    runner_name: str = "point_mass",
    progress_cb: Callable[[float, dict[str, Any]], None] | None = None,
    live: bool = False,
    config: dict[str, Any] | None = None,
    clean: bool = False,
) -> dict[str, Any]:
    import offline_bench as OB

    scene = Scene(simconf_path)
    radar_params = scene.systems["Radar"]
    radar_hz = radar_params["LoopFreqHz"]
    pos_noise = radar_params["Position_1sigma_m"] if pos_noise is None else pos_noise
    frame_accel = bool((config or {}).get("frame_accel", False))
    win_radius = scene.scoring["KeepOutRadius_m"]
    damage_threshold = scene.scoring["DamageThreshold_J"]
    rng = np.random.default_rng(seed)
    P0, V0, D, S, T = scene.field()
    if clean:
        P0 = np.zeros((0, 3), dtype=float)
        V0 = np.zeros((0, 3), dtype=float)
        D = np.zeros(0, dtype=float)
    attitudes = scene.debris_attitudes() if len(D) and log_run else None
    burn_name = burn or (config.get("burn") if config else None) or BasicController.BURN
    tracker_name = (config.get("tracker") if config else None) or BasicController.TRACKER
    C.configure(S, T, a_burn=Actuation.BURNS[burn_name].accel)
    P = P0.copy()
    V = V0 + rng.normal(0, vel_noise, V0.shape) if len(D) else V0.copy()
    N = len(D)
    true_size = (D / 2 * rng.uniform(1.0, 1.3, N) + 2.5) if N else np.zeros(0, dtype=float)
    thrust = 1 + rng.normal(0, thrust_sigma)
    r = S.copy()
    v = np.array(scene.shuttle["Velocity"][1:], dtype=float)
    fuel = 0.0
    t = 0.0
    step = 0
    max_speed = float(np.linalg.norm(v))
    main_burn_s = 0.0
    trim_burn_s = 0.0
    main_fuel_kg = 0.0
    trim_fuel_kg = 0.0

    parts = dict(planner=canonical_plan_name(plan_id), tracker=tracker_name, burn=burn_name, camera=camera, segmenter=segmenter)
    run_logger = RunLogger(
        plan_id,
        base_dir=log_dir or DEFAULT_LOG_DIR,
        flush_immediately=flush_immediately,
        runner=runner_name,
        parts=parts,
        config=config,
        seed=seed,
        live=live,
    ) if log_run else None

    class ConfigurableRadar(OB.Radar):
        def _measure(self, i, t_now, P_arr, V_arr, D_arr, r_arr, v_arr):
            rel = P_arr[i] - r_arr
            rho = np.linalg.norm(rel)
            g = self.rng
            return np.array([
                rel + g.normal(0, pos_noise, 3),
                [rho - 0.5 * D_arr[i] + g.normal(0, pos_noise),
                 (V_arr[i] - v_arr) @ rel / rho + g.normal(0, radar_params["Velocity_1sigma_m_s"]),
                 D_arr[i] * (1 + radar_params["Diameter_1sigma_fraction"] * g.normal())],
                [t_now, 0.0, 0.0],
            ])

    radar = ConfigurableRadar(rng, P, V, D, r, v) if N else None
    flight = Flight(plan_id, tracker=tracker_name, run_logger=run_logger, config=config, empty_field=clean)
    flight.radar(radar.published if radar else [], 0.0)
    cam = V_.camera_use(camera, segmenter) if N else None
    eye = OB.Camera(np.random.default_rng(seed + 1000)) if cam else None
    errs = []
    reports = []
    hits: dict[int, float] = {}
    clear = np.inf
    guard_steps = 0
    wall = time.time()
    last_prog = -10.0
    last_scene = -1.0

    try:
        while t < max_time:
            if attitudes is not None and t >= last_scene + 0.5:
                run_logger.log_scene(t, P, D, attitudes.at(t))
                last_scene = t
            mode, vec = flight.decide(t, r, v, fuel, dt)
            guard_steps += int(flight.guarding)
            lin_cmd = np.zeros(3)
            ang_cmd = np.zeros(3)
            if cam is not None and cam.due(t, mode == "coast" or (flight.departure is not None and not flight.guarding)):
                eye.scene = (np.vstack([P, T]), np.append(true_size - 2.5, 40.0), r)
                rep = cam(eye, flight.world, t, r, v)
                if isinstance(rep, dict):
                    reports.append(rep)
                if run_logger is not None:
                    run_logger.log_camera(t, True, 0.0, 0.0, V_.FOV, report=rep)
            acc = C.accel(r, v, frame_accel)
            if mode != "coast":
                a = (C.A_BURN if mode == "main" else C.A_TRIM) * thrust
                n = np.linalg.norm(vec)
                dv = min(a * dt, n)
                lin_cmd = 0.5 * vec / n
                v = v + dv * vec / n
                df = C.KF * dv * (1.0 if mode == "main" else C.TRIM_L1)
                fuel += df
                if mode == "main":
                    main_burn_s += dt
                    main_fuel_kg += df
                elif mode == "trim":
                    trim_burn_s += dt
                    trim_fuel_kg += df
            if run_logger is not None:
                run_logger.log_tick(
                    t, r, v, fuel, mode, vec, lin_cmd, ang_cmd,
                    flight.idx, flight.active_point, guarding=flight.guarding,
                    a_burn=C.A_BURN, barrier_active=flight.barrier_active,
                )
            v = v + acc * dt
            r = r + v * dt
            max_speed = max(max_speed, float(np.linalg.norm(v)))
            if N:
                V = V + C.accel(P, V, frame_accel) * dt
                P = P + V * dt
            t += dt
            step += 1
            if N and radar is not None:
                d = np.linalg.norm(P - r, axis=1)
                clear = min(clear, float((d - true_size).min()))
                for i in np.flatnonzero(d < true_size):
                    hits.setdefault(int(i), 0.5 * C.MASS * float(np.linalg.norm(v - V[i])) ** 2)
                for _ in range(int(round(t * radar_hz)) - int(round((t - dt) * radar_hz))):
                    radar.tick(t, P, V, D, r, v)
                if step % max(int(round(0.2 / dt)), 1) == 0:
                    flight.radar(radar.published, t)
                    if step % max(int(round(5.0 / dt)), 1) == 0:
                        errs.append(OB._track_error(flight, radar, P, V, r, t))
            if progress_cb is not None and t - last_prog >= 5.0:
                last_prog = t
                progress_cb(min(t / max_time, 1.0), {
                    "t": round(t, 1),
                    "dist_m": round(float(np.linalg.norm(r - T)), 1),
                    "fuel_kg": round(float(fuel), 1),
                    "mode": mode,
                    "hits": len(hits),
                })
            if sum(hits.values()) > damage_threshold or np.linalg.norm(r - T) <= win_radius:
                break

        damage = sum(hits.values())
        distance = float(np.linalg.norm(r - T))
        failed = damage > damage_threshold
        arrived = not failed and distance <= win_radius
        score = (scene.failure_score(distance, C.L, fuel, damage) if failed else
                 scene.arrival_score(t, fuel, damage) if arrived else None)
        errs = [e for e in errs if e is not None]
        rms = lambda k: float(np.sqrt(np.mean([e[k] for e in errs]))) if errs else float("nan")
        vc_now = float(flight.plan[1]) if flight.plan is not None else float(C.resolve_vc(config))
        ptot_now = round(float(flight.events[-1][3]), 6) if flight.events else None
        out = dict(
            plan=canonical_plan_name(plan_id),
            scenario=runner_name,
            seed=seed,
            score=round(score, 2) if score is not None else None,
            arrived=arrived,
            T=round(float(t), 2),
            fuel=round(float(fuel), 2),
            max_speed=round(max_speed, 2),
            main_burn_s=round(main_burn_s, 2),
            trim_burn_s=round(trim_burn_s, 2),
            main_fuel_kg=round(main_fuel_kg, 2),
            trim_fuel_kg=round(trim_fuel_kg, 2),
            hits=len(hits),
            damage_kJ=round(float(damage / 1e3), 2),
            min_clear=round(float(clear), 2) if np.isfinite(clear) else None,
            max_points=max([e[2] for e in flight.events], default=0),
            plan_changes=len(flight.events),
            accepted_replans=max(len(flight.events) - 1, 0),
            cruise_speed_mps=round(vc_now, 2),
            predicted_risk=ptot_now,
            guard_steps=int(guard_steps),
            cross_m=round(rms(0), 2) if np.isfinite(rms(0)) else None,
            along_m=round(rms(1), 2) if np.isfinite(rms(1)) else None,
            vel_mps=round(rms(2), 3) if np.isfinite(rms(2)) else None,
            images=len(reports),
            camera_used=sum(rep.get("used", 0) for rep in reports),
            wall_s=round(time.time() - wall, 2),
            run_dir=str(run_logger.run_dir) if run_logger else None,
            json_path=str(run_logger.json_path) if run_logger else None,
            log_path=str(run_logger.log_path) if run_logger else None,
        )
        return out
    finally:
        if run_logger is not None:
            dmg_now = sum(hits.values())
            distance = float(np.linalg.norm(r - T))
            failed = dmg_now > damage_threshold
            arrived_now = not failed and distance <= win_radius
            sc_now = (scene.failure_score(distance, C.L, fuel, dmg_now) if failed else
                      scene.arrival_score(t, fuel, dmg_now) if arrived_now else None)
            if step:
                zero = np.zeros(3)
                terminal_mode = "arrived" if arrived_now else "damage_limit" if failed else "stopped"
                run_logger.log_tick(t, r, v, fuel, terminal_mode, zero, zero, zero,
                                    flight.idx, flight.active_point, terminal=True)
                if attitudes is not None:
                    run_logger.log_scene(t, P, D, attitudes.at(t))
            run_logger.finalize({
                "arrived": arrived_now,
                "outcome": "damage_limit" if failed else "arrived" if arrived_now else "incomplete",
                "scoring_rules": scene.scoring,
                "duration_s": round(float(t), 4),
                "final_fuel_kg": round(float(fuel), 4),
                "final_position": r.tolist(),
                "final_velocity": v.tolist(),
                "hits": len(hits),
                "damage_kJ": round(float(dmg_now / 1e3), 2),
                "min_clear": round(float(clear), 4) if np.isfinite(clear) else None,
                "score": round(sc_now, 2) if sc_now is not None else None,
            })


def _flight_telemetry(flight, config: dict[str, Any] | None) -> dict[str, Any]:
    """Extract burn, replan, cruise-speed, and predicted-risk metrics from a completed Flight."""
    s = flight.run_logger._summary if getattr(flight, "run_logger", None) is not None else {}
    vc = float(flight.plan[1]) if flight.plan is not None else float(C.resolve_vc(config))
    ptot = round(float(flight.events[-1][3]), 6) if flight.events else None
    return {
        "main_burn_s": round(float(s["main_burn_s"]), 2) if "main_burn_s" in s else None,
        "trim_burn_s": round(float(s["trim_burn_s"]), 2) if "trim_burn_s" in s else None,
        "main_fuel_kg": round(float(s["main_fuel_kg"]), 2) if "main_fuel_kg" in s else None,
        "trim_fuel_kg": round(float(s["trim_fuel_kg"]), 2) if "trim_fuel_kg" in s else None,
        "accepted_replans": int(s.get("accepted_replans", max(len(flight.events) - 1, 0))),
        "cruise_speed_mps": round(float(s.get("cruise_speed_mps", vc)), 2),
        "predicted_risk": s.get("predicted_risk", ptot),
        **({"cutoff_time_s": flight.departure.cutoff_time, "peak_speed_mps": flight.departure.peak_speed}
           if flight.departure else {}),
    }


def run_single(
    plan: Any,
    scenario: str = "point_mass",
    seed: int = 0,
    burn: str | None = None,
    camera: str | None = None,
    segmenter: str = "classical",
    config_overrides: dict[str, Any] | None = None,
    dt: float | None = None,
    max_time: float = 420.0,
    flush_immediately: bool = False,
    log_dir: str | Path | None = None,
    log_run: bool = True,
    export_blender: bool = False,
    progress_cb: Callable[[float, dict[str, Any]], None] | None = None,
    raise_on_error: bool = False,
    ram: bool = False,
    param_json: str | Path | dict[str, Any] | None = None,
    live: bool = False,
    debris_count: int | None = None,
) -> dict[str, Any]:
    """Run one plan on one scenario and seed, returning structured metrics and log paths."""
    if scenario not in SCENARIOS:
        raise KeyError(f"Unknown scenario {scenario!r}; choose from {sorted(SCENARIOS)}")
    if debris_count is not None and scenario != "phobos_world":
        raise ValueError("debris_count experiments are supported by phobos_world only")

    resolved_log_dir = resolve_log_dir(log_dir, ram=ram)
    wall0 = time.time()
    try:
        config = effective_plan_config(plan, config_overrides, param_json)
        camera = camera or config.get("camera", "off")
        if debris_count is not None:
            config["experiment_debris_count"] = debris_count
            config["experiment_debris_selection"] = "evenly_spaced_original_indices"
        if scenario == "point_mass":
            res = _run_point_mass(
                plan, seed, FINAL_SIMCONF, dt=dt or 0.1, max_time=max_time,
                burn=burn, camera=camera, segmenter=segmenter,
                log_dir=resolved_log_dir, flush_immediately=flush_immediately, log_run=log_run,
                runner_name="point_mass", progress_cb=progress_cb, live=live, config=config,
            )
        elif scenario == "point_mass_clean":
            res = _run_point_mass(
                plan, seed, FINAL_SIMCONF, dt=dt or 0.1, max_time=max_time,
                burn=burn, camera="off", segmenter=segmenter,
                log_dir=resolved_log_dir, flush_immediately=flush_immediately, log_run=log_run,
                runner_name="point_mass_clean", progress_cb=progress_cb, live=live, config=config,
                clean=True,
            )
        elif scenario == "competition_simconf":
            res = _run_point_mass(
                plan, seed, COMPETITION_SIMCONF, dt=dt or 0.1, max_time=max_time,
                burn=burn, camera=camera, segmenter=segmenter,
                log_dir=resolved_log_dir, flush_immediately=flush_immediately, log_run=log_run,
                runner_name="competition_simconf", progress_cb=progress_cb, live=live, config=config,
            )
        elif scenario == "stress_storm":
            res = _run_point_mass(
                plan, seed, FINAL_SIMCONF, dt=dt or 0.1, max_time=max_time,
                burn=burn, camera=camera, segmenter=segmenter,
                vel_noise=0.4, pos_noise=3.0, thrust_sigma=0.05,
                log_dir=resolved_log_dir, flush_immediately=flush_immediately, log_run=log_run,
                runner_name="stress_storm", progress_cb=progress_cb, live=live, config=config,
            )
        elif scenario in ("jet_twin_field", "phobos_world", "real_fields"):
            wall = time.time()
            if scenario == "jet_twin_field":
                import sim_bench as SB
                sim = SB.FieldSim(seed, seconds=max_time, frame_accel=True)
            else:
                from tools.phobos_world import WorldSim, recorded_fields
                from tools.replay_bench import Field as RecordedField
                field = None
                if scenario == "real_fields":
                    rec = recorded_fields()
                    if not rec:
                        raise FileNotFoundError("No recorded fields: clone codernayeem/STU4-lab next to this repo and run restore_fields.py")
                    field = RecordedField(rec[seed % len(rec)])
                sim = WorldSim(seed, tmax=max_time, hz=1.0 / (dt or SCENARIOS[scenario]["default_dt"]), field=field,
                               debris_count=debris_count)
            flight = BasicController.fly(
                sim, planner=plan, burn=burn, camera=camera if scenario != "jet_twin_field" else "off", threaded=False,
                log_dir=resolved_log_dir, flush_immediately=flush_immediately,
                log_run=log_run, seed=seed, live=live, runner=scenario, config=config,
                score_summary=sim.score_summary,
            )
            summary = sim.score_summary()
            res = dict(
                plan=canonical_plan_name(plan),
                scenario=scenario,
                seed=seed,
                score=round(summary["score"], 2) if summary["score"] is not None else None,
                arrived=summary["arrived"],
                T=round(summary["scoring_time_s"], 2),
                fuel=round(float(sim.used), 2),
                hits=summary["hits"],
                damage_kJ=round(summary["damage_kJ"], 2),
                min_clear=round(summary["min_clear"], 2) if summary["min_clear"] is not None else None,
                max_points=max([e[2] for e in flight.events], default=0),
                plan_changes=len(flight.events),
                **_flight_telemetry(flight, config),
                images=getattr(sim, "camera_frames", 0),
                camera_used=getattr(flight.camera, "used", 0),
                guard_steps=flight.guard_ticks,
                wall_s=round(time.time() - wall, 2),
                run_dir=str(flight.run_dir) if flight.run_dir else None,
                json_path=str(flight.run_dir / "run.json") if flight.run_dir else None,
                log_path=str(flight.run_dir / "run.log") if flight.run_dir else None,
            )
        else:  # jet_twin_clean
            import actuator_twin as twin

            class _CleanTwinSim(twin.TwinSim):
                def __init__(self, seconds: float):
                    super().__init__(seconds=seconds, scene_attitude=True, frame_accel=True)
                    self.arrived: float | None = None
                    self.max_speed: float = float(np.linalg.norm(self.v))

                def wait(self) -> bool:
                    cont = super().wait()
                    self.max_speed = max(self.max_speed, float(np.linalg.norm(self.v)))
                    if np.linalg.norm(self.r - self.goal) <= twin.SCENE.scoring["KeepOutRadius_m"]:
                        self.arrived = self.t
                        return False
                    return cont

                def score_summary(self):
                    return self._score_summary(self.arrived)

            _, _, _, start, target = twin.SCENE.field()
            C.configure(start, target)
            wall = time.time()
            sim = _CleanTwinSim(seconds=max_time)
            flight = BasicController.fly(
                sim, planner=plan, burn=burn, camera="off", threaded=False,
                log_dir=resolved_log_dir, flush_immediately=flush_immediately,
                log_run=log_run, seed=seed, live=live, runner=scenario, config=config,
                score_summary=sim.score_summary,
            )
            summary = sim.score_summary()
            res = dict(
                plan=canonical_plan_name(plan),
                scenario="jet_twin_clean",
                seed=seed,
                score=round(summary["score"], 2) if summary["score"] is not None else None,
                arrived=summary["arrived"],
                T=round(summary["scoring_time_s"], 2),
                fuel=round(float(sim.used), 2),
                **_flight_telemetry(flight, config),
                max_speed=round(sim.max_speed, 2),
                hits=0,
                damage_kJ=0.0,
                min_clear=None,
                max_points=max([e[2] for e in flight.events], default=0),
                plan_changes=len(flight.events),
                guard_steps=0,
                wall_s=round(time.time() - wall, 2),
                run_dir=str(flight.run_dir) if flight.run_dir else None,
                json_path=str(flight.run_dir / "run.json") if flight.run_dir else None,
                log_path=str(flight.run_dir / "run.log") if flight.run_dir else None,
            )
        res.setdefault("error", None)
        res["debris_count"] = debris_count
    except Exception as exc:
        if raise_on_error:
            raise
        res = dict(
            plan=canonical_plan_name(plan),
            scenario=scenario,
            seed=seed,
            score=None,
            arrived=False,
            T=0.0,
            fuel=0.0,
            hits=0,
            damage_kJ=0.0,
            min_clear=None,
            max_points=0,
            plan_changes=0,
            guard_steps=0,
            wall_s=round(time.time() - wall0, 2),
            run_dir=None,
            json_path=None,
            log_path=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    if export_blender and res.get("json_path"):
        import subprocess
        blend_out = ROOT / "blender" / f"{Path(res['run_dir']).name}.blend"
        proc = subprocess.run(
            [sys.executable, str(TOOLS_DIR / "json_to_blender.py"), res["json_path"], "-o", str(blend_out)],
            capture_output=True, text=True,
        )
        res["blend_path"] = str(blend_out) if proc.returncode == 0 and blend_out.is_file() else None
        if proc.returncode != 0:
            res["blend_error"] = proc.stderr[-400:]
    else:
        res["blend_path"] = None
    return res


def _worker_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"


def _run_suite_item(kwargs: dict[str, Any]) -> dict[str, Any]:
    return run_single(**kwargs)


def run_suite(
    plans: list[Any],
    scenarios: list[str] | None = None,
    seeds: list[int] | None = None,
    burn: str | None = None,
    camera: str | None = None,
    segmenter: str = "classical",
    config_overrides: dict[str, Any] | None = None,
    dt: float | None = None,
    max_time: float = 420.0,
    flush_immediately: bool = False,
    log_dir: str | Path | None = None,
    log_run: bool = True,
    export_blender: bool = False,
    progress_cb: Callable[[int, int, dict[str, Any]], None] | None = None,
    workers: int = 1,
    ram: bool = False,
    param_json: str | Path | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a matrix of plans x scenarios x seeds and aggregate a suite comparison report."""
    scenarios = scenarios or ["point_mass"]
    seeds = [0] if seeds is None else list(seeds)
    base_out = resolve_log_dir(log_dir, ram=ram)
    jobs = [
        dict(
            plan=pl, scenario=sc, seed=s, burn=burn, camera=camera, segmenter=segmenter,
            config_overrides=config_overrides, dt=dt, max_time=max_time,
            flush_immediately=flush_immediately, log_dir=str(base_out), log_run=log_run,
            export_blender=export_blender, param_json=param_json,
        )
        for sc in scenarios
        for pl in plans
        for s in seeds
    ]
    total = len(jobs)
    results: list[dict[str, Any] | None] = [None] * total
    if workers > 1 and total > 1:
        max_w = min(workers, total)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_w, initializer=_worker_init) as pool:
            fut_to_idx = {pool.submit(_run_suite_item, j): i for i, j in enumerate(jobs)}
            done_cnt = 0
            for fut in concurrent.futures.as_completed(fut_to_idx):
                i = fut_to_idx[fut]
                r = fut.result()
                results[i] = r
                done_cnt += 1
                if progress_cb is not None:
                    progress_cb(done_cnt, total, r)
    else:
        for i, j in enumerate(jobs):
            r = run_single(**j)
            results[i] = r
            if progress_cb is not None:
                progress_cb(i + 1, total, r)

    clean_results: list[dict[str, Any]] = [r for r in results if r is not None]

    # Aggregate summary per (scenario, plan)
    summary = []
    for sc in scenarios:
        for pl in plans:
            pname = canonical_plan_name(pl)
            rows = [r for r in clean_results if r["scenario"] == sc and r["plan"] == pname]
            scores = [r["score"] for r in rows if r["score"] is not None]
            clears = [r["min_clear"] for r in rows if r["min_clear"] is not None]
            summary.append({
                "scenario": sc,
                "plan": pname,
                "runs": len(rows),
                "arrived": sum(1 for r in rows if r["arrived"]),
                "hit_runs": sum(1 for r in rows if r["hits"] > 0),
                "errors": sum(1 for r in rows if r.get("error")),
                "mean_score": round(float(np.mean(scores)), 2) if scores else None,
                "median_score": round(float(np.median(scores)), 2) if scores else None,
                "min_score": round(float(np.min(scores)), 2) if scores else None,
                "max_score": round(float(np.max(scores)), 2) if scores else None,
                "mean_fuel": round(float(np.mean([r["fuel"] for r in rows])), 2) if rows else None,
                "mean_T": round(float(np.mean([r["T"] for r in rows])), 2) if rows else None,
                "min_clear": round(float(np.min(clears)), 2) if clears else None,
                "mean_wall_s": round(float(np.mean([r["wall_s"] for r in rows])), 2) if rows else None,
            })

    suite_dir = base_out / f"suite__{time.strftime('%Y%m%d_%H%M%S')}"
    suite_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "plans": [canonical_plan_name(p) for p in plans],
        "scenarios": scenarios,
        "seeds": seeds,
        "config_overrides": config_overrides or {},
        "summary": summary,
        "results": clean_results,
    }
    report_path = suite_dir / "suite_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
