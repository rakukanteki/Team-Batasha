"""
Systems/BasicController.py - the main file: choose the parts, adapt the sim, fly.
Spec: docs/plans/00_FOUNDATION.md section 3. Called from the generated root Controller.py as run(api),
where api is that module: SetCommand, GetNearestDebrisLocations, PanTiltCamera, shuttle, target, frames.

Every sim call of the team's code is in SpaceTeamsSim below. It uses the organizers' helpers as they are.
"""
from __future__ import annotations
import json
import os
import threading
import time
import traceback
from pathlib import Path
import numpy as np

try:
    import spaceteams as st
except ImportError:                 # offline: tests and tools run fly() on a stand-in sim instead
    st = None

from Systems import Actuation, Core, Vision
from Systems.Flight import Flight
from Systems.Interfaces import pick
from Systems.Maneuvers import BODY_COMMANDS
from Systems.Recorder import Recorder
from Systems.RunLogger import DEFAULT_LOG_DIR, ENGINE_SOURCE, PYTHON_SOURCE, RunLogger
from Systems.UserAPI import CaptureCameraImage
from Systems.plans import PLAN_DIRS, get_config
from Systems.plans.common import extract_plan_overrides, read_config_json

# ---- The parts. Change a part by editing its one line; the names are listed in the registry on the right. -----
# ---- Interfaces and how to add a part: Systems/Interfaces.py. ------------------------------------------------
PLANNER = "v16_morning"        # submitted controller (Plan 17). Planners.PLANNERS: fewest_points, linear_program, lattice, team_pipeline, conjunction,
#                             barrier, velocity_obstacle, astar, cost_lattice, real_calibrated, v11
TRACKER = "kalman"          # Radar.TRACKERS: kalman, calibrated
BURN = "direct"             # Actuation.BURNS: direct, steer (nayeem/real-calibrated), boost, plain, jets (four-jet +z)
CAMERA = "off"              # Vision.CAMERAS: off, consensus (camera bearings into the tracker), inspect (outlines)
SEGMENTER = "classical"     # Vision.SEGMENTERS: classical, sam, yolo (the team's trained weights)
RECORD = True               # save each flight for tools/replay.py when the simconf does not set Record (organizer scenes)

RADAR_EVERY = 12            # ticks between radar reads: 5 Hz at a 60 Hz loop
LOG_RUN = True              # write flightlogs/<simulator>/<plan>__<timestamp>/{run.log, run.json, images/}
SNAPSHOT_S = 10.0           # engine runs: rewrite run.json this often (s), the engine may kill us before finalize
FLUSH_IMMEDIATELY = False  # False: keep logs in memory and write on completion (faster); True: flush every line


class SpaceTeamsSim:
    """Interfaces.Sim on the organizers' sim."""
    invalid = False             # set when the sim runs much faster than real time (sped up in the app)

    def __init__(self, api):
        self.api = api; self.pf = api.phobosFrame; self.sf = api.shuttleFrame
        self.t0 = st.SimGlobals.GetSimEntity().GetParam(st.VarType.timestamp, ["Time", "StartDateTAI"]).as_datetime()

    def wait(self):
        time.sleep(1.0 / self.api.this.GetParam(st.VarType.double, "LoopFreqHz"))
        self._check_speed()
        return True

    def _check_speed(self, limit=3.0, span=2.0):
        """The loop ticks ~50 times per wall second. In a sim sped up in the app it gets one command per many sim
        seconds and tumbles (FINDINGS 6.14), so such a run is flagged once and flown with the jets off."""
        now, t = time.monotonic(), self.time()
        w0, t0 = getattr(self, "_speed_ref", (now, t))
        if now - w0 < span:
            self._speed_ref = (w0, t0); return
        ratio = (t - t0) / (now - w0); self._speed_ref = (now, t)
        if ratio > limit and not self.invalid:
            self.invalid = True
            self.log(f"SIM RUNNING ~{ratio:.0f}x FASTER THAN REAL TIME - run invalid, jets off. Fly at 1x.", error=True)

    def time(self):
        return (st.SimGlobals.SimClock.GetTimeNow().as_datetime() - self.t0).total_seconds()

    def _where(self, entity):
        s = entity.getLocVelAcc()
        return np.array(s.loc_WRT_ExprIn(self.pf), float), np.array(s.vel_WRT_ExprIn(self.pf), float)

    def state(self):
        return self._where(self.api.shuttle)

    def target(self):
        return self._where(self.api.target)[0]

    def rates(self):
        return np.array(self.api.shuttle.getAngVelocity().WRT(self.pf).ExprIn(self.sf), float)

    def to_body(self, vec):
        return np.array(st.frames.FramedVector(np.asarray(vec, float), self.pf).ExprIn(self.sf), float)

    def body_quaternion(self):
        from scipy.spatial.transform import Rotation   # imported here: the sim clock runs during imports (v10 fast start)
        to_body = np.column_stack([self.to_body(axis) for axis in np.eye(3)])
        return Rotation.from_matrix(to_body.T).as_quat()

    def fuel(self):
        return self.api.shuttle.GetParam(st.VarType.double, ["Stats", "FuelUsed"])

    def damage(self):
        return self.api.shuttle.GetParam(st.VarType.double, ["Stats", "Damage"])

    def radar(self):
        return [np.array(e, float) for e in self.api.GetNearestDebrisLocations(500)]

    def command(self, lin, ang):
        if self.invalid:
            lin = ang = np.zeros(3)
        self.api.SetCommand(np.asarray(lin, float), np.asarray(ang, float))

    def capture(self, top, yaw_deg, pitch_deg, fov_deg=Vision.FOV):
        camera = self.api.topCamera if top else self.api.bottomCamera
        self.api.PanTiltCamera(camera, yaw_deg, pitch_deg)
        return self.capture_fixed(top, fov_deg)

    def capture_fixed(self, top, fov_deg=Vision.FOV):
        camera = self.api.topCamera if top else self.api.bottomCamera
        shot = CaptureCameraImage(camera, Vision.SIZE, Vision.SIZE, fov_deg)
        if shot is None:
            return None
        image = shot.as_RGB8(); n = Vision.SIZE
        return np.dstack([np.asarray(c).reshape(n, n) for c in (image.PixelsR, image.PixelsG, image.PixelsB)]).astype(np.uint8)

    def log(self, msg, error=False):
        (st.logger_error if error else st.logger_info)(msg)

    def choices(self):
        """Parts named in the simconf's Controller Inst_Parameters (sims/PLAN_*.simconf); a part the simconf
        does not name (the organizers' own simconfs name none) keeps its line at the top of this file."""
        out = {}
        for key in ("Planner", "Variant", "Tracker", "Burn", "Camera", "Segmenter", "Record", "FlushLog", "ManeuverTest"):
            try:
                value = self.api.this.GetParam(st.VarType.string, key)
            except Exception:               # no such parameter
                continue
            if value:
                if key == "FlushLog":
                    out["flush_immediately"] = str(value).strip().lower() in ("1", "true", "yes", "on")
                else:
                    out[key.lower()] = value
        return out

    def web_run(self):
        try:
            value = self.api.this.GetParam(st.VarType.string, "WebRun")
        except Exception:  # absent in ordinary scenes
            return None
        return json.loads(value) if value else None


class SpaceTeamsTrainingSource:
    """Unwired adapter for TrainingRecorder; constructing it does not start captures.

    Uses the existing organizer PanTiltCamera helper. Never queries debris entities.
    """
    def __init__(self, api):
        self.sim = SpaceTeamsSim(api)
        self.api = api

    def time(self):
        return self.sim.time()

    def _camera(self, spec):
        return self.api.topCamera if spec.camera == "top" else self.api.bottomCamera

    def point(self, spec):
        self.api.PanTiltCamera(self._camera(spec), spec.yaw_deg, spec.pitch_deg)

    def capture(self, spec):
        params = st.CaptureImageProperties()
        params.nonphys_EV = spec.exposure_ev
        params.FOV = spec.fov_deg
        params.ResolutionX, params.ResolutionY = spec.width, spec.height
        if hasattr(params, "disable_bloom"):
            params.disable_bloom = spec.disable_bloom
        elif spec.disable_bloom:
            raise RuntimeError("Installed engine does not support disable_bloom")
        shot = st.camera.CaptureImageBlocking(self._camera(spec), params)
        if shot is None:
            return None
        rgb = shot.as_RGB8()
        return dict(R=np.asarray(rgb.PixelsR), G=np.asarray(rgb.PixelsG), B=np.asarray(rgb.PixelsB))

    def read_radar(self):
        return self.sim.radar()

    def snapshot(self, spec):
        # API reads are sequential, so record the interval rather than claiming
        # all state and radar observations were acquired simultaneously.
        out = {"read_start_sim_s": self.time(), "frame": "Phobos body-fixed, metres and seconds"}
        queries = {
            "shuttle_state": self.sim.state,
            "target_position": self.sim.target,
            "body_rates": self.sim.rates,
            "body_from_phobos": lambda: np.column_stack([self.sim.to_body(e) for e in np.eye(3)]),
            "fuel_kg": self.sim.fuel,
            "damage_J": self.sim.damage,
            "radar_raw_3x3": self.read_radar,
            "camera_position_phobos": lambda: self.sim._where(self._camera(spec))[0],
            "camera_from_phobos": lambda: np.column_stack([
                np.array(st.frames.FramedVector(e, self.sim.pf).ExprIn(self._camera(spec).GetBodyFixedFrame()), float)
                for e in np.eye(3)]),
        }
        for name, query in queries.items():
            try:
                out[name] = query()
            except Exception as exc:
                out[name + "_error"] = str(exc)
        out["read_end_sim_s"] = self.time()
        return out


class _CameraAuditProxy:
    """Forwards Sim calls to `sim` while recording PanTiltCamera/CaptureCameraImage into `run_logger`."""

    def __init__(self, sim, run_logger, t_fn):
        self._sim = sim
        self._run_logger = run_logger
        self._t_fn = t_fn

    def __getattr__(self, name):
        return getattr(self._sim, name)

    def capture(self, top, yaw_deg, pitch_deg, fov_deg=Vision.FOV):
        img = self._sim.capture(top, yaw_deg, pitch_deg, fov_deg)
        if self._run_logger is not None:
            self._run_logger.log_camera(self._t_fn(), top, yaw_deg, pitch_deg, fov_deg, image=img)
        return img


def fly(sim, planner=None, tracker=None, burn=None, camera=None, segmenter=None, threaded=True,
        log_dir=None, flush_immediately=None, log_run=None, seed=None, config=None,
        source=PYTHON_SOURCE, live=False, max_time=None, stop_file=None, runner=None,
        score_summary=None):
    """The flight loop, on any Interfaces.Sim. Parts left as None are the choices at the top of this file.
    Returns the Flight when sim.wait() ends the loop (stand-in sims only)."""
    parts = dict(planner=PLANNER, tracker=TRACKER, burn=BURN, camera=CAMERA, segmenter=SEGMENTER)
    chosen_planner = planner if planner is not None else parts["planner"]
    plan_cfg = config if config is not None else get_config(chosen_planner)
    if plan_cfg:
        if camera is None and plan_cfg.get("camera"):
            parts["camera"] = plan_cfg["camera"]
        if tracker is None and plan_cfg.get("tracker"):
            parts["tracker"] = plan_cfg["tracker"]
        if burn is None and plan_cfg.get("burn"):
            parts["burn"] = plan_cfg["burn"]
    parts.update({k: x for k, x in dict(planner=planner, tracker=tracker, burn=burn, camera=camera, segmenter=segmenter).items() if x is not None})
    burn_law = pick(Actuation.BURNS, parts["burn"], "burn law")
    cam = Vision.camera_use(parts["camera"], parts["segmenter"])
    zero = np.zeros(3)

    # The sim clock is already running: scipy is imported where it is used (not at import time, which cost 3.3-3.6 s of
    # sim time before the first burn in the real sim) and warmed here while the shuttle already flies.
    threading.Thread(target=lambda: [__import__(m) for m in ("scipy.special", "scipy.optimize", "scipy.spatial.transform")], daemon=True).start()
    r, v = sim.state(); target = sim.target()
    plan_cfg = get_config(parts["planner"]) if config is None else config
    Actuation.THRUST = float(plan_cfg.get("max_thrust", 0.5))
    a_lo, a_hi = plan_cfg.get("a_burn_min", 0.06), plan_cfg.get("a_burn_max", 0.20)
    Core.configure(r, target, a_burn=plan_cfg.get("a_burn") or burn_law.accel)
    do_log = LOG_RUN if log_run is None else bool(log_run)
    # Engine-launched runs loop until the engine kills the process, so finalize/atexit never run: stream run.log to disk.
    engine_unbounded = source == ENGINE_SOURCE and max_time is None and stop_file is None
    flush_flag = (FLUSH_IMMEDIATELY or engine_unbounded) if flush_immediately is None else bool(flush_immediately)
    run_logger = RunLogger(
        parts["planner"],
        base_dir=log_dir or DEFAULT_LOG_DIR,
        flush_immediately=flush_flag,
        runner=runner or type(sim).__name__,
        parts=parts,
        config=config,
        seed=seed,
        source=source,
        live=live,
        snapshot_s=SNAPSHOT_S if engine_unbounded else None,
    ) if do_log else None

    if run_logger is not None:
        run_logger.log_api_call(0.0, "sim.state+sim.target", "", f"start={r.tolist()} target={target.tolist()}")

    flight = Flight(parts["planner"], parts["tracker"], threaded=threaded, log=sim.log,
                    run_logger=run_logger, config=config, empty_field=source == PYTHON_SOURCE)
    flight.camera = cam
    flight.run_logger = run_logger
    flight.run_dir = run_logger.run_dir if run_logger is not None else None
    if plan_cfg.get("timed_burns") or plan_cfg.get("predict_attitude"):
        flight.body_state = lambda: (np.array([sim.to_body(e) for e in np.eye(3)]), sim.rates())
    elif plan_cfg.get("w_turn"):                                     # the planner prices turning the burn (FINDINGS 7.17)
        flight.body_z = lambda: np.array([sim.to_body(e)[2] for e in np.eye(3)])
    sim.log("Team Batasha controller: " + ", ".join(f"{k} {x}" for k, x in parts.items()))

    memory = {"config": plan_cfg}; tick = 0; t_prev = sim.time(); arrived = False; t_tm = -9.0; n_tm = 0; v_tm = v.copy(); tm_main = False
    t = t_prev; fuel_now = 0.0
    arrival_time = None
    axial_window = True
    terminal_reason = None
    next_damage_check = -np.inf
    damage_now = 0.0
    read_damage = getattr(sim, "damage", None) if source == ENGINE_SOURCE else None
    body_quaternion = getattr(sim, "body_quaternion", lambda: None)
    log_scene = getattr(sim, "log_scene", None)
    next_scene_log = t
    cam_sim = _CameraAuditProxy(sim, run_logger, lambda: t) if run_logger is not None else sim

    def summary_now() -> dict:
        summary = {
            "arrived": bool(arrived),
            "duration_s": round(float(t), 4),
            "final_fuel_kg": round(float(fuel_now), 4),
            "final_position": r.tolist(),
            "final_velocity": v.tolist(),
            "ticks": int(tick),
        }
        if arrival_time is not None:
            summary["arrival_s"] = round(float(arrival_time), 4)
        if score_summary is not None:
            summary.update(score_summary())
        if terminal_reason is not None:
            summary.update(arrived=False, outcome=terminal_reason)
        if source == ENGINE_SOURCE:
            try:
                summary["damage_kJ"] = round(float(sim.damage()) / 1000.0, 4)
            except Exception:
                pass
            if arrival_time is not None:
                # The official score only reaches the engine's own logs (OnScreenAlert), never this process.
                # Scoring's clock starts ~1.3 s after ours, so this estimate runs ~0.4 points low.
                from Systems.ScoreMath import score_from_metrics
                est = score_from_metrics(arrival_time, fuel_now, summary.get("damage_kJ", 0.0))
                summary["score_estimate"] = round(float(est["final_score"]), 2)
        return summary

    try:
        while True:
            try:
                t = sim.time(); dt = max(t - t_prev, 1e-3); t_prev = t
                r, v = sim.state()
                fuel_now = sim.fuel()
                if run_logger is not None and log_scene is not None and t >= next_scene_log:
                    log_scene(run_logger)
                    next_scene_log = t + 0.5
                in_goal = np.linalg.norm(r - target) <= Core.WIN
                if not arrived and terminal_reason is None and read_damage is not None and (t >= next_damage_check or in_goal):
                    damage_now = float(read_damage())
                    next_damage_check = t + 0.2
                    if damage_now > Core.DMAX:
                        terminal_reason = "damage_limit"
                        message = f"Damage limit exceeded at t={t:.2f} s ({damage_now:.2f} J). Commands disabled."
                        sim.log(message)
                        if run_logger is not None:
                            run_logger.log_line(t, "NAV", message)
                if not arrived and terminal_reason is None and in_goal:
                    arrived = True; arrival_time = t
                    sim.log(f"Arrived at t={t:.1f} s")
                    if run_logger is not None:
                        run_logger.log_line(t, "NAV", f"Arrived within WIN={Core.WIN}m at pos={r.tolist()} fuel={fuel_now:.3f}")
                if arrived or terminal_reason is not None:
                    lin_cmd, ang_cmd = zero, zero
                    sim.command(lin_cmd, ang_cmd)
                    if run_logger is not None:
                        run_logger.log_tick(t, r, v, fuel_now, terminal_reason or "arrived", zero, lin_cmd, ang_cmd,
                                            flight.idx, flight.active_point, guarding=False, a_burn=Core.A_BURN,
                                            body_quaternion=body_quaternion())
                        if engine_unbounded:
                            # The engine kills this process seconds after arrival, before `finally` runs: save the whole run now.
                            run_logger.finalize(summary_now())
                            run_logger = None
                else:
                    if tick % RADAR_EVERY == 0:
                        entries = sim.radar()
                        flight.radar(entries, t)
                    w = sim.rates()
                    mode, vec = flight.decide(t, r, v, fuel_now, dt)
                    memory["launch_released"] = flight.launch is not None and flight.launch.done
                    flight.guard_ticks += int(flight.guarding)
                    lin_cmd, ang_cmd = zero, zero
                    if mode == "launch_failed":
                        terminal_reason = "launch_timeout"
                        message = f"Launch clearance deadline reached at t={t:.2f} s. Commands disabled."
                        sim.log(message)
                        if run_logger is not None:
                            run_logger.log_line(t, "NAV", message)
                    elif cam is not None and cam.due(t, mode == "coast" or (flight.departure is not None and not flight.guarding)):
                        sim.command(zero, zero)                          # a capture blocks 100-300 ms: leave nothing commanded
                        rep = cam(cam_sim, flight.world, t, r, v)
                        if isinstance(rep, dict):                        # Consensus: the line the team reads in the sim
                            sim.log(f"camera t={t:.1f} detected {rep['detected']} in_view {rep['in_view']} matched {rep['matched']} "
                                    f"shift {np.round(rep['shift'], 1).tolist()} layout {rep['layout']} used {rep['used']}")
                        if run_logger is not None:
                            run_logger.log_line(t, "CAMERA", f"camera_use report={rep}")
                    elif mode == "launch_evade":
                        lin_cmd = Actuation.THRUST * sim.to_body(vec)
                    elif mode in ("launch_lift", "launch_slew", "launch_slew_climb"):
                        lin_cmd = sim.to_body(vec)
                        if mode == "launch_lift":
                            ang_cmd = Actuation.coast_command(w)[1]
                        else:
                            ref = sim.to_body(target - r)
                            bearing = ref[None]
                            ready, _ = Actuation.x_forward_main_masks(bearing, bearing, w[None], False, plan_cfg)
                            memory["x_acquired"] = bool(ready[0])
                            ang_cmd = Actuation.launch_slew_attitude(ref, w, plan_cfg, early=mode == "launch_slew_climb")
                    elif mode == "body_command":
                        command = BODY_COMMANDS[flight.body_command]
                        lin_cmd, ang_cmd = command[:3], np.array([0.0, command[3], 0.0])
                    elif mode == "pulse":
                        lin_cmd = sim.to_body(vec) / 60.0
                        mode = "trim"
                    elif mode == "main":
                        d = sim.to_body(vec); ref = sim.to_body(target - r)
                        memory["ref"] = ref / np.linalg.norm(ref)
                        memory["ng"] = float(np.linalg.norm(d)); memory["dt"] = dt
                        lin_cmd, ang_cmd = burn_law(d / memory["ng"], w, memory)
                        if (flight.departure is not None or getattr(burn_law, "handles_main_attitude", False)) and not np.any(lin_cmd):
                            mode = "align"
                    elif mode == "trim":
                        trim_exact = bool(plan_cfg.get("trim_exact", False)) if plan_cfg else False
                        precision = plan_cfg.get("x_forward_precision_dv_mps")
                        if precision is not None and not flight.guarding and np.linalg.norm(vec) < precision:
                            lin_cmd = Actuation.x_forward_precision_command(sim.to_body(vec), plan_cfg)
                        elif trim_exact or parts["burn"] == "steer":
                            lin_cmd, ang_cmd = Actuation.trim_command(sim.to_body(vec), w=w, trim_exact=trim_exact or parts["burn"] == "steer")
                        else:
                            lin_cmd, ang_cmd = Actuation.trim_command(sim.to_body(vec))
                    elif mode == "evade":                                # full thrust any way: the escape guard's dodge
                        d = sim.to_body(vec); ref = sim.to_body(target - r); memory["ref"] = ref / np.linalg.norm(ref)
                        if plan_cfg.get("guard_translation_only"):
                            lin_cmd = Actuation.THRUST * d / np.linalg.norm(d)
                        elif parts["burn"] == "jets":                    # plan G1: one body axis, no turning command
                            lin_cmd, ang_cmd = Actuation.trim_command(d)
                        else:
                            lin_cmd, ang_cmd = Actuation.direct_burn(d / np.linalg.norm(d), w, memory)
                    else:
                        if flight.departure is None:
                            lin_cmd, ang_cmd = Actuation.coast_command(w)
                    attitude = getattr(burn_law, "attitude", None)
                    if attitude is not None and mode != "body_command" and not mode.startswith("launch_"):
                        ref = sim.to_body(target - r)
                        coast_attitude = getattr(burn_law, "coast_attitude", None)
                        if mode == "coast" and coast_attitude is not None and plan_cfg.get("coast_pointing_deadband"):
                            ang_cmd = coast_attitude(ref, w, memory, plan_cfg)
                        elif mode not in ("main", "align") or not getattr(burn_law, "handles_main_attitude", False):
                            memory.pop("coast_pointing", None)
                            ang_cmd = attitude(ref, w)
                    if mode != "coast":
                        memory.pop("coast_pointing", None)
                    flight.coast_pointing = memory.get("coast_pointing", False)
                    flight.main_axis_acquired = memory.get("x_acquired", False)
                    sim.command(lin_cmd, ang_cmd)
                    if plan_cfg.get("launch_clearance_m"):
                        axial_window &= (mode == "main" and lin_cmd[0] > 0
                                         and not np.any(lin_cmd[1:]) and not np.any(ang_cmd))
                    if run_logger is not None:
                        run_logger.log_tick(t, r, v, fuel_now, mode, vec, lin_cmd, ang_cmd,
                                            flight.idx, flight.active_point, guarding=flight.guarding,
                                            a_burn=Core.A_BURN, w=w, barrier_active=flight.barrier_active,
                                            body_quaternion=body_quaternion())
                    if t - t_tm >= 2.0:                                  # telemetry: the only record of what the sim did
                        forward = plan_cfg.get("pointing_axis", 2)
                        off = np.degrees(np.arccos(np.clip(sim.to_body(vec)[forward] / max(np.linalg.norm(vec), 1e-9), -1, 1))) if mode == "main" else 0.0
                        sim.log(f"tm t={t:.1f} dist={np.linalg.norm(r - target):.0f} speed={np.linalg.norm(v):.2f} closing={v @ (target - r) / np.linalg.norm(target - r):.2f} "
                                f"fuel={fuel_now:.1f} a_burn={Core.A_BURN:.3f} mode={mode}{'/guard' if flight.guarding else ''} gain={np.linalg.norm(vec):.2f} rate={np.linalg.norm(w):.3f} off={off:.1f}deg "
                                f"steer={'OFF' if memory.get('steer_off') else 'on'} rocks={flight.world.n} hz={(tick - n_tm) / max(t - t_tm, 1e-3):.0f}")
                        if mode == "main" and tm_main and axial_window and t - t_tm < 3.0:
                            gravity = Core.accel(r, v, frame=bool(plan_cfg.get("launch_clearance_m")
                                                                  or plan_cfg.get("launch_burn_height_m")))
                            a_now = float(np.linalg.norm(v - v_tm - gravity * (t - t_tm)) / (t - t_tm))
                            Core.A_BURN = float(np.clip(0.7 * Core.A_BURN + 0.3 * a_now, a_lo, a_hi))
                        t_tm = t; n_tm = tick; v_tm = v.copy(); tm_main = mode == "main"
                        axial_window = True
            except Exception:                                               # never leave the jets on after an error
                tb = traceback.format_exc()
                sim.log("Controller tick failed:\n" + tb, error=True)
                if run_logger is not None:
                    run_logger.log_line(t, "ERROR", "Controller tick failed:\n" + tb)
                try:
                    sim.command(zero, zero)
                except Exception:
                    pass
            tick += 1
            if max_time is not None or stop_file is not None:
                # Keep the controller connected briefly so the scoring system can observe arrival.
                arrived_and_settled = arrival_time is not None and t >= arrival_time + 1.0
                if terminal_reason is not None or arrived_and_settled or (max_time is not None and t >= max_time) or (
                    stop_file is not None and stop_file.is_file()
                ):
                    sim.command(zero, zero)
                    return flight
            if not sim.wait():
                if run_logger is not None:
                    terminal_time = sim.time()
                    if terminal_time > t:
                        t = terminal_time
                        r, v = sim.state()
                        fuel_now = sim.fuel()
                        if not arrived and terminal_reason is None:
                            if read_damage is not None and float(read_damage()) > Core.DMAX:
                                terminal_reason = "damage_limit"
                            elif np.linalg.norm(r - target) <= Core.WIN:
                                arrived = True
                                arrival_time = t
                        run_logger.log_tick(t, r, v, fuel_now, terminal_reason or ("arrived" if arrived else "stopped"),
                                            zero, zero, zero, flight.idx, flight.active_point, terminal=True,
                                            body_quaternion=body_quaternion())
                        if log_scene is not None:
                            log_scene(run_logger)
                        tick += 1
                return flight
    finally:
        if run_logger is not None:
            run_logger.finalize(summary_now())


def run(api) -> None:
    sim = SpaceTeamsSim(api); parts = sim.choices()
    if parts.pop("maneuvertest", None) == "lift_rotate":
        from Systems.LiftRotate import run as lift_rotate

        lift_rotate(sim)
        return
    variant = parts.pop("variant", None)
    if variant:                                 # sims/PLAN_<n>_<name>_<variant>.simconf: configs/<variant>.json of that plan
        plan = parts.get("planner", PLANNER)
        parts["config"] = {**get_config(plan), **read_config_json(PLAN_DIRS[plan] / "configs" / f"{variant}.json")}
    web = sim.web_run()
    if web is not None:
        plan = parts.get("planner", PLANNER)
        defaults = get_config(plan)
        config = {**defaults, **extract_plan_overrides(web["config"], plan, defaults)}
        log_dir = Path(web["log_dir"])
        parts.update(config=config, log_dir=log_dir, live=True, max_time=web["max_time"],
                     stop_file=log_dir / "stop", runner=web["scene"])
    if parts.pop("record", RECORD):             # save the flight for tools/replay.py
        sim = Recorder(sim, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "flightlogs"))
    fly(sim, source=ENGINE_SOURCE, **parts)
