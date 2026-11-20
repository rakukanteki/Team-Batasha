"""
Systems/RunLogger.py - auditable, retraceable per-run logging and JSON export.

Creates a per-run folder under `<base_dir>/<simulator>/`:
    run.log     full verbose, reproducible, auditable flight log
    run.json    structured telemetry (time->fuel, time->thrust, time->point, points array, radar_calls)
    images/     camera images captured during the run (and reserved for future vision captures)

By default (`flush_immediately=False`), all records stay in memory during the flight loop
and are written to disk only when `finalize()` runs, avoiding I/O overhead in the 60 Hz loop.
"""
from __future__ import annotations

import atexit
import json
import os
import platform
import struct
import subprocess
import threading
import time
import uuid
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from Systems import Core as C
from Systems.plans import get_config, plan_name as canonical_plan_name

# BATASHA_LOG_DIR redirects every default log (the test suite points it at a temp folder).
DEFAULT_LOG_DIR = Path(os.environ.get("BATASHA_LOG_DIR") or Path(__file__).resolve().parent.parent / "flightlogs")
PYTHON_SOURCE = "python"
ENGINE_SOURCE = "spaceteams"


def simulation_source(metadata: dict) -> str:
    source = metadata.get("source")
    if source in (PYTHON_SOURCE, ENGINE_SOURCE):
        return source
    runner = metadata.get("runner")
    if runner in ("SpaceTeamsSim", "Recorder"):
        return ENGINE_SOURCE
    if runner in ("offline_bench", "FieldSim", "TwinSim", "_CleanTwinSim", "point_mass",
                  "competition_simconf", "stress_storm", "jet_twin_field", "jet_twin_clean"):
        return PYTHON_SOURCE
    return "unknown"


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _fmt_vec(v) -> str:
    if v is None:
        return "None"
    arr = np.asarray(v, float).ravel()
    return "[" + ", ".join(f"{x:.4f}" for x in arr) + "]"


def _encode_png_rgb(img: np.ndarray) -> bytes:
    """Minimal pure-Python PNG encoder for HxWx3 uint8 arrays (no Pillow dependency)."""
    arr = np.ascontiguousarray(img, dtype=np.uint8)
    h, w = arr.shape[:2]
    if arr.ndim == 2:
        arr = np.dstack([arr, arr, arr])
    raw = b"".join(b"\x00" + arr[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, level=1)) + chunk(b"IEND", b"")


def _git_info() -> tuple[str | None, str | None]:
    root = Path(__file__).resolve().parent.parent
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
        return commit, branch
    except Exception:
        return None, None


class RunLogger:
    """Buffers verbose flight logs and structured JSON telemetry in memory until `finalize()`,
    unless `flush_immediately=True` is set."""

    def __init__(
        self,
        plan: Any,
        base_dir: str | Path | None = None,
        flush_immediately: bool = False,
        runner: str = "unknown",
        parts: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
        seed: int | None = None,
        source: str | None = None,
        live: bool = False,
        snapshot_s: float | None = None,
    ):
        self.plan_name = canonical_plan_name(plan)
        self.source = source or simulation_source({"runner": runner})
        root = Path(base_dir) if base_dir is not None else DEFAULT_LOG_DIR
        self.base_dir = root / self.source if self.source != "unknown" else root
        self.flush_immediately = bool(flush_immediately)
        self.runner = runner
        self.parts = dict(parts or {})
        base_cfg = get_config(plan)
        self.config = _to_jsonable(config if config is not None else dict(base_cfg))
        self.seed = seed
        self.timestamp_local = datetime.now().astimezone().isoformat(timespec="seconds")
        self.timestamp_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._stamp = time.strftime("%Y%m%d_%H%M%S")
        self.run_id = f"{self.plan_name}__{self._stamp}__{uuid.uuid4().hex[:8]}"
        self.run_dir = self.base_dir / self.run_id
        self.log_path = self.run_dir / "run.log"
        self.json_path = self.run_dir / "run.json"
        self.images_dir = self.run_dir / "images"

        self._lock = threading.Lock()
        self._log_lines: list[str] = []
        self._log_fp = None
        self._live_fp = None
        self._last_live = -float("inf")
        self._last_live_radar = -float("inf")
        self._live_radar = None
        self._finalized = False
        self._summary: dict[str, Any] = {}
        self._json_lock = threading.Lock()     # serialises run.json writers (snapshot thread vs finalize)
        self._stop_snapshots = threading.Event()

        # Structured mappings for JSON export
        self._time_to_fuel: dict[str, float] = {}
        self._time_to_thrust: dict[str, dict[str, Any]] = {}
        self._time_to_point: dict[str, dict[str, Any]] = {}
        self._points: list[dict[str, Any]] = []
        self._radar_raw = []
        self._scene_frames = []
        self._camera_images: list[dict[str, Any]] = []
        self._pending_images: list[tuple[str, np.ndarray]] = []

        self._git_commit, self._git_branch = _git_info()
        self._core_snapshot = {
            "start": C.S.tolist(),
            "target": C.T.tolist(),
            "L": float(C.L),
            "mu": float(C.MU),
            "mass": float(C.MASS),
            "kf": float(C.KF),
            "a_burn": float(C.A_BURN),
            "a_trim": float(C.A_TRIM),
            "trim_l1": float(C.TRIM_L1),
            "main_min": float(C.MAIN_MIN),
            "r_sh": float(C.R_SH),
            "shape_factor": float(C.SHAPE),
            "win_radius": float(C.WIN),
            "eps_on": float(C.EPS_ON),
            "terminal": float(C.TERMINAL),
            "lambda_node": float(C.LAMBDA_NODE),
            "p_safe": float(C.P_SAFE),
            "sig_ignore": float(C.SIG_IGNORE),
            "t_ignore": float(C.T_IGNORE),
            "switch_margin": float(C.SWITCH),
            "vcs": list(C.VCS),
            "dt_plan": float(C.DT_PLAN),
        }

        self._build_header()
        if self.flush_immediately:
            self._open_immediate()
        if live:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self._live_fp = (self.run_dir / "live.jsonl").open("w", encoding="utf-8", buffering=1)
            self._write_live({"metadata": {
                "source": self.source, "runner": self.runner, "plan_name": self.plan_name,
                "core": self._core_snapshot, "config": self.config,
            }})
        atexit.register(self._atexit_finalize)
        if snapshot_s:
            threading.Thread(target=self._snapshot_loop, args=(float(snapshot_s),), daemon=True,
                             name="RunLogger-snapshot").start()

    def _snapshot_loop(self, every_s: float) -> None:
        """Rewrite run.json every `every_s` so a run killed by the engine still leaves a usable file."""
        while not self._stop_snapshots.wait(every_s):
            try:
                self.snapshot()
            except Exception:
                pass

    def snapshot(self) -> None:
        """Write the run so far to run.json (marked partial) without finalizing."""
        if self._finalized:
            return
        with self._lock:
            summary = {**self._summary, **self._derived_summary(), "partial": True}
            parts = self._payload_parts(summary)
        self._write_json(*parts)

    def _write_live(self, row: dict) -> None:
        self._live_fp.write(json.dumps(row, separators=(",", ":")) + "\n")

    def _build_header(self) -> None:
        header = [
            "=" * 96,
            f"RUN LOG: {self.run_id}",
            f"timestamp_local={self.timestamp_local}  timestamp_utc={self.timestamp_utc}",
            f"runner={self.runner}  plan={self.plan_name}  seed={self.seed}  flush_immediately={self.flush_immediately}  source={self.source}",
            f"git_branch={self._git_branch}  git_commit={self._git_commit}  python={platform.python_version()}",
            f"parts={json.dumps(_to_jsonable(self.parts), sort_keys=True)}",
            f"plan_config={json.dumps(self.config, sort_keys=True)}",
            f"core_constants={json.dumps(self._core_snapshot, sort_keys=True)}",
            f"images_dir={self.images_dir}",
            f"run_dir={self.run_dir}",
            f"run_log={self.log_path}",
            f"run_json={self.json_path}",
            "=" * 96,
        ]
        self._log_lines.extend(header)

    def _open_immediate(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._log_fp = open(self.log_path, "w", encoding="utf-8", buffering=1)
        for line in self._log_lines:
            self._log_fp.write(line + "\n")
        self._log_fp.flush()

    def _append_line(self, line: str) -> None:
        with self._lock:
            self._log_lines.append(line)
            if self.flush_immediately and self._log_fp is not None:
                self._log_fp.write(line + "\n")
                self._log_fp.flush()

    def log_line(self, t: float, category: str, msg: str) -> None:
        self._append_line(f"[t={t:9.3f}s] [{category}] {msg}")

    def log_api_call(self, t: float, api_name: str, args_desc: str = "", result_desc: str = "") -> None:
        parts = [api_name]
        if args_desc:
            parts.append(f"args=({args_desc})")
        if result_desc:
            parts.append(f"-> {result_desc}")
        self.log_line(t, "API", " ".join(parts))

    def log_scene(self, t, positions, diameters, rotations):
        self._scene_frames.append({
            "t": round(t, 4), "positions_abs": np.round(positions, 4).tolist(),
            "diameters": np.asarray(diameters).tolist(), "rotations_xyzw": np.round(rotations, 7).tolist(),
        })

    def log_radar(self, t: float, shuttle_pos: np.ndarray, entries: list[np.ndarray],
                  changed_count: int | None = None, origins=None) -> None:
        n = len(entries)
        r = np.asarray(shuttle_pos, float)
        if n > 0:
            arr = np.asarray(entries, float)
            pos_rel = arr[:, 0, :].copy()
            diam = arr[:, 1, 2].copy()
            pos_abs = pos_rel + (r if origins is None else np.asarray(origins))
            t_min, t_max = float(arr[:, 2, 0].min()), float(arr[:, 2, 0].max())
            nearest_rel = pos_rel[0]
            with self._lock:
                self._radar_raw.append((float(t), r.copy(), pos_rel, pos_abs, diam))
                if self._live_fp is not None and t >= self._last_live_radar + 1:
                    self._last_live_radar = t
                    self._live_radar = {
                        "t": float(t), "positions_abs": np.round(pos_abs, 2).tolist(),
                        "diameters": np.round(diam, 2).tolist(),
                    }
            chg = f" changed={changed_count}" if changed_count is not None else ""
            self.log_api_call(
                t,
                "GetNearestDebrisLocations",
                "n=500",
                f"entries={n}{chg} radar_t=[{t_min:.3f}..{t_max:.3f}] shuttle={_fmt_vec(r)} nearest_rel={_fmt_vec(nearest_rel)}",
            )
        else:
            self.log_api_call(t, "GetNearestDebrisLocations", "n=500", f"entries=0 shuttle={_fmt_vec(r)}")

    def log_plan(self, t: float, plan: tuple, info: dict[str, Any]) -> None:
        nodes = [np.asarray(p, float).tolist() for p in plan[0]]
        turn_pts = nodes[:-1]
        vc = float(plan[1])
        hold = None if len(plan) <= 2 else float(plan[2])
        entry = {
            "t": round(float(t), 4),
            "event": "plan",
            "action": info.get("action"),
            "vc": vc,
            "hold": hold,
            "cost": float(info.get("cost", 0.0)),
            "ptot": float(info.get("ptot", 0.0)),
            "clean": float(info.get("clean", 0.0)),
            "clear": float(info.get("clear", 0.0)),
            "num_turns": int(info.get("K", max(len(nodes) - 1, 0))),
            "waypoints": nodes,
            "turn_points": turn_pts,
        }
        for key in ("lattice_points", "astar_points", "station_points", "cells", "stats",
                    "predicted_arrival_s", "predicted_fuel_kg"):
            if key in info and info[key] is not None:
                entry[key] = _to_jsonable(info[key])
        with self._lock:
            self._points.append(entry)

        extra_bits = []
        if "cells" in entry:
            extra_bits.append(f"cells={entry['cells']}")
        if "astar_points" in entry:
            extra_bits.append(f"astar_points={len(entry['astar_points'])}")
        if "lattice_points" in entry:
            extra_bits.append(f"lattice_points={len(entry['lattice_points'])}")
        if "stats" in entry:
            extra_bits.append(f"stats={json.dumps(entry['stats'])}")
        extra_str = (" " + " ".join(extra_bits)) if extra_bits else ""
        self.log_line(
            t,
            "PLAN",
            f"action={entry['action']} vc={vc:.2f} hold={hold} K={entry['num_turns']} "
            f"cost={entry['cost']:.3f} ptot={entry['ptot']:.6f} clean={entry['clean']:.3f} clear={entry['clear']:.2f} "
            f"waypoints={json.dumps([[round(x, 3) for x in p] for p in nodes])} "
            f"turns={json.dumps([[round(x, 3) for x in p] for p in turn_pts])}{extra_str}",
        )

    def log_turn_passed(self, t: float, node_idx: int, point: np.ndarray, shuttle_pos: np.ndarray) -> None:
        pt = np.asarray(point, float).tolist()
        pos = np.asarray(shuttle_pos, float).tolist()
        with self._lock:
            self._points.append({
                "t": round(float(t), 4),
                "event": "turn_passed",
                "node_idx": int(node_idx),
                "point": pt,
                "shuttle_pos": pos,
            })
        self.log_line(t, "NAV", f"passed_waypoint idx={node_idx} point={_fmt_vec(pt)} shuttle_pos={_fmt_vec(pos)}")

    def log_tick(
        self,
        t: float,
        r: np.ndarray,
        v: np.ndarray,
        fuel: float,
        mode: str,
        vec: np.ndarray,
        lin_cmd: np.ndarray,
        ang_cmd: np.ndarray,
        target_node_idx: int,
        target_point: np.ndarray,
        guarding: bool = False,
        a_burn: float | None = None,
        w: np.ndarray | None = None,
        barrier_active: bool = False,
        terminal: bool = False,
        body_quaternion=None,
    ) -> None:
        tk = f"{float(t):.4f}"
        r_arr = np.asarray(r, float)
        v_arr = np.asarray(v, float)
        vec_arr = np.asarray(vec, float)
        lin_arr = np.asarray(lin_cmd, float)
        ang_arr = np.asarray(ang_cmd, float)
        tp_arr = np.asarray(target_point, float)
        lin_mag = float(np.linalg.norm(lin_arr))
        ang_mag = float(np.linalg.norm(ang_arr))
        vec_mag = float(np.linalg.norm(vec_arr))
        thrust_dir_world = (vec_arr / vec_mag).tolist() if (mode != "coast" and vec_mag > 1e-9) else [0.0, 0.0, 0.0]

        with self._lock:
            self._time_to_fuel[tk] = round(float(fuel), 6)
            self._time_to_thrust[tk] = {
                "mode": mode,
                "lin_cmd": lin_arr.tolist(),
                "ang_cmd": ang_arr.tolist(),
                "lin_mag": round(lin_mag, 6),
                "ang_mag": round(ang_mag, 6),
                "thrust_dir_world": thrust_dir_world,
                "a_burn": float(C.A_BURN if a_burn is None else a_burn),
            }
            self._time_to_point[tk] = {
                "shuttle_pos": r_arr.tolist(),
                "shuttle_vel": v_arr.tolist(),
                "target_node_idx": int(target_node_idx),
                "target_point": tp_arr.tolist(),
                "mode": mode,
                "guarding": bool(guarding),
            }
            orientation = {} if body_quaternion is None else {
                "body_quaternion_xyzw": np.round(body_quaternion, 8).tolist(),
            }
            self._time_to_point[tk].update(orientation)
            if self._live_fp is not None and not self._finalized and (terminal or t >= self._last_live + 0.25):
                self._last_live = t
                self._write_live({
                    "t": float(t), "pos": r_arr.tolist(), "vel": v_arr.tolist(),
                    "fuel": float(fuel), "mode": mode, "guarding": bool(guarding),
                    "aim": tp_arr.tolist(), "node_idx": int(target_node_idx),
                    "speed": float(np.linalg.norm(v_arr)),
                    "dist_goal": float(np.linalg.norm(r_arr - C.T)),
                    "lin_mag": lin_mag, "ang_mag": ang_mag, "thrust_dir": thrust_dir_world,
                    "radar": self._live_radar,
                    **orientation,
                })
                self._live_radar = None

        dist_target = float(np.linalg.norm(r_arr - C.T))
        dist_node = float(np.linalg.norm(r_arr - tp_arr))
        w_str = f" w={_fmt_vec(w)}" if w is not None else ""
        guard_str = " GUARD=1" if guarding else (" BARRIER=1" if barrier_active else "")
        self._append_line(
            f"[t={t:9.3f}s] [NAV/CMD] mode={mode}{guard_str} node_idx={target_node_idx} "
            f"target_pt={_fmt_vec(tp_arr)} dist_node={dist_node:.2f} dist_goal={dist_target:.2f} "
            f"pos={_fmt_vec(r_arr)} vel={_fmt_vec(v_arr)} fuel={fuel:.3f} "
            f"vg={_fmt_vec(vec_arr)} SetCommand(lin={_fmt_vec(lin_arr)}, ang={_fmt_vec(ang_arr)}){w_str}"
        )

    def log_camera(
        self,
        t: float,
        top: bool,
        yaw_deg: float,
        pitch_deg: float,
        fov_deg: float,
        report: Any = None,
        image: np.ndarray | None = None,
    ) -> None:
        cam_name = "topCamera" if top else "bottomCamera"
        fname = None
        if image is not None:
            fname = f"cam_t{t:08.3f}_{cam_name}.png"
            if self.flush_immediately:
                self.images_dir.mkdir(parents=True, exist_ok=True)
                (self.images_dir / fname).write_bytes(_encode_png_rgb(image))
            else:
                with self._lock:
                    self._pending_images.append((fname, np.array(image, copy=True)))
        rec = {
            "t": round(float(t), 4),
            "camera": cam_name,
            "yaw_deg": float(yaw_deg),
            "pitch_deg": float(pitch_deg),
            "fov_deg": float(fov_deg),
            "filename": fname,
            "report": _to_jsonable(report),
        }
        with self._lock:
            self._camera_images.append(rec)
        self.log_api_call(
            t,
            "PanTiltCamera+CaptureCameraImage",
            f"cam={cam_name}, yaw={yaw_deg:.2f}, pitch={pitch_deg:.2f}, fov={fov_deg:.1f}",
            f"image={fname or 'none'} report={json.dumps(_to_jsonable(report))}",
        )

    def _atexit_finalize(self) -> None:
        if not self._finalized:
            try:
                self.finalize({"interrupted_or_atexit": True})
            except Exception:
                pass

    def _derived_summary(self) -> dict[str, Any]:
        """Compute burn, fuel, replan, cruise-speed, and predicted-risk metrics from logged ticks and plans."""
        main_s = trim_s = 0.0
        main_fuel = trim_fuel = avoidance_fuel = 0.0
        ks = sorted(self._time_to_thrust.keys(), key=float)
        ts = [float(k) for k in ks]
        for i, k in enumerate(ks[:-1]):
            dt = max(ts[i + 1] - ts[i], 0.0)
            df = max(float(self._time_to_fuel[ks[i + 1]]) - float(self._time_to_fuel[k]), 0.0)
            mode = self._time_to_thrust[k].get("mode")
            if self._time_to_point[k].get("guarding"):
                avoidance_fuel += df
            if mode == "main":
                main_s += dt
                main_fuel += df
            elif mode == "trim":
                trim_s += dt
                trim_fuel += df

        plan_events = [p for p in self._points if p.get("event") == "plan"]
        accepted_replans = sum(1 for p in plan_events[1:] if p.get("action") != "keep")
        cruise_speed = float(plan_events[-1]["vc"]) if plan_events else float(C.resolve_vc(self.config))
        out: dict[str, Any] = {
            "main_burn_s": round(main_s, 4),
            "trim_burn_s": round(trim_s, 4),
            "main_fuel_kg": round(main_fuel, 4),
            "trim_fuel_kg": round(trim_fuel, 4),
            "avoidance_fuel_kg": round(avoidance_fuel, 4),
            "accepted_replans": int(accepted_replans),
            "cruise_speed_mps": round(cruise_speed, 4),
            "predicted_risk": round(float(plan_events[-1]["ptot"]), 6) if plan_events else None,
        }
        if self._time_to_point:
            out["max_speed"] = round(
                max(float(np.linalg.norm(pt["shuttle_vel"])) for pt in self._time_to_point.values()),
                4,
            )
        return out

    def finalize(self, summary: dict[str, Any] | None = None) -> Path:
        with self._lock:
            if self._finalized:
                return self.run_dir
            self._finalized = True
            atexit.unregister(self._atexit_finalize)
            if self._live_fp is not None:
                self._live_fp.close()
                self._live_fp = None
            derived = self._derived_summary()
            if summary:
                self._summary.update(_to_jsonable(summary))
            for k, v in derived.items():
                self._summary.setdefault(k, v)

            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.images_dir.mkdir(parents=True, exist_ok=True)

            self._stop_snapshots.set()
            footer = [
                "=" * 96,
                f"RUN SUMMARY: {self.run_id}",
                f"ticks={len(self._time_to_point)}  plan_events={len(self._points)}  "
                f"radar_calls={len(self._radar_raw)}  camera_images={len(self._camera_images)}",
                f"summary={json.dumps(self._summary, sort_keys=True)}",
                f"saved: run_log={self.log_path}  run_json={self.json_path}  images_dir={self.images_dir}",
                "=" * 96,
            ]
            self._log_lines.extend(footer)

            if self._log_fp is not None:
                for line in footer:
                    self._log_fp.write(line + "\n")
                self._log_fp.flush()
                self._log_fp.close()
                self._log_fp = None
            else:
                self.log_path.write_text("\n".join(self._log_lines) + "\n", encoding="utf-8")

            for fname, img in self._pending_images:
                (self.images_dir / fname).write_bytes(_encode_png_rgb(img))
            self._pending_images.clear()

            self._write_json(*self._payload_parts(self._summary))
            return self.run_dir

    def _payload_parts(self, summary: dict[str, Any]) -> tuple:
        """Shallow copies of the run state, taken fast so JSON encoding can happen outside the lock."""
        return (dict(summary), dict(self._time_to_fuel), dict(self._time_to_thrust), dict(self._time_to_point),
                list(self._points), list(self._radar_raw), list(self._scene_frames), list(self._camera_images))

    def _write_json(self, summary, time_to_fuel, time_to_thrust, time_to_point, points, radar_raw,
                    scene_frames, camera_images) -> None:
        radar_calls = []
        for t, r_pos, pos_rel, pos_abs, diam in radar_raw:
            radar_calls.append({
                "t": round(float(t), 4),
                "shuttle_pos": r_pos.tolist(),
                "count": int(len(pos_rel)),
                "positions_rel": np.round(pos_rel, 4).tolist(),
                "positions_abs": np.round(pos_abs, 4).tolist(),
                "diameters": np.round(diam, 4).tolist(),
            })
        payload = {
            "metadata": {
                "run_id": self.run_id,
                "plan_name": self.plan_name,
                "plan_number": self.config.get("plan_number"),
                "runner": self.runner,
                "source": self.source,
                "coordinate_frame": "Phobos body-fixed",
                "radar_positions": "measurement_time",
                "timestamp_local": self.timestamp_local,
                "timestamp_utc": self.timestamp_utc,
                "git_commit": self._git_commit,
                "git_branch": self._git_branch,
                "seed": self.seed,
                "flush_immediately": self.flush_immediately,
                "parts": _to_jsonable(self.parts),
                "config": self.config,
                "core": self._core_snapshot,
                "images_dir": "images",
                "run_dir": str(self.run_dir),
                "run_log": str(self.log_path),
                "run_json": str(self.json_path),
                "summary": summary,
            },
            "time_to_fuel": time_to_fuel,
            "time_to_thrust": time_to_thrust,
            "time_to_point": time_to_point,
            "points": points,
            "radar_calls": radar_calls,
            "scene_frames": scene_frames,
            "camera_images": camera_images,
        }
        with self._json_lock:
            if summary.get("partial") and self._finalized:
                return                                  # finalize wrote (or is writing) the complete file
            self.run_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.json_path.with_suffix(".json.tmp")
            text = json.dumps(payload, separators=(",", ":"))  # one-shot C encoder: ~4x faster than json.dump
            with tmp.open("w", encoding="utf-8") as out:
                out.write(text)
                out.write("\n")
            tmp.replace(self.json_path)
