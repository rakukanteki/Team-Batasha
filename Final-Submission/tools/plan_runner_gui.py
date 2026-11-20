"""
Rigorous Plan Runner & Parameter Tuning Web GUI with Fast-Forward Replay.

Usage:
    python tools/plan_runner_gui.py
    python tools/plan_runner_gui.py --port 8765 --no-browser
"""
from __future__ import annotations

import argparse
import base64
import copy
from dataclasses import asdict
import json
import os
import shutil
import subprocess
import sys
import threading
from tempfile import TemporaryDirectory
import time
import uuid
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Systems import Actuation, Core as C, Radar, Vision  # noqa: E402
from Systems.RunLogger import DEFAULT_LOG_DIR, ENGINE_SOURCE, PYTHON_SOURCE, _to_jsonable, simulation_source  # noqa: E402
from Systems.ScoreMath import ACTUATOR_PRESETS, MathParams, analyze_max_score  # noqa: E402
from Systems.SimSuite import RAM_LOG_DIR, SCENARIOS, effective_plan_config, resolve_log_dir, run_single, run_suite  # noqa: E402
from tools.spaceteams_runner import SCENES, engine_score_for, installation, read_score_alert, run_engine
from Systems.EngineScore import SOURCE as ENGINE_SCORE_SOURCE, persist_engine_score  # noqa: E402
from Systems.plans.common import _variant_name, read_config_json  # noqa: E402
from Systems.plans import (  # noqa: E402
    PLAN_CONFIGS,
    PLAN_NAMES,
    active_variant_info,
    activate_variant,
    configure_plan,
    get_config,
    get_plan_dir,
    plan_name as canonical_plan_name,
    reload_plan_config,
    save_variant,
)

HTML_PATH = Path(__file__).resolve().parent / "gui_static.html"
_ORIGINAL_DEFAULT_LOG_DIR = DEFAULT_LOG_DIR.resolve()
EXTRA_LOG_DIRS: list[Path] = []
UI_INITIAL_MODE = "tune"  # "tune" | "browse"
UI_DEFAULT_RAM = False
STP_SERVER: Path | None = None
STP_ASSETS: Path | None = None

_RUN_INDEX_CACHE: dict[str, tuple[int, int, dict[str, Any]]] = {}
_SUITE_INDEX_CACHE: dict[str, tuple[int, int, dict[str, Any]]] = {}
_ENGINE_SCORE_CACHE: dict[str, dict[str, Any]] = {}  # run dir -> final engine score (only found scores are cached)
_PERSIST_BUSY: set[str] = set()                      # run dirs whose engine-score save is running right now

# Snapshot initial defaults at import time for reset support
_DEFAULT_CONFIGS = {
    name: copy.deepcopy(cfg)
    for key, cfg in PLAN_CONFIGS.items()
    if isinstance(key, str)
    for name in [key]
}

_JOBS: dict[str, dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()
_PYTHON_LOCK = threading.Lock()


class ConfigUpdateReq(BaseModel):
    updates: dict[str, Any]
    persist: bool = False


class SaveVariantReq(BaseModel):
    variant_name: str
    updates: dict[str, Any]
    activate: bool = True
    overwrite: bool = True


class ActivateVariantReq(BaseModel):
    variant_name: str


class AddRootReq(BaseModel):
    path: str


class MathMaxScoreReq(BaseModel):
    params: dict[str, Any] = Field(default_factory=dict)


class RunRequest(BaseModel):
    source: str = PYTHON_SOURCE
    engine_scene: str = "stp_final"
    mode: str = "single"  # single | suite | compare_all | sweep
    plan: str = "velocity_obstacle"
    variant: str | None = None
    debris_count: int | None = Field(default=None, ge=1, le=500)
    plans: list[str] | None = None
    scenario: str = "phobos_world"
    scenarios: list[str] | None = None
    seeds: list[int] = [0]
    burn: str | None = None
    camera: str = ""
    segmenter: str = "classical"
    dt: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    max_time: float = Field(default=420.0, gt=0, allow_inf_nan=False)
    flush_immediately: bool = False
    export_blender: bool = False
    ram: bool = False
    workers: int = 1
    config_overrides: dict[str, Any] | None = None
    param_json: dict[str, Any] | None = None
    sweep_param: str | None = None
    sweep_values: list[Any] | None = None


def _run_key(path: Path) -> str:
    return base64.urlsafe_b64encode(str(path.resolve()).encode("utf-8")).decode("ascii").rstrip("=")


def _maneuver_replay(run_dir: Path) -> dict[str, Any]:
    import csv
    with (run_dir / "pose.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = []
        required = {"sim_time_s", "phase", *(f"phobos_{axis}_m" for axis in "xyz"),
                    *(f"quat_{axis}" for axis in "xyzw")}
        for row in reader:
            try:
                if not required.issubset(row) or any(row[key] in (None, "") for key in required):
                    continue
                for key in required - {"phase"}:
                    float(row[key])
                rows.append(row)
            except (TypeError, ValueError):
                continue
    if not rows:
        raise ValueError(f"No pose samples in {run_dir / 'pose.csv'}")
    start = [float(rows[0][f"phobos_{axis}_m"]) for axis in "xyz"]
    goal = start.copy()
    radial = np.asarray(start, dtype=float)
    radial /= np.linalg.norm(radial)
    frames = []
    for row in rows:
        pos = [float(row[f"phobos_{axis}_m"]) for axis in "xyz"]
        if row["phase"] != "initial_pre_thrust" and (np.asarray(pos) - start) @ radial >= 30.0:
            goal = pos.copy()
        quat = [float(row[f"quat_{axis}"]) for axis in "xyzw"]
        frames.append({"t": float(row["sim_time_s"]), "pos": pos, "vel": [0, 0, 0],
                       "aim": goal, "dist_goal": float(np.linalg.norm(np.asarray(pos) - goal)),
                       "lin_mag": 0, "mode": row["phase"],
                       "body_quaternion_xyzw": quat, "phase": row["phase"]})
    if len(frames) == 1:
        frames.append({**frames[0], "t": frames[0]["t"] + 0.01})
    return {"metadata": {"plan_name": "Lift and rotate test", "runner": "lift_rotate",
                          "core": {"start": start, "target": goal}},
            "frames": frames, "points": [], "radar_frames": [], "scene_frames": [],
            "camera_images": []}


def _persist_plan_config(plan: str, cfg: dict[str, Any]) -> str:
    """Persist updates into the active config.json (following symlink if any)."""
    pname = canonical_plan_name(plan)
    pdir = get_plan_dir(pname)
    cfg_json = pdir / "config.json"
    target = cfg_json.resolve() if cfg_json.exists() else cfg_json
    with TemporaryDirectory(dir=target.parent) as temp:
        staged = Path(temp) / "preset.json"
        staged.write_text(json.dumps(_to_jsonable(cfg), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        staged.replace(target)
    return str(target)


def _create_app() -> FastAPI:
    app = FastAPI(title="Team Batasha - Rigorous Plan Runner & Parameter Tuner")

    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTML_PATH.read_text(encoding="utf-8")

    @app.get("/api/plans")
    def api_plans():
        try:
            installation(STP_SERVER, STP_ASSETS)
            engine_error = None
        except ValueError as exc:
            engine_error = str(exc)
        plans_out = []
        for num in (key for key in PLAN_NAMES if isinstance(key, int)):
            cfg = get_config(num)
            name = cfg.get("name")
            plans_out.append({
                "number": num,
                "name": name,
                "config": _to_jsonable(cfg),
                "defaults": _to_jsonable(_DEFAULT_CONFIGS.get(name, cfg)),
                "variant_info": active_variant_info(get_plan_dir(num)),
            })
        return {
            "plans": plans_out,
            "scenarios": SCENARIOS,
            "burns": sorted(Actuation.BURNS.keys()),
            "trackers": sorted(Radar.TRACKERS.keys()),
            "cameras": sorted(Vision.CAMERAS.keys()),
            "segmenters": sorted(Vision.SEGMENTERS.keys()),
            "ui_mode": UI_INITIAL_MODE,
            "engine_error": engine_error,
            "default_ram": UI_DEFAULT_RAM,
            "supports_ram": RAM_LOG_DIR.parent.is_dir(),
        }

    @app.post("/api/plans/{plan}/config")
    def api_update_config(plan: str, req: ConfigUpdateReq):
        pname = canonical_plan_name(plan)
        if pname not in PLAN_CONFIGS:
            raise HTTPException(status_code=404, detail=f"Unknown plan {plan!r}")
        cur = get_config(pname)
        clean_updates = {}
        for k, v in req.updates.items():
            # Preserve tuple types when original default was a tuple
            if isinstance(cur.get(k), tuple) and isinstance(v, list):
                clean_updates[k] = tuple(v)
            else:
                clean_updates[k] = v
        try:
            persisted_path = _persist_plan_config(pname, {**cur, **clean_updates}) if req.persist else None
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc
        updated = configure_plan(pname, **clean_updates)
        return {
            "plan": pname,
            "config": _to_jsonable(updated),
            "persisted": persisted_path,
            "variant_info": active_variant_info(get_plan_dir(pname)),
        }

    @app.post("/api/plans/{plan}/variants/save")
    def api_save_variant(plan: str, req: SaveVariantReq):
        pname = canonical_plan_name(plan)
        if pname not in PLAN_CONFIGS:
            raise HTTPException(status_code=404, detail=f"Unknown plan {plan!r}")
        cur = get_config(pname)
        clean_updates = {
            k: (tuple(v) if isinstance(cur.get(k), tuple) and isinstance(v, list) else v)
            for k, v in req.updates.items()
        }
        merged = {**cur, **clean_updates}
        try:
            info = save_variant(get_plan_dir(pname), req.variant_name, merged, activate=req.activate,
                                overwrite=req.overwrite)
        except FileExistsError as exc:
            raise HTTPException(409, "A preset with this name already exists.") from exc
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc
        if req.activate:
            reload_plan_config(pname)
        return {"plan": pname, "config": _to_jsonable(get_config(pname)), "variant_info": info}

    @app.post("/api/plans/{plan}/variants/activate")
    def api_activate_variant(plan: str, req: ActivateVariantReq):
        pname = canonical_plan_name(plan)
        if pname not in PLAN_CONFIGS:
            raise HTTPException(status_code=404, detail=f"Unknown plan {plan!r}")
        try:
            info = activate_variant(get_plan_dir(pname), req.variant_name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except (ValueError, OSError) as exc:
            raise HTTPException(422, str(exc)) from exc
        reloaded = reload_plan_config(pname)
        return {"plan": pname, "config": _to_jsonable(reloaded), "variant_info": info}

    @app.get("/api/plans/{plan}/variants/{variant}")
    def api_read_variant(plan: str, variant: str):
        try:
            path = get_plan_dir(plan) / "configs" / f"{_variant_name(variant)}.json"
            return {"config": _to_jsonable({**get_config(plan), **read_config_json(path)})}
        except (KeyError, ValueError, OSError) as exc:
            raise HTTPException(422, f"Unknown plan variant: {variant}") from exc

    @app.post("/api/plans/{plan}/reset")
    def api_reset_config(plan: str):
        pname = canonical_plan_name(plan)
        if pname not in _DEFAULT_CONFIGS:
            raise HTTPException(status_code=404, detail=f"Unknown plan {plan!r}")
        cfg = reload_plan_config(pname)
        return {"plan": pname, "config": _to_jsonable(cfg), "variant_info": active_variant_info(get_plan_dir(pname))}

    @app.get("/api/math/defaults")
    def api_math_defaults():
        return {
            "defaults": _to_jsonable(asdict(MathParams())),
            "presets": ACTUATOR_PRESETS,
        }

    @app.post("/api/math/max_score")
    def api_math_max_score(req: MathMaxScoreReq):
        try:
            return _to_jsonable(analyze_max_score(req.params))
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Math analysis failed: {exc}") from exc

    @app.post("/api/run")
    def api_run(req: RunRequest):
        if req.source not in (PYTHON_SOURCE, ENGINE_SOURCE):
            raise HTTPException(422, "Unknown simulator source.")
        if req.mode not in ("single", "suite", "compare_all", "sweep"):
            raise HTTPException(422, "Unknown run mode.")
        burn_arg = req.burn or None
        if (
            req.plan not in PLAN_CONFIGS or (burn_arg is not None and burn_arg not in Actuation.BURNS)
            or (req.camera and req.camera not in Vision.CAMERAS) or req.segmenter not in Vision.SEGMENTERS
        ):
            raise HTTPException(422, "Unknown plan, burn, camera or segmenter.")
        if req.variant:
            if req.mode == "compare_all" or (req.mode == "suite" and any(p != req.plan for p in req.plans or [])):
                raise HTTPException(422, "A subvariant belongs to one plan; select a single plan for this run.")
            try:
                preset = read_config_json(get_plan_dir(req.plan) / "configs" / f"{_variant_name(req.variant)}.json")
            except (ValueError, OSError) as exc:
                raise HTTPException(422, f"Unknown plan variant: {req.variant}") from exc
            req.config_overrides = {**preset, **(req.config_overrides or {}), "variant": req.variant}
        if req.debris_count is not None and (req.source != PYTHON_SOURCE or req.scenario != "phobos_world"
                                             or req.mode != "single"):
            raise HTTPException(422, "Debris count experiments require a single Python Phobos world run.")
        if req.source == ENGINE_SOURCE:
            if req.mode != "single" or req.engine_scene not in SCENES:
                raise HTTPException(422, "Select a single Space Teams PRO run and a valid scene.")
            try:
                installation(STP_SERVER, STP_ASSETS)
            except ValueError as exc:
                raise HTTPException(503, str(exc)) from exc
        elif req.scenario not in SCENARIOS or any(sc not in SCENARIOS for sc in req.scenarios or []):
            raise HTTPException(422, "Unknown Python scenario.")

        job_id = uuid.uuid4().hex[:10]
        run_log_root = resolve_log_dir(None if req.ram else DEFAULT_LOG_DIR, ram=req.ram)
        live_root = run_log_root / "web" / job_id
        with _JOBS_LOCK:
            if req.source == ENGINE_SOURCE and any(
                j["source"] == ENGINE_SOURCE and j["status"] == "running" for j in _JOBS.values()
            ):
                raise HTTPException(409, "A Space Teams PRO run is already active.")
            _JOBS[job_id] = {
                "job_id": job_id,
                "mode": req.mode,
                "source": req.source,
                "live_root": live_root,
                "status": "running",
                "progress": 0.0,
                "message": "Starting...",
                "live": {},
                "results": [],
                "summary": [],
                "error": None,
                "started_at": time.time(),
            }

        def worker():
            def result_keys(rows):
                for row in rows:
                    if row.get("run_dir"):
                        row["run_key"] = _run_key(Path(row["run_dir"]))

            try:
                if req.source == ENGINE_SOURCE:
                    config = effective_plan_config(req.plan, req.config_overrides, req.param_json)
                    path = run_engine(STP_SERVER, STP_ASSETS, live_root, req.engine_scene, {
                        "Planner": req.plan, "Burn": burn_arg or "", "Camera": req.camera or config.get("camera", "off"), "Segmenter": req.segmenter,
                    }, config, req.max_time, req.flush_immediately)
                    # The engine has exited, so nothing else writes this run any more: no settle wait.
                    _persist_quietly(path.parent, read_score_alert(live_root / "server.log") or None, settle_s=0)
                    meta, _ = _read_run_header(path, path.parent)
                    meta = _with_engine_score(meta, path.parent)
                    row = {**_score_breakdown(meta), "plan": req.plan, "scenario": req.engine_scene,
                           "run_dir": str(path.parent), "seed": None}
                    result_keys([row])
                    with _JOBS_LOCK:
                        _JOBS[job_id].update(status="done", progress=1.0, message="Space Teams PRO run saved", results=[row])
                elif req.mode == "single":
                    seed = req.seeds[0] if req.seeds else 0

                    def prog(pct, info):
                        with _JOBS_LOCK:
                            _JOBS[job_id]["progress"] = round(float(pct), 3)
                            _JOBS[job_id]["live"] = info
                            _JOBS[job_id]["message"] = f"t={info['t']}s dist={info['dist_m']}m fuel={info['fuel_kg']}kg"

                    res = run_single(
                        req.plan, scenario=req.scenario, seed=seed,
                        burn=burn_arg, camera=req.camera, segmenter=req.segmenter,
                        config_overrides=req.config_overrides, dt=req.dt, max_time=req.max_time,
                        flush_immediately=req.flush_immediately, log_dir=live_root,
                        export_blender=req.export_blender,
                        progress_cb=prog, param_json=req.param_json, live=True, raise_on_error=True,
                        debris_count=req.debris_count,
                    )
                    result_keys([res])
                    with _JOBS_LOCK:
                        _JOBS[job_id]["results"] = [res]
                        _JOBS[job_id]["progress"] = 1.0
                        _JOBS[job_id]["status"] = "done"
                        _JOBS[job_id]["message"] = f"Done: score={res['score']} arrived={res['arrived']}"
                elif req.mode in ("suite", "compare_all"):
                    plans = (
                        [name for key, name in PLAN_NAMES.items() if isinstance(key, int)]
                        if req.mode == "compare_all"
                        else (req.plans or [req.plan])
                    )
                    scenarios = req.scenarios or [req.scenario]

                    def on_step(done, total, row):
                        result_keys([row])
                        with _JOBS_LOCK:
                            _JOBS[job_id]["progress"] = round(done / max(total, 1), 3)
                            _JOBS[job_id]["results"].append(row)
                            _JOBS[job_id]["message"] = f"[{done}/{total}] {row['plan']} ({row['scenario']}, s={row['seed']}) -> score={row['score']}"

                    rep = run_suite(
                        plans=plans, scenarios=scenarios, seeds=req.seeds,
                        burn=burn_arg, camera=req.camera, segmenter=req.segmenter,
                        config_overrides=req.config_overrides, dt=req.dt, max_time=req.max_time,
                        flush_immediately=req.flush_immediately, log_dir=run_log_root,
                        export_blender=req.export_blender,
                        progress_cb=on_step, workers=req.workers, param_json=req.param_json,
                    )
                    with _JOBS_LOCK:
                        _JOBS[job_id]["summary"] = rep["summary"]
                        _JOBS[job_id]["progress"] = 1.0
                        _JOBS[job_id]["status"] = "done"
                        _JOBS[job_id]["message"] = f"Suite finished ({len(rep['results'])} runs)"
                elif req.mode == "sweep":
                    vals = req.sweep_values or []
                    if not req.sweep_param or not vals:
                        raise ValueError("sweep_param and sweep_values are required for sweep mode")
                    baseline = effective_plan_config(req.plan, req.config_overrides, req.param_json)
                    if req.sweep_param != "preset" and (req.sweep_param not in baseline or req.sweep_param in ("name", "plan_number")):
                        raise ValueError("Select a parameter belonging to the chosen plan.")
                    total = len(vals) * max(len(req.seeds), 1)
                    done = 0
                    for v in vals:
                        if req.sweep_param == "preset":         # A/B: a saved preset, then this run's JSON overrides
                            preset = read_config_json(get_plan_dir(req.plan) / "configs" / f"{_variant_name(str(v))}.json")
                            overrides = effective_plan_config(req.plan, preset, req.param_json)
                        else:
                            overrides = {**baseline, req.sweep_param: v}
                        sweep_burn = v if req.sweep_param == "burn" else burn_arg
                        for s in (req.seeds or [0]):
                            res = run_single(
                                req.plan, scenario=req.scenario, seed=s,
                                burn=sweep_burn, camera=req.camera, segmenter=req.segmenter,
                                config_overrides=overrides, dt=req.dt, max_time=req.max_time,
                                flush_immediately=req.flush_immediately, log_dir=run_log_root,
                                export_blender=req.export_blender,
                            )
                            res["sweep_param"] = req.sweep_param
                            res["sweep_value"] = v
                            result_keys([res])
                            done += 1
                            with _JOBS_LOCK:
                                _JOBS[job_id]["progress"] = round(done / max(total, 1), 3)
                                _JOBS[job_id]["results"].append(res)
                                _JOBS[job_id]["message"] = f"[{done}/{total}] {req.sweep_param}={v} -> score={res['score']}"
                    with _JOBS_LOCK:
                        _JOBS[job_id]["status"] = "done"
                        _JOBS[job_id]["progress"] = 1.0
                        _JOBS[job_id]["message"] = f"Sweep complete ({done} runs)"
            except Exception as exc:
                with _JOBS_LOCK:
                    _JOBS[job_id]["status"] = "error"
                    _JOBS[job_id]["error"] = repr(exc)
                    _JOBS[job_id]["message"] = f"Failed: {exc}"

        def execute():
            if req.source == ENGINE_SOURCE:
                worker()
            else:
                with _PYTHON_LOCK:
                    worker()

        threading.Thread(target=execute, daemon=True).start()
        return {"job_id": job_id}

    @app.get("/api/jobs/{job_id}")
    def api_job_status(job_id: str):
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail="Job not found")
            return copy.deepcopy({k: v for k, v in job.items() if k != "live_root"})

    @app.get("/api/jobs/{job_id}/live")
    def api_job_live(job_id: str, after: int = 0):
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
            if job is None:
                raise HTTPException(404, "Job not found")
            root = job["live_root"]
        path = next(root.glob(f"{job['source']}/*/live.jsonl"), None)
        if path is None:
            return {"cursor": 0, "frames": [], "metadata": None}
        with path.open("rb") as stream:
            header = stream.readline()
            if not header.endswith(b"\n"):
                return {"cursor": 0, "frames": [], "metadata": None}
            metadata = json.loads(header)["metadata"]
            stream.seek(max(after, stream.tell()))
            frames = []
            cursor = stream.tell()
            for _ in range(2000):
                line = stream.readline()
                if not line.endswith(b"\n"):
                    break
                frames.append(json.loads(line))
                cursor = stream.tell()
        return {"cursor": cursor, "frames": frames, "metadata": metadata}

    @app.post("/api/jobs/{job_id}/stop")
    def api_stop_job(job_id: str):
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
            if job is None or job["source"] != ENGINE_SOURCE or job["status"] != "running":
                raise HTTPException(404, "Active Space Teams PRO run not found")
            root = job["live_root"]
            root.mkdir(parents=True, exist_ok=True)
            (root / "stop").touch()
        return {"stopping": True}

    def _log_roots() -> list[Path]:
        cands: list[Path] = [DEFAULT_LOG_DIR, *EXTRA_LOG_DIRS]
        if DEFAULT_LOG_DIR.resolve() == _ORIGINAL_DEFAULT_LOG_DIR:
            cands.append(RAM_LOG_DIR)
            shm_root = Path("/dev/shm")
            if shm_root.is_dir():
                for p in sorted(shm_root.glob("batasha*")):
                    if p.is_dir():
                        cands.append(p)
            logs_dir = ROOT / "logs"
            if logs_dir.is_dir():
                cands.append(logs_dir)
        out: list[Path] = []
        seen: set[Path] = set()
        for p in cands:
            try:
                rp = p.resolve()
            except Exception:
                continue
            if rp.is_dir() and rp not in seen:
                seen.add(rp)
                out.append(rp)
        return out

    def _encode_key(p: Path) -> str:
        return _run_key(p)

    def _decode_key(key: str) -> Path | None:
        try:
            pad = "=" * (-len(key) % 4)
            raw = base64.urlsafe_b64decode((key + pad).encode("ascii")).decode("utf-8")
            cand = Path(raw).resolve()
            roots = _log_roots()
            if any(cand == r or r in cand.parents for r in roots):
                return cand
        except Exception:
            pass
        return None

    def _find_run_dir(run_ref: str) -> Path | None:
        # 1. URL-safe base64 run_key inside allowed roots
        decoded = _decode_key(run_ref)
        if decoded is not None and (decoded.name == "lift_rotate" and decoded.is_dir() or
                                     (decoded / "pose.csv").is_file() or
                                     (decoded / "run.json").is_file() or (decoded / "run.log").is_file()):
            return decoded
        # 2. Direct folder name or relative path under any root
        roots = _log_roots()
        for root in roots:
            cand = (root / run_ref).resolve()
            if (cand == root or root in cand.parents) and (cand.name == "lift_rotate" and cand.is_dir() or
                                                               (cand / "pose.csv").is_file() or
                                                             (cand / "run.json").is_file() or (cand / "run.log").is_file()):
                return cand
        # 3. Recursive fallback by folder name match
        if Path(run_ref).name == run_ref and not any(c in run_ref for c in "*?[]"):
            for root in roots:
                for cand in root.rglob(run_ref):
                    resolved = cand.resolve()
                    if root in resolved.parents and resolved.is_dir() and (
                        (resolved / "run.json").is_file() or (resolved / "run.log").is_file()
                    ):
                        return resolved
        return None

    def _persist_quietly(directory, alert, settle_s=30.0):
        try:
            return persist_engine_score(directory, alert, settle_s=settle_s)
        except Exception as exc:  # a locked or half-written run.json: shown from the cache, retried on the next listing
            print(f"Could not save the engine score into {directory}: {exc}")
            return {}
        finally:
            _PERSIST_BUSY.discard(str(directory))

    def _with_engine_score(meta, directory):
        if simulation_source(meta) != ENGINE_SOURCE or directory.parent.name != ENGINE_SOURCE:
            return meta
        summary = meta.get("summary") or {}
        if summary.get("score_source") == ENGINE_SCORE_SOURCE:   # already saved into run.json
            return meta
        key = str(directory)
        alert = _ENGINE_SCORE_CACHE.get(key) or read_score_alert(directory.parent.parent / "server.log")
        if not alert:  # run launched from the Space Teams PRO app, not this GUI: score is only in the app's log folder
            alert = engine_score_for(meta.get("timestamp_utc"), meta.get("plan_name"))
            alert = alert if "score" in alert else {}
        if not alert:
            return meta
        _ENGINE_SCORE_CACHE[key] = alert
        if key not in _PERSIST_BUSY:
            # Save it into run.json / run.log off the request thread (run.json can be tens of MB). Until a save lands
            # (a run still being written is skipped for 30 s), run.json has no score_source and the next listing retries.
            _PERSIST_BUSY.add(key)
            threading.Thread(target=_persist_quietly, args=(directory, alert), daemon=True).start()
        return {**meta, "summary": {**summary, **alert}}

    def _score_breakdown(meta: dict[str, Any], suite_row: dict[str, Any] | None = None) -> dict[str, Any]:
        summary = dict(meta.get("summary") or {})
        engine = simulation_source(meta) == ENGINE_SOURCE
        arrived = bool(suite_row["arrived"]) if suite_row and "arrived" in suite_row else bool(summary.get("arrived", False))
        T = suite_row.get("T") if suite_row and suite_row.get("T") is not None else summary.get(
            "scoring_time_s", summary.get("arrival_s", summary.get("duration_s"))
        )
        fuel = suite_row.get("fuel") if suite_row and suite_row.get("fuel") is not None else summary.get("final_fuel_kg")
        hits = suite_row.get("hits") if suite_row and suite_row.get("hits") is not None else summary.get("hits", None if engine else 0)
        damage_kJ = suite_row.get("damage_kJ") if suite_row and suite_row.get("damage_kJ") is not None else summary.get("damage_kJ")
        if damage_kJ is None and summary.get("damage") is not None:
            damage_kJ = float(summary["damage"]) / 1000.0
        if damage_kJ is not None or not engine:
            damage_kJ = float(damage_kJ or 0.0)
        min_clear = suite_row.get("min_clear") if suite_row and "min_clear" in suite_row else summary.get("min_clear")
        guard_steps = suite_row.get("guard_steps") if suite_row and "guard_steps" in suite_row else summary.get("guard_steps")
        score = suite_row.get("score") if suite_row and suite_row.get("score") is not None else summary.get("score")
        rules = summary.get("scoring_rules") or {}
        outcome = summary.get("outcome") or ("arrived" if arrived else "incomplete")

        time_term = fuel_term = damage_term = base_term = None
        if not engine and arrived and T and float(T) > 0:
            Tf = float(T)
            ff = float(fuel or 0.0)
            time_term = round(100.0 * (rules.get("NominalTime_s", 120.0) / Tf), 2)
            fuel_term = round(-100.0 * max(ff / rules.get("NominalFuel_kg", 75.0), 0.0), 2)
            damage_term = round(-200.0 * ((damage_kJ * 1000.0) / rules.get("DamageThreshold_J", C.DMAX)), 2)
            base_term = 50.0
            if score is None:
                score = round(time_term + fuel_term + damage_term + base_term, 2)

        return {
            "score": round(float(score), 2) if score is not None else None,
            "arrived": arrived,
            "outcome": outcome,
            "score_source": "Space Teams PRO" if engine and score is not None else "Python estimate" if score is not None else None,
            "T": round(float(T), 2) if T is not None else None,
            "fuel": round(float(fuel), 2) if fuel is not None else None,
            "hits": int(hits) if hits is not None else None,
            "damage_kJ": round(damage_kJ, 2) if damage_kJ is not None else None,
            "min_clear": round(float(min_clear), 2) if min_clear is not None else None,
            "guard_steps": int(guard_steps) if guard_steps is not None else None,
            "time_term": time_term,
            "fuel_term": fuel_term,
            "damage_term": damage_term,
            "base_term": base_term,
        }

    def _collect_suites(roots: list[Path]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        suites: list[dict[str, Any]] = []
        row_by_dir: dict[str, dict[str, Any]] = {}
        files: list[tuple[Path, Path]] = []
        for root in roots:
            for sp in root.rglob("suite_report.json"):
                files.append((root, sp))
        files.sort(key=lambda pair: pair[1].stat().st_mtime, reverse=True)
        for root, sp in files:
            try:
                st = sp.stat()
                ckey = str(sp.resolve())
                cached = _SUITE_INDEX_CACHE.get(ckey)
                if cached and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
                    rec = cached[2]
                else:
                    data = json.loads(sp.read_text(encoding="utf-8"))
                    results = data.get("results", [])
                    annotated = []
                    for r in results:
                        rr = dict(r)
                        if rr.get("run_dir"):
                            rpath = Path(rr["run_dir"]).resolve()
                            if (rpath / "run.json").is_file():
                                rr["run_key"] = _encode_key(rpath)
                                rr["run_id"] = rpath.name
                        annotated.append(rr)
                    rec = {
                        "suite_key": _encode_key(sp.parent),
                        "name": sp.parent.name,
                        "path": str(sp),
                        "root": str(root),
                        "timestamp": data.get("timestamp"),
                        "plans": data.get("plans", []),
                        "scenarios": data.get("scenarios", []),
                        "seeds": data.get("seeds", []),
                        "config_overrides": data.get("config_overrides", {}),
                        "summary": data.get("summary", []),
                        "results": annotated,
                        "total_runs": len(annotated),
                    }
                    _SUITE_INDEX_CACHE[ckey] = (st.st_mtime_ns, st.st_size, rec)
                suites.append(rec)
                for rr in rec["results"]:
                    if rr.get("run_dir"):
                        row_by_dir.setdefault(str(Path(rr["run_dir"]).resolve()), rr)
            except Exception:
                continue
        return suites, row_by_dir

    def _read_run_header(jp: Path, d: Path) -> tuple[dict[str, Any], dict[str, int]]:
        """Read only the small leading metadata JSON block (+ run.log counts) to avoid multi-GB parses."""
        dec = json.JSONDecoder()
        meta: dict[str, Any] | None = None
        counts: dict[str, int] = {"ticks": 0, "points_count": 0, "radar_calls": 0, "camera_images_count": 0}
        try:
            with open(jp, "r", encoding="utf-8") as f:
                head = f.read(65536)
            idx = head.find('"metadata"')
            if idx != -1:
                colon = head.find(":", idx + 10)
                if colon != -1:
                    parsed, _ = dec.raw_decode(head[colon + 1 :].lstrip())
                    if isinstance(parsed, dict):
                        meta = parsed
        except Exception:
            meta = None

        lp = d / "run.log"
        if lp.is_file():
            try:
                with open(lp, "rb") as lf:
                    lf.seek(0, 2)
                    sz = lf.tell()
                    lf.seek(max(0, sz - 768))
                    tail = lf.read().decode("utf-8", errors="replace")
                for token, key in (
                    ("ticks=", "ticks"),
                    ("plan_events=", "points_count"),
                    ("radar_calls=", "radar_calls"),
                    ("camera_images=", "camera_images_count"),
                ):
                    pos = tail.rfind(token)
                    if pos != -1:
                        val_str = tail[pos + len(token) :].split(None, 1)[0]
                        if val_str.isdigit():
                            counts[key] = int(val_str)
            except Exception:
                pass

        if meta is None:
            raw = json.loads(jp.read_text(encoding="utf-8"))
            meta = raw.get("metadata", {})
            counts = {
                "ticks": len(raw.get("time_to_point", {})),
                "points_count": len(raw.get("points", [])),
                "radar_calls": len(raw.get("radar_calls", [])),
                "camera_images_count": len(raw.get("camera_images", [])),
            }
        return meta, counts

    def _index_run(root: Path, jp: Path, row_by_dir: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
        d = jp.parent.resolve()
        st = jp.stat()
        ckey = str(jp.resolve())
        cached = _RUN_INDEX_CACHE.get(ckey)
        if cached and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
            rec = dict(cached[2])
        else:
            meta, counts = _read_run_header(jp, d)
            summary = meta.get("summary") or {}
            rec = {
                "run_id": d.name,
                "run_key": _encode_key(d),
                "run_dir": str(d),
                "root": str(root),
                "root_label": root.name if root != _ORIGINAL_DEFAULT_LOG_DIR else "flightlogs",
                "rel_path": str(d.relative_to(root)),
                "plan_name": meta.get("plan_name"),
                "plan_number": meta.get("plan_number"),
                "runner": meta.get("runner"),
                "source": simulation_source(meta),
                "seed": meta.get("seed"),
                "timestamp_local": meta.get("timestamp_local"),
                "meta": meta,
                "ticks": counts["ticks"] or int(summary.get("ticks") or 0),
                "radar_calls": counts["radar_calls"],
                "points_count": counts["points_count"],
                "camera_images_count": counts["camera_images_count"],
                "has_log": (d / "run.log").is_file(),
                "storage": "RAM" if str(d).startswith("/dev/shm") else "disk",
                "mtime": st.st_mtime,
            }
            _RUN_INDEX_CACHE[ckey] = (st.st_mtime_ns, st.st_size, rec)
            rec = dict(rec)  # the pop below must not strip "meta" from the cached record
        suite_row = row_by_dir.get(str(d))
        meta = rec.pop("meta", rec.get("metadata") or {})
        rec["summary"] = meta.get("summary", {})
        rec["parts"] = meta.get("parts", {})
        rec["scenario"] = (suite_row or {}).get("scenario") or rec.get("runner")
        rec["breakdown"] = _score_breakdown(_with_engine_score(meta, d), suite_row)
        rec["score"] = rec["breakdown"]["score"]
        rec["arrived"] = rec["breakdown"]["arrived"]
        rec["T"] = rec["breakdown"]["T"]
        rec["fuel"] = rec["breakdown"]["fuel"]
        rec["hits"] = rec["breakdown"]["hits"]
        rec["min_clear"] = rec["breakdown"]["min_clear"]
        return rec

    @app.get("/api/roots")
    def api_get_roots():
        roots = _log_roots()
        out = []
        for r in roots:
            rc = sum(1 for _ in r.rglob("run.json"))
            sc = sum(1 for _ in r.rglob("suite_report.json"))
            out.append({
                "path": str(r),
                "label": f"{r.name} ({'RAM' if str(r).startswith('/dev/shm') else 'disk'})",
                "run_count": rc,
                "suite_count": sc,
                "storage": "RAM" if str(r).startswith("/dev/shm") else "disk",
            })
        return {"roots": out, "ui_mode": UI_INITIAL_MODE}

    @app.post("/api/roots")
    def api_add_root(req: AddRootReq):
        p = Path(req.path).expanduser().resolve()
        if not p.is_dir():
            raise HTTPException(status_code=400, detail=f"Directory does not exist: {p}")
        if p not in EXTRA_LOG_DIRS:
            EXTRA_LOG_DIRS.append(p)
        return api_get_roots()

    @app.get("/api/suites")
    def api_list_suites():
        roots = _log_roots()
        suites, _ = _collect_suites(roots)
        return {"suites": suites}

    @app.get("/api/runs")
    def api_list_runs(dir: str = "", plan: str = "", scenario: str = "", source: str = "", q: str = "", limit: int = 500):
        roots = _log_roots()
        if dir:
            dpath = Path(dir).expanduser().resolve()
            if dpath in roots:
                roots = [dpath]
        _, row_by_dir = _collect_suites(roots)
        ql = q.strip().lower()
        found: list[tuple[Path, Path]] = []
        maneuver_root = DEFAULT_LOG_DIR / "lift_rotate"
        if maneuver_root.is_dir() and (not dir or Path(dir).resolve() == DEFAULT_LOG_DIR.resolve()):
            pose_candidates = [p for p in maneuver_root.glob("run_*/pose.csv") if p.stat().st_size > 80]
            latest_pose = max(pose_candidates, key=lambda p: p.stat().st_mtime, default=None)
            if latest_pose:
                maneuver_dir = latest_pose.parent
                image_dir = maneuver_dir / "images"
                pose_rows = []
                import csv
                with latest_pose.open(encoding="utf-8", newline="") as stream:
                    pose_rows = list(csv.DictReader(stream))
                image_names = sorted(p.name for p in image_dir.glob("*.png"))
                rec = {"run_id": maneuver_dir.name, "run_key": _encode_key(maneuver_dir),
                       "run_dir": str(maneuver_dir), "root": str(DEFAULT_LOG_DIR),
                       "root_label": "flightlogs", "rel_path": "lift_rotate",
                       "plan_name": "Lift and rotate test", "runner": "lift_rotate",
                       "source": ENGINE_SOURCE, "scenario": "Lift and rotate test",
                       "seed": None, "timestamp_local": None, "ticks": len(pose_rows),
                       "radar_calls": 0, "points_count": 0, "camera_images_count": len(image_names),
                       "has_log": False, "storage": "disk", "mtime": latest_pose.stat().st_mtime,
                       "summary": {"ticks": len(pose_rows)}, "parts": {}, "breakdown": {},
                       "score": None, "arrived": None, "T": None, "fuel": None,
                       "hits": 0, "min_clear": None}
                if (not source or source == ENGINE_SOURCE) and (not q or all(word in "lift rotate test" for word in ql.split())):
                    maneuver_record = rec
                    _RUN_INDEX_CACHE[str(maneuver_root.resolve())] = (0, 0, rec)
                else:
                    maneuver_record = None
            else:
                maneuver_record = None
        else:
            maneuver_record = None
        seen_dirs: set[Path] = set()
        for root in roots:
            if root.is_dir():
                for jp in root.rglob("run.json"):
                    rdir = jp.parent.resolve()
                    if rdir not in seen_dirs:
                        seen_dirs.add(rdir)
                        found.append((root, jp))

        runs = []
        for root, jp in found:
            try:
                rec = _index_run(root, jp, row_by_dir)
                if rec is None:
                    continue
                if plan and rec.get("plan_name") != plan:
                    continue
                if scenario and rec.get("scenario") != scenario:
                    continue
                if source and rec["source"] != source:
                    continue
                if ql:
                    hay = f"{rec.get('run_id','')} {rec.get('plan_name','')} {rec.get('scenario','')} {rec.get('runner','')} {rec.get('seed','')}".lower()
                    if ql not in hay:
                        continue
                runs.append(rec)
                if len(runs) >= limit:
                    break
            except Exception:
                continue
        if maneuver_record is not None:
            runs.insert(0, maneuver_record)
            runs = runs[:limit]
        runs.sort(key=lambda rec: rec.get("mtime", 0), reverse=True)
        return {"runs": runs, "total": len(runs)}

    def _replay_data(data, max_frames, max_radar):
        meta = data.get("metadata", {})
        core = meta.get("core", {})
        target = core.get("target") or [-850.0, 120.0, -9530.0]

        t2p = data.get("time_to_point", {})
        t2f = data.get("time_to_fuel", {})
        t2th = data.get("time_to_thrust", {})
        keys = sorted(t2p.keys(), key=lambda k: float(k))
        step = max(1, (len(keys) + max_frames - 3) // (max_frames - 1))
        sel_keys = keys[::step]
        if keys and sel_keys[-1] != keys[-1]:
            sel_keys.append(keys[-1])

        frames = []
        for k in sel_keys:
            p = t2p[k]
            th = t2th.get(k, {})
            pos = p["shuttle_pos"]
            vel = p.get("shuttle_vel", [0, 0, 0])
            if np.shape(pos) != (3,) or np.shape(vel) != (3,):
                raise ValueError("Expected three-dimensional shuttle vectors")
            ang_cmd = th.get("ang_cmd", [0, 0, 0])
            dist_goal = float(np.linalg.norm(np.array(pos, float) - np.array(target, float)))
            speed = float(np.linalg.norm(np.array(vel, float)))
            ang_mag = float(np.linalg.norm(np.array(ang_cmd, float))) if isinstance(ang_cmd, (list, tuple)) else 0.0
            frames.append({
                "t": float(k),
                "fuel": float(t2f.get(k, 0.0)),
                "pos": pos,
                "vel": vel,
                "speed": round(speed, 3),
                "dist_goal": round(dist_goal, 2),
                "aim": p.get("target_point", [0, 0, 0]),
                "node_idx": p.get("target_node_idx", 0),
                "mode": th.get("mode", p.get("mode", "coast")),
                "guarding": p.get("guarding", False),
                "lin_mag": float(th.get("lin_mag", 0.0)),
                "ang_mag": round(ang_mag, 3),
                "thrust_dir": th.get("thrust_dir_world", [0, 0, 0]),
                "body_quaternion_xyzw": p.get("body_quaternion_xyzw"),
            })

        rcalls = data.get("radar_calls", [])
        rstep = max(1, (len(rcalls) + max_radar - 1) // max_radar)
        radar_frames = []
        for rc in rcalls[::rstep]:
            radar_frames.append({
                "t": rc["t"],
                "shuttle_pos": rc.get("shuttle_pos"),
                "positions_abs": rc["positions_abs"],
                "diameters": rc.get("diameters", []),
            })

        scene_frames = data.get("scene_frames", [])
        scene_step = max(1, (len(scene_frames) + max_radar - 1) // max_radar)
        sampled_scene = scene_frames[::scene_step]
        if scene_frames and sampled_scene[-1] is not scene_frames[-1]:
            sampled_scene.append(scene_frames[-1])
        return {
            "metadata": meta, "source": simulation_source(meta), "frames": frames,
            "points": data.get("points", []), "radar_frames": radar_frames,
            "scene_frames": sampled_scene,
            "radar_calls_total": len(rcalls), "ticks_total": len(keys),
            "camera_images_meta": data.get("camera_images", []),
            "maneuver_images": data.get("maneuver_images", []),
            "maneuver_poses": data.get("maneuver_poses", []),
        }

    @app.post("/api/replay")
    def api_preview_json(data: dict[str, Any]):
        try:
            replay = _replay_data(data, 1000, 100)
            if not replay["frames"]:
                raise ValueError("No trajectory frames")
            return {
                **replay, "run_id": replay["metadata"].get("run_id") or "Imported JSON",
                "score_breakdown": _score_breakdown(replay["metadata"]),
            }
        except (KeyError, ValueError, TypeError, AttributeError, OverflowError) as exc:
            raise HTTPException(422, "Expected a Batasha run.json with trajectory frames.") from exc

    @app.get("/api/maneuver/replay")
    def api_maneuver_replay():
        root = DEFAULT_LOG_DIR / "lift_rotate"
        pose_candidates = [p for p in root.glob("run_*/pose.csv") if p.stat().st_size > 80]
        pose_file = max(pose_candidates, key=lambda p: p.stat().st_mtime, default=None)
        if pose_file is None:
            raise HTTPException(404, "No lift and rotate pose log found")
        run_dir = pose_file.parent
        replay = _replay_data(_maneuver_replay(run_dir), 10000, 1)
        replay["run_id"] = run_dir.name
        replay["run_key"] = _encode_key(run_dir)
        replay["run_dir"] = str(run_dir)
        replay["score_breakdown"] = {}
        replay["suite_row"] = None
        replay["camera_images"] = []
        replay["maneuver_images"] = sorted(p.name for p in (run_dir / "images").glob("*.png"))
        with pose_file.open(encoding="utf-8", newline="") as stream:
            import csv
            replay["maneuver_poses"] = list(csv.DictReader(stream))
        return replay

    @app.get("/api/runs/{run_ref}/replay")
    def api_run_replay(run_ref: str, max_frames: int = 1000, max_radar: int = 100):
        rdir = _find_run_dir(run_ref)
        if rdir is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if (rdir / "run.json").is_file():
            data = json.loads((rdir / "run.json").read_text(encoding="utf-8"))
        else:
            maneuver_dir = rdir
            if rdir.name == "lift_rotate":
                candidates = sorted(rdir.glob("run_*/pose.csv"), key=lambda p: p.stat().st_mtime)
                if candidates:
                    maneuver_dir = candidates[-1].parent
            if not (maneuver_dir / "pose.csv").is_file():
                raise HTTPException(status_code=404, detail="Run replay not found")
            data = _maneuver_replay(maneuver_dir)
        replay = _replay_data(data, max(2, max_frames), max(1, max_radar))
        _, row_by_dir = _collect_suites(_log_roots())
        suite_row = row_by_dir.get(str(rdir.resolve()))
        replay["metadata"] = _with_engine_score(replay["metadata"], rdir)
        breakdown = _score_breakdown(replay["metadata"], suite_row)

        img_names = []
        img_dir = rdir / "images"
        if img_dir.is_dir():
            img_names = sorted(p.name for p in img_dir.iterdir() if p.is_file())

        maneuver_dir = rdir
        if rdir.name == "lift_rotate" and not (rdir / "pose.csv").is_file():
            candidates = sorted(rdir.glob("run_*/pose.csv"), key=lambda p: p.stat().st_mtime)
            if candidates:
                maneuver_dir = candidates[-1].parent
        maneuver_images = []
        maneuver_poses = []
        pose_file = maneuver_dir / "pose.csv"
        if pose_file.is_file():
            import csv
            with pose_file.open(encoding="utf-8", newline="") as stream:
                maneuver_poses = list(csv.DictReader(stream))
            image_dir = maneuver_dir / "images"
            if image_dir.is_dir():
                maneuver_images = sorted(p.name for p in image_dir.iterdir()
                                         if p.is_file() and p.suffix.lower() == ".png")

        return {
            **replay,
            "run_id": rdir.name,
            "run_key": _encode_key(rdir),
            "run_dir": str(rdir),
            "score_breakdown": breakdown,
            "suite_row": suite_row,
            "camera_images": img_names,
            "maneuver_images": maneuver_images,
            "maneuver_poses": maneuver_poses,
        }

    @app.get("/api/runs/{run_ref}/json")
    def api_run_json_facts(run_ref: str):
        rdir = _find_run_dir(run_ref)
        if rdir is None or not (rdir / "run.json").is_file():
            raise HTTPException(status_code=404, detail="Run JSON not found")
        if (rdir / "run.json").is_file():
            data = json.loads((rdir / "run.json").read_text(encoding="utf-8"))
        else:
            maneuver_dir = rdir
            if rdir.name == "lift_rotate":
                candidates = sorted(rdir.glob("run_*/pose.csv"), key=lambda p: p.stat().st_mtime)
                if candidates:
                    maneuver_dir = candidates[-1].parent
            data = _maneuver_replay(maneuver_dir)
            data["maneuver_images"] = sorted(p.name for p in (maneuver_dir / "images").glob("*.png"))
            with (maneuver_dir / "pose.csv").open(encoding="utf-8", newline="") as stream:
                import csv
                data["maneuver_poses"] = list(csv.DictReader(stream))
        t2p = data.get("time_to_point", {})
        t2f = data.get("time_to_fuel", {})
        t2th = data.get("time_to_thrust", {})
        rcalls = data.get("radar_calls", [])
        keys = sorted(t2p.keys(), key=lambda k: float(k))
        sample_keys = keys[:5] + (keys[-5:] if len(keys) > 5 else [])
        _, row_by_dir = _collect_suites(_log_roots())
        suite_row = row_by_dir.get(str(rdir.resolve()))
        meta = data.get("metadata", {})
        return {
            "run_id": rdir.name,
            "run_dir": str(rdir),
            "metadata": meta,
            "score_breakdown": _score_breakdown(_with_engine_score(meta, rdir), suite_row),
            "suite_row": suite_row,
            "counts": {
                "ticks": len(t2p),
                "time_to_fuel": len(t2f),
                "time_to_thrust": len(t2th),
                "time_to_point": len(t2p),
                "points_events": len(data.get("points", [])),
                "radar_calls": len(rcalls),
                "camera_images": len(data.get("camera_images", [])),
            },
            "points": data.get("points", []),
            "time_to_fuel_sample": {k: t2f.get(k) for k in sample_keys},
            "time_to_thrust_sample": {k: t2th.get(k) for k in sample_keys},
            "time_to_point_sample": {k: t2p.get(k) for k in sample_keys},
            "radar_first": rcalls[0] if rcalls else None,
            "camera_images": data.get("camera_images", []),
        }

    @app.get("/api/runs/{run_ref}/raw_json")
    def api_run_raw_json(run_ref: str):
        rdir = _find_run_dir(run_ref)
        if rdir is None or not (rdir / "run.json").is_file():
            raise HTTPException(status_code=404, detail="Run JSON not found")
        return FileResponse(rdir / "run.json", media_type="application/json", filename=f"{rdir.name}_run.json")

    @app.get("/api/runs/{run_ref}/images/{filename}")
    def api_run_image(run_ref: str, filename: str):
        rdir = _find_run_dir(run_ref)
        if rdir is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if Path(filename).name != filename:
            raise HTTPException(status_code=400, detail="Invalid filename")
        ip = (rdir / "images" / filename).resolve()
        if not ip.is_file() or (rdir / "images").resolve() not in ip.parents:
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(ip)

    @app.get("/api/runs/{run_ref}/maneuver/{filename}")
    def api_maneuver_image(run_ref: str, filename: str):
        rdir = _find_run_dir(run_ref)
        if rdir is None or Path(filename).name != filename:
            raise HTTPException(status_code=404, detail="Maneuver image not found")
        maneuver_dir = rdir
        if rdir.name == "lift_rotate":
            candidates = sorted(rdir.glob("run_*/pose.csv"), key=lambda p: p.stat().st_mtime)
            if candidates:
                maneuver_dir = candidates[-1].parent
        image_dir = (maneuver_dir / "images").resolve()
        path = (image_dir / filename).resolve()
        if image_dir not in path.parents or not path.is_file():
            raise HTTPException(status_code=404, detail="Maneuver image not found")
        return FileResponse(path, media_type="image/png")

    @app.get("/api/runs/{run_ref}/maneuver/{filename}")
    def api_maneuver_image(run_ref: str, filename: str):
        rdir = _find_run_dir(run_ref)
        if rdir is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if Path(filename).name != filename:
            raise HTTPException(status_code=400, detail="Invalid image filename")
        maneuver_dir = rdir
        if rdir.name == "lift_rotate" and not (rdir / "pose.csv").is_file():
            candidates = sorted(rdir.glob("run_*/pose.csv"), key=lambda p: p.stat().st_mtime)
            if candidates:
                maneuver_dir = candidates[-1].parent
        image_dir = (maneuver_dir / "images").resolve()
        path = (image_dir / filename).resolve()
        if image_dir not in path.parents or not path.is_file():
            raise HTTPException(status_code=404, detail="Maneuver image not found")
        return FileResponse(path, media_type="image/png")

    @app.get("/api/runs/{run_ref}/log")
    def api_run_log(run_ref: str, category: str = "ALL", search: str = "", limit: int = 400):
        rdir = _find_run_dir(run_ref)
        if rdir is None or not (rdir / "run.log").is_file():
            raise HTTPException(status_code=404, detail="run.log not found")
        lines = (rdir / "run.log").read_text(encoding="utf-8").splitlines()
        if category and category != "ALL":
            tag = f"[{category}]"
            lines = [l for l in lines if tag in l or l.startswith("=") or l.startswith("RUN ")]
        if search:
            q = search.lower()
            lines = [l for l in lines if q in l.lower()]
        total = len(lines)
        if total > limit:
            lines = lines[:30] + [f"... ({total - limit} lines omitted; narrow filter/search) ..."] + lines[-(limit - 30):]
        return {"total_lines": total, "lines": lines}

    def _build_blend(run_ref: str) -> Path:
        """Build blender/<run>.blend from run.json unless an up-to-date one already exists."""
        from tools.json_to_blender import find_blender
        rdir = _find_run_dir(run_ref)
        if rdir is None or not (rdir / "run.json").is_file():
            raise HTTPException(status_code=404, detail="run.json not found")
        jp = rdir / "run.json"
        out_blend = ROOT / "blender" / f"{rdir.name}.blend"
        if out_blend.is_file() and out_blend.stat().st_mtime >= jp.stat().st_mtime:
            return out_blend
        if find_blender() is None:
            raise HTTPException(501, "Blender is not installed: install it from blender.org, add it to PATH, "
                                     "or start the GUI with BLENDER=<path to blender.exe>.")
        proc = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "json_to_blender.py"), str(jp), "-o", str(out_blend)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0 or not out_blend.is_file():
            raise HTTPException(status_code=500, detail=proc.stderr[-600:] or proc.stdout[-600:])
        return out_blend

    @app.post("/api/runs/{run_ref}/blender")
    def api_export_blender(run_ref: str):
        return {"blend_path": str(_build_blend(run_ref))}

    @app.get("/api/runs/{run_ref}/blend")
    def api_download_blend(run_ref: str):
        out = _build_blend(run_ref)  # lazy: built on first download, reused while run.json is unchanged
        return FileResponse(out, filename=out.name, media_type="application/octet-stream")

    return app


app = _create_app()


def main() -> None:
    global UI_INITIAL_MODE, UI_DEFAULT_RAM, STP_SERVER, STP_ASSETS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--ram", action="store_true", help="default log storage to /dev/shm RAM tmpfs")
    ap.add_argument("--dir", action="append", default=[], help="extra log directory to browse (repeatable)")
    ap.add_argument("--browse", action="store_true", help="open directly in the Log & Suite Browser view")
    ap.add_argument("--no-browser", action="store_true", help="do not open browser automatically")
    ap.add_argument("--stp-server", type=Path, help="Space Teams PRO Compute_Server/run_Server.exe")
    ap.add_argument("--stp-assets", type=Path, help="Space Teams PRO Assets directory")
    args = ap.parse_args()
    STP_SERVER = args.stp_server
    if STP_SERVER is None and (server := shutil.which("run_Server.exe")):
        STP_SERVER = Path(server)
    STP_ASSETS = args.stp_assets
    if STP_ASSETS is None and (local := os.environ.get("LOCALAPPDATA")):
        STP_ASSETS = Path(local) / "SimDynamX" / "STPro" / "Assets"
    if STP_SERVER is not None:
        STP_SERVER = STP_SERVER.expanduser().resolve()
    if STP_ASSETS is not None:
        STP_ASSETS = STP_ASSETS.expanduser().resolve()
    UI_DEFAULT_RAM = args.ram and RAM_LOG_DIR.parent.is_dir()
    for d in args.dir:
        for part in str(d).split(","):
            if part.strip():
                p = Path(part.strip()).expanduser().resolve()
                if p not in EXTRA_LOG_DIRS:
                    EXTRA_LOG_DIRS.append(p)
    if args.browse:
        UI_INITIAL_MODE = "browse"
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"Starting Team Batasha Plan Runner & 3D Web Visualizer at {url}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
