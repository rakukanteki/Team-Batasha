"""Tests for per-plan folders/configs, RunLogger buffering & JSON/log schema, and Blender replay export."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from Systems import BasicController, Core as C, Planners
from Systems.RunLogger import RunLogger
from Systems.plans import PLAN_CONFIGS, PLANNERS, configure_plan, get_config
from tools import actuator_twin, offline_bench
from tools.spaceteams_runner import read_score_alert

ROOT = Path(__file__).resolve().parent.parent


def test_every_plan_has_folder_with_config_and_algorithm_files():
    plans_dir = ROOT / "Systems" / "plans"
    expected_folders = [
        "plan_1_fewest_points",
        "plan_2_linear_program",
        "plan_3_lattice",
        "plan_4_team_pipeline",
        "plan_5_conjunction",
        "plan_6_barrier",
        "plan_7_velocity_obstacle",
        "plan_8_astar",
        "plan_9_cost_lattice",
        "plan_10_real_calibrated",
        "plan_11_v11",
    ]
    for folder in expected_folders:
        d = plans_dir / folder
        assert d.is_dir(), f"Missing plan folder {d}"
        assert (d / "config.py").is_file(), f"Missing config.py in {d}"
        py_files = [p.name for p in d.glob("*.py") if p.name not in ("__init__.py", "config.py")]
        assert len(py_files) >= 1, f"Expected algorithm file(s) alongside config.py in {d}"


def test_plan_configs_dict_covers_all_plans_and_overrides_take_effect(monkeypatch):
    for num, name in [
        (1, "fewest_points"),
        (2, "linear_program"),
        (3, "lattice"),
        (4, "team_pipeline"),
        (5, "conjunction"),
        (6, "barrier"),
        (7, "velocity_obstacle"),
        (8, "astar"),
        (9, "cost_lattice"),
        (10, "real_calibrated"),
        (11, "v11"),
    ]:
        assert name in PLAN_CONFIGS and num in PLAN_CONFIGS
        assert PLAN_CONFIGS[name] is PLAN_CONFIGS[num]
        assert PLANNERS[name] is PLANNERS[num]
        assert PLAN_CONFIGS[name]["name"] == name
        assert PLAN_CONFIGS[name]["plan_number"] == num

    # Test Plan 8 max_pops threshold override actually limits A* expansions
    C.configure()
    st = dict(r=C.S.copy(), v=np.zeros(3), burning=False, main=False, fuel=0.0, t=0.0)
    empty = (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0))
    monkeypatch.setitem(PLAN_CONFIGS["astar"], "max_pops", 1)
    with pytest.raises(RuntimeError, match="expansions"):
        PLANNERS["astar"](st, empty, None, False)

    # Test configure_plan helper
    old_alpha = get_config("barrier")["barrier_alpha"]
    try:
        configure_plan("barrier", barrier_alpha=0.75)
        assert get_config(6)["barrier_alpha"] == 0.75
    finally:
        configure_plan("barrier", barrier_alpha=old_alpha)


def test_run_logger_in_memory_buffering_and_immediate_flush(tmp_path):
    C.configure()
    # 1. Default flush_immediately=False: nothing written to disk until finalize()
    logger = RunLogger("astar", base_dir=tmp_path, flush_immediately=False, runner="unit_test", seed=42)
    assert not logger.run_dir.exists()
    assert not logger.log_path.exists()
    assert not logger.json_path.exists()

    origin = C.S - np.array([5.0, 0.0, 0.0])
    logger.log_radar(0.0, C.S, [np.array([[10.0, 20.0, 30.0], [37.4, -0.5, 4.0], [0.0, 0.0, 0.0]])],
                     changed_count=1, origins=[origin])
    logger.log_plan(
        0.0,
        ([C.S + 100 * C.SH, C.T], 8.4),
        {
            "action": "astar",
            "cost": -55.0,
            "ptot": 0.0001,
            "clean": 55.5,
            "clear": 12.0,
            "K": 1,
            "astar_points": [(C.S + 60 * C.SH).tolist(), C.T.tolist()],
            "cells": [40, 40],
            "stats": {"pops": 10, "priced_edges": 100},
        },
    )
    logger.log_turn_passed(5.0, 0, C.S + 100 * C.SH, C.S + 101 * C.SH)
    logger.log_tick(
        0.1, C.S, np.array([0.0, 0.0, -1.0]), 0.42, "main",
        C.SH, np.array([0.0, 0.0, 0.5]), np.array([0.0, 0.0, 0.0]),
        0, C.T, guarding=False, a_burn=0.125,
    )
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[4:12, 4:12] = 200
    logger.log_camera(1.0, True, 5.0, -2.0, 60.0, report={"used": True}, image=img)

    # Still nothing on disk!
    assert not logger.run_dir.exists()

    run_dir = logger.finalize({"arrived": True, "score": 60.0})
    assert run_dir.is_dir()
    assert run_dir.name.startswith("astar__")
    assert (run_dir / "run.log").is_file()
    assert (run_dir / "run.json").is_file()
    assert (run_dir / "images").is_dir()
    pngs = list((run_dir / "images").glob("*.png"))
    assert len(pngs) == 1 and pngs[0].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

    log_txt = (run_dir / "run.log").read_text(encoding="utf-8")
    assert "plan_config=" in log_txt
    assert "GetNearestDebrisLocations" in log_txt
    assert "SetCommand" in log_txt
    assert "passed_waypoint" in log_txt
    assert "PanTiltCamera+CaptureCameraImage" in log_txt

    data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert data["metadata"]["plan_name"] == "astar"
    assert data["metadata"]["config"]["max_pops"] == PLAN_CONFIGS["astar"]["max_pops"]
    assert "0.1000" in data["time_to_fuel"] and data["time_to_fuel"]["0.1000"] == pytest.approx(0.42)
    assert data["time_to_thrust"]["0.1000"]["mode"] == "main"
    assert data["time_to_point"]["0.1000"]["target_node_idx"] == 0
    assert len(data["points"]) == 2
    assert data["points"][0]["astar_points"] is not None
    assert data["points"][1]["event"] == "turn_passed"
    assert len(data["radar_calls"]) == 1
    assert data["radar_calls"][0]["positions_rel"] == [[10.0, 20.0, 30.0]]
    np.testing.assert_allclose(data["radar_calls"][0]["positions_abs"], [origin + [10, 20, 30]], atol=0.01)

    # 2. flush_immediately=True: writes run.log immediately during flight
    logger2 = RunLogger("barrier", base_dir=tmp_path, flush_immediately=True, runner="unit_test")
    assert logger2.log_path.is_file()
    logger2.log_line(0.5, "NAV", "immediate flush check")
    assert "immediate flush check" in logger2.log_path.read_text(encoding="utf-8")
    logger2.finalize()


def test_parse_official_engine_score_alert(tmp_path):
    console = tmp_path / "server.log"
    assert read_score_alert(console) == {}
    console.write_text(
        "Crashed into some debris and finished with a score of -219.3; Please try again :)\n"
        "Sim elapsed time   = 88.3 s\nTotal damage taken = 2.51e+05 J\nTotal fuel used    = 35.1 kg\n",
        encoding="utf-8",
    )
    assert read_score_alert(console) == dict(score=-219.3, arrived=False, outcome="damage_limit",
                                             scoring_time_s=88.3, damage_kJ=251, final_fuel_kg=35.1)


def test_engine_run_without_exit_condition_streams_log(tmp_path):
    # The engine kills the controller process at sim end, so in-memory logs would be lost.
    C.configure()
    flight = BasicController.fly(actuator_twin.TwinSim(3.0), planner="astar", threaded=False,
                                 log_dir=tmp_path, log_run=True, source=BasicController.ENGINE_SOURCE)
    assert flight.run_logger.flush_immediately
    assert (flight.run_dir / "run.log").stat().st_size > 0
    assert f"run_json={flight.run_dir / 'run.json'}" in (flight.run_dir / "run.log").read_text(encoding="utf-8")


def test_engine_run_saves_complete_run_json_on_arrival(tmp_path):
    # The engine kills the controller ~8 s after the Scoring alert, before `finally`: run.json must be final at arrival.
    C.configure()
    sim = actuator_twin.TwinSim(3.0)
    sim.goal = sim.r.copy()                              # arrive on the first tick
    flight = BasicController.fly(sim, planner="astar", threaded=False, log_dir=tmp_path, log_run=True,
                                 source=BasicController.ENGINE_SOURCE)
    summary = json.loads((flight.run_dir / "run.json").read_text(encoding="utf-8"))["metadata"]["summary"]
    assert summary["arrived"] is True and "partial" not in summary
    assert "score_estimate" in summary


def test_engine_damage_limit_cuts_thrust_and_prevents_false_arrival(tmp_path):
    class DamagedSim(actuator_twin.TwinSim):
        def damage(self):
            return C.DMAX + 1.0 if self.t >= 0.1 else 0.0

        def wait(self):
            keep_going = super().wait()
            if self.t >= 0.1:
                self.r = self.goal.copy()
            return keep_going

    C.configure()
    sim = DamagedSim(0.4)
    flight = BasicController.fly(sim, planner="v14", threaded=False, log_dir=tmp_path, log_run=True,
                                 source=BasicController.ENGINE_SOURCE)
    summary = json.loads((flight.run_dir / "run.json").read_text(encoding="utf-8"))["metadata"]["summary"]
    assert summary["arrived"] is False and summary["outcome"] == "damage_limit"
    assert "partial" not in summary and "score_estimate" not in summary
    assert all(not np.any(command) for command in sim.cmd)


def test_engine_score_for_reads_app_log_folder(tmp_path, monkeypatch):
    from tools.spaceteams_runner import engine_score_for
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    logs = tmp_path / "SimDynamX" / "STPro" / "Logs" / "261116_134417_PLAN_10_real_calibrated__e25"
    logs.mkdir(parents=True)
    (logs / "PLAN_10_real_calibrated_PySys_Scoring.log").write_text(
        "OnScreenAlert: Successfully escaped Phobos with a score of 37.0; congratulations!\n"
        "OnScreenAlert: Sim elapsed time   = 209.77 s\nOnScreenAlert: Total fuel used    = 52.65 kg\n", encoding="utf-8")
    r = engine_score_for("2026-11-16T13:44:24+00:00")
    assert r["score"] == 37.0 and r["arrived"] and r["scoring_time_s"] == 209.77
    assert engine_score_for("2026-11-16T15:00:00+00:00") == {}          # no folder within 10 min before the run


def _fake_engine_logs(root):
    logs = root / "SimDynamX" / "STPro" / "Logs"
    for name, score in (("261116_134417_PLAN_10_real_calibrated__e25", 37.0), ("261116_134420_PLAN_4_team_pipeline__9d3", 4.4)):
        (logs / name).mkdir(parents=True)
        (logs / name / "x_PySys_Scoring.log").write_text(
            f"OnScreenAlert: Successfully escaped Phobos with a score of {score}; congratulations!\n"
            "OnScreenAlert: Sim elapsed time   = 209.77 s\n", encoding="utf-8")


def _fake_engine_run(run, age_s=120.0):
    import os, time
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"metadata": {
        "runner": "Recorder", "source": "spaceteams", "plan_name": "real_calibrated",
        "timestamp_utc": "2026-11-16T13:44:24+00:00", "summary": {"partial": True, "final_fuel_kg": 40.0}}}), encoding="utf-8")
    (run / "run.log").write_text("[t=1.0s] NAV ...\n", encoding="utf-8")
    old = time.time() - age_s
    for f in ("run.json", "run.log"):
        os.utime(run / f, (old, old))


def test_persist_engine_score_writes_run_files_once(tmp_path, monkeypatch):
    from Systems.EngineScore import persist_engine_score
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    _fake_engine_logs(tmp_path)
    fresh = tmp_path / "fresh"
    _fake_engine_run(fresh, age_s=0)
    assert persist_engine_score(fresh) == {}             # still being written by the controller: left alone
    assert "score" not in json.loads((fresh / "run.json").read_text(encoding="utf-8"))["metadata"]["summary"]
    run = tmp_path / "run"
    _fake_engine_run(run)
    r = persist_engine_score(run)
    assert r["score"] == 37.0                            # the newer team_pipeline folder is a different plan: skipped
    s = json.loads((run / "run.json").read_text(encoding="utf-8"))["metadata"]["summary"]
    assert s["score"] == 37.0 and s["score_source"] == "Space Teams PRO" and s["partial"] is True
    assert s["scoring_time_s"] == 209.77 and s["final_fuel_kg"] == 40.0   # engine values win only where it has them
    before = (run / "run.json").stat().st_mtime_ns
    assert persist_engine_score(run)["score"] == 37.0    # second call: nothing rewritten, one log line
    assert (run / "run.json").stat().st_mtime_ns == before
    assert (run / "run.log").read_text(encoding="utf-8").count("ENGINE SCORE") == 1


def test_gui_listing_saves_engine_score_into_app_launched_run(tmp_path, monkeypatch):
    import time
    from fastapi.testclient import TestClient
    import tools.plan_runner_gui as gui
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    _fake_engine_logs(tmp_path)
    run = tmp_path / "logs" / "spaceteams" / "real_calibrated__20261116_194424__x"
    _fake_engine_run(run)
    monkeypatch.setattr(gui, "DEFAULT_LOG_DIR", tmp_path / "logs")
    rows = TestClient(gui.app).get("/api/runs").json()["runs"]
    assert rows[0]["score"] == 37.0 and rows[0]["breakdown"]["score_source"] == "Space Teams PRO"
    for _ in range(100):                                 # the save runs on a background thread
        summary = json.loads((run / "run.json").read_text(encoding="utf-8"))["metadata"]["summary"]
        if summary.get("score_source"):
            break
        time.sleep(0.05)
    assert summary["score"] == 37.0 and summary["score_source"] == "Space Teams PRO"


def test_gui_blend_download_is_lazy_and_cached(tmp_path, monkeypatch):
    import base64
    from fastapi.testclient import TestClient
    import tools.json_to_blender as j2b
    import tools.plan_runner_gui as gui
    run = tmp_path / "logs" / "python" / "vo__1"
    run.mkdir(parents=True)
    (run / "run.json").write_text('{"metadata": {"runner": "x"}}', encoding="utf-8")
    monkeypatch.setattr(gui, "DEFAULT_LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(gui, "ROOT", tmp_path)
    key = base64.urlsafe_b64encode(str(run.resolve()).encode()).decode().rstrip("=")
    client = TestClient(gui.app)
    monkeypatch.setattr(j2b, "find_blender", lambda *a: None)
    r = client.get(f"/api/runs/{key}/blend")
    assert r.status_code == 501 and "Blender" in r.json()["detail"]
    (tmp_path / "blender").mkdir()
    (tmp_path / "blender" / "vo__1.blend").write_bytes(b"BLENDER-cached")   # newer than run.json: served, not rebuilt
    r = client.get(f"/api/runs/{key}/blend")
    assert r.status_code == 200 and r.content == b"BLENDER-cached"


def test_run_logger_snapshot_writes_partial_run_json(tmp_path):
    from Systems.RunLogger import RunLogger
    C.configure()
    flight = BasicController.fly(actuator_twin.TwinSim(3.0), planner="astar", threaded=False, log_dir=tmp_path, log_run=False)
    lg = RunLogger("astar", base_dir=tmp_path, flush_immediately=True, runner="x", source="spaceteams")
    lg.snapshot()
    meta = json.loads((lg.run_dir / "run.json").read_text(encoding="utf-8"))["metadata"]
    assert meta["summary"]["partial"] is True and meta["run_json"] == str(lg.json_path)
    lg.finalize({"arrived": True})
    assert "partial" not in json.loads(lg.json_path.read_text(encoding="utf-8"))["metadata"]["summary"]
    lg.snapshot()                                        # after finalize: must not clobber the complete file
    assert "partial" not in json.loads(lg.json_path.read_text(encoding="utf-8"))["metadata"]["summary"]


@pytest.mark.slow
def test_fly_creates_run_folder_and_blender_export_builds_blend(tmp_path):
    C.configure()
    sim = actuator_twin.TwinSim(3.0)
    flight = BasicController.fly(
        sim, planner="astar", threaded=False, log_dir=tmp_path, flush_immediately=False, log_run=True
    )
    run_dir = flight.run_dir
    assert run_dir is not None and run_dir.is_dir()
    assert (run_dir / "run.log").is_file()
    json_path = run_dir / "run.json"
    assert json_path.is_file()

    # Check offline_bench also writes run_dir
    field = offline_bench.load_field()
    res = offline_bench.run("velocity_obstacle", 0, field=field, log_dir=tmp_path, log_run=True)
    vo_dir = Path(res["log_dir"])
    assert (vo_dir / "run.json").is_file()
    vo_data = json.loads((vo_dir / "run.json").read_text(encoding="utf-8"))
    assert len(vo_data["radar_calls"]) > 0
    assert len(vo_data["time_to_fuel"]) > 10

    # Test Blender conversion if blender is installed
    if shutil.which("blender") is not None:
        blend_out = tmp_path / "replay.blend"
        proc = subprocess.run(
            [sys.executable, str(ROOT / "tools" / "json_to_blender.py"), str(vo_dir / "run.json"), "-o", str(blend_out), "--radar-step", "20.0", "--max-rocks", "30"],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
        assert blend_out.is_file() and blend_out.stat().st_size > 10000


@pytest.mark.slow
def test_sim_suite_and_error_handling(tmp_path):
    from Systems import SimSuite

    assert {"point_mass", "jet_twin_field", "jet_twin_clean", "stress_storm", "competition_simconf"} <= set(SimSuite.SCENARIOS)

    # Error capture during infeasible parameter choice (should not crash sweeps/GUI)
    bad = SimSuite.run_single("astar", scenario="point_mass", seed=0, config_overrides={"max_pops": 10}, log_dir=tmp_path, log_run=False)
    assert bad["arrived"] is False and bad["error"] and "RuntimeError" in bad["error"]

    # Clean twin stops on arrival and computes score
    twin_res = SimSuite.run_single("astar", scenario="jet_twin_clean", seed=0, log_dir=tmp_path, live=True)
    assert twin_res["arrived"] is True and twin_res["score"] is not None
    replay = json.loads(Path(twin_res["json_path"]).read_text())
    summary = replay["metadata"]["summary"]
    last = max(replay["time_to_point"], key=float)
    assert summary["arrived"] and round(summary["score"], 2) == twin_res["score"]
    assert round(summary["scoring_time_s"], 2) == twin_res["T"]
    for key in ("main_burn_s", "trim_burn_s", "main_fuel_kg", "trim_fuel_kg",
                "accepted_replans", "cruise_speed_mps", "predicted_risk"):
        assert key in summary and key in twin_res
    assert summary["main_burn_s"] > 0 and summary["main_fuel_kg"] > 0
    assert replay["time_to_point"][last]["mode"] == "arrived"
    first = min(replay["time_to_point"], key=float)
    expected = actuator_twin.SCENE.body_rotation(actuator_twin.SCENE.shuttle).as_quat()
    actual = replay["time_to_point"][first]["body_quaternion_xyzw"]
    assert abs(np.dot(actual, expected)) == pytest.approx(1.0, abs=1e-7)
    assert np.linalg.norm(np.array(summary["final_position"]) - replay["metadata"]["core"]["target"]) <= C.WIN
    assert json.loads((Path(twin_res["run_dir"]) / "live.jsonl").read_text().splitlines()[-1])["mode"] == "arrived"

    # Suite report generation on point_mass with velocity_obstacle (fast)
    rep = SimSuite.run_suite(["velocity_obstacle"], scenarios=["point_mass"], seeds=[0], log_dir=tmp_path, log_run=True)
    assert Path(rep["report_path"]).is_file()
    assert rep["summary"][0]["arrived"] == 1
    pm_res = rep["results"][0]
    for key in ("main_burn_s", "trim_burn_s", "main_fuel_kg", "trim_fuel_kg",
                "accepted_replans", "cruise_speed_mps", "predicted_risk", "min_clear"):
        assert key in pm_res
    point_mass = json.loads(Path(pm_res["json_path"]).read_text())
    assert point_mass["metadata"]["summary"]["min_clear"] is not None
    scene = actuator_twin.SCENE
    frame = point_mass["scene_frames"][0]
    np.testing.assert_allclose(frame["positions_abs"][2], scene.entities["Debris2"]["Location"][1:], atol=1e-4)
    np.testing.assert_allclose(frame["rotations_xyzw"][2], scene.body_rotation(scene.entities["Debris2"]).as_quat(), atol=1e-6)


def test_damage_limit_score_uses_full_damage_before_arrival():
    sim = actuator_twin.TwinSim(10)
    sim.r = sim.goal.copy()
    sim.used = 30
    damage = actuator_twin.SCENE.scoring["DamageThreshold_J"] * 1.2
    summary = sim._score_summary(10, hits=[damage])
    assert summary["outcome"] == "damage_limit" and not summary["arrived"]
    assert summary["damage_kJ"] == pytest.approx(damage / 1000)
    assert summary["score"] == pytest.approx(50 - 40 - 240)


def test_plan_runner_gui_endpoints(tmp_path, monkeypatch):
    import time
    from fastapi.testclient import TestClient
    from Systems import SimSuite
    import tools.plan_runner_gui as gui

    monkeypatch.setattr(gui, "DEFAULT_LOG_DIR", tmp_path)
    monkeypatch.setattr(SimSuite, "DEFAULT_LOG_DIR", tmp_path)
    client = TestClient(gui.app)

    r_html = client.get("/")
    assert r_html.status_code == 200 and "Team Batasha" in r_html.text

    plans = client.get("/api/plans").json()
    assert {"v11", "v13", "v14", "musubi"} <= {plan["name"] for plan in plans["plans"]}
    assert "point_mass" in plans["scenarios"]
    v11 = next(plan for plan in plans["plans"] if plan["number"] == 11)
    assert v11["name"] == "v11"
    assert {"a", "b", "c"} <= {preset["name"] for preset in v11["variant_info"]["variants"]}
    assert "point_mass_clean" in plans["scenarios"]
    assert "calibrated" in plans["trackers"]
    orig_pops = next(p["config"]["max_pops"] for p in plans["plans"] if p["name"] == "astar")

    upd = client.post("/api/plans/astar/config", json={"updates": {"max_pops": orig_pops + 5}, "persist": False})
    assert upd.status_code == 200 and upd.json()["config"]["max_pops"] == orig_pops + 5
    rst = client.post("/api/plans/astar/reset")
    assert rst.status_code == 200 and rst.json()["config"]["max_pops"] == orig_pops

    # 15 s of flight is enough to exercise the endpoints; arrival is covered by the slow flight tests.
    job_id = client.post("/api/run", json={"mode": "single", "plan": "velocity_obstacle", "scenario": "point_mass",
                                           "seeds": [0], "max_time": 15}).json()["job_id"]
    state = {}
    for _ in range(600):
        state = client.get(f"/api/jobs/{job_id}").json()
        if state["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert state["status"] == "done" and state["results"][0]["run_key"]

    runs = client.get("/api/runs").json()["runs"]
    assert len(runs) == 1
    assert runs[0]["source"] == "python"
    assert Path(runs[0]["run_dir"]).parent.name == "python"
    assert not client.get("/api/runs", params={"source": "spaceteams"}).json()["runs"]
    live = client.get(f"/api/jobs/{job_id}/live").json()
    assert live["metadata"]["source"] == "python" and len(live["frames"]) > 10
    assert client.get(f"/api/jobs/{job_id}/live", params={"after": live["cursor"]}).json()["frames"] == []
    run_id = runs[0]["run_id"]
    rep = client.get(f"/api/runs/{run_id}/replay").json()
    assert len(rep["frames"]) > 10 and len(rep["points"]) >= 1
    lg = client.get(f"/api/runs/{run_id}/log", params={"category": "NAV"}).json()
    assert lg["total_lines"] > 0
    imported = client.post("/api/replay", json={
        "metadata": {"runner": "SpaceTeamsSim", "summary": {"arrived": True, "duration_s": 100}},
        "time_to_point": {"0": {"shuttle_pos": [0, 0, 0]}},
    })
    assert imported.status_code == 200
    assert imported.json()["source"] == "spaceteams"
    assert imported.json()["score_breakdown"]["score"] is None


def test_config_json_symlink_and_parameterized_override(tmp_path):
    from Systems import plans as P
    from Systems.SimSuite import effective_plan_config
    from Systems.plans.common import activate_variant, load_plan_config, save_variant

    # Copy plan_8_astar config layout into tmp_path so we test symlinks cleanly
    plan_tmp = tmp_path / "plan_8_astar"
    (plan_tmp / "configs").mkdir(parents=True)
    def_src = (P.get_plan_dir("astar") / "configs" / "default.json").read_text(encoding="utf-8")
    (plan_tmp / "configs" / "default.json").write_text(def_src, encoding="utf-8")
    try:
        (plan_tmp / "config.json").symlink_to(Path("configs") / "default.json")
    except OSError:                          # Windows without symlink privilege: git writes the link as this text
        (plan_tmp / "config.json").write_text("configs/default.json", encoding="utf-8")
    fake_cfg_py = plan_tmp / "config.py"
    defaults = P.PLAN_DEFAULTS["astar"]

    assert load_plan_config(str(fake_cfg_py), defaults)["max_pops"] == defaults["max_pops"]

    save_variant(plan_tmp, "fast", {**defaults, "max_pops": 7777}, activate=True)
    assert load_plan_config(str(fake_cfg_py), defaults)["max_pops"] == 7777

    activate_variant(plan_tmp, "default")
    assert load_plan_config(str(fake_cfg_py), defaults)["max_pops"] == defaults["max_pops"]

    # Parameterized JSON override takes highest priority
    pjson = tmp_path / "override.json"
    pjson.write_text(json.dumps({"*": {"p_safe": 0.03}, "astar": {"max_pops": 10}}), encoding="utf-8")
    loaded = load_plan_config(str(fake_cfg_py), defaults, extra_override=pjson)
    assert loaded["max_pops"] == 10 and loaded["p_safe"] == 0.03

    original = dict(P.get_config("astar"))
    eff = effective_plan_config("astar", {"max_pops": 999}, param_json=pjson)
    assert eff["max_pops"] == 10 and eff["p_safe"] == 0.03
    assert P.get_config("astar") == original


def test_plan10_real_calibrated_defaults_and_simconf():
    cfg = get_config(10)
    assert cfg["name"] == "real_calibrated"
    assert cfg["tracker"] == "calibrated"
    assert cfg["burn"] == "steer"
    assert cfg["trim_exact"] is True
    assert cfg["frame_accel"] is True
    assert cfg["rel_hit"] is False
    assert cfg["r_sh"] == 4.5
    assert cfg["main_min"] == 2.0
    assert cfg["hold_t"] == 12.0
    assert cfg["guard_plan"] is False
    assert cfg["keep_band"] == 3.0
    assert cfg["nudge"] is True

    sim_path = ROOT / "sims" / "PLAN_10_real_calibrated.simconf"
    assert sim_path.is_file()
    sim = json.loads(sim_path.read_text(encoding="utf-8"))
    controller = next(s for s in sim["SystemManager"]["Systems"] if s["Nametag"] == "Controller")
    assert controller["Inst_Parameters"]["Planner"] == "real_calibrated"


def test_unified_fast_accel_controls_and_clean_sweep():
    from Systems import Core as C, SimSuite

    required_keys = {
        "tracker", "burn", "accel_till_speed", "accel_dur_s",
        "hold_t", "main_min", "trim_exact", "frame_accel", "r_sh", "guard_plan",
    }
    for num in range(1, 12):
        cfg = get_config(num)
        missing = required_keys - set(cfg.keys())
        assert not missing, f"Plan {num} ({cfg.get('name')}) missing keys: {missing}"
        assert cfg["tracker"] == "calibrated"
        assert cfg["burn"] == "steer"
        assert cfg["trim_exact"] is True
        assert cfg["hold_t"] >= 10.0

    assert C.resolve_vc({"accel_till_speed": 9.2}) == 9.2
    assert C.resolve_vcs({"accel_till_speed": 9.2}) == [8.7, 9.2, 9.7]
    assert C.resolve_vcs({"vcs": [6.5, 7.5, 8.5]}) == [6.5, 7.5, 8.5]
    assert C.resolve_vcs({"vcs": [6.5, 7.5, 8.5], "accel_till_speed": 9.2}) == [8.2, 9.2, 10.2]

    r_short = SimSuite.run_single(
        "velocity_obstacle", scenario="point_mass_clean", seed=0,
        config_overrides={"accel_dur_s": 50.0}, log_run=False, raise_on_error=True,
    )
    r_long = SimSuite.run_single(
        "velocity_obstacle", scenario="point_mass_clean", seed=0,
        config_overrides={"accel_dur_s": 100.0}, log_run=False, raise_on_error=True,
    )
    assert r_short["arrived"] and r_long["arrived"]
    assert r_short["hits"] == 0 and r_long["hits"] == 0
    assert r_short["min_clear"] is None and r_long["min_clear"] is None
    assert r_long["max_speed"] > r_short["max_speed"] + 2.0
    assert r_long["T"] < r_short["T"] - 10.0                 # faster, and it still steers onto the target after the burn
