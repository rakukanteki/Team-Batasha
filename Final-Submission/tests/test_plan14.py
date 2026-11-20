import numpy as np
from scipy.spatial.transform import Rotation

from Systems import Actuation, Core
from Systems.Flight import Flight
from Systems.plans import PLAN_DIRS, get_config
from Systems.plans.common import read_config_json
from Systems.plans.plan_13_v13.planner import make_planner, rollout
from Systems.plans.plan_14_v14.launch import ClearanceLaunch
from tools.upstream_scene import Scene


def test_v14_initial_alignment_turns_negative_y_and_releases_axial_thrust():
    scene = Scene()
    *_, start, target = scene.field()
    orientation = scene.body_rotation(scene.shuttle)
    bearing = target - start
    body_bearing = orientation.inv().apply(bearing / np.linalg.norm(bearing))
    state = {"config": get_config(14), "ref": body_bearing}
    rates = np.zeros(3)
    burn = Actuation.BURNS["x_forward"]
    linear, angular = burn(body_bearing, rates, state)
    assert not np.any(linear)
    assert angular[1] < 0 and abs(angular[1]) > abs(angular[2])
    rotated = orientation * Rotation.from_euler("y", -90, degrees=True)
    state["ref"] = rotated.inv().apply(bearing / np.linalg.norm(bearing))
    linear, angular = burn(state["ref"], rates, state)
    np.testing.assert_array_equal(linear, Actuation.THRUST * Actuation.X_AXIS)
    np.testing.assert_array_equal(angular, 0)

    requested = np.array([0., 1., 0.])
    linear, _ = burn(requested, rates, state)
    np.testing.assert_array_equal(linear, Actuation.THRUST * requested)


def test_v14_coast_deadband_and_speed_presets():
    cfg = get_config(14)
    coast = Actuation.BURNS[cfg["burn"]].coast_attitude
    assert not np.any(coast(Actuation.X_AXIS, np.zeros(3), {}, cfg))
    assert np.any(coast(np.array([0., 0., 1.]), np.zeros(3), {}, cfg))
    assert Core.resolve_vcs(cfg) == [10.]
    preset = read_config_json(PLAN_DIRS[14] / "configs" / "s20.json")
    assert Core.resolve_vcs({**cfg, **preset}) == [20.]


def test_shared_launch_requires_clearance_and_acquisition_and_preserves_upward_thrust():
    up = Core.S / np.linalg.norm(Core.S)
    for plan in (14, 15):
        cfg = get_config(plan)
        launch = ClearanceLaunch(cfg)
        mode, acceleration = launch.command(Core.S, np.zeros(3), True, -up)
        assert mode == "launch_lift" and acceleration @ up > 0.0
        high = Core.S + (cfg["launch_clearance_m"] + 1.0) * up
        mode, acceleration = launch.command(high, 0.25 * up, False, None)
        assert mode == "launch_slew" and acceleration @ up > 0.0
        lateral = Core.E1 - (Core.E1 @ up) * up
        mode, acceleration = launch.command(high, 0.25 * up, False, -up + lateral)
        assert mode == "launch_evade" and acceleration @ lateral > 0.0
        np.testing.assert_allclose(acceleration @ up, 0.0, atol=1e-12)
        assert launch.command(high, -0.4 * up, True, None)[0] == "launch_lift"
        assert launch.command(high, 0.25 * up, True, None) is None and launch.done
        flight = Flight(plan)
        assert flight.decide(20.0, Core.S, np.zeros(3), 0.0, 0.02)[0] == "launch_lift"


def test_launch_timeouts_release_only_with_clearance():
    cfg = get_config(15)
    up = Core.S / np.linalg.norm(Core.S)
    high = Core.S + (cfg["launch_clearance_m"] + 5.0) * up
    stopped = np.zeros(3)
    launch = ClearanceLaunch(cfg)
    launch.command(Core.S, stopped, False, None, t=0.0)
    assert launch.command(high, stopped, False, None, t=15.0)[0] == "launch_slew"
    assert launch.command(high, stopped, False, None, t=15.0 + cfg["launch_slew_limit_s"]) is None
    assert launch.done and launch.timed_out

    bearing = np.array([0., 0., 1.])
    state = {"config": cfg, "ref": bearing, "launch_released": True}
    linear, angular = Actuation.BURNS["x_forward"](bearing, stopped, state)
    np.testing.assert_array_equal(linear, Actuation.THRUST * bearing)
    assert np.any(angular) and not state["x_acquired"]

    launch = ClearanceLaunch(cfg)
    launch.command(Core.S, stopped, False, None, t=0.0)
    mode, acceleration = launch.command(Core.S, stopped, False, None, t=cfg["launch_deadline_s"])
    assert mode == "launch_failed" and not launch.done
    np.testing.assert_array_equal(acceleration, 0)


def test_launch_turns_gently_during_climb_before_full_clearance():
    cfg = get_config(14)
    up = Core.S / np.linalg.norm(Core.S)
    height = Core.S + 6.0 * up
    launch = ClearanceLaunch(cfg)
    assert launch.command(height, 0.1 * up, False, None)[0] == "launch_lift"
    mode, acceleration = launch.command(height, 0.70 * up, True, None)
    assert mode == "launch_slew_climb" and not launch.done
    assert acceleration @ up > 0.0

    bearing = np.array([0., 0., 1.])
    slow_turn = Actuation.launch_slew_attitude(bearing, np.zeros(3), cfg, early=True)
    full_turn = Actuation.launch_slew_attitude(bearing, np.zeros(3), cfg)
    assert full_turn[1] < slow_turn[1] < 0.0


def test_small_velocity_correction_uses_proportional_thrust():
    cfg = get_config(14)
    state = {"config": cfg, "ref": Actuation.X_AXIS, "ng": 0.1}
    burn = Actuation.BURNS["x_forward"]
    small, _ = burn(Actuation.X_AXIS, np.zeros(3), state)
    state["ng"] = 0.5
    large, _ = burn(Actuation.X_AXIS, np.zeros(3), state)
    assert 0 < small[0] < large[0] == Actuation.THRUST


def test_fuel40_bounds_burn_heading_and_leaves_small_trims_docking_facing():
    cfg = {**get_config(14), **read_config_json(PLAN_DIRS[14] / "configs" / "fuel40.json")}
    assert cfg["fuel_goal_kg"] == 40.0
    assert Core.resolve_vcs(cfg) == [4.8, 5.0, 5.25]
    assert cfg["stop_dv_mps"] < cfg["ignite_dv_mps"]
    docking = np.array([[1., 0., 0.], [1., 0., 0.]])
    gain = np.array([[1., 4., 0.], [0., 0.1, 0.]])
    reference = Actuation.x_forward_burn_bearings(gain, docking, cfg)
    heading = np.rad2deg(np.arccos(np.clip(reference[0] @ docking[0], -1.0, 1.0)))
    np.testing.assert_allclose(heading, cfg["x_forward_burn_heading_deg"])
    np.testing.assert_array_equal(reference[1], docking[1])


def test_fuel40_retains_waypoint_progress_when_no_replacement_is_available():
    cfg = {**get_config(14), **read_config_json(PLAN_DIRS[14] / "configs" / "fuel40.json")}
    flight = Flight("v14", config=cfg, threaded=False)
    route = ([Core.S + 30.0 * Core.SH, Core.T], 5.5)
    flight.plan = route
    flight.idx = 1
    flight.committed = True
    flight._replan(50.0, Core.S, np.zeros(3), 8.0)
    flight._accept_result(51.0, Core.S, np.zeros(3), None, {"action": "v13_no_safe_route"})
    assert flight.plan is route and flight.idx == 1 and flight.committed


def test_fuel40_accepts_a_small_docking_offset_but_keeps_waypoints(monkeypatch):
    monkeypatch.setattr(Core, "accel", lambda r, v, frame=False: np.zeros_like(r))
    cfg = {**get_config(14), **read_config_json(PLAN_DIRS[14] / "configs" / "fuel40.json")}
    start = Core.T - 300.0 * Core.SH
    grazing = 4.0 * Core.SH + 0.24 * Core.E1
    wide = 4.0 * Core.SH + 0.55 * Core.E1
    assert Core.coast_pass_near(start, grazing, Core.T, cfg["coast_docking_radius_m"], 360.0)
    assert not Core.coast_pass_near(start, wide, Core.T, cfg["coast_docking_radius_m"], 360.0)

    flight = Flight("v14", config=cfg, threaded=False)
    flight.launch.done = True
    assert flight.decide(50.0, start, grazing, 20.0, 0.02)[0] == "coast"
    waypoint = start + 100.0 * Core.SH - 30.0 * Core.E1
    flight.plan = ([waypoint, Core.T], 5.5)
    assert flight.decide(51.0, start, grazing, 20.0, 0.02)[0] != "coast"
    waypoint = start + 100.0 * Core.SH
    flight.plan = ([waypoint, Core.T], 5.5)
    assert flight.decide(52.0, start, grazing, 20.0, 0.02)[0] == "coast"


def test_fuel40_late_cruise_uses_open_path_and_fuel_headroom():
    cfg = {**get_config(14), **read_config_json(PLAN_DIRS[14] / "configs" / "fuel40.json")}
    planner = make_planner(cfg)
    direction = Core.SH
    attitude = np.column_stack([direction, Core.E1, np.cross(direction, Core.E1)])
    state = {"r": Core.S + 600.0 * direction, "v": 4.2 * direction, "t": 170.0, "fuel": 22.0,
             "burning": False, "main": False, "body": attitude, "rates": np.zeros(3),
             "main_axis_acquired": True, "launch_released": True}
    empty = (np.empty((0, 3)), np.empty((0, 3)), np.empty(0), np.empty(0), np.empty(0))
    conservative = ([Core.T], 5.0)
    fast, info = planner(state, empty, conservative, True)
    assert fast[1] in cfg["late_cruise_speeds_mps"] and info["action"] == "v14_late_cruise"
    encounter = rollout([fast], state, empty, cfg)["path"][55, 0]
    crossing_speed = 3.0 * Core.E1
    rocks = ((encounter - 55.0 * crossing_speed)[None], crossing_speed[None],
             np.array([12.0]), np.array([1.0]), np.array([0.08]))
    slower, info = planner(state, rocks, conservative, True)
    assert slower[1] < fast[1] and info["clear"] >= cfg["route_clearance_m"]
    plan, info = planner({**state, "fuel": 39.0}, empty, conservative, True)
    assert plan[1] == 5.0 and info["action"] == "keep"


def test_fuel40_allows_a_steady_off_angle_final_coast():
    cfg = {**get_config(14), **read_config_json(PLAN_DIRS[14] / "configs" / "fuel40.json")}
    angle = np.deg2rad(25.0)
    bearing = np.array([np.cos(angle), 0.0, np.sin(angle)])
    calm = np.array([0.0, 0.013, 0.0])
    unstable = np.array([0.0, 0.09, 0.0])
    assert not np.any(Actuation.x_forward_coast_attitude(110.0 * bearing, calm, {}, cfg))
    assert np.any(Actuation.x_forward_coast_attitude(110.0 * bearing, unstable, {}, cfg))
    assert np.any(Actuation.x_forward_coast_attitude(200.0 * bearing, calm, {}, cfg))
