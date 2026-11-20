from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from Systems import Actuation, Core as C
from Systems.Flight import Flight
from Systems.plans import PLAN_DIRS, get_config
from Systems.plans.common import read_config_json
from Systems.plans.plan_13_v13 import planner as v13


def test_handoff_requires_elapsed_time_distinct_samples_and_known_tracks():
    cfg = {**get_config(13), "route_expected_tracks": 2}
    flight = Flight("v13", config=cfg)
    calls = []
    flight.planner = lambda *args: (calls.append(args[0]) or None, {"action": "v13_no_safe_route"})
    flight.world = SimpleNamespace(
        n=2, count=np.array([3, 2]), t=np.array([12.0, 12.0]),
        predict=lambda t: (np.zeros((2, 3)), np.zeros((2, 3)),
                           np.ones(2), np.ones(2), np.ones(2)),
    )
    flight._replan(14.0, C.S, np.zeros(3), 0.0)
    flight.world.count[1] = 3
    flight._replan(13.9, C.S, np.zeros(3), 0.0)
    assert not calls
    flight._replan(14.1, C.S, np.zeros(3), 0.0)
    assert len(calls) == 1 and len(flight.events) == 0
    assert flight.plan is None


def test_route_keeps_progress_and_rejects_pre_dodge_result():
    flight = Flight(13)
    old = ([C.S + 200 * C.SH, C.T], 8.4)
    flight.plan = old
    flight.idx = 1
    flight._accept_result(16.0, C.S, np.zeros(3), ([C.T], 8.4), {"action": "keep"})
    assert flight.plan is old and flight.idx == 1
    flight.guard_end = 16.5
    info = {"action": "v13_direct", "snapshot_t": 16.0, "ptot": 0.0, "K": 0, "cost": 0.0}
    flight._accept_result(17.0, C.S, np.zeros(3), ([C.T], 8.4), info)
    assert flight.plan is old


def test_target_z_attitude_is_independent_of_avoidance_thrust():
    direction = np.array([1., 0., 0.])
    rates = np.zeros(3)
    burn = Actuation.BURNS["target_z"]
    lin, ang = burn(direction, rates, {"ref": np.array([0., 0., 1.])})
    assert lin[0] > 0
    np.testing.assert_array_equal(ang, 0)
    assert burn.attitude(np.array([1., 0., 0.]), rates)[1] > 0


def test_clear_field_selects_zero_intermediate_points_and_keeps_them():
    st = dict(r=C.S.copy(), v=np.zeros(3), burning=False, main=False, fuel=0., t=14.)
    rocks = (np.empty((0, 3)), np.empty((0, 3)), np.empty(0), np.empty(0), np.empty(0))
    plan, info = v13.v13(st, rocks, None, False)
    assert info["action"] == "v13_direct" and info["K"] == 0
    assert v13.v13(st, rocks, plan, True)[1]["action"] == "keep"


def test_coast_pointing_hysteresis_suppresses_small_corrections():
    cfg = get_config(13)
    rates = np.zeros(3)
    memory = {}

    def command(deg, w=rates):
        ref = Rotation.from_euler("y", deg, degrees=True).apply([0., 0., 1.])
        return Actuation.target_z_coast_attitude(ref, w, memory, cfg)

    assert not np.any(command(4))
    assert np.any(command(6))
    assert np.any(command(4))
    assert not np.any(command(1))
    assert np.any(command(0, np.array([0.021, 0., 0.])))


def test_v13_speed_presets_select_only_the_requested_target():
    for speed in (10.0, 20.0):
        cfg = {**get_config(13), **read_config_json(PLAN_DIRS[13] / "configs" / f"s{int(speed)}.json")}
        assert C.resolve_vc(cfg) == speed
        assert C.resolve_vcs(cfg) == [speed]


def test_rollout_prices_torque_only_coast(monkeypatch):
    def coast_step(p, v, burning, main, index, *args, **kwargs):
        return p, v, burning, main, index, np.zeros(len(p)), None

    monkeypatch.setattr(C, "guid_step", coast_step)
    cfg = {**get_config(13), "tmax": 1.0}
    attitude = np.column_stack((C.E1, C.E2, C.SH)) @ Rotation.from_euler("y", 4, degrees=True).as_matrix()
    st = dict(r=C.S.copy(), v=np.zeros(3), burning=False, main=False, fuel=0., t=14., body=attitude)
    empty = (np.empty((0, 3)), np.empty((0, 3)), np.empty(0), np.empty(0), np.empty(0))
    plans = [([C.T], 8.4)]
    continuous = v13.rollout(plans, st, empty, {**cfg, "coast_pointing_deadband": False})
    quiet = v13.rollout(plans, st, empty, cfg)
    assert continuous["fuel"][0] > 0
    assert quiet["fuel"][0] == 0
