"""Verify timed v11 maneuvers across the planner and flight boundary."""
import numpy as np

from Systems import Core as C
from Systems.BasicController import fly
from Systems.Flight import Flight
from Systems.Maneuvers import BODY_COMMANDS, Maneuver, Segment, maneuver_of
from Systems.plans import PLANNER_FACTORIES, get_config, get_plan_dir
from Systems.plans.common import read_config_json
from Systems.plans.plan_11_v11.planner import _scenario_scores, candidates, make_planner, roll_out

EMPTY_ROCKS = (np.empty((0, 3)), np.empty((0, 3)), np.empty(0), np.empty(0), np.empty(0))


def config(variant):
    return {**get_config(11), **read_config_json(get_plan_dir(11) / "configs" / f"{variant}.json")}


def coast_state():
    return dict(r=C.T - 300 * C.SH, v=10 * C.SH, t=100.0, fuel=35.0, burning=False, main=False)


def test_ballistic_entry_executes_and_a_late_track_cancels_it(monkeypatch):
    flight = Flight("v11", config=config("b"))
    state = coast_state()
    t, r, v = state["t"], state["r"], state["v"]
    plan, info = flight.planner(state, EMPTY_ROCKS, None, False)
    assert info["action"] == "coast_entry"
    flight._accept_result(t + 0.5, r + 0.5 * v, v, plan, info)
    active = maneuver_of(flight.plan)
    assert active is not None and active.start == t + 0.5

    flight.next_replan = t + 3
    mode, _ = flight.decide(t + 0.5, r + 0.5 * v, v, state["fuel"], 0.02)
    assert mode == "coast"

    rocks = ((r + 50 * C.SH)[None, :], np.zeros((1, 3)), np.array([10.0]), np.ones(1), np.zeros(1))
    monkeypatch.setattr(flight, "_nearby", lambda *args: rocks)
    mode, _ = flight.decide(t + 0.8, r + 0.8 * v, v, state["fuel"], 0.02)
    assert mode == "trim"
    assert maneuver_of(flight.plan) is None
    assert flight.guarding


def test_expired_result_cannot_restart_a_timed_command():
    flight = Flight(11, config=config("b"))
    st = coast_state()
    plan, info = flight.planner(st, EMPTY_ROCKS, None, False)
    flight._accept_result(st["t"] + 2, st["r"] + 2 * st["v"], st["v"], plan, info)
    assert flight.plan is None


def test_c_searches_body_commands_and_scores_uncertain_fields():
    st = {**coast_state(), "body": np.eye(3), "rates": np.zeros(3), "u": np.eye(3)[:, 2]}
    a, b, c = (config(variant) for variant in ("a", "b", "c"))
    assert a["w_turn"] and a["ignore_per_rock"] and not a["timed_burns"]
    assert all(segs[0].body_command is None for segs, _ in candidates(st, b, None))
    slower = {**st, "v": 5 * C.SH}
    assert any(segs[0].body_command == 4 for segs, _ in candidates(slower, c, None))
    command = Segment(2, body_command=0)
    options = [((command, Segment(300)), "test")]
    rollout = roll_out(st, options, c)
    delta_v = rollout.velocity[1, 0] - st["v"] - C.accel(st["r"], st["v"], frame=True)
    np.testing.assert_allclose(delta_v, BODY_COMMANDS[0, 4:7], atol=1e-12)

    distant = (C.T + 140 * C.E1)[None, :]
    rocks = (distant, np.zeros((1, 3)), np.array([5.0]), np.ones(1), np.full(1, 0.02))
    plan, info = make_planner(c)(st, rocks, None, False)
    assert maneuver_of(plan) is not None
    assert np.isfinite(info["cost"]) and info["ptot"] <= c["p_safe"]


def test_raw_body_segment_reaches_the_sim_command_interface(monkeypatch):
    cfg = config("c")
    st = coast_state()
    r, v = st["r"], st["v"]

    class Sim:
        def time(self):
            return st["t"]

        def state(self):
            return r.copy(), v.copy()

        def target(self):
            return C.T.copy()

        def rates(self):
            return np.zeros(3)

        def to_body(self, vector):
            return vector

        def fuel(self):
            return st["fuel"]

        def radar(self):
            return []

        def command(self, lin, ang):
            self.sent = lin.copy(), ang.copy()

        def wait(self):
            return False

        def log(self, message, error=False):
            assert not error, message

    class Planner:
        def __call__(self, snapshot, rocks, previous, committed):
            segments = (Segment(2, body_command=4), Segment(300))
            rollout = roll_out(snapshot, [(segments, "command_sequence")], cfg)
            fallback = ([C.T.copy()], 8.4)
            maneuver = Maneuver(snapshot["t"], segments, 1.0, rollout.path[:, 0],
                                rollout.velocity[:, 0], fallback, True, snapshot["body"])
            return (*fallback, snapshot["t"], maneuver), {
                "action": "command_sequence", "cost": -60, "K": 0, "ptot": 0.0,
            }

    monkeypatch.setitem(PLANNER_FACTORIES, "v11", lambda cfg: Planner())
    sim = Sim()
    old_geometry = C.S.copy(), C.T.copy(), C.A_BURN
    try:
        flight = fly(sim, planner="v11", config=cfg, threaded=False, log_run=False)
    finally:
        C.configure(*old_geometry)
    assert flight.body_command == 4
    np.testing.assert_array_equal(sim.sent[0], [0.5, 0, 0])
    np.testing.assert_array_equal(sim.sent[1], [0, 0.02, 0])


def test_c_prices_a_damage_failure_in_the_sampled_field():
    st = coast_state()
    step = 20 * C.SH
    path = np.stack([st["r"] + t * step for t in range(4)])
    alternatives = np.stack((path, path + 25 * C.E1), axis=1)
    rocks = (path[1][None, :], np.zeros((1, 3)), np.array([5.0]), np.full(1, 0.1), np.zeros(1))
    mean, worst = _scenario_scores(alternatives, np.array([3, 3]), np.zeros(2), rocks, st, config("c"))
    assert mean[0] < 0 < mean[1]
    assert worst[0] < 0 < worst[1]
