import numpy as np

from Systems import Core as C
from Systems.Flight import Flight
from Systems.Maneuvers import Maneuver, Segment, maneuver_of
from Systems.plans import PLANNER_FACTORIES, get_config


def test_v16_lifts_immediately_and_guard_can_interrupt(monkeypatch):
    flight = Flight(16)
    up = C.S / np.linalg.norm(C.S)
    mode, aim = flight.decide(0.0, C.S, np.zeros(3), 0.0, 0.02)
    assert mode == "main"
    assert aim @ up / np.linalg.norm(aim) >= np.cos(np.deg2rad(8.0))

    monkeypatch.setattr(flight, "_nearby", lambda *_: ())
    monkeypatch.setattr(flight, "_guard", lambda *_: -up)
    mode, escape = flight.decide(1.0, C.S, np.zeros(3), 0.0, 0.02)
    assert mode == "evade" and flight.guarding
    assert escape @ up >= 0.0


def test_v16_schedule_survives_old_age_limit_and_checks_thirty_seconds():
    cfg = get_config(16)
    planner = PLANNER_FACTORIES[16](cfg)
    state = dict(t=18.0, r=C.S + 17.0 * C.SH, v=2.15 * C.SH, u=C.SH, fuel=10.2,
                 burning=True, main=True)
    empty = (np.empty((0, 3)), np.empty((0, 3)), np.empty(0), np.empty(0), np.empty(0))
    plan, info = planner(state, empty, None, False)
    schedule = maneuver_of(plan)
    assert info["action"] == "v16_burn" and info["predicted_fuel_kg"] < 45.0
    assert info["predicted_arrival_s"] < 200.0
    assert schedule.review_horizon == 30.0
    assert schedule.action(state["t"] + 10.0)[1] == "main"
    assert schedule.check(state["t"] + 10.0, schedule.path[10], schedule.velocity[10], None)
    rock = schedule.path[35] - 0.5 * C.accel(schedule.path[35], np.zeros(3), frame=True) * 25.0**2
    rocks = (rock[None], np.zeros((1, 3)), np.array([8.0]), np.array([2.0]), np.array([0.1]))
    assert not schedule.check(state["t"] + 10.0, schedule.path[10], schedule.velocity[10], rocks)


def test_small_scheduled_correction_has_a_bounded_impulse():
    pulse = Segment(10.0, (0.0, 1.0, 0.0), pulse_accel=0.05)
    mode, acceleration = pulse.control()
    assert mode == "pulse"
    assert np.linalg.norm(acceleration) * pulse.duration == 0.5


def test_v16_guard_uses_earlier_burn_encounter_even_if_later_coast_misses_by_less():
    flight = Flight(16)
    r = C.S + 15.0 * C.SH
    v = 0.4 * C.SH
    rocks = ((r + 12.0 * C.SH)[None], np.zeros((1, 3)), np.array([4.0]),
             np.ones(1), np.zeros(1))
    flight.body_z = lambda: C.SH
    assert flight._guard_escape(r, v, rocks) is None
    flight.burning = flight.main = True
    assert flight._guard_escape(r, v, rocks) is not None


def test_v16_late_burn_requires_a_faster_schedule_within_the_fuel_reserve():
    cfg = get_config(16)
    planner = PLANNER_FACTORIES[16](cfg)
    r, v = C.S + 600.0 * C.SH, 5.7 * C.SH
    state = dict(t=105.0, r=r, v=v, u=C.SH, fuel=25.0, burning=False, main=False)
    fallback = ([C.T], 8.9)
    coast = Maneuver(105.0, (Segment(170.0),), 1.0, r + np.arange(171)[:, None] * v,
                     np.tile(v, (171, 1)), fallback, True, review_horizon=30.0)
    plan = (*fallback, 105.0, coast)
    empty = (np.empty((0, 3)), np.empty((0, 3)), np.empty(0), np.empty(0), np.empty(0))
    _, info = planner(state, empty, plan, True)
    assert info["action"] == "v16_late_burn"
    assert info["predicted_fuel_kg"] <= cfg["fuel_goal_kg"]
    retained, info = planner({**state, "fuel": 40.0}, empty, plan, True)
    assert retained is plan and info["action"] == "keep"


def test_escape_guard_trusts_a_route_burn_that_clears_the_rock():
    cfg = {**get_config(16), "guard_horizon_s": 30.0, "guard_abort_s": 24.0, "guard_sigma": 1.0,
           "guard_max_sigma_m": 15.0, "guard_assume_burn": False}
    flight = Flight(16, config=cfg)
    r, v = C.S + 50.0 * C.SH, 5.0 * C.SH
    rock = (np.array([r + 100.0 * C.SH]), np.zeros((1, 3)), np.array([10.0]), np.array([1.0]), np.array([0.05]))
    assert flight._guard_escape(r, v, rock) is not None              # coasting runs straight into it at 20 s
    assert flight._guard_escape(r, v, rock, 1.0 * C.E1, 0.1) is None  # the route's sideways burn passes 15 m off
    assert flight._guard_escape(r, v, rock, 0.1 * C.E1, 0.1) is not None
