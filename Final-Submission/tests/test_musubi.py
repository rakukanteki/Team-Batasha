import numpy as np

from Systems import Core as C
from Systems.plans import PLANNERS, get_config
from Systems.plans.plan_15_musubi import planner as search


def state():
    return dict(r=C.S.copy(), v=6.0 * C.SH, burning=True, main=True, fuel=0., t=14.,
                body=np.column_stack((C.SH, C.E1, C.E2)), main_axis_acquired=True)


def test_musubi_inserts_one_knot_only_when_straight_route_is_blocked():
    cfg = {**get_config(15), "musubi_budget_s": 30.0}
    st = state()
    empty = (np.empty((0, 3)), np.empty((0, 3)), np.empty(0), np.empty(0), np.empty(0))
    planner = search.make_planner(cfg)
    assert PLANNERS[15] is PLANNERS["musubi"]
    direct, info = planner(st, empty, None, False)
    assert info["action"] == "musubi_direct" and info["K"] == 0
    initial = search.rollout([direct], st, empty, cfg)
    encounter = initial["path"][55, 0]
    position = encounter - 0.5 * C.accel(encounter, np.zeros(3), frame=True) * 55**2
    rocks = (position[None], np.zeros((1, 3)), np.array([9.]), np.array([1.]), np.array([0.02]))
    blocked = search.rollout([direct], st, rocks, cfg)
    assert blocked["clear"][0] < 0
    plan, info = planner(st, rocks, None, False)
    assert info["action"] == "musubi_knots" and info["K"] == 1
    assert info["clear"] >= cfg["route_clearance_m"] and info["ptot"] <= cfg["p_safe"]
    np.testing.assert_array_equal(plan[0][-1], C.T)


def test_musubi_restarts_from_straight_with_alternate_knot_locations(monkeypatch):
    cfg = {**get_config(15), "musubi_budget_s": 10.0, "musubi_max_knots": 2}
    attempts = []

    def evaluate(plans, *args):
        valid = [len(nodes) > 1 and (nodes[0] - C.S) @ C.E1 > 10.0 for nodes, _ in plans]
        n = len(plans)
        return {"T": np.full(n, 200.), "clear": np.where(valid, 10., -5.),
                "ptot": np.where(valid, 0., 0.5), "cost": np.zeros(n), "clean": np.zeros(n)}

    def alternate(plan, _, i, st, rocks, attempt, cfg):
        attempts.append((attempt, len(plan[0]) - 1))
        shift = (5.0, 15.0)[min(attempt, 1)]
        yield [C.S + (80 + 30 * len(plan[0])) * C.SH + shift * C.E1, *plan[0]], plan[1]

    monkeypatch.setattr(search, "rollout", evaluate)
    monkeypatch.setattr(search, "insertions", alternate)
    plan, info = search.make_planner(cfg)(state(), (), None, False)
    assert attempts == [(0, 0), (0, 1), (1, 0)]
    assert plan is not None and info["K"] == 1


def test_musubi_does_not_adopt_unsafe_route_when_knot_budget_is_zero(monkeypatch):
    monkeypatch.setattr(search, "rollout", lambda *args: {"T": [200.], "clear": [-3.], "ptot": [0.1]})
    cfg = {**get_config(15), "musubi_max_knots": 0}
    plan, info = search.make_planner(cfg)(state(), (), None, False)
    assert plan is None and info["action"] == "musubi_no_safe_route"
