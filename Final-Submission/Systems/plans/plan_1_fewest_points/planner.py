"""Planner entrypoint for Plan 1 (fewest_points)."""
from __future__ import annotations
from Systems.plans.common import _one, _settle
from Systems.plans.plan_1_fewest_points.config import CONFIG
from Systems.plans.plan_1_fewest_points.search import repair, search


def make_planner(config=None):
    def planner(st, rocks, plan, committed):
        cfg = CONFIG if config is None else config
        if plan is None:
            return search(st, rocks, cfg=cfg)
        if not committed:
            new, ni = search(st, rocks, (plan[1],), cfg=cfg)   # speed parameter fixed by first plan
            _, cur = _one(plan, st, rocks, 'keep', cfg=cfg)
            return _settle(plan, cur, new, ni, committed, cfg=cfg)
        return repair(st, rocks, plan, cfg=cfg)

    planner.__name__ = "plan1"
    return planner


plan1 = make_planner()
