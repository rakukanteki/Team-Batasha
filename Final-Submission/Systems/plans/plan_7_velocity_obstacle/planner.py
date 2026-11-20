"""Planner entrypoint for Plan 7 (velocity_obstacle)."""
from __future__ import annotations
from Systems.plans.plan_7_velocity_obstacle.config import CONFIG
from Systems.plans.plan_7_velocity_obstacle.vo import select_velocity_obstacle


def make_planner(config=None):
    def planner(st, rocks, plan, committed):
        cfg = CONFIG if config is None else config
        return select_velocity_obstacle(st, rocks, plan, committed, cfg=cfg)

    planner.__name__ = "plan7"
    return planner


plan7 = make_planner()
