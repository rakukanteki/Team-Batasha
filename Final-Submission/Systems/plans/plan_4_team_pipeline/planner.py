"""Planner entrypoint for Plan 4 (team_pipeline)."""
from __future__ import annotations
from Systems.plans.plan_4_team_pipeline.config import CONFIG
from Systems.plans.plan_4_team_pipeline.pipeline import run_pipeline


def make_planner(config=None):
    def planner(st, rocks, plan, committed):
        cfg = CONFIG if config is None else config
        return run_pipeline(st, rocks, plan, committed, cfg)

    planner.__name__ = "plan4"
    return planner


plan4 = make_planner()
