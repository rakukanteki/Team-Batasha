"""Planner entrypoint for Plan 6 (barrier)."""
from __future__ import annotations
from Systems import Core as C
from Systems.plans.common import _one
from Systems.plans.plan_6_barrier.config import CONFIG


def make_planner(config=None):
    def planner(st, rocks, plan, committed):
        cfg = CONFIG if config is None else config
        planner.barrier_alpha = cfg.get("barrier_alpha", 0.4)
        vc = C.resolve_vc(cfg)
        plan = ([C.T.copy()], vc) if plan is None else plan
        return plan, _one(plan, st, rocks, 'keep', cfg=cfg)[1]

    planner.barrier_alpha = (CONFIG if config is None else config).get("barrier_alpha", 0.4)
    planner.__name__ = "plan6"
    return planner


plan6 = make_planner()
