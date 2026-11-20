"""Planner entrypoint for Plan 10 (real_calibrated)."""
from __future__ import annotations

from Systems.plans.plan_10_real_calibrated.calibrated import select_real_calibrated
from Systems.plans.plan_10_real_calibrated.config import CONFIG


def make_planner(config=None):
    """Build a Plan 10 real_calibrated planner bound to ``config``."""
    cfg = CONFIG if config is None else config

    def plan10(st, rocks, plan, committed):
        return select_real_calibrated(st, rocks, plan, committed, cfg=cfg)

    return plan10


real_calibrated = make_planner()
