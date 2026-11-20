"""Planner entrypoint for Plan 2 (linear_program)."""
from __future__ import annotations
from Systems import Core as C
from Systems.plans.common import _best, _one, _settle
from Systems.plans.plan_2_linear_program.config import CONFIG
from Systems.plans.plan_2_linear_program.lp import _lp_solve_detail


def make_planner(config=None):
    def planner(st, rocks, plan, committed):
        cfg = CONFIG if config is None else config
        vcs = C.resolve_vcs(cfg)
        if plan is None:
            solved = [_lp_solve_detail(st, rocks, vc, ([C.T.copy()], vc), cfg=cfg) for vc in vcs]
            cands = [p for p, _ in solved]
            extras = [ex for _, ex in solved]
            return _best(cands, C.score_plans(cands, st, rocks, cfg=cfg), 'search', extra=extras)
        _, cur = _one(plan, st, rocks, 'keep', cfg=cfg)
        new, ex = _lp_solve_detail(st, rocks, plan[1], plan, cfg=cfg)
        _, ni = _one(new, st, rocks, 'search', cfg=cfg)
        ni.update(ex)
        return _settle(plan, cur, new, ni, committed, cfg=cfg)

    planner.__name__ = "plan2"
    return planner


plan2 = make_planner()
