"""Planner entrypoint for Plan 3 (lattice)."""
from __future__ import annotations
from Systems import Core as C
from Systems.plans.common import _best, _one, _settle
from Systems.plans.plan_3_lattice.config import CONFIG
from Systems.plans.plan_3_lattice.lattice import _lattice_detail, _previous


def make_planner(config=None):
    def planner(st, rocks, plan, committed):
        cfg = CONFIG if config is None else config
        vcs = C.resolve_vcs(cfg)
        if plan is None:
            solved = [_lattice_detail(st, rocks, vc, cfg=cfg) for vc in vcs]
            cands = [(nd, vc) for (nd, _), vc in zip(solved, vcs)]
            extras = [ex for _, ex in solved]
            return _best(cands, C.score_plans(cands, st, rocks, cfg=cfg), 'search', extra=extras)
        _, cur = _one(plan, st, rocks, 'keep', cfg=cfg)
        nd, ex = _lattice_detail(st, rocks, plan[1], _previous(plan, st) if committed else None, cfg=cfg)
        new = (nd, plan[1])
        _, ni = _one(new, st, rocks, 'search', cfg=cfg)
        ni.update(ex)
        return _settle(plan, cur, new, ni, committed, cfg=cfg)

    planner.__name__ = "plan3"
    return planner


plan3 = make_planner()
