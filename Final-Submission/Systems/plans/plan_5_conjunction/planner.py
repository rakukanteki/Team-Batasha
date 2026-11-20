"""Planner entrypoint for Plan 5 (conjunction)."""
from __future__ import annotations
from Systems import Core as C
from Systems.plans.common import _one, _settle
from Systems.plans.plan_1_fewest_points.search import repair, search
from Systems.plans.plan_5_conjunction.avoidance import _push
from Systems.plans.plan_5_conjunction.config import CONFIG


def make_planner(config=None):
    def planner(st, rocks, plan, committed):
        cfg = CONFIG if config is None else config
        vcs = C.resolve_vcs(cfg)
        p_safe = cfg.get("p_safe", C.P_SAFE)
        drop_margin = cfg.get("drop_margin", 0.2)
        max_pts = cfg.get("max_points", 3)
        max_cost_delta = cfg.get("max_cost_delta", 3.0)

        if plan is None:
            return search(st, rocks, vcs, kmax=0, cfg=cfg)
        _, cur = _one(plan, st, rocks, 'keep', cfg=cfg)
        if cur['ptot'] <= p_safe:                       # shed points that are no longer needed: all at once, or one
            fewer = [list(plan[0][-1:])] if len(plan[0]) > 2 else []
            fewer += [plan[0][:j] + plan[0][j + 1:] for j in range(len(plan[0]) - 1)]
            for nd in fewer:
                _, d = _one((list(nd), plan[1]), st, rocks, 'drop', cfg=cfg)
                if d['ptot'] <= p_safe and d['cost'] < cur['cost'] - drop_margin:
                    return (list(nd), plan[1]), d
            return plan, cur
        new, ni = _push(plan, st, rocks, cfg=cfg)
        if ni['ptot'] > p_safe or ni['K'] > max_pts or ni['cost'] > -cur['clean'] + max_cost_delta:
            # the single-rock push failed, piled up points, or is expensive: search all sides instead
            base = (list(plan[0][-1:]), plan[1]) if len(plan[0]) - 1 > max_pts else plan
            alt, ai = repair(st, rocks, base, cfg=cfg)
            if ai['ptot'] > p_safe:
                alt, ai = search(st, rocks, (plan[1],), cfg=cfg)
            if (ai['ptot'] > p_safe, ai['cost']) < (ni['ptot'] > p_safe, ni['cost']):
                new, ni = alt, ai
        return _settle(plan, cur, new, ni, committed, cfg=cfg)

    planner.__name__ = "plan5"
    return planner


plan5 = make_planner()
