"""Planner entrypoint for Plan 8 (astar)."""
from __future__ import annotations
from Systems import Core as C
from Systems.plans.common import _best, _one, _settle
from Systems.plans.plan_8_astar.astar import solve_astar_detail
from Systems.plans.plan_8_astar.config import CONFIG
from Systems.plans.plan_8_astar.graph import Graph


def make_planner(config=None):
    def planner(st, rocks, plan, committed):
        cfg = CONFIG if config is None else config
        vcs = C.resolve_vcs(cfg)

        def path(vc):
            G = Graph(st, rocks, vc, cfg=cfg)
            cells, _, stats = solve_astar_detail(G)
            extra = {
                "astar_points": G.station_points(cells),
                "cells": list(cells),
                "stats": stats,
            }
            return (G.nodes(cells), vc), extra

        if plan is None:
            solved = [path(vc) for vc in vcs]
            cands = [p for p, _ in solved]
            extras = [ex for _, ex in solved]
            return _best(cands, C.score_plans(cands, st, rocks, cfg=cfg), 'astar', extra=extras)
        _, cur = _one(plan, st, rocks, 'keep', cfg=cfg)
        new, ex = path(plan[1])
        _, ni = _one(new, st, rocks, 'astar', cfg=cfg)
        ni.update(ex)
        return _settle(plan, cur, new, ni, committed, cfg=cfg)

    planner.__name__ = "astar"
    return planner


astar = make_planner()
