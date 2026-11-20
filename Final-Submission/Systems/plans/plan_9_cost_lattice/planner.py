"""Planner entrypoint for Plan 9 (cost_lattice)."""
from __future__ import annotations
from Systems import Core as C
from Systems.plans.common import _best, _one, _settle
from Systems.plans.plan_8_astar.graph import Graph
from Systems.plans.plan_9_cost_lattice.config import CONFIG
from Systems.plans.plan_9_cost_lattice.dp import solve_lattice_detail


def make_planner(config=None):
    def planner(st, rocks, plan, committed):
        cfg = CONFIG if config is None else config
        vcs = C.resolve_vcs(cfg)

        def path(vc):
            G = Graph(st, rocks, vc, cfg=cfg)
            cells, _, stats = solve_lattice_detail(G)
            extra = {
                "lattice_points": G.station_points(cells),
                "cells": list(cells),
                "stats": stats,
            }
            return (G.nodes(cells), vc), extra

        if plan is None:
            solved = [path(vc) for vc in vcs]
            cands = [p for p, _ in solved]
            extras = [ex for _, ex in solved]
            return _best(cands, C.score_plans(cands, st, rocks, cfg=cfg), 'lattice', extra=extras)
        _, cur = _one(plan, st, rocks, 'keep', cfg=cfg)
        new, ex = path(plan[1])
        _, ni = _one(new, st, rocks, 'lattice', cfg=cfg)
        ni.update(ex)
        return _settle(plan, cur, new, ni, committed, cfg=cfg)

    planner.__name__ = "lattice"
    return planner


lattice = make_planner()
