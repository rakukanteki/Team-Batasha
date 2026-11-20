"""
Systems/CostSearch.py - facade over Plans 8 (astar) and 9 (cost_lattice) in Systems/plans/.
Spec: docs/plans/PLAN_8_9_COST_SEARCH.md. Pure numpy/scipy, no sim calls.
"""
from __future__ import annotations

from Systems.plans.plan_8_astar import (  # noqa: F401
    C_V,
    DS,
    EPS_TURN,
    GOAL,
    GRID,
    NCELL,
    YY,
    Graph,
    astar,
    solve_astar,
    solve_astar_detail,
)
from Systems.plans.plan_9_cost_lattice import (  # noqa: F401
    lattice,
    solve_lattice,
    solve_lattice_detail,
)
