"""
Systems/Planners.py - facade & registry over the per-plan modules in Systems/plans/.
Specs: docs/plans/PLAN_1..9. Pure numpy/scipy, no sim calls.

Every planner has the same signature and can be swapped freely:
    planner(st, rocks, plan, committed) -> (plan, info)
"""
from __future__ import annotations

from Systems import CostSearch  # noqa: F401
from Systems.plans import (  # noqa: F401
    PLAN_CONFIGS,
    PLAN_NAMES,
    PLANNERS,
    configure_plan,
    get_config,
    plan_name,
)
from Systems.plans.common import (  # noqa: F401
    _beside,
    _best,
    _early,
    _insert,
    _one,
    _reference,
    _settle,
    _start_slope,
    _to_nodes,
)
from Systems.plans.plan_6_barrier import barrier, plan6  # noqa: F401
from Systems.plans.plan_10_real_calibrated import (  # noqa: F401
    KEEP_BAND,
    LA_MULTS,
    NUDGE,
    SIDE_M,
    real_calibrated,
    real_calibrated as plan10,
)
