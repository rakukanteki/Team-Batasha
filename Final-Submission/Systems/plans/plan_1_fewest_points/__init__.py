"""Plan 1: fewest intermediate points."""
from Systems.plans.plan_1_fewest_points.config import CONFIG
from Systems.plans.plan_1_fewest_points.planner import make_planner, plan1
from Systems.plans.plan_1_fewest_points.search import repair, search

__all__ = ["CONFIG", "make_planner", "plan1", "repair", "search"]
