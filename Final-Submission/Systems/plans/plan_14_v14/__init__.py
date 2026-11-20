from Systems.plans.plan_13_v13.planner import make_planner as make_route_planner
from Systems.plans.plan_14_v14 import config


def make_planner(overrides=None):
    planner = make_route_planner(config.CONFIG if overrides is None else overrides)
    planner.__name__ = "v14"
    return planner


v14 = make_planner()
