"""Configuration for Plan 4 (team_pipeline): Plan 7's settings plus the route-zone switches."""
from Systems.plans.common import load_plan_config
from Systems.plans.plan_7_velocity_obstacle.config import _DEFAULTS as PLAN_7

_DEFAULTS = dict(
    PLAN_7,                      # Plan 4 follows Plan 7's real-sim calibration; its zone switches are below
    name="team_pipeline",
    plan_number=4,
    # zones along the route (rocks within 40 m of it, 42 recorded real fields, FINDINGS 7.10):
    # 200-300 m is the densest stretch, 600-900 m the second, and nothing is left past 900 m
    corridor=False,              # first cluster: also offer a lattice route around it as a whole (before corridor_t)
    wide_fallback=False,         # no safe plan anywhere: also try long waits and 48-72 m detours
    late_after_m=0.0,            # clear stretch: past this many metres along the route (0: off) ...
    late_vc=0.0,                 # ... cruise at least this fast, m/s (Flight.decide)
)

CONFIG = load_plan_config(__file__, _DEFAULTS)
