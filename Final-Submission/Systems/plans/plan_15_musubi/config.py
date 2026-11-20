"""Musubi uses V14's navigation and body +X policy."""
from Systems.plans.common import load_plan_config
from Systems.plans.plan_14_v14.config import _DEFAULTS as V14_DEFAULTS

_DEFAULTS = {
    **V14_DEFAULTS,
    "name": "musubi",
    "plan_number": 15,
    "musubi_max_knots": 3,
    "musubi_beam_width": 2,
    "musubi_restarts": 2,
    "musubi_candidate_budget": 42,
    "musubi_batch_size": 8,
    "musubi_budget_s": 3.5,
}
CONFIG = load_plan_config(__file__, _DEFAULTS)
