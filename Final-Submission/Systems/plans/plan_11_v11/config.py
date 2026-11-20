"""v11 starts from the tracked v10 preset, independent of Plan 10's active config."""
from pathlib import Path

from Systems.plans.common import load_plan_config, read_config_json

_V10 = Path(__file__).resolve().parent.parent / "plan_10_real_calibrated/configs/v10.json"
_DEFAULTS = {
    **read_config_json(_V10),
    "name": "v11",
    "plan_number": 11,
    "variant": "a",
    "w_turn": 0.035,
    "ignore_per_rock": True,
    "timed_burns": False,
    "robust_samples": 0,
    "maneuver_margin": 1.0,
}
CONFIG = load_plan_config(__file__, _DEFAULTS)
