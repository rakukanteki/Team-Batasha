"""V14 uses the V13 route policy with body +X toward docking."""
from Systems.plans.common import load_plan_config
from Systems.plans.plan_13_v13.config import _DEFAULTS as V13_DEFAULTS

_DEFAULTS = {
    **V13_DEFAULTS,
    "name": "v14",
    "plan_number": 14,
    "burn": "x_forward",
    "pointing_axis": 0,
    "x_forward_acquire_deg": 5.0,
    "x_forward_rate_rad_s": 0.018,
    "x_forward_cone_deg": 10.0,
    "launch_clearance_m": 10.0,
    "launch_climb_mps": 0.85,
    "launch_early_slew_m": 5.0,
    "launch_early_slew_min_climb_mps": 0.60,
    "launch_early_slew_rate_rad_s": 0.025,
    "launch_slew_min_torque": 0.002,
    "launch_slew_cruise_torque": 0.006,
    "launch_slew_limit_s": 32.0,
    "launch_deadline_s": 70.0,
    "x_forward_precision_dv_mps": 0.30,
    "x_forward_precision_settle_s": 1.25,
    "r_sh": 6.0,  # interim orientation-independent buffer; not a measured hull dimension
    "accel_till_speed": 10.0,
    "vcs": [10.0],
}
CONFIG = load_plan_config(__file__, _DEFAULTS)
