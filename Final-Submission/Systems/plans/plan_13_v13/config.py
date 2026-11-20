"""V13 defaults for the official final scene."""
from Systems.plans.common import load_plan_config
from Systems.plans.plan_9_cost_lattice.config import _DEFAULTS as LATTICE_DEFAULTS

_DEFAULTS = {
    **LATTICE_DEFAULTS,
    "name": "v13",
    "plan_number": 13,
    "burn": "target_z",
    "camera": "off",
    "coast_pointing_deadband": True,
    "coast_heading_enter_deg": 5.0,
    "coast_heading_release_deg": 2.0,
    "coast_rate_enter_rad_s": 0.018,
    "coast_rate_release_rad_s": 0.007,
    "coast_torque_fuel_slope": 300.0,   # kg/s per rad/s² commanded, fit to selector samples
    "coast_torque_fuel_cap_kg_s": 0.50,
    "frame_accel": True,
    "predict_attitude": True,
    "hold_t": 0.0,
    "route_handoff_s": 14.0,
    "route_expected_tracks": 500,
    "route_min_samples": 3,
    "route_max_sigma_m": 12.0,
    "route_max_track_age_s": 9.0,
    "route_max_age_s": 5.0,
    "route_clearance_m": 3.0,
    "route_candidate_limit": 24,
    "guard_plan": "union",
    "ds": 100.0,
    "grid_half_cells": 2,
    "grid_step_m": 20.0,
    "dt_plan": 1.0,                 # C.evaluate sweeps continuously between steps
    "t_ignore": 45.0,               # retain nominal clearance even past the confidence horizon
    "sig_ignore": 12.0,
    "rel_hit": True,
}
CONFIG = load_plan_config(__file__, _DEFAULTS)
