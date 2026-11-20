"""Configuration for Plan 3 (lattice)."""
from Systems import Core as C

from Systems.plans.common import load_plan_config

_DEFAULTS = {
    "name": "lattice",
    "plan_number": 3,
    # thrust: the same keys in every plan
    "max_thrust": 0.5,           # m/s^2 every burn commands (SetCommand); 0.5 is the organizers' limit
    "a_burn": None,              # m/s^2 the plan assumes at the start; None: the burn law's own figure
    "a_burn_min": 0.06,          # bounds on the burn acceleration the controller learns in flight
    "a_burn_max": 0.20,
    # guard: the last-resort dodge, the same keys in every plan
    "guard_horizon_s": 20.0,     # s ahead a predicted hit triggers the dodge
    "guard_margin_m": 1.0,       # m kept beyond the rock's hard radius ... (fires only for a real hit:
    "guard_sigma": 0.0,          # ... plus this many sigma; wider margins fought the planner, FINDINGS 6.14)
    "guard_max_sigma_m": 6.0,    # m: a rock known worse than this is left to the planner (velocities still guesses)
    "guard_release_m": 5.0,      # m further out before the dodge lets go
    "replan_risk_p": None,       # replan at once while the plan's hit probability is above this (None: never, v5)
    "risk_weight": 1.0,          # expected hit loss charged this many times over (v5: 1; FINDINGS 6.14: 3)
    "rel_hit": False,            # hit loss at the speed relative to the rock (nayeem: -2.4 on the replay bench)
    "guard_mode": "v5",          # last-resort guard: "v5" (nayeem, 12 s, push away) or "escape" (FINDINGS 6.14)
    "tracker": "calibrated",
    "burn": "steer",
    "accel_till_speed": float(C.VCS[1]),
    "accel_dur_s": 0.0,
    "hold_t": 12.0,
    "main_min": 2.0,
    "trim_exact": True,
    "frame_accel": False,
    "r_sh": float(C.R_SH),
    "guard_plan": False,
    "vcs": C.VCS,
    "ds": 60.0,
    "grid_half_cells": 4,
    "grid_step_m": 8.0,
    "nq": 6,
    "w_dev": 0.02,
    "goal_radius_m": 40.0,
    "merge_threshold": 0.5,
    "near_margin_m": 45.0,
    "near_lat_base_m": 46.0,
    "safety_sigma": 3.0,
    "safety_margin_m": 3.0,
    "burn_turn_weight": 0.15,
    "time_weight_num": 12000.0,
    "slope_eps": 1e-6,
    "simplify_margin": 0.5,
    "switch_margin": C.SWITCH,
    "p_safe": C.P_SAFE,
    "sig_ignore": C.SIG_IGNORE,
    "t_ignore": C.T_IGNORE,
    "lambda_node": C.LAMBDA_NODE,
    "dt_plan": C.DT_PLAN,
    "tmax": 420.0,
}

CONFIG = load_plan_config(__file__, _DEFAULTS)
