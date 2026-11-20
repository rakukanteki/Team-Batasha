"""Configuration for Plan 7 (velocity_obstacle)."""
from Systems import Core as C

from Systems.plans.common import load_plan_config

_DEFAULTS = {
    "name": "velocity_obstacle",
    "plan_number": 7,
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
    "risk_weight": 3.0,          # expected hit loss charged this many times over (v5: 1; replay bench +2.2, 0 hits)
    "rel_hit": False,            # hit loss at the speed relative to the rock (nayeem: -2.4 on the replay bench)
    "guard_mode": "v5",          # last-resort guard: "v5" (nayeem, 12 s, push away) or "escape" (FINDINGS 6.14)
    "vcs": C.VCS,
    "default_vc": C.VCS[1],
    "holds": (4.0, 8.0, 12.0, 18.0),  # s a plan may wait before burning on (v5: none; replay bench +3.0, 0 hits)
    "speeds": False,             # also aim at the other cruise speeds
    "nudge": True,               # v5: 'carry on along v, then turn' and the flown plan nudged 4/8/16 m
    "la_mults": (1.0, 2.0, 3.0),  # v5: side points at these multiples of the lookahead (earlier dodges are cheaper)
    "keep_band": 3.0,            # v5: the flown plan stays 'safe' up to this many times p_safe (hysteresis)
    "selection": "p_safe",       # v5: cheapest among plans under p_safe; "cost": cheapest in expected score
    "wide_fallback": False,      # when the best still risks a hit, also try long waits and 48-72 m detours
    "corridor": False,           # nayeem's cluster fix: also offer Plan 3's lattice route over a wide grid
    "corridor_step_m": 16.0,     # m between lattice cells (9 x 9: +-64 m)
    "corridor_t": 80.0,          # s: offered only before this (later it never changed the flight)
    "corridor_pad_m": 0.0,       # m added to every rock in the corridor search
    "tracker": "calibrated",
    "burn": "steer",
    "accel_till_speed": float(C.VCS[1]),
    "accel_dur_s": 0.0,
    "hold_t": 12.0,
    "main_min": 2.0,
    "trim_exact": True,
    "frame_accel": True,         # v5: rotating-frame physics (FINDINGS 7.2); tested with holds + risk 3
    "r_sh": float(C.R_SH),
    "guard_plan": False,
    "holds_wide": (26.0, 36.0, 48.0),
    "p_wide": 0.03,
    "lookahead_min_m": 60.0,
    "lookahead_time_s": 15.0,
    "lookahead_min_v": 2.0,
    "stop_ahead_margin_m": 80.0,
    "side_offsets": (4.0, 8.0, 16.0, 32.0),
    "wide_side_offsets": (48.0, 72.0),
    "keep_committed_bonus": 0.5,     # "cost" selection only
    "switch_margin": C.SWITCH,
    "p_safe": C.P_SAFE,
    "sig_ignore": C.SIG_IGNORE,
    "t_ignore": C.T_IGNORE,
    "lambda_node": C.LAMBDA_NODE,
    "dt_plan": C.DT_PLAN,
    "tmax": 420.0,
}

CONFIG = load_plan_config(__file__, _DEFAULTS)
