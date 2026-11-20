"""Default configuration for Plan 10 (real-calibrated Nayeem v5 stack)."""
from __future__ import annotations

from Systems import Core as C
from Systems.plans.common import load_plan_config

_DEFAULTS = {
    "name": "real_calibrated",
    "plan_number": 10,
    "tracker": "calibrated",
    "burn": "steer",
    "accel_till_speed": float(C.VCS[1] if len(C.VCS) > 1 else C.VCS[0]),
    "accel_dur_s": 0.0,
    "hold_t": 12.0,
    "main_min": 2.0,
    "trim_exact": True,
    "frame_accel": True,
    "rel_hit": False,
    "r_sh": 4.5,
    "guard_plan": False,
    "vcs": list(C.VCS),
    "la_mults": [1.0, 2.0, 3.0, 4.0, 5.0],  # v10: earlier side dodges (nayeem 85d512e)
    "side_offsets": [4.0, 8.0, 16.0, 32.0],
    "drift_offsets": [4.0, 8.0],
    "nudge_offsets": [4.0, 8.0, 16.0],
    "nudge": True,
    "keep_band": 3.0,
    "node_ahead_m": 60.0,
    "nudge_target_max_offset_m": 45.0,
    "lookahead_min_m": 60.0,
    "lookahead_time_s": 15.0,
    "lookahead_min_v": 2.0,
    "stop_ahead_margin_m": 80.0,
    "switch_margin": float(C.SWITCH),
    "p_safe": float(C.P_SAFE),
    "sig_ignore": float(C.SIG_IGNORE),
    "t_ignore": float(C.T_IGNORE),
    "lambda_node": float(C.LAMBDA_NODE),
    "dt_plan": float(C.DT_PLAN),
    "tmax": 420.0,
    "q_vel": 0.02 ** 2,
    "v0_var": 3.0 ** 2,
    "q_burst": 1.0 ** 2,
    "burst_t": 12.0,
    "adapt_gate": 16.0,
    "rr_var": 0.02 ** 2,
}

CONFIG = load_plan_config(__file__, _DEFAULTS)
