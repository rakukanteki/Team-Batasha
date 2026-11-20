"""Tests for Systems.ScoreMath exact optimization & GUI Max Score Math endpoints."""
from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from Systems import Core
from Systems.ScoreMath import (
    MathParams,
    _eval_nograv_at_v,
    analytical_bang_coast,
    analyze_max_score,
    optimize_gravity_ode,
    simulate_guidance_batch,
)
from tools.plan_runner_gui import app


def test_analytical_bang_coast_optimality_and_bounds():
    p_direct = MathParams(a_burn=0.130, eta_l1=1.017)
    res = analytical_bang_coast(p_direct)
    v_star = res["v_cruise"]
    # First-order optimality dS/dv == 0 at interior optimum
    left = _eval_nograv_at_v(v_star - 0.01, p_direct)["score"]
    right = _eval_nograv_at_v(v_star + 0.01, p_direct)["score"]
    assert res["score"] >= left - 1e-3
    assert res["score"] >= right - 1e-3
    assert abs(res["score"] - 67.184) < 0.05

    # Body +x axis bound (a=0.175, eta=1.0) and unconstrained SetCommand ceiling (0.500, eta=1.0)
    px = analytical_bang_coast(MathParams(a_burn=0.175, eta_l1=1.0))
    unc = analytical_bang_coast(MathParams(a_burn=0.500, eta_l1=1.0))
    assert abs(px["score"] - 70.78) < 0.1
    assert abs(unc["score"] - 85.13) < 0.1


def test_gravity_ode_optimization_and_penalties():
    res_direct, _ = optimize_gravity_ode(MathParams(a_burn=0.130, eta_l1=1.017))
    assert res_direct["arrived"] is True
    assert abs(res_direct["score"] - 58.958) < 0.1

    res_px, _ = optimize_gravity_ode(MathParams(a_burn=0.175, eta_l1=1.0))
    assert res_px["arrived"] is True
    assert abs(res_px["score"] - 63.89) < 0.1

    # Hold delay strictly reduces max achievable score
    res_hold, _ = optimize_gravity_ode(MathParams(a_burn=0.130, eta_l1=1.017, hold_t_s=15.0))
    assert res_hold["arrived"] is True
    assert res_hold["score"] < res_direct["score"] - 5.0

    # Damage deducts exact 200 * D / Dmax without altering trajectory
    d_kj = 20.92  # 10% of Dmax = 209.2 kJ => exact -20.0 pts
    res_dmg, _ = optimize_gravity_ode(MathParams(a_burn=0.130, eta_l1=1.017, damage_kJ=d_kj))
    assert res_dmg["arrived"] is True
    assert abs((res_direct["score"] - res_dmg["score"]) - 20.0) < 1e-2


def test_guidance_batch_matches_core_score_plans_and_preserves_globals():
    vcs = [6.5, 7.9, 8.4, 8.9, 10.5]
    orig_frame = Core.FRAME_B.copy()
    orig_omega = Core.OMEGA.copy()
    plans = [([Core.T], v) for v in vcs]
    st = dict(r=Core.S, v=np.zeros(3), burning=False, main=False, fuel=0.0, t=0.0)
    rocks = (
        np.empty((0, 3)),
        np.empty((0, 3)),
        np.empty(0),
        np.empty(0),
        np.empty(0),
    )

    for frame_accel in (False, True):
        c_res = Core.score_plans(plans, st, rocks, cfg={"frame_accel": frame_accel})
        m_res = simulate_guidance_batch(
            vcs, MathParams(a_burn=float(Core.A_BURN), eta_l1=1.0, frame_accel=frame_accel)
        )
        assert np.all(m_res["arrived"])
        max_score_diff = float(np.max(np.abs(m_res["score"] - c_res["clean"])))
        max_time_diff = float(np.max(np.abs(m_res["T_flight_s"] - c_res["T"])))
        max_fuel_diff = float(np.max(np.abs(m_res["fuel_kg"] - c_res["fuel"])))
        assert max_score_diff < 1e-10
        assert max_time_diff < 1e-10
        assert max_fuel_diff < 1e-10

    # Verify ScoreMath analysis does not mutate Core globals
    analysis = analyze_max_score({"a_burn": 0.175, "eta_l1": 1.0, "hold_t_s": 5.0, "frame_accel": True})
    assert analysis["guidance"]["arrived"] is True
    assert len(analysis["presets_comparison"]) >= 6
    assert np.allclose(Core.FRAME_B, orig_frame)
    assert np.allclose(Core.OMEGA, orig_omega)


def test_gui_math_endpoints_and_html():
    client = TestClient(app)

    r_html = client.get("/")
    assert r_html.status_code == 200
    assert "tabMath" in r_html.text
    assert "Max Score Math" in r_html.text

    r_def = client.get("/api/math/defaults")
    assert r_def.status_code == 200
    d_body = r_def.json()
    assert "defaults" in d_body
    assert "presets" in d_body
    assert "direct" in d_body["presets"]

    r_calc = client.post(
        "/api/math/max_score",
        json={"params": {"a_burn": 0.175, "eta_l1": 1.0, "hold_t_s": 5.0, "detour_extra_m": 25.0}},
    )
    assert r_calc.status_code == 200
    body = r_calc.json()
    for key in (
        "params",
        "analytical",
        "gravity_ode",
        "guidance",
        "configured_vcs",
        "loss_attribution",
        "sensitivities",
        "curves",
        "presets_comparison",
    ):
        assert key in body
    assert body["analytical"]["score"] > body["gravity_ode"]["score"]
    assert body["guidance"]["arrived"] is True
