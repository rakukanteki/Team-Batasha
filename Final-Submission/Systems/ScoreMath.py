"""
Systems/ScoreMath.py - Exact mathematical optimization & sensitivity analysis for Escape from Phobos max score.

Pure numpy/scipy (no sim calls). Computes:
  1. Closed-form analytical bang-coast max score (zero-gravity bound + optimality condition).
  2. Exact 3D Phobos-gravity (+ optional rotating-frame Coriolis/bias) burn-coast trajectory optimization.
  3. Guidance-loop (Core.fly equivalent) optimization over coast speed vc without mutating Core globals.
  4. Score attribution breakdown, local sensitivities at optimum, actuator-axis comparison table,
     and Speed-vs-Score / Time-vs-Fuel curves for the interactive Web GUI.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from Systems import Core as C
from Systems.UserAPI import compute_competition_score


ACTUATOR_PRESETS: dict[str, dict[str, Any]] = {
    "direct": {
        "label": "Direct burn (1° +x tilt, default)",
        "a_burn": 0.130,
        "eta_l1": 1.017,
        "slew_fuel_kg": 0.0,
    },
    "boost": {
        "label": "Body +z torque-shaped (boost)",
        "a_burn": 0.135,
        "eta_l1": 1.000,
        "slew_fuel_kg": 0.0,
    },
    "steer": {
        "label": "Steer burn (pointing-gated)",
        "a_burn": 0.125,
        "eta_l1": 1.017,
        "slew_fuel_kg": 0.0,
    },
    "body_px": {
        "label": "Body +x zero-torque (strongest axis)",
        "a_burn": 0.175,
        "eta_l1": 1.000,
        "slew_fuel_kg": 0.0,
    },
    "body_py": {
        "label": "Body +y zero-torque",
        "a_burn": 0.143,
        "eta_l1": 1.000,
        "slew_fuel_kg": 0.0,
    },
    "body_mz": {
        "label": "Body -z zero-torque",
        "a_burn": 0.117,
        "eta_l1": 1.000,
        "slew_fuel_kg": 0.0,
    },
    "plain": {
        "label": "Body +z zero-torque (plain)",
        "a_burn": 0.096,
        "eta_l1": 1.000,
        "slew_fuel_kg": 0.0,
    },
    "unconstrained": {
        "label": "Unconstrained SetCommand ceiling (0.50 m/s²)",
        "a_burn": 0.500,
        "eta_l1": 1.000,
        "slew_fuel_kg": 0.0,
    },
}


@dataclass
class MathParams:
    # Propulsion & actuator
    a_burn: float = 0.130
    eta_l1: float = 1.017
    mass_kg: float = float(C.MASS)
    isp_s: float = 80.0
    g0_mps2: float = 9.8

    # Environment & mission geometry
    mu: float = 7.075e5
    frame_accel: bool = False
    distance_m: float = float(C.L)
    win_radius_m: float = float(C.WIN)
    v0_along_mps: float = 0.0

    # Operational delays, detours & penalties
    hold_t_s: float = 0.0
    clock_offset_s: float = 0.0
    extra_dv_mps: float = 0.0
    slew_fuel_kg: float = 0.0
    detour_extra_m: float = 0.0
    damage_kJ: float = 0.0

    # Guidance & scoring constants
    a_trim: float = float(C.A_TRIM)
    trim_l1: float = float(C.TRIM_L1)
    main_min: float = float(C.MAIN_MIN)
    eps_on: float = float(C.EPS_ON)
    terminal_m: float = float(C.TERMINAL)
    dt_plan: float = float(C.DT_PLAN)
    tmax: float = 420.0
    nominal_time_s: float = 120.0
    nominal_fuel_kg: float = 75.0
    max_damage_kJ: float = float(C.DMAX / 1000.0)
    vcs: list[float] = field(default_factory=lambda: [float(x) for x in C.VCS])

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "MathParams":
        defaults = cls()
        if not data:
            return defaults
        kwargs: dict[str, Any] = {}
        for f_name, f_val in asdict(defaults).items():
            if f_name not in data or data[f_name] is None:
                kwargs[f_name] = f_val
                continue
            raw = data[f_name]
            if f_name == "vcs":
                if isinstance(raw, str):
                    vals = [float(x.strip()) for x in raw.strip("[]").split(",") if x.strip()]
                else:
                    vals = [float(x) for x in raw]
                kwargs[f_name] = [v for v in vals if np.isfinite(v) and v > 0] or f_val
            elif isinstance(f_val, bool):
                kwargs[f_name] = bool(raw)
            else:
                kwargs[f_name] = float(raw)
        # Clamp physically positive quantities to safe bounds
        kwargs["a_burn"] = max(kwargs["a_burn"], 1e-3)
        kwargs["eta_l1"] = max(kwargs["eta_l1"], 1.0)
        kwargs["mass_kg"] = max(kwargs["mass_kg"], 1.0)
        kwargs["isp_s"] = max(kwargs["isp_s"], 1.0)
        kwargs["g0_mps2"] = max(kwargs["g0_mps2"], 0.1)
        kwargs["mu"] = max(kwargs["mu"], 0.0)
        kwargs["distance_m"] = max(kwargs["distance_m"], 10.0)
        kwargs["win_radius_m"] = float(np.clip(kwargs["win_radius_m"], 1.0, kwargs["distance_m"] - 1.0))
        kwargs["hold_t_s"] = max(kwargs["hold_t_s"], 0.0)
        kwargs["extra_dv_mps"] = max(kwargs["extra_dv_mps"], 0.0)
        kwargs["slew_fuel_kg"] = max(kwargs["slew_fuel_kg"], 0.0)
        kwargs["detour_extra_m"] = max(kwargs["detour_extra_m"], 0.0)
        kwargs["damage_kJ"] = max(kwargs["damage_kJ"], 0.0)
        kwargs["a_trim"] = max(kwargs["a_trim"], 1e-3)
        kwargs["trim_l1"] = max(kwargs["trim_l1"], 1.0)
        kwargs["main_min"] = max(kwargs["main_min"], 0.0)
        kwargs["eps_on"] = max(kwargs["eps_on"], 1e-4)
        kwargs["terminal_m"] = max(kwargs["terminal_m"], 0.0)
        kwargs["dt_plan"] = float(np.clip(kwargs["dt_plan"], 0.05, 2.0))
        kwargs["tmax"] = float(np.clip(kwargs["tmax"], 30.0, 1200.0))
        kwargs["nominal_time_s"] = max(kwargs["nominal_time_s"], 1.0)
        kwargs["nominal_fuel_kg"] = max(kwargs["nominal_fuel_kg"], 1.0)
        kwargs["max_damage_kJ"] = max(kwargs["max_damage_kJ"], 1.0)
        return cls(**kwargs)

    @property
    def kf(self) -> float:
        return self.mass_kg / (self.isp_s * self.g0_mps2)

    @property
    def effective_distance_m(self) -> float:
        return max(self.distance_m - self.win_radius_m + self.detour_extra_m, 1.0)


def score_from_metrics(
    time_s: float,
    fuel_kg: float,
    damage_kJ: float = 0.0,
    nominal_time_s: float = 120.0,
    nominal_fuel_kg: float = 75.0,
    max_damage_kJ: float = 209.2,
) -> dict[str, float]:
    """Evaluate official scoring equation and return component terms."""
    bd = compute_competition_score(
        actual_time=max(float(time_s), 1e-3),
        actual_damage=max(float(damage_kJ), 0.0) * 1000.0,
        actual_fuel=max(float(fuel_kg), 0.0),
        nominal_time=nominal_time_s,
        max_damage=max_damage_kJ * 1000.0,
        nominal_fuel=nominal_fuel_kg,
    )
    return {
        "base_term": 50.0,
        "time_term": round(100.0 * bd.time_score, 4),
        "fuel_term": round(-100.0 * bd.fuel_penalty, 4),
        "damage_term": round(-100.0 * bd.damage_penalty, 4),
        "raw_score": round(bd.raw_score, 6),
        "final_score": round(bd.final_score, 4),
    }


def _eval_nograv_at_v(v: float, p: MathParams) -> dict[str, float]:
    d = p.effective_distance_m
    a = p.a_burn
    v0 = p.v0_along_mps
    kf = p.kf
    x_hold = v0 * p.hold_t_s
    if v0 > 1e-6 and x_hold >= d:
        t_flight = d / v0
        dv_main = 0.0
        t_burn = 0.0
    else:
        d_rem = d - x_hold
        v_max = float(np.sqrt(max(v0 * v0 + 2.0 * a * d_rem, 0.0)))
        v_eff = min(max(v, v0), v_max)
        if v_eff <= v0:
            dv_main = 0.0
            t_burn = 0.0
            t_flight = p.hold_t_s + (d_rem / v0 if v0 > 1e-6 else 1e6)
        else:
            dv_main = v_eff - v0
            t_burn = dv_main / a
            x_burn = v0 * t_burn + 0.5 * a * t_burn * t_burn
            t_coast = max(d_rem - x_burn, 0.0) / max(v_eff, 1e-6)
            t_flight = p.hold_t_s + t_burn + t_coast
    t_score = max(t_flight + p.clock_offset_s, 1e-3)
    dv_total = dv_main + p.extra_dv_mps
    fuel_kg = kf * (p.eta_l1 * dv_main + p.extra_dv_mps) + p.slew_fuel_kg
    sc = score_from_metrics(
        t_score, fuel_kg, p.damage_kJ,
        p.nominal_time_s, p.nominal_fuel_kg, p.max_damage_kJ,
    )
    return {
        "v_cruise": round(float(v), 4),
        "t_burn_s": round(float(t_burn), 3),
        "T_flight_s": round(float(t_flight), 3),
        "T_score_s": round(float(t_score), 3),
        "dv_main_mps": round(float(dv_main), 4),
        "dv_total_mps": round(float(dv_total), 4),
        "fuel_kg": round(float(fuel_kg), 4),
        "score": round(float(sc["final_score"]), 3),
        "breakdown": sc,
    }


def analytical_bang_coast(params: MathParams) -> dict[str, Any]:
    """Closed-form analytical zero-gravity bang-coast maximum score."""
    d = params.effective_distance_m
    v0 = params.v0_along_mps
    x_hold = v0 * params.hold_t_s
    d_rem = max(d - x_hold, 0.0)
    v_max = float(np.sqrt(max(v0 * v0 + 2.0 * params.a_burn * d_rem, 1e-6)))
    v_lo = max(v0, 0.5)
    if v_max <= v_lo + 1e-6:
        best = _eval_nograv_at_v(v_lo, params)
    else:
        res = minimize_scalar(
            lambda vv: -_eval_nograv_at_v(vv, params)["score"],
            bounds=(v_lo, v_max),
            method="bounded",
            options={"xatol": 1e-4},
        )
        candidates = [
            _eval_nograv_at_v(float(res.x), params),
            _eval_nograv_at_v(v_max, params),
        ]
        if v0 > 0.5:
            candidates.append(_eval_nograv_at_v(v0, params))
        best = max(candidates, key=lambda c: c["score"])
    best["v_max_reachable"] = round(v_max, 3)
    best["regime"] = "full_burn" if abs(best["v_cruise"] - v_max) < 0.05 else "burn_then_coast"
    return best


def _corridor_geometry(params: MathParams) -> tuple[np.ndarray, np.ndarray, float]:
    """Start r0, target T, and |r0| keeping Phobos pad radius invariant when distance_m changes."""
    r0 = C.S.copy()
    target = C.S + params.distance_m * C.SH
    if params.detour_extra_m > 0.0:
        # Represent geometric detour as equivalent extra straight corridor travel
        target = C.S + (params.distance_m + params.detour_extra_m) * C.SH
    return r0, target, float(np.linalg.norm(r0))


def _env_accel_batch(r: np.ndarray, v: np.ndarray, mu: float, frame_accel: bool) -> np.ndarray:
    r2 = np.einsum("ij,ij->i", r, r)
    inv_r3 = -mu / np.maximum(r2 * np.sqrt(r2), 1.0)
    g = r * inv_r3[:, None]
    if frame_accel:
        return g + C.FRAME_B - 2.0 * np.cross(C.OMEGA, v)
    return g


def _simulate_ode_batch(tbs: np.ndarray | list[float], params: MathParams, dt: float = 0.5) -> dict[str, np.ndarray]:
    """Vectorized 3D RK2 burn-then-coast trajectory simulator for an array of burn durations `tbs`."""
    tbs = np.asarray(tbs, dtype=float)
    r0, target, _ = _corridor_geometry(params)
    Nc = len(tbs)
    r = np.repeat(r0[None], Nc, axis=0)
    v = np.repeat((params.v0_along_mps * C.SH)[None], Nc, axis=0)
    dv_main = np.zeros(Nc)
    arr_t = np.full(Nc, np.nan)
    failed = np.zeros(Nc, dtype=bool)
    v_burnout = np.zeros(Nc)
    burned_out = np.zeros(Nc, dtype=bool)
    sh = C.SH
    win_r = params.win_radius_m
    win_r2 = win_r * win_r
    hold = params.hold_t_s
    a_b = params.a_burn
    mu = params.mu
    frame = params.frame_accel
    a_lat = (C.FRAME_B - (C.FRAME_B @ sh) * sh) if frame else None

    diff0 = r - target[None]
    prev_dist = np.sqrt(np.einsum("ij,ij->i", diff0, diff0))
    min_dist = prev_dist.copy()

    t = 0.0
    n_steps = int(np.ceil(params.tmax / dt))
    t_end_burn = hold + tbs

    for _ in range(n_steps):
        live = np.isnan(arr_t) & (~failed)
        if not live.any():
            break

        # Exact overlap of [t, t+dt] with [hold, hold+tb]
        ov = np.minimum(t + dt, t_end_burn) - np.maximum(t, hold)
        ov = np.where(live & (ov > 0.0), ov, 0.0)
        burn_frac = ov / dt

        to_t = target[None] - r
        dn = np.sqrt(np.maximum(np.einsum("ij,ij->i", to_t, to_t), 1e-18))

        if frame:
            spd = np.maximum(np.sqrt(np.einsum("ij,ij->i", v, v)), 3.0)
            t_go = np.minimum(np.maximum((dn - win_r) / spd, 5.0), 220.0)
            aim_pt = target[None] - (0.175 * (t_go * t_go))[:, None] * a_lat[None]
            to_aim = aim_pt - r
            an = np.sqrt(np.maximum(np.einsum("ij,ij->i", to_aim, to_aim), 1e-18))
            u_burn = to_aim / an[:, None]
        else:
            u_burn = to_t / dn[:, None]

        a_th1 = (a_b * burn_frac)[:, None] * u_burn
        a1 = _env_accel_batch(r, v, mu, frame) + a_th1
        r_mid = r + 0.5 * v * dt
        v_mid = v + 0.5 * a1 * dt
        a2 = _env_accel_batch(r_mid, v_mid, mu, frame) + a_th1

        r_next = r + v_mid * dt
        v_next = v + a2 * dt
        dv_main += a_b * ov

        just_finished = (~burned_out) & live & (t + dt >= t_end_burn) & (tbs > 0.0)
        if just_finished.any():
            v_burnout = np.where(just_finished, np.sqrt(np.einsum("ij,ij->i", v_next, v_next)), v_burnout)
            burned_out |= just_finished

        diff_next = r_next - target[None]
        next_dist2 = np.einsum("ij,ij->i", diff_next, diff_next)
        next_dist = np.sqrt(next_dist2)
        min_dist = np.minimum(min_dist, next_dist)

        crossed = live & (next_dist2 <= win_r2)
        if crossed.any():
            denom = np.maximum(prev_dist - next_dist, 1e-9)
            alpha = np.minimum(np.maximum((prev_dist - win_r) / denom, 0.0), 1.0)
            arr_t = np.where(crossed, t + alpha * dt, arr_t)

        # Early-stop trajectories already coasting back toward Phobos after burnout
        if t > hold:
            receding = live & (t >= t_end_burn) & ((v_next @ sh) < -0.25)
            failed |= receding

        r = np.where(live[:, None], r_next, r)
        v = np.where(live[:, None], v_next, v)
        prev_dist = np.where(live, next_dist, prev_dist)
        t += dt

    still_burning_at_arr = (~burned_out) & np.isfinite(arr_t)
    if still_burning_at_arr.any():
        v_burnout = np.where(still_burning_at_arr, np.sqrt(np.einsum("ij,ij->i", v, v)), v_burnout)

    kf = params.kf
    t_score = np.maximum(arr_t + params.clock_offset_s, 1e-3)
    fuel_kg = kf * (params.eta_l1 * dv_main + params.extra_dv_mps) + params.slew_fuel_kg
    dmg_term = 200.0 * (params.damage_kJ / max(params.max_damage_kJ, 1e-3))
    scores = np.where(
        np.isfinite(arr_t),
        50.0 + 100.0 * (params.nominal_time_s / t_score) - 100.0 * (fuel_kg / params.nominal_fuel_kg) - dmg_term,
        np.nan,
    )
    return {
        "tbs": tbs,
        "arrived": np.isfinite(arr_t),
        "T_flight_s": arr_t,
        "T_score_s": t_score,
        "dv_main_mps": dv_main,
        "fuel_kg": fuel_kg,
        "v_burnout_mps": v_burnout,
        "min_dist_m": min_dist,
        "score": scores,
    }


def optimize_gravity_ode(params: MathParams, polish: bool = True) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Find global optimal burn duration under 3D Phobos gravity (+ optional rotating frame)."""
    d = params.effective_distance_m
    tb_upper = float(min(params.tmax - params.hold_t_s, max(20.0, np.sqrt(2.5 * d / params.a_burn) + 20.0)))
    tbs_grid = np.linspace(0.0, tb_upper, 35)
    coarse = _simulate_ode_batch(tbs_grid, params, dt=1.0)

    valid_idx = np.flatnonzero(coarse["arrived"])
    if not len(valid_idx):
        return {
            "arrived": False,
            "t_burn_s": None,
            "v_burnout_mps": None,
            "T_flight_s": None,
            "T_score_s": None,
            "dv_main_mps": None,
            "dv_total_mps": None,
            "fuel_kg": None,
            "score": None,
            "breakdown": None,
        }, coarse

    best_i = int(valid_idx[np.nanargmax(coarse["score"][valid_idx])])
    lo = float(tbs_grid[max(0, best_i - 2)])
    hi = float(tbs_grid[min(len(tbs_grid) - 1, best_i + 2)])

    fine_grid = np.linspace(lo, hi, 25 if polish else 17)
    fine = _simulate_ode_batch(fine_grid, params, dt=0.2 if polish else 0.4)
    f_valid = np.flatnonzero(fine["arrived"])
    if not len(f_valid):
        src, fi = coarse, best_i
    else:
        fi = int(f_valid[np.nanargmax(fine["score"][f_valid])])
        src = fine
        if polish:
            lo2 = float(fine_grid[max(0, fi - 1)])
            hi2 = float(fine_grid[min(len(fine_grid) - 1, fi + 1)])
            zoom_grid = np.linspace(lo2, hi2, 15)
            zoom = _simulate_ode_batch(zoom_grid, params, dt=0.1)
            z_valid = np.flatnonzero(zoom["arrived"])
            if len(z_valid):
                src = zoom
                fi = int(z_valid[np.nanargmax(zoom["score"][z_valid])])

    opt_tb = float(src["tbs"][fi])
    t_fl = float(src["T_flight_s"][fi])
    t_sc = float(src["T_score_s"][fi])
    fuel = float(src["fuel_kg"][fi])
    dv_m = float(src["dv_main_mps"][fi])
    v_bo = float(src["v_burnout_mps"][fi])
    bd = score_from_metrics(
        t_sc, fuel, params.damage_kJ,
        params.nominal_time_s, params.nominal_fuel_kg, params.max_damage_kJ,
    )
    return {
        "arrived": True,
        "t_burn_s": round(opt_tb, 3),
        "v_burnout_mps": round(v_bo, 4),
        "T_flight_s": round(t_fl, 3),
        "T_score_s": round(t_sc, 3),
        "dv_main_mps": round(dv_m, 4),
        "dv_total_mps": round(dv_m + params.extra_dv_mps, 4),
        "fuel_kg": round(fuel, 4),
        "score": round(bd["final_score"], 3),
        "breakdown": bd,
    }, coarse


def simulate_guidance_batch(vcs: np.ndarray | list[float], params: MathParams) -> dict[str, np.ndarray]:
    """Thread-safe pure-numpy reproduction of Core.fly + Core.guid_step parameterized by MathParams."""
    vcs = np.asarray(vcs, dtype=float)
    r0, target, rs = _corridor_geometry(params)
    Nc = len(vcs)
    r = np.repeat(r0[None], Nc, axis=0)
    v = np.repeat((params.v0_along_mps * C.SH)[None], Nc, axis=0)
    burning = np.zeros(Nc, dtype=bool)
    main = np.zeros(Nc, dtype=bool)
    dv_main = np.zeros(Nc)
    dv_trim = np.zeros(Nc)
    arr_t = np.full(Nc, np.nan)

    dt = params.dt_plan
    n_steps = int(params.tmax / dt)
    mu = params.mu
    frame = params.frame_accel
    a_burn = params.a_burn
    a_trim = params.a_trim
    trim_l1 = params.trim_l1
    main_min = params.main_min
    eps_on = params.eps_on
    terminal_m = params.terminal_m
    win_r = params.win_radius_m
    hold = params.hold_t_s

    def _vset(rp: np.ndarray) -> np.ndarray:
        rn = np.linalg.norm(rp, axis=-1)
        return np.sqrt(np.maximum(vcs ** 2 - 2.0 * mu * (1.0 / rs - 1.0 / np.maximum(rn, 1.0)), 4.0))

    v_t = _vset(np.repeat(target[None], Nc, axis=0))

    for k in range(n_steps + 1):
        t = k * dt
        live = np.isnan(arr_t)
        if not live.any():
            break
        dist_now = np.linalg.norm(r - target, axis=1)
        arrived_now = live & (dist_now <= win_r)
        arr_t = np.where(arrived_now, t, arr_t)
        live = np.isnan(arr_t)
        if not live.any():
            break

        d = target[None] - r
        dist = np.linalg.norm(d, axis=1)
        v_r = _vset(r)
        tgo = np.maximum(dist / (0.5 * (v_r + v_t)), 1e-3)
        if frame:
            vm = d / tgo[:, None]
            vreq = vm - (tgo[:, None] / 6.0) * (
                _env_accel_batch(r, vm, mu, True) + 2.0 * _env_accel_batch(r + 0.5 * d, vm, mu, True)
            )
        else:
            vreq = d / tgo[:, None] - (tgo[:, None] / 6.0) * (
                _env_accel_batch(r, v, mu, False) + 2.0 * _env_accel_batch(r + 0.5 * d, v, mu, False)
            )
        vg = vreq - v
        ng = np.linalg.norm(vg, axis=1)

        go = t >= hold
        if not go:
            burning[:] = False
        else:
            ignite = (~burning) & (ng > eps_on) & ~(dist < terminal_m)
            main = np.where(ignite, ng > main_min, main)
            burning = burning | ignite

        a = np.where(main, a_burn, a_trim)
        step = a * dt
        fin = burning & (ng <= step)
        on = burning & ~fin
        dv_raw = np.where(fin, ng, 0.0) + np.where(on, step, 0.0)
        dv_main += np.where(live & main, dv_raw, 0.0)
        dv_trim += np.where(live & ~main, dv_raw * trim_l1, 0.0)

        dv_vec = np.where(fin[:, None], vg, 0.0) + np.where(on[:, None], (step / np.maximum(ng, 1e-9))[:, None] * vg, 0.0)
        v_next = v + np.where(live[:, None], dv_vec, 0.0)
        burning = burning & ~fin

        a_env = _env_accel_batch(r, v_next, mu, frame)
        v_next = np.where(live[:, None], v_next + a_env * dt, v)
        r_next = np.where(live[:, None], r + v_next * dt, r)
        r, v = r_next, v_next

    kf = params.kf
    t_score = np.maximum(arr_t + params.clock_offset_s, 1e-3)
    fuel_kg = kf * (params.eta_l1 * dv_main + dv_trim + params.extra_dv_mps) + params.slew_fuel_kg
    dmg_term = 200.0 * (params.damage_kJ / max(params.max_damage_kJ, 1e-3))
    scores = np.where(
        np.isfinite(arr_t),
        50.0 + 100.0 * (params.nominal_time_s / t_score) - 100.0 * (fuel_kg / params.nominal_fuel_kg) - dmg_term,
        np.nan,
    )
    return {
        "vcs": vcs,
        "arrived": np.isfinite(arr_t),
        "T_flight_s": arr_t,
        "T_score_s": t_score,
        "dv_main_mps": dv_main,
        "dv_trim_mps": dv_trim,
        "fuel_kg": fuel_kg,
        "score": scores,
    }


def optimize_guidance(params: MathParams) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Sweep coast speed vc in guidance simulator to find best guidance-achievable score."""
    vc_lo = max(3.5, params.v0_along_mps + 0.5)
    vc_hi = max(18.0, min(32.0, np.sqrt(2.0 * params.a_burn * params.effective_distance_m) + 2.0))
    vc_grid = np.linspace(vc_lo, vc_hi, 65)
    curve = simulate_guidance_batch(vc_grid, params)
    valid = np.flatnonzero(curve["arrived"])
    if not len(valid):
        return {"arrived": False, "vc_opt": None, "score": None}, curve

    bi = int(valid[np.nanargmax(curve["score"][valid])])
    lo = float(vc_grid[max(0, bi - 2)])
    hi = float(vc_grid[min(len(vc_grid) - 1, bi + 2)])
    fine_vcs = np.linspace(lo, hi, 33)
    fine = simulate_guidance_batch(fine_vcs, params)
    fv = np.flatnonzero(fine["arrived"])
    fi = int(fv[np.nanargmax(fine["score"][fv])]) if len(fv) else 0
    best_src, idx = (fine, fi) if len(fv) else (curve, bi)

    vc_opt = float(best_src["vcs"][idx])
    t_fl = float(best_src["T_flight_s"][idx])
    t_sc = float(best_src["T_score_s"][idx])
    fuel = float(best_src["fuel_kg"][idx])
    bd = score_from_metrics(
        t_sc, fuel, params.damage_kJ,
        params.nominal_time_s, params.nominal_fuel_kg, params.max_damage_kJ,
    )
    return {
        "arrived": True,
        "vc_opt": round(vc_opt, 3),
        "T_flight_s": round(t_fl, 3),
        "T_score_s": round(t_sc, 3),
        "fuel_kg": round(fuel, 4),
        "dv_main_mps": round(float(best_src["dv_main_mps"][idx]), 4),
        "dv_trim_mps": round(float(best_src["dv_trim_mps"][idx]), 4),
        "score": round(bd["final_score"], 3),
        "breakdown": bd,
    }, curve


def _compute_sensitivities(ode_opt: dict[str, Any], params: MathParams) -> dict[str, float]:
    """Compute local score sensitivities (partial derivatives) around the 3D gravity optimum."""
    if not ode_opt.get("arrived"):
        return {}
    T = float(ode_opt["T_score_s"])
    s0 = float(ode_opt["score"])
    kf = params.kf
    dS_dT = -100.0 * params.nominal_time_s / (T * T)
    dS_dF = -100.0 / params.nominal_fuel_kg
    dS_ddv = dS_dF * kf * params.eta_l1
    dS_ddamage_kJ = -200.0 / params.max_damage_kJ

    def _reopt_delta(**changes: Any) -> float:
        p2 = MathParams.from_dict({**asdict(params), **changes})
        res, _ = optimize_gravity_ode(p2, polish=False)
        return float(res["score"]) - s0 if res.get("arrived") else 0.0

    ds_da = _reopt_delta(a_burn=params.a_burn + 0.01)
    ds_dhold = _reopt_delta(hold_t_s=params.hold_t_s + 1.0)
    ds_dmu = _reopt_delta(mu=params.mu * 1.1) if params.mu > 0 else 0.0

    return {
        "pts_per_s": round(dS_dT, 4),
        "pts_per_kg": round(dS_dF, 4),
        "pts_per_mps_dv": round(dS_ddv, 4),
        "pts_per_kJ_damage": round(dS_ddamage_kJ, 4),
        "pts_per_001_accel": round(ds_da, 3),
        "pts_per_1s_hold": round(ds_dhold, 3),
        "pts_per_10pct_mu": round(ds_dmu, 3),
    }


def analyze_max_score(raw_params: dict[str, Any] | MathParams | None = None) -> dict[str, Any]:
    """Run full analytical, 3D-gravity ODE, and Core.fly guidance optimization for max score."""
    params = raw_params if isinstance(raw_params, MathParams) else MathParams.from_dict(raw_params)
    r0, target, rs = _corridor_geometry(params)
    rt = float(np.linalg.norm(target))
    dphi = params.mu * (1.0 / max(rs, 1.0) - 1.0 / max(rt, 1.0))
    v_esc = float(np.sqrt(max(2.0 * dphi, 0.0)))

    anal = analytical_bang_coast(params)
    ode_opt, ode_coarse = optimize_gravity_ode(params)
    guid_opt, guid_curve = optimize_guidance(params)

    # Evaluate configured plan `vcs`
    cfg_vcs = np.array(params.vcs, dtype=float)
    cfg_res = simulate_guidance_batch(cfg_vcs, params)
    configured_vcs = []
    for i, vc_val in enumerate(cfg_vcs):
        configured_vcs.append({
            "vc": round(float(vc_val), 3),
            "arrived": bool(cfg_res["arrived"][i]),
            "T_score_s": round(float(cfg_res["T_score_s"][i]), 2) if cfg_res["arrived"][i] else None,
            "fuel_kg": round(float(cfg_res["fuel_kg"][i]), 2) if cfg_res["arrived"][i] else None,
            "score": round(float(cfg_res["score"][i]), 3) if cfg_res["arrived"][i] else None,
        })
    valid_cfg = [row for row in configured_vcs if row["score"] is not None]
    best_cfg_vc = max(valid_cfg, key=lambda r: r["score"]) if valid_cfg else None

    # Loss attribution cascade from clean zero-gravity bound -> actual guidance
    clean_p = MathParams.from_dict({
        **asdict(params),
        "mu": 0.0, "frame_accel": False, "hold_t_s": 0.0, "clock_offset_s": 0.0,
        "eta_l1": 1.0, "extra_dv_mps": 0.0, "slew_fuel_kg": 0.0, "detour_extra_m": 0.0, "damage_kJ": 0.0,
    })
    clean_nograv = analytical_bang_coast(clean_p)
    grav_only_p = MathParams.from_dict({
        **asdict(clean_p),
        "mu": params.mu, "frame_accel": params.frame_accel,
    })
    grav_only, _ = optimize_gravity_ode(grav_only_p, polish=False)
    delay_p = MathParams.from_dict({
        **asdict(grav_only_p),
        "hold_t_s": params.hold_t_s, "clock_offset_s": params.clock_offset_s,
    })
    delay_ode, _ = optimize_gravity_ode(delay_p, polish=False)

    loss_attribution = {
        "ideal_nograv_score": clean_nograv["score"],
        "gravity_and_frame_loss": round((grav_only["score"] or 0.0) - clean_nograv["score"], 3) if grav_only.get("arrived") else None,
        "hold_and_clock_loss": round((delay_ode["score"] or 0.0) - (grav_only["score"] or 0.0), 3) if (delay_ode.get("arrived") and grav_only.get("arrived")) else None,
        "ops_l1_slew_detour_dmg_loss": round((ode_opt["score"] or 0.0) - (delay_ode["score"] or 0.0), 3) if (ode_opt.get("arrived") and delay_ode.get("arrived")) else None,
        "guidance_quantization_gap": round((guid_opt["score"] or 0.0) - (ode_opt["score"] or 0.0), 3) if (guid_opt.get("arrived") and ode_opt.get("arrived")) else None,
    }

    # Curves for plotting (sampled cleanly over speed v)
    v_grid = guid_curve["vcs"]
    nograv_pts = [_eval_nograv_at_v(float(vv), params) for vv in v_grid]

    # Map ODE coarse points by burnout speed for curve overlay
    ode_ok = ode_coarse["arrived"]
    curves = {
        "v_grid": [round(float(x), 3) for x in v_grid],
        "score_nograv": [pt["score"] for pt in nograv_pts],
        "T_nograv": [pt["T_score_s"] for pt in nograv_pts],
        "F_nograv": [pt["fuel_kg"] for pt in nograv_pts],
        "score_guidance": [round(float(x), 3) if np.isfinite(x) else None for x in guid_curve["score"]],
        "T_guidance": [round(float(x), 3) if np.isfinite(x) else None for x in guid_curve["T_score_s"]],
        "F_guidance": [round(float(x), 3) if np.isfinite(x) else None for x in guid_curve["fuel_kg"]],
        "v_ode": [round(float(x), 3) for x in ode_coarse["v_burnout_mps"][ode_ok]],
        "score_ode": [round(float(x), 3) for x in ode_coarse["score"][ode_ok]],
        "T_ode": [round(float(x), 3) for x in ode_coarse["T_score_s"][ode_ok]],
        "F_ode": [round(float(x), 3) for x in ode_coarse["fuel_kg"][ode_ok]],
    }

    # Actuator preset comparison under the current environment/ops parameters
    presets_comparison = []
    for key, preset in ACTUATOR_PRESETS.items():
        p_pre = MathParams.from_dict({
            **asdict(params),
            "a_burn": preset["a_burn"],
            "eta_l1": preset["eta_l1"],
            "slew_fuel_kg": preset["slew_fuel_kg"],
        })
        an_pre = analytical_bang_coast(p_pre)
        od_pre, _ = optimize_gravity_ode(p_pre, polish=False)
        presets_comparison.append({
            "key": key,
            "label": preset["label"],
            "a_burn": preset["a_burn"],
            "eta_l1": preset["eta_l1"],
            "nograv_score": an_pre["score"],
            "ode_arrived": od_pre["arrived"],
            "ode_score": od_pre["score"],
            "ode_v": od_pre["v_burnout_mps"],
            "ode_t_burn": od_pre["t_burn_s"],
            "ode_T": od_pre["T_score_s"],
            "ode_fuel": od_pre["fuel_kg"],
        })

    sensitivities = _compute_sensitivities(ode_opt, params)

    return {
        "params": {
            **asdict(params),
            "kf": round(params.kf, 6),
            "effective_distance_m": round(params.effective_distance_m, 3),
            "r_start_m": round(rs, 2),
            "r_target_m": round(rt, 2),
            "delta_phi_J_kg": round(dphi, 4),
            "v_escape_corridor_mps": round(v_esc, 4),
        },
        "analytical": anal,
        "gravity_ode": ode_opt,
        "guidance": guid_opt,
        "configured_vcs": configured_vcs,
        "best_configured_vc": best_cfg_vc,
        "loss_attribution": loss_attribution,
        "sensitivities": sensitivities,
        "curves": curves,
        "presets_comparison": presets_comparison,
    }
