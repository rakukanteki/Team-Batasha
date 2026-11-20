"""Shortlist +Z burn-and-coast schedules using the measured flight state."""
from __future__ import annotations

from time import monotonic

import numpy as np

from Systems import Core as C
from Systems.Maneuvers import Maneuver, Segment, maneuver_of
from Systems.plans.plan_10_real_calibrated import make_planner as make_fallback
from Systems.plans.plan_17_v16_morning.config import CONFIG

PULSE_ACCEL = 0.05


def _controls(st, aim, eta, pulse, delay, accel):
    """Solve a constant-gravity starting guess for each departure direction and burn duration."""
    pulse_dt = np.linalg.norm(pulse, axis=1) / PULSE_ACCEL
    gravity = C.accel(st["r"], st["v"], frame=True)
    halfway = 0.5 * (st["r"] + aim)
    gravity = (gravity + 2.0 * C.accel(halfway, (aim - st["r"]) / eta[:, None], frame=True)) / 3.0
    need = aim - st["r"] - eta[:, None] * st["v"] - 0.5 * eta[:, None] ** 2 * gravity
    need -= pulse * (eta - 0.5 * pulse_dt)[:, None]
    distance = np.linalg.norm(need, axis=1)
    free = eta - pulse_dt - delay
    disc = free**2 - 2.0 * distance / accel
    duration = free - np.sqrt(np.maximum(disc, 0.0))
    direction = need / np.maximum(distance[:, None], 1e-9)
    return direction, duration, disc > 0.0


def _rollout(st, direction, duration, pulse, delay, accel, horizon, dt):
    """Propagate a pulse, a fixed +Z burn, and a ballistic coast for every candidate."""
    n = len(direction)
    p = np.broadcast_to(st["r"], (n, 3)).copy()
    v = np.broadcast_to(st["v"], (n, 3)).copy()
    axis = np.broadcast_to(st["u"], (n, 3)).copy()
    pulse_dt = np.linalg.norm(pulse, axis=1) / PULSE_ACCEL
    pulse_rate = pulse / np.maximum(pulse_dt[:, None], 1e-9)
    arrive = np.full(n, -1, int)
    fuel = np.zeros(n)
    times = np.arange(0.0, horizon + dt, dt)
    path = np.empty((len(times), n, 3))
    velocities = np.empty_like(path)
    for i, t in enumerate(times):
        path[i], velocities[i] = p, v
        arrive[(arrive < 0) & (np.linalg.norm(p - C.T, axis=1) <= C.WIN - 3.0)] = i
        live = arrive < 0
        pulse_step = np.clip(pulse_dt - t, 0.0, dt)
        burn_start = pulse_dt + delay
        burn_step = np.maximum(0.0, np.minimum(t + dt, burn_start + duration) - np.maximum(t, burn_start))
        axis = C.turn_toward(axis, direction, 0.055 * burn_step)
        cosine = np.sum(axis * direction, axis=1)
        aligned = cosine >= np.cos(np.deg2rad(10.0))
        mixed = cosine < np.cos(np.deg2rad(35.0))
        delivered = np.where(aligned, accel, np.minimum(accel, np.where(mixed, 0.082, 0.103)))
        thrust = np.where(mixed[:, None], direction, axis) * delivered[:, None]
        impulse = pulse_rate * pulse_step[:, None] + thrust * burn_step[:, None]
        gravity = C.accel(p, v, frame=True)
        p += v * dt + 0.5 * gravity * dt**2 + 0.5 * impulse * dt
        v += gravity * dt + impulse
        fuel += live * (delivered * burn_step * np.where(aligned, 1.015, np.where(mixed, 1.38, 1.12))
                        + PULSE_ACCEL * pulse_step * C.TRIM_L1) * C.KF
    return path, velocities, arrive, fuel


def _candidates(st, cfg, aim, eta, pulse, delay, accel):
    direction, duration, viable = _controls(st, aim, eta, pulse, delay, accel)
    horizon = max(85.0, float(np.max(eta)) + 4.0)
    dt = cfg["dt_plan"]
    # Correct the analytic guess once for rotating-frame dynamics and the measured attitude.
    path, _, _, _ = _rollout(st, direction, duration, pulse, delay, accel, horizon, dt)
    endpoint = path[np.minimum(np.rint(eta / dt).astype(int), len(path) - 1), np.arange(len(eta))]
    direction, duration, refined = _controls(st, aim + (aim - endpoint), eta, pulse, delay, accel)
    keep = viable & refined & (duration >= 0.0)
    return direction[keep], duration[keep], pulse[keep], delay[keep], horizon


def _review(st, rocks, cfg, direction, duration, pulse, delay, accel, horizon, fuel_cap=np.inf):
    if not len(duration):
        return None
    dt = cfg["dt_plan"]
    path, velocity, arrive, fuel = _rollout(st, direction, duration, pulse, delay, accel, horizon, dt)
    total_fuel = fuel + st["fuel"]
    time = st["t"] + arrive * dt
    arrival = np.flatnonzero((arrive >= 0) & (total_fuel <= fuel_cap))
    if not len(arrival):
        return None
    key = (time + 4.0 * np.maximum(total_fuel - cfg["fuel_goal_kg"], 0.0)
           + 0.75 * np.maximum(time - cfg["planning_time_s"], 0.0))
    # Include the lowest-fuel finishes so the shortlist retains alternatives to the fastest path.
    ordered = arrival[np.argsort(key[arrival])]
    # Keep alternatives at different departure times and in different lateral directions.
    lateral = direction[ordered] @ np.array([C.E1, C.E2]).T
    sectors = np.floor((np.arctan2(lateral[:, 1], lateral[:, 0]) + np.pi) * 2.0 / np.pi).astype(int)
    diversity = np.column_stack([delay[ordered], sectors])
    _, first = np.unique(diversity, return_index=True, axis=0)
    chosen = np.union1d(ordered[:12], ordered[first])
    chosen = np.union1d(chosen, arrival[np.argsort(total_fuel[arrival])[:6]])
    result = C.evaluate(path[:, chosen], np.linalg.norm(velocity[:, chosen], axis=2),
                        arrive[chosen], fuel[chosen] / C.KF, np.ones(len(chosen)), rocks,
                        cfg=cfg, fuel0=st["fuel"], t0=st["t"])
    _, _, hard, sp, sv = rocks
    near_time = result["kmin"] * dt
    uncertainty = 1.5 * np.hypot(sp, np.minimum(near_time, 30.0) * sv)
    pad = 2.0 + np.where(near_time <= cfg["review_horizon_s"], uncertainty, 0.0)
    clear = np.all(result["mind"] - hard >= pad, axis=1)
    ok = np.flatnonzero(clear & (result["ptot"] <= cfg["p_safe"]))
    if not len(ok):
        return None
    feasible = ok[(time[chosen[ok]] < 200.0) & (total_fuel[chosen[ok]] < 45.0)]
    reserve = feasible[(time[chosen[feasible]] <= cfg["planning_time_s"])
                       & (total_fuel[chosen[feasible]] <= cfg["fuel_goal_kg"])]
    if len(reserve):
        i = int(min(reserve, key=lambda k: (total_fuel[chosen[k]], time[chosen[k]])))
    else:
        pool = feasible if len(feasible) else ok
        i = int(pool[np.argmin(key[chosen[pool]])])
    j = chosen[i]
    stop = arrive[j] + 1
    end = stop * dt + 2.0
    pulse_dt = np.linalg.norm(pulse[j]) / PULSE_ACCEL
    segments = []
    if pulse_dt > 0.0:
        segments.append(Segment(float(pulse_dt), tuple(pulse[j] / np.linalg.norm(pulse[j])),
                                pulse_accel=PULSE_ACCEL))
    if delay[j] > 0.0:
        segments.append(Segment(float(delay[j])))
    if duration[j] > 0.0:
        segments.append(Segment(float(duration[j]), tuple(direction[j])))
    segments.append(Segment(max(dt, end - pulse_dt - delay[j] - duration[j])))
    summary = dict(cost=float(key[j]), ptot=float(result["ptot"][i]), clear=float(result["clear"][i]),
                   K=0, predicted_arrival_s=float(time[j]), predicted_fuel_kg=float(total_fuel[j]))
    return tuple(segments), path[:stop + 1, j], velocity[:stop + 1, j], summary


def make_planner(config=None):
    cfg = CONFIG if config is None else config
    fallback = make_fallback(cfg)

    def planner(st, rocks, plan, committed):
        current = maneuver_of(plan)
        late = (st["r"] - C.S) @ C.SH >= 500.0 and np.linalg.norm(C.T - st["r"]) >= 200.0
        can_speed_up = (current is not None and late and not st["burning"]
                        and st["fuel"] < cfg["fuel_goal_kg"] - 2.0 and st["t"] < 175.0)
        if current is not None and not can_speed_up:
            return plan, {"action": "keep"}

        previous = current.fallback if current is not None else plan
        deadline = st.get("deadline", np.inf)
        if st["t"] < cfg["route_handoff_s"] or "u" not in st:
            return fallback(st, rocks, previous, committed)

        accel = float(np.clip(C.A_BURN, 0.108, 0.137))
        coast_horizon = max(75.0, 220.0 - st["t"])
        coast = _review(st, rocks, cfg, np.asarray([st["u"]]), np.zeros(1),
                        np.zeros((1, 3)), np.zeros(1), accel, coast_horizon)
        result = coast
        if (coast is None or coast[3]["predicted_arrival_s"] > cfg["planning_time_s"]) and monotonic() < deadline:
            times = np.asarray(cfg["arrival_targets_s"]) - st["t"]
            times = times[times > 8.0]
            if len(times):
                center = C.T - 24.0 * C.SH
                aims = np.array([center, *(center + offset * d for offset in (12.0, 24.0) for d in C.DIRS)])
                aim = np.tile(aims, (len(times), 1))
                eta = np.repeat(times, len(aims))
                pulse = np.zeros_like(aim)
                direction, duration, pulse, delay, horizon = _candidates(
                    st, cfg, aim, eta, pulse, np.zeros(len(aim)), accel)
                fuel_cap = cfg["fuel_goal_kg"] if can_speed_up else np.inf
                powered = _review(st, rocks, cfg, direction, duration, pulse, delay, accel, horizon, fuel_cap)
                if powered is None and coast is None and monotonic() < deadline:
                    # Use a single early lateral pulse to move either side of a crossing.
                    pulse = np.array([dv * d for dv in (0.25, 0.5) for d in C.DIRS])
                    delayed_aim = np.tile(aims, (3, 1))
                    aim = np.vstack([np.broadcast_to(center, pulse.shape), delayed_aim])
                    eta = np.full(len(aim), times[-1])
                    delay = np.r_[np.zeros(len(pulse)), np.repeat([3.0, 6.0, 9.0], len(aims))]
                    pulse = np.vstack([pulse, np.zeros_like(delayed_aim)])
                    direction, duration, pulse, delay, horizon = _candidates(st, cfg, aim, eta, pulse, delay, accel)
                    powered = _review(st, rocks, cfg, direction, duration, pulse, delay, accel, horizon, fuel_cap)
                if powered is not None and (coast is None or
                        (powered[3]["predicted_arrival_s"] < coast[3]["predicted_arrival_s"] - 5.0
                         and powered[3]["predicted_fuel_kg"] <= cfg["fuel_goal_kg"])):
                    result = powered

        if current is not None:
            scheduled_arrival = current.start + (len(current.path) - 2) * current.dt
            if result is None or result is coast or result[3]["predicted_arrival_s"] >= scheduled_arrival - 5.0:
                return plan, {"action": "keep"}
        if result is None:
            return fallback(st, rocks, previous, committed)
        segments, path, velocity, info = result
        fallback_route = previous if previous is not None else ([C.T.copy()], C.resolve_vc(cfg))
        action = "v16_late_burn" if can_speed_up else "v16_coast" if result is coast else "v16_burn"
        maneuver = Maneuver(st["t"], segments, cfg["dt_plan"], path, velocity, fallback_route, True,
                            review_horizon=cfg["review_horizon_s"])
        return ([C.T.copy()], C.resolve_vc(cfg), st["t"], maneuver), {**info, "action": action}

    return planner


v16 = make_planner()
