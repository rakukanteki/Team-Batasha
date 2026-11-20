"""V13 sparse route search with a target-facing trajectory rollout."""
from itertools import combinations, islice

import numpy as np

from Systems import Actuation, Core as C
from Systems.plans.plan_8_astar.graph import Graph
from Systems.plans.plan_9_cost_lattice.dp import solve_lattice_detail
from Systems.plans.plan_13_v13.config import CONFIG


def rollout(plans, st, rocks, cfg):
    """Carry position, velocity, attitude and angular rates through each waypoint.

    Translation shares the live guidance law. Delivered acceleration is an
    approximation based on thrust direction in body axes, not a per-jet solve.
    """
    from scipy.spatial.transform import Rotation

    nodes, counts, cruise = C.pack(plans)
    n = len(plans)
    p = np.tile(st["r"], (n, 1))
    v = np.tile(st["v"], (n, 1))
    initial = st.get("body", np.column_stack((C.E1, C.E2, C.SH)))
    body = Rotation.from_matrix(np.repeat(initial[None], n, axis=0))
    rates = np.tile(st.get("rates", np.zeros(3)), (n, 1))
    burning = np.full(n, st["burning"])
    main = np.full(n, st["main"])
    index = np.zeros(n, dtype=int)
    arrived = np.full(n, -1, dtype=int)
    fuel_dv = np.zeros(n)
    coast_pointing = np.full(n, st.get("coast_pointing", False), dtype=bool)
    acquired = np.full(n, st.get("main_axis_acquired", False), dtype=bool)
    forward = cfg.get("pointing_axis", 2)
    path, speed = [], []
    dt = cfg["dt_plan"]
    ignite_dv = cfg.get("ignite_dv_mps", C.EPS_ON)
    stop_dv = cfg.get("stop_dv_mps", 0.0)
    coast_docking_radius = cfg.get("coast_docking_radius_m", 0.0)
    coast_waypoint_radius = cfg.get("coast_waypoint_radius_m", 0.0)
    fast_cruise = C.late_cruise(cruise, cfg)
    for k in range(int(cfg["tmax"] / dt) + 1):
        path.append(p)
        speed.append(np.linalg.norm(v, axis=1))
        arrived[(arrived < 0) & (np.linalg.norm(p - C.T, axis=1) <= C.WIN)] = k
        live = arrived < 0
        if not live.any():
            break

        inv = body.inv()
        bearing = inv.apply(C.T - p)
        gain, _ = C.to_gain(p, v, nodes[np.arange(n), index], cruise, frame=True)
        body_gain = inv.apply(gain)
        gain_norm = np.linalg.norm(body_gain, axis=1)
        axis_mix = np.abs(body_gain).sum(axis=1) / np.maximum(gain_norm, 1e-9)
        # Mixed-axis thrust pays an acceleration penalty. Use the live learned
        # on-axis estimate and the calibrated trim estimate as the starting scales.
        factor = np.maximum(axis_mix, 1.0)
        go = True
        main_direction = None
        axial = np.zeros(n, dtype=bool)
        burn_accel = C.A_BURN / factor
        if cfg.get("x_forward_acquire_deg") is not None:
            want_main = np.where(burning, main, gain_norm > cfg["main_min"])
            burn_bearing = Actuation.x_forward_burn_bearings(body_gain, bearing, cfg)
            ready, axial = Actuation.x_forward_main_masks(body_gain, burn_bearing, rates, acquired, cfg)
            acquired |= want_main & ready
            go = ~want_main | acquired | st.get("launch_released", False)
            axial &= want_main & (gain_norm >= cfg.get("x_forward_precision_dv_mps", 0.0))
            main_direction = np.where(axial[:, None], body.apply(np.tile(Actuation.X_AXIS, (n, 1))),
                                      gain / np.maximum(gain_norm[:, None], 1e-9))
            burn_accel = np.where(axial, C.A_BURN, 0.096 / factor)
        trim_accel = C.A_TRIM / factor
        if cfg.get("launch_clearance_m"):
            precision = gain_norm < cfg["x_forward_precision_dv_mps"]
            nominal = gain_norm / cfg["x_forward_precision_settle_s"]
            burn_accel = np.where(precision, np.minimum(burn_accel, nominal), burn_accel)
            trim_accel = np.where(precision, np.minimum(trim_accel, nominal), trim_accel)
        new_p, new_v, burning, main, index, spent, _ = C.guid_step(
            p, v, burning, main, index, nodes, counts, cruise, dt,
            a_burn=burn_accel, a_trim=trim_accel, go=go,
            frame=True, main_min=cfg["main_min"], main_direction=main_direction,
            ignite_dv=ignite_dv, stop_dv=stop_dv,
            coast_docking_radius=coast_docking_radius,
            coast_waypoint_radius=coast_waypoint_radius,
            build_cruise=fast_cruise & C.cruise_needs_thrust(p, v, cruise) if np.any(fast_cruise) else False,
        )
        coast = spent <= 1e-12
        pointing = bearing
        if cfg.get("x_forward_acquire_deg") is not None:
            pointing = np.where((main & ~coast)[:, None], burn_bearing, bearing)
        angular = Actuation.target_axis_attitudes(pointing, rates, forward)
        if cfg["coast_pointing_deadband"]:
            coast_pointing = coast & Actuation.coast_pointing_mask(bearing, rates, coast_pointing, cfg, forward)
            angular[coast & ~coast_pointing] = 0.0
            if cfg.get("coast_max_turn_rate_rad_s") is not None:
                angular[coast_pointing] = Actuation.target_axis_attitudes(
                    bearing[coast_pointing], rates[coast_pointing], forward, cfg["coast_max_turn_rate_rad_s"]
                )
            quiet = coast & (np.linalg.norm(angular, axis=1) < cfg.get("coast_min_torque", 0.0))
            angular[quiet] = 0.0
        angular[axial & ~coast] = 0.0
        # For torque alone, the selector spends propellant and scales down
        # delivered torque when the longest jet pulse exceeds its tick.
        requested_fuel = cfg["coast_torque_fuel_slope"] * np.linalg.norm(angular, axis=1)
        torque_fuel = np.minimum(requested_fuel, cfg["coast_torque_fuel_cap_kg_s"])
        scale = torque_fuel / np.maximum(requested_fuel, 1e-12)
        angular[coast] *= scale[coast, None]
        extra_dv = np.where(coast, torque_fuel * dt / C.KF, 0.0)
        if cfg.get("launch_clearance_m"):
            # Main guidance's fuel baseline is single-axis. Charge off-axis
            # components and use the larger torque estimate without adding
            # both in full, since one jet selection can supply both impulses.
            translation = spent * np.where(main & ~axial, factor, 1.0)
            spent = np.where(coast, 0.0, np.maximum(translation, torque_fuel * dt / C.KF))
        rates += angular * dt
        body = body * Rotation.from_rotvec(rates * dt)
        p = np.where(live[:, None], new_p, p)
        v = np.where(live[:, None], new_v, v)
        fuel_dv += np.where(live, spent + extra_dv, 0.0)

    result = C.evaluate(np.array(path), np.array(speed), arrived, fuel_dv, counts, rocks,
                        fuel0=st["fuel"], t0=st["t"], cfg=cfg)
    if cfg.get("fuel_goal_kg") is not None:
        result["cost"] += 3.0 * np.maximum(result["fuel"] - cfg["fuel_goal_kg"], 0.0)
    return result


def sparse_candidates(nodes, cruise, limit):
    """Try short subsequences of the lattice route and always retain the full route."""
    waypoints, goal = nodes[:-1], nodes[-1]
    for size in range(1, len(waypoints)):
        for subset in islice(combinations(waypoints, size), min(6, limit)):
            yield [*subset, goal], cruise
            limit -= 1
        if limit <= 0:
            break
    yield nodes, cruise


def make_planner(config=None):
    cfg = CONFIG if config is None else config

    def choose(plans, result, action, limit=None):
        safe_mask = ((result["ptot"] <= cfg["p_safe"]) &
                     (result["clear"] >= cfg["route_clearance_m"]) &
                     (result["T"] < 1e6))
        safe = np.flatnonzero(safe_mask[:limit])
        if not len(safe):
            return None
        best = min(safe, key=lambda i: (len(plans[i][0]), result["cost"][i]))
        return plans[best], C.info(plans, result, best, action)

    def planner(st, rocks, plan, committed):
        if plan is not None:
            faster = cfg.get("late_cruise_speeds_mps", ())
            can_upgrade = (faster and len(plan[0]) == 1 and not C.late_cruise(plan[1], cfg)
                           and (st["r"] - C.S) @ C.SH >= cfg["late_cruise_after_m"]
                           and np.linalg.norm(C.T - st["r"]) >= cfg["late_cruise_min_remaining_m"]
                           and st["fuel"] < cfg["fuel_goal_kg"] - cfg["late_fuel_reserve_kg"])
            plans = [plan, *((plan[0], speed) for speed in faster)] if can_upgrade else [plan]
            result = rollout(plans, st, rocks, cfg)
            kept = choose(plans, result, "keep", limit=1)
            if kept is not None:
                if can_upgrade:
                    _, _, hard, sp, sv = rocks
                    encounter_t = np.minimum(result["kmin"] * cfg["dt_plan"], 45.0)
                    pad = cfg["route_clearance_m"] + 2.0 * np.hypot(sp, encounter_t * sv)
                    clear = np.all(result["mind"] - hard >= pad, axis=1)
                    affordable = result["fuel"] <= cfg["fuel_goal_kg"] - cfg["late_fuel_reserve_kg"]
                    safer = result["ptot"] <= cfg["p_safe"]
                    faster_arrival = result["T"] <= result["T"][0] - 8.0
                    viable = np.flatnonzero(clear & affordable & safer & faster_arrival)
                    if len(viable):
                        best = int(viable[np.argmin(result["T"][viable])])
                        return plans[best], C.info(plans, result, best, "v14_late_cruise")
                return kept

        cruise = C.resolve_vc(cfg)
        direct = [([C.T.copy()], speed) for speed in C.resolve_vcs(cfg)]
        result = rollout(direct, st, rocks, cfg)
        straight = choose(direct, result, "v13_direct")
        if straight is not None:
            return straight
        if not len(rocks[0]):
            return None, {"action": "v13_no_safe_route"}

        # The docking rule is a sphere. Try direct approaches to other points
        # inside it before spending an intermediate waypoint.
        approaches = [([C.T + 35.0 * d], cruise) for d in C.DIRS]
        straight = choose(approaches, rollout(approaches, st, rocks, cfg), "v13_direct")
        if straight is not None:
            return straight

        # Seed one-waypoint detours before the most limiting straight-line encounter.
        best_direct = int(np.argmin(result["ptot"]))
        worst = int(np.argmin(result["mind"][best_direct] - rocks[2]))
        encounter = result["path"][result["kmin"][best_direct, worst], best_direct]
        advance = (encounter - st["r"]) @ C.SH
        if advance > 30.0:
            anchor = st["r"] + 0.6 * (encounter - st["r"])
            detours = [([anchor + offset * d, C.T], cruise) for offset in (25.0, 60.0) for d in C.DIRS]
            single = choose(detours, rollout(detours, st, rocks, cfg), "v13_route")
            if single is not None:
                return single

        graph = Graph(st, rocks, cruise, cfg=cfg)
        cells, _, _ = solve_lattice_detail(graph)
        candidates = list(sparse_candidates(graph.nodes(cells), cruise, cfg["route_candidate_limit"]))
        selected = choose(candidates, rollout(candidates, st, rocks, cfg), "v13_route")
        return selected if selected is not None else (None, {"action": "v13_no_safe_route"})

    planner.__name__ = "v13"
    return planner


v13 = make_planner()
