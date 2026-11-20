"""Search routes with progressively more knots around predicted encounters."""
from itertools import zip_longest
from time import monotonic

import numpy as np

from Systems import Core as C
from Systems.plans.plan_13_v13.planner import rollout
from Systems.plans.plan_15_musubi.config import CONFIG


def safe(result, i, cfg):
    return (result["T"][i] < 1e6 and result["clear"][i] >= cfg["route_clearance_m"]
            and result["ptot"][i] <= cfg["p_safe"])


def priority(result, i, cfg):
    """Prefer smaller clearance deficits, lower risk, and higher projected score."""
    deficit = max(0.0, cfg["route_clearance_m"] - result["clear"][i])
    return result["cost"][i] + 5.0 * deficit + 100.0 * result["ptot"][i]


def insertions(plan, result, i, st, rocks, attempt, cfg):
    """Place alternate knots before the worst encounters on the flown forecast."""
    nodes, cruise = plan
    rp, rv, hard, sp, sv = rocks
    if not len(rp):
        return
    margin = cfg["route_clearance_m"]
    gaps = result["mind"][i] - hard - margin
    risk = result["ph"][i]
    blockers = np.argsort(np.maximum(-gaps, 0.0) + 40.0 * risk)[::-1][:2]
    stations = np.array([(node - st["r"]) @ C.SH for node in nodes])
    for rock in blockers:
        if gaps[rock] >= 0 and risk[rock] <= cfg["p_safe"] / max(len(rp), 1):
            continue
        tick = result["kmin"][i, rock]
        encounter = result["path"][tick, i]
        distance = (encounter - st["r"]) @ C.SH
        leg = min(int(np.searchsorted(stations, distance)), len(nodes) - 1)
        previous = st["r"] if leg == 0 else nodes[leg - 1]
        gap = (encounter - previous) @ C.SH
        if gap <= 12.0 or stations[leg] <= 15.0:
            continue
        fraction = (0.60, 0.38, 0.78)[attempt % 3]
        anchor = previous + fraction * (encounter - previous)
        tau = tick * cfg["dt_plan"]
        rock_at_encounter = rp[rock] + rv[rock] * tau + 0.5 * C.accel(rp[rock], rv[rock], frame=True) * tau**2
        away = encounter - rock_at_encounter
        away -= (away @ C.SH) * C.SH
        length = np.linalg.norm(away)
        away = away / length if length > 1e-6 else C.E1
        side = np.cross(C.SH, away)
        radius = hard[rock] + margin + min(3.0 * np.hypot(sp[rock], tau * sv[rock]), 55.0)
        offset = np.clip(radius * (1.0 + 0.35 * attempt), 18.0, 120.0)
        for direction in (away, side, -side, -away):
            knot = anchor + offset * direction
            station = (knot - st["r"]) @ C.SH
            lower = 0.0 if leg == 0 else stations[leg - 1]
            if lower + 6.0 < station < stations[leg] - 6.0:
                yield [*nodes[:leg], knot, *nodes[leg:]], cruise


def route_key(plan):
    nodes, speed = plan
    return speed, tuple(tuple(np.round(node, 1)) for node in nodes)


def make_planner(config=None):
    cfg = CONFIG if config is None else config

    def planner(st, rocks, plan, committed):
        deadline = monotonic() + cfg["musubi_budget_s"]
        direct = [([C.T.copy()], speed) for speed in C.resolve_vcs(cfg)]
        first = rollout(direct, st, rocks, cfg)
        winners = [i for i in range(len(direct)) if safe(first, i, cfg)]
        if winners:
            if plan is not None and len(plan[0]) == 1:
                current = rollout([plan], st, rocks, cfg)
                if safe(current, 0, cfg):
                    return plan, C.info([plan], current, 0, "keep")
            best = min(winners, key=lambda i: first["cost"][i])
            return direct[best], C.info(direct, first, best, "musubi_direct")

        if plan is not None:
            current = rollout([plan], st, rocks, cfg)
            if safe(current, 0, cfg):
                return plan, C.info([plan], current, 0, "keep")

        budget = cfg["musubi_candidate_budget"]
        beam = cfg["musubi_beam_width"]
        seen = {route_key(candidate) for candidate in direct}
        best_route = None
        for attempt in range(1 + cfg["musubi_restarts"]):
            parents, result = direct, first
            depth_limit = cfg["musubi_max_knots"] if best_route is None else best_route[1]["K"] - 1
            for _ in range(depth_limit):
                if monotonic() >= deadline or budget <= 0:
                    break
                children = []
                order = sorted(range(len(parents)), key=lambda i: priority(result, i, cfg))[:beam]
                batch = min(budget, cfg["musubi_batch_size"])
                branches = [insertions(parents[i], result, i, st, rocks, attempt, cfg) for i in order]
                for layer in zip_longest(*branches):
                    for child in layer:
                        if child is None:
                            continue
                        key = route_key(child)
                        if key not in seen:
                            seen.add(key)
                            children.append(child)
                        if len(children) >= batch:
                            break
                    if len(children) >= batch:
                        break
                if not children:
                    break
                budget -= len(children)
                parents = children
                result = rollout(parents, st, rocks, cfg)
                winners = [i for i in range(len(parents)) if safe(result, i, cfg)]
                if winners:
                    best = min(winners, key=lambda i: result["cost"][i])
                    best_route = parents[best], C.info(parents, result, best, "musubi_knots")
                    break
            if monotonic() >= deadline or budget <= 0 or (best_route and best_route[1]["K"] == 1):
                break
        return best_route if best_route else (None, {"action": "musubi_no_safe_route"})

    planner.__name__ = "musubi"
    return planner


musubi = make_planner()
