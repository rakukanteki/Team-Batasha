"""Plan 10: real-calibrated Nayeem v5 planner (calibrated plan7 + drift/nudge/keep_band)."""
from __future__ import annotations

import numpy as np

from Systems import Core as C
from Systems.plans.plan_10_real_calibrated.config import CONFIG

LA_MULTS = tuple(CONFIG["la_mults"])
KEEP_BAND = float(CONFIG["keep_band"])
SIDE_M = tuple(CONFIG["side_offsets"])
NUDGE = bool(CONFIG["nudge"])


def apf_waypoint(st, rocks, cfg):
    """Attract toward the target and repel sideways from predicted nearby rock surfaces.

    Returns a local waypoint; the ordinary Plan 10 rollout checks its complete route.
    A deterministic sideways direction breaks symmetry for a rock directly ahead.
    """
    r, v = st["r"], st["v"]
    ahead = float(cfg.get("apf_lookahead_m", 100.0))
    influence = float(cfg.get("apf_influence_m", 60.0))
    horizon = float(cfg.get("apf_horizon_s", 20.0))
    rp, rv, hard, sp, sv = rocks
    delta = C.T - r
    distance = np.linalg.norm(delta)
    if distance <= C.TERMINAL:
        return None
    toward = delta / distance
    rel = rp - r
    closing = v - rv
    tau = np.clip(np.sum(rel * closing, axis=1) / np.maximum(np.sum(closing**2, axis=1), 1e-9), 0, horizon)
    encounter = rel - closing * tau[:, None]
    clearance = np.linalg.norm(encounter, axis=1) - hard - 2 * np.sqrt(sp**2 + (tau * sv)**2)
    forward = rel @ toward
    use = (forward > 0) & (forward < ahead + influence) & (clearance < influence)
    if not use.any():
        return None
    sideways = -encounter[use]
    sideways -= (sideways @ toward)[:, None] * toward
    n = np.linalg.norm(sideways, axis=1)
    fallback = C.E1 - (C.E1 @ toward) * toward
    if np.linalg.norm(fallback) < 1e-6:
        fallback = C.E2 - (C.E2 @ toward) * toward
    fallback /= np.linalg.norm(fallback)
    away = np.where((n > 1e-6)[:, None], sideways / np.maximum(n[:, None], 1e-6), fallback)
    gap = np.maximum(clearance[use], 1.0)
    weight = np.maximum(1 / gap - 1 / influence, 0) / gap**2
    offset = float(cfg.get("apf_gain", 20000.0)) * np.sum(weight[:, None] * away, axis=0)
    limit = float(cfg.get("apf_max_offset_m", 32.0))
    offset *= min(1.0, limit / max(np.linalg.norm(offset), 1e-9))
    return r + min(ahead, distance - C.TERMINAL) * toward + offset


def select_real_calibrated(st, rocks, plan, committed, cfg=None):
    """Selects among aim, lookahead side-step, drift, plan-nudge, and keep candidates."""
    cfg = CONFIG if cfg is None else cfg
    vcs = C.resolve_vcs(cfg)
    vc = C.resolve_vc(cfg, vcs[1] if len(vcs) > 1 else vcs[0]) if plan is None else plan[1]
    la_min = float(cfg.get("lookahead_min_m", 60.0))
    la_time = float(cfg.get("lookahead_time_s", 15.0))
    la_min_v = float(cfg.get("lookahead_min_v", 2.0))
    stop_margin = float(cfg.get("stop_ahead_margin_m", 80.0))
    la_mults = cfg.get("la_mults", (1.0, 2.0, 3.0))
    side_m = cfg.get("side_offsets", (4.0, 8.0, 16.0, 32.0))
    drift_m = cfg.get("drift_offsets", (4.0, 8.0))
    nudge_m = cfg.get("nudge_offsets", (4.0, 8.0, 16.0))
    nudge = bool(cfg.get("nudge", True))
    keep_band = float(cfg.get("keep_band", 3.0))
    p_safe = float(cfg.get("p_safe", C.P_SAFE))
    sw_margin = float(cfg.get("switch_margin", C.SWITCH))
    nudge_target_max = float(cfg.get("nudge_target_max_offset_m", 45.0))

    la = max(la_min, la_time * max(np.linalg.norm(st["v"]), la_min_v))
    s0 = (st["r"] - C.S) @ C.SH
    cands = [([C.T + C.lat(o)], vc) for o in C.AIMS]
    kind = ["aim"] * len(cands)
    if cfg.get("apf", False):
        waypoint = apf_waypoint(st, rocks, cfg)
        if waypoint is not None:
            cands.append(([waypoint, C.T.copy()], vc))
            kind.append("apf")

    for mult in la_mults:
        if s0 + mult * la < C.L - stop_margin:
            base = C.S + (s0 + mult * la) * C.SH + C.lat(C.lat_of(st["r"]))
            for d in C.DIRS:
                for m in side_m:
                    cands.append(([base + m * d, C.T.copy()], vc))
                    kind.append("side")

    if nudge:
        vs = st["v"] @ C.SH
        if vs > 1.0:
            for mult in la_mults:
                w_pt = st["r"] + st["v"] * (mult * la / vs)
                if (w_pt - C.S) @ C.SH < C.L - stop_margin:
                    cands += [
                        ([w_pt + m * d, C.T.copy()], vc)
                        for d in [np.zeros(3)] + C.DIRS
                        for m in ((0.0,) if not d.any() else drift_m)
                    ]
                    kind += ["drift"] * (len(cands) - len(kind))
        if plan is not None:
            nodes = list(plan[0])
            for j in sorted({0, len(nodes) - 1}):
                for d in C.DIRS:
                    for m in nudge_m:
                        nd = list(nodes)
                        nd[j] = nd[j] + m * d
                        if j == len(nodes) - 1 and np.linalg.norm(nd[j] - C.T) > nudge_target_max:
                            continue
                        cands.append((nd, vc))
                        kind.append("nudge")

    if plan is not None:
        cands.append(plan)
        kind.append("keep")

    ev = C.score_plans(cands, st, rocks, cfg=cfg)
    cost = ev["cost"].copy()
    ok = ev["ptot"] <= p_safe
    if plan is not None:
        cost[-1] -= sw_margin
        ok[-1] = ev["ptot"][-1] <= keep_band * p_safe

    safe = np.flatnonzero(ok)
    pool = safe if len(safe) else np.arange(len(cands))
    c = int(pool[np.argmin(cost[pool])])
    return cands[c], C.info(cands, ev, c, kind[c])
