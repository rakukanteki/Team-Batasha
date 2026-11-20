"""Velocity-obstacle candidate search algorithm for Plan 7 (velocity_obstacle).

Defaults reproduce nayeem's v5 (real sim 13/13 arrivals, 0 damage, mean 44.4; docs/REAL_SIM_RESULTS.md): aim points,
side points at 1-3 lookaheads, 'carry on along v, then turn' and the flown plan nudged, chosen as the cheapest among
plans under p_safe, with keep_band hysteresis. Extras, each a config switch for A/B: other cruise speeds ("speeds"),
holding the burn ("holds"), a wide fallback ("wide_fallback"), and pure expected-cost choice ("selection": "cost").
"""
from __future__ import annotations
import numpy as np
from Systems import Core as C
from Systems.plans.plan_3_lattice.config import CONFIG as LATTICE
from Systems.plans.plan_3_lattice.lattice import _lattice
from Systems.plans.plan_7_velocity_obstacle.config import CONFIG

HOLDS = CONFIG["holds"]
HOLDS_WIDE = CONFIG["holds_wide"]
P_WIDE = CONFIG["p_wide"]


def select_velocity_obstacle(st, rocks, plan, committed, cfg=None):
    cfg = CONFIG if cfg is None else cfg
    vcs = C.resolve_vcs(cfg)
    vc = C.resolve_vc(cfg, vcs[1] if len(vcs) > 1 else vcs[0]) if plan is None else plan[1]
    t = st['t']
    la = max(cfg.get("lookahead_min_m", 60.0), cfg.get("lookahead_time_s", 15.0) * max(np.linalg.norm(st['v']), cfg.get("lookahead_min_v", 2.0)))
    stop_margin = cfg.get("stop_ahead_margin_m", 80.0)
    p_safe = cfg.get("p_safe", C.P_SAFE)
    s0 = (st['r'] - C.S) @ C.SH

    cands = [([C.T + C.lat(o)], vc) for o in C.AIMS]; kind = ['aim'] * len(cands)
    if cfg.get("speeds", False):
        cands += [([C.T + C.lat(o)], v) for v in vcs if v != vc for o in C.AIMS5]; kind += ['speed'] * (len(cands) - len(kind))
    cands += [([C.T + C.lat(o)], vc, t + h) for h in cfg.get("holds", ()) for o in C.AIMS5]
    kind += ['hold'] * (len(cands) - len(kind))
    for mult in cfg.get("la_mults", (1.0,)):           # also earlier, cheaper dodges: dv ~ offset / lookahead time
        if s0 + mult * la < C.L - stop_margin:
            base = C.S + (s0 + mult * la) * C.SH + C.lat(C.lat_of(st['r']))
            for d in C.DIRS:
                for m in cfg.get("side_offsets", (4.0, 8.0, 16.0, 32.0)):
                    cands.append(([base + m * d, C.T.copy()], vc)); kind.append('side')
    if cfg.get("nudge", False):
        vs = st['v'] @ C.SH
        if vs > 1.0:                                     # carry on along the velocity already paid for, then turn
            for mult in cfg.get("la_mults", (1.0,)):
                W = st['r'] + st['v'] * (mult * la / vs)
                if (W - C.S) @ C.SH < C.L - stop_margin:
                    cands += [([W + m * d, C.T.copy()], vc) for d in [np.zeros(3)] + C.DIRS for m in ((0.0,) if not d.any() else (4.0, 8.0))]
                    kind += ['drift'] * (len(cands) - len(kind))
        if plan is not None:                             # the plan being flown, nudged: small changes stay cheap
            nodes = list(plan[0])
            for j in sorted({0, len(nodes) - 1}):
                for d in C.DIRS:
                    for m in (4.0, 8.0, 16.0):
                        nd = list(nodes); nd[j] = nd[j] + m * d
                        if j == len(nodes) - 1 and np.linalg.norm(nd[j] - C.T) > 45:
                            continue
                        cands.append((nd, vc)); kind.append('nudge')
    if cfg.get("corridor", False) and t < cfg.get("corridor_t", 80.0):    # nayeem's corridor: route around a dense cluster
        pad = cfg.get("corridor_pad_m", 0.0)                                  # as a whole, Plan 3's lattice on a wide grid
        rk = rocks if not pad else (rocks[0], rocks[1], rocks[2] + pad, rocks[3], rocks[4])
        cands.append((_lattice(st, rk, vc, None, dict(LATTICE, grid_step_m=cfg.get("corridor_step_m", 16.0))), vc))
        kind.append('corridor')
    if plan is not None:
        cands.append(plan); kind.append('keep')

    ev = C.score_plans(cands, st, rocks, cfg=cfg); cost = ev['cost'].copy()
    if cfg.get("selection", "p_safe") == "p_safe":       # v5: cheapest among the safe plans
        ok = ev['ptot'] <= p_safe
        if plan is not None:
            cost[-1] -= cfg.get("switch_margin", C.SWITCH)                     # prefer the velocity already flown
            ok[-1] = ev['ptot'][-1] <= cfg.get("keep_band", 3.0) * p_safe       # p_hit noise must not flip plans
        safe = np.flatnonzero(ok); pool = safe if len(safe) else np.arange(len(cands))
        c = int(pool[np.argmin(cost[pool])])
    else:                                                # "cost": cheapest in expected score, risk priced in
        if plan is not None:
            cost[-1] -= cfg.get("keep_committed_bonus", 0.5) if committed else cfg.get("switch_margin", C.SWITCH)
        c = int(np.argmin(cost))
    best = (cands[c], C.info(cands, ev, c, kind[c]), cost[c])
    if cfg.get("wide_fallback", False) and ev['ptot'][c] > cfg.get("p_wide", P_WIDE):   # still risky: look further afield
        wide = [([C.T.copy()], v, t + h) for h in cfg.get("holds_wide", HOLDS_WIDE) for v in vcs]; kw = ['wait'] * len(wide)
        for ahead in (la, 2 * la):
            if s0 + ahead < C.L - stop_margin:
                base = C.S + (s0 + ahead) * C.SH + C.lat(C.lat_of(st['r']))
                wide += [([base + m * d, C.T.copy()], vc) for d in C.DIRS for m in cfg.get("wide_side_offsets", (48.0, 72.0))]
        kw += ['around'] * (len(wide) - len(kw))
        ev2 = C.score_plans(wide, st, rocks, cfg=cfg); c2 = int(np.argmin(ev2['cost']))
        if ev2['cost'][c2] < best[2]:
            return wide[c2], C.info(wide, ev2, c2, kw[c2])
    return best[0], best[1]
