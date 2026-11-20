"""Search and repair algorithms for Plan 1 (fewest_points)."""
from __future__ import annotations
import numpy as np
from Systems import Core as C
from Systems.plans.common import _beside, _best, _early, _insert
from Systems.plans.plan_1_fewest_points.config import CONFIG


def search(st, rocks, vcs=None, kmax=None, cfg=None):
    """Zero points first; one or two only if nothing with fewer is safe."""
    cfg = CONFIG if cfg is None else cfg
    vcs = C.resolve_vcs(cfg) if vcs is None else vcs
    kmax = cfg.get("kmax", 2) if kmax is None else kmax
    p_safe = cfg.get("p_safe", C.P_SAFE)
    top_seeds = cfg.get("top_seeds", 3)

    plans = [([C.T + C.lat(o)], vc) for vc in vcs for o in C.AIMS]
    ev = C.score_plans(plans, st, rocks, cfg=cfg)
    found = [_best(plans, ev, 'search')]
    seeds = [(plans[c], ev, c) for c in np.argsort(ev['cost'])[:top_seeds]]
    for _ in range(kmax):
        if min(i['ptot'] for _, i in found) <= p_safe:
            break
        nxt = [(_insert(nd[:-1] + [A], X), vc)
               for (nd, vc), e, c in seeds for W in _beside(e, c, rocks, st, cfg=cfg)
               for A in (C.T + C.lat(o) for o in C.AIMS5) for X in (W, _early(W, A, st['r']))]
        if not nxt:
            break
        ev = C.score_plans(nxt, st, rocks, cfg=cfg)
        found.append(_best(nxt, ev, 'search'))
        seeds = [(nxt[c], ev, c) for c in np.argsort(ev['cost'])[:top_seeds]]
    safe = [f for f in found if f[1]['ptot'] <= p_safe]
    return min(safe or found, key=lambda f: f[1]['cost'])


def repair(st, rocks, plan, cfg=None):
    """Smallest change that keeps the plan safe: keep or drop a point, nudge, insert one point, least bad."""
    cfg = CONFIG if cfg is None else cfg
    p_safe = cfg.get("p_safe", C.P_SAFE)
    nudges = cfg.get("nudge_distances", (3.0, 6.0, 12.0, 24.0))
    aim_max = cfg.get("aim_max_offset", 45.0)
    move_pen = cfg.get("move_penalty", 0.02)
    drop_margin = cfg.get("drop_margin", 0.2)

    nodes, vc = plan
    drops = [(list(nodes[:j]) + list(nodes[j + 1:]), vc) for j in range(len(nodes) - 1)]
    if len(nodes) > 2:
        drops.append((list(nodes[-1:]), vc))
    keep_or_drop = [(list(nodes), vc), *drops]
    base = C.score_plans(keep_or_drop, st, rocks, cfg=cfg)
    if base['ptot'][0] <= p_safe:
        simpler = [c for c in range(1, len(keep_or_drop))
                   if base['ptot'][c] <= p_safe and base['cost'][c] < base['cost'][0] - drop_margin]
        c = min(simpler, key=lambda c: base['cost'][c]) if simpler else 0
        return keep_or_drop[c], C.info(keep_or_drop, base, c, 'drop' if c else 'keep')

    cands = []
    moved = []
    for j in sorted({0, len(nodes) - 1}):                # the next point, and the aim point
        for d in C.DIRS:
            for m in nudges:
                nd = list(nodes)
                nd[j] = nd[j] + m * d
                if j == len(nodes) - 1 and np.linalg.norm(nd[j] - C.T) > aim_max:
                    continue
                cands.append((nd, vc))
                moved.append(m)
    if cands:
        nudged = C.score_plans(cands, st, rocks, cfg=cfg)
        ev = {key: np.concatenate((base[key][:1], nudged[key], base[key][1:]))
              for key in ("cost", "ptot", "clean", "clear", "ph", "kmin")}
    else:
        ev = base
    cands = [keep_or_drop[0], *cands, *drops]
    moved = np.array([0.0, *moved, *([0.0] * len(drops))])

    safe = np.flatnonzero(ev['ptot'] <= p_safe)
    if len(safe):
        c = int(safe[np.argmin(ev['cost'][safe] + move_pen * moved[safe])])
        return cands[c], C.info(cands, ev, c, 'nudge')
    ins = [(_insert(list(nodes), X), vc)
           for W in _beside(ev, 0, rocks, st, cfg=cfg)
           for X in (W, _early(W, nodes[-1], st['r']))]
    best = _best(cands, ev, 'least-bad')
    if ins:
        ev2 = C.score_plans(ins, st, rocks, cfg=cfg)
        s2 = np.flatnonzero(ev2['ptot'] <= p_safe)
        if len(s2):
            c = int(s2[np.argmin(ev2['cost'][s2])])
            return ins[c], C.info(ins, ev2, c, 'insert')
        alt = _best(ins, ev2, 'least-bad')
        if alt[1]['cost'] < best[1]['cost']:
            best = alt
    return best
