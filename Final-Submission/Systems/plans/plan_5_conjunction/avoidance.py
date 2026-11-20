"""Analytic conjunction avoidance algorithm for Plan 5 (conjunction)."""
from __future__ import annotations
import numpy as np
from Systems import Core as C
from Systems.plans.common import _early, _insert, _one
from Systems.plans.plan_5_conjunction.config import CONFIG


def _push(plan, st, rocks, iters=None, cfg=None):
    """Repeatedly add a point beside the worst rock, on the side the path already favours."""
    cfg = CONFIG if cfg is None else cfg
    iters = cfg.get("push_iters", 3) if iters is None else iters
    p_hit_min = cfg.get("p_hit_min", 1e-4)
    min_along = cfg.get("min_along_ahead", 5.0)
    sig_mult = cfg.get("safety_sigma", 3.0)
    margin_m = cfg.get("safety_margin_m", 3.0)
    p_safe = cfg.get("p_safe", C.P_SAFE)
    dt_plan = cfg.get("dt_plan", C.DT_PLAN)

    rp, rv, hard, sp, sv = rocks
    g = C.accel(rp, rv)
    ev, ni = _one(plan, st, rocks, 'push', cfg=cfg)
    best = (plan, ni)
    for _ in range(iters):
        ph = ev['ph'][0]
        i = int(np.argmax(ph))
        if ph[i] < p_hit_min:
            break
        path = ev['path']
        k = ev['kmin'][0, i]
        tau = k * dt_plan
        pc = rp[i] + rv[i] * tau + 0.5 * g[i] * tau * tau
        if (pc - st['r']) @ C.SH < min_along:
            break
        kk = min(k, len(path) - 2)
        ps = path[min(k, len(path) - 1), 0]
        w = (path[kk + 1, 0] - path[kk, 0]) / dt_plan - (rv[i] + g[i] * tau)
        w = w / np.linalg.norm(w)
        m = (ps - pc) - ((ps - pc) @ w) * w              # miss vector in the encounter plane
        mh = m / np.linalg.norm(m) if np.linalg.norm(m) > 0.5 else C.E1
        need = hard[i] + sig_mult * np.sqrt(sp[i] ** 2 + (tau * sv[i]) ** 2) + margin_m
        W = pc + need * mh                               # turn at the rock, or already halfway there: keep the cheaper
        opts = [(_insert(list(plan[0]), X), plan[1]) for X in (W, _early(W, plan[0][-1], st['r']))]
        evs = [_one(o, st, rocks, 'push', cfg=cfg) for o in opts]
        j = min((0, 1), key=lambda j: (evs[j][1]['ptot'] > p_safe, evs[j][1]['cost']))
        plan = opts[j]
        ev, ni = evs[j]
        if ni['cost'] < best[1]['cost'] or ni['ptot'] <= p_safe < best[1]['ptot']:
            best = (plan, ni)
        if ni['ptot'] <= p_safe:
            break
    return best
