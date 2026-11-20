"""Linear-programming path optimizer for Plan 2 (linear_program)."""
from __future__ import annotations
import itertools
import numpy as np
from Systems import Core as C
from Systems.plans.common import _reference, _start_slope, _station_points, _to_nodes
from Systems.plans.plan_2_linear_program.config import CONFIG

YMAX, AIMBOX = CONFIG["ymax"], CONFIG["aimbox"]
DIR8 = [np.array([np.cos(a), np.sin(a)]) for a in np.arange(8) * np.pi / 4]


def _lp_path_detail(st, rocks, base_plan, enum_max=None, cfg=None):
    cfg = CONFIG if cfg is None else cfg
    ds = cfg.get("ds", 60.0)
    ymax = cfg.get("ymax", 60.0)
    aimbox = cfg.get("aimbox", 28.0)
    enum_max = cfg.get("enum_max", 2) if enum_max is None else enum_max
    active_margin = cfg.get("active_margin_m", 25.0)
    sig_mult = cfg.get("safety_sigma", 3.0)
    margin_m = cfg.get("safety_margin_m", 2.0)
    burn_w = cfg.get("burn_turn_weight", 0.15)
    slope_eps = cfg.get("slope_eps", 1e-6)
    dt_plan = cfg.get("dt_plan", C.DT_PLAN)
    sig_ign = cfg.get("sig_ignore", C.SIG_IGNORE)
    t_ign = cfg.get("t_ignore", C.T_IGNORE)

    rp, rv, hard, sp, sv = rocks
    ev = C.score_plans([base_plan], st, rocks, cfg=cfg)
    path, sprof, tprof, vprof = _reference(st, base_plan, cfg=cfg)
    kb = int(np.argmax(vprof))
    s0 = (st['r'] - C.S) @ C.SH
    n = max(int(np.ceil((C.L - s0) / ds)), 1)
    ss = np.r_[s0, s0 + ds * np.arange(1, n), C.L]
    J = len(ss) - 1
    vj = np.interp(ss, sprof, vprof)
    tj = np.interp(ss, sprof, tprof)
    wk = (100 / 75) * C.KF * vj * np.where(tj < tprof[kb], burn_w, C.TRIM_L1)
    y0 = C.lat_of(st['r'])
    slope0 = _start_slope(st)

    tau = ev['kmin'][0] * dt_plan
    pc = rp + rv * tau[:, None] + 0.5 * C.accel(rp, rv) * tau[:, None] ** 2 if len(rp) else rp
    sig = np.sqrt(sp ** 2 + (tau * sv) ** 2)
    need = hard + sig_mult * sig + margin_m
    known = ~((sig > sig_ign) & (tau > t_ign))
    sc = (pc - C.S) @ C.SH
    cl = np.stack([(pc - C.S) @ C.E1, (pc - C.S) @ C.E2], 1)
    pl = np.stack([np.interp(sc, sprof, (path - C.S) @ C.E1), np.interp(sc, sprof, (path - C.S) @ C.E2)], 1)
    off = pl - cl
    dist = np.linalg.norm(off, axis=1)
    act = np.flatnonzero(known & (dist < need + active_margin) & (sc > s0 + 2) & (sc < C.L + 5))
    blocked = sorted([i for i in act if dist[i] < need[i]], key=lambda i: dist[i] - need[i])[:enum_max]

    nv = 4 * J
    iy = lambda j, a: 2 * (j - 1) + a
    iu = lambda j, a: 2 * J + 2 * j + a
    c = np.zeros(nv)
    A = []
    b = []
    for j in range(J):                                   # |direction change at station j| <= u_j, per sideways axis
        if j == 0 and slope0 is None:
            continue
        for a in (0, 1):
            row = np.zeros(nv)
            const = 0.0
            go = ss[j + 1] - ss[j]
            row[iy(j + 1, a)] += 1 / go
            if j >= 1:
                gi = ss[j] - ss[j - 1]
                row[iy(j, a)] -= 1 / go + 1 / gi
                if j >= 2:
                    row[iy(j - 1, a)] += 1 / gi
                else:
                    const += y0[a] / gi
            else:
                const -= y0[a] / go + slope0[a]
            c[iu(j, a)] = wk[j]
            r1 = row.copy(); r1[iu(j, a)] -= 1; A.append(r1); b.append(-const)
            r2 = -row; r2[iu(j, a)] -= 1; A.append(r2); b.append(const)

    def rock_row(i, nvec):                               # nvec . (y(sc_i) - c_i) >= need_i
        j = int(np.clip(np.searchsorted(ss, sc[i]) - 1, 0, J - 1))
        w = np.clip((sc[i] - ss[j]) / (ss[j + 1] - ss[j]), 0, 1)
        row = np.zeros(nv)
        const = -nvec @ cl[i]
        for a in (0, 1):
            row[iy(j + 1, a)] += w * nvec[a]
            if j >= 1:
                row[iy(j, a)] += (1 - w) * nvec[a]
            else:
                const += (1 - w) * nvec[a] * y0[a]
        return -row, const - need[i]

    bounds = [(-ymax, ymax)] * (2 * (J - 1)) + [(-aimbox, aimbox)] * 2 + [(0, None)] * (2 * J)
    fixed = [(i, off[i] / max(dist[i], 1e-6)) for i in act if i not in blocked]
    from scipy.optimize import linprog      # imported here: the sim clock runs during imports (v10 fast start)
    best = None
    for combo in itertools.product(DIR8, repeat=len(blocked)):      # try every side of each blocker
        A2 = list(A); b2 = list(b)
        for i, nvec in fixed + list(zip(blocked, combo)):
            r, rhs = rock_row(i, np.asarray(nvec)); A2.append(r); b2.append(rhs)
        res = linprog(c, A_ub=np.array(A2) if A2 else None, b_ub=np.array(b2) if b2 else None,
                      bounds=bounds, method='highs')
        if res.status == 0 and (best is None or res.fun < best[0]):
            best = (res.fun, res.x)
    if best is None:
        return None, None
    ys = np.vstack([y0, best[1][:2 * J].reshape(J, 2)])
    return _to_nodes(ys, ss, slope_eps=slope_eps), [p.tolist() for p in _station_points(ys, ss)]


def _lp_path(st, rocks, base_plan, enum_max=None, cfg=None):
    return _lp_path_detail(st, rocks, base_plan, enum_max=enum_max, cfg=cfg)[0]


def _lp_solve_detail(st, rocks, vc, start_plan, cfg=None):
    cfg = CONFIG if cfg is None else cfg
    iters = cfg.get("relinearize_iters", 3)
    plan = start_plan
    station_pts = None
    for _ in range(iters):                               # re-linearise around the latest path
        nd, spts = _lp_path_detail(st, rocks, plan, cfg=cfg)
        if nd is None:
            break
        plan = (nd, vc)
        station_pts = spts
    extra = {"station_points": station_pts} if station_pts is not None else {}
    return plan, extra


def _lp_solve(st, rocks, vc, start_plan, cfg=None):
    return _lp_solve_detail(st, rocks, vc, start_plan, cfg=cfg)[0]
