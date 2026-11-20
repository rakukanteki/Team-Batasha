"""Lattice dynamic-programming search algorithm for Plan 3 (lattice)."""
from __future__ import annotations
import numpy as np
from Systems import Core as C
from Systems.plans.common import _reference, _start_slope, _station_points, _to_nodes
from Systems.plans.plan_3_lattice.config import CONFIG

DS = CONFIG["ds"]
GRID = np.arange(-CONFIG["grid_half_cells"], CONFIG["grid_half_cells"] + 1) * CONFIG["grid_step_m"]
YY = np.array([(a, b) for a in GRID for b in GRID])
NCELL = len(YY)
NQ = CONFIG["nq"]
W_DEV = CONFIG["w_dev"]


def _grid(cfg):
    half = int(cfg.get("grid_half_cells", 4))
    step = float(cfg.get("grid_step_m", 8.0))
    if half == CONFIG["grid_half_cells"] and step == CONFIG["grid_step_m"]:
        return YY, NCELL
    g = np.arange(-half, half + 1) * step
    yy = np.array([(a, b) for a in g for b in g])
    return yy, len(yy)


def _lattice_detail(st, rocks, vc, prev=None, cfg=None):
    cfg = CONFIG if cfg is None else cfg
    ds = cfg.get("ds", 60.0)
    nq = int(cfg.get("nq", 6))
    w_dev = cfg.get("w_dev", 0.02)
    goal_r = cfg.get("goal_radius_m", 40.0)
    merge_thr = cfg.get("merge_threshold", 0.5)
    near_margin = cfg.get("near_margin_m", 45.0)
    near_lat_base = cfg.get("near_lat_base_m", 46.0)
    sig_mult = cfg.get("safety_sigma", 3.0)
    margin_m = cfg.get("safety_margin_m", 3.0)
    burn_w = cfg.get("burn_turn_weight", 0.15)
    tw_num = cfg.get("time_weight_num", 12000.0)
    slope_eps = cfg.get("slope_eps", 1e-6)
    lam_node = cfg.get("lambda_node", C.LAMBDA_NODE)
    sig_ign = cfg.get("sig_ignore", C.SIG_IGNORE)
    t_ign = cfg.get("t_ignore", C.T_IGNORE)

    yy, ncell = _grid(cfg)
    rp, rv, hard, sp, sv = rocks
    g = C.accel(rp, rv) if len(rp) else rp
    _, sprof, tprof, vprof = _reference(st, ([C.T.copy()], vc), cfg=cfg)
    kb = int(np.argmax(vprof))
    s0 = (st['r'] - C.S) @ C.SH
    n = max(int((C.L - s0) // ds), 1)
    merge = n > 1 and C.L - (s0 + ds * (n - 1)) < merge_thr * ds      # avoid a very short last gap
    stn = np.r_[s0 + ds * np.arange(1, n - 1 if merge else n), C.L]
    J = len(stn)
    tj = np.interp(stn, sprof, tprof, right=tprof[-1] + (stn[-1] - sprof[-1]) / max(vprof[-1], 1.0))
    vj = np.interp(stn, sprof, vprof)
    wt = tw_num / max(st['t'] + tprof[-1], 1.0) ** 2
    wk = (100 / 75) * C.KF * vj * np.where(tj < tprof[kb], burn_w, C.TRIM_L1)
    y0 = C.lat_of(st['r'])
    slope0 = _start_slope(st)
    cells = lambda s: C.S + s * C.SH + yy[:, 0:1] * C.E1 + yy[:, 1:2] * C.E2
    X = [st['r'][None]] + [cells(s) for s in stn]
    Ys = [y0[None]] + [yy] * J
    ts = np.r_[0.0, tj]
    ss = np.r_[s0, stn]
    q = (np.arange(nq) + 0.5) / nq

    edge = []
    for j in range(J):                                   # edge layer j: station j -> j+1
        A, B = X[j], X[j + 1]
        tm = ts[j] + q * (ts[j + 1] - ts[j])
        tmid = 0.5 * (ts[j] + ts[j + 1])
        r = np.zeros((len(A), len(B)))
        if len(rp):
            pm = rp + rv * tmid + 0.5 * g * tmid ** 2
            al = (pm - C.S) @ C.SH
            la = np.linalg.norm((pm - C.S) - np.outer(al, C.SH), axis=1)
            near = np.flatnonzero((al > ss[j] - hard - near_margin) & (al < ss[j + 1] + hard + near_margin)
                                  & (la < near_lat_base + hard + near_margin + np.linalg.norm(y0)))
            if len(near):
                seg = A[:, None, None, :] * (1 - q)[None, None, :, None] + B[None, :, None, :] * q[None, None, :, None]
                pr = rp[near][None] + rv[near][None] * tm[:, None, None] + 0.5 * g[near][None] * tm[:, None, None] ** 2
                d = np.linalg.norm(seg[:, :, :, None, :] - pr[None, None], axis=4).min(2)        # [a, b, rock]
                sig = np.sqrt(sp[near] ** 2 + (tmid * sv[near]) ** 2)
                sh = (d.shape[0] * d.shape[1], len(near))
                ph = C.p_hit(d.reshape(sh), np.broadcast_to(hard[near], sh).copy(), np.broadcast_to(sig, sh).copy()).reshape(d.shape)
                if tmid > t_ign:
                    ph = ph * (sig <= sig_ign)[None, None, :]
                r = (ph * 200 * min(1.0, 0.5 * C.MASS * vj[j] ** 2 / C.DMAX)).sum(2)
        r = r + wt * (np.linalg.norm(B[None] - A[:, None], axis=2) - (ss[j + 1] - ss[j])) / max(vj[j], 1.0)
        if prev is not None:                             # pull toward the plan being flown
            r = r + w_dev * np.linalg.norm(Ys[j + 1] - prev(ss[j + 1])[None], axis=1)[None, :]
        edge.append(r)

    # backward pass. V[a, b]: best cost-to-go at station j+1 in cell b, having come from a at station j
    V = np.broadcast_to(np.where(np.linalg.norm(yy, axis=1) <= goal_r, 0.0, 1e9)[None, :], (len(Ys[J - 1]), ncell)).copy()
    choice = [None] * J
    for j in range(J - 1, 0, -1):
        gi = ss[j] - ss[j - 1]
        go = ss[j + 1] - ss[j]
        kv = np.linalg.norm((yy[None, None, :, :] - yy[None, :, None, :]) / go
                            - (yy[None, :, None, :] - Ys[j - 1][:, None, None, :]) / gi, axis=3)   # [a, b, c]
        tot = wk[j - 1] * kv + lam_node * (kv > 1e-9) + (edge[j] + V)[None, :, :]
        choice[j] = tot.argmin(2)
        V = tot.min(2)
    k0 = np.zeros(ncell) if slope0 is None else \
        (100 / 75) * C.KF * np.linalg.norm(st['v']) * (burn_w if st['burning'] and st['main'] else C.TRIM_L1) \
        * np.linalg.norm((yy - y0[None]) / (ss[1] - ss[0]) - slope0[None], axis=1)
    seq = [int((edge[0][0] + k0 + V[0]).argmin())]
    a = 0
    for j in range(1, J):
        c = int(choice[j][a, seq[-1]])
        a = seq[-1]
        seq.append(c)
    ys = np.vstack([y0] + [yy[c] for c in seq])
    nodes = _to_nodes(ys, ss, slope_eps=slope_eps)
    lat_pts = [p.tolist() for p in _station_points(ys, ss)]
    return nodes, dict(lattice_points=lat_pts, cells=list(seq))


def _lattice(st, rocks, vc, prev=None, cfg=None):
    return _lattice_detail(st, rocks, vc, prev=prev, cfg=cfg)[0]


def _previous(plan, st):
    pts = [st['r']] + list(plan[0])
    s = np.array([(p - C.S) @ C.SH for p in pts])
    l = np.array([C.lat_of(p) for p in pts])
    return lambda x: np.array([np.interp(x, s, l[:, 0]), np.interp(x, s, l[:, 1])])
