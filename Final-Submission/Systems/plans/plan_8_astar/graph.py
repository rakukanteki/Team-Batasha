"""Priced layered lattice graph shared by Plans 8 (astar) and 9 (cost_lattice)."""
from __future__ import annotations
import numpy as np
from Systems import Core as C
from Systems.plans.common import _reference, _start_slope, _station_points, _to_nodes
from Systems.plans.plan_8_astar.config import CONFIG

DS = CONFIG["ds"]
GRID = np.arange(-CONFIG["grid_half_cells"], CONFIG["grid_half_cells"] + 1) * CONFIG["grid_step_m"]
YY = np.array([(a, b) for a in GRID for b in GRID])
NCELL = len(YY)
GOAL = np.linalg.norm(YY, axis=1) <= CONFIG["goal_radius_m"]
C_V = (100 / 75) * C.KF
EPS_TURN = CONFIG["eps_turn"]


class Graph:
    """The priced lattice from state st, for coast-speed parameter vc. Edges are priced lazily and cached."""

    def __init__(self, st, rocks, vc, cfg=None):
        cfg = CONFIG if cfg is None else cfg
        self.cfg = cfg
        ds = cfg.get("ds", DS)
        half = int(cfg.get("grid_half_cells", 4))
        step = float(cfg.get("grid_step_m", 10.0))
        goal_r = float(cfg.get("goal_radius_m", 40.0))
        merge_thr = float(cfg.get("merge_threshold", 0.5))
        burn_w = float(cfg.get("burn_turn_weight", 0.15))

        if half == CONFIG["grid_half_cells"] and step == CONFIG["grid_step_m"]:
            self.yy = YY
            self.goal = GOAL if goal_r == CONFIG["goal_radius_m"] else (np.linalg.norm(YY, axis=1) <= goal_r)
        else:
            g = np.arange(-half, half + 1) * step
            self.yy = np.array([(a, b) for a in g for b in g])
            self.goal = np.linalg.norm(self.yy, axis=1) <= goal_r
        self.ncell = len(self.yy)
        self.goal_r = goal_r
        self.eps_turn = float(cfg.get("eps_turn", EPS_TURN))
        self.lam_node = float(cfg.get("lambda_node", C.LAMBDA_NODE))
        self.sig_ign = float(cfg.get("sig_ignore", C.SIG_IGNORE))
        self.t_ign = float(cfg.get("t_ignore", C.T_IGNORE))
        self.near_margin_m = float(cfg.get("near_margin_m", 30.0))
        self.near_sigma_mult = float(cfg.get("near_sigma_mult", 6.0))
        self.slope_eps = float(cfg.get("slope_eps", 1e-6))
        self.risk_weight = float(cfg.get("risk_weight", C.RISK_WEIGHT))

        self.rocks = rocks
        rp = rocks[0]
        self.g = C.accel(rp, rocks[1], frame=bool(cfg.get("frame_accel", False))) if len(rp) else rp
        _, sprof, tprof, vprof = _reference(st, ([C.T.copy()], vc), cfg=cfg)
        kb = int(np.argmax(vprof))
        s0 = (st['r'] - C.S) @ C.SH
        n = max(int((C.L - s0) // ds), 1)
        merge = n > 1 and C.L - (s0 + ds * (n - 1)) < merge_thr * ds  # no very short last gap
        stn = np.r_[s0 + ds * np.arange(1, n - 1 if merge else n), C.L]
        self.J = len(stn)
        self.ss = np.r_[s0, stn]
        self.ts = np.r_[0.0, np.interp(stn, sprof, tprof, right=tprof[-1] + (stn[-1] - sprof[-1]) / max(vprof[-1], 1.0))]
        self.vs = np.maximum(np.r_[np.linalg.norm(st['v']), np.interp(stn, sprof, vprof)], 1.0)
        self.v_cruise = max(float(vprof.max()), 1.0)
        self.c_T = 100 * 120 / max(st['t'] + tprof[-1], 1.0) ** 2    # one second, at the planned arrival time
        burn = self.ts < tprof[kb]
        burn[0] = bool(st['burning'] and st['main']) or np.linalg.norm(st['v']) < 0.5
        self.w = C_V * self.vs * np.where(burn, burn_w, C.TRIM_L1)   # points per radian of turn, per station
        self.y0 = C.lat_of(st['r'])
        self.slope0 = _start_slope(st)
        self.Y = [self.y0[None]] + [self.yy] * self.J
        self.X = [st['r'][None]] + [C.S + s * C.SH + self.yy[:, :1] * C.E1 + self.yy[:, 1:] * C.E2 for s in stn]
        self.cache = {}
        self.priced = 0

    def edges(self, j, b):
        """Cost of the edges from cell b at station j to every cell at station j+1: time and expected loss."""
        if (j, b) not in self.cache:
            self.cache[j, b] = self.price(j, np.array([b]))[0]
        return self.cache[j, b]

    def price(self, j, bs):
        """[len(bs), cells at j+1]: the edge costs from cells bs at station j, in one vectorised pass."""
        A = self.X[j][bs]
        B = self.X[j + 1]
        dt = max(self.ts[j + 1] - self.ts[j], 1e-3)
        length = np.linalg.norm(B[None] - A[:, None], axis=2)
        cost = self.c_T * (length - (self.ss[j + 1] - self.ss[j])) / self.vs[j]
        rp, rv, hard, sp, sv = self.rocks
        self.priced += cost.size
        if not len(rp):
            return cost
        t0 = self.ts[j]
        R0 = rp + rv * t0 + 0.5 * self.g * t0 ** 2
        V0 = rv + self.g * t0
        reach = np.linalg.norm(A[:, None] - R0[None], axis=2).min(0)
        near = np.flatnonzero(reach < length.max() + np.linalg.norm(V0, axis=1) * dt + hard + self.near_margin_m
                              + self.near_sigma_mult * np.sqrt(sp ** 2 + ((t0 + dt) * sv) ** 2))
        if not len(near):
            return cost
        u = (B[None] - A[:, None]) / dt                                # shuttle velocity on each edge [b, c]
        d0 = A[:, None, :] - R0[near][None]                            # [b, rock]
        wv = u[:, :, None, :] - V0[near][None, None]                   # [b, c, rock]
        tau = np.clip(-(d0[:, None] * wv).sum(3) / np.maximum((wv * wv).sum(3), 1e-9), 0.0, dt)
        miss = np.linalg.norm(d0[:, None] + wv * tau[..., None], axis=3)
        tm = t0 + tau
        sig = np.sqrt(sp[near] ** 2 + (tm * sv[near]) ** 2)
        ph = C.p_hit(miss.ravel(), np.broadcast_to(hard[near], miss.shape).ravel().copy(), sig.ravel()).reshape(miss.shape)
        ph = np.where((sig > self.sig_ign) & (tm > self.t_ign), 0.0, ph)   # cannot localize it yet
        loss = 200 * np.minimum(1.0, 0.5 * C.MASS * (wv * wv).sum(3) / C.DMAX)
        cost = cost + self.risk_weight * (ph * loss).sum(2)
        if "route_clearance_m" in self.cfg:
            blocked = (miss < hard[near] + self.cfg["route_clearance_m"]).any(axis=2)
            cost += 200 * blocked  # the full rollout, rather than this approximate graph, decides admissibility
        return cost

    def turn(self, j, slope_in, b):
        """Cost of leaving cell b at station j toward every cell at j+1, arriving with slope_in (None: free)."""
        out = (self.yy - self.Y[j][b]) / (self.ss[j + 1] - self.ss[j])
        if slope_in is None:
            return np.zeros(self.ncell)
        k = np.linalg.norm(out - slope_in, axis=-1)
        return self.w[j] * k + self.lam_node * (k > self.eps_turn)

    def slope(self, j, a, b):
        """Slope of the edge from cell a at station j-1 to cell b at station j."""
        return (self.Y[j][b] - self.Y[j - 1][a]) / (self.ss[j] - self.ss[j - 1])

    def h(self, j, b):
        """COST_FUNCTIONS 3.3: the seconds the shortest way to the goal disc still adds. Admissible, consistent."""
        s_rem = C.L - self.ss[j]
        rho = max(np.linalg.norm(self.Y[j][b]) - self.goal_r, 0.0)
        return self.c_T * (np.hypot(s_rem, rho) - s_rem) / self.v_cruise

    def nodes(self, cells):
        """Cells at stations 1..J -> the plan's node list (only where the direction changes, and the aim point)."""
        return _to_nodes(np.vstack([self.y0] + [self.yy[c] for c in cells]), self.ss, slope_eps=self.slope_eps)

    def station_points(self, cells):
        """All 3D coordinates at stations 1..J for the chosen cell sequence."""
        return [p.tolist() for p in _station_points(np.vstack([self.y0] + [self.yy[c] for c in cells]), self.ss)]
