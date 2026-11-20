"""
Systems/Radar.py - radar world model: which entry is which rock, and one filter per rock.
Spec: docs/plans/00_FOUNDATION.md section 5. Pure numpy/scipy, no sim calls.
"""
from __future__ import annotations
import numpy as np
from Systems import Core as C

POS_VAR = 4.0            # (2 m)^2 radar position noise per axis
RR_VAR = 0.02 ** 2       # (0.02 m/s)^2 radar range-rate noise
Q_VEL = 0.02 ** 2        # (m/s)^2 per s velocity random walk. Real-sim rocks are ballistic within ~1 m over 60 s except
#                          for rock-on-rock bumps (4-13% per minute). With the start-up burst handled below, 0.02 keeps
#                          the claimed sigma honest (97% inside 3 sigma, tools/tracker_bench.py) and predicts 20 s ahead
#                          to p90 7 m; 0.002 was overconfident, 0.1 too noisy (replay bench mean score 37 vs 19)
V0_VAR = 3.0 ** 2        # (m/s)^2 velocity uncertainty of a new track (real rocks reach 3-17 m/s in the first 10 s)
Q_BURST, BURST_T = 1.0 ** 2, 12.0    # extra random walk while the sim's start-up push is still on (t < BURST_T)
ADAPT_GATE = 16.0        # chi-square (3 dof) innovation gate for the bump response (0 = off): a rock bumped by a
#                          neighbour is re-learned instead of trusted (removed a replay-bench hit at Q_VEL 0.01)
# the original 'kalman' tracker's settings (tasmirz's configurable tracker defaults to them)
VEL_VAR = 4.0 ** 2       # (m/s)^2 prior on a new rock's velocity, per axis
Q_KICK = 0.5 ** 2        # velocity random walk per second while the rocks may still be accelerating
T_SETTLE = 12.0          # s on our clock after which the rocks coast
JUMP = 9.0               # squared position innovation, in its own variance, that marks a disturbed rock
JUMP_DV = 3.0            # on a jump the velocity may be off by this times |innovation| / time since the last look
# v10 (nayeem 85d512e) values below. Off by default: turn on per plan with rr_gate 100, rr_adapt 0.2, adapt_2pt 2.
# nayeem's 109-field bench: hit flights 5 -> 1 of 436. Ours, 12 generated fields x 2 seeds (2026-11-16): Plan 7 -0.2 +/- 0.9,
# Plan 10 one hit in 24 (field 4, seed 22) against none with v5 (FINDINGS 7.13)
RR_GATE = 100.0           # chi-square (1 dof) range-rate gate for the bump response (0 = off). A bump shows in range rate
#                          (0.02 m/s noise) a cycle before the position gate sees it; on a fire the line-of-sight
#                          velocity is re-learned from this same measurement (lab RESULTS2, crash case 161108 s22)
RR_TF = 1.0             # transverse share of that re-learn (0 = line of sight only, 1 = all axes by the same amount)
BUMP_K = 0.0             # rock-rock bump prediction (0 = off): two rocks whose predicted paths pass within r1 + r2 +
#                          BUMP_K * sigma in the next BUMP_H s get a kick-sized velocity uncertainty from the contact
#                          time on. Real bumps are seen ~20 s ahead (lab RESULTS2); the kick direction is not predictable
BUMP_H, BUMP_R, BUMP_E = 20.0, 400.0, 0.5    # horizon s, pair search radius m, restitution for the kick size
RR_ADAPT = 0.2           # per-rock range-rate noise (0 = off): each rock's variance follows its own range-rate innovations
#                          (moving average with this weight, minus what the filter's own uncertainty explains), kept in
#                          [RR_VAR, RR_MAX]. A wobbling rock gets the loose 0.1^2 that fixed its tracking (lab sweep28-33);
#                          ballistic rocks keep the tight RR_VAR. Lab pick: RR_ADAPT 0.2 with RR_GATE 100, RR_TF 1 (sweep36/39)
RR_MAX = 0.1 ** 2
ADAPT_2PT = 2            # second gate fire in a row: the kick came before the previous fix, so re-estimate the velocity
#                          from the last two measured positions (v = dz / gap, with its covariance) instead of inflating P.
#                          A transverse kick is invisible to range rate and the plain re-learn corrects it slowly (rock 433)
RR_JOINT = 0.0           # joint bump gate (0 = off): position d2 (3 dof) + range-rate d2 (1 dof) above this fires the
#                          position re-learn even when each test alone stays under its gate (field 074006 rock 433:
#                          d2 10.7 < 16 and range rate 1.0 m/s under a loosened per-rock variance; hit 13 s later)


class WorldModel:
    """Feed it every radar read; ask it for all rocks predicted to any time."""

    def __init__(
        self,
        *,
        rr_var=RR_VAR,
        q_vel=Q_VEL,
        v0_var=VEL_VAR,
        q_burst=Q_KICK,
        burst_t=T_SETTLE,
        adapt_gate=None,
        r_sh=None,
        frame_accel=False,
        continuous_burst=False,
        rr_gate=0.0,
        rr_adapt=0.0,
        adapt_2pt=0,
    ):
        self.keys = {}          # bytes of a published 3x3 entry -> track index
        self.offset = None      # our clock minus the radar's clock
        self.n = 0; self.bumps = 0; self.rbumps = 0; self.tcheck = -np.inf; self.npred = 0
        self.rr_gate = rr_gate
        self.rr_adapt = rr_adapt
        self.adapt_2pt = adapt_2pt
        self.rr_var = rr_var
        self.q_vel = q_vel
        self.v0_var = v0_var
        self.q_burst = q_burst
        self.burst_t = burst_t
        self.adapt_gate = adapt_gate
        self.r_sh = r_sh
        self.frame_accel = frame_accel
        self.continuous_burst = continuous_burst

    def configure_from_plan(self, cfg):
        """Apply per-plan tracker overrides when provided in a plan config dict."""
        if not cfg:
            return
        for key, attr in (
            ("rr_var", "rr_var"),
            ("q_vel", "q_vel"),
            ("v0_var", "v0_var"),
            ("q_burst", "q_burst"),
            ("burst_t", "burst_t"),
            ("adapt_gate", "adapt_gate"),
            ("r_sh", "r_sh"),
            ("frame_accel", "frame_accel"),
            ("rr_gate", "rr_gate"),
            ("rr_adapt", "rr_adapt"),
            ("adapt_2pt", "adapt_2pt"),
        ):
            if key in cfg and cfg[key] is not None:
                setattr(self, attr, cfg[key])

    def _alloc(self, n):
        self.n = n
        self.x = np.zeros((n, 6)); self.P = np.zeros((n, 6, 6)); self.t = np.zeros(n)
        self.count = np.zeros(n, int); self.dsum = np.zeros(n)
        self.rcam = np.full(n, np.nan)      # radius measured by a camera, where available (Vision.py)
        self.tb = np.full(n, np.inf); self.kick = np.zeros(n)   # predicted bump time and kick variance per axis
        self.rv = np.full(n, self.rr_var)   # range-rate noise variance per rock (rr_adapt)
        self.zl = np.zeros((n, 3)); self.tl = np.full(n, np.nan); self.fired = np.zeros(n, bool)   # last fix (ADAPT_2PT)

    def read(self, entries, t_now, own_at):
        """entries: 3x3 arrays from GetNearestDebrisLocations(500). own_at(t) -> (pos, vel) of the shuttle at
        our-clock time t. Returns [(track, entry)] for the entries that changed since the last read."""
        ent = {m.tobytes(): m for m in (np.ascontiguousarray(e, dtype=float) for e in entries)}
        if not ent:
            return []
        lag = t_now - max(m[2, 0] for m in ent.values())
        self.offset = lag if self.offset is None else min(self.offset, lag)
        if not self.keys:
            self._alloc(len(ent))
            changed = []
            for i, (k, m) in enumerate(ent.items()):
                self.keys[k] = i; changed.append((i, m))
        else:
            gone = [k for k in self.keys if k not in ent]            # re-measured since the last read
            new = [k for k in ent if k not in self.keys]
            if not gone or len(gone) != len(new):
                return []
            gm = [np.frombuffer(k).reshape(3, 3) for k in gone]; nm = [ent[k] for k in new]
            ts_old = np.array([m[2, 0] for m in gm])
            if (ts_old > 0).all() and len(set(ts_old)) == len(ts_old):
                pairs = zip(np.argsort(ts_old), np.argsort([m[2, 0] for m in nm]))   # same order in, same order out
            else:                                                    # first cycle: nearest neighbour inside the changed set
                A = np.array([m[0] + own_at(m[2, 0] + self.offset)[0] for m in gm])
                B = np.array([m[0] + own_at(m[2, 0] + self.offset)[0] for m in nm])
                dA = np.array([m[1, 2] for m in gm]); dB = np.array([m[1, 2] for m in nm])
                cost = np.linalg.norm(A[:, None] - B[None], axis=2) \
                    + 20 * np.abs(dA[:, None] - dB[None]) / np.maximum(dA[:, None], 1)
                from scipy.optimize import linear_sum_assignment
                pairs = zip(*linear_sum_assignment(cost))
            changed = []
            for a, b in pairs:
                i = self.keys.pop(gone[a]); self.keys[new[b]] = i; changed.append((i, ent[new[b]]))
        self._update(changed, own_at)
        if BUMP_K and t_now > BURST_T and t_now - self.tcheck >= 1.0:
            self.tcheck = t_now; self._bumps(t_now)
        return changed

    def _bumps(self, t):
        """Rock pairs about to touch: arm a kick at the predicted contact time (consumed by the next fix after it)."""
        old = (self.tb > t) | (self.tb <= self.t)          # not reached yet (re-armed below if still predicted) or used up
        self.tb[old] = np.inf; self.kick[old] = 0.0
        idx = np.flatnonzero(self.count >= 3)
        if len(idx) < 2:
            return
        p, v, _, sp, _ = self.predict(t); r = self.radius()
        from scipy.spatial import cKDTree
        pr = cKDTree(p[idx]).query_pairs(BUMP_R, output_type="ndarray")
        if not len(pr):
            return
        a, b = idx[pr[:, 0]], idx[pr[:, 1]]; dp = p[b] - p[a]; dv = v[b] - v[a]
        R = r[a] + r[b] + BUMP_K * np.hypot(sp[a], sp[b])
        A = (dv * dv).sum(1); B = 2 * (dp * dv).sum(1); D = B * B - 4 * A * ((dp * dp).sum(1) - R * R)
        hit = (D > 0) & (A > 1e-9) & (B < 0)
        tc = np.where(hit, (-B - np.sqrt(np.maximum(D, 0))) / (2 * np.maximum(A, 1e-9)), np.inf)
        hit &= tc < BUMP_H
        if not hit.any():
            return
        a, b, tc, dp, dv = a[hit], b[hit], np.maximum(tc[hit], 0), dp[hit], dv[hit]
        n = dp + dv * tc[:, None]; n /= np.linalg.norm(n, axis=1)[:, None]; vn = np.abs((dv * n).sum(1))
        ma, mb = r[a] ** 3, r[b] ** 3
        for j, m_other, m_self in ((a, mb, ma), (b, ma, mb)):
            k = ((1 + BUMP_E) * m_other / (m_self + m_other) * vn) ** 2
            np.minimum.at(self.tb, j, t + tc); np.maximum.at(self.kick, j, k)
        self.npred += int(hit.sum())

    def _predict(self, i, t):
        dt = t - self.t[i]; p = self.x[i, :3]; v = self.x[i, 3:]
        g = C.accel(p, v, frame=True) if self.frame_accel else C.grav(p)
        x = np.concatenate([p + v * dt[:, None] + 0.5 * g * dt[:, None] ** 2, v + g * dt[:, None]], 1)
        F = np.tile(np.eye(6), (len(i), 1, 1)); F[:, 0, 3] = F[:, 1, 4] = F[:, 2, 5] = dt
        Q = np.zeros((len(i), 6, 6)); ad = np.abs(dt)        # a radar entry can be older than the last camera fix
        if self.continuous_burst:
            overlap = np.minimum(np.maximum(t, self.t[i]), self.burst_t) - np.minimum(np.minimum(t, self.t[i]), self.burst_t)
            q = self.q_vel + self.q_burst * overlap / np.maximum(ad, 1e-9)
        else:
            q = np.where(self.t[i] < self.burst_t, self.q_burst, self.q_vel)
        for a in range(3):
            Q[:, a, a] = q * ad ** 3 / 3; Q[:, a, a + 3] = Q[:, a + 3, a] = q * ad ** 2 / 2
            Q[:, a + 3, a + 3] = q * ad
        if BUMP_K:                                       # a predicted bump between the last fix and t: an impulse at tb
            tau = t - self.tb[i]; on = (self.tb[i] > self.t[i]) & (tau > 0)
            if on.any():
                k = np.where(on, self.kick[i], 0.0); ta = np.where(on, tau, 0.0)
                for a in range(3):
                    Q[:, a, a] += k * ta ** 2; Q[:, a, a + 3] += k * ta; Q[:, a + 3, a] += k * ta; Q[:, a + 3, a + 3] += k
        return x, F @ self.P[i] @ F.transpose(0, 2, 1) + Q

    def _update(self, changed, own_at):
        if not changed:
            return
        i = np.array([c[0] for c in changed]); M = np.array([c[1] for c in changed])
        t = M[:, 2, 0] + self.offset
        own = [own_at(tt) for tt in t]
        op = np.array([o[0] for o in own]); ov = np.array([o[1] for o in own])
        z = M[:, 0, :] + op; rr = M[:, 1, 1]
        new = self.count[i] == 0
        if new.any():
            j = i[new]; self.x[j, :3] = z[new]; self.x[j, 3:] = 0.0
            self.P[j] = np.diag([POS_VAR] * 3 + [self.v0_var] * 3); self.t[j] = t[new]
        x, P = self._predict(i, t)
        H = np.zeros((len(i), 3, 6)); H[:, 0, 0] = H[:, 1, 1] = H[:, 2, 2] = 1
        two = np.zeros(len(i), bool)
        y = z - x[:, :3]; S = P[:, :3, :3] + POS_VAR * np.eye(3)
        rr0 = self.rr_var
        if self.adapt_gate is not None:
            if self.adapt_gate:                          # a rock bumped by another: re-learn it instead of trusting the old path
                d2 = np.einsum("ni,nij,nj->n", y, np.linalg.inv(S), y)
                bad = (d2 > self.adapt_gate) & ~new
                if RR_JOINT:
                    rel = x[:, :3] - op; rho = np.linalg.norm(rel, axis=1); u = rel / rho[:, None]; vrel = x[:, 3:] - ov
                    h = (vrel * u).sum(1); Hr = np.concatenate([(vrel - h[:, None] * u) / rho[:, None], u], 1)[:, None, :]
                    d2r = (rr - h) ** 2 / ((Hr @ P @ Hr.transpose(0, 2, 1))[:, 0, 0] + (self.rv[i] if self.rr_adapt else rr0))
                    bad |= (d2 + d2r > RR_JOINT) & (self.count[i] >= 3) & (t > self.burst_t)
                if self.adapt_2pt:
                    g2 = t - self.tl[i]; two = bad & (self.fired[i] | (self.adapt_2pt == 2)) & (g2 >= 2.0)
                    if two.any():                        # mode 2: on every fire; the kick may sit inside this gap, so
                        g = g2[two]; I3 = np.eye(3)      # keep the plain response's extra velocity variance on top
                        x[two, :3] = z[two]; x[two, 3:] = (z[two] - self.zl[i][two]) / g[:, None]
                        ex = (y[two] ** 2).sum(1) / 3 / g ** 2 if self.adapt_2pt == 2 else 0.0
                        P[two] = 0.0; P[two, :3, :3] = POS_VAR * I3
                        P[two, 3:, 3:] = (2 * POS_VAR / g ** 2 + ex)[:, None, None] * I3
                        P[two, :3, 3:] = (POS_VAR / g)[:, None, None] * I3; P[two, 3:, :3] = P[two, :3, 3:]
                    self.fired[i] = bad; bad = bad & ~two
                if bad.any():
                    gap = np.maximum(np.abs(t - self.t[i]), 1.0)[bad]; e2 = (y[bad] ** 2).sum(1) / 3
                    P[bad, :3, :3] += e2[:, None, None] * np.eye(3)
                    P[bad, 3:, 3:] += (e2 / gap ** 2)[:, None, None] * np.eye(3)
                    self.bumps += int(bad.sum())
        else:
            jump = (np.einsum("ni,nij,nj->n", y, np.linalg.inv(S), y) > JUMP) & ~new
            if jump.any():                                   # off its path: its velocity is unknown again
                dt = np.maximum(t[jump] - self.t[i[jump]], 1.0)
                for a in range(3):
                    P[jump, a + 3, a + 3] += (JUMP_DV ** 2 / 3) * (y[jump] ** 2).sum(1) / dt ** 2
        if self.rr_gate:                                 # range rate says the velocity changed: open it along the line of sight
            rel = x[:, :3] - op; rho = np.linalg.norm(rel, axis=1); u = rel / rho[:, None]; vrel = x[:, 3:] - ov
            h = (vrel * u).sum(1)
            Hr = np.concatenate([(vrel - h[:, None] * u) / rho[:, None], u], 1)[:, None, :]
            e = rr - h; Sr = (Hr @ P @ Hr.transpose(0, 2, 1))[:, 0, 0] + (self.rv[i] if self.rr_adapt else rr0)
            bad = (e * e / Sr > self.rr_gate) & (self.count[i] >= 3) & (t > self.burst_t)
            if bad.any():
                ub = u[bad]; uu = ub[:, :, None] * ub[:, None, :]; A = uu + RR_TF ** 2 * (np.eye(3) - uu)
                e2 = e[bad] ** 2; gap = np.maximum(np.abs(t - self.t[i]), 1.0)[bad]
                P[bad, 3:, 3:] += e2[:, None, None] * A
                P[bad, :3, :3] += (e2 * gap ** 2 / 3)[:, None, None] * A    # the bump came some time in the last gap
                self.rbumps += int(bad.sum())
        K = P @ H.transpose(0, 2, 1) @ np.linalg.inv(H @ P @ H.transpose(0, 2, 1) + POS_VAR * np.eye(3))
        upd = ~new & ~two                                # a two-point re-estimate already holds this fix
        x = np.where(upd[:, None], x + (K @ (z - x[:, :3])[:, :, None])[:, :, 0], x)
        P = np.where(upd[:, None, None], (np.eye(6) - K @ H) @ P, P)
        rel = x[:, :3] - op; rho = np.linalg.norm(rel, axis=1); u = rel / rho[:, None]; vrel = x[:, 3:] - ov
        h = (vrel * u).sum(1)
        Hr = np.concatenate([(vrel - h[:, None] * u) / rho[:, None], u], 1)[:, None, :]
        s0 = (Hr @ P @ Hr.transpose(0, 2, 1))[:, 0, 0]
        if self.rr_adapt:
            e2 = (rr - h) ** 2; on = (self.count[i] >= 2) & (t > self.burst_t)
            est = (1 - self.rr_adapt) * self.rv[i] + self.rr_adapt * np.maximum(e2 - s0, 0.0)
            self.rv[i] = np.where(on, np.clip(est, self.rr_var, RR_MAX), self.rv[i])
        Sr = s0 + (self.rv[i] if self.rr_adapt else self.rr_var)
        Kr = (P @ Hr.transpose(0, 2, 1))[:, :, 0] / Sr[:, None]
        x = x + Kr * (rr - h)[:, None]; P = P - Kr[:, :, None] * (Hr @ P)
        self.x[i] = x; self.P[i] = P; self.t[i] = t; self.count[i] += 1; self.dsum[i] += M[:, 1, 2]
        self.zl[i] = z; self.tl[i] = t

    def sight(self, i, t, H, e, var):
        """Camera fixes (Vision.Consensus). For rocks i (k) at time t: e (k x 2) is what the camera saw minus what
        the filter expected, H (k x 2 x 3) maps a position error to it, var (k) is its noise variance."""
        i = np.asarray(i, int); x, P = self._predict(i, np.full(len(i), float(t)))
        H = np.concatenate([H, np.zeros_like(H)], 2)
        S = H @ P @ H.transpose(0, 2, 1) + np.asarray(var)[:, None, None] * np.eye(2)
        K = P @ H.transpose(0, 2, 1) @ np.linalg.inv(S)
        self.x[i] = x + (K @ np.asarray(e)[:, :, None])[:, :, 0]; self.P[i] = (np.eye(6) - K @ H) @ P; self.t[i] = t

    def radius(self):
        """Best radius per rock: the radar's mean, or the camera's where it has seen more."""
        rad = self.dsum / np.maximum(self.count, 1) / 2
        return np.where(np.isnan(self.rcam), rad, np.maximum(self.rcam, rad))

    def set_radius(self, i, radius):
        """A camera-measured outline radius. The largest seen is kept; it replaces the blanket SHAPE margin."""
        self.rcam[i] = radius if np.isnan(self.rcam[i]) else max(self.rcam[i], radius)

    def predict(self, t):
        """All rocks at time t: (pos, vel, hard_radius, sig_pos, sig_vel)."""
        i = np.arange(self.n); x, P = self._predict(i, np.full(self.n, float(t)))
        sp = np.sqrt(np.trace(P[:, :3, :3], axis1=1, axis2=2) / 3)
        sv = np.sqrt(np.trace(P[:, 3:, 3:], axis1=1, axis2=2) / 3)
        radius = self.dsum / np.maximum(self.count, 1) / 2
        r_sh = C.R_SH if self.r_sh is None else float(self.r_sh)
        hard = np.where(np.isnan(self.rcam), C.SHAPE * radius, np.maximum(self.rcam, radius)) + r_sh
        return x[:, :3], x[:, 3:], hard, sp, sv


class CalibratedWorldModel(WorldModel):
    """Real-sim calibrated radar tracker (nayeem/real-calibrated): honest Q_VEL, start-up burst, bump gate."""

    def __init__(self):
        super().__init__(
            rr_var=0.02 ** 2,
            q_vel=0.02 ** 2,
            v0_var=3.0 ** 2,
            q_burst=1.0 ** 2,
            burst_t=12.0,
            adapt_gate=16.0,
            r_sh=4.5,
            frame_accel=True,
            continuous_burst=True,
        )


TRACKERS = {"kalman": WorldModel, "calibrated": CalibratedWorldModel}
