"""
Systems/Core.py - shared flight math: frames, node guidance, plan simulation and the cost function.
Spec: docs/plans/00_FOUNDATION.md sections 1, 2, 6, 7. Pure numpy/scipy, no sim calls.

A plan is (nodes, vc): a list of world points to fly through (the last is the aim point near the
target) and one coast-speed parameter. `rocks` is the tuple (pos, vel, hard_radius, sig_pos, sig_vel).
"""
from __future__ import annotations
import os

import numpy as np

MASS = 3388.7
KF = MASS / (80.0 * 9.8)            # kg of fuel per m/s on one body axis
DMAX = 2.092e5                      # J, damage that ends the run
A_TRIM = 0.096                      # m/s^2, weakest zero-torque body axis
TRIM_L1 = 1.27                      # fuel factor of a sideways push split over body axes
MAIN_MIN = 2.0                      # m/s to gain above which a burn is a main burn (1.0 slewed 100 deg for 1.3 m/s in the replay bench)
R_SH = 4.5                          # m, shuttle reach past its centre (real-sim contact at 4.6 m, 2026-11-13)
SHAPE = 1.3                         # rock mesh reach beyond the radar radius, until a camera measures it
WIN = 50.0                          # m, win radius around the target
EPS_ON = 0.08                       # m/s, coast error that starts a correction burn
TERMINAL = 150.0                    # m to the aim point inside which no new burn starts
LAMBDA_NODE = 0.5                   # points charged per intermediate point
P_SAFE = 1e-3                       # a plan is safe below this total hit probability
RISK_WEIGHT = 1.0                   # expected hit loss is charged this many times over (plan config: risk_weight)
SIG_IGNORE, T_IGNORE = 8.0, 25.0    # ignore a rock while it is this uncertain and this far away in time
SWITCH = 1.5                        # points a new plan must gain to replace an uncommitted one
VCS = (7.9, 8.4, 8.9)               # coast-speed options (speed the coast would have at pad altitude)
DT_PLAN = 0.5                       # s, simulation step used when scoring plans
# Real-sim rotating Phobos frame + residual bias measured across 11 runs on nayeem/real-calibrated:
# a = grav(r) - 2 OMEGA x v + FRAME_B (enabled per-plan via cfg['frame_accel']).
OMEGA = np.array([0.0, 0.0, 2.2803e-4])            # rad/s, Phobos spin (sidereal 7.654 h) about frame z axis
FRAME_B = np.array([0.00215, -0.00100, 0.00148])   # m/s^2, constant extra acceleration near the route

AIMS = [np.zeros(2)] \
     + [20 * np.array([np.cos(a), np.sin(a)]) for a in np.arange(8) * np.pi / 4] \
     + [40 * np.array([np.cos(a), np.sin(a)]) for a in np.arange(12) * np.pi / 6]
AIMS5 = [AIMS[0]] + [AIMS[i] for i in (9, 12, 15, 18)]


def configure(start=(-651.0228484243066, 95.19902620579852, -8239.668001620705),
              target=(-850.0, 120.0, -9530.0), mu=7.075e5, a_burn=0.130):
    """Set the mission geometry. Defaults are the final simconf; the controller calls this with live values."""
    global S, T, L, SH, E1, E2, RS, MU, A_BURN, DIRS
    S = np.array(start, float); T = np.array(target, float)
    L = float(np.linalg.norm(T - S)); SH = (T - S) / L
    E1 = np.cross(SH, [0.0, 1.0, 0.0]); E1 /= np.linalg.norm(E1); E2 = np.cross(SH, E1)
    RS = float(np.linalg.norm(S)); MU = float(mu); A_BURN = float(a_burn)
    DIRS = [np.cos(a) * E1 + np.sin(a) * E2 for a in np.arange(8) * np.pi / 4]


configure()


def grav(r):
    return -MU * r / np.linalg.norm(r, axis=-1, keepdims=True) ** 3


def accel(r, v, frame=False):
    """Acceleration of a coasting body at r with velocity v (Phobos rotating-frame terms when frame=True)."""
    a = grav(r)
    return a + FRAME_B - 2.0 * np.cross(OMEGA, v) if frame else a


def vset(r, vc):
    """Coast speed at radius |r| for speed parameter vc (energy is conserved along a coast)."""
    return np.sqrt(np.maximum(vc ** 2 - 2 * MU * (1 / RS - 1 / np.linalg.norm(r, axis=-1)), 4.0))


def resolve_vc(cfg=None, default=None):
    """Resolve scalar target coast speed (m/s) from accel_till_speed / default_vc."""
    base = VCS[1] if default is None else default
    if not cfg:
        return float(base)
    if cfg.get("accel_till_speed") is not None:
        return float(cfg["accel_till_speed"])
    return float(cfg.get("default_vc", base))


def resolve_vcs(cfg=None, base_vcs=None):
    """Resolve coast-speed candidate list, shifted around accel_till_speed when tuned."""
    base = [float(x) for x in (VCS if base_vcs is None else base_vcs)]
    if not cfg:
        return base
    out = [float(x) for x in cfg.get("vcs", base)]
    sp = cfg.get("accel_till_speed")
    if sp is not None and out:
        sp = float(sp)
        mid = out[1] if len(out) > 1 else out[0]
        if abs(sp - mid) > 1e-6:
            out = [max(2.0, round(sp + (v - mid), 3)) for v in out]
    return out


def lat(o):
    return o[0] * E1 + o[1] * E2


def lat_of(p):
    d = p - S
    return np.array([d @ E1, d @ E2])


def to_gain(r, v, W, vc, frame=False):
    """Velocity still to gain so that a coast from r reaches W. Works on [Nc,3] arrays."""
    d = W - r; dist = np.linalg.norm(d, axis=-1)
    tgo = np.maximum(dist / (0.5 * (vset(r, vc) + vset(W, vc))), 1e-3)
    if frame:
        vm = d / tgo[..., None]
        vreq = vm - (tgo[..., None] / 6) * (accel(r, vm, frame=True) + 2 * accel(r + 0.5 * d, vm, frame=True))
    else:
        vreq = d / tgo[..., None] - (tgo[..., None] / 6) * (grav(r) + 2 * grav(r + 0.5 * d))
    return vreq - v, dist


def coast_pass_near(r, v, point, radius, horizon, frame=False):
    """Check whether the projected ballistic path passes near a point."""
    d = point - r
    distance = np.linalg.norm(d, axis=-1)
    axis = d / np.maximum(distance[..., None], 1e-9)
    closing = np.sum(v * axis, axis=-1)
    a0 = accel(r, v, frame=frame)
    eta = np.minimum(distance / np.maximum(closing, 1e-9), horizon)
    mid_velocity = v + 0.5 * eta[..., None] * a0
    average = (a0 + 2.0 * accel(r + 0.5 * d, mid_velocity, frame=frame)) / 3.0
    closing2 = closing**2 + 2.0 * distance * np.sum(average * axis, axis=-1)
    finish_speed = np.sqrt(np.maximum(closing2, 0.0))
    eta = 2.0 * distance / np.maximum(closing + finish_speed, 1e-9)
    offset = v * eta[..., None] + 0.5 * average * eta[..., None] ** 2 - d
    return ((closing > 0.0) & (finish_speed >= 1.0) & (eta <= horizon)
            & (np.linalg.norm(offset, axis=-1) <= radius))


def late_cruise(cruise, cfg):
    speeds = cfg.get("late_cruise_speeds_mps")
    return np.asarray(cruise) >= min(speeds) if speeds else False


def cruise_needs_thrust(r, v, cruise):
    """Allow the faster approach to build speed before settling into a coast."""
    toward = T - r
    closing = np.sum(v * toward, axis=-1) / np.maximum(np.linalg.norm(toward, axis=-1), 1e-9)
    return closing < vset(r, cruise) - 0.65


OFF_CONE = np.radians(35.0)         # past this the main burn thrusts off body +z (steer_burn's "along d" branch) ...
OFF_ACC, OFF_FUEL = 0.6, 1.1        # ... at 0.6x the thrust for 1.1x the fuel rate (engine runs, FINDINGS 7.17)


def turn_toward(u, g_hat, max_angle):
    """Unit vectors u turned toward unit g_hat by at most max_angle (rad), per row."""
    c = np.clip((u * g_hat).sum(1), -1.0, 1.0); th = np.arccos(c)
    perp = g_hat - c[:, None] * u; n = np.linalg.norm(perp, axis=1)
    perp = np.where(n[:, None] > 1e-9, perp / np.maximum(n, 1e-9)[:, None], 0.0)
    phi = np.minimum(th, max_angle)
    out = np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * perp
    return np.where((th <= max_angle)[:, None], g_hat, out)


def guid_step(r, v, burning, main, idx, nodes, nn, vc, dt, a_burn=None, a_trim=A_TRIM, go=True,
              frame=False, main_min=MAIN_MIN, u=None, w_turn=0.0, main_direction=None,
              ignite_dv=EPS_ON, stop_dv=0.0, coast_docking_radius=0.0,
              coast_waypoint_radius=0.0, build_cruise=False):
    """One guidance + dynamics step for Nc simulated vehicles. Returns the new state, the dv spent and u.
    go: per vehicle, False while its plan holds the burn (a plan may carry a time before which it only coasts).
    u: per vehicle, the main burn's thrust direction (body +z in the world), turned toward the wanted direction at
    w_turn rad/s. Without it the main burn thrusts along the wanted direction at once, so a mid-burn dodge looks as
    cheap as one before the burn; the real shuttle must turn first or thrust off-axis (FINDINGS 7.17)."""
    a_burn = A_BURN if a_burn is None else a_burn
    ar = np.arange(len(r))
    vg, dist = to_gain(r, v, nodes[ar, idx], vc, frame=frame); ng = np.linalg.norm(vg, axis=1)
    last = idx >= nn - 1
    if coast_docking_radius > 0.0 or coast_waypoint_radius > 0.0:
        point = np.where(last[:, None], T, nodes[ar, idx])
        radius = np.where(last, coast_docking_radius, coast_waypoint_radius)
        horizon = np.where(last, 360.0, 35.0)
        go = go & ~((radius > 0.0) & ~np.asarray(build_cruise)
                    & coast_pass_near(r, v, point, radius, horizon, frame=frame))
    burning = burning & go
    ignite = (~burning) & go & (ng > ignite_dv) & ~(last & (dist < TERMINAL))
    main = np.where(ignite, ng > main_min, main); burning = burning | ignite
    a = np.where(main, a_burn, a_trim); fac = np.where(main, 1.0, TRIM_L1); step = a * dt
    if stop_dv > 0.0:
        burning = burning & (ng > stop_dv)
    fin = burning & (ng <= step); on = burning & ~fin
    g_hat = vg / np.maximum(ng, 1e-9)[:, None]
    if main_direction is not None:
        g_hat = np.where(main[:, None], main_direction, g_hat)
    push = step[:, None] * g_hat
    if u is not None:
        turning = on & main
        u = np.where(turning[:, None], turn_toward(u, g_hat, w_turn * dt), u)
        off = turning & ((u * g_hat).sum(1) < np.cos(OFF_CONE))
        push = np.where(off[:, None], OFF_ACC * push, np.where(turning[:, None], step[:, None] * u, push))
        fac = np.where(off, OFF_FUEL, fac)
    dv = np.where(fin, ng * fac, 0.0) + np.where(on, step * fac, 0.0)
    finish = vg if main_direction is None else ng[:, None] * g_hat
    v = v + np.where(fin[:, None], finish, 0.0) + np.where(on[:, None], push, 0.0)
    burning = burning & ~fin
    v = v + (accel(r, v, frame=True) if frame else grav(r)) * dt; r = r + v * dt
    passed = ((nodes[ar, idx] - r) @ SH <= 1.0) & ~last
    return r, v, burning, main, np.where(passed, idx + 1, idx), dv, u


def hold_of(plan):
    """The time before which a plan only coasts: its optional third element, else never."""
    return plan[2] if len(plan) > 2 else -np.inf


def pack(plans):
    """List of (nodes, vc[, hold]) -> nodes [Nc,M,3] padded with each plan's last node, node counts, vc array."""
    M = max(len(p[0]) for p in plans)
    nodes = np.empty((len(plans), M, 3)); nn = np.empty(len(plans), int); vc = np.empty(len(plans))
    for c, (nd, v) in enumerate(p[:2] for p in plans):
        nn[c] = len(nd); vc[c] = v
        for m in range(M):
            nodes[c, m] = nd[min(m, len(nd) - 1)]
    return nodes, nn, vc


def fly(r0, v0, burning0, main0, nodes, nn, vc, dt=DT_PLAN, tmax=420.0, hold=None, cfg=None, u0=None):
    """Simulate Nc plans from one common state until each enters the win sphere.
    hold: per plan, seconds from now before which it only coasts.
    u0: the shuttle's body +z in the world; with cfg w_turn > 0 the main burn has to turn it (guid_step)."""
    frame = bool(cfg.get("frame_accel", False)) if cfg else False
    main_min = float(cfg.get("main_min", MAIN_MIN)) if cfg else MAIN_MIN
    w_turn = float(cfg.get("w_turn", 0.0) or 0.0) if cfg else 0.0
    Nc = len(nn); r = np.repeat(r0[None], Nc, 0); v = np.repeat(v0[None], Nc, 0)
    u = np.repeat(np.asarray(u0, float)[None] / np.linalg.norm(u0), Nc, 0) if (w_turn > 0 and u0 is not None) else None
    burning = np.full(Nc, bool(burning0)); main = np.full(Nc, bool(main0)); idx = np.zeros(Nc, int)
    Nt = int(tmax / dt); path = np.empty((Nt + 1, Nc, 3)); spd = np.empty((Nt + 1, Nc))
    dvs = np.zeros(Nc); arr = np.full(Nc, -1)
    for k in range(Nt + 1):
        path[k] = r; spd[k] = np.linalg.norm(v, axis=1)
        arr[(arr < 0) & (np.linalg.norm(r - T, axis=1) <= WIN)] = k
        if (arr >= 0).all():
            path = path[:k + 1]; spd = spd[:k + 1]; break
        r2, v2, b2, m2, i2, dv, u2 = guid_step(r, v, burning, main, idx, nodes, nn, vc, dt,
                                               go=True if hold is None else k * dt >= hold,
                                               frame=frame, main_min=main_min, u=u, w_turn=w_turn)
        live = arr < 0
        r = np.where(live[:, None], r2, r); v = np.where(live[:, None], v2, v)
        if u is not None:
            u = np.where(live[:, None], u2, u)
        burning = np.where(live, b2, burning); main = np.where(live, m2, main); idx = np.where(live, i2, idx)
        dvs += np.where(live, dv, 0.0)
    return path, spd, arr, dvs


def p_hit(miss, Rh, sig):
    """Probability that the true miss distance is inside Rh: 2-D Gaussian over a disc in the encounter plane."""
    z = (miss - Rh) / sig; out = np.where(z < -6, 1.0, 0.0); mid = (z >= -6) & (z <= 6)
    if mid.any():
        from scipy.special import chndtr          # = scipy.stats.ncx2.cdf, without importing scipy.stats
        out[mid] = chndtr((Rh[mid] / sig[mid]) ** 2, 2, (miss[mid] / sig[mid]) ** 2)
    return out


SWEEP_BLOCK = int(os.environ.get("BATASHA_SWEEP_BLOCK", "48"))   # rocks per block in _sweep (speed only)


def _sweep(path, rp, rv, g_half, arr, dt):
    """Per plan and rock: the closest approach squared over the flight (anywhere within each step, not only at its
    ends: at ~12 m/s relative a step is 6 m), the step it happens at, and the relative speed squared there. Ties keep
    the earliest encounter; nothing counts after a plan's arrival. Works on x, y, z planes instead of [..., 3] arrays
    for speed, summing in einsum's order, (x + z) + y, so the numbers are the same as the einsum version's. Rocks go in
    blocks of SWEEP_BLOCK so the arrays stay in cache (they don't depend on each other)."""
    X = [np.ascontiguousarray(path[:, :, a]) for a in range(3)]
    if len(rp) <= SWEEP_BLOCK:
        return _sweep_block(X, rp, rv, g_half, arr, dt)
    outs = [_sweep_block(X, rp[i:i + SWEEP_BLOCK], rv[i:i + SWEEP_BLOCK], g_half[i:i + SWEEP_BLOCK], arr, dt)
            for i in range(0, len(rp), SWEEP_BLOCK)]
    return tuple(np.concatenate([o[j] for o in outs], axis=1) for j in range(3))


def _sweep_block(X, rp, rv, g_half, arr, dt):
    K, Nc = X[0].shape; Nr = len(rp)
    mind2 = np.full((Nc, Nr), np.inf); kmin = np.zeros((Nc, Nr), int); w2 = np.zeros((Nc, Nr)); thr = mind2 - 1e-6
    tau = np.arange(K) * dt
    P = [rp[:, a][None] + rv[:, a][None] * tau[:, None] + g_half[:, a][None] * (tau * tau)[:, None] for a in range(3)]
    done = arr >= 0; prev = None; dk2 = np.empty((Nc, Nr)); tmp = np.empty((Nc, Nr)); ee = s = None
    for k in range(K):
        diff = [X[a][k][:, None] - P[a][k][None] for a in range(3)]
        if prev is None:
            np.multiply(diff[0], diff[0], out=dk2); np.multiply(diff[2], diff[2], out=tmp); dk2 += tmp
            np.multiply(diff[1], diff[1], out=tmp); dk2 += tmp
        else:
            e = [diff[a] - prev[a] for a in range(3)]
            ee = e[0] * e[0]; ee += e[2] * e[2]; ee += e[1] * e[1]
            pe = prev[0] * e[0]; pe += prev[2] * e[2]; pe += prev[1] * e[1]
            s = np.clip(-pe / np.maximum(ee, 1e-12), 0.0, 1.0)
            c = prev[0] + s * e[0]; np.multiply(c, c, out=dk2)
            c = prev[2] + s * e[2]; np.multiply(c, c, out=tmp); dk2 += tmp
            c = prev[1] + s * e[1]; np.multiply(c, c, out=tmp); dk2 += tmp
        prev = diff
        gone = done & (arr < k)
        if gone.any():
            dk2[gone] = np.inf
        b = dk2 < thr
        if b.any():
            mind2[b] = dk2[b]; thr[b] = mind2[b] - 1e-6
            if ee is not None:
                kmin[b] = k - 1 + (s[b] >= 0.5); w2[b] = ee[b] / (dt * dt)
    return mind2, kmin, w2


def evaluate(path, spd, arr, dvs, nn, rocks, dt=None, fuel0=0.0, t0=0.0, cfg=None):
    """Cost in score points (lower is better) of Nc flown plans against the tracked rocks."""
    dt = (cfg.get("dt_plan", DT_PLAN) if cfg else DT_PLAN) if dt is None else dt
    lam_node = cfg.get("lambda_node", LAMBDA_NODE) if cfg else LAMBDA_NODE
    sig_ign = cfg.get("sig_ignore", SIG_IGNORE) if cfg else SIG_IGNORE
    t_ign = cfg.get("t_ignore", T_IGNORE) if cfg else T_IGNORE
    frame = bool(cfg.get("frame_accel", False)) if cfg else False
    rp, rv, hard, sp, sv = rocks
    Nc = path.shape[1]
    g = accel(rp, rv, frame=True) if (frame and len(rp)) else (grav(rp) if len(rp) else rp)
    g_half = 0.5 * g
    risk_w = cfg.get("risk_weight", RISK_WEIGHT) if cfg else RISK_WEIGHT   # >1: the hit model is overconfident in the sim
    mind2, kmin, w2 = _sweep(path, rp, rv, g_half, arr, dt)
    mind = np.sqrt(mind2)
    sig = np.sqrt(sp[None] ** 2 + (kmin * dt * sv[None]) ** 2)
    ph = p_hit(mind, np.broadcast_to(hard, mind.shape).copy(), sig)
    ign = (sig > sig_ign) & (kmin * dt > t_ign)                           # cannot localize it yet: decide later
    if cfg and cfg.get("ignore_per_rock", False) and len(rp):
        ign = np.broadcast_to(ign.all(axis=0), ign.shape)                # one decision per rock: no plan is safe only
    ph = np.where(ign, 0.0, ph)                                            # by meeting it later than the others (7.16)
    rel = cfg.get("rel_hit", False) if cfg else False                      # damage at the speed relative to the rock is
    spd2 = np.take_along_axis(spd.T, kmin, axis=1) ** 2                    # physically right but dodges more (-2.4 on the
    vhit2 = np.where(w2 > 0, w2, spd2) if rel else spd2                    # replay bench, FINDINGS 7): off by default
    loss = 200 * np.minimum(1.0, 0.5 * MASS * vhit2 / DMAX)
    Tarr = np.where(arr >= 0, t0 + arr * dt, 1e6); fuel = fuel0 + KF * dvs
    clean = 100 * (120 / Tarr - fuel / 75 + 0.5)
    cost = np.where(arr >= 0, -clean + lam_node * (nn - 1) + risk_w * (ph * loss).sum(1), 1e6)
    clear = (mind - hard).min(1) if len(rp) else np.full(Nc, np.inf)
    return dict(cost=cost, clean=clean, T=Tarr, fuel=fuel, ptot=1 - np.prod(1 - ph, axis=1),
                ph=ph, kmin=kmin, mind=mind, clear=clear, path=path)


def score_plans(plans, st, rocks, cfg=None):
    """Simulate and cost a list of plans from the state st = dict(r, v, burning, main, fuel, t)."""
    dt = cfg.get("dt_plan", DT_PLAN) if cfg else DT_PLAN
    tmax = cfg.get("tmax", 420.0) if cfg else 420.0
    nodes, nn, vc = pack(plans); hold = np.array([hold_of(p) for p in plans]) - st['t']
    path, spd, arr, dvs = fly(st['r'], st['v'], st['burning'], st['main'], nodes, nn, vc,
                              dt=dt, tmax=tmax, hold=hold if (hold > 0).any() else None, cfg=cfg, u0=st.get('u'))
    return evaluate(path, spd, arr, dvs, nn, rocks, dt=dt, fuel0=st['fuel'], t0=st['t'], cfg=cfg)


def info(plans, ev, c, action):
    """Summary of candidate c, returned by every planner next to the plan."""
    return dict(cost=float(ev['cost'][c]), ptot=float(ev['ptot'][c]), clean=float(ev['clean'][c]),
                clear=float(ev['clear'][c]), K=len(plans[c][0]) - 1, action=action)


def prune(r_now, rp, hard, tube=220.0):
    """Mask of rocks that can come near the remaining path."""
    rel = rp - S; al = rel @ SH; la = np.linalg.norm(rel - np.outer(al, SH), axis=1)
    return (la < tube + hard) & (al > (r_now - S) @ SH - 80) & (al < L + 60)
