"""
Systems/Actuation.py - turns a wanted thrust into SetCommand(lin, ang) vectors, in body axes.
Spec: docs/plans/00_FOUNDATION.md section 4. Pure numpy, no sim calls.

Every function returns (lin, ang) ready for SetCommand.
"""
from __future__ import annotations
import numpy as np

from Systems import Core as C

PHI_B = np.radians(40.0)        # boost direction for the body +z burn, from the offline jet solver
ZERO = np.zeros(3)
THRUST = 0.5                    # m/s^2 every burn commands; set per flight from the plan config (max_thrust)


def _wanted_rate(d, kp=0.15, wmax=0.03):
    """Body rate (x, y) that turns body +z toward the unit direction d (body axes)."""
    w = kp * np.array([-d[1], d[0]]); n = np.linalg.norm(w)
    return w * (wmax / n) if n > wmax else w


# ---------------------------------------------------------------------------------------------------------------
# main burn laws (Interfaces.BurnLaw): burn(d, w, state) -> (lin, ang)
# d: unit thrust direction in body axes. w: body rates. state: a dict the caller keeps between ticks.
# ---------------------------------------------------------------------------------------------------------------
def plain_burn(d, w, state, kw=2.0):
    """Body +z burn steered by a plain rate loop: about 0.10 m/s^2. Use it if the boost sector does not
    reproduce in the sim (foundation section 11, check 4)."""
    a = np.clip(kw * (_wanted_rate(d) - w[:2]), -0.03, 0.03)
    return np.array([0.0, 0.0, THRUST]), np.array([a[0], a[1], -0.4 * w[2]])


def boost_burn(d, w, state, kw=2.0, m_b=0.0005, m_max=0.02, m_k=0.02, w_hi=0.001, w_top=0.001,
               half=np.radians(60)):
    """Body +z burn that keeps the torque command inside the boosted sector: about 0.13 m/s^2."""
    e = _wanted_rate(d) - w[:2]
    B = np.array([np.cos(PHI_B), np.sin(PHI_B)]); BP = np.array([-B[1], B[0]])
    e_par, e_perp = e @ B, e @ BP
    if state.setdefault('kick', False):
        if e_par > w_top:
            state['kick'] = False
    elif e_par < -w_hi:
        state['kick'] = True
    if state['kick']:                                    # short strong correction against the boost direction
        a = -m_k * B + np.clip(kw * e_perp, -m_k, m_k) * BP
    else:                                                # stay inside the boosted torque sector
        m = np.clip(kw * max(e_par, 0.0), m_b, m_max)
        ang = np.clip(np.arctan2(kw * e_perp, m), -half, half)
        a = m * (B + np.tan(ang) * BP)
    return np.array([0.0, 0.0, THRUST]), np.array([a[0], a[1], -0.4 * w[2]])


def slew_rate_cmd(d, w, wmax=0.06, kp=0.25, kw=0.6, amax=0.05):
    """Angular acceleration command (body) that turns body +z toward unit d at up to wmax rad/s, from any start
    (also 180 deg away), and damps every other rate, including a tumble after a hit."""
    zc = np.clip(d[2], -1.0, 1.0); err = np.arccos(zc)
    axis = np.array([-d[1], d[0], 0.0]); n = np.linalg.norm(axis)
    axis = axis / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])        # d straight behind: turn about body x
    w_des = axis * min(wmax, kp * err)
    a = kw * (w_des - np.asarray(w, float)); na = np.linalg.norm(a)
    return a * (amax / na) if na > amax else a, err


def steer_burn(d, w, state, gate=np.radians(10.0), cone=np.radians(35.0), rate_gate=0.02, slew_min=3.0):
    """Pointing-gated main burn. Real-sim flight 2026-11-13 showed the boost/plain laws thrusting 40-60 deg off the
    wanted direction for minutes (their torque is limited to a sector) and thrusting through a tumble after a hit.
      aligned (< gate) and calm  -> boost_burn, 0.13 m/s^2 on +z
      inside the cone            -> full +z thrust while slewing with full authority (cos loss < 18%)
      outside, much left to gain -> translate exactly along d (the jet solver keeps the commanded direction) + slew
      outside, little left       -> translate along d, no slew (a slew costs more than the small correction)
    state['ng'] (m/s still to gain) is set by the caller. Never pushes away from d."""
    ang, err = slew_rate_cmd(d, w); ng = state.get('ng', 10.0)
    if err < gate and np.linalg.norm(w) < rate_gate:
        return boost_burn(d, w, state)
    state['kick'] = False
    if err < cone:
        return np.array([0.0, 0.0, THRUST]), ang
    if ng >= slew_min:
        return THRUST * np.asarray(d, float), ang
    damp = ZERO if np.linalg.norm(w) <= 0.01 else np.clip(-0.6 * np.asarray(w, float), -0.05, 0.05)
    return THRUST * np.asarray(d, float), damp


TILT = np.sin(np.radians(1.0))  # the direct burn settles with the thrust this far from body +z toward +x


def direct_burn(d, w, state, kw=2.0, lost=np.radians(25.0), w_lost=0.06):
    """Thrust commanded along d itself, so it points the right way at any attitude: 0.096 to 0.135 m/s^2.
    A slow rate loop turns the body until d sits 1 deg from +z toward +x. That is for thrust and fuel only:
    the jets are axis-aligned (5 deg off an axis costs 9 % more fuel), and the sim's jet solver gives
    0.135 m/s^2 instead of 0.096 once the command has a small +x part, with zero torque (tools/actuator_twin.py).
    While the thrust is far from the flight's long-term direction (state["ref"], set by the flight loop: a sideways
    avoidance push), the body is held on that direction instead, so it is aligned again when the thrust comes back.
    If the loop ever runs away it is switched off for good and only the rates are damped."""
    d = np.asarray(d, float); w = np.asarray(w, float)
    if np.linalg.norm(w) > w_lost:                        # only a rate runaway means the loop failed (it held 1-2 deg
        state["steer_off"] = True                         # in the sim); a swinging thrust direction does not
    if state.get("steer_off"):
        return THRUST * d, -0.5 * w
    aim = state.get("ref", d)                             # the flight's long-term direction (the target line), in body axes
    if d @ aim > np.cos(lost):                            # thrust near it: follow the thrust itself
        aim = d
    if aim[2] < 0.0:                                      # more than 90 deg from +z (never in a normal flight): only damp
        return THRUST * d, -0.5 * w
    a = np.clip(kw * (_wanted_rate(aim - np.array([TILT, 0.0, 0.0]), kp=0.3) - w[:2]), -0.03, 0.03)
    return THRUST * d, np.array([a[0], a[1], -0.4 * w[2]])


# Four-jet body +z commands (tools/jet_bases.py, FINDINGS 7.12). Each command is the exact sum of four +z jets, so the
# sim's jet selector fires those four at full thrust: 0.193 m/s^2 at 4.32 kg per m/s. Each set also turns the body;
# (command lin, command ang, delivered angular acceleration in body axes, rad/s^2).
# Keep every digit: a change of 1e-5 relative already lets the selector pick other jets (0.10-0.18 m/s^2).
ZJETS = (
    ((9.001452270606754e-07, -1.5654267678453691e-07, 0.15654731195788618),
     (-0.04812958247842996, 0.013281539265152376, 0.0026728271923607645), (-0.0593, 0.01637, 0.00329)),   # jets 0 29 31 35
    ((-8.61379526713346e-08, 6.891437786243048e-08, 0.06891383323976878),
     (0.03562833660876962, 0.03484540297408026, -0.004052100944715607), (0.09973, 0.09753, -0.01134)),     # jets 0 28 29 33
    ((2.6828724083213313e-07, 3.156467236313324e-08, 0.06312674131975228),
     (-0.0032138249543314816, -0.04983227804047237, 0.002532862897981518), (-0.00982, -0.15227, 0.00774)), # jets 0 20 21 31
    ((3.2039866234702283e-07, -4.576873154259141e-08, 0.09154254266471135),
     (-0.031154836474305843, -0.0389087380444058, 0.003935259591319391), (-0.06565, -0.08199, 0.00829)),   # jets 0 21 31 35
    ((-5.176678482169121e-07, 6.090518471635941e-08, 0.12180651888133133),
     (0.023714655330190078, 0.043866209531860455, -0.0036566082477075537), (0.03755, 0.06947, -0.00579)),  # jets 0 29 33 35
)
def jets_burn(d, w, state, on=np.radians(3.0), off=np.radians(6.0), i_max=0.01):
    """Plan variant A: body +z at 0.193 m/s^2 that steers by choosing which four-jet set fires.
    Each tick picks the set whose turn keeps the running sum of (wanted - delivered) angular acceleration smallest
    (error diffusion), so on average the body turns as slew_rate_cmd asks while all four jets stay on.
    The sets turn the body at only 0.02 rad/s, and thrust off d turns d away at a sin(err) / |vg|, so they hold
    only a small error: within `off` of d they fly; past it steer_burn's full-authority slew takes over until the
    body is back within `on` and calm. The old 20 deg cone ran away on the engine (FINDINGS 7.15)."""
    ang_des, err = slew_rate_cmd(d, w, wmax=0.02, amax=0.04)
    hold = err < off if state.get("hold") else (err < on and np.linalg.norm(w[:2]) < 0.01)
    state["hold"] = hold
    if not hold:
        state["i_err"] = np.zeros(2)
        return steer_burn(d, w, state)
    dt = state.get("dt", 1 / 60); i_err = state.get("i_err", np.zeros(2))
    outs = np.array([s[2][:2] for s in ZJETS])
    k = int(np.argmin(np.linalg.norm(i_err + (ang_des[:2] - outs) * dt, axis=1)))
    i_err = i_err + (ang_des[:2] - outs[k]) * dt
    n = np.linalg.norm(i_err)
    state["i_err"] = i_err * (i_max / n) if n > i_max else i_err
    lin, ang, _ = ZJETS[k]
    return np.array(lin), np.array(ang)


def axis_attitude(body_axis, ref, w):
    """Align a body axis to the desired bearing and damp body rates."""
    ref = np.asarray(ref, float)
    ref = ref / max(np.linalg.norm(ref), 1e-12)
    err = np.arccos(np.clip(ref @ body_axis, -1.0, 1.0))
    axis = np.cross(body_axis, ref)
    n = np.linalg.norm(axis)
    axis = axis / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])
    ang = 0.6 * (axis * min(0.06, 0.25 * err) - np.asarray(w, float))
    return ang * min(1.0, 0.05 / max(np.linalg.norm(ang), 1e-12))


def target_x_attitude(ref, w):
    return axis_attitude(np.array([1., 0., 0.]), ref, w)


def aligned_axis_burn(axis, acceleration):
    """Slew with no translation, then fire along the selected local axis."""
    def burn(direction, rates, state):
        cosine = np.clip(axis @ direction, -1.0, 1.0)
        # Small simultaneous torques can substantially reduce axial acceleration.
        # Acquire a steady bearing first, then let the pure translation run.
        if state.get("axis_locked", False):
            locked = cosine >= np.cos(np.deg2rad(5.0)) and np.linalg.norm(rates) < 0.015
        else:
            locked = cosine >= np.cos(np.deg2rad(1.5)) and np.linalg.norm(rates) < 0.006
        state["axis_locked"] = locked
        if locked:
            return THRUST * axis, ZERO
        return ZERO, axis_attitude(axis, direction, rates)
    burn.accel = acceleration
    return burn


def target_x_burn(d, w, state):
    """Translate along the requested gain while body +X tracks the mothership."""
    return THRUST * np.asarray(d, float), target_x_attitude(state.get("ref", d), w)


# The loop also applies this attitude policy during trim, evasions and coast.
target_x_burn.attitude = target_x_attitude
target_x_burn.accel = 0.096


def target_z_attitude(ref, w):
    return target_z_attitudes(np.asarray(ref, float)[None], np.asarray(w, float)[None])[0]


def target_z_attitudes(bearings, rates):
    """Target-facing angular control shared by the controller and V13's batched rollout."""
    return target_axis_attitudes(bearings, rates, 2)


def target_axis_attitudes(bearings, rates, forward, max_rate=0.06):
    distance = np.linalg.norm(bearings, axis=1)
    direction = bearings / np.maximum(distance[:, None], 1e-12)
    axis_body = np.eye(3)[forward]
    angle = np.arccos(np.clip(direction[:, forward], -1.0, 1.0))
    axis = np.cross(axis_body, direction)
    length = np.linalg.norm(axis, axis=1)
    fallback = np.eye(3)[(forward + 1) % 3]
    axis = np.where((length > 1e-6)[:, None], axis / np.maximum(length[:, None], 1e-12), fallback)
    wanted = np.minimum(max_rate, 0.25 * angle)
    wanted[distance < 1e-12] = 0.0
    angular = 0.6 * (axis * wanted[:, None] - rates)
    return angular * np.minimum(1.0, 0.05 / np.maximum(np.linalg.norm(angular, axis=1), 1e-12))[:, None]


def coast_pointing_mask(bearings, rates, engaged, cfg, forward=2):
    """Acquire on the outer limits; release only inside both inner limits."""
    distance = np.linalg.norm(bearings, axis=1)
    cosine = bearings[:, forward] / np.maximum(distance, 1e-12)
    angle = np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))
    rate = np.linalg.norm(rates, axis=1)
    enter = (angle >= cfg["coast_heading_enter_deg"]) | (rate >= cfg["coast_rate_enter_rad_s"])
    release = (angle <= cfg["coast_heading_release_deg"]) & (rate <= cfg["coast_rate_release_rad_s"])
    active = np.where(engaged, ~release, enter)
    terminal = cfg.get("coast_terminal_attitude_m", 0.0)
    if terminal > 0.0:
        steady = (angle < cfg["coast_terminal_heading_deg"]) & (rate < cfg["coast_terminal_rate_rad_s"])
        active &= ~((distance <= terminal) & steady)
    return active


def target_z_coast_attitude(ref, w, state, cfg):
    return target_axis_coast_attitude(ref, w, state, cfg, 2)


def target_axis_coast_attitude(ref, w, state, cfg, forward):
    bearings, rates = np.asarray(ref, float)[None], np.asarray(w, float)[None]
    engaged = coast_pointing_mask(bearings, rates, state.get("coast_pointing", False), cfg, forward)[0]
    state["coast_pointing"] = bool(engaged)
    if not engaged:
        return ZERO
    max_rate = cfg.get("coast_max_turn_rate_rad_s", 0.06)
    angular = target_axis_attitudes(bearings, rates, forward, max_rate)[0]
    return angular if np.linalg.norm(angular) >= cfg.get("coast_min_torque", 0.0) else ZERO


X_AXIS = np.array([1., 0., 0.])


def x_forward_main_masks(gain, bearing, rates, acquired, cfg):
    """Hold main thrust until initial acquisition, then prefer aligned +X thrust."""
    angle = np.rad2deg(np.arccos(np.clip(bearing[:, 0] / np.maximum(np.linalg.norm(bearing, axis=1), 1e-12), -1., 1.)))
    rate = np.linalg.norm(rates, axis=1)
    calm = rate <= cfg["x_forward_rate_rad_s"]
    acquired = acquired | ((angle <= cfg["x_forward_acquire_deg"]) & calm)
    limit = cfg["x_forward_cone_deg"]
    on_axis = gain[:, 0] >= np.cos(np.deg2rad(limit)) * np.linalg.norm(gain, axis=1)
    return acquired, acquired & calm & (angle <= limit) & on_axis


def x_forward_attitude(ref, w):
    return target_axis_attitudes(np.asarray(ref, float)[None], np.asarray(w, float)[None], 0)[0]


def x_forward_burn_bearings(gain, docking, cfg):
    """Allow a planned main burn to turn toward its thrust direction within the docking cone."""
    limit = cfg.get("x_forward_burn_heading_deg", 0.0)
    if not limit:
        return docking
    bearing = docking / np.maximum(np.linalg.norm(docking, axis=1)[:, None], 1e-9)
    length = np.linalg.norm(gain, axis=1)
    direction = gain / np.maximum(length[:, None], 1e-9)
    aim = C.turn_toward(bearing, direction, np.deg2rad(limit))
    return np.where((length >= cfg["main_min"])[:, None], aim, docking)


def x_forward_coast_attitude(ref, w, state, cfg):
    return target_axis_coast_attitude(ref, w, state, cfg, 0)


def launch_slew_attitude(ref, w, cfg, early=False):
    """Leave the jets quiet for small rate errors while the initial turn is underway."""
    if early:
        max_rate = cfg["launch_early_slew_rate_rad_s"]
        angular = target_axis_attitudes(ref[None], w[None], 0, max_rate)[0]
    else:
        angular = x_forward_attitude(ref, w)
    angle = np.rad2deg(np.arccos(np.clip(ref[0] / max(np.linalg.norm(ref), 1e-12), -1., 1.)))
    deadband = cfg["launch_slew_cruise_torque"] if angle > 20.0 else cfg["launch_slew_min_torque"]
    return angular if np.linalg.norm(angular) >= deadband else ZERO


def x_forward_precision_command(gain_body, cfg):
    # The official selector interprets small solutions as on-times; scale a
    # desired average acceleration by its nominal 60 Hz actuator period.
    return np.asarray(gain_body, float) / (60.0 * cfg["x_forward_precision_settle_s"])


def x_forward_burn(d, w, state):
    cfg = state["config"]
    remaining = state.get("ng", np.inf)
    ref = state["ref"]
    if cfg.get("x_forward_burn_heading_deg") and np.isfinite(remaining):
        ref = x_forward_burn_bearings((remaining * d)[None], ref[None], cfg)[0]
    ready, axial = x_forward_main_masks(d[None], ref[None], w[None], state.get("x_acquired", False), cfg)
    state["x_acquired"] = bool(ready[0])
    if not ready[0] and not state.get("launch_released"):
        return ZERO, x_forward_attitude(ref, w)
    if remaining < cfg["x_forward_precision_dv_mps"]:
        return x_forward_precision_command(remaining * d, cfg), x_forward_attitude(ref, w)
    if axial[0]:
        return THRUST * X_AXIS, ZERO
    return THRUST * d, x_forward_attitude(ref, w)


x_forward_burn.attitude = x_forward_attitude
x_forward_burn.coast_attitude = x_forward_coast_attitude
x_forward_burn.handles_main_attitude = True
x_forward_burn.accel = 0.17456


def target_z_burn(d, w, state):
    """Translate along the commanded gain while body +Z tracks the docking target."""
    return THRUST * np.asarray(d, float), target_z_attitude(state.get("ref", d), w)


target_z_burn.attitude = target_z_attitude
target_z_burn.coast_attitude = target_z_coast_attitude
target_z_burn.accel = 0.096
boost_burn.accel = 0.130; plain_burn.accel = 0.100; direct_burn.accel = 0.125; steer_burn.accel = 0.125
jets_burn.accel = 0.190
BURNS = {"direct": direct_burn, "steer": steer_burn, "boost": boost_burn, "plain": plain_burn, "jets": jets_burn,
         "target_x": target_x_burn, "target_z": target_z_burn, "x_forward": x_forward_burn,
         "aligned_x": aligned_axis_burn(np.array([1., 0., 0.]), 0.17456),
         "aligned_y": aligned_axis_burn(np.array([0., 1., 0.]), 0.14318)}


def trim_command(vg_body, eps=0.01, w=None, trim_exact=False):
    """Small correction: single-axis thrust by default, or exact-direction thrust + rate damping when enabled."""
    vg_body = np.asarray(vg_body, float); n = np.linalg.norm(vg_body)
    ang = ZERO if w is None or np.linalg.norm(w) <= 0.01 else np.clip(-0.6 * np.asarray(w, float), -0.05, 0.05)
    if trim_exact:
        if n < eps:
            return ZERO, ang
        return THRUST * vg_body / n, ang
    k = int(np.argmax(np.abs(vg_body)))
    if abs(vg_body[k]) < eps:
        return ZERO, ang
    lin = np.zeros(3); lin[k] = THRUST * np.sign(vg_body[k])
    return lin, ang


def coast_command(w):
    """No thrust. Null the body rate only if it has grown past 0.01 rad/s."""
    if np.linalg.norm(w) > 0.01:
        return ZERO, 0.3 * -np.asarray(w)       # with no linear command a small ang is delivered per tick
    return ZERO, ZERO
