"""v11 uses Plan 10 for waypoint guidance and as the fallback for timed flight."""
from dataclasses import dataclass
from time import monotonic

import numpy as np

from Systems import Core as C
from Systems.Maneuvers import BODY_COMMANDS, Maneuver, Segment, maneuver_of
from Systems.plans.plan_10_real_calibrated import select_real_calibrated
from Systems.plans.plan_11_v11.config import CONFIG

# Durations are defined on the rollout grid. The real controller switches at
# the first control tick at or after each boundary.
STEP = 1.0
HORIZON = 300.0
ENTRY_RADIUS = 40.0


def _unit(vectors):
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-9)


def candidates(st, cfg, active):
    """A small fan around the frame-corrected velocity gain, with bracketed cutoff times."""
    to_target = C.T - st["r"]
    forward = _unit(to_target)
    coast = Segment(HORIZON)
    if active is not None:
        remaining = active.remaining(st["t"])
        if remaining:
            yield remaining, "keep"
    yield (coast,), "coast_entry"
    gain, _ = C.to_gain(st["r"], st["v"], C.T, C.resolve_vc(cfg), frame=True)
    for axis in (C.E1, C.E2):
        # Use both signs of each lateral axis. A 0.4 m/s aiming change is
        # already several tens of metres over the ballistic portion.
        for offset in (-0.8, -0.4, 0.4, 0.8):
            direction = _unit(gain + offset * axis)
            yield from _burns(direction, gain, coast)
    yield from _burns(_unit(gain), gain, coast)
    if not cfg["robust_samples"]:
        return

    # The wider search includes waiting and early lateral changes before the
    # main burn. The raw body commands are only considered with fresh attitude.
    prefixes = [Segment(3.0)]
    if "body" in st and np.linalg.norm(st["rates"]) < 0.03:
        prefixes.extend(Segment(2.0, body_command=i) for i in range(len(BODY_COMMANDS)))
    else:
        prefixes.extend(Segment(2.0, tuple(sign * axis)) for axis in (C.E1, C.E2) for sign in (-1, 1))
    for prefix in prefixes:
        corrected = gain.copy()
        if prefix.body_command is not None:
            corrected -= prefix.duration * (st["body"] @ BODY_COMMANDS[prefix.body_command, 4:7])
        elif prefix.direction is not None:
            corrected -= prefix.duration * C.A_TRIM * np.asarray(prefix.direction)
        if corrected @ forward < -0.5:
            continue
        for sequence, _ in _burns(_unit(corrected), corrected, coast):
            yield (prefix, Segment(2.0), *sequence), "command_sequence"


def _burns(direction, gain, coast):
    if np.linalg.norm(direction) < 0.5:
        return
    nominal = float(gain @ direction) / max(C.A_BURN, 0.06)
    for duration in sorted({round(float(np.clip(nominal + offset, STEP, 85.0))) for offset in (-7, -3, 0, 3, 7)}):
        yield (Segment(duration, tuple(direction)), coast), "ballistic_burn"


@dataclass
class Rollouts:
    path: np.ndarray
    velocity: np.ndarray
    arrival: np.ndarray
    fuel: np.ndarray
    feasible: np.ndarray


def roll_out(st, options, cfg):
    count = len(options)
    steps = int(HORIZON / STEP)
    path = np.empty((steps + 1, count, 3))
    velocity = np.empty_like(path)
    fuel = np.zeros(count)
    arrival = np.full(count, -1, dtype=int)
    inside = np.zeros(count, dtype=bool)
    dwell = np.zeros(count, dtype=bool)
    last_inside = np.zeros(count, dtype=bool)
    terrain = np.ones(count, dtype=bool)
    r = np.tile(st["r"], (count, 1))
    v = np.tile(st["v"], (count, 1))
    pointing = np.tile(st.get("u", C.SH), (count, 1))
    body = st.get("body", np.eye(3))
    body_rate = st.get("rates", np.zeros(3))
    # Build the controls once, then propagate every candidate in one batch.
    directions = np.zeros((steps, count, 3))
    commands = np.full((steps, count), -1, dtype=int)
    for column, (segments, _) in enumerate(options):
        start = 0.0
        for segment in segments:
            end = start + segment.duration
            ticks = slice(int(np.ceil(start / STEP)), int(np.ceil(end / STEP)))
            if segment.body_command is not None:
                commands[ticks, column] = segment.body_command
            elif segment.direction is not None:
                directions[ticks, column] = segment.direction
            start = end

    launch_radius = np.linalg.norm(C.S) + 10.0
    turn_rate = cfg["w_turn"]
    use_attitude = "u" in st and turn_rate > 0
    frame = cfg["frame_accel"]
    for k in range(steps + 1):
        path[k], velocity[k] = r, v
        distance = np.linalg.norm(r - C.T, axis=1)
        newly_arrived = (arrival < 0) & (distance <= C.WIN)
        arrival[newly_arrived] = k
        inside |= distance <= ENTRY_RADIUS
        current_inside = distance <= C.WIN
        dwell |= current_inside & last_inside
        last_inside = current_inside
        terrain &= (arrival >= 0) | (np.linalg.norm(r, axis=1) >= launch_radius)
        if k == steps or (inside & dwell).all():
            path, velocity = path[:k + 1], velocity[:k + 1]
            break

        direction = directions[k]
        thrusting = np.any(direction, axis=1)
        accel = C.A_BURN * direction
        fuel_rate = C.KF * C.A_BURN * thrusting
        if use_attitude and thrusting.any():
            turning = np.flatnonzero(thrusting)
            pointing[turning] = C.turn_toward(pointing[turning], direction[turning], turn_rate * STEP)
            off = thrusting & (np.einsum("ij,ij->i", pointing, direction) < np.cos(C.OFF_CONE))
            aligned = thrusting & ~off
            accel[aligned] = C.A_BURN * pointing[aligned]
            accel[off] *= C.OFF_ACC
            fuel_rate[off] *= C.OFF_FUEL

        # Short raw commands precede the steer burn. Integrate their delivered
        # angular acceleration and the measured starting body rate.
        for column in np.flatnonzero(commands[k] >= 0):
            command = BODY_COMMANDS[commands[k, column]]
            from scipy.spatial.transform import Rotation
            rate = body_rate + np.array([0.0, command[7] * k * STEP, 0.0])
            rotation = body @ Rotation.from_rotvec(
                (k + 0.5) * STEP * body_rate
                + np.array([0.0, 0.5 * command[7] * ((k + 0.5) * STEP) ** 2, 0.0])
            ).as_matrix()
            accel[column] = rotation @ command[4:7]
            fuel_rate[column] = command[8]
            pointing[column] = rotation[:, 2] + STEP * np.cross(rotation @ rate, rotation[:, 2]) * 0.5
            pointing[column] = _unit(pointing[column])
        v_next = v + STEP * (accel + C.accel(r, v, frame=frame))
        r = r + (v + v_next) * (0.5 * STEP)
        v = v_next
        fuel += STEP * fuel_rate * (arrival < 0)
    return Rollouts(path, velocity, arrival, fuel, terrain & inside & dwell & (arrival > 0))


def _contacts(path, arrival, rocks, cfg):
    rp, rv, hard, _, _ = rocks
    frame = cfg["frame_accel"]
    g_half = 0.5 * C.accel(rp, rv, frame=frame)
    last = len(path) if (arrival < 0).any() else min(int(arrival.max()) + 1, len(path))
    d2, encounter, speed2 = C._sweep(path[:last], rp, rv, g_half, arrival, STEP)
    return np.sqrt(d2), encounter, speed2


def _risk(path, arrival, rocks, cfg):
    _, _, hard, sp, sv = rocks
    distance, encounter, speed2 = _contacts(path, arrival, rocks, cfg)
    sigma = np.maximum(np.hypot(sp, encounter * STEP * sv), 1e-6)
    radius = np.broadcast_to(hard, distance.shape)
    return C.p_hit(distance, radius, sigma), speed2, distance - radius


def _scenario_scores(path, arrival, fuel, rocks, st, cfg):
    """Common nominal and antithetic radar perturbations for all finalists."""
    clean = 50.0 + 12000.0 / np.maximum(st["t"] + STEP * arrival, STEP) - (4 / 3) * (st["fuel"] + fuel)
    clean = np.where(arrival > 0, clean, -1e6)
    rp, rv, hard, sp, sv = rocks
    rng = np.random.default_rng(11)
    scores = []
    for sample in range(cfg["robust_samples"]):
        if monotonic() >= st.get("deadline", np.inf):
            return None
        if sample == 0:
            delta_r, delta_v = np.zeros_like(rp), np.zeros_like(rv)
        elif sample % 2:
            delta_r = rng.normal(size=rp.shape) * sp[:, None]
            delta_v = rng.normal(size=rv.shape) * sv[:, None]
        else:
            delta_r, delta_v = -delta_r, -delta_v
        sampled = rp + delta_r, rv + delta_v, hard, sp, sv
        distance, encounter, speed2 = _contacts(path, arrival, sampled, cfg)
        damage = np.where(distance < hard, 0.5 * C.MASS * speed2, 0.0)
        total_damage = damage.sum(axis=1)
        loss = clean - 200.0 * total_damage / C.DMAX
        for i in np.flatnonzero((arrival > 0) & (total_damage > C.DMAX)):
            order = np.argsort(encounter[i])
            fatal = np.flatnonzero(np.cumsum(damage[i, order]) > C.DMAX)[0]
            k = encounter[i, order[fatal]]
            # Approximate fuel at contact by prorating the remaining flight fuel.
            fraction = k / max(arrival[i], 1)
            spent = st["fuel"] + fraction * fuel[i]
            distance = np.linalg.norm(path[k, i] - C.T)
            loss[i] = 50.0 - 100.0 * distance / C.L - (4 / 3) * spent
            loss[i] -= 200.0 * damage[i, order[:fatal + 1]].sum() / C.DMAX
        scores.append(loss)
    samples = np.array(scores or [clean])
    return samples.mean(axis=0), samples.min(axis=0)


def _choose_timed(st, rocks, baseline, active, cfg):
    plans = list(candidates(st, cfg, active))
    rollout = roll_out(st, plans, cfg)
    feasible = np.flatnonzero(rollout.feasible)
    if not len(feasible) or monotonic() >= st.get("deadline", np.inf):
        return None

    clean = 50 + 12000 / (st["t"] + rollout.arrival[feasible] * STEP) - (4 / 3) * (st["fuel"] + rollout.fuel[feasible])
    finalists = feasible[np.argsort(-clean, kind="stable")[:14]]
    if active is not None and feasible[0] == 0 and plans[0][1] == "keep" and 0 not in finalists:
        finalists = np.r_[finalists, 0]
    path = rollout.path[:, finalists]
    arrival, fuel = rollout.arrival[finalists], rollout.fuel[finalists]
    probability, speed2, clearance = _risk(path, arrival, rocks, cfg)
    p_hit = 1.0 - np.prod(1.0 - probability, axis=1)
    nominal = 50.0 + 12000.0 / (st["t"] + STEP * arrival) - (4 / 3) * (st["fuel"] + fuel)
    impact_loss = 200.0 * (0.5 * C.MASS * speed2) / C.DMAX
    expected = nominal - (impact_loss * probability).sum(axis=1)
    safe = np.flatnonzero(p_hit <= cfg["p_safe"])
    ranked = safe[np.argsort(-expected[safe], kind="stable")[:5]]
    if not len(ranked):
        return None

    base, base_info = baseline
    if cfg["robust_samples"] and len(rocks[0]):
        nodes, counts, speed = C.pack([base])
        base_path, _, base_arrival, base_dv = C.fly(
            st["r"], st["v"], st["burning"], st["main"], nodes, counts, speed,
            dt=STEP, tmax=HORIZON, hold=np.array([C.hold_of(base) - st["t"]]), cfg=cfg, u0=st.get("u"),
        )
        # Join the baseline with the finalists so every rollout sees the same perturbations.
        length = max(len(path), len(base_path))
        base_path = np.pad(base_path, ((0, length - len(base_path)), (0, 0), (0, 0)), mode="edge")
        selected = np.pad(path[:, ranked], ((0, length - len(path)), (0, 0), (0, 0)), mode="edge")
        combined = np.concatenate((selected, base_path), axis=1)
        scenarios = _scenario_scores(combined, np.r_[arrival[ranked], base_arrival],
                                     np.r_[fuel[ranked], C.KF * base_dv], rocks, st, cfg)
        if scenarios is None:
            return None
        mean, worst = scenarios
        baseline_score = min(-base_info["cost"], mean[-1])
        ranking = sorted(range(len(ranked)), key=lambda i: (-round(mean[i], 1), -worst[i]))
        scores = [(ranked[i], mean[i]) for i in ranking]
    else:
        baseline_score = -base_info["cost"]
        scores = [(i, expected[i]) for i in ranked]

    for i, score in scores:
        if monotonic() >= st.get("deadline", np.inf):
            return None
        index = finalists[i]
        segments, action = plans[index]
        depth = np.flatnonzero(np.linalg.norm(rollout.path[:, index] - C.T, axis=1) <= ENTRY_RADIUS)[0]
        end = min(max(depth, rollout.arrival[index] + 1) + 1, len(rollout.path))
        maneuver = Maneuver(st["t"], segments, STEP, rollout.path[:end, index].copy(),
                            rollout.velocity[:end, index].copy(), base, cfg["frame_accel"], st.get("body"))
        margin = 0.0 if action == "keep" else cfg["maneuver_margin"]
        if score <= baseline_score + margin or not maneuver.check(st["t"], st["r"], st["v"], rocks):
            continue
        info = dict(cost=float(-score), clean=float(nominal[i]), ptot=float(p_hit[i]),
                    clear=float(clearance[i].min(initial=np.inf)), K=0, action=action,
                    stats={"segments_s": [s.duration for s in segments]})
        return ([C.T.copy()], base[1], st["t"], maneuver), info
    return None


def make_planner(config=None):
    cfg = CONFIG if config is None else config

    def planner(st, rocks, plan, committed):
        active = maneuver_of(plan)
        fallback = active.fallback if active is not None else plan
        baseline = select_real_calibrated(st, rocks, fallback, committed, cfg)
        altitude = np.linalg.norm(st["r"]) - np.linalg.norm(C.S)
        if cfg["timed_burns"] and altitude >= 30.0 and monotonic() < st.get("deadline", np.inf):
            return _choose_timed(st, rocks, baseline, active, cfg) or baseline
        return baseline

    return planner


v11 = make_planner()
