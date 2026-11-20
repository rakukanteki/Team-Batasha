"""
Systems/Flight.py - the flight brain: world model, replanning schedule, node guidance and the guard.
Spec: docs/plans/00_FOUNDATION.md sections 6, 8, 9. No sim calls: the offline bench and the real
controller both drive this same class.

    flight = Flight(planner, tracker)              # names from Planners.PLANNERS and Radar.TRACKERS
    flight.radar(entries, t)                       # at about 5 Hz
    mode, vec = flight.decide(t, r, v, fuel, dt)   # every tick
mode: "coast", "main" (burn toward vec), "trim" (burn vec on single body axes) or "evade" (full thrust along the
unit vector vec, any direction: the guard and the barrier filter).
A plan is (nodes, vc), (nodes, vc, hold) for a hold before burn, or (nodes, vc, hold, maneuver).
For a timed "body_command", body_command indexes the legal command pair and vec is its body linear command.
Otherwise vec is expressed in world axes.
"""
from __future__ import annotations
import threading
from collections import deque
from time import monotonic
import numpy as np
from Systems import Core as C
from Systems import Planners as P
from Systems.Interfaces import pick
from Systems.Maneuvers import maneuver_of
from Systems.Radar import TRACKERS
from Systems.plans import PLANNER_FACTORIES, get_config
from Systems.plans.plan_11_v11.departure import AxisDeparture
from Systems.plans.plan_14_v14.launch import ClearanceLaunch

GUARD_PLAN = False      # True/"union": guard tests the path the burn in progress will fly. Bench +1.7 and 0 hits, but
#                         real sim 2026-11-13: ~54 kg every run vs ~45 kg without (8 vs 10 runs) -> off
HOLD_T = 12.0           # s: no replanning before this. Real-sim rocks are shoved for ~10 s at start-up; plans made
#                         then chased noise (an 80 deg turn at t=6 in the replay bench, 2026-11-13)


class Flight:
    def __init__(self, planner=1, tracker=None, threaded=False, log=None, run_logger=None, config=None,
                 empty_field=False):
        self.planner_choice = planner
        self.config = get_config(planner) if config is None else config
        self.planner = pick(P.PLANNERS, planner, "planner")
        if config is not None and planner in PLANNER_FACTORIES:
            self.planner = PLANNER_FACTORIES[planner](config)
        self.alpha = getattr(self.planner, "barrier_alpha", None)
        self.threaded = threaded; self.log = log or (lambda msg: None)
        self.empty_field = empty_field
        self.run_logger = run_logger
        tracker_name = tracker if tracker is not None else (self.config.get("tracker", "kalman") if self.config else "kalman")
        self.world = pick(TRACKERS, tracker_name, "tracker")
        if hasattr(self.world, "configure_from_plan"):
            self.world.configure_from_plan(self.config)
        self.hist = deque(maxlen=1500)                                     # (t, r, v) of the shuttle
        self.plan = None; self.idx = 0; self.committed = False
        self.burning = False; self.main = False; self.guarding = False; self.barrier_active = False
        self.active_nodes = [C.T]; self.active_point = C.T.copy()
        self.next_replan = 0.0; self.near = None; self.near_t = -1.0
        self._job = None; self._result = None; self._lock = threading.Lock(); self._busy = False
        self.events = []                                                   # (t, action, points, p_hit, cost)
        self.plan_t = -np.inf; self.guard_end = None                       # replan_after_guard (plan R1)
        self.guard_ticks = 0
        self.guard_pulse_accel = 0.0
        self.route_ready_at = -np.inf
        self.route_ready = False
        self.coast_pointing = False
        self.main_axis_acquired = False
        self.body_z = None                                                 # callable -> body +z in the world, set by the loop
        self.body_state = None                                             # callable -> body-to-world matrix and body rates
        self.maneuver_check_t = -np.inf
        self.body_command = None
        self.departure = AxisDeparture(self.config) if (self.config or {}).get("lift_height_m") else None
        self.launch = ClearanceLaunch(self.config) if self.config.get("launch_clearance_m") else None

    # -- radar ------------------------------------------------------------------------------------------------
    def own_at(self, t):
        """Shuttle (pos, vel) at our-clock time t, from recent history."""
        h = self.hist
        if not h:
            return C.S, np.zeros(3)
        if t >= h[-1][0]:
            return h[-1][1], h[-1][2]
        for a in reversed(h):                                              # newest first; lookups are a few ticks back
            if a[0] <= t:
                return a[1], a[2]
        return h[0][1], h[0][2]

    def radar(self, entries, t):
        changed = self.world.read(entries, t, self.own_at)
        if self.run_logger is not None:
            own_r, _ = self.own_at(t)
            offset = getattr(self.world, "offset", None)
            origins = [self.own_at(float(e[2, 0]) + offset)[0] for e in entries] if offset is not None else None
            self.run_logger.log_radar(t, own_r, entries, changed_count=len(changed), origins=origins)
        return changed

    # -- planning ---------------------------------------------------------------------------------------------
    def _snapshot(self, t, r, v, fuel):
        tube = self.config.get("prune_tube", 220.0) if self.config else 220.0
        rocks = self.world.predict(t) if self.world.n else (
            np.empty((0, 3)), np.empty((0, 3)), np.empty(0), np.empty(0), np.empty(0),
        )
        keep = slice(None) if self.config.get("route_handoff_s") is not None else C.prune(r, rocks[0], rocks[2], tube=tube)
        st = dict(r=r.copy(), v=v.copy(), burning=self.burning, main=self.main, fuel=fuel, t=t)
        if self.body_z is not None:                                        # body +z in the world: the planner's turn model
            st['u'] = self.body_z()
        if self.body_state is not None:
            st["body"], st["rates"] = self.body_state()
            st["u"] = st["body"][:, 2]
            st["coast_pointing"] = self.coast_pointing
            st["main_axis_acquired"] = self.main_axis_acquired
            st["launch_released"] = self.launch is not None and self.launch.done
        rem = None if self.plan is None else (list(self.plan[0][self.idx:]), *self.plan[1:])
        return st, tuple(a[keep] for a in rocks), rem, self.committed

    def _adopt(self, t, plan, info):
        first = self.plan is None
        self.plan = plan; self.idx = 0; self.plan_t = t
        self.maneuver_check_t = -np.inf
        p_safe = self.config.get("p_safe", C.P_SAFE) if self.config else C.P_SAFE
        self.committed = self.committed or info['ptot'] <= p_safe
        risk = self.config.get("replan_risk_p", 0.03) if self.config else 0.03
        if risk is not None and info['ptot'] > risk:
            self.next_replan = t                                           # still risky: look again at once
        if info['action'] != 'keep':
            self.events.append((round(t, 1), info['action'], info['K'], round(info['ptot'], 5), round(info['cost'], 2)))
            self.log(f"plan t={t:.1f} {info['action']} points={info['K']} p_hit={info['ptot']:.5f} cost={info['cost']:.2f}")
            if self.run_logger is not None:
                self.run_logger.log_plan(t, plan, info)
        elif self.run_logger is not None:
            if first:
                self.run_logger.log_plan(t, plan, info)
            else:
                self.run_logger.log_line(t, "PLAN", f"keep K={info['K']} cost={info['cost']:.3f} ptot={info['ptot']:.6f}")

    def _work(self, job):
        timed = (self.config or {}).get("timed_burns")
        if timed:
            job[0]["deadline"] = monotonic() + 1.1
        try:
            res = self.planner(*job)
            if timed or self.config.get("route_handoff_s") is not None:
                plan, info = res
                res = plan, {**info, "snapshot_t": job[0]["t"]}
                if not timed:
                    res[1].update(snapshot_r=job[0]["r"], snapshot_v=job[0]["v"])
        except Exception as e:                                             # a planner failure must never stop the flight
            res = None; self.log(f"planner error: {e!r}")
            if self.run_logger is not None:
                self.run_logger.log_line(job[0]['t'], "ERROR", f"planner error: {e!r}")
        with self._lock:
            self._result = res; self._busy = False

    def _accept_result(self, t, r, v, plan, info):
        if t - info.get("snapshot_t", t) > self.config.get("route_max_age_s", 1.2):
            return
        if self.config.get("route_handoff_s") is not None:
            if self.guarding or info.get("snapshot_t", t) < (self.guard_end if self.guard_end is not None else -np.inf):
                return
            if "snapshot_r" in info:
                projected = info["snapshot_r"] + (t - info["snapshot_t"]) * info["snapshot_v"]
                if np.linalg.norm(r - projected) > 4.0 or np.linalg.norm(v - info["snapshot_v"]) > 0.8:
                    return
            if plan is None:
                if not self.config.get("route_keep_on_miss"):
                    self.plan = None
                    self.idx = 0
                    self.committed = False
                return
            if self.plan is not None and info["action"] == "keep":
                return  # retain the exact waypoint list and progress while it remains safe
        maneuver = maneuver_of(plan)
        if maneuver is not None:
            # Worker estimates start at the captured state. Start the command
            # clock at adoption, only if the measured state is still close.
            if self.guarding:
                return
            near = self._nearby(t, r)
            if maneuver.segments[0].body_command is not None:
                if self.body_state is None or maneuver.initial_body is None:
                    return
                body, rates = self.body_state()
                if np.linalg.norm(rates) >= 0.04 or np.trace(maneuver.initial_body.T @ body) < 2.98:
                    return
            maneuver = maneuver.rebase(t, r, v)
            if maneuver is None or not maneuver.check(t, r, v, near):
                return
            plan = (*plan[:2], t, maneuver)
        self._adopt(t, plan, info)

    def _replan(self, t, r, v, fuel):
        hold_t = float(self.config.get("hold_t", 0.0)) if self.config else 0.0
        handoff = self.config.get("route_handoff_s")
        if handoff is not None:
            if t < handoff or self.guarding:
                return
            launch_height = self.config.get("launch_burn_height_m")
            if launch_height is not None and (r - C.S) @ (C.S / np.linalg.norm(C.S)) < launch_height:
                return
            if not self._route_ready(t):
                if not self.config.get("route_keep_on_miss"):
                    self.plan = None
                    self.idx = 0
                    self.committed = False
                return
        plan_without_radar = self.empty_field and (self.config.get("timed_burns") or handoff is not None)
        if ((self.world.n == 0 and not plan_without_radar)
                or (self.plan is not None and t < hold_t)):
            return
        if self.threaded:
            with self._lock:
                res, self._result = self._result, None
                busy = self._busy
            if res is not None:
                self._accept_result(t, r, v, *res)
            if t >= self.next_replan and not busy:
                snapshot = self._snapshot(t, r, v, fuel)
                self._busy = True
                threading.Thread(target=self._work, args=(snapshot,), daemon=True).start()
                self.next_replan = t + self._period()
        elif t >= self.next_replan:
            self._accept_result(t, r, v, *self.planner(*self._snapshot(t, r, v, fuel)))
            self.next_replan = t + self._period()

    def _route_ready(self, t):
        if not self.world.n:
            return self.empty_field
        cfg = self.config
        if self.world.n < cfg["route_expected_tracks"]:
            return False
        tube = cfg.get("route_ready_tube_m")
        if tube is None and np.any(self.world.count < cfg["route_min_samples"]):
            return False
        if t - self.route_ready_at >= 0.2:
            positions, _, radii, sigma, _ = self.world.predict(t)
            relevant = slice(None) if tube is None else C.prune(C.S, positions, radii + sigma, tube=tube)
            self.route_ready = bool(np.all(self.world.count[relevant] >= cfg["route_min_samples"])
                                    and np.all(sigma[relevant] <= cfg["route_max_sigma_m"])
                                    and np.all(t - self.world.t[relevant] <= cfg["route_max_track_age_s"]))
            self.route_ready_at = t
        return self.route_ready

    def _period(self):
        return 2.0 if (self.burning and self.main) or not self.committed else 3.0

    # -- guard and barrier filter -------------------------------------------------------------------------------
    def _nearby(self, t, r):
        if self.world.n and t - self.near_t >= 0.2:
            radius = self.config.get("guard_range_m", 250.0)
            if self.plan is not None and C.late_cruise(self.plan[1], self.config):
                radius = max(radius, 400.0)
            rp, rv, hard, sp, sv = self.world.predict(t); k = np.linalg.norm(rp - r, axis=1) < radius
            self.near = (rp[k], rv[k], hard[k], sp[k], sv[k]); self.near_t = t
        if self.near is None:
            return None
        rp, rv, hard, sp, sv = self.near
        return rp + rv * (t - self.near_t), rv, hard, sp, sv

    def _guard(self, r, v, near, vg=None, acc=0.0):
        """Last resort. guard_mode "v5": nayeem's (12 s, push straight away; 13/13 real arrivals). "escape": picks the
        best of 13 directions at full thrust (FINDINGS 6.14). Returns a world direction or None."""
        if (self.config or {}).get("guard_mode", "v5") == "escape":
            return self._guard_escape(r, v, near, vg, acc)
        return self._guard_v5(r, v, near, vg, acc)

    def _guard_v5(self, r, v, near, vg=None, acc=0.0):
        """Last resort: a rock predicted inside its hard radius within 12 s. Returns a push direction or None."""
        rp, rv, hard = near[:3]
        if not len(rp):
            return None
        guard_plan = self.config.get("guard_plan", False) if self.config else False
        if guard_plan and vg is not None and acc > 0 and np.linalg.norm(vg) > 1e-6:
            ng = np.linalg.norm(vg); u = vg / ng; tb = ng / acc
            ts = np.arange(0.25, 12.01, 0.25); sb = np.where(ts < tb, 0.5 * acc * ts ** 2, 0.5 * acc * tb ** 2 + ng * (ts - tb))
            path = r + np.outer(ts, v) + np.outer(sb, u)
            rk = rp[None] + ts[:, None, None] * rv[None]
            d = np.linalg.norm(rk - path[:, None], axis=2); k = d.argmin(0); miss = d.min(0)
            limit = hard + (5.0 if self.guarding else 0.0)
            bad = miss < limit
            if bad.any():
                i = int(np.flatnonzero(bad)[np.argmin(k[bad])]); m = rk[k[i], i] - path[k[i]]
                return -m / np.linalg.norm(m) if np.linalg.norm(m) > 0.5 else C.E1
            if guard_plan != "union":
                return None
        dr = rp - r; w = v - rv; ww = np.maximum((w * w).sum(1), 1e-9)
        tca = np.clip((dr * w).sum(1) / ww, 0.0, None); m = dr - w * tca[:, None]; miss = np.linalg.norm(m, axis=1)
        limit = hard + (5.0 if self.guarding else 0.0)                     # hysteresis: release 5 m further out
        bad = (miss < limit) & (tca < 12.0) & (tca > 0.0)
        if not bad.any():
            return None
        i = int(np.flatnonzero(bad)[np.argmin(tca[bad])])
        return -m[i] / miss[i] if miss[i] > 0.5 else C.E1                  # push the shuttle away from the rock

    def _guard_escape(self, r, v, near, vg=None, acc=0.0):
        """Last resort, any plan: a rock predicted inside its hard radius + guard_sigma*sigma + guard_margin_m within
        guard_horizon_s. Returns the world thrust direction that leaves the most clearance to every nearby rock, or None.
        Candidates: straight away from the rock, eight sideways, braking, speeding up, along and against the route.
        Each is flown at full thrust until well past closest approach, in the rock frame (FINDINGS 6.14)."""
        rp, rv, hard, sp, sv = near
        if not len(rp):
            return None
        cfg = self.config or {}
        horizon = cfg.get("guard_horizon_s", 20.0); margin = cfg.get("guard_margin_m", 1.0)
        if self.plan is not None and C.late_cruise(self.plan[1], cfg):
            horizon = max(horizon, cfg.get("late_guard_horizon_s", horizon))
        nsig = cfg.get("guard_sigma", 0.0); release = cfg.get("guard_release_m", 5.0)
        dr = r - rp; w = v - rv; ww = np.maximum((w * w).sum(1), 1e-9)          # the shuttle seen from each rock
        tca = np.clip(-(dr * w).sum(1) / ww, 0.0, None); m = dr + w * tca[:, None]; miss = np.linalg.norm(m, axis=1)
        abort = cfg.get("guard_abort_s")
        if (cfg.get("guard_assume_burn") and ((self.burning and self.main) or self.guarding)
                and self.body_z is not None):
            # Check where continuing the current burn takes us as well as a straight coast.
            times = np.arange(0.5, min(horizon, abort or horizon) + 0.01, 0.5)
            thrust = C.A_BURN * self.body_z()
            path = dr[:, None] + times[None, :, None] * w[:, None] + 0.5 * times[None, :, None]**2 * thrust
            dist = np.linalg.norm(path, axis=2)
            k = dist.argmin(1)
            row = np.arange(len(rp))
            projected_t = times[k]
            projected_need = hard + margin + nsig * np.hypot(sp, projected_t * sv)
            earlier_hit = (projected_t < tca) & (dist[row, k] < projected_need)
            use_burn = (dist[row, k] < miss) | earlier_hit
            tca = np.where(use_burn, projected_t, tca)
            m = np.where(use_burn[:, None], path[row, k], m)
            miss = np.linalg.norm(m, axis=1)
        sig = np.sqrt(sp ** 2 + (tca * sv) ** 2)                            # where the rock will be, how well known
        known = sig <= cfg.get("guard_max_sigma_m", 6.0)                    # early on the velocities are still guesses
        need = hard + nsig * sig + margin
        bad = known & (miss < need + (release if self.guarding else 0.0)) & (tca < horizon) & (tca > 0.0)
        if abort is not None:
            bad &= tca <= abort
        if bad.any() and vg is not None and acc and np.linalg.norm(vg) > 1e-6:
            # guard_trust_plan: a rock counts only if the route's own burn (vg at acc, then coast) also misses short.
            ng = np.linalg.norm(vg); tb = ng / acc; ts = np.arange(0.5, horizon + 0.01, 0.5)
            sb = np.where(ts < tb, 0.5 * acc * ts ** 2, 0.5 * acc * tb ** 2 + ng * (ts - tb))
            path = dr[bad][:, None] + ts[None, :, None] * w[bad][:, None] + sb[None, :, None] * (vg / ng)
            k = np.linalg.norm(path, axis=2).argmin(1)
            planned = np.linalg.norm(path[np.arange(len(k)), k], axis=1)
            sig_p = np.sqrt(sp[bad] ** 2 + (ts[k] * sv[bad]) ** 2)
            idx = np.flatnonzero(bad)
            bad[idx[planned >= hard[bad] + nsig * sig_p + margin]] = False
        if not bad.any():
            return None
        i = int(np.flatnonzero(bad)[np.argmin(tca[bad])])
        watch = np.flatnonzero((tca < horizon + 10.0) & (miss < need + 40.0))  # the rocks a dodge could run into
        away = m[i] / miss[i] if miss[i] > 0.5 else C.E1
        u = w[i] / np.sqrt(ww[i])
        cands = np.array([away, u, -u, C.SH, -C.SH] + [np.cos(a) * C.E1 + np.sin(a) * C.E2 for a in np.arange(8) * np.pi / 4])
        cands /= np.linalg.norm(cands, axis=1, keepdims=True)
        pulse_accel = cfg.get("guard_pulse_accel", 0.0) if tca[i] >= 12.0 else 0.0
        self.guard_pulse_accel = pulse_accel
        accel = pulse_accel or C.A_TRIM
        ts = np.linspace(0.0, max(2.0 * tca[i], 10.0), 60)

        def clearance(a):
            q = (dr[watch][None, :, None, :] + w[watch][None, :, None, :] * ts[None, None, :, None]
                 + 0.5 * a * cands[:, None, None, :] * ts[None, None, :, None] ** 2)
            return (np.linalg.norm(q, axis=3).min(2) - need[watch][None]).min(1)
        clear = clearance(accel)
        if pulse_accel and cfg.get("guard_pulse_check") and clear.max() < cfg.get("guard_pulse_margin_m", 2.0):
            # A gentle pulse cannot clear this rock in time (live run 261119_223116 was hit this way): full thrust.
            self.guard_pulse_accel = 0.0
            clear = clearance(C.A_TRIM)
        return cands[int(np.argmax(clear))]

    # -- every tick -------------------------------------------------------------------------------------------
    def decide(self, t, r, v, fuel, dt):
        self.hist.append((t, r.copy(), v.copy()))
        self.body_command = None
        self.guard_pulse_accel = 0.0
        if self.launch is not None and not self.launch.done:
            near = self._nearby(t, r)
            push = self._guard(r, v, near) if near is not None else None
            launch = self.launch.command(r, v, self.main_axis_acquired, push, t)
            if launch is not None:
                self.guarding = push is not None
                self.burning = self.main = False
                return launch
            self.guarding = False
            status = "slew deadline reached, steering in flight" if self.launch.timed_out else "+X acquired"
            self.log(f"launch clear at t={t:.1f} s ({status}); enabling route guidance")
        if self.departure is not None:
            near = self._nearby(t, r)
            push = self._guard(r, v, near) if near is not None else None
            mode, vec = self.departure.decide(t, r, v, near, push)
            self.guarding = mode == "trim" and push is not None
            self.burning, self.main = mode != "coast", mode == "main"
            return mode, vec
        self._replan(t, r, v, fuel)
        near = self._nearby(t, r)
        maneuver = maneuver_of(self.plan)
        invalidated = False
        if maneuver is not None and t >= self.maneuver_check_t:
            self.maneuver_check_t = t + 0.2
            if not maneuver.check(t, r, v, near):
                if maneuver.review_horizon:
                    self.log(f"burn schedule invalidated at t={t:.1f} s; replanning from measured state")
                self.plan = maneuver.fallback
                self.idx = 0
                self.next_replan = t
                self.burning = False
                maneuver = None
                invalidated = True
        nodes, vc = self.plan[:2] if self.plan is not None else ([C.T], C.resolve_vc(self.config))   # straight until the first plan arrives
        late = float(self.config.get("late_after_m", 0.0) or 0.0) if self.config else 0.0
        if late > 0.0 and (r - C.S) @ C.SH > late:                         # the clear last stretch: cruise faster
            vc = max(vc, float(self.config.get("late_vc", 0.0)))           # shortcut: the planner still costs plans at their own speed, fine while no rocks are left there
        self.active_nodes = list(nodes)
        while self.idx < len(nodes) - 1 and (nodes[self.idx] - r) @ C.SH <= 1.0:
            if self.run_logger is not None:
                self.run_logger.log_turn_passed(t, self.idx, nodes[self.idx], r)
            self.idx += 1                                                  # passed this node: aim at the next
        idx = min(self.idx, len(nodes) - 1)
        self.active_point = np.asarray(nodes[idx], float)
        self.barrier_active = False
        frame = bool(self.config.get("frame_accel", False)) if self.config else False
        main_min = float(self.config.get("main_min", C.MAIN_MIN)) if self.config else C.MAIN_MIN
        ignite_dv = self.config.get("ignite_dv_mps", C.EPS_ON)
        guard_plan = self.config.get("guard_plan", False) if self.config else False
        last = idx == len(nodes) - 1
        radius_key = "coast_docking_radius_m" if last else "coast_waypoint_radius_m"
        radius = self.config.get(radius_key, 0.0)
        coast_horizon = 360.0 if last else 35.0
        deadline = self.config.get("local_coast_deadline_s")
        if last and deadline is not None and t < deadline and fuel < self.config["fuel_goal_kg"]:
            coast_horizon = max(1.0, deadline - t)
        coast_clear = (radius > 0.0 and C.coast_pass_near(r, v, C.T if last else self.active_point,
                                                        radius, coast_horizon, frame=frame))
        if coast_clear and last and C.late_cruise(vc, self.config):
            fuel_limit = self.config["fuel_goal_kg"] - self.config["late_fuel_reserve_kg"]
            coast_clear = (fuel >= fuel_limit or not self._route_ready(t)
                           or not C.cruise_needs_thrust(r, v, vc))

        if near is not None:
            vg0 = acc0 = None
            trust = self.config.get("guard_trust_plan", False) and not self.guarding
            trust_route = trust and maneuver is None
            trust_gate = self.config.get("guard_trust_gate", False)
            if trust_route and trust_gate:
                launch_height = self.config.get("launch_burn_height_m")
                in_launch = launch_height is not None and (r - C.S) @ (C.S / np.linalg.norm(C.S)) < launch_height
                trust_route = (not invalidated and self.plan is not None and not in_launch
                               and t >= C.hold_of(self.plan)
                               and t - self.plan_t <= self.config.get("guard_trust_max_age_s", 6.0))
            if (guard_plan or trust_route) and not coast_clear:
                i0 = min(self.idx, len(nodes) - 1); vg0, _ = C.to_gain(r, v, nodes[i0], vc, frame=frame)
                n0 = float(np.linalg.norm(vg0)); main0 = self.main if self.burning else n0 > main_min
                guard_ignite_dv = ignite_dv if trust_route and trust_gate else C.EPS_ON
                acc0 = (C.A_BURN if main0 else C.A_TRIM) if (self.burning or n0 > guard_ignite_dv) else 0.0
            # A schedule that passes its own 30 s check (hard + 2 sigma + 1 m) already clears every rock the escape guard
            # would flag; the guard only judges a coast and would fight the burn that makes the clearance (engine 100758).
            if trust and maneuver is not None and maneuver.check(t, r, v, near):
                push = None
            else:
                push = self._guard(r, v, near, vg0, acc0 if trust or guard_plan else 0.0)
            was_guarding = self.guarding
            self.guarding = push is not None
            if self.guarding:
                self.burning = False
                if self.config.get("route_handoff_s") is not None:
                    self.plan = None
                    self.idx = 0
                    self.committed = False
                    self.guard_end = t
                if maneuver is not None:
                    self.plan = maneuver.fallback
                    self.idx = 0
                    self.next_replan = t
                launch_height = self.config.get("launch_burn_height_m")
                if launch_height is not None:
                    up = C.S / np.linalg.norm(C.S)
                    if (r - C.S) @ up < launch_height:
                        push = push - min(0.0, push @ up) * up
                        length = np.linalg.norm(push)
                        push = push / length if length > 1e-6 else up
                if self.guard_pulse_accel:
                    return "pulse", self.guard_pulse_accel * push
                return ("evade" if (self.config or {}).get("guard_mode", "v5") == "escape" else "trim"), push
            if was_guarding and (self.config or {}).get("replan_after_guard", False):
                self.guard_end = t; self.next_replan = t              # plan R1: replan from where the dodge left us
            if was_guarding and self.config.get("route_handoff_s") is not None:
                self.next_replan = t
        wait = float((self.config or {}).get("replan_after_guard", False) or 0.0)
        if self.guard_end is not None and self.plan_t < self.guard_end and t - self.guard_end < wait:
            self.burning = False                                      # coast instead of burning back onto the old plan
            return "coast", np.zeros(3)
            alpha = getattr(self.planner, "barrier_alpha", self.alpha)
            if alpha is not None:                                          # barrier-function filter (Plan 6)
                rp, rv, hard, sp, _ = near
                sig_m = self.config.get("safety_sigma", 3.0) if self.config else 3.0
                mar_m = self.config.get("safety_margin_m", 3.0) if self.config else 3.0
                thr = self.config.get("barrier_threshold", 0.01) if self.config else 0.01
                u = P.barrier(r, v, rp, rv, hard + sig_m * sp + mar_m, alpha)
                if np.linalg.norm(u) > thr:
                    self.burning = False
                    self.barrier_active = True
                    return "evade", u / np.linalg.norm(u)

        if maneuver is not None:
            action = maneuver.action(t)
            if action is not None:
                segment, mode, vec = action
                self.body_command = segment.body_command
                self.burning = mode != "coast"
                self.main = mode in ("main", "body_command")
                return mode, vec

        accel_dur = float(self.config.get("accel_dur_s", 0.0) or 0.0) if self.config else 0.0
        if accel_dur > 0.0 and t >= accel_dur:                            # after the timed burn: keep the speed it
            vc = max(vc, float(np.linalg.norm(v)))                           # built, still steer (no drift past the target)
        elif accel_dur > 0.0:
            d_goal = nodes[idx] - r
            dist_goal = float(np.linalg.norm(d_goal))
            u_goal = d_goal / max(dist_goal, 1e-6)
            v_est = max(float(np.linalg.norm(v)) + C.A_BURN * max(accel_dur - t, 0.0), 4.0)
            v_req_lead, _ = C.to_gain(r, v, nodes[idx], v_est, frame=frame)
            v_lat = v_req_lead - (v_req_lead @ u_goal) * u_goal
            self.burning = True
            self.main = True
            return "main", max(main_min + 20.0, 25.0) * u_goal + v_lat

        if self.plan is not None and t < C.hold_of(self.plan):            # the plan waits: a rock crosses ahead
            self.burning = False
            return "coast", np.zeros(3)
        launch_height = self.config.get("launch_burn_height_m")
        if launch_height is not None:
            up = C.S / np.linalg.norm(C.S)
            if (r - C.S) @ up < launch_height:
                bearing = (C.T - r) / np.linalg.norm(C.T - r)
                aim = C.turn_toward(up[None], bearing[None], np.deg2rad(self.config["launch_burn_cone_deg"]))[0]
                self.burning = self.main = True
                return "main", 4.0 * aim
        vg, dist = C.to_gain(r, v, nodes[idx], vc, frame=frame); ng = float(np.linalg.norm(vg))
        if coast_clear:
            self.burning = False
            return "coast", np.zeros(3)
        stop_dv = self.config.get("stop_dv_mps", 0.0)
        if not self.burning and ng > ignite_dv and not (last and dist < C.TERMINAL):
            self.burning = True; self.main = ng > main_min
        if self.burning and ng <= max((C.A_BURN if self.main else C.A_TRIM) * dt, 0.0 if self.main else 0.02, stop_dv):
            self.burning = False
        if not self.burning:
            return "coast", np.zeros(3)
        return ("main" if self.main else "trim"), vg
