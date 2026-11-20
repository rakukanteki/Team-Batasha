"""Flight code: radar world model, planners, actuation laws, vision geometry, and one closed-loop flight.
Specs: docs/plans/. The closed-loop test needs the STU4-Sim submodule (the debris list is test data only)."""
import numpy as np
import pytest

from Systems import Actuation, BasicController, Core as C, Interfaces, Radar, Vision
from Systems.Flight import Flight
from Systems.plans import get_config
from Systems.Planners import PLANNERS, barrier
from Systems.Radar import WorldModel
from tools import actuator_twin, offline_bench

FIELD = offline_bench.load_field()


def rest_state():
    return dict(r=C.S.copy(), v=np.zeros(3), burning=False, main=False, fuel=0.0, t=0.0)


def one_rock(dist=300.0, hard=16.0):
    """A well-localized rock sitting on the start-to-target line."""
    return (np.array([C.S + dist * C.SH]), np.zeros((1, 3)), np.array([hard]), np.array([0.7]), np.array([0.03]))


EMPTY = (np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0), np.zeros(0))


# ---- plug points ----------------------------------------------------------------------------------------------
def test_every_registered_part_has_its_interface():
    assert all(isinstance(f, Interfaces.Planner) for f in PLANNERS.values())
    assert all(isinstance(f, Interfaces.BurnLaw) for f in Actuation.BURNS.values())
    assert all(isinstance(c(), Interfaces.Tracker) for c in Radar.TRACKERS.values())
    assert isinstance(Vision.ClassicalSegmenter(), Interfaces.Segmenter)        # the model-based ones need ultralytics
    assert all(isinstance(c(None), Interfaces.CameraUse) for c in Vision.CAMERAS.values() if c)
    assert isinstance(actuator_twin.TwinSim(1.0), Interfaces.Sim)
    assert all(hasattr(BasicController.SpaceTeamsSim, m) for m in dir(Interfaces.Sim) if not m.startswith("_"))
    with pytest.raises(KeyError):
        Interfaces.pick(PLANNERS, "no_such_planner", "planner")


def test_the_real_flight_loop_burns_toward_the_target_on_the_jet_twin():
    C.configure(); early = actuator_twin.TwinSim(20.0); sim = actuator_twin.TwinSim(30.0)
    BasicController.fly(early, threaded=False); BasicController.fly(sim, threaded=False)   # the loop the sim runs
    thrust = sim.v - C.grav(C.S) * sim.t; late = (sim.v - early.v) - C.grav(C.S) * (sim.t - early.t)
    assert np.linalg.norm(late) / (sim.t - early.t) > 0.128        # aligned 1 deg toward +x: the solver's strong jet set
    assert sim.used / np.linalg.norm(thrust) < 4.55 and np.linalg.norm(sim.w) < 0.002
    assert thrust @ C.SH > 0.999 * np.linalg.norm(thrust) and not any(l.startswith("ERROR") for l in sim.lines)


def test_every_plan_config_sets_the_thrust_and_the_flight_uses_it(monkeypatch):
    from Systems.plans import PLAN_CONFIGS, get_config
    keys = ("max_thrust", "a_burn", "a_burn_min", "a_burn_max")
    assert all(k in PLAN_CONFIGS[n] for n in range(1, 10) for k in keys)
    monkeypatch.setattr(Actuation, "THRUST", Actuation.THRUST)              # restored after the test
    cfg = dict(get_config(BasicController.PLANNER), max_thrust=0.3, a_burn=0.11)
    C.configure(); sim = actuator_twin.TwinSim(3.0); sent = []; a_first = []
    command = sim.command
    sim.command = lambda lin, ang: (sent.append(np.linalg.norm(lin)), a_first.append(C.A_BURN), command(lin, ang))
    BasicController.fly(sim, threaded=False, config=cfg, log_run=False)
    assert max(sent) == pytest.approx(0.3) and a_first[0] == pytest.approx(0.11)     # then learned in flight


def test_direct_burn_thrusts_along_the_wanted_direction_at_any_attitude():
    rng = np.random.default_rng(0)
    for _ in range(50):                                     # tumbling or not, the thrust command is the wanted direction
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        lin, ang = Actuation.direct_burn(d, rng.normal(0, 0.05, 3), {})
        assert np.allclose(lin, 0.5 * d) and np.isfinite(ang).all()
    state = {}; d = np.array([0.05, 0.0, 1.0]); d /= np.linalg.norm(d)
    assert np.linalg.norm(Actuation.direct_burn(d, np.zeros(3), state)[1]) > 0 and not state.get("steer_off")
    far = np.array([0.6, 0.0, 0.8]); state["ref"] = np.array([0.0, 0.0, 1.0])     # a sideways push: thrust goes there,
    lin, ang = Actuation.direct_burn(far, np.zeros(3), state)                      # the body stays on the target line
    assert np.allclose(lin, 0.5 * far) and np.linalg.norm(ang) < 0.02 and ang[1] < 0 and not state.get("steer_off")
    w = np.array([0.08, 0.0, 0.0])                          # the alignment loop ran away: only damp from now on
    assert np.allclose(Actuation.direct_burn(d, w, state)[1], -0.5 * w) and state["steer_off"]
    assert np.allclose(Actuation.direct_burn(d, np.zeros(3), state)[1], 0.0)


def test_the_recorder_saves_what_the_flight_saw_and_replays_the_radar(tmp_path):
    from Systems.Recorder import Recorder, reads
    C.configure(); sim = actuator_twin.TwinSim(2.0); feed = [np.full((3, 3), float(k)) for k in range(4)]
    sim.radar = lambda: [feed[0], feed[1 + int(sim.t > 1.0)]]              # one entry is replaced half way
    rec = Recorder(sim, str(tmp_path)); BasicController.fly(rec, threaded=False); rec.save(wait=True)
    log = np.load(rec.path); ticks = log["ticks"]
    assert ticks.shape[1] == 26 and len(ticks) > 100 and np.allclose(ticks[-1, 1:4], sim.r, atol=0.01)
    assert np.allclose(log["target"], C.T) and ticks[-1, 25] > 0 and np.allclose(np.linalg.norm(ticks[:, 19:22], axis=1), 0.5)
    seen = [sorted(float(m[0, 0]) for m in entries) for _, entries in reads(log)]
    assert seen[0] == [0.0, 1.0] and seen[-1] == [0.0, 2.0]


def test_a_plan_can_hold_the_burn_and_the_flight_waits():
    C.configure(); none = tuple(np.zeros(shape) for shape in ((0, 3), (0, 3), 0, 0, 0))
    st = dict(r=C.S.copy(), v=np.zeros(3), burning=False, main=False, fuel=0.0, t=10.0)
    ev = C.score_plans([([C.T], 8.4), ([C.T], 8.4, 22.0)], st, none)
    assert 11.0 < ev['T'][1] - ev['T'][0] < 14.5 and abs(ev['fuel'][1] - ev['fuel'][0]) < 1.0    # ~12 s later, same fuel
    flight = Flight(7); flight.plan = ([C.T], 8.4, 22.0)
    assert flight.decide(10.0, C.S, np.zeros(3), 0.0, 0.1)[0] == "coast"
    assert flight.decide(22.5, C.S, np.zeros(3), 0.0, 0.1)[0] == "main"


def test_the_fast_closest_approach_sweep_gives_the_einsum_numbers(monkeypatch):
    rng = np.random.default_rng(4); K, Nc, Nr, dt = 60, 9, 70, 0.5
    path = np.cumsum(rng.normal(0, 3, (K, Nc, 3)), axis=0); rp = rng.normal(0, 40, (Nr, 3)); rv = rng.normal(0, 2, (Nr, 3))
    gh = rng.normal(0, 0.01, (Nr, 3)); arr = np.where(rng.random(Nc) < 0.4, rng.integers(10, K, Nc), -1)
    mind2 = np.full((Nc, Nr), np.inf); kmin = np.zeros((Nc, Nr), int); w2 = np.zeros((Nc, Nr)); prev = None
    for k in range(K):                                       # the einsum version the sweep replaced
        tau = k * dt; diff = path[k][:, None, :] - (rp + rv * tau + gh * (tau * tau))[None]
        if prev is None:
            dk2 = np.einsum("ijk,ijk->ij", diff, diff); wk2 = np.zeros_like(dk2); kn = np.zeros(dk2.shape, int)
        else:
            e = diff - prev; ee = np.einsum("ijk,ijk->ij", e, e)
            s = np.clip(-np.einsum("ijk,ijk->ij", prev, e) / np.maximum(ee, 1e-12), 0.0, 1.0)
            c = prev + s[..., None] * e; dk2 = np.einsum("ijk,ijk->ij", c, c); wk2 = ee / (dt * dt); kn = k - 1 + (s >= 0.5)
        prev = diff; dk2[(arr >= 0) & (arr < k)] = np.inf
        b = dk2 < mind2 - 1e-6; mind2[b] = dk2[b]; kmin[b] = kn[b]; w2[b] = wk2[b]
    for block in (7, 48, 1000):
        monkeypatch.setattr(C, "SWEEP_BLOCK", block)
        got = C._sweep(path, rp, rv, gh, arr, dt)
        assert all(np.array_equal(x, y) for x, y in zip(got, (mind2, kmin, w2)))


def test_plan_4_is_plan_7_plus_zone_switches():
    from Systems.plans.plan_4_team_pipeline import make_planner
    C.configure(); st = rest_state(); rocks = one_rock()
    off = dict(get_config("team_pipeline"), corridor=False, wide_fallback=False, late_after_m=0.0)
    p4, i4 = make_planner(off)(st, rocks, None, False); p7, i7 = PLANNERS[7](st, rocks, None, False)
    assert i4["cost"] == i7["cost"] and all(np.allclose(a, b) for a, b in zip(p4[0], p7[0]))
    r = C.S + 950.0 * C.SH; v = C.to_gain(r, np.zeros(3), C.T, 8.4, frame=True)[0]     # coasting on plan, past 900 m
    modes = []
    for cfg in (off, dict(off, late_after_m=900.0, late_vc=10.4)):
        flight = Flight("team_pipeline", config=cfg); flight.plan = ([C.T], 8.4)
        modes.append(flight.decide(180.0, r, v, 40.0, 0.1)[0])
    assert modes[0] == "coast" and modes[1] in ("main", "trim")                        # the clear stretch: speed up


def test_the_tracker_learns_the_fast_rocks_of_the_sim():
    from tools import sim_bench                              # tools/sim_field.npz: the field as the sim flew it
    sim = sim_bench.FieldSim(1); world = WorldModel(); hist = []
    own = lambda t: next(((r, v) for tt, r, v in reversed(hist) if tt <= t), (sim.r, sim.v))
    for k in range(int(26 * 60)):                           # to t = 30 s, no thrust
        hist.append((sim.t, sim.r.copy(), sim.v.copy()))
        if k % 12 == 0:
            world.read(sim.radar(), sim.t, own)
        sim.wait()
    ids = {world.keys[m.tobytes()]: i for m, i in zip(sim.scan.published, sim.scan.truth) if m.tobytes() in world.keys}
    trk = np.array(list(ids)); tru = np.array([ids[k] for k in trk]); rp, rv, _, sp, sv = world.predict(sim.t)
    assert np.median(np.linalg.norm(sim.V, axis=1)) > 3.0                          # these rocks are fast
    assert np.median(np.linalg.norm(rv[trk] - sim.V[tru], axis=1)) < 0.25          # and their velocity is known
    assert np.median(np.linalg.norm(rp[trk] - sim.P[tru], axis=1)) < 3.5 and 0.05 < np.median(sv) < 0.3


def test_astar_and_the_cost_lattice_find_the_same_cheapest_path():
    from Systems import CostSearch
    from tools import sim_bench
    sim = sim_bench.FieldSim(1); world = WorldModel(); hist = []
    own = lambda t: next(((r, v) for tt, r, v in reversed(hist) if tt <= t), (sim.r, sim.v))
    for k in range(int(36 * 60)):                           # burn straight at the target to t = 40 s
        hist.append((sim.t, sim.r.copy(), sim.v.copy()))
        if k % 12 == 0:
            world.read(sim.radar(), sim.t, own)
        d = sim.to_body(C.T - sim.r); sim.command(0.5 * d / np.linalg.norm(d), np.zeros(3)); sim.wait()
    rocks = world.predict(sim.t); keep = C.prune(sim.r, rocks[0], rocks[2]); rocks = tuple(a[keep] for a in rocks)
    st = dict(r=sim.r.copy(), v=sim.v.copy(), burning=True, main=True, fuel=sim.used, t=sim.t)
    G1 = CostSearch.Graph(st, rocks, 8.4); G2 = CostSearch.Graph(st, rocks, 8.4)
    (p1, c1), (p2, c2) = CostSearch.solve_lattice(G1), CostSearch.solve_astar(G2)
    assert abs(c1 - c2) < 1e-6 and len(p1) == len(p2) == G1.J and CostSearch.GOAL[p1[-1]] and CostSearch.GOAL[p2[-1]]
    assert G2.priced < 0.3 * G1.priced                      # A* priced only a fraction of the edges (26% on this field)
    plan, info = PLANNERS["astar"](st, rocks, None, False)
    assert np.linalg.norm(plan[0][-1] - C.T) <= 40.0 + 1e-6 and info["action"] == "astar"


# ---- radar ----------------------------------------------------------------------------------------------------
def test_radar_identity_and_velocity():
    P, V, D, S, T = FIELD; C.configure(S, T)
    rng = np.random.default_rng(0); P = P.copy(); V = V.copy(); r = S.copy(); v = np.zeros(3)
    radar = offline_bench.Radar(rng, P, V, D, r, v); world = WorldModel(); fed = {}
    own = lambda t: (r, v)

    def read(t):
        truth = {m.tobytes(): i for m, i in zip(radar.published, radar.truth)}
        for track, entry in world.read(radar.published, t, own):
            fed.setdefault(track, set()).add(truth[entry.tobytes()])

    read(0.0); t = 0.0
    for step in range(int(30 * 120)):                       # 30 s of radar at 120 Hz, read at 5 Hz
        t += 1 / 120; V += C.grav(P) / 120; P += V / 120
        radar.tick(t, P, V, D, r, v)
        if step % 24 == 0:
            read(t)
    assert len(fed) == len(D) and all(len(ids) == 1 for ids in fed.values())     # no track ever got another rock's data
    true_of = np.array([next(iter(fed[k])) for k in range(len(D))])
    err = world.predict(t)[1] - V[true_of]
    assert np.sqrt((err ** 2).mean()) < 0.2                 # ~0.08 m/s per axis after 30 s. Q_VEL is sized for real-sim
    #                                                         rocks that bump each other (tools/tracker_bench.py), so ideal
    #                                                         constant-velocity rocks converge a little slower than before


# ---- planners -------------------------------------------------------------------------------------------------
def test_plan_clearance_uses_the_earliest_encounter_through_arrival():
    C.configure()
    rock = C.S + 20 * C.SH
    path = np.broadcast_to(rock + 30 * C.E1, (70, 2, 3)).copy()
    path[[6, 35], 0] = rock + 6 * C.E1
    path[50, 0] = rock
    path[50, 1] = rock
    rocks = (rock[None], np.zeros((1, 3)), np.array([3.0]), np.ones(1), np.zeros(1))
    C.configure(mu=0.0)                                          # pure geometry: no gravity (frame terms are off without a plan config)
    try:
        result = C.evaluate(path, np.zeros((70, 2)), np.array([35, -1]), np.zeros(2),
                            np.ones(2, dtype=int), rocks)
        np.testing.assert_array_equal(result["kmin"][:, 0], [6, 50])
        np.testing.assert_allclose(result["mind"][:, 0], [6, 0])
    finally:
        C.configure()


@pytest.mark.parametrize("plan_id", [1, 2, 3, 5, 7, 10])
def test_planner_flies_straight_in_an_empty_field(plan_id):
    C.configure()
    plan, info = PLANNERS[plan_id](rest_state(), EMPTY, None, False)
    assert info["K"] == 0 and info["ptot"] == 0.0 and 55.0 < info["clean"] < 62.0


@pytest.mark.parametrize("plan_id", [1, 2, 3, 5, 7, 10])
def test_planner_avoids_a_rock_on_the_line(plan_id):
    C.configure(); st = rest_state(); rocks = one_rock()
    _, clear = PLANNERS[plan_id](st, EMPTY, None, False)
    plan, info = PLANNERS[plan_id](st, rocks, None, False)
    for _ in range(2):                                      # the avoiding logic of some planners runs on a replan
        if info["ptot"] > C.P_SAFE:
            plan, info = PLANNERS[plan_id](st, rocks, plan, False)
    assert info["ptot"] <= C.P_SAFE
    assert info["cost"] < clear["cost"] + 5.0               # the detour costs a few points, not more


def test_barrier_pushes_away_only_when_closing():
    r = np.zeros(3); v = np.array([8.0, 0, 0]); need = np.array([20.0])
    ahead = barrier(r, v, np.array([[60.0, 2.0, 0]]), np.zeros((1, 3)), need, 0.4)
    assert ahead @ (r - np.array([60.0, 2.0, 0])) > 0
    assert not barrier(r, v, np.array([[-200.0, 0, 0]]), np.zeros((1, 3)), need, 0.4).any()


def test_guard_fires_only_for_a_rock_inside_its_hard_radius():
    C.configure(); f = Flight(1); r = C.S.copy(); v = 8.0 * C.SH
    near = lambda offset: (np.array([r + 60 * C.SH + offset * C.E1]), np.zeros((1, 3)), np.array([15.0]),
                           np.array([0.7]), np.array([0.02]))
    push = f._guard(r, v, near(5.0))
    assert push is not None and push @ C.E1 < 0.0           # away from the rock's side
    assert f._guard(r, v, near(40.0)) is None


# ---- actuation ------------------------------------------------------------------------------------------------
def test_burn_laws_against_the_jet_solver():
    accel, kg_per_mps, _ = actuator_twin.fly_burn(Actuation.boost_burn, 0.0, 4.0)
    assert accel > 0.125 and kg_per_mps < 4.4               # boosted thrust at the minimum fuel cost
    accel, kg_per_mps, _ = actuator_twin.fly_burn(Actuation.plain_burn, 0.0, 2.0)
    assert 0.09 < accel < 0.11 and kg_per_mps < 4.4
    for lin, ang, turn in Actuation.ZJETS:                   # each command fires its four +z jets at full thrust
        acc, _, _ = actuator_twin.tick(lin, ang)
        assert abs(acc[2] - 0.1929) < 1e-3 and np.allclose(acc[3:], turn, atol=1e-4)
    accel, kg_per_mps, err = actuator_twin.fly_burn(Actuation.jets_burn, 2.0, 20.0)
    assert accel > 0.185 and kg_per_mps < 4.4 and err < 2.0   # steers by choosing the set, all four jets stay on
    _, _, err = actuator_twin.fly_burn(Actuation.jets_burn, 25.0, 40.0)
    assert err < 3.0                                          # far off: slews back in instead of running away


def test_trim_and_coast_commands():
    lin, ang = Actuation.trim_command(np.array([0.02, -0.3, 0.1]))
    assert list(lin) == [0.0, -0.5, 0.0] and not np.any(ang)                       # the single-axis law
    assert not np.any(Actuation.trim_command(np.array([0.001, 0.002, 0.0]))[0])
    lin_ex, ang_ex = Actuation.trim_command(np.array([0.0, -0.4, 0.3]), w=np.array([0.02, 0.0, 0.0]), trim_exact=True)
    assert np.allclose(lin_ex, [0.0, -0.4, 0.3]) and ang_ex[0] < 0                 # exact direction, tumble damped
    assert not np.any(Actuation.coast_command(np.array([0.001, 0.0, 0.0]))[1])
    assert Actuation.coast_command(np.array([0.02, 0.0, 0.0]))[1][0] < 0


def test_steer_burn_and_calibrated_stack():
    aligned = np.array([0.0, 0.1, 0.995]) / np.linalg.norm([0.0, 0.1, 0.995])
    lin_al, _ = Actuation.steer_burn(aligned, np.zeros(3), {"ng": 5.0})
    assert np.allclose(lin_al, [0.0, 0.0, Actuation.THRUST])
    wide = np.array([0.8, 0.0, 0.6])
    lin_far, ang_far = Actuation.steer_burn(wide, np.zeros(3), {"ng": 5.0})
    assert np.allclose(lin_far, 0.5 * wide) and np.linalg.norm(ang_far) > 0
    lin_low, ang_low = Actuation.steer_burn(wide, np.array([0.03, 0.0, 0.0]), {"ng": 1.0})
    assert np.allclose(lin_low, 0.5 * wide) and ang_low[0] < 0
    a_off = C.accel(C.S, np.array([0.0, 8.0, 0.0]), frame=False)
    a_on = C.accel(C.S, np.array([0.0, 8.0, 0.0]), frame=True)
    assert np.allclose(a_on - a_off, C.FRAME_B - 2.0 * np.cross(C.OMEGA, np.array([0.0, 8.0, 0.0])))
    w = Radar.TRACKERS["calibrated"]()
    assert isinstance(w, Radar.CalibratedWorldModel) and w.rr_var == pytest.approx(0.02 ** 2) and w.frame_accel is True


def test_planner_prices_turning_the_burn():
    r = C.S + 100.0 * C.SH; v = 3.0 * C.SH                                           # mid-burn, pointed along the track
    st = dict(r=r, v=v, burning=True, main=True, fuel=10.0, t=20.0)
    plans = [([C.T.copy()], 8.4), ([r + 60.0 * C.SH + 25.0 * C.E1, C.T.copy()], 8.4)]   # straight, and a dodge
    none = tuple(np.zeros((0, 3)) if i < 2 else np.zeros(0) for i in range(5))
    old = C.score_plans(plans, st, none, cfg={})
    new = C.score_plans(plans, dict(st, u=C.SH.copy()), none, cfg={"w_turn": 0.035})
    assert new["fuel"][0] == pytest.approx(old["fuel"][0], abs=0.2)                  # no turn: same price
    assert new["fuel"][1] > old["fuel"][1] + 2.0                                       # the dodge's swing now costs fuel
    rp = np.array([C.S + 400.0 * C.SH, C.S + 900.0 * C.SH]); rocks = (rp, np.zeros((2, 3)), np.full(2, 10.0),
                                                                     np.full(2, 9.0), np.full(2, 0.2))
    ev = C.score_plans(plans, st, rocks, cfg={"ignore_per_rock": True})
    ign = ev["ph"] == 0.0
    assert (ign == ign[:1]).all()                                                      # one ignore decision per rock


# ---- vision ---------------------------------------------------------------------------------------------------
def test_vision_geometry_and_classical_segmentation():
    p_body = Vision.TOP_MOUNT + np.array([200.0, 30.0, 40.0])
    top, yaw, pitch, lost = Vision.aim(p_body)
    u, v, rng = Vision.project(p_body, top, yaw, pitch)
    assert top and lost == 0.0 and abs(u - 256) < 1e-6 and abs(v - 256) < 1e-6      # aimed at it: image centre
    image = np.zeros((512, 512, 3), np.uint8); vv, uu = np.mgrid[:512, :512]
    image[np.hypot(uu - 300, vv - 200) <= 20] = 200
    seg = Vision.ClassicalSegmenter(); mask = seg.segment(image, [(300, 200, 15)])[0]
    radius = Vision.measure(mask, 300, 200, 200.0)
    assert abs(radius - 200.0 * 20 / Vision.FOCAL) < 0.6    # 20 px at 200 m is about 9 m
    assert seg.segment(image, [(60, 60, 10)])[0] is None


def lit_disc(kappa, rad=14.0, c=(100.0, 90.0), sun=(1.0, 0.0)):
    """A ball lit from `sun` as the bench camera draws it: kappa 1 full, 0 half, below 0 a crescent."""
    vv, uu = np.mgrid[:200, :200]; du, dv = uu - c[0], vv - c[1]; s = du * sun[0] + dv * sun[1]
    return (du ** 2 + dv ** 2 <= rad ** 2) & (s >= -kappa * np.sqrt(np.maximum(rad ** 2 - (du ** 2 + dv ** 2 - s ** 2), 0)))


def test_circle_finds_the_centre_of_a_partly_lit_ball():
    for kappa in (1.0, 0.0, -0.4):
        u, v, rad = Vision.circle(lit_disc(kappa))
        assert np.hypot(u - 100, v - 90) < 2.5 and abs(rad - 14) < 1.5
    rows, cols = np.nonzero(lit_disc(0.0))
    assert np.hypot(cols.mean() - 100, rows.mean() - 90) > 4.0      # the centroid leans to the sun: why circle is used


def test_classical_segmenter_without_prompts_finds_every_rock_but_not_a_star():
    image = np.zeros((200, 200, 3), np.uint8); image[lit_disc(1.0)] = 180
    image[lit_disc(1.0, 8.0, (40.0, 150.0))] = 150; image[10, 10] = 255
    masks = Vision.ClassicalSegmenter().segment(image)
    assert len(masks) == 2 and not any(m[10, 10] for m in masks)


def test_consensus_matches_through_a_shift_and_strays():
    rng = np.random.default_rng(1); pred = rng.uniform(40, 470, (40, 2)); gate = np.full(40, 6.0)
    det = np.vstack([pred[:30] + (6.0, -4.0) + rng.normal(0, 1, (30, 2)), rng.uniform(40, 470, (15, 2))])
    pairs, shift = Vision.consensus(pred, gate, det)
    assert (pairs[:, 0] == pairs[:, 1]).all() and len(pairs) >= 28 and np.linalg.norm(shift - (6, -4)) < 0.5
    assert len(Vision.consensus(pred, gate, rng.uniform(40, 470, (45, 2)))[0]) < 5


def small_world(P, radius, sig=2.0):
    """A tracker that already holds rocks P (k x 3) with position error sig per axis."""
    w = WorldModel(); w._alloc(len(P)); w.x[:, :3] = P; w.count[:] = 1; w.dsum[:] = 2 * np.asarray(radius)
    w.P[:] = np.diag([sig ** 2] * 3 + [0.01] * 3); return w


def test_a_camera_fix_corrects_across_the_line_of_sight_only():
    C.configure(); w = small_world(np.array([[300.0, 3.0, 0.0]]), [5.0])       # truth (300, 0, 0), seen from 0
    q = w.x[0, :3]; H = np.array([[[-q[1] / q[0] ** 2, 1 / q[0], 0], [-q[2] / q[0] ** 2, 0, 1 / q[0]]]])   # bearing
    e = H[0] @ (np.array([300.0, 0, 0]) - q); P0 = w.P[0].copy()
    w.sight([0], 0.0, H, e[None], [(0.5 / 300) ** 2])
    assert abs(w.x[0, 1]) < 1.0 and abs(w.x[0, 0] - 300.0) < 0.1 and np.trace(w.P[0]) < np.trace(P0)


def cross_error(x, P, r):
    """RMS error across the line of sight from r, rows of x against the truth P."""
    los = (P - r) / np.linalg.norm(P - r, axis=1, keepdims=True); e = x - P
    return np.sqrt(np.mean(np.sum((e - (e * los).sum(1, keepdims=True) * los) ** 2, 1)))


def test_consensus_on_the_emulated_camera_finds_the_layout_and_improves_the_rocks(monkeypatch):
    monkeypatch.setattr(Vision, "LAYOUT", Vision.LAYOUT)
    P, _, D, S, T = FIELD; C.configure(S, T); rng = np.random.default_rng(3)
    w = small_world(P + rng.normal(0, 2, P.shape), D / 2); before = w.x[:, :3].copy()
    eye = offline_bench.Camera(np.random.default_rng(4)); eye.scene = (np.vstack([P, T]), np.append(D / 2, 40.0), S)
    rep = Vision.Consensus(Vision.ClassicalSegmenter())(eye, w, 0.0, S, np.zeros(3))
    moved = np.flatnonzero((w.x[:, :3] != before).any(1))
    assert rep["used"] and rep["layout"] == eye.layout and len(moved) >= 10
    assert cross_error(w.x[moved, :3], P[moved], S) < 0.7 * cross_error(before[moved], P[moved], S)


class StillSim:
    """A sim whose camera returns one fixed image."""
    def __init__(self, image): self.image = image
    def to_body(self, vec): return np.asarray(vec, float)
    def capture(self, *a): return self.image


def test_consensus_leaves_the_tracker_alone_on_an_empty_sky(monkeypatch):
    monkeypatch.setattr(Vision, "LAYOUT", Vision.LAYOUT)
    C.configure(); w = small_world(np.array([[300.0, 0, 0], [400.0, 20.0, 0]]), [5.0, 5.0]); x0 = w.x.copy()
    stars = np.zeros((512, 512, 3), np.uint8); stars[np.random.default_rng(0).integers(0, 512, (2, 300)).tolist()] = 255
    for image in (np.zeros((512, 512, 3), np.uint8), stars):
        Vision.Consensus(Vision.ClassicalSegmenter())(StillSim(image), w, 0.0, np.zeros(3), np.zeros(3))
        assert (w.x == x0).all()


def test_inspect_measures_a_rock_the_top_camera_can_reach():
    P, _, D, S, T = FIELD; C.configure(S, T)
    eye = offline_bench.Camera(np.random.default_rng(5)); eye.err = np.eye(3)
    rock = S + eye.B.T @ (Vision.TOP_MOUNT + [200.0, 0, 150.0])                   # 37 degrees up from body +x
    w = small_world(rock[None], [8.0], sig=0.5); eye.scene = (rock[None], np.array([10.0]), S)
    v = 8.0 * (rock - S) / np.linalg.norm(rock - S)                               # closest approach in 31 s
    i, radius = Vision.Inspect(Vision.ClassicalSegmenter())(eye, w, 0.0, S, v)
    assert i == 0 and abs(radius - 10.0) < 1.5


# ---- closed loop ----------------------------------------------------------------------------------------------
@pytest.mark.slow
def test_closed_loop_flight_arrives_without_a_hit():
    res = offline_bench.run(BasicController.PLANNER, 0, FIELD)                  # the slow simconf field
    assert res["arrived"] and res["hits"] == 0 and res["score"] > 45.0 and res["guard_steps"] < 100     # old slow field
    from tools import sim_bench                                                 # the field as the sim flew it
    res = sim_bench.run(BasicController.PLANNER, 0)
    assert res["arrived"] and res["hits"] == 0 and res["score"] > 40.0      # 44 at the 11-13 defaults
    res10 = sim_bench.run("real_calibrated", 0)
    assert res10["arrived"] and res10["hits"] == 0 and res10["score"] > 35.0


def test_the_guard_dodges_a_rock_crossing_the_route():
    C.configure(); flight = Flight("velocity_obstacle", config=dict(get_config("velocity_obstacle"), guard_mode="escape"))
    for ahead, side, speed, label in ((132.0, 88.0, 4.0, "crossing"), (198.0, 0.0, -3.0, "head-on")):
        r = C.S.copy(); v = 6.0 * C.SH                          # coasting at 6 m/s; both rocks meet it in 22 s,
        rp = C.S + ahead * C.SH + side * C.E1                    # just past the guard's 20 s horizon
        rv = (-speed * C.E1) if side else (speed * C.SH)
        hard, sp = np.array([12.0]), np.array([1.0]); dt = 0.1; closest = np.inf; dodged = False
        for _ in range(400):                                     # 40 s, the guard re-deciding every tick
            d = flight._guard(r, v, (rp[None], rv[None], hard, sp, np.array([0.02]))); flight.guarding = d is not None
            if d is not None:
                v = v + C.A_TRIM * d * dt; dodged = True
            r = r + v * dt; rp = rp + rv * dt; closest = min(closest, np.linalg.norm(r - rp))
        assert dodged and closest > hard[0], f"{label}: closest {closest:.1f} m"


def test_hit_risk_is_priced_at_relative_speed_and_between_plan_samples():
    C.configure(); K = 40; dt = 0.5; x = np.arange(K) * 5.0   # from the start, along the line at 10 m/s
    path = (C.S + x[:, None] * C.SH)[:, None, :]; spd = np.full((K, 1), 10.0)
    arr = np.array([K - 1]); dvs = np.zeros(1); nn = np.array([1])
    rock = lambda s, u, hard=5.0: (np.array([C.S + s * C.SH]), np.array([u * C.SH]), np.array([hard]),
                                   np.array([0.3]), np.array([0.01]))
    ev = lambda r: C.evaluate(path, spd, arr, dvs, nn, r, dt=dt, cfg={"rel_hit": True})
    head_on, same_way = ev(rock(100, -4)), ev(rock(30, 6))   # meet at 14 m/s and at 4 m/s
    assert head_on["ph"][0, 0] > 0.99 and same_way["ph"][0, 0] > 0.99
    assert head_on["cost"][0] > same_way["cost"][0] + 20     # 14 m/s relative costs more than 4 m/s
    between = ev(rock(12.5, 0, hard=1.5))                     # 2.5 m from both neighbouring samples
    assert between["mind"][0, 0] < 0.1 and between["ph"][0, 0] > 0.99


def test_a_sim_sped_up_in_the_app_is_flagged_and_flown_with_the_jets_off(monkeypatch):
    from types import SimpleNamespace
    wall = [0.0]
    monkeypatch.setattr(BasicController, "st", SimpleNamespace(VarType=SimpleNamespace(double="double")))
    monkeypatch.setattr(BasicController, "time", SimpleNamespace(sleep=lambda s: wall.__setitem__(0, wall[0] + s),
                                                               monotonic=lambda: wall[0]))
    for speed, flagged in ((1.0, False), (50.0, True)):
        sent = []; sim = BasicController.SpaceTeamsSim.__new__(BasicController.SpaceTeamsSim)
        sim.api = SimpleNamespace(this=SimpleNamespace(GetParam=lambda kind, key: 60.0),
                                  SetCommand=lambda lin, ang: sent.append(np.linalg.norm(lin)))
        sim.time = lambda: speed * wall[0]; sim.log = lambda msg, error=False: None
        for _ in range(4 * 60):                             # 4 s of wall time at 60 Hz
            sim.command(np.array([0.0, 0.0, 0.5]), np.zeros(3)); sim.wait()
        assert sim.invalid == flagged and (sent[-1] == 0.0) == flagged


def test_the_simconf_part_names_reach_the_flight(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(BasicController, "st", SimpleNamespace(VarType=SimpleNamespace(string="string")))
    params = {"Planner": "conjunction", "Camera": "off"}

    def get(kind, key):
        if key not in params:
            raise RuntimeError(f"no parameter {key}")
        return params[key]
    sim = BasicController.SpaceTeamsSim.__new__(BasicController.SpaceTeamsSim)
    sim.api = SimpleNamespace(this=SimpleNamespace(GetParam=get))
    assert sim.choices() == {"planner": "conjunction", "camera": "off"}
    params.clear()
    assert sim.choices() == {}                              # the organizers' simconfs: the lines at the top apply


def test_world_field_follows_the_world_definition():
    from tools.phobos_world import WorldField, spawn_impulses
    dv = spawn_impulses(np.full(500, 1e5))
    assert np.allclose(dv, spawn_impulses(np.full(500, 1e5))) and 0.015 < dv.std() < 0.025   # SpawnDebris seed 8193
    C.configure(); f = WorldField(3, seconds=40.0, bumps=False)
    v0 = f.P[1] - f.P[0]; push = np.median((f.P[16] - f.P[14]) / 2 - v0, axis=0)
    assert -3.8 < push[0] < -2.2 and abs(push[1]) < 0.1                            # the start-up push along -x
    late = (f.P[40] - f.P[38]) / 2 - (f.P[32] - f.P[30]) / 2
    assert np.median(np.linalg.norm(late, axis=1)) < 0.1                           # then smooth gravity and frame terms
