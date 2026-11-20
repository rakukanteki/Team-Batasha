"""Body-axis pointing and APF candidate checks for Plan 10 variants."""
import json
from pathlib import Path

import numpy as np
import pytest

from Systems import Actuation, BasicController, Core as C
from Systems.plans import get_config
from Systems.plans.plan_10_real_calibrated.calibrated import apf_waypoint, select_real_calibrated
from tools.actuator_twin import TwinSim

ROOT = Path(__file__).resolve().parents[1]


def config(variant):
    path = ROOT / 'Systems/plans/plan_10_real_calibrated/configs' / f'{variant}.json'
    return {**get_config(10), **json.loads(path.read_text())}


def test_x_attitude_handles_alignment_opposition_and_rate_damping():
    assert np.allclose(Actuation.target_x_attitude([1, 0, 0], [0, 0, 0]), 0)
    assert Actuation.target_x_attitude([-1, 0, 0], [0, 0, 0])[2] > 0
    assert Actuation.target_x_attitude([0, 1, 0], [0, 0, 0])[2] > 0
    assert Actuation.target_x_attitude([0, 0, 1], [0, 0, 0])[1] < 0
    assert Actuation.target_x_attitude([1, 0, 0], [0.01, 0, 0])[0] < 0
    lin, _ = Actuation.target_x_burn([0, 1, 0], np.zeros(3), {'ref': [1, 0, 0]})
    assert np.allclose(lin, [0, Actuation.THRUST, 0])


@pytest.mark.parametrize('mode', ['main', 'trim', 'coast', 'evade'])
def test_body_x_tracks_target_in_every_flight_mode(mode, monkeypatch):
    from Systems.Flight import Flight
    C.configure()
    monkeypatch.setattr(Actuation, 'THRUST', Actuation.THRUST)
    monkeypatch.setattr(Flight, 'decide', lambda self, *args: (mode, C.SH.copy()))
    sim = TwinSim(80.0)
    BasicController.fly(sim, planner='real_calibrated', config=config('x_forward'),
                        threaded=False, log_run=False)
    ref = sim.to_body(sim.goal - sim.r)
    assert ref[0] / np.linalg.norm(ref) > np.cos(np.radians(5))
    assert not any('ERROR' in line for line in sim.lines)


def test_apf_repels_from_surface_and_adds_scored_candidate(monkeypatch):
    C.configure()
    cfg = config('x_forward_apf')
    st = dict(r=C.S.copy(), v=6 * C.SH, burning=False, main=False, fuel=0., t=20.)
    empty = (np.empty((0, 3)), np.empty((0, 3)), np.empty(0), np.empty(0), np.empty(0))
    assert apf_waypoint(st, empty, cfg) is None
    rocks = (np.array([C.S + 80 * C.SH + 10 * C.E1]), np.zeros((1, 3)),
             np.array([10.]), np.array([.5]), np.array([.02]))
    waypoint = apf_waypoint(st, rocks, cfg)
    assert (waypoint - C.S) @ C.SH > 0
    assert (waypoint - C.S) @ C.E1 < 0
    captured = []
    def score(plans, *args, **kwargs):
        captured.extend(plans)
        n = len(plans)
        return dict(cost=np.arange(n, dtype=float), ptot=np.zeros(n), clean=np.zeros(n), clear=np.ones(n))
    monkeypatch.setattr(C, 'score_plans', score)
    select_real_calibrated(st, rocks, None, False, cfg)
    assert any(np.allclose(p[0][0], waypoint) for p in captured)
    assert all(np.allclose(p[0][-1], C.T) or np.linalg.norm(p[0][-1] - C.T) <= 45 for p in captured)


@pytest.mark.parametrize('variant', ['x_forward', 'x_forward_apf'])
def test_generated_sim_selects_variant(variant):
    sim = json.loads((ROOT / f'sims/PLAN_10_real_calibrated_{variant}.simconf').read_text())
    params = next(s for s in sim['SystemManager']['Systems'] if s['Nametag'] == 'Controller')['Inst_Parameters']
    assert params['Planner'] == 'real_calibrated'
    assert params['Variant'] == variant
    assert config(variant)['burn'] == 'target_x'
