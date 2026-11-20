"""Fail when organizer updates (submodules/) drift from what our code assumes.

Fix a failure with `python tools/sync_upstream.py --no-pull` (Controller.py) or by
updating the constant the message names. Never edit Controller.py by hand.
"""
import inspect
import json
from pathlib import Path

import numpy as np

from Systems import UserAPI
from tools.sync_upstream import OUTPUT, build_controller

SIMCONF = Path(__file__).resolve().parent.parent / "submodules/STU4-Sim/Sims/FINAL_SUBMISSION.simconf"
SIM = json.loads(SIMCONF.read_text(encoding="utf-8"))


def system_params(nametag):
    return next(s["Inst_Parameters"] for s in SIM["SystemManager"]["Systems"] if s["Nametag"] == nametag)


def find_entity(name, node=SIM):
    if isinstance(node, dict):
        if node.get("#Required", {}).get("Name") == name:
            return node
        children = node.values()
    elif isinstance(node, list):
        children = node
    else:
        return None
    return next((e for c in children if (e := find_entity(name, c)) is not None), None)


def test_controller_matches_upstream_boilerplate():
    assert OUTPUT.read_text(encoding="utf-8") == build_controller(), \
        "Controller.py is stale: run `python tools/sync_upstream.py --no-pull`"


def test_scoring_defaults_match_simconf():
    p = system_params("Scoring")
    d = inspect.signature(UserAPI.compute_competition_score).parameters
    assert d["nominal_time"].default == p["NominalTime_s"]
    assert d["max_damage"].default == p["DamageThreshold_J"]
    assert d["nominal_fuel"].default == p["NominalFuel_kg"]


def test_radar_noise_matches_simconf():
    p = system_params("Radar")
    assert UserAPI.RADAR_POS_STD_DEV == p["Position_1sigma_m"]
    assert UserAPI.RADAR_RANGE_STD_DEV == p["Position_1sigma_m"]
    assert UserAPI.RADAR_RANGE_RATE_STD_DEV == p["Velocity_1sigma_m_s"]
    assert UserAPI.RADAR_DIAMETER_STD_PERCENT == p["Diameter_1sigma_fraction"]


def test_shuttle_mass_properties_match_simconf():
    dyn = find_entity("SpaceBuggy")["Dynamics"]
    assert UserAPI.SHUTTLE_MASS == dyn["Mass"]
    np.testing.assert_allclose(UserAPI.SHUTTLE_INERTIA_TENSOR, np.reshape(dyn["Inertia_BodyFrame"][1:], (3, 3)))


def test_plan_sims_are_the_organizers_sim_with_only_our_part_names():
    from Systems.Planners import PLANNERS
    from Systems.plans import PLAN_DIRS
    from tools.sync_upstream import SIM_PARTS, SIM_PLANS, SIMS, build_sims, sim_variants
    built = build_sims()
    assert len(built) == len(SIM_PLANS) + sum(len(sim_variants(n, m)) for n, m in SIM_PLANS.items())
    for path, sim in built.items():
        n = int(path.name.split("_")[1]); name = SIM_PLANS[n]
        assert PLANNERS[name] is PLANNERS[n], f"{path.name}: {name} is not plan {n}"
        assert path.exists() and json.loads(path.read_text(encoding="utf-8")) == sim, \
            f"{path.name} is stale: run `python tools/sync_upstream.py --no-pull`"
        ours = next(s for s in sim["SystemManager"]["Systems"] if s["Nametag"] == "Controller")["Inst_Parameters"]
        variant = ours.pop("Variant", None)                      # a config variant: configs/<variant>.json must exist
        if variant is not None:
            assert path.stem.endswith("_" + variant) and (PLAN_DIRS[name] / "configs" / f"{variant}.json").is_file()
        expected = {"Planner": name, **SIM_PARTS}
        if variant:
            preset = json.loads((PLAN_DIRS[name] / "configs" / f"{variant}.json").read_text())
            expected["Camera"] = preset.get("camera", expected["Camera"])
        assert {k: ours.pop(k) for k in expected} == expected
        assert sim == SIM, f"{path.name} changes the organizers' sim beyond the Controller's part names"
    training_path = SIMS / "TRAINING_RADAR_CAPTURE.simconf"
    assert set(built) | {training_path, SIMS / "TEST_LIFT_ROTATE.simconf"} == set(SIMS.glob("*.simconf")), \
        "sims/ holds a simconf the generator does not make"


def test_lift_rotate_scene_only_changes_controller_mode():
    from tools.sync_upstream import SIMS, build_lift_rotate_scene

    scene = json.loads((SIMS / "TEST_LIFT_ROTATE.simconf").read_text(encoding="utf-8"))
    assert scene == build_lift_rotate_scene()
    controller = next(s for s in scene["SystemManager"]["Systems"] if s["Nametag"] == "Controller")
    assert controller["Inst_Parameters"].pop("ManeuverTest") == "lift_rotate"
    assert scene == SIM
