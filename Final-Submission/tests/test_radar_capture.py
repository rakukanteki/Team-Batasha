import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from Radar_Capture import TrainingSource
from Systems.BasicController import SpaceTeamsTrainingSource
from Systems.TrainingCapture import CaptureSpec, TrainingRecorder
from tools.sync_upstream import SIMS, UPSTREAM_SIM, build_radar_training_scene, build_sims


def test_training_scene_is_separate_and_keeps_stock_radar():
    scene = json.loads((SIMS / "TRAINING_RADAR_CAPTURE.simconf").read_text(encoding="utf-8"))
    assert scene == build_radar_training_scene()
    original = json.loads(UPSTREAM_SIM.read_text(encoding="utf-8"))
    systems = lambda data: {system["Nametag"]: system for system in data["SystemManager"]["Systems"]}
    assert systems(scene)["Radar"] == systems(original)["Radar"]
    capture = systems(scene)["Radar_Capture"]
    assert capture["Inst_Parameters"]["NumDebris"] == systems(scene)["Radar"]["Inst_Parameters"]["NumDebris"]
    assert (Path(__file__).resolve().parents[1] / Path(capture["Source"]).name).is_file()
    assert not systems(scene)["Scoring"]["Active"]
    assert all("Radar_Capture" not in systems(plan) for plan in build_sims().values())


def test_training_frame_pairs_camera_with_noise_free_truth(tmp_path, monkeypatch):
    source = TrainingSource.__new__(TrainingSource)
    source.sim = SimpleNamespace(sf=object(), pf=object(), time=lambda: 4.0)
    position = np.array([3., 4., 0.])
    velocity = np.array([1., 2., 0.])
    state = SimpleNamespace(loc_WRT=lambda frame: SimpleNamespace(ExprIn=lambda axes: position),
                            vel_WRT=lambda frame: SimpleNamespace(ExprIn=lambda axes: velocity))
    source.debris = [("Debris12", SimpleNamespace(getLocVelAcc=lambda: state), 2.0)]
    noisy = np.ones((1, 3, 3))
    monkeypatch.setattr(SpaceTeamsTrainingSource, "snapshot", lambda self, spec: {"radar_raw_3x3": noisy})
    monkeypatch.setattr(source, "capture", lambda spec: {c: np.arange(6, dtype=np.uint8) for c in "RGB"})
    recorder = TrainingRecorder(tmp_path, specs=(CaptureSpec(width=3, height=2),),
                                debris_labels="noise_free_simulator_state")

    event = recorder.poll(source)
    assert event["status"] == "saved", event
    frame = json.loads((recorder.directory / event["metadata"]).read_text())
    assert (recorder.directory / frame["image"]).is_file()
    assert frame["labels"]["debris"] == "noise_free_simulator_state"
    assert frame["before"]["radar_raw_3x3"] == noisy.tolist()
    for snapshot in (frame["before"], frame["after"]):
        truth = snapshot["radar_truth"]["objects"][0]
        assert truth["entity_name"] == "Debris12"
        assert truth["closest_range_m"] == 4.0
        assert truth["relative_position_m"] == position.tolist()
        assert truth["range_rate_m_s"] == pytest.approx(2.2)
