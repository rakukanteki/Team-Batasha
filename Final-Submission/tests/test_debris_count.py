import numpy as np
import pytest
from tools.phobos_world import WorldField, WorldSim


def test_debris_subset_keeps_original_paths_and_logs_matching_geometry():
    field = WorldField(0, seconds=12)
    sim = WorldSim(0, field=field, tmax=1, debris_count=50)
    assert len(field.dia) == 500
    assert len(sim.f.dia) == 50
    assert sim.f.P.shape[1] == len(sim.f.vis) == 50
    np.testing.assert_array_equal(sim.f.P, field.P[:, sim.scene_indices])
    np.testing.assert_array_equal(sim.f.at(3)[0], field.at(3)[0][sim.scene_indices])
    class Logger:
        def log_scene(self, t, positions, diameters, attitudes):
            assert len(positions) == len(diameters) == len(attitudes) == 50
    sim.log_scene(Logger())
    with pytest.raises(ValueError): WorldSim(0, field=field, tmax=1, debris_count=501)
