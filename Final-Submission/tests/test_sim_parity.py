import numpy as np
from scipy.spatial.transform import Rotation

from Systems import Core
from tools.actuator_twin import TwinSim, tick
from tools.compare_run import check_dynamics
from tools.sim_bench import FIELD, FieldSim


def test_command_replay_uses_logged_body_to_phobos_attitude():
    rotation = Rotation.from_euler("y", 90, degrees=True)
    position = Core.S.copy()
    command = [0.0, 0.0, 0.1]
    acceleration, _, _ = tick(command, [0.0, 0.0, 0.0])
    from tools.replay_bench import TRUE_B

    velocity = rotation.apply(acceleration[:3]) + Core.grav(position) + TRUE_B
    control = {"lin_cmd": command, "ang_cmd": [0.0, 0.0, 0.0]}
    run = {
        "metadata": {"core": {"start": position, "target": Core.T, "a_burn": Core.A_BURN}},
        "time_to_point": {
            "0.0000": {"shuttle_pos": position, "shuttle_vel": np.zeros(3),
                       "body_quaternion_xyzw": rotation.as_quat()},
            "1.0000": {"shuttle_pos": position + velocity, "shuttle_vel": velocity},
        },
        "time_to_thrust": {"0.0000": control, "1.0000": control},
    }
    metrics = check_dynamics(run)
    assert metrics["final_pos_error_m"] == metrics["final_speed_error_mps"] == 0.0


def test_recorded_field_starts_at_measured_state():
    sim = FieldSim(0, field=FIELD)
    with np.load(FIELD) as field:
        np.testing.assert_array_equal(sim.r, field["r_first"])
        np.testing.assert_array_equal(sim.v, field["v_first"])
        np.testing.assert_array_equal(sim.R, field["axes_first"])
        assert sim.time() == field["t_first"]


def test_offline_step_size_does_not_set_actuator_tick_length():
    sims = [TwinSim(seconds=1, hz=hz) for hz in (30, 60)]
    for sim in sims:
        sim.command([0.001, 0.0, 0.0], [0.0, 0.0, 0.0])
        while sim.time() < 1 - sim.dt / 2:
            sim.wait()
    np.testing.assert_allclose(sims[0].used, sims[1].used, atol=1e-12)
