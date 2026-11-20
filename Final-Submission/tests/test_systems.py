"""
Unit tests for UserAPI constants, camera transforms, radar entry parsing and the scoring model.
"""

import math
import numpy as np
import pytest

from Systems.UserAPI import (
    RadarObservation,
    compute_camera_body_to_cam_transform,
    compute_competition_score,
    TOP_CAMERA_YAW_LIMITS,
    TOP_CAMERA_PITCH_LIMITS,
    BOTTOM_CAMERA_YAW_LIMITS,
    BOTTOM_CAMERA_PITCH_LIMITS,
    SHUTTLE_MASS,
    SHUTTLE_INERTIA_TENSOR
)


def test_vehicle_constants():
    assert SHUTTLE_MASS == 3388.7
    assert SHUTTLE_INERTIA_TENSOR.shape == (3, 3)
    assert SHUTTLE_INERTIA_TENSOR[0, 0] == 3025.8
    assert SHUTTLE_INERTIA_TENSOR[1, 1] == 5927.5
    assert SHUTTLE_INERTIA_TENSOR[2, 2] == 6756.0


def test_camera_transform():
    # Test zero angles -> identity matrix
    t_zero = compute_camera_body_to_cam_transform(0.0, 0.0)
    assert np.allclose(t_zero, np.eye(3))

    # Test 90 degree yaw
    t_yaw90 = compute_camera_body_to_cam_transform(90.0, 0.0)
    expected_yaw90 = np.array([
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    assert np.allclose(t_yaw90, expected_yaw90, atol=1e-5)


def test_radar_observation_parsing():
    raw_matrix = np.array([
        [10.0, 20.0, 30.0],
        [37.4, -2.5, 5.0],
        [1.5, 0.0, 0.0]
    ])
    obs = RadarObservation.from_matrix(raw_matrix)
    assert np.allclose(obs.rel_center, np.array([10.0, 20.0, 30.0]))
    assert obs.closest_range == 37.4
    assert obs.range_rate == -2.5
    assert obs.diameter == 5.0
    assert obs.timestamp == 1.5

    shuttle_loc = np.array([100.0, 200.0, 300.0])
    abs_pos = obs.abs_position(shuttle_loc)
    assert np.allclose(abs_pos, np.array([110.0, 220.0, 330.0]))


def test_scoring_model():
    # Example from competition wiki:
    # 80% of nominal time (Time Score = 1.25)
    # Damage = 5% of max (Penalty = 0.10)
    # Fuel = 90% of nominal (Penalty = 0.90)
    # Raw Score = 1.25 - 0.10 - 0.90 = 0.25 -> Final Score = (0.25 + 0.5) * 100 = 75
    score = compute_competition_score(
        actual_time=96.0,       # 120 * 0.8
        actual_damage=5.0,      # 100 * 0.05
        actual_fuel=450.0,      # 500 * 0.9
        nominal_time=120.0,
        max_damage=100.0,
        nominal_fuel=500.0
    )
    assert math.isclose(score.time_score, 1.25, rel_tol=1e-4)
    assert math.isclose(score.damage_penalty, 0.10, rel_tol=1e-4)
    assert math.isclose(score.fuel_penalty, 0.90, rel_tol=1e-4)
    assert math.isclose(score.raw_score, 0.25, rel_tol=1e-4)
    assert math.isclose(score.final_score, 75.0, rel_tol=1e-4)
