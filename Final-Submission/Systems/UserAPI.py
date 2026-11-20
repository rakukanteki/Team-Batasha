"""
Systems/UserAPI.py - Core API, Data Structures, and Sensor Interfaces
for Space Teams University 2026: Escape from Phobos Competition.
"""

from __future__ import annotations
import math
import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import numpy as np

try:
    import spaceteams as st
except ImportError:
    # Allow mock / local testing outside Space Teams PRO environment
    st = None


# ==============================================================================
# Vehicle and Competition Constants
# ==============================================================================

SHUTTLE_MASS: float = 3388.7  # kg, constant during flight
SHUTTLE_COM: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float64)  # m

SHUTTLE_INERTIA_TENSOR: np.ndarray = np.array([
    [3025.8, -171.7,  461.7],
    [-171.7, 5927.5,  313.6],
    [ 461.7,  313.6, 6756.0]
], dtype=np.float64)  # kg * m^2

# Camera specs
TOP_CAMERA_POS: np.ndarray = np.array([1.27, 0.0, 0.9], dtype=np.float64)
TOP_CAMERA_YAW_LIMITS: Tuple[float, float] = (-89.0, 89.0)     # degrees
TOP_CAMERA_PITCH_LIMITS: Tuple[float, float] = (-50.0, 0.0)    # degrees (turns downward)

BOTTOM_CAMERA_POS: np.ndarray = np.array([1.675, 0.0, -1.025], dtype=np.float64)
BOTTOM_CAMERA_YAW_LIMITS: Tuple[float, float] = (-89.0, 89.0)  # degrees
BOTTOM_CAMERA_PITCH_LIMITS: Tuple[float, float] = (0.0, 50.0)  # degrees (turns downward)

# Radar measurement standard deviations
RADAR_POS_STD_DEV: float = 2.0         # meters
RADAR_RANGE_STD_DEV: float = 2.0       # meters
RADAR_RANGE_RATE_STD_DEV: float = 0.02 # m/s
RADAR_DIAMETER_STD_PERCENT: float = 0.05  # 5% of true diameter


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass
class VehicleState:
    """State of the escape shuttle."""
    loc: np.ndarray              # [x, y, z] in Phobos-fixed frame (m)
    vel: np.ndarray              # [vx, vy, vz] in Phobos-fixed frame (m/s)
    acc: np.ndarray              # [ax, ay, az] in Phobos-fixed frame (m/s^2)
    angular_velocity: np.ndarray # [wx, wy, wz] (rad/s)
    rotation: any                # Quaternion or rotation object from Space Teams
    mass: float = SHUTTLE_MASS
    damage: float = 0.0
    fuel_used: float = 0.0


@dataclass
class TargetState:
    """State of the Earth-return spacecraft."""
    loc: np.ndarray  # [x, y, z] in Phobos-fixed frame (m)
    vel: np.ndarray  # [vx, vy, vz] in Phobos-fixed frame (m/s)


@dataclass
class RadarObservation:
    """
    Parsed 3x3 Radar observation matrix R:
    [ x      y      z     ]  <- Center of mass with respect to shuttle (Phobos frame)
    [ rc     vr     d     ]  <- Closest range, range rate, average diameter
    [ dt     0      0     ]  <- Measurement timestamp since start
    """
    rel_center: np.ndarray       # [x, y, z] offset from shuttle in Phobos frame (m)
    closest_range: float         # rc (m)
    range_rate: float            # vr (m/s, positive = receding)
    diameter: float              # d (m)
    timestamp: float             # delta t (seconds since simulation start)
    raw_matrix: np.ndarray

    @classmethod
    def from_matrix(cls, mat: np.ndarray) -> "RadarObservation":
        rel_pos = np.array(mat[0, :], dtype=np.float64)
        closest_r = float(mat[1, 0])
        vr = float(mat[1, 1])
        dia = float(mat[1, 2])
        dt = float(mat[2, 0])
        return cls(
            rel_center=rel_pos,
            closest_range=closest_r,
            range_rate=vr,
            diameter=dia,
            timestamp=dt,
            raw_matrix=mat
        )

    def abs_position(self, shuttle_loc: np.ndarray) -> np.ndarray:
        """Returns estimated absolute position of debris in Phobos frame."""
        return shuttle_loc + self.rel_center

    def estimated_velocity(self, shuttle_vel: np.ndarray) -> np.ndarray:
        """Approximates debris velocity vector from range rate and radial unit vector."""
        r_norm = np.linalg.norm(self.rel_center)
        if r_norm < 1e-4:
            return shuttle_vel.copy()
        radial_unit = self.rel_center / r_norm
        # Radial component in Phobos frame:
        return shuttle_vel + self.range_rate * radial_unit


# ==============================================================================
# Helper Functions
# ==============================================================================

def get_sim_time(t0: datetime.datetime) -> float:
    """Get elapsed simulation time in seconds."""
    if st is None:
        return 0.0
    t_now = st.SimGlobals.SimClock.GetTimeNow().as_datetime()
    return (t_now - t0).total_seconds()






def compute_camera_body_to_cam_transform(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """
    Calculates the 3-2-1 Euler body-to-camera rotation matrix:
    T = R_pitch(theta) * R_yaw(psi)
    Positive yaw turns camera left; positive pitch turns camera down.
    """
    psi = math.radians(yaw_deg)
    theta = math.radians(pitch_deg)

    r_pitch = np.array([
        [ math.cos(theta), 0.0, -math.sin(theta)],
        [ 0.0,             1.0,  0.0            ],
        [ math.sin(theta), 0.0,  math.cos(theta)]
    ], dtype=np.float64)

    r_yaw = np.array([
        [ math.cos(psi),  math.sin(psi), 0.0],
        [-math.sin(psi),  math.cos(psi), 0.0],
        [ 0.0,            0.0,           1.0]
    ], dtype=np.float64)

    return np.matmul(r_pitch, r_yaw)




def CaptureCameraImage(camera_entity: any, res_x: int = 512, res_y: int = 512, fov: float = 60.0):
    """
    Captures an image from the camera using Unreal client capture.
    Note: Blocks until returned from Unreal engine (typically 100-300ms).
    """
    if st is None or camera_entity is None:
        return None
    params = st.CaptureImageProperties()
    params.nonphys_EV = 12.0
    params.FOV = fov
    params.ResolutionX = res_x
    params.ResolutionY = res_y
    return st.camera.CaptureImageBlocking(camera_entity, params)


# ==============================================================================
# Scoring Model
# ==============================================================================

@dataclass
class ScoreBreakdown:
    time_score: float
    damage_penalty: float
    fuel_penalty: float
    raw_score: float
    final_score: float


def compute_competition_score(
    actual_time: float,
    actual_damage: float,
    actual_fuel: float,
    nominal_time: float = 120.0,
    max_damage: float = 2.092e5,
    nominal_fuel: float = 75.0
) -> ScoreBreakdown:
    """
    Computes official competition score based on rules:
    - Time Score = Nominal Time / Actual Time
    - Damage Penalty = 2 * (Actual Damage / Max Damage)
    - Fuel Penalty = Actual Fuel / Nominal Fuel
    - Raw Score = Time Score - Damage Penalty - Fuel Penalty
    - Final Score = (Raw Score + 0.5) * 100
    """
    time_score = nominal_time / max(actual_time, 1e-3)
    damage_penalty = 2.0 * (actual_damage / max(max_damage, 1e-3))
    fuel_penalty = actual_fuel / max(nominal_fuel, 1e-3)
    raw_score = time_score - damage_penalty - fuel_penalty
    final_score = (raw_score + 0.5) * 100.0
    return ScoreBreakdown(
        time_score=time_score,
        damage_penalty=damage_penalty,
        fuel_penalty=fuel_penalty,
        raw_score=raw_score,
        final_score=final_score
    )
