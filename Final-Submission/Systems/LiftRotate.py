"""Rise 30 m, turn, and burn forward while recording the test."""
import csv
from pathlib import Path
from tempfile import mkdtemp
from threading import Thread
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from Systems.RunLogger import DEFAULT_LOG_DIR, _encode_png_rgb


def save_photos(sim, directory, sim_time):
    # Capture sequentially on one worker to avoid overlapping engine readbacks.
    for name, top in (("top", True), ("bottom", False)):
        try:
            image = sim.capture_fixed(top=top)
            if image is None:
                raise RuntimeError("no image returned")
            (directory / f"{name}_t{sim_time:010.3f}.png").write_bytes(_encode_png_rgb(image))
        except Exception as exc:
            sim.log(f"[lift_rotate] {name} camera capture failed at {sim_time:.3f} s: {exc}", error=True)


def run(sim):
    root = DEFAULT_LOG_DIR / "lift_rotate"
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(mkdtemp(prefix="run_", dir=root))
    photos = directory / "images"
    photos.mkdir()
    sim.log(f"[lift_rotate] Saving pose and photos to {directory}")
    with (directory / "pose.csv").open("w", encoding="utf-8", newline="", buffering=1) as file:
        pose = csv.writer(file)
        pose.writerow(("sim_time_s", "phase", "phobos_x_m", "phobos_y_m", "phobos_z_m",
                       "quat_x", "quat_y", "quat_z", "quat_w", "roll_deg", "pitch_deg", "yaw_deg"))
        fly(sim, pose, photos)


def log_pose(pose, sim_time, phase, position, orientation):
    # At gimbal lock, SciPy sets the third Euler angle to zero. The quaternion stays valid.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Gimbal lock detected.*")
        euler = orientation.as_euler("xyz", degrees=True)
    pose.writerow((sim_time, phase, *position, *orientation.as_quat(), *euler))


def fly(sim, pose, photos):
    start, _ = sim.state()
    up = start / np.linalg.norm(start)
    goal = start + 30.0 * up
    attitude = Rotation.from_quat(sim.body_quaternion())
    started_at = next_photo = sim.time()
    log_pose(pose, started_at, "initial_pre_thrust", start, attitude)
    camera_worker = None
    turning = accelerating = False

    # Axis is expressed in the shuttle's body frame at the start of the turn.
    #turn = Rotation.from_euler("x", 90, degrees=True)
    turn = Rotation.from_euler("y", -90, degrees=True)
    # turn = Rotation.from_euler("z", 90, degrees=True)

    sim.log("[lift_rotate] Climbing 30 m from the starting position")
    while True:
        sim_time = sim.time()
        if sim_time >= next_photo:
            next_photo = started_at + int(sim_time - started_at) + 1
            # Camera readback blocks. Keep it off the flight-control loop.
            if camera_worker is None or not camera_worker.is_alive():
                camera_worker = Thread(target=save_photos, args=(sim, photos, sim_time), daemon=True)
                camera_worker.start()
            else:
                sim.log(f"[lift_rotate] Camera still busy at {sim_time:.3f} s", error=True)

        position, velocity = sim.state()
        orientation = Rotation.from_quat(sim.body_quaternion())
        rates = sim.rates()
        height = (position - start) @ up
        if not turning and abs(height - 30.0) < 0.75 and np.linalg.norm(velocity) < 0.15:
            attitude = orientation * turn
            turning = True
            sim.log(f"[lift_rotate] Starting turn at {height:.2f} m")

        rotation_error = (orientation.inv() * attitude).as_rotvec()
        error_deg = np.rad2deg(np.linalg.norm(rotation_error))
        if turning and not accelerating and error_deg < 2.0 and np.linalg.norm(rates) < 0.02:
            sim.log(f"[lift_rotate] Turn settled at {height:.2f} m; accelerating along the new +Z heading")
            accelerating = True

        if accelerating:
            phase = "forward_burn"
            acceleration = attitude.apply([0.0, 0.0, 0.10])
        else:
            phase = "turn" if turning else "climb"
            # Brake at the setpoint and hold position during the turn.
            remaining_height = 30.0 - height
            desired_velocity = np.clip(0.18 * remaining_height, -1.5, 1.5) * up
            lateral_error = goal - position - remaining_height * up
            acceleration = 0.55 * (desired_velocity + 0.12 * lateral_error - velocity)
        acceleration += 0.0104 * up
        desired_rates = np.clip(0.8 * rotation_error, -0.20, 0.20)
        angular_acceleration = np.clip(2.0 * (desired_rates - rates), -0.12, 0.12)
        log_pose(pose, sim_time, phase, position, orientation)
        sim.command(np.clip(sim.to_body(acceleration), -0.25, 0.25), angular_acceleration)
        if not sim.wait():
            return
