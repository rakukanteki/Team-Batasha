"""
Convert a Team Batasha `run.json` telemetry file into a Blender `.blend` scene with timeline replay support.

Usage (from normal Python - automatically launches `blender --background`):
    python tools/json_to_blender.py flightlogs/<plan_name__timestamp>/run.json
    python tools/json_to_blender.py flightlogs/<plan_name__timestamp> --output blender/my_replay.blend --fps 30

Or directly inside Blender:
    blender --background --python tools/json_to_blender.py -- flightlogs/<plan_name__timestamp>/run.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BLENDER_DIR = ROOT / "blender"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert run.json to a Blender .blend replay file.")
    ap.add_argument("input_path", help="Path to run.json or a run folder containing run.json")
    ap.add_argument("-o", "--output", default=None, help="Output .blend path (default: blender/<run_id>.blend)")
    ap.add_argument("--fps", type=int, default=30, help="Blender timeline FPS (default: 30)")
    ap.add_argument("--time-scale", type=float, default=1.0, help="Sim seconds per replay second (default: 1.0)")
    ap.add_argument("--radar-step", type=float, default=1.0, help="Min seconds between radar debris keyframes (0 = all)")
    ap.add_argument("--max-rocks", type=int, default=500, help="Maximum debris rocks to animate (default: 500)")
    ap.add_argument("--blender-bin", default="blender", help="Blender executable when invoked outside Blender")
    return ap.parse_args(argv)


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, dict]:
    inp = Path(args.input_path).resolve()
    json_path = inp / "run.json" if inp.is_dir() else inp
    if not json_path.is_file():
        raise FileNotFoundError(f"Could not find run.json at {json_path}")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    run_id = data.get("metadata", {}).get("run_id") or json_path.parent.name or json_path.stem
    if args.output:
        out_path = Path(args.output).resolve()
    else:
        DEFAULT_BLENDER_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DEFAULT_BLENDER_DIR / f"{run_id}.blend"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return json_path, out_path, data


def find_blender(blender_bin: str = "blender") -> str | None:
    """Blender from $BLENDER, PATH, or the default Windows install folder (newest version)."""
    for cand in (os.environ.get("BLENDER"), blender_bin):
        if cand and (exe := shutil.which(cand)):
            return exe
    installs = sorted(Path(os.environ.get("ProgramFiles", r"C:\Program Files")).glob("Blender Foundation/*/blender.exe"))
    return str(installs[-1]) if installs else None


def _launch_via_blender(argv: list[str], blender_bin: str) -> int:
    exe = find_blender(blender_bin)
    if exe is None:
        raise RuntimeError("Blender not found: install it from blender.org, add it to PATH, or set BLENDER=<path to blender>.")
    cmd = [exe, "--background", "--python", str(Path(__file__).resolve()), "--", *argv]
    print(f"Launching Blender: {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    return proc.returncode


def _build_blend(json_path: Path, out_path: Path, data: dict, args: argparse.Namespace) -> None:
    import bpy
    import mathutils
    import numpy as np

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    fps = max(int(args.fps), 1)
    time_scale = max(float(args.time_scale), 1e-3)
    scene.render.fps = fps

    def make_collection(name: str):
        col = bpy.data.collections.new(name)
        scene.collection.children.link(col)
        return col

    def link_only(obj, col):
        for c in list(obj.users_collection):
            c.objects.unlink(obj)
        col.objects.link(obj)

    def make_material(name: str, rgba, emission_strength: float = 0.0, roughness: float = 0.5, alpha: float = 1.0):
        mat = bpy.data.materials.new(name=name)
        if not mat.use_nodes:
            mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = (rgba[0], rgba[1], rgba[2], 1.0)
            bsdf.inputs["Roughness"].default_value = roughness
            if "Alpha" in bsdf.inputs:
                bsdf.inputs["Alpha"].default_value = alpha
            if emission_strength > 0.0:
                if "Emission Color" in bsdf.inputs:
                    bsdf.inputs["Emission Color"].default_value = (rgba[0], rgba[1], rgba[2], 1.0)
                if "Emission Strength" in bsdf.inputs:
                    bsdf.inputs["Emission Strength"].default_value = emission_strength
        return mat

    def add_sphere(name: str, loc, radius: float, mat, col, subdivisions: int = 2):
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=subdivisions, radius=radius, location=tuple(loc))
        obj = bpy.context.active_object
        obj.name = name
        if mat is not None:
            obj.data.materials.append(mat)
        link_only(obj, col)
        return obj

    def add_curve(name: str, pts, bevel_depth: float, mat, col):
        if len(pts) < 2:
            return None
        curve_data = bpy.data.curves.new(name=name, type="CURVE")
        curve_data.dimensions = "3D"
        curve_data.bevel_depth = bevel_depth
        spline = curve_data.splines.new("POLY")
        spline.points.add(len(pts) - 1)
        for i, p in enumerate(pts):
            spline.points[i].co = (float(p[0]), float(p[1]), float(p[2]), 1.0)
        obj = bpy.data.objects.new(name, curve_data)
        if mat is not None:
            obj.data.materials.append(mat)
        col.objects.link(obj)
        return obj

    def quat_from_dir(vec, fallback=(0.0, 0.0, 1.0)):
        v = mathutils.Vector(vec)
        if v.length < 1e-6:
            v = mathutils.Vector(fallback)
        return v.normalized().to_track_quat("Z", "Y")

    def to_frame(t: float, t0: float) -> int:
        return max(1, int(round((t - t0) * fps / time_scale)) + 1)

    col_mission = make_collection("Mission")
    col_shuttle = make_collection("Shuttle")
    col_planned = make_collection("PlannedPath")
    col_debris = make_collection("RadarDebris")
    col_hud = make_collection("HUD")

    mat_start = make_material("Mat_Start", (0.1, 0.85, 0.25), emission_strength=0.6)
    mat_target = make_material("Mat_Target", (0.1, 0.85, 0.95), emission_strength=1.0)
    mat_win = make_material("Mat_WinSphere", (0.1, 0.8, 0.95), emission_strength=0.2, alpha=0.15)
    mat_nominal = make_material("Mat_Nominal", (0.4, 0.4, 0.45), emission_strength=0.2)
    mat_shuttle = make_material("Mat_Shuttle", (1.0, 0.45, 0.05), emission_strength=0.8)
    mat_thrust = make_material("Mat_Thrust", (1.0, 0.85, 0.15), emission_strength=3.0)
    mat_traj = make_material("Mat_Trajectory", (0.05, 0.9, 1.0), emission_strength=1.2)
    mat_aim = make_material("Mat_AimPoint", (0.95, 0.15, 0.85), emission_strength=1.5)
    mat_plan = make_material("Mat_PlannedPath", (1.0, 0.55, 0.05), emission_strength=1.0)
    mat_turn = make_material("Mat_TurnPoint", (1.0, 0.2, 0.1), emission_strength=1.5)
    mat_astar = make_material("Mat_AStarLattice", (0.45, 0.35, 1.0), emission_strength=1.0)
    mat_rock = make_material("Mat_DebrisRock", (0.55, 0.48, 0.42), roughness=0.9)

    meta = data.get("metadata", {})
    core = meta.get("core", {})
    start_pos = np.array(core.get("start", [0.0, 0.0, 0.0]), float)
    target_pos = np.array(core.get("target", [0.0, 0.0, -1000.0]), float)
    win_r = float(core.get("win_radius", 50.0))
    r_sh = float(core.get("r_sh", 3.0))

    # World background: dark space
    world = bpy.data.worlds.new("SpaceWorld")
    scene.world = world
    if not world.use_nodes:
        world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs["Color"].default_value = (0.01, 0.012, 0.02, 1.0)
        bg.inputs["Strength"].default_value = 1.0

    # Mission landmarks
    add_sphere("StartPad", start_pos, 6.0, mat_start, col_mission, subdivisions=2)
    add_sphere("Target_AsteroidCycler", target_pos, 10.0, mat_target, col_mission, subdivisions=2)
    win_obj = add_sphere("Target_WinSphere", target_pos, win_r, mat_win, col_mission, subdivisions=2)
    win_obj.display_type = "WIRE"
    add_curve("NominalLine", [start_pos, target_pos], 0.4, mat_nominal, col_mission)

    # Sort time-series keys chronologically
    t2p = data.get("time_to_point", {})
    t2f = data.get("time_to_fuel", {})
    t2th = data.get("time_to_thrust", {})
    times_sorted = sorted((float(k), k) for k in t2p.keys())
    t0 = times_sorted[0][0] if times_sorted else 0.0
    t_end = times_sorted[-1][0] if times_sorted else 1.0
    max_frame = to_frame(t_end, t0)

    # Shuttle + Thrust indicator + Aim point
    bpy.ops.mesh.primitive_cone_add(vertices=16, radius1=r_sh, radius2=0.5, depth=2.5 * r_sh, location=tuple(start_pos))
    shuttle = bpy.context.active_object
    shuttle.name = "Shuttle"
    shuttle.rotation_mode = "QUATERNION"
    shuttle.data.materials.append(mat_shuttle)
    link_only(shuttle, col_shuttle)

    bpy.ops.mesh.primitive_cone_add(vertices=12, radius1=1.6, radius2=0.05, depth=12.0, location=tuple(start_pos))
    thrust_cone = bpy.context.active_object
    thrust_cone.name = "ThrustVector"
    thrust_cone.rotation_mode = "QUATERNION"
    thrust_cone.data.materials.append(mat_thrust)
    link_only(thrust_cone, col_shuttle)

    aim_obj = add_sphere("ActiveAimPoint", target_pos, 4.0, mat_aim, col_shuttle, subdivisions=1)

    # Deduplicate per timeline frame so 60 Hz logs replay smoothly at target fps
    sampled = []
    seen_frames = set()
    for t_val, key in times_sorted:
        fr = to_frame(t_val, t0)
        if fr not in seen_frames or t_val == t_end:
            seen_frames.add(fr)
            sampled.append((fr, t_val, key))

    traj_pts = []
    hud_samples = {}
    nom_dir = target_pos - start_pos
    for fr, t_val, key in sampled:
        p_info = t2p[key]
        pos = p_info["shuttle_pos"]
        vel = p_info.get("shuttle_vel", [0.0, 0.0, 0.0])
        aim_pt = p_info.get("target_point", target_pos.tolist())
        fuel_val = float(t2f.get(key, 0.0))
        th_info = t2th.get(key, {})
        mode = th_info.get("mode", p_info.get("mode", "coast"))
        lin_mag = float(th_info.get("lin_mag", 0.0))
        th_dir = th_info.get("thrust_dir_world", [0.0, 0.0, 0.0])

        traj_pts.append(pos)
        shuttle.location = tuple(pos)
        shuttle.rotation_quaternion = quat_from_dir(vel, fallback=nom_dir)
        shuttle.keyframe_insert(data_path="location", frame=fr)
        shuttle.keyframe_insert(data_path="rotation_quaternion", frame=fr)

        shuttle["sim_time"] = float(t_val)
        shuttle["fuel_kg"] = fuel_val
        shuttle["thrust_mag"] = lin_mag
        shuttle.keyframe_insert(data_path='["sim_time"]', frame=fr)
        shuttle.keyframe_insert(data_path='["fuel_kg"]', frame=fr)
        shuttle.keyframe_insert(data_path='["thrust_mag"]', frame=fr)

        aim_obj.location = tuple(aim_pt)
        aim_obj.keyframe_insert(data_path="location", frame=fr)

        if mode in ("main", "trim", "evade") and lin_mag > 1e-4 and np.linalg.norm(th_dir) > 1e-4:
            d_vec = np.asarray(th_dir, float)
            d_vec /= np.linalg.norm(d_vec)
            # Place exhaust plume opposite thrust direction behind the shuttle
            plume_pos = np.asarray(pos, float) - d_vec * (r_sh + 6.0)
            thrust_cone.location = tuple(plume_pos.tolist())
            thrust_cone.rotation_quaternion = quat_from_dir(-d_vec)
            sc = 0.6 + 1.4 * lin_mag
            thrust_cone.scale = (1.0, 1.0, sc)
        else:
            thrust_cone.location = tuple(pos)
            thrust_cone.scale = (0.001, 0.001, 0.001)
        thrust_cone.keyframe_insert(data_path="location", frame=fr)
        thrust_cone.keyframe_insert(data_path="rotation_quaternion", frame=fr)
        thrust_cone.keyframe_insert(data_path="scale", frame=fr)

        dist_goal = float(np.linalg.norm(np.asarray(pos, float) - target_pos))
        hud_samples[str(fr)] = f"Plan: {meta.get('plan_name')} | t={t_val:.1f}s | mode={mode} | fuel={fuel_val:.1f}kg | thrust={lin_mag:.2f} | dist={dist_goal:.0f}m"

    add_curve("ShuttleTrajectory", traj_pts, 0.8, mat_traj, col_shuttle)

    # Planned waypoints, turn points, lattice / A* points
    points_arr = data.get("points", [])
    plan_events = [e for e in points_arr if e.get("event") == "plan"]
    if plan_events:
        last_plan = plan_events[-1]
        wps = last_plan.get("waypoints", [])
        if wps:
            add_curve("PlannedWaypoints_Curve", [start_pos.tolist()] + wps, 0.6, mat_plan, col_planned)
        for idx, tp in enumerate(last_plan.get("turn_points", [])):
            add_sphere(f"TurnPoint_{idx}", tp, 4.5, mat_turn, col_planned, subdivisions=1)
        dense_pts = last_plan.get("astar_points") or last_plan.get("lattice_points") or last_plan.get("station_points")
        if dense_pts:
            add_curve("SearchGridPath_Curve", [start_pos.tolist()] + dense_pts, 0.35, mat_astar, col_planned)
            for idx, gp in enumerate(dense_pts):
                add_sphere(f"GridPoint_{idx}", gp, 2.0, mat_astar, col_planned, subdivisions=1)

    for idx, ev in enumerate(e for e in points_arr if e.get("event") == "turn_passed"):
        add_sphere(f"TurnExecuted_{idx}", ev["point"], 3.5, mat_turn, col_planned, subdivisions=1)

    # Radar debris animation
    radar_calls = data.get("radar_calls", [])
    if radar_calls:
        sel_calls = []
        last_rt = -1e9
        for rc in radar_calls:
            if args.radar_step <= 0 or rc["t"] - last_rt >= args.radar_step or rc is radar_calls[-1]:
                sel_calls.append(rc)
                last_rt = rc["t"]

        first_pos = np.asarray(sel_calls[0]["positions_abs"], float)[:args.max_rocks]
        first_diam = np.asarray(sel_calls[0].get("diameters", [4.0] * len(first_pos)), float)[:args.max_rocks]
        n_rocks = len(first_pos)
        rock_objs = []
        for i in range(n_rocks):
            rad = max(float(first_diam[i]) * 0.5, 1.0)
            rock_objs.append(add_sphere(f"Rock_{i:03d}", first_pos[i], rad, mat_rock, col_debris, subdivisions=1))

        # Match rocks between consecutive snapshots by nearest neighbor so sorted-list reorders stay smooth
        prev_pos = first_pos.copy()
        for rc in sel_calls:
            fr = to_frame(float(rc["t"]), t0)
            max_frame = max(max_frame, fr)
            cur_pos = np.asarray(rc["positions_abs"], float)
            if len(cur_pos) >= n_rocks and rc is not sel_calls[0]:
                # Vectorized nearest assignment from prev_pos -> cur_pos
                d2 = np.sum((prev_pos[:, None, :] - cur_pos[None, :, :]) ** 2, axis=2)
                matched = np.empty_like(prev_pos)
                used = np.zeros(len(cur_pos), dtype=bool)
                order = np.argsort(d2.min(axis=1))
                for r_idx in order:
                    row = d2[r_idx].copy()
                    row[used] = np.inf
                    c_idx = int(np.argmin(row))
                    used[c_idx] = True
                    matched[r_idx] = cur_pos[c_idx]
                prev_pos = matched
            for i in range(n_rocks):
                rock_objs[i].location = tuple(prev_pos[i].tolist())
                rock_objs[i].keyframe_insert(data_path="location", frame=fr)

    scene.frame_start = 1
    scene.frame_end = max_frame
    scene.frame_current = 1

    # Lighting + Tracking Camera
    light_data = bpy.data.lights.new(name="SunLight", type="SUN")
    light_data.energy = 3.0
    light_obj = bpy.data.objects.new(name="SunLight", object_data=light_data)
    light_obj.rotation_euler = (0.7, 0.3, 0.5)
    col_mission.objects.link(light_obj)

    mid = 0.5 * (start_pos + target_pos)
    corridor = target_pos - start_pos
    side = np.cross(corridor, [0.0, 1.0, 0.0])
    if np.linalg.norm(side) < 1e-6:
        side = np.array([1.0, 0.0, 0.0])
    side = side / np.linalg.norm(side)
    cam_pos = mid + side * 420.0 + np.array([0.0, 220.0, 180.0])

    cam_data = bpy.data.cameras.new(name="ReplayCamera")
    cam_data.clip_start = 1.0
    cam_data.clip_end = 50000.0
    cam_obj = bpy.data.objects.new("ReplayCamera", cam_data)
    cam_obj.location = tuple(cam_pos.tolist())
    track = cam_obj.constraints.new(type="TRACK_TO")
    track.target = shuttle
    track.track_axis = "TRACK_NEGATIVE_Z"
    track.up_axis = "UP_Y"
    col_mission.objects.link(cam_obj)
    scene.camera = cam_obj

    # 3D Telemetry HUD + embedded frame-change handler
    bpy.ops.object.text_add(location=tuple((start_pos + np.array([0.0, 40.0, 40.0])).tolist()))
    hud_obj = bpy.context.active_object
    hud_obj.name = "TelemetryHUD"
    hud_obj.data.body = hud_samples.get("1", f"Plan: {meta.get('plan_name')}")
    hud_obj.data.size = 14.0
    link_only(hud_obj, col_hud)

    scene["batasha_hud_json"] = json.dumps(hud_samples)
    hud_script = bpy.data.texts.new("replay_hud.py")
    hud_script.from_string(
        "import json, bpy\n"
        "def update_hud(scene):\n"
        "    hud = bpy.data.objects.get('TelemetryHUD')\n"
        "    raw = scene.get('batasha_hud_json')\n"
        "    if not hud or not raw:\n"
        "        return\n"
        "    samples = json.loads(raw)\n"
        "    keys = sorted(int(k) for k in samples)\n"
        "    f = scene.frame_current\n"
        "    k = max((x for x in keys if x <= f), default=keys[0]) if keys else None\n"
        "    if k is not None:\n"
        "        hud.data.body = samples[str(k)]\n"
        "bpy.app.handlers.frame_change_post[:] = [h for h in bpy.app.handlers.frame_change_post if getattr(h, '__name__', '') != 'update_hud']\n"
        "bpy.app.handlers.frame_change_post.append(update_hud)\n"
    )
    hud_script.use_module = True

    # Ensure viewport clipping handles Phobos coordinates cleanly when opened interactively
    for scr in bpy.data.screens:
        for area in scr.areas:
            if area.type == "VIEW_3D":
                for space in area.spaces:
                    if space.type == "VIEW_3D":
                        space.clip_start = 1.0
                        space.clip_end = 50000.0

    bpy.ops.wm.save_as_mainfile(filepath=str(out_path))
    print(f"Saved Blender replay file: {out_path} (frames 1..{max_frame} @ {fps} fps)")


def main() -> None:
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
    else:
        argv = sys.argv[1:]
    args = _parse_args(argv)
    try:
        import bpy  # noqa: F401
        has_bpy = True
    except ImportError:
        has_bpy = False

    if not has_bpy:
        rc = _launch_via_blender(argv, args.blender_bin)
        sys.exit(rc)

    json_path, out_path, data = _resolve_paths(args)
    _build_blend(json_path, out_path, data, args)


if __name__ == "__main__":
    main()
