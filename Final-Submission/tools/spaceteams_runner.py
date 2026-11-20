"""Launch a local Space Teams PRO Compute_Server for a web run."""
from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path

from Systems.EngineScore import engine_score_for, persist_engine_score, read_score_alert  # noqa: F401 (re-exported)
from Systems.SimSuite import COMPETITION_SIMCONF, FINAL_SIMCONF, ROOT

SCENES = {"stp_final": FINAL_SIMCONF, "stp_practice": COMPETITION_SIMCONF}


def installation(server: Path | None, assets: Path | None) -> tuple[Path, Path, str]:
    if server is None or not server.is_file():
        raise ValueError("Set --stp-server to the installed Compute_Server/run_Server.exe.")
    if assets is None or not assets.is_dir():
        raise ValueError("Set --stp-assets to the Space Teams PRO Assets directory.")
    repos = assets / "Repos"
    try:
        repo = next(p for p in repos.iterdir() if p.is_dir() and p.resolve() == ROOT.resolve())
    except (OSError, StopIteration) as exc:
        raise ValueError("This checkout must be available under the Space Teams PRO Assets/Repos folder.") from exc
    if not (repos / "STU4-Sim" / "Systems" / "Scoring.py").is_file():
        raise ValueError("Install the official STU4-Sim repository under Space Teams PRO Assets/Repos.")
    return server, assets, f"Local/Repos/{repo.name}/Controller.py"


def run_engine(server, assets, job_dir, scene, parts, config, max_time, flush_log):
    server, assets, controller_path = installation(server, assets)
    data = json.loads(SCENES[scene].read_text(encoding="utf-8"))
    controller = next(s for s in data["SystemManager"]["Systems"] if s["Nametag"] == "Controller")
    controller["Source"] = controller_path
    controller["Inst_Parameters"].update({
        **parts,
        "FlushLog": str(flush_log).lower(),
        "WebRun": json.dumps({"config": config, "log_dir": str(job_dir), "max_time": max_time, "scene": scene},
                             allow_nan=False),
    })
    job_dir.mkdir(parents=True, exist_ok=True)
    console_path = job_dir / "server.log"
    with tempfile.TemporaryDirectory(prefix="batasha-", dir=assets) as tmp:
        scene_path = Path(tmp) / "run.simconf"
        scene_path.write_text(json.dumps(data), encoding="utf-8")
        sim_arg = "Local/" + scene_path.relative_to(assets).as_posix()
        with console_path.open("w", encoding="utf-8") as console:
            proc = subprocess.Popen([str(server), f"--sim={sim_arg}"], cwd=server.parent,
                                    stdout=console, stderr=subprocess.STDOUT)
            started = time.monotonic()
            deadline = started + 60 + max_time * 4
            try:
                while True:
                    completed = next(job_dir.glob("spaceteams/*/run.json"), None)
                    if completed is not None:
                        return completed
                    elapsed = time.monotonic() - started
                    if proc.poll() is not None:
                        raise RuntimeError(f"Compute_Server exited with code {proc.returncode}. See {console_path}")
                    if elapsed > 60 and not any(job_dir.glob("spaceteams/*/live.jsonl")):
                        raise RuntimeError(f"No controller telemetry after 60 seconds. See {console_path}")
                    if time.monotonic() >= deadline:
                        (job_dir / "stop").touch()
                        raise RuntimeError(f"Space Teams PRO run timed out. See {console_path}")
                    time.sleep(0.25)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
