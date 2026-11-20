"""The official Space Teams PRO score, read from the engine's own logs and saved into our run files.

The official score exists only as on-screen alerts from STU4-Sim/Systems/Scoring.py. Those alerts land in the engine's
log folder (%LOCALAPPDATA%/SimDynamX/STPro/Logs/<yymmdd_HHMMSS>_PLAN_<n>_<plan>__xxx), never in our controller, which can
be stopped any time after them (8 s later in one run, still running 37 s later in another). So a later process (the web
GUI, or the command below) copies them into run.json and run.log.

    python -m Systems.EngineScore flightlogs/spaceteams          # backfill every engine run under a folder
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_NUMBER = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
_SCORE_ALERT = re.compile(r"(Successfully escaped Phobos with a score of|Crashed into some debris and finished with a score of)\s+" + _NUMBER)
_SCORE_DETAILS = {
    "scoring_time_s": re.compile(r"Sim elapsed time\s*=\s*" + _NUMBER + r"\s+s"),
    "final_fuel_kg": re.compile(r"Total fuel used\s*=\s*" + _NUMBER + r"\s+kg"),
    "damage_kJ": re.compile(r"Total damage taken\s*=\s*" + _NUMBER + r"\s+J"),
}
SOURCE = "Space Teams PRO"
LOG_MARK = "ENGINE SCORE"


def read_score_alert(console_path: Path) -> dict:
    """Score, outcome, elapsed time, fuel and damage from one engine log file; {} if it holds no score line."""
    if not console_path.is_file():
        return {}
    summary = {}
    with console_path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if match := _SCORE_ALERT.search(line):
                arrived = match[1].startswith("Successfully")
                summary.update(score=float(match[2]), arrived=arrived,
                               outcome="arrived" if arrived else "damage_limit")
            for key, pattern in _SCORE_DETAILS.items():
                if match := pattern.search(line):
                    summary[key] = float(match[1]) / (1000 if key == "damage_kJ" else 1)
    return summary if "score" in summary else {}


def engine_logs_dir() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", "")) / "SimDynamX" / "STPro" / "Logs"


def engine_score_for(timestamp_utc: str | None, plan_name: str | None = None) -> dict:
    """The engine's score for a run started at `timestamp_utc`: from the newest engine log folder that started within
    10 min before it (the controller can start minutes after the engine) and, when `plan_name` is given, names that plan. {} when no folder matches; no "score" key when the
    folder has no score line yet."""
    if not timestamp_utc:
        return {}
    try:
        start = datetime.fromisoformat(timestamp_utc)
    except ValueError:
        return {}
    base = engine_logs_dir()
    best = None
    for d in base.glob("*_PLAN_*") if base.is_dir() else []:
        if plan_name and not re.search(rf"_PLAN_\d+_{re.escape(plan_name)}(_[^_].*)?__", d.name):
            continue                       # other plans must not swap scores; a config suffix ("..._jets__") still matches
        try:
            t = datetime.strptime(d.name[:13], "%y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if 0 <= (start - t).total_seconds() < 600 and (best is None or t > best[0]):
            best = (t, d)
    if best is None:
        return {}
    # Scoring system log first: it is small and holds the score lines.
    for f in sorted(best[1].glob("*.log"), key=lambda p: (not p.name.endswith("Scoring.log"), p.name)):
        if r := read_score_alert(f):
            return {"engine_log_dir": str(best[1]), **r}
    return {"engine_log_dir": str(best[1])}


def persist_engine_score(run_dir: Path, alert: dict | None = None, settle_s: float = 30.0) -> dict:
    """Write the engine's score into run_dir/run.json (metadata.summary) and run.log. Idempotent: a run that already
    has it is left alone. `alert` skips the log search (a GUI-launched run has its own server.log).
    Skips a run whose files changed in the last `settle_s` seconds: the controller may still be saving it, and
    rewriting an older snapshot over its final save would truncate the run. Pass 0 once the controller has exited.
    Returns the engine score dict, or {} when none is available or the run is not settled yet."""
    run_dir = Path(run_dir)
    jp = run_dir / "run.json"
    if not jp.is_file():
        return {}
    newest = max(p.stat().st_mtime for p in (jp, run_dir / "run.log") if p.is_file())
    data = json.loads(jp.read_text(encoding="utf-8"))
    meta = data.setdefault("metadata", {})
    summary = meta.setdefault("summary", {})
    if summary.get("score_source") == SOURCE and "score" in summary:
        return {k: summary[k] for k in ("score", "arrived", "outcome", "scoring_time_s", "final_fuel_kg", "damage_kJ")
                if k in summary}
    if time.time() - newest < settle_s:
        return {}
    if alert is None:
        alert = engine_score_for(meta.get("timestamp_utc"), meta.get("plan_name"))
    if "score" not in (alert or {}):
        return {}
    summary.update({k: v for k, v in alert.items()})   # the engine's numbers override our own estimates
    summary["score_source"] = SOURCE
    text = json.dumps(data, separators=(",", ":"))
    tmp = jp.with_suffix(".json.score-tmp")       # not RunLogger's run.json.tmp: never share a temp file with it
    tmp.write_text(text + "\n", encoding="utf-8")
    tmp.replace(jp)
    log = run_dir / "run.log"
    if log.is_file():
        with log.open("rb") as f:
            f.seek(max(0, log.stat().st_size - 4096))
            tail = f.read().decode("utf-8", errors="replace")
        if LOG_MARK not in tail:
            fields = ", ".join(f"{k}={alert[k]}" for k in ("score", "outcome", "scoring_time_s", "final_fuel_kg", "damage_kJ")
                               if k in alert)
            with log.open("a", encoding="utf-8") as f:
                f.write(f"{LOG_MARK} ({SOURCE}, from {alert.get('engine_log_dir', 'server.log')}): {fields}\n")
    return alert


def main(argv: list[str]) -> int:
    roots = [Path(a) for a in argv] or [Path(__file__).resolve().parent.parent / "flightlogs" / "spaceteams"]
    found = 0
    for root in roots:
        for jp in sorted(root.rglob("run.json")) if root.is_dir() else []:
            r = persist_engine_score(jp.parent)
            found += bool(r)
            print(f"{jp.parent.name}: " + (f"score {r['score']} ({r.get('outcome')}, {r.get('scoring_time_s')} s, "
                                            f"{r.get('final_fuel_kg')} kg)" if r else
                                            "no engine score yet (or the run changed in the last 30 s)"))
    return 0 if found else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
