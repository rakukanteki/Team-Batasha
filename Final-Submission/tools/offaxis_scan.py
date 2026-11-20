"""Fuel burned with the thrust far off body +z, per engine run (FINDINGS 7.17).

    python tools/offaxis_scan.py [RUN_DIR ...]          # default: every flightlogs/spaceteams/<run>

Off-axis thrust (commanded thrust more than 35 deg from body +z) runs at about 0.07 m/s^2 and 8 kg per m/s, against
4.4 aligned. Every bad engine run of 2026-11-16/17 spent 13-21 kg of its main burn this way; the good runs under 1 kg.
tools/replay_bench.py reports the same number (`off_axis`) for bench flights.
"""
import glob
import re
import sys

import numpy as np

OFF_AXIS = np.radians(35.0)
CMD = re.compile(r"\[t=\s*([\d.]+)s\] \[NAV/CMD\] mode=(\S+) .*?fuel=([\d.]+) vg=\[[^\]]+\] SetCommand\(lin=\[([^\]]+)\]")
NAME = re.compile(r'plan_config=\{.*?"name": "([^"]+)"')
SCORE = re.compile(r"ENGINE SCORE.*?score=([\d.]+)")


def scan(run_dir):
    """(config name, engine score or None, fuel, off-axis kg in main mode, off-axis kg in any mode)."""
    txt = open(f"{run_dir}/run.log", encoding="utf-8").read()
    rows = [(m[2], float(m[3]), np.array([float(x) for x in m[4].split(",")])) for m in CMD.finditer(txt)]
    main = anymode = 0.0
    for (mode, fuel, lin), (_, fuel2, _) in zip(rows, rows[1:]):
        n = np.linalg.norm(lin)
        if n > 1e-9 and lin[2] < np.cos(OFF_AXIS) * n:
            anymode += fuel2 - fuel
            main += (fuel2 - fuel) if mode == "main" else 0.0
    name = NAME.search(txt); score = SCORE.findall(txt)
    return (name[1] if name else "?", float(score[-1]) if score else None, rows[-1][1] if rows else 0.0, main, anymode)


def main():
    dirs = sys.argv[1:] or sorted(glob.glob("flightlogs/spaceteams/*"))
    print(f"{'run':44s} {'config':18s} {'score':>5s} {'fuel':>6s} {'off-axis main':>14s} {'any mode':>9s}")
    for d in dirs:
        try:
            name, score, fuel, main_kg, any_kg = scan(d)
        except (OSError, IndexError):
            continue
        print(f"{d.replace(chr(92), '/').split('/')[-1]:44s} {name:18s} {score if score is not None else '-':>5} "
              f"{fuel:6.1f} {main_kg:12.1f} kg {any_kg:6.1f} kg")


if __name__ == "__main__":
    main()
