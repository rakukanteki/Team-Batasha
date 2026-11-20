"""Burn replay: fly burn laws against the wanted velocity a real engine run asked for.

    python tools/burn_replay.py RUN_DIR NPZ [--laws jets,steer,...] [--until 80]

RUN_DIR is a flightlogs/spaceteams/<run> folder (run.log NAV/CMD lines: t, mode, vel, vg); NPZ the Recorder file of the
same run (start attitude, positions). The wanted velocity v + vg is taken from the log at the logged tick times
(stalls included) and in the logged modes; each law then flies its own shuttle (jet twin + rigid body), seeing its
state one tick late as the engine does. Guidance does not react to the law, so this compares burn laws only.
Offline testing only (FINDINGS 7.15).
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from Systems import Actuation as A                                     # noqa: E402
from Systems import Core as C                                          # noqa: E402
from tools.actuator_twin import tick, rotation, INERTIA, INERTIA_INV   # noqa: E402
from tools.replay_bench import TRUE_B, TRUE_W                          # noqa: E402

TGT = np.array([-850.0, 120.0, -9530.0])                   # the competition target (run.log header)
PAT = re.compile(r"\[t=\s*([\d.]+)s\] \[NAV/CMD\] mode=(\S+) .*?pos=\[([^\]]+)\] vel=\[([^\]]+)\] fuel=[\d.]+ vg=\[([^\]]+)\]")


def load(run_dir):
    rows = []
    for ln in open(Path(run_dir) / "run.log", encoding="utf-8"):
        m = PAT.search(ln)
        if m:
            f = lambda s: np.array([float(x) for x in s.split(",")])
            rows.append((float(m[1]), m[2], f(m[3]), f(m[4]), f(m[5])))
    return rows


def replay(rows, R0, law, until=80.0, sub=1 / 120, main_min=2.0, trace=None):
    """The wanted velocity follows the log (the last one logged while burning holds through coasts); when to burn
    follows Flight.decide's rule on this shuttle's own error, so a faster law also stops sooner.
    trace: a list to append (t, mode, |wanted - v|, pointing error deg, fuel so far) to, every tick."""
    R = R0.copy(); w = np.zeros(3); v = rows[0][3].copy(); fuel = 0.0; state = {}; seen = (R.copy(), w.copy())
    main_s = 0.0; want = rows[0][3] + rows[0][4]; burning = main = False; a = getattr(law, "accel", C.A_BURN)
    for (t, mode, r, vl, vg), nxt in zip(rows, rows[1:]):
        if t > until:
            break
        T = nxt[0] - t; Rs, ws = seen                                     # the law sees the previous tick's state
        if mode != "coast":
            want = vl + vg
        g = want - v; ng = float(np.linalg.norm(g))
        if not burning and ng > C.EPS_ON:
            burning = True; main = ng > main_min
        if burning and ng <= max((a if main else C.A_TRIM) * T, 0.0 if main else 0.02):
            burning = False
        if mode == "evade":
            lin, ang = A.trim_command(Rs.T @ g)
        elif burning and main:
            d = Rs.T @ g
            state["ng"] = ng; state["dt"] = T; state["ref"] = Rs.T @ (TGT - r) / np.linalg.norm(TGT - r)
            lin, ang = law(d / ng, ws, state); main_s += T
        elif burning:
            lin, ang = A.trim_command(Rs.T @ g)
        else:
            lin, ang = A.coast_command(ws)
        if trace is not None:
            z = R[:, 2]; err = np.degrees(np.arccos(np.clip(z @ g / max(ng, 1e-9), -1, 1)))
            trace.append((t, "main" if burning and main else ("trim" if burning or mode == "evade" else "coast"), ng, err, fuel))
        seen = (R.copy(), w.copy())
        k = max(1, int(round(T / sub))); h = T / k
        acc, rate, _ = tick(lin, ang, h)                                  # the selector's answer holds over the tick
        for _ in range(k):
            w = w + (acc[3:] - INERTIA_INV @ np.cross(w, INERTIA @ w)) * h
            v = v + (R @ acc[:3] + C.grav(r) + TRUE_B - 2 * np.cross(TRUE_W, v)) * h
            if np.linalg.norm(w) > 0:
                R = R @ rotation(w, np.linalg.norm(w) * h)
            fuel += rate * h
    return dict(fuel=round(fuel, 1), resid=round(float(np.linalg.norm(want - v)), 2), main_s=round(main_s, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir"); ap.add_argument("npz"); ap.add_argument("--laws", default="jets,steer,direct")
    ap.add_argument("--until", type=float, default=80.0)
    a = ap.parse_args()
    rows = load(a.run_dir); T = np.load(a.npz)["ticks"]; R0 = T[0, 10:19].reshape(3, 3)
    C.configure(rows[0][2], TGT)
    want0 = np.linalg.norm(rows[0][4])
    for name in a.laws.split(","):
        res = replay(rows, R0, A.BURNS[name], a.until)
        gained = want0 - res["resid"]
        print(f"{name:8s} fuel {res['fuel']:5.1f} kg  residual |vg| {res['resid']:5.2f} m/s  main {res['main_s']} s  "
              f"~{res['fuel'] / max(gained, 1e-6):.2f} kg per m/s of the first vg", flush=True)


if __name__ == "__main__":
    main()
