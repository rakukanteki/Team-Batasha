"""Build tools/sim_field.npz, the rock field as the sim really flew it, from a flight recording.

Usage:  python tools/sim_field.py [flightlogs/<file>.npz]      # default: the newest recording

The simconf gives every rock under 0.8 m/s, but in the sim the rocks gain several m/s in their first seconds
(FINDINGS 6.13), so a bench on the simconf field alone is far too easy. This replays the recorded radar through the
tracker to keep each rock's identity, fits a ballistic path to each rock's own measurements, and saves it for
tools/offline_bench.py. Offline testing only: the flight code never reads this file.
"""
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from Systems import Core as C                    # noqa: E402
from Systems.Radar import WorldModel             # noqa: E402
from Systems.Recorder import reads               # noqa: E402
import offline_bench as OB                       # noqa: E402

OUT = ROOT / "tools" / "sim_field.npz"
T_REF = 10.0                                     # s on the controller clock: the rocks coast from before here
T_KICK = 8.0                                     # s: until about here the rocks are still accelerating


def build(path, out=OUT):
    log = np.load(path); tk = log["ticks"]; t, r, v = tk[:, 0], tk[:, 1:4], tk[:, 4:7]
    # Keep generated indices compatible with the bundled calibration's ordering.
    P0, V0, D, S0, T0 = OB.load_field(serialized=True); C.configure(S0, T0)

    def own_at(tt):                              # before the first tick: the start-up push, taken as uniform
        if tt <= t[0]:
            f = np.clip(tt / t[0], 0, 1); return S0 + (r[0] - S0) * f * f, v[0] * f
        i = min(np.searchsorted(t, tt) - 1, len(t) - 1); return r[i], v[i]

    world = WorldModel(); hist = {}
    for tr, entries in reads(log):
        for i, m in world.read(entries, tr, own_at):
            tm = m[2, 0] + world.offset if m[2, 0] > 0 else 0.0
            hist.setdefault(i, []).append((tm, *(own_at(tm)[0] + m[0]), m[1, 2]))
    P = np.zeros((world.n, 3)); V = np.zeros((world.n, 3)); dia = np.zeros(world.n); rock = np.full(world.n, -1); rms = []
    for i in range(world.n):
        h = np.array(hist[i]); dia[i] = h[:, 4].mean()
        d = np.linalg.norm(P0 + V0 * h[0, 0] - h[0, 1:4], axis=1) + 40 * np.abs(D - dia[i]) / D; j = int(d.argmin())
        if h[0, 0] < 3.5 and np.linalg.norm(P0[j] + V0[j] * h[0, 0] - h[0, 1:4]) < 10:
            rock[i] = j                          # which simconf rock this is: known while it has barely moved
        m = (h[:, 0] > T_REF) & (h[:, 0] < 120)
        tt = h[m, 0] - T_REF; g = C.grav(h[m, 1:4].mean(0))
        A = np.c_[np.ones(m.sum()), tt]; y = h[m, 1:4] - 0.5 * g * tt[:, None] ** 2
        coef = np.linalg.lstsq(A, y, rcond=None)[0]; P[i], V[i] = coef; rms.append(np.sqrt(((y - A @ coef) ** 2).mean()))
    uniq = np.unique(rock[rock >= 0], return_counts=True); dup = set(uniq[0][uniq[1] > 1]); rock[np.isin(rock, list(dup))] = -1
    np.savez_compressed(out, P=P, V=V, D=dia, rock=rock, t_ref=T_REF, t_kick=T_KICK, t_first=t[0], r_first=r[0], v_first=v[0],
                        axes_first=tk[0, 10:19].reshape(3, 3))
    spd = np.linalg.norm(V, axis=1)
    print(f"{Path(out).name}: {world.n} rocks from {Path(path).name}; fit rms median {np.median(rms):.2f} m; "
          f"speed median {np.median(spd):.2f} m/s, 90% {np.percentile(spd, 90):.2f}; mean velocity {np.round(V.mean(0), 2)}; "
          f"{(rock >= 0).sum()} matched to a simconf rock; first tick t={t[0]:.2f} s, shuttle already at {np.round(r[0] - S0, 2)} m, {np.round(v[0], 2)} m/s")
    return float(np.median(rms))


FIELDS = ROOT / "tools" / "sim_fields"        # one field per valid recording, for sim_bench --fields all


def real_time(path):
    """A recording flown at 1x: about 50 ticks per sim second. A sim sped up in the app gives far fewer."""
    t = np.load(path)["ticks"][:, 0]
    return len(t) > 600 and float(np.median(np.diff(t))) < 0.05


if __name__ == "__main__":
    if sys.argv[1:] == ["--all"]:                # every 1x recording -> tools/sim_fields/<recording>.npz
        FIELDS.mkdir(exist_ok=True)
        for p in sorted(glob.glob(str(ROOT / "flightlogs" / "*.npz"))):
            if p.endswith(".tmp.npz"):
                continue
            if not real_time(p):
                print(f"skip {Path(p).name}: not flown at 1x"); continue
            if build(p, FIELDS / Path(p).name) > 5.0:  # the rocks did not fit straight paths: a bad recording
                (FIELDS / Path(p).name).unlink(); print(f"dropped {Path(p).name}: fit too poor")
    else:
        build(sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(str(ROOT / "flightlogs" / "*.npz")))[-1])
