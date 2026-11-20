"""How closely the generated Phobos world (tools/phobos_world.py) matches the rock fields recorded in real runs.

    python tools/world_check.py [N]          # N generated fields against every recorded field (default 20)

Per field: the start-up push (median rock velocity change along x by t = 14 s), how far the smooth model
(gravity + frame terms) drifts from the field over 60 s (median and 90th percentile), the rate of velocity jumps
over 1 m/s, and how many rocks sit within 40 m of the straight route at t = 60 s and 120 s. Prints the mean of each
over the recorded and the generated fields and the agreement, 1 - |generated - recorded| / recorded. "jumps" counts
velocity jumps between 3 s windows, transient ones included; the generator models only lasting changes, so it is shown
but left out of the mean agreement.
"""
from __future__ import annotations

import sys

import numpy as np

from Systems import Core as C
from tools.phobos_world import WorldField, recorded_fields
from tools.replay_bench import TRUE_B, TRUE_W, Field

TANK_DIA = 17.0                              # ExplodedTank pieces: their reported location wobbles as they tumble


def stats(f):
    t, X = f.t, f.P; tank = np.abs(f.dia - TANK_DIA) < 1e-6
    vel = lambda k: (X[k + 1] - X[k - 1]) / (t[k + 1] - t[k - 1])
    k14 = int(np.searchsorted(t, 14.0))
    push = float(np.median(vel(k14)[:, 0] - (X[1, :, 0] - X[0, :, 0]) / (t[1] - t[0])))
    def fit_v(t0, half=5.0):                 # least-squares velocity over a window: the recordings' timestamps jitter
        m = np.abs(t - t0) <= half; tt = t[m] - t0; A = np.vstack([np.ones_like(tt), tt]).T
        c = np.linalg.lstsq(A, X[m].reshape(m.sum(), -1), rcond=None)[0].reshape(2, -1, 3); return c[0], c[1]
    p, v = fit_v(25.0); b = int(np.searchsorted(t, 85.0)); n = 600; h = (t[b] - 25.0) / n
    for _ in range(n):
        v = v + (C.grav(p) + TRUE_B - 2 * np.cross(TRUE_W, v)) * h; p = p + v * h
    e = np.linalg.norm(p - X[b], axis=1)[~tank]
    W = [fit_v(c, 1.5)[1] for c in np.arange(18.0, min(t[-1] - 2, 140.0), 3.0)]        # 3 s window velocities
    J = np.linalg.norm(np.diff(np.array(W), axis=0), axis=2)[:, ~tank]
    rate = (J > 1.0).sum() / ((~tank).sum() * 3.0 * len(J) / 60)
    S, T = f.r0, f.T; u = (T - S) / np.linalg.norm(T - S)
    def corridor(tt):
        k = int(np.searchsorted(t, tt)); d = X[k] - S; s = d @ u
        lat = np.linalg.norm(d - s[:, None] * u, axis=1)
        return int(((lat < 40 + f.vis) & (s > 0) & (s < np.linalg.norm(T - S))).sum())
    return dict(push=push, drift50=float(np.median(e)), drift90=float(np.percentile(e, 90)), jumps=float(rate),
                route60=corridor(60.0), route120=corridor(120.0))


def main(n_gen=20):
    C.configure()
    rec = [stats(f) for f in map(Field, recorded_fields()) if len(f.t) > 130]
    gen = [stats(WorldField(s, seconds=140.0)) for s in range(n_gen)]
    print(f"{'metric':10s} {'recorded':>9s} {'generated':>9s} {'agreement':>9s}   (recorded n={len(rec)}, generated n={len(gen)})")
    agree = []
    for key in rec[0]:
        r = np.mean([x[key] for x in rec]); g = np.mean([x[key] for x in gen]); a = 1 - abs(g - r) / abs(r)
        if key != "jumps":
            agree.append(a)
        print(f"{key:10s} {r:9.3f} {g:9.3f} {100 * a:8.1f}%" + ("  (info)" if key == "jumps" else ""))
    print(f"mean agreement {100 * np.mean(agree):.1f}%")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
