"""Tracker bench: the radar world model against rocks recorded in the real sim, along the recorded shuttle path.

    python tools/tracker_bench.py FIELD_DIR [--qvel 0.002] [--seeds 2]

No flight code but the tracker runs, so it is fast. Reports, for rocks within 400 m, the median and 90th
percentile position error now and predicted 10 s and 20 s ahead, and the consistency of the claimed sigma
(fraction of rocks whose error is inside 3 sigma: should be about 0.97+ for a planner that trusts sigma).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from Systems import Core as C, Radar as RD                     # noqa: E402
from tools.offline_bench import Radar                           # noqa: E402
from tools.replay_bench import Field                            # noqa: E402


def run(field, seed, tmax=300.0, dt=1 / 30, read_every=6, radar_hz=120.0, verbose=False):
    rng = np.random.default_rng(seed); shift = rng.uniform(-3, 3)
    S = np.genfromtxt(Path(field.folder) / "shuttle.csv", delimiter=",", names=True)
    st = S["t"]; sr = np.c_[S["rx"], S["ry"], S["rz"]]; sv = np.c_[S["vx"], S["vy"], S["vz"]]
    own = lambda t: (np.array([np.interp(t, st, sr[:, k]) for k in range(3)]), np.array([np.interp(t, st, sv[:, k]) for k in range(3)]))
    C.configure(field.r0, field.T)
    P, V = field.at(shift); r, v = own(0.0)
    em = Radar(rng, P, V, field.dia, r, v); W = RD.WorldModel(); W.read(em.published, 0.0, own)
    t = 0.0; k = 0; out = []; per = radar_hz * dt; acc = 0.0
    while t < min(tmax, st[-1] - 25):
        t += dt; k += 1; P, V = field.at(t + shift); r, v = own(t)
        acc += per
        while acc >= 1:
            em.tick(t, P, V, field.dia, r, v); acc -= 1
        if k % read_every == 0:
            W.read(em.published, t, own)
        if k % 60 == 0:
            trk = {}
            for m, i in zip(em.published, em.truth):
                j = W.keys.get(np.ascontiguousarray(m, dtype=float).tobytes())
                if j is not None:
                    trk[i] = j
            ids = np.array(sorted(trk)); tk = np.array([trk[i] for i in ids])
            near = np.linalg.norm(P[ids] - r, axis=1) < 400
            if near.sum() < 3:
                continue
            row = [t]
            for h in (0.0, 10.0, 20.0):
                pp = W.predict(t + h); Ph, _ = field.at(t + h + shift)
                e = np.linalg.norm(pp[0][tk[near]] - Ph[ids[near]], axis=1)
                sig = np.sqrt(pp[3][tk[near]] ** 2 * 3)                     # 3-D sigma from the per-axis value
                row += [np.median(e), np.percentile(e, 90), np.mean(e < 3 * sig)]
            out.append(row)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("field"); ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--qvel", type=float); ap.add_argument("--v0", type=float); ap.add_argument("--radar-hz", type=float, default=120.0)
    ap.add_argument("--tmax", type=float, default=300.0); ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if a.qvel is not None:
        RD.Q_VEL = a.qvel ** 2
    if a.v0 is not None:
        RD.V0_VAR = a.v0 ** 2
    f = Field(a.field); f.folder = a.field
    allr = np.vstack([run(f, s, a.tmax, radar_hz=a.radar_hz) for s in range(a.seeds)])
    early = allr[allr[:, 0] < 60]; late = allr[allr[:, 0] >= 60]
    for name, x in (("t<60", early), ("t>=60", late)):
        m = x[:, 1:].mean(0)
        print(f"{name:6s} now med {m[0]:5.1f} p90 {m[1]:5.1f} in3sig {m[2]:.2f} | +10s med {m[3]:5.1f} p90 {m[4]:5.1f} in3sig {m[5]:.2f} "
              f"| +20s med {m[6]:5.1f} p90 {m[7]:5.1f} in3sig {m[8]:.2f}")


if __name__ == "__main__":
    main()
