"""Plan step R2 (docs/plans/SCORE_70_PLAN.md): do fewer rocks come near a route bent toward Phobos +x?

Rocks are pushed toward Phobos -x at start-up (FINDINGS 7.9), so the +x side may empty first. For each recorded field of
codernayeem/STU4-lab, a lane that drifts sideways linearly to `offset` m at the target is checked: how many rocks come
within 15 m and 40 m (plus their radius) of the shuttle's place on it, at the time a typical flight is there.
Recorded true rock positions are used here for analysis only, never by the flight code (FINDINGS 1.4).

Usage:  python tools/lane_count.py [../STU4-lab/fields_all41.json]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.upstream_scene import Scene

# along-route distance (m) -> time (s) of a typical flight (FINDINGS 7.10)
PROFILE = np.array([(0, 0), (50, 37), (150, 61), (250, 80), (350, 95), (450, 110), (550, 123), (650, 136), (750, 149),
                    (850, 162), (950, 176), (1100, 197), (1256, 210)], float)
OFFSETS = (-40.0, -20.0, 0.0, 20.0, 40.0)        # sideways at the target, + toward Phobos +x


def main(field_list):
    lab = Path(field_list).resolve().parent
    pos0, _, diam, S, T = Scene().field()
    L = np.linalg.norm(T - S); SH = (T - S) / L
    ex = np.array([1.0, 0, 0]) - SH[0] * SH; ex /= np.linalg.norm(ex)          # Phobos +x, square to the route
    s = np.arange(0.0, L - 50.0, 10.0); ts = np.interp(s, *PROFILE.T)
    counts = {o: [] for o in OFFSETS}
    for name in json.load(open(field_list)):
        f = lab / "runs" / Path(name).name / "debris.csv"
        if not f.is_file():
            continue
        D = np.loadtxt(f, delimiter=","); t = D[:, 0]; P = D[:, 1:].reshape(len(D), -1, 3)
        first = P[0]
        r = (diam / 2)[np.argmin(np.linalg.norm(first[:, None] - pos0[None], axis=2), axis=1)]   # radius per column
        rows = np.clip(np.searchsorted(t, ts), 0, len(t) - 1)
        for o in OFFSETS:
            lane = S + np.outer(s, SH) + np.outer(o * s / L, ex)
            d = np.linalg.norm(P[rows] - lane[:, None], axis=2) - r[None]               # [station, rock]
            counts[o].append(((d < 15).any(0).sum(), (d < 40).any(0).sum()))
    n = len(counts[0.0])
    print(f"{n} fields; rocks that come within 15 m / 40 m of the lane (mean, p90 over fields)")
    for o in OFFSETS:
        c = np.array(counts[o])
        print(f"  {o:+5.0f} m at the target: within 15 m {c[:, 0].mean():.2f} (p90 {np.percentile(c[:, 0], 90):.0f}), "
              f"within 40 m {c[:, 1].mean():.2f} (p90 {np.percentile(c[:, 1], 90):.0f})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "../STU4-lab/fields_all41.json")
