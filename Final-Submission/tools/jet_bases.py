"""Exact list of what the sim's jet selector can output (plan step 2, docs/plans/SCORE_70_PLAN.md).

The selector solves  min sum(x)  s.t.  A x = b, x >= 0  (tools/actuator_twin.py). A set B of six jets is its answer
for every command b inside the cone of B exactly when B is dual feasible: with y solving A_B^T y = 1, every other jet
has reduced cost 1 - A_j . y >= 0. That test does not depend on b, so every six-jet set is checked once. With all
reduced costs strictly positive the answer is unique, so the engine's revised simplex must return it too.

For each such set the jets that push along a body direction can all fire together: command b = sum of those columns.
The script reports, per thrust direction, the best single tick and the best zero-torque mix of those commands, and
verifies every command it reports through actuator_twin.tick.

Usage:  python tools/jet_bases.py
"""
import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools.actuator_twin import A, MAX_ACCEL, MAX_ANG_ACCEL, tick

STRICT = 1e-9          # reduced cost margin for a unique answer
K_MAX = 4.4            # kg per m/s allowed in a mix


def optimal_bases(chunk=200_000):
    """Every six-jet set that the selector returns uniquely, as an (n, 6) index array."""
    M = A.T                                                   # 6 x 38: column j is jet j's [linear, angular] output
    combos = np.array(list(itertools.combinations(range(M.shape[1]), 6)), dtype=np.int16)
    keep = []
    for s in range(0, len(combos), chunk):
        C = combos[s:s + chunk]
        B = M[:, C].transpose(1, 0, 2)                        # (n, 6, 6), columns = jets
        ok = np.abs(np.linalg.det(B)) > 1e-14
        C, B = C[ok], B[ok]
        y = np.linalg.solve(B.transpose(0, 2, 1), np.ones((len(C), 6, 1)))[..., 0]      # A_B^T y = 1
        red = 1.0 - y @ M                                     # (n, 38) reduced costs
        red[np.arange(len(C))[:, None], C] = np.inf           # basic jets do not count
        keep.append(C[(red > STRICT).all(1)])
    return np.concatenate(keep)


def commands(bases, ax):
    """Distinct commands b = sum of the basis jets that push along ax (at least two jets), scaled into the clips."""
    seen = {}
    for C in bases:
        S = tuple(sorted(int(j) for j in C if A[j, :3] @ ax > 0.9 * np.linalg.norm(A[j, :3])))
        if len(S) >= 2 and S not in seen:
            b = A[list(S)].sum(0)
            b = b * min(1.0, MAX_ACCEL / np.linalg.norm(b[:3]), MAX_ANG_ACCEL / max(np.linalg.norm(b[3:]), 1e-12))
            seen[S] = b
    return seen


def best_mix(R, ax):
    """Best time-shared mix of tick outputs R (n, 7) with zero mean torque and sideways thrust."""
    along = R[:, :3] @ ax; lat = R[:, :3] - np.outer(along, ax)
    side = [v for v in np.eye(3) if abs(v @ ax) < 0.5]
    G = np.vstack([R[:, 3:6].T, [lat @ v for v in side]])
    res = linprog(-along, A_ub=np.vstack([R[:, 6] - K_MAX * along, G, -G]), b_ub=np.r_[0.0, np.full(10, 1e-5)],
                  A_eq=[np.ones(len(R))], b_eq=[1.0], method="highs")
    return (-res.fun, res.x) if res.success else (float("nan"), None)


def main():
    bases = optimal_bases()
    print(f"{len(bases)} six-jet sets are a unique answer of the jet selector")
    for name, ax in zip(("+x", "+y", "+z", "-x", "-y", "-z"), np.vstack([np.eye(3), -np.eye(3)])):
        cmds = commands(bases, ax)
        sets = list(cmds); R = np.array([np.r_[acc, rate] for acc, rate, _ in (tick(cmds[S][:3], cmds[S][3:]) for S in sets)])
        along = R[:, :3] @ ax; i = int(np.argmax(along))
        a, w = best_mix(R, ax)
        k = (w @ R[:, 6]) / a if w is not None else float("nan")
        print(f"{name}: {len(sets)} jet subsets; best tick {along[i]:.3f} from {len(sets[i])} jets {sets[i]}; "
              f"zero-torque mix {a:.3f} m/s^2 at {k:.2f} kg/(m/s)")
        if w is not None:
            for j in np.argsort(-w)[:5]:
                if w[j] > 1e-3:
                    print(f"    {w[j]:.3f}  jets {sets[j]}  lin {np.round(cmds[sets[j]][:3], 5).tolist()}  "
                          f"ang {np.round(cmds[sets[j]][3:], 6).tolist()}  -> {along[j]:.3f}")


if __name__ == "__main__":
    main()
