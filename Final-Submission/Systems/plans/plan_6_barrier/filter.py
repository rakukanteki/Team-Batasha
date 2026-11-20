"""Control barrier function safety filter for Plan 6 (barrier)."""
from __future__ import annotations
import numpy as np


def barrier(r, v, rp, rv, need, alpha):
    """Thrust acceleration (world) the barrier condition demands, or zeros. need: keep-out radius per rock."""
    if not len(rp):
        return np.zeros(3)
    dr = r - rp
    dv = v - rv
    n = np.linalg.norm(dr, axis=1)
    b = -(2 * (dv ** 2).sum(1) + 4 * alpha * (dr * dv).sum(1) + alpha ** 2 * (n ** 2 - need ** 2))
    req = b / (2 * n)
    i = int(np.argmax(req))
    return req[i] * dr[i] / n[i] if req[i] > 0 else np.zeros(3)
