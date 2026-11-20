"""Exact dynamic-programming search over the priced lattice for Plan 9 (cost_lattice)."""
from __future__ import annotations
import numpy as np


def solve_lattice_detail(G):
    """Exact search: every edge priced, then a forward dynamic program over (previous cell, cell)."""
    D = (G.edges(0, 0) + G.turn(0, G.slope0, 0))[None, :]          # [a, b] at station 1; a: the start
    back = []
    for j in range(1, G.J):
        E = G.price(j, np.arange(G.ncell))                          # [b, c]
        gi = G.ss[j] - G.ss[j - 1]
        go = G.ss[j + 1] - G.ss[j]
        k = np.linalg.norm((G.yy[None, None] - G.yy[None, :, None]) / go
                           - ((G.yy[None, :] - G.Y[j - 1][:, None]) / gi)[:, :, None], axis=3)   # [a, b, c]
        tot = D[:, :, None] + G.w[j] * k + G.lam_node * (k > G.eps_turn) + E[None]
        back.append(tot.argmin(0))
        D = tot.min(0)
    last = np.where(G.goal[None, :], D, np.inf)
    b, c = np.unravel_index(int(np.argmin(last)), last.shape)
    cells = [int(c), int(b)]
    for j in range(G.J - 2, 0, -1):
        cells.append(int(back[j][cells[-1], cells[-2]]))
    seq = cells[::-1][-G.J:]
    return seq, float(last[b, c]), {"priced_edges": G.priced, "stations": G.J}


def solve_lattice(G):
    cells, cost, _ = solve_lattice_detail(G)
    return cells, cost
