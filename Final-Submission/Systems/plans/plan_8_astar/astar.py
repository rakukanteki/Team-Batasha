"""Forward A* search algorithm for Plan 8 (astar)."""
from __future__ import annotations
import heapq
import numpy as np
from Systems.plans.plan_8_astar.config import CONFIG


def solve_astar_detail(G, max_pops=None):
    """A* forward: returns (cells, cost, stats)."""
    if max_pops is None:
        max_pops = int(getattr(G, "cfg", CONFIG).get("max_pops", CONFIG["max_pops"]))
    best = {}
    parent = {}
    heap = []
    g1 = G.edges(0, 0) + G.turn(0, G.slope0, 0)
    for b in range(G.ncell):
        best[1, 0, b] = g1[b]
        heapq.heappush(heap, (g1[b] + G.h(1, b), g1[b], 1, 0, b))
    pops = 0
    while heap and pops < max_pops:
        f, g, j, a, b = heapq.heappop(heap)
        pops += 1
        if g > best[j, a, b]:
            continue
        if j == G.J:
            if G.goal[b]:
                cells = [b]
                key = (j, a, b)
                while key[0] > 1:
                    key = parent[key]
                    cells.append(key[2])
                return cells[::-1], g, {"pops": pops, "priced_edges": G.priced, "stations": G.J}
            continue
        cost = g + G.edges(j, b) + G.turn(j, G.slope(j, a, b), b)
        for c in np.flatnonzero(cost < np.inf):
            key = (j + 1, b, int(c))
            if cost[c] < best.get(key, np.inf):
                best[key] = cost[c]
                parent[key] = (j, a, b)
                heapq.heappush(heap, (cost[c] + G.h(j + 1, c), cost[c], j + 1, b, int(c)))
    raise RuntimeError(f"A* found no path in {pops} expansions")


def solve_astar(G, max_pops=None):
    cells, g, _ = solve_astar_detail(G, max_pops=max_pops)
    return cells, g
