"""
Systems/Recorder.py - flight recorder: wraps any Interfaces.Sim and saves what the flight saw and commanded.
No sim calls of its own. Switched on by "Record" in the simconf's Controller Inst_Parameters (sims/PLAN_*.simconf);
the organizers' simconfs do not set it. tools/replay.py feeds a recording back through the tracker and planner.

File (flightlogs/<time>.npz, rewritten every 5 s so a stopped sim still leaves one):
    ticks  [n, 26]  t, r(3), v(3), body rates(3), body axes(9, rows = world x, y, z in body axes), lin(3), ang(3), fuel
    ents   [m, 3, 3]  every distinct radar entry seen
    read_t [k], add_n [k], rem_n [k], add [sum add_n], rem [sum rem_n]   per radar read: its time, and the indices
    into ents of the entries that appeared and disappeared since the read before
    target [3]
"""
from __future__ import annotations
import os
import threading
import time
import numpy as np


class Recorder:
    def __init__(self, sim, folder):
        self.sim = sim; os.makedirs(folder, exist_ok=True)
        self.path = os.path.join(folder, time.strftime("%y%m%d_%H%M%S") + ".npz")
        self.ticks = []; self.ents = []; self.index = {}; self.cur = set()
        self.read_t = []; self.add = []; self.rem = []; self.add_n = []; self.rem_n = []
        self.t = 0.0; self.rv = (np.zeros(3), np.zeros(3)); self.w = np.zeros(3); self.used = 0.0
        self.axes = np.eye(3); self.n = 0; self.saved = -1e9; self.busy = False
        self.goal = np.asarray(sim.target(), float)

    def __getattr__(self, name):                # wait, target, to_body, capture, log: straight through
        return getattr(self.sim, name)

    def time(self):
        self.t = self.sim.time(); return self.t

    def state(self):
        self.rv = self.sim.state(); return self.rv

    def rates(self):
        self.w = self.sim.rates(); return self.w

    def fuel(self):
        self.used = self.sim.fuel(); return self.used

    def radar(self):
        entries = self.sim.radar(); now = set()
        for m in entries:
            k = np.ascontiguousarray(m, dtype=float).tobytes(); i = self.index.get(k)
            if i is None:
                i = self.index[k] = len(self.ents); self.ents.append(np.array(m, float))
            now.add(i)
        new = sorted(now - self.cur); gone = sorted(self.cur - now); self.cur = now
        self.read_t.append(self.t); self.add += new; self.rem += gone; self.add_n.append(len(new)); self.rem_n.append(len(gone))
        return entries

    def command(self, lin, ang):
        self.sim.command(lin, ang)
        if self.n % 6 == 0:                     # body axes change slowly: three extra sim calls every sixth tick
            self.axes = np.array([self.sim.to_body(e) for e in np.eye(3)])
        self.n += 1
        self.ticks.append(np.r_[self.t, self.rv[0], self.rv[1], self.w, self.axes.ravel(), lin, ang, self.used])
        if self.t - self.saved >= 5.0 and not self.busy:
            self.saved = self.t; self.save()

    def save(self, wait=False):
        data = dict(ticks=np.array(self.ticks), ents=np.array(self.ents), read_t=np.array(self.read_t),
                    add=np.array(self.add, int), rem=np.array(self.rem, int), add_n=np.array(self.add_n, int),
                    rem_n=np.array(self.rem_n, int), target=self.goal)
        self.busy = True
        job = threading.Thread(target=self._write, args=(data,), daemon=True); job.start()
        if wait:
            job.join()

    def _write(self, data):
        try:                                    # a recording must never stop the flight
            tmp = self.path + ".tmp.npz"; np.savez_compressed(tmp, **data); os.replace(tmp, self.path)
        except Exception:
            pass
        self.busy = False


def reads(log):
    """Yield (t, entries) for every radar read of a loaded recording (np.load of a Recorder file)."""
    ents = log["ents"]; cur = set(); a = b = 0
    for t, na, nr in zip(log["read_t"], log["add_n"], log["rem_n"]):
        cur -= set(log["rem"][b:b + nr].tolist()); cur |= set(log["add"][a:a + na].tolist()); a += na; b += nr
        yield float(t), [ents[i] for i in sorted(cur)]
