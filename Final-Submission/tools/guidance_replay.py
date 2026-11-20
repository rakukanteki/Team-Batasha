"""Guidance replay: the planner and tracker of a config, fed an engine run's own radar reads along its own path.

    python tools/guidance_replay.py NPZ [RUN_DIR] [--set key=json ...] [--verbose]

NPZ is the run's Recorder file (ticks and radar reads); RUN_DIR its flightlogs/spaceteams folder (the config is read
from run.json, else Plan 10 default). The path is the one flown, so the plans do not move the shuttle: this counts how
often a guidance setting changes its mind on real radar data, and why (the kept plan's p_hit at each switch).
Offline testing only (FINDINGS 7.16).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from Systems import Core as C                                          # noqa: E402
from Systems.Flight import Flight                                      # noqa: E402
from Systems.Recorder import reads                                     # noqa: E402
from Systems.plans import get_config                                   # noqa: E402

TGT = np.array([-850.0, 120.0, -9530.0])


def load_config(run_dir, plan="real_calibrated"):
    cfg = dict(get_config(plan))
    if run_dir and (Path(run_dir) / "run.json").is_file():
        cfg.update(json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))["metadata"]["config"])
    return cfg


def replay(npz, cfg, until=None):
    log = np.load(npz); T = log["ticks"]; R = list(reads(log))
    C.configure(T[0, 1:4], TGT, a_burn=cfg.get("a_burn") or 0.125)
    f = Flight("real_calibrated", None, threaded=False, config=cfg)
    seen = {}; orig = C.score_plans

    def score(cands, st, rocks, cfg=None):                             # remember the kept plan's score at each call
        ev = orig(cands, st, rocks, cfg=cfg); seen["keep"] = (float(ev["ptot"][-1]), float(ev["cost"][-1])); return ev
    C.score_plans = score
    switches = []; k = 0; prev = T[0, 0]
    try:
        for row in T:
            t = row[0]
            if until is not None and t > until:
                break
            while k < len(R) and R[k][0] <= t:
                f.radar(R[k][1], R[k][0]); k += 1
            n = len(f.events); seen.clear(); had = f.plan is not None
            f.decide(t, row[1:4], row[4:7], row[25], max(t - prev, 1e-3)); prev = t
            if len(f.events) > n and had:
                tt, action, K, p, cost = f.events[-1]
                kp, kc = seen.get("keep", (np.nan, np.nan))
                switches.append(dict(t=tt, action=action, p_new=p, cost_new=cost, p_keep=kp, cost_keep=kc))
    finally:
        C.score_plans = orig
    return switches


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("npz"); ap.add_argument("run_dir", nargs="?"); ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--until", type=float); ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.run_dir)
    for kv in a.set:
        key, val = kv.split("=", 1); cfg[key] = json.loads(val)
    sw = replay(a.npz, cfg, a.until)
    if a.verbose:
        for s in sw:
            print(f"  t {s['t']:6.1f} {s['action']:6s} new p {s['p_new']:.5f} cost {s['cost_new']:7.2f} | "
                  f"kept p {s['p_keep']:.5f} cost {s['cost_keep']:7.2f}")
    p_safe = cfg.get("p_safe", C.P_SAFE); band = cfg.get("keep_band", 3.0) * p_safe
    unsafe = sum(s["p_keep"] > band for s in sw)
    print(f"switches {len(sw)}  (kept plan over the keep band: {unsafe}, cheaper by the margin: {len(sw) - unsafe})")


if __name__ == "__main__":
    main()
