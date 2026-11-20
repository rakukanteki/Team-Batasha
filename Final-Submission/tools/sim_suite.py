"""
Python Simulator Suite CLI for Team Batasha.

Examples:
    python tools/sim_suite.py --plans astar,velocity_obstacle --scenarios point_mass,jet_twin_field --seeds 0,1
    python tools/sim_suite.py --all-plans --scenarios point_mass --seeds 0
    python tools/sim_suite.py --plans astar --sweep max_pops=2000,10000,50000,200000 --seeds 0
    python tools/sim_suite.py --plans 7 --set p_wide=0.05 --export-blender
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from Systems.SimSuite import SCENARIOS, run_single, run_suite  # noqa: E402
from Systems.plans import PLAN_NAMES, plan_name as canonical_plan_name  # noqa: E402


def _parse_val(raw: str):
    try:
        return ast.literal_eval(raw)
    except Exception:
        return raw


def _parse_plans(spec: str, all_plans: bool) -> list:
    if all_plans:
        return [name for key, name in PLAN_NAMES.items() if isinstance(key, int)]
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok) if tok.isdigit() else tok)
    return out


def _parse_seeds(spec: str) -> list[int]:
    if "," in spec:
        return [int(x.strip()) for x in spec.split(",") if x.strip()]
    n = int(spec)
    return list(range(n)) if n > 0 else [0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--plans", default="velocity_obstacle", help="comma-separated plan names or numbers")
    ap.add_argument("--all-plans", action="store_true", help="run every registered plan")
    ap.add_argument("--scenarios", default="point_mass", help=f"comma-separated scenarios: {','.join(SCENARIOS)}")
    ap.add_argument("--seeds", default="0", help="seed list (e.g. 0,1,2) or count (e.g. 3 -> 0..2)")
    ap.add_argument("--burn", default=None, help="burn law: direct, boost, plain, steer")
    ap.add_argument("--camera", default=None, help="camera use: off, consensus, inspect")
    ap.add_argument("--segmenter", default="classical", help="segmenter: classical, sam, yolo")
    ap.add_argument("--dt", type=float, default=None, help="override simulation step dt (s) for fast-forward sweeps")
    ap.add_argument("--max-time", type=float, default=420.0, help="max simulated time (s)")
    ap.add_argument("--set", dest="overrides", action="append", default=[], help="override plan config key=value")
    ap.add_argument("--config-json", default=None, help="parameterized JSON override file (highest priority)")
    ap.add_argument("--sweep", default=None, help="sweep one parameter: key=v1,v2,v3")
    ap.add_argument("--flush-immediately", action="store_true", help="flush run.log immediately on each entry")
    ap.add_argument("--log-dir", default=None, help="output root directory (default: flightlogs)")
    ap.add_argument("--ram", action="store_true", help="store run logs and suite reports in /dev/shm RAM tmpfs")
    ap.add_argument("-j", "--workers", type=int, default=1, help="parallel worker processes for run_suite")
    ap.add_argument("--fast", action="store_true", help="fast mode: enable --ram and set --dt 0.2 for point-mass runs")
    ap.add_argument("--no-log", action="store_true", help="disable run.log/run.json generation")
    ap.add_argument("--export-blender", action="store_true", help="export each run.json to blender/<run_id>.blend")
    ap.add_argument("--list-scenarios", action="store_true", help="list available simulator scenarios and exit")
    args = ap.parse_args()
    if args.fast:
        args.ram = True
        if args.dt is None:
            args.dt = 0.2

    if args.list_scenarios:
        for k, v in SCENARIOS.items():
            print(f"{k:20s}  dt={v['default_dt']:.4f}s  cam={v['supports_camera']}  - {v['description']}")
        return

    plans = _parse_plans(args.plans, args.all_plans)
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    seeds = _parse_seeds(args.seeds)

    overrides = {}
    for item in args.overrides:
        if "=" not in item:
            raise ValueError(f"Expected key=value in --set, got {item!r}")
        k, v = item.split("=", 1)
        overrides[k.strip()] = _parse_val(v.strip())

    if args.sweep:
        if "=" not in args.sweep:
            raise ValueError(f"Expected key=v1,v2,... in --sweep, got {args.sweep!r}")
        sk, svals = args.sweep.split("=", 1)
        sk = sk.strip()
        vals = [_parse_val(x.strip()) for x in svals.split(",") if x.strip()]
        print(f"Parameter sweep {sk} = {vals}")
        for val in vals:
            ov = {**overrides, sk: val}
            for sc in scenarios:
                for pl in plans:
                    for s in seeds:
                        res = run_single(
                            pl, scenario=sc, seed=s, burn=args.burn, camera=args.camera, segmenter=args.segmenter,
                            config_overrides=ov, dt=args.dt, max_time=args.max_time,
                            flush_immediately=args.flush_immediately, log_dir=args.log_dir,
                            log_run=not args.no_log, export_blender=args.export_blender, ram=args.ram,
                            param_json=args.config_json,
                        )
                        err_note = f" ERR={res['error']}" if res.get("error") else ""
                        print(f"[{sk}={val}] {res['scenario']} | {res['plan']} | seed={s} | "
                              f"score={res['score']} arrived={res['arrived']} T={res['T']}s fuel={res['fuel']}kg "
                              f"hits={res['hits']} min_clear={res['min_clear']}m wall={res['wall_s']}s{err_note}")
        return

    def on_progress(done, total, row):
        print(f"[{done}/{total}] {row['scenario']} | {row['plan']} | seed={row['seed']} -> "
              f"score={row['score']} arrived={row['arrived']} T={row['T']}s fuel={row['fuel']}kg "
              f"hits={row['hits']} clear={row['min_clear']}m wall={row['wall_s']}s", flush=True)

    rep = run_suite(
        plans=plans, scenarios=scenarios, seeds=seeds, burn=args.burn, camera=args.camera,
        segmenter=args.segmenter, config_overrides=overrides, dt=args.dt, max_time=args.max_time,
        flush_immediately=args.flush_immediately, log_dir=args.log_dir, log_run=not args.no_log,
        export_blender=args.export_blender, progress_cb=on_progress, workers=args.workers, ram=args.ram,
        param_json=args.config_json,
    )
    print("\n=== SUITE SUMMARY ===")
    for row in rep["summary"]:
        print(f"{row['scenario']:18s} | {row['plan']:18s} | n={row['runs']} arr={row['arrived']}/{row['runs']} "
              f"hits={row['hit_runs']} mean_score={row['mean_score']} min={row['min_score']} max={row['max_score']} "
              f"fuel={row['mean_fuel']} T={row['mean_T']} clear={row['min_clear']}m wall={row['mean_wall_s']}s")
    print(f"Suite report saved to: {rep['report_path']}")


if __name__ == "__main__":
    main()
