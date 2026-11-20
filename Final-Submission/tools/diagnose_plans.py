"""Paired debris-density experiment on the rigid-body Phobos world (not official engine scores).

python tools/diagnose_plans.py --plan 8 --count 50 --seed 0
Retained rocks are evenly spaced scene indices, with their full-field trajectories unchanged.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Systems.Flight import Flight
from Systems.SimSuite import run_single
from Systems.plans import get_plan_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=int, required=True)
    parser.add_argument('--count', type=int, choices=(50, 500), default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--variant')
    parser.add_argument('--output', type=Path, default=ROOT / 'flightlogs/diagnosis_20261118')
    args = parser.parse_args()
    metrics = dict(planner_calls=0, planner_wall_s=0., max_planner_s=0., planner_errors=[], max_speed=0.,
                   last_sim_time=0., modes={}, accepted_actions={})
    original_init, original_decide = Flight.__init__, Flight.decide
    def init(self, *a, **kw):
        original_init(self, *a, **kw)
        planner = self.planner
        def timed(*a):
            start = time.monotonic()
            try:
                result = planner(*a)
                action = result[1]['action']
                metrics['accepted_actions'][action] = metrics['accepted_actions'].get(action, 0) + 1
                return result
            except Exception as exc:
                metrics['planner_errors'].append(repr(exc))
                raise
            finally:
                elapsed = time.monotonic() - start
                metrics['planner_calls'] += 1
                metrics['planner_wall_s'] += elapsed
                metrics['max_planner_s'] = max(metrics['max_planner_s'], elapsed)
        self.planner = timed
    def decide(self, t, r, v, *rest):
        import numpy as np
        metrics['last_sim_time'] = t
        metrics['max_speed'] = max(metrics['max_speed'], float(np.linalg.norm(v)))
        mode, vec = original_decide(self, t, r, v, *rest)
        metrics['modes'][mode] = metrics['modes'].get(mode, 0) + 1
        return mode, vec
    Flight.__init__, Flight.decide = init, decide
    overrides = None
    if args.variant:
        overrides = json.loads((get_plan_dir(args.plan) / 'configs' / f'{args.variant}.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    name = f'plan{args.plan}_{args.variant or "active"}_{args.count}_seed{args.seed}'
    # No full image/radar logging: the metrics below isolate search time, outcomes and burn duration.
    result = run_single(args.plan, scenario='phobos_world', seed=args.seed, debris_count=args.count,
                        config_overrides=overrides, log_run=False, raise_on_error=False)
    result.update(metrics)
    (args.output / f'{name}.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
