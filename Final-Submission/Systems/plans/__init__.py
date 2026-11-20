"""
Systems/plans - per-plan packages (config.py + algorithm modules + planner.py) and unified registries.

Usage:
    from Systems.plans import PLAN_CONFIGS, PLANNERS, configure_plan, get_config, plan_name
    PLAN_CONFIGS["astar"]["max_pops"] = 50000
    PLAN_CONFIGS[1]["kmax"] = 3
"""
from __future__ import annotations

from pathlib import Path
from Systems.plans.common import (
    active_variant_info,
    activate_variant,
    load_plan_config,
    save_variant,
)
from Systems.plans import (
    plan_6_barrier,
    plan_8_astar,
    plan_9_cost_lattice,
    plan_10_real_calibrated,
    plan_11_v11,
    plan_13_v13,
    plan_14_v14,
    plan_16_v16,
    plan_17_v16_morning,
)

_BY_NUMBER = [
    (6, "barrier", plan_6_barrier.plan6, plan_6_barrier),
    (8, "astar", plan_8_astar.astar, plan_8_astar),
    (9, "cost_lattice", plan_9_cost_lattice.lattice, plan_9_cost_lattice),
    (10, "real_calibrated", plan_10_real_calibrated.real_calibrated, plan_10_real_calibrated),
    (11, "v11", plan_11_v11.v11, plan_11_v11),
    (13, "v13", plan_13_v13.v13, plan_13_v13),
    (14, "v14", plan_14_v14.v14, plan_14_v14),
    (16, "v16", plan_16_v16.v16, plan_16_v16),
    (17, "v16_morning", plan_17_v16_morning.v16, plan_17_v16_morning),
]

PLANNERS = {}
PLANNER_FACTORIES = {}
PLAN_CONFIGS = {}
PLAN_DEFAULTS = {}
PLAN_DIRS = {}
PLAN_NAMES = {}
for _num, _name, _fn, _module in _BY_NUMBER:
    _cfg_mod = _module.config
    _cfg = _cfg_mod.CONFIG
    _dir = Path(_cfg_mod.__file__).resolve().parent
    PLANNERS[_name] = _fn
    PLANNERS[_num] = _fn
    PLANNER_FACTORIES[_name] = PLANNER_FACTORIES[_num] = _module.make_planner
    PLAN_CONFIGS[_name] = _cfg
    PLAN_CONFIGS[_num] = _cfg
    PLAN_DEFAULTS[_name] = _cfg_mod._DEFAULTS
    PLAN_DEFAULTS[_num] = _cfg_mod._DEFAULTS
    PLAN_DIRS[_name] = _dir
    PLAN_DIRS[_num] = _dir
    PLAN_NAMES[_name] = _name
    PLAN_NAMES[_num] = _name


def plan_name(choice) -> str:
    """Canonical string name for a plan name, number, or custom callable."""
    if choice in PLAN_NAMES:
        return PLAN_NAMES[choice]
    if hasattr(choice, "__name__"):
        return str(choice.__name__)
    return str(choice)


def get_config(choice) -> dict:
    """Return the mutable configuration dict for a plan (by name or number)."""
    if choice in PLAN_CONFIGS:
        return PLAN_CONFIGS[choice]
    return {}


def configure_plan(choice, **overrides) -> dict:
    """Update configuration parameters/thresholds for a plan and return its config dict."""
    cfg = get_config(choice)
    cfg.update(overrides)
    return cfg


def reload_plan_config(choice, override_json=None) -> dict:
    """Reload plan config from config.json (following symlinks) + optional parameterized JSON override."""
    cfg = get_config(choice)
    if choice not in PLAN_DIRS:
        return cfg
    fresh = load_plan_config(str(PLAN_DIRS[choice] / "config.py"), PLAN_DEFAULTS[choice], extra_override=override_json)
    cfg.clear()
    cfg.update(fresh)
    return cfg


def get_plan_dir(choice) -> Path:
    return PLAN_DIRS[choice]


__all__ = [
    "PLAN_CONFIGS",
    "PLAN_DEFAULTS",
    "PLAN_DIRS",
    "PLAN_NAMES",
    "PLANNERS",
    "active_variant_info",
    "activate_variant",
    "configure_plan",
    "get_config",
    "get_plan_dir",
    "plan_name",
    "reload_plan_config",
    "save_variant",
    "plan_6_barrier",
    "plan_8_astar",
    "plan_9_cost_lattice",
    "plan_10_real_calibrated",
    "plan_11_v11",
    "plan_13_v13",
    "plan_14_v14",
]
