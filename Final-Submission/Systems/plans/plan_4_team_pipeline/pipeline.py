"""Plan 4 (team_pipeline): Plan 7's candidate search flown with Plan 4's config, which adds settings by route zone
(docs/plans/PLAN_4_TEAM_PIPELINE.md): a lattice route around the first cluster ("corridor"), a wide fallback where no
plan is safe ("wide_fallback"), and a faster cruise in the clear last stretch ("late_after_m", "late_vc", Flight.decide)."""
from __future__ import annotations
from Systems.plans.plan_7_velocity_obstacle.vo import select_velocity_obstacle


def run_pipeline(st, rocks, plan, committed, cfg):
    return select_velocity_obstacle(st, rocks, plan, committed, cfg)
