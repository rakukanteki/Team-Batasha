# Team Batasha — code used for the submitted runs

This folder is an exact copy of the files that flew runs `261120_082545_FINAL_SUBMISSION__2bc` and `261120_082915_FINAL_SUBMISSION__b21`, taken file by file from commit `e9dfff2` (branch `faysal/plan17-guard`; same flight settings as `68d0967`, plus opt-in pulse-check and trust-gate code paths that are switched off).

## What runs in the sim

- The scene is the **unmodified official** `STU4-Sim/Sims/FINAL_SUBMISSION.simconf` (STU4-Sim `0d753a2`). Nothing in the sim config was changed.
- That scene runs `Local/Repos/STU4-Solution/Controller.py`. It is generated from the official STU4-Solution starter and hands the sim helpers to `Systems.BasicController.run`.
- `Systems/BasicController.py` defaults to **Plan 17 `v16_morning`** (`PLANNER = "v16_morning"`), which selects the calibrated tracker and the steered burn. `RECORD = True` saves each flight for offline replay.
- `Systems/plans/plan_17_v16_morning/config.py` sets `guard_trust_plan: True`. With it, the last-resort escape guard ignores a rock that the active route's own burn already clears. This removed most of the double dodging that wasted fuel in earlier runs.

## How to run

1. Put this folder in Space Teams PRO as the `STU4-Solution` asset repo, at `AppData/Local/SimDynamX/STPro/Assets/Repos/STU4-Solution`.
2. Open `Repos/STU4-Sim/Sims/FINAL_SUBMISSION.simconf` in Space Teams PRO 0.56.0. In the lobby, choose the Orbit Pawn role and press Start Simulation.
3. The controller needs `numpy` and `scipy` in the Space Teams Python runtime (`requirements.txt`).

## Other folders

## Files

- `Controller.py`, `Systems/`: the controller code the sim runs. `Systems/plans/` keeps only Plan 17 and the plans it builds on (6, 8, 9, 10, 11, 13, 14, 16). The unused plans 1-5, 7 and 15 were removed along with their registry imports; the flight code itself is unchanged.
- `requirements.txt`: Python packages the controller needs.
- `Batasha-sim-logs.zip`: the two engine log folders.
- `CLAIM.md`: the claimed time, fuel and damage.
