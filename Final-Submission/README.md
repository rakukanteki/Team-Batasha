# Team Batasha — code used for the submitted runs

This folder is an exact copy of the files that flew the three runs in this video (engine logs `261120_033653_FINAL_SUBMISSION__801`, `261120_034030_FINAL_SUBMISSION__4c4`, `261120_034406_FINAL_SUBMISSION__ef8`), taken file by file from commit `68d0967` (branch `faysal/plan17-guard`).

## What runs in the sim

- The scene is the **unmodified official** `STU4-Sim/Sims/FINAL_SUBMISSION.simconf` (STU4-Sim `0d753a2`). Nothing in the sim config was changed.
- That scene runs `Local/Repos/STU4-Solution/Controller.py`. It is generated from the official STU4-Solution starter and hands the sim helpers to `Systems.BasicController.run`.
- `Systems/BasicController.py` defaults to **Plan 17 `v16_morning`** (`PLANNER = "v16_morning"`), which selects the calibrated tracker and the steered burn. `RECORD = True` saves each flight for offline replay.
- `Systems/plans/plan_17_v16_morning/config.py` sets `guard_trust_plan: True`. With it, the last-resort escape guard ignores a rock that the active route's own burn already clears. This removed most of the double dodging that wasted fuel in earlier runs.

## How to run

1. Put this folder in Space Teams PRO as the `STU4-Solution` asset repo, at `AppData/Local/SimDynamX/STPro/Assets/Repos/STU4-Solution`.
2. Open `Repos/STU4-Sim/Sims/FINAL_SUBMISSION.simconf` in Space Teams PRO 0.56.0. In the lobby, choose the Orbit Pawn role and press Start Simulation.
3. The controller needs `numpy` and `scipy` in the Space Teams Python runtime.

## Other folders

- `tools/`: offline benchmarks and replay utilities used during development. Not needed in the sim.
- `tests/`: offline tests. Some need the official upstream repos as Git submodules, which are not included here.
