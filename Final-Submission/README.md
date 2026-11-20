# Team Batasha: STU4 Escape from Phobos (2026)

Final submission controller for Space Teams University 2026, *Escape from Phobos*.

| Claimed result (average of 2 consecutive runs) | Score | Time | Fuel | Damage |
|---|---:|---:|---:|---:|
| Run 1 (`261120_082545_FINAL_SUBMISSION__2bc`) | 56.1 | 188.24 s | 43.25 kg | 0 J |
| Run 2 (`261120_082915_FINAL_SUBMISSION__b21`) | 56.1 | 195.00 s | 41.61 kg | 0 J |
| **Claim** | **56.1** | **191.62 s** | **42.43 kg** | **0 J** |

The code here is exactly what flew those runs: commit `e9dfff2`, branch `faysal/plan17-guard`, Plan 17 `v16_morning` with `guard_trust_plan`. The sim logs are in `Batasha-sim-logs.zip`, and `CLAIM.md` has the details.

# Approach

Lift off at once, track every rock with a Kalman filter on radar, plan timed burns in a background thread, and keep a last-resort escape guard. The controller is radar-only; the cameras are not used.

```
INITIALIZATION (Controller.py: official starter block, unchanged)
→ Connect to the sim, get entity refs (Shuttle, Target, Phobos, cameras) and frames
→ Wait until debris spawning finishes
→ Hand the starter helpers (SetCommand, GetNearestDebrisLocations, PanTiltCamera) to Systems.BasicController.run

MAIN LOOP (BasicController.fly → Flight.step, at the scene's LoopFreqHz)
→ Read sim time, the shuttle's position, velocity, attitude and body rates (Phobos frame), fuel used and damage
→ Read radar: GetNearestDebrisLocations(500)  (noisy Radar.SortedDebrisData only)

  RADAR TRACKING (Radar.py, CalibratedWorldModel)
  → Match each radar return to an existing track (gated nearest neighbour / assignment)
  → Kalman update of each rock's position, velocity and covariance
  → Start-up burst: fast convergence in the first seconds; bump gate rejects outliers

  LAUNCH (first seconds)
  → Thrust within 8° of straight up until 10 m of clearance from Phobos

  ROUTE PLANNING (background thread, Plan 17 → plan_16_v16 planner)
  → Wait until tracks along the route are reliable (≥ 3 samples, σ ≤ 12 m, age ≤ 9 s)
  → Pass a snapshot of shuttle + tracks to a planning thread; the flight loop keeps flying meanwhile
  → Planner:
     → First check whether a pure coast reaches the 50 m arrival sphere around the target
     → Simulate timed burns in Phobos's rotating frame, including turn time and fuel use
     → Prefer a burn over a viable coast only if it arrives > 5 s earlier and predicted fuel ≤ 43 kg
  → Reject results older than 1.2 s; accept the newest valid schedule

  SAFETY
  → Recheck the next 30 s of the accepted trajectory against fresh radar every cycle
  → If the schedule becomes unsafe → fall back to waypoint guidance
  → ESCAPE GUARD (last resort): if a rock still threatens the shuttle after the planned burn is
    accounted for, test 13 thrust directions at full thrust and fly the one with the most clearance

  ACTUATION (Actuation.py, "steer" burn)
  → Main burn on body +Z: slew towards the burn direction; full thrust once within a 35° cone,
    boost thrust once aligned within 10° and calm
  → Coast inside the 24 m docking radius; stop when the target is reached
  → SetCommand(linear_accel, angular_accel)

END
→ The sim's scoring system ends the run on arrival; the run is closed with Exit Sim
```

**Two design choices from testing**
- **Body +Z for the main burn.** It lets us accelerate immediately. Body +X has more thrust once aligned, but the ~90° turn it needs costs fuel and time, which lowered the score.
- **Radar only.** Radar gives 3D positions and closing speeds in any lighting. Adding camera bearings made little difference to tracking accuracy in our tests, so `CAMERA = "off"`.

# Environment setup

1. Install **Space Teams PRO v0.56.0** (Beta) and sign in.
2. In Space Teams PRO, download the official asset repos **STU4-Sim** and **STU4-Solution**. They land in `%LOCALAPPDATA%\SimDynamX\STPro\Assets\Repos\`.
3. Replace the contents of `Assets\Repos\STU4-Solution` with this folder. A junction also works:
   ```powershell
   New-Item -ItemType Junction -Path "$env:LOCALAPPDATA\SimDynamX\STPro\Assets\Repos\STU4-Solution" -Target "<path to>\Final-Submission"
   ```
4. Make sure the Space Teams Python runtime has the packages in `requirements.txt`, which are `numpy` and `scipy`.

# Running

1. Open Space Teams PRO, then **Sim → `FINAL_SUBMISSION`**. This is the unmodified official `STU4-Sim/Sims/FINAL_SUBMISSION.simconf`, at STU4-Sim `0d753a2`.
2. In the lobby, choose the **Orbit Pawn** role and press **Start Simulation**.
3. The controller starts by itself. The result appears on screen and in `%LOCALAPPDATA%\SimDynamX\STPro\Logs\<timestamp>_FINAL_SUBMISSION__xxx\`.
4. Use **Exit Sim** after each run.

No sim config changes are needed. Each run also writes its own flight log to `flightlogs/` next to this file, and git ignores that folder.

# Sim API used

Everything goes through the official STU4-Solution starter helpers in `Controller.py`.

| Call | What we use it for |
|---|---|
| `GetNearestDebrisLocations(500)` | Noisy radar returns as 3×3 arrays: row 0 = relative position; row 1 = closest point, range rate, diameter; row 2 = measurement time |
| `shuttle.getLocVelAcc()`, `getRotation()`, `getAngVelocity()` | Own position, velocity, attitude and body rates in the Phobos frame |
| `target.getLocVelAcc()` | Target position |
| `shuttle.GetParam(["Stats","FuelUsed"])`, `["Stats","Damage"]` | Fuel and damage, for logging and the fuel cap |
| `SetCommand(linear_accel, angular_accel)` | Thrust and attitude commands in the body frame (the sim's jet selector fires the jets) |
| `SimClock.GetTimeNow()`, `LoopFreqHz` | Sim time and loop rate |

The controller never reads debris entities directly or uses any noise-free data.

# Code layout

| Path | Role |
|---|---|
| `Controller.py` | Official starter; its protected block is unchanged. Calls `Systems.BasicController.run` |
| `Systems/BasicController.py` | The sim loop, and the only module that talks to the sim. Picks `PLANNER`, `TRACKER`, `BURN`, `CAMERA` by name |
| `Systems/Flight.py` | Flight logic: launch, route hand-off, review, waypoint fallback, escape guard |
| `Systems/Radar.py` | Radar tracking (Kalman / calibrated Kalman) |
| `Systems/Actuation.py` | Burn and attitude laws (`steer` burn) |
| `Systems/Core.py`, `Maneuvers.py` | Dynamics in the rotating frame and maneuver data types |
| `Systems/Planners.py`, `Systems/plans/` | Planner registry. Plan 17 (`plan_17_v16_morning`) is flown, and the plans it builds on are kept (6, 8, 9, 10, 11, 13, 14, 16) |
| `Systems/RunLogger.py`, `Recorder.py` | Local flight logs for offline replay |
| `Systems/UserAPI.py` | Vehicle constants and data structures |
| `requirements.txt` | Python packages |
| `Batasha-sim-logs.zip`, `CLAIM.md` | Submission evidence |

The unused plans 1–5, 7 and 15 were removed along with their registry imports. The flight code is unchanged: an offline bench over 5 seeds matches `e9dfff2` exactly, and a live re-run of this folder in Space Teams PRO escaped with 53.2 points and 0 damage.
