# Team Batasha — Final Submission package F (claimed result)

Deadline: **Nov. 20, 11:59 pm Central** = Nov. 21 05:59 UTC = Nov. 21 11:59 am Bangladesh time.

## Claimed result: runs 1 and 2, consecutive in one uncut recording

| | Run 1 | Run 2 | Average (claim) |
|---|---:|---:|---:|
| Score | 56.1 | 56.1 | **56.1** |
| Time (s) | 188.24 | 195.00 | 191.62 |
| Fuel (kg) | 43.25 | 41.61 | **42.43** |
| Damage (J) | 0.0 | 0.0 | **0.0** |

Engine runs: `261120_082545_FINAL_SUBMISSION__2bc` (run 1) and `261120_082915_FINAL_SUBMISSION__b21` (run 2). These are the only two runs in the video.

## Files

- **`Batasha-final-run-video.mkv`**: 8:04, 1920×1080, 30 fps. One continuous desktop recording that ffmpeg finished itself, with no cuts, edits or re-encoding.
  - It is a byte-identical copy of `E:\Batasha-videos\Batasha-trust-live-20261120-142519.mkv`.
  - SHA-256 `06ADE23EA4C41B0623939D523C89F93CB882F1BDA13F83B8A4A5DE8539D1E9E2`.
  - Each run is started by hand in Space Teams PRO and ended with the sim's own Exit Sim.
  - The last few seconds, after run 2 ended, show the desktop with the agent windows.
- **`Batasha-sim-logs.zip`**: exactly the two engine log folders. SHA-256 `1A6227BE03E5A4EE57C4F5EFA71ADC0B5D919E660A7C8F053F9B11070D3C3B5C`.
- **`Batasha-code/`**: the exact competitor code (193 files) from commit `e9dfff2` on `faysal/plan17-guard`. See `Batasha-code/SUBMISSION_CODE.md`.

## How the runs were made

- Scene: the **unmodified official** `STU4-Sim/Sims/FINAL_SUBMISSION.simconf` (STU4-Sim `0d753a2`). Space Teams PRO Beta v0.56.0, viewer role Orbit Pawn.
- Controller: Plan 17 `v16_morning`, tracker calibrated, steered burn.
  - Both runs' logs show `guard_trust_plan: true`, `coast_docking_radius_m: 24`, and no pulse check.
  - These are the same flight settings as the 54.35 build `68d0967`. The extra opt-in code paths in `e9dfff2` are switched off.

## Still to do before submitting

1. Push `Batasha-code/` (commit `e9dfff2`) to `Team-Batasha/Final-Submission`, replacing the 54.35 code, so the code matches this video.
2. Upload the video and get a shareable link.
3. Write the "additional information about your solution" text.
4. Fill in the form: https://forms.gle/19HRACS55Js82K8d9
