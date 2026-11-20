"""
Team Batasha flight code (Space Teams University 2026: Escape from Phobos).

Core        frames, node guidance, plan simulation, cost function
Radar       radar association and per-rock filters
Planners    the seven planners, one interface
Flight      the flight brain: replanning schedule, guidance state, guard
Actuation   SetCommand laws
Vision      camera geometry and the segmentation interface
BasicController   the only module that calls the sim
"""
