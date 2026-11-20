"""
Systems/Interfaces.py - every plug point of the controller, in one place.

Each part is chosen by one line at the top of Systems/BasicController.py and looked up by name in a registry
that sits next to its implementations:

    part         chosen by     registry              interface below
    planner      PLANNER       Planners.PLANNERS     Planner
    tracker      TRACKER       Radar.TRACKERS        Tracker
    burn law     BURN          Actuation.BURNS       BurnLaw
    camera use   CAMERA        Vision.CAMERAS        CameraUse
    segmenter    SEGMENTER     Vision.SEGMENTERS     Segmenter
    simulator    (the caller)  -                     Sim

To add a part: write a function or class with the interface, add one line to its registry, and name it in
BasicController.py. An object with the interface can also be given directly in place of a name.
These are typing.Protocol classes: an implementation needs the same methods; it does not inherit from them.
tests/test_flight_code.py checks every registry entry against its interface.
"""
from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class Sim(Protocol):
    """Adapter around a simulator: the flight loop (BasicController.fly) uses nothing else.
    SpaceTeamsSim (BasicController.py) wraps the organizers' helpers; TwinSim (tools/actuator_twin.py) is the
    offline stand-in. Vectors are numpy arrays; positions and velocities are in the Phobos frame."""

    def wait(self) -> bool:
        """Let one control period pass. Returning False ends the flight loop."""
    def time(self) -> float:
        """Seconds since the start of the run."""
    def state(self):
        """(position, velocity) of the shuttle."""
    def target(self):
        """Position of the target."""
    def rates(self):
        """Body rates in body axes, rad/s."""
    def to_body(self, vec):
        """A Phobos-frame direction expressed in body axes."""
    def fuel(self) -> float:
        """Fuel used so far, kg."""
    def radar(self):
        """The published radar list: 3x3 arrays as GetNearestDebrisLocations returns them."""
    def command(self, lin, ang) -> None:
        """Send the linear and angular acceleration command, body axes."""
    def capture(self, top, yaw_deg, pitch_deg, fov_deg):
        """Point the top or bottom camera and return an H x W x 3 uint8 image, or None. May block 100-300 ms."""
    def log(self, msg, error=False) -> None:
        """Write a line to the sim's log."""


@runtime_checkable
class Tracker(Protocol):
    """The world model: which radar entry is which rock, and each rock's state estimate. `n` is the rock count."""
    n: int

    def read(self, entries, t_now, own_at):
        """Take one radar list. own_at(t) -> (position, velocity) of the shuttle at time t."""
    def predict(self, t):
        """Every rock at time t: (pos n x 3, vel n x 3, hard_radius n, sig_pos n, sig_vel n)."""
    def radius(self):
        """Best radius estimate per rock, metres."""
    def sight(self, i, t, H, e, var) -> None:
        """Camera fixes for rocks i at time t: e (k x 2) = H (k x 2 x 3) @ position error, noise variance var (k)."""
    def set_radius(self, i, radius) -> None:
        """A camera-measured outline radius for rock i."""


@runtime_checkable
class Planner(Protocol):
    """planner(st, rocks, plan, committed) -> (plan, info). See the top of Systems/Planners.py.
    Optional attribute `barrier_alpha`: Flight then also runs the barrier filter with that gain.
    A plan may carry a fourth item, a Maneuver with timed actions and a waypoint fallback."""

    def __call__(self, st, rocks, plan, committed): ...


@runtime_checkable
class BurnLaw(Protocol):
    """burn(d, w, state) -> (lin, ang) for the main burn. d: unit thrust direction in body axes, w: body rates,
    state: a dict the caller keeps between ticks. `accel` is the thrust the planners should assume, m/s^2."""
    accel: float

    def __call__(self, d, w, state): ...


class AttitudePolicy(Protocol):
    """Optional BurnLaw.attitude: body angular command tracking a body-frame target vector in every flight mode."""

    def __call__(self, ref, w): ...


@runtime_checkable
class Segmenter(Protocol):
    """Finds rocks in an image. The slot for pretrained and custom-trained models."""

    def segment(self, image, prompts=None):
        """image: H x W x 3 uint8. prompts: list of (u, v, expected_radius_px) from the radar, or None.
        With prompts: one boolean H x W mask per prompt, or None where nothing was found.
        Without: a list with one boolean mask per object found anywhere in the image."""


@runtime_checkable
class CameraUse(Protocol):
    """What the camera is used for. Built as Use(segmenter)."""

    def due(self, t, coasting) -> bool:
        """Whether to capture now. The loop zeroes the jets first, because a capture blocks."""
    def __call__(self, sim, world, t, r, v):
        """Capture through sim.capture and feed what is learned into the tracker `world`."""


class TrainingCaptureSource(Protocol):
    """Unwired image dataset source. No control commands and no debris truth queries."""

    def time(self) -> float: ...
    def point(self, spec) -> None: ...
    def snapshot(self, spec) -> dict: ...
    def capture(self, spec) -> dict | None:
        """Original RGB8 channel buffers keyed R, G, B, or None on failure."""


def pick(registry, choice, what="part", *args):
    """The entry of `registry` named `choice`, or `choice` itself if it is not a name. A class is built with args."""
    if isinstance(choice, (str, int)):
        if choice not in registry:
            raise KeyError(f"unknown {what} {choice!r}: choose one of {sorted(map(str, registry))}")
        choice = registry[choice]
    return choice(*args) if isinstance(choice, type) else choice
