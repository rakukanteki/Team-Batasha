"""Read the organizer scene parameters used by the offline simulators."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SIM_ROOT = Path(__file__).resolve().parents[1] / "submodules" / "STU4-Sim"
FINAL_SCENE = SIM_ROOT / "Sims" / "FINAL_SUBMISSION.simconf"
PRACTICE_SCENE = SIM_ROOT / "Sims" / "COMPETITION_SIM.simconf"


class Scene:
    def __init__(self, path=FINAL_SCENE):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.entities = {
            e["#Required"]["Name"]: e
            for e in data["DataManager"]["entities"]
            if e.get("#Required", {}).get("Name")
        }
        self.systems = {
            system["Nametag"]: system["Inst_Parameters"]
            for system in data["SystemManager"]["Systems"]
            if "Inst_Parameters" in system
        }
        self.scoring = self.systems["Scoring"]
        self.shuttle = self.entities[self.scoring["Shuttle"][1]]
        self.target = self.entities[self.scoring["Target"][1]]

    def field(self):
        debris = self.debris
        return (
            np.array([e["Location"][1:] for e in debris], dtype=float).reshape(-1, 3),
            np.array([e["Velocity"][1:] for e in debris], dtype=float).reshape(-1, 3),
            np.array([e["RadarDiameter_m"] for e in debris], dtype=float),
            np.array(self.shuttle["Location"][1:], dtype=float),
            np.array(self.target["Location"][1:], dtype=float),
        )

    @property
    def debris(self):
        count = self.systems["Radar"]["NumDebris"]
        return [self.entities[f"Debris{i}"] for i in range(int(count[-1] if isinstance(count, list) else count))]

    def serialized_debris_order(self):
        """Map the original scene entity order to radar's numerical Debris indices."""
        return np.array([int(name.removeprefix("Debris")) for name in self.entities
                         if name.startswith("Debris") and name.removeprefix("Debris").isdigit()])

    @staticmethod
    def body_rotation(entity):
        """Active body-to-Phobos rotation from the scene's passive xyzw quaternion."""
        if entity["Dynamics"]["ResidentFrame"][1] != "Phobos":
            raise ValueError("Expected an entity resident in the Phobos frame")
        return Rotation.from_quat(entity["Rotation"][1:]).inv()

    def debris_attitudes(self, indices=None):
        debris = self.debris
        if indices is not None:
            debris = [debris[i] for i in indices]
        return DebrisAttitudes(debris)

    def arrival_score(self, seconds, fuel_kg, damage_j=0.0):
        scoring = self.scoring
        return float(100 * (
            0.5 + (scoring["NominalTime_s"] / seconds if seconds else 0.0)
            - max(fuel_kg / scoring["NominalFuel_kg"], 0.0)
            - 2 * damage_j / scoring["DamageThreshold_J"]
        ))

    def failure_score(self, distance_m, initial_distance_m, fuel_kg, damage_j):
        return float(100 * (
            0.5 - distance_m / initial_distance_m
            - max(fuel_kg / self.scoring["NominalFuel_kg"], 0.0)
            - 2 * damage_j / self.scoring["DamageThreshold_J"]
        ))


class DebrisAttitudes:
    """Constant scene angular velocities expressed in the Phobos frame."""

    def __init__(self, debris):
        self.initial = Rotation.concatenate([Scene.body_rotation(entity) for entity in debris])
        self.rates = np.array([entity["AngVelocity"][1:] for entity in debris])

    def at(self, seconds):
        return (Rotation.from_rotvec(self.rates * seconds) * self.initial).as_quat()
