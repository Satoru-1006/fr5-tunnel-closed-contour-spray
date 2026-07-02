from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SprayTool:
    spray_distance: float = 0.18
    spray_angle_deg: float = 35.0

    @property
    def footprint_width(self) -> float:
        return 2.0 * self.spray_distance * np.tan(np.deg2rad(self.spray_angle_deg) / 2.0)

    def radius_at_wall(self) -> float:
        return self.footprint_width / 2.0


@dataclass(frozen=True)
class ContactTool:
    stand_off: float = 0.035
    contact_width: float = 0.12

    @property
    def footprint_width(self) -> float:
        return self.contact_width


ToolModel = SprayTool | ContactTool
