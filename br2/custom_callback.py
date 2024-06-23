__all__ = ["BlenderRodCallback"]

import bpy
import numpy as np
from elastica import CallBackBaseClass
from elastica.typing import RodType

from bsr import Rod


class BlenderRodCallback(CallBackBaseClass):
    """
    PyElastica callback to save rod state to Blender.
    """

    def __init__(self, step_skip: int, time_interval: int, callback_params=None,**kwargs) -> None:
        CallBackBaseClass.__init__(self, **kwargs)
        self.every = step_skip
        self.time_interval = time_interval
        self.key_frame = 0
        self.bpy_objs: Rod
        self.stop = False

    def make_callback(
        self, system: RodType, time: np.floating, current_step: int
    ) -> None:
        if self.stop or current_step % self.every != 0:
            return
        if self.time_interval is not None and (
            time < self.time_interval[0] or time > self.time_interval[1]
        ):
            return
        if np.isnan(system.position_collection).any() or np.isnan(system.radius).any():
            self.stop = True
            return
        if current_step == 0:
            self.bpy_objs = Rod(
                system.position_collection,
                system.radius,
            )
        else:
            self.bpy_objs.update_states(
                positions=system.position_collection,
                radii=system.radius,
            )
        self.bpy_objs.set_keyframe(self.key_frame)
        self.key_frame += 1
