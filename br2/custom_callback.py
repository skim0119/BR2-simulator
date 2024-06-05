__all__ = ["BlenderCallback"]

import bpy
import numpy as np
from elastica import CallBackBaseClass
from elastica.typing import RodType

import bsr
from bsr.geometry import Cylinder, Sphere


class BlenderCallback(CallBackBaseClass):
    """
    PyElastica callback to save rod state to Blender.
    """

    def __init__(self, step_skip: int, time_interval: int, callback_params=None,**kwargs) -> None:
        CallBackBaseClass.__init__(self, **kwargs)
        self.every = step_skip
        self.time_interval = time_interval
        self.key_frame = 0
        self.bpy_objs = bsr.Rod()
        self.stop = False

    def make_callback(
        self, system: RodType, time: np.floating, current_step: int
    ) -> None:
        if current_step % self.every != 0:
            return
        if self.time_interval is not None and (
            time < self.time_interval[0] or time > self.time_interval[1]
        ):
            return
        if np.isnan(system.position_collection).any():
            self.stop = True
            return
        if self.stop:
            return
        self.bpy_objs.update(
            keyframe=self.key_frame,
            positions=system.position_collection,
            radii=system.radius*2,
        )
        self.key_frame += 1
