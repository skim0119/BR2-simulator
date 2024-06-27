__doc__ = """
Base System
-----------

Basic coordinating for multiple, smaller systems that have an independently integrable
interface (i.e. works with symplectic or explicit routines `timestepper.py`.)
"""
from typing import Iterable, Callable, AnyStr
import elastica as ea

from br2.modules.memory_block import (
    construct_memory_block_structures,
)
from br2.rod.cosserat_rod import FreeCosseratRod


class BaseSystemCollection(ea.BaseSystemCollection):

    def finalize(self):
        """
        This method finalizes the simulator class. When it is called, it is assumed that the user has appended
        all rod-like objects to the simulator as well as all boundary conditions, callbacks, etc.,
        acting on these rod-like objects. After the finalize method called,
        the user cannot add new features to the simulator class.
        """

        assert not self._finalize_flag, "The finalize cannot be called twice."
        self._finalize_flag = True

        # construct memory block
        self.__final_blocks = construct_memory_block_structures(self.systems())
        self.__systems.extend(self.__final_blocks)

        # Recurrent call finalize functions for all components.
        for finalize in self._feature_group_finalize:
            finalize()

        # Clear the finalize feature group, just for the safety.
        self._feature_group_finalize.clear()
        del self._feature_group_finalize
