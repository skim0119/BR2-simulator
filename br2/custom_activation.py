import numpy as np
import numba
from numba import njit
from elastica._linalg import _batch_product_i_k_to_ik
from elastica._linalg import _batch_matvec
from elastica._external_forces import NoForces


class PointForces(NoForces):
    """
    This class applies constant forces on the endpoint nodes.

        Attributes
        ----------
        start_force: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type. Force applied to first node of the rod-like object.
        end_force: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type. Force applied to last node of the rod-like object.
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.

    """

    def __init__(self, force, location, ramp_up_time=1e-3):
        """

        Parameters
        ----------
        start_force: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
            Force applied to first node of the rod-like object.
        end_force: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
            Force applied to last node of the rod-like object.
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.

        """
        super().__init__()
        self.force = force
        assert ramp_up_time >= 0.0
        self.ramp_up_time = ramp_up_time
        assert location < 1.0 and location > 0.0
        self.location = location

    def apply_forces(self, system, time=0.0):
        force_index = int(system.external_forces.shape[-1]*self.location)
        self.compute_end_point_forces(
            system.external_forces,
            self.force,
            force_index,
            time,
            self.ramp_up_time,
        )

    @staticmethod
    @njit(cache=True)
    def compute_end_point_forces(
        external_forces, force, force_index, time, ramp_up_time
    ):
        """
        Compute end point forces that are applied on the rod using numba njit decorator.

        Parameters
        ----------
        external_forces: numpy.ndarray
            2D (dim, blocksize) array containing data with 'float' type. External force vector.
        force: numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
        force_index: int
            index of the rod
        time: float
        ramp_up_time: float
            Applied forces are ramped up until ramp up time.

        Returns
        -------

        """
        factor = min(1.0, time / ramp_up_time)
        external_forces[..., force_index] += force * factor
