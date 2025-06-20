import numpy as np
from elastica._rotations import _get_rotation_matrix
from elastica.external_forces import NoForces

import numba
from numba import njit

class PivotSpring(NoForces):
    """
    This boundary condition class fixes one end of the rod. Currently,
    this boundary condition fixes position and directors
    at the first node and first element of the rod.
        Attributes
        ----------
        fixed_positions : numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
        fixed_directors : numpy.ndarray
            3D (dim, dim, 1) array containing data with 'float' type.
    """

    def __init__(self, R:float, k:float, order:int, loc:np.ndarray, pivot_idx:int, time_interval:tuple[float,float], **kwargs):
        super().__init__(**kwargs)
        self.R = R
        self.k = k
        self.loc = loc
        self.pivot_idx = pivot_idx  # Element
        self.time_interval = time_interval
        self.order = order

    def apply_forces(self, system, time):
        if self.time_interval[1] < time or time < self.time_interval[0]:
            return
        self.compute(
            system.position_collection,
            system.radius,
            system.director_collection,
            system.external_forces,
            system.external_torques,
            self.R,
            self.k,
            self.loc,
            self.pivot_idx,
            self.order
        )

    def apply_torques(self, system, time):
        pass
        # self.compute_constrain_rates(rod.velocity_collection, rod.omega_collection)

    @staticmethod
    @njit(cache=True)
    def compute(
        positions, 
        radius,
        directors,
        external_forces,
        external_torques,
        R,
        k,
        loc,
        pivot_idx,
        order,
    ):
        elem_p = 0.5 * (positions[:, pivot_idx] + positions[:, pivot_idx+1])
        arm = -directors[2, :, pivot_idx] * radius[pivot_idx]
        rp = elem_p + arm
        v = loc - rp
        r = np.linalg.norm(v)
        if r > R or r < 1e-8:
            return
        v_norm = v / r
        force = k * r * (1 - ((r**2)/(R**2)))**order
        torque = np.cross(arm, force * v_norm)

        external_forces[:, pivot_idx] += 0.5 * force * v_norm
        external_forces[:, pivot_idx+1] += 0.5 * force * v_norm
        external_torques[:, pivot_idx] += directors[:, :, pivot_idx] @ torque

    # @staticmethod
    # @njit(cache=True)
    # def compute_constrain_rates(velocity_collection, omega_collection):
    #     velocity_collection[:] = 0.0
    #     omega_collection[:] = 0.0

