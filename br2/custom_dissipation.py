__doc__ = """
Custom modules for damper implementations
"""

from elastica.dissipation import DamperBase

from numba import njit

import numpy as np


class AnalyticalLinearDamperV2(DamperBase):
    """
    Analytical damper with single-coefficient #354
    """

    def __init__(self, damping_constant, time_step, **kwargs):
        super().__init__(**kwargs)
        # Compute the damping coefficient for translational velocity
        nodal_mass = self._system.mass

        # Compute the damping coefficient for exponential velocity
        if self._system.ring_rod_flag:
            element_mass = nodal_mass
        else:
            element_mass = 0.5 * (nodal_mass[1:] + nodal_mass[:-1])
            element_mass[0] += 0.5 * nodal_mass[0]
            element_mass[-1] += 0.5 * nodal_mass[-1]

        self.translational_damping_coefficient = np.exp(-damping_constant * time_step)
        self.rotational_damping_coefficient = np.exp(
            -damping_constant
            * time_step
            #* element_mass
            #* np.diagonal(self._system.inv_mass_second_moment_of_inertia).T
        )

    def dampen_rates(self, rod, time: float):
        nb_dampen_rates(rod.velocity_collection, rod.omega_collection, self.translational_damping_coefficient, rod.dilatation)


@njit(cache=True)
def nb_dampen_rates(v, w, k, e):
    """
    v - velocity_collection : numpy.ndarray (3, n_node)
    w - omega_collection : numpy.ndarray (3, n_node-1)
    k - damping_coefficient : float
    e - dilatation : numpy.ndarray (n_node-1)
    """

    n_node = v.shape[1]
    for i in range(n_node):
        v[:, i] = v[:, i] * k

    for i in range(n_node - 1):
        w[:, i] = w[:, i] * k * e[i]


