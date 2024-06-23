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
        w[:, i] = w[:, i] * k ** e[i]


class LaplaceDissipationFilterV2(DamperBase):
    def __init__(self, filter_order: int, **kwargs):
        """
        Filter damper initializer

        Parameters
        ----------
        filter_order : int, even number
            Filter order, which corresponds to the number of times the Laplacian
            operator is applied. Increasing `filter_order` implies higher-order/weaker
            filtering.
        """
        super().__init__(**kwargs)
        if not (filter_order > 0 and isinstance(filter_order, int)):
            raise ValueError(
                "Invalid filter order! Filter order must be a positive integer."
            )
        self.filter_order = filter_order

        self.velocity_filter_term = np.zeros_like(self._system.velocity_collection)
        self.omega_filter_term = np.zeros_like(self._system.omega_collection)
        #self.acceleration_filter_term = np.zeros_like(self._system.acceleration_collection)
        #self.alpha_filter_term = np.zeros_like(self._system.alpha_collection)
        #self.kappa_filter_term = np.zeros_like(self._system.kappa)
        #self.sigma_filter_term = np.zeros_like(self._system.sigma)

    def dampen_rates(self, rod, time: float) -> None:

        nb_filter_rate(
            rod.velocity_collection,
            self.velocity_filter_term,
            self.filter_order,
        )
        nb_filter_rate(
            rod.omega_collection,
            self.omega_filter_term,
            self.filter_order,
        )
        # nb_filter_rate(
        #     rod.acceleration_collection,
        #     self.acceleration_filter_term,
        #     self.filter_order,
        # )
        # nb_filter_rate(
        #     rod.alpha_collection,
        #     self.alpha_filter_term,
        #     self.filter_order,
        # )
        # nb_filter_rate(
        #     rod.kappa,
        #     self.kappa_filter_term,
        #     self.filter_order,
        # )
        # nb_filter_rate(
        #     rod.sigma,
        #     self.sigma_filter_term,
        #     self.filter_order,
        # )


@njit(cache=True)
def nb_filter_rate(
    rate_collection: np.ndarray, filter_term: np.ndarray, filter_order: int
) -> None:
    """
    Filters the rod rates (velocities) in numba njit decorator

    Parameters
    ----------
    rate_collection : numpy.ndarray
        2D array containing data with 'float' type.
        Array containing rod rates (velocities).
    filter_term: numpy.ndarray
        2D array containing data with 'float' type.
        Filter term that modifies rod rates (velocities).
    filter_order : int
        Filter order, which corresponds to the number of times the Laplacian
        operator is applied. Increasing `filter_order` implies higher order/weaker
        filtering.

    Notes
    -----
    For details regarding the numerics behind the filtering, refer to:

    .. [1] Jeanmart, H., & Winckelmans, G. (2007). Investigation of eddy-viscosity
       models modified using discrete filters: a simplified “regularized variational
       multiscale model” and an “enhanced field model”. Physics of fluids, 19(5), 055110.
    .. [2] Lorieul, G. (2018). Development and validation of a 2D Vortex Particle-Mesh
       method for incompressible multiphase flows (Doctoral dissertation,
       Université Catholique de Louvain).
    """

    filter_term[...] = rate_collection
    for i in range(filter_order):
        filter_term[..., 1:-1] = (
            -filter_term[..., 2:] - filter_term[..., :-2] + 2.0 * filter_term[..., 1:-1]
        ) / 4.0
        # dont touch boundary values
        filter_term[..., 0] = 0.0
        filter_term[..., -1] = 0.0
    rate_collection[...] = rate_collection - filter_term
