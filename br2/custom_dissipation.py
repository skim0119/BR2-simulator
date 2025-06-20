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
            -(damping_constant*5)
            * time_step
            # * element_mass
            # * np.diagonal(self._system.inv_mass_second_moment_of_inertia).T
        )

    def dampen_rates(self, rod, time):
        nb_dampen_rates(
            rod.velocity_collection,
            rod.omega_collection,
            self.translational_damping_coefficient,
            self.rotational_damping_coefficient,
            rod.dilatation,
        )


@njit(cache=True)
def nb_dampen_rates(v, w, k, kr, e):
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
        w[:, i] = w[:, i] * kr ** e[i]


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

        self.position_filter_term = np.zeros_like(self._system.position_collection)
        self.velocity_filter_term = np.zeros_like(self._system.velocity_collection)
        self.omega_filter_term = np.zeros_like(self._system.omega_collection)
        self.kappa_filter_term = np.zeros_like(self._system.kappa)
        self.sigma_filter_term = np.zeros_like(self._system.sigma)
        self.director_filter_term = np.zeros_like(self._system.director_collection)

    def dampen_rates(self, rod, time) -> None:

        nb_filter_rate(
            rod.position_collection,
            self.position_filter_term,
            self.filter_order,
        )

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
        nb_filter_rate(
            rod.kappa,
            self.kappa_filter_term,
            self.filter_order,
        )
        nb_filter_rate(
            rod.sigma,
            self.sigma_filter_term,
            self.filter_order,
        )
        
        # Filter SO3 rotation matrices (not working)
        # nb_filter_rate_so3(
        #     rod.director_collection,
        #     self.director_filter_term,
        #     #self.filter_order,
        #     3,
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
       models modified using discrete filters: a simplified "regularized variational
       multiscale model" and an "enhanced field model". Physics of fluids, 19(5), 055110.
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
        # filter_term[..., 0] = filter_term[..., 1]
        # filter_term[..., -1] = filter_term[..., -2]
        # filter_term[..., 0] = -(-2.0 * filter_term[..., 0] + 5 * filter_term[..., 1] - 4 * filter_term[..., 2] + filter_term[..., 3]) / 4.0
        # filter_term[..., -1] = -(-2.0 * filter_term[..., -1] + 5 * filter_term[..., -2] - 4 * filter_term[..., -3] + filter_term[..., -4]) / 4.0
    rate_collection[...] = rate_collection - filter_term


@njit(cache=True)
def nb_filter_rate_so3(
    director_collection: np.ndarray, filter_term: np.ndarray, filter_order: int
) -> None:
    """
    Filters SO3 rotation matrices using quaternion representation

    Parameters
    ----------
    director_collection : numpy.ndarray
        3D array of shape (3, 3, n) containing SO3 rotation matrices.
    filter_term: numpy.ndarray
        3D array of shape (3, 3, n) for storing filter terms.
    filter_order : int
        Filter order, which corresponds to the number of times the Laplacian
        operator is applied.

    Notes
    -----
    This function filters SO3 rotation matrices by:
    1. Converting rotation matrices to quaternions
    2. Applying the Laplacian filter to the quaternion components
    3. Converting back to rotation matrices
    
    The filtering preserves the SO3 structure by working with quaternions.
    """
    
    n_elements = director_collection.shape[2]
    
    # Convert rotation matrices to quaternions
    quaternions = np.zeros((4, n_elements))
    for k in range(n_elements):
        # Extract rotation matrix for current element
        R = director_collection[:, :, k].T
        
        # Branchless quaternion conversion using sign handling
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        
        # Compute all squared components
        qw_sq = 0.25 * (1.0 + trace)
        qx_sq = 0.25 * (1.0 + 2.0 * R[0, 0] - trace)
        qy_sq = 0.25 * (1.0 + 2.0 * R[1, 1] - trace)
        qz_sq = 0.25 * (1.0 + 2.0 * R[2, 2] - trace)
        
        # Use the largest component for numerical stability
        # Compute all possible quaternions
        qw = np.sqrt(qw_sq + 1e-10)
        qx_from_w = (R[2, 1] - R[1, 2]) / (4.0 * qw)
        qy_from_w = (R[0, 2] - R[2, 0]) / (4.0 * qw)
        qz_from_w = (R[1, 0] - R[0, 1]) / (4.0 * qw)
        
        qx = np.sqrt(qx_sq + 1e-10)
        qw_from_x = (R[2, 1] - R[1, 2]) / (4.0 * qx)
        qy_from_x = (R[0, 1] + R[1, 0]) / (4.0 * qx)
        qz_from_x = (R[0, 2] + R[2, 0]) / (4.0 * qx)
        
        qy = np.sqrt(qy_sq + 1e-10)
        qw_from_y = (R[0, 2] - R[2, 0]) / (4.0 * qy)
        qx_from_y = (R[0, 1] + R[1, 0]) / (4.0 * qy)
        qz_from_y = (R[1, 2] + R[2, 1]) / (4.0 * qy)
        
        qz = np.sqrt(qz_sq + 1e-10)
        qw_from_z = (R[1, 0] - R[0, 1]) / (4.0 * qz)
        qx_from_z = (R[0, 2] + R[2, 0]) / (4.0 * qz)
        qy_from_z = (R[1, 2] + R[2, 1]) / (4.0 * qz)
        
        # Branchless selection using mathematical operations
        # Use the component with the largest squared value
        max_sq = max(qw_sq, qx_sq, qy_sq, qz_sq)
        
        # Create masks for selection (1 if largest, 0 otherwise)
        w_mask = (max_sq == qw_sq)
        x_mask = (max_sq == qx_sq)
        y_mask = (max_sq == qy_sq)
        z_mask = (max_sq == qz_sq)
        
        # Combine using masks (only one will be non-zero)
        quaternions[0, k] = w_mask * qw + x_mask * qw_from_x + y_mask * qw_from_y + z_mask * qw_from_z
        quaternions[1, k] = w_mask * qx_from_w + x_mask * qx + y_mask * qx_from_y + z_mask * qx_from_z
        quaternions[2, k] = w_mask * qy_from_w + x_mask * qy_from_x + y_mask * qy + z_mask * qy_from_z
        quaternions[3, k] = w_mask * qz_from_w + x_mask * qz_from_x + y_mask * qz_from_y + z_mask * qz
    
    # Apply Laplacian filter to quaternion components
    filter_term_quaternions = np.zeros_like(quaternions)
    pfilter_term_quaternions = np.zeros_like(quaternions)
    filter_term_quaternions[...] = quaternions
    pfilter_term_quaternions[...] = quaternions
    
    for i in range(filter_order):
        filter_term_quaternions[:, 1:-1] = (
            -pfilter_term_quaternions[:, 2:] - pfilter_term_quaternions[:, :-2] + 2.0 * pfilter_term_quaternions[:, 1:-1]
        ) / 4.0
        
        # Handle boundary conditions for quaternions
        # Option 1: Extrapolation (extend the interior values)
        filter_term_quaternions[:, 0] = -(-2.0 * pfilter_term_quaternions[:, 0] + 5 * pfilter_term_quaternions[:, 1] - 4 * pfilter_term_quaternions[:, 2] + pfilter_term_quaternions[:, 3]) / 4.0
        filter_term_quaternions[:, -1] = -(-2.0 * pfilter_term_quaternions[:, -1] + 5 * pfilter_term_quaternions[:, -2] - 4 * pfilter_term_quaternions[:, -3] + pfilter_term_quaternions[:, -4]) / 4.0
        
        # Option 2: Zero gradient (reflect interior values)
        # filter_term_quaternions[:, 0] = filter_term_quaternions[:, 1]
        # filter_term_quaternions[:, -1] = filter_term_quaternions[:, -2]
        # filter_term_quaternions[:, 0] = 0.0
        # filter_term_quaternions[:, -1] = 0.0

        pfilter_term_quaternions[...] = filter_term_quaternions
    
    # Compute filtered quaternions
    filtered_quaternions = quaternions - filter_term_quaternions
    
    # Normalize quaternions to ensure unit length
    for k in range(n_elements):
        norm = np.sqrt(filtered_quaternions[0, k]**2 + filtered_quaternions[1, k]**2 + 
                      filtered_quaternions[2, k]**2 + filtered_quaternions[3, k]**2)
        if norm > 1e-12:
            filtered_quaternions[:, k] /= norm
        else:
            continue
        
        # Extract quaternion components
        qw = filtered_quaternions[0, k]
        qx = filtered_quaternions[1, k]
        qy = filtered_quaternions[2, k]
        qz = filtered_quaternions[3, k]
        
        # Convert quaternion to rotation matrix
        R_filtered = np.array([
            [1.0 - 2.0*qy*qy - 2.0*qz*qz, 2.0*qx*qy - 2.0*qz*qw, 2.0*qx*qz + 2.0*qy*qw],
            [2.0*qx*qy + 2.0*qz*qw, 1.0 - 2.0*qx*qx - 2.0*qz*qz, 2.0*qy*qz - 2.0*qx*qw],
            [2.0*qx*qz - 2.0*qy*qw, 2.0*qy*qz + 2.0*qx*qw, 1.0 - 2.0*qx*qx - 2.0*qy*qy]
        ])
        
        # Store filtered rotation matrix
        director_collection[:, :, k] = R_filtered.T
