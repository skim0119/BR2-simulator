from elastica import *

from elastica.utils import Tolerance

from elastica._calculus import _clip_array
from elastica._linalg import _batch_cross, _batch_dot, _batch_norm, _batch_matvec
from elastica._linalg import _batch_product_i_k_to_ik
from elastica.boundary_conditions import ConstraintBase

from br2.surface_connection_parallel_rod_numba import _single_inv_rotate, _single_get_rotation_matrix

import numpy as np
import numba
from numba import njit

# Defining Variable Bending
class FreeBendActuation(NoForces):
    # TODO

    def __init__(self, actuation_ref, z_angle, scale, ramp_up_time=1.0):
        super(FreeBendActuation, self).__init__()
        self.actuation_ref = actuation_ref
        self.z_angle = z_angle
        self.magnitude_scale = scale # TODO
        '''
        Currently we are using scale to externally compute the moment scale.
        We are assuming the radius is not changing as it deform.
        If we want to incorporate the change in radius, we cannot use default
        CR radius, since the basic CR is implemented assuming solid cylinder
        '''
        self.ramp_up_time = ramp_up_time

    ''' (deprecated)
    def apply_forces(self, system, time: np.float = 0.0):
        factor = min(1.0, time / self.ramp_up_time)
        force_on_one_element = self.actuation_ref[0] * factor
        tangents = system.tangents[...]
        system.external_forces[..., 0] -= factor * 0.5 * force_on_one_element * tangents[..., 0]
        system.external_forces[..., -1] += factor * 0.5 * force_on_one_element * tangents[..., -1]
        system.external_forces[..., 1:-1] += (
            factor * 0.5 * force_on_one_element
            * (tangents[..., :-1] - tangents[..., 1:])
        )
    '''

    def apply_torques(self, system, time: np.float = 0.0):
        factor = min(1.0, time / self.ramp_up_time) # Not sure
        torque_mag = self.actuation_ref[0] * self.magnitude_scale # * factor
        local_unit_vector = np.array([np.cos(self.z_angle),
                                      np.sin(self.z_angle),
                                      0.0
                                      ])
        torque = torque_mag * local_unit_vector
        system.external_torques[...,-1] += torque
        #system.external_torques[...,-1] += system.director_collection[...,-1] @ torque
        ''' Not used: (uniformly distributed torque) '''
        return
        n_elems = system.n_elems
        torque_on_one_element = (
            _batch_product_i_k_to_ik(torque, np.ones((n_elems)))
        )
        system.external_torques += _batch_matvec(
            system.director_collection, torque_on_one_element
        )


# Defining Variable Torque
class FreeTwistActuation(NoForces):
    # TODO

    def __init__(self, actuation_ref, scale, ramp_up_time=1.0):
        """

        Parameters
        ----------
        actuation_ref: list
        scale: float
        """
        super(FreeTwistActuation, self).__init__()
        self.actuation_ref = actuation_ref
        self.direction = np.array([0.0, 0.0, 1.0]) # Rod always in tangent
        self.scale = scale
        self.ramp_up_time = ramp_up_time

    '''
    def apply_forces(self, system, time: np.float = 0.0):
        factor = min(1.0, time / self.ramp_up_time)
        force_on_one_element = self.torque[0] / 3.14742e-2 * factor # TODO: double check the ratio
        tangents = system.tangents[...]
        system.external_forces[..., 0] -= factor * 0.5 * force_on_one_element * tangents[..., 0]
        system.external_forces[..., -1] += factor * 0.5 * force_on_one_element * tangents[..., -1]
        system.external_forces[..., 1:-1] += (
            factor * 0.5 * force_on_one_element
            * (tangents[..., :-1] - tangents[..., 1:])
        )
    '''

    def apply_torques(self, system, time: np.float = 0.0):
        #factor = min(1.0, time / self.ramp_up_time)
        torque = self.actuation_ref[0] * self.scale * self.direction #* factor
        n_elems = system.n_elems
        torques = (
            _batch_product_i_k_to_ik(torque, np.ones((n_elems))) / n_elems
        )
        system.external_torques += torques
        '''
        system.external_torques += _batch_matvec(
            system.director_collection, torque_on_one_element
        )
        '''

class FreeBaseEndSoftFixed(ConstraintBase):

    def __init__(self, fixed_position, fixed_directors, k, nu, kt, **kwargs):
        """
        Parameters
        ----------
        fixed_position : numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
        fixed_directors : numpy.ndarray
            3D (dim, dim, 1) array containing data with 'float' type.
        """
        super().__init__(**kwargs)
        self.fixed_position = fixed_position # Initial position
        self.fixed_directors = fixed_directors # Initial directors
        self.k = k
        self.nu = nu
        self.kt = kt

        # Accumulated rotation
        self.rev = 0

    def constrain_values(self, rod, time):
        super().constrain_values(rod, time)
        #rod.position_collection[...,0] = self.fixed_position
        self.restrict_position(
            rod.position_collection[...,0],
            self.fixed_position,
            rod.velocity_collection[...,0],
            rod.external_forces,
            self.k,
            self.nu,
        )
        rod.director_collection[...,0] = self.fixed_directors
        return
        self.rev = self.restrict_orientation(
                self.kt,
                rod.director_collection[...,0],
                self.fixed_directors,
                rod.external_torques,
                self.rev,
            )

    def constrain_rates(self, rod, time):
        super().constrain_rates(rod, time)
        rod.velocity_collection[..., 0] = 0.0
        rod.omega_collection[..., 0] = 0.0
        rod.acceleration_collection[..., 0] = 0.0
        rod.alpha_collection[..., 0] = 0.0

    @staticmethod
    @njit(cache=True)
    def restrict_position(
            base_position, fixed_position, base_velocity, rod_external_force,
            k, nu # damping coefficient
        ):
        end_distance_vector = fixed_position - base_position

        # Calculate norm of end_distance_vector
        # this implementation timed: 2.48 µs ± 126 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        end_distance = np.sqrt(np.dot(end_distance_vector, end_distance_vector))

        # Below if check is not efficient find something else
        # We are checking if end of rod1 and start of rod2 are at the same point in space
        # If they are at the same point in space, it is a zero vector.
        if end_distance <= 1e-8:
            normalized_end_distance_vector = np.array([0.0, 0.0, 0.0])
        else:
            normalized_end_distance_vector = end_distance_vector / end_distance

        elastic_force = k * end_distance_vector

        relative_velocity = -base_velocity
        normal_relative_velocity = (
                np.dot(relative_velocity, normalized_end_distance_vector)
                * normalized_end_distance_vector
        )
        damping_force = -nu * normal_relative_velocity

        contact_force = elastic_force + damping_force
        rod_external_force[...,0] += contact_force

    @staticmethod
    @njit(cache=True)
    def restrict_orientation(
        kt, # damping coefficient
        base_directors,
        fixed_directors,
        rod_external_torques,
        rev,
    ):
        return 0
        BAt = np.eye(3) # Initial misalignment

        # Alignment Torque
        Tp = (fixed_directors @ BAt) @ base_directors.T
            
        omega = _single_inv_rotate(Tp) / 2.0
        theta = np.linalg.norm(omega)
        if theta <= 1e-8:
            normalized_omega = np.array([0.0, 0.0, 0.0])
        else:
            normalized_omega = omega / theta
        new_theta = theta + 2 * np.pi * rev
        torque_on_rod = normalized_omega * kt * theta # new_theta

        # Change coordinate
        torque_on_rod_material_frame = base_directors @ torque_on_rod

        # Add torque
        rod_external_torques[..., 0] += torque_on_rod_material_frame

        return 0 #np.fix(new_theta / (2*np.pi))

