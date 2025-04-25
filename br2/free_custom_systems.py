from elastica import *
from elastica import NoForces

from elastica.utils import Tolerance

from elastica._linalg import _batch_cross, _batch_dot
from elastica._linalg import _batch_product_i_k_to_ik
from elastica.boundary_conditions import ConstraintBase
from elastica.external_forces import EndpointForces
from elastica.contact_forces import RodRodContact

from br2.linalg import (
    _single_inv_rotate,
)

import math
import numpy as np
import numba
from numba import njit
import matplotlib.pyplot as plt


class TipLoad(EndpointForces):
    def apply_forces(self, system, time):
        if time > 28.3:
            super().apply_forces(system, time)


# Defining Variable Bending
class FreeBendActuation(NoForces):

    def __init__(self, actuation_ref, z_angle, scale, ramp_up_time=1.0, gamma_tilt=0.0):
        super(FreeBendActuation, self).__init__()
        self.actuation_ref = actuation_ref
        self.z_angle = z_angle
        self.gamma_tilt = gamma_tilt
        self.scale = scale
        self.ramp_up_time = ramp_up_time

    def apply_torques(self, system, time=0.0):
        factor = min(1.0, time / self.ramp_up_time)
        pressure = self.actuation_ref() * self.scale * factor
        # torque = pressure * self.local_unit_vector / system.n_elems
        # torque_on_each_element = (
        #    _batch_product_i_k_to_ik(torque, np.ones((system.n_elems)))
        # )
        # system.external_torques += torque_on_each_element
        # return

        self.nb_apply_moment(
            self.z_angle,
            self.gamma_tilt,
            system.lengths,
            rod_external_forces=system.external_forces,
            rod_external_torques=system.external_torques,
            pressure=pressure,
            radius=system.radius,
            alpha_angle=system.alpha_angle,
            beta_angle=system.beta_angle,
            # alpha_angle=system.initial_alpha_angle,
            # beta_angle=system.initial_beta_angle,
            tangents=system.tangents,
            n_elems=system.n_elems,
            skip_element_pre=0,
            skip_element_post=0,
        )

    @staticmethod
    @njit(cache=True)
    def nb_apply_moment(
        z_angle,
        gamma_tilt,
        lengths,
        rod_external_forces,
        rod_external_torques,
        pressure,
        radius,
        alpha_angle,
        beta_angle,
        tangents,
        n_elems,
        skip_element_pre,
        skip_element_post,
    ):
        Sa = np.sin(alpha_angle)
        Sb = np.sin(beta_angle)
        Ca = np.cos(alpha_angle)
        Cb = np.cos(beta_angle)
        Sa_b = np.sin(alpha_angle - beta_angle)
        pSaSb = Sa * Sb + 2 * Ca * Cb  # Pitch * sin(a) * sin(b)

        # Compute axial force per elements
        F_scale = (pSaSb * Sa * Sb * Sa_b**2) / (
            (Sa * Sb * Sa_b) ** 2 + (Sa**2 - Sb**2) ** 2
        )
        force_on_one_element = pressure * np.pi * radius**2 * F_scale / n_elems

        # Compute moment
        _some_scale_linear_actuation_to_moment = 1.0
        _some_scale_linear_actuation = 1.0

        # print(f"alpha: {np.rad2deg(alpha_angle.min())}-{np.rad2deg(alpha_angle.max())}, beta: {np.rad2deg(beta_angle.min())}-{np.rad2deg(beta_angle.max())}")

        # Angular actuation
        ff = 15
        fi = 15
        for i in range(skip_element_pre, n_elems - skip_element_post - 1):
            _gamma_tilt_z = gamma_tilt * min(1.0, (-i + 60) / fi, i / ff)
            moment_arm = np.array(
                [
                    np.cos(z_angle),
                    np.sin(z_angle),
                    _gamma_tilt_z,
                ]
            )
            moment_arm /= np.linalg.norm(moment_arm)
            rod_external_torques[0, i] -= (
                force_on_one_element[i + 1]
                * moment_arm[0]
                * _some_scale_linear_actuation_to_moment
            )
            rod_external_torques[1, i] -= (
                force_on_one_element[i + 1]
                * moment_arm[1]
                * _some_scale_linear_actuation_to_moment
            )
            rod_external_torques[2, i] -= (
                force_on_one_element[i + 1]
                * moment_arm[2]
                * _some_scale_linear_actuation_to_moment
            )
        for i in range(skip_element_pre + 1, n_elems - skip_element_post):
            _gamma_tilt_z = gamma_tilt * min(1.0, (-i + 60) / fi, i / ff)
            moment_arm = np.array(
                [
                    np.cos(z_angle),
                    np.sin(z_angle),
                    _gamma_tilt_z,
                ]
            )
            moment_arm /= np.linalg.norm(moment_arm)
            rod_external_torques[0, i] += (
                force_on_one_element[i - 1]
                * moment_arm[0]
                * _some_scale_linear_actuation_to_moment
            )
            rod_external_torques[1, i] += (
                force_on_one_element[i - 1]
                * moment_arm[1]
                * _some_scale_linear_actuation_to_moment
            )
            rod_external_torques[2, i] += (
                force_on_one_element[i - 1]
                * moment_arm[2]
                * _some_scale_linear_actuation_to_moment
            )

        # Linear Actuation
        _force_on_one_element = force_on_one_element * _some_scale_linear_actuation
        rod_external_forces[..., skip_element_pre] -= (
            _force_on_one_element[skip_element_pre] * tangents[..., skip_element_pre]
        )
        rod_external_forces[..., n_elems - skip_element_post] += (
            _force_on_one_element[n_elems - skip_element_post - 1]
            * tangents[..., n_elems - skip_element_post - 1]
        )
        for i in range(skip_element_pre + 1, n_elems - skip_element_post):
            rod_external_forces[..., i] += (
                _force_on_one_element[i - 1] * tangents[..., i - 1]
            )
            rod_external_forces[..., i] -= _force_on_one_element[i] * tangents[..., i]


class FreeBaseEndSoftFixed(ConstraintBase):
    def __init__(
        self,
        fixed_position1,
        fixed_position2,
        fixed_position3,
        fixed_position4,
        fixed_position5,
        fixed_directors1,
        fixed_directors2,
        fixed_directors3,
        fixed_directors4,
        fixed_directors5,
        k,
        nu,
        kt,
        **kwargs,
    ):
        """
        Parameters
        ----------
        fixed_position : numpy.ndarray
            2D (dim, 1) array containing data with 'float' type.
        fixed_directors : numpy.ndarray
            3D (dim, dim, 1) array containing data with 'float' type.
        """
        super().__init__(**kwargs)
        self.fixed_position = np.stack(
            [
                fixed_position1,
                fixed_position2,
                fixed_position3,
                fixed_position4,
                fixed_position5,
            ],
            axis=-1,
        )
        self.fixed_directors = np.stack(
            [
                fixed_directors1,
                fixed_directors2,
                fixed_directors3,
                fixed_directors4,
                fixed_directors5,
            ],
            axis=-1,
        )
        self.k = k
        self.nu = nu
        self.kt = kt

        self.fixing_slice = 5

        # Accumulated rotation
        self.rev = 0

    def constrain_values(self, system, time):
        system.position_collection[..., : self.fixing_slice] = self.fixed_position
        system.director_collection[..., : self.fixing_slice] = self.fixed_directors
        return
        self.restrict_position(
            system.position_collection[..., 0],
            self.fixed_position,
            system.velocity_collection[..., 0],
            system.external_forces,
            self.k,
            self.nu,
        )
        self.rev = self.restrict_orientation(
            self.kt,
            system.director_collection[..., 0],
            self.fixed_directors,
            system.external_torques,
            self.rev,
        )

    def constrain_rates(self, system, time):
        super().constrain_rates(system, time)
        system.velocity_collection[..., : self.fixing_slice] = 0.0
        system.omega_collection[..., : self.fixing_slice] = 0.0
        system.acceleration_collection[..., : self.fixing_slice] = 0.0
        system.alpha_collection[..., : self.fixing_slice] = 0.0

        self.constrain_shear(system.sigma, system.kappa, self.fixing_slice)

    @staticmethod
    @njit(cache=True)
    def constrain_shear(sigma, kappa, fixing_slice):
        pass
        # sigma[..., :fixing_slice] = 0.0
        # kappa[..., :fixing_slice] = 0.0

    @staticmethod
    @njit(cache=True)
    def restrict_position(
        base_position,
        fixed_position,
        base_velocity,
        system_external_force,
        k,
        nu,  # damping coefficient
    ):
        end_distance_vector = fixed_position - base_position

        # Calculate norm of end_distance_vector
        # this implementation timed: 2.48 µs ± 126 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        end_distance = np.sqrt(np.dot(end_distance_vector, end_distance_vector))

        # Below if check is not efficient find something else
        # We are checking if end of system1 and start of system2 are at the same point in space
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
        system_external_force[..., 0] += contact_force

    @staticmethod
    @njit(cache=True)
    def restrict_orientation(
        kt,  # damping coefficient
        base_directors,
        fixed_directors,
        system_external_torques,
        rev,
    ):
        return 0
        BAt = np.eye(3)  # Initial misalignment

        # Alignment Torque
        Tp = (fixed_directors @ BAt) @ base_directors.T

        omega = _single_inv_rotate(Tp) / 2.0
        theta = np.linalg.norm(omega)
        if theta <= 1e-8:
            normalized_omega = np.array([0.0, 0.0, 0.0])
        else:
            normalized_omega = omega / theta
        new_theta = theta + 2 * np.pi * rev
        torque_on_system = normalized_omega * kt * theta  # new_theta

        # Change coordinate
        torque_on_system_material_frame = base_directors @ torque_on_system

        # Add torque
        system_external_torques[..., 0] += torque_on_system_material_frame

        # return 0  # np.fix(new_theta / (2*np.pi))


class FreeCombinedActuation(NoForces):
    """
    Based on B. Joshua 2013
    """

    def __init__(self, actuation_ref, scale, ramp_up_time=1.0):
        """

        Parameters
        ----------
        actuation_ref: list
            Pressure
        scale: float
        ramp_up_time: float
        """
        self.actuation_ref = actuation_ref
        self.scale = scale
        self.ramp_up_time = ramp_up_time

        self.direction = np.array([0.0, 0.0, 1.0])  # system always in tangent

        self._counter = 0
        self._time = []
        self._force = []
        self._moment = []
        fig = plt.figure()
        self.ax = fig.add_subplot(111)
        self.pl1 = self.ax.plot(self._force, label="Force")[0]
        self.pl2 = self.ax.plot(self._moment, label="Moment")[0]
        plt.legend()

    def apply_forces(self, system: "FreeCosseratRod", time):
        factor = min(1.0, time / self.ramp_up_time)
        pressure = self.actuation_ref() * self.scale * factor
        if math.isclose(
            pressure, 0.0
        ):  # This is to save some unnecessary activation with small pressure
            a = np.zeros(system.n_elems, dtype=np.float_)
            b = np.zeros(system.n_elems, dtype=np.float_)
        else:
            a, b = self.nb_apply_forces_and_torques(
                rod_external_forces=system.external_forces,
                rod_external_torques=system.external_torques,
                pressure=pressure,
                radius=system.radius,
                alpha_angle=system.alpha_angle,
                beta_angle=system.beta_angle,
                # alpha_angle=system.initial_alpha_angle,
                # beta_angle=system.initial_beta_angle,
                tangents=system.tangents,
                n_elems=system.n_elems,
                skip_element_pre=0,
                skip_element_post=0,
            )
        if self._counter % 10000 == 0:
            self._counter = 0
            # self._time.append(time)
            # self._force.append(a.max())
            # self._moment.append(b.max())
            # self.pl1.set_xdata(self._time)
            # self.pl1.set_ydata(self._force)
            # self.pl2.set_xdata(self._time)
            # self.pl2.set_ydata(self._moment)
            # self.ax.relim()
            # self.ax.autoscale_view()
            # plt.show(block=False)
            # plt.pause(0.01)

            # print(f"{time=:.2f} | Pressure: {pressure:.02e} || Force: {np.abs(a).max():.02f}, Torque: {np.abs(b).max():.02f}, maxDelta={system.delta_turn.max():.02f}, minDelta={system.delta_turn.min():.02f}, maxDil={system.dilatation.max():.02f}, minDil={system.dilatation.min():.02f}")
            # print(f"delta: {system.delta_turn=}")
            # print(f"delta: {system.alpha_angle=}")
            # print(f"delta: {system.beta_angle=}")
            # print(system.compute_twist())
        self._counter += 1

    @staticmethod
    @njit(cache=True)
    def nb_apply_forces_and_torques(
        rod_external_forces,
        rod_external_torques,
        pressure,
        radius,
        alpha_angle,
        beta_angle,
        tangents,
        n_elems,
        skip_element_pre,
        skip_element_post,
    ):
        # if alpha or beta is nan, breakpoint
        # if np.isnan(rod_external_forces).any() or np.isnan(beta_angle).any():
        #    breakpoint()
        Sa = np.sin(alpha_angle)
        Sb = np.sin(beta_angle)
        Ca = np.cos(alpha_angle)
        Cb = np.cos(beta_angle)
        Sa_b = np.sin(alpha_angle - beta_angle)
        pSaSb = Sa * Sb + 2 * Ca * Cb  # Pitch * sin(a) * sin(b)

        # Compute axial force per elements
        F_scale = -(
            (pSaSb * Sa * Sb * Sa_b**2) / ((Sa * Sb * Sa_b) ** 2 + (Sa**2 - Sb**2) ** 2)
        )
        force_on_one_element = pressure * np.pi * radius**2 * F_scale / n_elems

        # Compute moment
        M_scale = -(pSaSb * Sa_b * (Sa**2 - Sb**2)) / (
            (Sa * Sb * Sa_b) ** 2 + (Sa**2 - Sb**2) ** 2
        )
        torque_on_one_element = pressure * np.pi * radius**3 * M_scale / n_elems

        # Add on angular actuation
        for i in range(skip_element_pre, n_elems - skip_element_post - 1):
            rod_external_torques[2, i] -= torque_on_one_element[i + 1]
        for i in range(skip_element_pre + 1, n_elems - skip_element_post):
            rod_external_torques[2, i] += torque_on_one_element[i - 1]

        # Add on linear actuation
        rod_external_forces[..., skip_element_pre] -= (
            force_on_one_element[skip_element_pre] * tangents[..., skip_element_pre]
        )
        rod_external_forces[..., n_elems - skip_element_post] += (
            force_on_one_element[n_elems - skip_element_post - 1]
            * tangents[..., n_elems - skip_element_post - 1]
        )
        for i in range(skip_element_pre + 1, n_elems - skip_element_post):
            rod_external_forces[..., i] += (
                force_on_one_element[i - 1] * tangents[..., i - 1]
            )
            rod_external_forces[..., i] -= force_on_one_element[i] * tangents[..., i]

        # for i in range(skip_element_pre, n_elems - skip_element_post):
        #     rod_external_forces[..., i] += (
        #         force_on_one_element[i] * tangents[..., i]
        #     ) * 0.5
        #     rod_external_forces[..., i + 1] += (
        #         force_on_one_element[i] * tangents[..., i]
        #     ) * 0.5

        # rod_external_forces[..., 0] += 0.5 * force_on_one_element[0] * tangents[..., 0]
        # rod_external_forces[..., -1] += (
        #     0.5 * force_on_one_element[-1] * tangents[..., -1]
        # )
        return force_on_one_element, torque_on_one_element

    def apply_torques(self, system: "FreeCosseratRod", time):
        # Force and torque included together
        pass


class RodRodContactInterval(RodRodContact):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def apply_contact(self, system_one, system_two):
        pass
