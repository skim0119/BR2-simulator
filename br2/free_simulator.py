import os
import copy
import numpy as np
from itertools import combinations
import json

from elastica import *
from elastica._calculus import _isnan_check
from elastica.timestepper import extend_stepper_interface
from elastica.external_forces import (
    UniformTorques,
    # EndPointTorques,
    GravityForces,
)


from elastica._calculus import _clip_array
from elastica._linalg import _batch_cross, _batch_dot, _batch_norm, _batch_matvec
from elastica._linalg import _batch_product_i_k_to_ik
from elastica.restart import save_state, load_state

# from elastica.experimental.connection_contact_joint.parallel_connection import (
from br2.rod.cosserat_rod import FreeCosseratRod
from br2.surface_connection import (
    SurfaceJointSideBySide,
    get_connection_vector_straight_straight_rod,
    get_connection_directors_straight_straight_rod,
)

from br2.free_custom_systems import (
    TipLoad,
    FreeBendActuation,
    FreeBaseEndSoftFixed,
    FreeCombinedActuation,
)
from br2.custom_callback import FreeCallback
from br2.custom_dissipation import AnalyticalLinearDamperV2, LaplaceDissipationFilterV2
from br2.modules.base_system import BaseSystemCollection as CustomBaseSystemCollection
from br2.modules.base_system import BaseSystemCollection as CustomBaseSystemCollection
from br2.modules.block_connections import MemoryBlockConnections


# Set base elastica simulator class
class BR2Simulator(
    CustomBaseSystemCollection,
    Constraints,
    # Connections,
    MemoryBlockConnections,
    Forcing,
    Contact,
    Damping,
    CallBacks,
):
    pass


class FreeAssembly:
    def __init__(self, env, gravity=False, REMOVE_CONNECTION:bool=False, **kwargs):
        self.env = env

        self.simulator = BR2Simulator()
        self.actuation = defaultdict(lambda: [0.0])
        self.free = {}  # Key: <segment name>_<order>_<rod name>

        # Parameter
        self.toggle_gravity: bool = gravity
        self.num_activation: int = 0
        self.num_segment: int = 0

        # Scale by modulus
        self.k_multiplier = kwargs.get("k_multiplier", 0.166) * 1e6
        self.nu_multiplier = kwargs.get("nu_multiplier", 0)  # Scale from 0 to 1
        self.k_torsion_multiplier = kwargs.get("k_torsion_multiplier", 5e3)
        self.k_torsion_multiplier_serial = kwargs.get("k_torsion_multiplier_serial", 5e3)
        self.k_repulsive_multiplier = kwargs.get("k_repulsive", 2) * 1e6
        # DEBUG
        # print(
        #     f"  {self.k_multiplier=} {self.nu_multiplier=} {self.kt_multiplier=} {self.k_repulsive_multiplier=}"
        # )

        # DEBUG_CONFIGURATION
        self.REMOVE_CONNECTION = REMOVE_CONNECTION

    def save_state(self, **kwargs):
        # kwargs: directory, time, verbose
        save_state(self.simulator, **kwargs)

    def load_state(self, **kwargs):
        # kwargs: directory, verbose
        load_state(self.simulator, **kwargs)

    def build(
        self,
        rod_info,
        connect_info,
        verbose: bool = True,
        debug: bool = False,
        prepend_tag: str = "",
    ):
        """
        if not verbose:
            print = lambda *x: False  # Bypass the print call
        """

        # import rod configuration
        with open(rod_info, "r") as json_data_file:
            rod_config = json.load(json_data_file)
        rod_specs = rod_config["Rods"]
        default_rod_spec = rod_config["DefaultParams"]
        for k in rod_specs.keys():  # Update default parameter if doesn't existe
            rod_spec = rod_specs[k]
            rod_spec.update(
                {
                    dk: default_rod_spec[dk]
                    for dk in default_rod_spec.keys()
                    if dk not in rod_spec
                }
            )

        # import segments and activations. construct rod_name
        with open(connect_info) as json_data_file:
            connect_spec = json.load(json_data_file)
            segments = connect_spec["Segments"]
            activations = connect_spec["Activations"]
        self.num_activation += len(activations)

        # create segment
        self.num_segment += len(segments)
        start_y_position = 0.0
        prev_seg_rods = None
        frees = {}
        for seg_idx, seg_name in enumerate(segments):  # create segment rods

            # create rods
            seg = segments[seg_name]
            seg_rods = []
            seg_lengths, seg_n_elements = [], []  # for assertion
            for rod_i in range(len(seg["rod_order"])):
                rod_type = seg["rod_order"][rod_i]
                base_position = seg["base_position"][rod_i]
                y_rotation = seg["y-rotation"][rod_i]
                rod_name = "%s_%d_%s" % (seg_name, rod_i, rod_type)
                if prepend_tag:
                    rod_name = f"{prepend_tag}_{rod_name}"

                rod_spec = rod_specs[rod_type].copy()

                rod_spec["direction"] = np.array(rod_spec["direction"])
                rod_spec["normal"] = np.array(rod_spec["normal"])
                seg_lengths.append(rod_spec["base_length"])
                seg_n_elements.append(rod_spec["n_elements"])
                rod_pos = np.array(base_position)
                longitudinal_index = np.argmax(np.abs(rod_spec["direction"]))
                sign = np.sign(rod_spec["direction"][longitudinal_index])
                rod_pos = np.insert(
                    rod_pos, longitudinal_index, sign * start_y_position
                )
                rod_spec["start"] = rod_pos
                rod_spec["is_first_segment"] = True if seg_idx == 0 else False
                rod_spec["base_radius"] = rod_spec["outer_radius"]

                # Set fiber angles
                if "gamma" in rod_spec:
                    rod_spec["gamma"] = rod_spec["gamma"] + y_rotation
                else:
                    rod_spec["gamma"] = None
                if "alpha" in rod_spec:
                    rod_spec["alpha_fiber_angle"] = rod_spec["alpha"]
                if "beta" in rod_spec:
                    rod_spec["beta_fiber_angle"] = rod_spec["beta"]

                # link activation
                actuation_name = None
                for _actuation_name, rod_list in activations.items():
                    for data in rod_list:
                        if data[0] == seg_name and data[1] == rod_i:
                            actuation_name = _actuation_name
                    if actuation_name is not None:
                        break
                if prepend_tag:
                    actuation_name = f"{prepend_tag}_{actuation_name}"

                rod = self.add_free(
                    rod_name, actuation_name, verbose=verbose, **rod_spec
                )

                frees[rod_name] = rod
                seg_rods.append(rod_name)
            assert seg_lengths.count(seg_lengths[0]) == len(
                seg_lengths
            ), "rods' lengths should all be the same within a segment"
            assert seg_n_elements.count(seg_n_elements[0]) == len(
                seg_n_elements
            ), "rods' number of elements should all be the same within a segment"
            start_y_position += seg_lengths[0]

            """Parallel Connection"""
            if not self.REMOVE_CONNECTION and len(seg_rods) > 1:
                if verbose:
                    print("connecting in parallel...")
                for rod_i in range(len(seg_rods)):
                    first_rod_name = seg_rods[rod_i - 1]
                    second_rod_name = seg_rods[rod_i]
                    if verbose:
                        print(
                            f"    connecting seg {seg_idx}: {first_rod_name} || {second_rod_name}"
                        )
                    self.add_parallel_connection(
                        first_rod_name,
                        second_rod_name,
                    )

            if not self.REMOVE_CONNECTION and seg_idx > 0:
                """Serial Connection"""
                if verbose:
                    print("connecting in serial...")
                    print("  connecting seg-%d and seg-%d" % (seg_idx, seg_idx + 1))
                    print(f"  previous segment rods: {prev_seg_rods}")
                    print(f"  current segment rods: {seg_rods}")
                self.add_serial_connection(
                    prev_seg_rods,
                    seg_rods,
                )

                # FIXME : Temporary
                # self.add_tip_load(seg_rods)

            prev_seg_rods = seg_rods.copy()

        self.free.update(frees)

        return frees

    def add_tip_load(self, rods):
        weight = 0.027 / len(rods)
        for rod_name in rods:
            rod = self.free[rod_name]
            self.simulator.add_forcing_to(rod).using(
                TipLoad,
                start_force=np.zeros(3),
                end_force=np.array([0, 0, -1]) * weight,
                ramp_up_time=1.0,
            )

    def generate_callbacks(self, step_skip, time_interval=None, callback_class=None, **kwargs):
        data_rods = {}
        for rod_name in self.free.keys():
            data_rods[rod_name] = self.add_callback(
                rod_name,
                step_skip,
                time_interval=time_interval,
                callback=callback_class,
                **kwargs,
            )
        return data_rods

    def set_actuation(self, actuation: dict):
        """
        Set actuation for each rods
        """
        for k, v in actuation.items():
            self.actuation[k][0] = v

    def get_actuation_reference(self, actuation_name=None):
        if actuation_name is not None:
            actuation_ref = self.actuation[actuation_name]
            # actuation_ref.append(0.0)  # Remove?
            return actuation_ref
        else:
            return None

    def create_rod(self, name, is_first_segment=True, verbose=False, **rod_spec):
        if rod_spec.get("prebuilt", False):
            smoothing_path = rod_spec["smoothing_path"]
            smoothing_slice = rod_spec["smoothing_slice"]
            data = np.load(smoothing_path)
            positions = data["position"][
                0, :, int(smoothing_slice[0]) : int(smoothing_slice[1]) + 1
            ]
            directors = data["director"][
                0, :, :, int(smoothing_slice[0]) : int(smoothing_slice[1])
            ]
            rod = FreeCosseratRod.straight_rod(
                **rod_spec, position=positions, directors=directors
            )
            rod.compute_internal_forces_and_torques(0)
            rod = FreeCosseratRod.straight_rod(
                **rod_spec,
                position=positions,
                directors=directors,
                rest_sigma=rod.sigma,
                rest_kappa=rod.kappa,
            )
        else:
            # Create new rod
            rod = FreeCosseratRod.straight_rod(**rod_spec)
        rod.outer_radius = rod_spec["outer_radius"]
        rod.inner_radius = rod_spec["inner_radius"]
        self.free[name] = rod

        # Append rod to simulator
        self.simulator.append(rod)

        if rod_spec.get("hollow", True):  # By default, hollow
            outer_radius = rod.outer_radius
            inner_radius = rod.inner_radius
            hollow_scale_bend = ((2 * outer_radius) ** 4 - (2 * inner_radius) ** 4) / (
                (2 * outer_radius) ** 4
            )
            hollow_scale_shear = ((2 * outer_radius) ** 2 - (2 * inner_radius) ** 2) / (
                (2 * outer_radius) ** 2
            )
            rod.bend_matrix[:2, :2, :] *= hollow_scale_bend
            rod.shear_matrix[:2, :2, :] *= hollow_scale_shear

            rod.mass[:] = 0.0
            rod.mass[:-1] += 0.5 * (
                rod.density
                * np.pi
                * (outer_radius**2 - inner_radius**2)
                * rod.rest_lengths
            )
            rod.mass[1:] += 0.5 * (
                rod.density
                * np.pi
                * (outer_radius**2 - inner_radius**2)
                * rod.rest_lengths
            )

        # add damping
        if "damping_constant" in rod_spec:
            damping_constant = rod_spec["damping_constant"]
            self.simulator.dampen(rod).using(
                AnalyticalLinearDamperV2,
                damping_constant=damping_constant,
                time_step=self.env.time_step,
            )
            self.simulator.dampen(rod).using(
                LaplaceDissipationFilterV2,
                filter_order=3,
            )

        # Constrain one end of the rod (TODO : Modify for serial connection)
        if True and is_first_segment:
            self.simulator.constrain(rod).using(
                FreeBaseEndSoftFixed,
                constrained_position_idx=(0, 1, 2, 3, 4),
                constrained_director_idx=(0, 1, 2, 3, 4),
                k=1e9,
                nu=0,
                kt=0.0,
            )

        # Gravity
        if True and self.toggle_gravity:
            self.simulator.add_forcing_to(rod).using(
                GravityForces,
                # acc_gravity=np.array([0.0, 9.80665, 0.0]),  # Reverse direction
                acc_gravity=np.array([0.0, 0.0, -9.80665]),  # Reverse direction
            )

        return rod

    def add_free(
        self,
        name,
        actuation_name,
        alpha: float,
        beta: float,
        gamma: float | None = None,
        verbose: bool = True,
        **rod_spec,
    ):
        # Create rod
        rod = self.create_rod(name, verbose=verbose, **rod_spec)
        actuation_ref = self.get_actuation_reference(actuation_name)

        # Add fiber
        if actuation_ref is not None:
            ramp_up_time = rod_spec.get("ramp_up_time", 1.0)
            if gamma is not None:
                scale = rod_spec.get("moment_scale", 1.0)
                gamma_tilt = rod_spec.get("gamma_tilt", 0.0)
                self.simulator.add_forcing_to(rod).using(
                    FreeBendActuation,
                    actuation_ref,
                    z_angle=gamma * np.pi / 180.0,
                    scale=scale,
                    gamma_tilt=gamma_tilt,
                    ramp_up_time=ramp_up_time,
                )
            else:
                scale = rod_spec.get("twist_scale", 1.0)
                self.simulator.add_forcing_to(rod).using(
                    FreeCombinedActuation,
                    actuation_ref,
                    scale=scale,
                    ramp_up_time=ramp_up_time,
                )

        return rod

    def add_serial_connection(self, rod_list1: list, rod_list2: list):
        # Connect tip of rod_list1 to base of rod_list2
        for rod1_name in rod_list1:
            for rod2_name in rod_list2:
                rod1 = self.free[rod1_name]
                rod2 = self.free[rod2_name]
                self.tip_to_base_connection(rod1, rod2)

    def add_callback(self, name, step_skip, callback=None, **kwargs):
        rod = self.free[name]
        # list which collected data will be append
        callback_params = defaultdict(list)
        if callback is None:
            callback = FreeCallback
        self.simulator.collect_diagnostics(rod).using(
            callback, step_skip=step_skip, callback_params=callback_params, **kwargs
        )
        return callback_params

    def add_parallel_connection(self, rod_one_name, rod_two_name):
        rod_one = self.free[rod_one_name]
        rod_two = self.free[rod_two_name]
        (
            rod_one_direction_vec_in_material_frame,
            rod_two_direction_vec_in_material_frame,
            offset_btw_rods,
        ) = get_connection_vector_straight_straight_rod(
            rod_one=rod_one,
            rod_two=rod_two,
            rod_one_idx=(0, rod_one.n_elems),
            rod_two_idx=(0, rod_two.n_elems),
        )
        offset_rotation_btw_rods = get_connection_directors_straight_straight_rod(
            rod_one=rod_one,
            rod_two=rod_two,
            rod_one_idx=(0, rod_one.n_elems),
            rod_two_idx=(0, rod_two.n_elems),
        )

        n_elems = rod_one.n_elems

        for i in range(n_elems):
            k_conn = (
                rod_one.radius[i]
                * rod_two.radius[i]
                / (rod_one.radius[i] + rod_two.radius[i])
                * rod_one.lengths[i]
                / (rod_one.radius[i] + rod_two.radius[i])
            )

            self.simulator.connect(
                first_rod=rod_one,
                second_rod=rod_two,
                first_connect_idx=i,
                second_connect_idx=i,
            ).using(
                SurfaceJointSideBySide,
                k=k_conn * self.k_multiplier,
                nu=self.nu_multiplier,
                k_repulsive=k_conn * self.k_repulsive_multiplier,
                k_torsion=k_conn * self.k_torsion_multiplier,
                rod_one_direction_vec_in_material_frame=rod_one_direction_vec_in_material_frame[
                    ..., i
                ],
                rod_two_direction_vec_in_material_frame=rod_two_direction_vec_in_material_frame[
                    ..., i
                ],
                offset_btw_rods=offset_btw_rods[i],
                offset_rotation_btw_rods=offset_rotation_btw_rods[..., i],
            )

    def tip_to_base_connection(self, rod_one, rod_two):
        (
            rod_one_direction_vec_in_material_frame,
            rod_two_direction_vec_in_material_frame,
            offset_btw_rods,
        ) = get_connection_vector_straight_straight_rod(
            rod_one=rod_one,
            rod_two=rod_two,
            rod_one_idx=(rod_one.n_elems - 1, rod_one.n_elems),
            rod_two_idx=(0, 1),
        )
        offset_rotation_btw_rods = get_connection_directors_straight_straight_rod(
            rod_one=rod_one,
            rod_two=rod_two,
            rod_one_idx=(rod_one.n_elems - 1, rod_one.n_elems),
            rod_two_idx=(0, 1),
        )
        rod_one_direction_vec_in_material_frame = (
            rod_one_direction_vec_in_material_frame[..., 0]
        )
        rod_two_direction_vec_in_material_frame = (
            rod_two_direction_vec_in_material_frame[..., 0]
        )
        offset_btw_rods = offset_btw_rods[0]
        offset_rotation_btw_rods = offset_rotation_btw_rods[..., 0]
        rod_one_idx = rod_one.n_elems - 1
        rod_two_idx = 0

        k_conn = (
            rod_one.radius[rod_one_idx]
            * rod_two.radius[rod_two_idx]
            / (rod_one.radius[rod_one_idx] + rod_two.radius[rod_two_idx])
            * rod_one.lengths[rod_one_idx]
            / (rod_one.radius[rod_one_idx] + rod_two.radius[rod_two_idx])
        )

        print(f"  connecting tip to base: {rod_one} || {rod_two}")
        print(f"    {k_conn=}, {self.k_multiplier=}, {self.nu_multiplier=}")
        print(f"    {self.k_torsion_multiplier=}")

        self.simulator.connect(
            first_rod=rod_one,
            second_rod=rod_two,
            first_connect_idx=rod_one_idx,
            second_connect_idx=rod_two_idx,
        ).using(
            SurfaceJointSideBySide,
            k=k_conn * self.k_multiplier * 1e1,
            nu=self.nu_multiplier + 1e-4,
            k_repulsive=0.0,
            k_torsion=k_conn * self.k_torsion_multiplier_serial,
            rod_one_direction_vec_in_material_frame=rod_one_direction_vec_in_material_frame,
            rod_two_direction_vec_in_material_frame=rod_two_direction_vec_in_material_frame,
            offset_btw_rods=offset_btw_rods,
            offset_rotation_btw_rods=offset_rotation_btw_rods,
        )
