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
)
from elastica.external_forces import GravityForces


from elastica._calculus import _clip_array
from elastica._linalg import _batch_cross, _batch_dot, _batch_norm, _batch_matvec
from elastica._linalg import _batch_product_i_k_to_ik
from elastica.restart import save_state, load_state
#from elastica.experimental.connection_contact_joint.parallel_connection import (
from br2.surface_connection import (
    SurfaceJointSideBySide,
    get_connection_vector_straight_straight_rod,
)

from br2.legacy_surface_connection_parallel_rod_numba import TipToTipStraightJoint

from br2.free_custom_systems import (
    FreeBendActuation,
    FreeTwistActuation,
    FreeBaseEndSoftFixed,
)
from br2.custom_dissipation import AnalyticalLinearDamperV2
from br2.custom_modules import MemoryBlockConnections


# Set base elastica simulator class
class BR2Simulator(
    BaseSystemCollection,
    Constraints,
    #Connections,
    MemoryBlockConnections,
    Forcing,
    Damping,
    CallBacks
):
    pass


class FreeCallback(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict, time_interval=None):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.time_interval = time_interval
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every != 0:
            return
        if self.time_interval is not None and (
            time < self.time_interval[0] or time > self.time_interval[1]
        ):
            return
        self.callback_params["time"].append(time)
        self.callback_params["step"].append(current_step)
        self.callback_params["position"].append(system.position_collection.copy())
        self.callback_params["velocity"].append(system.velocity_collection.copy())
        self.callback_params["acceleration"].append(
            system.acceleration_collection.copy()
        )
        self.callback_params["omega"].append(system.omega_collection.copy())
        self.callback_params["director"].append(system.director_collection.copy())
        self.callback_params["external_forces"].append(system.external_forces.copy())
        self.callback_params["external_torques"].append(system.external_torques.copy())
        self.callback_params["internal_forces"].append(system.internal_forces.copy())
        self.callback_params["internal_torques"].append(system.internal_torques.copy())
        self.callback_params["lengths"].append(system.lengths.copy())
        self.callback_params["dilatation"].append(system.dilatation.copy())
        self.callback_params["radius"].append(system.radius.copy())
        self.callback_params["com"].append(system.compute_position_center_of_mass())
        # self.callback_params["vcom"].append(system.compute_velocity_center_of_mass())


class FreeAssembly:
    def __init__(self, env, gravity=False, **kwargs):
        self.env = env

        self.simulator = BR2Simulator()
        self.actuation = defaultdict(list)
        self.free = {}  # Key: <segment name>_<order>_<rod name>

        # Parameter
        self.toggle_gravity = gravity
        
        # Scale by modulus
        self.k_multiplier = kwargs.get("k_multiplier", 0.166) * 1e6
        self.nu_multiplier = kwargs.get("nu_multiplier", 0) * 1e6
        self.kt_multiplier = kwargs.get("kt_multiplier", 1)
        self.k_repulsive_multiplier = kwargs.get("k_repulsive", 2) * 1e6
        # DEBUG
        print(f"  {self.k_multiplier=} {self.nu_multiplier=} {self.kt_multiplier=} {self.k_repulsive_multiplier=}")

    def save_state(self, **kwargs):
        # kwargs: directory, time, verbose
        save_state(self.simulator, **kwargs)

    def load_state(self, **kwargs):
        # kwargs: directory, verbose
        load_state(self.simulator, **kwargs)

    def build(self, rod_info, connect_info, verbose=True, debug=False):
        """
        if not verbose:
            print = lambda *x: False  # Bypass the print call
        """

        """import rods"""
        with open(rod_info) as json_data_file:
            rod_info = json.load(json_data_file)
        rod_specs = rod_info["Rods"]
        default_rod_spec = rod_info["DefaultParams"]
        for k in rod_specs.keys():  # Update default parameter if doesn't existe
            rod_spec = rod_specs[k]
            rod_spec.update(
                {
                    dk: default_rod_spec[dk]
                    for dk in default_rod_spec.keys()
                    if dk not in rod_spec
                }
            )

        """import segments and activations. construct rod_name"""
        with open(connect_info) as json_data_file:
            connect_spec = json.load(json_data_file)
            segments = connect_spec["Segments"]
            activations = connect_spec["Activations"]
        self.num_activation = len(activations)

        """create segment"""
        self.num_segment = len(segments)
        start_y_position = 0.0
        prev_seg_rods = None
        for seg_idx, seg_name in enumerate(segments):  # create segment rods

            """create rods"""
            seg = segments[seg_name]
            seg_rods = []
            seg_lengths, seg_n_elements = [], []  # for assertion
            for rod_i in range(len(seg["rod_order"])):
                rod_type = seg["rod_order"][rod_i]
                base_position = seg["base_position"][rod_i]
                y_rotation = seg["y-rotation"][rod_i]
                rod_name = "%s_%d_%s" % (seg_name, rod_i, rod_type)

                rod_spec = rod_specs[rod_type].copy()

                # TODO: DEBUG nu
                # rod_spec['nu'] = 10**lognu

                rod_spec["direction"] = np.array(rod_spec["direction"])
                rod_spec["normal"] = np.array(rod_spec["normal"])
                seg_lengths.append(rod_spec["base_length"])
                seg_n_elements.append(rod_spec["n_elements"])
                rod_pos = np.array(base_position)
                longitudinal_index = np.argmax(rod_spec["direction"])
                rod_pos = np.insert(rod_pos, longitudinal_index, start_y_position)
                rod_spec["start"] = rod_pos
                rod_spec["is_first_segment"] = True if seg_idx == 0 else False
                rod_spec["base_radius"] = rod_spec[
                    "outer_radius"
                ]  # TODO:  - rod_spec['inner_radius']
                # rod_spec['base_radius'] = np.sqrt(
                #                            (rod_i_spec['outer_radius'] ** 2 * np.pi - rod_i_spec['inner_radius'] ** 2 * np.pi) / np.pi)
                if "gamma" in rod_spec:
                    rod_spec["gamma"] = [
                        angle + y_rotation for angle in rod_spec["gamma"]
                    ]

                """link activation"""
                actuation_name = None
                for _actuation_name, rod_list in activations.items():
                    for data in rod_list:
                        if data[0] == seg_name and data[1] == rod_i:
                            actuation_name = _actuation_name
                    if actuation_name is not None:
                        break

                rod = self.add_free(rod_name, actuation_name, **rod_spec)

                self.free[rod_name] = rod
                seg_rods.append(rod_name)
            assert seg_lengths.count(seg_lengths[0]) == len(
                seg_lengths
            ), "rods' lengths should all be the same within a segment"
            assert seg_n_elements.count(seg_n_elements[0]) == len(
                seg_n_elements
            ), "rods' number of elements should all be the same within a segment"
            start_y_position += seg_lengths[0]

            """Parallel Connection"""
            if len(seg_rods) > 1:
                print("connecting in parallel...")
                outer_radius = rod_spec["outer_radius"]
                base_length = rod_spec["base_length"]
                E = rod_spec["youngs_modulus"]
                n_elem = rod_spec["n_elements"]
                nu_connection = base_length / n_elem * self.nu_multiplier  # TODO: fix this
                for rod_i in range(len(seg_rods)):
                    first_rod_name = seg_rods[rod_i - 1]
                    second_rod_name = seg_rods[rod_i]
                    print(
                        f"    connecting seg {seg_idx}: {first_rod_name} || {second_rod_name}"
                    )
                    self.add_parallel_connection(
                        first_rod_name,
                        second_rod_name,
                        nu=nu_connection,
                    )

            if seg_idx > 0:
                """Serial Connection"""
                print("connecting in serial...")
                outer_radius = rod_spec["outer_radius"]
                base_length = rod_spec["base_length"]
                E = rod_spec["youngs_modulus"]
                n_elem = rod_spec["n_elements"]
                k_connection = (
                    np.pi * outer_radius * E / n_elem * self.k_multiplier
                )  # 50  # 1e5
                nu_connection = base_length / n_elem * self.nu_multiplier
                kt_connection = outer_radius / 2 * self.kt_multiplier  # 1e-3
                print("  connecting seg-%d and seg-%d" % (seg_idx, seg_idx + 1))
                # print(f"  {k_connection=} {nu_connection=} {kt_connection=}")
                print(f"  previous segment rods: {prev_seg_rods}")
                print(f"  current segment rods: {seg_rods}")
                self.add_serial_connection(
                    prev_seg_rods,
                    seg_rods,
                    k=k_connection,
                    nu=nu_connection,
                    kt=kt_connection,
                )

            prev_seg_rods = seg_rods.copy()

        # Debug Mode
        if debug:
            # Plot base of the first segment
            # TODO: check
            import matplotlib.pyplot as plt

            plt.plot(
                self.free["seg1_0_R2"].position_collection[0, 0],
                self.free["seg1_0_R2"].position_collection[2, 0],
                "rx",
            )
            plt.plot(
                self.free["seg1_1_R4"].position_collection[0, 0],
                self.free["seg1_1_R4"].position_collection[2, 0],
                "bx",
            )
            plt.plot(
                self.free["seg1_2_R2"].position_collection[0, 0],
                self.free["seg1_2_R2"].position_collection[2, 0],
                "gx",
            )
            plt.show(block=False)

        self.shearable_rods = self.free

        return self.shearable_rods

    def generate_callbacks(self, step_skip, time_interval=None):
        data_rods = []
        for rod_name in self.free.keys():
            data_rods.append(self.add_callback(rod_name, step_skip, time_interval))
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
            actuation_ref.append(0.0)
            return actuation_ref
        else:
            return None

    def create_rod(self, name, is_first_segment=True, verbose=False, **rod_spec):
        # Create new rod
        rod = CosseratRod.straight_rod(**rod_spec)
        rod.outer_radius = rod_spec["outer_radius"]
        rod.inner_radius = rod_spec["inner_radius"]
        self.free[name] = rod

        # Append rod to simulator
        self.simulator.append(rod)

        if getattr(rod_spec, "hollow", False):
            outer_radius = rod.outer_radius
            inner_radius = rod.inner_radius
            hollow_scale_bend = (
                (2 * outer_radius) ** 4 - (2 * inner_radius) ** 4
            ) / ((2 * outer_radius) ** 4)
            hollow_scale_shear = (
                (2 * outer_radius) ** 2 - (2 * inner_radius) ** 2
            ) / ((2 * outer_radius) ** 2)
            rod.bend_matrix = hollow_scale_bend * rod.bend_matrix
            rod.shear_matrix = hollow_scale_shear * rod.shear_matrix

        # add damping
        if "damping_constant" in rod_spec:
            damping_constant = rod_spec["damping_constant"]
            self.simulator.dampen(rod).using(
                AnalyticalLinearDamperV2,
                damping_constant=damping_constant,
                time_step=self.env.time_step,
            )
            self.simulator.dampen(rod).using(
                LaplaceDissipationFilter,
                filter_order=5,
            )

        # Constrain one end of the rod (TODO : Modify for serial connection)
        if is_first_segment:
            self.simulator.constrain(rod).using(
                FreeBaseEndSoftFixed,
                constrained_position_idx=(0,),
                constrained_director_idx=(0,),
                k=1e9,
                nu=0,
                kt=0.0,
            )

        # Gravity
        if self.toggle_gravity:
            self.simulator.add_forcing_to(rod).using(
                GravityForces,
                acc_gravity=np.array([0.0, 9.80665, 0.0]),  # Reverse direction
            )

        return rod

    def add_angled_fibers(self, rod, actuation_ref, fiber_angles: list):
        for alpha in fiber_angles:
            angle = alpha * np.pi / 180
            scale = (
                np.pi
                * (rod.inner_radius ** 3)
                * ((np.sin(angle) ** 2) + 2 * (np.cos(angle) ** 2))
                / (np.sin(2 * angle))
            )
            self.simulator.add_forcing_to(rod).using(
                FreeTwistActuation, actuation_ref, scale=scale
            )

    def add_straight_fibers(self, rod, actuation_ref, fiber_angles: list):
        scale = np.pi * rod.radius[-1] * (rod.inner_radius ** 2)
        for gamma in fiber_angles:
            self.simulator.add_forcing_to(rod).using(
                FreeBendActuation,
                actuation_ref,
                z_angle=gamma * np.pi / 180.0,
                scale=scale,
            )

    def add_free(self, name, actuation_name, alpha=[], beta=[], gamma=[], **rod_spec):
        # Create rod
        rod = self.create_rod(name, **rod_spec)
        actuation_ref = self.get_actuation_reference(actuation_name)

        # Add fiber
        if actuation_ref is not None:
            self.add_angled_fibers(rod, actuation_ref, alpha)
            self.add_angled_fibers(rod, actuation_ref, beta)
            self.add_straight_fibers(rod, actuation_ref, gamma)

        return rod

    def add_parallel_connection(self, name1, name2, **param):
        rod1 = self.free[name1]
        rod2 = self.free[name2]
        self.glue_rods_surface_connection(rod1, rod2, **param)

    def add_serial_connection(self, rod_list1: list, rod_list2: list, **param):
        # Connect tip of rod_list1 to base of rod_list2
        for rod1_name in rod_list1:
            for rod2_name in rod_list2:
                rod1 = self.free[rod1_name]
                rod2 = self.free[rod2_name]
                self.tip_to_base_connection(rod1, rod2, **param)

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

    def glue_rods_surface_connection(self, rod_one, rod_two, nu):
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

        n_elems = rod_one.n_elems

        for i in range(n_elems):
            k_conn = (
                rod_one.radius[i] * rod_two.radius[i]
                / (rod_one.radius[i] + rod_two.radius[i])
                * rod_one.lengths[i] / (rod_one.radius[i] + rod_two.radius[i])
            )

            self.simulator.connect(
                first_rod=rod_one,
                second_rod=rod_two,
                first_connect_idx=i,
                second_connect_idx=i,
            ).using(
                SurfaceJointSideBySide,
                k=k_conn*self.k_multiplier,
                nu=nu,
                k_repulsive=k_conn*self.k_repulsive_multiplier,
                rod_one_direction_vec_in_material_frame=rod_one_direction_vec_in_material_frame[
                    ..., i
                ],
                rod_two_direction_vec_in_material_frame=rod_two_direction_vec_in_material_frame[
                    ..., i
                ],
                offset_btw_rods=offset_btw_rods[i],
            )

    def tip_to_base_connection(self, rod_one, rod_two, k, nu, kt):
        rod1_elem_pos = 0.5 * (
            rod_one.position_collection[..., -1] + rod_one.position_collection[..., -2]
        )
        rod2_elem_pos = 0.5 * (
            rod_two.position_collection[..., 0] + rod_two.position_collection[..., 1]
        )
        rod1_Q = rod_one.director_collection[..., -1]
        rod2_Q = rod_two.director_collection[..., 0]
        distance = np.linalg.norm(rod2_elem_pos - rod1_elem_pos)
        connection_lab = (rod2_elem_pos - rod1_elem_pos) / distance
        rod1_rd2_local = rod1_Q @ connection_lab  # local frame
        rod2_rd2_local = rod2_Q @ (-connection_lab)  # local frame

        self.simulator.connect(
            first_rod=rod_one, second_rod=rod_two, first_connect_idx=-1, second_connect_idx=0
        ).using(
            TipToTipStraightJoint,
            k=k,
            nu=nu,
            kt=kt,
            rod1_rd2_local=rod1_rd2_local,
            rod2_rd2_local=rod2_rd2_local,
            stability_check=False,
        )
