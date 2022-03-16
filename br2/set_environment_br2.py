from typing import Optional 

import os
import copy
import time

import numpy as np

from elastica._calculus import _isnan_check
from elastica import *
from elastica.timestepper import extend_stepper_interface

from post_processing import (
    plot_video_with_surface,
)

from br2.free_simulator import FreeAssembly


class Environment:
    def __init__(
        self,
        fps,
        rendering_fps:int=25,
        time_step:float=2.0e-5,
        final_time:Optional[float]=None,
        flag_collect_data:bool=False,
    ):
        # Integrator type
        self.StatefulStepper = PositionVerlet()

        # Simulation parameters
        self.final_time = final_time
        self.time_step = time_step

        # Recording speed
        self.rendering_fps = fps
        self.step_skip = int(1.0 / (self.rendering_fps * self.time_step)) # match rendering fps to physical time
        self.capture_interval = None #(0.3, 0.5)#None
        print(f'{self.step_skip=}')

        # Rod
        self.shearable_rods = {}

        # Steady State Thresholds
        self.position_threshold = 1.0e-7
        self.director_threshold = 1.0e-5
        self.velocity_threshold = 1.0e-4
        self.omega_threshold = 1.0e-3
        self.acceleration_threshold = 10**(0)
        self.alpha_threshold = 1.0e+1

        # Collect data is a boolean. If it is true callback function collects
        # rod parameters defined by user in a list.
        self.flag_collect_data = flag_collect_data

    def save_state(self, directory: str='', verbose:bool=False):
        """
        Save state parameters of each rod.

        TODO : environment list variable is not uniform at the current stage of development. 
        It would be nice if we have set list (like env.system) that iterates all the rods.

        Parameters
        ----------
        directory: str
            Directory path name. The path must exist.
        """
        self.assy.save_state(directory=directory, time=self.time, verbose=verbose)

    def load_state(self, directory: str='', clear_callback:bool=False, verbose:bool=False):
        """
        Load the rod-state.
        Compatibale with 'save_state' method.

        If the save-file does not exist, it returns error.

        Parameters
        ----------
        directory : str
            Directory path name. 
        """
        self.assy.load_state(directory=directory, verbose=verbose)

        # Clear callback
        if clear_callback:
            for callback_defaultdict in self.data_rods:
                callback_defaultdict.clear()
            if verbose:
                print('callback cleared')

    def reset(self, rod_database_path:str, assembly_config_path:str, start_time:float=0.0, **kwargs) -> None:
        """
        Creates the simulation environment.

        Parameters
        ----------
        rod_database_path : str
        assembly_config_path : str
        """

        assert os.path.exists(rod_database_path), "Rod database path does not exists."
        assert os.path.exists(assembly_config_path), "Assembly configuration does not exists."

        self.assy = FreeAssembly(**kwargs)

        '''rod name -> [seg,rod]'''
        self.shearable_rods = self.assy.build(rod_info, connect_info)
        self.simulator = self.assy.simulator

        if self.flag_collect_data:
            # Collect data using callback function for postprocessing
            # set the diagnostics for rod and collect data
            self.data_rods = self.assy.generate_callbacks(self.step_skip, time_interval=self.capture_interval) #[seg,rod]
        # Finalize simulation environment. After finalize, you cannot add
        # any forcing, constrain or call back functions
        self.simulator.finalize()

        # do_step, stages_and_updates will be used in step function
        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.StatefulStepper, self.simulator
        )
        self.time = start_time # simulation time

    def run(self, action:dict, duration:float, check_nan:bool=False, check_steady_state:bool=False) -> bool:
        # Set action
        self.assy.set_actuation(action)

        # Simulation
        time = self.time
        while time >= self.time + duration:
            time = self.do_step(
                self.StatefulStepper,
                self.stages_and_updates,
                self.simulator,
                self.time,
                self.time_step,
            )

        # Check steady state
        if check_steady_state == 1:
            # velocity test
            velocities = [np.array(self.shearable_rods[key].position_collection) for key in self.shearable_rods.keys()]
            max_velocity = max([np.linalg.norm(v, axis=0).max() for v in velocities])
            info['max_velocity'] = max_velocity
            if max_velocity < self.velocity_threshold:
                #print("Steady-state (minimum velocity), exiting simulation now")
                done = True
                info['done_by_steady_state'] = 'maximum_velocity'
        elif check_steady_state == 2:
            keys = list(self.shearable_rods.keys())
            prev_position = np.concatenate([self.shearable_rods[name].position_collection for name in keys], axis=-1)
            prev_velocity = np.concatenate([self.shearable_rods[name].velocity_collection for name in keys], axis=-1)
            prev_acceleration = np.concatenate([self.shearable_rods[name].acceleration_collection for name in keys], axis=-1)
            prev_director = np.concatenate([self.shearable_rods[name].director_collection for name in keys], axis=-1)
            prev_omega = np.concatenate([self.shearable_rods[name].omega_collection for name in keys], axis=-1)
            prev_alpha = np.concatenate([self.shearable_rods[name].alpha_collection for name in keys], axis=-1)
            # convergence
            keys = list(self.shearable_rods.keys())
            position_delta = np.nanmax(np.linalg.norm(prev_position - np.concatenate([self.shearable_rods[name].position_collection for name in keys], axis=-1), axis=0))
            velocity_delta = np.nanmax(np.linalg.norm(prev_velocity - np.concatenate([self.shearable_rods[name].velocity_collection for name in keys], axis=-1), axis=0))
            acceleration_delta = np.nanmax(np.linalg.norm(prev_acceleration - np.concatenate([self.shearable_rods[name].acceleration_collection for name in keys], axis=-1), axis=0))
            director_delta = np.nanmax(np.linalg.norm(prev_director - np.concatenate([self.shearable_rods[name].director_collection for name in keys], axis=-1), axis=(0,1)))
            omega_delta = np.nanmax(np.linalg.norm(prev_omega - np.concatenate([self.shearable_rods[name].omega_collection for name in keys], axis=-1), axis=0))
            alpha_delta = np.nanmax(np.linalg.norm(prev_alpha - np.concatenate([self.shearable_rods[name].alpha_collection for name in keys], axis=-1), axis=0))
            info['position_delta'] = position_delta
            info['velocity_delta'] = velocity_delta
            info['acceleration_delta'] = acceleration_delta
            info['director_delta'] = director_delta
            info['omega_delta'] = omega_delta
            info['alpha_delta'] = alpha_delta
            criteria = (position_delta < self.position_threshold) and \
                       (velocity_delta < self.velocity_threshold) and \
                       (acceleration_delta < self.acceleration_threshold) and \
                       (director_delta < self.director_threshold) and \
                       (omega_delta < self.omega_threshold) and \
                       (alpha_delta < self.alpha_threshold)
            if criteria:
                #print("Steady-state (delta), exiting simulation now")
                done = True
                info['done_by_steady_state'] = 'delta convergence'

        # Check NaN
        if check_nan:
            # Position of the rod cannot be NaN, it is not valid, stop the simulation
            invalid_values_conditions = [_isnan_check(self.shearable_rods[name].position_collection)
                                            for name in self.shearable_rods.keys()] + \
                                        [_isnan_check(self.shearable_rods[name].velocity_collection)
                                            for name in self.shearable_rods.keys()] + \
                                        [_isnan_check(self.shearable_rods[name].director_collection)
                                            for name in self.shearable_rods.keys()] + \
                                        [_isnan_check(self.shearable_rods[name].omega_collection)
                                            for name in self.shearable_rods.keys()]

            if any(invalid_values_conditions):
                done = True
                info['done_due_to_nan'] = True
                #print("Nan detected, exiting simulation now")

        self.time = time

    def post_processing(self, filename_video, save_folder, data_tag=0, **kwargs):
        """
        Make video 3D rod movement in time.
        Parameters
        ----------
        filename_video
        Returns
        -------

        """

        if self.flag_collect_data:
            plot_video_with_surface(
                self.data_rods,
                video_name=filename_video,
                fps=self.rendering_fps,
                step=1,
                save_folder=save_folder,
                **kwargs
            )

            position_data_path = os.path.join(save_folder, f"br2_data_{data_tag}.npz")
            self.save_data(position_data_path)

        else:
            raise RuntimeError(
                "call back function is not called anytime during simulation, "
                "change COLLECT_DATA=True"
            )

    def save_data(self, path):
        # TODO
        # TEMP
        position_rod = np.array(self.data_rods[0]["position"])
        position_rod = 0.5 * (position_rod[..., 1:] + position_rod[..., :-1])
        np.savez(
            path,
            time=np.array(self.data_rods[0]["time"]),
            position_rod=position_rod,
        )
        return
        # Transform nodal to elemental positions
        position_rod1 = np.array(self.data_rod1["position"])
        position_rod1 = 0.5 * (position_rod1[..., 1:] + position_rod1[..., :-1])

        # Transform nodal to elemental positions
        position_rod2 = np.array(self.data_rod2["position"])
        position_rod2 = 0.5 * (position_rod2[..., 1:] + position_rod2[..., :-1])

        # Transform nodal to element positions
        position_rod3 = np.array(self.data_rod3["position"])
        position_rod3 = 0.5 * (position_rod3[..., 1:] + position_rod3[..., :-1])

        # Save rod position (for povray)
        np.savez(
            path,
            time=np.array(self.data_rod1["time"]),
            position_rod1=position_rod1,
            radii_rod1=np.array(self.data_rod1["radius"]),
            director_rod1=np.array(self.data_rod1["director"]),
            position_rod2=position_rod2,
            radii_rod2=np.array(self.data_rod2["radius"]),
            director_rod2=np.array(self.data_rod2["director"]),
            position_rod3=position_rod3,
            radii_rod3=np.array(self.data_rod3["radius"]),
            director_rod3=np.array(self.data_rod3["director"]),
        )
