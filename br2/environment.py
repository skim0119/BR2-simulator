from typing import Optional, Union

import os
import copy
import time

from dataclasses import dataclass

from tqdm import tqdm

import numpy as np

from elastica import *
from elastica._calculus import _isnan_check
from elastica.timestepper import extend_stepper_interface

from br2.post_processing import plot_video_with_surface

from br2.free_simulator import FreeAssembly


@dataclass
class DataPaths:
    """
    Data path collection

        Attributes
        ----------
        tag : str
            Simulation tag that will be appended at the end of the result directory.
        paths : str
            Parent directory path
        simulation : str
            Path to save the simulation data for pause/restart purpose.
        renderings : str
            Path to save all the rendering videos.
        data : str
            Path to save data for the post-processing.

    """

    tag: str

    @property
    def paths(self) -> str:
        return f"result_{self.tag}"

    @property
    def simulation(self) -> str:
        return os.path.join(self.paths, "simulation_saves")

    @property
    def renderings(self) -> str:
        return os.path.join(self.paths, "renderings")

    @property
    def data(self) -> str:
        return os.path.join(self.paths, "data")

    def initialize(self):
        """
        Initialization step: create missing directories
        """
        os.makedirs(self.paths, exist_ok=1)
        os.makedirs(self.simulation, exist_ok=1)
        os.makedirs(self.renderings, exist_ok=1)
        os.makedirs(self.data, exist_ok=1)


@dataclass
class TerminalInfo:
    """

        Attributes
        ----------
        end_status : bool
            Indicate if simulation reached end.
        <variable>_nan_status : bool
            Indicate if NaN exists in <variable>. Only given when `check_nan` is True.
        <variable>_steady_state_status : bool
            Indicate if <variable> is in steady-state. Only given when `check_steady_state` is given.
        max_velocity : float
            Maximum velocity at the end of the run. Only given when `check_steady_state=1`.

    """

    end_status: bool = False

    def __str__(self):
        """Print all status"""
        messages = []
        for name in self.__dir__():
            if name.endswith("status"):
                messages.append(f"{name} = {getattr(self, name)}")
        return '\n'.join(messages)

    @property
    def combined_nan_status(self) -> bool:
        """
        Combined status for if NaN exists. Return true if any NaN status is true.

        Notes
        -----
        If check_nan is not given in simulation, this property does not give correct indication.
        """
        status_list = [
            getattr(self, name)
            for name in self.__dir__()
            if name.endswith("status") and "_nan_" in name and ("combined_" not in name)
        ]
        return all(status_list)

    @property
    def combined_steady_state_status(self) -> bool:
        """
        Combined status for steady state. Return true if all steady-state status is true.

        Notes
        -----
        If check_steady_state is not given in simulation, this property does not give correct indication.
        """
        status_list = [
            getattr(self, name)
            for name in self.__dir__()
            if name.endswith("status") and ("_steady_state_" in name) and ("combined_" not in name)
        ]
        return any(status_list)


class Environment:
    """

    Attributes
    ----------
    rendering_fps : int
        Rendering fps for output videos. (default=25)
    time_step : float
        Simulation timestep. Faster time-step could reduce the simulation walltime,
        but the simulation may be unstable. (default=2.0e-5)
    final_time : Optional[float]
    """

    def __init__(
        self,
        run_tag: str,
        rendering_fps: int = 25,
        time_step: float = 2.0e-5,
        final_time: Optional[float] = None,
    ):
        # Integrator type (pyelastica==0.2.2 only provide PositionVerlet)
        self.StatefulStepper = PositionVerlet()

        # Set paths
        self.paths = DataPaths(run_tag)
        self.paths.initialize()

        # Simulation parameters
        self.final_time = final_time  # TODO: either remove or implement stopper
        self.time_step = time_step

        # Recording speed
        self.rendering_fps = rendering_fps
        self.step_skip = int(
            1.0 / (self.rendering_fps * self.time_step)
        )  # match rendering fps to physical time
        self.capture_interval = None  # TODO: ex.(0.3, 0.5)

        # Rod
        self.shearable_rods = {}

        # Steady State Thresholds
        self.position_threshold = 1.0e-7
        self.director_threshold = 1.0e-5
        self.velocity_threshold = 1.0e-4
        self.omega_threshold = 1.0e-3
        self.acceleration_threshold = 10 ** (0)
        self.alpha_threshold = 1.0e1

    def reset(
        self,
        rod_database_path: str,
        assembly_config_path: str,
        start_time: float = 0.0,
        **kwargs,
    ) -> None:
        """
        Creates the simulation environment.

        Parameters
        ----------
        rod_database_path : str
        assembly_config_path : str
        start_time : float
        """

        assert os.path.exists(rod_database_path), "Rod database path does not exists."
        assert os.path.exists(
            assembly_config_path
        ), "Assembly configuration does not exists."

        self.assy = FreeAssembly(**kwargs)

        """rod name -> [seg,rod]"""
        self.shearable_rods = self.assy.build(rod_database_path, assembly_config_path)
        self.simulator = self.assy.simulator

        # Collect data using callback function for postprocessing
        # set the diagnostics for rod and collect data
        self.data_rods = self.assy.generate_callbacks(
            self.step_skip, time_interval=self.capture_interval
        )  # [seg,rod]
        # Finalize simulation environment. After finalize, you cannot add
        # any forcing, constrain or call back functions
        self.simulator.finalize()

        # do_step, stages_and_updates will be used in step function
        self.do_step, self.stages_and_updates = extend_stepper_interface(
            self.StatefulStepper, self.simulator
        )
        self.time = start_time  # simulation time

    def run(
        self,
        action: dict,
        duration: Optional[float] = None,
        disable_progress_bar: bool = False,
        check_nan: bool = False,
        check_steady_state: Optional[int] = None,
    ) -> Optional[TerminalInfo]:
        """
        Run simulation for a duration given action.

        Parameters
        ----------
        action : dict
            Action specified for each rods.
        duration : Optional[float]
            If duration is not specified, run a single step (duration=step_size)
        disable_progress_bar : bool
        check_nan : bool
            If True, check if any Nan is detected in the simulation at the end of the run. (default=False)
        check_steady_state : Optional[int]
            Check steady state condition at the end of the run. If 1, the velocity steady-
            state is checked. If 2, dynamic steady-state condition is checked.(default=None)
        """
        # Initialize status
        status = TerminalInfo()

        # Set action
        self.assy.set_actuation(action)

        # Record previous-step
        if check_steady_state == 2:
            # fmt: off
            keys = list(self.shearable_rods.keys())
            prev_position = np.concatenate([self.shearable_rods[name].position_collection for name in keys], axis=-1)
            prev_velocity = np.concatenate([self.shearable_rods[name].velocity_collection for name in keys], axis=-1)
            prev_acceleration = np.concatenate([self.shearable_rods[name].acceleration_collection for name in keys], axis=-1)
            prev_director = np.concatenate([self.shearable_rods[name].director_collection for name in keys], axis=-1)
            prev_omega = np.concatenate([self.shearable_rods[name].omega_collection for name in keys], axis=-1)
            prev_alpha = np.concatenate([self.shearable_rods[name].alpha_collection for name in keys], axis=-1)
            # fmt: on

        # Simulation
        time = self.time
        if not duration:
            duration = self.time_step
        with tqdm(
            total=duration, mininterval=0.5, disable=disable_progress_bar
        ) as pbar:
            while time < self.time + duration:
                time = self.do_step(
                    self.StatefulStepper,
                    self.stages_and_updates,
                    self.simulator,
                    self.time,
                    self.time_step,
                )
                pbar.update(self.time_step)
        self.time = time

        # Check steady state
        if check_steady_state == 1:
            # Velocity steady-state test
            velocities = [
                np.array(self.shearable_rods[key].position_collection)
                for key in self.shearable_rods.keys()
            ]
            max_velocity = max([np.linalg.norm(v, axis=0).max() for v in velocities])
            status.velocity_steady_state_status = max_velocity < self.velocity_threshold
            status.max_velocity = max_velocity
        elif check_steady_state == 2:
            # fmt: off
            # convergence
            position_delta = np.nanmax(
                np.linalg.norm(prev_position - np.concatenate(
                    [self.shearable_rods[name].position_collection for name in keys],
                    axis=-1
                ), axis=0))
            velocity_delta = np.nanmax(
                np.linalg.norm(prev_velocity - np.concatenate(
                    [self.shearable_rods[name].velocity_collection for name in keys],
                    axis=-1,
                ), axis=0))
            acceleration_delta = np.nanmax(
                np.linalg.norm(prev_acceleration - np.concatenate(
                    [self.shearable_rods[name].acceleration_collection for name in keys],
                    axis=-1
                ), axis=0))
            director_delta = np.nanmax(
                np.linalg.norm(prev_director - np.concatenate(
                    [self.shearable_rods[name].director_collection for name in keys],
                    axis=-1
                ), axis=(0, 1)))
            omega_delta = np.nanmax(
                np.linalg.norm(prev_omega - np.concatenate(
                    [self.shearable_rods[name].omega_collection for name in keys],
                    axis=-1
                ), axis=0))
            alpha_delta = np.nanmax(
                np.linalg.norm(prev_alpha - np.concatenate(
                    [self.shearable_rods[name].alpha_collection for name in keys],
                    axis=-1
                ), axis=0))
            # Check steady state status
            position_steady_state_status = (position_delta < self.position_threshold)
            velocity_steady_state_status = (velocity_delta < self.velocity_threshold)
            acceleration_steady_state_status = (acceleration_delta < self.acceleration_threshold)
            director_steady_state_status = (director_delta < self.director_threshold)
            omega_steady_state_status = (omega_delta < self.omega_threshold)
            alpha_steady_state_status = (alpha_delta < self.alpha_threshold)
            # fmt: on

        # Check NaN
        if check_nan:
            # fmt: off
            # Position of the rod cannot be NaN, it is not valid, stop the simulation
            status.position_nan_status = any([_isnan_check(self.shearable_rods[name].position_collection) for name in self.shearable_rods.keys()] )
            status.velocity_nan_status = any([_isnan_check(self.shearable_rods[name].velocity_collection) for name in self.shearable_rods.keys()])
            status.director_nan_status = any([_isnan_check(self.shearable_rods[name].director_collection) for name in self.shearable_rods.keys()])
            status.omega_nan_status = any([_isnan_check(self.shearable_rods[name].omega_collection) for name in self.shearable_rods.keys()])
            # fmt: on

        status.end_status = True
        return status

    def save_state(
        self, directory: Optional[str] = None, verbose: bool = False
    ) -> None:
        """
        Save state parameters of each rod.

        Parameters
        ----------
        directory: Optional[str]
            Directory path name. The path must exist.
        """
        if not directory:
            directory = self.paths.simulation
        self.assy.save_state(directory=directory, time=self.time, verbose=verbose)

    def load_state(
        self, directory: str = "", clear_callback: bool = False, verbose: bool = False
    ) -> None:
        """
        Load the rod-state.
        Compatibale with 'save_state' method.

        If the save-file does not exist, it returns error.

        Parameters
        ----------
        directory : Optional[str]
            Directory path name.
        """
        if not directory:
            directory = self.paths.simulation
        self.assy.load_state(directory=directory, verbose=verbose)

        # Clear callback
        if clear_callback:
            for callback_defaultdict in self.data_rods:
                callback_defaultdict.clear()
            if verbose:
                print("callback cleared")

    def render_video(
        self,
        **kwargs,
    ) -> None:
        """
        Make video 3D rod movement in time.
        """

        filename_video = "br2_simulation"
        save_folder = self.paths.renderings

        plot_video_with_surface(
            self.data_rods,
            video_name=filename_video,
            fps=self.rendering_fps,
            step=1,
            save_folder=save_folder,
            **kwargs,
        )

        position_data_path = os.path.join(save_folder, f"br2_data_{data_tag}.npz")
        self.save_data(position_data_path)

    def save_data(self, tag:Optional[str]=None) -> None:
        """save_data.

        Parameters
        ----------
        tag : Optional[str]
            String tag that appends to the file name.
        """
        if not tag:
            filename = f"br2_data.npz"
        else:
            filename = f"br2_data_{tag}.npz"
        path = os.path.join(self.paths.data, filename)
        position_rod = np.array(self.data_rods[0]["position"])
        position_rod = 0.5 * (position_rod[..., 1:] + position_rod[..., :-1])
        np.savez(
            path,
            time=np.array(self.data_rods[0]["time"]),
            position_rod=position_rod,
        )

    def close(self):
        """
        Close the simulator.
        """
        pass
