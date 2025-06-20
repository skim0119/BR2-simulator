import os
import sys

import csv
import pandas as pd
import datetime
from functools import partial

# sys.settrace

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

np.set_printoptions(precision=4)

from br2.environment import Environment
from br2.constants import psi2Nm2
from br2.configurations import RodLibrary

RTIME = 21.0


def add_tip_weight(simulator, rods):
    from br2.free_custom_systems import TipLoad

    weight = 0.027 / 3

    for name, rod in rods.items():
        if "seg2" in name:
            simulator.add_forcing_to(rod).using(
                TipLoad,
                time_interval=(33.0, 65.0),
                start_force=np.zeros(3),
                end_force=np.array([0, 0, -1]) * weight,
                ramp_up_time=1.0,
            )


def add_hole_constraints(simulator, rods, rendering_fps, time_step):
    # Adjust factor for inner-tube for actuation
    # rods["seg1_2_RodElongation"].bend_matrix[0, 0, :] *= 2
    # rods["seg1_2_RodElongation"].bend_matrix[1, 1, :] *= 2
    # rods["seg1_2_RodElongation"].bend_matrix[2, 2, :] *= 2

    import elastica as ea
    from collections import defaultdict
    from br2.callbacks import BlenderRodCallback
    from br2.custom_constraint import FixFix

    ring_center_position = np.array([0.12, 0.12, -0.18])
    ring_base_length = 0.26
    ring_R = ring_base_length / (2 * np.pi)
    ring_r = 0.0040
    torus = ea.CosseratRod.ring_rod(
        n_elements=48,
        ring_center_position=ring_center_position,
        direction=np.array([0, 1, 0.0]),
        normal=np.array([1, 0, 0.0]),
        base_length=ring_base_length,
        base_radius=ring_r,
        density=5000,
        youngs_modulus=1e7,
    )
    simulator.append(torus)
    simulator.constrain(torus).using(FixFix)
    # Contact between two rods
    step_skip = max(1, int(1.0 / (rendering_fps * time_step)))
    for name, rod in rods.items():
        if "seg2" in name:
            simulator.detect_contact_between(torus, rod).using(
                ea.RodRodContact, k=1e3, nu=0.0
            )
    callback_params = defaultdict(list)
    simulator.collect_diagnostics(torus).using(
        BlenderRodCallback,
        step_skip=step_skip,
        time_interval=None,
        callback_params=callback_params,
        visualize_alpha_beta=False,
        is_ring=True,
    )

    # Add soft spring between last rod and torus in close-proximity
    from br2.custom_pivot import PivotSpring
    from br2.surface_connection import (
        SurfaceJointSideBySide,
        get_connection_vector_straight_straight_rod,
        get_connection_directors_straight_straight_rod,
    )

    zminus = np.array([0.0, 0.0, -1.0])
    yplus = np.array([0.0, 1.0, 0.0])
    pivot_angle = np.pi / 4
    pivot_loc = (
        ring_center_position
        + zminus * (ring_R - ring_r) * np.cos(pivot_angle)
        + yplus * (ring_R - ring_r) * np.sin(pivot_angle)
    )

    def add_spring(rod_one, rod_two):
        simulator.add_forcing_to(rod_two).using(
            PivotSpring,
            R=0.04,
            # k=1e3,
            k=4e3,
            order=3,
            loc=pivot_loc,
            pivot_idx=52,
            # time_interval=(12.0, 26.2),
            # time_interval=(0.0, 26.2),
            time_interval=(0.0, 38.0),
        )

    for name, rod in rods.items():
        if "seg2_0_RodBend" in name:
            add_spring(torus, rod)


def run(
    filename,
    rod_database_path,
    assembly_config_path,
    tag,
    ratio,
    optimizing=False,
    inputs=None,  # arbitrary to sweep
    restart_save_path=None,
    skip_stime=0.0,  # Use with restart
):
    DEBUG = not optimizing
    EXPORT_BLENDER = True and DEBUG

    with open(filename, newline="") as f:
        reader = csv.reader(f)
        recorded_data_list = list(reader)

        # print(time_diff_btw_counters)
        # time.sleep(time_diff_btw_counters)
        # print(bending, twisting, bending2, twisting2 )

    # Prepare environment
    time_step = 4.0e-5
    rendering_fps = 10
    env = Environment(
        time_step=time_step,
        run_tag=tag,
        export_blender=EXPORT_BLENDER,
        visualize_alpha_beta=False,
        # capture_interval=(25.8, 27),  # This can be used with rendering_fps=1 to visualize every steps
        # rendering_fps=None, #rendering_fps,
        rendering_fps=rendering_fps,
    )
    env.reset(
        rod_database_path=rod_database_path,
        assembly_config_path=assembly_config_path,
        k_multiplier=7.0,
        k_repulsive=8,  # Default 2
        nu_multiplier=0.0000,  # Default 0
        k_torsion_multiplier=1e4,  # Default 1e4,
        k_torsion_multiplier_serial=1e4 * 3.0,  # Default 1e4,
        custom_callbacks=[
            add_tip_weight,
            partial(
                add_hole_constraints, rendering_fps=rendering_fps, time_step=time_step
            ),
        ],
        REMOVE_CONNECTION=False,  # Debug to remove connection and run
        restart_save_path=restart_save_path,
        start_time=skip_stime,
    )

    time_delta = [float(data[1]) for data in recorded_data_list]
    timestamps = np.cumsum(time_delta)
    total_time = timestamps[-1]  # 21: approach
    bending_pressure = [float(data[2]) for data in recorded_data_list]
    twisting_pressure = [float(data[3]) for data in recorded_data_list]
    bending2_pressure = [float(data[4]) for data in recorded_data_list]
    twisting2_pressure = [float(data[5]) for data in recorded_data_list]

    # Spline Interpolate the pressure data for each seconds
    duration = 0.1
    time = np.arange(skip_stime, int(total_time), duration)

    bending_interp = np.interp(time, timestamps, bending_pressure)
    twisting_interp = np.interp(time, timestamps, twisting_pressure)
    bending2_interp = np.interp(time, timestamps, bending2_pressure)
    twisting2_interp = np.interp(time, timestamps, twisting2_pressure)

    pbar = tqdm(total=total_time, disable=False)
    for bending, twisting, bending2, twisting2 in zip(
        bending_interp, twisting_interp, bending2_interp, twisting2_interp
    ):
        pbar.set_description(
            f"a1={bending:.2e} | a2={twisting:.2e} | a3={bending2:.2e} | a4={twisting2:.2e}"
        )
        # action = {
        #     "action1": 0 * psi2Nm2 * ratio,
        #     "action2": 0 * psi2Nm2 * ratio,
        #     "action3": 0 * psi2Nm2 * ratio,
        #     "action4": 20 * psi2Nm2 * ratio,
        # }
        action = {
            "action1": bending * psi2Nm2 * ratio,
            "action2": twisting * psi2Nm2 * ratio,
            "action3": bending2 * psi2Nm2 * ratio,
            "action4": twisting2 * psi2Nm2 * ratio,
        }

        # for current_data in recorded_data_list:
        #    _, time_diff_btw_counters, bending, twisting, bending2, twisting2 = current_data
        #    time_diff_btw_counters = float(time_diff_btw_counters)
        #    bending = float(bending)
        #    twisting = float(twisting)
        #    bending2 = float(bending2)
        #    twisting2 = float(twisting2)
        #    duration = time_diff_btw_counters

        # Simulation
        status = env.run(
            action=action,
            duration=duration,
            check_nan=True,
            check_steady_state=False,
            pbar=pbar,
        )

        # BREAK
        if skip_stime > 0.0 and env.time > RTIME:
            env.save_state(directory=restart_save_path, verbose=verbose)
            print("BREAK run!: simulation time over. File saved.")
            break
        if status.combined_nan_status:
            print(status)
            break

    # Post Processing
    # env.render_video(
    #     # The following parameters are optional
    #     x_limits=(-0.13, 0.13),  # Set bounds on x-axis
    #     y_limits=(-0.05, 0.5),  # Set bounds on y-axis
    #     z_limits=(-0.13, 0.13),  # Set bounds on z-axis
    #     dpi=100,  # Set the quality of the image
    #     vis3D=True,  # Turn on 3D visualization
    #     vis2D=True,  # Turn on projected (2D) visualization
    #     vis3D_director=False,
    #     vis2D_director_lastelement=False,
    #     visualize_twist_angle=False,
    #     max_fps=30,
    # )
    # env.debug_data()
    # env.save_data()

    if DEBUG:
        env.save_rod_data()

    env.save_state()

    # Terminate
    env.close()
    pbar.close()

    save_path = env.paths.paths
    os.system(f"mv {save_path} /home/sharing/Elastica-BR2-gulam/")

    # Return results in useful format
    # Test serial connection maintains parallel
    L = 0
    v1 = env.data_rods["seg1_0_RodElongation"]["director"][-1][:, :, -1]
    v2 = env.data_rods["seg2_0_RodBend"]["director"][-1][:, :, 0]
    L += np.linalg.norm(v1 - v2, ord="fro")
    v1 = env.data_rods["seg1_1_RodElongationSide"]["director"][-1][:, :, -1]
    v2 = env.data_rods["seg2_1_RodLeftTwist"]["director"][-1][:, :, 0]
    L += np.linalg.norm(v1 - v2, ord="fro")
    v1 = env.data_rods["seg1_2_RodElongationSide"]["director"][-1][:, :, -1]
    v2 = env.data_rods["seg2_2_RodRightTwist"]["director"][-1][:, :, 0]
    L += np.linalg.norm(v1 - v2, ord="fro")
    return L


def plot_pressure(file_path):
    data = pd.read_csv(file_path, delimiter=",", header=None)

    time = pd.to_datetime(data.iloc[:, 0])
    bending = data.iloc[:, 2]
    twisting = data.iloc[:, 3]
    bending_2 = data.iloc[:, 4]
    twisting_2 = data.iloc[:, 5]

    fig, axs = plt.subplots(4, 1, figsize=(8, 8))

    axs[0].plot(time, bending, label="Bending pressue")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Pressue")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(time, twisting, label="Twisting pressue")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Pressue")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(time, bending_2, label="Bending_2 pressue")
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Pressue")
    axs[2].legend()
    axs[2].grid(True)

    axs[3].plot(time, twisting_2, label="Twisting_2 pressue")
    axs[3].set_xlabel("Time")
    axs[3].set_ylabel("Pressue")
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    ratio = 0.70

    # action_path = "action.csv"
    tag = "1m6s"
    action_path = "2024_09_15_0100_practice/1m6s.csv"

    RESTART = False
    if RESTART:
        restart_save_path = "cache_restart"
        skip_stime = RTIME
    else:
        restart_save_path = None
        skip_stime = 0.0
    # plot_pressure(action_path)

    import time
    import datetime
    import multiprocessing as mp

    timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M")

    stime = time.time()
    ms = 2.0 ** np.linspace(-2, 8, 50)
    if False:
        import multiprocessing as mp

        args_list = [
            (
                action_path,
                "rods.json",
                "b3br2.json",
                f"case_hole_contact_{ratio}_{tag}_{timestamp}",
                ratio,
                True,
                m,
                restart_save_path,
                skip_stime,
            )
            for m in ms
        ]
        with mp.Pool() as pool:
            results = pool.starmap(run, args_list)
        inputs = ms
        vals = results
    else:
        inputs, vals = [], []
        for m in [3.0]:
            results = run(
                action_path,
                "rods.json",
                "b3br2.json",
                tag=f"case_hole_contact_{ratio}_{tag}_{timestamp}",
                ratio=ratio,
                inputs=m,
                restart_save_path=restart_save_path,
                skip_stime=skip_stime,
            )
            inputs.append(m)
            vals.append(results)
    total_time = time.time() - stime

    for i, v in zip(inputs, vals):
        print(i, v)

    _min, _sec = divmod(total_time, 60)
    print(f"Time taken: {_min:.0f} min {_sec:.0f} sec")
