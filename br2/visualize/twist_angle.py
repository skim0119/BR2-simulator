import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import proj3d, Axes3D
from tqdm import tqdm

from typing import Dict, Sequence
from elastica.rod.knot_theory import compute_twist


def visual_twist_with_surface(
    rods_history: Sequence[Dict],
    video_name="video",
    fps=60,
    step=1,
    save_folder="",
    vis2D=True,
    **kwargs,
):

    # simulation time
    sim_time = np.array(rods_history[0]["time"])

    # Rod
    n_visualized_rods = len(rods_history)  # should be one for now
    # Rod info
    def rod_history_unpacker(rod_idx, t_idx):
        return (
            rods_history[rod_idx]["position"][t_idx],
            rods_history[rod_idx]["radius"][t_idx],
        )

    # Rod center of mass
    def com_history_unpacker(rod_idx, t_idx):
        return rods_history[rod_idx]["com"][t_idx]

    # Director
    def director_history_unpacker(rod_idx, t_idx):
        return rods_history[rod_idx]["director"][t_idx]

    # centerline
    def centerline_history_unpacker(rod_idx, t_idx):
        center_line = []
        position, _ = rod_history_unpacker(rod_idx, t_idx)
        center_line = np.array(position)
        return center_line

    # video pre-processing
    # print("plot scene visualization video")
    FFMpegWriter = animation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    dpi = kwargs.get("dpi", 100)

    xlim = kwargs.get("x_limits", (-1.0, 1.0))
    ylim = kwargs.get("y_limits", (-1.0, 1.0))
    zlim = kwargs.get("z_limits", (-0.05, 1.0))

    difference = lambda x: x[1] - x[0]
    max_axis_length = max(difference(xlim), difference(ylim))
    # The scaling factor from physical space to matplotlib space
    scaling_factor = (2 * 0.1) / max_axis_length  # Octopus head dimension
    scaling_factor *= 2.6e3  # Along one-axis

    if kwargs.get("vis2D", True):

        # Plot zx

        fig = plt.figure(2, figsize=(10, 8), frameon=True, dpi=dpi)
        ax = fig.add_subplot(111)
        ax.set_xlim(*zlim)
        ax.set_ylim(*xlim)

        time_idx = 0
        rod_lines = [None for _ in range(n_visualized_rods)]
        rod_com_lines = [None for _ in range(n_visualized_rods)]
        rod_scatters = [None for _ in range(n_visualized_rods)]
        # initial dictionary to store rotation_degree
        rot_degrees_all = np.zeros(3)

        for rod_idx in range(n_visualized_rods):
            inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
            inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
            rod_lines[rod_idx] = ax.plot(
                inst_position[2], inst_position[0], "r", lw=0.5
            )[0]
            inst_com = com_history_unpacker(rod_idx, time_idx)
            rod_com_lines[rod_idx] = ax.plot(inst_com[2], inst_com[0], "k--", lw=2.0)[0]

            rod_scatters[rod_idx] = ax.scatter(
                inst_position[2],
                inst_position[0],
                s=np.pi * (scaling_factor * inst_radius) ** 2,
            )

            director_total = director_history_unpacker(rod_idx, time_idx)
            normal = director_total[0, :, :]
            center_line = centerline_history_unpacker(rod_idx, time_idx)
            normal_cal_degrees = normal[
                np.newaxis, :, :
            ]  # change to 3 dimension so compute_twist can calculate it
            center_line = center_line[np.newaxis, :, :]
            rot_degrees_temp, _ = compute_twist(center_line, normal_cal_degrees)
            rot_degrees_temp = np.degrees(rot_degrees_temp[-1] * 2 * np.pi)
            rot_degrees_all[rod_idx] = rot_degrees_temp

            # the beginning point of arrow
            x_position = inst_position[2][..., -1]
            z_position = inst_position[0][..., -1]
            # calculate the direction of arrow
            U = normal[2, -1] * inst_radius[-1]
            V = normal[0, -1] * inst_radius[-1]
            quiver = ax.quiver(x_position, z_position, U, V)

        ax.set_aspect("equal")
        video_name_2D = os.path.join(
            save_folder, video_name + "_2D_visual_twist_angle.mp4"
        )

        with writer.saving(fig, video_name_2D, dpi):
            with plt.style.context("seaborn-v0_8-whitegrid"):
                for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):

                    current_time = sim_time[time_idx]

                    for rod_idx in range(n_visualized_rods):
                        ax.set_title(
                            f"Simulation Time: {current_time:.2f} seconds\n"
                            f"Twist_angle_0: {rot_degrees_all[0]:.2f}\n"
                            f"Twist_angle_rod_1: {rot_degrees_all[1]:.2f}\n"
                            f"Twist_angle_rod_2: {rot_degrees_all[2]:.2f}"
                        )
                        if "quiver" in locals():
                            quiver.remove()

                        inst_position, inst_radius = rod_history_unpacker(
                            rod_idx, time_idx
                        )
                        inst_position = 0.5 * (
                            inst_position[..., 1:] + inst_position[..., :-1]
                        )

                        rod_lines[rod_idx].set_xdata(inst_position[2])
                        rod_lines[rod_idx].set_ydata(inst_position[0])

                        com = com_history_unpacker(rod_idx, time_idx)
                        rod_com_lines[rod_idx].set_xdata([com[2]])
                        rod_com_lines[rod_idx].set_ydata([com[0]])

                        rod_scatters[rod_idx].set_offsets(
                            np.vstack((inst_position[2], inst_position[0])).T
                        )
                        rod_scatters[rod_idx].set_sizes(
                            np.pi * (scaling_factor * inst_radius) ** 2
                        )

                        director_total = director_history_unpacker(rod_idx, time_idx)
                        normal = director_total[0, :, :]
                        center_line = centerline_history_unpacker(rod_idx, time_idx)
                        normal_cal_degrees = normal[
                            np.newaxis, :, :
                        ]  # change to 3 dimension so compute_twist can calculate it
                        center_line = center_line[np.newaxis, :, :]
                        rot_degrees_temp, _ = compute_twist(
                            center_line, normal_cal_degrees
                        )
                        rot_degrees_temp = np.degrees(rot_degrees_temp[-1] * 2 * np.pi)
                        rot_degrees_all[rod_idx] = rot_degrees_temp

                        # the beginning point of arrow
                        x_position = inst_position[2][..., -1]
                        z_position = inst_position[0][..., -1]
                        # calculate the direction of arrow
                        U = normal[2, -1] * inst_radius[-1]
                        V = normal[0, -1] * inst_radius[-1]
                        quiver = ax.quiver(x_position, z_position, U, V)

                    writer.grab_frame()

        # Be a good boy and close figures
        # https://stackoverflow.com/a/37451036
        # plt.close(fig) alone does not suffice
        # See https://github.com/matplotlib/matplotlib/issues/8560/
        plt.close(plt.gcf())
