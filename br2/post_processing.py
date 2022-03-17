import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d import proj3d, Axes3D
#from tqdm import tqdm

from typing import Dict, Sequence

def tqdm(obj): # tqdm suppressor
    return obj

def plot_video_2d(
    plot_params1: dict, plot_params2: dict, video_name="video.mp4", margin=0.2, fps=40
):  # (time step, x/y/z, node)
    import matplotlib.animation as manimation

    plt.rcParams["animation.ffmpeg_path"] = "/usr/local/bin/ffmpeg"

    positions_over_time1 = np.array(plot_params1["position"])
    positions_over_time2 = np.array(plot_params2["position"])

    forces1 = np.array(plot_params1["external_forces"])
    forces2 = np.array(plot_params2["external_forces"])
    time_array = np.array(plot_params1["time"])
    print("plot video")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure()
    plt.axis("equal")
    with writer.saving(fig, video_name, dpi=200):
        for time in range(1, len(plot_params1["time"])):
            x1 = positions_over_time1[time][0]
            y1 = positions_over_time1[time][1]
            x2 = positions_over_time2[time][0]
            y2 = positions_over_time2[time][1]
            fig.clf()
            plt.plot(x1, y1, ".-r")
            plt.plot(x2, y2, ".-b")

            # plot line to indicate force ramp up
            # time_real = time_array[time]
            # ramp_time = 1.0
            # factor = min(1.0, time_real / ramp_time)
            # plt.plot([0.5,0.501],[0.0,factor],'-k', linewidth = 2)

            plt.xlim([-1.0 - margin, 1.0 + margin])
            plt.ylim([-0.5 - margin, 1.5 + margin])
            writer.grab_frame()

    # Be a good boy and close figures
    # https://stackoverflow.com/a/37451036
    # plt.close(fig) alone does not suffice
    # See https://github.com/matplotlib/matplotlib/issues/8560/
    plt.close(plt.gcf())



def plot_video_with_surface(
    rods_history: Sequence[Dict],
    video_name="video",
    fps=60,
    step=1,
    save_folder="",
    vis2D=True,
    **kwargs,
):
    #plt.rcParams.update({"font.size": 22})

    # 2d case <always 2d case for now>
    import matplotlib.animation as animation

    # simulation time
    sim_time = np.array(rods_history[0]["time"])

    # Rod
    n_visualized_rods = len(rods_history)  # should be one for now
    # Rod info
    rod_history_unpacker = lambda rod_idx, t_idx: (
        rods_history[rod_idx]["position"][t_idx],
        rods_history[rod_idx]["radius"][t_idx],
    )
    # Rod center of mass
    com_history_unpacker = lambda rod_idx, t_idx: rods_history[rod_idx]["com"][time_idx]

    # video pre-processing
    #print("plot scene visualization video")
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

    if kwargs.get("vis2D_director_lastelement", False):
        _length = 0.070
        color_scheme = plt.rcParams['axes.prop_cycle'].by_key()['color']
        directors = np.array([data['director'] for data in rods_history])[:,:,:,[0,2],...]
        positions = np.array([data['position'] for data in rods_history])[:,:,[0,2],...]
        positions = 0.5 * (positions[...,1:] + positions[...,:-1]) # Get element position
        directors = directors[...,-1]
        positions = positions[...,-1]
        n_elem = 1
        n_rod = directors.shape[0]

        fig = plt.figure(1, figsize=(10, 8), frameon=True, dpi=dpi)
        ax = plt.axes()

        ax.set_xlabel("z")
        ax.set_ylabel("x")

        ax.set_xlim(*zlim)
        ax.set_ylim(*xlim)

        video_name_2d_quiv = os.path.join(save_folder, video_name + "_2D_directors_last.mp4")
        with writer.saving(fig, video_name_2d_quiv, dpi), plt.style.context("seaborn-whitegrid"):
            time_idx = 0
            quiver_axes = [[] for _ in range(n_rod)]
            for rod_idx in range(n_rod):
                position = positions[rod_idx, time_idx, ...]
                director = directors[rod_idx, time_idx, ...] * _length
                for i in range(3):
                    quiver_axes[rod_idx].append(ax.quiver(*position, *director[i], color=color_scheme[rod_idx]))
                ax.set_aspect("auto")
            writer.grab_frame()
            #ax.set_aspect("equal")
            for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):
                for rod_idx in range(n_rod):
                    position = positions[rod_idx, time_idx, ...]
                    director = directors[rod_idx, time_idx, ...] * _length
                    for i in range(3):
                        quiver_axes[rod_idx][i].set_offsets([position.tolist()])
                        quiver_axes[rod_idx][i].set_UVC(*director[i,:].tolist())
                writer.grab_frame()

        # Be a good boy and close figures
        # https://stackoverflow.com/a/37451036
        # plt.close(fig) alone does not suffice
        # See https://github.com/matplotlib/matplotlib/issues/8560/
        plt.close(plt.gcf())
    if kwargs.get("vis3D_director", False):
        _length = 0.070
        skip_element = 5
        color_scheme = plt.rcParams['axes.prop_cycle'].by_key()['color']
        directors = np.array([data['director'] for data in rods_history])[...,::skip_element]
        positions = np.array([data['position'] for data in rods_history])[...,::skip_element]
        #positions = 0.5 * (positions[...,1:] + positions[...,:-1]) # Get element position
        #positions[:,[0,1,2],:] = positions[:,[2,0,1],:] # Swap axis for visualization purpose
        #directors[:,:,[0,1,2],:] = directors[:,:,[2,0,1],:] # TODO: need to rotate
        n_elem = positions.shape[-1]

        fig = plt.figure(1, figsize=(10, 8), frameon=True, dpi=dpi)
        ax = plt.axes(projection="3d")

        ax.set_xlabel("z")
        ax.set_ylabel("x")
        ax.set_zlabel("y")

        ax.set_xlim(*zlim)
        ax.set_ylim(*xlim)
        ax.set_zlim(*ylim)

        video_name_3D = os.path.join(save_folder, video_name + "_3D_directors.mp4")
        with writer.saving(fig, video_name_3D, dpi), plt.style.context("seaborn-whitegrid"):
            time_idx = 0
            quiver_axes = [[] for _ in range(n_visualized_rods)]
            for rod_idx in range(0, n_visualized_rods):
                position = positions[rod_idx, time_idx, ...]
                director = directors[rod_idx, time_idx, ...]
                for i in range(3):
                    quiver_axes[rod_idx].append(ax.quiver(*position, *director[i], length=_length, color=color_scheme[rod_idx]))
            writer.grab_frame()

            #ax.set_aspect("equal")
            ax.set_aspect("auto")
            for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):
                for rod_idx in range(0, n_visualized_rods):
                    position = positions[rod_idx, time_idx, ...]
                    director = directors[rod_idx, time_idx, ...] * _length
                    for i in range(3):
                        segs = [[position[:,j].tolist(), (position[:,j]+director[i,:,j]).tolist()] for j in range(n_elem)]
                        quiver_axes[rod_idx][i].set_segments(segs)
                writer.grab_frame()

        # Be a good boy and close figures
        # https://stackoverflow.com/a/37451036
        # plt.close(fig) alone does not suffice
        # See https://github.com/matplotlib/matplotlib/issues/8560/
        plt.close(plt.gcf())

    if kwargs.get("vis3D", True):
        fig = plt.figure(1, figsize=(10, 8), frameon=True, dpi=dpi)
        ax = plt.axes(projection="3d")

        ax.set_xlabel("z")
        ax.set_ylabel("x")
        ax.set_zlabel("y")

        ax.set_xlim(*zlim)
        ax.set_ylim(*xlim)
        ax.set_zlim(*ylim)

        time_idx = 0
        rod_lines = [None for _ in range(n_visualized_rods)]
        rod_com_lines = [None for _ in range(n_visualized_rods)]
        rod_scatters = [None for _ in range(n_visualized_rods)]

        _scaling_factor = scaling_factor * 0.6

        for rod_idx in range(n_visualized_rods):
            inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
            _s = np.pi * (_scaling_factor * inst_radius[0]) ** 2
            
            try:
                rod_scatters[rod_idx] = ax.scatter(
                    -inst_position[2],
                    -inst_position[0],
                    inst_position[1],
                    s=_s,
                    #s=inst_radius[0]**2
                )
            except ValueError:
                print(_s)
                input('')

        #ax.set_aspect("equal")
        ax.set_aspect("auto")
        video_name_3D = os.path.join(save_folder, video_name + "_3D.mp4")

        with writer.saving(fig, video_name_3D, dpi):
            with plt.style.context("seaborn-whitegrid"):
                for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):

                    for rod_idx in range(n_visualized_rods):
                        inst_position, inst_radius = rod_history_unpacker(
                            rod_idx, time_idx
                        )
                        inst_position = 0.5 * (
                            inst_position[..., 1:] + inst_position[..., :-1]
                        )

                        rod_scatters[rod_idx]._offsets3d = (
                            -inst_position[2],
                            -inst_position[0],
                            inst_position[1],
                        )

                        # rod_scatters[rod_idx].set_offsets(inst_position[:2].T)
                        rod_scatters[rod_idx].set_sizes(
                            np.pi * (_scaling_factor * inst_radius) ** 2,
                        )

                    writer.grab_frame()

        # Be a good boy and close figures
        # https://stackoverflow.com/a/37451036
        # plt.close(fig) alone does not suffice
        # See https://github.com/matplotlib/matplotlib/issues/8560/
        plt.close(plt.gcf())

    if kwargs.get("vis2D", True):
        # Plot xy
        fig = plt.figure(2, figsize=(10, 8), frameon=True, dpi=dpi)
        ax = fig.add_subplot(111)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        time_idx = 0
        rod_lines = [None for _ in range(n_visualized_rods)]
        rod_com_lines = [None for _ in range(n_visualized_rods)]
        rod_scatters = [None for _ in range(n_visualized_rods)]

        for rod_idx in range(n_visualized_rods):
            inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
            inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
            rod_lines[rod_idx] = ax.plot(
                inst_position[0], inst_position[1], "r", lw=0.5
            )[0]
            inst_com = com_history_unpacker(rod_idx, time_idx)
            rod_com_lines[rod_idx] = ax.plot(inst_com[0], inst_com[1], "k--", lw=2.0)[0]

            rod_scatters[rod_idx] = ax.scatter(
                inst_position[0],
                inst_position[1],
                s=np.pi * (scaling_factor * inst_radius) ** 2,
            )

        ax.set_aspect("equal")
        video_name_2D = os.path.join(save_folder, video_name + "_2D_xy.mp4")

        with writer.saving(fig, video_name_2D, dpi):
            with plt.style.context("seaborn-whitegrid"):
                for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):

                    for rod_idx in range(n_visualized_rods):
                        inst_position, inst_radius = rod_history_unpacker(
                            rod_idx, time_idx
                        )
                        inst_position = 0.5 * (
                            inst_position[..., 1:] + inst_position[..., :-1]
                        )

                        rod_lines[rod_idx].set_xdata(inst_position[0])
                        rod_lines[rod_idx].set_ydata(inst_position[1])

                        com = com_history_unpacker(rod_idx, time_idx)
                        rod_com_lines[rod_idx].set_xdata(com[0])
                        rod_com_lines[rod_idx].set_ydata(com[1])

                        rod_scatters[rod_idx].set_offsets(inst_position[:2].T)
                        rod_scatters[rod_idx].set_sizes(
                            np.pi * (scaling_factor * inst_radius) ** 2
                        )

                    writer.grab_frame()

        # Be a good boy and close figures
        # https://stackoverflow.com/a/37451036
        # plt.close(fig) alone does not suffice
        # See https://github.com/matplotlib/matplotlib/issues/8560/
        plt.close(plt.gcf())

        # Plot yz

        fig = plt.figure(2, figsize=(10, 8), frameon=True, dpi=dpi)
        ax = fig.add_subplot(111)
        ax.set_xlim(*ylim)
        ax.set_ylim(*zlim)

        time_idx = 0
        rod_lines = [None for _ in range(n_visualized_rods)]
        rod_com_lines = [None for _ in range(n_visualized_rods)]
        rod_scatters = [None for _ in range(n_visualized_rods)]

        for rod_idx in range(n_visualized_rods):
            inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
            inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
            rod_lines[rod_idx] = ax.plot(
                inst_position[1], inst_position[2], "r", lw=0.5
            )[0]
            inst_com = com_history_unpacker(rod_idx, time_idx)
            rod_com_lines[rod_idx] = ax.plot(inst_com[1], inst_com[2], "k--", lw=2.0)[0]

            rod_scatters[rod_idx] = ax.scatter(
                inst_position[1],
                inst_position[2],
                s=np.pi * (scaling_factor * inst_radius) ** 2,
            )

        ax.set_aspect("equal")
        video_name_2D = os.path.join(save_folder, video_name + "_2D_yz.mp4")

        with writer.saving(fig, video_name_2D, dpi):
            with plt.style.context("seaborn-whitegrid"):
                for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):

                    for rod_idx in range(n_visualized_rods):
                        inst_position, inst_radius = rod_history_unpacker(
                            rod_idx, time_idx
                        )
                        inst_position = 0.5 * (
                            inst_position[..., 1:] + inst_position[..., :-1]
                        )

                        rod_lines[rod_idx].set_xdata(inst_position[1])
                        rod_lines[rod_idx].set_ydata(inst_position[2])

                        com = com_history_unpacker(rod_idx, time_idx)
                        rod_com_lines[rod_idx].set_xdata(com[1])
                        rod_com_lines[rod_idx].set_ydata(com[2])

                        rod_scatters[rod_idx].set_offsets(
                            np.vstack((inst_position[1], inst_position[2])).T
                        )
                        rod_scatters[rod_idx].set_sizes(
                            np.pi * (scaling_factor * inst_radius) ** 2
                        )

                    writer.grab_frame()

        # Be a good boy and close figures
        # https://stackoverflow.com/a/37451036
        # plt.close(fig) alone does not suffice
        # See https://github.com/matplotlib/matplotlib/issues/8560/
        plt.close(plt.gcf())

        # Plot zx

        fig = plt.figure(2, figsize=(10, 8), frameon=True, dpi=dpi)
        ax = fig.add_subplot(111)
        ax.set_xlim(*zlim)
        ax.set_ylim(*xlim)

        time_idx = 0
        rod_lines = [None for _ in range(n_visualized_rods)]
        rod_com_lines = [None for _ in range(n_visualized_rods)]
        rod_scatters = [None for _ in range(n_visualized_rods)]

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

        ax.set_aspect("equal")
        video_name_2D = os.path.join(save_folder, video_name + "_2D_zx.mp4")

        with writer.saving(fig, video_name_2D, dpi):
            with plt.style.context("seaborn-whitegrid"):
                for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):

                    for rod_idx in range(n_visualized_rods):
                        inst_position, inst_radius = rod_history_unpacker(
                            rod_idx, time_idx
                        )
                        inst_position = 0.5 * (
                            inst_position[..., 1:] + inst_position[..., :-1]
                        )

                        rod_lines[rod_idx].set_xdata(inst_position[2])
                        rod_lines[rod_idx].set_ydata(inst_position[0])

                        com = com_history_unpacker(rod_idx, time_idx)
                        rod_com_lines[rod_idx].set_xdata(com[2])
                        rod_com_lines[rod_idx].set_ydata(com[0])

                        rod_scatters[rod_idx].set_offsets(
                            np.vstack((inst_position[2], inst_position[0])).T
                        )
                        rod_scatters[rod_idx].set_sizes(
                            np.pi * (scaling_factor * inst_radius) ** 2
                        )

                    writer.grab_frame()

        # Be a good boy and close figures
        # https://stackoverflow.com/a/37451036
        # plt.close(fig) alone does not suffice
        # See https://github.com/matplotlib/matplotlib/issues/8560/
        plt.close(plt.gcf())
