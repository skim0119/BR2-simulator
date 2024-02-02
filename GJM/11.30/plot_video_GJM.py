import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, Sequence
from tqdm import tqdm

def plot_video_2D(
    rods_history: Sequence[dict],
    video_name="video.mp4",
    fps=30,
    step=1,
    **kwargs,            
):
    dpi = kwargs.get("dpi", 100)
    time_idx = 0
    #设置全局的字体大小
    plt.rcParams.update({"font.size": 22})

    # simulation time
    sim_time = np.array(rods_history[0]["time"])

    # Rod
    n_visualized_rods = len(rods_history)
    
    # Rod info
    rod_history_unpacker = lambda rod_idx, t_idx: (
        rods_history[rod_idx]["position"][t_idx],
        rods_history[rod_idx]["radius"][t_idx],
    )

    # Rod center of mass
    com_history_unpacker = lambda rod_idx, t_idx: rods_history[rod_idx]["com"][time_idx]

    # video pre-processing
    print("plot scene visualization video")
    FFMpegWriter = animation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata) #,extra_args=['-v', 'debug'])

    #设置x轴的缩放
    xlim = kwargs.get("x_limits", (-1.0, 1.0))
    ylim = kwargs.get("y_limits", (-1.0, 1.0))
    difference = lambda x: x[1] - x[0]
    max_axis_length = max(difference(xlim), difference(ylim))
    # The scaling factor from physical space to matplotlib space
    scaling_factor = (2 * 0.1) / max_axis_length  # Octopus head dimension
    scaling_factor *= 2.6e3  # Along one-axis

    #设置图形和轴
    fig = plt.figure(2, figsize=(10, 8), frameon=True, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


    # xy plot
    video_name_xy = "xy_" + video_name
    #列表推导式，用于生成一个长度为n_visualized_rods的列表
    rod_lines = [None for _ in range(n_visualized_rods)]
    rod_scatters = [None for _ in range(n_visualized_rods)]
    for rod_idx in range(n_visualized_rods):
        inst_position,inst_radius= rod_history_unpacker(rod_idx, time_idx)
        inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
        rod_lines[rod_idx] = ax.plot(inst_position[0], inst_position[1], "r", lw=0.5)[0]

        #用来看数组的格式，在这里似乎都是1维的
        print(np.size(inst_position))
        print(np.size(inst_radius))


        rod_scatters[rod_idx] = ax.scatter(
            inst_position[0],
            inst_position[1],
            # s=np.pi * (scaling_factor * inst_radius) ** 2,
            s=np.pi * (scaling_factor * inst_radius) ** 2,
        )    
    ax.set_aspect("equal")

    with writer.saving(fig, video_name_xy, dpi):
        with plt.style.context('seaborn-v0_8-white'):
            for time_idx in tqdm(range(sim_time.shape[0])):
                for rod_idx in range(n_visualized_rods):
                    inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
                    inst_position = 0.5 * (
                        inst_position[..., 1:] + inst_position[..., :-1]
                    )

                    rod_lines[rod_idx].set_xdata(inst_position[0])
                    rod_lines[rod_idx].set_ydata(inst_position[1])


                    rod_scatters[rod_idx].set_offsets(inst_position[:2].T)
                    # rod_scatters[rod_idx].set_sizes([2] * len(rod_scatters[rod_idx].get_offsets()))
                    rod_scatters[rod_idx].set_sizes(np.pi * (scaling_factor * inst_radius) ** 2)   
                    
                writer.grab_frame()
    
    #plot xz
    # xz plot
    video_name_xz = "xz_" + video_name
    #列表推导式，用于生成一个长度为n_visualized_rods的列表
    rod_lines = [None for _ in range(n_visualized_rods)]
    rod_scatters = [None for _ in range(n_visualized_rods)]
    for rod_idx in range(n_visualized_rods):
        inst_position,inst_radius= rod_history_unpacker(rod_idx, time_idx)
        inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
        rod_lines[rod_idx] = ax.plot(inst_position[0], inst_position[1], "r", lw=0.5)[0]

        #用来看数组的格式，在这里似乎都是1维的
        print(np.size(inst_position))
        print(np.size(inst_radius))


        rod_scatters[rod_idx] = ax.scatter(
            inst_position[0],
            inst_position[2],
            # s=np.pi * (scaling_factor * inst_radius) ** 2,
            s=np.pi * (scaling_factor * inst_radius) ** 2,
        )    
    ax.set_aspect("equal")

    with writer.saving(fig, video_name_xz, dpi):
        with plt.style.context("seaborn-v0_8-white"):
            for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):

                for rod_idx in range(n_visualized_rods):
                    inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
                    inst_position = 0.5 * (
                        inst_position[..., 1:] + inst_position[..., :-1]
                    )

                    rod_lines[rod_idx].set_xdata(inst_position[0])
                    rod_lines[rod_idx].set_ydata(inst_position[2])

                    rod_scatters[rod_idx].set_offsets(
                        np.vstack((inst_position[0], inst_position[2])).T
                    )
                    rod_scatters[rod_idx].set_sizes(
                        np.pi * (scaling_factor * inst_radius) ** 2
                    )

                writer.grab_frame()

    
    # #plot yz
    # # yz plot
    # video_name_yz = "yz_" + video_name
    # #列表推导式，用于生成一个长度为n_visualized_rods的列表
    # rod_lines = [None for _ in range(n_visualized_rods)]
    # rod_scatters = [None for _ in range(n_visualized_rods)]
    # for rod_idx in range(n_visualized_rods):
    #     inst_position,inst_radius= rod_history_unpacker(rod_idx, time_idx)
    #     inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
    #     rod_lines[rod_idx] = ax.plot(inst_position[0], inst_position[1], "r", lw=0.5)[0]

    #     #用来看数组的格式，在这里似乎都是1维的
    #     print(np.size(inst_position))
    #     print(np.size(inst_radius))


    #     rod_scatters[rod_idx] = ax.scatter(
    #         inst_position[1],
    #         inst_position[2],
    #         # s=np.pi * (scaling_factor * inst_radius) ** 2,
    #         s=np.pi * (scaling_factor * inst_radius) ** 2,
    #     )    
    # ax.set_aspect("equal")

    # with writer.saving(fig, video_name_yz, dpi):
    #     with plt.style.context("seaborn-v0_8-white"):
    #         for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):

    #             for rod_idx in range(n_visualized_rods):
    #                 inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
    #                 inst_position = 0.5 * (
    #                     inst_position[..., 1:] + inst_position[..., :-1]
    #                 )

    #                 rod_lines[rod_idx].set_xdata(inst_position[1])
    #                 rod_lines[rod_idx].set_ydata(inst_position[2])

    #                 rod_scatters[rod_idx].set_offsets(
    #                     np.vstack((inst_position[0], inst_position[2])).T
    #                 )
    #                 rod_scatters[rod_idx].set_sizes(
    #                     np.pi * (scaling_factor * inst_radius) ** 2
    #                 )

    #             writer.grab_frame()

    # # Be a good boy and close figures
    # # https://stackoverflow.com/a/37451036
    # # plt.close(fig) alone does not suffice
    # # See https://github.com/matplotlib/matplotlib/issues/8560/
    plt.close(plt.gcf())



    # #用于了解数组形式，为什么inst_position，会出现（三维数组，）这样的一个格式为什么后面会空一个
    # print(rod_history_unpacker(0,0))
    # print('-----------------------------------')
    # # print(inst_position[0][:,0])
    # print(inst_position)
    # print(np.size(inst_position))


def plot_video(
    rods_history: Sequence[Dict],
    video_name="video.mp4",
    fps=60,
    step=1,
    **kwargs,
):
    plt.rcParams.update({"font.size": 22})

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
    print("plot scene visualization video")
    FFMpegWriter = animation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    dpi = kwargs.get("dpi", 100)

    xlim = kwargs.get("x_limits", (-1.0, 1.00))
    ylim = kwargs.get("y_limits", (-1.0, 1.00))
    zlim = kwargs.get("z_limits", (-0.0, 1.0))

    difference = lambda x: x[1] - x[0]
    max_axis_length = max(difference(xlim), difference(ylim))
    # The scaling factor from physical space to matplotlib space
    scaling_factor = (2 * 0.1) / max_axis_length  # Octopus head dimension
    scaling_factor *= 2.6e3  # Along one-axis

    fig = plt.figure(2, figsize=(10, 8), frameon=True, dpi=dpi)
    # ax = fig.add_subplot(111)
    ax = plt.axes(projection="3d")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    time_idx = 0
    rod_lines = [None for _ in range(n_visualized_rods)]
    rod_com_lines = [None for _ in range(n_visualized_rods)]
    rod_scatters = [None for _ in range(n_visualized_rods)]

    for rod_idx in range(n_visualized_rods):
        inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
        inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
        # rod_lines[rod_idx] = ax.plot(inst_position[0], inst_position[1],inst_position[2], "r", lw=0.5)[0]
        # inst_com = com_history_unpacker(rod_idx, time_idx)
        # rod_com_lines[rod_idx] = ax.plot(inst_com[0], inst_com[1],inst_com[2], "k--", lw=2.0)[0]
        
        rod_scatters[rod_idx] = ax.scatter(
            inst_position[2],
            inst_position[0],
            inst_position[1],
            s=np.pi * (scaling_factor * inst_radius) ** 2,
        )

    # ax.set_aspect("equal")
    title_name = kwargs.get("title", 'title')

    with writer.saving(fig, video_name, dpi):
        with plt.style.context('seaborn-v0_8-white'):
            for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):
                current_time = sim_time[time_idx]
                ax.set_title(f"{title_name}   Simulation Time: {current_time:.2f} seconds")  # Update title with current simulation time

                for rod_idx in range(n_visualized_rods):
                    inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
                    inst_position = 0.5 * (
                        inst_position[..., 1:] + inst_position[..., :-1]
                    )

                    # rod_lines[rod_idx].set_xdata(inst_position[0])
                    # rod_lines[rod_idx].set_ydata(inst_position[1])
                    # rod_lines[rod_idx].set_zdata(inst_position[2])

                    # com = com_history_unpacker(rod_idx, time_idx)
                    # rod_com_lines[rod_idx].set_xdata(com[0])
                    # rod_com_lines[rod_idx].set_ydata(com[1])
                    # rod_com_lines[rod_idx].set_zdata(com[2])

                    # rod_scatters[rod_idx].set_offsets(inst_position[:3].T)
                    rod_scatters[rod_idx]._offsets3d = (
                        inst_position[2],
                        inst_position[0],
                        inst_position[1],
                    )

                    rod_scatters[rod_idx].set_sizes(
                        np.pi * (scaling_factor * inst_radius) ** 2 * 0.1
                    )

                writer.grab_frame()

    # Be a good boy and close figures
    # https://stackoverflow.com/a/37451036
    # plt.close(fig) alone does not suffice
    # See https://github.com/matplotlib/matplotlib/issues/8560/
    plt.close(plt.gcf())








