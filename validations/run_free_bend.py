import os
import sys

# sys.settrace

import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4)

from br2.environment import Environment
from br2.constants import psi2Nm2


def single_free_bend(
    actuation, rod_database_path, assembly_config_path, tag="bend_validation"
):
    # Actuation Profile
    action = {"action1": actuation * psi2Nm2}

    # Prepare environment
    env = Environment(run_tag=tag)
    env.reset(
        rod_database_path=rod_database_path,
        assembly_config_path=assembly_config_path,
    )

    # Simulation
    status = env.run(
        action=action,
        duration=5.0,
        check_nan=True,
        check_steady_state=True,
        disable_progress_bar=True,
    )
    # print(status)

    # "Pure" bending angle
    rod_data = env.data_rods[0]
    time = rod_data["time"]
    positions = np.asarray(rod_data["position"]).copy()  # (time, 3, n_nodes)
    tangents = positions[..., :-1] - positions[..., 1:]
    normalized_tangents = tangents / np.linalg.norm(tangents, axis=1)[:, None, :]
    angles = np.arccos(
        np.clip(
            (normalized_tangents[..., :-1] * normalized_tangents[..., 1:]).sum(axis=1),
            -1,
            1,
        )
    ).sum(axis=1)
    bend = angles / (2 * np.pi)
    lengths = np.linalg.norm(tangents, axis=1).sum(axis=1)

    # Post Processing
    # env.render_video(
    #     # The following parameters are optional
    #     x_limits=(-0.13, 0.13),  # Set bounds on x-axis
    #     y_limits=(-0.00, 0.5),  # Set bounds on y-axis
    #     z_limits=(-0.13, 0.13),  # Set bounds on z-axis
    #     dpi=100,  # Set the quality of the image
    #     vis3D=True,  # Turn on 3D visualization
    #     vis2D=True,  # Turn on projected (2D) visualization
    #     vis3D_director=False,
    #     vis2D_director_lastelement=False,
    # )
    # env.save_data()

    # Terminate
    env.close()

    # time = np.asarray(time)
    # plt.plot(time, bend)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Bending Angle (revolutions)")

    # # Second axis
    # ax2 = plt.gca().twinx()
    # ax2.plot(time, lengths, color="red")
    # ax2.set_ylabel("Total Length (m)", color="red")

    # plt.show()

    return bend, lengths


if __name__ == "__main__":
    bend, lengths = single_free_bend(
        30,
        "rod_library/standard_18.json",
        "assembly/free_bend_18cm_85_85.json",
    )
