import os
import sys

# sys.settrace

import numpy as np

np.set_printoptions(precision=4)

from elastica.rod.knot_theory import compute_twist

from br2.environment import Environment
from br2.constants import psi2Nm2

import matplotlib.pyplot as plt


def single_free_twist(
    actuation, rod_database_path, assembly_config_path, tag="twist_validation"
):
    # Actuation Profile
    action = {"action1": actuation * psi2Nm2}

    # Prepare environment
    env = Environment(run_tag=tag)
    env.reset(
        rod_database_path=rod_database_path,
        assembly_config_path=assembly_config_path,
        verbose=False,
    )

    # Simulation
    status = env.run(
        action=action,
        duration=1.0,
        check_nan=True,
        check_steady_state=True,
        disable_progress_bar=True,
    )
    # print(status)

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

    #     visualize_twist_angle = True,
    # )
    # env.debug_data()
    # env.save_data()

    # Query position and director
    rod_data = env.data_rods[0]
    time = rod_data["time"]
    positions = np.asarray(rod_data["position"]).copy()
    normals = np.asarray(rod_data["director"])[:, 0, :, :].copy()

    # Terminate
    env.close()

    total_twist, local_twist = compute_twist(
        positions,  # shape (time, 3, n_nodes)
        normals,  # shape (time, 3, n_elems)
    )

    # Plot in 3D quiver
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # ax.quiver(
    #     positions[-1, 0, :-1],
    #     positions[-1, 1, :-1],
    #     positions[-1, 2, :-1],
    #     normals[-1, 0, :],
    #     normals[-1, 1, :],
    #     normals[-1, 2, :],
    #     length=0.02,
    # )
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")

    # print(f"Total Twist: {total_twist.shape}")
    # print(f"Local Twist: {local_twist.shape}")
    # print(total_twist)

    # plt.figure()
    # plt.plot(time, total_twist)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Total Twist (rad)")
    # plt.title(f"Total Twist vs Time (Actuation: {actuation} psi)")

    # plt.show()

    return total_twist[~np.isnan(total_twist)][-1]


if __name__ == "__main__":
    from loader import load_data

    actuations, twists_exp = load_data("data/single_free_twist.csv")

    actuation = 35
    twists = []
    for actuation in actuations:
        twist = single_free_twist(
            actuation,
            "rod_library/standard_18.json",
            "assembly/free_twist_18cm_60_0.json",
        )
        twists.append(twist)

    plt.plot(actuations, twists_exp, "o", label="exp")
    plt.plot(actuations, twists, "o", label="sim")
    plt.xlabel("Actuation")
    plt.ylabel("Twist (turns)")
    plt.show()
