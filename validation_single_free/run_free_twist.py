import os
import sys

# sys.settrace

import numpy as np

np.set_printoptions(precision=4)

from elastica.rod.knot_theory import compute_twist

from br2.environment import BatchEnvironment
from br2.constants import psi2Nm2

import matplotlib.pyplot as plt

DEBUG = True
EXPORT_BLENDER = False and DEBUG


def single_free_twist(
    actuation_list, rod_database_path, assembly_config_path, tag="twist_validation"
):
    # Actuation Profile

    # Prepare environment
    env = BatchEnvironment(run_tag=tag)
    for idx, actuation in enumerate(actuation_list):
        action = {"action1": actuation * psi2Nm2}
        env.build(
            action=action,
            rod_database_path=rod_database_path,
            assembly_config_path=assembly_config_path,
            verbose=DEBUG,
            prepend_tag=f"run_{idx}",
        )

    # Simulation
    status = env.run(
        duration=3.0,
        disable_progress_bar=not DEBUG,
    )
    if DEBUG:
        print(status)

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
    # )
    # env.debug_data()
    # env.save_data()

    # Terminate
    env.close()

    twists = []
    for key, rod_data in env.data_rods.items():
        positions = np.asarray(rod_data["position"]).copy()
        normals = np.asarray(rod_data["director"])[:, 0, :, :].copy()

        total_twist, local_twist = compute_twist(
            positions,  # shape (time, 3, n_nodes)
            normals,  # shape (time, 3, n_elems)
        )

        if np.all(np.isnan(total_twist)):
            twist = 0.0
        else:
            twist = total_twist[~np.isnan(total_twist)][-1]
        twists.append(twist)
    print(twists)

    return twists

    # Query position and director
    rod_data = env.data_rods[0]
    positions = np.asarray(rod_data["position"]).copy()
    normals = np.asarray(rod_data["director"])[:, 0, :, :].copy()

    total_twist, local_twist = compute_twist(
        positions,  # shape (time, 3, n_nodes)
        normals,  # shape (time, 3, n_elems)
    )
    # print(f"Total Twist: {total_twist[-1]}")
    # print(f"Local Twist: {local_twist[-1]}")

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

    if np.all(np.isnan(total_twist)):
        # print("All NaNs")
        return 0.0
    else:
        # print("twist: ", total_twist[~np.isnan(total_twist)])
        return total_twist[~np.isnan(total_twist)][-1]


if __name__ == "__main__":
    from loader import load_data

    actuations, twists_exp = load_data(
        "data/single_free_twist.csv", x_key="Actuation (psi)", y_key="Twist (turns)"
    )

    print(actuations)
    print(twists_exp)

    plt.scatter(actuations, twists_exp)
    plt.show()

    sys.exit()

    # DEBUG
    twists = single_free_twist(
        actuations,
        "rod_library/standard_18.json",
        "assembly/free_twist_18cm_60_0.json",
    )

    print(f"{twists_exp=}")
    print(f"{twists=}")

    plt.plot(actuations, twists_exp, "o", label="exp")
    plt.plot(actuations, twists, "o", label="sim")
    plt.xlabel("Actuation")
    plt.ylabel("Twist (turns)")
    plt.show()
