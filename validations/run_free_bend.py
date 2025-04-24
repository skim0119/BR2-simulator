import os
import sys

# sys.settrace

import tempfile
import numpy as np
from pydantic_core import from_json
import json
import matplotlib.pyplot as plt

np.set_printoptions(precision=4)

from br2.environment import Environment
from br2.constants import psi2Nm2
from br2.configurations import RodLibrary


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
        duration=1.0,
        check_nan=True,
        check_steady_state=True,
        disable_progress_bar=False,
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
    bend = angles
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
    if True:
        # Query data
        from loader import load_data

        actuations, bend_angle, info = load_data(
            "data/single_free_bend.csv",
            x_key="Actuation (psi)",
            y_key="Bend Angle (rad)",
            keys=["Fiber angles (alpha)", "Fiber angles (beta)", "Length (cm)"],
        )

        # Setup configuration for simulation
        assembly_config_path = "assembly/free_bend.json"

        # Set New Configuration
        original_database_path = "rod_library/standard_18.json"
        with open(original_database_path, "r") as f:
            json_data = f.read()
        rod_library = RodLibrary.model_validate(from_json(json_data))

        # Run
        act = []
        exp = []
        sim = []
        for actuation, bend_angle_experimental, alpha, beta, length in zip(
            actuations,
            bend_angle,
            info["Fiber angles (alpha)"],
            info["Fiber angles (beta)"],
            info["Length (cm)"],
        ):
            if length != 18:
                continue
            if actuation > 36:
                continue
            if actuation < 14:
                continue
            rod_library.Rods["RodBend"]["alpha"] = alpha.item()
            rod_library.Rods["RodBend"]["beta"] = -beta.item()

            # Create new configuration
            new_json_data = rod_library.model_dump()
            tfile = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8")
            rod_database_path = tfile.name
            json.dump(new_json_data, tfile)
            tfile.flush()

            # Run simulation
            bend_angle_simulated, _ = single_free_bend(
                actuation,
                rod_database_path,
                assembly_config_path,
            )
            bend_sim = np.nanmax(bend_angle_simulated, initial=0.0)

            act.append(actuation)
            sim.append(bend_sim)
            exp.append(bend_angle_experimental)

        # Sort data
        # label = lambda a, b: f"αβ{a}"
        # labels = np.array([label(a, b) for a, b in zip(info["Fiber angles (alpha)"][mask], info["Fiber angles (beta)"][mask])])
        # unique_labels = np.unique(labels)

        # for label in unique_labels:
        #     mask = np.logical_and(np.logical_and(labels == label, actuations > 10), actuations <= 35)
        #     plt.scatter(actuations[mask], bend_angle[mask], label=label)
        for a, e, s in zip(act, exp, sim):
            plt.plot([a, a], [e, s], alpha=0.8, marker="o", color="black")
        plt.scatter(act, exp, label="Experimental", color="red")
        plt.scatter(act, sim, label="Simulated", color="blue")
        plt.legend()
        plt.xlabel("Actuation (psi)")
        plt.ylabel("Bending Angle (rad)")
        plt.show()
        plt.savefig("bend_validation.png")

        # Save data in npz
        np.savez("bend_validation.npz", act=act, exp=exp, sim=sim)

    # bend, lengths = single_free_bend(
    #     30,
    #     "rod_library/standard_18.json",
    #     "assembly/free_bend_18cm_85_85.json",
    # )
    # print(bend, lengths)
