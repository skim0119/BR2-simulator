import os
import sys

from dataclasses import dataclass

import tempfile
import numpy as np
from pydantic_core import from_json
import json
import matplotlib.pyplot as plt

np.set_printoptions(precision=4)

from br2.environment import BatchEnvironment
from br2.constants import psi2Nm2
from br2.configurations import RodLibrary

DEBUG = True
EXPORT_BLENDER = False and DEBUG


@dataclass
class RunConfig:
    length: float = 0.18
    actuation: float
    alpha: float
    beta: float


def single_free_bend(
    actuation_list,
    rod_database_path,
    assembly_config_path,
    alpha_list,
    beta_list,
    tag="validation",
):
    # Prepare environment
    env = BatchEnvironment(run_tag=tag)
    for idx, (actuation, alpha, beta) in enumerate(
        zip(actuation_list, alpha_list, beta_list)
    ):
        # Setup
        original_database_path = "rod_library/standard_18.json"
        with open(original_database_path, "r") as f:
            json_data = f.read()
        rod_library = RodLibrary.model_validate(from_json(json_data))

        rod_library.Rods["RodBend"]["alpha"] = alpha.item()
        rod_library.Rods["RodBend"]["beta"] = -beta.item()

        # Create new configuration
        new_json_data = rod_library.model_dump()
        tfile = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8")
        new_rod_database_path = tfile.name
        json.dump(new_json_data, tfile)
        tfile.flush()

        action = {"action1": actuation * psi2Nm2}
        env.build(
            action=action,
            rod_database_path=new_rod_database_path,
            assembly_config_path=assembly_config_path,
            verbose=DEBUG,
            prepend_tag=f"run_{idx}",
        )

    # Simulation
    status = env.run(
        duration=5.0,
        disable_progress_bar=not DEBUG,
    )
    # print(status)

    # Terminate
    env.close()

    # "Pure" bending angle
    bends = []
    lengths = []
    for key, rod_data in env.data_rods.items():
        # rod_data = env.data_rods[0]
        time = rod_data["time"]
        positions = np.asarray(rod_data["position"]).copy()  # (time, 3, n_nodes)
        tangents = positions[..., :-1] - positions[..., 1:]
        normalized_tangents = tangents / np.linalg.norm(tangents, axis=1)[:, None, :]
        angles = np.arccos(
            np.clip(
                (normalized_tangents[..., :-1] * normalized_tangents[..., 1:]).sum(
                    axis=1
                ),
                -1,
                1,
            )
        ).sum(axis=1)
        bend = angles
        length = np.linalg.norm(tangents, axis=1).sum(axis=1)

        bends.append(np.nanmax(bend, initial=0.0))
        lengths.append(length[-1])

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

    # time = np.asarray(time)
    # plt.plot(time, bend)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Bending Angle (revolutions)")

    # # Second axis
    # ax2 = plt.gca().twinx()
    # ax2.plot(time, lengths, color="red")
    # ax2.set_ylabel("Total Length (m)", color="red")

    # plt.show()

    return bends, lengths


if __name__ == "__main__":

    # Query data
    from loader import load_data

    actuations, bend_angle_experiment, info = load_data(
        "data/single_free_bend.csv",
        x_key="Actuation (psi)",
        y_key="Bend Angle (rad)",
        keys=["Fiber angles (alpha)", "Fiber angles (beta)", "Length (cm)"],
    )

    def filter_(length_list, actuation_list, alpha_list, beta_list, bend_angle_list):
        actuations = []
        alphas = []
        betas = []
        bend_angle = []
        for length, actuation, alpha, beta, angle in zip(
            length_list, actuation_list, alpha_list, beta_list, bend_angle_list
        ):
            if length != 18:
                continue
            if actuation > 36:
                continue
            if actuation < 14:
                continue
            actuations.append(actuation)
            alphas.append(alpha)
            betas.append(beta)
            bend_angle.append(angle)
        return actuations, alphas, betas, bend_angle

    actuations, alpha_list, beta_list, bend_angle = filter_(
        info["Length (cm)"],
        actuations,
        info["Fiber angles (alpha)"],
        info["Fiber angles (beta)"],
        bend_angle_experiment,
    )

    bends, lengths = single_free_bend(
        actuations,
        "rod_library/standard.json",
        "assembly/single.json",
        alpha_list,
        beta_list,
    )
    print(bends)
    print(lengths)

    angle_category = [65, 70, 75, 80, 85]
    fig, axes = plt.subplots(1, len(angle_category), figsize=(15, 5), sharey=True)
    for idx, angle in enumerate(angle_category):
        mask = angle == np.array(alpha_list)
        a = np.array(actuations)[mask]
        be = np.array(bend_angle)[mask]
        b = np.array(bends)[mask]

        axes[idx].scatter(a, b, label="Simulated")
        axes[idx].scatter(a, be, label="Experimental")
        axes[idx].set_title(f"Actuation: {angle}")
        axes[idx].set_xlabel("Actuation (psi)")
        axes[idx].set_ylabel("Bending Angle (radian)")
    plt.show()
