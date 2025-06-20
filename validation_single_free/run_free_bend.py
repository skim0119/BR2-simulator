import os
import sys

from dataclasses import dataclass

import tempfile
import numpy as np
from pydantic_core import from_json
import json
import matplotlib.pyplot as plt

np.set_printoptions(precision=4)

from elastica.rod.knot_theory import compute_twist
from br2.environment import BatchEnvironment
from br2.constants import psi2Nm2
from br2.configurations import RodLibrary

DEBUG = True


@dataclass
class RunConfig:
    actuation: float
    alpha: float
    beta: float
    length: float = 0.18
    gamma: float | None = None

    @classmethod
    def zip(cls, length_list, actuation_list, alpha_list, beta_list, gamma_list = None):
        return [
            cls(length=length, actuation=actuation, alpha=alpha, beta=beta, gamma=gamma)
            for length, actuation, alpha, beta, gamma in zip(length_list, actuation_list, alpha_list, beta_list, gamma_list)
        ]


def single_free_bend(
    configs: list[RunConfig],
    assembly_config_path: str,
    tag: str = "validation",
    plot: bool = False,
):
    # Prepare environment
    timestep = 4.0e-5
    env = BatchEnvironment(run_tag=tag, time_step=timestep, rendering_fps=1)
    for idx, config in enumerate(configs):
        # Setup
        original_database_path = f"rod_library/standard.json"
        with open(original_database_path, "r") as f:
            json_data = f.read()
        rod_library = RodLibrary.model_validate(from_json(json_data))

        rod_library.Rods["Rod"]["alpha"] = config.alpha
        rod_library.Rods["Rod"]["beta"] = config.beta
        rod_library.Rods["Rod"]["base_length"] = config.length
        rod_library.Rods["Rod"]["gamma"] = config.gamma

        # Create new configuration
        new_json_data = rod_library.model_dump()
        tfile = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8")
        new_rod_database_path = tfile.name
        json.dump(new_json_data, tfile)
        tfile.flush()

        action = {"action": config.actuation * psi2Nm2}
        env.build(
            action=action,
            prepend_tag=f"run_{idx}",
            rod_database_path=new_rod_database_path,
            assembly_config_path=assembly_config_path,
            verbose=DEBUG,
        )

    # Simulation
    # Run until the position is stable
    status = env.run(
        duration=5.0,
        disable_progress_bar=not DEBUG,
    )

    bends = []
    twists = []
    lengths = []

    for k, rod_data in env.data_rods.items():
        fig, axes = plt.subplots(3, 3, figsize=(10, 8), sharex=True)

        for tidx, time in enumerate(rod_data["time"]):
            positions_at_time = rod_data["position"][tidx] # (3, n_nodes)

            axes[0, 0].plot(positions_at_time[0, :], label=f"t={time:.1f}s")
            axes[0, 1].plot(positions_at_time[1, :], label=f"t={time:.1f}s")
            axes[0, 2].plot(positions_at_time[2, :], label=f"t={time:.1f}s")

            tangents = positions_at_time[..., :-1] - positions_at_time[..., 1:]
            normalized_tangents = tangents / np.linalg.norm(tangents, axis=0, keepdims=True)
            angles = np.arccos(
                np.clip(
                    (normalized_tangents[..., :-1] * normalized_tangents[..., 1:]).sum(
                        axis=0
                    ),
                    -1,
                    1,
                )
            )
            bend = angles.sum()
            dilatations = np.asarray(rod_data["dilatation"])[tidx, :].copy()
            total_dilatation = dilatations.mean()

            normals_at_time = np.asarray(rod_data["director"])[tidx, 0, :, :].copy()

            total_twist, local_twist = compute_twist(
                positions_at_time[None, ...],  # shape (time, 3, n_nodes)
                normals_at_time[None, ...],  # shape (time, 3, n_elems)
            )
            axes[1, 0].plot(dilatations, label=f"t={time:.1f}s")
            axes[1, 1].plot(angles, label=f"t={time:.1f}s")
            axes[1, 2].plot(local_twist[0], label=f"t={time:.1f}s")

            axes[2, 0].plot(rod_data["alpha_angle"][tidx], label=f"t={time:.1f}s")
            axes[2, 1].plot(rod_data["beta_angle"][tidx], label=f"t={time:.1f}s")
            axes[2, 2].plot(rod_data["delta_turn"][tidx], label=f"t={time:.1f}s")

        axes[2, 0].set_xlabel("Node")
        axes[2, 1].set_xlabel("Node")
        axes[2, 2].set_xlabel("Node")

        axes[0, 0].legend()
        axes[0, 1].legend()
        axes[0, 2].legend()
        axes[1, 0].legend()
        axes[1, 1].legend()
        axes[1, 2].legend()
        axes[2, 0].legend()
        axes[2, 1].legend()
        axes[2, 2].legend()

        axes[0, 0].set_ylabel("px")
        axes[0, 1].set_ylabel("py")
        axes[0, 2].set_ylabel("pz")

        axes[1, 0].set_ylabel("Dilatation")
        axes[1, 1].set_ylabel("Bend")
        axes[1, 2].set_ylabel("Twist")

        axes[2, 0].set_ylabel("Alpha")
        axes[2, 1].set_ylabel("Beta")
        axes[2, 2].set_ylabel("Delta Turn")

        # TITLE
        fig.suptitle(f"Rod {k} (bend={bend*180/np.pi:.1f} deg, twist={total_twist[0]*180/np.pi:.1f} deg, dilatation={total_dilatation:.2e})")

        bends.append(bend)
        twists.append(total_twist[0])
        lengths.append(dilatations.sum())
    if plot:
        plt.show()
    plt.close('all')

    # Terminate
    env.close()
    return bends, twists, lengths

if __name__ == "__main__":
    actuations = [5, 6, 8]  # np.linspace(0, 50, 10) * 2/50
    alpha_list = [60] * len(actuations)
    beta_list = [0] * len(actuations)
    gamma_list = [None] * len(actuations)
    lengths = [0.18] * len(actuations)

    configs = RunConfig.zip(lengths, actuations, alpha_list, beta_list, gamma_list)

    bends, twists, lengths = single_free_bend(
        configs,
        "assembly/single.json",
        tag="validation_single_free_configs",
        plot=True,
    )

    plt.plot(actuations,  bends, label="bends")
    plt.plot(actuations,  twists, label="twists")
    plt.legend()
    plt.show()
