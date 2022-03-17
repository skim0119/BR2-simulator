import os
import sys

# sys.settrace

import numpy as np

np.set_printoptions(precision=4)

from br2.environment import Environment

import argparse


def main():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, required=True)
    args = parser.parse_args()

    # Action Configuration
    psi2Nm2 = 6895

    # Actuation Profile
    action = {"action1": 35 * psi2Nm2, "action2": 0 * psi2Nm2}

    # Prepare environment
    env = Environment(
        run_tag=args.tag,
        final_time=simulation_time,
    )
    env.reset(
        rod_database_path="sample_database/sample_rod_library.json",
        assembly_config_path="sample_assembly/single_br2_v1.json",
    )

    # Simulation
    env.run(action=action, duration=20.0)

    # Post Processing
    env.render_video(
        filename_video="br2_simulation",
        save_folder=PATH,
        data_id=0,
        # The following parameters are optional
        x_limits=(-0.13, 0.13),  # Set bounds on x-axis
        y_limits=(-0.00, 0.5),  # Set bounds on y-axis
        z_limits=(-0.13, 0.13),  # Set bounds on z-axis
        dpi=100,  # Set the quality of the image
        vis3D=True,  # Turn on 3D visualization
        vis2D=True,  # Turn on projected (2D) visualization
        vis3D_director=False,
        vis2D_director_lastelement=False,
    )

    # Terminate
    env.close()


if __name__ == "__main__":
    main()
