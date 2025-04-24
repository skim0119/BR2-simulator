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
    action = {"action1": 90 * psi2Nm2}

    # Prepare environment
    env = Environment(
        time_step=1.0e-4, run_tag=args.tag, rendering_fps=60
    )  # , capture_interval=(0.5,0.8))
    env.reset(
        rod_database_path="database/rod_library.json",
        assembly_config_path="assembly/single_br2_twist.json",
        gravity=True,
        k_multiplier=1e0,
        k_repulsive=1e3,  # Default 2
        nu_multiplier=0.0000,  # Default 0
    )

    # Simulation
    status = env.run(
        action=action, duration=1.0, check_nan=True, check_steady_state=True
    )
    print(status)

    # Post Processing
    env.render_video(
        # The following parameters are optional
        x_limits=(-0.13, 0.13),  # Set bounds on x-axis
        y_limits=(-0.05, 0.5),  # Set bounds on y-axis
        z_limits=(-0.13, 0.13),  # Set bounds on z-axis
        dpi=100,  # Set the quality of the image
        vis3D=True,  # Turn on 3D visualization
        vis2D=True,  # Turn on projected (2D) visualization
        vis3D_director=False,
        vis2D_director_lastelement=False,
        visualize_twist_angle=True,
        max_fps=30,
    )
    env.debug_data()
    env.save_data()

    # Terminate
    env.close()


if __name__ == "__main__":
    main()
