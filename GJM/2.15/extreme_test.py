import os
import sys

# sys.settrace
sys.path.append('f:\\Soft_arm\\Code_br2\\BR2-simulator')
# print(sys.path)

import br2
# print(br2.__file__)

import numpy as np

np.set_printoptions(precision=4)

from br2.environment import Environment

import argparse

from calculate_rotation_degree import calculate_degrees

from tqdm import tqdm


def main():
    # Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, required=True)
    args = parser.parse_args()

    # Action Configuration
    psi2Nm2 = 6895
    
    # Prepare environment4
    env = Environment(run_tag=args.tag, time_step=1e-5)#change dt to 1.5e-5, original magnitude is 2e-5
    
    #initial twist pressure and rotation_degree
    rot_degrees= []
    # Actuation Profile
    for j in tqdm(range(0,200,5)):
        rot_degrees_prev = rot_degrees
        action = {"action1": 40 * psi2Nm2, "action2": j * psi2Nm2} #0-40 range for pressure


        env.reset(
            rod_database_path="F:\\Soft_arm\\Code_br2\\BR2-simulator\\GJM\\2.11\\rod_library_GJM.json",
            assembly_config_path="F:\\Soft_arm\\Code_br2\\BR2-simulator\\GJM\\2.11\\single_br2_GJM.json",
            gravity=True
        )

        # Simulation
        status = env.run(action=action, duration=1.0, check_nan=True, check_steady_state=True)
        print(status)
        
        env.save_data()
        
        degrees = calculate_degrees()
        
        rot_degrees = np.append(rot_degrees,degrees)
        
    print(rot_degrees)

    # Post Processing
    env.render_video(
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
    
    rot_degrees = np.array(rot_degrees)
    
    with open('F:\\Soft_arm\\Code_br2\\BR2-simulator\\GJM\\2.15\\rot_degree_bend.txt','w') as file:
        file.writelines([str(d)+' ' for d in rot_degrees])
        file.write('\n')


if __name__ == "__main__":
    main()
