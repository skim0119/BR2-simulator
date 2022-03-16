import multiprocessing
from multiprocessing import Process, Pool
import subprocess
from subprocess import call
import os
import sys
import copy

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from functools import partial
from tqdm import tqdm

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#sys.settrace
from br2.set_environment_br2 import (
    Environment,
)

PATH = 'result_br2_17'

np.set_printoptions(precision=4)

SAVE_PATH = os.path.join(PATH, 'rod_save')
os.makedirs(PATH, exist_ok=1)
os.makedirs(SAVE_PATH, exist_ok=1)

# Downsampling
target_fps = 200  # Targeted frames per second: comparison made at each frames
simulation_time = 3.0  # total simulation time
simulation_frames = int(simulation_time * target_fps)
print(f'simulation time (sec): {simulation_time}')
print(f'simulation frames: {simulation_frames}')

pbar_update_interval = 5000
save_interval = 500000
check_nan_interval = 100
check_steady_state_interval = 100
check_nan_type = 2 # 1: simple check and stop. 2: print location of nan
check_steady_state_type = 2 # 1: maximum velocity stopper. 2: steady state check

def main(action):

    # Set simulation final time
    RENDER = 1
    DEBUG = 1

    env = Environment(
        final_time=simulation_time,
        fps=target_fps,
        COLLECT_DATA_FOR_POSTPROCESSING=True,
    )

    # Reset the environment before the new episode and get total number of simulation steps
    total_steps = env.reset(
        rod_database_path='sample_database/sample_rod_library.json',
        assembly_config_path='sample_assembly/single_br2_v1.json'
    )
    print(f'Total simulation steps: {total_steps}')

    # Simulation loop starts
    user_defined_condition = False
    reward = 0.0
    done = False

    # Simulation
    if DEBUG:
        pos = [] # TODO: debug
        vel = [] # TODO: debug
        acc = [] # TODO: debug
        dir = [] # TODO: debug
        ome = [] # TODO: debug
        alp = [] # TODO: debug
        tmp = [] # TODO: debug
        int_forc = []
        int_torq = []
        dam_forc = []
        dam_torq = []
        dilatati = []

    with tqdm(total=simulation_time) as pbar:
        prev_time = 0
        time = np.float64(0.0)
        i_sim = 0
        #for i_sim in range(1, simulation_frames):
        while not done:
            i_sim += 1

            # Ramp Action
            _action = action.copy()
            _action['action1'] = min(time, 1.0) * _action['action1']

            # Simulation
            check_nan = check_nan_type if i_sim % check_nan_interval == 0 else 0
            check_steady_state = check_steady_state_type if i_sim % check_steady_state_interval == 0 else 0
            time, reward, done, info = env.step(
                _action,
                time,
                check_nan=check_nan,
                check_steady_state=check_steady_state
            )
            if DEBUG and check_steady_state:
                pos.append(info['position_delta'])
                vel.append(info['velocity_delta'])
                acc.append(info['acceleration_delta'])
                dir.append(info['director_delta'])
                ome.append(info['omega_delta'])
                alp.append(info['alpha_delta'])
                tmp.append(time)
                int_torq.append(np.nanmax(np.linalg.norm(info['system']['seg1_0_RodBend'].internal_torques, axis=0)))
                int_forc.append(np.nanmax(np.linalg.norm(info['system']['seg1_0_RodBend'].internal_forces, axis=0)))
                dam_torq.append(np.nanmax(np.linalg.norm(info['system']['seg1_0_RodBend'].damping_torques, axis=0)))
                dam_forc.append(np.nanmax(np.linalg.norm(info['system']['seg1_0_RodBend'].damping_forces, axis=0)))
                dilatati.append(np.nanmax(info['system']['seg1_0_RodBend'].dilatation))

            # Progress bar update
            if (i_sim + 1) % pbar_update_interval == 0:
                pbar.update(time - prev_time)
                pbar.set_description("Processing ({}/{})"
                        .format(i_sim, total_steps))
                prev_time = time

            # Post-processing
            if done or i_sim % save_interval == save_interval-1:
                # Make a video of octopus for current simulation episode. Note that
                # in order to make a video, COLLECT_DATA_FOR_POSTPROCESSING=True
                #env.save_data(os.path.join(PATH, f'damping/{lognu}.npz'))
                env.save_data(os.path.join(PATH, f'data.npz'))
                if RENDER:
                    env.post_processing(
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

            # If done=True, NaN detected in simulation.
            # Exit the simulation loop before, reaching final time
            if done:
                print(f" Episode finished after {time=} {i_sim=}")
                break
    # Simulation loop ends
    print("Final time of simulation is : ", time)
    for key, msg in info.items():
        print(f'{key}: {msg}')


    # DEBUG
    if DEBUG:
        plt.figure()
        plt.semilogy(tmp, pos, label='pos')
        plt.semilogy(tmp, vel, label='vel')
        plt.semilogy(tmp, acc, label='acc')
        plt.semilogy(tmp, dir, label='dir')
        plt.semilogy(tmp, ome, label='ome')
        plt.semilogy(tmp, alp, label='alp')
        plt.legend(loc='upper left')

        plt.figure()
        plt.semilogy(tmp, int_forc, label='int_F')
        plt.semilogy(tmp, int_torq, label='int_T')
        plt.legend(loc='upper left')

        plt.figure()
        plt.semilogy(tmp, dam_forc, label='damp_F')
        plt.semilogy(tmp, dam_torq, label='damp_T')
        plt.legend(loc='upper left')

        plt.figure()
        plt.plot(env.data_rods[0]['time'],
                 np.array(env.data_rods[0]['position'])[:,1,-1] - 0.18,
                 label='seg1')
        plt.plot(env.data_rods[3]['time'],
                 np.array(env.data_rods[3]['position'])[:,1,-1] - 0.18*2,
                 label='seg2')
        plt.plot(env.data_rods[6]['time'],
                 np.array(env.data_rods[6]['position'])[:,1,-1] - 0.18*3,
                 label='seg3')
        plt.legend()
        plt.show()


if __name__ == "__main__":
    psi2Nm2 = 6895

    # Actuation Profile
    # maxBend = 35.0 * psi2Nm2
    # maxTorque = 10.0 * psi2Nm2
    action = {
        "action1": 50 * psi2Nm2,
        "action2": 0 * psi2Nm2
    }

    # run
    main(action)

    '''
    # test nu(deprecated)
    testrange = np.linspace(np.log10(0.01),np.log10(0.30), 10)
    for lognu in testrange:
        #lognu = np.log10(0.10)
        main(action, lognu)
        #break

    nus = []
    for lognu in testrange:
        nu = 10 ** lognu
        nus.append(nu)

        data = np.load(os.path.join(PATH, f'damping/{lognu}.npz'))
        time = data['time']
        tip_position = data['position_rod'][:, 1, -1]
        plt.plot(time, tip_position)
        print(tip_position[-1])

    plt.legend(nus)
    plt.show()
    '''
