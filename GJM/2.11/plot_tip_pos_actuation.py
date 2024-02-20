import os
import sys

# sys.settrace
sys.path.append('f:\\Soft_arm\\Code_br2\\BR2-simulator')

import br2
import numpy as np
np.set_printoptions(precision=4)

from br2.environment import Environment
from elastica.rod.knot_theory import compute_twist
import argparse

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
    
    #initial tip_pos relating array
    tip_pos_row = []
    tip_pos_total = [] 
    rot_degrees = []  
    
    # Actuation Profile
    for i in tqdm(range(0,41,5)):#include 40
        for j in range(0,41,5):
            action = {"action1": i * psi2Nm2, "action2": j * psi2Nm2} #0-40 range for pressure


            env.reset(
                rod_database_path="F:\\Soft_arm\\Code_br2\\BR2-simulator\\GJM\\2.11\\rod_library_GJM.json",
                assembly_config_path="F:\\Soft_arm\\Code_br2\\BR2-simulator\\GJM\\2.11\\single_br2_GJM.json",
                gravity=True,
                k_multiplier=1.0,
            )

            # Simulation
            status = env.run(action=action, duration=1.0, check_nan=True, check_steady_state=True)
            print(status)
            
            env.save_data()
            
            #save the tip position
            data = np.load("F:\\Soft_arm\\Code_br2\\BR2-simulator\\result_2_19_tip_pos\\data\\br2_data.npz") #tip pos relating to bend/twist pressure will be stocked here

            tip_pos_temp = data["position_rod_0"][-1,...]#just have the last step tip position
            tip_pos_temp = tip_pos_temp.reshape([3,1])
            
            if j==0:
                tip_pos_row = tip_pos_temp
            else:
                tip_pos_row = np.hstack((tip_pos_row,tip_pos_temp))#make its shape be [3,j]    
                
            #calculate the rotation angle
            director_total = data["director_rod_0"]
            center_line = data["center_line"]
            normal_total = np.array(director_total[:,0,...])
            rot_degrees_temp,_ = compute_twist(center_line,normal_total)
            rot_degrees_temp = np.degrees(rot_degrees_temp[-1])
            rot_degrees = np.append(rot_degrees,rot_degrees_temp)
            # Terminate
            env.close()
        
        #save it in the first dimension of tip_pos_total
        if i==0:
            tip_pos_total = tip_pos_row
        else:
            tip_pos_total = np.dstack((tip_pos_total,tip_pos_row))#make its shape be [3,j,i]
    
    #save it to npz filw        
    np.savez("F:\\Soft_arm\\Code_br2\\BR2-simulator\\GJM\\2.19\\tip_pose.npz", 
             tip_pos = tip_pos_total,
             rot_degrees = rot_degrees)

    
    
    
    
if __name__ == "__main__":
    main()