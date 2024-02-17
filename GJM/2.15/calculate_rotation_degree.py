import numpy as np
from knot_theory import compute_twist



def calculate_degrees():

    data = np.load("F:\\Soft_arm\\Code_br2\\BR2-simulator\\result_2_16_extreme_test\\data\\br2_data.npz")
    # data = np.load("F:\\Soft_arm\\Code_br2\\result_2_15\\data\\br2_data.npz")

    director_total = data["director_rod_0"][-1,...]

    theta = 0
    rotation_degree = 0

    #not sure it should be done from left to right or opposite
    for i in range(0,director_total.shape[2]-1,1):
        director_1 = director_total[:,:,i]
        director_2 = director_total[:,:,i+1]
        rot_matrix = cal_rotate(director_1,director_2)
        theta = np.arccos((np.trace(rot_matrix)-1)/2)
        rotation_degree = rotation_degree + theta

    

    print(np.degrees(rotation_degree))
    
    return np.degrees(rotation_degree)

def calculate_twist_only():
    data = np.load("F:\\Soft_arm\\Code_br2\\BR2-simulator\\result_2_16_extreme_test\\data\\br2_data.npz")
    # data = np.load("F:\\Soft_arm\\Code_br2\\result_2_15\\data\\br2_data.npz")
    
    director_total = data["director_rod_0"]

    normal_total = np.array(director_total[:,0,...])
    binormal_total = np.array(director_total[:,1,...])

    
    center_line = np.zeros((26,3,41))
    center_line[:,0,:] = 1
    center_line[:,1,:] = 0
    center_line[:,2,:] = 0
    # center_line = np.dstack((center_line,binormal_total))
        
    
    # print(np.shape(center_line))
    
    total_twist,local_twist=compute_twist(center_line,normal_total)
    print(np.shape(total_twist))
    print(np.shape(local_twist))
    
    total_twist_degree = np.degrees(total_twist[-1])
    local_twist_degree = np.degrees(np.sum(local_twist[-1,:]))
    
    print(total_twist_degree)
    print(local_twist_degree)
    
    return  total_twist,local_twist
        

def cal_rotate(R1,R2):
    rot_matrix = R1.T @ R2
    return rot_matrix

def main():
    """test cal_rotate
    R1 = np.array([[0, 0, 1],
               [0, 1, 0],
               [1, 0, 0]])

    R2 = np.array([[1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]])
    rot_matrix=cal_rotate(R1,R2)
    print(np.degrees(np.arccos((np.trace(rot_matrix)-1)/2)))
    """
    
    calculate_degrees()
    calculate_twist_only()
    
if __name__ == "__main__" :
    main()