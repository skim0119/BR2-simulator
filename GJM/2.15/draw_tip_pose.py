import numpy as np
import matplotlib.pyplot as plt

def main():
    data = np.load("F:\\Soft_arm\\Code_br2\\BR2-simulator\\GJM\\2.19\\tip_pose.npz")
    tip_pos = data["tip_pos"]#（3,j,i) j is the twist pressure; i is the bend pressure; may need to regenerate to have bend pressure in front
    rot_degrees = data["rot_degrees"]
    rot_degrees = rot_degrees.reshape(9,9)
    # print(rot_degrees[0,...]) to check if reshape is right
    
    x_tip_pos = np.abs(tip_pos[0,...])
    y_tip_pos = 0.15-np.abs(tip_pos[1,...])
    z_tip_pos = np.abs(tip_pos[2,...])
   
    step = 5
    x = np.linspace(0,1,9)
    y = np.linspace(0,1,9)
    X, Y=np.meshgrid(x,y)
    
    plt.figure()
    plot_x = plt.pcolormesh(X,Y,x_tip_pos,cmap='Greens')
    plt.colorbar(plot_x)
    plt.title('tip_movement in x-axis', fontweight='bold')
    plt.xlabel("bend pressure(normalized)")
    plt.ylabel("twist pressure(normalized)")
    plt   
    
    plt.figure()
    plot_x = plt.pcolormesh(X,Y,y_tip_pos,cmap='Greens')
    plt.colorbar(plot_x)
    plt.title('tip_movement in y-axis', fontweight='bold')
    plt.xlabel("bend pressure(normalized)")
    plt.ylabel("twist pressure(normalized)")
    plt   
    
    plt.figure()
    plot_x = plt.pcolormesh(X,Y,z_tip_pos,cmap='Greens')
    plt.colorbar(plot_x)
    plt.title('tip_movement in z-axis', fontweight='bold')
    plt.xlabel("bend pressure(normalized)")
    plt.ylabel("twist pressure(normalized)")
    plt   
    
    plt.figure()
    plot_x = plt.pcolormesh(X,Y,rot_degrees,cmap='Greens')
    plt.colorbar(plot_x)
    plt.title('twist angle under different pressure', fontweight='bold')
    plt.xlabel("bend pressure(normalized)")
    plt.ylabel("twist pressure(normalized)")
    plt  
    
    plt.show()
    
    
    
if __name__ =="__main__":
    main()