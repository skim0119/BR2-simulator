import numpy as np
import matplotlib.pyplot as plt

def main():
    data = np.load("F:\\Soft_arm\\Code_br2\\BR2-simulator\\GJM\\2.15\\tip_pose.npz")
    tip_pos = data["tip_pos"]#（3,j,i) j is the twist pressure; i is the bend pressure; may need to regenerate to have bend pressure in front
    
    x_tip_pos = np.abs(tip_pos[0,...])
    x_tip_pos = x_tip_pos.T
    y_tip_pos = 0.15-np.abs(tip_pos[1,...])
    y_tip_pos = y_tip_pos.T
    z_tip_pos = np.abs(tip_pos[2,...])
    z_tip_pos = z_tip_pos.T
   
    step = 5
    x = np.arange(-1,40,step)
    y = np.arange(-1,40,step)
    X, Y=np.meshgrid(x,y)
    
    plt.figure()
    plot_x = plt.pcolormesh(X,Y,x_tip_pos,cmap='Greens')
    plt.colorbar(plot_x)
    plt.title('tip_movement in x-axis', fontweight='bold')
    plt.xlabel("twist pressure(Pa)")
    plt.ylabel("bend pressure(Pa)")
    plt   
    
    plt.figure()
    plot_x = plt.pcolormesh(X,Y,y_tip_pos,cmap='Greens')
    plt.colorbar(plot_x)
    plt.title('tip_movement in y-axis', fontweight='bold')
    plt.xlabel("twist pressure(Pa)")
    plt.ylabel("bend pressure(Pa)")
    plt   
    
    plt.figure()
    plot_x = plt.pcolormesh(X,Y,z_tip_pos,cmap='Greens')
    plt.colorbar(plot_x)
    plt.title('tip_movement in z-axis', fontweight='bold')
    plt.xlabel("twist pressure(Pa)")
    plt.ylabel("bend pressure(Pa)")
    plt   
    
    plt.show()
    
    
    
if __name__ =="__main__":
    main()