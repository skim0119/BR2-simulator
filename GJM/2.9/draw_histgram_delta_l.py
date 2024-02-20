import numpy as np
import matplotlib.pyplot as plt

def print_histgram(data):
    out_radius = 0.007522
    
    data = data.reshape(-1,)
    x = data / out_radius
    
    plt.figure()
    
    n, *_=plt.hist(x, bins=30, alpha=0.5, edgecolor='black')
    
    max_value = x.max()

    plt.title('Histogram of delta L(max_penetration=%f)'%max_value) 
    plt.xlabel('delta_L/R (R is the out_radius)')  
    plt.ylabel('Count')  
    plt.axvline(x=0.3, color='r', linestyle='--')
    
 
    plt.text(0.295, 0.95*max(n),'0.3', color='red', fontsize=14, verticalalignment='bottom', horizontalalignment='right')


    
def print_deltaL_3d(data):
    out_radius = 0.007522
    time = 1
    
    length_elem = data.shape[1]
    length = np.linspace(0,1,length_elem)
    time_steps = np.linspace(0,1000*time,data.shape[0])

    
    # define data on X,Y plane
    x,y = np.meshgrid(length, time_steps)
    z = data / out_radius
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(x,y,z,cmap='viridis')# viridis define the color
    
    ax.set_xlabel('Position/rod_length')
    ax.set_ylabel('time (ms)')
    ax.set_zlabel('delta_L/R (R is the out_radius)')
    

    

def main():
    data = np.loadtxt('F:\\Soft_arm\\Code_br2\\BR2-simulator\\GJM\\2.9\\Disatance_Record_only_twist.txt')
    print_histgram(data)
    print_deltaL_3d(data)
    plt.show()
    
    


if __name__ == "__main__":
    main()