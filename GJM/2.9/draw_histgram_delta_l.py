import numpy as np
import matplotlib.pyplot as plt

def Print_Histgram(data):
    data = data.reshape(-1,)
    
    plt.figure()
    
    plt.hist(data, bins=30, alpha=0.5, edgecolor='black',density=True)

    plt.title('Histogram of delta L')  # 设置标题
    plt.xlabel('delta L')  # 设置X轴标签
    plt.ylabel('Frequency')  # 设置Y轴标签


    
def Print_deltaL_3d(data):
    n_elem = data.shape[1]
    length = np.linspace(0,1,n_elem)
    time_steps = np.arange(data.shape[0])
    
    # define data on X,Y plane
    X,Y = np.meshgrid(length, time_steps)
    Z = data
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X,Y,Z,cmap='viridis')# viridis define the color
    
    ax.set_xlabel('L')
    ax.set_ylabel('time')
    ax.set_zlabel('delta_L')
    

    

def main():
    data = np.loadtxt('F:\\Soft_arm\\Code_br2\\BR2-simulator\\GJM\\2.9\\Disatance_Record.txt')
    Print_Histgram(data)
    Print_deltaL_3d(data)
    plt.show()
    
    


if __name__ == "__main__":
    main()