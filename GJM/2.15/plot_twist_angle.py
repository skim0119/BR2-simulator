import matplotlib.pyplot as plt
import numpy as np

def main():
    y = np.loadtxt("F:\\Soft_arm\\Code_br2\\BR2-simulator\\GJM\\2.19\\rot_degree_without_bend.txt")
    x = range(0,200,5)
    y_max = np.max(np.abs(y))
    y_under_40 = y[0:8]
    y_under_40_max = np.max(y_under_40)

    plt.figure()

    plt.plot(x,y)

    plt.title(f'Max Twist Angle={y_max:.2f}°\n Max Twist Angle from 0-40pa={y_under_40_max:.2f}°')
    plt.xlabel('Pressure(Pa)')
    plt.ylabel('Twist angel(°)')

    plt.show()
    
if __name__ =='__main__':
    main()

