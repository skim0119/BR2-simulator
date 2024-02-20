import matplotlib.pyplot as plt
import numpy as np

def main():
    y = np.loadtxt("F:\\Soft_arm\\Code_br2\\BR2-simulator\\GJM\\2.19\\rot_degree_without_bend.txt")
    x = range(0,200,5)

    plt.figure()

    plt.plot(x,y)

    # plt.title('Twist angle in different pressure(bend=40)')
    plt.xlabel('Pressure(Pa)')
    plt.ylabel('Twist angel(°)')

    plt.show()
    
if __name__ =='__main__':
    main()

