import numpy as np

def Count_Distance(distance):
    distance = np.array(distance)
    with open('F:\\Soft_arm\\Code_br2\\BR2-simulator\\GJM\\2.9\\Disatance_Record_only_twist.txt','a') as file:
        file.writelines([str(d)+' ' for d in distance])
        file.write('\n')