import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('F:\\Soft_arm\\Code_br2\\BR2-simulator\\GJM\\2.9\\Disatance_Record.txt')
data = data.reshape(-1,)
print(np.shape(data))

plt.hist(data, bins=30, alpha=0.5, edgecolor='black',density=True)

plt.title('Histogram of delta L')  # 设置标题
plt.xlabel('delta L')  # 设置X轴标签
plt.ylabel('Frequency')  # 设置Y轴标签

plt.show()  # 显示图形