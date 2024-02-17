import numpy as np

data = np.load("F:\\Soft_arm\\Code_br2\\result_2_11\\data\\br2_data.npz")

print(np.shape(data["position_rod"]))
print(np.shape(data["external_force_rod_0"]))

# print(data.files)
# print(data["external_force_rod_0"])
# print("---------------------------------------")
# print(data["external_force_rod_1"])
# print("---------------------------------------")
# print(data["external_force_rod_2"])
