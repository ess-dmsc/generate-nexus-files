def calculate_slit_edges(theta_list, phi_list):
    res = []
    for c in range(len(theta_list)):
        res.append(phi_list[c] - theta_list[c] / 2)
        res.append(phi_list[c] + theta_list[c] / 2)
    return res


# WFM1
theta_1 = [5.7, 9.0, 12.0, 14.9, 17.5, 20.0, ]
phi_1 = [91.93, 142.23, 189.40, 233.63, 275.10, 313.99, ]
slit_edges_1 = calculate_slit_edges(theta_1, phi_1)
print("WFM1")
print(slit_edges_1)
print("___________________________________")

# WFM2
theta_2 = [5.7, 9.0, 12.0, 14.9, 17.5, 20.0, ]
phi_2 = [97.67, 151.21, 201.41, 248.48, 292.62, 334.01, ]
slit_edges_2 = calculate_slit_edges(theta_2, phi_2)
print("WFM2")
print(slit_edges_2)
print("___________________________________")

# FOC1
theta_3 = [11.06, 13.06, 14.94, 16.71, 18.36, 16.72, ]
phi_3 = [81.12, 127.82, 171.60, 212.66, 251.16, 288.85, ]
slit_edges_3 = calculate_slit_edges(theta_3, phi_3)
print("FOC1")
print(slit_edges_3)
print("___________________________________")

# BPC1
theta_4 = [46.71, ]
phi_4 = [30.73, ]
slit_edges_4 = calculate_slit_edges(theta_4, phi_4)
print("BPC1")
print(slit_edges_4)
print("___________________________________")

# FOC2
theta_5 = [32.90, 33.54, 34.15, 34.37, 34.90, 34.31, ]
phi_5 = [106.57, 174.42, 238.04, 297.53, 353.48, 46.65, ]
slit_edges_5 = calculate_slit_edges(theta_5, phi_5)
print("FOC2")
print(slit_edges_5)
print("___________________________________")

# BPC2
theta_6 = [67.49, ]
phi_6 = [43.40, ]
slit_edges_6 = calculate_slit_edges(theta_6, phi_6)
print("BPC2")
print(slit_edges_6)
print("___________________________________")

# T0
theta_7 = [294.74, ]
phi_7 = [179.34, ]
slit_edges_7 = calculate_slit_edges(theta_7, phi_7)
print("T0")
print(slit_edges_7)
print("___________________________________")

# FOC3
theta_8 = [40.32, 39.61, 38.94, 38.31, 37.72, 36.05, ]
phi_8 = [92.47, 155.52, 214.65, 270.09, 322.08, 371.39, ]
slit_edges_8 = calculate_slit_edges(theta_8, phi_8)
print("FOC3")
print(slit_edges_8)
print("___________________________________")

# FOC4
theta_9 = [32.98, 31.82, 30.74, 29.72, 28.77, 26.76, ]
phi_9 = [61.17, 105.110, 146.32, 184.96, 221.19, 255.72, ]
slit_edges_9 = calculate_slit_edges(theta_9, phi_9)
print("FOC4")
print(slit_edges_9)
print("___________________________________")

# FOC5
theta_10 = [50.81, 48.54, 45.49, 41.32, 37.45, 37.74, ]
phi_10 = [81.94, 143.17, 200.11, 254.19, 304.47, 353.76, ]
slit_edges_10 = calculate_slit_edges(theta_10, phi_10)
print("FOC5")
print(slit_edges_10)

