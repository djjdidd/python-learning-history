import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
dose_data =np.array( [0.80, 0.90, 1.00, 1.10, 1.20])
cd_data =np.array( [42.0, 39.0, 36.0, 34.0, 33.0])
f_linear=interp1d(dose_data,cd_data,kind='linear')
f_cubic=interp1d(dose_data,cd_data,kind='cubic')
cd_1=f_linear(0.95)
cd_2=f_linear(1.05)
cd_3=f_linear(1.15)
dose_fine=np.array([0.95,1.05,1.15])
cd_fine=([37.5,35.0,33.5])#cd_fine = f_linear(dose_fine)这行写错了
print("dose = 0.95 时，CD = ",float(cd_1))
print("dose = 1.05 时，CD = ",float(cd_2))
print("dose = 1.15 时，CD = ",float(cd_3))
dose_fin=np.linspace(0.8,1.2,500)
cd_fin=f_linear(dose_fin)
cd_fin1=f_cubic(dose_fin)
plt.figure()
plt.scatter(dose_data, cd_data, label='original data')#原始数据点
plt.plot(dose_fin, cd_fin, label='linear interpolation')#插值数据线
plt.plot(dose_fin, cd_fin1, label='cubic interpolation')
plt.xlabel("Dose")
plt.ylabel("CD")
plt.title("Interpolation Example for Lithography")
plt.grid(True)
plt.legend()
plt.show()