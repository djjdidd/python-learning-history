import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

dose_data = np.array([0.80, 0.90, 1.00, 1.10, 1.20])
cd_data = np.array([42.0, 39.0, 36.0, 34.0, 33.0])

def cd_model1(dose, a, b, c):
    return a * dose**2 + b * dose + c
def cd_model2(dose, a, b):
    return a * dose + b

params1, params_co1 = curve_fit(cd_model1, dose_data, cd_data)
params2, params_co2 = curve_fit(cd_model2, dose_data, cd_data)
# 拟合得到的参数
a_fit, b_fit, c_fit = params1
a_fit1, b_fit1 = params2

dose_fine = np.linspace(0.80, 1.20, 300)
cd_fit1 = cd_model1(dose_fine, a_fit, b_fit, c_fit)
cd_fit2 = cd_model2(dose_fine, a_fit1, b_fit1)
print("二次函数拟合得到的参数：")
print(f"a = {a_fit:.4f}")
print(f"b = {b_fit:.4f}")
print(f"c = {c_fit:.4f}")
print("一次函数拟合得到的参数：")
print(f"a = {a_fit1:.4f}")
print(f"b = {b_fit1:.4f}")


plt.figure(figsize=(8, 5))
plt.scatter(dose_data, cd_data, label='original data')
plt.plot(dose_fine, cd_fit1, label='fitted curve1')
plt.plot(dose_fine, cd_fit2, label='fitted curve2')
plt.xlabel("Dose")
plt.ylabel("CD")
plt.title("Dose-CD Curve Fitting")
plt.grid(True)
plt.legend()
plt.show()

# 总结：
# 1. 对这组 dose-CD 数据来说，二次函数比一次函数更贴近原始数据。
# 2. 二次函数有额外的曲率项，因此更适合描述非线性变化趋势。
# 3. 如果数据有噪声，不能只看谁贴得更紧，还要考虑模型是否过于复杂。