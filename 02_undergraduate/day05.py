#从数据反推模型——SciPy 曲线拟合与光刻参数标定
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==============================
# 1. 构造一组 dose-CD 数据
# ==============================
dose_data = np.array([0.80, 0.90, 1.00, 1.10, 1.20])
cd_data = np.array([42.0, 39.0, 36.0, 34.0, 33.0])

# ==============================
# 2. 定义拟合模型
# ==============================
# 这里先用一个简单的二次函数：
# CD = a*dose^2 + b*dose + c
def cd_model(dose, a, b, c):
    return a * dose**2 + b * dose + c

# ==============================
# 3. 用 curve_fit 拟合参数
# ==============================
params, params_cov = curve_fit(cd_model, dose_data, cd_data)

# 拟合得到的参数
a_fit, b_fit, c_fit = params

print("拟合得到的参数：")
print(f"a = {a_fit:.4f}")
print(f"b = {b_fit:.4f}")
print(f"c = {c_fit:.4f}")

# ==============================
# 4. 生成更密的 dose 点，用来画拟合曲线
# ==============================
dose_fine = np.linspace(0.80, 1.20, 300)
cd_fit = cd_model(dose_fine, a_fit, b_fit, c_fit)

# ==============================
# 5. 画图比较
# ==============================
plt.figure(figsize=(8, 5))
plt.scatter(dose_data, cd_data, label='original data')
plt.plot(dose_fine, cd_fit, label='fitted curve')
plt.xlabel("Dose")
plt.ylabel("CD")
plt.title("Dose-CD Curve Fitting")
plt.grid(True)
plt.legend()
plt.show()