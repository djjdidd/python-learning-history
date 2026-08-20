#least_squares 与“残差”概念
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# ==============================
# 1. 原始数据
# ==============================
dose_data = np.array([0.80, 0.90, 1.00, 1.10, 1.20])
cd_data = np.array([42.0, 39.0, 36.0, 34.0, 33.0])

# ==============================
# 2. 定义模型函数
# ==============================
def cd_model(dose, a, b, c):
    return a * dose**2 + b * dose + c

# ==============================
# 3. 定义残差函数
# ==============================
def residuals(params, x, y):
    a, b, c = params
    y_pred = cd_model(x, a, b, c)
    return y_pred - y

# ==============================
# 4. 给一个参数初值
# ==============================
initial_guess = np.array([1.0, 1.0, 1.0])

# ==============================
# 5. 调 least_squares
# ==============================
result = least_squares(residuals, x0=initial_guess, args=(dose_data, cd_data))

# ==============================
# 6. 取出最优参数
# ==============================
a_fit, b_fit, c_fit = result.x

print("least_squares 拟合得到的参数：")
print(f"a = {a_fit:.4f}")
print(f"b = {b_fit:.4f}")
print(f"c = {c_fit:.4f}")

# ==============================
# 7. 生成更密的点，画拟合曲线
# ==============================
dose_fine = np.linspace(0.80, 1.20, 300)
cd_fit = cd_model(dose_fine, a_fit, b_fit, c_fit)

# ==============================
# 8. 画图
# ==============================
plt.figure(figsize=(8, 5))
plt.scatter(dose_data, cd_data, label='original data')
plt.plot(dose_fine, cd_fit, label='least_squares fit')
plt.xlabel("Dose")
plt.ylabel("CD")
plt.title("Dose-CD Fitting by least_squares")
plt.grid(True)
plt.legend()
plt.show()