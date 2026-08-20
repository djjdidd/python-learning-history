#用误差来评价拟合效果——RMSE 与 R² 比较模型好坏
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==============================
# 1. 原始数据
# ==============================
dose_data = np.array([0.80, 0.90, 1.00, 1.10, 1.20])
cd_data = np.array([42.0, 39.0, 36.0, 34.0, 33.0])

# ==============================
# 2. 定义两个模型
# ==============================
def quadratic_model(dose, a, b, c):
    return a * dose**2 + b * dose + c

def linear_model(dose, a, b):
    return a * dose + b

# ==============================
# 3. 拟合参数
# ==============================
params_quadratic, _ = curve_fit(quadratic_model, dose_data, cd_data)
params_linear, _ = curve_fit(linear_model, dose_data, cd_data)

# 计算拟合值
cd_fit_quadratic = quadratic_model(dose_data, *params_quadratic)
cd_fit_linear = linear_model(dose_data, *params_linear)

# ==============================
# 4. 计算误差指标
# ==============================
# 计算 RMSE
def rmse(y_actual, y_pred):
    return np.sqrt(np.mean((y_actual - y_pred) ** 2))

rmse_quadratic = rmse(cd_data, cd_fit_quadratic)
rmse_linear = rmse(cd_data, cd_fit_linear)

# 计算 R²
def r_squared(y_actual, y_pred):
    ss_total = np.sum((y_actual - np.mean(y_actual)) ** 2)
    ss_residual = np.sum((y_actual - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

r2_quadratic = r_squared(cd_data, cd_fit_quadratic)
r2_linear = r_squared(cd_data, cd_fit_linear)

# 打印 RMSE 和 R²
print(f"Quadratic Model RMSE: {rmse_quadratic:.4f}")
print(f"Linear Model RMSE: {rmse_linear:.4f}")
print(f"Quadratic Model R²: {r2_quadratic:.4f}")
print(f"Linear Model R²: {r2_linear:.4f}")

# ==============================
# 5. 画图比较
# ==============================
plt.figure(figsize=(8, 5))
plt.scatter(dose_data, cd_data, label='Original Data')
plt.plot(dose_data, cd_fit_quadratic, label='Quadratic Fit', linestyle='--')
plt.plot(dose_data, cd_fit_linear, label='Linear Fit', linestyle='-')
plt.xlabel("Dose")
plt.ylabel("CD")
plt.title("Model Fitting Comparison")
plt.grid(True)
plt.legend()
plt.show()