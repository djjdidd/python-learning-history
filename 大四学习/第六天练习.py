import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

dose_data = np.array([0.80, 0.90, 1.00, 1.10, 1.20])
cd_data = np.array([42.0, 39.0, 36.0, 34.0, 33.0])

def quad(dose, a, b, c):
    return a * dose**2 + b * dose + c

def linear(dose, a, b):
    return a * dose + b

def exp_func(dose, a, b):
    return a * np.exp(b * dose)

params_qu, _ = curve_fit(quad, dose_data, cd_data)
params_lin, _ = curve_fit(linear, dose_data, cd_data)
params_exp, _ = curve_fit(exp_func, dose_data, cd_data, p0=(40, -0.2))

cd_qua = quad(dose_data, *params_qu)# * 把一个序列里的元素拆开，一个一个传给函数
cd_lin = linear(dose_data, *params_lin)
cd_exp = exp_func(dose_data, *params_exp)

def rmse(y_ac, y_pr):
    y_ac = np.array(y_ac)
    y_pr = np.array(y_pr)
    return np.sqrt(np.mean((y_ac - y_pr)**2))

def r2(y_ac, y_pr):
    y_ac = np.array(y_ac)
    y_pr = np.array(y_pr)
    ss_res = np.sum((y_ac - y_pr)**2)
    ss_tot = np.sum((y_ac - np.mean(y_ac))**2)
    return 1 - ss_res / ss_tot

rmse_qua = rmse(cd_data, cd_qua)
rmse_lin = rmse(cd_data, cd_lin)
rmse_exp = rmse(cd_data, cd_exp)

r_qua = r2(cd_data, cd_qua)
r_lin = r2(cd_data, cd_lin)
r_exp = r2(cd_data, cd_exp)

print(f"一次函数模型 RMSE: {rmse_lin:.4f}")
print(f"二次函数模型 RMSE: {rmse_qua:.4f}")
print(f"指数函数模型 RMSE: {rmse_exp:.4f}")

print(f"一次函数模型 R²: {r_lin:.4f}")
print(f"二次函数模型 R²: {r_qua:.4f}")
print(f"指数函数模型 R²: {r_exp:.4f}")

plt.figure(figsize=(8, 5))
plt.scatter(dose_data, cd_data, label='original data')
plt.plot(dose_data, cd_qua, label='quadratic fit', linestyle='--')
plt.plot(dose_data, cd_lin, label='linear fit', linestyle='-')
plt.plot(dose_data, cd_exp, label='exp fit', linestyle='-', color='green')
plt.xlabel("dose")
plt.ylabel("cd")
plt.title("model fitting comparison")
plt.grid(True)
plt.legend()
plt.show()