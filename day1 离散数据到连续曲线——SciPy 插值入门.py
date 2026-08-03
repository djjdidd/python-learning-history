import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# -----------------------------
# 第一步：准备已知数据
# -----------------------------
# 这些数据可以理解为实验测出来的结果：
# 不同曝光剂量 dose 对应不同的线宽 CD
dose_data = np.array([0.80, 0.90, 1.00, 1.10, 1.20])
cd_data = np.array([42.0, 39.0, 36.0, 34.0, 33.0])

# -----------------------------
# 第二步：建立插值函数
# -----------------------------
# linear 表示线性插值：相邻点之间连直线
f_linear = interp1d(dose_data, cd_data, kind='linear')

# cubic 表示三次插值：曲线更平滑
f_cubic = interp1d(dose_data, cd_data, kind='cubic')

# -----------------------------
# 第三步：查询中间某个点
# -----------------------------
# 假设我们想知道 dose=0.95 时，CD 大概是多少
dose_query = 0.95

cd_linear = f_linear(dose_query)
cd_cubic = f_cubic(dose_query)

print("当 dose =", dose_query, "时：")
print("线性插值估计 CD =", float(cd_linear))
print("三次插值估计 CD =", float(cd_cubic))

# -----------------------------
# 第四步：为了画平滑曲线，生成更多中间点
# -----------------------------
dose_fine = np.linspace(0.80, 1.20, 200)

# 在这些更密的点上做插值
cd_fine_linear = f_linear(dose_fine)
cd_fine_cubic = f_cubic(dose_fine)

# -----------------------------
# 第五步：画图
# -----------------------------
plt.figure(figsize=(8, 5))

# 原始数据点
plt.scatter(dose_data, cd_data, label='original data')

# 线性插值曲线
plt.plot(dose_fine, cd_fine_linear, label='linear interpolation')

# 三次插值曲线
plt.plot(dose_fine, cd_fine_cubic, label='cubic interpolation')

plt.xlabel("Dose")
plt.ylabel("CD")
plt.title("Interpolation Example for Lithography")
plt.grid(True)
plt.legend()
plt.show()
