# 第18天：目标函数与自动找最优 dose

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy import optimize

# ==============================
# 1. 建立二维空间网格
# ==============================
x = np.linspace(-1.0, 1.0, 201)
y = np.linspace(-1.0, 1.0, 201)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)

# ==============================
# 2. 构造两个相邻矩形 mask
# ==============================
left_rect = ((X >= -0.45) & (X <= -0.10) & (np.abs(Y) <= 0.25))
right_rect = ((X >= 0.10) & (X <= 0.45) & (np.abs(Y) <= 0.25))
mask_2d = np.where(left_rect | right_rect, 1.0, 0.0)

# ==============================
# 3. 构造二维高斯 PSF
# ==============================
sigma = 0.08
psf_2d = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
psf_2d = psf_2d / np.sum(psf_2d)

# ==============================
# 4. 二维卷积得到 aerial image
# ==============================
aerial_2d = convolve2d(mask_2d, psf_2d, mode='same')

# ==============================
# 5. 固定 threshold
# ==============================
threshold = 0.24

# ==============================
# 6. 定义函数：给定 dose，返回打印图形
# ==============================
def get_printed_pattern(dose):
    printed_2d = (dose * aerial_2d >= threshold).astype(float)
    return printed_2d

# ==============================
# 7. 定义函数：提取中间横截面 CD
# ==============================
def extract_cd_from_middle_row(binary_2d, x_coords):
    row_mid = binary_2d.shape[0] // 2
    profile = binary_2d[row_mid, :]

    printed_indices = np.where(profile == 1)[0]

    if len(printed_indices) == 0:
        return 0.0

    left_edge = x_coords[printed_indices[0]]
    right_edge = x_coords[printed_indices[-1]]
    cd_value = right_edge - left_edge
    return cd_value

# ==============================
# 8. 设定目标 CD
# ==============================
target_cd = 1.00

# ==============================
# 9. 定义目标函数
# ==============================
def objective(dose_array):
    # optimize.minimize 传进来的通常是数组，即使只有一个变量
    dose = dose_array[0]

    # 如果 dose <= 0，就给一个很大的惩罚值
    if dose <= 0:
        return 1e6

    # 得到当前 dose 下的打印图形
    printed_2d = get_printed_pattern(dose)

    # 测量当前 dose 下的中间横截面 CD
    cd_mid = extract_cd_from_middle_row(printed_2d, x)

    # 目标函数：让 CD 尽量接近 target_cd
    value = (cd_mid - target_cd) ** 2

    return value

# ==============================
# 10. 用 optimize.minimize 自动找最优 dose
# ==============================
initial_guess = np.array([1.00])

result = optimize.minimize(
    objective,
    x0=initial_guess,
    method='Nelder-Mead'
)

best_dose = result.x[0]
best_printed = get_printed_pattern(best_dose)
best_cd = extract_cd_from_middle_row(best_printed, x)

print("最优 dose =", best_dose)
print("对应的中间横截面 CD =", best_cd)
print("目标 CD =", target_cd)
print("目标函数最小值 =", result.fun)#result.fun就是最优值

# ==============================
# 11. 再额外做一组手动扫描，方便和优化结果比较
# ==============================
dose_scan = np.linspace(0.70, 1.30, 100)
objective_values = []
cd_values = []

for d in dose_scan:
    printed_temp = get_printed_pattern(d)
    cd_temp = extract_cd_from_middle_row(printed_temp, x)
    obj_temp = (cd_temp - target_cd) ** 2

    cd_values.append(cd_temp)
    objective_values.append(obj_temp)

cd_values = np.array(cd_values)
objective_values = np.array(objective_values)

# ==============================
# 12. 画图
# ==============================
plt.figure(figsize=(15, 10))

# ---- 图1：原始 mask ----
plt.subplot(2, 2, 1)
plt.imshow(mask_2d, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='gray')
plt.title("2D Mask")
plt.xlabel("x")
plt.ylabel("y")

# ---- 图2：最优 dose 下的打印图形 ----
plt.subplot(2, 2, 2)
plt.imshow(best_printed, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='gray')
plt.title(f"Best Printed Pattern\n(best dose = {best_dose:.4f})")
plt.xlabel("x")
plt.ylabel("y")

# ---- 图3：dose-CD 曲线 ----
plt.subplot(2, 2, 3)
plt.plot(dose_scan, cd_values, label='CD(dose)')
plt.axhline(target_cd, color='red', linestyle='--', label='target CD')
plt.axvline(best_dose, color='green', linestyle='--', label='best dose')
plt.xlabel("Dose")
plt.ylabel("Middle-row CD")
plt.title("Dose-CD Curve")
plt.grid(True)
plt.legend()

# ---- 图4：目标函数曲线 ----
plt.subplot(2, 2, 4)
plt.plot(dose_scan, objective_values, label='objective(dose)')
plt.axvline(best_dose, color='green', linestyle='--', label='best dose')
plt.xlabel("Dose")
plt.ylabel("Objective Value")
plt.title("Objective Function Curve")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
#疑问解答
#中间有“阈值化”和“像素边缘提取”这两层离散化，所以最后曲线自然很容易变成阶梯状。

