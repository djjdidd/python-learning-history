import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
x = np.linspace(-1.0, 1.0, 201)
y = np.linspace(-1.0, 1.0, 201)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)
mask=np.where((np.abs(X)<=0.35)&(np.abs(Y)<=0.25),1.0,0.0)
sigma = 0.08
psf_2d = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
psf_2d = psf_2d / np.sum(psf_2d)
aerial_2d = convolve2d(mask, psf_2d, mode='same')
dose=1.0
target_cd=0.80
threshold_scan=np.linspace(0.10,0.40,150)
cd_values = []
objective_values = []
for i in threshold_scan:
    printed_2d=(dose * aerial_2d >= i).astype(float)
    row_mid = printed_2d.shape[0] // 2
    profile = printed_2d[row_mid, :]
    printed_indices = np.where(profile == 1)[0]
    left_edge = x[printed_indices[0]]
    right_edge = x[printed_indices[-1]]
    cd_value = right_edge - left_edge
    obj_temp = (cd_value - target_cd) ** 2
    cd_values.append(cd_value)
    objective_values.append(obj_temp)
cd_values = np.array(cd_values)
objective_values = np.array(objective_values)
best_index = np.argmin(objective_values)
best_threshold=threshold_scan[best_index]
best_objective = objective_values[best_index]
best_printed=(dose * aerial_2d >= best_threshold).astype(float)
best_cd = cd_values[best_index]

print("最优 threshold =", best_threshold)
print("对应的中间横截面 CD =", best_cd)
print("目标 CD =", target_cd)
print("目标函数最小值 =", best_objective)
plt.figure(figsize=(15, 10))

# ---- 图1：原始 mask ----
plt.subplot(2, 2, 1)
plt.imshow(mask, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='gray')
plt.title("2D Mask")
plt.xlabel("x")
plt.ylabel("y")

# ---- 图2：最优 threshold 下的打印图形 ----
plt.subplot(2, 2, 2)
plt.imshow(best_printed, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='gray')
plt.title(f"Best Printed Pattern\n(best threshold = {best_threshold:.4f})")
plt.xlabel("x")
plt.ylabel("y")

# ---- 图3：threshold-CD 曲线 ----
plt.subplot(2, 2, 3)
plt.plot(threshold_scan, cd_values, label='CD(threshold)')
plt.axhline(target_cd, color='red', linestyle='--', label='target CD')
plt.axvline(best_threshold, color='green', linestyle='--', label='best threshold')
plt.xlabel("threshold")
plt.ylabel("Middle-row CD")
plt.title("threshold-CD Curve")
plt.grid(True)
plt.legend()

# ---- 图4：目标函数曲线 ----
plt.subplot(2, 2, 4)
plt.plot(threshold_scan, objective_values, label='objective(dose)')
plt.axvline(best_threshold, color='green', linestyle='--', label='best dose')
plt.xlabel("threshold")
plt.ylabel("Objective Value")
plt.title("Objective Function Curve")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()