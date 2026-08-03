import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, optimize
from scipy.interpolate import interp1d

# ==============================
# 1. 空间网格
# ==============================
x = np.linspace(-2.0, 2.0, 2000)
dx = x[1] - x[0]

# ==============================
# 2. 掩模定义
# ==============================
mask = np.where(np.abs(x) <= 0.25, 1.0, 0.0)

# ==============================
# 3. 光学模糊核 PSF
# ==============================
sigma = 0.12#sigma 变大表示 光学系统模糊更强 ，所以成像会 更模糊、更平滑
psf = np.exp(-x**2 / (2 * sigma**2))
psf = psf / np.sum(psf)

# ==============================
# 4. 成像卷积
# ==============================
aerial_image = signal.fftconvolve(mask, psf, mode='same')

# ==============================
# 5. 定义打印过程
# ==============================
threshold = 0.40

def print_pattern(dose):
    return (dose * aerial_image >= threshold).astype(float)

def measure_width(binary_pattern):
    return np.sum(binary_pattern) * dx

# ==============================
# 6. 先扫描几组剂量，得到 dose-width 数据
# ==============================
dose_samples = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
width_samples = []

for d in dose_samples:
    binary = print_pattern(d)
    w = measure_width(binary)
    width_samples.append(w)

width_samples = np.array(width_samples)

print("采样剂量:", dose_samples)
print("对应线宽:", width_samples)

# ==============================
# 7. 用插值把离散数据变成更平滑的曲线
# ==============================
f_width = interp1d(dose_samples, width_samples, kind='linear')
dose_fine = np.linspace(0.8, 1.2, 200)
width_fine = f_width(dose_fine)

# ==============================
# 8. 再做剂量优化
# ==============================
target_width = 0.40

def objective(dose_array):
    dose = dose_array[0]
    if dose <= 0:
        return 1e6
    w = measure_width(print_pattern(dose))
    return (w - target_width) ** 2

result = optimize.minimize(objective, x0=np.array([1.0]), method='Nelder-Mead')
best_dose = result.x[0]
best_width = measure_width(print_pattern(best_dose))

print("最优剂量 =", best_dose)
print("最优线宽 =", best_width)

# ==============================
# 9. 画图
# ==============================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(x, mask, label='mask')
plt.plot(x, aerial_image, label='aerial image')
plt.plot(x, print_pattern(best_dose), label='printed pattern')
plt.axhline(threshold, linestyle='--', label='threshold')
plt.title("Imaging and Printing")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(dose_samples, width_samples, label='samples')
plt.plot(dose_fine, width_fine, label='interpolated curve')
plt.axhline(target_width, linestyle='--', label='target width')
plt.axvline(best_dose, linestyle='--', label='best dose')
plt.title("Dose-Width Curve")
plt.xlabel("Dose")
plt.ylabel("Printed Width")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 第七天代码主流程总结：
# 1. 先建立 mask（掩模），表示原始图形长什么样。
# 2. 再用 PSF（点扩散函数）和 mask 做卷积，得到 aerial image（空中像/成像强度分布）。
# 3. 然后把 dose（剂量）乘到 aerial image 上，表示曝光强度随剂量变化。
# 4. 接着判断各位置的强度是否超过 threshold（阈值）：
#    - 超过阈值的位置记为 1
#    - 没超过阈值的位置记为 0
#    这样就得到二值打印图形 binary pattern。
# 5. 再对 binary pattern 求和，并乘上 dx，
#    就可以得到打印线宽 width。
# 6. 然后设定目标线宽 target width，
#    通过优化让“当前线宽”和“目标线宽”的误差尽量小。
# 7. 最后找到最合适的剂量 best dose。
