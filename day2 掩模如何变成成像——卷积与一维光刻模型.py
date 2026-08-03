import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# ==============================
# 1. 构造一维空间坐标
# ==============================
x = np.linspace(-2.0, 2.0, 2000)  # 空间坐标
dx = x[1] - x[0]  # 相邻采样点间距

# ==============================
# 2. 构造一条简单的掩模（mask）
# ==============================
# 当 |x| <= 0.25 时，掩模取值为 1，表示透光
# 否则为 0，表示不透光
mask = np.where(np.abs(x) <= 0.25, 1.0, 0.0)

# ==============================
# 3. 定义光学系统的点扩散函数（PSF）
# ==============================
# 使用高斯函数来近似光学系统的模糊效应
sigma = 0.08  # 高斯标准差，决定模糊程度
psf = np.exp(-x**2 / (2 * sigma**2))  # 计算高斯函数
psf = psf / np.sum(psf)  # 归一化，使得 PSF 总能量为 1

# ==============================
# 4. 卷积得到 aerial image（成像图）
# ==============================
aerial_image = signal.fftconvolve(mask, psf, mode='same')

# ==============================
# 5. 画图
# ==============================
plt.figure(figsize=(8, 5))

# 掩模图形
plt.subplot(1, 3, 1)
plt.plot(x, mask, label="Mask Pattern")
plt.title("Mask")
plt.xlabel("Position")
plt.ylabel("Intensity")
plt.grid(True)

# 点扩散函数（PSF）
plt.subplot(1, 3, 2)
plt.plot(x, psf, label="PSF", color="orange")
plt.title("Point Spread Function (PSF)")
plt.xlabel("Position")
plt.ylabel("Intensity")
plt.grid(True)

# 成像图（aerial image）
plt.subplot(1, 3, 3)
plt.plot(x, aerial_image, label="Aerial Image", color="green")
plt.title("Aerial Image")
plt.xlabel("Position")
plt.ylabel("Intensity")
plt.grid(True)

plt.tight_layout()
plt.show()