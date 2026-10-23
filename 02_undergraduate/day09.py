#二维阈值打印与桥连/断裂直觉
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# ==============================
# 1. 建立二维空间网格
# ==============================
x = np.linspace(-1.0, 1.0, 201)
y = np.linspace(-1.0, 1.0, 201)
X, Y = np.meshgrid(x, y)

# ==============================
# 2. 建立两个相邻矩形掩模
# ==============================
left_rect = ((X >= -0.45) & (X <= -0.10) & (np.abs(Y) <= 0.25))
right_rect = ((X >= 0.10) & (X <= 0.45) & (np.abs(Y) <= 0.25))

mask_2d = np.where(left_rect | right_rect, 1.0, 0.0)

# ==============================
# 3. 建立二维高斯 PSF
# ==============================
sigma = 0.08
psf_2d = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
psf_2d = psf_2d / np.sum(psf_2d)

# ==============================
# 4. 二维卷积得到 aerial image
# ==============================
aerial_2d = convolve2d(mask_2d, psf_2d, mode='same')

# ==============================
# 5. 加剂量和阈值，得到打印图形
# ==============================
dose = 1.0
threshold = 0.20

printed_2d = (dose * aerial_2d >= threshold).astype(float)

# ==============================
# 6. 画图
# ==============================
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(mask_2d, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
plt.title("2D Mask")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(aerial_2d, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
plt.title("2D Aerial Image")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(printed_2d, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
plt.title("2D Printed Pattern")
plt.colorbar()

plt.tight_layout()
plt.show()