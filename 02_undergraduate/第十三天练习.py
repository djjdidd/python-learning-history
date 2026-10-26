import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, ndimage

# ==============================
# 1. 建立二维空间网格
# ==============================
N = 201
x = np.linspace(-1.0, 1.0, N)
y = np.linspace(-1.0, 1.0, N)
X, Y = np.meshgrid(x, y)

# ==============================
# 2. 构造一个二维小孔 mask
# ==============================
radius = 0.25
mask = (X**2 + Y**2 <= radius**2).astype(float)

# ==============================
# 3. 构造二维高斯 PSF
# ==============================
sigma = 0.08
psf = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
psf = psf / np.sum(psf)

# ==============================
# 4. 二维卷积得到 aerial image
# ==============================
aerial_image = signal.fftconvolve(mask, psf, mode='same')

noise_1=0.01
noise_2=0.05
noise1 = noise_1 * np.random.randn(N, N)#会生成一个二维随机数组，均值大致是 0，正负都有。
aerial_noisy1 = aerial_image + noise1
noise2 = noise_2 * np.random.randn(N, N)#会生成一个二维随机数组，均值大致是 0，正负都有。
aerial_noisy2 = aerial_image + noise2
aerial_noisy1 = np.clip(aerial_noisy1, 0.0, 1.0)
aerial_noisy2 = np.clip(aerial_noisy2, 0.0, 1.0)
threshold = 0.2
printed_clean = (aerial_image >= threshold).astype(float)
printed_noisy1 = (aerial_noisy1 >= threshold).astype(float)
printed_noisy2 = (aerial_noisy2 >= threshold).astype(float)
printed_filtered1 = ndimage.median_filter(printed_noisy2, size=3)
printed_filtered2 = ndimage.median_filter(printed_noisy2, size=5)
plt.figure(figsize=(12, 8))

plt.subplot(2,4, 1)
plt.imshow(mask, cmap='gray')#灰度图
plt.title("Mask")
plt.axis('off')#关闭坐标轴

plt.subplot(2, 4, 2)
plt.imshow(aerial_image, cmap='hot')#热度颜色图
plt.title("Clean Aerial Image")
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(printed_noisy1, cmap='gray')
plt.title("Noisy1 printed Image")
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(printed_clean, cmap='gray')
plt.title("Clean Printed Pattern")
plt.axis('off')

plt.subplot(2, 4, 5)
plt.imshow(printed_noisy2, cmap='gray')
plt.title("Noisy2 Printed Pattern")
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(printed_filtered1, cmap='gray')
plt.title("Filtered1 Printed Pattern")
plt.axis('off')
plt.subplot(2, 4, 7)
plt.imshow(printed_filtered2, cmap='gray')
plt.title("Filtered2 Printed Pattern")
plt.axis('off')

plt.tight_layout()
plt.show()