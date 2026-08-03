#二维图形的噪声、阈值敏感性与简单后处理
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

# ==============================
# 5. 加入简单随机噪声
# ==============================
noise_level = 0.03
noise = noise_level * np.random.randn(N, N)#会生成一个二维随机数组，均值大致是 0，正负都有。
aerial_noisy = aerial_image + noise

# 防止数值太离谱，截断到合理范围
aerial_noisy = np.clip(aerial_noisy, 0.0, 1.0)

# ==============================
# 6. 阈值打印
# ==============================
threshold = 0.15
printed_clean = (aerial_image >= threshold).astype(float)
printed_noisy = (aerial_noisy >= threshold).astype(float)

# ==============================
# 7. 简单后处理：中值滤波
# ==============================
printed_filtered = ndimage.median_filter(printed_noisy, size=2)
#会用邻域里的“中间值”来替换当前点
# ==============================
# 8. 画图比较
# ==============================
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(mask, cmap='gray')#灰度图
plt.title("Mask")
plt.axis('off')#关闭坐标轴

plt.subplot(2, 3, 2)
plt.imshow(aerial_image, cmap='hot')#热度颜色图
plt.title("Clean Aerial Image")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(aerial_noisy, cmap='hot')
plt.title("Noisy Aerial Image")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(printed_clean, cmap='gray')
plt.title("Clean Printed Pattern")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(printed_noisy, cmap='gray')
plt.title("Noisy Printed Pattern")
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(printed_filtered, cmap='gray')
plt.title("Filtered Printed Pattern")
plt.axis('off')

plt.tight_layout()
plt.show()
#拓扑变化是“连没连、断没断、区域个数变没变”
#核越大，平滑越强，但保真度越差