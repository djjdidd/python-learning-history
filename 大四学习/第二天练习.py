import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 1. 一维坐标
x = np.linspace(-2.0,2.0,2000)
dx = x[1] - x[0]

# 2. 双线掩模
line1 = np.where(np.abs(x+0.35) <= 0.10, 1.0, 0.0)
line2 = np.where(np.abs(x-0.35) <= 0.10, 1.0, 0.0)
mask = np.where((line1==True) | (line2==True), 1.0, 0.0)#mask = np.maximum(line1, line2)

# 3. 第一组 sigma
sigma1 = 0.05
psf1 = np.exp(-x**2 / (2 * sigma1**2))
psf1 = psf1 / np.sum(psf1)
aerial_image_1 = signal.fftconvolve(mask, psf1, mode='same')

# 4. 第二组 sigma
sigma2 =0.15 
psf2 = np.exp(-x**2 / (2 * sigma2**2))
psf2 = psf2 / np.sum(psf2)
aerial_image_2 = signal.fftconvolve(mask, psf2, mode='same')

# 5. 画图
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(x, mask)
plt.title("Double-Line Mask")
plt.xlabel("Position")
plt.ylabel("Intensity")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(x, aerial_image_1)
plt.title("Aerial Image (sigma=0.05)")
plt.xlabel("Position")
plt.ylabel("Intensity")
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(x, aerial_image_2)
plt.title("Aerial Image (sigma=0.15)")
plt.xlabel("Position")
plt.ylabel("Intensity")
plt.grid(True)

plt.tight_layout()
plt.show()

#双线掩模和单线掩模相比，成像观察上有什么不同？
# 单线掩模通常对应一个主峰双线掩模通常对应两个峰当模糊增强时，这两个峰的分离度会下降，甚至中间抬高、趋于粘连