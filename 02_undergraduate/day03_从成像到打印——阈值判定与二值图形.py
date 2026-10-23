import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

x=np.linspace(-2.0,2.0,2000)
dx=x[1]-x[0]
mask=np.where(np.abs(x)<=0.25,1.0,0.0)
sigma = 0.08  # 高斯标准差，决定模糊程度
psf = np.exp(-x**2 / (2 * sigma**2))  # 计算高斯函数
psf = psf / np.sum(psf)
aerial_image=signal.fftconvolve(mask,psf,mode='same')
threshold = 0.3 * np.max(aerial_image)
printed_pattern = np.where(aerial_image >= threshold, 1.0, 0.0)
# 掩模图形
plt.figure(figsize=(10,6))
plt.subplot(3, 1, 1)
plt.plot(x, mask, label="Mask Pattern")
plt.title("Mask")
plt.xlabel("Position")
plt.ylabel("Intensity")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(x, aerial_image, label="Aerial Image")
plt.axhline(threshold, color='red', linestyle='--', label="Threshold")
plt.title("Aerial Image with Threshold")
plt.ylabel("Intensity")
plt.legend()
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(x, printed_pattern)
plt.title("Printed Pattern")
plt.xlabel("Position")
plt.ylabel("Binary Value")
plt.grid(True)

plt.tight_layout()
plt.show()