import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
x=np.linspace(-2.0,2.0,2000)
dx=x[1]-x[0]
line_1=np.where(np.abs(x+0.35)<=0.1)
line_2=np.where(np.abs(x-0.35)<=0.1)
mask=np.where((np.abs(x+0.35)<=0.1)|(np.abs(x-0.35)<=0.1),1.0,0.0)
sigma = 0.12  # 高斯标准差，决定模糊程度
psf = np.exp(-x**2 / (2 * sigma**2))  # 计算高斯函数
psf = psf / np.sum(psf)
aerial_image=signal.fftconvolve(mask,psf,mode='same')
threshold1 = 0.3 * np.max(aerial_image)
threshold2 = 0.7 * np.max(aerial_image)
printed_pattern1 = np.where(aerial_image >= threshold1, 1.0, 0.0)
printed_pattern2 = np.where(aerial_image >= threshold2, 1.0, 0.0)
# 掩模图形
plt.figure(figsize=(10,6))
plt.subplot(4, 1, 1)
plt.plot(x, mask, label="Mask Pattern")
plt.title("Mask")
plt.xlabel("Position")
plt.ylabel("Intensity")
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(x, aerial_image, label="Aerial Image")
plt.axhline(threshold1, color='red', linestyle='--', label="Threshold1")
plt.axhline(threshold2, color='blue', linestyle='--', label="Threshold2")
plt.title("Aerial Image with Threshold")
plt.ylabel("Intensity")
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(x, printed_pattern1)
plt.title("Printed Pattern1")
plt.xlabel("Position")
plt.ylabel("Binary Value")
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(x, printed_pattern2)
plt.title("Printed Pattern2")
plt.xlabel("Position")
plt.ylabel("Binary Value")
plt.grid(True)

plt.tight_layout()
plt.show()

# 总结：
# 阈值较低时，中间较弱的成像区域也可能超过阈值，因此两条线更容易桥连。
# 阈值较高时，只有高强度区域被保留下来，打印区域会缩小，更容易保持分离。
# 这说明图形是否能被正确分辨，不仅取决于成像本身，还取决于阈值判定条件。