#二维掩模与二维成像入门

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
x=np.linspace(-1.0,1.0,201)
y=np.linspace(-1.0,1.0,201)
X,Y=np.meshgrid(x,y)
mask_2d=np.where((np.abs(X)<=0.4)&(np.abs(Y)<=0.4),1.0,0.0)
sigma=0.12
psf_2d=np.exp(-(X**2+Y**2)/(2*sigma**2))
psf_2d=psf_2d/np.sum(psf_2d)
aerial_2d=convolve2d(mask_2d,psf_2d,mode='same')
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(mask_2d,extent=[x.min(),x.max(),y.min(),y.max()],origin='lower')#xy坐标轴范围
plt.title("2D mask")
plt.subplot(1,3,2)
plt.imshow(psf_2d,extent=[x.min(),x.max(),y.min(),y.max()],origin='lower')
plt.title("2D psf")
plt.subplot(1,3,3)
plt.imshow(aerial_2d,extent=[x.min(),x.max(),y.min(),y.max()],origin='lower')
plt.title("2D aerial image")
plt.tight_layout()
plt.show()

#二维卷积表示掩模上每个透光点都按照二维 PSF 在平面中扩散
# 并把所有点的扩散结果叠加起来
# 最终得到二维连续的成像强度分布
# 为什么 psf_2d 要做归一化？
# 因为 PSF 表示的是系统对能量的空间重新分配，
# 不是随意放大总强度，所以通常要归一化，让总和等于 1。
