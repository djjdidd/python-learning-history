
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy import ndimage
x=np.linspace(-1.0,1.0,201)
y=np.linspace(-1.0,1.0,201)
dx=x[1]-x[0]
dy=y[1]-y[0]
X,Y=np.meshgrid(x,y)
left_block=((X>=-0.35)&(X<=-0.15)&(np.abs(Y)<=0.15))
right_block=((X>=0.15)&(X<=0.35)&(np.abs(Y)<=0.15))
mask_2d=np.where(left_block|right_block,0.0,1.0)
sigma=0.08
psf_2d=np.exp(-(X**2+Y**2)/(2*sigma**2))
psf_2d=psf_2d/np.sum(psf_2d)
aerial_2d=convolve2d(mask_2d,psf_2d,mode='same')
dose=1.0
threshold=0.50
printed_2d=(dose*aerial_2d>=threshold).astype(float)
hole_pattern=1-printed_2d
hole_central=hole_pattern[40:-40,40:-40]
# 找到二维打印图中间那一行
mid_row = hole_central.shape[0] // 2

# 取出中间这一整行，得到一维剖面
profile = hole_central[mid_row, :]

# 统计这一行有多少位置被打印出来，并乘上 dx 得到横向宽度
width_x = np.sum(profile) * dx

# 统计整张图打印出来的像素数，并乘上单像素面积 dx*dy 得到总打印面积
area = np.sum(hole_central) * dx * dy

# 标记连通区域，并返回连通区域个数
labeled_array, num_features = ndimage.label(hole_central)

print("中间横截面宽度=",width_x)
print("总打印面积：",area)
print("连通区域个数：",num_features)
plt.figure(figsize=(8,7))
plt.subplot(2,2,1)
plt.imshow(mask_2d,extent=[x.min(),x.max(),y.min(),y.max()],origin='lower')
plt.title("2d mask")
plt.colorbar()
plt.subplot(2,2,2)
plt.imshow(printed_2d,extent=[x.min(),x.max(),y.min(),y.max()],origin='lower')
plt.title("2d printed pattern")
plt.colorbar()
plt.subplot(2,2,3)
plt.imshow(aerial_2d,extent=[x.min(),x.max(),y.min(),y.max()],origin='lower')
plt.title("2d aerial image")
plt.colorbar()
plt.subplot(2,2,4)
plt.imshow(hole_central,extent=[x.min(),x.max(),y.min(),y.max()],origin='lower')
plt.title("hole pattern")
plt.colorbar()

plt.tight_layout()
plt.show()