#二维图形测量入门
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy import ndimage
x=np.linspace(-1.0,1.0,201)
y=np.linspace(-1.0,1.0,201)
dx=x[1]-x[0]
dy=y[1]-y[0]
X,Y=np.meshgrid(x,y)
left_block=((X>=-0.55)&(X<=-0.20)&(np.abs(Y)<=0.20))
right_block=((X>=0.20)&(X<=0.55)&(np.abs(Y)<=0.20))
bridge=((np.abs(X)<=0.20)&(np.abs(Y)<=0.02))
mask_2d=np.where(left_block|right_block|bridge,1.0,0.0)
sigma=0.08
psf_2d=np.exp(-(X**2+Y**2)/(2*sigma**2))
psf_2d=psf_2d/np.sum(psf_2d)
aerial_2d=convolve2d(mask_2d,psf_2d,mode='same')
dose=1.0
threshold=0.20
printed_2d=(dose*aerial_2d>=threshold).astype(float)

# 找到二维打印图中间那一行
mid_row = printed_2d.shape[0] // 2

# 取出中间这一整行，得到一维剖面
profile = printed_2d[mid_row, :]

# 统计这一行有多少位置被打印出来，并乘上 dx 得到横向宽度
width_x = np.sum(profile) * dx

# 统计整张图打印出来的像素数，并乘上单像素面积 dx*dy 得到总打印面积
area = np.sum(printed_2d) * dx * dy

# 标记连通区域，并返回连通区域个数
labeled_array, num_features = ndimage.label(printed_2d)

print("中间横截面宽度=",width_x)
print("总打印面积：",area)
print("连通区域个数：",num_features)
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(mask_2d,extent=[x.min(),x.max(),y.min(),y.max()],origin='lower')
plt.title("2d mask")
plt.colorbar()
plt.subplot(1,3,2)
plt.imshow(printed_2d,extent=[x.min(),x.max(),y.min(),y.max()],origin='lower')
plt.title("2d printed pattern")
plt.colorbar()
plt.subplot(1,3,3)
plt.plot(x,profile)
plt.title("middle row profile")
plt.xlabel("x")
plt.ylabel("printed")
plt.grid(True)
plt.tight_layout()
plt.show()