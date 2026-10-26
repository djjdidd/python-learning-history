#构造“上宽下窄”的 mask
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
x = np.linspace(-1.0, 1.0, 201)
y = np.linspace(-1.0, 1.0, 201)
X,Y=np.meshgrid(x,y)
z=0.20+0.25*(Y+1.0)/0.5
mask=np.where((np.abs(X)<=z)&(np.abs(Y)<=0.35),1.0,0.0)
sigma = 0.14
psf_2d = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
psf_2d = psf_2d / np.sum(psf_2d)
aerial_2d=convolve2d(mask,psf_2d,mode='same')
dose = 1.0
threshold = 0.20
printed_2d=np.where(dose*aerial_2d>=threshold,1.0,0.0)
def myfunc(binary,x_roots,row):
    profile=binary[row,:]
    printed_indice=np.where(profile==1)[0]
    if len(printed_indice)==0:
        return None,None,0.0,profile
    left=x_roots[printed_indice[0]]
    right=x_roots[printed_indice[-1]]
    cd_value=right-left
    return left,right,cd_value,profile
row_1=75
row_2=printed_2d.shape[0]//2#表示二维数组总共有多少行
row_3=125
left_top, right_top, cd_top, profile_top = myfunc(printed_2d, x, row_1)
left_mid, right_mid, cd_mid, profile_mid = myfunc(printed_2d, x, row_2)
left_bottom, right_bottom, cd_bottom, profile_bottom = myfunc(printed_2d, x, row_3)
print("上方横截面:")
print("左边缘 =", left_top, "右边缘 =", right_top, "CD =", cd_top)
print("中间横截面:")
print("左边缘 =", left_mid, "右边缘 =", right_mid, "CD =", cd_mid)
print("下方横截面:")
print("左边缘 =", left_bottom, "右边缘 =", right_bottom, "CD =", cd_bottom)
plt.figure(figsize=(14,8))
plt.subplot(2,2,1)
plt.imshow(
    printed_2d,
    cmap='gray',
    extent=[x.min(),x.max(),y.min(),y.max()],
    origin='lower'
)
plt.axhline(y[row_1],linestyle='--',color='green',label='1 row')
plt.axhline(y[row_2],linestyle='--',color='blue',label='2 row')
plt.axhline(y[row_3],linestyle='--',color='pink',label='3 row')
plt.title("Printed Pattern with Measured Rows")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.subplot(2,2,2)
plt.plot(x,profile_top,label='top profile')
if left_top is not None:
    plt.axvline(left_top, color='red', linestyle='--', label='left edge')
    plt.axvline(right_top, color='purple', linestyle='--', label='right edge')
plt.title(f"Top Row Profile, CD = {cd_top:.4f}")
plt.xlabel("x")
plt.ylabel("printed")
plt.grid(True)
plt.legend()
plt.subplot(2,2,3)
plt.plot(x,profile_mid,label='mid profile')
if left_mid is not None:
    plt.axvline(left_mid, color='red', linestyle='--', label='left edge')
    plt.axvline(right_mid, color='purple', linestyle='--', label='right edge')
plt.title(f"middle Row Profile, CD = {cd_mid:.4f}")
plt.xlabel("x")
plt.ylabel("printed")
plt.grid(True)
plt.legend()
plt.subplot(2,2,4)
plt.plot(x,profile_bottom,label='bottom profile')
if left_bottom is not None:
    plt.axvline(left_bottom, color='red', linestyle='--', label='left edge')
    plt.axvline(right_bottom, color='purple', linestyle='--', label='right edge')
plt.title(f"bottom Row Profile, CD = {cd_bottom:.4f}")
plt.xlabel("x")
plt.ylabel("printed")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()