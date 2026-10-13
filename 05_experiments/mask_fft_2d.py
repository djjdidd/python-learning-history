import numpy as np
import matplotlib.pyplot as plt


# ==========================
# 1. 创建Mask
# ==========================

size = 256


# 大矩形Mask
mask_large = np.zeros((size, size))

mask_large[90:166, 80:176] = 1



# 小线条Mask
mask_small = np.zeros((size, size))

mask_small[120:136, 50:206] = 1



# ==========================
# 2. FFT
# ==========================

def get_fft(mask):

    # 二维傅里叶变换
    fft = np.fft.fft2(mask)

    # 将低频移动到中心
    fft_shift = np.fft.fftshift(fft)

    # 取幅值
    magnitude = np.abs(fft_shift)

    # 对数显示，方便观察
    spectrum = np.log(1 + magnitude)

    return spectrum



fft_large = get_fft(mask_large)

fft_small = get_fft(mask_small)



# ==========================
# 3. 显示结果
# ==========================

plt.figure(figsize=(12,8))


# 大矩形
plt.subplot(2,2,1)
plt.imshow(mask_large,cmap="gray")
plt.title("Large Rectangle Mask")
plt.axis("off")


plt.subplot(2,2,2)
plt.imshow(fft_large,cmap="gray")
plt.title("FFT Spectrum of Large Mask")
plt.axis("off")



# 小线条
plt.subplot(2,2,3)
plt.imshow(mask_small,cmap="gray")
plt.title("Small Line Mask")
plt.axis("off")


plt.subplot(2,2,4)
plt.imshow(fft_small,cmap="gray")
plt.title("FFT Spectrum of Small Line")
plt.axis("off")


plt.tight_layout()
plt.show()