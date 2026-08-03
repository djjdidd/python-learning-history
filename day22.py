# 第22天：线端缩短（Line-End Shortening）与二维图形局部失真

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# ==============================
# 1. 建立二维空间网格
# ==============================
x = np.linspace(-1.0, 1.0, 301)
y = np.linspace(-1.0, 1.0, 301)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)

# ==============================
# 2. 构造一个“水平线” mask
# ==============================
mask_2d = np.where((np.abs(X) <= 0.45) & (np.abs(Y) <= 0.06), 1.0, 0.0)

# ==============================
# 3. 构造二维高斯 PSF
# ==============================
sigma = 0.07
psf_2d = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
psf_2d = psf_2d / np.sum(psf_2d)

# ==============================
# 4. 二维卷积得到 aerial image
# ==============================
aerial_2d = convolve2d(mask_2d, psf_2d, mode='same')

# ==============================
# 5. 阈值打印
# ==============================
dose = 1.0
threshold = 0.18
printed_2d = (dose * aerial_2d >= threshold).astype(float)

# ==============================
# 6. 定义函数：测量中间行宽度
# ==============================
def measure_middle_width(binary_2d, x_coords):
    row_mid = binary_2d.shape[0] // 2
    profile = binary_2d[row_mid, :]
    printed_indices = np.where(profile == 1)[0]

    if len(printed_indices) == 0:
        return 0.0, None, None, profile

    left_edge = x_coords[printed_indices[0]]
    right_edge = x_coords[printed_indices[-1]]
    width = right_edge - left_edge

    return width, left_edge, right_edge, profile

# ==============================
# 7. 定义函数：测量中间列长度
# ==============================
def measure_middle_length(binary_2d, y_coords):
    col_mid = binary_2d.shape[1] // 2
    profile = binary_2d[:, col_mid]
    printed_indices = np.where(profile == 1)[0]

    if len(printed_indices) == 0:
        return 0.0, None, None, profile

    bottom_edge = y_coords[printed_indices[0]]
    top_edge = y_coords[printed_indices[-1]]
    length = top_edge - bottom_edge

    return length, bottom_edge, top_edge, profile

# ==============================
# 8. 测量 mask 本身
# ==============================
mask_width, mask_left, mask_right, mask_row_profile = measure_middle_width(mask_2d, x)
mask_length, mask_bottom, mask_top, mask_col_profile = measure_middle_length(mask_2d, y)

# ==============================
# 9. 测量 printed pattern
# ==============================
printed_width, printed_left, printed_right, printed_row_profile = measure_middle_width(printed_2d, x)
printed_length, printed_bottom, printed_top, printed_col_profile = measure_middle_length(printed_2d, y)

# ==============================
# 10. 计算线端缩短量
# ==============================
line_end_shortening = mask_length - printed_length

print("设计宽度 =", mask_width)
print("打印宽度 =", printed_width)
print("设计长度 =", mask_length)
print("打印长度 =", printed_length)
print("线端缩短 LES =", line_end_shortening)

# ==============================
# 11. 画图
# ==============================
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(mask_2d, extent=[x.min(), x.max(), y.min(), y.max()],
           origin='lower', cmap='gray')
plt.title("Mask")
plt.xlabel("x")
plt.ylabel("y")

plt.subplot(2, 3, 2)
plt.imshow(aerial_2d, extent=[x.min(), x.max(), y.min(), y.max()],
           origin='lower', cmap='hot')
plt.title("Aerial Image")
plt.xlabel("x")
plt.ylabel("y")

plt.subplot(2, 3, 3)
plt.imshow(printed_2d, extent=[x.min(), x.max(), y.min(), y.max()],
           origin='lower', cmap='gray')
plt.title("Printed Pattern")
plt.xlabel("x")
plt.ylabel("y")

plt.subplot(2, 3, 4)
plt.plot(x, mask_row_profile, label='mask row')
plt.plot(x, printed_row_profile, label='printed row')
if printed_left is not None:
    plt.axvline(printed_left, color='red', linestyle='--', label='printed left edge')
    plt.axvline(printed_right, color='purple', linestyle='--', label='printed right edge')
plt.title(f"Middle Row Profile\nPrinted Width = {printed_width:.4f}")
plt.xlabel("x")
plt.ylabel("value")
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 5)
plt.plot(y, mask_col_profile, label='mask column')
plt.plot(y, printed_col_profile, label='printed column')
if printed_bottom is not None:
    plt.axvline(printed_bottom, color='red', linestyle='--', label='printed bottom edge')
    plt.axvline(printed_top, color='purple', linestyle='--', label='printed top edge')
plt.title(f"Middle Column Profile\nPrinted Length = {printed_length:.4f}")
plt.xlabel("y")
plt.ylabel("value")
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 6)
plt.axis('off')
summary_text = (
    f"Mask Width = {mask_width:.4f}\n"
    f"Printed Width = {printed_width:.4f}\n\n"
    f"Mask Length = {mask_length:.4f}\n"
    f"Printed Length = {printed_length:.4f}\n\n"
    f"LES = {line_end_shortening:.4f}"
)
plt.text(0.1, 0.5, summary_text, fontsize=12)

plt.tight_layout()
plt.show()

#sigma 变大时，线端缩短通常会更严重，LES 往往变大，因为模糊增强会让端部更难维持