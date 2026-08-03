# 第23天：line-space-line 结构的多指标测量

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy import ndimage

# ==============================
# 1. 建立二维空间网格
# ==============================
x = np.linspace(-1.0, 1.0, 301)
y = np.linspace(-1.0, 1.0, 301)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)

# ==============================
# 2. 构造 line-space-line mask
# ==============================
left_line = ((X >= -0.55) & (X <= -0.38) & (np.abs(Y) <= 0.25))
right_line = ((X >= 0.18) & (X <= 0.55) & (np.abs(Y) <= 0.25))
mask_2d = np.where(left_line | right_line, 1.0, 0.0)

# ==============================
# 3. 构造二维高斯 PSF
# ==============================
sigma = 0.12
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
threshold = 0.20
printed_2d = (dose * aerial_2d >= threshold).astype(float)

# ==============================
# 6. 定义函数：提取左线宽、右线宽、gap
# ==============================
def extract_line_and_gap_metrics(binary_2d, x_coords):
    row_mid = binary_2d.shape[0] // 2
    profile = binary_2d[row_mid, :]

    printed_indices = np.where(profile == 1)[0]

    # 如果这一行什么都没打印出来
    if len(printed_indices) == 0:
        return 0.0, 0.0, 0.0, profile

    # 找到相邻索引的差值
    diff_indices = np.diff(printed_indices)

    # 找到断开的地方
    split_points = np.where(diff_indices > 1)[0]

    # 如果没有断点，说明只剩下一整段
    # 通常意味着桥连或无法分成左右两段
    if len(split_points) < 1:
        left_cd = x_coords[printed_indices[-1]] - x_coords[printed_indices[0]]
        right_cd = 0.0
        gap_cd = 0.0
        return left_cd, right_cd, gap_cd, profile

    # 用第一处断点，把打印区域分成左右两段
    split_idx = split_points[0]
    left_segment = printed_indices[:split_idx + 1]
    right_segment = printed_indices[split_idx + 1:]

    left_cd = x_coords[left_segment[-1]] - x_coords[left_segment[0]]
    right_cd = x_coords[right_segment[-1]] - x_coords[right_segment[0]]
    gap_cd = x_coords[right_segment[0]] - x_coords[left_segment[-1]]

    return left_cd, right_cd, gap_cd, profile

# ==============================
# 7. 调用函数测量打印结果
# ==============================
left_cd, right_cd, gap_cd, profile_mid = extract_line_and_gap_metrics(printed_2d, x)

# 连通区域个数
labeled_array, num_features = ndimage.label(printed_2d)

print("左线宽 left_cd =", left_cd)
print("右线宽 right_cd =", right_cd)
print("中间空隙 gap_cd =", gap_cd)
print("连通区域个数 num_features =", num_features)

# ==============================
# 8. 画图
# ==============================
plt.figure(figsize=(14, 8))

# ---- 图1：mask ----
plt.subplot(2, 2, 1)
plt.imshow(mask_2d, extent=[x.min(), x.max(), y.min(), y.max()],
           origin='lower', cmap='gray')
plt.title("Line-Space-Line Mask")
plt.xlabel("x")
plt.ylabel("y")

# ---- 图2：printed pattern ----
plt.subplot(2, 2, 2)
plt.imshow(printed_2d, extent=[x.min(), x.max(), y.min(), y.max()],
           origin='lower', cmap='gray')
plt.title("Printed Pattern")
plt.xlabel("x")
plt.ylabel("y")

# ---- 图3：中间横截面 profile ----
plt.subplot(2, 2, 3)
plt.plot(x, profile_mid, label='middle row profile')
plt.title("Middle Row Profile")
plt.xlabel("x")
plt.ylabel("printed")
plt.grid(True)
plt.legend()

# ---- 图4：结果文字总结 ----
plt.subplot(2, 2, 4)
plt.axis('off')
summary_text = (
    f"left_cd = {left_cd:.4f}\n"
    f"right_cd = {right_cd:.4f}\n"
    f"gap_cd = {gap_cd:.4f}\n"
    f"num_features = {num_features}"
)
plt.text(0.1, 0.5, summary_text, fontsize=12)

plt.tight_layout()
plt.show()