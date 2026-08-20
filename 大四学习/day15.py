#二维图形边缘提取与 CD 测量入门

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

# ==============================
# 1. 建立二维空间网格
# ==============================
# x, y 分别表示横向和纵向坐标
# 一共取 201 个点，范围是 -1 到 1
x = np.linspace(-1.0, 1.0, 201)
y = np.linspace(-1.0, 1.0, 201)

# dx, dy 是网格间距
dx = x[1] - x[0]
dy = y[1] - y[0]

# meshgrid 把一维坐标扩展成二维坐标网格
# X 和 Y 的形状都会是 (201, 201)
X, Y = np.meshgrid(x, y)

# ==============================
# 2. 构造一个二维矩形 mask
# ==============================
# 这里定义一个中心矩形：
# 横向范围 |X| <= 0.35
# 纵向范围 |Y| <= 0.30
mask_2d = np.where((np.abs(X) <= 0.35) & (np.abs(Y) <= 0.30), 1.0, 0.0)

# ==============================
# 3. 构造二维高斯 PSF
# ==============================
# sigma 控制模糊强弱
sigma = 0.14

# 二维高斯函数
psf_2d = np.exp(-(X**2 + Y**2) / (2 * sigma**2))

# 归一化，让 PSF 总和为 1
psf_2d = psf_2d / np.sum(psf_2d)

# ==============================
# 4. 二维卷积，得到 aerial image
# ==============================
# 表示掩模经过二维光学模糊后得到连续成像强度分布
aerial_2d = convolve2d(mask_2d, psf_2d, mode='same')

# ==============================
# 5. 阈值打印
# ==============================
dose = 1.0
threshold = 0.20

# 强度超过阈值的位置打印为 1，否则为 0
printed_2d = (dose * aerial_2d >= threshold).astype(float)

# ==============================
# 6. 定义一个函数：提取某一行的左右边缘和 CD
# ==============================
def extract_cd_from_row(binary_2d, x_coords, row_index):
    """
    输入:
        binary_2d : 二值打印图形，元素只有 0 和 1
        x_coords  : 横向坐标数组
        row_index : 要测量的那一行编号

    输出:
        left_edge  : 左边缘坐标
        right_edge : 右边缘坐标
        cd_value   : 这一行的宽度
        profile    : 这一行的一维横截面
    """

    # 取出指定行的横截面
    profile = binary_2d[row_index, :]

    # 找出这一行里所有被打印出来的位置索引
    printed_indices = np.where(profile == 1)[0]

    # 如果这一行没有任何打印区域，说明这一行是空的
    # 这时边缘和 CD 都没法定义，返回 None
    if len(printed_indices) == 0:
        return None, None, 0.0, profile

    # 左边缘：第一个打印点对应的横坐标
    left_edge = x_coords[printed_indices[0]]

    # 右边缘：最后一个打印点对应的横坐标
    right_edge = x_coords[printed_indices[-1]]

    # CD = 右边缘 - 左边缘
    cd_value = right_edge - left_edge

    return left_edge, right_edge, cd_value, profile

# ==============================
# 7. 选三条横截面：上、中、下
# ==============================
# 注意数组行号从上到下编号，但图像显示时我们常用 origin='lower'
# 这里直接用行索引即可
row_top = 70
row_mid = printed_2d.shape[0] // 2
row_bottom = 130

# 分别提取三条横截面的边缘和 CD
left_top, right_top, cd_top, profile_top = extract_cd_from_row(printed_2d, x, row_top)
left_mid, right_mid, cd_mid, profile_mid = extract_cd_from_row(printed_2d, x, row_mid)
left_bottom, right_bottom, cd_bottom, profile_bottom = extract_cd_from_row(printed_2d, x, row_bottom)

# 输出结果
print("上方横截面:")
print("左边缘 =", left_top, "右边缘 =", right_top, "CD =", cd_top)

print("中间横截面:")
print("左边缘 =", left_mid, "右边缘 =", right_mid, "CD =", cd_mid)

print("下方横截面:")
print("左边缘 =", left_bottom, "右边缘 =", right_bottom, "CD =", cd_bottom)

# ==============================
# 8. 画图
# ==============================
plt.figure(figsize=(14, 8))

# ---- 图1：打印图形，并标出三条测量横线 ----
plt.subplot(2, 2, 1)
plt.imshow(
    printed_2d,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    cmap='gray'
)
plt.axhline(y[row_top], color='red', linestyle='--', label='top row')
plt.axhline(y[row_mid], color='blue', linestyle='--', label='middle row')
plt.axhline(y[row_bottom], color='green', linestyle='--', label='bottom row')
plt.title("Printed Pattern with Measured Rows")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

# ---- 图2：上方横截面 ----
plt.subplot(2, 2, 2)
plt.plot(x, profile_top, label='top profile')
if left_top is not None:
    plt.axvline(left_top, color='red', linestyle='--', label='left edge')
    plt.axvline(right_top, color='purple', linestyle='--', label='right edge')
plt.title(f"Top Row Profile, CD = {cd_top:.4f}")
plt.xlabel("x")
plt.ylabel("printed")
plt.grid(True)
plt.legend()

# ---- 图3：中间横截面 ----
plt.subplot(2, 2, 3)
plt.plot(x, profile_mid, label='middle profile')
if left_mid is not None:
    plt.axvline(left_mid, color='red', linestyle='--', label='left edge')
    plt.axvline(right_mid, color='purple', linestyle='--', label='right edge')
plt.title(f"Middle Row Profile, CD = {cd_mid:.4f}")
plt.xlabel("x")
plt.ylabel("printed")
plt.grid(True)
plt.legend()

# ---- 图4：下方横截面 ----
plt.subplot(2, 2, 4)
plt.plot(x, profile_bottom, label='bottom profile')
if left_bottom is not None:
    plt.axvline(left_bottom, color='red', linestyle='--', label='left edge')
    plt.axvline(right_bottom, color='purple', linestyle='--', label='right edge')
plt.title(f"Bottom Row Profile, CD = {cd_bottom:.4f}")
plt.xlabel("x")
plt.ylabel("printed")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

#我们是用离散网格上的第一个和最后一个打印点来近似边缘，
# 真实边缘可能落在采样点之间，所以结果会受网格分辨率和阈值影响。