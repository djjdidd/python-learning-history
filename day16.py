# dose-threshold 参数扫描入门

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy import ndimage

# ==============================
# 1. 建立二维空间网格
# ==============================
x = np.linspace(-1.0, 1.0, 201)
y = np.linspace(-1.0, 1.0, 201)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)

# ==============================
# 2. 构造两个相邻矩形 mask
# ==============================
# 这样做的好处是：
# 参数变化时，更容易观察桥连 / 断裂 / 连通区域变化
left_rect = ((X >= -0.45) & (X <= -0.10) & (np.abs(Y) <= 0.25))
right_rect = ((X >= 0.10) & (X <= 0.45) & (np.abs(Y) <= 0.25))
mask_2d = np.where(left_rect | right_rect, 1.0, 0.0)

# ==============================
# 3. 构造二维高斯 PSF
# ==============================
sigma = 0.08
psf_2d = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
psf_2d = psf_2d / np.sum(psf_2d)

# ==============================
# 4. 二维卷积得到 aerial image
# ==============================
aerial_2d = convolve2d(mask_2d, psf_2d, mode='same')

# ==============================
# 5. 定义一个函数：计算中间横截面 CD
# ==============================
def extract_cd_from_middle_row(binary_2d, x_coords):
    # 取中间行
    row_mid = binary_2d.shape[0] // 2
    profile = binary_2d[row_mid, :]

    # 找到所有打印位置
    printed_indices = np.where(profile == 1)[0]

    # 如果这一行没有打印区域，就返回 0
    if len(printed_indices) == 0:
        return 0.0

    left_edge = x_coords[printed_indices[0]]
    right_edge = x_coords[printed_indices[-1]]
    cd_value = right_edge - left_edge
    return cd_value

# ==============================
# 6. 定义一个函数：对某组 dose 和 threshold 做测量
# ==============================
def measure_metrics(dose, threshold):
    # 阈值打印
    printed_2d = (dose * aerial_2d >= threshold).astype(float)

    # 中间横截面 CD
    cd_mid = extract_cd_from_middle_row(printed_2d, x)

    # 总打印面积
    area = np.sum(printed_2d) * dx * dy

    # 连通区域个数
    labeled_array, num_features = ndimage.label(printed_2d)

    return printed_2d, cd_mid, area, num_features

# ==============================
# 7. 准备扫描的参数列表
# ==============================
dose_list = np.array([0.80, 0.90, 1.00, 1.10, 1.20])
threshold_list = np.array([0.12,0.15, 0.18, 0.21, 0.24,0.27])

# ==============================
# 8. 创建结果矩阵
# ==============================
# 行对应 threshold
# 列对应 dose
cd_matrix = np.zeros((len(threshold_list), len(dose_list)))
area_matrix = np.zeros((len(threshold_list), len(dose_list)))
feature_matrix = np.zeros((len(threshold_list), len(dose_list)))

# ==============================
# 9. 双循环扫描参数
# ==============================
for i, threshold in enumerate(threshold_list):
    for j, dose in enumerate(dose_list):
        printed_2d, cd_mid, area, num_features = measure_metrics(dose, threshold)

        cd_matrix[i, j] = cd_mid
        area_matrix[i, j] = area
        feature_matrix[i, j] = num_features

        print(f"threshold={threshold:.2f}, dose={dose:.2f} -> "
              f"CD={cd_mid:.4f}, area={area:.4f}, features={num_features}")

# ==============================
# 10. 选一组参数，画出对应打印图形
# ==============================
example_dose = 1.00
example_threshold = 0.18
printed_example, cd_example, area_example, features_example = measure_metrics(example_dose, example_threshold)

# ==============================
# 11. 画图
# ==============================
plt.figure(figsize=(14, 10))

# ---- 图1：mask ----
plt.subplot(2, 2, 1)
plt.imshow(mask_2d, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='gray')
plt.title("2D Mask")
plt.xlabel("x")
plt.ylabel("y")

# ---- 图2：示例打印图 ----
plt.subplot(2, 2, 2)
plt.imshow(printed_example, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='gray')
plt.title(f"Printed Pattern\n(dose={example_dose}, threshold={example_threshold})")
plt.xlabel("x")
plt.ylabel("y")

# ---- 图3：CD 矩阵热图 ----
plt.subplot(2, 2, 3)
plt.imshow(
    cd_matrix,
    extent=[dose_list.min(), dose_list.max(), threshold_list.min(), threshold_list.max()],
    origin='lower',
    aspect='auto'
)
plt.colorbar(label='Middle-row CD')
plt.xlabel("Dose")
plt.ylabel("Threshold")
plt.title("Dose-Threshold CD Matrix")

# ---- 图4：连通区域个数矩阵热图 ----
plt.subplot(2, 2, 4)
plt.imshow(
    feature_matrix,
    extent=[dose_list.min(), dose_list.max(), threshold_list.min(), threshold_list.max()],
    origin='lower',
    aspect='auto'
)
plt.colorbar(label='Number of Connected Components')
plt.xlabel("Dose")
plt.ylabel("Threshold")
plt.title("Dose-Threshold Connectivity Matrix")

plt.tight_layout()
plt.show()