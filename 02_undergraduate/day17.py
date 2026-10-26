# 参数扫描结果热图与二维工艺窗口入门

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy import ndimage

x = np.linspace(-1.0, 1.0, 201)
y = np.linspace(-1.0, 1.0, 201)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)

left_rect = ((X >= -0.45) & (X <= -0.10) & (np.abs(Y) <= 0.25))
right_rect = ((X >= 0.10) & (X <= 0.45) & (np.abs(Y) <= 0.25))
mask_2d = np.where(left_rect | right_rect, 1.0, 0.0)

sigma = 0.08
psf_2d = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
psf_2d = psf_2d / np.sum(psf_2d)

aerial_2d = convolve2d(mask_2d, psf_2d, mode='same')

def extract_cd_from_middle_row(binary_2d, x_coords):
    row_mid = binary_2d.shape[0] // 2
    profile = binary_2d[row_mid, :]

    printed_indices = np.where(profile == 1)[0]

    if len(printed_indices) == 0:
        return 0.0

    left_edge = x_coords[printed_indices[0]]
    right_edge = x_coords[printed_indices[-1]]
    cd_value = right_edge - left_edge
    return cd_value

def measure_metrics(dose, threshold):
    printed_2d = (dose * aerial_2d >= threshold).astype(float)
    cd_mid = extract_cd_from_middle_row(printed_2d, x)
    area = np.sum(printed_2d) * dx * dy
    labeled_array, num_features = ndimage.label(printed_2d)
    return printed_2d, cd_mid, area, num_features

dose_list = np.array([0.80, 0.90, 1.00, 1.10, 1.20])
threshold_list = np.array([0.15, 0.18, 0.21, 0.24, 0.27])

cd_matrix = np.zeros((len(threshold_list), len(dose_list)))
area_matrix = np.zeros((len(threshold_list), len(dose_list)))
feature_matrix = np.zeros((len(threshold_list), len(dose_list)))

for i, threshold in enumerate(threshold_list):
    for j, dose in enumerate(dose_list):
        printed_2d, cd_mid, area, num_features = measure_metrics(dose, threshold)

        cd_matrix[i, j] = cd_mid
        area_matrix[i, j] = area
        feature_matrix[i, j] = num_features

# ==============================
# 10. 构造“目标 CD 区域”布尔矩阵
# ==============================
# 这里假设我们希望中间 CD 大约在 0.85 到 1.10 之间
target_region = ((cd_matrix >= 0.85) & (cd_matrix <= 1.10)).astype(float)
# ==============================
# 11. 构造“拓扑正确区域”布尔矩阵
# ==============================
# 这里假设我们希望还是两个连通区域
topology_region = (feature_matrix == 2).astype(float)
# ==============================
# 12. 构造“同时满足尺寸和拓扑要求”的区域
# ==============================
good_region = ((target_region == 1) & (topology_region == 1)).astype(float)


example_dose = 1.00
example_threshold = 0.18
printed_example, cd_example, area_example, features_example = measure_metrics(example_dose, example_threshold)

plt.figure(figsize=(15, 12))

# ---- 图1：原始 mask ----
plt.subplot(2, 3, 1)
plt.imshow(mask_2d, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='gray')
plt.title("2D Mask")
plt.xlabel("x")
plt.ylabel("y")

# ---- 图2：示例打印图形 ----
plt.subplot(2, 3, 2)
plt.imshow(printed_example, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='gray')
plt.title(f"Printed Pattern\n(dose={example_dose}, threshold={example_threshold})")
plt.xlabel("x")
plt.ylabel("y")

# ---- 图3：CD 热图 ----
plt.subplot(2, 3, 3)
plt.imshow(
    cd_matrix,
    extent=[dose_list.min(), dose_list.max(), threshold_list.min(), threshold_list.max()],
    origin='lower',
    aspect='auto'
)
plt.colorbar(label='Middle-row CD')
plt.xlabel("Dose")
plt.ylabel("Threshold")
plt.title("CD Heatmap")

# ---- 图4：面积热图 ----
plt.subplot(2, 3, 4)
plt.imshow(
    area_matrix,
    extent=[dose_list.min(), dose_list.max(), threshold_list.min(), threshold_list.max()],
    origin='lower',
    aspect='auto'
)
plt.colorbar(label='Printed Area')
plt.xlabel("Dose")
plt.ylabel("Threshold")
plt.title("Area Heatmap")

# ---- 图5：连通区域个数热图 ----
plt.subplot(2, 3, 5)
plt.imshow(
    feature_matrix,
    extent=[dose_list.min(), dose_list.max(), threshold_list.min(), threshold_list.max()],
    origin='lower',
    aspect='auto'
)
plt.colorbar(label='Number of Connected Components')
plt.xlabel("Dose")
plt.ylabel("Threshold")
plt.title("Connectivity Heatmap")

# ---- 图6：满足条件的“较好区域” ----
plt.subplot(2, 3, 6)
plt.imshow(
    good_region,
    extent=[dose_list.min(), dose_list.max(), threshold_list.min(), threshold_list.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.xlabel("Dose")
plt.ylabel("Threshold")
plt.title("Good Parameter Region")

plt.tight_layout()
plt.show()

# ==============================
# 15. 打印矩阵结果
# ==============================
print("CD matrix =")
print(cd_matrix)

print("\nArea matrix =")
print(area_matrix)

print("\nFeature matrix =")
print(feature_matrix)

print("\nGood region matrix =")
print(good_region)