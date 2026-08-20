#线-空-线结构的可用工艺窗口分析
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy import ndimage
x = np.linspace(-1.0, 1.0, 201)
y = np.linspace(-1.0, 1.0, 201)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)
left_rect = ((X >= -0.55) & (X <= -0.25) & (np.abs(Y) <= 0.30))
right_rect = ((X >= 0.25) & (X <= 0.55) & (np.abs(Y) <= 0.30))
mask_2d = np.where(left_rect | right_rect, 1.0, 0.0)
sigma = 0.08
psf_2d = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
psf_2d = psf_2d / np.sum(psf_2d)
aerial_2d = convolve2d(mask_2d, psf_2d, mode='same')
def extract_line_and_gap_metrics(binary_2d, x_coords):
    # 取中间行
    row_mid = binary_2d.shape[0] // 2
    profile = binary_2d[row_mid, :]

    # 找到所有打印点的索引
    printed_indices = np.where(profile == 1)[0]

    # 如果完全没有打印，直接返回 0
    if len(printed_indices) == 0:
        return 0.0, 0.0, 0.0, profile

    # 找相邻索引之间的差
    # 如果差值 > 1，说明中间断开了
    diff_indices = np.diff(printed_indices)

    # 找到断点位置
    split_points = np.where(diff_indices > 1)[0]

    # 如果 split_points 长度 < 1，说明只有一整段连续打印
    # 很可能已经桥连成一整块了
    if len(split_points) < 1:
        left_cd = x_coords[printed_indices[-1]] - x_coords[printed_indices[0]]
        right_cd = 0.0
        gap_cd = 0.0
        return left_cd, right_cd, gap_cd, profile

    # 取第一处断点，把打印区域分成左段和右段
    split_idx = split_points[0]

    left_segment = printed_indices[:split_idx + 1]
    right_segment = printed_indices[split_idx + 1:]

    # 左线宽
    left_cd = x_coords[left_segment[-1]] - x_coords[left_segment[0]]

    # 右线宽
    right_cd = x_coords[right_segment[-1]] - x_coords[right_segment[0]]

    # 中间空隙宽度 = 右线左边缘 - 左线右边缘
    gap_cd = x_coords[right_segment[0]] - x_coords[left_segment[-1]]

    return left_cd, right_cd, gap_cd, profile
def measure_metrics(dose, threshold):
    # 阈值打印
    printed_2d = (dose * aerial_2d >= threshold).astype(float)

    # 中间横截面 CD
    left_cd, right_cd, gap_cd, profile = extract_line_and_gap_metrics(printed_2d, x)

    # 总面积
    area = np.sum(printed_2d) * dx * dy

    # 连通区域个数
    labeled_array, num_features = ndimage.label(printed_2d)

    return printed_2d, left_cd, right_cd, gap_cd, profile, area, num_features
def evaluate_robustness(dose, threshold, noise_level=0.02, num_trials=20):
    left_cd_results = []
    right_cd_results = []
    gap_cd_results = []
    feature_results = []

    for _ in range(num_trials):
        # 加随机高斯噪声
        noise = noise_level * np.random.randn(*aerial_2d.shape)
        aerial_noisy = aerial_2d + noise

        # 防止数值太离谱
        aerial_noisy = np.clip(aerial_noisy, 0.0, 1.0)

        # 阈值打印
        printed_noisy = (dose * aerial_noisy >= threshold).astype(float)

        # 统计 CD
        left_cd, right_cd, gap_cd, profile = extract_line_and_gap_metrics(printed_noisy, x)
        left_cd_results.append(left_cd)
        right_cd_results.append(right_cd)
        gap_cd_results.append(gap_cd)

        # 统计连通区域数
        labeled_array, num_features = ndimage.label(printed_noisy)
        feature_results.append(num_features)

    left_cd_results = np.array(left_cd_results)
    right_cd_results = np.array(right_cd_results)
    gap_cd_results = np.array(gap_cd_results)
    feature_results = np.array(feature_results)

    gap_std = np.std(gap_cd_results)

    # 连通区域数最常出现的值
    most_common_feature = np.bincount(feature_results.astype(int)).argmax()
#np.bincount()统计每个整数出现了多少次.argmax()找最大值所在的位置
    return gap_std, most_common_feature
dose_list = np.array([0.80, 0.90, 1.00, 1.10, 1.20])
threshold_list = np.array([0.15, 0.18, 0.21, 0.24, 0.27])

# ==============================
# 9. 创建结果矩阵
# ==============================
left_cd_matrix = np.zeros((len(threshold_list), len(dose_list)))
right_cd_matrix = np.zeros((len(threshold_list), len(dose_list)))
feature_matrix = np.zeros((len(threshold_list), len(dose_list)))
gap_matrix = np.zeros((len(threshold_list), len(dose_list)))
gap_std_matrix = np.zeros((len(threshold_list), len(dose_list)))
for i, threshold in enumerate(threshold_list):
    for j, dose in enumerate(dose_list):
        printed_2d, left_cd, right_cd, gap_cd, profile, area, num_features = measure_metrics(dose, threshold)
        gap_std, most_common_feature = evaluate_robustness(
            dose, threshold, noise_level=0.02, num_trials=20
        )

        left_cd_matrix[i, j] = left_cd
        right_cd_matrix[i, j] = right_cd
        feature_matrix[i, j] = num_features
        gap_matrix[i, j] = gap_cd
target_cd_region = (left_cd_matrix > 0) & (right_cd_matrix > 0)

# 条件 2：连通区域个数正确
# 这里假设希望保持两个分离区域，不桥连
topology_region = (feature_matrix == 2)

# 条件 3：面积不要太离谱
# 这里先给一个简化范围
area_region = (gap_matrix >= 0.15) & (gap_matrix <= 0.35)

# 条件 4：噪声下要相对稳定
# 这里用 CD 标准差小于某个阈值作为“稳”的近似标准
robust_region = gap_std_matrix <= 0.04

# 最终“可用区域”
good_region = (
    target_cd_region &
    topology_region &
    area_region &
    robust_region
).astype(float)
example_dose = 1.00
example_threshold = 0.21
printed_1, left_1, right_1, gap_1, profile1, area1, num1_features = measure_metrics(
    example_dose, example_threshold
)

# ==============================
# 13. 画图
# ==============================
plt.figure(figsize=(16, 12))

# ---- 图1：原始 mask ----
plt.subplot(2, 3, 1)
plt.imshow(mask_2d, extent=[x.min(), x.max(), y.min(), y.max()],
           origin='lower', cmap='gray')
plt.title("2D Mask")
plt.xlabel("x")
plt.ylabel("y")

# ---- 图2：示例打印图 ----
plt.subplot(2, 3, 2)
plt.imshow(printed_1, extent=[x.min(), x.max(), y.min(), y.max()],
           origin='lower', cmap='gray')
plt.title(f"Printed Pattern\n(dose={example_dose}, threshold={example_threshold})")
plt.xlabel("x")
plt.ylabel("y")

# ---- 图3：CD 热图 ----
plt.subplot(2, 3, 3)
plt.imshow(left_cd_matrix,
           extent=[dose_list.min(), dose_list.max(),
                   threshold_list.min(), threshold_list.max()],
           origin='lower', aspect='auto')
plt.colorbar(label='left CD')
plt.xlabel("Dose")
plt.ylabel("Threshold")
plt.title("left CD Heatmap")

# ---- 图4：面积热图 ----
plt.subplot(2, 3, 4)
plt.imshow(right_cd_matrix,
           extent=[dose_list.min(), dose_list.max(),
                   threshold_list.min(), threshold_list.max()],
           origin='lower', aspect='auto')
plt.colorbar(label='right cd')
plt.xlabel("Dose")
plt.ylabel("Threshold")
plt.title("right cd Heatmap")

# ---- 图5：CD 标准差热图 ----
plt.subplot(2, 3, 5)
plt.imshow(gap_matrix,
           extent=[dose_list.min(), dose_list.max(),
                   threshold_list.min(), threshold_list.max()],
           origin='lower', aspect='auto')
plt.colorbar(label='gap')
plt.xlabel("Dose")
plt.ylabel("Threshold")
plt.title("gap Heatmap")

# ---- 图6：最终可用区域 ----
plt.subplot(2, 3, 6)
plt.imshow(good_region,
           extent=[dose_list.min(), dose_list.max(),
                   threshold_list.min(), threshold_list.max()],
           origin='lower', aspect='auto', cmap='gray')
plt.xlabel("Dose")
plt.ylabel("Threshold")
plt.title("Good Parameter Region")

plt.tight_layout()
plt.show()