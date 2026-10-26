#前20天总结
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy import ndimage


#Step 1：建立空间网格
x = np.linspace(-1.0, 1.0, 201)
y = np.linspace(-1.0, 1.0, 201)
dx = x[1] - x[0]
dy = y[1] - y[0]
X, Y = np.meshgrid(x, y)


#Step 2：定义 mask
left_rect = ((X >= -0.45) & (X <= -0.10) & (np.abs(Y) <= 0.25))
right_rect = ((X >= 0.10) & (X <= 0.45) & (np.abs(Y) <= 0.25))
mask_2d = np.where(left_rect | right_rect, 1.0, 0.0)


#Step 3：定义 PSF
sigma = 0.08
psf_2d = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
psf_2d = psf_2d / np.sum(psf_2d)


#Step 4：卷积得到 aerial image
aerial_2d = convolve2d(mask_2d, psf_2d, mode='same')


#Step 5：提取几何量
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


#Step 6：加入噪声
def evaluate_robustness(dose, threshold, noise_level=0.02, num_trials=20):
    cd_results = []
    feature_results = []
    for _ in range(num_trials):
        noise = noise_level * np.random.randn(*aerial_2d.shape)
        aerial_noisy = aerial_2d + noise
        aerial_noisy = np.clip(aerial_noisy, 0.0, 1.0)
        printed_noisy = (dose * aerial_noisy >= threshold).astype(float)
        cd_mid = extract_cd_from_middle_row(printed_noisy, x)
        cd_results.append(cd_mid)
        labeled_array, num_features = ndimage.label(printed_noisy)
        feature_results.append(num_features)
    cd_results = np.array(cd_results)
    feature_results = np.array(feature_results)
    cd_mean = np.mean(cd_results)
    cd_std = np.std(cd_results)
    most_common_feature = np.bincount(feature_results.astype(int)).argmax()
    return cd_mean, cd_std, most_common_feature


#Step 7：参数扫描
# 设置扫描参数
dose_list = np.array([0.80, 0.90, 1.00, 1.10, 1.20])
threshold_list = np.array([0.15, 0.18, 0.21, 0.24, 0.27])
# 创建结果矩阵
cd_matrix = np.zeros((len(threshold_list), len(dose_list)))
area_matrix = np.zeros((len(threshold_list), len(dose_list)))
feature_matrix = np.zeros((len(threshold_list), len(dose_list)))
cd_std_matrix = np.zeros((len(threshold_list), len(dose_list)))
# 双循环扫描参数
for i, threshold in enumerate(threshold_list):
    for j, dose in enumerate(dose_list):
        printed_2d, cd_mid, area, num_features = measure_metrics(dose, threshold)
        cd_mean, cd_std, most_common_feature = evaluate_robustness(
            dose, threshold, noise_level=0.02, num_trials=20
        )
        cd_matrix[i, j] = cd_mid
        area_matrix[i, j] = area
        feature_matrix[i, j] = num_features
        cd_std_matrix[i, j] = cd_std
        print(
            f"threshold={threshold:.2f}, dose={dose:.2f} -> "
            f"CD={cd_mid:.4f}, area={area:.4f}, "
            f"features={num_features}, CD_std={cd_std:.4f}"
        )


#Step 8：定义评价标准
# 条件 1：CD 在目标范围内
target_cd_region = ((cd_matrix >= 0.85) & (cd_matrix <= 1.10))
# 条件 2：连通区域个数正确
# 这里假设希望保持两个分离区域，不桥连
topology_region = (feature_matrix == 2)
# 条件 3：面积不要太离谱
# 这里先给一个简化范围
area_region = ((area_matrix >= 0.30) & (area_matrix <= 0.50))
# 条件 4：噪声下要相对稳定
# 这里用 CD 标准差小于某个阈值作为“稳”的近似标准
robust_region = (cd_std_matrix <= 0.05)


#Step 9：筛选可用区域
good_region = (
    target_cd_region &
    topology_region &
    area_region &
    robust_region
).astype(float)


#Step 10：可视化结果
example_dose = 1.00
example_threshold = 0.21
printed_example, cd_example, area_example, features_example = measure_metrics(
    example_dose, example_threshold
)
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
plt.imshow(printed_example, extent=[x.min(), x.max(), y.min(), y.max()],
           origin='lower', cmap='gray')
plt.title(f"Printed Pattern\n(dose={example_dose}, threshold={example_threshold})")
plt.xlabel("x")
plt.ylabel("y")
# ---- 图3：CD 热图 ----
plt.subplot(2, 3, 3)
plt.imshow(cd_matrix,
           extent=[dose_list.min(), dose_list.max(),
                   threshold_list.min(), threshold_list.max()],
           origin='lower', aspect='auto')
plt.colorbar(label='Middle-row CD')
plt.xlabel("Dose")
plt.ylabel("Threshold")
plt.title("CD Heatmap")
# ---- 图4：面积热图 ----
plt.subplot(2, 3, 4)
plt.imshow(area_matrix,
           extent=[dose_list.min(), dose_list.max(),
                   threshold_list.min(), threshold_list.max()],
           origin='lower', aspect='auto')
plt.colorbar(label='Printed Area')
plt.xlabel("Dose")
plt.ylabel("Threshold")
plt.title("Area Heatmap")
# ---- 图5：CD 标准差热图 ----
plt.subplot(2, 3, 5)
plt.imshow(cd_std_matrix,
           extent=[dose_list.min(), dose_list.max(),
                   threshold_list.min(), threshold_list.max()],
           origin='lower', aspect='auto')
plt.colorbar(label='CD Std under Noise')
plt.xlabel("Dose")
plt.ylabel("Threshold")
plt.title("Robustness Heatmap")
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