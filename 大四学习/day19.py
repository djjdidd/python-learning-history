# 噪声下的鲁棒性分析

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
# 4. 卷积得到无噪声的 aerial image
# ==============================
aerial_2d = convolve2d(mask_2d, psf_2d, mode='same')

# ==============================
# 5. 固定一组待分析的参数
# ==============================
dose = 1.00
threshold = 0.20

# ==============================
# 6. 定义函数：提取中间横截面 CD
# ==============================
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

# ==============================
# 7. 定义函数：给 aerial image 加噪声并打印
# ==============================
def simulate_with_noise(noise_level):
    # 生成和 aerial_2d 同尺寸的随机高斯噪声
    noise = noise_level * np.random.randn(*aerial_2d.shape)

    # 加到原始 aerial image 上.最简化的“工艺波动 / 信号噪声”模型
    aerial_noisy = aerial_2d + noise

    # 为了防止数值太离谱，截断到 0~1
    aerial_noisy = np.clip(aerial_noisy, 0.0, 1.0)

    # 阈值打印
    printed_2d = (dose * aerial_noisy >= threshold).astype(float)

    # 提取中间横截面 CD
    cd_mid = extract_cd_from_middle_row(printed_2d, x)

    # 连通区域个数
    labeled_array, num_features = ndimage.label(printed_2d)

    return aerial_noisy, printed_2d, cd_mid, num_features

# ==============================
# 8. 设置噪声强度和重复次数
# ==============================
noise_level = 0.02
num_trials = 50

# ==============================
# 9. 重复仿真很多次，统计结果
# ==============================
cd_results = []
feature_results = []

for _ in range(num_trials):
    aerial_noisy, printed_2d, cd_mid, num_features = simulate_with_noise(noise_level)

    cd_results.append(cd_mid)
    feature_results.append(num_features)

cd_results = np.array(cd_results)
feature_results = np.array(feature_results)

# ==============================
# 10. 计算统计量
# ==============================
cd_mean = np.mean(cd_results)
cd_std = np.std(cd_results)#很多次结果围绕平均值的波动大小

print("固定参数：")
print("dose =", dose)
print("threshold =", threshold)
print("noise level =", noise_level)
print("重复次数 =", num_trials)

print("\nCD 统计：")
print("平均值 =", cd_mean)
print("标准差 =", cd_std)

print("\n连通区域个数统计：")#连通区域个数分别出现了多少次
unique_features = np.unique(feature_results)#把数组里出现过的不同值找出来，并且去重
for f in unique_features:
    count = np.sum(feature_results == f)
    print(f"num_features = {f}, 次数 = {count}")

# ==============================
# 11. 再拿一次无噪声结果做对比
# ==============================
printed_clean = (dose * aerial_2d >= threshold).astype(float)
cd_clean = extract_cd_from_middle_row(printed_clean, x)
labeled_clean, features_clean = ndimage.label(printed_clean)

# ==============================
# 12. 再生成一次示例噪声结果用于画图
# ==============================
aerial_example, printed_example, cd_example, features_example = simulate_with_noise(noise_level)

# ==============================
# 13. 画图
# ==============================
plt.figure(figsize=(15, 10))

# ---- 图1：无噪声打印图 ----
plt.subplot(2, 3, 1)
plt.imshow(printed_clean, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='gray')
plt.title(f"Clean Printed Pattern\nCD={cd_clean:.4f}, features={features_clean}")
plt.xlabel("x")
plt.ylabel("y")

# ---- 图2：一次有噪声的 aerial image ----
plt.subplot(2, 3, 2)
plt.imshow(aerial_example, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='hot')
plt.title("One Noisy Aerial Image")
plt.xlabel("x")
plt.ylabel("y")

# ---- 图3：一次有噪声的打印图 ----
plt.subplot(2, 3, 3)
plt.imshow(printed_example, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', cmap='gray')
plt.title(f"One Noisy Printed Pattern\nCD={cd_example:.4f}, features={features_example}")
plt.xlabel("x")
plt.ylabel("y")

# ---- 图4：CD 结果直方图 ----
plt.subplot(2, 3, 4)
plt.hist(cd_results, bins=15, edgecolor='black')#直方图，表示把 CD 的取值范围分成 15 个小区间，然后统计每个区间里有多少次试验结果落进去
plt.axvline(cd_mean, color='red', linestyle='--', label='mean')
plt.axvline(cd_clean, color='green', linestyle='--', label='clean CD')
plt.title("Histogram of CD under Noise")
plt.xlabel("CD")
plt.ylabel("Count")
plt.legend()

# ---- 图5：每次试验的 CD 变化 ----
plt.subplot(2, 3, 5)
plt.plot(cd_results, marker='o')#折线图，表示每个点都画一个圆点
plt.axhline(cd_mean, color='red', linestyle='--', label='mean')
plt.axhline(cd_clean, color='green', linestyle='--', label='clean CD')
plt.title("CD over Trials")
plt.xlabel("Trial index")
plt.ylabel("CD")
plt.grid(True)
plt.legend()

# ---- 图6：连通区域个数变化 ----
plt.subplot(2, 3, 6)
plt.plot(feature_results, marker='o')
plt.axhline(features_clean, color='green', linestyle='--', label='clean features')
plt.title("Connected Components over Trials")
plt.xlabel("Trial index")
plt.ylabel("num_features")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()