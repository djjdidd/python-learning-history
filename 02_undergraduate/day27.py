import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

# ============================================================
# 第27天：二维参数扫描下的局部工艺窗口热图
# 目标：
# 1. 同时扫描 dose 和 threshold
# 2. 每组参数下都做逐行状态分类
# 3. 统计 safe_ratio / fail_ratio
# 4. 画出二维 heatmap
# ============================================================


# ============================================================
# 1. 建立二维空间网格
# ============================================================
nx, ny = 401, 241
x = np.linspace(-200, 200, nx)
y = np.linspace(-120, 120, ny)
X, Y = np.meshgrid(x, y)

# y方向网格间距，后面画色带时常用
dy = y[1] - y[0]


# ============================================================
# 2. 构造一个“中间最危险”的 line-space-line mask
# 思路：
# - 左右线宽度固定
# - 中间 y=0 附近 gap 更小
# - 上下两端 gap 更大
# ============================================================
mask = np.zeros((ny, nx), dtype=float)

left_width = 36
right_width = 36

y_min_pattern = -80
y_max_pattern = 80

for i in range(ny):
    yy = y[i]

    # 只在图形有效高度范围内画结构
    if y_min_pattern <= yy <= y_max_pattern:

        # 中间更危险的设计：
        # 当 yy 接近 0 时，left_center 更靠右，gap 更小
        # 当 yy 接近 ±80 时，left_center 更靠左，gap 更大
        left_center = -70 + 60 * (1 - (yy / 80)**2)

        # 右线中心固定
        right_center =55

        # 根据中心 + 宽度算边界
        left_x1 = left_center - left_width / 2
        left_x2 = left_center + left_width / 2

        right_x1 = right_center - right_width / 2
        right_x2 = right_center + right_width / 2

        # 在第 i 行中，把左右线所在区间设成 1
        mask[i, (x >= left_x1) & (x <= left_x2)] = 1.0
        mask[i, (x >= right_x1) & (x <= right_x2)] = 1.0


# ============================================================
# 3. 定义二维高斯 PSF
# ============================================================
sigma = 12.0
psf = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
psf = psf / psf.sum()


# ============================================================
# 4. 先卷积得到 aerial image
# 注意：
# 这个只和 mask + PSF 有关
# 后面 dose 变化时不需要重新卷积
# ============================================================
aerial = fftconvolve(mask, psf, mode='same')


# ============================================================
# 5. 定义：测量一行中的 left / gap / right / num_features
# ============================================================
def measure_row_lsl(row, x):
    # 找出这一行中所有 printed 点的位置
    idx = np.where(row > 0)[0]

    # 如果这一行没有 printed 点
    if len(idx) == 0:
        return None, None, None, 0

    # 相邻 printed 点下标之差
    diff_idx = np.diff(idx)

    # 找连续区域断点
    split_points = np.where(diff_idx > 1)[0]

    # feature 数 = 断点数 + 1
    num_features = len(split_points) + 1

    # 如果不是两个 feature，就不定义 left/gap/right
    if num_features != 2:
        return None, None, None, num_features

    # 切成左右两部分
    split = split_points[0]
    left_part = idx[:split + 1]
    right_part = idx[split + 1:]

    # 计算几何量
    left_cd = x[left_part[-1]] - x[left_part[0]]
    right_cd = x[right_part[-1]] - x[right_part[0]]
    gap_cd = x[right_part[0]] - x[left_part[-1]]

    return left_cd, gap_cd, right_cd, num_features


# ============================================================
# 6. 定义：对一个 printed pattern 做逐行分类
# 输出：
#   valid_y_arr, status_arr
# 这里只返回今天最核心的状态结果，先不返回全部几何量
# ============================================================
def classify_pattern_rows(printed, x, y,
                          y_min_pattern, y_max_pattern,
                          fail_left_cd, fail_gap_cd, fail_right_cd,
                          weak_left_cd, weak_gap_cd, weak_right_cd):
    valid_y_list = []
    status_list = []

    for i in range(len(y)):
        yy = y[i]

        # 只分析图形有效区，避免把图形外空白误判成 fail
        if y_min_pattern <= yy <= y_max_pattern:
            row = printed[i, :]

            left_cd, gap_cd, right_cd, num_features = measure_row_lsl(row, x)

            valid_y_list.append(yy)

            # 先处理 feature 数异常
            if num_features != 2:
                status_list.append('fail')
            else:
                # 先判 fail，再判 weak，最后 safe
                if (left_cd < fail_left_cd) or (gap_cd < fail_gap_cd) or (right_cd < fail_right_cd):
                    status_list.append('fail')
                elif (left_cd < weak_left_cd) or (gap_cd < weak_gap_cd) or (right_cd < weak_right_cd):
                    status_list.append('weak')
                else:
                    status_list.append('safe')

    valid_y_arr = np.array(valid_y_list)
    status_arr = np.array(status_list)

    return valid_y_arr, status_arr


# ============================================================
# 7. 设定分类阈值
# ============================================================
fail_left_cd = 24
fail_gap_cd = 20
fail_right_cd = 24

weak_left_cd = 30
weak_gap_cd = 30
weak_right_cd = 30


# ============================================================
# 8. 设定二维参数扫描范围
# dose_list: 横向扫描
# threshold_list: 纵向扫描
# ============================================================
dose_list = np.linspace(0.85, 1.15, 50)
threshold_list = np.linspace(0.14, 0.24, 50)

# 预先分配二维数组，用来存每组参数下的状态比例
# safe_ratio_map[i, j] 表示：
#   threshold_list[i], dose_list[j] 这一组参数下的 safe 比例
safe_ratio_map = np.zeros((len(threshold_list), len(dose_list)))
weak_ratio_map = np.zeros((len(threshold_list), len(dose_list)))
fail_ratio_map = np.zeros((len(threshold_list), len(dose_list)))


# ============================================================
# 9. 双层循环：对每组 (dose, threshold) 做分类统计
# 外层扫 threshold
# 内层扫 dose
# ============================================================
for i, threshold in enumerate(threshold_list):
    for j, dose in enumerate(dose_list):

        # 当前参数组合下的打印结果
        printed = (dose * aerial >= threshold).astype(int)

        # 做逐行分类
        valid_y_arr, status_arr = classify_pattern_rows(
            printed, x, y,
            y_min_pattern, y_max_pattern,
            fail_left_cd, fail_gap_cd, fail_right_cd,
            weak_left_cd, weak_gap_cd, weak_right_cd
        )

        # 统计三种状态比例
        safe_ratio = np.mean(status_arr == 'safe')
        weak_ratio = np.mean(status_arr == 'weak')
        fail_ratio = np.mean(status_arr == 'fail')

        # 存到二维 map 中
        safe_ratio_map[i, j] = safe_ratio
        weak_ratio_map[i, j] = weak_ratio
        fail_ratio_map[i, j] = fail_ratio


# ============================================================
# 10. 找 safe_ratio 最大的位置
# ============================================================
# np.argmax(safe_ratio_map) 先找到二维数组最大值的“一维展开编号”
# np.unravel_index(..., safe_ratio_map.shape) 再把它还原成二维下标 (行, 列)
# 这样就能知道最大 safe_ratio 对应的是哪个 threshold 和哪个 dose
max_idx = np.unravel_index(np.argmax(safe_ratio_map), safe_ratio_map.shape)

best_i, best_j = max_idx
best_threshold = threshold_list[best_i]
best_dose = dose_list[best_j]
best_safe_ratio = safe_ratio_map[best_i, best_j]

# 对这组最优参数，再算一次 printed 和状态，用于展示
best_printed = (best_dose * aerial >= best_threshold).astype(int)
best_valid_y_arr, best_status_arr = classify_pattern_rows(
    best_printed, x, y,
    y_min_pattern, y_max_pattern,
    fail_left_cd, fail_gap_cd, fail_right_cd,
    weak_left_cd, weak_gap_cd, weak_right_cd
)

# 把 best_status_arr 转成数字，方便画 local failure map
best_status_num = np.zeros(len(best_status_arr), dtype=float)
best_status_num[best_status_arr == 'fail'] = 0
best_status_num[best_status_arr == 'weak'] = 1
best_status_num[best_status_arr == 'safe'] = 2


# ============================================================
# 11. 画图
# 这次画 4 张图：
# (1) safe_ratio heatmap
# (2) fail_ratio heatmap
# (3) best condition 下的 printed pattern
# (4) best condition 下的 local failure map
# ============================================================
plt.figure(figsize=(15, 11))

# ------------------------------------------------------------
# 图1：safe_ratio heatmap
# extent 用真实参数坐标显示横纵轴
# origin='lower' 让小 threshold 在下方
# ------------------------------------------------------------
plt.subplot(2, 2, 1)
plt.imshow(safe_ratio_map,
           extent=[dose_list.min(), dose_list.max(),
                   threshold_list.min(), threshold_list.max()],
           origin='lower',
           aspect='auto',
           cmap='viridis')
plt.colorbar(label='safe ratio')
plt.scatter(best_dose, best_threshold, marker='x', s=100, label='best point')
plt.xlabel('dose')
plt.ylabel('threshold')
plt.title('Safe Ratio Heatmap')
plt.legend()

# ------------------------------------------------------------
# 图2：fail_ratio heatmap
# ------------------------------------------------------------
plt.subplot(2, 2, 2)
plt.imshow(fail_ratio_map,
           extent=[dose_list.min(), dose_list.max(),
                   threshold_list.min(), threshold_list.max()],
           origin='lower',
           aspect='auto',
           cmap='magma')
plt.colorbar(label='fail ratio')
plt.scatter(best_dose, best_threshold, marker='x', s=100, label='best point')
plt.xlabel('dose')
plt.ylabel('threshold')
plt.title('Fail Ratio Heatmap')
plt.legend()

# ------------------------------------------------------------
# 图3：best condition 下的 printed pattern
# ------------------------------------------------------------
plt.subplot(2, 2, 3)
plt.imshow(best_printed,
           extent=[x.min(), x.max(), y.min(), y.max()],
           origin='lower',
           cmap='gray')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Printed Pattern at Best Point')

# ------------------------------------------------------------
# 图4：best condition 下的局部失效图
# ------------------------------------------------------------
plt.subplot(2, 2, 4)
plt.plot(best_valid_y_arr, best_status_num, 'ko')
plt.yticks([0, 1, 2], ['fail', 'weak', 'safe'])
plt.xlabel('y')
plt.ylabel('status')
plt.title('Local Failure Map at Best Point')

summary_text = (
    f"best dose = {best_dose:.3f}\n"
    f"best threshold = {best_threshold:.3f}\n"
    f"max safe ratio = {best_safe_ratio:.2%}"
)

plt.gcf().text(0.67, 0.03, summary_text, fontsize=11,
               bbox=dict(facecolor='white', alpha=0.85))

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()