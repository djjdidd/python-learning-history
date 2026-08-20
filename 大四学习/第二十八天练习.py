import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

# ============================================================
# 第28天练习：从比例热图到整体通过/失败窗口图
# 目标：改变标准，在pass区中找best safe ratio
# 1. 同时扫描 dose 和 threshold
# 2. 每组参数下先做局部状态分类
# 3. 统计 safe / weak / fail 比例
# 4. 再把每个参数点整体判成：
#    pass / borderline / fail
# 5. 画出二维整体窗口图
# ============================================================


# ============================================================
# 1. 建立二维空间网格
# ============================================================
nx, ny = 401, 241
x = np.linspace(-200, 200, nx)
y = np.linspace(-120, 120, ny)
X, Y = np.meshgrid(x, y)

# y方向网格间距（有时候画色带或标注会用到）
dy = y[1] - y[0]


# ============================================================
# 2. 构造一个“中间更危险”的 line-space-line mask
# 思路：
# - 左右线宽度固定
# - y=0 附近 gap 更小
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

        # 中间最危险：yy接近0时，left_center更靠右
        left_center = -60 + 68 * (1 - (yy / 80)**2)

        # 右线固定
        right_center = 85

        # 算左右边界
        left_x1 = left_center - left_width / 2
        left_x2 = left_center + left_width / 2

        right_x1 = right_center - right_width / 2
        right_x2 = right_center + right_width / 2

        # 在第 i 行上画左右两条线
        mask[i, (x >= left_x1) & (x <= left_x2)] = 1.0
        mask[i, (x >= right_x1) & (x <= right_x2)] = 1.0


# ============================================================
# 3. 定义二维高斯 PSF
# ============================================================
sigma = 12.0
psf = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
psf = psf / psf.sum()


# ============================================================
# 4. 卷积得到 aerial image
# 注意：
# 后面扫描 dose 时，不需要重复卷积
# 因为 dose 只是乘在 aerial 前面的缩放
# ============================================================
aerial = fftconvolve(mask, psf, mode='same')


# ============================================================
# 5. 定义：测量一行中的 left / gap / right / num_features
# ============================================================
def measure_row_lsl(row, x):
    # 找出这一行里所有 printed 点的位置
    idx = np.where(row > 0)[0]

    # 如果这一行完全没有 printed 点
    if len(idx) == 0:
        return None, None, None, 0

    # 相邻 printed 点的下标差
    diff_idx = np.diff(idx)

    # 找连续区域断点
    split_points = np.where(diff_idx > 1)[0]

    # feature 数 = 断点数 + 1
    num_features = len(split_points) + 1

    # 如果不是两个独立 feature，则 left/gap/right 不再可靠
    if num_features != 2:
        return None, None, None, num_features

    # 恰好两个 feature 时，切成左右两部分
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
# status_arr 中每个元素属于：
#   safe / weak / fail
# ============================================================
def classify_pattern_rows(printed, x, y,
                          y_min_pattern, y_max_pattern,
                          fail_left_cd, fail_gap_cd, fail_right_cd,
                          weak_left_cd, weak_gap_cd, weak_right_cd):
    valid_y_list = []
    status_list = []

    for i in range(len(y)):
        yy = y[i]

        # 只分析图形有效高度区
        if y_min_pattern <= yy <= y_max_pattern:
            row = printed[i, :]

            left_cd, gap_cd, right_cd, num_features = measure_row_lsl(row, x)

            valid_y_list.append(yy)

            # 如果 feature 数不是 2，直接判 fail
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
# 7. 分类阈值
# ============================================================
fail_left_cd = 24
fail_gap_cd = 20
fail_right_cd = 24

weak_left_cd = 30
weak_gap_cd = 30
weak_right_cd = 30


# ============================================================
# 8. 扫描的 dose 和 threshold 范围
# ============================================================
dose_list = np.linspace(0.45, 1.15, 23)
threshold_list = np.linspace(0.06, 0.24, 21)

# 三个局部比例图
safe_ratio_map = np.zeros((len(threshold_list), len(dose_list)))
weak_ratio_map = np.zeros((len(threshold_list), len(dose_list)))
fail_ratio_map = np.zeros((len(threshold_list), len(dose_list)))

# 整体窗口图：
# 我们用数字编码整体状态，方便后面画图
# pass = 2
# borderline = 1
# fail = 0
window_map = np.zeros((len(threshold_list), len(dose_list)))


# ============================================================
# 9. 双层循环：对每组 (dose, threshold) 计算
#    局部比例 + 整体状态
# ============================================================
for i, threshold in enumerate(threshold_list):
    for j, dose in enumerate(dose_list):

        # 当前参数点下的打印结果
        printed = (dose * aerial >= threshold).astype(int)

        # 做逐行局部分类
        valid_y_arr, status_arr = classify_pattern_rows(
            printed, x, y,
            y_min_pattern, y_max_pattern,
            fail_left_cd, fail_gap_cd, fail_right_cd,
            weak_left_cd, weak_gap_cd, weak_right_cd
        )

        # 统计三种局部状态比例
        safe_ratio = np.mean(status_arr == 'safe')
        weak_ratio = np.mean(status_arr == 'weak')
        fail_ratio = np.mean(status_arr == 'fail')

        # 存进比例图
        safe_ratio_map[i, j] = safe_ratio
        weak_ratio_map[i, j] = weak_ratio
        fail_ratio_map[i, j] = fail_ratio

        # ----------------------------------------------------
        # 根据局部比例，定义这个参数点的整体状态
        #
        # 严格规则：
        # 1) 只要有任何 fail，就整体 fail
        # 2) 如果没有 fail，但有 weak，就整体 borderline
        # 3) 只有 safe_ratio=1 时，才整体 pass
        # ----------------------------------------------------
        if fail_ratio > 0:
            window_map[i, j] = 0      # fail
        elif weak_ratio > 0.2:
            window_map[i, j] = 1      # borderline
        else:
            window_map[i, j] = 2      # pass


# ============================================================
# 10. 找到 pass 区域
# 这里我们也顺便找 所有pass中safe_ratio 最大点，用于比较
# ============================================================
pass_map=(window_map==2)
if np.any(pass_map):#只要 pass_mask 里有任何一个 True，就返回 True
    safe=np.where(pass_map,safe_ratio_map,-1)
    max_idx = np.unravel_index(np.argmax(safe), safe.shape)
    best_i, best_j = max_idx

    best_threshold = threshold_list[best_i]
    best_dose = dose_list[best_j]
    best_safe_ratio = safe_ratio_map[best_i, best_j]
else:
    best_pass_i, best_pass_j = None, None
    best_pass_threshold = None
    best_pass_dose = None
    best_pass_safe_ratio = None
# 对这个 best point 再重新计算一遍 printed 和局部状态
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
# (2) overall window map
# (3) best point 下的 printed pattern
# (4) best point 下的 local failure map
# ============================================================
plt.figure(figsize=(15, 11))

# ------------------------------------------------------------
# 图1：safe_ratio heatmap
# 表示每个参数点下，有多少比例的局部区域是 safe
# ------------------------------------------------------------
plt.subplot(2, 2, 1)
plt.imshow(safe_ratio_map,
           extent=[dose_list.min(), dose_list.max(),
                   threshold_list.min(), threshold_list.max()],
           origin='lower',
           aspect='auto',
           cmap='viridis')
plt.colorbar(label='safe ratio')
plt.scatter(best_dose, best_threshold, marker='x', s=100, label='best safe-ratio point')
plt.xlabel('dose')
plt.ylabel('threshold')
plt.title('Safe Ratio Heatmap')
plt.legend()

# ------------------------------------------------------------
# 图2：整体窗口图
# 0 = fail
# 1 = borderline
# 2 = pass
# ------------------------------------------------------------
plt.subplot(2, 2, 2)
plt.imshow(window_map,
           extent=[dose_list.min(), dose_list.max(),
                   threshold_list.min(), threshold_list.max()],
           origin='lower',
           aspect='auto',
           cmap='plasma')
plt.colorbar(label='overall state code')
plt.scatter(best_dose, best_threshold, marker='x', s=100, label='best safe-ratio point')
plt.xlabel('dose')
plt.ylabel('threshold')
plt.title('Overall Pass / Borderline / Fail Window')
plt.legend()

# ------------------------------------------------------------
# 图3：best point 下的 printed pattern
# ------------------------------------------------------------
plt.subplot(2, 2, 3)
plt.imshow(best_printed,
           extent=[x.min(), x.max(), y.min(), y.max()],
           origin='lower',
           cmap='gray')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Printed Pattern at Best Safe-Ratio Point')

# ------------------------------------------------------------
# 图4：best point 下的局部失效图
# ------------------------------------------------------------
plt.subplot(2, 2, 4)
plt.plot(best_valid_y_arr, best_status_num, 'ko')
plt.yticks([0, 1, 2], ['fail', 'weak', 'safe'])
plt.xlabel('y')
plt.ylabel('local status')
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