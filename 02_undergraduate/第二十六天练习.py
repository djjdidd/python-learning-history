import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

# ============================================================
# 第26天练习：把今天的“最佳 dose 标准”从：safe_ratio 最大改成一个新的综合评价
# 目标：
# 1. 对多个 dose 做扫描
# 2. 每个 dose 下都做逐行状态分类
# 3. 比较 safe / weak / fail 比例
# ============================================================


# ============================================================
# 1. 建立二维空间网格
# x: 横向坐标
# y: 纵向坐标
# X, Y: 二维网格
# ============================================================
nx, ny = 401, 241
x = np.linspace(-200, 200, nx)
y = np.linspace(-120, 120, ny)
X, Y = np.meshgrid(x, y)

# y方向网格间距，后面画 overlay 色带时会用到
dy = y[1] - y[0]


# ============================================================
# 2. 构造二维 mask
# 这里故意让中间 y=0 附近 gap 更小
# 这样中间会更危险，更容易先变 weak / fail
# ============================================================
mask = np.zeros((ny, nx), dtype=float)

# 左右两条线的宽度固定
left_width = 36
right_width = 36

# 图形只出现在这个高度范围内
y_min_pattern = -80
y_max_pattern = 80

for i in range(ny):
    yy = y[i]

    # 只在图形有效高度范围内画结构
    if y_min_pattern <= yy <= y_max_pattern:

        # 设计一个“中间最危险”的结构：
        # y=0 附近 left_center 更靠右，gap 更小
        # 上下两端 left_center 更靠左，gap 更大
        #
        # 这里使用一个简单的二次型变化：
        # abs(yy) 越小（越靠近中间），left_center 越大（越往右）
        # abs(yy) 越大（越靠近上下边缘），left_center 越小（越往左）
        left_center = -70 + 22 * (1 - (yy / 80)**2)

        # 右线中心固定
        right_center = 30

        # 根据中心位置和宽度，算出左右边界
        left_x1 = left_center - left_width / 2
        left_x2 = left_center + left_width / 2

        right_x1 = right_center - right_width / 2
        right_x2 = right_center + right_width / 2

        # 在第 i 行中，把左右线所在的 x 区间设成 1
        mask[i, (x >= left_x1) & (x <= left_x2)] = 1.0
        mask[i, (x >= right_x1) & (x <= right_x2)] = 1.0


# ============================================================
# 3. 定义二维高斯 PSF
# 这里仍然使用简化成像模型
# ============================================================
sigma = 12.0
psf = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
psf = psf / psf.sum()


# ============================================================
# 4. 先卷积得到 aerial image
# 注意：
# aerial image 只和 mask + PSF 有关
# dose 是后面乘上去的，所以这里卷积只做一次
# ============================================================
aerial = fftconvolve(mask, psf, mode='same')


# ============================================================
# 5. 定义：测量一行中的 left / gap / right / num_features
# 输入:
#   row : 一维打印结果（某一行）
#   x   : x坐标
# 输出:
#   left_cd, gap_cd, right_cd, num_features
# 如果这一行不是两个独立 feature，则 left/gap/right 返回 None
# ============================================================
def measure_row_lsl(row, x):
    # 找出这一行中所有 printed 点的位置下标
    idx = np.where(row > 0)[0]

    # 如果这一行完全没有 printed 点
    if len(idx) == 0:
        return None, None, None, 0

    # 相邻 printed 点下标之差
    diff_idx = np.diff(idx)

    # 找到断点（大于1说明不连续）
    split_points = np.where(diff_idx > 1)[0]

    # feature 数 = 断点数 + 1
    num_features = len(split_points) + 1

    # 如果不是两个独立 feature，就不再定义 left/gap/right
    if num_features != 2:
        return None, None, None, num_features

    # 只有两个 feature 时，找到左右两部分
    split = split_points[0]

    left_part = idx[:split + 1]
    right_part = idx[split + 1:]

    # 计算左线宽度、右线宽度、中间 gap
    left_cd = x[left_part[-1]] - x[left_part[0]]
    right_cd = x[right_part[-1]] - x[right_part[0]]
    gap_cd = x[right_part[0]] - x[left_part[-1]]

    return left_cd, gap_cd, right_cd, num_features


# ============================================================
# 6. 定义：对一个 printed pattern 做逐行分类
# 作用：
#   给每一行打 safe / weak / fail 标签
# 输出：
#   valid_y_arr, left_cd_arr, gap_cd_arr, right_cd_arr,
#   num_features_arr, status_arr
# ============================================================
def classify_pattern_rows(printed, x, y,
                          y_min_pattern, y_max_pattern,
                          fail_left_cd, fail_gap_cd, fail_right_cd,
                          weak_left_cd, weak_gap_cd, weak_right_cd):
    valid_y_list = []
    left_cd_list = []
    gap_cd_list = []
    right_cd_list = []
    num_features_list = []
    status_list = []

    for i in range(len(y)):
        yy = y[i]

        # 这里只分析图形有效高度范围内的行
        # 避免把图形外的空白区域误判成 fail
        if y_min_pattern <= yy <= y_max_pattern:
            row = printed[i, :]

            left_cd, gap_cd, right_cd, num_features = measure_row_lsl(row, x)

            valid_y_list.append(yy)
            num_features_list.append(num_features)

            # 如果 feature 数不是 2，直接判 fail
            if num_features != 2:
                status_list.append('fail')

                # 这些几何量此时不再可靠，用 np.nan 占位
                left_cd_list.append(np.nan)
                gap_cd_list.append(np.nan)
                right_cd_list.append(np.nan)

            else:
                # 先存数值
                left_cd_list.append(left_cd)
                gap_cd_list.append(gap_cd)
                right_cd_list.append(right_cd)

                # 先判 fail，再判 weak，最后 safe
                if (left_cd < fail_left_cd) or (gap_cd < fail_gap_cd) or (right_cd < fail_right_cd):
                    status_list.append('fail')
                elif (left_cd < weak_left_cd) or (gap_cd < weak_gap_cd) or (right_cd < weak_right_cd):
                    status_list.append('weak')
                else:
                    status_list.append('safe')

    # 转成 numpy 数组，后面画图和统计更方便
    valid_y_arr = np.array(valid_y_list)
    left_cd_arr = np.array(left_cd_list)
    gap_cd_arr = np.array(gap_cd_list)
    right_cd_arr = np.array(right_cd_list)
    num_features_arr = np.array(num_features_list)
    status_arr = np.array(status_list)

    return valid_y_arr, left_cd_arr, gap_cd_arr, right_cd_arr, num_features_arr, status_arr


# ============================================================
# 7. 设定分类阈值
# 这些阈值不是唯一正确值，而是今天用来训练多条件判定的规则
# ============================================================
fail_left_cd = 24
fail_gap_cd = 20
fail_right_cd = 24

weak_left_cd = 30
weak_gap_cd = 30
weak_right_cd = 30


# ============================================================
# 8. 扫描多个 dose
# threshold 固定
# ============================================================
threshold = 0.18
dose_list = np.linspace(0.80, 1.20, 100)

safe_ratio_list = []
weak_ratio_list = []
fail_ratio_list = []

# 用来记录“最优 dose”对应的结果
best_dose = None
best_score=-1

best_printed = None
best_valid_y_arr = None
best_left_cd_arr = None
best_gap_cd_arr = None
best_right_cd_arr = None
best_status_arr = None


# ============================================================
# 9. 对每个 dose 重复：
#    printed -> classify -> 统计比例
# ============================================================
for dose in dose_list:
    # 当前 dose 下的打印结果
    printed = (dose * aerial >= threshold).astype(int)

    # 对当前 printed pattern 做逐行分类
    valid_y_arr, left_cd_arr, gap_cd_arr, right_cd_arr, num_features_arr, status_arr = classify_pattern_rows(
        printed, x, y,
        y_min_pattern, y_max_pattern,
        fail_left_cd, fail_gap_cd, fail_right_cd,
        weak_left_cd, weak_gap_cd, weak_right_cd
    )

    # 统计三种状态的比例
    safe_ratio = np.mean(status_arr == 'safe')
    weak_ratio = np.mean(status_arr == 'weak')
    fail_ratio = np.mean(status_arr == 'fail')

    safe_ratio_list.append(safe_ratio)
    weak_ratio_list.append(weak_ratio)
    fail_ratio_list.append(fail_ratio)
    score = safe_ratio - 0.5 * weak_ratio - 1.0 * fail_ratio
    # 如果当前 dose 的 safe_ratio 更大，就更新最优结果
    if score > best_score:
        
        best_score = score
        best_dose = dose

        best_printed = printed.copy()
        best_valid_y_arr = valid_y_arr.copy()
        best_left_cd_arr = left_cd_arr.copy()
        best_gap_cd_arr = gap_cd_arr.copy()
        best_right_cd_arr = right_cd_arr.copy()
        best_status_arr = status_arr.copy()


# 转成 numpy 数组，方便画图
safe_ratio_arr = np.array(safe_ratio_list)
weak_ratio_arr = np.array(weak_ratio_list)
fail_ratio_arr = np.array(fail_ratio_list)


# ============================================================
# 10. 把 best_status_arr 转成数字，方便画局部失效图
# fail=0, weak=1, safe=2
# ============================================================
best_status_num = np.zeros(len(best_status_arr), dtype=float)
best_status_num[best_status_arr == 'fail'] = 0
best_status_num[best_status_arr == 'weak'] = 1
best_status_num[best_status_arr == 'safe'] = 2


# ============================================================
# 11. 画图
# 这次画 4 张图：
# (1) safe/weak/fail 比例 vs dose
# (2) 最优 dose 下的 printed pattern
# (3) 最优 dose 下的 row-wise metrics
# (4) 最优 dose 下的 local failure map
# ============================================================
plt.figure(figsize=(15, 11))

# ------------------------------------------------------------
# 图1：不同 dose 下，safe / weak / fail 比例怎么变化
# ------------------------------------------------------------
plt.subplot(2, 2, 1)
plt.plot(dose_list, safe_ratio_arr, 'o-', label='safe ratio')
plt.plot(dose_list, weak_ratio_arr, 'o-', label='weak ratio')
plt.plot(dose_list, fail_ratio_arr, 'o-', label='fail ratio')
plt.axvline(best_dose, color='red', linestyle='--', label='best dose')
plt.xlabel('dose')
plt.ylabel('ratio')
plt.title('Local Status Ratios vs Dose')
plt.legend()

# ------------------------------------------------------------
# 图2：最优 dose 下的 printed pattern
# ------------------------------------------------------------
plt.subplot(2, 2, 2)
plt.imshow(best_printed, extent=[x.min(), x.max(), y.min(), y.max()],
           origin='lower', cmap='gray')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Printed Pattern at Best Dose = {best_dose:.2f}')

# ------------------------------------------------------------
# 图3：最优 dose 下的逐行指标
# ------------------------------------------------------------
plt.subplot(2, 2, 3)
plt.plot(best_valid_y_arr, best_left_cd_arr, label='left_cd')
plt.plot(best_valid_y_arr, best_gap_cd_arr, label='gap_cd')
plt.plot(best_valid_y_arr, best_right_cd_arr, label='right_cd')

# 画出 weak/fail 阈值线，帮助看哪些地方接近危险
plt.axhline(fail_gap_cd, color='red', linestyle='--', alpha=0.6, label='fail gap threshold')
plt.axhline(weak_gap_cd, color='orange', linestyle='--', alpha=0.6, label='weak gap threshold')

plt.xlabel('y')
plt.ylabel('CD / gap')
plt.title('Row-wise Metrics at Best Dose')
plt.legend()

# ------------------------------------------------------------
# 图4：最优 dose 下的局部失效图
# 横轴：y位置
# 纵轴：状态（fail / weak / safe）
# ------------------------------------------------------------
plt.subplot(2, 2, 4)
plt.plot(best_valid_y_arr, best_status_num, 'ko')
plt.yticks([0, 1, 2], ['fail', 'weak', 'safe'])
plt.xlabel('y')
plt.ylabel('status')
plt.title('Local Failure Map at Best Dose')

summary_text = (
    f"best dose = {best_dose:.2f}\n"
    f"max score = {best_score:.2%}"
)

plt.gcf().text(0.68, 0.03, summary_text, fontsize=11,
               bbox=dict(facecolor='white', alpha=0.85))

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()