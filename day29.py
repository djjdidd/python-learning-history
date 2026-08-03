#整体窗口规则从“硬判定”升级到“带容忍度的判定”
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve


# =========================================================
# 1. 建立二维网格
# =========================================================
nx, ny = 401, 241
x = np.linspace(-2.0, 2.0, nx)
y = np.linspace(-1.2, 1.2, ny)
dx = x[1] - x[0]
dy = y[1] - y[0]

xx, yy = np.meshgrid(x, y)


# =========================================================
# 2. 构造二维 line-space-line mask
#    左线、右线中间留一个 gap
#    再让中间一部分高度区域稍微更难打印（人为加一点局部挑战）
# =========================================================
mask = np.zeros((ny, nx), dtype=float)

# 图形有效高度范围
y_min_pattern = -0.75
y_max_pattern = 0.75

# 基础 line-space-line 参数
left_x1, left_x2 = -1.10, -0.35
right_x1, right_x2 = 0.35, 1.10

# 先做一个基础 line-space-line
pattern_region = (yy >= y_min_pattern) & (yy <= y_max_pattern)
left_line = (xx >= left_x1) & (xx <= left_x2) & pattern_region
right_line = (xx >= right_x1) & (xx <= right_x2) & pattern_region

mask[left_line] = 1.0
mask[right_line] = 1.0

# 为了让中间局部更薄弱一些：
# 在中间高度范围内，把右边线条稍微缩窄一点
middle_weak_region = (yy >= -0.20) & (yy <= 0.20)
right_line_trim = (xx >= 0.35) & (xx <= 0.48) & middle_weak_region
mask[right_line_trim] = 0.0


# =========================================================
# 3. 构造二维高斯 PSF
# =========================================================
sigma = 0.12
psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
psf /= psf.sum()


# =========================================================
# 4. 卷积得到 aerial image
#    注意：aerial 只由 mask 和 PSF 决定
# =========================================================
aerial = fftconvolve(mask, psf, mode='same')


# =========================================================
# 5. 定义一个函数：测量一行里的 line-space-line 几何信息
#    返回：
#    - left_cd
#    - gap_cd
#    - right_cd
#    - num_features
# =========================================================
def measure_lsl_row(row_binary, x):
    """
    输入：
        row_binary : 某一行的 0/1 打印结果
        x         : x 坐标数组
    输出：
        left_cd, gap_cd, right_cd, num_features
    如果结构不完整，就返回 None 或较少信息
    """
    idx = np.where(row_binary > 0)[0]

    # 如果这一行完全没有打印出来
    if len(idx) == 0:
        return None, None, None, 0

    # 用 np.diff 找连续区域的断点
    diffs = np.diff(idx)
    split_points = np.where(diffs > 1)[0]

    # 每个 feature 对应一个连续区间
    starts = [idx[0]]
    ends = []

    for sp in split_points:
        ends.append(idx[sp])
        starts.append(idx[sp + 1])

    ends.append(idx[-1])

    num_features = len(starts)

    # 如果少于两个 feature，通常意味着桥连或严重失真
    if num_features < 2:
        total_cd = x[idx[-1]] - x[idx[0]]
        return total_cd, 0.0, None, num_features

    # 只取前两个 feature 来描述 left / gap / right
    left_start = starts[0]
    left_end = ends[0]
    right_start = starts[1]
    right_end = ends[1]

    left_cd = x[left_end] - x[left_start]
    gap_cd = x[right_start] - x[left_end]
    right_cd = x[right_end] - x[right_start]

    return left_cd, gap_cd, right_cd, num_features


# =========================================================
# 6. 定义逐行分类函数
#    输出每一行是 safe / weak / fail
#    规则：
#    - 如果不在图形有效高度范围内，不参与统计
#    - 如果 num_features < 2，则 fail
#    - 否则根据 left/gap/right 是否低于阈值判断 weak / fail
# =========================================================
def classify_pattern_rows(
    printed,
    x,
    y,
    y_min_pattern,
    y_max_pattern,
    weak_left_cd,
    fail_left_cd,
    weak_gap_cd,
    fail_gap_cd,
    weak_right_cd,
    fail_right_cd
):
    """
    返回：
        row_status : 长度为 ny 的数组
                     np.nan = 不参与统计
                     0 = safe
                     1 = weak
                     2 = fail
        summary    : 一个字典，里面有 safe_ratio / weak_ratio / fail_ratio
    """
    ny = printed.shape[0]#取出 printed 的总行数
    row_status = np.full(ny, np.nan)#创建一个长度为 ny 的一维数组，并且一开始全部填成 np.nan

    for i in range(ny):
        yy_value = y[i]

        # 只在有效图形高度范围内做分类
        if not (y_min_pattern <= yy_value <= y_max_pattern):#如果当前这一行不在有效图形高度范围内，就跳过，不分析这一行。
            continue

        row_binary = printed[i, :]
        left_cd, gap_cd, right_cd, num_features = measure_lsl_row(row_binary, x)

        # 先判 fail
        if num_features < 2:
            row_status[i] = 2
            continue

        if left_cd is None or gap_cd is None or right_cd is None:
            row_status[i] = 2
            continue

        if (left_cd < fail_left_cd) or (gap_cd < fail_gap_cd) or (right_cd < fail_right_cd):
            row_status[i] = 2

        # 再判 weak
        elif (left_cd < weak_left_cd) or (gap_cd < weak_gap_cd) or (right_cd < weak_right_cd):
            row_status[i] = 1

        # 最后 safe
        else:
            row_status[i] = 0

    valid_mask = ~np.isnan(row_status)#如果某个位置是 np.nan，就得到 True.~表示取反
    valid_status = row_status[valid_mask]#从 row_status 里，只取那些 valid_mask = True 的位置

    if len(valid_status) == 0:
        safe_ratio = 0.0
        weak_ratio = 0.0
        fail_ratio = 0.0
    else:
        safe_ratio = np.mean(valid_status == 0)
        weak_ratio = np.mean(valid_status == 1)
        fail_ratio = np.mean(valid_status == 2)

    summary = {
        "safe_ratio": safe_ratio,
        "weak_ratio": weak_ratio,
        "fail_ratio": fail_ratio
    }

    return row_status, summary


# =========================================================
# 7. 定义整体判定函数（今天的新主角）
#    用可调容忍度来决定：
#    0 = fail
#    1 = borderline
#    2 = pass
# =========================================================
def evaluate_global_status_with_tolerance(fail_ratio, weak_ratio, fail_tol, weak_tol):
    """
    规则：
    - fail_ratio > fail_tol   -> fail
    - 否则 weak_ratio > weak_tol -> borderline
    - 否则 pass
    """
    if fail_ratio > fail_tol:
        return 0
    elif weak_ratio > weak_tol:
        return 1
    else:
        return 2


# =========================================================
# 8. 设置逐行分类阈值
# =========================================================
weak_left_cd = 0.55
fail_left_cd = 0.40

weak_gap_cd = 0.55
fail_gap_cd = 0.38

weak_right_cd = 0.55
fail_right_cd = 0.40


# =========================================================
# 9. 扫描 dose 和 threshold
# =========================================================
dose_list = np.linspace(0.85, 1.20, 36)
threshold_list = np.linspace(0.17, 0.42, 36)

safe_ratio_map = np.zeros((len(threshold_list), len(dose_list)))
weak_ratio_map = np.zeros((len(threshold_list), len(dose_list)))
fail_ratio_map = np.zeros((len(threshold_list), len(dose_list)))

# 三套整体规则对应的窗口图
window_map_strict = np.zeros((len(threshold_list), len(dose_list)))
window_map_medium = np.zeros((len(threshold_list), len(dose_list)))
window_map_relaxed = np.zeros((len(threshold_list), len(dose_list)))


# =========================================================
# 10. 双层循环扫描参数
# =========================================================
for i, threshold in enumerate(threshold_list):
    for j, dose in enumerate(dose_list):

        # printed = dose * aerial 再和 threshold 比较
        printed = (dose * aerial >= threshold).astype(float)

        row_status, summary = classify_pattern_rows(
            printed,
            x,
            y,
            y_min_pattern,
            y_max_pattern,
            weak_left_cd,
            fail_left_cd,
            weak_gap_cd,
            fail_gap_cd,
            weak_right_cd,
            fail_right_cd
        )

        safe_ratio = summary["safe_ratio"]
        weak_ratio = summary["weak_ratio"]
        fail_ratio = summary["fail_ratio"]

        safe_ratio_map[i, j] = safe_ratio
        weak_ratio_map[i, j] = weak_ratio
        fail_ratio_map[i, j] = fail_ratio

        # 规则1：严格规则（等价于第28天）
        window_map_strict[i, j] = evaluate_global_status_with_tolerance(
            fail_ratio=fail_ratio,
            weak_ratio=weak_ratio,
            fail_tol=0.00,
            weak_tol=0.00
        )

        # 规则2：中等规则
        window_map_medium[i, j] = evaluate_global_status_with_tolerance(
            fail_ratio=fail_ratio,
            weak_ratio=weak_ratio,
            fail_tol=0.02,
            weak_tol=0.08
        )

        # 规则3：宽松规则
        window_map_relaxed[i, j] = evaluate_global_status_with_tolerance(
            fail_ratio=fail_ratio,
            weak_ratio=weak_ratio,
            fail_tol=0.05,
            weak_tol=0.18
        )


# =========================================================
# 11. 统计三套规则下的 pass 点数量
# =========================================================
pass_count_strict = np.sum(window_map_strict == 2)
pass_count_medium = np.sum(window_map_medium == 2)
pass_count_relaxed = np.sum(window_map_relaxed == 2)


# =========================================================
# 12. 选一个代表性参数点，画局部分类图
# =========================================================
example_dose = 0.72
example_threshold = 0.28

printed_example = (example_dose * aerial >= example_threshold).astype(float)
row_status_example, summary_example = classify_pattern_rows(
    printed_example,
    x,
    y,
    y_min_pattern,
    y_max_pattern,
    weak_left_cd,
    fail_left_cd,
    weak_gap_cd,
    fail_gap_cd,
    weak_right_cd,
    fail_right_cd
)

safe_ratio_ex = summary_example["safe_ratio"]
weak_ratio_ex = summary_example["weak_ratio"]
fail_ratio_ex = summary_example["fail_ratio"]

status_strict_ex = evaluate_global_status_with_tolerance(
    fail_ratio_ex, weak_ratio_ex, fail_tol=0.00, weak_tol=0.00
)
status_medium_ex = evaluate_global_status_with_tolerance(
    fail_ratio_ex, weak_ratio_ex, fail_tol=0.02, weak_tol=0.08
)
status_relaxed_ex = evaluate_global_status_with_tolerance(
    fail_ratio_ex, weak_ratio_ex, fail_tol=0.05, weak_tol=0.18
)

status_name = {0: "fail", 1: "borderline", 2: "pass"}


# =========================================================
# 13. 作图
# =========================================================
fig = plt.figure(figsize=(16, 11))

# (1) mask
ax1 = plt.subplot(2, 3, 1)
plt.imshow(mask, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', aspect='auto', cmap='gray')
plt.title("Mask")
plt.xlabel("x")
plt.ylabel("y")

# (2) example printed
ax2 = plt.subplot(2, 3, 2)
plt.imshow(printed_example, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower', aspect='auto', cmap='gray')
plt.title(f"Printed Example\n(dose={example_dose:.2f}, threshold={example_threshold:.2f})")
plt.xlabel("x")
plt.ylabel("y")

# (3) example row status map
ax3 = plt.subplot(2, 3, 3)
status_plot = np.full((len(y), 1), np.nan)
for i in range(len(y)):
    status_plot[i, 0] = row_status_example[i]

plt.imshow(status_plot, extent=[0, 1, y.min(), y.max()], origin='lower', aspect='auto', cmap='coolwarm', vmin=0, vmax=2)
plt.title("Local Row Status Example\n(0=safe, 1=weak, 2=fail)")
plt.xlabel("dummy x")
plt.ylabel("y")
plt.colorbar()
# (4) strict window map
ax4 = plt.subplot(2, 3, 4)
plt.imshow(
    window_map_strict,
    extent=[dose_list.min(), dose_list.max(), threshold_list.min(), threshold_list.max()],
    origin='lower',
    aspect='auto',
    cmap='viridis',
    vmin=0,
    vmax=2
)
plt.title(f"Strict Window Map\npass count = {pass_count_strict}")
plt.xlabel("dose")
plt.ylabel("threshold")
plt.colorbar()

# (5) medium window map
ax5 = plt.subplot(2, 3, 5)
plt.imshow(
    window_map_medium,
    extent=[dose_list.min(), dose_list.max(), threshold_list.min(), threshold_list.max()],
    origin='lower',
    aspect='auto',
    cmap='viridis',
    vmin=0,
    vmax=2
)
plt.title(f"Medium-Tolerance Window Map\npass count = {pass_count_medium}")
plt.xlabel("dose")
plt.ylabel("threshold")
plt.colorbar()

# (6) relaxed window map
ax6 = plt.subplot(2, 3, 6)
plt.imshow(
    window_map_relaxed,
    extent=[dose_list.min(), dose_list.max(), threshold_list.min(), threshold_list.max()],
    origin='lower',
    aspect='auto',
    cmap='viridis',
    vmin=0,
    vmax=2
)
plt.title(f"Relaxed-Tolerance Window Map\npass count = {pass_count_relaxed}")
plt.xlabel("dose")
plt.ylabel("threshold")
plt.colorbar()


plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.show()