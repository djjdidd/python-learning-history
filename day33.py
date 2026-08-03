#目标函数设计
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve


# =========================================================
# 1. 建立二维网格
# =========================================================
nx, ny = 401, 241
x = np.linspace(-2.0, 2.0, nx)
y = np.linspace(-1.2, 1.2, ny)
xx, yy = np.meshgrid(x, y)


# =========================================================
# 2. 构造二维 line-space-line mask
# =========================================================
mask = np.zeros((ny, nx), dtype=float)

y_min_pattern = -0.75
y_max_pattern = 0.75

left_x1, left_x2 = -1.10, -0.35
right_x1, right_x2 = 0.35, 1.10

pattern_region = (yy >= y_min_pattern) & (yy <= y_max_pattern)
left_line = (xx >= left_x1) & (xx <= left_x2) & pattern_region
right_line = (xx >= right_x1) & (xx <= right_x2) & pattern_region

mask[left_line] = 1.0
mask[right_line] = 1.0

# 中间局部薄弱区
middle_weak_region = (yy >= -0.20) & (yy <= 0.20)
right_line_trim = (xx >= 0.35) & (xx <= 0.48) & middle_weak_region
mask[right_line_trim] = 0.0


# =========================================================
# 3. 高斯 PSF
# =========================================================
sigma = 0.12
psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
psf /= psf.sum()


# =========================================================
# 4. aerial image
# =========================================================
aerial = fftconvolve(mask, psf, mode='same')


# =========================================================
# 5. 单行测量函数
# =========================================================
def measure_lsl_row(row_binary, x):
    idx = np.where(row_binary > 0)[0]

    if len(idx) == 0:
        return None, None, None, 0

    diffs = np.diff(idx)
    split_points = np.where(diffs > 1)[0]

    starts = [idx[0]]
    ends = []

    for sp in split_points:
        ends.append(idx[sp])
        starts.append(idx[sp + 1])

    ends.append(idx[-1])

    num_features = len(starts)

    if num_features < 2:
        total_cd = x[idx[-1]] - x[idx[0]]
        return total_cd, 0.0, None, num_features

    left_start = starts[0]
    left_end = ends[0]
    right_start = starts[1]
    right_end = ends[1]

    left_cd = x[left_end] - x[left_start]
    gap_cd = x[right_start] - x[left_end]
    right_cd = x[right_end] - x[right_start]

    return left_cd, gap_cd, right_cd, num_features


# =========================================================
# 6. 逐行分析函数
# =========================================================
def analyze_pattern_rows(
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
    ny = printed.shape[0]

    row_status = np.full(ny, np.nan)
    left_cd_rows = np.full(ny, np.nan)
    gap_cd_rows = np.full(ny, np.nan)
    right_cd_rows = np.full(ny, np.nan)

    for i in range(ny):
        yy_value = y[i]

        if not (y_min_pattern <= yy_value <= y_max_pattern):
            continue

        row_binary = printed[i, :]
        left_cd, gap_cd, right_cd, num_features = measure_lsl_row(row_binary, x)

        if left_cd is not None:
            left_cd_rows[i] = left_cd
        if gap_cd is not None:
            gap_cd_rows[i] = gap_cd
        if right_cd is not None:
            right_cd_rows[i] = right_cd

        if num_features < 2:
            row_status[i] = 2
            continue

        if left_cd is None or gap_cd is None or right_cd is None:
            row_status[i] = 2
            continue

        if (left_cd < fail_left_cd) or (gap_cd < fail_gap_cd) or (right_cd < fail_right_cd):
            row_status[i] = 2
        elif (left_cd < weak_left_cd) or (gap_cd < weak_gap_cd) or (right_cd < weak_right_cd):
            row_status[i] = 1
        else:
            row_status[i] = 0

    valid_status = row_status[~np.isnan(row_status)]

    if len(valid_status) == 0:
        safe_ratio = 0.0
        weak_ratio = 0.0
        fail_ratio = 0.0
    else:
        safe_ratio = np.mean(valid_status == 0)
        weak_ratio = np.mean(valid_status == 1)
        fail_ratio = np.mean(valid_status == 2)

    valid_left = left_cd_rows[~np.isnan(left_cd_rows)]
    valid_gap = gap_cd_rows[~np.isnan(gap_cd_rows)]
    valid_right = right_cd_rows[~np.isnan(right_cd_rows)]

    min_left_cd = np.min(valid_left) if len(valid_left) > 0 else np.nan
    min_gap_cd = np.min(valid_gap) if len(valid_gap) > 0 else np.nan
    min_right_cd = np.min(valid_right) if len(valid_right) > 0 else np.nan

    summary = {
        "safe_ratio": safe_ratio,
        "weak_ratio": weak_ratio,
        "fail_ratio": fail_ratio,
        "min_left_cd": min_left_cd,
        "min_gap_cd": min_gap_cd,
        "min_right_cd": min_right_cd
    }

    return row_status, left_cd_rows, gap_cd_rows, right_cd_rows, summary


# =========================================================
# 7. 多指标整体判定函数（沿用第32天）
# =========================================================
def evaluate_global_status_multi_metric(
    fail_ratio,
    weak_ratio,
    min_left_cd,
    min_gap_cd,
    min_right_cd,
    fail_tol,
    weak_tol,
    fail_left_limit,
    weak_left_limit,
    fail_gap_limit,
    weak_gap_limit,
    fail_right_limit,
    weak_right_limit
):
    if fail_ratio > fail_tol:
        return 0

    if (not np.isnan(min_left_cd)) and (min_left_cd < fail_left_limit):
        return 0

    if (not np.isnan(min_gap_cd)) and (min_gap_cd < fail_gap_limit):
        return 0

    if (not np.isnan(min_right_cd)) and (min_right_cd < fail_right_limit):
        return 0

    if weak_ratio > weak_tol:
        return 1

    if (not np.isnan(min_left_cd)) and (min_left_cd < weak_left_limit):
        return 1

    if (not np.isnan(min_gap_cd)) and (min_gap_cd < weak_gap_limit):
        return 1

    if (not np.isnan(min_right_cd)) and (min_right_cd < weak_right_limit):
        return 1

    return 2


# =========================================================
# 8. 今天的新重点：目标函数
# =========================================================
def compute_objective(
    safe_ratio,
    weak_ratio,
    fail_ratio,
    min_gap_cd,
    target_gap=0.55,
    weak_weight=0.5,
    fail_weight=2.0,
    gap_penalty_weight=1.5
):
    """
    目标函数思路：
    score = reward - penalty

    reward:
        safe_ratio 越大越好

    penalty:
        weak_ratio 越大越差
        fail_ratio 越大越差，而且更严重
        min_gap_cd 如果低于 target_gap，要额外扣分
    """

    # gap 不足时的缺口
    if np.isnan(min_gap_cd):
        gap_shortage = 1.0   # 没有有效 gap，直接给较大惩罚
    else:
        gap_shortage = max(0.0, target_gap - min_gap_cd)
#如果当前最差 gap 比目标 gap 小，就计算它差了多少；如果已经不小于目标 gap，就缺口记为 0。
    score = (
        1.0 * safe_ratio
        - weak_weight * weak_ratio
        - fail_weight * fail_ratio
        - gap_penalty_weight * gap_shortage
    )

    return score


# =========================================================
# 9. 设置逐行分类阈值
# =========================================================
weak_left_cd = 0.55
fail_left_cd = 0.40

weak_gap_cd = 0.55
fail_gap_cd = 0.38

weak_right_cd = 0.55
fail_right_cd = 0.40


# =========================================================
# 10. 设置整体规则阈值
# =========================================================
fail_tol = 0.02
weak_tol = 0.08

fail_left_limit = 0.40
weak_left_limit = 0.50

fail_gap_limit = 0.38
weak_gap_limit = 0.50

fail_right_limit = 0.40
weak_right_limit = 0.50


# =========================================================
# 11. 扫描 dose 和 threshold
# =========================================================
dose_list = np.linspace(0.25, 1.20, 36)
threshold_list = np.linspace(0.07, 0.42, 36)

safe_ratio_map = np.zeros((len(threshold_list), len(dose_list)))
weak_ratio_map = np.zeros((len(threshold_list), len(dose_list)))
fail_ratio_map = np.zeros((len(threshold_list), len(dose_list)))

min_gap_cd_map = np.full((len(threshold_list), len(dose_list)), np.nan)

window_map_multi = np.zeros((len(threshold_list), len(dose_list)))
objective_map = np.zeros((len(threshold_list), len(dose_list)))


# =========================================================
# 12. 双层循环
# =========================================================
for i, threshold in enumerate(threshold_list):
    for j, dose in enumerate(dose_list):

        printed = (dose * aerial >= threshold).astype(float)

        row_status, left_cd_rows, gap_cd_rows, right_cd_rows, summary = analyze_pattern_rows(
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
        min_gap_cd = summary["min_gap_cd"]

        safe_ratio_map[i, j] = safe_ratio
        weak_ratio_map[i, j] = weak_ratio
        fail_ratio_map[i, j] = fail_ratio
        min_gap_cd_map[i, j] = min_gap_cd

        window_map_multi[i, j] = evaluate_global_status_multi_metric(
            fail_ratio=fail_ratio,
            weak_ratio=weak_ratio,
            min_left_cd=summary["min_left_cd"],
            min_gap_cd=summary["min_gap_cd"],
            min_right_cd=summary["min_right_cd"],
            fail_tol=fail_tol,
            weak_tol=weak_tol,
            fail_left_limit=fail_left_limit,
            weak_left_limit=weak_left_limit,
            fail_gap_limit=fail_gap_limit,
            weak_gap_limit=weak_gap_limit,
            fail_right_limit=fail_right_limit,
            weak_right_limit=weak_right_limit
        )

        objective_map[i, j] = compute_objective(
            safe_ratio=safe_ratio,
            weak_ratio=weak_ratio,
            fail_ratio=fail_ratio,
            min_gap_cd=min_gap_cd,
            target_gap=0.55,
            weak_weight=0.5,
            fail_weight=2.0,
            gap_penalty_weight=1.5
        )


# =========================================================
# 13. 只在 pass 区中找目标函数最优点
# =========================================================
pass_mask = (window_map_multi == 2)
has_pass = np.any(pass_mask)

if has_pass:
    objective_in_pass = np.where(pass_mask, objective_map, -999)
    best_flat_idx = np.argmax(objective_in_pass)
    best_i, best_j = np.unravel_index(best_flat_idx, objective_map.shape)

    best_dose = dose_list[best_j]
    best_threshold = threshold_list[best_i]

    best_safe = safe_ratio_map[best_i, best_j]
    best_weak = weak_ratio_map[best_i, best_j]
    best_fail = fail_ratio_map[best_i, best_j]
    best_gap = min_gap_cd_map[best_i, best_j]
    best_objective = objective_map[best_i, best_j]
else:
    best_i, best_j = None, None


# =========================================================
# 14. 选例子点
# =========================================================
example_dose = 1.02
example_threshold = 0.28

printed_example = (example_dose * aerial >= example_threshold).astype(float)

row_status_ex, left_cd_rows_ex, gap_cd_rows_ex, right_cd_rows_ex, summary_ex = analyze_pattern_rows(
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

objective_ex = compute_objective(
    safe_ratio=summary_ex["safe_ratio"],
    weak_ratio=summary_ex["weak_ratio"],
    fail_ratio=summary_ex["fail_ratio"],
    min_gap_cd=summary_ex["min_gap_cd"],
    target_gap=0.55,
    weak_weight=0.5,
    fail_weight=2.0,
    gap_penalty_weight=1.5
)


# =========================================================
# 15. 作图
# =========================================================
plt.figure(figsize=(16, 11))

# (1) safe ratio map
plt.subplot(2, 3, 1)
plt.imshow(
    safe_ratio_map,
    extent=[dose_list.min(), dose_list.max(), threshold_list.min(), threshold_list.max()],
    origin='lower',
    aspect='auto',
    cmap='viridis'
)
plt.colorbar()
plt.title("Safe Ratio Map")
plt.xlabel("dose")
plt.ylabel("threshold")

# (2) objective map
plt.subplot(2, 3, 2)
plt.imshow(
    objective_map,
    extent=[dose_list.min(), dose_list.max(), threshold_list.min(), threshold_list.max()],
    origin='lower',
    aspect='auto',
    cmap='plasma'
)
plt.colorbar()
plt.title("Objective Map")
plt.xlabel("dose")
plt.ylabel("threshold")

if has_pass:
    plt.plot(best_dose, best_threshold, 'bx', markersize=10)

# (3) multi-metric window map
plt.subplot(2, 3, 3)
plt.imshow(
    window_map_multi,
    extent=[dose_list.min(), dose_list.max(), threshold_list.min(), threshold_list.max()],
    origin='lower',
    aspect='auto',
    cmap='coolwarm',
    vmin=0,
    vmax=2
)
plt.colorbar()
plt.title("Multi-Metric Window Map")
plt.xlabel("dose")
plt.ylabel("threshold")

if has_pass:
    plt.plot(best_dose, best_threshold, 'ko', markersize=8)

# (4) printed example
plt.subplot(2, 3, 4)
plt.imshow(
    printed_example,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.title(f"Printed Example\n(dose={example_dose:.2f}, threshold={example_threshold:.2f})")
plt.xlabel("x")
plt.ylabel("y")

# (5) row-wise gap curve
plt.subplot(2, 3, 5)
plt.plot(y, gap_cd_rows_ex)
plt.axhline(0.55, linestyle='--')
plt.title("Row-wise Gap CD")
plt.xlabel("y")
plt.ylabel("gap_cd")

# (6) summary text
plt.subplot(2, 3, 6)
plt.axis('off')

if has_pass:
    summary_text = (
        f"[Best point in pass region]\n"
        f"dose = {best_dose:.3f}\n"
        f"threshold = {best_threshold:.3f}\n"
        f"safe_ratio = {best_safe:.3f}\n"
        f"weak_ratio = {best_weak:.3f}\n"
        f"fail_ratio = {best_fail:.3f}\n"
        f"min_gap_cd = {best_gap:.3f}\n"
        f"objective = {best_objective:.3f}\n\n"
        f"[Example point]\n"
        f"safe_ratio = {summary_ex['safe_ratio']:.3f}\n"
        f"weak_ratio = {summary_ex['weak_ratio']:.3f}\n"
        f"fail_ratio = {summary_ex['fail_ratio']:.3f}\n"
        f"min_gap_cd = {summary_ex['min_gap_cd']:.3f}\n"
        f"objective = {objective_ex:.3f}"
    )
else:
    summary_text = "No pass point found."

plt.text(0.02, 0.98, summary_text, va='top', fontsize=11)
plt.title("Summary")

plt.tight_layout()
plt.show()