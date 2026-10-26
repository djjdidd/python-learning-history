#约束和目标，到底分别在干什么
#window_map 用于定义哪些点可行，objective_map 用于区分这些可行点中的优劣
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

# 人为制造局部薄弱区
middle_weak_region = (yy >= -0.20) & (yy <= 0.20)
right_line_trim = (xx >= 0.35) & (xx <= 0.48) & middle_weak_region
mask[right_line_trim] = 0.0


# =========================================================
# 3. PSF 和 aerial
# =========================================================
sigma = 0.12
psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
psf /= psf.sum()

aerial = fftconvolve(mask, psf, mode='same')


# =========================================================
# 4. 单行测量函数
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
# 5. 逐行分析函数
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

    return row_status, gap_cd_rows, summary


# =========================================================
# 6. 多指标整体判定函数
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
# 7. 目标函数
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
    if np.isnan(min_gap_cd):
        gap_shortage = 1.0
    else:
        gap_shortage = max(0.0, target_gap - min_gap_cd)

    score = (
        1.0 * safe_ratio
        - weak_weight * weak_ratio
        - fail_weight * fail_ratio
        - gap_penalty_weight * gap_shortage
    )
    return score


# =========================================================
# 8. 总评估函数
# =========================================================
def evaluate_condition(
    dose,
    threshold,
    aerial,
    x,
    y,
    y_min_pattern,
    y_max_pattern,
    weak_left_cd,
    fail_left_cd,
    weak_gap_cd,
    fail_gap_cd,
    weak_right_cd,
    fail_right_cd,
    fail_tol,
    weak_tol,
    fail_left_limit,
    weak_left_limit,
    fail_gap_limit,
    weak_gap_limit,
    fail_right_limit,
    weak_right_limit
):
    printed = (dose * aerial >= threshold).astype(float)

    row_status, gap_cd_rows, summary = analyze_pattern_rows(
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

    global_status = evaluate_global_status_multi_metric(
        fail_ratio=summary["fail_ratio"],
        weak_ratio=summary["weak_ratio"],
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

    objective = compute_objective(
        safe_ratio=summary["safe_ratio"],
        weak_ratio=summary["weak_ratio"],
        fail_ratio=summary["fail_ratio"],
        min_gap_cd=summary["min_gap_cd"],
        target_gap=0.55,
        weak_weight=0.5,
        fail_weight=2.0,
        gap_penalty_weight=1.5
    )

    return printed, row_status, gap_cd_rows, summary, global_status, objective


# =========================================================
# 9. 设置阈值
# =========================================================
weak_left_cd = 0.55
fail_left_cd = 0.40

weak_gap_cd = 0.55
fail_gap_cd = 0.38

weak_right_cd = 0.55
fail_right_cd = 0.40

fail_tol = 0.02
weak_tol = 0.08

fail_left_limit = 0.40
weak_left_limit = 0.50

fail_gap_limit = 0.38
weak_gap_limit = 0.50

fail_right_limit = 0.40
weak_right_limit = 0.50


# =========================================================
# 10. 二维参数扫描
# =========================================================
dose_list = np.linspace(0.85, 1.20, 36)
threshold_list = np.linspace(0.17, 0.42, 36)

window_map = np.zeros((len(threshold_list), len(dose_list)))
objective_map = np.zeros((len(threshold_list), len(dose_list)))
safe_ratio_map = np.zeros((len(threshold_list), len(dose_list)))

for i, threshold in enumerate(threshold_list):
    for j, dose in enumerate(dose_list):
        printed, row_status, gap_cd_rows, summary, global_status, objective = evaluate_condition(
            dose=dose,
            threshold=threshold,
            aerial=aerial,
            x=x,
            y=y,
            y_min_pattern=y_min_pattern,
            y_max_pattern=y_max_pattern,
            weak_left_cd=weak_left_cd,
            fail_left_cd=fail_left_cd,
            weak_gap_cd=weak_gap_cd,
            fail_gap_cd=fail_gap_cd,
            weak_right_cd=weak_right_cd,
            fail_right_cd=fail_right_cd,
            fail_tol=fail_tol,
            weak_tol=weak_tol,
            fail_left_limit=fail_left_limit,
            weak_left_limit=weak_left_limit,
            fail_gap_limit=fail_gap_limit,
            weak_gap_limit=weak_gap_limit,
            fail_right_limit=fail_right_limit,
            weak_right_limit=weak_right_limit
        )

        window_map[i, j] = global_status
        objective_map[i, j] = objective
        safe_ratio_map[i, j] = summary["safe_ratio"]


# =========================================================
# 11. 全图 objective 最大点
# =========================================================
global_best_flat_idx = np.argmax(objective_map)
global_best_i, global_best_j = np.unravel_index(global_best_flat_idx, objective_map.shape)

global_best_dose = dose_list[global_best_j]
global_best_threshold = threshold_list[global_best_i]

global_best_printed, _, _, global_best_summary, global_best_status, global_best_objective = evaluate_condition(
    dose=global_best_dose,
    threshold=global_best_threshold,
    aerial=aerial,
    x=x,
    y=y,
    y_min_pattern=y_min_pattern,
    y_max_pattern=y_max_pattern,
    weak_left_cd=weak_left_cd,
    fail_left_cd=fail_left_cd,
    weak_gap_cd=weak_gap_cd,
    fail_gap_cd=fail_gap_cd,
    weak_right_cd=weak_right_cd,
    fail_right_cd=fail_right_cd,
    fail_tol=fail_tol,
    weak_tol=weak_tol,
    fail_left_limit=fail_left_limit,
    weak_left_limit=weak_left_limit,
    fail_gap_limit=fail_gap_limit,
    weak_gap_limit=weak_gap_limit,
    fail_right_limit=fail_right_limit,
    weak_right_limit=weak_right_limit
)


# =========================================================
# 12. pass 区 objective 最大点
# =========================================================
pass_mask = (window_map == 2)
has_pass = np.any(pass_mask)

if has_pass:
    objective_in_pass = np.where(pass_mask, objective_map, -999)
    pass_best_flat_idx = np.argmax(objective_in_pass)
    pass_best_i, pass_best_j = np.unravel_index(pass_best_flat_idx, objective_map.shape)

    pass_best_dose = dose_list[pass_best_j]
    pass_best_threshold = threshold_list[pass_best_i]

    pass_best_printed, _, _, pass_best_summary, pass_best_status, pass_best_objective = evaluate_condition(
        dose=pass_best_dose,
        threshold=pass_best_threshold,
        aerial=aerial,
        x=x,
        y=y,
        y_min_pattern=y_min_pattern,
        y_max_pattern=y_max_pattern,
        weak_left_cd=weak_left_cd,
        fail_left_cd=fail_left_cd,
        weak_gap_cd=weak_gap_cd,
        fail_gap_cd=fail_gap_cd,
        weak_right_cd=weak_right_cd,
        fail_right_cd=fail_right_cd,
        fail_tol=fail_tol,
        weak_tol=weak_tol,
        fail_left_limit=fail_left_limit,
        weak_left_limit=weak_left_limit,
        fail_gap_limit=fail_gap_limit,
        weak_gap_limit=weak_gap_limit,
        fail_right_limit=fail_right_limit,
        weak_right_limit=weak_right_limit
    )

status_name = {0: "fail", 1: "borderline", 2: "pass"}


# =========================================================
# 13. 作图
# =========================================================
plt.figure(figsize=(16, 10))

# (1) objective map
plt.subplot(2, 3, 1)
plt.imshow(
    objective_map,
    extent=[dose_list.min(), dose_list.max(), threshold_list.min(), threshold_list.max()],
    origin='lower',
    aspect='auto',
    cmap='inferno'
)
plt.colorbar()
plt.title("Objective Map")
plt.xlabel("dose")
plt.ylabel("threshold")

plt.plot(global_best_dose, global_best_threshold, 'go', label='global best objective')
if has_pass:
    plt.plot(pass_best_dose, pass_best_threshold, 'gx', markersize=10, label='best in pass')
plt.legend()

# (2) window map
plt.subplot(2, 3, 2)
plt.imshow(
    window_map,
    extent=[dose_list.min(), dose_list.max(), threshold_list.min(), threshold_list.max()],
    origin='lower',
    aspect='auto',
    cmap='coolwarm',
    vmin=0,
    vmax=2
)
plt.colorbar()
plt.title("Window Map")
plt.xlabel("dose")
plt.ylabel("threshold")

plt.plot(global_best_dose, global_best_threshold, 'go', label='global best objective')
if has_pass:
    plt.plot(pass_best_dose, pass_best_threshold, 'gx', markersize=10, label='best in pass')
plt.legend()

# (3) safe ratio map
plt.subplot(2, 3, 3)
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

# (4) global best printed
plt.subplot(2, 3, 4)
plt.imshow(
    global_best_printed,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.title("Global Best Objective Point")
plt.xlabel("x")
plt.ylabel("y")

# (5) pass best printed
plt.subplot(2, 3, 5)
if has_pass:
    plt.imshow(
        pass_best_printed,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin='lower',
        aspect='auto',
        cmap='gray'
    )
    plt.title("Best Objective Point in Pass Region")
else:
    plt.text(0.1, 0.5, "No pass point")
plt.xlabel("x")
plt.ylabel("y")

# (6) summary
plt.subplot(2, 3, 6)
plt.axis('off')

summary_text = (
    "[Global best in whole map]\n"
    f"dose = {global_best_dose:.3f}\n"
    f"threshold = {global_best_threshold:.3f}\n"
    f"status = {status_name[global_best_status]}\n"
    f"objective = {global_best_objective:.3f}\n"
    f"safe_ratio = {global_best_summary['safe_ratio']:.3f}\n"
    f"fail_ratio = {global_best_summary['fail_ratio']:.3f}\n"
    f"min_gap_cd = {global_best_summary['min_gap_cd']:.3f}\n\n"
)

if has_pass:
    summary_text += (
        "[Best objective in pass region]\n"
        f"dose = {pass_best_dose:.3f}\n"
        f"threshold = {pass_best_threshold:.3f}\n"
        f"status = {status_name[pass_best_status]}\n"
        f"objective = {pass_best_objective:.3f}\n"
        f"safe_ratio = {pass_best_summary['safe_ratio']:.3f}\n"
        f"fail_ratio = {pass_best_summary['fail_ratio']:.3f}\n"
        f"min_gap_cd = {pass_best_summary['min_gap_cd']:.3f}"
    )

plt.text(0.02, 0.98, summary_text, va='top', fontsize=10)
plt.title("Constraint vs Objective Summary")

plt.tight_layout()
plt.show()