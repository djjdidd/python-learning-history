#瓶颈占比图——频次和占比到底有什么区别
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

# 局部薄弱区
middle_weak_region = (yy >= -0.20) & (yy <= 0.20)
right_line_trim = (xx >= 0.35) & (xx <= 0.48) & middle_weak_region
mask[right_line_trim] = 0.0


# =========================================================
# 3. PSF 和 aerial image
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
    topology_bad_rows = np.zeros(ny, dtype=int)

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
            topology_bad_rows[i] = 1
            continue

        if left_cd is None or gap_cd is None or right_cd is None:
            row_status[i] = 2
            topology_bad_rows[i] = 1
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

    topology_bad_ratio = np.mean(topology_bad_rows)

    summary = {
        "safe_ratio": safe_ratio,
        "weak_ratio": weak_ratio,
        "fail_ratio": fail_ratio,
        "min_left_cd": min_left_cd,
        "min_gap_cd": min_gap_cd,
        "min_right_cd": min_right_cd,
        "topology_bad_ratio": topology_bad_ratio
    }

    return row_status, left_cd_rows, gap_cd_rows, right_cd_rows, topology_bad_rows, summary


# =========================================================
# 6. 多原因诊断函数
# =========================================================
def diagnose_failure_reason_multi(
    summary,
    fail_tol,
    weak_tol,
    fail_left_limit,
    weak_left_limit,
    fail_gap_limit,
    weak_gap_limit,
    fail_right_limit,
    weak_right_limit
):
    reason_list = []

    fail_ratio = summary["fail_ratio"]
    weak_ratio = summary["weak_ratio"]
    min_left_cd = summary["min_left_cd"]
    min_gap_cd = summary["min_gap_cd"]
    min_right_cd = summary["min_right_cd"]
    topology_bad_ratio = summary["topology_bad_ratio"]

    # fail 级别
    if topology_bad_ratio > 0:
        reason_list.append("topology problem")

    if fail_ratio > fail_tol:
        reason_list.append("fail_ratio too high")

    if (not np.isnan(min_left_cd)) and (min_left_cd < fail_left_limit):
        reason_list.append("min_left_cd too small")

    if (not np.isnan(min_gap_cd)) and (min_gap_cd < fail_gap_limit):
        reason_list.append("min_gap_cd too small")

    if (not np.isnan(min_right_cd)) and (min_right_cd < fail_right_limit):
        reason_list.append("min_right_cd too small")

    if len(reason_list) > 0:
        return 0, reason_list

    # borderline 级别
    if weak_ratio > weak_tol:
        reason_list.append("weak_ratio too high")

    if (not np.isnan(min_left_cd)) and (min_left_cd < weak_left_limit):
        reason_list.append("min_left_cd in weak zone")

    if (not np.isnan(min_gap_cd)) and (min_gap_cd < weak_gap_limit):
        reason_list.append("min_gap_cd in weak zone")

    if (not np.isnan(min_right_cd)) and (min_right_cd < weak_right_limit):
        reason_list.append("min_right_cd in weak zone")

    if len(reason_list) > 0:
        return 1, reason_list

    return 2, ["pass"]


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
# 8. 统一评估函数
# =========================================================
def evaluate_condition_and_diagnose(
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

    (
        row_status,
        left_cd_rows,
        gap_cd_rows,
        right_cd_rows,
        topology_bad_rows,
        summary
    ) = analyze_pattern_rows(
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

    global_status, reason_list = diagnose_failure_reason_multi(
        summary,
        fail_tol,
        weak_tol,
        fail_left_limit,
        weak_left_limit,
        fail_gap_limit,
        weak_gap_limit,
        fail_right_limit,
        weak_right_limit
    )

    objective = compute_objective(
        safe_ratio=summary["safe_ratio"],
        weak_ratio=summary["weak_ratio"],
        fail_ratio=summary["fail_ratio"],
        min_gap_cd=summary["min_gap_cd"]
    )

    return (
        printed,
        row_status,
        left_cd_rows,
        gap_cd_rows,
        right_cd_rows,
        topology_bad_rows,
        summary,
        global_status,
        reason_list,
        objective
    )


# =========================================================
# 9. 统计频次函数
# =========================================================
def count_reason_frequency(reason_lists):
    reason_count = {}

    for reasons in reason_lists:
        for r in reasons:
            if r not in reason_count:
                reason_count[r] = 1
            else:
                reason_count[r] += 1

    return reason_count


# =========================================================
# 10. 今天的新函数：把频次变成占比
# =========================================================
def compute_reason_ratio(reason_count, total_points):
    """
    输入：
        reason_count: 某一类点（例如 fail 点）里，各原因出现次数的字典
        total_points: 这一类点总共有多少个

    输出：
        reason_ratio: 各原因占比（百分比）的字典
    """
    reason_ratio = {}

    if total_points == 0:
        return reason_ratio

    for reason in reason_count:
        reason_ratio[reason] = 100.0 * reason_count[reason] / total_points

    return reason_ratio


# =========================================================
# 11. 设置阈值
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
# 12. 参数扫描并收集统计信息
# =========================================================
dose_list = np.linspace(0.85, 1.20, 36)
threshold_list = np.linspace(0.17, 0.42, 36)

window_map = np.zeros((len(threshold_list), len(dose_list)))
objective_map = np.zeros((len(threshold_list), len(dose_list)))

fail_reason_lists = []
borderline_reason_lists = []

for i, threshold in enumerate(threshold_list):
    for j, dose in enumerate(dose_list):
        (
            printed,
            row_status,
            left_cd_rows,
            gap_cd_rows,
            right_cd_rows,
            topology_bad_rows,
            summary,
            global_status,
            reason_list,
            objective
        ) = evaluate_condition_and_diagnose(
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

        if global_status == 0:
            fail_reason_lists.append(reason_list)
        elif global_status == 1:
            borderline_reason_lists.append(reason_list)


# =========================================================
# 13. 统计 fail / borderline 的频次和占比
# =========================================================
fail_reason_count = count_reason_frequency(fail_reason_lists)
border_reason_count = count_reason_frequency(borderline_reason_lists)

num_fail = np.sum(window_map == 0)
num_borderline = np.sum(window_map == 1)
num_pass = np.sum(window_map == 2)

fail_reason_ratio = compute_reason_ratio(fail_reason_count, num_fail)
border_reason_ratio = compute_reason_ratio(border_reason_count, num_borderline)


# =========================================================
# 14. 整理柱状图数据
# =========================================================
fail_labels = list(fail_reason_count.keys())
fail_values = list(fail_reason_count.values())

fail_ratio_labels = list(fail_reason_ratio.keys())
fail_ratio_values = list(fail_reason_ratio.values())

border_labels = list(border_reason_count.keys())
border_values = list(border_reason_count.values())

border_ratio_labels = list(border_reason_ratio.keys())
border_ratio_values = list(border_reason_ratio.values())


# =========================================================
# 15. 选一个例子点
# =========================================================
example_dose = 0.98
example_threshold = 0.28

(
    printed_ex,
    row_status_ex,
    left_cd_rows_ex,
    gap_cd_rows_ex,
    right_cd_rows_ex,
    topology_bad_rows_ex,
    summary_ex,
    global_status_ex,
    reason_list_ex,
    objective_ex
) = evaluate_condition_and_diagnose(
    dose=example_dose,
    threshold=example_threshold,
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
# 16. 作图
# =========================================================
plt.figure(figsize=(16, 11))

# (1) window map
plt.subplot(2, 3, 1)
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
plt.plot(example_dose, example_threshold, 'ko', markersize=8)

# (2) objective map
plt.subplot(2, 3, 2)
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
plt.plot(example_dose, example_threshold, 'wo', markersize=8)

# (3) fail 频次图
plt.subplot(2, 3, 3)
if len(fail_labels) > 0:
    plt.bar(range(len(fail_labels)), fail_values)
    plt.xticks(range(len(fail_labels)), fail_labels, rotation=15, ha='right')
plt.title("Fail Reason Frequency")
plt.xlabel("reason")
plt.ylabel("count")

# (4) fail 占比图
plt.subplot(2, 3, 4)
if len(fail_ratio_labels) > 0:
    plt.bar(range(len(fail_ratio_labels)), fail_ratio_values)
    plt.xticks(range(len(fail_ratio_labels)), fail_ratio_labels, rotation=35, ha='right')
plt.title("Fail Reason Ratio (%)")
plt.xlabel("reason")
plt.ylabel("percent")

# (5) borderline 频次图
plt.subplot(2, 3, 5)
if len(border_labels) > 0:
    plt.bar(range(len(border_labels)), border_values)
    plt.xticks(range(len(border_labels)), border_labels, rotation=35, ha='right')
plt.title("Borderline Reason Frequency")
plt.xlabel("reason")
plt.ylabel("count")


plt.subplot(2, 3, 6)
if len(border_ratio_labels) > 0:
    plt.bar(range(len(border_ratio_labels)), border_ratio_values)
    plt.xticks(range(len(border_ratio_labels)), border_ratio_labels, rotation=35, ha='right')

plt.title("Borderline Reason Ratio (%)")
plt.xlabel("reason")
plt.ylabel("percent")
plt.tight_layout()
plt.show()