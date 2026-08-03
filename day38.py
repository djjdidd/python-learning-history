# 第38天：多指标整体判定再升级——到底是谁让它 fail
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

# 人为制造局部薄弱区：让右侧线在中间局部更容易出问题
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
    """
    输入一行二值图形 row_binary
    返回：
    left_cd, gap_cd, right_cd, num_features
    """
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

    # 如果不足两个 feature，就说明 topology 出问题
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

        # 只分析图形有效高度范围
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

        # topology 出问题：不是两条独立线
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

    topology_bad_ratio = np.mean(topology_bad_rows)  # 有多少行 topology 出问题

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
# 6. 整体状态判定函数
#    0 = fail
#    1 = borderline
#    2 = pass
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
# 7. 新函数：诊断到底是谁导致 fail / borderline
# =========================================================
def diagnose_failure_reason(
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
    """
    返回：
    global_status
    primary_reason_code
    primary_reason_text

    reason_code 约定：
    0  -> pass
    1  -> fail_ratio too high
    2  -> min_left_cd too small
    3  -> min_gap_cd too small
    4  -> min_right_cd too small
    5  -> weak_ratio too high
    6  -> left_cd in weak zone
    7  -> gap_cd in weak zone
    8  -> right_cd in weak zone
    9  -> topology problem
    """

    fail_ratio = summary["fail_ratio"]
    weak_ratio = summary["weak_ratio"]
    min_left_cd = summary["min_left_cd"]
    min_gap_cd = summary["min_gap_cd"]
    min_right_cd = summary["min_right_cd"]
    topology_bad_ratio = summary["topology_bad_ratio"]

    # 先看 topology，因为这是结构层面的严重问题
    if topology_bad_ratio > 0:
        return 0, 9, "fail: topology problem"

    # fail 级别诊断
    if fail_ratio > fail_tol:
        return 0, 1, "fail: fail_ratio too high"

    if (not np.isnan(min_left_cd)) and (min_left_cd < fail_left_limit):
        return 0, 2, "fail: min_left_cd too small"

    if (not np.isnan(min_gap_cd)) and (min_gap_cd < fail_gap_limit):
        return 0, 3, "fail: min_gap_cd too small"

    if (not np.isnan(min_right_cd)) and (min_right_cd < fail_right_limit):
        return 0, 4, "fail: min_right_cd too small"

    # borderline 级别诊断
    if weak_ratio > weak_tol:
        return 1, 5, "borderline: weak_ratio too high"

    if (not np.isnan(min_left_cd)) and (min_left_cd < weak_left_limit):
        return 1, 6, "borderline: min_left_cd in weak zone"

    if (not np.isnan(min_gap_cd)) and (min_gap_cd < weak_gap_limit):
        return 1, 7, "borderline: min_gap_cd in weak zone"

    if (not np.isnan(min_right_cd)) and (min_right_cd < weak_right_limit):
        return 1, 8, "borderline: min_right_cd in weak zone"

    return 2, 0, "pass"


# =========================================================
# 8. 目标函数（沿用前几天思路）
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
# 9. 统一评估函数
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

    row_status, left_cd_rows, gap_cd_rows, right_cd_rows, topology_bad_rows, summary = analyze_pattern_rows(
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

    global_status, reason_code, reason_text = diagnose_failure_reason(
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
        min_gap_cd=summary["min_gap_cd"],
        target_gap=0.55,
        weak_weight=0.5,
        fail_weight=2.0,
        gap_penalty_weight=1.5
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
        reason_code,
        reason_text,
        objective
    )


# =========================================================
# 10. 设置阈值
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
# 11. 扫描 dose 和 threshold
# =========================================================
dose_list = np.linspace(0.85, 1.20, 36)
threshold_list = np.linspace(0.17, 0.42, 36)

window_map = np.zeros((len(threshold_list), len(dose_list)))
reason_map = np.zeros((len(threshold_list), len(dose_list)))
objective_map = np.zeros((len(threshold_list), len(dose_list)))

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
            reason_code,
            reason_text,
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
        reason_map[i, j] = reason_code
        objective_map[i, j] = objective


# =========================================================
# 12. 选一个例子点来详细看诊断
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
    reason_code_ex,
    reason_text_ex,
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

reason_name = {
    0: "pass",
    1: "fail_ratio too high",
    2: "min_left_cd too small",
    3: "min_gap_cd too small",
    4: "min_right_cd too small",
    5: "weak_ratio too high",
    6: "min_left_cd in weak zone",
    7: "min_gap_cd in weak zone",
    8: "min_right_cd in weak zone",
    9: "topology problem"
}


# =========================================================
# 13. 作图
# =========================================================
plt.figure(figsize=(16, 11))

# (1) 整体 window map
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
plt.title("Window Map (0 fail, 1 borderline, 2 pass)")
plt.xlabel("dose")
plt.ylabel("threshold")
plt.plot(example_dose, example_threshold, 'ko', markersize=8)

# (2) 诊断原因图
plt.subplot(2, 3, 2)
plt.imshow(
    reason_map,
    extent=[dose_list.min(), dose_list.max(), threshold_list.min(), threshold_list.max()],
    origin='lower',
    aspect='auto',
    cmap='tab10',
    vmin=0,
    vmax=9
)
plt.colorbar()
plt.title("Reason Map")
plt.xlabel("dose")
plt.ylabel("threshold")
plt.plot(example_dose, example_threshold, 'wo', markersize=8)

# (3) objective map
plt.subplot(2, 3, 3)
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

# (4) 例子点 printed pattern
plt.subplot(2, 3, 4)
plt.imshow(
    printed_ex,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.title(f"Printed Example\n(dose={example_dose:.2f}, threshold={example_threshold:.2f})")
plt.xlabel("x")
plt.ylabel("y")

# (5) 例子点三条曲线
plt.subplot(2, 3, 5)
plt.plot(y, left_cd_rows_ex, label='left_cd')
plt.plot(y, gap_cd_rows_ex, label='gap_cd')
plt.plot(y, right_cd_rows_ex, label='right_cd')
plt.axhline(0.55, linestyle='--', label='weak limit')
plt.axhline(0.40, linestyle='--', label='fail limit')
plt.title("Row-wise Metrics")
plt.xlabel("y")
plt.ylabel("CD / gap")
plt.legend()

# (6) 总结文字
plt.subplot(2, 3, 6)
plt.axis('off')

summary_text = (
    "[Example point diagnosis]\n"
    f"global_status = {status_name[global_status_ex]}\n"
    f"primary_reason = {reason_name[reason_code_ex]}\n"
    f"objective = {objective_ex:.3f}\n\n"
    f"safe_ratio = {summary_ex['safe_ratio']:.3f}\n"
    f"weak_ratio = {summary_ex['weak_ratio']:.3f}\n"
    f"fail_ratio = {summary_ex['fail_ratio']:.3f}\n"
    f"min_left_cd = {summary_ex['min_left_cd']:.3f}\n"
    f"min_gap_cd = {summary_ex['min_gap_cd']:.3f}\n"
    f"min_right_cd = {summary_ex['min_right_cd']:.3f}\n"
    f"topology_bad_ratio = {summary_ex['topology_bad_ratio']:.3f}\n\n"
    "[Reason code legend]\n"
    "0 pass\n"
    "1 fail_ratio too high\n"
    "2 min_left_cd too small\n"
    "3 min_gap_cd too small\n"
    "4 min_right_cd too small\n"
    "5 weak_ratio too high\n"
    "6 min_left_cd in weak zone\n"
    "7 min_gap_cd in weak zone\n"
    "8 min_right_cd in weak zone\n"
    "9 topology problem"
)

plt.text(0.02, 0.98, summary_text, va='top', fontsize=10)
plt.title("Diagnosis Summary")

plt.tight_layout()
plt.show()
