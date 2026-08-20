#最简单的“迭代改参数”直觉
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
# 3. PSF 与 aerial
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
# 6. 多指标整体判定函数（沿用第32天）
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
# 7. 目标函数（沿用第33天）
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
# 8. 把“给定 dose，算 objective”封装成函数
#    这是今天非常关键的一步
# =========================================================
def evaluate_dose_objective(
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
    """
    输入一个 dose，输出：
    - objective 分数
    - overall_status
    - printed
    - summary
    """
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

    overall_status = evaluate_global_status_multi_metric(
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

    score = compute_objective(
        safe_ratio=summary["safe_ratio"],
        weak_ratio=summary["weak_ratio"],
        fail_ratio=summary["fail_ratio"],
        min_gap_cd=summary["min_gap_cd"],
        target_gap=0.55,
        weak_weight=0.5,
        fail_weight=2.0,
        gap_penalty_weight=1.5
    )

    return score, overall_status, printed, gap_cd_rows, summary


# =========================================================
# 9. 设置各种阈值
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
# 10. 今天固定 threshold，只优化 dose
# =========================================================
fixed_threshold = 0.28

# 初始 dose
current_dose = 0.95

# 每次试探的步长
step = 0.02

# 最多迭代次数
max_iters = 12

# dose 边界，防止跑太远
dose_min = 0.80
dose_max = 1.25

# 用来记录迭代历史
dose_history = []
score_history = []
status_history = []


# =========================================================
# 11. 迭代过程
# =========================================================
for it in range(max_iters):
    # 当前点
    current_score, current_status, current_printed, current_gap_rows, current_summary = evaluate_dose_objective(
        dose=current_dose,
        threshold=fixed_threshold,
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

    # 记录当前状态
    dose_history.append(current_dose)
    score_history.append(current_score)
    status_history.append(current_status)

    # 左边试一点
    left_dose = max(dose_min, current_dose - step)
    left_score, left_status, _, _, _ = evaluate_dose_objective(
        dose=left_dose,
        threshold=fixed_threshold,
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

    # 右边试一点
    right_dose = min(dose_max, current_dose + step)
    right_score, right_status, _, _, _ = evaluate_dose_objective(
        dose=right_dose,
        threshold=fixed_threshold,
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

    # -----------------------------------------------------
    # 今天先用最简单的规则：
    # 比较 current / left / right，谁 score 最大就走向谁
    # -----------------------------------------------------
    candidate_scores = [left_score, current_score, right_score]
    candidate_doses = [left_dose, current_dose, right_dose]

    best_local_idx = np.argmax(candidate_scores)
    best_local_dose = candidate_doses[best_local_idx]
    best_local_score = candidate_scores[best_local_idx]

    # 如果当前点已经不比左右差，就停止
    if best_local_dose == current_dose:
        break

    # 否则往更好的方向更新
    current_dose = best_local_dose


# =========================================================
# 12. 最终最优点再评估一次
# =========================================================
final_score, final_status, final_printed, final_gap_rows, final_summary = evaluate_dose_objective(
    dose=current_dose,
    threshold=fixed_threshold,
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

# (1) dose 更新轨迹
plt.subplot(2, 3, 1)
plt.plot(range(len(dose_history)), dose_history, marker='o')
plt.title("Dose Update History")
plt.xlabel("iteration")
plt.ylabel("dose")

# (2) score 更新轨迹
plt.subplot(2, 3, 2)
plt.plot(range(len(score_history)), score_history, marker='o')
plt.title("Objective History")
plt.xlabel("iteration")
plt.ylabel("objective")

# (3) 最终 printed pattern
plt.subplot(2, 3, 3)
plt.imshow(
    final_printed,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.title("Final Printed Pattern")
plt.xlabel("x")
plt.ylabel("y")

# (4) 最终 row-wise gap
plt.subplot(2, 3, 4)
plt.plot(y, final_gap_rows)
plt.axhline(0.55, linestyle='--')
plt.title("Final Row-wise Gap CD")
plt.xlabel("y")
plt.ylabel("gap_cd")

# (5) 迭代状态文字
plt.subplot(2, 3, 5)
plt.axis('off')

history_text = "Iteration history\n\n"
for k in range(len(dose_history)):
    history_text += (
        f"iter {k}: "
        f"dose={dose_history[k]:.3f}, "
        f"score={score_history[k]:.3f}, "
        f"status={status_name[status_history[k]]}\n"
    )

plt.text(0.02, 0.98, history_text, va='top', fontsize=10)#vertical alignment竖直方向的对齐方式。让这段文字的“顶部”对齐到你给的位置
plt.title("History Detail")

# (6) 最终总结
plt.subplot(2, 3, 6)
plt.axis('off')

summary_text = (
    f"Fixed threshold = {fixed_threshold:.3f}\n"
    f"Initial dose = {dose_history[0]:.3f}\n"
    f"Final dose = {current_dose:.3f}\n"
    f"Final objective = {final_score:.3f}\n"
    f"Final status = {status_name[final_status]}\n\n"
    f"safe_ratio = {final_summary['safe_ratio']:.3f}\n"
    f"weak_ratio = {final_summary['weak_ratio']:.3f}\n"
    f"fail_ratio = {final_summary['fail_ratio']:.3f}\n"
    f"min_gap_cd = {final_summary['min_gap_cd']:.3f}"
)

plt.text(0.02, 0.98, summary_text, va='top', fontsize=11)
plt.title("Final Summary")

plt.tight_layout()
plt.show()