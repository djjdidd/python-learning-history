#在 pass 区里选最优点，不只看 safe_ratio
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

# 中间局部做一个薄弱区，让结果更有层次
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
# 6. 逐行分类函数
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
    ny = printed.shape[0]
    row_status = np.full(ny, np.nan)

    for i in range(ny):
        yy_value = y[i]

        if not (y_min_pattern <= yy_value <= y_max_pattern):
            continue

        row_binary = printed[i, :]
        left_cd, gap_cd, right_cd, num_features = measure_lsl_row(row_binary, x)

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

    valid_mask = ~np.isnan(row_status)
    valid_status = row_status[valid_mask]

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
# 7. 带容忍度的整体判定函数
# =========================================================
def evaluate_global_status_with_tolerance(fail_ratio, weak_ratio, fail_tol, weak_tol):
    if fail_ratio > fail_tol:
        return 0
    elif weak_ratio > weak_tol:
        return 1
    else:
        return 2


# =========================================================
# 8. 今天的新函数：综合评分函数
#    先用一个简化版本
# =========================================================
def compute_score(safe_ratio, weak_ratio, fail_ratio):
    """
    一个简单的综合评分：
    - safe_ratio 越大越好
    - weak_ratio 越大越差
    - fail_ratio 越大越差，而且惩罚更重
    """
    score = 1.0 * safe_ratio - 0.5 * weak_ratio - 2.0 * fail_ratio
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
# 10. 设置整体容忍度规则（今天先固定一套中等规则）
# =========================================================
fail_tol = 0.02
weak_tol = 0.08


# =========================================================
# 11. 扫描 dose 和 threshold
# =========================================================
dose_list = np.linspace(0.0, 1.20, 36)
threshold_list = np.linspace(0.00, 0.42, 36)

safe_ratio_map = np.zeros((len(threshold_list), len(dose_list)))
weak_ratio_map = np.zeros((len(threshold_list), len(dose_list)))
fail_ratio_map = np.zeros((len(threshold_list), len(dose_list)))
score_map = np.zeros((len(threshold_list), len(dose_list)))
window_map = np.zeros((len(threshold_list), len(dose_list)))


# =========================================================
# 12. 双层循环
# =========================================================
for i, threshold in enumerate(threshold_list):
    for j, dose in enumerate(dose_list):

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

        score_map[i, j] = compute_score(safe_ratio, weak_ratio, fail_ratio)

        window_map[i, j] = evaluate_global_status_with_tolerance(
            fail_ratio=fail_ratio,
            weak_ratio=weak_ratio,
            fail_tol=fail_tol,
            weak_tol=weak_tol
        )


# =========================================================
# 13. 先找所有 pass 点
# =========================================================
pass_mask = (window_map == 2)

# 有没有至少一个 pass 点
has_pass = np.any(pass_mask)


# =========================================================
# 14. 在 pass 区里找 safe_ratio 最大的点
# =========================================================
if has_pass:
    safe_in_pass = np.where(pass_mask, safe_ratio_map, -1)#如果某个位置是 pass，就保留它原来的 safe_ratio。如果某个位置不是 pass，就把它改成 -1
    best_safe_flat_idx = np.argmax(safe_in_pass)#在这个新数组里找最大值的位置
    best_safe_i, best_safe_j = np.unravel_index(best_safe_flat_idx, safe_ratio_map.shape)#把刚才那个一维编号，再还原成二维下标 (i, j)

    best_safe_dose = dose_list[best_safe_j]
    best_safe_threshold = threshold_list[best_safe_i]
    best_safe_value = safe_ratio_map[best_safe_i, best_safe_j]
    best_safe_score = score_map[best_safe_i, best_safe_j]
    best_safe_weak = weak_ratio_map[best_safe_i, best_safe_j]
    best_safe_fail = fail_ratio_map[best_safe_i, best_safe_j]
else:
    best_safe_i, best_safe_j = None, None


# =========================================================
# 15. 在 pass 区里找 score 最大的点
# =========================================================
if has_pass:
    score_in_pass = np.where(pass_mask, score_map, -999)
    best_score_flat_idx = np.argmax(score_in_pass)
    best_score_i, best_score_j = np.unravel_index(best_score_flat_idx, score_map.shape)

    best_score_dose = dose_list[best_score_j]
    best_score_threshold = threshold_list[best_score_i]
    best_score_value = score_map[best_score_i, best_score_j]
    best_score_safe = safe_ratio_map[best_score_i, best_score_j]
    best_score_weak = weak_ratio_map[best_score_i, best_score_j]
    best_score_fail = fail_ratio_map[best_score_i, best_score_j]
else:
    best_score_i, best_score_j = None, None


# =========================================================
# 16. 作图
# =========================================================
plt.figure(figsize=(16, 10))

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

if has_pass:
    plt.plot(best_safe_dose, best_safe_threshold, 'ro', label='best by safe_ratio')
    plt.plot(best_score_dose, best_score_threshold, 'wx', markersize=10, label='best by score')
    plt.legend()

# (2) score map
plt.subplot(2, 3, 2)
plt.imshow(
    score_map,
    extent=[dose_list.min(), dose_list.max(), threshold_list.min(), threshold_list.max()],
    origin='lower',
    aspect='auto',
    cmap='plasma'
)
plt.colorbar()
plt.title("Score Map")
plt.xlabel("dose")
plt.ylabel("threshold")

if has_pass:
    plt.plot(best_safe_dose, best_safe_threshold, 'wo', label='best by safe_ratio')
    plt.plot(best_score_dose, best_score_threshold, 'rx', markersize=10, label='best by score')
    plt.legend()

# (3) window map
plt.subplot(2, 3, 3)
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
plt.title("Window Map (0=fail, 1=borderline, 2=pass)")
plt.xlabel("dose")
plt.ylabel("threshold")

if has_pass:
    plt.plot(best_safe_dose, best_safe_threshold, 'ko', label='best by safe_ratio')
    plt.plot(best_score_dose, best_score_threshold, 'gx', markersize=10, label='best by score')
    plt.legend()

# (4) best-safe 点对应的 printed
if has_pass:
    printed_best_safe = (best_safe_dose * aerial >= best_safe_threshold).astype(float)

    plt.subplot(2, 3, 4)
    plt.imshow(
        printed_best_safe,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin='lower',
        aspect='auto',
        cmap='gray'
    )
    plt.title("Printed Pattern\nBest by Safe Ratio")
    plt.xlabel("x")
    plt.ylabel("y")

# (5) best-score 点对应的 printed
if has_pass:
    printed_best_score = (best_score_dose * aerial >= best_score_threshold).astype(float)

    plt.subplot(2, 3, 5)
    plt.imshow(
        printed_best_score,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin='lower',
        aspect='auto',
        cmap='gray'
    )
    plt.title("Printed Pattern\nBest by Score")
    plt.xlabel("x")
    plt.ylabel("y")

# (6) 文字总结
plt.subplot(2, 3, 6)
plt.axis('off')

if has_pass:
    summary_text = (
        "Best point in pass region\n\n"
        f"[Best by safe_ratio]\n"
        f"dose = {best_safe_dose:.3f}\n"
        f"threshold = {best_safe_threshold:.3f}\n"
        f"safe_ratio = {best_safe_value:.3f}\n"
        f"weak_ratio = {best_safe_weak:.3f}\n"
        f"fail_ratio = {best_safe_fail:.3f}\n"
        f"score = {best_safe_score:.3f}\n\n"
        f"[Best by score]\n"
        f"dose = {best_score_dose:.3f}\n"
        f"threshold = {best_score_threshold:.3f}\n"
        f"safe_ratio = {best_score_safe:.3f}\n"
        f"weak_ratio = {best_score_weak:.3f}\n"
        f"fail_ratio = {best_score_fail:.3f}\n"
        f"score = {best_score_value:.3f}"
    )
else:
    summary_text = "No pass point found."

plt.text(0.02, 0.98, summary_text, va='top', fontsize=11)
plt.title("Summary")

plt.tight_layout()
plt.show()