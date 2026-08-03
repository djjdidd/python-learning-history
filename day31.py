#从“比例指标”走向“最差值指标”
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
right_x1, right_x2 = 0.25, 1.10

pattern_region = (yy >= y_min_pattern) & (yy <= y_max_pattern)
left_line = (xx >= left_x1) & (xx <= left_x2) & pattern_region
right_line = (xx >= right_x1) & (xx <= right_x2) & pattern_region

mask[left_line] = 1.0
mask[right_line] = 1.0

# 中间局部薄弱区
middle_weak_region = (yy >= -0.20) & (yy <= 0.20)
right_line_trim = (xx >= 0.25) & (xx <= 0.48) & middle_weak_region
mask[right_line_trim] = 0.0


# =========================================================
# 3. 构造二维高斯 PSF
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
# 6. 逐行分类 + 逐行几何记录函数
#    今天的新重点：除了分类，还收集每一行的 left/gap/right
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

    # 记录每一行的几何量
    left_cd_rows = np.full(ny, np.nan)
    gap_cd_rows = np.full(ny, np.nan)
    right_cd_rows = np.full(ny, np.nan)

    for i in range(ny):
        yy_value = y[i]

        # 图形外不参与分析
        if not (y_min_pattern <= yy_value <= y_max_pattern):
            continue

        row_binary = printed[i, :]
        left_cd, gap_cd, right_cd, num_features = measure_lsl_row(row_binary, x)

        # 先把当前行测量值记下来
        # 注意：如果是 None，就不写进去，保持 nan
        if left_cd is not None:
            left_cd_rows[i] = left_cd
        if gap_cd is not None:
            gap_cd_rows[i] = gap_cd
        if right_cd is not None:
            right_cd_rows[i] = right_cd

        # 开始分类
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

    # 有效分类行
    valid_status_mask = ~np.isnan(row_status)
    valid_status = row_status[valid_status_mask]

    if len(valid_status) == 0:
        safe_ratio = 0.0
        weak_ratio = 0.0
        fail_ratio = 0.0
    else:
        safe_ratio = np.mean(valid_status == 0)
        weak_ratio = np.mean(valid_status == 1)
        fail_ratio = np.mean(valid_status == 2)

    # 今天的新重点：求 worst-case（最小值）
    # 注意要排除 nan.先把 np.nan 去掉，只保留真正有数值的行
    valid_left = left_cd_rows[~np.isnan(left_cd_rows)]
    valid_gap = gap_cd_rows[~np.isnan(gap_cd_rows)]
    valid_right = right_cd_rows[~np.isnan(right_cd_rows)]

    if len(valid_left) == 0:
        min_left_cd = np.nan
    else:
        min_left_cd = np.min(valid_left)

    if len(valid_gap) == 0:
        min_gap_cd = np.nan
    else:
        min_gap_cd = np.min(valid_gap)

    if len(valid_right) == 0:
        min_right_cd = np.nan
    else:
        min_right_cd = np.min(valid_right)

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
# 7. 整体判定函数（先沿用第29天的容忍度规则）
# =========================================================
def evaluate_global_status_with_tolerance(fail_ratio, weak_ratio, fail_tol, weak_tol):
    if fail_ratio > fail_tol:
        return 0
    elif weak_ratio > weak_tol:
        return 1
    else:
        return 2


# =========================================================
# 8. 设置阈值
# =========================================================
weak_left_cd = 0.55
fail_left_cd = 0.40

weak_gap_cd = 0.55
fail_gap_cd = 0.38

weak_right_cd = 0.55
fail_right_cd = 0.40

fail_tol = 0.02
weak_tol = 0.08


# =========================================================
# 9. 扫描 dose 和 threshold
# =========================================================
dose_list = np.linspace(0.85, 1.20, 36)
threshold_list = np.linspace(0.17, 0.42, 36)

safe_ratio_map = np.zeros((len(threshold_list), len(dose_list)))
weak_ratio_map = np.zeros((len(threshold_list), len(dose_list)))
fail_ratio_map = np.zeros((len(threshold_list), len(dose_list)))

min_left_cd_map = np.full((len(threshold_list), len(dose_list)), np.nan)
min_gap_cd_map = np.full((len(threshold_list), len(dose_list)), np.nan)
min_right_cd_map = np.full((len(threshold_list), len(dose_list)), np.nan)

window_map = np.zeros((len(threshold_list), len(dose_list)))


# =========================================================
# 10. 双层循环分析
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

        safe_ratio_map[i, j] = summary["safe_ratio"]
        weak_ratio_map[i, j] = summary["weak_ratio"]
        fail_ratio_map[i, j] = summary["fail_ratio"]

        min_left_cd_map[i, j] = summary["min_left_cd"]
        min_gap_cd_map[i, j] = summary["min_gap_cd"]
        min_right_cd_map[i, j] = summary["min_right_cd"]

        window_map[i, j] = evaluate_global_status_with_tolerance(
            fail_ratio=summary["fail_ratio"],
            weak_ratio=summary["weak_ratio"],
            fail_tol=fail_tol,
            weak_tol=weak_tol
        )


# =========================================================
# 11. 选一个代表性参数点做逐行展示
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


# =========================================================
# 12. 作图
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

# (2) min gap map
plt.subplot(2, 3, 2)
plt.imshow(
    min_gap_cd_map,
    extent=[dose_list.min(), dose_list.max(), threshold_list.min(), threshold_list.max()],
    origin='lower',
    aspect='auto',
    cmap='plasma'
)
plt.colorbar()
plt.title("Min Gap CD Map")
plt.xlabel("dose")
plt.ylabel("threshold")

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

# (5) 行方向的 gap_cd 曲线
plt.subplot(2, 3, 5)
plt.plot(y, gap_cd_rows_ex)
plt.axhline(fail_gap_cd, linestyle='--')
plt.axhline(weak_gap_cd, linestyle='--')
plt.title("Row-wise Gap CD")
plt.xlabel("y")
plt.ylabel("gap_cd")

# (6) 文字总结
plt.subplot(2, 3, 6)
plt.axis('off')

summary_text = (
    f"Example point: dose={example_dose:.2f}, threshold={example_threshold:.2f}\n\n"
    f"safe_ratio = {summary_ex['safe_ratio']:.3f}\n"
    f"weak_ratio = {summary_ex['weak_ratio']:.3f}\n"
    f"fail_ratio = {summary_ex['fail_ratio']:.3f}\n\n"
    f"min_left_cd = {summary_ex['min_left_cd']:.3f}\n"
    f"min_gap_cd = {summary_ex['min_gap_cd']:.3f}\n"
    f"min_right_cd = {summary_ex['min_right_cd']:.3f}"
)

plt.text(0.02, 0.98, summary_text, va='top', fontsize=11)
plt.title("Summary")

plt.tight_layout()
plt.show()