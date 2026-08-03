# 第56天：真实 OPC 为什么更复杂
# 多工艺条件下评估 OPC：nominal 好，不代表整体稳

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
# 2. 固定几何参数
# =========================================================
y_min_pattern = -0.75
y_max_pattern = 0.75

left_x1, left_x2 = -1.10, -0.35

# 三段式右线 target
right_x1_target = 0.35      # 下段、上段目标右内边缘
right_x3_target = 0.55      # 中段目标右内边缘
right_x2 = 1.10             # 右线外边缘

threshold = 0.28


# =========================================================
# 3. 三段式 segment 定义
# =========================================================
segments = [
    (-0.75, -0.25),   # 下段
    (-0.25,  0.25),   # 中段
    ( 0.25,  0.75),   # 上段
]

segment_target_x1_list = [
    right_x1_target,
    right_x3_target,
    right_x1_target,
]


# =========================================================
# 4. 根据 y 位置返回 target right inner edge
# =========================================================
def get_target_right_inner_edge(yy_value):
    """
    根据 y 位置返回目标右线内边缘。

    下段、上段 target = 0.35
    中段 target = 0.55
    """

    if -0.25 <= yy_value <= 0.25:
        return right_x3_target
    else:
        return right_x1_target


# =========================================================
# 5. 构造三段式 mask
# =========================================================
def build_mask_three_part(
    lower_right_x1,
    middle_right_x1,
    upper_right_x1
):
    """
    构造三段式右边 mask。

    左边是一条完整矩形线。
    右边由下、中、上三个小矩形组成。
    """

    mask = np.zeros((ny, nx), dtype=float)

    pattern_region = (yy >= y_min_pattern) & (yy <= y_max_pattern)

    # 左线
    left_line = (
        (xx >= left_x1)
        & (xx <= left_x2)
        & pattern_region
    )

    # 右线下段
    lower_region = (
        (yy >= -0.75)
        & (yy <= -0.25)
    )

    right_lower = (
        (xx >= lower_right_x1)
        & (xx <= right_x2)
        & lower_region
    )

    # 右线中段
    middle_region = (
        (yy >= -0.25)
        & (yy <= 0.25)
    )

    right_middle = (
        (xx >= middle_right_x1)
        & (xx <= right_x2)
        & middle_region
    )

    # 右线上段
    upper_region = (
        (yy >= 0.25)
        & (yy <= 0.75)
    )

    right_upper = (
        (xx >= upper_right_x1)
        & (xx <= right_x2)
        & upper_region
    )

    mask[left_line] = 1.0
    mask[right_lower] = 1.0
    mask[right_middle] = 1.0
    mask[right_upper] = 1.0

    return mask


# =========================================================
# 6. 用 segment_right_x1_list 构造 OPC mask
# =========================================================
def build_mask_segmented(segment_right_x1_list):
    """
    segment_right_x1_list:
        [下段 right_x1, 中段 right_x1, 上段 right_x1]
    """

    return build_mask_three_part(
        lower_right_x1=segment_right_x1_list[0],
        middle_right_x1=segment_right_x1_list[1],
        upper_right_x1=segment_right_x1_list[2]
    )


# =========================================================
# 7. 成像打印
# =========================================================
def simulate_print(mask, sigma_value, dose_value):
    """
    mask -> aerial image -> printed pattern

    sigma_value:
        模拟成像 blur / focus 变化。

    dose_value:
        模拟曝光剂量变化。
    """

    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma_value**2))
    psf /= psf.sum()

    aerial = fftconvolve(mask, psf, mode="same")

    printed = (dose_value * aerial >= threshold).astype(float)

    return aerial, printed


# =========================================================
# 8. 单行边缘测量
# =========================================================
def measure_edges_one_row(row_binary, x):
    """
    对一行 printed pattern 测左右两条线的边缘。

    返回：
        left_line_left_edge
        left_line_right_edge
        right_line_left_edge
        right_line_right_edge
        num_features
    """

    idx = np.where(row_binary > 0)[0]

    if len(idx) == 0:
        return None, None, None, None, 0

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
        return None, None, None, None, num_features

    left_line_left_edge = x[starts[0]]
    left_line_right_edge = x[ends[0]]

    right_line_left_edge = x[starts[1]]
    right_line_right_edge = x[ends[1]]

    return (
        left_line_left_edge,
        left_line_right_edge,
        right_line_left_edge,
        right_line_right_edge,
        num_features
    )


# =========================================================
# 9. 评估 right inner EPE
# =========================================================
def evaluate_printed(printed):
    """
    逐行测量 right inner EPE。

    EPE = printed right inner edge - target right inner edge
    """

    right_inner_epe_rows = np.full(ny, np.nan)
    fail_feature_rows = 0

    for i in range(ny):
        yy_value = y[i]

        if not (y_min_pattern <= yy_value <= y_max_pattern):
            continue

        row_binary = printed[i, :]

        (
            left_line_left_edge,
            left_line_right_edge,
            right_line_left_edge,
            right_line_right_edge,
            num_features
        ) = measure_edges_one_row(row_binary, x)

        if num_features < 2:
            fail_feature_rows += 1
            continue

        target_right_inner_edge = get_target_right_inner_edge(yy_value)

        right_inner_epe_rows[i] = right_line_left_edge - target_right_inner_edge

    valid_epe = right_inner_epe_rows[~np.isnan(right_inner_epe_rows)]

    mean_epe = np.mean(valid_epe) if len(valid_epe) > 0 else 0.0
    mean_abs_epe = np.mean(np.abs(valid_epe)) if len(valid_epe) > 0 else np.nan
    worst_abs_epe = np.max(np.abs(valid_epe)) if len(valid_epe) > 0 else np.nan

    if len(valid_epe) > 0:
        valid_mask = ~np.isnan(right_inner_epe_rows)
        valid_indices = np.where(valid_mask)[0]

        local_idx = np.argmax(np.abs(valid_epe))
        worst_index = valid_indices[local_idx]

        worst_y = y[worst_index]
        worst_epe = right_inner_epe_rows[worst_index]
    else:
        worst_y = np.nan
        worst_epe = 0.0

    return {
        "epe_rows": right_inner_epe_rows,
        "mean_epe": mean_epe,
        "mean_abs_epe": mean_abs_epe,
        "worst_abs_epe": worst_abs_epe,
        "worst_epe": worst_epe,
        "worst_y": worst_y,
        "fail_feature_rows": fail_feature_rows,
    }


# =========================================================
# 10. 计算每个 segment 的 mean EPE
# =========================================================
def compute_segment_mean_epe(epe_rows, segments):
    """
    对每个 segment 计算 mean EPE。

    注意：
        用有符号 mean EPE。
        因为修正需要方向。
    """

    segment_mean_epe_list = []

    for segment in segments:
        y_start, y_end = segment

        region = (
            (y >= y_start)
            & (y <= y_end)
            & (~np.isnan(epe_rows))
        )

        values = epe_rows[region]

        if len(values) == 0:
            mean_epe = 0.0
        else:
            mean_epe = np.mean(values)

        segment_mean_epe_list.append(mean_epe)

    return segment_mean_epe_list


# =========================================================
# 11. 计算 OPC 修正
# =========================================================
def compute_segment_correction_limited(
    segment_mean_epe_list,
    segment_target_x1_list,
    correction_gain,
    max_shift
):
    """
    加 correction range 的 OPC。

    raw_correction = -gain * mean_epe

    再用 np.clip 限制：
        -max_shift <= correction <= max_shift
    """

    segment_right_x1_list = []

    for mean_epe, target_x1 in zip(segment_mean_epe_list, segment_target_x1_list):
        raw_correction = -correction_gain * mean_epe

        limited_correction = np.clip(
            raw_correction,
            -max_shift,
            max_shift
        )

        corrected_x1 = target_x1 + limited_correction

        segment_right_x1_list.append(corrected_x1)

    return segment_right_x1_list


# =========================================================
# 12. 平滑中段修正
# =========================================================
def smooth_segment_right_x1(segment_right_x1_list, smooth_weight=0.3):
    """
    三段式里只平滑中段。

    新中段 = 原中段 * (1 - smooth_weight)
             + 上下段平均 * smooth_weight
    """

    values = np.array(segment_right_x1_list, dtype=float)
    smoothed = values.copy()

    i = 1
    neighbor_mean = 0.5 * (values[i - 1] + values[i + 1])

    smoothed[i] = (
        (1 - smooth_weight) * values[i]
        + smooth_weight * neighbor_mean
    )

    return list(smoothed)


# =========================================================
# 13. 多工艺条件评估
# =========================================================
def evaluate_over_process_conditions(mask, process_conditions):
    """
    在多个 process condition 下评估同一个 mask。

    process_conditions:
        [(sigma1, dose1), (sigma2, dose2), ...]

    返回：
        results: 每个条件下的指标列表
    """

    results = []

    for sigma_value, dose_value in process_conditions:
        aerial, printed = simulate_print(
            mask=mask,
            sigma_value=sigma_value,
            dose_value=dose_value
        )

        metrics = evaluate_printed(printed)

        result = {
            "sigma": sigma_value,
            "dose": dose_value,
            "mean_abs_epe": metrics["mean_abs_epe"],
            "worst_abs_epe": metrics["worst_abs_epe"],
            "fail_feature_rows": metrics["fail_feature_rows"],
        }

        results.append(result)

    return results


# =========================================================
# 14. 从多条件结果中取 worst-over-process
# =========================================================
def summarize_process_results(results):
    """
    汇总多工艺条件结果。

    真实 OPC 中很重要的一点：
        不是只看 nominal，
        而是看所有工艺条件中的最差情况。
    """

    mean_list = [r["mean_abs_epe"] for r in results]
    worst_list = [r["worst_abs_epe"] for r in results]
    fail_list = [r["fail_feature_rows"] for r in results]

    summary = {
        "avg_mean_abs_epe": np.nanmean(mean_list),
        "worst_over_process": np.nanmax(worst_list),
        "max_fail_feature_rows": np.nanmax(fail_list),
    }

    return summary


# =========================================================
# 15. 主流程：nominal 条件下做 OPC
# =========================================================
nominal_sigma = 0.12
nominal_dose = 0.98

correction_gain = 1.0
max_shift = 0.04
smooth_weight = 0.3

# 原始 target-like mask
mask_before = build_mask_three_part(
    lower_right_x1=right_x1_target,
    middle_right_x1=right_x3_target,
    upper_right_x1=right_x1_target
)

# nominal 条件下打印
aerial_before_nominal, printed_before_nominal = simulate_print(
    mask=mask_before,
    sigma_value=nominal_sigma,
    dose_value=nominal_dose
)

metrics_before_nominal = evaluate_printed(printed_before_nominal)

# 根据 nominal 条件下的 EPE 做一次 OPC
segment_mean_epe_list = compute_segment_mean_epe(
    epe_rows=metrics_before_nominal["epe_rows"],
    segments=segments
)

segment_right_x1_limited = compute_segment_correction_limited(
    segment_mean_epe_list=segment_mean_epe_list,
    segment_target_x1_list=segment_target_x1_list,
    correction_gain=correction_gain,
    max_shift=max_shift
)

segment_right_x1_smooth = smooth_segment_right_x1(
    segment_right_x1_list=segment_right_x1_limited,
    smooth_weight=smooth_weight
)

mask_opc = build_mask_segmented(
    segment_right_x1_list=segment_right_x1_smooth
)

# nominal 下 OPC 后打印
aerial_opc_nominal, printed_opc_nominal = simulate_print(
    mask=mask_opc,
    sigma_value=nominal_sigma,
    dose_value=nominal_dose
)

metrics_opc_nominal = evaluate_printed(printed_opc_nominal)


# =========================================================
# 16. 设置多个工艺条件
# =========================================================
process_conditions = [
    (0.10, 0.94),
    (0.10, 0.98),
    (0.10, 1.02),

    (0.12, 0.94),
    (0.12, 0.98),
    (0.12, 1.02),

    (0.14, 0.94),
    (0.14, 0.98),
    (0.14, 1.02),
]


# =========================================================
# 17. 多条件评估 before / OPC
# =========================================================
results_before = evaluate_over_process_conditions(
    mask=mask_before,
    process_conditions=process_conditions
)

results_opc = evaluate_over_process_conditions(
    mask=mask_opc,
    process_conditions=process_conditions
)

summary_before = summarize_process_results(results_before)
summary_opc = summarize_process_results(results_opc)


# =========================================================
# 18. 作图
# =========================================================
plt.figure(figsize=(16, 10))


# (1) Before mask
plt.subplot(2, 3, 1)
plt.imshow(
    mask_before,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin="lower",
    aspect="auto",
    cmap="gray"
)
plt.axhline(-0.25, linestyle=":")
plt.axhline(0.25, linestyle=":")
plt.title("Before Mask")
plt.xlabel("x")
plt.ylabel("y")


# (2) OPC mask
plt.subplot(2, 3, 2)
plt.imshow(
    mask_opc,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin="lower",
    aspect="auto",
    cmap="gray"
)
plt.axhline(-0.25, linestyle=":")
plt.axhline(0.25, linestyle=":")
plt.title("OPC Mask")
plt.xlabel("x")
plt.ylabel("y")


# (3) Printed before at nominal
plt.subplot(2, 3, 3)
plt.imshow(
    printed_before_nominal,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin="lower",
    aspect="auto",
    cmap="gray"
)
plt.title("Printed Before: Nominal")
plt.xlabel("x")
plt.ylabel("y")


# (4) Printed after OPC at nominal
plt.subplot(2, 3, 4)
plt.imshow(
    printed_opc_nominal,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin="lower",
    aspect="auto",
    cmap="gray"
)
plt.title("Printed OPC: Nominal")
plt.xlabel("x")
plt.ylabel("y")


# (5) Nominal EPE comparison
plt.subplot(2, 3, 5)
plt.plot(y, metrics_before_nominal["epe_rows"], label="before nominal")
plt.plot(y, metrics_opc_nominal["epe_rows"], label="opc nominal")
plt.axhline(0.0, linestyle="--")
plt.axvline(-0.25, linestyle=":")
plt.axvline(0.25, linestyle=":")
plt.title("Nominal EPE Comparison")
plt.xlabel("y")
plt.ylabel("EPE")
plt.legend()


# (6) Summary text
plt.subplot(2, 3, 6)
plt.axis("off")

summary_text = (
    "[Nominal before]\n"
    f"mean |EPE| = {metrics_before_nominal['mean_abs_epe']:.4f}\n"
    f"worst |EPE| = {metrics_before_nominal['worst_abs_epe']:.4f}\n\n"
    "[Nominal OPC]\n"
    f"mean |EPE| = {metrics_opc_nominal['mean_abs_epe']:.4f}\n"
    f"worst |EPE| = {metrics_opc_nominal['worst_abs_epe']:.4f}\n\n"
    "[Process summary before]\n"
    f"avg mean |EPE| = {summary_before['avg_mean_abs_epe']:.4f}\n"
    f"worst over process = {summary_before['worst_over_process']:.4f}\n\n"
    "[Process summary OPC]\n"
    f"avg mean |EPE| = {summary_opc['avg_mean_abs_epe']:.4f}\n"
    f"worst over process = {summary_opc['worst_over_process']:.4f}\n\n"
    "[Key idea]\n"
    "Real OPC must work across\n"
    "multiple process conditions."
)

plt.text(0.02, 0.98, summary_text, va="top", fontsize=9)
plt.title("Summary")

plt.tight_layout()
plt.show()


# =========================================================
# 19. 输出多条件表格
# =========================================================
print("Segment correction:")
print("id | target_x1 | mean_EPE | opc_x1")

for i in range(len(segments)):
    print(
        f"{i:2d} | "
        f"{segment_target_x1_list[i]:9.4f} | "
        f"{segment_mean_epe_list[i]:8.4f} | "
        f"{segment_right_x1_smooth[i]:7.4f}"
    )

print("\nProcess condition comparison:")
print("sigma | dose | before_worst | opc_worst | before_mean | opc_mean")

for rb, ro in zip(results_before, results_opc):
    print(
        f"{rb['sigma']:.2f} | "
        f"{rb['dose']:.2f} | "
        f"{rb['worst_abs_epe']:.4f} | "
        f"{ro['worst_abs_epe']:.4f} | "
        f"{rb['mean_abs_epe']:.4f} | "
        f"{ro['mean_abs_epe']:.4f}"
    )

print("\nSummary before:")
print(summary_before)

print("\nSummary OPC:")
print(summary_opc)