# 第53天：OPC 为什么需要约束
# 三段式右边 mask + 三段 target + 三段 OPC
# correction range + smoothness constraint

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
# 2. 固定参数
# =========================================================
y_min_pattern = -0.75
y_max_pattern = 0.75

# 左边线条范围
left_x1, left_x2 = -1.10, -0.35

# 右边线条的三个目标内边缘位置
# 下段和上段的右线内边缘目标位置
right_x1_target = 0.35

# 中段的右线内边缘目标位置
# 这里故意让中段更靠右，相当于右边 mask 由三个小矩形组成
right_x3_target = 0.55

# 右边线条外边缘
right_x2 = 1.10

# 成像参数
sigma = 0.12
dose = 0.98
threshold = 0.28


# =========================================================
# 3. 三段式 y 区域定义
# =========================================================
# 这三个 segment 同时用于：
# 1. 构造原始 mask
# 2. 定义 target right inner edge
# 3. 做 segmented OPC
segments = [
    (-0.75, -0.25),   # 下段
    (-0.25,  0.25),   # 中段
    ( 0.25,  0.75),   # 上段
]

# 每一段对应自己的目标 right inner edge
# 下段 target = 0.35
# 中段 target = 0.55
# 上段 target = 0.35
segment_target_x1_list = [
    right_x1_target,
    right_x3_target,
    right_x1_target,
]


# =========================================================
# 4. 根据 y 位置返回目标 right inner edge
# =========================================================
def get_target_right_inner_edge(yy_value):
    """
    根据当前 y 位置，返回对应的目标右线内边缘位置。

    下段：target = right_x1_target
    中段：target = right_x3_target
    上段：target = right_x1_target

    这样 EPE 的定义就和三段式 target 对齐。
    """

    if -0.25 <= yy_value <= 0.25:
        return right_x3_target
    else:
        return right_x1_target


# =========================================================
# 5. 构造三段式原始 mask
# =========================================================
def build_mask_three_part(
    lower_right_x1,
    middle_right_x1,
    upper_right_x1
):
    """
    构造一个右边由三个小矩形组成的 mask。

    参数：
        lower_right_x1:
            下段右线内边缘位置。

        middle_right_x1:
            中段右线内边缘位置。

        upper_right_x1:
            上段右线内边缘位置。

    返回：
        mask:
            左边一条矩形线 + 右边三段式矩形线。
    """

    mask = np.zeros((ny, nx), dtype=float)

    pattern_region = (yy >= y_min_pattern) & (yy <= y_max_pattern)

    # -------------------------
    # 左边线条：保持完整矩形
    # -------------------------
    left_line = (
        (xx >= left_x1)
        & (xx <= left_x2)
        & pattern_region
    )
    mask[left_line] = 1.0

    # -------------------------
    # 右边线条：下段
    # -------------------------
    lower_region = (
        (yy >= -0.75)
        & (yy <= -0.25)
    )

    right_line_lower = (
        (xx >= lower_right_x1)
        & (xx <= right_x2)
        & lower_region
    )

    # -------------------------
    # 右边线条：中段
    # -------------------------
    middle_region = (
        (yy >= -0.25)
        & (yy <= 0.25)
    )

    right_line_middle = (
        (xx >= middle_right_x1)
        & (xx <= right_x2)
        & middle_region
    )

    # -------------------------
    # 右边线条：上段
    # -------------------------
    upper_region = (
        (yy >= 0.25)
        & (yy <= 0.75)
    )

    right_line_upper = (
        (xx >= upper_right_x1)
        & (xx <= right_x2)
        & upper_region
    )

    # 写入 mask
    mask[right_line_lower] = 1.0
    mask[right_line_middle] = 1.0
    mask[right_line_upper] = 1.0

    return mask


# =========================================================
# 6. 根据 segment_right_x1_list 构造三段式 OPC mask
# =========================================================
def build_mask_segmented(segment_right_x1_list):
    """
    根据每个 segment 的 right_x1 构造三段式 OPC mask。

    segment_right_x1_list 长度必须等于 3：
        [下段 right_x1, 中段 right_x1, 上段 right_x1]

    注意：
        这里不再额外挖洞，也不再使用 5 段。
        因为现在三段 mask、三段 target、三段 OPC 是统一的。
    """

    lower_right_x1 = segment_right_x1_list[0]
    middle_right_x1 = segment_right_x1_list[1]
    upper_right_x1 = segment_right_x1_list[2]

    mask = build_mask_three_part(
        lower_right_x1=lower_right_x1,
        middle_right_x1=middle_right_x1,
        upper_right_x1=upper_right_x1
    )

    return mask


# =========================================================
# 7. 成像打印
# =========================================================
def simulate_print(mask):
    """
    简化成像模型：
        mask -> PSF 卷积 -> aerial image -> threshold -> printed pattern
    """

    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    psf /= psf.sum()

    aerial = fftconvolve(mask, psf, mode='same')

    printed = (dose * aerial >= threshold).astype(float)

    return aerial, printed


# =========================================================
# 8. 单行边缘测量
# =========================================================
def measure_edges_one_row(row_binary, x):
    """
    对一行二值图形测量左右两条线的边缘。

    返回：
        left_line_left_edge
        left_line_right_edge
        right_line_left_edge
        right_line_right_edge
        num_features

    其中 right_line_left_edge 就是我们今天关注的 right inner edge。
    """

    # 找出这一行中 printed = 1 的像素位置
    idx = np.where(row_binary > 0)[0]

    # 如果这一行没有图形，返回空
    if len(idx) == 0:
        return None, None, None, None, 0

    # 找相邻 index 的差值
    diffs = np.diff(idx)

    # 如果差值 > 1，说明中间断开了，是两个不同 feature
    split_points = np.where(diffs > 1)[0]

    starts = [idx[0]]
    ends = []

    for sp in split_points:
        ends.append(idx[sp])
        starts.append(idx[sp + 1])

    ends.append(idx[-1])

    num_features = len(starts)

    # 如果少于两个 feature，就无法识别左右两条线
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

    关键变化：
        target_right_inner_edge 不再是固定 0.35，
        而是根据 y 位置使用三段式 target。
    """

    right_inner_epe_rows = np.full(ny, np.nan)

    for i in range(ny):
        yy_value = y[i]

        # 只在有效图形高度内测量
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

        # 如果这一行没有成功测到左右两条线，就跳过
        if num_features < 2:
            continue

        # 根据当前 y 位置得到三段式 target edge
        target_right_inner_edge = get_target_right_inner_edge(yy_value)

        # EPE = printed edge - target edge
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
        worst_index = None
        worst_y = np.nan
        worst_epe = 0.0

    return {
        "epe_rows": right_inner_epe_rows,
        "mean_epe": mean_epe,
        "mean_abs_epe": mean_abs_epe,
        "worst_abs_epe": worst_abs_epe,
        "worst_epe": worst_epe,
        "worst_y": worst_y,
        "worst_index": worst_index,
    }


# =========================================================
# 10. 计算每个 segment 的平均 EPE
# =========================================================
def compute_segment_mean_epe(epe_rows, segments):
    """
    对每个 segment 计算 mean EPE。

    注意：
        这里用的是有符号 mean EPE，不是 mean |EPE|。
        因为 correction 需要知道误差方向。
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
# 11. 不加约束的三段式 OPC 修正
# =========================================================
def compute_segment_correction_unconstrained(
    segment_mean_epe_list,
    segment_target_x1_list,
    correction_gain
):
    """
    无约束修正。

    每个 segment 围绕自己的 target_x1 修正：

        corrected_x1 = target_x1 - gain * mean_epe

    如果 mean_epe > 0：
        printed edge 比 target 偏右，
        所以 mask edge 要往左移动。

    如果 mean_epe < 0：
        printed edge 比 target 偏左，
        所以 mask edge 要往右移动。
    """

    segment_right_x1_list = []

    for mean_epe, target_x1 in zip(segment_mean_epe_list, segment_target_x1_list):
        corrected_right_x1 = target_x1 - correction_gain * mean_epe
        segment_right_x1_list.append(corrected_right_x1)

    return segment_right_x1_list


# =========================================================
# 12. 加 correction range 约束的三段式 OPC
# =========================================================
def compute_segment_correction_limited(
    segment_mean_epe_list,
    segment_target_x1_list,
    correction_gain,
    max_shift
):
    """
    加 correction range 限制。

    raw_correction = -gain * mean_epe

    但每段最多只能移动 max_shift：

        -max_shift <= correction <= max_shift
    """

    segment_right_x1_list = []

    for mean_epe, target_x1 in zip(segment_mean_epe_list, segment_target_x1_list):
        raw_correction = -correction_gain * mean_epe

        # 限制修正量范围，避免过度 OPC
        limited_correction = np.clip(
            raw_correction,
            -max_shift,
            max_shift
        )

        corrected_right_x1 = target_x1 + limited_correction

        segment_right_x1_list.append(corrected_right_x1)

    return segment_right_x1_list


# =========================================================
# 13. smoothness 约束
# =========================================================
def smooth_segment_right_x1(segment_right_x1_list, smooth_weight=0.1):
    """
    保持上下段不变，只把中段的 right_x1 往上下段平均值靠近一点，从而减少中段和上下段之间的跳变。
    对三段 right_x1 做简单平滑。

    对三段情况：
        下段 index = 0
        中段 index = 1
        上段 index = 2

    这里只平滑中段，让中段不要和上下段跳变过大。

    smooth_weight = 0:
        完全不平滑。

    smooth_weight = 1:
        中段完全等于上下段平均值。

    smooth_weight = 0.5:
        中段一半保留原值，一半靠近上下段平均值。
    """

    values = np.array(segment_right_x1_list, dtype=float)
    smoothed = values.copy()#复制一份 values，作为后面要修改的结果

    # 三段式里，只平滑中间段
    i = 1

    neighbor_mean = 0.5 * (values[i - 1] + values[i + 1])

    smoothed[i] = (
        (1 - smooth_weight) * values[i]
        + smooth_weight * neighbor_mean
    )

    return list(smoothed)


# =========================================================
# 14. 原始 mask 和原始 printed
# =========================================================
mask_before = build_mask_three_part(
    lower_right_x1=right_x1_target,
    middle_right_x1=right_x3_target,
    upper_right_x1=right_x1_target
)

aerial_before, printed_before = simulate_print(mask_before)
metrics_before = evaluate_printed(printed_before)


# =========================================================
# 15. 设置 OPC 参数
# =========================================================
correction_gain = 1.2
max_shift = 0.04
smooth_weight = 0.5


# =========================================================
# 16. 计算每段 mean EPE
# =========================================================
segment_mean_epe_list = compute_segment_mean_epe(
    epe_rows=metrics_before["epe_rows"],
    segments=segments
)


# =========================================================
# 17. 无约束 segmented OPC
# =========================================================
segment_right_x1_unconstrained = compute_segment_correction_unconstrained(
    segment_mean_epe_list=segment_mean_epe_list,
    segment_target_x1_list=segment_target_x1_list,
    correction_gain=correction_gain
)

mask_unconstrained = build_mask_segmented(
    segment_right_x1_list=segment_right_x1_unconstrained
)

aerial_unconstrained, printed_unconstrained = simulate_print(mask_unconstrained)
metrics_unconstrained = evaluate_printed(printed_unconstrained)


# =========================================================
# 18. 加 correction range 约束
# =========================================================
segment_right_x1_limited = compute_segment_correction_limited(
    segment_mean_epe_list=segment_mean_epe_list,
    segment_target_x1_list=segment_target_x1_list,
    correction_gain=correction_gain,
    max_shift=max_shift
)

mask_limited = build_mask_segmented(
    segment_right_x1_list=segment_right_x1_limited
)

aerial_limited, printed_limited = simulate_print(mask_limited)
metrics_limited = evaluate_printed(printed_limited)


# =========================================================
# 19. 加 smoothness 约束
# =========================================================
segment_right_x1_smooth = smooth_segment_right_x1(
    segment_right_x1_list=segment_right_x1_limited,
    smooth_weight=smooth_weight
)

mask_smooth = build_mask_segmented(
    segment_right_x1_list=segment_right_x1_smooth
)

aerial_smooth, printed_smooth = simulate_print(mask_smooth)
metrics_smooth = evaluate_printed(printed_smooth)


# =========================================================
# 20. 作图
# =========================================================
plt.figure(figsize=(16, 10))


# -------------------------
# (1) 原始 mask
# -------------------------
plt.subplot(2, 3, 1)
plt.imshow(
    mask_before,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.axhline(-0.25, linestyle=':', linewidth=1)
plt.axhline(0.25, linestyle=':', linewidth=1)
plt.title("Original Three-Part Mask")
plt.xlabel("x")
plt.ylabel("y")


# -------------------------
# (2) 无约束 OPC mask
# -------------------------
plt.subplot(2, 3, 2)
plt.imshow(
    mask_unconstrained,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.axhline(-0.25, linestyle=':', linewidth=1)
plt.axhline(0.25, linestyle=':', linewidth=1)
plt.title("Unconstrained Segmented OPC")
plt.xlabel("x")
plt.ylabel("y")


# -------------------------
# (3) range-limited OPC mask
# -------------------------
plt.subplot(2, 3, 3)
plt.imshow(
    mask_limited,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.axhline(-0.25, linestyle=':', linewidth=1)
plt.axhline(0.25, linestyle=':', linewidth=1)
plt.title("Range-Limited OPC")
plt.xlabel("x")
plt.ylabel("y")


# -------------------------
# (4) range + smoothness OPC mask
# -------------------------
plt.subplot(2, 3, 4)
plt.imshow(
    mask_smooth,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.axhline(-0.25, linestyle=':', linewidth=1)
plt.axhline(0.25, linestyle=':', linewidth=1)
plt.title("Range + Smoothness OPC")
plt.xlabel("x")
plt.ylabel("y")


# -------------------------
# (5) EPE 曲线比较
# -------------------------
plt.subplot(2, 3, 5)

plt.plot(y, metrics_before["epe_rows"], label='before')
plt.plot(y, metrics_unconstrained["epe_rows"], label='unconstrained')
plt.plot(y, metrics_limited["epe_rows"], label='range limited')
plt.plot(y, metrics_smooth["epe_rows"], label='range + smoothness')

plt.axhline(0.0, linestyle='--')

# 标出三段边界
plt.axvline(-0.25, linestyle=':', linewidth=1)
plt.axvline(0.25, linestyle=':', linewidth=1)

plt.title("Right Inner EPE Comparison")
plt.xlabel("y")
plt.ylabel("EPE")
plt.legend()


# -------------------------
# (6) summary
# -------------------------
plt.subplot(2, 3, 6)
plt.axis('off')

summary_text = (
    "[Parameters]\n"
    f"correction_gain = {correction_gain:.2f}\n"
    f"max_shift = {max_shift:.3f}\n"
    f"smooth_weight = {smooth_weight:.2f}\n\n"
    "[Before]\n"
    f"mean |EPE| = {metrics_before['mean_abs_epe']:.4f}\n"
    f"worst |EPE| = {metrics_before['worst_abs_epe']:.4f}\n\n"
    "[Unconstrained]\n"
    f"mean |EPE| = {metrics_unconstrained['mean_abs_epe']:.4f}\n"
    f"worst |EPE| = {metrics_unconstrained['worst_abs_epe']:.4f}\n\n"
    "[Range limited]\n"
    f"mean |EPE| = {metrics_limited['mean_abs_epe']:.4f}\n"
    f"worst |EPE| = {metrics_limited['worst_abs_epe']:.4f}\n\n"
    "[Range + Smoothness]\n"
    f"mean |EPE| = {metrics_smooth['mean_abs_epe']:.4f}\n"
    f"worst |EPE| = {metrics_smooth['worst_abs_epe']:.4f}\n\n"
    "[Key idea]\n"
    "Three-part mask, target, and OPC\n"
    "segments are now consistent.\n"
    "OPC needs constraints to remain stable."
)

plt.text(0.02, 0.98, summary_text, va='top', fontsize=9)
plt.title("Summary")

plt.tight_layout()
plt.show()


# =========================================================
# 21. 输出每段修正信息
# =========================================================
print("Segment correction comparison:")
print("id | y_start | y_end | target_x1 | mean_EPE | unconstrained | limited | smooth")

for i, segment in enumerate(segments):
    y_start, y_end = segment

    print(
        f"{i:2d} | "
        f"{y_start:7.3f} | "
        f"{y_end:5.3f} | "
        f"{segment_target_x1_list[i]:9.4f} | "
        f"{segment_mean_epe_list[i]:8.4f} | "
        f"{segment_right_x1_unconstrained[i]:13.4f} | "
        f"{segment_right_x1_limited[i]:7.4f} | "
        f"{segment_right_x1_smooth[i]:7.4f}"
    )

print("\nBefore:")
print("mean |EPE| =", metrics_before["mean_abs_epe"])
print("worst |EPE| =", metrics_before["worst_abs_epe"])

print("\nUnconstrained:")
print("mean |EPE| =", metrics_unconstrained["mean_abs_epe"])
print("worst |EPE| =", metrics_unconstrained["worst_abs_epe"])

print("\nRange limited:")
print("mean |EPE| =", metrics_limited["mean_abs_epe"])
print("worst |EPE| =", metrics_limited["worst_abs_epe"])

print("\nRange + Smoothness:")
print("mean |EPE| =", metrics_smooth["mean_abs_epe"])
print("worst |EPE| =", metrics_smooth["worst_abs_epe"])