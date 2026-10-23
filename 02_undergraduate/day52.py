# 第52天：分段 OPC——不同 y 段使用不同修正量

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

left_x1, left_x2 = -1.10, -0.35
right_x1_target, right_x2 = 0.35, 1.10

sigma = 0.12
dose = 0.98
threshold = 0.28


# =========================================================
# 3. 构造原始 mask
# =========================================================
def build_mask_global(right_x1):
    mask = np.zeros((ny, nx), dtype=float)

    pattern_region = (yy >= y_min_pattern) & (yy <= y_max_pattern)

    left_line = (xx >= left_x1) & (xx <= left_x2) & pattern_region
    right_line = (xx >= right_x1) & (xx <= right_x2) & pattern_region

    mask[left_line] = 1.0
    mask[right_line] = 1.0

    # 局部薄弱区
    # 这里故意把右线中间靠内侧的一小段挖掉
    # 目的是制造一个局部更难打印的 hotspot
    middle_weak_region = (yy >= -0.20) & (yy <= 0.20)
    right_line_trim = (
        (xx >= right_x1)
        & (xx <= right_x1 + 0.17)
        & middle_weak_region
    )
    mask[right_line_trim] = 0.0

    return mask


# =========================================================
# 4. 构造分段 OPC mask
# =========================================================
def build_mask_segmented(segments, segment_right_x1_list):
    """
    构造分段 OPC 后的 mask。

    参数:
        segments:
            y 方向上的分段列表。
            例如:
            [(-0.75, -0.45), (-0.45, -0.15), ...]

        segment_right_x1_list:
            每个 segment 对应的 right_x1。
            例如:
            [0.35, 0.34, 0.30, 0.34, 0.35]

    返回:
        mask:
            分段修正后的 mask。
    """

    mask = np.zeros((ny, nx), dtype=float)

    pattern_region = (yy >= y_min_pattern) & (yy <= y_max_pattern)

    # 左线保持不变
    left_line = (xx >= left_x1) & (xx <= left_x2) & pattern_region
    mask[left_line] = 1.0

    # 右线分段构造
    for segment, segment_right_x1 in zip(segments, segment_right_x1_list):#zip() 会把两个列表一一配对
        y_start, y_end = segment

        segment_region = (
            (yy >= y_start)
            & (yy <= y_end)
            & pattern_region
        )

        right_line_segment = (
            (xx >= segment_right_x1)
            & (xx <= right_x2)
            & segment_region
        )

        mask[right_line_segment] = 1.0

    # 仍然保留局部薄弱区
    # 注意：这里用 right_x1_target 附近来挖掉一小块
    # 让原本中间区域仍然具有较强 hotspot 特征
    middle_weak_region = (yy >= -0.20) & (yy <= 0.20)
    right_line_trim = (
        (xx >= right_x1_target)
        & (xx <= right_x1_target + 0.17)
        & middle_weak_region
    )
    mask[right_line_trim] = 0.0

    return mask


# =========================================================
# 5. 成像打印
# =========================================================
def simulate_print(mask):
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    psf /= psf.sum()

    aerial = fftconvolve(mask, psf, mode='same')
    printed = (dose * aerial >= threshold).astype(float)

    return aerial, printed


# =========================================================
# 6. 单行边缘测量
# =========================================================
def measure_edges_one_row(row_binary, x):
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
# 7. 评估 right inner EPE
# =========================================================
def evaluate_printed(printed):
    right_inner_epe_rows = np.full(ny, np.nan)

    target_right_inner_edge = right_x1_target

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
            continue

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
# 8. 计算每个 segment 的平均 EPE
# =========================================================
def compute_segment_mean_epe(epe_rows, segments):
    """
    对每个 y segment 计算 mean EPE。

    参数:
        epe_rows:
            每一行的 right inner EPE。

        segments:
            y 方向分段列表。

    返回:
        segment_mean_epe_list:
            每一段的平均 EPE。
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
# 9. 根据 segment EPE 计算每段修正后的 right_x1
# =========================================================
def compute_segment_correction(segment_mean_epe_list, correction_gain):
    """
    根据每段的 mean EPE,计算每段修正后的 right_x1。

    核心公式:
        corrected_right_x1 = right_x1_target - gain * mean_epe

    如果 mean_epe > 0:
        printed edge 偏右，所以 mask edge 往左修。

    如果 mean_epe < 0:
        printed edge 偏左，所以 mask edge 往右修。
    """

    segment_right_x1_list = []

    for mean_epe in segment_mean_epe_list:
        corrected_right_x1 = right_x1_target - correction_gain * mean_epe
        segment_right_x1_list.append(corrected_right_x1)

    return segment_right_x1_list


# =========================================================
# 10. 原始结果
# =========================================================
mask_before = build_mask_global(right_x1_target)
aerial_before, printed_before = simulate_print(mask_before)
metrics_before = evaluate_printed(printed_before)

worst_epe_before = metrics_before["worst_epe"]


# =========================================================
# 11. 全局修正作为对比
# =========================================================
correction_gain = 0.5

global_corrected_right_x1 = right_x1_target - correction_gain * worst_epe_before

mask_global = build_mask_global(global_corrected_right_x1)
aerial_global, printed_global = simulate_print(mask_global)
metrics_global = evaluate_printed(printed_global)


# =========================================================
# 12. 分段 OPC
# =========================================================
segments = [
    (-0.75, -0.45),
    (-0.45, -0.15),
    (-0.15,  0.15),
    ( 0.15,  0.45),
    ( 0.45,  0.75),
]

segment_mean_epe_list = compute_segment_mean_epe(
    epe_rows=metrics_before["epe_rows"],
    segments=segments
)

segment_right_x1_list = compute_segment_correction(
    segment_mean_epe_list=segment_mean_epe_list,
    correction_gain=correction_gain
)

mask_segmented = build_mask_segmented(
    segments=segments,
    segment_right_x1_list=segment_right_x1_list
)

aerial_segmented, printed_segmented = simulate_print(mask_segmented)
metrics_segmented = evaluate_printed(printed_segmented)


# =========================================================
# 13. 作图
# =========================================================
plt.figure(figsize=(16, 10))


# (1) 原始 mask
plt.subplot(2, 3, 1)
plt.imshow(
    mask_before,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.title("Original Mask")
plt.xlabel("x")
plt.ylabel("y")


# (2) 全局修正 mask
plt.subplot(2, 3, 2)
plt.imshow(
    mask_global,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.title("Global Correction Mask")
plt.xlabel("x")
plt.ylabel("y")


# (3) 分段 OPC mask
plt.subplot(2, 3, 3)
plt.imshow(
    mask_segmented,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)

# 用横线标出 segment 边界
for y_start, y_end in segments:
    plt.axhline(y_start, linestyle=':', linewidth=1)
    plt.axhline(y_end, linestyle=':', linewidth=1)

plt.title("Segmented OPC Mask")
plt.xlabel("x")
plt.ylabel("y")


# (4) 分段 OPC 后的 printed pattern
plt.subplot(2, 3, 4)
plt.imshow(
    printed_segmented,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)

for y_start, y_end in segments:
    plt.axhline(y_start, linestyle=':', linewidth=1)
    plt.axhline(y_end, linestyle=':', linewidth=1)

plt.title("Printed after Segmented OPC")
plt.xlabel("x")
plt.ylabel("y")


# (5) EPE 曲线对比
plt.subplot(2, 3, 5)
plt.plot(y, metrics_before["epe_rows"], label='before')
plt.plot(y, metrics_global["epe_rows"], label='global correction')
plt.plot(y, metrics_segmented["epe_rows"], label='segmented OPC')

plt.axhline(0.0, linestyle='--')

# 标出 segment 边界
for y_start, y_end in segments:
    plt.axvline(y_start, linestyle=':', linewidth=1)
    plt.axvline(y_end, linestyle=':', linewidth=1)

plt.title("Right Inner EPE Comparison")
plt.xlabel("y")
plt.ylabel("EPE")
plt.legend()


# (6) summary
plt.subplot(2, 3, 6)
plt.axis('off')

summary_text = (
    "[Before]\n"
    f"mean |EPE| = {metrics_before['mean_abs_epe']:.4f}\n"
    f"worst |EPE| = {metrics_before['worst_abs_epe']:.4f}\n"
    f"worst EPE = {metrics_before['worst_epe']:.4f}\n"
    f"worst y = {metrics_before['worst_y']:.4f}\n\n"
    "[Global correction]\n"
    f"right_x1 = {global_corrected_right_x1:.4f}\n"
    f"mean |EPE| = {metrics_global['mean_abs_epe']:.4f}\n"
    f"worst |EPE| = {metrics_global['worst_abs_epe']:.4f}\n\n"
    "[Segmented OPC]\n"
    f"mean |EPE| = {metrics_segmented['mean_abs_epe']:.4f}\n"
    f"worst |EPE| = {metrics_segmented['worst_abs_epe']:.4f}\n\n"
    "[Key idea]\n"
    "Global correction uses one edge shift.\n"
    "Segmented OPC uses different shifts\n"
    "for different y segments."
)

plt.text(0.02, 0.98, summary_text, va='top', fontsize=9)
plt.title("Summary")

plt.tight_layout()
plt.show()


# =========================================================
# 14. 输出每个 segment 的修正信息
# =========================================================
print("Segment correction table:")
print("segment_id | y_start | y_end | mean_EPE | corrected_right_x1")

for i, segment in enumerate(segments):
    y_start, y_end = segment
    mean_epe = segment_mean_epe_list[i]
    corrected_right_x1 = segment_right_x1_list[i]

    print(
        f"{i:9d} | "
        f"{y_start:7.3f} | "
        f"{y_end:5.3f} | "
        f"{mean_epe:8.4f} | "
        f"{corrected_right_x1:18.4f}"
    )

print("\nBefore:")
print("mean |EPE| =", metrics_before["mean_abs_epe"])
print("worst |EPE| =", metrics_before["worst_abs_epe"])

print("\nGlobal correction:")
print("mean |EPE| =", metrics_global["mean_abs_epe"])
print("worst |EPE| =", metrics_global["worst_abs_epe"])

print("\nSegmented OPC:")
print("mean |EPE| =", metrics_segmented["mean_abs_epe"])
print("worst |EPE| =", metrics_segmented["worst_abs_epe"])