#从全局边缘平移到局部 hotspot 修正
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
    middle_weak_region = (yy >= -0.20) & (yy <= 0.20)
    right_line_trim = (xx >= right_x1) & (xx <= right_x1 + 0.17) & middle_weak_region
    mask[right_line_trim] = 0.0

    return mask


# =========================================================
# 4. 构造局部修正 mask
# =========================================================
def build_mask_local(
    original_right_x1,
    corrected_right_x1,
    hotspot_y,
    hotspot_half_width
):
    mask = np.zeros((ny, nx), dtype=float)

    pattern_region = (yy >= y_min_pattern) & (yy <= y_max_pattern)

    left_line = (xx >= left_x1) & (xx <= left_x2) & pattern_region
    mask[left_line] = 1.0

    hotspot_region = (
        (yy >= hotspot_y - hotspot_half_width)
        & (yy <= hotspot_y + hotspot_half_width)
        & pattern_region
    )

    non_hotspot_region = pattern_region & (~hotspot_region)

    # 非 hotspot 区域：保持原始 right_x1
    right_line_normal = (
        (xx >= original_right_x1)
        & (xx <= right_x2)
        & non_hotspot_region
    )

    # hotspot 区域：使用 corrected_right_x1
    right_line_hotspot = (
        (xx >= corrected_right_x1)
        & (xx <= right_x2)
        & hotspot_region
    )

    mask[right_line_normal] = 1.0
    mask[right_line_hotspot] = 1.0

    # 局部薄弱区也保留，模拟原本较难打印的位置
    middle_weak_region = (yy >= -0.20) & (yy <= 0.20)
    right_line_trim = (
        (xx >= corrected_right_x1)
        & (xx <= corrected_right_x1 + 0.17)
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
# 8. 原始结果
# =========================================================
mask_before = build_mask_global(right_x1_target)
aerial_before, printed_before = simulate_print(mask_before)
metrics_before = evaluate_printed(printed_before)

worst_epe_before = metrics_before["worst_epe"]
hotspot_y = metrics_before["worst_y"]


# =========================================================
# 9. 全局修正：整条 right_x1 一起动
# =========================================================
correction_gain = 0.5

global_corrected_right_x1 = right_x1_target - correction_gain * worst_epe_before

mask_global = build_mask_global(global_corrected_right_x1)
aerial_global, printed_global = simulate_print(mask_global)
metrics_global = evaluate_printed(printed_global)


# =========================================================
# 10. 局部修正：只修 hotspot 附近
# =========================================================
hotspot_half_width = 0.18

local_corrected_right_x1 = right_x1_target - correction_gain * worst_epe_before

mask_local = build_mask_local(
    original_right_x1=right_x1_target,
    corrected_right_x1=local_corrected_right_x1,
    hotspot_y=hotspot_y,
    hotspot_half_width=hotspot_half_width
)

aerial_local, printed_local = simulate_print(mask_local)
metrics_local = evaluate_printed(printed_local)


# =========================================================
# 11. 作图
# =========================================================
plt.figure(figsize=(16, 10))

# (1) 初始 mask
plt.subplot(2, 3, 1)
plt.imshow(
    mask_before,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.axhline(hotspot_y, linestyle='--')
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
plt.axhline(hotspot_y, linestyle='--')
plt.title("Global Correction Mask")
plt.xlabel("x")
plt.ylabel("y")

# (3) 局部修正 mask
plt.subplot(2, 3, 3)
plt.imshow(
    mask_local,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.axhline(hotspot_y, linestyle='--')
plt.title("Local Hotspot Correction Mask")
plt.xlabel("x")
plt.ylabel("y")

# (4) printed 对比
plt.subplot(2, 3, 4)
plt.imshow(
    printed_local,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.axhline(hotspot_y, linestyle='--')
plt.title("Printed after Local Correction")
plt.xlabel("x")
plt.ylabel("y")

# (5) EPE 曲线对比
plt.subplot(2, 3, 5)
plt.plot(y, metrics_before["epe_rows"], label='before')
plt.plot(y, metrics_global["epe_rows"], label='global correction')
plt.plot(y, metrics_local["epe_rows"], label='local correction')
plt.axhline(0.0, linestyle='--')
plt.axvspan(#标记横坐标从 x1 到 x2 的竖向区域
    hotspot_y - hotspot_half_width,
    hotspot_y + hotspot_half_width,
    alpha=0.2,
    label='local correction region'
)
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
    f"hotspot y = {hotspot_y:.4f}\n\n"
    "[Global correction]\n"
    f"right_x1 = {global_corrected_right_x1:.4f}\n"
    f"mean |EPE| = {metrics_global['mean_abs_epe']:.4f}\n"
    f"worst |EPE| = {metrics_global['worst_abs_epe']:.4f}\n\n"
    "[Local correction]\n"
    f"local right_x1 = {local_corrected_right_x1:.4f}\n"
    f"hotspot half width = {hotspot_half_width:.2f}\n"
    f"mean |EPE| = {metrics_local['mean_abs_epe']:.4f}\n"
    f"worst |EPE| = {metrics_local['worst_abs_epe']:.4f}\n\n"
    "[Key idea]\n"
    "Global correction moves the whole edge.\n"
    "Local correction modifies only the hotspot region."
)

plt.text(0.02, 0.98, summary_text, va='top', fontsize=9)
plt.title("Summary")

plt.tight_layout()
plt.show()