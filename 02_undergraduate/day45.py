#为什么需要 OPC——从误差诊断走向图形修正
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
# 2. 一些固定参数
# =========================================================
y_min_pattern = -0.75
y_max_pattern = 0.75

left_x1, left_x2 = -1.10, -0.35
right_x1_original, right_x2 = 0.35, 1.10

sigma = 0.12
dose = 0.98
threshold = 0.28


# =========================================================
# 3. 构造 mask 的函数
# =========================================================
def build_lsl_mask(right_x1):
    """
    根据 right_x1 构造 line-space-line mask。
    今天我们只让右线内边缘 right_x1 可以被修改。
    """
    mask = np.zeros((ny, nx), dtype=float)

    pattern_region = (yy >= y_min_pattern) & (yy <= y_max_pattern)

    left_line = (xx >= left_x1) & (xx <= left_x2) & pattern_region
    right_line = (xx >= right_x1) & (xx <= right_x2) & pattern_region

    mask[left_line] = 1.0
    mask[right_line] = 1.0

    # 人为制造右线局部薄弱区
    middle_weak_region = (yy >= -0.20) & (yy <= 0.20)
    right_line_trim = (xx >= right_x1) & (xx <= right_x1 + 0.17) & middle_weak_region
    mask[right_line_trim] = 0.0

    return mask


# =========================================================
# 4. 成像和打印函数
# =========================================================
def print_mask(mask):
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    psf /= psf.sum()

    aerial = fftconvolve(mask, psf, mode='same')
    printed = (dose * aerial >= threshold).astype(float)

    return aerial, printed


# =========================================================
# 5. 单行边缘测量函数
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
# 6. 给定 right_x1，计算 right_inner EPE 曲线
# =========================================================
def evaluate_right_inner_epe(right_x1):
    mask = build_lsl_mask(right_x1)
    aerial, printed = print_mask(mask)

    right_inner_epe_rows = np.full(ny, np.nan)

    # 注意：target 仍然是原始目标 right_x1_original
    # OPC 修改的是 mask，但目标图形不变。
    target_right_inner_edge = right_x1_original

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

    valid = right_inner_epe_rows[~np.isnan(right_inner_epe_rows)]

    mean_abs_epe = np.mean(np.abs(valid)) if len(valid) > 0 else np.nan
    worst_abs_epe = np.max(np.abs(valid)) if len(valid) > 0 else np.nan

    return mask, printed, right_inner_epe_rows, mean_abs_epe, worst_abs_epe


# =========================================================
# 7. 修正前评估
# =========================================================
(
    mask_before,
    printed_before,
    epe_before,
    mean_abs_before,
    worst_abs_before
) = evaluate_right_inner_epe(right_x1_original)


# =========================================================
# 8. 根据 worst-case EPE 做一步简单 OPC
# =========================================================
valid_before = epe_before[~np.isnan(epe_before)]

if len(valid_before) > 0:
    worst_idx = np.argmax(np.abs(valid_before))
    worst_epe = valid_before[worst_idx]
else:
    worst_epe = 0.0

correction_gain = 0.5

right_x1_corrected = right_x1_original - correction_gain * worst_epe


# =========================================================
# 9. 修正后重新评估
# =========================================================
(
    mask_after,
    printed_after,
    epe_after,
    mean_abs_after,
    worst_abs_after
) = evaluate_right_inner_epe(right_x1_corrected)


# =========================================================
# 10. 作图
# =========================================================
plt.figure(figsize=(16, 10))

# (1) 修正前 mask
plt.subplot(2, 3, 1)
plt.imshow(
    mask_before,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.title("Mask Before OPC")
plt.xlabel("x")
plt.ylabel("y")

# (2) 修正后 mask
plt.subplot(2, 3, 2)
plt.imshow(
    mask_after,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.title("Mask After Simple OPC")
plt.xlabel("x")
plt.ylabel("y")

# (3) 修正前 printed
plt.subplot(2, 3, 3)
plt.imshow(
    printed_before,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.title("Printed Before OPC")
plt.xlabel("x")
plt.ylabel("y")

# (4) 修正后 printed
plt.subplot(2, 3, 4)
plt.imshow(
    printed_after,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.title("Printed After Simple OPC")
plt.xlabel("x")
plt.ylabel("y")

# (5) EPE 对比
plt.subplot(2, 3, 5)
plt.plot(y, epe_before, label='before OPC')
plt.plot(y, epe_after, label='after simple OPC')
plt.axhline(0.0, linestyle='--')
plt.title("Right Inner EPE Before / After")
plt.xlabel("y")
plt.ylabel("EPE")
plt.legend()

# (6) summary
plt.subplot(2, 3, 6)
plt.axis('off')

summary_text = (
    "[Simple OPC update]\n"
    f"original right_x1 = {right_x1_original:.4f}\n"
    f"worst EPE before = {worst_epe:.4f}\n"
    f"correction_gain = {correction_gain:.2f}\n"
    f"corrected right_x1 = {right_x1_corrected:.4f}\n\n"
    "[Right inner EPE]\n"
    f"mean |EPE| before = {mean_abs_before:.4f}\n"
    f"mean |EPE| after  = {mean_abs_after:.4f}\n"
    f"worst |EPE| before = {worst_abs_before:.4f}\n"
    f"worst |EPE| after  = {worst_abs_after:.4f}\n\n"
    "[Interpretation]\n"
    "This is a toy one-step OPC.\n"
    "The target edge is fixed.\n"
    "Only the mask edge is pre-shifted."
)

plt.text(0.02, 0.98, summary_text, va='top', fontsize=10)
plt.title("Summary")

plt.tight_layout()
plt.show()