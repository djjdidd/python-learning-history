#一周总结——从 EPE / contour / hotspot 到 toy OPC
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
# 3. 构造 line-space-line mask
# =========================================================
def build_mask(right_x1):
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
# 4. 成像打印
# =========================================================
def simulate_print(mask):
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    psf /= psf.sum()

    aerial = fftconvolve(mask, psf, mode='same')
    printed = (dose * aerial >= threshold).astype(float)

    return aerial, printed


# =========================================================
# 5. 单行边缘提取
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
# 6. 计算 right inner EPE
# =========================================================
def evaluate_right_inner_epe(right_x1_mask):
    mask = build_mask(right_x1_mask)
    aerial, printed = simulate_print(mask)

    right_inner_epe_rows = np.full(ny, np.nan)

    # 注意：target 永远是目标边缘，不随 OPC mask 改变
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
        worst_epe = np.nan

    return (
        mask,
        printed,
        right_inner_epe_rows,
        mean_abs_epe,
        worst_abs_epe,
        worst_epe,
        worst_y,
        worst_index
    )


# =========================================================
# 7. 修正前评估
# =========================================================
(
    mask_before,
    printed_before,
    epe_before,
    mean_abs_before,
    worst_abs_before,
    worst_epe_before,
    worst_y_before,
    worst_index_before
) = evaluate_right_inner_epe(right_x1_target)


# =========================================================
# 8. 一步 toy OPC 修正
# =========================================================
correction_gain = 0.5
right_x1_corrected = right_x1_target - correction_gain * worst_epe_before


# =========================================================
# 9. 修正后评估
# =========================================================
(
    mask_after,
    printed_after,
    epe_after,
    mean_abs_after,
    worst_abs_after,
    worst_epe_after,
    worst_y_after,
    worst_index_after
) = evaluate_right_inner_epe(right_x1_corrected)


# =========================================================
# 10. 提取简单 contour，用于总结图展示
# =========================================================
def extract_boundary(binary_image):
    ny_img, nx_img = binary_image.shape
    boundary = np.zeros_like(binary_image)

    for i in range(1, ny_img - 1):
        for j in range(1, nx_img - 1):
            if binary_image[i, j] == 0:
                continue

            up = binary_image[i - 1, j]
            down = binary_image[i + 1, j]
            left = binary_image[i, j - 1]
            right = binary_image[i, j + 1]

            if (up == 0) or (down == 0) or (left == 0) or (right == 0):
                boundary[i, j] = 1

    return boundary


target_contour = extract_boundary(mask_before)
printed_contour_before = extract_boundary(printed_before)
printed_contour_after = extract_boundary(printed_after)


# =========================================================
# 11. 作图：一周总结图
# =========================================================
plt.figure(figsize=(16, 11))

# (1) target mask
plt.subplot(2, 3, 1)
plt.imshow(
    mask_before,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.title("Target / Original Mask")
plt.xlabel("x")
plt.ylabel("y")

# (2) printed before
plt.subplot(2, 3, 2)
plt.imshow(
    printed_before,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.axhline(worst_y_before, linestyle='--')
plt.title("Printed Before OPC\nwith Hotspot Row")
plt.xlabel("x")
plt.ylabel("y")

# (3) contour before
plt.subplot(2, 3, 3)
plt.imshow(
    printed_contour_before + 2 * target_contour,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='viridis'
)
plt.colorbar()
plt.title("Contour Before OPC\n(target + printed)")
plt.xlabel("x")
plt.ylabel("y")

# (4) mask after OPC
plt.subplot(2, 3, 4)
plt.imshow(
    mask_after,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.title("Mask After Toy OPC")
plt.xlabel("x")
plt.ylabel("y")

# (5) printed after
plt.subplot(2, 3, 5)
plt.imshow(
    printed_after,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.axhline(worst_y_after, linestyle='--')
plt.title("Printed After OPC\nwith Hotspot Row")
plt.xlabel("x")
plt.ylabel("y")

# (6) EPE before / after
plt.subplot(2, 3, 6)
plt.plot(y, epe_before, label='before OPC')
plt.plot(y, epe_after, label='after OPC')
plt.axhline(0.0, linestyle='--')
plt.title("Right Inner EPE Before / After")
plt.xlabel("y")
plt.ylabel("EPE")
plt.legend()

plt.tight_layout()
plt.show()


# =========================================================
# 12. 总结文字图
# =========================================================
plt.figure(figsize=(10, 5))
plt.axis('off')

summary_text = (
    "[One-week summary]\n\n"
    "1. EPE measures edge displacement relative to the target edge.\n"
    "2. Contour comparison visualizes 2D boundary mismatch.\n"
    "3. Worst-case EPE identifies the most dangerous local hotspot.\n"
    "4. Toy OPC uses the measured EPE to pre-shift the mask edge.\n"
    "5. The target remains fixed; the mask is modified to improve the printed result.\n\n"
    "[Before / After]\n"
    f"right_x1 target/original = {right_x1_target:.4f}\n"
    f"right_x1 after toy OPC   = {right_x1_corrected:.4f}\n\n"
    f"mean |EPE| before = {mean_abs_before:.4f}\n"
    f"mean |EPE| after  = {mean_abs_after:.4f}\n\n"
    f"worst |EPE| before = {worst_abs_before:.4f}\n"
    f"worst |EPE| after  = {worst_abs_after:.4f}\n\n"
    "[Key logic]\n"
    "diagnose error -> locate hotspot -> update mask -> re-simulate -> compare"
)

plt.text(0.02, 0.98, summary_text, va='top', fontsize=11)
plt.title("Week Summary: EPE / Contour / Hotspot / Toy OPC")
plt.tight_layout()
plt.show()