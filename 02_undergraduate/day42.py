#从 CD / gap 过渡到 EPE——什么叫边缘位置误差 EPE = Edge Placement Error
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

# 加一个局部薄弱区
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
# 4. 阈值打印
# =========================================================
dose = 0.98
threshold = 0.28
printed = (dose * aerial >= threshold).astype(float)


# =========================================================
# 5. 单行边缘测量函数
#    返回四个边缘：
#    左线左边缘、左线右边缘、右线左边缘、右线右边缘
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
# 6. 目标边缘位置（来自 mask 几何定义）
# =========================================================
target_left_line_left_edge = left_x1
target_left_line_right_edge = left_x2
target_right_line_left_edge = right_x1
target_right_line_right_edge = right_x2


# =========================================================
# 7. 逐行计算 EPE
# =========================================================
left_outer_epe_rows = np.full(ny, np.nan)
left_inner_epe_rows = np.full(ny, np.nan)
right_inner_epe_rows = np.full(ny, np.nan)
right_outer_epe_rows = np.full(ny, np.nan)

num_feature_rows = np.full(ny, np.nan)

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

    num_feature_rows[i] = num_features

    if num_features < 2:
        continue

    # EPE = printed edge - target edge
    left_outer_epe_rows[i] = left_line_left_edge - target_left_line_left_edge
    left_inner_epe_rows[i] = left_line_right_edge - target_left_line_right_edge
    right_inner_epe_rows[i] = right_line_left_edge - target_right_line_left_edge
    right_outer_epe_rows[i] = right_line_right_edge - target_right_line_right_edge


# =========================================================
# 8. 统计 worst-case EPE
# =========================================================
valid_left_outer = left_outer_epe_rows[~np.isnan(left_outer_epe_rows)]
valid_left_inner = left_inner_epe_rows[~np.isnan(left_inner_epe_rows)]
valid_right_inner = right_inner_epe_rows[~np.isnan(right_inner_epe_rows)]
valid_right_outer = right_outer_epe_rows[~np.isnan(right_outer_epe_rows)]

max_abs_left_outer_epe = np.max(np.abs(valid_left_outer)) if len(valid_left_outer) > 0 else np.nan
max_abs_left_inner_epe = np.max(np.abs(valid_left_inner)) if len(valid_left_inner) > 0 else np.nan
max_abs_right_inner_epe = np.max(np.abs(valid_right_inner)) if len(valid_right_inner) > 0 else np.nan
max_abs_right_outer_epe = np.max(np.abs(valid_right_outer)) if len(valid_right_outer) > 0 else np.nan


# =========================================================
# 9. 作图
# =========================================================
plt.figure(figsize=(16, 10))

# (1) mask
plt.subplot(2, 3, 1)
plt.imshow(
    mask,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.title("Target Mask")
plt.xlabel("x")
plt.ylabel("y")

# (2) printed
plt.subplot(2, 3, 2)
plt.imshow(
    printed,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.title(f"Printed Pattern\n(dose={dose:.2f}, threshold={threshold:.2f})")
plt.xlabel("x")
plt.ylabel("y")

# (3) 左线两条边缘 EPE
plt.subplot(2, 3, 3)
plt.plot(y, left_outer_epe_rows, label='left outer EPE')
plt.plot(y, left_inner_epe_rows, label='left inner EPE')
plt.axhline(0.0, linestyle='--')
plt.title("Left Line Edge Placement Error")
plt.xlabel("y")
plt.ylabel("EPE")
plt.legend()

# (4) 右线两条边缘 EPE
plt.subplot(2, 3, 4)
plt.plot(y, right_inner_epe_rows, label='right inner EPE')
plt.plot(y, right_outer_epe_rows, label='right outer EPE')
plt.axhline(0.0, linestyle='--')
plt.title("Right Line Edge Placement Error")
plt.xlabel("y")
plt.ylabel("EPE")
plt.legend()

# (5) feature 数量检查
plt.subplot(2, 3, 5)
plt.plot(y, num_feature_rows)
plt.axhline(2, linestyle='--')
plt.title("Number of Features per Row")
plt.xlabel("y")
plt.ylabel("num_features")

# (6) 总结文字
plt.subplot(2, 3, 6)
plt.axis('off')

summary_text = (
    "[Worst-case absolute EPE]\n"
    f"left outer  = {max_abs_left_outer_epe:.4f}\n"
    f"left inner  = {max_abs_left_inner_epe:.4f}\n"
    f"right inner = {max_abs_right_inner_epe:.4f}\n"
    f"right outer = {max_abs_right_outer_epe:.4f}\n\n"
    "[Interpretation]\n"
    "EPE > 0 : printed edge shifts to +x direction\n"
    "EPE < 0 : printed edge shifts to -x direction\n"
    "Large |EPE| means the edge is far from its target position."
)

plt.text(0.02, 0.98, summary_text, va='top', fontsize=10)
plt.title("Summary")

plt.tight_layout()
plt.show()