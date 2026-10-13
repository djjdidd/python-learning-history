#worst-case EPE 与 hotspot——怎么从边缘误差里找最危险位置
#hotspot 可以理解成局部最危险位置
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

# 局部薄弱区：让右边线中间更容易出现局部热点
middle_weak_region = (yy >= -0.20) & (yy <= 0.20)
right_line_trim = (xx >= 0.35) & (xx <= 0.52) & middle_weak_region
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
# 6. 目标边缘位置
# =========================================================
target_left_line_left_edge = left_x1
target_left_line_right_edge = left_x2
target_right_line_left_edge = right_x1
target_right_line_right_edge = right_x2


# =========================================================
# 7. 逐行计算四条边的 EPE
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

    left_outer_epe_rows[i] = left_line_left_edge - target_left_line_left_edge
    left_inner_epe_rows[i] = left_line_right_edge - target_left_line_right_edge
    right_inner_epe_rows[i] = right_line_left_edge - target_right_line_left_edge
    right_outer_epe_rows[i] = right_line_right_edge - target_right_line_right_edge


# =========================================================
# 8. 新重点：找每条边的 worst-case EPE
# =========================================================
def find_worst_case_epe(epe_rows, y):
    """
    输入：
        epe_rows: 某条边在每一行上的 EPE
        y: y 坐标数组

    输出：
        worst_epe: 最差 EPE 原始值，带正负号
        worst_abs_epe: 最差 EPE 绝对值
        worst_y: 最差位置对应的 y 坐标
        worst_index: 最差位置对应的数组索引
    """

    valid_mask = ~np.isnan(epe_rows)

    if np.sum(valid_mask) == 0:
        return np.nan, np.nan, np.nan, None

    valid_epe = epe_rows[valid_mask]
    valid_y = y[valid_mask]
    valid_indices = np.where(valid_mask)[0]

    local_idx = np.argmax(np.abs(valid_epe))

    worst_epe = valid_epe[local_idx]
    worst_abs_epe = np.abs(worst_epe)
    worst_y = valid_y[local_idx]
    worst_index = valid_indices[local_idx]

    return worst_epe, worst_abs_epe, worst_y, worst_index


left_outer_worst = find_worst_case_epe(left_outer_epe_rows, y)
left_inner_worst = find_worst_case_epe(left_inner_epe_rows, y)
right_inner_worst = find_worst_case_epe(right_inner_epe_rows, y)
right_outer_worst = find_worst_case_epe(right_outer_epe_rows, y)

edge_names = [
    "left_outer",
    "left_inner",
    "right_inner",
    "right_outer"
]

worst_results = [
    left_outer_worst,
    left_inner_worst,
    right_inner_worst,
    right_outer_worst
]


# =========================================================
# 9. 在四条边里找全局 worst-case EPE
# =========================================================
#r[0] -> worst_epe
#r[1] -> worst_abs_epe
#r[2] -> worst_y
#r[3] -> worst_index
all_abs_values = np.array([r[1] for r in worst_results])
#从四条边的 worst-case 结果里，把每条边的 worst_abs_epe 取出来，组成一个数组
global_edge_idx = np.nanargmax(all_abs_values)
#在四条边的 worst absolute EPE 里，找出最大值的位置
global_edge_name = edge_names[global_edge_idx]
global_worst_epe = worst_results[global_edge_idx][0]
global_worst_abs_epe = worst_results[global_edge_idx][1]
global_worst_y = worst_results[global_edge_idx][2]
global_worst_index = worst_results[global_edge_idx][3]


# =========================================================
# 10. 作图
# =========================================================
plt.figure(figsize=(16, 10))

# (1) printed pattern，并标出全局 hotspot 行
plt.subplot(2, 3, 1)
plt.imshow(
    printed,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.axhline(global_worst_y, linestyle='--')
plt.title("Printed Pattern with Hotspot Row")
plt.xlabel("x")
plt.ylabel("y")

# (2) 左边线 EPE
plt.subplot(2, 3, 2)
plt.plot(y, left_outer_epe_rows, label='left outer')
plt.plot(y, left_inner_epe_rows, label='left inner')

plt.scatter(left_outer_worst[2], left_outer_worst[0], marker='o')
plt.scatter(left_inner_worst[2], left_inner_worst[0], marker='x')

plt.axhline(0.0, linestyle='--')
plt.title("Left Line EPE with Worst Points")
plt.xlabel("y")
plt.ylabel("EPE")
plt.legend()

# (3) 右边线 EPE
plt.subplot(2, 3, 3)
plt.plot(y, right_inner_epe_rows, label='right inner')
plt.plot(y, right_outer_epe_rows, label='right outer')

plt.scatter(right_inner_worst[2], right_inner_worst[0], marker='o')
plt.scatter(right_outer_worst[2], right_outer_worst[0], marker='x')

plt.axhline(0.0, linestyle='--')
plt.title("Right Line EPE with Worst Points")
plt.xlabel("y")
plt.ylabel("EPE")
plt.legend()

# (4) num_features 检查
plt.subplot(2, 3, 4)
plt.plot(y, num_feature_rows)
plt.axhline(2, linestyle='--')
plt.title("Number of Features per Row")
plt.xlabel("y")
plt.ylabel("num_features")

# (5) 四条边 worst-case absolute EPE 对比
plt.subplot(2, 3, 5)
plt.bar(edge_names, all_abs_values)
plt.title("Worst-case |EPE| by Edge")
plt.xlabel("edge")
plt.ylabel("worst |EPE|")
plt.xticks(rotation=20)

# (6) summary
plt.subplot(2, 3, 6)
plt.axis('off')

summary_text = (
    "[Global worst-case EPE]\n"
    f"edge = {global_edge_name}\n"
    f"worst EPE = {global_worst_epe:.4f}\n"
    f"worst |EPE| = {global_worst_abs_epe:.4f}\n"
    f"hotspot y = {global_worst_y:.4f}\n\n"
    "[Interpretation]\n"
    "The hotspot is the row where |EPE| is largest.\n"
    "Positive EPE means +x shift.\n"
    "Negative EPE means -x shift.\n"
    "Worst-case |EPE| highlights the most dangerous local edge error."
)

plt.text(0.02, 0.98, summary_text, va='top', fontsize=10)
plt.title("Summary")

plt.tight_layout()
plt.show()