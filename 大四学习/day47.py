#从一步 OPC 到迭代 OPC——为什么修一次通常不够 iterative迭代的
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
# 3. 构造 mask
# =========================================================
def build_mask(right_x1):
    mask = np.zeros((ny, nx), dtype=float)

    pattern_region = (yy >= y_min_pattern) & (yy <= y_max_pattern)

    left_line = (xx >= left_x1) & (xx <= left_x2) & pattern_region
    right_line = (xx >= right_x1) & (xx <= right_x2) & pattern_region

    mask[left_line] = 1.0
    mask[right_line] = 1.0

    # 局部薄弱区跟着 right_x1 移动
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
# 5. 单行边缘测量
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
# 6. 给定 right_x1，计算 right inner EPE
# =========================================================
def evaluate_right_inner_epe(right_x1_mask):
    mask = build_mask(right_x1_mask)
    aerial, printed = simulate_print(mask)

    right_inner_epe_rows = np.full(ny, np.nan)

    # target 不随 mask 改变
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

    mean_epe = np.mean(valid_epe) if len(valid_epe) > 0 else np.nan
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

    return (
        mask,
        printed,
        right_inner_epe_rows,
        mean_epe,
        mean_abs_epe,
        worst_abs_epe,
        worst_epe,
        worst_y,
        worst_index
    )


# =========================================================
# 7. 迭代 OPC 参数
# =========================================================
current_right_x1 = right_x1_target

correction_gain = 0.1
max_iters = 20
stop_tol = 0.005

right_x1_history = []
mean_abs_history = []
worst_abs_history = []
worst_epe_history = []


# =========================================================
# 8. 迭代 OPC 主循环
# =========================================================
for it in range(max_iters):

    (
        mask_current,
        printed_current,
        epe_current,
        mean_epe_current,
        mean_abs_current,
        worst_abs_current,
        worst_epe_current,
        worst_y_current,
        worst_index_current
    ) = evaluate_right_inner_epe(current_right_x1)

    # 记录当前状态
    right_x1_history.append(current_right_x1)
    mean_abs_history.append(mean_abs_current)
    worst_abs_history.append(worst_abs_current)
    worst_epe_history.append(worst_epe_current)

    # 如果 worst |EPE| 已经足够小，就停止
    if worst_abs_current < stop_tol:
        break

    # 根据当前 worst EPE 更新 mask 边缘
    current_right_x1 = current_right_x1 - correction_gain * worst_epe_current


# =========================================================
# 9. 最终结果再评估一次
# =========================================================
(
    mask_final,
    printed_final,
    epe_final,
    mean_epe_final,
    mean_abs_final,
    worst_abs_final,
    worst_epe_final,
    worst_y_final,
    worst_index_final
) = evaluate_right_inner_epe(current_right_x1)


# 初始结果用于对比
(
    mask_initial,
    printed_initial,
    epe_initial,
    mean_epe_initial,
    mean_abs_initial,
    worst_abs_initial,
    worst_epe_initial,
    worst_y_initial,
    worst_index_initial
) = evaluate_right_inner_epe(right_x1_target)


# =========================================================
# 10. 作图
# =========================================================
plt.figure(figsize=(16, 10))

# (1) 初始 printed
plt.subplot(2, 3, 1)
plt.imshow(
    printed_initial,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.axhline(worst_y_initial, linestyle='--')
plt.title("Printed Initial")
plt.xlabel("x")
plt.ylabel("y")

# (2) 最终 printed
plt.subplot(2, 3, 2)
plt.imshow(
    printed_final,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin='lower',
    aspect='auto',
    cmap='gray'
)
plt.axhline(worst_y_final, linestyle='--')
plt.title("Printed Final after Iterative OPC")
plt.xlabel("x")
plt.ylabel("y")

# (3) EPE before / after
plt.subplot(2, 3, 3)
plt.plot(y, epe_initial, label='initial')
plt.plot(y, epe_final, label='final')
plt.axhline(0.0, linestyle='--')
plt.title("Right Inner EPE Initial vs Final")
plt.xlabel("y")
plt.ylabel("EPE")
plt.legend()

# (4) right_x1 history
plt.subplot(2, 3, 4)
plt.plot(range(len(right_x1_history)), right_x1_history, marker='o')
plt.title("Mask Edge Update History")
plt.xlabel("iteration")
plt.ylabel("right_x1")

# (5) EPE history
plt.subplot(2, 3, 5)
plt.plot(range(len(mean_abs_history)), mean_abs_history, marker='o', label='mean |EPE|')
plt.plot(range(len(worst_abs_history)), worst_abs_history, marker='x', label='worst |EPE|')
plt.title("EPE History")
plt.xlabel("iteration")
plt.ylabel("EPE")
plt.legend()

# (6) summary
plt.subplot(2, 3, 6)
plt.axis('off')

summary_text = (
    "[Iterative Toy OPC]\n"
    f"correction_gain = {correction_gain:.2f}\n"
    f"max_iters = {max_iters}\n"
    f"actual_iters = {len(right_x1_history)}\n\n"
    "[Mask edge]\n"
    f"initial right_x1 = {right_x1_target:.4f}\n"
    f"final right_x1 = {current_right_x1:.4f}\n\n"
    "[EPE]\n"
    f"mean |EPE| initial = {mean_abs_initial:.4f}\n"
    f"mean |EPE| final   = {mean_abs_final:.4f}\n"
    f"worst |EPE| initial = {worst_abs_initial:.4f}\n"
    f"worst |EPE| final   = {worst_abs_final:.4f}\n\n"
    "[Logic]\n"
    "simulate -> measure EPE -> update mask -> repeat"
)

plt.text(0.02, 0.98, summary_text, va='top', fontsize=10)
plt.title("Summary")

plt.tight_layout()
plt.show()