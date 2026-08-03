#不同 objective 会导致不同 OPC 行为
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

    # 局部薄弱区跟随 right_x1 移动
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
# 6. 评估 right inner EPE
# =========================================================
def evaluate_right_inner_epe(right_x1_mask):
    mask = build_mask(right_x1_mask)
    aerial, printed = simulate_print(mask)

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
        "mask": mask,
        "printed": printed,
        "epe_rows": right_inner_epe_rows,
        "mean_epe": mean_epe,
        "mean_abs_epe": mean_abs_epe,
        "worst_abs_epe": worst_abs_epe,
        "worst_epe": worst_epe,
        "worst_y": worst_y,
        "worst_index": worst_index,
    }


# =========================================================
# 7. 根据策略计算 objective
# =========================================================
def compute_strategy_objective(metrics, strategy_name, worst_weight=2.0):
    mean_abs = metrics["mean_abs_epe"]
    worst_abs = metrics["worst_abs_epe"]

    if strategy_name == "mean":
        objective = mean_abs

    elif strategy_name == "worst":
        objective = worst_abs

    elif strategy_name == "combined":
        objective = mean_abs + worst_weight * worst_abs

    else:
        raise ValueError("Unknown strategy_name. Use 'mean', 'worst', or 'combined'.")

    return objective


# =========================================================
# 8. 通用 OPC 迭代函数
# =========================================================
def run_opc_with_strategy(
    strategy_name,
    correction_gain=0.2,
    min_gain=0.01,
    gain_shrink=0.5,
    max_iters=20,
    stop_tol=0.005,
    worst_weight=2.0
):
    current_right_x1 = right_x1_target

    current_metrics = evaluate_right_inner_epe(current_right_x1)
    current_objective = compute_strategy_objective(
        current_metrics,
        strategy_name,
        worst_weight=worst_weight
    )

    right_x1_history = []
    mean_abs_history = []
    worst_abs_history = []
    objective_history = []
    gain_history = []
    accepted_history = []

    for it in range(max_iters):
        right_x1_history.append(current_right_x1)
        mean_abs_history.append(current_metrics["mean_abs_epe"])
        worst_abs_history.append(current_metrics["worst_abs_epe"])
        objective_history.append(current_objective)
        gain_history.append(correction_gain)

        if current_metrics["worst_abs_epe"] < stop_tol:
            accepted_history.append(True)
            break

        # 更新方向仍然由当前 worst EPE 决定
        trial_right_x1 = current_right_x1 - correction_gain * current_metrics["worst_epe"]

        trial_metrics = evaluate_right_inner_epe(trial_right_x1)
        trial_objective = compute_strategy_objective(
            trial_metrics,
            strategy_name,
            worst_weight=worst_weight
        )

        if trial_objective < current_objective:
            current_right_x1 = trial_right_x1
            current_metrics = trial_metrics
            current_objective = trial_objective
            accepted_history.append(True)

        else:
            correction_gain = correction_gain * gain_shrink
            accepted_history.append(False)

            if correction_gain < min_gain:
                break

    result = {
        "strategy_name": strategy_name,
        "final_right_x1": current_right_x1,
        "final_metrics": current_metrics,
        "final_objective": current_objective,
        "right_x1_history": right_x1_history,
        "mean_abs_history": mean_abs_history,
        "worst_abs_history": worst_abs_history,
        "objective_history": objective_history,
        "gain_history": gain_history,
        "accepted_history": accepted_history,
        "final_gain": correction_gain,
    }

    return result


# =========================================================
# 9. 分别运行三种策略
# =========================================================
result_mean = run_opc_with_strategy("mean", worst_weight=2.0)
result_worst = run_opc_with_strategy("worst", worst_weight=2.0)
result_combined = run_opc_with_strategy("combined", worst_weight=2.0)

results = [result_mean, result_worst, result_combined]
strategy_names = ["mean", "worst", "combined"]


# =========================================================
# 10. 初始结果
# =========================================================
initial_metrics = evaluate_right_inner_epe(right_x1_target)
initial_mean_abs = initial_metrics["mean_abs_epe"]
initial_worst_abs = initial_metrics["worst_abs_epe"]


# =========================================================
# 11. 整理最终结果
# =========================================================
final_mean_abs_values = [r["final_metrics"]["mean_abs_epe"] for r in results]
final_worst_abs_values = [r["final_metrics"]["worst_abs_epe"] for r in results]
final_objective_values = [r["final_objective"] for r in results]
final_right_x1_values = [r["final_right_x1"] for r in results]


# =========================================================
# 12. 作图
# =========================================================
plt.figure(figsize=(16, 10))

# (1) 三种策略的 mean |EPE| history
plt.subplot(2, 3, 1)
for r in results:
    plt.plot(
        range(len(r["mean_abs_history"])),
        r["mean_abs_history"],
        marker='o',
        label=r["strategy_name"]
    )
plt.axhline(initial_mean_abs, linestyle='--', label='initial')
plt.title("Mean |EPE| History")
plt.xlabel("iteration")
plt.ylabel("mean |EPE|")
plt.legend()

# (2) 三种策略的 worst |EPE| history
plt.subplot(2, 3, 2)
for r in results:
    plt.plot(
        range(len(r["worst_abs_history"])),
        r["worst_abs_history"],
        marker='x',
        label=r["strategy_name"]
    )
plt.axhline(initial_worst_abs, linestyle='--', label='initial')
plt.title("Worst |EPE| History")
plt.xlabel("iteration")
plt.ylabel("worst |EPE|")
plt.legend()

# (3) 三种策略的 objective history
plt.subplot(2, 3, 3)
for r in results:
    plt.plot(
        range(len(r["objective_history"])),
        r["objective_history"],
        marker='s',
        label=r["strategy_name"]
    )
plt.title("Objective History")
plt.xlabel("iteration")
plt.ylabel("objective")
plt.legend()

# (4) 最终 mean |EPE| 对比
plt.subplot(2, 3, 4)
plt.bar(strategy_names, final_mean_abs_values)
plt.axhline(initial_mean_abs, linestyle='--')
plt.title("Final Mean |EPE|")
plt.xlabel("strategy")
plt.ylabel("mean |EPE|")

# (5) 最终 worst |EPE| 对比
plt.subplot(2, 3, 5)
plt.bar(strategy_names, final_worst_abs_values)
plt.axhline(initial_worst_abs, linestyle='--')
plt.title("Final Worst |EPE|")
plt.xlabel("strategy")
plt.ylabel("worst |EPE|")

# (6) summary
plt.subplot(2, 3, 6)
plt.axis('off')

summary_text = (
    "[Initial]\n"
    f"mean |EPE| = {initial_mean_abs:.4f}\n"
    f"worst |EPE| = {initial_worst_abs:.4f}\n\n"
    "[Final by strategy]\n"
)

for r in results:
    name = r["strategy_name"]
    final_mean = r["final_metrics"]["mean_abs_epe"]
    final_worst = r["final_metrics"]["worst_abs_epe"]
    final_obj = r["final_objective"]
    final_x1 = r["final_right_x1"]
    accepted = sum(r["accepted_history"])
    rejected = len(r["accepted_history"]) - accepted

    summary_text += (
        f"{name}:\n"
        f"  final right_x1 = {final_x1:.4f}\n"
        f"  mean |EPE| = {final_mean:.4f}\n"
        f"  worst |EPE| = {final_worst:.4f}\n"
        f"  objective = {final_obj:.4f}\n"
        f"  accepted/rejected = {accepted}/{rejected}\n\n"
    )

plt.text(0.02, 0.98, summary_text, va='top', fontsize=9)
plt.title("Summary")

plt.tight_layout()
plt.show()


# =========================================================
# 13. 额外画三种策略最终 EPE 曲线
# =========================================================
plt.figure(figsize=(10, 5))

plt.plot(y, initial_metrics["epe_rows"], label='initial')

for r in results:
    plt.plot(
        y,
        r["final_metrics"]["epe_rows"],
        label=f"final - {r['strategy_name']}"
    )

plt.axhline(0.0, linestyle='--')
plt.title("Final EPE Curves under Different Strategies")
plt.xlabel("y")
plt.ylabel("EPE")
plt.legend()
plt.tight_layout()
plt.show()