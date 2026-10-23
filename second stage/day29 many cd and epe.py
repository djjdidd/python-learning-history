
import numpy as np
import torch
import matplotlib.pyplot as plt

# ==============================
# 1. 创建 Target
# ==============================

N = 64
dx = 2.0 / N

x = (np.arange(N) - N // 2) * dx

X, Y = np.meshgrid(x, x, indexing="ij")

target_np = (
    (np.abs(X) <= 0.3) &
    (np.abs(Y) <= 0.2)
).astype(np.float32)

target = torch.tensor(target_np)

# ==============================
# 2. 初始化 Z
# ==============================

Z = (4 * target - 2).detach().clone()
Z.requires_grad_(True)

# ==============================
# 3. 构造 Pupil
# ==============================

freq = np.fft.fftfreq(N, d=dx)

FX, FY = np.meshgrid(freq, freq, indexing="ij")

pupil_np = (
    FX**2 + FY**2 <= 3.0**2
).astype(np.float32)

pupil = torch.tensor(pupil_np)

beta = 10.0
threshold = 0.5

# ==============================
# 4. Forward Model
# ==============================

def forward(Z):

    M = torch.sigmoid(Z)

    F = torch.fft.fft2(M)

    E = torch.fft.ifft2(F * pupil)

    I = torch.abs(E) ** 2

    R = torch.sigmoid(
        beta * (I - threshold)
    )

    return M, I, R

# ==============================
# 5. 保存初始打印结果
# ==============================

with torch.no_grad():

    _, _, R_initial = forward(Z)

    R_initial = R_initial.clone()

# ==============================
# 6. Adam 优化 30 次
# ==============================

optimizer = torch.optim.Adam([Z], lr=0.1)

for step in range(30):

    optimizer.zero_grad()

    M, I, R = forward(Z)

    loss = torch.mean((R - target) ** 2)

    loss.backward()

    optimizer.step()

# ==============================
# 7. 得到最终打印结果
# ==============================

with torch.no_grad():

    _, _, R_final = forward(Z)

    initial_loss = torch.mean(
        (R_initial - target) ** 2
    )

    final_loss = torch.mean(
        (R_final - target) ** 2
    )

    # 二值化
    print_initial = (R_initial >= 0.5).float()

    print_final = (R_final >= 0.5).float()

    # 找出与 Target 不相同的位置
    error_initial = (
        print_initial != target
    ).float()

    error_final = (
        print_final != target
    ).float()

    # 统计错误像素
    wrong_initial = int(error_initial.sum().item())

    wrong_final = int(error_final.sum().item())

    # 计算错误率
    total_pixels = target.numel()

    rate_initial = wrong_initial / total_pixels * 100

    rate_final = wrong_final / total_pixels * 100


# =====================================
# 10. 定义边缘提取函数
# =====================================

def extract_edges(line):

    indices = torch.where(line > 0.5)[0]

    # 整条截线都没有打印
    if indices.numel() == 0:
        return None, "Missing"

    # 出现多段不连续的打印区域
    gaps = indices[1:] - indices[:-1]

    if torch.any(gaps != 1).item():
        return None, "Fragmented"

    left = x[indices[0].item()] - dx / 2
    right = x[indices[-1].item()] + dx / 2

    return (float(left), float(right)), "OK"


# =====================================
# 11. 在指定位置计算 EPE
# =====================================

def measure_epe(axis, index, printed):

    if axis == "fixed_x":

        # 固定 x，沿 y 方向测量
        target_line = target[index, :]
        printed_line = printed[index, :]

    else:

        # 固定 y，沿 x 方向测量
        target_line = target[:, index]
        printed_line = printed[:, index]

    target_edges, target_status = extract_edges(target_line)
    printed_edges, printed_status = extract_edges(printed_line)

    if target_status != "OK":
        raise ValueError("所选 Target 截线不适合当前测量方法")

    # 无法提取一段完整的打印区域时，不编造 EPE
    if printed_status != "OK":
        return None, printed_status

    left_epe = printed_edges[0] - target_edges[0]
    right_epe = printed_edges[1] - target_edges[1]

    return (left_epe, right_epe), "OK"


# =====================================
# 12. 选择多条测量截线
# =====================================

# 第一组：矩形内部及靠近边缘的位置
interior_samples = [
    ("fixed_x", 24),
    ("fixed_x", 28),
    ("fixed_x", 32),
    ("fixed_x", 36),
    ("fixed_x", 40),

    ("fixed_y", 28),
    ("fixed_y", 30),
    ("fixed_y", 32),
    ("fixed_y", 34),
    ("fixed_y", 36)
]

# 第二组：专门检查目标矩形的外侧边界
corner_samples = [
    ("fixed_x", 23),
    ("fixed_x", 41),
    ("fixed_y", 26),
    ("fixed_y", 38)
]


# =====================================
# 13. 计算并显示多位置 EPE
# =====================================

def format_result(result):

    epe, status = result

    if status != "OK":
        return status

    return f"({epe[0]:+.5f}, {epe[1]:+.5f})"


initial_interior_errors = []
final_interior_errors = []

for group_name, samples in [
    ("Interior Samples", interior_samples),
    ("Corner-near Samples", corner_samples)
]:

    print(f"\n===== {group_name} =====")

    for axis, index in samples:

        initial_result = measure_epe(
            axis, index, print_initial
        )

        final_result = measure_epe(
            axis, index, print_final
        )

        print(f"\n{axis}, index = {index}")

        print("Initial:", format_result(initial_result))
        print("Final:  ", format_result(final_result))

        # 只统计两种方法都有有效测量结果的内部截线
        if group_name == "Interior Samples":

            if (
                initial_result[1] == "OK"
                and final_result[1] == "OK"
            ):

                initial_interior_errors.extend(
                    initial_result[0]
                )

                final_interior_errors.extend(
                    final_result[0]
                )


# =====================================
# 14. 汇总平均绝对 EPE 和最大绝对 EPE
# =====================================

def summarize_epe(errors):

    values = np.abs(np.array(errors))

    mean_epe = np.mean(values)
    max_epe = np.max(values)

    return mean_epe, max_epe


mean_initial, max_initial = summarize_epe(
    initial_interior_errors
)

mean_final, max_final = summarize_epe(
    final_interior_errors
)

print("\n===== Interior EPE Summary =====")

print("Measured edge samples =", len(initial_interior_errors))

print(f"Initial Mean |EPE| = {mean_initial:.6f}")
print(f"Final Mean |EPE|   = {mean_final:.6f}")

print(f"Initial Max |EPE| = {max_initial:.6f}")
print(f"Final Max |EPE|   = {max_final:.6f}")


# =====================================
# 15. 查看角落与中心的打印截线
# =====================================

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, fixed_x, title in [
    (axes[0], 23, "Near Corner: x index 23"),
    (axes[1], 32, "Center: x index 32")
]:

    ax.step(
        x, target[fixed_x, :].numpy(),
        where="mid",
        label="Target",
        linewidth=2
    )

    ax.step(
        x, print_initial[fixed_x, :].numpy(),
        where="mid",
        label="Initial"
    )

    ax.step(
        x, print_final[fixed_x, :].numpy(),
        where="mid",
        label="Optimized"
    )

    ax.set_title(title)
    ax.set_xlabel("Y coordinate")
    ax.set_ylim(-0.1, 1.2)
    ax.legend()
    ax.grid(True)

axes[2].imshow(
    error_final.numpy(),
    cmap="gray",
    vmin=0,
    vmax=1,
    interpolation="nearest"
)

axes[2].set_title("Final Error Map")
axes[2].axis("off")

plt.tight_layout()
plt.show()
