
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

# ==============================
# 8. 输出评价结果
# ==============================

print("===== Printing Loss =====")

print("Initial Loss =", initial_loss.item())
print("Final Loss =", final_loss.item())

print("\n===== Binary Evaluation =====")

print("Target pixels =", int(target.sum().item()))

print(
    "Initial printed pixels =",
    int(print_initial.sum().item())
)

print(
    "Final printed pixels =",
    int(print_final.sum().item())
)

print("Initial wrong pixels =", wrong_initial)
print("Final wrong pixels =", wrong_final)

print(f"Initial Pixel Error = {rate_initial:.4f}%")
print(f"Final Pixel Error = {rate_final:.4f}%")

# ==============================
# 9. 显示四张对比图
# ==============================

images = [
    target,
    print_initial,
    print_final,
    error_final
]

titles = [
    "Target",
    "Initial Printed Pattern",
    "Optimized Printed Pattern",
    "Final Error Map"
]

fig, axes = plt.subplots(1, 4, figsize=(14, 4))

for ax, image, title in zip(axes, images, titles):

    ax.imshow(
        image.detach().cpu().numpy(),
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()

# =====================================
# 10. 提取 x 和 y 方向的中心截线
# =====================================

center = N // 2

# x 方向：固定 y，读取所有 x
target_x = target[:, center]
initial_x = print_initial[:, center]
final_x = print_final[:, center]

# y 方向：固定 x，读取所有 y
target_y = target[center, :]
initial_y = print_initial[center, :]
final_y = print_final[center, :]


# =====================================
# 11. 定义 CD 测量函数
# =====================================

def calculate_cd(line, dx):

    pixels = torch.sum(line).item()

    cd = pixels * dx

    return cd


# =====================================
# 12. 测量 CD
# =====================================

target_cdx = calculate_cd(target_x, dx)
initial_cdx = calculate_cd(initial_x, dx)
final_cdx = calculate_cd(final_x, dx)

target_cdy = calculate_cd(target_y, dx)
initial_cdy = calculate_cd(initial_y, dx)
final_cdy = calculate_cd(final_y, dx)


print("\n===== CD Measurement =====")

print("Target CDx =", target_cdx)
print("Initial CDx =", initial_cdx)
print("Final CDx =", final_cdx)

print()

print("Target CDy =", target_cdy)
print("Initial CDy =", initial_cdy)
print("Final CDy =", final_cdy)


print("\n===== CD Error =====")

print("Initial CDx Error =", initial_cdx - target_cdx)
print("Final CDx Error =", final_cdx - target_cdx)

print("Initial CDy Error =", initial_cdy - target_cdy)
print("Final CDy Error =", final_cdy - target_cdy)

# =====================================
# 13. 定义边缘提取函数
# =====================================

def find_edges(line, x, dx):

    # 找出所有打印值为 1 的位置
    indices = torch.where(line > 0.5)[0]

    # 如果截线上没有打印图形
    if indices.numel() == 0:
        return None

    # 检查是否为一段连续打印区域
    if torch.any(indices[1:] - indices[:-1] != 1):
        raise ValueError("存在多段打印区域，不能直接测量单一线宽")

    left_index = indices[0].item()
    right_index = indices[-1].item()

    # 从像素中心转换为像素区间边界
    left_edge = x[left_index] - dx / 2
    right_edge = x[right_index] + dx / 2

    return left_edge, right_edge


# =====================================
# 14. 分别测量 x 和 y 方向的边缘
# =====================================

target_edges_x = find_edges(target_x, x, dx)
initial_edges_x = find_edges(initial_x, x, dx)
final_edges_x = find_edges(final_x, x, dx)

target_edges_y = find_edges(target_y, x, dx)
initial_edges_y = find_edges(initial_y, x, dx)
final_edges_y = find_edges(final_y, x, dx)


# =====================================
# 15. 计算 EPE
# =====================================

def calculate_epe(printed_edges, target_edges):

    if printed_edges is None or target_edges is None:
        return None

    left_epe = printed_edges[0] - target_edges[0]

    right_epe = printed_edges[1] - target_edges[1]

    return left_epe, right_epe


initial_epe_x = calculate_epe(
    initial_edges_x, target_edges_x
)

final_epe_x = calculate_epe(
    final_edges_x, target_edges_x
)

initial_epe_y = calculate_epe(
    initial_edges_y, target_edges_y
)

final_epe_y = calculate_epe(
    final_edges_y, target_edges_y
)


print("\n===== EPE Measurement =====")

print("Initial EPE x =", initial_epe_x)
print("Final EPE x =", final_epe_x)

print("Initial EPE y =", initial_epe_y)
print("Final EPE y =", final_epe_y)

# =====================================
# 16. 可视化中心截线
# =====================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# x 方向
axes[0].step(
    x, target_x.numpy(),
    where="mid",
    label="Target",
    linewidth=2
)

axes[0].step(
    x, initial_x.numpy(),
    where="mid",
    label="Initial"
)

axes[0].step(
    x, final_x.numpy(),
    where="mid",
    label="Optimized"
)

axes[0].set_title("Center Cross-section: X")
axes[0].set_xlabel("X")
axes[0].set_ylabel("Binary Pattern")
axes[0].legend()
axes[0].grid(True)


# y 方向
axes[1].step(
    x, target_y.numpy(),
    where="mid",
    label="Target",
    linewidth=2
)

axes[1].step(
    x, initial_y.numpy(),
    where="mid",
    label="Initial"
)

axes[1].step(
    x, final_y.numpy(),
    where="mid",
    label="Optimized"
)

axes[1].set_title("Center Cross-section: Y")
axes[1].set_xlabel("Y")
axes[1].set_ylabel("Binary Pattern")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()
