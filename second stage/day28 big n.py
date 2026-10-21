import numpy as np
import torch
import matplotlib.pyplot as plt

# =========================
# 1. 建立二维空间网格
# =========================

N = 64

dx = 2.0 / N

x = (np.arange(N) - N // 2) * dx

X, Y = np.meshgrid(x, x, indexing="ij")

# =========================
# 2. 构造 Target
# =========================

target_np = (
    (np.abs(X) <= 0.3) &
    (np.abs(Y) <= 0.2)
).astype(np.float32)

target = torch.tensor(target_np)

# =========================
# 3. 初始化 Z 和 Mask
# =========================

Z = (4 * target - 2).detach().clone()
Z.requires_grad_(True)

M = torch.sigmoid(Z)

# =========================
# 4. 构造 Pupil
# =========================

cutoff = 3.0

freq = np.fft.fftfreq(N, d=dx)

FX, FY = np.meshgrid(
    freq, freq, indexing="ij"
)

pupil_np = (
    FX**2 + FY**2 <= cutoff**2
).astype(np.float32)

pupil = torch.tensor(pupil_np)

# =========================
# 5. Optical Imaging
# =========================

F = torch.fft.fft2(M)

E = torch.fft.ifft2(F * pupil)

I = torch.abs(E) ** 2

# =========================
# 6. Soft Resist
# =========================

beta = 10.0
threshold = 0.5

R = torch.sigmoid(
    beta * (I - threshold)
)

# =========================
# 7. 计算初始 Printing Loss
# =========================

loss = torch.mean((R - target) ** 2)

print("Target shape =", target.shape)

print("Pupil passed points =", pupil.sum().item())

print("Initial Printing Loss =", loss.item())

# =========================
# 8. 可视化
# =========================

images = [
    target.detach().numpy(),
    M.detach().numpy(),
    I.detach().numpy(),
    R.detach().numpy()
]

titles = [
    "Target",
    "Initial Mask",
    "Aerial Image",
    "Soft Resist"
]

fig, axes = plt.subplots(1, 4, figsize=(14, 4))

for ax, image, title in zip(axes, images, titles):

    im = ax.imshow(
        image,
        cmap="gray"
    )

    ax.set_title(title)

    ax.axis("off")

    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()

plt.show()

# =========================
# 9. 定义完整的 ILT Forward
# =========================

def forward_ilt(Z):

    # Mask 参数化
    M = torch.sigmoid(Z)

    # 光学成像
    F = torch.fft.fft2(M)

    E = torch.fft.ifft2(F * pupil)

    I = torch.abs(E) ** 2

    # Soft Resist
    R = torch.sigmoid(
        beta * (I - threshold)
    )

    return M, I, R


# =========================
# 10. 保存优化前的结果
# =========================

with torch.no_grad():

    M_initial, I_initial, R_initial = forward_ilt(Z)

    # 保存独立副本，避免和后续结果混淆
    M_initial = M_initial.clone()
    I_initial = I_initial.clone()
    R_initial = R_initial.clone()

    initial_loss = torch.mean(
        (R_initial - target) ** 2
    )

print("\nInitial Printing Loss =", initial_loss.item())


# =========================
# 11. 创建优化器
# =========================

optimizer = torch.optim.Adam(
    [Z],
    lr=0.1
)


# =========================
# 12. 连续优化 30 次
# =========================

for step in range(30):

    optimizer.zero_grad()

    # Forward
    M, I, R = forward_ilt(Z)

    # Printing Loss
    loss = torch.mean((R - target) ** 2)

    # Backward
    loss.backward()

    # 更新 Z
    optimizer.step()

    # 记录更新后的 Loss
    with torch.no_grad():

        _, _, R_new = forward_ilt(Z)

        new_loss = torch.mean(
            (R_new - target) ** 2
        )

    if (step + 1) in [1, 5, 10, 20, 30]:

        print(
            f"Step {step+1}: "
            f"Printing Loss = {new_loss.item():.8f}"
        )


# =========================
# 13. 保存最终结果
# =========================

with torch.no_grad():

    M_final, I_final, R_final = forward_ilt(Z)

    final_loss = torch.mean(
        (R_final - target) ** 2
    )

print("\nFinal Printing Loss =", final_loss.item())


# =========================
# 14. 对比优化前后的图像
# =========================

before = [
    M_initial.numpy(),
    I_initial.numpy(),
    R_initial.numpy()
]

after = [
    M_final.numpy(),
    I_final.numpy(),
    R_final.numpy()
]

titles = [
    "Mask",
    "Aerial Image",
    "Soft Resist"
]

# Aerial Image 可能超过 1。
# 为保证上下两张图的显示尺度相同，统一设置色标范围。
aerial_max = max(
    I_initial.max().item(),
    I_final.max().item()
)

vmax_list = [1.0, aerial_max, 1.0]

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for j in range(3):

    axes[0, j].imshow(
        before[j],
        cmap="gray",
        vmin=0,
        vmax=vmax_list[j]
    )

    axes[0, j].set_title("Initial " + titles[j])

    axes[1, j].imshow(
        after[j],
        cmap="gray",
        vmin=0,
        vmax=vmax_list[j]
    )

    axes[1, j].set_title("Optimized " + titles[j])

    axes[0, j].axis("off")
    axes[1, j].axis("off")

plt.tight_layout()
plt.show()

# =========================
# 15. 将 Resist 转换为二值图形
# =========================

with torch.no_grad():

    print_initial = (R_initial >= 0.5).float()

    print_final = (R_final >= 0.5).float()

    # 与 Target 不一致的位置
    error_initial = (print_initial != target).float()

    error_final = (print_final != target).float()

    # 统计错误像素数
    wrong_initial = int(error_initial.sum().item())

    wrong_final = int(error_final.sum().item())

    # 总像素数
    total_pixels = target.numel()

    # 计算像素错误率
    error_rate_initial = wrong_initial / total_pixels * 100

    error_rate_final = wrong_final / total_pixels * 100

print("\n===== Binary Printing Evaluation =====")

print("Target pixels =", int(target.sum().item()))

print("Initial printed pixels =", int(print_initial.sum().item()))
print("Final printed pixels =", int(print_final.sum().item()))

print("\nInitial wrong pixels =", wrong_initial)
print("Final wrong pixels =", wrong_final)

print(f"Initial Pixel Error = {error_rate_initial:.4f}%")
print(f"Final Pixel Error = {error_rate_final:.4f}%")


# =========================
# 16. 显示二值打印结果
# =========================

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

fig, axes = plt.subplots(1, 4, figsize=(13, 4))

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