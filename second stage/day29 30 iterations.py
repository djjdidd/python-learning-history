
import numpy as np
import torch
import matplotlib.pyplot as plt

# ====================================
# 1. 创建二维 Target
# ====================================

N = 64
dx = 2.0 / N

x = (np.arange(N) - N // 2) * dx

X, Y = np.meshgrid(x, x, indexing="ij")

target_np = (
    (np.abs(X) <= 0.3) &
    (np.abs(Y) <= 0.2)
).astype(np.float32)

target = torch.tensor(target_np)


# ====================================
# 2. 初始化 Z
# ====================================

Z = (4 * target - 2).detach().clone()

Z.requires_grad_(True)


# ====================================
# 3. 构造 Pupil
# ====================================

cutoff = 3.0

freq = np.fft.fftfreq(N, d=dx)

FX, FY = np.meshgrid(
    freq, freq, indexing="ij"
)

pupil_np = (
    FX**2 + FY**2 <= cutoff**2
).astype(np.float32)

pupil = torch.tensor(pupil_np)


# ====================================
# 4. Soft Resist 参数
# ====================================

beta = 10.0
threshold = 0.5


# ====================================
# 5. 定义完整 Forward Model
# ====================================

def forward(Z):

    # Mask 参数化
    M = torch.sigmoid(Z)

    # Optical Imaging
    F = torch.fft.fft2(M)

    E = torch.fft.ifft2(F * pupil)

    I = torch.abs(E) ** 2

    # Soft Resist
    R = torch.sigmoid(
        beta * (I - threshold)
    )

    return M, I, R


# ====================================
# 6. 保存优化前的结果
# ====================================

with torch.no_grad():

    M_initial, I_initial, R_initial = forward(Z)

    initial_loss = torch.mean(
        (R_initial - target) ** 2
    )

print("Target pixels =", target.sum().item())

print("Pupil passed points =", pupil.sum().item())

print("Initial Loss =", initial_loss.item())


# ====================================
# 7. 创建 Adam 优化器
# ====================================

optimizer = torch.optim.Adam(
    [Z],
    lr=0.1
)


# ====================================
# 8. 连续优化 30 次
# ====================================

for step in range(30):

    # 清空上一轮梯度
    optimizer.zero_grad()

    # Forward
    M, I, R = forward(Z)

    # Printing Loss
    loss = torch.mean(
        (R - target) ** 2
    )

    # Backward
    loss.backward()

    # 更新 Z
    optimizer.step()

    # 计算更新后的 Loss
    with torch.no_grad():

        _, _, R_new = forward(Z)

        new_loss = torch.mean(
            (R_new - target) ** 2
        )

    if (step + 1) in [1, 5, 10, 20, 30]:

        print(
            f"Step {step + 1}: "
            f"Loss = {new_loss.item():.8f}"
        )


# ====================================
# 9. 保存优化后的结果
# ====================================

with torch.no_grad():

    M_final, I_final, R_final = forward(Z)

    final_loss = torch.mean(
        (R_final - target) ** 2
    )

print("\nFinal Loss =", final_loss.item())


# ====================================
# 10. 可视化优化前后的图像
# ====================================

before = [
    M_initial.detach().numpy(),
    I_initial.detach().numpy(),
    R_initial.detach().numpy()
]

after = [
    M_final.detach().numpy(),
    I_final.detach().numpy(),
    R_final.detach().numpy()
]

titles = [
    "Mask",
    "Aerial Image",
    "Soft Resist"
]

# 保证优化前后 Aerial Image 的色标一致
aerial_max = max(
    I_initial.max().item(),
    I_final.max().item()
)

vmax_list = [1.0, aerial_max, 1.0]

fig, axes = plt.subplots(
    2, 3, figsize=(12, 8)
)

for j in range(3):

    axes[0, j].imshow(
        before[j],
        cmap="gray",
        vmin=0,
        vmax=vmax_list[j]
    )

    axes[0, j].set_title(
        "Initial " + titles[j]
    )

    axes[1, j].imshow(
        after[j],
        cmap="gray",
        vmin=0,
        vmax=vmax_list[j]
    )

    axes[1, j].set_title(
        "Optimized " + titles[j]
    )

    axes[0, j].axis("off")
    axes[1, j].axis("off")

plt.tight_layout()

plt.show()
