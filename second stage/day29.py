import numpy as np
import torch
import matplotlib.pyplot as plt

# ====================================
# 1. 创建二维空间网格
# ====================================

N = 64

dx = 2.0 / N

x = (np.arange(N) - N // 2) * dx

X, Y = np.meshgrid(x, x, indexing="ij")


# ====================================
# 2. 创建 Target
# ====================================

target_np = (
    (np.abs(X) <= 0.3) &
    (np.abs(Y) <= 0.2)
).astype(np.float32)

target = torch.tensor(target_np)


# ====================================
# 3. 初始化参数化 Mask
# ====================================

Z = 4 * target - 2

Z = Z.clone().detach().requires_grad_(True)

M = torch.sigmoid(Z)


# ====================================
# 4. 创建 Pupil
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
# 5. 计算 Aerial Image
# ====================================

F = torch.fft.fft2(M)

E = torch.fft.ifft2(F * pupil)

I = torch.abs(E) ** 2


# ====================================
# 6. Soft Resist
# ====================================

beta = 10.0
threshold = 0.5

R = torch.sigmoid(
    beta * (I - threshold)
)


# ====================================
# 7. 计算 Printing Loss
# ====================================

loss = torch.mean(
    (R - target) ** 2
)


# ====================================
# 8. 检查数值结果
# ====================================

print("Target shape =", target.shape)

print("Target pixels =", target.sum().item())

print("Pupil passed points =", pupil.sum().item())

print("Initial Printing Loss =", loss.item())


# ====================================
# 9. 可视化四个计算结果
# ====================================

images = [
    target.detach().cpu().numpy(),
    M.detach().cpu().numpy(),
    I.detach().cpu().numpy(),
    R.detach().cpu().numpy()
]

titles = [
    "Target",
    "Initial Mask",
    "Aerial Image",
    "Soft Resist"
]

fig, axes = plt.subplots(
    1, 4, figsize=(14, 4)
)

for ax, image, title in zip(axes, images, titles):

    im = ax.imshow(
        image,
        cmap="gray",
        interpolation="nearest"
    )

    ax.set_title(title)

    ax.axis("off")

    plt.colorbar(
        im,
        ax=ax,
        fraction=0.046
    )

plt.tight_layout()

plt.show()