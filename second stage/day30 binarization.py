
import numpy as np
import torch
import matplotlib.pyplot as plt

# ====================================
# 1. Target
# ====================================

N = 64
dx = 2.0 / N

x = (np.arange(N) - N // 2) * dx
X, Y = np.meshgrid(x, x, indexing="ij")

target = torch.tensor(
    ((np.abs(X) <= 0.3) &
     (np.abs(Y) <= 0.2)).astype(np.float32)
)

Z0 = 4 * target - 2

# ====================================
# 2. 光学参数与 Dose-Focus
# ====================================

freq = np.fft.fftfreq(N, d=dx)
FX, FY = np.meshgrid(freq, freq, indexing="ij")

f2 = torch.tensor(
    (FX**2 + FY**2).astype(np.float32)
)

pupil0 = torch.tensor(
    (FX**2 + FY**2 <= 3.0**2).astype(np.float32)
)

wavelength = 0.193
beta = 10.0
threshold = 0.5

train_doses = [0.9, 1.0, 1.1]
train_focus = [-50, 0, 50]

eval_doses = [0.90, 0.95, 1.0, 1.05, 1.10]
eval_focus = [-50, -25, 0, 25, 50]

pupils = {}

for focus in eval_focus:

    phase = (
        -torch.pi * wavelength
        * (focus / 1000.0) * f2
    )

    pupils[focus] = (
        pupil0 * torch.exp(1j * phase)
    )

# ====================================
# 3. 多工艺条件 Printing Loss
# ====================================

def loss_grid(M, focuses, doses):

    F = torch.fft.fft2(M)
    rows = []

    for focus in focuses:

        E = torch.fft.ifft2(
            F * pupils[focus]
        )

        I = torch.abs(E) ** 2

        row = []

        for dose in doses:

            R = torch.sigmoid(
                beta * (dose * I - threshold)
            )

            loss = torch.mean(
                (R - target) ** 2
            )

            row.append(loss)

        rows.append(torch.stack(row))

    return torch.stack(rows)

# ====================================
# 4. TV 正则化
# ====================================

def tv_loss(M):

    tv_x = torch.abs(
        M[1:, :] - M[:-1, :]
    ).mean()

    tv_y = torch.abs(
        M[:, 1:] - M[:, :-1]
    ).mean()

    return tv_x + tv_y

# ====================================
# 5. 新增：Binarization Loss
# ====================================

def bin_loss(M):

    return torch.mean(
        M * (1 - M)
    )

# ====================================
# 6. 优化：Robust + TV + Binarization
# ====================================

def optimize(bin_weight):

    Z = Z0.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam(
        [Z], lr=0.1
    )

    tv_weight = 0.01

    for step in range(30):

        optimizer.zero_grad()

        M = torch.sigmoid(Z)

        # 九个工艺条件的平均 Printing Loss
        printing = loss_grid(
            M, train_focus, train_doses
        ).mean()

        # Mask TV
        tv = tv_loss(M)

        # Mask Binarization Loss
        binary_penalty = bin_loss(M)

        # 完整优化目标
        total_loss = (
            printing
            + tv_weight * tv
            + bin_weight * binary_penalty
        )

        total_loss.backward()
        optimizer.step()

    return torch.sigmoid(Z.detach())

# ====================================
# 7. 标称条件的 Resist
# ====================================

def nominal_resist(M):

    E = torch.fft.ifft2(
        torch.fft.fft2(M) * pupils[0]
    )

    I = torch.abs(E) ** 2

    return torch.sigmoid(
        beta * (I - threshold)
    )

# ====================================
# 8. 实验：三个 Binarization 权重
# ====================================

bin_weights = [0.0, 0.01, 0.1]

results = {}

for weight in bin_weights:

    M = optimize(weight)

    with torch.no_grad():

        # 连续 Mask 的 Printing Loss
        continuous_losses = loss_grid(
            M, eval_focus, eval_doses
        )

        # 将 Mask 硬二值化
        M_binary = (M >= 0.5).float()

        # 二值 Mask 的 Printing Loss
        binary_losses = loss_grid(
            M_binary, eval_focus, eval_doses
        )

        # 两种 Mask 在标称条件下的 Resist
        R_cont = nominal_resist(M)
        R_binary = nominal_resist(M_binary)

        # 测量 Mask 接近二值的程度
        mixed_pixels = (
            (M > 0.1) & (M < 0.9)
        ).sum().item()

        mixed_ratio = mixed_pixels / M.numel() * 100

        # 二值 Mask 的实际打印错误数
        binary_wrong = int(
            ((R_binary >= 0.5) != target.bool())
            .sum().item()
        )

        mean_cont = continuous_losses.mean().item()
        mean_binary = binary_losses.mean().item()

        results[weight] = (
            M.numpy(),
            M_binary.numpy(),
            R_cont.numpy(),
            R_binary.numpy()
        )

        print(f"\n===== Bin Weight = {weight} =====")

        print(f"Continuous Mean Loss = {mean_cont:.8f}")
        print(f"Binary Mask Mean Loss = {mean_binary:.8f}")
        print(f"Mask TV               = {tv_loss(M).item():.8f}")
        print(f"Binarization Loss     = {bin_loss(M).item():.8f}")
        print(f"Mixed Pixels (%)      = {mixed_ratio:.2f}")
        print(f"Binary Printed Errors = {binary_wrong}")

# ====================================
# 9. 图像对比
# ====================================

fig, axes = plt.subplots(
    3, 4, figsize=(14, 10)
)

titles = [
    "Continuous Mask",
    "Binary Mask",
    "Continuous Mask Resist",
    "Binary Mask Resist"
]

for row, weight in enumerate(bin_weights):

    for col, image in enumerate(results[weight]):

        axes[row, col].imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest"
        )

        axes[row, col].set_title(
            f"Bin={weight}: {titles[col]}"
        )

        axes[row, col].axis("off")

plt.tight_layout()
plt.show()
