
import numpy as np
import torch
import matplotlib.pyplot as plt

# ====================================
# 1. Target 和初始参数
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
# 2. Dose-Focus 光学模型
# ====================================

freq = np.fft.fftfreq(N, d=dx)
FX, FY = np.meshgrid(freq, freq, indexing="ij")

f2 = torch.tensor(
    (FX**2 + FY**2).astype(np.float32)
)

pupil0 = torch.tensor(
    (FX**2 + FY**2 <= 3.0**2).astype(np.float32)
)

wavelength = 0.193  # μm
beta = 10.0
threshold = 0.5

train_doses = [0.9, 1.0, 1.1]
train_focus = [-50, 0, 50]

eval_doses = [0.90, 0.95, 1.0, 1.05, 1.10]
eval_focus = [-50, -25, 0, 25, 50]

pupils = {}

for focus_nm in eval_focus:

    focus_um = focus_nm / 1000.0

    phase = (
        -torch.pi * wavelength
        * focus_um * f2
    )

    pupils[focus_nm] = (
        pupil0 * torch.exp(1j * phase)
    )

# ====================================
# 3. 多条件 Printing Loss
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
# 4. 新增：TV Regularization
# ====================================

def tv_loss(M):

    # 第一个空间方向的相邻差分
    tv_x = torch.mean(
        torch.abs(M[1:, :] - M[:-1, :])
    )

    # 第二个空间方向的相邻差分
    tv_y = torch.mean(
        torch.abs(M[:, 1:] - M[:, :-1])
    )

    return tv_x + tv_y


# ====================================
# 5. 根据不同 TV 权重优化
# ====================================

def optimize(tv_weight):

    Z = Z0.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam(
        [Z], lr=0.1
    )

    for step in range(30):

        optimizer.zero_grad()

        M = torch.sigmoid(Z)

        # 3x3 Dose-Focus Printing Loss
        printing = loss_grid(
            M, train_focus, train_doses
        ).mean()

        # Mask TV
        tv = tv_loss(M)

        # 总优化目标
        total_loss = printing + tv_weight * tv

        total_loss.backward()
        optimizer.step()

    return Z.detach()


# ====================================
# 6. 标称条件的 Resist
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
# 7. 运行三组对照实验
# ====================================

tv_weights = [0.0, 0.01, 0.05]

results = {}

print("===== TV Regularization Experiment =====")

for weight in tv_weights:

    Z_final = optimize(weight)

    with torch.no_grad():

        M = torch.sigmoid(Z_final)

        # 统一使用 5x5 评价网格
        losses = loss_grid(
            M, eval_focus, eval_doses
        )

        tv_value = tv_loss(M).item()

        nominal_loss = losses[2, 2].item()
        mean_loss = losses.mean().item()
        worst_loss = losses.max().item()

        # 仅用于检查：将 Mask 硬二值化
        binary_mask = (M >= 0.5).float()

        # 统计二值 Mask 的相邻边界变化数量
        edge_changes = int((
            (binary_mask[1:, :] != binary_mask[:-1, :]).sum()
            + (binary_mask[:, 1:] != binary_mask[:, :-1]).sum()
        ).item())

        # 检查 Mask 二值化后，成像损失的变化
        binary_losses = loss_grid(
            binary_mask, eval_focus, eval_doses
        )

        binary_mean = binary_losses.mean().item()

        R = nominal_resist(M)

        wrong_pixels = int(
            ((R >= 0.5) != target.bool()).sum().item()
        )

        results[weight] = (
            M.numpy(),
            binary_mask.numpy(),
            R.numpy()
        )

        print(f"\nTV weight = {weight}")
        print(f"Nominal Printing Loss = {nominal_loss:.8f}")
        print(f"Mean Printing Loss    = {mean_loss:.8f}")
        print(f"Worst Printing Loss   = {worst_loss:.8f}")
        print(f"Mask TV               = {tv_value:.8f}")
        print(f"Binary edge changes   = {edge_changes}")
        print(f"Printed wrong pixels  = {wrong_pixels}")
        print(f"Binary Mask Mean Loss = {binary_mean:.8f}")


# ====================================
# 8. 可视化
# ====================================

fig, axes = plt.subplots(
    3, 3, figsize=(11, 10)
)

titles = [
    "Continuous Mask",
    "Thresholded Mask",
    "Nominal Soft Resist"
]

for row, weight in enumerate(tv_weights):

    images = results[weight]

    for col, image in enumerate(images):

        axes[row, col].imshow(
            image,
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest"
        )

        axes[row, col].set_title(
            f"TV={weight}: {titles[col]}"
        )

        axes[row, col].axis("off")

plt.tight_layout()
plt.show()
