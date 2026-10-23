
import numpy as np
import torch
import matplotlib.pyplot as plt

# =====================================
# 1. Target 和初始参数
# =====================================

N = 64
dx = 2.0 / N

x = (np.arange(N) - N // 2) * dx

X, Y = np.meshgrid(x, x, indexing="ij")

target_np = (
    (np.abs(X) <= 0.3) &
    (np.abs(Y) <= 0.2)
).astype(np.float32)

target = torch.tensor(target_np)

Z0 = 4 * target - 2


# =====================================
# 2. Pupil
# =====================================

freq = np.fft.fftfreq(N, d=dx)

FX, FY = np.meshgrid(
    freq, freq, indexing="ij"
)

pupil_np = (
    FX**2 + FY**2 <= 3.0**2
).astype(np.float32)

pupil = torch.tensor(pupil_np)


# =====================================
# 3. 设置 Dose 和 Resist 参数
# =====================================

beta = 10.0
threshold = 0.5

train_doses = [0.9, 1.0, 1.1]


# =====================================
# 4. 加入 Dose 的 Forward Model
# =====================================

def forward(Z, dose):

    # Mask 参数化
    M = torch.sigmoid(Z)

    # Optical Imaging
    F = torch.fft.fft2(M)

    E = torch.fft.ifft2(F * pupil)

    I = torch.abs(E) ** 2

    # Dose-dependent Soft Resist
    R = torch.sigmoid(
        beta * (dose * I - threshold)
    )

    return M, I, R


# =====================================
# 5. 同一张 Mask 在三个 Dose 下的 Loss
# =====================================

def dose_losses(Z):

    values = []

    for dose in train_doses:

        _, _, R = forward(Z, dose)

        loss = torch.mean(
            (R - target) ** 2
        )

        values.append(loss)

    return torch.stack(values)


# =====================================
# 6. 定义统一的优化函数
# =====================================

def optimize(method):

    # 每种方法都从同一个 Z0 开始
    Z = Z0.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam(
        [Z],
        lr=0.1
    )

    for step in range(30):

        optimizer.zero_grad()

        # 计算三个 Dose 下的 Loss
        each_loss = dose_losses(Z)

        # 两种不同的优化目标
        if method == "Nominal":

            # 只优化 Dose = 100%
            loss = each_loss[1]

        else:

            # 优化三个 Dose 的平均 Loss
            loss = each_loss.mean()

        # 自动求导
        loss.backward()

        # 更新共同的 Z
        optimizer.step()

        # 查看更新后的结果
        if (step + 1) in [1, 5, 10, 20, 30]:

            with torch.no_grad():

                now = dose_losses(Z)

                print(
                    f"{method:7s} Step {step+1:2d}: "
                    f"D90={now[0].item():.8f}, "
                    f"D100={now[1].item():.8f}, "
                    f"D110={now[2].item():.8f}"
                )

    return Z.detach()


# =====================================
# 7. 记录初始状态
# =====================================

with torch.no_grad():

    initial = dose_losses(Z0)

print(
    "Initial losses (90%, 100%, 110%) =",
    [round(v, 8) for v in initial.tolist()]
)


# =====================================
# 8. 分别运行两种优化方法
# =====================================

Z_nominal = optimize("Nominal")

Z_robust = optimize("Robust")


# =====================================
# 9. 统一评价两种方法
# =====================================

print("\n===== Common Evaluation =====")

for name, Z in [
    ("Nominal", Z_nominal),
    ("Robust", Z_robust)
]:

    with torch.no_grad():

        ls = dose_losses(Z)

        mean_loss = ls.mean().item()

        worst_loss = ls.max().item()

    print(
        f"{name:7s}: "
        f"90%={ls[0].item():.8f}, "
        f"100%={ls[1].item():.8f}, "
        f"110%={ls[2].item():.8f}, "
        f"Mean={mean_loss:.8f}, "
        f"Worst={worst_loss:.8f}"
    )


# =====================================
# 10. 可视化两种 Mask 和 Resist
# =====================================

fig, axes = plt.subplots(
    2, 4, figsize=(13, 7)
)

methods = [
    ("Nominal", Z_nominal),
    ("Robust", Z_robust)
]

for row, (name, Z) in enumerate(methods):

    with torch.no_grad():

        mask = torch.sigmoid(Z).cpu().numpy()

    axes[row, 0].imshow(
        mask,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[row, 0].set_title(f"{name} Mask")

    for col, dose in enumerate(train_doses, start=1):

        with torch.no_grad():

            _, _, R = forward(Z, dose)

        axes[row, col].imshow(
            R.cpu().numpy(),
            cmap="gray",
            vmin=0,
            vmax=1
        )

        axes[row, col].set_title(
            f"{name}: Dose {dose:.0%}"
        )

    for ax in axes[row]:

        ax.axis("off")

plt.tight_layout()
plt.show()
