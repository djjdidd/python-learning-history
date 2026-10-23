
import numpy as np
import torch
import matplotlib.pyplot as plt

# =====================================
# 1. 二维 Target
# 空间坐标单位：μm
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
# 2. 构造原始 Pupil
# =====================================

cutoff = 3.0

freq = np.fft.fftfreq(N, d=dx)

FX, FY = np.meshgrid(freq, freq, indexing="ij")

f_squared = torch.tensor(
    (FX**2 + FY**2).astype(np.float32)
)

pupil0 = torch.tensor(
    (FX**2 + FY**2 <= cutoff**2).astype(np.float32)
)


# =====================================
# 3. Dose、Focus 和物理参数
# =====================================

wavelength_um = 0.193

dose_values = [0.9, 1.0, 1.1]

# Focus 单位：nm
focus_values_nm = [-50, 0, 50]

beta = 10.0
threshold = 0.5


# =====================================
# 4. 为三个 Focus 分别生成 Pupil
# =====================================

pupils = {}

for focus_nm in focus_values_nm:

    # nm 转换为 μm
    focus_um = focus_nm / 1000.0

    # Defocus Phase
    phase = (
        -torch.pi
        * wavelength_um
        * focus_um
        * f_squared
    )

    pupil_F = pupil0 * torch.exp(1j * phase)

    pupils[focus_nm] = pupil_F


# =====================================
# 5. 同时考虑 Dose 和 Focus 的 Forward
# =====================================

def forward(Z, dose, focus_nm):

    M = torch.sigmoid(Z)

    # 不同 Focus 使用不同 Pupil
    pupil_F = pupils[focus_nm]

    F = torch.fft.fft2(M)

    E = torch.fft.ifft2(F * pupil_F)

    I = torch.abs(E) ** 2

    # 不同 Dose 对应不同 Resist
    R = torch.sigmoid(
        beta * (dose * I - threshold)
    )

    return M, I, R


# =====================================
# 6. 单个工艺条件的 Printing Loss
# =====================================

def printing_loss(Z, dose, focus_nm):

    _, _, R = forward(Z, dose, focus_nm)

    return torch.mean((R - target) ** 2)


# =====================================
# 7. 统一评价九个工艺条件
# =====================================

def evaluate(Z):

    with torch.no_grad():

        values = torch.stack([

            torch.stack([
                printing_loss(Z, dose, focus_nm)
                for dose in dose_values
            ])

            for focus_nm in focus_values_nm

        ])

    return values


# =====================================
# 8. 先检查 Focus 模型是否正确
# =====================================

with torch.no_grad():

    M0 = torch.sigmoid(Z0)

    # 旧模型：没有 Defocus Phase
    E_old = torch.fft.ifft2(
        torch.fft.fft2(M0) * pupil0
    )

    I_old = torch.abs(E_old) ** 2

    # 新模型：Focus = 0
    _, I_zero, _ = forward(Z0, 1.0, 0)

    # 新模型：Focus = ±50 nm
    _, I_minus, _ = forward(Z0, 1.0, -50)
    _, I_plus, _ = forward(Z0, 1.0, 50)

    zero_difference = torch.max(
        torch.abs(I_old - I_zero)
    )

    defocus_change = torch.max(
        torch.abs(I_plus - I_zero)
    )

    symmetry_difference = torch.max(
        torch.abs(I_minus - I_plus)
    )

print("===== Model Verification =====")

print("Zero-focus difference =", zero_difference.item())

print("Defocus image change =", defocus_change.item())

print(
    "Plus/minus focus difference =",
    symmetry_difference.item()
)

print("\nInitial 3x3 Loss:")
print(evaluate(Z0))


# =====================================
# 9. 定义优化过程
# =====================================

def optimize(name):

    # 两种方法使用相同的初始 Z
    Z = Z0.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([Z], lr=0.1)

    for step in range(30):

        optimizer.zero_grad()

        if name == "Nominal":

            # 只优化 Dose=100%, Focus=0
            loss = printing_loss(Z, 1.0, 0)

        else:

            # 九个工艺条件使用同一张 Mask
            losses = torch.stack([

                printing_loss(Z, dose, focus_nm)

                for focus_nm in focus_values_nm
                for dose in dose_values

            ])

            loss = losses.mean()

        loss.backward()

        optimizer.step()

        # 统一评价更新后的九个工艺条件
        if (step + 1) in [1, 5, 10, 20, 30]:

            ls = evaluate(Z)

            print(
                f"{name:7s} Step {step+1:2d}: "
                f"Nominal={ls[1,1].item():.8f}  "
                f"Mean={ls.mean().item():.8f}  "
                f"Worst={ls.max().item():.8f}"
            )

    return Z.detach()


# =====================================
# 10. 运行两种方法
# =====================================

Z_nominal = optimize("Nominal")

Z_robust = optimize("Robust")


# =====================================
# 11. 统一评价
# =====================================

nominal_table = evaluate(Z_nominal)

robust_table = evaluate(Z_robust)

for name, values in [
    ("Nominal", nominal_table),
    ("Robust", robust_table)
]:

    print(f"\n===== {name} Evaluation =====")

    print("Rows: Focus -50, 0, +50 nm")
    print("Cols: Dose 90%, 100%, 110%")

    print(values)

    print("Nominal Loss =", values[1, 1].item())
    print("Mean Loss =", values.mean().item())
    print("Worst Loss =", values.max().item())


# =====================================
# 12. 绘制 Dose-Focus Loss Map
# =====================================

vmax = max(
    nominal_table.max().item(),
    robust_table.max().item()
)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for ax, title, values in zip(
    axes,
    ["Nominal ILT", "Robust ILT"],
    [nominal_table, robust_table]
):

    im = ax.imshow(
        values.numpy(),
        vmin=0,
        vmax=vmax,
        cmap="viridis"
    )

    ax.set_xticks(
        range(3),
        ["90%", "100%", "110%"]
    )

    ax.set_yticks(
        range(3),
        ["-50", "0", "+50"]
    )

    ax.set_xlabel("Dose")
    ax.set_ylabel("Focus (nm)")
    ax.set_title(title)

    # 将具体 Loss 标在对应工艺条件上
    for i in range(3):

        for j in range(3):

            value = values[i, j].item()

            ax.text(
                j, i,
                f"{value:.4f}",
                ha="center",
                va="center",
                color=(
                    "white" if value < vmax / 2
                    else "black"
                )
            )

fig.colorbar(
    im,
    ax=axes.ravel().tolist(),
    shrink=0.8,
    label="Printing Loss"
)

plt.show()
