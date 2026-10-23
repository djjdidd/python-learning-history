
import numpy as np
import torch
import matplotlib.pyplot as plt

# =====================================
# 1. Target：空间单位为 μm
# =====================================

N = 64
dx = 2.0 / N

x = (np.arange(N) - N // 2) * dx
X, Y = np.meshgrid(x, x, indexing="ij")

target = torch.tensor(
    (
        (np.abs(X) <= 0.3) &
        (np.abs(Y) <= 0.2)
    ).astype(np.float32)
)

Z0 = 4 * target - 2

# =====================================
# 2. Pupil 和工艺参数
# =====================================

freq = np.fft.fftfreq(N, d=dx)
FX, FY = np.meshgrid(freq, freq, indexing="ij")

f2 = torch.tensor(
    (FX**2 + FY**2).astype(np.float32)
)

pupil0 = torch.tensor(
    (FX**2 + FY**2 <= 3.0**2).astype(np.float32)
)

wavelength_um = 0.193
beta = 10.0
threshold = 0.5

# 训练使用 3×3
train_doses = [0.9, 1.0, 1.1]
train_focus = [-50, 0, 50]

# 评价使用 5×5
eval_doses = [0.90, 0.95, 1.00, 1.05, 1.10]
eval_focus = [-50, -25, 0, 25, 50]

# 提前生成评价需要的五个 Pupil
pupils = {}

for f_nm in eval_focus:

    focus_um = f_nm / 1000.0

    phase = (
        -torch.pi
        * wavelength_um
        * focus_um
        * f2
    )

    pupils[f_nm] = pupil0 * torch.exp(1j * phase)

# =====================================
# 3. Forward Model
# =====================================

def forward(Z, dose, focus_nm):

    M = torch.sigmoid(Z)

    E = torch.fft.ifft2(
        torch.fft.fft2(M) * pupils[focus_nm]
    )

    I = torch.abs(E) ** 2

    R = torch.sigmoid(
        beta * (dose * I - threshold)
    )

    return R


def loss_at(Z, dose, focus_nm):

    R = forward(Z, dose, focus_nm)

    return torch.mean((R - target) ** 2)

# =====================================
# 4. 分别训练两种 ILT
# =====================================

def optimize(method):

    Z = Z0.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([Z], lr=0.1)

    for step in range(30):

        optimizer.zero_grad()

        if method == "Nominal":

            loss = loss_at(Z, 1.0, 0)

        else:

            losses = torch.stack([
                loss_at(Z, d, f)

                for f in train_focus
                for d in train_doses
            ])

            loss = losses.mean()

        loss.backward()
        optimizer.step()

    return Z.detach()


Z_nominal = optimize("Nominal")
Z_robust = optimize("Robust")

# =====================================
# 5. 定义 CD 和 EPE 测量
# =====================================

center = N // 2

# Target 内部所有有效的行、列
x_rows = torch.where(
    target[:, center] > 0.5
)[0].tolist()

y_cols = torch.where(
    target[center, :] > 0.5
)[0].tolist()

target_cdx = len(x_rows) * dx
target_cdy = len(y_cols) * dx


def line_edges(line):

    indices = torch.where(line > 0.5)[0]

    if indices.numel() == 0:
        return None

    # 目前只接受一段连续打印区域
    if torch.any(
        indices[1:] - indices[:-1] != 1
    ).item():
        return None

    return (
        int(indices[0].item()),
        int(indices[-1].item())
    )


# =====================================
# 6. 评价一个 Dose-Focus 条件
# =====================================

def evaluate_one(Z, dose, focus_nm):

    with torch.no_grad():

        R = forward(Z, dose, focus_nm)

        loss = torch.mean(
            (R - target) ** 2
        ).item()

        B = (R >= 0.5).float()

        # 错误像素数
        wrong = int((B != target).sum().item())

        # 中心 CD
        edge_x = line_edges(B[:, center])
        edge_y = line_edges(B[center, :])

        cdx = (
            (edge_x[1] - edge_x[0] + 1) * dx
            if edge_x is not None else None
        )

        cdy = (
            (edge_y[1] - edge_y[0] + 1) * dx
            if edge_y is not None else None
        )

        # 在 Target 所有有效行、列上检查边缘
        errors = []
        valid = True

        # 固定 x，沿 y 方向测量
        for i in x_rows:

            target_edge = line_edges(target[i, :])
            printed_edge = line_edges(B[i, :])

            if printed_edge is None:
                valid = False
                break

            errors.extend([
                abs(printed_edge[k] - target_edge[k]) * dx
                for k in (0, 1)
            ])

        # 固定 y，沿 x 方向测量
        if valid:

            for j in y_cols:

                target_edge = line_edges(target[:, j])
                printed_edge = line_edges(B[:, j])

                if printed_edge is None:
                    valid = False
                    break

                errors.extend([
                    abs(printed_edge[k] - target_edge[k]) * dx
                    for k in (0, 1)
                ])

        # 缺失或断裂时，不将 EPE 错误地设为 0
        max_epe = max(errors) if valid else None

        # 本次教学实验的合格判据
        passed = (
            cdx is not None
            and cdy is not None
            and max_epe is not None

            and abs(cdx - target_cdx) <= dx + 1e-8
            and abs(cdy - target_cdy) <= dx + 1e-8

            and max_epe <= dx + 1e-8
            and wrong <= 4
        )

        return loss, wrong, max_epe, passed


# =====================================
# 7. 使用 5×5 网格统一评价
# =====================================

results = {}

for name, Z in [
    ("Nominal", Z_nominal),
    ("Robust", Z_robust)
]:

    loss_map = np.zeros((5, 5))
    pass_map = np.zeros((5, 5), dtype=int)

    print(f"\n===== {name} Evaluation =====")

    print("Target CDx =", target_cdx)
    print("Target CDy =", target_cdy)

    for i, focus in enumerate(eval_focus):

        for j, dose in enumerate(eval_doses):

            loss, wrong, max_epe, passed = evaluate_one(
                Z, dose, focus
            )

            loss_map[i, j] = loss
            pass_map[i, j] = int(passed)

    print("\nPASS Map (1=PASS, 0=FAIL):")
    print(pass_map)

    print(
        "PASS Count =",
        int(pass_map.sum()),
        "/ 25"
    )

    print("Average Loss =", loss_map.mean())
    print("Worst Loss =", loss_map.max())

    results[name] = (loss_map, pass_map)


# =====================================
# 8. 可视化 Loss Map 和 PASS Map
# =====================================

fig, axes = plt.subplots(
    2, 2, figsize=(11, 9),
    constrained_layout=True
)

vmax = max(
    results[name][0].max()
    for name in results
)

for row, name in enumerate(["Nominal", "Robust"]):

    loss_map, pass_map = results[name]

    maps = [loss_map, pass_map]

    for col, values in enumerate(maps):

        ax = axes[row, col]

        if col == 0:

            ax.imshow(
                values,
                cmap="viridis",
                vmin=0,
                vmax=vmax
            )

            title = "Printing Loss"

        else:

            ax.imshow(
                values,
                cmap="gray",
                vmin=0,
                vmax=1
            )

            title = "PASS Map (White = PASS)"

        ax.set_title(f"{name}: {title}")

        ax.set_xticks(
            range(5),
            [f"{d:.0%}" for d in eval_doses]
        )

        ax.set_yticks(
            range(5),
            [str(f) for f in eval_focus]
        )

        ax.set_xlabel("Dose")
        ax.set_ylabel("Focus (nm)")

        for i in range(5):

            for j in range(5):

                if col == 0:
                    label = f"{values[i, j]:.4f}"
                else:
                    label = "PASS" if values[i, j] else "FAIL"

                ax.text(
                    j, i, label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=(
                        "black"
                        if col == 1 and values[i, j] == 1
                        else "white"
                    )
                )

plt.show()
