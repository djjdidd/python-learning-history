import time
import numpy as np
import torch
import matplotlib.pyplot as plt

# ==========================================
# 1. 创建 Target 和初始 Mask 参数
# ==========================================

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

target_np = target.numpy().astype(bool)

Z0 = 4.0 * target - 2.0


# ==========================================
# 2. 建立 Dose-Focus 光学模型
# ==========================================

freq = np.fft.fftfreq(N, d=dx)

FX, FY = np.meshgrid(
    freq, freq, indexing="ij"
)

f2 = torch.tensor(
    (FX**2 + FY**2).astype(np.float32)
)

pupil0 = torch.tensor(
    (
        FX**2 + FY**2 <= 3.0**2
    ).astype(np.float32)
)

wavelength_um = 0.193

beta = 10.0
threshold = 0.5

# 训练工艺条件：3×3
train_doses = [0.90, 1.00, 1.10]
train_focus = [-50, 0, 50]

# 评价工艺条件：5×5
eval_doses = [0.90, 0.95, 1.00, 1.05, 1.10]
eval_focus = [-50, -25, 0, 25, 50]


# ==========================================
# 3. 创建不同 Focus 下的 Pupil
# ==========================================

pupils = {}

for focus_nm in eval_focus:

    focus_um = focus_nm / 1000.0

    phase = (
        -torch.pi
        * wavelength_um
        * focus_um
        * f2
    )

    pupils[focus_nm] = (
        pupil0 * torch.exp(1j * phase)
    )


# ==========================================
# 4. Resist Model
# ==========================================

def resist_from_field(I, dose):

    R = torch.sigmoid(
        beta * (dose * I - threshold)
    )

    return R


# ==========================================
# 5. 多工艺条件 Printing Loss
# ==========================================

def loss_grid(M, focuses, doses):

    mask_fft = torch.fft.fft2(M)

    rows = []

    for focus_nm in focuses:

        E = torch.fft.ifft2(
            mask_fft * pupils[focus_nm]
        )

        I = torch.abs(E) ** 2

        row = []

        for dose in doses:

            R = resist_from_field(I, dose)

            loss = torch.mean(
                (R - target) ** 2
            )

            row.append(loss)

        rows.append(torch.stack(row))

    return torch.stack(rows)


# ==========================================
# 6. TV Regularization
# ==========================================

def tv_loss(M):

    tv_x = torch.mean(
        torch.abs(M[1:, :] - M[:-1, :])
    )

    tv_y = torch.mean(
        torch.abs(M[:, 1:] - M[:, :-1])
    )

    return tv_x + tv_y


# ==========================================
# 7. Binarization Loss
# ==========================================

def bin_loss(M):

    return torch.mean(
        M * (1.0 - M)
    )


# ==========================================
# 8. 优化函数
# ==========================================

def optimize(tv_weight, bin_weight, steps=30):

    # 每组实验使用相同的初始 Z
    Z = Z0.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam(
        [Z],
        lr=0.1
    )

    t0 = time.perf_counter()

    for step in range(steps):

        optimizer.zero_grad()

        M = torch.sigmoid(Z)

        # Robust Printing Loss
        robust_loss = loss_grid(
            M,
            train_focus,
            train_doses
        ).mean()

        # 总损失
        objective = (
            robust_loss
            + tv_weight * tv_loss(M)
            + bin_weight * bin_loss(M)
        )

        objective.backward()

        optimizer.step()

    elapsed = time.perf_counter() - t0

    M_final = torch.sigmoid(Z.detach())

    return M_final, elapsed


# ==========================================
# 9. CD 和 EPE 评价准备
# ==========================================

center = N // 2

x_rows = np.flatnonzero(
    target_np[:, center]
)

y_cols = np.flatnonzero(
    target_np[center, :]
)

cdx_target = len(x_rows) * dx
cdy_target = len(y_cols) * dx


# ==========================================
# 10. 提取二值图形边缘
# ==========================================

def edges(binary_line):

    indices = np.flatnonzero(binary_line)

    if len(indices) == 0:
        return None

    # 如果图形不连续，不进行单段边缘测量
    if np.any(np.diff(indices) != 1):
        return None

    return (
        int(indices[0]),
        int(indices[-1])
    )


# ==========================================
# 11. 单个工艺条件的 PASS / FAIL
# ==========================================

def condition_pass(binary_print):

    wrong = int(
        np.count_nonzero(
            binary_print != target_np
        )
    )

    edge_x = edges(
        binary_print[:, center]
    )

    edge_y = edges(
        binary_print[center, :]
    )

    if edge_x is None or edge_y is None:
        return False, wrong

    # 计算中心 CD
    cdx = (
        edge_x[1] - edge_x[0] + 1
    ) * dx

    cdy = (
        edge_y[1] - edge_y[0] + 1
    ) * dx

    errors = []

    # 固定 x，测量 y 方向的两侧边缘
    for i in x_rows:

        actual = edges(
            binary_print[i, :]
        )

        if actual is None:
            return False, wrong

        truth = edges(
            target_np[i, :]
        )

        errors.extend(
            abs(actual[k] - truth[k]) * dx
            for k in (0, 1)
        )

    # 固定 y，测量 x 方向的两侧边缘
    for j in y_cols:

        actual = edges(
            binary_print[:, j]
        )

        if actual is None:
            return False, wrong

        truth = edges(
            target_np[:, j]
        )

        errors.extend(
            abs(actual[k] - truth[k]) * dx
            for k in (0, 1)
        )

    # 统一工艺合格判据
    passed = (
        abs(cdx - cdx_target) <= dx + 1e-8
        and abs(cdy - cdy_target) <= dx + 1e-8
        and max(errors) <= dx + 1e-8
        and wrong <= 4
    )

    return passed, wrong


# ==========================================
# 12. 在 5×5 工艺网格下统一评价
# ==========================================

def evaluate_printing(M):

    with torch.no_grad():

        loss_values = loss_grid(
            M,
            eval_focus,
            eval_doses
        )

        M_fft = torch.fft.fft2(M)

        n_pass = 0
        nominal_wrong = None

        for focus in eval_focus:

            E = torch.fft.ifft2(
                M_fft * pupils[focus]
            )

            I = torch.abs(E) ** 2

            for dose in eval_doses:

                R = resist_from_field(I, dose)

                B = (
                    R >= 0.5
                ).cpu().numpy()

                passed, wrong = condition_pass(B)

                n_pass += int(passed)

                # 保存标称条件下的错误像素数
                if focus == 0 and dose == 1.0:
                    nominal_wrong = wrong

        return (
            float(loss_values.mean()),
            float(loss_values.max()),
            n_pass,
            nominal_wrong
        )


# ==========================================
# 13. 定义四组消融实验
# ==========================================

cases = [
    ("Robust",         0.0,  0.0),
    ("Robust+TV",      0.01, 0.0),
    ("Robust+Bin",     0.0,  0.1),
    ("Robust+TV+Bin",  0.01, 0.1)
]

records = []
images = []

# 简单预热，减少首次运行开销
_ = optimize(0.0, 0.0, steps=1)


# ==========================================
# 14. 开始四组对照实验
# ==========================================

for name, tv_weight, bin_weight in cases:

    # 优化连续 Mask
    M, seconds = optimize(
        tv_weight,
        bin_weight
    )

    # 对 Mask 进行硬二值化
    M_binary = (M >= 0.5).float()

    # 连续 Mask 的打印评价
    cm, cw, cp, cwrong = evaluate_printing(M)

    # 二值 Mask 的打印评价
    bm, bw, bp, bwrong = evaluate_printing(
        M_binary
    )

    with torch.no_grad():

        tv = float(tv_loss(M))

        binpen = float(bin_loss(M))

        gray = float(
            (
                (M > 0.1) &
                (M < 0.9)
            ).float().mean() * 100
        )

    records.append((
        name,
        cm, cw, cp,
        bm, bw, bp,
        tv, binpen, gray,
        seconds,
        cwrong, bwrong
    ))

    images.append((
        M.numpy(),
        M_binary.numpy()
    ))


# ==========================================
# 15. 打印实验对比表
# ==========================================

print("\n===== ABLATION STUDY =====")

for row in records:

    (
        name,
        cm, cw, cp,
        bm, bw, bp,
        tv, binpen, gray,
        seconds,
        cwrong, bwrong
    ) = row

    print(f"\nMethod: {name}")

    print(f"Continuous Mean Loss = {cm:.8f}")
    print(f"Continuous Worst Loss = {cw:.8f}")
    print(f"Continuous PASS       = {cp}/25")

    print(f"Binary Mean Loss     = {bm:.8f}")
    print(f"Binary Worst Loss    = {bw:.8f}")
    print(f"Binary PASS          = {bp}/25")

    print(f"Mask TV              = {tv:.8f}")
    print(f"Binarization Loss    = {binpen:.8f}")
    print(f"Gray Pixels (%)      = {gray:.2f}")

    print(f"Training Time (s)    = {seconds:.4f}")

    print(f"Wrong Pixels (Cont.) = {cwrong}")
    print(f"Wrong Pixels (Bin.)  = {bwrong}")


# ==========================================
# 16. 绘制四组 Mask 对比图
# ==========================================

fig, axes = plt.subplots(
    4, 2, figsize=(8, 13)
)

for row, (case, image_pair) in enumerate(
    zip(cases, images)
):

    name = case[0]

    M, B = image_pair

    axes[row, 0].imshow(
        M,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    axes[row, 0].set_title(
        name + ": Continuous Mask"
    )

    axes[row, 1].imshow(
        B,
        cmap="gray",
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    axes[row, 1].set_title(
        name + ": Binary Mask"
    )

    axes[row, 0].axis("off")
    axes[row, 1].axis("off")

plt.tight_layout()

# 保存在当前运行目录
plt.savefig(
    "lesson10_57_masks.png",
    dpi=160,
    bbox_inches="tight"
)

plt.show()