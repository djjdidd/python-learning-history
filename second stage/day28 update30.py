import numpy as np
import torch

# =========================
# 1. 定义 Target
# =========================

target = torch.tensor([
    [0., 0., 0., 0.],
    [0., 1., 1., 0.],
    [0., 1., 1., 0.],
    [0., 0., 0., 0.]
])

# =========================
# 2. 初始化优化变量 Z
# =========================

Z = torch.tensor([
    [-2., -2., -2., -2.],
    [-2.,  2.,  2., -2.],
    [-2.,  2.,  2., -2.],
    [-2., -2., -2., -2.]
], requires_grad=True)

# =========================
# 3. 构造 Pupil
# =========================

N = 4
cutoff = 0.26

freq = np.fft.fftfreq(N)

FX, FY = np.meshgrid(
    freq, freq, indexing="ij"
)

pupil_np = (
    FX**2 + FY**2 <= cutoff**2
).astype(np.float32)

pupil = torch.tensor(pupil_np)

# =========================
# 4. 定义 Forward Model
# =========================

def forward(Z):

    M = torch.sigmoid(Z)

    F = torch.fft.fft2(M)

    E = torch.fft.ifft2(F * pupil)

    I = torch.abs(E) ** 2

    return I

# =========================
# 5. 创建优化器
# =========================

optimizer = torch.optim.SGD(
    [Z],
    lr=1.0
)

# =========================
# 6. 初始 Loss
# =========================

with torch.no_grad():

    initial_loss = torch.mean(
        (forward(Z) - target) ** 2
    )

print("Initial Loss =", initial_loss.item())

# =========================
# 7. 连续优化 30 次
# =========================

for step in range(30):

    optimizer.zero_grad()

    # Forward
    I = forward(Z)

    # Loss
    loss = torch.mean((I - target) ** 2)

    # Backward
    loss.backward()

    # 更新 Z
    optimizer.step()

    # 查看更新后的 Loss
    with torch.no_grad():

        new_loss = torch.mean(
            (forward(Z) - target) ** 2
        )

    if (step + 1) in [1, 5, 10, 20, 30]:

        print(
            f"Step {step+1}: "
            f"Loss = {new_loss.item():.8f}"
        )

# =========================
# 8. 输出优化后的 Mask
# =========================

with torch.no_grad():

    final_mask = torch.sigmoid(Z)

print("\nOptimized Mask:")
print(final_mask)