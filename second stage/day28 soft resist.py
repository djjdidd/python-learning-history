import numpy as np
import torch

# =========================
# 1. Target
# =========================

target = torch.tensor([
    [0., 0., 0., 0.],
    [0., 1., 1., 0.],
    [0., 1., 1., 0.],
    [0., 0., 0., 0.]
])

# =========================
# 2. 初始化 Z
# =========================

Z = torch.tensor([
    [-2., -2., -2., -2.],
    [-2.,  2.,  2., -2.],
    [-2.,  2.,  2., -2.],
    [-2., -2., -2., -2.]
], requires_grad=True)

# =========================
# 3. Pupil
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

# Resist 参数
beta = 10.0
threshold = 0.5

# =========================
# 4. Forward Model
# =========================

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

    return R

# =========================
# 5. 创建优化器
# =========================

optimizer = torch.optim.SGD(
    [Z],
    lr=1.0
)

# =========================
# 6. 初始 Printing Loss
# =========================

with torch.no_grad():

    R_initial = forward(Z)

    initial_loss = torch.mean(
        (R_initial - target) ** 2
    )

print("Initial Printing Loss =", initial_loss.item())

# =========================
# 7. 连续优化五次
# =========================

for step in range(5):

    optimizer.zero_grad()

    # Forward
    R = forward(Z)

    # Printing Loss
    loss = torch.mean((R - target) ** 2)

    # Backward
    loss.backward()

    # 更新 Z
    optimizer.step()

    # 查看更新后的 Loss
    with torch.no_grad():

        R_new = forward(Z)

        new_loss = torch.mean(
            (R_new - target) ** 2
        )

    print(
        f"Step {step+1}: "
        f"Loss = {new_loss.item():.8f}"
    )

# =========================
# 8. 输出最终结果
# =========================

with torch.no_grad():

    final_mask = torch.sigmoid(Z)

    final_resist = forward(Z)

print("\nFinal Mask:")
print(final_mask)

print("\nFinal Resist:")
print(final_resist)