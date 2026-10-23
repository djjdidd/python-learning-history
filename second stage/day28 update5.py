import numpy as np
import torch

# =========================
# 1. 初始化 Mask 和 Target
# =========================

mask_np = np.array([
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0]
], dtype=np.float32)

mask = torch.tensor(
    mask_np,
    requires_grad=True
)

target = torch.tensor(mask_np.copy())


# =========================
# 2. 构造 Pupil
# =========================

N = 4
cutoff = 0.26

freq = np.fft.fftfreq(N)

FX, FY = np.meshgrid(freq, freq, indexing="ij")

pupil_np = (
    FX**2 + FY**2 <= cutoff**2
).astype(np.float32)

pupil = torch.tensor(pupil_np)


# =========================
# 3. 定义光刻 Forward Model
# =========================

def forward(M):

    F = torch.fft.fft2(M)

    E = torch.fft.ifft2(F * pupil)

    I = torch.abs(E) ** 2

    return I


# =========================
# 4. 创建优化器
# =========================

optimizer = torch.optim.SGD(
    [mask],
    lr=0.5
)


# =========================
# 5. 连续优化
# =========================

initial_loss = torch.mean(
    (forward(mask) - target) ** 2
)

print("Initial Loss =", initial_loss.item())

for step in range(5):

    # 清除上一次的梯度
    optimizer.zero_grad()

    # Forward
    I = forward(mask)

    # Loss
    loss = torch.mean((I - target) ** 2)

    # Backward
    loss.backward()

    # 更新 Mask
    optimizer.step()

    # 限制 Mask 的范围
    with torch.no_grad():
        mask.clamp_(0.0, 1.0)

        # 计算更新后的 Loss
        new_loss = torch.mean(
            (forward(mask) - target) ** 2
        )

    print(
        f"Step {step+1}: "
        f"Loss = {new_loss.item():.8f}"
    )


# =========================
# 6. 输出最终结果
# =========================

print("\nOptimized Mask:")
print(mask.detach())


# =========================
# 7. 对比优化前后的成像
# =========================

with torch.no_grad():

    initial_mask = torch.tensor(mask_np)

    I_initial = forward(initial_mask)

    I_final = forward(mask)

print("\nInitial Aerial Image:")
print(I_initial)

print("\nFinal Aerial Image:")
print(I_final)